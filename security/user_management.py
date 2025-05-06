import os
import logging
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
import json
from datetime import datetime
import pymongo
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

from security.auth import AuthManager
from security.metrics import record_auth_attempt, record_security_event, inc_active_users, dec_active_users, set_active_users

# Configure logging
logger = logging.getLogger(__name__)

class UserManager:
    """User management system for the RL application."""
    
    # Define standard roles and their permissions
    ROLES = {
        'admin': ['user:create', 'user:read', 'user:update', 'user:delete', 'user:manage_roles', 
                  'model:create', 'model:read', 'model:update', 'model:delete',
                  'data:read', 'data:write', 'data:delete', 'system:admin'],
                  
        'data_scientist': ['model:create', 'model:read', 'model:update', 
                           'data:read', 'data:write'],
                           
        'analyst': ['model:read', 'data:read'],
        
        'developer': ['model:read', 'model:update', 'data:read'],
        
        'guest': ['model:read', 'data:read']
    }
    
    # Define permissions
    PERMISSIONS = {
        'user:create': 'Create user accounts',
        'user:read': 'View user details',
        'user:update': 'Update user details',
        'user:delete': 'Delete user accounts',
        'user:manage_roles': 'Assign and modify user roles',
        'model:create': 'Create ML models',
        'model:read': 'View ML models',
        'model:update': 'Update ML models',
        'model:delete': 'Delete ML models',
        'data:read': 'View data',
        'data:write': 'Create or update data',
        'data:delete': 'Delete data',
        'system:admin': 'Administrative operations'
    }
    
    def __init__(self, auth_manager: AuthManager, db_uri: Optional[str] = None):
        """Initialize the user management system.
        
        Args:
            auth_manager: Authentication manager instance
            db_uri: MongoDB connection URI. If None, use env var MONGODB_URI
        """
        self.auth_manager = auth_manager
        self.db_uri = db_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/rl_results')
        self.client = None
        self.db = None
        self.users_collection = None
        
        # Initialize database connection
        self._init_db()
        
    def _init_db(self) -> None:
        """Initialize database connection and collections."""
        try:
            self.client = MongoClient(self.db_uri)
            self.client.admin.command('ping')  # Test connection
            
            # Get database name from URI
            db_name = self.db_uri.split('/')[-1]
            self.db = self.client[db_name]
            
            # Access users collection
            self.users_collection = self.db.users
            
            # Create indexes
            self.users_collection.create_index([("username", ASCENDING)], unique=True)
            self.users_collection.create_index([("email", ASCENDING)], unique=True)
            
            logger.info(f"Connected to MongoDB for user management: {self.db_uri}")
            
            # Create admin user if no users exist
            if self.users_collection.count_documents({}) == 0:
                self._create_initial_admin()
                
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            record_security_event("database_connection_error", "error")
            self.client = None
            self.db = None
    
    def _create_initial_admin(self) -> None:
        """Create the initial admin user"""
        try:
            admin_password = os.getenv('INITIAL_ADMIN_PASSWORD') or str(uuid.uuid4())
            
            # Create admin user
            admin_salt, admin_key = self.auth_manager.hash_password(admin_password)
            
            admin_user = {
                'username': 'admin',
                'email': os.getenv('ADMIN_EMAIL', 'admin@example.com'),
                'password_salt': admin_salt,
                'password_key': admin_key,
                'role': 'admin',
                'created_at': datetime.utcnow(),
                'last_login': None,
                'active': True,
                'metadata': {}
            }
            
            self.users_collection.insert_one(admin_user)
            logger.info(f"Created initial admin user with username 'admin'")
            
            if not os.getenv('INITIAL_ADMIN_PASSWORD'):
                logger.info(f"Generated admin password: {admin_password}")
                logger.info("Please change this password immediately after first login!")
                
            record_security_event("admin_user_created", "info")
            
        except Exception as e:
            logger.error(f"Failed to create admin user: {str(e)}")
            record_security_event("admin_user_creation_failed", "error")
            
    def create_user(self, 
                   username: str, 
                   password: str, 
                   email: str, 
                   roles: List[str] = None, 
                   full_name: Optional[str] = None) -> Tuple[bool, str]:
        """Create a new user.
        
        Args:
            username: Username
            password: Password
            email: Email address
            roles: List of roles (default: ['guest'])
            full_name: User's full name (optional)
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Validate roles
        if roles is None:
            roles = ['guest']
            
        for role in roles:
            if role not in self.ROLES:
                return False, f"Invalid role: {role}"
                
        # Hash password
        hashed_password = self.auth_manager.hash_password(password)
        
        # Create user document
        user = {
            'user_id': str(uuid.uuid4()),
            'username': username,
            'password': hashed_password,
            'email': email,
            'full_name': full_name,
            'roles': roles,
            'active': True,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        try:
            self.users_collection.insert_one(user)
            logger.info(f"Created user: {username}")
            return True, "User created successfully"
        except DuplicateKeyError:
            logger.warning(f"Failed to create user {username}: Username or email already exists")
            return False, "Username or email already exists"
        except Exception as e:
            logger.error(f"Failed to create user {username}: {str(e)}")
            return False, f"Failed to create user: {str(e)}"
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Authenticate a user and generate a token.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple[bool, str, dict]: (success, token, user_data) or (False, None, None) if authentication fails
        """
        # Find user by username
        user = self.users_collection.find_one({'username': username})
        
        if not user:
            logger.warning(f"Authentication failed: User {username} not found")
            return False, None, None
            
        # Check if user is active
        if not user.get('active', False):
            logger.warning(f"Authentication failed: User {username} is inactive")
            return False, None, None
            
        # Verify password
        if not self.auth_manager.verify_password(user['password'], password):
            logger.warning(f"Authentication failed: Invalid password for user {username}")
            return False, None, None
            
        # Generate token
        token = self.auth_manager.generate_token(
            user_id=user['user_id'],
            username=user['username'],
            roles=user['roles']
        )
        
        # Prepare user data to return (excluding sensitive fields)
        user_data = {k: v for k, v in user.items() if k not in ['password']}
        
        logger.info(f"User {username} authenticated successfully")
        return True, token, user_data
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            dict: User document or None if not found
        """
        user = self.users_collection.find_one({'user_id': user_id})
        
        if user:
            # Remove sensitive fields
            user.pop('password', None)
            user.pop('_id', None)
            
        return user
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username.
        
        Args:
            username: Username
            
        Returns:
            dict: User document or None if not found
        """
        user = self.users_collection.find_one({'username': username})
        
        if user:
            # Remove sensitive fields
            user.pop('password', None)
            user.pop('_id', None)
            
        return user
    
    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Update user information.
        
        Args:
            user_id: User ID
            update_data: Data to update
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Make sure password gets hashed if it's being updated
        if 'password' in update_data:
            update_data['password'] = self.auth_manager.hash_password(update_data['password'])
            
        # Don't allow updating user_id
        if 'user_id' in update_data:
            del update_data['user_id']
            
        # Validate roles if they're being updated
        if 'roles' in update_data:
            for role in update_data['roles']:
                if role not in self.ROLES:
                    return False, f"Invalid role: {role}"
        
        # Add updated_at field
        update_data['updated_at'] = datetime.utcnow()
        
        try:
            result = self.users_collection.update_one(
                {'user_id': user_id},
                {'$set': update_data}
            )
            
            if result.matched_count == 0:
                logger.warning(f"Failed to update user: User ID {user_id} not found")
                return False, "User not found"
                
            logger.info(f"Updated user: {user_id}")
            return True, "User updated successfully"
        except DuplicateKeyError:
            logger.warning(f"Failed to update user {user_id}: Username or email already exists")
            return False, "Username or email already exists"
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {str(e)}")
            return False, f"Failed to update user: {str(e)}"
    
    def delete_user(self, user_id: str) -> Tuple[bool, str]:
        """Delete a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            result = self.users_collection.delete_one({'user_id': user_id})
            
            if result.deleted_count == 0:
                logger.warning(f"Failed to delete user: User ID {user_id} not found")
                return False, "User not found"
                
            logger.info(f"Deleted user: {user_id}")
            return True, "User deleted successfully"
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {str(e)}")
            return False, f"Failed to delete user: {str(e)}"
    
    def deactivate_user(self, user_id: str) -> Tuple[bool, str]:
        """Deactivate a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            result = self.users_collection.update_one(
                {'user_id': user_id},
                {'$set': {'active': False, 'updated_at': datetime.utcnow()}}
            )
            
            if result.matched_count == 0:
                logger.warning(f"Failed to deactivate user: User ID {user_id} not found")
                return False, "User not found"
                
            logger.info(f"Deactivated user: {user_id}")
            return True, "User deactivated successfully"
        except Exception as e:
            logger.error(f"Failed to deactivate user {user_id}: {str(e)}")
            return False, f"Failed to deactivate user: {str(e)}"
    
    def activate_user(self, user_id: str) -> Tuple[bool, str]:
        """Activate a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            result = self.users_collection.update_one(
                {'user_id': user_id},
                {'$set': {'active': True, 'updated_at': datetime.utcnow()}}
            )
            
            if result.matched_count == 0:
                logger.warning(f"Failed to activate user: User ID {user_id} not found")
                return False, "User not found"
                
            logger.info(f"Activated user: {user_id}")
            return True, "User activated successfully"
        except Exception as e:
            logger.error(f"Failed to activate user {user_id}: {str(e)}")
            return False, f"Failed to activate user: {str(e)}"
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if a user has a specific permission.
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            bool: True if user has permission, False otherwise
        """
        user = self.users_collection.find_one({'user_id': user_id})
        
        if not user:
            logger.warning(f"Permission check failed: User ID {user_id} not found")
            return False
            
        # Check if user is active
        if not user.get('active', False):
            logger.warning(f"Permission check failed: User {user_id} is inactive")
            return False
            
        # Check if user has admin role (admins have all permissions)
        if 'admin' in user['roles']:
            return True
            
        # Check if permission exists
        if permission not in self.PERMISSIONS:
            logger.warning(f"Permission check failed: Unknown permission '{permission}'")
            return False
            
        # Check if any of the user's roles have the permission
        return any(role in self.ROLES[permission] for role in user['roles'])
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List[str]: List of permissions
        """
        user = self.users_collection.find_one({'user_id': user_id})
        
        if not user or not user.get('active', False):
            return []
            
        # If user is admin, return all permissions
        if 'admin' in user['roles']:
            return list(self.PERMISSIONS.keys())
            
        # Otherwise return permissions based on roles
        permissions = []
        for perm, perm_data in self.PERMISSIONS.items():
            if any(role in self.ROLES[perm] for role in user['roles']):
                permissions.append(perm)
                
        return permissions
    
    def create_admin_if_missing(self) -> None:
        """Create an admin user if no users exist in the database."""
        user_count = self.users_collection.count_documents({})
        
        if user_count == 0:
            admin_username = os.getenv('ADMIN_USERNAME', 'admin')
            admin_password = os.getenv('ADMIN_PASSWORD', 'admin')
            admin_email = os.getenv('ADMIN_EMAIL', 'admin@example.com')
            
            logger.warning("No users found. Creating admin user...")
            
            if admin_username == 'admin' and admin_password == 'admin':
                logger.warning("Using default admin credentials! Please change them immediately.")
                
            self.create_user(
                username=admin_username,
                password=admin_password,
                email=admin_email,
                roles=['admin'],
                full_name='Administrator'
            )
            
            logger.info(f"Created admin user: {admin_username}")

    def close(self) -> None:
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection for user management") 