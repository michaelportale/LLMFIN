from flask import Blueprint, request, jsonify, current_app, g
import logging
from typing import Dict, Any, Optional, Tuple
import traceback

# Import security components
from security.auth import AuthManager
from security.user_management import UserManager
from security.api_keys import APIKeyManager
from security.middleware import USER_CREATE_SCHEMA, LOGIN_SCHEMA, API_KEY_CREATE_SCHEMA

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
security_bp = Blueprint('security', __name__)

# Helper function to get security middleware instance within application context
def get_security():
    return current_app.security

@security_bp.route('/register', methods=['POST'])
def register_user():
    """Register a new user.
    
    Required JSON body:
    {
        "username": "string",
        "email": "string",
        "password": "string",
        "first_name": "string" (optional),
        "last_name": "string" (optional),
        "role": "string" (optional)
    }
    """
    try:
        # Validate request against schema using middleware
        schema_errors = get_security()._validate_against_schema(request.json, USER_CREATE_SCHEMA)
        if schema_errors:
            return jsonify({
                'error': 'Invalid request data',
                'message': 'The provided data does not match the required format',
                'details': schema_errors
            }), 400
            
        # Extract data
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        first_name = data.get('first_name', '')
        last_name = data.get('last_name', '')
        role = data.get('role', 'guest')
        
        # Create user through the user manager
        user_manager = get_security().user_manager
        success, message, user_id = user_manager.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            role=role
        )
        
        if not success:
            return jsonify({
                'error': 'Registration failed',
                'message': message
            }), 400
            
        # Return success
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user_id': user_id
        }), 201
        
    except Exception as e:
        logger.error(f"Error during user registration: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred during registration'
        }), 500

@security_bp.route('/login', methods=['POST'])
def login():
    """Login a user and get access token.
    
    Required JSON body:
    {
        "username": "string",
        "password": "string"
    }
    """
    try:
        # Validate request against schema
        schema_errors = get_security()._validate_against_schema(request.json, LOGIN_SCHEMA)
        if schema_errors:
            return jsonify({
                'error': 'Invalid request data',
                'message': 'The provided data does not match the required format',
                'details': schema_errors
            }), 400
            
        # Extract credentials
        username = request.json.get('username')
        password = request.json.get('password')
        
        # Authenticate user
        user_manager = get_security().user_manager
        success, message, user_data = user_manager.authenticate_user(username, password)
        
        if not success:
            return jsonify({
                'error': 'Authentication failed',
                'message': message
            }), 401
            
        # Generate token
        auth_manager = get_security().auth_manager
        token_data = {
            'sub': user_data['user_id'],
            'username': user_data['username'],
            'email': user_data['email'],
            'roles': user_data['roles']
        }
        token = auth_manager.generate_token(token_data)
        
        # Return token
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'token': token,
            'user': {
                'user_id': user_data['user_id'],
                'username': user_data['username'],
                'email': user_data['email'],
                'roles': user_data['roles'],
                'first_name': user_data.get('first_name', ''),
                'last_name': user_data.get('last_name', '')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred during login'
        }), 500

@security_bp.route('/api-keys', methods=['POST'])
@get_security().login_required
def create_api_key():
    """Create a new API key for the current user.
    
    Required JSON body:
    {
        "name": "string",
        "scopes": ["string"],
        "expires_in_days": number (optional)
    }
    """
    try:
        # Validate request against schema
        schema_errors = get_security()._validate_against_schema(request.json, API_KEY_CREATE_SCHEMA)
        if schema_errors:
            return jsonify({
                'error': 'Invalid request data',
                'message': 'The provided data does not match the required format',
                'details': schema_errors
            }), 400
            
        # Extract data
        data = request.json
        name = data.get('name')
        scopes = data.get('scopes', [])
        expires_in_days = data.get('expires_in_days', 365)
        
        # Get user ID from token
        user_id = g.user_id
        
        # Create API key
        api_key_manager = get_security().api_key_manager
        success, message, api_key = api_key_manager.generate_api_key(
            name=name,
            owner_id=user_id,
            scopes=scopes,
            expires_in_days=expires_in_days
        )
        
        if not success:
            return jsonify({
                'error': 'API key creation failed',
                'message': message
            }), 400
            
        # Return the API key (only shown once)
        return jsonify({
            'success': True,
            'message': 'API key created successfully',
            'api_key': api_key,
            'name': name,
            'scopes': scopes,
            'expires_in_days': expires_in_days
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating API key: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while creating API key'
        }), 500

@security_bp.route('/api-keys', methods=['GET'])
@get_security().login_required
def list_api_keys():
    """List all API keys for the current user."""
    try:
        # Get user ID from token
        user_id = g.user_id
        
        # Get API keys
        api_key_manager = get_security().api_key_manager
        keys = api_key_manager.get_api_keys(owner_id=user_id)
        
        # Return API keys (without the actual key values)
        return jsonify({
            'success': True,
            'api_keys': keys
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing API keys: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while listing API keys'
        }), 500

@security_bp.route('/api-keys/<key_id>', methods=['DELETE'])
@get_security().login_required
def revoke_api_key(key_id):
    """Revoke an API key."""
    try:
        # Get user ID from token
        user_id = g.user_id
        
        # Get API key manager
        api_key_manager = get_security().api_key_manager
        
        # Verify ownership before revoking
        keys = api_key_manager.get_api_keys(owner_id=user_id)
        is_owner = any(key.get('key_id') == key_id for key in keys)
        
        # Allow admins to revoke any key
        is_admin = 'admin' in g.current_user.get('roles', [])
        
        if not is_owner and not is_admin:
            return jsonify({
                'error': 'Forbidden',
                'message': 'You do not own this API key'
            }), 403
            
        # Revoke the API key
        success, message = api_key_manager.revoke_api_key(key_id)
        
        if not success:
            return jsonify({
                'error': 'Revocation failed',
                'message': message
            }), 400
            
        return jsonify({
            'success': True,
            'message': 'API key revoked successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error revoking API key: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while revoking API key'
        }), 500

@security_bp.route('/data-sources', methods=['POST'])
@get_security().login_required
def register_data_source():
    """Register a new data source.
    
    Required JSON body:
    {
        "name": "string",
        "source_type": "string",
        "config": { ... }
    }
    """
    try:
        # Extract data
        data = request.json
        name = data.get('name')
        source_type = data.get('source_type')
        config = data.get('config', {})
        
        # Validate required fields
        if not name or not source_type:
            return jsonify({
                'error': 'Invalid request data',
                'message': 'Name and source_type are required'
            }), 400
            
        # Get user ID from token
        user_id = g.user_id
        
        # Register data source
        api_key_manager = get_security().api_key_manager
        success, message, source_id = api_key_manager.register_data_source(
            name=name,
            source_type=source_type,
            config=config,
            owner_id=user_id
        )
        
        if not success:
            return jsonify({
                'error': 'Data source registration failed',
                'message': message
            }), 400
            
        # Return success
        return jsonify({
            'success': True,
            'message': 'Data source registered successfully',
            'source_id': source_id
        }), 201
        
    except Exception as e:
        logger.error(f"Error registering data source: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while registering data source'
        }), 500

@security_bp.route('/data-sources', methods=['GET'])
@get_security().login_required
def list_data_sources():
    """List all data sources for the current user."""
    try:
        # Get user ID from token
        user_id = g.user_id
        
        # Check if admin wants to see all sources
        show_all = (
            request.args.get('all') == 'true' and 
            'admin' in g.current_user.get('roles', [])
        )
        
        # Get data sources
        api_key_manager = get_security().api_key_manager
        if show_all:
            sources = api_key_manager.get_data_sources()
        else:
            sources = api_key_manager.get_data_sources(owner_id=user_id)
        
        # Return data sources
        return jsonify({
            'success': True,
            'data_sources': sources
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing data sources: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while listing data sources'
        }), 500

@security_bp.route('/data-sources/<source_id>', methods=['GET'])
@get_security().login_required
def get_data_source(source_id):
    """Get a data source by ID."""
    try:
        # Get user ID from token
        user_id = g.user_id
        
        # Get API key manager
        api_key_manager = get_security().api_key_manager
        
        # Get data source
        data_source = api_key_manager.get_data_source(source_id)
        
        if not data_source:
            return jsonify({
                'error': 'Not found',
                'message': 'Data source not found'
            }), 404
            
        # Check ownership or admin status
        is_owner = data_source.get('owner_id') == user_id
        is_admin = 'admin' in g.current_user.get('roles', [])
        
        if not is_owner and not is_admin:
            return jsonify({
                'error': 'Forbidden',
                'message': 'You do not have permission to access this data source'
            }), 403
            
        # Return data source without sensitive config
        if 'config' in data_source:
            data_source['config'] = {'encrypted': True}
            
        return jsonify({
            'success': True,
            'data_source': data_source
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting data source: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while getting data source'
        }), 500

@security_bp.route('/data-sources/<source_id>', methods=['PUT'])
@get_security().login_required
def update_data_source(source_id):
    """Update a data source."""
    try:
        # Get user ID from token
        user_id = g.user_id
        
        # Get API key manager
        api_key_manager = get_security().api_key_manager
        
        # Get data source to check ownership
        data_source = api_key_manager.get_data_source(source_id)
        
        if not data_source:
            return jsonify({
                'error': 'Not found',
                'message': 'Data source not found'
            }), 404
            
        # Check ownership or admin status
        is_owner = data_source.get('owner_id') == user_id
        is_admin = 'admin' in g.current_user.get('roles', [])
        
        if not is_owner and not is_admin:
            return jsonify({
                'error': 'Forbidden',
                'message': 'You do not have permission to update this data source'
            }), 403
            
        # Extract update data
        update_data = request.json
        
        # Update data source
        success, message = api_key_manager.update_data_source(source_id, update_data)
        
        if not success:
            return jsonify({
                'error': 'Update failed',
                'message': message
            }), 400
            
        return jsonify({
            'success': True,
            'message': 'Data source updated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating data source: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while updating data source'
        }), 500

@security_bp.route('/data-sources/<source_id>', methods=['DELETE'])
@get_security().login_required
def delete_data_source(source_id):
    """Delete a data source."""
    try:
        # Get user ID from token
        user_id = g.user_id
        
        # Get API key manager
        api_key_manager = get_security().api_key_manager
        
        # Get data source to check ownership
        data_source = api_key_manager.get_data_source(source_id)
        
        if not data_source:
            return jsonify({
                'error': 'Not found',
                'message': 'Data source not found'
            }), 404
            
        # Check ownership or admin status
        is_owner = data_source.get('owner_id') == user_id
        is_admin = 'admin' in g.current_user.get('roles', [])
        
        if not is_owner and not is_admin:
            return jsonify({
                'error': 'Forbidden',
                'message': 'You do not have permission to delete this data source'
            }), 403
            
        # Delete data source
        success, message = api_key_manager.delete_data_source(source_id)
        
        if not success:
            return jsonify({
                'error': 'Deletion failed',
                'message': message
            }), 400
            
        return jsonify({
            'success': True,
            'message': 'Data source deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting data source: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while deleting data source'
        }), 500

@security_bp.route('/users/me', methods=['GET'])
@get_security().login_required
def get_current_user():
    """Get current user information."""
    try:
        # Get user ID from token
        user_id = g.user_id
        
        # Get user manager
        user_manager = get_security().user_manager
        
        # Get user data
        success, message, user_data = user_manager.get_user(user_id)
        
        if not success:
            return jsonify({
                'error': 'User not found',
                'message': message
            }), 404
            
        # Remove sensitive data
        if 'password_hash' in user_data:
            del user_data['password_hash']
            
        return jsonify({
            'success': True,
            'user': user_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting current user: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while getting user information'
        }), 500

@security_bp.route('/users/me', methods=['PUT'])
@get_security().login_required
def update_current_user():
    """Update current user information.
    
    Allowed fields to update:
    {
        "first_name": "string",
        "last_name": "string",
        "email": "string"
    }
    """
    try:
        # Get user ID from token
        user_id = g.user_id
        
        # Extract update data
        data = request.json
        
        # Only allow specific fields to be updated
        allowed_fields = ['first_name', 'last_name', 'email']
        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        if not update_data:
            return jsonify({
                'error': 'Invalid request data',
                'message': 'No valid fields to update'
            }), 400
            
        # Get user manager
        user_manager = get_security().user_manager
        
        # Update user
        success, message = user_manager.update_user(user_id, update_data)
        
        if not success:
            return jsonify({
                'error': 'Update failed',
                'message': message
            }), 400
            
        return jsonify({
            'success': True,
            'message': 'User updated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while updating user information'
        }), 500

@security_bp.route('/users/me/password', methods=['PUT'])
@get_security().login_required
def change_password():
    """Change current user's password.
    
    Required JSON body:
    {
        "current_password": "string",
        "new_password": "string"
    }
    """
    try:
        # Get user ID from token
        user_id = g.user_id
        
        # Extract data
        data = request.json
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({
                'error': 'Invalid request data',
                'message': 'Current password and new password are required'
            }), 400
            
        # Get user manager
        user_manager = get_security().user_manager
        
        # Verify current password and change to new password
        success, message = user_manager.change_password(user_id, current_password, new_password)
        
        if not success:
            return jsonify({
                'error': 'Password change failed',
                'message': message
            }), 400
            
        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while changing password'
        }), 500

# Admin-only routes
@security_bp.route('/admin/users', methods=['GET'])
@get_security().role_required('admin')
def list_users():
    """List all users (admin only)."""
    try:
        # Get user manager
        user_manager = get_security().user_manager
        
        # Get all users
        users = user_manager.get_all_users()
        
        # Remove sensitive data
        for user in users:
            if 'password_hash' in user:
                del user['password_hash']
                
        return jsonify({
            'success': True,
            'users': users
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while listing users'
        }), 500

@security_bp.route('/admin/users/<user_id>/roles', methods=['PUT'])
@get_security().role_required('admin')
def update_user_roles(user_id):
    """Update a user's roles (admin only).
    
    Required JSON body:
    {
        "roles": ["string"]
    }
    """
    try:
        # Extract data
        data = request.json
        roles = data.get('roles', [])
        
        if not roles:
            return jsonify({
                'error': 'Invalid request data',
                'message': 'Roles are required'
            }), 400
            
        # Get user manager
        user_manager = get_security().user_manager
        
        # Update roles
        success, message = user_manager.update_user(user_id, {'roles': roles})
        
        if not success:
            return jsonify({
                'error': 'Update failed',
                'message': message
            }), 400
            
        return jsonify({
            'success': True,
            'message': 'User roles updated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating user roles: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while updating user roles'
        }), 500

@security_bp.route('/admin/users/<user_id>/activate', methods=['PUT'])
@get_security().role_required('admin')
def activate_user(user_id):
    """Activate a user account (admin only)."""
    try:
        # Get user manager
        user_manager = get_security().user_manager
        
        # Activate user
        success, message = user_manager.activate_user(user_id)
        
        if not success:
            return jsonify({
                'error': 'Activation failed',
                'message': message
            }), 400
            
        return jsonify({
            'success': True,
            'message': 'User activated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error activating user: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while activating user'
        }), 500

@security_bp.route('/admin/users/<user_id>/deactivate', methods=['PUT'])
@get_security().role_required('admin')
def deactivate_user(user_id):
    """Deactivate a user account (admin only)."""
    try:
        # Get user manager
        user_manager = get_security().user_manager
        
        # Deactivate user
        success, message = user_manager.deactivate_user(user_id)
        
        if not success:
            return jsonify({
                'error': 'Deactivation failed',
                'message': message
            }), 400
            
        return jsonify({
            'success': True,
            'message': 'User deactivated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deactivating user: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Server error',
            'message': 'An error occurred while deactivating user'
        }), 500 