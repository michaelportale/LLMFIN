import os
import logging
import uuid
import secrets
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Configure logging
logger = logging.getLogger(__name__)

class APIKeyManager:
    """API key management system for external data sources and services."""
    
    # Define API key scopes
    SCOPES = {
        'data_source:read': 'Read access to data sources',
        'data_source:write': 'Write access to data sources',
        'model:read': 'Read access to models',
        'model:write': 'Write access to models',
        'prediction:create': 'Create predictions using models',
        'admin': 'Full administrative access'
    }
    
    def __init__(self, db_uri: Optional[str] = None):
        """Initialize the API key manager.
        
        Args:
            db_uri: MongoDB connection URI. If None, use env var MONGODB_URI
        """
        self.db_uri = db_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/rl_results')
        self.client = None
        self.db = None
        self.api_keys_collection = None
        self.data_sources_collection = None
        
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
            
            # Access collections
            self.api_keys_collection = self.db.api_keys
            self.data_sources_collection = self.db.data_sources
            
            # Create indexes
            self.api_keys_collection.create_index([("key", pymongo.ASCENDING)], unique=True)
            self.api_keys_collection.create_index([("name", pymongo.ASCENDING)], unique=True)
            self.data_sources_collection.create_index([("name", pymongo.ASCENDING)], unique=True)
            
            logger.info(f"Connected to MongoDB for API key management: {self.db_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def generate_api_key(self, 
                        name: str, 
                        owner_id: str, 
                        scopes: List[str], 
                        expires_in_days: Optional[int] = 365) -> Tuple[bool, str, Optional[str]]:
        """Generate a new API key.
        
        Args:
            name: Name/description of the API key
            owner_id: User ID of the key owner
            scopes: List of permission scopes for the key
            expires_in_days: Number of days until key expires (None for no expiry)
            
        Returns:
            Tuple[bool, str, str]: (success, message, api_key)
        """
        # Validate scopes
        for scope in scopes:
            if scope not in self.SCOPES:
                return False, f"Invalid scope: {scope}", None
                
        # Generate a new API key
        api_key = secrets.token_urlsafe(32)
        
        # Create key document
        key_doc = {
            'key_id': str(uuid.uuid4()),
            'key': api_key,
            'name': name,
            'owner_id': owner_id,
            'scopes': scopes,
            'created_at': datetime.utcnow(),
            'active': True,
            'last_used': None
        }
        
        # Set expiry date if specified
        if expires_in_days is not None:
            key_doc['expires_at'] = datetime.utcnow() + timedelta(days=expires_in_days)
        else:
            key_doc['expires_at'] = None
            
        try:
            self.api_keys_collection.insert_one(key_doc)
            logger.info(f"Generated API key: {name}")
            return True, "API key generated successfully", api_key
        except DuplicateKeyError:
            logger.warning(f"Failed to generate API key: Name {name} already exists")
            return False, "Key name already exists", None
        except Exception as e:
            logger.error(f"Failed to generate API key: {str(e)}")
            return False, f"Failed to generate API key: {str(e)}", None
    
    def validate_api_key(self, api_key: str, required_scope: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate an API key and update its last used timestamp.
        
        Args:
            api_key: API key to validate
            required_scope: Required permission scope (if any)
            
        Returns:
            Tuple[bool, dict]: (is_valid, key_data) pair
        """
        # Find key in database
        key_doc = self.api_keys_collection.find_one({'key': api_key})
        
        if not key_doc:
            logger.warning("API key validation failed: Key not found")
            return False, None
            
        # Check if key is active
        if not key_doc.get('active', False):
            logger.warning(f"API key validation failed: Key {key_doc['key_id']} is inactive")
            return False, None
            
        # Check if key has expired
        if key_doc.get('expires_at') and datetime.utcnow() > key_doc['expires_at']:
            logger.warning(f"API key validation failed: Key {key_doc['key_id']} has expired")
            return False, None
            
        # Check required scope
        if required_scope and required_scope not in key_doc['scopes'] and 'admin' not in key_doc['scopes']:
            logger.warning(f"API key validation failed: Key {key_doc['key_id']} lacks required scope '{required_scope}'")
            return False, None
            
        # Update last used timestamp
        self.api_keys_collection.update_one(
            {'key_id': key_doc['key_id']},
            {'$set': {'last_used': datetime.utcnow()}}
        )
        
        # Return success and key data (excluding sensitive fields)
        key_data = {k: v for k, v in key_doc.items() if k != 'key'}
        
        return True, key_data
    
    def revoke_api_key(self, key_id: str) -> Tuple[bool, str]:
        """Revoke an API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            result = self.api_keys_collection.update_one(
                {'key_id': key_id},
                {'$set': {'active': False}}
            )
            
            if result.matched_count == 0:
                logger.warning(f"Failed to revoke API key: Key ID {key_id} not found")
                return False, "API key not found"
                
            logger.info(f"Revoked API key: {key_id}")
            return True, "API key revoked successfully"
        except Exception as e:
            logger.error(f"Failed to revoke API key {key_id}: {str(e)}")
            return False, f"Failed to revoke API key: {str(e)}"
    
    def get_api_keys(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all API keys, optionally filtered by owner.
        
        Args:
            owner_id: Filter by owner's user ID
            
        Returns:
            List[Dict[str, Any]]: List of API key documents
        """
        query = {}
        if owner_id:
            query['owner_id'] = owner_id
            
        keys = list(self.api_keys_collection.find(query, {'key': 0}))  # Exclude the key itself
        
        # Convert MongoDB ObjectId to string
        for key in keys:
            if '_id' in key:
                key['_id'] = str(key['_id'])
                
        return keys
    
    # Data source management methods
    
    def register_data_source(self, 
                           name: str, 
                           source_type: str, 
                           config: Dict[str, Any], 
                           owner_id: str) -> Tuple[bool, str, Optional[str]]:
        """Register a new data source with encrypted credentials.
        
        Args:
            name: Data source name
            source_type: Type of data source (e.g., 'database', 'api', 'file')
            config: Configuration including credentials (will be encrypted)
            owner_id: User ID of the data source owner
            
        Returns:
            Tuple[bool, str, str]: (success, message, data_source_id)
        """
        # Create data source document
        data_source = {
            'source_id': str(uuid.uuid4()),
            'name': name,
            'source_type': source_type,
            'config': self._encrypt_sensitive_data(config),
            'owner_id': owner_id,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'active': True
        }
        
        try:
            self.data_sources_collection.insert_one(data_source)
            logger.info(f"Registered data source: {name}")
            return True, "Data source registered successfully", data_source['source_id']
        except DuplicateKeyError:
            logger.warning(f"Failed to register data source: Name {name} already exists")
            return False, "Data source name already exists", None
        except Exception as e:
            logger.error(f"Failed to register data source: {str(e)}")
            return False, f"Failed to register data source: {str(e)}", None
    
    def get_data_source(self, source_id: str, decrypt: bool = False) -> Optional[Dict[str, Any]]:
        """Get a data source by ID.
        
        Args:
            source_id: Data source ID
            decrypt: Whether to decrypt sensitive data
            
        Returns:
            dict: Data source document or None if not found
        """
        data_source = self.data_sources_collection.find_one({'source_id': source_id})
        
        if data_source:
            # Remove MongoDB ObjectId
            if '_id' in data_source:
                data_source['_id'] = str(data_source['_id'])
                
            # Decrypt config if requested
            if decrypt and 'config' in data_source:
                data_source['config'] = self._decrypt_sensitive_data(data_source['config'])
                
        return data_source
    
    def get_data_sources(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all data sources, optionally filtered by owner.
        
        Args:
            owner_id: Filter by owner's user ID
            
        Returns:
            List[Dict[str, Any]]: List of data source documents
        """
        query = {}
        if owner_id:
            query['owner_id'] = owner_id
            
        data_sources = list(self.data_sources_collection.find(query))
        
        # Convert MongoDB ObjectId to string and don't return decrypted configs
        for source in data_sources:
            if '_id' in source:
                source['_id'] = str(source['_id'])
            
            # Replace config with placeholder to avoid returning encrypted data
            if 'config' in source:
                source['config'] = {'encrypted': True}
                
        return data_sources
    
    def update_data_source(self, source_id: str, update_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Update a data source.
        
        Args:
            source_id: Data source ID
            update_data: Data to update
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Don't allow updating source_id
        if 'source_id' in update_data:
            del update_data['source_id']
            
        # Encrypt config if it's being updated
        if 'config' in update_data:
            update_data['config'] = self._encrypt_sensitive_data(update_data['config'])
            
        # Add updated_at field
        update_data['updated_at'] = datetime.utcnow()
        
        try:
            result = self.data_sources_collection.update_one(
                {'source_id': source_id},
                {'$set': update_data}
            )
            
            if result.matched_count == 0:
                logger.warning(f"Failed to update data source: ID {source_id} not found")
                return False, "Data source not found"
                
            logger.info(f"Updated data source: {source_id}")
            return True, "Data source updated successfully"
        except DuplicateKeyError:
            logger.warning(f"Failed to update data source {source_id}: Name already exists")
            return False, "Data source name already exists"
        except Exception as e:
            logger.error(f"Failed to update data source {source_id}: {str(e)}")
            return False, f"Failed to update data source: {str(e)}"
    
    def delete_data_source(self, source_id: str) -> Tuple[bool, str]:
        """Delete a data source.
        
        Args:
            source_id: Data source ID
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            result = self.data_sources_collection.delete_one({'source_id': source_id})
            
            if result.deleted_count == 0:
                logger.warning(f"Failed to delete data source: ID {source_id} not found")
                return False, "Data source not found"
                
            logger.info(f"Deleted data source: {source_id}")
            return True, "Data source deleted successfully"
        except Exception as e:
            logger.error(f"Failed to delete data source {source_id}: {str(e)}")
            return False, f"Failed to delete data source: {str(e)}"
    
    def _encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive data in a configuration.
        
        In a production environment, this would use a secure encryption method.
        For simplicity in this implementation, we'll just mark the data as encrypted.
        
        Args:
            data: Configuration data to encrypt
            
        Returns:
            dict: Encrypted configuration data
        """
        # IMPORTANT: In a real implementation, use a proper encryption method
        # such as Fernet encryption from cryptography library
        
        # This is a placeholder that just marks data as encrypted
        encrypted_data = {
            '_encrypted': True,
            '_encrypted_time': datetime.utcnow().isoformat(),
            'data': data  # In a real implementation, this would be encrypted
        }
        
        return encrypted_data
    
    def _decrypt_sensitive_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive data from a configuration.
        
        In a production environment, this would use a secure decryption method.
        For simplicity in this implementation, we'll just return the original data.
        
        Args:
            encrypted_data: Encrypted configuration data
            
        Returns:
            dict: Decrypted configuration data
        """
        # IMPORTANT: In a real implementation, use a proper decryption method
        # corresponding to the encryption method used
        
        # This is a placeholder that just returns the original data
        if isinstance(encrypted_data, dict) and encrypted_data.get('_encrypted', False):
            return encrypted_data.get('data', {})
        
        return encrypted_data
    
    def close(self) -> None:
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection for API key management") 