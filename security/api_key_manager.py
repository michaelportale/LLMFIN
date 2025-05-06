import os
import uuid
import time
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from pymongo import MongoClient, ASCENDING
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from security.metrics import record_api_key_usage, inc_active_api_keys, dec_active_api_keys, set_active_api_keys, record_security_event

# Configure logging
logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manager for API key generation, validation, and access control"""
    
    def __init__(self, mongo_uri: str = None, db_name: str = 'security'):
        """Initialize the API key manager
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
        """
        self.mongo_uri = mongo_uri or os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
        self.db_name = db_name
        self._connect_db()
        
    def _connect_db(self) -> None:
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            
            # Create collection if it doesn't exist
            if 'api_keys' not in self.db.list_collection_names():
                self.db.create_collection('api_keys')
                self.db.api_keys.create_index([('key_id', ASCENDING)], unique=True)
                self.db.api_keys.create_index([('key_hash', ASCENDING)], unique=True)
                self.db.api_keys.create_index([('user_id', ASCENDING)])
                
            logger.info("Connected to MongoDB for API key management")
            
            # Count active keys and update metric
            active_keys = self.db.api_keys.count_documents({'active': True})
            set_active_api_keys(active_keys)
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            record_security_event("database_connection_error", "error")
            self.client = None
            self.db = None
            
    def generate_key(self, user_id: str, name: str, scopes: List[str], 
                     expires_in_days: int = 365) -> Tuple[str, Dict]:
        """Generate a new API key
        
        Args:
            user_id: User ID the key belongs to
            name: Name/description of the key
            scopes: List of access scopes for the key
            expires_in_days: Number of days until key expires
            
        Returns:
            tuple: (api_key, key_metadata)
        """
        if not self.db:
            self._connect_db()
            if not self.db:
                raise ConnectionError("Could not connect to database")
                
        # Generate a secure random key
        key_id = uuid.uuid4().hex
        api_key = f"{key_id}.{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(api_key)
        
        expiry_date = datetime.utcnow() + timedelta(days=expires_in_days)
        
        key_doc = {
            'key_id': key_id,
            'key_hash': key_hash,
            'user_id': user_id,
            'name': name,
            'scopes': scopes,
            'created_at': datetime.utcnow(),
            'expires_at': expiry_date,
            'last_used': None,
            'active': True
        }
        
        try:
            self.db.api_keys.insert_one(key_doc.copy())
            inc_active_api_keys()
            record_security_event("api_key_created", "info")
            # Don't return the key_hash
            key_doc.pop('key_hash')
            return api_key, key_doc
        except DuplicateKeyError:
            record_security_event("api_key_creation_failed", "warning")
            logger.warning(f"Duplicate key ID generated: {key_id}")
            # Try again with a new key ID
            return self.generate_key(user_id, name, scopes, expires_in_days)
    
    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for storage
        
        Args:
            api_key: Full API key
            
        Returns:
            str: Hashed key
        """
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def validate_key(self, api_key: str, required_scope: Optional[str] = None) -> Tuple[bool, Optional[Dict]]:
        """Validate an API key and check its scope
        
        Args:
            api_key: API key to validate
            required_scope: Required scope (if any)
            
        Returns:
            tuple: (is_valid, key_metadata)
        """
        if not self.db:
            self._connect_db()
            if not self.db:
                return False, None
                
        if not api_key or '.' not in api_key:
            record_security_event("invalid_api_key_format", "warning")
            return False, None
            
        try:
            key_id, key_secret = api_key.split('.', 1)
            key_hash = self._hash_key(api_key)
            
            key_doc = self.db.api_keys.find_one({
                'key_id': key_id,
                'key_hash': key_hash,
                'active': True
            })
            
            if not key_doc:
                record_security_event("api_key_not_found", "warning")
                return False, None
                
            # Check if key has expired
            if key_doc['expires_at'] < datetime.utcnow():
                record_security_event("expired_api_key_used", "warning")
                return False, None
                
            # Check scope if required
            if required_scope and required_scope not in key_doc['scopes']:
                record_security_event("insufficient_api_key_scope", "warning")
                return False, None
                
            # Update last used timestamp
            self.db.api_keys.update_one(
                {'_id': key_doc['_id']},
                {'$set': {'last_used': datetime.utcnow()}}
            )
            
            # Record usage for metrics
            current_endpoint = required_scope or "unknown"
            record_api_key_usage(key_id, current_endpoint)
            
            # Don't return the key_hash in the response
            key_doc.pop('key_hash')
            return True, key_doc
            
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            record_security_event("api_key_validation_error", "error")
            return False, None
    
    def revoke_key(self, key_id: str, user_id: Optional[str] = None) -> bool:
        """Revoke an API key
        
        Args:
            key_id: ID of the key to revoke
            user_id: If provided, only revoke if key belongs to this user
            
        Returns:
            bool: True if key was revoked
        """
        if not self.db:
            self._connect_db()
            if not self.db:
                return False
                
        query = {'key_id': key_id}
        if user_id:
            query['user_id'] = user_id
            
        result = self.db.api_keys.update_one(
            query,
            {'$set': {'active': False}}
        )
        
        if result.modified_count > 0:
            dec_active_api_keys()
            record_security_event("api_key_revoked", "info")
            return True
            
        return False
    
    def get_user_keys(self, user_id: str) -> List[Dict]:
        """Get all API keys for a user
        
        Args:
            user_id: User ID
            
        Returns:
            list: List of key metadata (without the hashed keys)
        """
        if not self.db:
            self._connect_db()
            if not self.db:
                return []
                
        keys = list(self.db.api_keys.find({'user_id': user_id}))
        for key in keys:
            # Remove the hashed key for security
            key.pop('key_hash', None)
            
        return keys 