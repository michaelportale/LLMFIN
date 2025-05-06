import os
import json
import base64
import logging
from typing import Dict, Any, Optional, Tuple, Union
import hmac
import hashlib
from datetime import datetime
import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Cryptography for proper encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("cryptography package not available. Falling back to simpler encryption.")

# Configure logging
logger = logging.getLogger(__name__)

class SecureStorage:
    """Secure storage system for sensitive data."""
    
    # Storage categories
    CATEGORIES = {
        'credentials': 'API credentials and access tokens',
        'model_params': 'Sensitive model parameters',
        'user_data': 'Personal user information',
        'system_secrets': 'System-level secrets and keys'
    }
    
    def __init__(self, db_uri: Optional[str] = None, encryption_key: Optional[str] = None):
        """Initialize the secure storage system.
        
        Args:
            db_uri: MongoDB connection URI. If None, use env var MONGODB_URI
            encryption_key: Key for encryption/decryption. If None, use env var ENCRYPTION_KEY
        """
        self.db_uri = db_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/rl_results')
        
        # Set up encryption
        self.encryption_key = encryption_key or os.getenv('ENCRYPTION_KEY')
        if not self.encryption_key:
            # Generate a random key if not provided
            if CRYPTOGRAPHY_AVAILABLE:
                self.encryption_key = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
            else:
                self.encryption_key = os.urandom(32).hex()
            logger.warning("No encryption key provided. Generated a random key. This key will not persist across restarts.")
        
        # Initialize Fernet for encryption if available
        if CRYPTOGRAPHY_AVAILABLE:
            # Convert string key to bytes if it's a string
            if isinstance(self.encryption_key, str):
                # If the key is already base64 encoded
                try:
                    key_bytes = base64.urlsafe_b64decode(self.encryption_key.encode())
                    if len(key_bytes) != 32:
                        # Not a valid Fernet key, derive a new one
                        key_bytes = self._derive_key(self.encryption_key)
                except Exception:
                    # Not base64 encoded, derive a key
                    key_bytes = self._derive_key(self.encryption_key)
            else:
                key_bytes = self.encryption_key
                
            # Ensure the key is properly formatted for Fernet
            self.fernet_key = base64.urlsafe_b64encode(key_bytes[:32])
            self.fernet = Fernet(self.fernet_key)
        
        # Initialize DB connection
        self.client = None
        self.db = None
        self.secure_data_collection = None
        self._init_db()
        
    def _derive_key(self, password_str: str) -> bytes:
        """Derive a 32-byte key from a password string using PBKDF2.
        
        Args:
            password_str: Password string to derive key from
            
        Returns:
            bytes: 32-byte derived key
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            # Simple fallback using hashlib
            return hashlib.sha256(password_str.encode()).digest()
            
        # Use PBKDF2 with a static salt (in production, use a secure stored salt)
        salt = b'static_salt_for_key_derivation'  # In production, use a secure stored salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password_str.encode())
    
    def _init_db(self) -> None:
        """Initialize database connection and collections."""
        try:
            self.client = MongoClient(self.db_uri)
            self.client.admin.command('ping')  # Test connection
            
            # Get database name from URI
            db_name = self.db_uri.split('/')[-1]
            self.db = self.client[db_name]
            
            # Access collections
            self.secure_data_collection = self.db.secure_data
            
            # Create indexes
            self.secure_data_collection.create_index([
                ("data_id", pymongo.ASCENDING)
            ], unique=True)
            
            self.secure_data_collection.create_index([
                ("category", pymongo.ASCENDING),
                ("owner_id", pymongo.ASCENDING)
            ])
            
            logger.info(f"Connected to MongoDB for secure storage: {self.db_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def store_data(self, 
                  data: Dict[str, Any], 
                  category: str, 
                  name: str, 
                  owner_id: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Optional[str]]:
        """Store sensitive data securely.
        
        Args:
            data: Data to store securely
            category: Data category (must be in CATEGORIES)
            name: Descriptive name for the data
            owner_id: ID of the data owner
            metadata: Additional metadata about the data (not encrypted)
            
        Returns:
            Tuple[bool, str, str]: (success, message, data_id)
        """
        # Validate category
        if category not in self.CATEGORIES:
            return False, f"Invalid category: {category}", None
            
        # Encrypt the data
        encrypted_data = self._encrypt_data(data)
        
        # Create a document for storage
        data_id = hashlib.sha256(f"{name}_{owner_id}_{datetime.utcnow().isoformat()}".encode()).hexdigest()
        doc = {
            'data_id': data_id,
            'name': name,
            'category': category,
            'owner_id': owner_id,
            'encrypted_data': encrypted_data,
            'metadata': metadata or {},
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'checksum': self._compute_checksum(encrypted_data)
        }
        
        try:
            self.secure_data_collection.insert_one(doc)
            logger.info(f"Stored secure data: {name} (category: {category})")
            return True, "Data stored securely", data_id
        except DuplicateKeyError:
            logger.warning(f"Failed to store data: ID {data_id} already exists")
            return False, "Data ID already exists", None
        except Exception as e:
            logger.error(f"Failed to store data: {str(e)}")
            return False, f"Failed to store data: {str(e)}", None
    
    def retrieve_data(self, data_id: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Retrieve and decrypt sensitive data.
        
        Args:
            data_id: ID of the data to retrieve
            
        Returns:
            Tuple[bool, str, dict]: (success, message, decrypted_data)
        """
        try:
            doc = self.secure_data_collection.find_one({'data_id': data_id})
            
            if not doc:
                logger.warning(f"Failed to retrieve data: ID {data_id} not found")
                return False, "Data not found", None
                
            # Verify checksum to ensure data integrity
            if not self._verify_checksum(doc['encrypted_data'], doc['checksum']):
                logger.error(f"Data integrity check failed for ID {data_id}")
                return False, "Data integrity check failed", None
                
            # Decrypt the data
            try:
                decrypted_data = self._decrypt_data(doc['encrypted_data'])
                
                # Return a result with metadata
                result = {
                    'data': decrypted_data,
                    'name': doc['name'],
                    'category': doc['category'],
                    'owner_id': doc['owner_id'],
                    'metadata': doc['metadata'],
                    'created_at': doc['created_at'],
                    'updated_at': doc['updated_at']
                }
                
                logger.info(f"Retrieved secure data: {doc['name']} (ID: {data_id})")
                return True, "Data retrieved successfully", result
            except Exception as e:
                logger.error(f"Failed to decrypt data: {str(e)}")
                return False, f"Failed to decrypt data: {str(e)}", None
                
        except Exception as e:
            logger.error(f"Failed to retrieve data: {str(e)}")
            return False, f"Failed to retrieve data: {str(e)}", None
    
    def update_data(self, 
                   data_id: str, 
                   data: Dict[str, Any], 
                   metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Update sensitive data.
        
        Args:
            data_id: ID of the data to update
            data: New data to store
            metadata: New metadata (if None, keep existing)
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Check if data exists
            doc = self.secure_data_collection.find_one({'data_id': data_id})
            if not doc:
                logger.warning(f"Failed to update data: ID {data_id} not found")
                return False, "Data not found"
                
            # Encrypt the new data
            encrypted_data = self._encrypt_data(data)
            
            # Prepare update
            update_doc = {
                'encrypted_data': encrypted_data,
                'updated_at': datetime.utcnow(),
                'checksum': self._compute_checksum(encrypted_data)
            }
            
            # Update metadata if provided
            if metadata is not None:
                update_doc['metadata'] = metadata
                
            # Update the document
            self.secure_data_collection.update_one(
                {'data_id': data_id},
                {'$set': update_doc}
            )
            
            logger.info(f"Updated secure data: {doc['name']} (ID: {data_id})")
            return True, "Data updated successfully"
        except Exception as e:
            logger.error(f"Failed to update data: {str(e)}")
            return False, f"Failed to update data: {str(e)}"
    
    def delete_data(self, data_id: str) -> Tuple[bool, str]:
        """Delete sensitive data.
        
        Args:
            data_id: ID of the data to delete
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            result = self.secure_data_collection.delete_one({'data_id': data_id})
            
            if result.deleted_count == 0:
                logger.warning(f"Failed to delete data: ID {data_id} not found")
                return False, "Data not found"
                
            logger.info(f"Deleted secure data: {data_id}")
            return True, "Data deleted successfully"
        except Exception as e:
            logger.error(f"Failed to delete data: {str(e)}")
            return False, f"Failed to delete data: {str(e)}"
    
    def list_data(self, 
                 category: Optional[str] = None, 
                 owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List stored data entries (without the actual encrypted data).
        
        Args:
            category: Filter by category
            owner_id: Filter by owner ID
            
        Returns:
            List[Dict[str, Any]]: List of data entries
        """
        query = {}
        if category:
            query['category'] = category
        if owner_id:
            query['owner_id'] = owner_id
            
        try:
            # Exclude the encrypted data from results
            projection = {'encrypted_data': 0}
            cursor = self.secure_data_collection.find(query, projection)
            
            # Convert MongoDB ObjectId to string
            results = []
            for doc in cursor:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                results.append(doc)
                
            return results
        except Exception as e:
            logger.error(f"Failed to list data: {str(e)}")
            return []
    
    def _encrypt_data(self, data: Any) -> Dict[str, Any]:
        """Encrypt data using the configured encryption method.
        
        Args:
            data: Data to encrypt (will be serialized to JSON)
            
        Returns:
            dict: Encrypted data container
        """
        # Serialize data to JSON
        json_data = json.dumps(data)
        
        if CRYPTOGRAPHY_AVAILABLE:
            # Use Fernet symmetric encryption
            encrypted_bytes = self.fernet.encrypt(json_data.encode())
            encrypted_text = base64.b64encode(encrypted_bytes).decode('utf-8')
        else:
            # Fallback simple "encryption" (not secure, just for demo)
            # In a real implementation, require the cryptography package
            encoded = base64.b64encode(json_data.encode()).decode('utf-8')
            encrypted_text = encoded  # This is NOT actual encryption
            
        return {
            'version': '1.0',
            'method': 'fernet' if CRYPTOGRAPHY_AVAILABLE else 'base64',
            'data': encrypted_text,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _decrypt_data(self, encrypted_container: Dict[str, Any]) -> Any:
        """Decrypt data using the configured encryption method.
        
        Args:
            encrypted_container: Container with encrypted data
            
        Returns:
            Any: Decrypted data
        """
        encrypted_text = encrypted_container['data']
        method = encrypted_container['method']
        
        if method == 'fernet' and CRYPTOGRAPHY_AVAILABLE:
            # Use Fernet symmetric decryption
            encrypted_bytes = base64.b64decode(encrypted_text.encode())
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            json_data = decrypted_bytes.decode('utf-8')
        elif method == 'base64':
            # Fallback simple "decryption" (not secure, just for demo)
            json_data = base64.b64decode(encrypted_text.encode()).decode('utf-8')
        else:
            raise ValueError(f"Unsupported encryption method: {method}")
            
        # Deserialize from JSON
        return json.loads(json_data)
    
    def _compute_checksum(self, data: Dict[str, Any]) -> str:
        """Compute a checksum for data integrity verification.
        
        Args:
            data: Data to compute checksum for
            
        Returns:
            str: Checksum value
        """
        # Create a deterministic JSON representation
        serialized = json.dumps(data, sort_keys=True)
        
        # Compute HMAC using the encryption key
        if isinstance(self.encryption_key, str):
            key_bytes = self.encryption_key.encode()
        else:
            key_bytes = self.encryption_key
            
        return hmac.new(key_bytes, serialized.encode(), hashlib.sha256).hexdigest()
    
    def _verify_checksum(self, data: Dict[str, Any], checksum: str) -> bool:
        """Verify the integrity of data using its checksum.
        
        Args:
            data: Data to verify
            checksum: Expected checksum
            
        Returns:
            bool: True if checksum is valid
        """
        computed = self._compute_checksum(data)
        return hmac.compare_digest(computed, checksum)
    
    def close(self) -> None:
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection for secure storage") 