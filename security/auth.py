import os
import jwt
import time
import uuid
import hashlib
import logging
from typing import Dict, Optional, Any, List, Union, Tuple
from datetime import datetime, timedelta
from functools import wraps
import secrets
import base64
from flask import request, jsonify, g, current_app

from security.metrics import record_auth_attempt, record_security_event

# Configure logging
logger = logging.getLogger(__name__)

class AuthManager:
    """Authentication and authorization manager for the RL application."""
    
    def __init__(self, secret_key: Optional[str] = None, token_expiry: int = 86400):
        """Initialize the authentication manager.
        
        Args:
            secret_key: Secret key for JWT token encryption. If None, use env var JWT_SECRET_KEY
            token_expiry: Token expiry time in seconds (default: 24 hours)
        """
        self.secret_key = secret_key or os.environ.get('JWT_SECRET_KEY') or self._generate_secret()
        self.token_expiry = token_expiry
        logger.info("AuthManager initialized")
        
    def _generate_secret(self):
        """Generate a random secret key"""
        random_secret = uuid.uuid4().hex
        logger.warning("No secret key provided. Generated a random one - this will invalidate existing tokens if server restarts")
        record_security_event("secret_key_generated", "warning")
        return random_secret
        
    def hash_password(self, password: str) -> Tuple[str, str]:
        """Hash a password using SHA-256 with a salt
        
        Args:
            password: Plain text password
            
        Returns:
            tuple: (salt, hash)
        """
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return salt.hex(), key.hex()
        
    def verify_password(self, password: str, salt_hex: str, key_hex: str) -> bool:
        """Verify a password against a hash
        
        Args:
            password: Plain text password
            salt_hex: Salt as a hex string
            key_hex: Key as a hex string
            
        Returns:
            bool: True if password matches, False otherwise
        """
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        is_valid = key.hex() == key_hex
        record_auth_attempt(is_valid)
        return is_valid
        
    def generate_token(self, user_id: str, username: str, role: str, additional_data: Optional[Dict[str, Any]] = None) -> str:
        """Generate a JWT token
        
        Args:
            user_id: User ID
            username: Username
            role: User role
            additional_data: Additional data to include in the token
            
        Returns:
            str: JWT token
        """
        now = datetime.utcnow()
        payload = {
            'sub': user_id,
            'username': username,
            'role': role,
            'iat': now,
            'exp': now + timedelta(seconds=self.token_expiry)
        }
        
        if additional_data:
            payload.update(additional_data)
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        record_auth_attempt(True, 'token_generation')
        record_security_event("token_generated", "info")
        return token
        
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT token
        
        Args:
            token: JWT token
            
        Returns:
            dict: Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            record_security_event("token_verified", "info")
            return payload
        except jwt.ExpiredSignatureError:
            record_auth_attempt(False, 'token_expired')
            record_security_event("token_expired", "warning")
            logger.warning("Expired token attempted")
            return None
        except jwt.InvalidTokenError:
            record_auth_attempt(False, 'token_invalid')
            record_security_event("invalid_token", "warning")
            logger.warning("Invalid token attempted")
            return None
            
    def refresh_token(self, token: str, refresh_threshold: int = 3600) -> Tuple[Optional[str], bool]:
        """Refresh a token if it's close to expiry
        
        Args:
            token: JWT token
            refresh_threshold: Threshold in seconds to refresh (default: 1 hour)
            
        Returns:
            str: New token if refreshed, original token otherwise
            bool: True if refreshed, False otherwise
        """
        payload = self.verify_token(token)
        if not payload:
            return None, False
            
        now = datetime.utcnow()
        exp = datetime.fromtimestamp(payload['exp'])
        
        # If token is about to expire within threshold, refresh it
        if (exp - now).total_seconds() < refresh_threshold:
            new_token = self.generate_token(
                payload['sub'],
                payload['username'],
                payload['role'],
                {k: v for k, v in payload.items() if k not in ['sub', 'username', 'role', 'exp', 'iat']}
            )
            record_security_event("token_refreshed", "info")
            return new_token, True
        
        return token, False


# Flask authentication helpers

def login_required(f):
    """Decorator for endpoints that require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in the Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            record_auth_attempt(False, 'no_token')
            record_security_event("access_denied_no_token", "warning")
            return jsonify({'message': 'Authentication token is missing'}), 401
            
        auth_manager = current_app.config.get('AUTH_MANAGER')
        if not auth_manager:
            return jsonify({'message': 'Authentication manager not configured'}), 500
            
        payload = auth_manager.verify_token(token)
        
        if not payload:
            record_security_event("access_denied_invalid_token", "warning")
            return jsonify({'message': 'Invalid authentication token'}), 401
            
        # Store user info in Flask's g object
        g.user = {
            'user_id': payload['sub'],
            'username': payload['username'],
            'role': payload['role']
        }
        
        return f(*args, **kwargs)
        
    return decorated


def role_required(roles: Union[str, List[str]]):
    """Decorator for endpoints that require specific roles.
    
    Args:
        roles: Role or list of roles required to access the endpoint
    """
    if isinstance(roles, str):
        roles = [roles]
        
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # First check if user is authenticated
            if not hasattr(g, 'user'):
                return jsonify({'message': 'Authentication required'}), 401
                
            # Check if user has required role
            user_role = g.user.get('role')
            
            if not user_role or user_role not in roles:
                record_security_event("access_denied_unauthorized_role", "warning")
                return jsonify({'message': 'Insufficient permissions'}), 403
                
            return f(*args, **kwargs)
            
        return decorated
        
    return decorator 