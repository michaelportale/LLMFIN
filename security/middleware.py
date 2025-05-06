import os
import functools
import logging
import time
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from flask import Flask, request, jsonify, g, Response, current_app
import json
import traceback
import ipaddress
from datetime import datetime

# Import our security components
from security.auth import AuthManager
from security.user_management import UserManager
from security.api_keys import APIKeyManager

# Configure logging
logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Middleware for securing Flask API endpoints.
    
    This middleware provides authentication, authorization, rate limiting,
    and request validation for Flask applications.
    """
    
    def __init__(self, app: Optional[Flask] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the security middleware.
        
        Args:
            app: Flask application instance (optional)
            config: Configuration dictionary (optional)
        """
        self.app = app
        self.config = config or {}
        
        # Initialize security components
        self.auth_manager = None
        self.user_manager = None
        self.api_key_manager = None
        
        # Rate limiting storage
        self.rate_limit_data = {}
        
        # Initialize components immediately if app is provided
        if app is not None:
            self.init_app(app, config)
    
    def init_app(self, app: Flask, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the middleware with a Flask application.
        
        Args:
            app: Flask application instance
            config: Configuration dictionary (optional)
        """
        if config:
            self.config.update(config)
            
        # Set default configurations if not provided
        self._set_default_config(app)
        
        # Use provided security components if available in config
        if isinstance(config, dict):
            self.auth_manager = config.get('auth_manager', self.auth_manager)
            self.user_manager = config.get('user_manager', self.user_manager)
            self.api_key_manager = config.get('api_key_manager', self.api_key_manager)
        
        # Initialize auth manager if not provided
        if self.auth_manager is None:
            secret_key = app.config.get('JWT_SECRET_KEY') or app.config.get('SECRET_KEY')
            self.auth_manager = AuthManager(
                secret_key=secret_key,
                token_expiry=app.config.get('JWT_TOKEN_EXPIRY', 3600)
            )
        
        # Initialize user manager if not provided
        if self.user_manager is None:
            db_uri = app.config.get('MONGODB_URI')
            self.user_manager = UserManager(auth_manager=self.auth_manager, db_uri=db_uri)
        
        # Initialize API key manager if not provided
        if self.api_key_manager is None:
            db_uri = app.config.get('MONGODB_URI')
            self.api_key_manager = APIKeyManager(db_uri=db_uri)
        
        # Register error handlers
        app.errorhandler(401)(self._handle_unauthorized_error)
        app.errorhandler(403)(self._handle_forbidden_error)
        app.errorhandler(429)(self._handle_rate_limit_error)
        
        # Register before request handlers
        app.before_request(self._before_request)
        
        # Register middleware components
        if app.config.get('SECURITY_ENABLE_CORS', True):
            self._configure_cors(app)
            
        if app.config.get('SECURITY_ENABLE_LOGGING', True):
            app.after_request(self._log_request)
            
        # Store middleware in app context
        app.security = self
        
        logger.info("Security middleware initialized")
    
    def _set_default_config(self, app: Flask) -> None:
        """Set default security configuration.
        
        Args:
            app: Flask application instance
        """
        # JWT Settings
        app.config.setdefault('JWT_TOKEN_EXPIRY', 3600)  # 1 hour
        app.config.setdefault('JWT_SECRET_KEY', app.config.get('SECRET_KEY', os.urandom(32).hex()))
        
        # Rate limiting settings
        app.config.setdefault('RATE_LIMIT_DEFAULT', 100)  # per minute
        app.config.setdefault('RATE_LIMIT_WINDOW', 60)  # seconds
        
        # IP Allow/Deny lists
        app.config.setdefault('IP_ALLOWLIST', [])
        app.config.setdefault('IP_DENYLIST', [])
        
        # Security features
        app.config.setdefault('SECURITY_ENABLE_CORS', True)
        app.config.setdefault('SECURITY_ENABLE_LOGGING', True)
        app.config.setdefault('SECURITY_ENABLE_BRUTE_FORCE_PROTECTION', True)
        
        # Request validation settings
        app.config.setdefault('MAX_CONTENT_LENGTH', 10 * 1024 * 1024)  # 10MB
        app.config.setdefault('SECURITY_ALLOWED_METHODS', ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
    
    def _before_request(self) -> Optional[Response]:
        """Handle pre-request security checks.
        
        Returns:
            Optional[Response]: Error response or None to continue
        """
        # Skip security checks for OPTIONS requests (CORS preflight)
        if request.method == 'OPTIONS':
            return None
            
        # Set start time for request duration tracking
        g.start_time = time.time()
        
        # 1. Check IP restrictions
        ip_check_result = self._check_ip_restrictions()
        if ip_check_result:
            return ip_check_result
            
        # 2. Check rate limits
        rate_limit_result = self._check_rate_limits()
        if rate_limit_result:
            return rate_limit_result
            
        # 3. Method validation
        if request.method not in current_app.config.get('SECURITY_ALLOWED_METHODS', ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']):
            return jsonify({
                'error': 'Method not allowed',
                'message': f"Method {request.method} is not allowed"
            }), 405
        
        # Set default authentication information
        g.current_user = None
        g.current_api_key = None
        
        # 4. Extract authentication (if present)
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            # Try JWT authentication (Bearer token)
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
                valid, payload = self.auth_manager.verify_token(token)
                
                if valid and payload:
                    # Store user info in g for route handlers
                    g.current_user = payload
                    g.user_id = payload.get('sub')
                    logger.debug(f"Authenticated user: {g.user_id}")
            
            # Try API key authentication
            elif auth_header.startswith('ApiKey '):
                api_key = auth_header.split(' ')[1]
                
                # Get required scope from endpoint if it's registered
                endpoint_func = current_app.view_functions.get(request.endpoint)
                required_scope = getattr(endpoint_func, '_required_scope', None)
                
                valid, key_data = self.api_key_manager.validate_api_key(api_key, required_scope)
                
                if valid and key_data:
                    # Store API key info in g for route handlers
                    g.current_api_key = key_data
                    g.api_key_id = key_data.get('key_id')
                    logger.debug(f"Authenticated with API key: {g.api_key_id}")
        
        return None
    
    def _check_ip_restrictions(self) -> Optional[Response]:
        """Check if the client IP is allowed.
        
        Returns:
            Optional[Response]: Error response or None to continue
        """
        client_ip = request.remote_addr
        
        # Check deny list first
        deny_list = current_app.config.get('IP_DENYLIST', [])
        for banned_ip in deny_list:
            if self._is_ip_in_range(client_ip, banned_ip):
                logger.warning(f"Blocked request from denied IP: {client_ip}")
                return jsonify({
                    'error': 'Access denied',
                    'message': 'Your IP address is not allowed to access this resource'
                }), 403
        
        # Then check allow list if it's not empty
        allow_list = current_app.config.get('IP_ALLOWLIST', [])
        if allow_list:
            # If allow list exists, IP must be in it
            is_allowed = any(self._is_ip_in_range(client_ip, allowed_ip) for allowed_ip in allow_list)
            
            if not is_allowed:
                logger.warning(f"Blocked request from non-allowed IP: {client_ip}")
                return jsonify({
                    'error': 'Access denied', 
                    'message': 'Your IP address is not allowed to access this resource'
                }), 403
                
        return None
    
    def _is_ip_in_range(self, ip: str, ip_range: str) -> bool:
        """Check if an IP is within a range.
        
        Args:
            ip: IP address to check
            ip_range: IP range (single IP or CIDR notation)
            
        Returns:
            bool: True if IP is in range
        """
        try:
            if '/' in ip_range:
                # CIDR notation
                return ipaddress.ip_address(ip) in ipaddress.ip_network(ip_range)
            else:
                # Single IP
                return ip == ip_range
        except Exception as e:
            logger.error(f"Error checking IP range: {str(e)}")
            return False
    
    def _check_rate_limits(self) -> Optional[Response]:
        """Check if the request exceeds rate limits.
        
        Returns:
            Optional[Response]: Error response or None to continue
        """
        # Get rate limit settings
        default_limit = current_app.config.get('RATE_LIMIT_DEFAULT', 100)
        window_size = current_app.config.get('RATE_LIMIT_WINDOW', 60)
        
        # Determine rate limit key (user_id > API key > IP address)
        if hasattr(g, 'user_id') and g.user_id:
            rate_key = f"user:{g.user_id}"
            # Users might have custom rate limits
            limit = getattr(g, 'current_user', {}).get('rate_limit', default_limit)
        elif hasattr(g, 'api_key_id') and g.api_key_id:
            rate_key = f"api_key:{g.api_key_id}"
            limit = default_limit
        else:
            # Unauthenticated requests use IP-based limiting
            rate_key = f"ip:{request.remote_addr}"
            limit = default_limit // 2  # Lower limit for anonymous requests
            
        # Get endpoint-specific limit if defined
        endpoint = request.endpoint
        if endpoint:
            endpoint_func = current_app.view_functions.get(endpoint)
            endpoint_limit = getattr(endpoint_func, '_rate_limit', None)
            if endpoint_limit is not None:
                limit = endpoint_limit
                
        # Check rate limit
        now = time.time()
        window_start = now - window_size
        
        # Initialize or clean old data
        if rate_key not in self.rate_limit_data:
            self.rate_limit_data[rate_key] = []
        else:
            # Remove requests outside the current time window
            self.rate_limit_data[rate_key] = [
                t for t in self.rate_limit_data[rate_key] if t > window_start
            ]
        
        # Count requests in current window
        request_count = len(self.rate_limit_data[rate_key])
        
        # Set rate limit headers
        remaining = max(0, limit - request_count)
        reset_time = int(window_start + window_size)
        
        # Add headers to response
        headers = {
            'X-RateLimit-Limit': str(limit),
            'X-RateLimit-Remaining': str(remaining),
            'X-RateLimit-Reset': str(reset_time)
        }
        
        # Check if limit exceeded
        if request_count >= limit:
            logger.warning(f"Rate limit exceeded for {rate_key}")
            response = jsonify({
                'error': 'Too many requests',
                'message': 'Rate limit exceeded. Please try again later.'
            })
            response.status_code = 429
            
            # Add headers to error response
            for header, value in headers.items():
                response.headers[header] = value
                
            return response
        
        # Record this request
        self.rate_limit_data[rate_key].append(now)
        
        # Clean up old rate limit data periodically
        if len(self.rate_limit_data) > 10000:  # Arbitrary cleanup threshold
            self._cleanup_rate_limit_data()
            
        return None
    
    def _cleanup_rate_limit_data(self) -> None:
        """Clean up old rate limit data to prevent memory leak."""
        now = time.time()
        window_size = current_app.config.get('RATE_LIMIT_WINDOW', 60)
        window_start = now - window_size
        
        # Keep only active keys with recent requests
        cleaned_data = {}
        for key, timestamps in self.rate_limit_data.items():
            recent_timestamps = [t for t in timestamps if t > window_start]
            if recent_timestamps:
                cleaned_data[key] = recent_timestamps
                
        self.rate_limit_data = cleaned_data
        logger.debug(f"Cleaned rate limit data. Current keys: {len(self.rate_limit_data)}")
    
    def _configure_cors(self, app: Flask) -> None:
        """Configure CORS headers.
        
        Args:
            app: Flask application instance
        """
        @app.after_request
        def add_cors_headers(response):
            # Get CORS settings
            allowed_origins = app.config.get('CORS_ALLOWED_ORIGINS', '*')
            allowed_methods = app.config.get('CORS_ALLOWED_METHODS', 'GET, POST, PUT, DELETE, OPTIONS')
            allowed_headers = app.config.get('CORS_ALLOWED_HEADERS', 
                'Authorization, Content-Type, X-Requested-With, X-API-Key')
            
            # Set CORS headers
            if allowed_origins == '*' or request.origin in allowed_origins:
                response.headers['Access-Control-Allow-Origin'] = request.origin or '*'
            response.headers['Access-Control-Allow-Methods'] = allowed_methods
            response.headers['Access-Control-Allow-Headers'] = allowed_headers
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            
            return response
    
    def _log_request(self, response: Response) -> Response:
        """Log API request details.
        
        Args:
            response: Flask response object
            
        Returns:
            Response: Unmodified response
        """
        # Calculate request duration
        duration = time.time() - getattr(g, 'start_time', time.time())
        duration_ms = int(duration * 1000)
        
        # Determine authentication context
        auth_context = "anonymous"
        if hasattr(g, 'user_id') and g.user_id:
            auth_context = f"user:{g.user_id}"
        elif hasattr(g, 'api_key_id') and g.api_key_id:
            auth_context = f"api_key:{g.api_key_id}"
            
        # Log the request
        log_level = logging.INFO if 200 <= response.status_code < 400 else logging.WARNING
        logger.log(log_level, 
            f"{request.method} {request.path} - {response.status_code} - {duration_ms}ms - {auth_context}")
            
        return response
    
    def _handle_unauthorized_error(self, error) -> Response:
        """Handle 401 Unauthorized errors.
        
        Args:
            error: The error object
            
        Returns:
            Response: JSON error response
        """
        return jsonify({
            'error': 'Unauthorized',
            'message': str(error) or 'Authentication is required to access this resource'
        }), 401
    
    def _handle_forbidden_error(self, error) -> Response:
        """Handle 403 Forbidden errors.
        
        Args:
            error: The error object
            
        Returns:
            Response: JSON error response
        """
        return jsonify({
            'error': 'Forbidden',
            'message': str(error) or 'You do not have permission to access this resource'
        }), 403
    
    def _handle_rate_limit_error(self, error) -> Response:
        """Handle 429 Too Many Requests errors.
        
        Args:
            error: The error object
            
        Returns:
            Response: JSON error response
        """
        return jsonify({
            'error': 'Too Many Requests',
            'message': str(error) or 'Rate limit exceeded. Please try again later.'
        }), 429

    # Decorator for API routes requiring authentication
    def login_required(self, f: Callable) -> Callable:
        """Decorator for routes that require user authentication.
        
        Args:
            f: The route function
            
        Returns:
            Callable: Decorated function
        """
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'current_user') or not g.current_user:
                return jsonify({
                    'error': 'Unauthorized',
                    'message': 'Authentication is required to access this resource'
                }), 401
            return f(*args, **kwargs)
        return decorated_function
    
    # Decorator for API routes requiring specific roles
    def role_required(self, role: str) -> Callable:
        """Decorator for routes that require specific user roles.
        
        Args:
            role: Required role name
            
        Returns:
            Callable: Decorator function
        """
        def decorator(f: Callable) -> Callable:
            @functools.wraps(f)
            def decorated_function(*args, **kwargs):
                # Must be authenticated
                if not hasattr(g, 'current_user') or not g.current_user:
                    return jsonify({
                        'error': 'Unauthorized',
                        'message': 'Authentication is required to access this resource'
                    }), 401
                
                # Check roles
                user_roles = g.current_user.get('roles', [])
                if role not in user_roles and 'admin' not in user_roles:
                    return jsonify({
                        'error': 'Forbidden',
                        'message': f'Required role: {role}'
                    }), 403
                    
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    # Decorator for API routes requiring API keys with specific scopes
    def api_key_required(self, scope: Optional[str] = None) -> Callable:
        """Decorator for routes that require API key authentication.
        
        Args:
            scope: Required API key scope
            
        Returns:
            Callable: Decorator function
        """
        def decorator(f: Callable) -> Callable:
            # Store the required scope on the function object
            f._required_scope = scope
            
            @functools.wraps(f)
            def decorated_function(*args, **kwargs):
                # Must have API key
                if not hasattr(g, 'current_api_key') or not g.current_api_key:
                    return jsonify({
                        'error': 'Unauthorized',
                        'message': 'Valid API key is required to access this resource'
                    }), 401
                
                # Scope already checked in _before_request
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    # Decorator for custom rate limits per endpoint
    def rate_limit(self, limit: int) -> Callable:
        """Decorator for setting custom rate limits on specific endpoints.
        
        Args:
            limit: Maximum number of requests per time window
            
        Returns:
            Callable: Decorator function
        """
        def decorator(f: Callable) -> Callable:
            # Store the rate limit on the function object
            f._rate_limit = limit
            
            @functools.wraps(f)
            def decorated_function(*args, **kwargs):
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    # Decorator for validating request JSON against a schema
    def validate_json(self, schema: Dict[str, Any]) -> Callable:
        """Decorator for validating JSON request body against a schema.
        
        Args:
            schema: JSON schema dictionary
            
        Returns:
            Callable: Decorator function
        """
        def decorator(f: Callable) -> Callable:
            @functools.wraps(f)
            def decorated_function(*args, **kwargs):
                # Only validate POST, PUT, PATCH with JSON content
                if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
                    try:
                        json_data = request.get_json()
                        errors = self._validate_against_schema(json_data, schema)
                        
                        if errors:
                            return jsonify({
                                'error': 'Invalid request data',
                                'message': 'The provided data does not match the required format',
                                'details': errors
                            }), 400
                    except Exception as e:
                        return jsonify({
                            'error': 'Invalid JSON',
                            'message': str(e)
                        }), 400
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Simple schema validation without external dependencies.
        
        Args:
            data: JSON data to validate
            schema: Schema to validate against
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Check required fields
        required = schema.get('required', [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                
        # Check field types
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field in data:
                field_type = field_schema.get('type')
                
                # Skip validation if type is not specified
                if not field_type:
                    continue
                    
                # Check type
                if field_type == 'string' and not isinstance(data[field], str):
                    errors.append(f"Field '{field}' must be a string")
                elif field_type == 'number' and not isinstance(data[field], (int, float)):
                    errors.append(f"Field '{field}' must be a number")
                elif field_type == 'integer' and not isinstance(data[field], int):
                    errors.append(f"Field '{field}' must be an integer")
                elif field_type == 'boolean' and not isinstance(data[field], bool):
                    errors.append(f"Field '{field}' must be a boolean")
                elif field_type == 'array' and not isinstance(data[field], list):
                    errors.append(f"Field '{field}' must be an array")
                elif field_type == 'object' and not isinstance(data[field], dict):
                    errors.append(f"Field '{field}' must be an object")
                    
                # Check string pattern
                if field_type == 'string' and 'pattern' in field_schema:
                    import re
                    pattern = field_schema['pattern']
                    if not re.match(pattern, data[field]):
                        errors.append(f"Field '{field}' does not match required pattern")
                        
                # Check min/max for numbers
                if field_type in ['number', 'integer']:
                    if 'minimum' in field_schema and data[field] < field_schema['minimum']:
                        errors.append(f"Field '{field}' must be >= {field_schema['minimum']}")
                    if 'maximum' in field_schema and data[field] > field_schema['maximum']:
                        errors.append(f"Field '{field}' must be <= {field_schema['maximum']}")
                        
                # Check min/max length for strings
                if field_type == 'string':
                    if 'minLength' in field_schema and len(data[field]) < field_schema['minLength']:
                        errors.append(f"Field '{field}' must have at least {field_schema['minLength']} characters")
                    if 'maxLength' in field_schema and len(data[field]) > field_schema['maxLength']:
                        errors.append(f"Field '{field}' cannot exceed {field_schema['maxLength']} characters")
                        
                # Check enum values
                if 'enum' in field_schema and data[field] not in field_schema['enum']:
                    errors.append(f"Field '{field}' must be one of: {', '.join(str(x) for x in field_schema['enum'])}")
                    
        return errors

# Example API schema definitions for use with validate_json
USER_CREATE_SCHEMA = {
    'type': 'object',
    'required': ['username', 'email', 'password'],
    'properties': {
        'username': {'type': 'string', 'minLength': 3, 'maxLength': 50},
        'email': {'type': 'string', 'format': 'email'},
        'password': {'type': 'string', 'minLength': 8},
        'first_name': {'type': 'string'},
        'last_name': {'type': 'string'},
        'role': {'type': 'string', 'enum': ['admin', 'data_scientist', 'analyst', 'developer', 'guest']}
    }
}

LOGIN_SCHEMA = {
    'type': 'object',
    'required': ['username', 'password'],
    'properties': {
        'username': {'type': 'string'},
        'password': {'type': 'string'}
    }
}

API_KEY_CREATE_SCHEMA = {
    'type': 'object',
    'required': ['name', 'scopes'],
    'properties': {
        'name': {'type': 'string', 'minLength': 3, 'maxLength': 100},
        'scopes': {'type': 'array', 'items': {'type': 'string'}},
        'expires_in_days': {'type': 'integer', 'minimum': 1}
    }
} 