import time
import logging
from typing import Dict, Optional, Tuple, List, Callable
from functools import wraps
from flask import request, jsonify, g

import redis
from redis.exceptions import RedisError

from security.metrics import record_rate_limit_event, record_security_event

# Configure logging
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API requests with Redis backend"""
    
    def __init__(self, redis_uri: str = None):
        """Initialize the rate limiter
        
        Args:
            redis_uri: Redis connection URI
        """
        self.redis_uri = redis_uri or "redis://localhost:6379/0"
        self.redis = None
        self._connect_redis()
        
    def _connect_redis(self) -> None:
        """Connect to Redis"""
        try:
            self.redis = redis.from_url(self.redis_uri)
            self.redis.ping()  # Test connection
            logger.info("Connected to Redis for rate limiting")
        except (RedisError, ConnectionError) as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            record_security_event("redis_connection_error", "error")
            self.redis = None
            
    def _get_limit_key(self, identifier: str, endpoint: str) -> str:
        """Get Redis key for a rate limit counter
        
        Args:
            identifier: Client identifier (IP, user ID, etc.)
            endpoint: API endpoint
            
        Returns:
            str: Redis key
        """
        return f"ratelimit:{identifier}:{endpoint}"
        
    def check_rate_limit(self, identifier: str, endpoint: str, 
                         limit: int, period: int) -> Tuple[bool, Dict]:
        """Check if a request is within rate limits
        
        Args:
            identifier: Client identifier (IP, user ID, etc.)
            endpoint: API endpoint
            limit: Maximum number of requests
            period: Time period in seconds
            
        Returns:
            tuple: (is_allowed, limit_info)
        """
        if not self.redis:
            self._connect_redis()
            if not self.redis:
                # If Redis is unavailable, allow the request to avoid blocking legitimate users
                logger.warning("Redis unavailable, bypassing rate limiting")
                return True, {'limit': limit, 'remaining': 1, 'reset': int(time.time()) + period}
                
        key = self._get_limit_key(identifier, endpoint)
        
        try:
            # Use pipelining for atomic operations
            pipe = self.redis.pipeline()
            
            # Check if key exists and get current count
            current_count = self.redis.get(key)
            if current_count is None:
                # Key doesn't exist, set new counter with expiry
                pipe.set(key, 1)
                pipe.expire(key, period)
                pipe.execute()
                
                return True, {'limit': limit, 'remaining': limit - 1, 'reset': int(time.time()) + period}
            
            # Key exists, check count against limit
            current_count = int(current_count)
            ttl = self.redis.ttl(key)
            
            if current_count >= limit:
                # Rate limit exceeded
                record_rate_limit_event(endpoint, identifier)
                record_security_event("rate_limit_exceeded", "warning")
                return False, {'limit': limit, 'remaining': 0, 'reset': int(time.time()) + ttl}
                
            # Increment counter
            pipe.incr(key)
            if ttl < 0:
                pipe.expire(key, period)  # Reset expiry if needed
            pipe.execute()
            
            return True, {'limit': limit, 'remaining': limit - current_count - 1, 'reset': int(time.time()) + ttl}
            
        except (RedisError, ConnectionError) as e:
            logger.error(f"Rate limiting error: {str(e)}")
            record_security_event("rate_limit_error", "error")
            return True, {'limit': limit, 'remaining': 1, 'reset': int(time.time()) + period}
            
    def limit_by_ip(self, limit: int = 100, period: int = 60):
        """Decorator to apply rate limiting by IP address
        
        Args:
            limit: Maximum number of requests
            period: Time period in seconds
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Extract IP address
                ip = request.remote_addr
                endpoint = request.endpoint or "unknown"
                
                allowed, limit_info = self.check_rate_limit(ip, endpoint, limit, period)
                
                if not allowed:
                    record_security_event("ip_rate_limit_exceeded", "warning")
                    response = jsonify({
                        'error': 'Rate limit exceeded',
                        'limit': limit_info['limit'],
                        'remaining': limit_info['remaining'],
                        'reset': limit_info['reset']
                    })
                    response.headers['X-RateLimit-Limit'] = str(limit_info['limit'])
                    response.headers['X-RateLimit-Remaining'] = str(limit_info['remaining'])
                    response.headers['X-RateLimit-Reset'] = str(limit_info['reset'])
                    return response, 429
                
                # Add rate limit headers to response
                response = f(*args, **kwargs)
                
                # Check if response is a tuple (response, status) or just response
                if isinstance(response, tuple):
                    resp, status = response
                    if hasattr(resp, 'headers'):
                        resp.headers['X-RateLimit-Limit'] = str(limit_info['limit'])
                        resp.headers['X-RateLimit-Remaining'] = str(limit_info['remaining'])
                        resp.headers['X-RateLimit-Reset'] = str(limit_info['reset'])
                    return resp, status
                else:
                    if hasattr(response, 'headers'):
                        response.headers['X-RateLimit-Limit'] = str(limit_info['limit'])
                        response.headers['X-RateLimit-Remaining'] = str(limit_info['remaining'])
                        response.headers['X-RateLimit-Reset'] = str(limit_info['reset'])
                    return response
                    
            return decorated_function
        return decorator
        
    def limit_by_user(self, limit: int = 1000, period: int = 3600):
        """Decorator to apply rate limiting by user ID
        
        Args:
            limit: Maximum number of requests
            period: Time period in seconds
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Extract user ID from request context
                user_id = getattr(g, 'user', {}).get('user_id', 'anonymous')
                endpoint = request.endpoint or "unknown"
                
                allowed, limit_info = self.check_rate_limit(user_id, endpoint, limit, period)
                
                if not allowed:
                    record_security_event("user_rate_limit_exceeded", "warning")
                    response = jsonify({
                        'error': 'Rate limit exceeded',
                        'limit': limit_info['limit'],
                        'remaining': limit_info['remaining'],
                        'reset': limit_info['reset']
                    })
                    response.headers['X-RateLimit-Limit'] = str(limit_info['limit'])
                    response.headers['X-RateLimit-Remaining'] = str(limit_info['remaining'])
                    response.headers['X-RateLimit-Reset'] = str(limit_info['reset'])
                    return response, 429
                
                # Add rate limit headers to response
                response = f(*args, **kwargs)
                
                # Check if response is a tuple (response, status) or just response
                if isinstance(response, tuple):
                    resp, status = response
                    if hasattr(resp, 'headers'):
                        resp.headers['X-RateLimit-Limit'] = str(limit_info['limit'])
                        resp.headers['X-RateLimit-Remaining'] = str(limit_info['remaining'])
                        resp.headers['X-RateLimit-Reset'] = str(limit_info['reset'])
                    return resp, status
                else:
                    if hasattr(response, 'headers'):
                        response.headers['X-RateLimit-Limit'] = str(limit_info['limit'])
                        response.headers['X-RateLimit-Remaining'] = str(limit_info['remaining'])
                        response.headers['X-RateLimit-Reset'] = str(limit_info['reset'])
                    return response
                    
            return decorated_function
        return decorator 