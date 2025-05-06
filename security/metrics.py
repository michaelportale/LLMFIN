import time
from functools import wraps
from flask import request, Blueprint
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST

# Initialize metrics
AUTH_ATTEMPTS = Counter('auth_attempts_total', 'Total number of authentication attempts', ['success', 'method'])
API_KEY_USAGE = Counter('api_key_usage_total', 'Total number of API key uses', ['key_id', 'endpoint'])
RATE_LIMIT_EVENTS = Counter('rate_limit_events_total', 'Total number of rate limiting events', ['endpoint', 'user_id'])
SECURITY_EVENTS = Counter('security_events_total', 'Security related events', ['event_type', 'severity'])
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
ACTIVE_API_KEYS = Gauge('active_api_keys', 'Number of active API keys')
REQUEST_RATE = Gauge('request_rate', 'Request rate per second')

# Request latency
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['endpoint', 'method'])

# Set up a blueprint for metrics endpoint
metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/metrics')
def metrics():
    """Endpoint to expose Prometheus metrics"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

def record_auth_attempt(success, method='password'):
    """Record an authentication attempt"""
    AUTH_ATTEMPTS.labels(str(success).lower(), method).inc()
    
def record_api_key_usage(key_id, endpoint):
    """Record API key usage"""
    API_KEY_USAGE.labels(key_id, endpoint).inc()
    
def record_rate_limit_event(endpoint, user_id):
    """Record a rate limit event"""
    RATE_LIMIT_EVENTS.labels(endpoint, user_id).inc()
    
def record_security_event(event_type, severity='info'):
    """Record a security event"""
    SECURITY_EVENTS.labels(event_type, severity).inc()
    
def set_active_users(count):
    """Set the number of active users"""
    ACTIVE_USERS.set(count)
    
def inc_active_users():
    """Increment the number of active users"""
    ACTIVE_USERS.inc()
    
def dec_active_users():
    """Decrement the number of active users"""
    ACTIVE_USERS.dec()
    
def set_active_api_keys(count):
    """Set the number of active API keys"""
    ACTIVE_API_KEYS.set(count)
    
def inc_active_api_keys():
    """Increment the number of active API keys"""
    ACTIVE_API_KEYS.inc()
    
def dec_active_api_keys():
    """Decrement the number of active API keys"""
    ACTIVE_API_KEYS.dec()
    
def track_request_rate():
    """Track the request rate"""
    REQUEST_RATE.inc()
    
def measure_latency(endpoint, method):
    """Measure request latency"""
    return REQUEST_LATENCY.labels(endpoint, method).time()

def request_metrics_middleware():
    """Middleware to record request metrics"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            track_request_rate()
            endpoint = request.endpoint or 'unknown'
            method = request.method
            
            with measure_latency(endpoint, method):
                return f(*args, **kwargs)
                
        return decorated_function
    return decorator

# Function to initialize metrics with Flask app
def init_metrics(app):
    """Initialize metrics with Flask app"""
    app.register_blueprint(metrics_bp)
    
    # Apply middleware to all routes
    for endpoint, view_func in app.view_functions.items():
        if endpoint != 'metrics.metrics':  # Skip the metrics endpoint itself
            app.view_functions[endpoint] = request_metrics_middleware()(view_func)
    
    return app 