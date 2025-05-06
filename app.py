import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from flask import Flask, jsonify, request, g
from flask_cors import CORS

# Import security components
from security.middleware import SecurityMiddleware
from security.auth import AuthManager
from security.user_management import UserManager
from security.api_key_manager import APIKeyManager
from security.rate_limiter import RateLimiter
from security.metrics import init_metrics, record_security_event, inc_active_users, dec_active_users, set_active_users

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_app(config=None):
    """Create and configure Flask application.
    
    Args:
        config: Configuration dictionary or path to config file
        
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    
    # Load default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY') or os.urandom(32).hex(),
        MONGODB_URI=os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/rl_results'),
        JWT_TOKEN_EXPIRY=int(os.environ.get('JWT_TOKEN_EXPIRY', 3600)),
        DEBUG=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true',
        CORS_ALLOWED_ORIGINS=os.environ.get('CORS_ALLOWED_ORIGINS', '*'),
        IP_ALLOWLIST=os.environ.get('IP_ALLOWLIST', '').split(',') if os.environ.get('IP_ALLOWLIST') else [],
        IP_DENYLIST=os.environ.get('IP_DENYLIST', '').split(',') if os.environ.get('IP_DENYLIST') else [],
        ALLOWED_ORIGINS=os.environ.get("ALLOWED_ORIGINS", "*")
    )
    
    # Override with custom config if provided
    if config:
        if isinstance(config, dict):
            app.config.update(config)
        else:
            app.config.from_pyfile(config)
    
    try:
        # Initialize security components
        auth_manager = AuthManager(
            secret_key=os.environ.get('JWT_SECRET_KEY'),
            token_expiry=int(os.environ.get('TOKEN_EXPIRY_SECONDS', 86400))
        )
        
        user_manager = UserManager(
            auth_manager=auth_manager,
            db_uri=os.environ.get('MONGO_URI')
        )
        
        api_key_manager = APIKeyManager(
            mongo_uri=os.environ.get('MONGO_URI')
        )
        
        rate_limiter = RateLimiter(
            redis_uri=os.environ.get('REDIS_URI')
        )
        
        # Initialize security middleware with proper components
        security = SecurityMiddleware(app, {
            'auth_manager': auth_manager,
            'user_manager': user_manager,
            'api_key_manager': api_key_manager,
            'rate_limiter': rate_limiter
        })
        
        # Initialize metrics
        app = init_metrics(app)
        
        # Counter for active user sessions
        active_users = user_manager.count_active_users()
        set_active_users(active_users)
    except Exception as e:
        logger.warning(f"Security components couldn't be fully initialized: {str(e)}")
        logger.warning("Running in limited functionality mode")
        
    # Enable CORS with secure defaults
    CORS(app, resources={r"/api/*": {"origins": app.config.get("ALLOWED_ORIGINS")}})

    # Now import blueprints after security is initialized
    from api.stock_search_routes import stock_search_bp
    
    try:
        from api.security_routes import security_bp
        # Register security blueprint if available
        app.register_blueprint(security_bp, url_prefix='/api/security')
    except Exception as e:
        logger.warning(f"Security routes not available: {str(e)}")
    
    # Register stock search blueprint
    app.register_blueprint(stock_search_bp, url_prefix='/api/stocks')
    
    # Set up error handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found"}), 404
        
    @app.errorhandler(500)
    def server_error(e):
        try:
            record_security_event("server_error", "error")
        except:
            pass
        return jsonify({"error": "Internal server error"}), 500
    
    # Main index route - redirect to the default page
    @app.route('/', methods=['GET'])
    def index():
        return app.send_static_file('index.html')
        
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "ok", "version": "1.0.0"}), 200
        
    # Stock search UI
    @app.route('/stocks/search', methods=['GET'])
    def stock_search_ui():
        return app.send_static_file('stock_search.html')
    
    # API Authentication routes if security components are available
    if 'security' in dir(app):
        @app.route('/api/auth/login', methods=['POST'])
        def login():
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
                
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                return jsonify({"error": "Username and password required"}), 400
                
            try:
                user = user_manager.authenticate_user(username, password)
                if not user:
                    record_security_event("failed_login", "warning")
                    return jsonify({"error": "Invalid credentials"}), 401
                    
                # Generate token
                token = auth_manager.generate_token(
                    user_id=str(user['_id']),
                    username=user['username'],
                    role=user['role']
                )
                
                inc_active_users()
                record_security_event("successful_login", "info")
                
                return jsonify({
                    "message": "Login successful",
                    "token": token,
                    "user": {
                        "id": str(user['_id']),
                        "username": user['username'],
                        "role": user['role']
                    }
                }), 200
            except Exception as e:
                logger.error(f"Login error: {str(e)}")
                return jsonify({"error": "Authentication failed"}), 401
    
    logger.info(f"Application initialized with environment: {app.config.get('FLASK_ENV', 'development')}")
    
    return app

if __name__ == '__main__':
    # Create and run the application
    app = create_app()
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))  # Use port 5001 to avoid macOS AirPlay conflicts
    
    # Run the application
    logger.info(f"Starting application on {host}:{port}")
    app.run(host=host, port=port, debug=app.config.get('DEBUG', False))