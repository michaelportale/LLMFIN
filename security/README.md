# Reinforcement Learning Security System

This security system provides a comprehensive authentication, authorization, and secure data handling framework for the reinforcement learning application.

## Features

### Authentication System
- **JWT-based authentication** for token management
- **Password hashing and security** using bcrypt
- **Role-based access control** with flexible permission system
- **User account management** (creation, activation, deactivation, password reset)

### API Key Management
- **Granular API key creation** with defined scopes and permissions
- **Automatic key expiration** for enhanced security
- **API key validation** and permission checking
- **Usage tracking** for all API keys

### Data Source Security
- **Secure storage of external data source credentials**
- **Encrypted configuration management** for sensitive connection details
- **Credential rotation** support

### Secure Storage for Sensitive Data
- **Encryption at rest** for all sensitive data
- **Categorized secure storage** for different types of sensitive information
- **Data integrity verification** through checksums
- **Simple API** for storing, retrieving, and managing secure data

### API Protection Middleware
- **Rate limiting** to prevent abuse
- **IP-based filtering** (allowlists/denylists)
- **CORS configuration** for browser security
- **Request validation** against schemas
- **Detailed logging** of security events
- **Brute force protection**

## Architecture

The security system is built around these key components:

1. **AuthManager** (`auth.py`) - Handles JWT token creation, verification, and user authentication flows
2. **UserManager** (`user_management.py`) - Manages user accounts, roles, and permissions
3. **APIKeyManager** (`api_keys.py`) - Handles API key lifecycle and validation
4. **SecureStorage** (`secure_storage.py`) - Provides encrypted storage for sensitive data
5. **SecurityMiddleware** (`middleware.py`) - Flask middleware for protecting API endpoints
6. **API Routes** (`../api/security_routes.py`) - Implementation of security-related API endpoints

## Configuration

The security system is configured through environment variables or a configuration file loaded by the Flask application:

- `JWT_SECRET_KEY` - Secret key for JWT token signing (critical for security)
- `JWT_TOKEN_EXPIRY` - Token expiration time in seconds (default: 3600)
- `MONGODB_URI` - MongoDB connection string
- `ENCRYPTION_KEY` - Key for encrypting sensitive data
- `RATE_LIMIT_DEFAULT` - Default rate limit per minute
- `RATE_LIMIT_WINDOW` - Time window for rate limiting in seconds
- `IP_ALLOWLIST` - Comma-separated list of allowed IP addresses/ranges
- `IP_DENYLIST` - Comma-separated list of banned IP addresses/ranges
- `CORS_ALLOWED_ORIGINS` - Allowed origins for CORS

## Usage Examples

### Authentication

```python
# Initialize components
auth_manager = AuthManager(secret_key="your-secret-key")
user_manager = UserManager(db_uri="mongodb://localhost:27017/mydb")

# Create a user
success, message, user_id = user_manager.create_user(
    username="johndoe",
    email="john@example.com",
    password="secure_password",
    role="data_scientist"
)

# Authenticate
success, message, user_data = user_manager.authenticate_user(
    username="johndoe",
    password="secure_password"
)

# Generate JWT token
token_data = {
    'sub': user_data['user_id'],
    'username': user_data['username'],
    'roles': user_data['roles']
}
token = auth_manager.generate_token(token_data)

# Verify token
is_valid, payload = auth_manager.verify_token(token)
```

### API Keys

```python
# Initialize API key manager
api_key_manager = APIKeyManager(db_uri="mongodb://localhost:27017/mydb")

# Generate new API key
success, message, api_key = api_key_manager.generate_api_key(
    name="Data Science Pipeline",
    owner_id="user_123",
    scopes=["data_source:read", "model:read", "prediction:create"],
    expires_in_days=90
)

# Validate an API key
is_valid, key_data = api_key_manager.validate_api_key(
    api_key="your-api-key-here",
    required_scope="data_source:read"
)
```

### Secure Storage

```python
# Initialize secure storage
secure_store = SecureStorage(
    db_uri="mongodb://localhost:27017/mydb",
    encryption_key="your-encryption-key"
)

# Store sensitive data
success, message, data_id = secure_store.store_data(
    data={"api_key": "secret_api_key", "password": "secret_password"},
    category="credentials",
    name="AWS Credentials",
    owner_id="user_123"
)

# Retrieve and decrypt
success, message, result = secure_store.retrieve_data(data_id)
credentials = result['data']
```

### Middleware Protection

```python
# In your Flask application
from flask import Flask
from security.middleware import SecurityMiddleware

app = Flask(__name__)
security = SecurityMiddleware(app)

# Protected route with authentication
@app.route('/api/protected')
@security.login_required
def protected_route():
    return {"message": "This route is protected"}

# Role-based protection
@app.route('/api/admin')
@security.role_required('admin')
def admin_route():
    return {"message": "Admin only"}

# API key protected route
@app.route('/api/data')
@security.api_key_required(scope='data_source:read')
def data_route():
    return {"message": "API key access"}

# Rate limited route
@app.route('/api/limited')
@security.rate_limit(100)
def limited_route():
    return {"message": "Rate limited route"}
```

## Security Best Practices

- Ensure `JWT_SECRET_KEY` and `ENCRYPTION_KEY` are strong, unique, and kept secure
- Store these keys in environment variables, not in code
- Use HTTPS in production to protect data in transit
- Rotate API keys periodically
- Use the most restrictive scopes possible for API keys
- Monitor logs for suspicious activity
- Implement the principle of least privilege for roles

## API Documentation

For full API documentation, see the API routes defined in `../api/security_routes.py`.

## Dependencies

- Flask
- PyMongo
- PyJWT
- Cryptography
- BCrypt
- IPAddress

## Testing

Run the security system tests:

```
pytest security/tests/
``` 