"""API Authentication using X-API-Key header.

Provides a FastAPI dependency to secure endpoints.
"""

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from src.config import get_settings

# Extract the API key from the "X-API-Key" HTTP header.
# auto_error=False allows us to handle the missing key logic ourselves.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key_header_value: str = Security(api_key_header)) -> str | None:
    """Validate the incoming API key against configured valid keys.
    
    If the server has no API keys configured in the environment (.env), 
    authentication is temporarily bypassed. This makes local development easier.
    
    Args:
        api_key_header_value: The value of the X-API-Key header extracted by FastAPI.
        
    Returns:
        The valid API key if successful, or None if auth is bypassed.
        
    Raises:
        HTTPException: 403 Forbidden if the API key is missing or invalid.
    """
    settings = get_settings()
    valid_keys = settings.get_api_keys_list()
    
    # 1. Bypassed Auth: If no keys are configured, skip authentication entirely.
    if not valid_keys:
        return None
        
    # 2. Missing Key
    if not api_key_header_value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authentication required: Missing X-API-Key header"
        )
        
    # 3. Invalid Key
    if api_key_header_value not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authentication failed: Invalid API Key"
        )
        
    return api_key_header_value


def rate_limit_dependency(
    request: Request,
    api_key: str | None = Depends(verify_api_key),
) -> None:
    """Enforce the sliding window rate limit for the API.
    
    This dependency should be added to routers or endpoints to protect them.
    It automatically invokes verify_api_key first to authenticate the user.
    
    Args:
        request: The FastAPI request object.
        api_key: The authenticated API key (if auth is enabled).
        
    Raises:
        HTTPException: 429 Too Many Requests if the limit is exceeded.
    """
    from src.storage.cache import check_rate_limit
    
    # Use the API key if provided, otherwise fallback to the client's IP address
    # If the client is behind a proxy, we try to use the x-forwarded-for header
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
        
    client_id = api_key if api_key else client_ip
    
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down."
        )
