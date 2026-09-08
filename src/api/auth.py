"""API Authentication using X-API-Key header.

Provides a FastAPI dependency to secure endpoints.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from src.config import get_settings
from src.storage.db.database import get_db
from src.storage.db.models import User

settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.app_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Optional[User]:
    """Validate JWT and retrieve User from DB.
    
    Bypass auth if token is None and no API keys/Auth are required locally, 
    but for Enterprise RBAC we enforce token presence.
    """
    if not token:
        # We can bypass auth for completely open local dev if desired, 
        # but Enterprise RBAC demands authentication.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.app_secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

def rate_limit_dependency(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user),
) -> None:
    """Enforce the sliding window rate limit for the API."""
    from src.storage.cache import check_rate_limit
    
    client_id = current_user.username if current_user else None
    
    if not client_id:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_id = forwarded_for.split(",")[0].strip()
        else:
            client_id = request.client.host if request.client else "unknown"
        
    # ── Context Observability: Set the User ID for Logs/Traces ──
    from src.utils.request_context import set_user_id
    set_user_id(client_id)
    
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down."
        )
