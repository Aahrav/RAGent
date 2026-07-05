"""Redis low-level operations and caching logic.

Provides a singleton connection to the Redis server and handles
getting/setting cached query responses to speed up the RAG pipeline.
"""

import json
from typing import Any

import redis

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_redis_client: redis.Redis | None = None

def get_client() -> redis.Redis:
    """Return a singleton Redis client instance.
    
    Returns:
        An active redis.Redis connection object.
        
    Raises:
        redis.ConnectionError: If the server is unreachable.
    """
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        logger.info("Initializing Redis connection", extra={"url": settings.redis_url})
        try:
            # decode_responses=True ensures we get strings back instead of bytes
            _redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            
            # Ping immediately to verify the connection is alive
            _redis_client.ping()
            logger.info("Connected to Redis successfully")
            
        except redis.ConnectionError as e:
            logger.error("Failed to connect to Redis", extra={"error": str(e)})
            raise
            
    return _redis_client
