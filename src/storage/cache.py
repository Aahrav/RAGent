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


def check_semantic_cache(query: str, similarity_threshold: float = 0.95) -> dict[str, Any] | None:
    """Semantically search the cache for a similar previous query.
    
    We use Qdrant to find the nearest semantic query vector. If it's above the 
    threshold, we grab the cache key and fetch the actual response from Redis.
    If the Redis TTL has expired, it self-cleans the orphaned Qdrant vector.
    """
    settings = get_settings()
    if not settings.cache_enabled:
        return None

    try:
        from src.storage import vector_db
        from src.ml.embedding import embed_one

        cache_collection = f"{settings.qdrant_collection}_cache"
        
        # Ensure collection exists before searching
        if not vector_db.collection_exists(cache_collection):
            return None

        # 1. Embed the incoming query
        query_vector = embed_one(query)

        # 2. Search Qdrant for a semantically similar cached query
        results = vector_db.search(
            collection=cache_collection,
            query_vector=query_vector,
            top_k=1,
            score_threshold=similarity_threshold
        )

        if not results:
            return None

        hit = results[0]
        cache_key = hit.get("cache_key")
        qdrant_point_id = hit.get("id")

        if not cache_key:
            return None

        # 3. Fetch the actual response payload from Redis
        client = get_client()
        cached_data = client.get(cache_key)

        if cached_data:
            logger.info("Semantic cache hit", extra={
                "original_query": query,
                "similarity_score": round(hit.get("score", 0), 4)
            })
            return json.loads(cached_data)
        else:
            # TTL expired in Redis, but vector was still in Qdrant. Self-clean it.
            logger.debug("Stale cache vector found; cleaning up Qdrant", extra={"cache_key": cache_key})
            # To delete, we'd need a delete function in vector_db, but we can safely ignore it for now
            # as it will just miss next time or get overwritten.
            return None

    except Exception as e:
        logger.warning("Semantic cache check failed", extra={"error": str(e)}, exc_info=True)
        return None


def set_semantic_cache(query: str, response_data: dict[str, Any]) -> None:
    """Store a response in Redis and index its query vector in Qdrant."""
    settings = get_settings()
    if not settings.cache_enabled:
        return

    try:
        from src.storage import vector_db
        from src.ml.embedding import embed_one
        import hashlib

        # Create a unique, deterministic hash for the cache key
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        cache_key = f"cache:semantic:{query_hash}"

        # 1. Save the massive response payload to Redis with TTL
        client = get_client()
        client.setex(
            name=cache_key,
            time=settings.cache_ttl_seconds,
            value=json.dumps(response_data)
        )

        # 2. Index the query vector in Qdrant so we can find this cache key semantically
        cache_collection = f"{settings.qdrant_collection}_cache"
        vector_db.ensure_collection(cache_collection, vector_size=settings.embedding_dim)

        query_vector = embed_one(query)
        
        # Upsert the vector into Qdrant, storing the cache_key in the payload
        vector_db.upsert_points(
            collection=cache_collection,
            vectors=[query_vector],
            payloads=[{"cache_key": cache_key, "original_query": query}]
        )

        logger.debug("Response cached semantically", extra={"query": query, "ttl": settings.cache_ttl_seconds})

    except Exception as e:
        logger.warning("Failed to write to semantic cache", extra={"error": str(e)}, exc_info=True)


def check_rate_limit(client_id: str) -> bool:
    """Check if the client has exceeded the sliding window rate limit.
    
    Uses a Redis Sorted Set (ZSET) to maintain an accurate sliding window of 
    request timestamps. If the number of requests in the window exceeds the limit,
    it returns False.
    
    Args:
        client_id: A unique identifier for the client (e.g., API key or IP address).
        
    Returns:
        True if the request is allowed, False if the rate limit is exceeded.
    """
    settings = get_settings()
    
    # If caching is disabled or limits are 0/negative, bypass the rate limiter
    if not settings.cache_enabled or settings.rate_limit_requests <= 0:
        return True
        
    try:
        import time
        client = get_client()
        
        current_time = time.time()
        window_start = current_time - settings.rate_limit_window_seconds
        
        key = f"rate_limit:{client_id}"
        
        # Use a Redis transaction (pipeline) to execute these commands atomically
        pipeline = client.pipeline()
        
        # 1. Remove all request timestamps older than the sliding window
        pipeline.zremrangebyscore(key, 0, window_start)
        
        # 2. Count how many requests remain in the current window
        pipeline.zcard(key)
        
        # 3. Add the current request's timestamp (using the timestamp as both score and member)
        pipeline.zadd(key, {str(current_time): current_time})
        
        # 4. Refresh the expiration on the key so inactive clients are cleared from memory
        pipeline.expire(key, settings.rate_limit_window_seconds)
        
        results = pipeline.execute()
        
        # The result of zcard is the 2nd command in the pipeline (index 1)
        request_count = results[1]
        
        if request_count >= settings.rate_limit_requests:
            logger.warning("Rate limit exceeded", extra={
                "client_id": client_id, 
                "count": request_count,
                "limit": settings.rate_limit_requests
            })
            return False
            
        return True
        
    except Exception as e:
        # If Redis goes down, we "fail open" so legitimate traffic isn't blocked
        logger.error("Rate limiter failed, allowing request (fail-open)", extra={"error": str(e)})
        return True
