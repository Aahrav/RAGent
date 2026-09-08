"""Chat history storage using Redis.

Provides functions to store and retrieve short-term conversational memory
(chat history) for a given session.
"""

import json
from src.storage.cache import get_client
from src.utils.logger import get_logger
from src.config import get_settings

logger = get_logger(__name__)

def get_chat_history(session_id: str, limit: int = 10) -> list[dict[str, str]]:
    """Retrieve the last N messages for a session from Redis.
    
    Returns:
        List of dictionaries with 'role' and 'content', oldest first.
    """
    settings = get_settings()
    if not settings.cache_enabled:
        return []
        
    try:
        client = get_client()
        key = f"chat_history:{session_id}"
        
        # Get the last `limit` messages (lrange is inclusive, so -limit to -1)
        raw_history = client.lrange(key, -limit, -1)
        
        history = []
        for item in raw_history:
            history.append(json.loads(item))
            
        return history
    except Exception as e:
        logger.warning("Failed to retrieve chat history", extra={"error": str(e), "session_id": session_id})
        return []

def add_to_chat_history(session_id: str, role: str, content: str) -> None:
    """Append a single message to the session's chat history in Redis.
    
    Args:
        session_id: The unique identifier for the conversation.
        role: 'user' or 'assistant'.
        content: The text content of the message.
    """
    settings = get_settings()
    if not settings.cache_enabled:
        return
        
    try:
        client = get_client()
        key = f"chat_history:{session_id}"
        
        msg = json.dumps({"role": role, "content": content})
        
        pipeline = client.pipeline()
        pipeline.rpush(key, msg)
        # Keep only the last 20 messages to prevent memory bloat
        pipeline.ltrim(key, -20, -1)
        # Expire history after 24 hours of inactivity
        pipeline.expire(key, 86400) 
        pipeline.execute()
        
    except Exception as e:
        logger.warning("Failed to save chat history", extra={"error": str(e), "session_id": session_id})
