"""Content Safety Filter.

A rapid pre-flight check to block inappropriate or off-topic queries
before they reach the LLM, saving compute and preventing misuse.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

# A simple set of blocked keywords. In a real production system,
# this might be loaded from a database or use a dedicated moderation API.
BLOCKLIST = {
    "hack",
    "password",
    "secret",
    "ignore previous instructions",
    "system prompt",
    "bypass",
    "malware",
}


def check_safety(query: str) -> bool:
    """Check if the query contains any blocked keywords.
    
    Args:
        query: The user's input string.
        
    Raises:
        ValueError: If the query triggers the blocklist.
        
    Returns:
        True if the query is safe.
    """
    query_lower = query.lower()
    
    for blocked_word in BLOCKLIST:
        if blocked_word in query_lower:
            logger.warning(
                "Query blocked by safety filter",
                extra={
                    "blocked_word": blocked_word,
                    "query": query
                }
            )
            raise ValueError(
                f"Query blocked: contains prohibited content."
            )
            
    return True
