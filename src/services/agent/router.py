"""Semantic Query Router.

Uses embedding distances to decide if a user query should be handled by the 
standard (fast) RAG pipeline or the complex LangGraph Agent.
"""

from src.ml.embedding import embed
from src.services.guardrails.scorer import cosine_similarity
from src.utils.logger import get_logger

logger = get_logger(__name__)

# We define "anchors". An anchor is a prototypical query or description 
# representing a specific route.
AGENT_ANCHOR = "math calculation addition multiply current events live news live weather stock price search the internet compare difference between complex reasoning"
RAG_ANCHOR = "internal company documents policies project apollo database architectural overview employee handbook private records"

# Pre-compute the embeddings for our anchors so we don't recalculate them on every request
# (When the app starts, this module is imported once, caching these vectors in memory)
_agent_vector = embed([AGENT_ANCHOR])[0]
_rag_vector = embed([RAG_ANCHOR])[0]


def route_query(query: str) -> bool:
    """Determine if a query requires the Agent using semantic routing.
    
    Args:
        query: The user's input string.
        
    Returns:
        True if the Agent should be used, False to use standard RAG.
    """
    # Embed the incoming user query
    query_vector = embed([query])[0]
    
    # Calculate how semantically similar the query is to both anchors
    agent_score = cosine_similarity(query_vector, _agent_vector)
    rag_score = cosine_similarity(query_vector, _rag_vector)
    
    # Route based on which anchor is closer in the vector space
    if agent_score > rag_score:
        logger.info(
            "Query semantically routed to Agent",
            extra={"query": query, "agent_score": round(agent_score, 3), "rag_score": round(rag_score, 3)}
        )
        return True
        
    # Default to the much faster standard RAG pipeline
    logger.info(
        "Query semantically routed to RAG",
        extra={"query": query, "agent_score": round(agent_score, 3), "rag_score": round(rag_score, 3)}
    )
    return False
