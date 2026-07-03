"""Semantic Query Router.

Uses embedding distances to decide if a user query should be handled by the 
standard (fast) RAG pipeline or the complex LangGraph Agent.
"""

from src.ml.embedding import embed
from src.services.guardrails.scorer import cosine_similarity
from src.utils.logger import get_logger

logger = get_logger(__name__)

# We define multiple specific anchors for better semantic clustering.
# If the query is close to ANY agent anchor, we route to the agent.
AGENT_ANCHORS = [
    "math calculation arithmetic numbers addition multiplication",
    "live news current events stock prices real-time information",
    "what is the live weather forecast temperature right now",
    "who is the ceo of meta public figures tech companies general knowledge internet search",
    "compare and contrast difference between complex multi-step reasoning"
]

RAG_ANCHORS = [
    "internal company documents policies procedures employee handbook",
    "project apollo database architecture engineering specs internal records"
]

# Pre-compute embeddings for all anchors
_agent_vectors = embed(AGENT_ANCHORS)
_rag_vectors = embed(RAG_ANCHORS)


def route_query(query: str) -> bool:
    """Determine if a query requires the Agent using semantic routing.
    
    Args:
        query: The user's input string.
        
    Returns:
        True if the Agent should be used, False to use standard RAG.
    """
    # Embed the incoming user query
    query_vector = embed([query])[0]
    
    # Calculate how semantically similar the query is to all anchors
    # We take the MAXIMUM score for both categories
    agent_score = max(cosine_similarity(query_vector, vec) for vec in _agent_vectors)
    rag_score = max(cosine_similarity(query_vector, vec) for vec in _rag_vectors)
    
    # Route based on which cluster is closer in the vector space
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
