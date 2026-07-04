"""LLM-based Query Router.

Uses a lightweight LLM call to decide if a user query should be handled by the 
standard (fast) RAG pipeline or the complex LangGraph Agent.
"""

from typing import Literal
from pydantic import BaseModel, Field

from src.ml.llm import get_llm
from src.utils.logger import get_logger
from langsmith import traceable

logger = get_logger(__name__)


class RouteDecision(BaseModel):
    """The structured output for the router."""
    route: Literal["RAG", "AGENT"] = Field(
        ...,
        description="Choose 'RAG' for internal company knowledge. Choose 'AGENT' for math, live internet searches, consumer products, weather, or complex reasoning."
    )


@traceable(name="llm_router")
def route_query(query: str) -> bool:
    """Determine if a query requires the Agent using LLM classification.
    
    Args:
        query: The user's input string.
        
    Returns:
        True if the Agent should be used, False to use standard RAG.
    """
    logger.debug("Routing query via LLM", extra={"query": query})
    
    llm = get_llm()
    
    try:
        # We use structured output to guarantee we get exactly "RAG" or "AGENT"
        router = llm.with_structured_output(RouteDecision)
        
        system_prompt = (
            "You are a routing expert. Your job is to classify user queries into one of two buckets:\n"
            "1. 'RAG' - The query asks about internal company policies, Project Apollo, or proprietary data.\n"
            "2. 'AGENT' - The query asks about general knowledge, public figures, consumer products, live weather, stock prices, or math calculations.\n"
            "If in doubt or the query is very vague (like 'loq laptop' or 'meta ceo'), route to 'AGENT'."
        )
        
        messages = [
            ("system", system_prompt),
            ("human", query)
        ]
        
        result: RouteDecision = router.invoke(messages)
        
        if result.route == "AGENT":
            logger.info("Query LLM-routed to Agent", extra={"query": query})
            return True
        else:
            logger.info("Query LLM-routed to RAG", extra={"query": query})
            return False
            
    except Exception as e:
        logger.warning(
            "LLM router failed, falling back to Agent", 
            extra={"error": str(e), "query": query}
        )
        # Safest fallback is the Agent since it can handle both internal and external queries
        return True

