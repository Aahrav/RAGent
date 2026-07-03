"""Agent Tools.

These are the functions the LLM can decide to execute during its reasoning loop.
Every tool must have a clear docstring, as the LLM reads the docstring to understand
when and how to use the tool.
"""

from duckduckgo_search import DDGS
from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)


@tool
def web_search(query: str) -> str:
    """Search the live internet for up-to-date information.
    
    Use this tool when you need current events, weather, stock prices,
    or information that is likely not contained in the local database.
    
    Args:
        query: The search query string.
        
    Returns:
        A concatenated string of the top search result snippets.
    """
    logger.info("Agent invoked tool: web_search", extra={"query": query})
    
    try:
        # DDGS is the free DuckDuckGo search client
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            
        if not results:
            return "No results found on the web."
            
        # Format the results into a readable string for the LLM context
        formatted_results = []
        for i, res in enumerate(results, start=1):
            title = res.get("title", "No Title")
            body = res.get("body", "No content")
            formatted_results.append(f"Result {i} ({title}): {body}")
            
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error("Web search tool failed", extra={"error": str(e), "query": query})
        return f"Error performing web search: {str(e)}"
