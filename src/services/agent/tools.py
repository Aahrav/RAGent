"""Agent Tools.

These are the functions the LLM can decide to execute during its reasoning loop.
Every tool must have a clear docstring, as the LLM reads the docstring to understand
when and how to use the tool.
"""

from duckduckgo_search import DDGS
from langchain_core.tools import tool

from src.services.rag import retriever
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


@tool
def rag_search(query: str) -> str:
    """Search the internal database for proprietary company information.
    
    Use this tool when you need information about internal projects (e.g. Project Apollo),
    internal policies, or proprietary documents that are not on the public internet.
    
    Args:
        query: The search query string.
        
    Returns:
        A concatenated string of the retrieved document chunks.
    """
    logger.info("Agent invoked tool: rag_search", extra={"query": query})
    
    try:
        chunks = retriever.retrieve(query)
        if not chunks:
            return "No information found in the internal database."
            
        formatted_chunks = []
        for i, chunk in enumerate(chunks, start=1):
            # Extract just the filename if possible, otherwise use the full source
            source_name = chunk.metadata.get("filename", chunk.source.split("/")[-1])
            formatted_chunks.append(f"Document {i} ({source_name}): {chunk.text}")
            
        return "\n\n".join(formatted_chunks)
        
    except Exception as e:
        logger.error("RAG search tool failed", extra={"error": str(e), "query": query})
        return f"Error performing internal search: {str(e)}"

