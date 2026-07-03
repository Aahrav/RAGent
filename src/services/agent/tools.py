"""Agent Tools.

These are the functions the LLM can decide to execute during its reasoning loop.
Every tool must have a clear docstring, as the LLM reads the docstring to understand
when and how to use the tool.
"""

from datetime import datetime

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


@tool
def get_current_datetime() -> str:
    """Get the current date and time.
    
    Use this tool when you need to know today's date or the current time
    to answer questions about "today", "yesterday", or time-sensitive events.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Use this tool to perform math calculations safely. 
    Supported operators: +, -, *, /, **, (, )
    
    Args:
        expression: A mathematical string, e.g. "(124 * 3) + 42"
    """
    logger.info("Agent invoked tool: calculator", extra={"expression": expression})
    
    # Restrict characters to prevent arbitrary code execution via eval()
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in mathematical expression. Only numbers and basic operators are allowed."
        
    try:
        # pylint: disable=eval-used
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        logger.error("Calculator tool failed", extra={"error": str(e), "expression": expression})
        return f"Error evaluating expression: {str(e)}"


