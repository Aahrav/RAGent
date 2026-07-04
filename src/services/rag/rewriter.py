"""Query Rewriter (Multi-Query Retrieval).

Takes a vague user query and uses an LLM to expand it into multiple variations.
This massively improves retrieval by covering different synonyms and semantic structures.
"""

from langchain_core.prompts import PromptTemplate

from src.ml.llm import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

REWRITE_PROMPT = """You are an AI language model assistant. Your task is to generate 3 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Do not include numbering, bullet points, or introductory text. Just output the 3 alternative questions separated by newlines.

Original question: {question}
"""

def generate_multi_queries(question: str) -> list[str]:
    """Generate multiple variations of a search query using the LLM.
    
    Args:
        question: The raw user input string.
        
    Returns:
        A list of unique search queries (always starting with the original query).
    """
    llm = get_llm()
    prompt = PromptTemplate(template=REWRITE_PROMPT, input_variables=["question"])
    
    chain = prompt | llm
    
    logger.debug("Rewriting query for Multi-Query Retrieval", extra={"original": question})
    
    try:
        response = chain.invoke({"question": question})
        content = response.content if hasattr(response, "content") else str(response)
        
        # Split by newline and remove any markdown list artifacts like "1." or "-"
        queries = []
        for line in content.split("\n"):
            clean = line.strip(" -*0123456789.")
            if clean:
                queries.append(clean)
                
        # Always include the original question at the very front
        all_queries = [question] + queries
        
        # Deduplicate while preserving order
        unique_queries = list(dict.fromkeys(all_queries))
        
        logger.info(
            "Query expanded successfully", 
            extra={
                "original": question, 
                "expanded": unique_queries,
                "count": len(unique_queries)
            }
        )
        return unique_queries
        
    except Exception as e:
        logger.error("Failed to rewrite query, falling back to original", extra={"error": str(e)}, exc_info=True)
        return [question]
