"""User memory extraction and injection.

This module is responsible for analyzing user messages to extract long-term
facts and preferences, saving them to Qdrant, and retrieving them for future context.
"""
import json
from typing import List

from langchain_core.prompts import PromptTemplate

from src.ml.llm import get_llm
from src.ml.embedding import embed, embed_sparse
from src.storage import vector_db
from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXTRACT_PROMPT = """You are a memory extraction assistant.
Analyze the following user message and extract any explicit personal facts, preferences, or details about the user that should be remembered for future conversations.
If the message contains no personal facts, return an empty JSON list: []
Otherwise, return a JSON list of strings, where each string is a concise fact.

Examples of facts to extract:
- User is a software engineer
- User prefers Python over Java
- User's name is Alice
- User works at Google

User Message: {message}

Respond ONLY with valid JSON (a list of strings). Do not include markdown blocks like ```json.
"""

def extract_and_store_facts(user_id: str, message: str) -> None:
    """Extract facts from the user message and store them in long-term memory."""
    llm = get_llm()
    prompt = PromptTemplate(template=EXTRACT_PROMPT, input_variables=["message"])
    chain = prompt | llm
    
    try:
        response = chain.invoke({"message": message})
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip().strip("`").removeprefix("json").strip()
        
        facts = json.loads(content)
        if not facts or not isinstance(facts, list):
            logger.debug("No facts extracted", extra={"user_id": user_id, "message": message})
            return
            
        settings = get_settings()
        collection = "user_memory"
        
        # Ensure collection exists
        vector_db.ensure_collection(collection, settings.embedding_dim)
        
        # Embed and store
        dense_vectors = embed(facts)
        sparse_vectors = embed_sparse(facts)
        
        payloads = [{"user_id": user_id, "text": fact, "source": "user_memory", "page": 0, "chunk_index": 0} for fact in facts]
        
        vector_db.upsert_points(
            collection=collection,
            vectors=dense_vectors,
            payloads=payloads,
            sparse_vectors=sparse_vectors
        )
        logger.info("Extracted and stored user facts", extra={"user_id": user_id, "facts": facts})
        
    except json.JSONDecodeError:
        logger.warning("Failed to parse facts JSON from LLM", extra={"content": content})
    except Exception as e:
        logger.error("Error extracting user facts", extra={"error": str(e)}, exc_info=True)
