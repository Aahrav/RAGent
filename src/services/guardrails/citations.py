"""Citation Extractor.

Filters retrieved chunks to only return those that actually contributed
to the LLM's generated answer, based on the cosine similarity scores.
"""
from pathlib import Path

from src.services.rag.models import Chunk, Citation
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_citations(chunks: list[Chunk]) -> list[Citation]:
    """Map the retrieved chunks to formatted Citation objects.
    
    Since the LLM-as-a-Judge verifies the answer against the entire context,
    we consider all retrieved chunks as the cited source material.
    
    Args:
        chunks: The original list of chunks retrieved from the DB.
        
    Returns:
        List of formatted Citation objects.
    """
    citations: list[Citation] = []
    
    for chunk in chunks:
        citations.append(
            Citation(
                document=chunk.metadata.get("filename", Path(chunk.source).name),
                source=chunk.source,
                page=chunk.page,
                text=chunk.text,
                score=1.0,  # Score is obsolete since we use an LLM judge
            )
        )
            
    logger.debug(
        "Citations extracted",
        extra={
            "retrieved_chunks": len(chunks),
            "valid_citations": len(citations)
        }
    )
            
    return citations
