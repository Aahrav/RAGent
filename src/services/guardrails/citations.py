"""Citation Extractor.

Filters retrieved chunks to only return those that actually contributed
to the LLM's generated answer, based on the cosine similarity scores.
"""
from pathlib import Path

from src.services.rag.models import Chunk, Citation
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_citations(
    chunks: list[Chunk], chunk_scores: list[float], threshold: float
) -> list[Citation]:
    """Map the LLM's answer back to specific chunks and return formal citations.
    
    Only chunks with a semantic similarity score above the threshold
    are considered to have been used in the answer.
    
    Args:
        chunks: The original list of chunks retrieved from the DB.
        chunk_scores: The cosine similarity scores from the groundedness scorer.
        threshold: Minimum score required to be cited.
        
    Returns:
        List of formatted Citation objects.
    """
    if len(chunks) != len(chunk_scores):
        logger.error(
            "Mismatch between chunks and scores length",
            extra={"chunks_len": len(chunks), "scores_len": len(chunk_scores)}
        )
        # Fallback: return everything if there is a mismatch
        threshold = -1.0
        
    citations: list[Citation] = []
    
    for chunk, score in zip(chunks, chunk_scores):
        # We only cite chunks that meet the semantic threshold
        if score >= threshold:
            citations.append(
                Citation(
                    document=chunk.metadata.get("filename", Path(chunk.source).name),
                    source=chunk.source,
                    page=chunk.page,
                    text=chunk.text,
                    score=score,
                )
            )
            
    logger.debug(
        "Citations extracted",
        extra={
            "retrieved_chunks": len(chunks),
            "valid_citations": len(citations),
            "threshold": threshold
        }
    )
            
    return citations
