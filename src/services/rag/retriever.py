"""Qdrant-backed retrieval logic.

Converts a user's text query into an embedding vector, then searches
the Qdrant vector database for the most semantically similar chunks.

Flow:
    User Query (str) 
      → embed_one() 
      → search Qdrant 
      → parse payloads 
      → list[Chunk]
"""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.ml.embedding import embed_one
from src.services.rag.models import Chunk
from src.storage import vector_db
from src.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve(
    query: str,
    top_k: int | None = None,
    score_threshold: float = 0.0,
    filters: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Retrieve the most relevant document chunks for a given query.

    Args:
        query:           The user's question or search terms.
        top_k:           Max number of chunks to return. Defaults to config setting.
        score_threshold: Minimum cosine similarity score (0.0 to 1.0).
        filters:         Optional Qdrant metadata filters (e.g. {"source": "report.pdf"}).

    Returns:
        List of :class:`Chunk` objects, sorted by relevance (highest score first).
        Returns an empty list if the collection doesn't exist or no matches are found.
    """
    settings = get_settings()
    collection = settings.qdrant_collection
    top_k = top_k or settings.retrieval_top_k

    if not query.strip():
        logger.warning("Empty query provided to retriever")
        return []

    # If nothing has been ingested yet, the collection won't exist
    if not vector_db.collection_exists(collection):
        logger.warning(
            "Vector collection does not exist — ingest documents first",
            extra={"collection": collection},
        )
        return []

    logger.debug(
        "Retrieving chunks",
        extra={"query": query, "top_k": top_k, "collection": collection},
    )

    # 1. Convert the query text into a semantic vector
    query_vector = embed_one(query)

    # 2. Search Qdrant for the closest chunk vectors
    try:
        results = vector_db.search(
            collection=collection,
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )
    except Exception as exc:
        logger.error(
            "Qdrant search failed",
            extra={"error": str(exc), "query": query},
            exc_info=True,
        )
        # We don't want the whole API to crash if search fails, return empty
        return []

    # 3. Parse the Qdrant payloads back into our typed Chunk models
    chunks: list[Chunk] = []
    for hit in results:
        # hit is a dict with "id", "score", and all payload fields
        score = hit.pop("score", 0.0)
        chunk = Chunk.from_payload(hit, score=score)
        chunks.append(chunk)

    logger.info(
        "Retrieval complete",
        extra={
            "retrieved_count": len(chunks),
            "top_score": round(chunks[0].score, 4) if chunks else 0.0,
        },
    )

    return chunks
