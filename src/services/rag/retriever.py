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

import concurrent.futures
from typing import Any

from src.config import get_settings
from src.ml.embedding import embed_one
from src.services.rag.models import Chunk
from src.storage import vector_db
from src.utils.logger import get_logger

logger = get_logger(__name__)


from langsmith import traceable

@traceable(name="qdrant_retriever")
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


def reciprocal_rank_fusion(results_list: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    """Fuse multiple retrieval results using Reciprocal Rank Fusion (RRF).
    
    Args:
        results_list: A list containing lists of retrieved chunks for each query.
        k: Smoothing constant for RRF.
        
    Returns:
        A list of deduplicated chunks, ranked by their fused score.
    """
    fused_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for results in results_list:
        for rank, chunk in enumerate(results):
            # Create a unique ID for the chunk since it doesn't have an explicit one
            chunk_id = f"{chunk.doc_id}_{chunk.chunk_index}"
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
                chunk_map[chunk_id] = chunk
            # Rank is 0-indexed, so we add 1
            fused_scores[chunk_id] += 1.0 / (rank + 1 + k)

    # Sort chunks by their fused score in descending order
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_results = []
    for chunk_id, score in sorted_items:
        chunk = chunk_map[chunk_id]
        chunk.score = score
        fused_results.append(chunk)

    return fused_results


@traceable(name="multi_query_retriever")
def multi_retrieve(
    queries: list[str],
    top_k: int | None = None,
    score_threshold: float = 0.0,
    filters: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Retrieve chunks for multiple queries in parallel and fuse them with RRF.
    
    Args:
        queries: A list of search query variations.
        top_k: Max chunks to return.
        
    Returns:
        A deduplicated list of top_k Chunk objects.
    """
    if not queries:
        return []

    settings = get_settings()
    final_top_k = top_k or settings.retrieval_top_k

    results_list = []
    
    # Run Qdrant retrieval for all queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = [
            executor.submit(
                retrieve, 
                query=q, 
                top_k=final_top_k, 
                score_threshold=score_threshold, 
                filters=filters
            )
            for q in queries
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                if results:
                    results_list.append(results)
            except Exception as e:
                logger.error("Parallel retrieval failed", extra={"error": str(e)}, exc_info=True)
                
    # Deduplicate and re-rank the results
    fused_results = reciprocal_rank_fusion(results_list)
    
    logger.info(
        "Multi-query retrieval complete",
        extra={
            "queries_run": len(queries),
            "total_fused_chunks": len(fused_results),
            "returning_top_k": final_top_k
        }
    )
    
    return fused_results[:final_top_k]
