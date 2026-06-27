"""Qdrant vector database client wrapper.

Abstracts all Qdrant operations behind a clean interface.
The rest of the codebase never imports qdrant-client directly.

Key concepts:
  - Collection: like a table, stores vectors + payloads
  - Point: one record = vector + payload (metadata) + unique ID
  - Payload: arbitrary JSON metadata attached to each vector
  - Search: find the N most similar vectors to a query vector (cosine similarity)
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Client singleton ───────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Return the cached Qdrant client.

    Connects using QDRANT_URL (and QDRANT_API_KEY if provided).
    """
    settings = get_settings()
    kwargs: dict[str, Any] = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key

    client = QdrantClient(**kwargs)
    logger.info("Qdrant client created", extra={"url": settings.qdrant_url})
    return client


# ── Collection management ──────────────────────────────────────────────────────


def ensure_collection(name: str, vector_size: int) -> None:
    """Create the collection if it doesn't already exist.

    Uses cosine distance — the standard for text embeddings.

    Args:
        name: Collection name (e.g. "ragent_docs").
        vector_size: Dimension of the embedding vectors (e.g. 384).
    """
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info(
            "Qdrant collection created",
            extra={"collection": name, "vector_size": vector_size},
        )
    else:
        logger.debug("Qdrant collection already exists", extra={"collection": name})


def collection_exists(name: str) -> bool:
    """Return True if the collection exists."""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    return name in existing


# ── Write ──────────────────────────────────────────────────────────────────────


def upsert_points(
    collection: str,
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    ids: list[str] | None = None,
) -> list[str]:
    """Store vectors with their metadata payloads.

    Args:
        collection: Target collection name.
        vectors: List of embedding vectors (one per document chunk).
        payloads: List of metadata dicts — same length as vectors.
                  Should contain at least: text, source, page, chunk_index.
        ids: Optional list of string IDs. Generated (UUID4) if not provided.

    Returns:
        List of IDs that were upserted.
    """
    if not vectors:
        return []

    if ids is None:
        ids = [str(uuid.uuid4()) for _ in vectors]

    points = [
        qmodels.PointStruct(
            id=_str_to_uuid(point_id),
            vector=vector,
            payload=payload,
        )
        for point_id, vector, payload in zip(ids, vectors, payloads)
    ]

    client = get_client()
    client.upsert(collection_name=collection, points=points, wait=True)

    logger.debug(
        "Upserted points to Qdrant",
        extra={"collection": collection, "count": len(points)},
    )
    return ids


# ── Read / Search ──────────────────────────────────────────────────────────────


def search(
    collection: str,
    query_vector: list[float],
    top_k: int = 5,
    score_threshold: float = 0.0,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Find the top-k most similar vectors to the query vector.

    Args:
        collection: Collection to search in.
        query_vector: Embedding of the user's query.
        top_k: Number of results to return.
        score_threshold: Minimum cosine similarity score (0.0 = no filter).
        filters: Optional Qdrant filter dict for metadata-based pre-filtering.

    Returns:
        List of dicts with keys: ``id``, ``score``, and all payload fields
        (``text``, ``source``, ``page``, ``chunk_index``).
    """
    client = get_client()

    qdrant_filter = None
    if filters:
        conditions = [
            qmodels.FieldCondition(
                key=k,
                match=qmodels.MatchValue(value=v),
            )
            for k, v in filters.items()
        ]
        qdrant_filter = qmodels.Filter(must=conditions)

    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold if score_threshold > 0 else None,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    hits = []
    for r in results:
        hit = {"id": str(r.id), "score": r.score}
        if r.payload:
            hit.update(r.payload)
        hits.append(hit)

    logger.debug(
        "Qdrant search complete",
        extra={"collection": collection, "top_k": top_k, "hits": len(hits)},
    )
    return hits


def ping() -> bool:
    """Return True if Qdrant is reachable."""
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False


# ── Helpers ────────────────────────────────────────────────────────────────────


def _str_to_uuid(s: str) -> str:
    """Qdrant requires UUIDs or unsigned ints as IDs. Convert any string."""
    try:
        uuid.UUID(s)
        return s
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))
