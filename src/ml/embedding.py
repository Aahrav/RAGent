"""Embedding model wrapper.

Loads sentence-transformers once at startup (singleton pattern).
All other modules call embed() — they never touch the model directly.

Model: sentence-transformers/all-MiniLM-L6-v2
  - Output dim: 384
  - Fast, lightweight, strong general-purpose embeddings
  - Runs fully locally — no API calls, no cost
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from src.config import get_settings
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding

logger = get_logger(__name__)

# ── Singleton ──────────────────────────────────────────────────────────────────

_model: "SentenceTransformer | None" = None
_lock = threading.Lock()


def _get_model() -> "SentenceTransformer":
    """Load the embedding model once and cache it for the lifetime of the process."""
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:  # double-checked locking
            return _model

        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        logger.info(
            "Loading embedding model",
            extra={"model": settings.embedding_model},
        )
        _model = SentenceTransformer(settings.embedding_model)
        logger.info(
            "Embedding model loaded",
            extra={"model": settings.embedding_model, "dim": settings.embedding_dim},
        )

    return _model

_sparse_model: "SparseTextEmbedding | None" = None
_sparse_lock = threading.Lock()

def _get_sparse_model() -> "SparseTextEmbedding":
    """Load the sparse embedding model (SPLADE)."""
    global _sparse_model
    if _sparse_model is not None:
        return _sparse_model

    with _sparse_lock:
        if _sparse_model is not None:
            return _sparse_model

        from fastembed import SparseTextEmbedding

        logger.info("Loading sparse embedding model", extra={"model": "prithivida/Splade_PP_en_v1"})
        _sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        logger.info("Sparse embedding model loaded")

    return _sparse_model


# ── Public API ─────────────────────────────────────────────────────────────────


def embed(texts: list[str]) -> list[list[float]]:
    """Convert a list of text strings into embedding vectors.

    Args:
        texts: List of strings to embed. Can be a single item.

    Returns:
        List of float vectors, one per input text.
        Each vector has length ``settings.embedding_dim`` (384 for MiniLM-L6-v2).

    Example:
        >>> vectors = embed(["Hello world", "How are you?"])
        >>> len(vectors)        # 2
        >>> len(vectors[0])     # 384
    """
    if not texts:
        return []

    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    """Convenience wrapper — embed a single string and return one vector."""
    return embed([text])[0]


def embed_sparse(texts: list[str]):
    """Convert a list of text strings into sparse embedding vectors (SPLADE).
    
    Returns:
        List of SparseEmbedding objects (with .indices and .values).
    """
    if not texts:
        return []

    model = _get_sparse_model()
    # fastembed returns a generator, convert to list
    return list(model.embed(texts))


def embed_sparse_one(text: str):
    """Convenience wrapper — embed a single string and return one sparse vector."""
    return embed_sparse([text])[0]


def warmup() -> None:
    """Pre-load the model so the first real request isn't slow.

    Call this during application startup.
    """
    _get_model()
    _get_sparse_model()
