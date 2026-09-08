"""Document metadata store.

Tracks every document that has been ingested into the vector DB.
Persists metadata to a JSON file on disk — simple, human-readable,
requires no extra database.

Stored per document:
  - doc_id      : deterministic hash of the source path (stable across runs)
  - source      : original file path string
  - filename    : basename of the file
  - file_type   : "pdf" | "txt" | "md"
  - chunk_count : how many chunks were created and stored in Qdrant
  - ingested_at : ISO-8601 timestamp of the ingestion
  - file_size_bytes : size of the source file at ingestion time

Why JSON file instead of a database?
  - Zero extra dependencies
  - Human-readable — you can open it and see exactly what's stored
  - Easy to reset (just delete the file)
  - Can be swapped for SQLite or Postgres later without changing the interface
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Types ──────────────────────────────────────────────────────────────────────


class DocMeta(TypedDict):
    """Metadata record for one ingested document."""

    doc_id: str           # SHA-256 of the source path (first 16 chars)
    source: str           # full path as string
    filename: str         # basename, e.g. "report.pdf"
    file_type: str        # "pdf", "txt", or "md"
    chunk_count: int      # number of chunks stored in Qdrant
    ingested_at: str      # ISO-8601 UTC timestamp
    file_size_bytes: int  # file size at ingestion time


# ── Store path ─────────────────────────────────────────────────────────────────

def _store_path() -> Path:
    """Return the path to the JSON store file.

    Creates the parent directory if it doesn't exist yet.
    Default: <project_root>/data/doc_store.json
    """
    # Walk up from this file to find the project root (where pyproject.toml lives)
    here = Path(__file__).resolve()
    project_root = here.parents[3]   # src/storage/document_store.py → project root
    store_dir = project_root / "data"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / "doc_store.json"


# ── Thread safety ──────────────────────────────────────────────────────────────

_lock = threading.Lock()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_store() -> dict[str, DocMeta]:
    """Read the JSON file and return all records as a dict keyed by doc_id.

    Returns an empty dict if the file doesn't exist yet.
    IMPORTANT: Always call this inside ``_lock``.
    """
    path = _store_path()
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # data is a dict: { doc_id -> DocMeta }
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.error(
            "Failed to read doc store — returning empty store",
            extra={"path": str(path), "error": str(exc)},
        )
        return {}


def _save_store(store: dict[str, DocMeta]) -> None:
    """Write the full store dict back to the JSON file.

    IMPORTANT: Always call this inside ``_lock``.
    """
    path = _store_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.error(
            "Failed to write doc store",
            extra={"path": str(path), "error": str(exc)},
        )
        raise


# ── Public helpers ─────────────────────────────────────────────────────────────

def make_doc_id(source: str) -> str:
    """Create a stable, short ID for a document from its source path.

    Uses UUID5 of the path string.
    This is deterministic — the same path always produces the same ID.

    Example:
        >>> make_doc_id("data/reports/Q3.pdf")
        'b79148d8-795a-5e7e-8c34-eb58d4a6de14'
    """
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source))


# ── Public API ─────────────────────────────────────────────────────────────────

def save_doc_meta(
    source: str,
    chunk_count: int,
) -> DocMeta:
    """Create or overwrite the metadata record for a document.

    Call this after a successful ingestion so the store reflects the latest state.

    Args:
        source: The full file path of the ingested document.
        chunk_count: How many chunks were stored in Qdrant.

    Returns:
        The ``DocMeta`` dict that was written to disk.

    Raises:
        OSError: If the JSON file cannot be written.
    """
    path = Path(source)
    file_size = path.stat().st_size if path.exists() else 0
    suffix = path.suffix.lstrip(".").lower()

    meta: DocMeta = {
        "doc_id": make_doc_id(source),
        "source": source,
        "filename": path.name,
        "file_type": suffix if suffix in {"pdf", "txt", "md"} else "unknown",
        "chunk_count": chunk_count,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": file_size,
    }

    with _lock:
        store = _load_store()
        store[meta["doc_id"]] = meta
        _save_store(store)

    logger.info(
        "Document metadata saved",
        extra={
            "doc_id": meta["doc_id"],
            "file_name": meta["filename"],
            "chunk_count": chunk_count,
        },
    )
    return meta


def get_doc_meta(source: str) -> DocMeta | None:
    """Return the metadata for a document by its source path.

    Args:
        source: The file path that was used during ingestion.

    Returns:
        The ``DocMeta`` dict, or ``None`` if this document hasn't been ingested.
    """
    doc_id = make_doc_id(source)
    with _lock:
        store = _load_store()
    return store.get(doc_id)


def get_doc_meta_by_id(doc_id: str) -> DocMeta | None:
    """Return the metadata for a document by its ``doc_id``.

    Args:
        doc_id: The 16-character hex ID returned by :func:`make_doc_id`.

    Returns:
        The ``DocMeta`` dict, or ``None`` if not found.
    """
    with _lock:
        store = _load_store()
    return store.get(doc_id)


def list_docs() -> list[DocMeta]:
    """Return all ingested documents, sorted by ingestion time (newest first).

    Returns:
        List of ``DocMeta`` dicts. Empty list if nothing has been ingested yet.
    """
    with _lock:
        store = _load_store()

    docs = list(store.values())
    docs.sort(key=lambda d: d["ingested_at"], reverse=True)
    return docs


def is_ingested(source: str) -> bool:
    """Return True if this source path has already been ingested.

    Useful for skipping re-ingestion of unchanged documents.

    Args:
        source: File path to check.
    """
    return get_doc_meta(source) is not None


def delete_doc_meta(source: str) -> bool:
    """Remove the metadata record for a document.

    Call this when you delete a document's chunks from Qdrant.

    Args:
        source: The file path of the document to remove.

    Returns:
        ``True`` if the record existed and was deleted, ``False`` if not found.
    """
    doc_id = make_doc_id(source)
    with _lock:
        store = _load_store()
        if doc_id not in store:
            return False
        del store[doc_id]
        _save_store(store)

    logger.info("Document metadata deleted", extra={"doc_id": doc_id, "source": source})
    return True


def clear_all() -> int:
    """Delete all document metadata records (resets the store).

    Returns:
        Number of records that were deleted.

    Warning:
        This only clears the metadata store. It does NOT delete chunks from Qdrant.
        Use with caution — only for dev/testing resets.
    """
    with _lock:
        store = _load_store()
        count = len(store)
        _save_store({})

    logger.warning("Doc store cleared", extra={"deleted_count": count})
    return count
