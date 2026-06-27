"""RAG pipeline: ingest and query orchestration.

This module is the conductor of the RAG system.
It calls the individual modules in the correct order and passes data between them.

──────────────────────────────────────────────────────────
INGEST FLOW (POST /ingest):

    sources (file paths)
        │
        ▼
    document_loader.load_documents()     ← read files from disk
        │  [list[Document]]
        ▼
    text_splitter.split_documents()      ← cut into fixed-size chunks
        │  [list[Chunk]]
        ▼
    embedding.embed()                    ← convert text → vectors (batched)
        │  [list[list[float]]]
        ▼
    vector_db.ensure_collection()        ← create Qdrant collection if needed
    vector_db.upsert_points()            ← store vectors + payloads
        │
        ▼
    document_store.save_doc_meta()       ← record ingestion metadata
        │
        ▼
    IngestResult                         ← returned to the API route

──────────────────────────────────────────────────────────
QUERY FLOW (POST /chat) — added in Step 1.4:
    Will be added below after ingest is complete.
──────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from pathlib import Path

from src.config import get_settings
from src.ml.embedding import embed
from src.services.rag import generator, retriever
from src.services.rag.document_loader import load_documents
from src.services.rag.models import Chunk, Citation, IngestResult, QueryResult
from src.services.rag.text_splitter import split_documents
from src.storage import document_store, vector_db
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Maximum number of chunks to embed in one batch.
# Larger batches are faster but use more RAM.
# 64 is a safe default for the MiniLM-L6-v2 model on CPU.
_EMBED_BATCH_SIZE = 64


# ── Ingest pipeline ────────────────────────────────────────────────────────────

def ingest(sources: list[str]) -> IngestResult:
    """Ingest documents from file paths or directories into the vector store.

    Steps:
      1. Load documents from disk (PDF, txt, md).
      2. Split documents into fixed-size chunks with overlap.
      3. Embed all chunks in batches.
      4. Ensure the Qdrant collection exists (create if needed).
      5. Upsert all chunk vectors + payloads into Qdrant.
      6. Save per-document metadata to the document store.

    Args:
        sources: List of file paths or directory paths to ingest.
                 Directories are walked recursively for supported file types.

    Returns:
        :class:`IngestResult` with counts, model name, and duration.

    Raises:
        ValueError: If ``sources`` is empty.
        RuntimeError: If embedding or Qdrant upsert fails.

    Example:
        >>> result = ingest(["data/reports/Q3.pdf", "data/notes/"])
        >>> result.ingested_documents
        4
        >>> result.total_chunks
        87
        >>> result.duration_sec
        12.4
    """
    if not sources:
        raise ValueError("No sources provided — pass at least one file or directory path.")

    settings = get_settings()
    start_time = time.perf_counter()

    logger.info("Ingest started", extra={"sources": sources})

    # ── Step 1: Load documents from disk ──────────────────────────────────────
    logger.info("Step 1/5 — Loading documents from disk")
    docs = load_documents(sources)

    if not docs:
        logger.warning(
            "No documents were loaded — check that the source paths exist "
            "and contain supported file types (.pdf, .txt, .md)",
            extra={"sources": sources},
        )
        return IngestResult(
            ingested_documents=0,
            total_chunks=0,
            embedding_model=settings.embedding_model,
            duration_sec=_elapsed(start_time),
            sources=sources,
        )

    logger.info(
        "Documents loaded",
        extra={"count": len(docs), "sources": len(sources)},
    )

    # ── Step 2: Split documents into chunks ────────────────────────────────────
    logger.info("Step 2/5 — Splitting documents into chunks")
    chunks: list[Chunk] = split_documents(docs)

    if not chunks:
        logger.warning("Splitting produced zero chunks — check document content")
        return IngestResult(
            ingested_documents=len(docs),
            total_chunks=0,
            embedding_model=settings.embedding_model,
            duration_sec=_elapsed(start_time),
            sources=sources,
        )

    logger.info(
        "Chunks created",
        extra={
            "chunk_count": len(chunks),
            "avg_chars": int(sum(len(c.text) for c in chunks) / len(chunks)),
        },
    )

    # ── Step 3: Embed all chunks (batched) ─────────────────────────────────────
    logger.info(
        "Step 3/5 — Embedding chunks",
        extra={"total": len(chunks), "batch_size": _EMBED_BATCH_SIZE},
    )

    all_vectors: list[list[float]] = _embed_in_batches(
        [c.text for c in chunks],
        batch_size=_EMBED_BATCH_SIZE,
    )

    logger.info("Embedding complete", extra={"vectors": len(all_vectors)})

    # ── Step 4: Ensure Qdrant collection exists ────────────────────────────────
    logger.info("Step 4/5 — Ensuring Qdrant collection exists")
    vector_db.ensure_collection(
        name=settings.qdrant_collection,
        vector_size=settings.embedding_dim,
    )

    # ── Step 5: Upsert vectors + payloads into Qdrant ─────────────────────────
    logger.info(
        "Step 5/5 — Upserting into Qdrant",
        extra={"collection": settings.qdrant_collection},
    )

    payloads = [chunk.to_payload() for chunk in chunks]

    vector_db.upsert_points(
        collection=settings.qdrant_collection,
        vectors=all_vectors,
        payloads=payloads,
        # IDs are auto-generated (UUID4) — no need to pass them
    )

    # ── Save document metadata ─────────────────────────────────────────────────
    # Group chunks by source to count chunks per document
    chunks_per_source: dict[str, int] = {}
    for chunk in chunks:
        chunks_per_source[chunk.source] = chunks_per_source.get(chunk.source, 0) + 1

    for source_path, chunk_count in chunks_per_source.items():
        document_store.save_doc_meta(
            source=source_path,
            chunk_count=chunk_count,
        )

    duration = _elapsed(start_time)

    result = IngestResult(
        ingested_documents=len(chunks_per_source),
        total_chunks=len(chunks),
        embedding_model=settings.embedding_model,
        duration_sec=duration,
        sources=list(chunks_per_source.keys()),
    )

    logger.info(
        "Ingest complete",
        extra={
            "ingested_documents": result.ingested_documents,
            "total_chunks": result.total_chunks,
            "duration_sec": round(duration, 2),
        },
    )

    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _embed_in_batches(
    texts: list[str],
    batch_size: int = _EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed a large list of texts in fixed-size batches.

    Why batch?
      The sentence-transformers model processes texts in parallel internally,
      so a batch of 64 is much faster than 64 individual calls.
      Batching also prevents OOM errors on machines with limited RAM.

    Args:
        texts:      All texts to embed.
        batch_size: Number of texts per batch.

    Returns:
        Flat list of embedding vectors, same order as ``texts``.
    """
    all_vectors: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch = texts[start_idx:end_idx]

        logger.debug(
            "Embedding batch",
            extra={
                "batch": f"{batch_num + 1}/{total_batches}",
                "size": len(batch),
            },
        )

        vectors = embed(batch)
        all_vectors.extend(vectors)

    return all_vectors


def _elapsed(start: float) -> float:
    """Return seconds elapsed since ``start`` (from ``time.perf_counter()``)."""
    return round(time.perf_counter() - start, 2)


# ── Query pipeline ───────────────────────────────────────────────────────────────

def query(user_input: str) -> QueryResult:
    """Execute the full RAG query pipeline.

    Steps:
      1. Retrieve relevant chunks from the vector store.
      2. Generate an answer using the LLM.
      3. Map the retrieved chunks into Citation objects.
      4. (Phase 2) Evaluate hallucination / safety.

    Args:
        user_input: The question asked by the user.

    Returns:
        :class:`QueryResult` containing the answer, citations, and metadata.
    """
    start_time = time.perf_counter()
    logger.info("Query started", extra={"query": user_input})

    # 1. Retrieve
    chunks = retriever.retrieve(query=user_input)

    if not chunks:
        # Fallback if the database is empty or nothing matches
        return QueryResult(
            answer="I don't have any ingested documents to answer that question.",
            citations=[],
            latency_ms=_elapsed(start_time) * 1000,
        )

    # 2. Generate
    answer = generator.generate_answer(
        query=user_input,
        context_chunks=chunks,
    )

    # 3. Build citations (simple mapping for Phase 1)
    # In Phase 2, this will be replaced by the hallucination evaluator
    # which will only cite the specific chunks actually used in the answer.
    citations: list[Citation] = []
    for chunk in chunks:
        citations.append(
            Citation(
                document=chunk.metadata.get("filename", Path(chunk.source).name),
                source=chunk.source,
                page=chunk.page,
                text=chunk.text,
                score=chunk.score,
            )
        )

    latency_ms = _elapsed(start_time) * 1000

    logger.info(
        "Query complete",
        extra={
            "latency_ms": round(latency_ms, 2),
            "citations_returned": len(citations),
        },
    )

    return QueryResult(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms,
    )
