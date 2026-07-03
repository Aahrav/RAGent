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

def query(user_input: str, use_agent: bool | None = None) -> QueryResult:
    """Execute the full RAG query pipeline with Guardrails and Agent Routing.

    Steps:
      0. (Phase 2) Pre-flight content safety check.
      0.5. (Phase 3) Semantic router decides between Fast RAG and LangGraph Agent.
      1. Retrieve relevant chunks from the vector store.
      2. Generate an answer using the LLM.
      3. (Phase 2) Evaluate groundedness (hallucination detection).
      4. (Phase 2) Extract strict citations based on confidence threshold.

    Args:
        user_input: The question asked by the user.
        use_agent: Optional override for the semantic router.

    Returns:
        :class:`QueryResult` containing the answer, citations, and metadata.
    """
    start_time = time.perf_counter()
    logger.info("Query started", extra={"query": user_input, "use_agent_override": use_agent})

    # 0. Safety Check
    from src.services.guardrails.safety import check_safety
    try:
        check_safety(user_input)
    except ValueError as e:
        logger.warning("Safety check failed", extra={"error": str(e)})
        return QueryResult(
            answer="I'm sorry, I cannot fulfill this request because it violates safety policies.",
            citations=[],
            latency_ms=_elapsed(start_time) * 1000,
            confidence=0.0,
            fallback_triggered=True
        )

    # =========================================================================
    # ROUTER (Phase 3)
    # =========================================================================
    from src.services.agent.router import route_query
    
    # If the user didn't explicitly override it, use the semantic router
    should_use_agent = use_agent if use_agent is not None else route_query(user_input)
    
    if should_use_agent:
        logger.info("Executing LangGraph Agent Pipeline")
        from langchain_core.messages import HumanMessage
        from src.services.agent.graph import agent_app
        
        # Invoke the LangGraph agent
        final_state = agent_app.invoke({"messages": [HumanMessage(content=user_input)]})
        
        # The final message is the last AI message in the state
        final_message = final_state["messages"][-1].content
        
        # Extract tools used from the state history
        tools_used = []
        for msg in final_state["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    tools_used.append(call["name"])
        
        latency_ms = _elapsed(start_time) * 1000
        return QueryResult(
            answer=final_message,
            citations=[],  # Agent handles its own citing directly in text for now
            latency_ms=latency_ms,
            confidence=1.0,
            fallback_triggered=False,
            tools_used=tools_used
        )

    # =========================================================================
    # FAST RAG PIPELINE (Phase 1 & 2)
    # =========================================================================
    logger.info("Executing Fast RAG Pipeline")
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

    # 3 & 4. Guardrails (Groundedness and Citations)
    from src.services.guardrails.scorer import calculate_groundedness
    from src.services.guardrails.citations import extract_citations
    from src.config import get_settings
    
    settings = get_settings()
    
    overall_confidence, chunk_scores = calculate_groundedness(answer, chunks)
    citations = extract_citations(chunks, chunk_scores, settings.confidence_threshold)
    
    fallback_triggered = False
    if overall_confidence < settings.confidence_threshold:
        logger.warning(
            "Hallucination detected (low confidence)",
            extra={"confidence": overall_confidence, "threshold": settings.confidence_threshold}
        )
        answer = "I'm sorry, I couldn't find a reliable answer in the provided documents."
        citations = []
        fallback_triggered = True

    latency_ms = _elapsed(start_time) * 1000

    logger.info(
        "Query complete",
        extra={
            "latency_ms": round(latency_ms, 2),
            "citations_returned": len(citations),
            "confidence": round(overall_confidence, 4),
            "fallback_triggered": fallback_triggered
        },
    )

    return QueryResult(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms,
        confidence=overall_confidence,
        fallback_triggered=fallback_triggered
    )
