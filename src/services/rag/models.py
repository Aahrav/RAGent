"""Shared data models for the RAG pipeline.

All data structures that flow between RAG pipeline stages are defined here.
Every other module imports types from this file — never redefines them.

Flow:
    DocumentLoader → [Document]
        → TextSplitter → [Chunk]
            → Retriever → [Chunk]  (subset, ranked by score)
                → Generator → GeneratorResult
                    → Pipeline → QueryResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Stage 1: Raw document (output of DocumentLoader) ──────────────────────────

@dataclass
class Document:
    """A raw piece of content loaded from a source file.

    One Document typically represents one page (PDF) or one whole file (txt/md).
    The TextSplitter will break Documents into smaller Chunks.

    Attributes:
        text:     The raw text content of this document/page.
        source:   Original file path, e.g. "data/reports/Q3.pdf".
        page:     Page number (1-indexed). For non-paginated files, always 1.
        metadata: Any extra key-value pairs (e.g. author, title from PDF).
    """
    text: str
    source: str
    page: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Strip leading/trailing whitespace — loaders sometimes leave it
        self.text = self.text.strip()

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Document(source={self.source!r}, page={self.page}, text={preview!r}...)"


# ── Stage 2: Chunk (output of TextSplitter, input to Retriever) ───────────────

@dataclass
class Chunk:
    """A fixed-size text segment created by splitting a Document.

    Chunks are what actually get embedded and stored in Qdrant.
    Each Chunk carries enough metadata to trace it back to its source.

    Attributes:
        text:        The chunk text content.
        source:      Original file path (inherited from the parent Document).
        page:        Source page number (inherited from the parent Document).
        chunk_index: Position of this chunk within its parent document (0-indexed).
        doc_id:      The document store ID of the parent document.
        score:       Similarity score from Qdrant (only set after retrieval).
        metadata:    Extra key-value pairs passed through from the Document.
    """
    text: str
    source: str
    page: int = 1
    chunk_index: int = 0
    doc_id: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a dict suitable for storing as a Qdrant point payload.

        Everything in this dict can be retrieved alongside the vector during search.
        """
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
            **self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], score: float = 0.0) -> "Chunk":
        """Reconstruct a Chunk from a Qdrant search result payload."""
        return cls(
            text=payload.get("text", ""),
            source=payload.get("source", ""),
            page=payload.get("page", 1),
            chunk_index=payload.get("chunk_index", 0),
            doc_id=payload.get("doc_id", ""),
            score=score,
            metadata={
                k: v for k, v in payload.items()
                if k not in {"text", "source", "page", "chunk_index", "doc_id"}
            },
        )

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(source={self.source!r}, page={self.page}, "
            f"idx={self.chunk_index}, score={self.score:.3f}, text={preview!r}...)"
        )


# ── Stage 3: Citation (produced by Guardrails, included in QueryResult) ────────

@dataclass
class Citation:
    """A reference pointing to the specific source chunk that supports an answer.

    Attributes:
        document:  Filename of the source (e.g. "Q3_Report.pdf").
        source:    Full source path.
        page:      Page number within the source document.
        text:      The exact chunk text that supports the answer.
        score:     Relevance score from retrieval (cosine similarity, 0–1).
    """
    document: str
    source: str
    page: int
    text: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "source": self.source,
            "page": self.page,
            "text": self.text,
            "score": round(self.score, 4),
        }


# ── Ingest result ──────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    """Summary returned by the pipeline after ingesting documents.

    This is what the POST /ingest endpoint returns to the caller.
    """
    ingested_documents: int
    total_chunks: int
    embedding_model: str
    duration_sec: float
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingested_documents": self.ingested_documents,
            "total_chunks": self.total_chunks,
            "embedding_model": self.embedding_model,
            "duration_sec": round(self.duration_sec, 2),
            "sources": self.sources,
        }


# ── Query result ───────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Full response returned by the pipeline after a /chat query.

    This is what the POST /chat endpoint returns to the caller.
    """
    answer: str
    citations: list[Citation] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    confidence: float = 0.0
    fallback_triggered: bool = False
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "tools_used": self.tools_used,
            "latency_ms": round(self.latency_ms, 1),
            "confidence": round(self.confidence, 4),
            "fallback_triggered": self.fallback_triggered,
            "session_id": self.session_id,
        }
