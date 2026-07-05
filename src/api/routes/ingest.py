"""POST /ingest — document ingestion endpoint.

Accepts a list of file or directory paths, runs the full ingest pipeline,
and returns a summary of what was processed.

Flow:
    POST /ingest  →  validate request  →  pipeline.ingest()  →  IngestResponse
"""

from __future__ import annotations

import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.services.rag import pipeline
from src.storage import document_store
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Router ─────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/ingest", tags=["Ingest"])


# ── Request / Response schemas ─────────────────────────────────────────────────


class IngestRequest(BaseModel):
    """Body for POST /ingest.

    Attributes:
        sources: One or more file paths or directory paths to ingest.
                 Supported file types: .pdf, .txt, .md
                 Directories are walked recursively.

    Example::

        {
          "sources": [
            "data/reports/Q3_Report.pdf",
            "data/manuals/"
          ]
        }
    """

    sources: list[str] = Field(
        ...,
        min_length=1,
        description="File or directory paths to ingest. At least one required.",
        examples=[["data/reports/Q3_Report.pdf", "data/notes/"]],
    )

    @field_validator("sources")
    @classmethod
    def sources_must_not_be_empty_strings(cls, v: list[str]) -> list[str]:
        """Reject requests that contain blank path strings."""
        cleaned = [s.strip() for s in v]
        blanks = [s for s in cleaned if not s]
        if blanks:
            raise ValueError("Sources list must not contain empty strings.")
        return cleaned


class IngestResponse(BaseModel):
    """Response body for POST /ingest.

    Attributes:
        ingested_documents: Number of distinct source files processed.
        total_chunks:       Total number of chunks stored in Qdrant.
        embedding_model:    The embedding model that was used.
        duration_sec:       Wall-clock time for the full ingest pipeline.
        sources:            Resolved list of files that were actually ingested.

    Example::

        {
          "ingested_documents": 3,
          "total_chunks": 87,
          "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
          "duration_sec": 14.2,
          "sources": ["data/reports/Q3_Report.pdf"]
        }
    """

    ingested_documents: int
    total_chunks: int
    embedding_model: str
    duration_sec: float
    sources: list[str]


class IngestedDocumentsResponse(BaseModel):
    """Response body for GET /ingest/documents."""

    documents: list[dict]
    total: int


# ── Routes ─────────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest documents into the vector store",
    description=(
        "Loads documents from the provided file or directory paths, splits them "
        "into chunks, embeds the chunks, and stores them in Qdrant. "
        "Re-ingesting the same file overwrites its previous chunks."
    ),
)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    """POST /ingest — run the full ingest pipeline for the provided sources.

    Args:
        request: JSON body with a ``sources`` list of file/directory paths.

    Returns:
        Summary of the ingestion: document count, chunk count, model, duration.

    Raises:
        422: If the request body is invalid (handled automatically by FastAPI).
        400: If no loadable documents are found at the provided paths.
        500: If embedding or Qdrant upsert fails unexpectedly.
    """
    logger.info(
        "Ingest request received",
        extra={"sources": request.sources, "count": len(request.sources)},
    )

    try:
        result = pipeline.ingest(sources=request.sources)
    except ValueError as exc:
        # Raised by pipeline when sources list is empty or paths don't exist
        logger.warning("Ingest rejected — bad input", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Unexpected failures (Qdrant unreachable, embedding model error, etc.)
        logger.error(
            "Ingest failed with unexpected error",
            extra={"error": str(exc), "sources": request.sources},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Ingest failed due to an internal error. "
                "Check server logs for details."
            ),
        ) from exc

    if result.ingested_documents == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No documents were ingested. Verify that the provided paths exist "
                "and contain supported file types (.pdf, .txt, .md)."
            ),
        )

    logger.info(
        "Ingest request completed",
        extra={
            "ingested_documents": result.ingested_documents,
            "total_chunks": result.total_chunks,
            "duration_sec": result.duration_sec,
        },
    )

    return IngestResponse(
        ingested_documents=result.ingested_documents,
        total_chunks=result.total_chunks,
        embedding_model=result.embedding_model,
        duration_sec=result.duration_sec,
        sources=result.sources,
    )


@router.get(
    "/documents",
    response_model=IngestedDocumentsResponse,
    summary="List all ingested documents",
    description="Returns metadata for every document currently stored in the system.",
)
def list_ingested_documents() -> IngestedDocumentsResponse:
    """GET /ingest/documents — list all documents in the document store.

    Returns:
        List of document metadata records (filename, chunk count, ingested_at, etc.)

    This is useful for building a UI that shows what's been ingested,
    or for debugging whether a document was successfully processed.
    """
    docs = document_store.list_docs()

    logger.info("Listed ingested documents", extra={"total": len(docs)})

    return IngestedDocumentsResponse(
        documents=list(docs),
        total=len(docs),
    )
