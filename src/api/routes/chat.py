"""POST /chat — query the RAG system.

Accepts a user question, runs the retrieval and generation pipeline,
and returns the answer along with source citations.

Flow:
    POST /chat  →  validate request  →  pipeline.query()  →  ChatResponse
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.services.rag import pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Router ─────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Request / Response schemas ─────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Body for POST /chat.

    Attributes:
        query: The user's question.
        use_agent: Optional flag to force the query to use the Agent pipeline 
            (True) or the standard RAG pipeline (False). If None, the system's 
            semantic router will decide automatically.

    Example::

        {
          "query": "What were the key takeaways from the Q3 report?",
          "use_agent": false
        }
    """

    query: str = Field(
        ...,
        min_length=1,
        description="The question to ask the RAG system.",
        examples=["What were the key takeaways from the Q3 report?"],
    )
    use_agent: bool | None = Field(
        default=None,
        description="Override the automatic router. True forces the Agent, False forces standard RAG.",
    )


class CitationResponse(BaseModel):
    """A source reference for the generated answer."""
    document: str
    source: str
    page: int
    text: str
    score: float


class ChatResponse(BaseModel):
    """Response body for POST /chat.

    Attributes:
        answer:     The generated text response.
        citations:  List of source chunks used to generate the answer.
        latency_ms: Time taken to process the query.
        tools_used: List of tools the Agent executed (if any).
        confidence: Groundedness score (0.0 to 1.0) indicating hallucination risk.
        fallback_triggered: True if the primary answer was blocked by a guardrail.

    Example::

        {
          "answer": "Revenue grew by 15% due to strong enterprise sales.",
          "citations": [
            {
              "document": "Q3_Report.pdf",
              "source": "data/Q3_Report.pdf",
              "page": 4,
              "text": "...enterprise sales drove a 15% revenue increase...",
              "score": 0.892
            }
          ],
          "latency_ms": 1450.2,
          "tools_used": ["rag_search"],
          "confidence": 0.94,
          "fallback_triggered": false
        }
    """
    answer: str
    citations: list[CitationResponse]
    latency_ms: float
    tools_used: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    fallback_triggered: bool = False


# ── Routes ─────────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question",
    description=(
        "Retrieves the most relevant document chunks from the vector database "
        "and uses them to generate an answer. Includes citations indicating "
        "exactly where the information came from."
    ),
)
def chat(request: ChatRequest) -> ChatResponse:
    """POST /chat — execute a RAG query.

    Args:
        request: JSON body containing the user's query string.

    Returns:
        The generated answer, latency, and a list of citations.

    Raises:
        422: If the request body is invalid (handled automatically by FastAPI).
        500: If the retrieval or generation pipeline fails unexpectedly.
    """
    logger.info("Chat request received", extra={"query": request.query})

    try:
        result = pipeline.query(
            user_input=request.query,
            use_agent=request.use_agent
        )
    except Exception as exc:
        logger.error(
            "Chat pipeline failed",
            extra={"error": str(exc), "query": request.query},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your query.",
        ) from exc

    # Map the internal QueryResult to the Pydantic API response
    citations = [
        CitationResponse(
            document=c.document,
            source=c.source,
            page=c.page,
            text=c.text,
            score=c.score,
        )
        for c in result.citations
    ]

    return ChatResponse(
        answer=result.answer,
        citations=citations,
        latency_ms=result.latency_ms,
        tools_used=result.tools_used,
        confidence=result.confidence,
        fallback_triggered=result.fallback_triggered,
    )
