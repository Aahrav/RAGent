"""GET /health — system healthcheck endpoint.

Used by load balancers, Kubernetes, or Docker to verify the application
is running and its dependencies (like Qdrant) are reachable.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, status
from pydantic import BaseModel

from src.config import get_settings
from src.storage import vector_db
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Router ─────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["Health"])


# ── Response Schema ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    environment: str
    qdrant_connected: bool
    timestamp: float


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System healthcheck",
    description="Returns the operational status of the API and its dependencies.",
)
def check_health() -> HealthResponse:
    """GET /health — verify system connectivity.

    Pings the Qdrant database to ensure the storage layer is accessible.
    Even if Qdrant is down, this endpoint returns 200 OK so the API itself
    isn't marked as dead, but the `qdrant_connected` flag will be False.
    """
    settings = get_settings()
    
    # Check Qdrant connectivity
    try:
        # vector_db.ping() isn't explicitly implemented yet, but we can check if 
        # the client is alive by fetching collections
        vector_db.get_client().get_collections()
        qdrant_ok = True
    except Exception as exc:
        logger.warning(
            "Healthcheck: Qdrant unreachable",
            extra={"error": str(exc)},
        )
        qdrant_ok = False

    return HealthResponse(
        status="healthy" if qdrant_ok else "degraded",
        environment=settings.environment,
        qdrant_connected=qdrant_ok,
        timestamp=time.time(),
    )
