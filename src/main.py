"""RAGent Application Entrypoint.

This is the main FastAPI application. It wires together all the routes,
middleware, and startup/shutdown lifecycle events.

To run locally:
    uvicorn src.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.middleware import RequestContextMiddleware, RequestLoggingMiddleware
from src.api.routes import auth, chat, health, ingest
from src.config import get_settings
from src.ml import embedding
from src.utils.logger import get_logger
from src.services.observability.telemetry import init_telemetry

# ── Part 1: Application Shell & Lifespan ───────────────────────────────────────

# Grab our custom JSON logger
logger = get_logger("src.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown events.
    
    This replaces the old @app.on_event("startup") decorator.
    Everything before `yield` runs when the server starts.
    Everything after `yield` runs when the server shuts down.
    """
    settings = get_settings()
    
    logger.info(
        "Starting RAGent server",
        extra={
            "environment": settings.environment,
            "log_level": settings.log_level,
        },
    )

    # 1. Initialize OpenTelemetry tracing
    logger.info("Initializing OpenTelemetry...")
    init_telemetry()
    
    # 2. Warm up the embedding model
    # The first time you embed a sentence, it takes ~2 seconds to load the model
    # weights from disk into memory. We do this at startup so the very first
    # user query doesn't experience a massive latency spike.
    try:
        logger.info("Warming up embedding model...")
        embedding.warmup()
        logger.info("Embedding model ready")
    except Exception as exc:
        logger.error(
            "Failed to load embedding model during startup",
            extra={"error": str(exc)},
            exc_info=True,
        )
        # We don't crash the server here, but the first ingest/chat will fail

    # --- Server is now actively running and accepting requests ---
    yield
    # --- Server is now shutting down ---

    logger.info("Shutting down RAGent server")


# Initialize the FastAPI application
app = FastAPI(
    title="RAGent API",
    description="Production-grade RAG assistant with agentic capabilities",
    version="1.0.0",
    lifespan=lifespan,
    # We disable the default swagger UI for production environments
    docs_url="/docs" if get_settings().environment != "production" else None,
    redoc_url=None,
)


# ── Part 2: Middleware & Routes ────────────────────────────────────────────────

# Middleware is executed in the REVERSE order of how it is added.
# We want the RequestContext (ID generator) to run absolutely first, 
# and Logging to run second, so we add Logging first, then Context.

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestContextMiddleware)

# Instrument FastAPI with Prometheus metrics
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

# Instrument FastAPI with OpenTelemetry (Jaeger Traces)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)


# Mount the API routers
# Each router handles a specific domain (e.g. all /chat endpoints)
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(chat.router)


@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    """Root redirect.
    
    If someone visits localhost:8000 in their browser, point them to the docs
    or return a simple greeting so they know the server is alive.
    """
    if get_settings().environment == "production":
        return JSONResponse({"message": "RAGent API is running."})
    
    return JSONResponse(
        {
            "message": "RAGent API is running.",
            "docs": "/docs",
            "health": "/health",
        }
    )


