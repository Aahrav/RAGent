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

from src.config import get_settings
from src.ml import embedding
from src.utils.logger import get_logger

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

    # 1. Warm up the embedding model
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

