"""FastAPI Middleware.

Middleware intercepts every incoming HTTP request before it hits the route,
and every outgoing response before it leaves the server.

This file contains two critical backend patterns:
1. Request ID Context: Assigns a unique ID to every request so logs can be traced.
2. Logging Middleware: Automatically logs the start, end, and duration of requests.
"""

import time
import uuid
from contextvars import ContextVar

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from prometheus_client import Counter, Histogram, Gauge

from src.utils.logger import get_logger

logger = get_logger(__name__)


from src.utils.request_context import set_request_id, reset_request_id

# ── Part 1: Request ID Middleware ──────────────────────────────────────────────

class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique ID to every incoming request.
    
    It also checks if the client sent an 'X-Request-ID' header and uses that
    if present (useful for microservice tracing).
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        
        # 1. Look for an existing ID from the client, or generate a new UUID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # 2. Set the context variable for the duration of this request
        token = set_request_id(request_id)
        
        try:
            # 3. Pass control to the rest of the application
            response = await call_next(request)
            
            # 4. Attach the ID to the outgoing response headers so the client sees it
            response.headers["X-Request-ID"] = request_id
            return response
            
        finally:
            # 5. Clean up the context variable
            reset_request_id(token)


# ── Part 2: Request Logging & Timing Middleware ────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs the start, end, and duration of every request.
    
    This provides an automatic "access log" for the API. It records the URL,
    HTTP method, status code, and exactly how many milliseconds the request took.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        
        start_time = time.perf_counter()
        
        # Don't log healthchecks to avoid spamming the logs every 10 seconds
        is_healthcheck = request.url.path == "/health"
        
        if not is_healthcheck:
            logger.info(
                "Request started",
                extra={
                    "method": request.method,
                    "url": str(request.url.path),
                    "client": request.client.host if request.client else "unknown",
                }
            )

        try:
            # Wait for the route (/chat, /ingest) to process the request
            response = await call_next(request)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if not is_healthcheck:
                logger.info(
                    "Request completed",
                    extra={
                        "method": request.method,
                        "url": str(request.url.path),
                        "status_code": response.status_code,
                        "latency_ms": round(latency_ms, 2),
                    }
                )
            
            return response

        except Exception as exc:
            # If the application crashed entirely, log the crash and the latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Request failed with unhandled exception",
                extra={
                    "method": request.method,
                    "url": str(request.url.path),
                    "status_code": 500,
                    "latency_ms": round(latency_ms, 2),
                    "error": str(exc),
                },
                exc_info=True
            )
            raise  # Re-raise so FastAPI can handle returning the 500 to the client

# ── Part 3: Prometheus Metrics Middleware ──────────────────────────────────────

# Define our Prometheus metrics at the module level so they are only registered once
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)

ACTIVE_REQUESTS = Gauge(
    "http_requests_active",
    "Number of active HTTP requests",
    ["method", "endpoint"]
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks HTTP request metrics for Prometheus."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        
        # Don't track metrics for the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        # We group endpoints by their path. In a REST API, you'd want to normalize paths (e.g. /users/{id}) 
        # but for this simple agent, request.url.path is fine.
        endpoint = request.url.path

        ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception:
            # If there's an unhandled exception, it's a 500 Internal Server Error
            status_code = "500"
            raise
        finally:
            duration = time.perf_counter() - start_time
            
            # Record metrics
            ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
            
        return response
