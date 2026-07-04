"""Structured JSON logging setup.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened", extra={"request_id": "abc123", "latency_ms": 42})

Every log line is emitted as JSON so it can be parsed by ELK / Loki / any log aggregator.
"""

import logging
import sys
from typing import Any

from src.config import get_settings

# ── JSON formatter ─────────────────────────────────────────────────────────────


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import traceback

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Automatically inject the request ID if we are inside a request context
        try:
            from src.utils.request_context import get_request_id
            req_id = get_request_id()
            if req_id:
                payload["request_id"] = req_id
        except ImportError:
            # Middleware module might not be fully initialized yet
            pass

        # Merge any extra fields the caller passed in
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            } and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = traceback.format_exception(*record.exc_info)

        return json.dumps(payload, default=str)


# ── Root setup ─────────────────────────────────────────────────────────────────

_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.setLevel(level)

    # Remove default handlers to avoid duplicate / unformatted output
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


# ── Public API ─────────────────────────────────────────────────────────────────


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for structured JSON output.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A standard :class:`logging.Logger` that emits JSON to stdout.
    """
    _configure_root_logger()
    return logging.getLogger(name)
