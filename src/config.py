"""Application configuration via pydantic-settings.

All settings are read from environment variables (or a .env file).
Use get_settings() everywhere — it is cached so the file is parsed once.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for RAGent.

    Reads from environment variables; falls back to .env file if present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM provider ──────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "ollama", "anthropic"] = Field(
        default="ollama",
        description="Which LLM backend to use.",
    )
    llm_model: str = Field(
        default="llama3",
        description="Model name passed to the provider.",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key.")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama server.",
    )

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key.")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL.",
    )
    qdrant_api_key: str = Field(default="", description="Qdrant API key (cloud only).")
    qdrant_collection: str = Field(
        default="ragent_docs",
        description="Default Qdrant collection name.",
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL.",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model name.",
    )
    embedding_dim: int = Field(
        default=384,
        description="Output dimension of the embedding model.",
    )

    # ── RAG tuning ────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=500, description="Max characters per chunk.")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks.")
    retrieval_top_k: int = Field(default=5, description="Number of chunks to retrieve.")
    confidence_threshold: float = Field(
        default=0.65,
        description="Minimum groundedness score before fallback is triggered.",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_secret_key: str = Field(
        default="changeme-replace-in-production",
        description="Secret key for signing tokens.",
    )
    api_keys: str = Field(
        default="",
        description="Comma-separated list of valid API keys.",
    )
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    app_version: str = Field(default="0.1.0")

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_ttl_seconds: int = Field(
        default=300,
        description="How long to cache query responses (seconds).",
    )
    cache_enabled: bool = Field(default=True)

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_requests: int = Field(
        default=60,
        description="Max requests per window per API key.",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate limit window in seconds.",
    )

    # ── Observability ─────────────────────────────────────────────────────────
    otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OpenTelemetry collector gRPC endpoint.",
    )
    enable_tracing: bool = Field(default=False)
    enable_metrics: bool = Field(default=True)

    @field_validator("api_keys")
    @classmethod
    def parse_api_keys(cls, v: str) -> str:
        # Store as-is; use get_api_keys_list() helper below
        return v

    def get_api_keys_list(self) -> list[str]:
        """Return the list of valid API keys parsed from the comma-separated string."""
        if not self.api_keys:
            return []
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Use this everywhere instead of constructing Settings() directly.
    """
    return Settings()
