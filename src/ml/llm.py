"""LLM provider wrapper.

Acts as a factory for LangChain chat models.
Supports multiple providers seamlessly based on the LLM_PROVIDER env var.
The rest of the app just calls get_llm() and doesn't care which provider is active.

Supported providers:
  - ollama    (Local, free, private)
  - openai    (Cloud, state-of-the-art)
  - anthropic (Cloud, excellent context handling)
  - openrouter (Cloud aggregator, any model)
  - huggingface (Cloud inference endpoints)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from src.config import get_settings
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)

# ── Singleton ──────────────────────────────────────────────────────────────────

_llm: "BaseChatModel | None" = None
_lock = threading.Lock()


def get_llm() -> "BaseChatModel":
    """Return a configured LangChain ChatModel instance.

    The instance is created once (singleton) based on the current configuration,
    and reused for all subsequent calls.

    Returns:
        A subclass of ``BaseChatModel`` (e.g., ``ChatOpenAI``, ``ChatOllama``).

    Raises:
        ValueError: If the configured provider is not supported.
    """
    global _llm
    if _llm is not None:
        return _llm

    with _lock:
        if _llm is not None:  # double-checked locking
            return _llm

        settings = get_settings()
        provider = settings.llm_provider.lower()
        model_name = settings.llm_model

        logger.info(
            "Initializing LLM provider",
            extra={"provider": provider, "model": model_name},
        )

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            
            if not settings.openai_api_key:
                logger.warning("OpenAI API key is missing from config")

            _llm = ChatOpenAI(
                model=model_name,
                api_key=settings.openai_api_key,
                temperature=0,  # 0 for factual RAG answers
            )

        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            
            _llm = ChatOllama(
                model=model_name,
                base_url=settings.ollama_base_url,
                temperature=0,
            )

        elif provider == "anthropic":
            # Note: Requires `langchain-anthropic` package to be installed if used
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:
                raise ImportError(
                    "langchain-anthropic is required for the anthropic provider. "
                    "Run: pip install langchain-anthropic"
                ) from exc

            if not settings.anthropic_api_key:
                logger.warning("Anthropic API key is missing from config")

            _llm = ChatAnthropic(
                model_name=model_name,
                api_key=settings.anthropic_api_key,
                temperature=0,
            )

        elif provider == "openrouter":
            from langchain_openai import ChatOpenAI
            
            if not settings.openrouter_api_key:
                logger.warning("OpenRouter API key is missing from config")

            _llm = ChatOpenAI(
                model=model_name,
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0,
            )

        elif provider == "huggingface":
            try:
                from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            except ImportError as exc:
                raise ImportError(
                    "langchain-huggingface is required for the huggingface provider. "
                    "Run: pip install langchain-huggingface"
                ) from exc
                
            if not settings.huggingface_api_key:
                logger.warning("HuggingFace API key is missing from config")
                
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=settings.huggingface_api_key,
                temperature=0.1,  # HF endpoints often fail with exactly 0.0
            )
            _llm = ChatHuggingFace(llm=llm_endpoint)

        else:
            raise ValueError(
                f"Unsupported LLM provider '{provider}'. "
                f"Supported: openai, ollama, anthropic, openrouter, huggingface"
            )

        logger.info("LLM initialized successfully")

    return _llm
