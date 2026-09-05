"""Prometheus Metrics for the RAG pipeline.

This module defines custom business-level metrics to track the performance
and accuracy of our Retrieval-Augmented Generation system.
"""

from prometheus_client import Counter, Histogram

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
    ["status"]  # 'success', 'error', 'safety_blocked'
)

SEMANTIC_CACHE_HITS = Counter(
    "rag_semantic_cache_hits_total",
    "Total semantic cache hits"
)

SEMANTIC_CACHE_MISSES = Counter(
    "rag_semantic_cache_misses_total",
    "Total semantic cache misses"
)

LLM_CONFIDENCE_SCORE = Histogram(
    "rag_llm_confidence_score",
    "Groundedness/Confidence score of LLM answers",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0)
)

RETRIEVAL_CHUNKS_COUNT = Histogram(
    "rag_retrieval_chunks_count",
    "Number of chunks retrieved per query",
    buckets=(0, 1, 3, 5, 10, 20)
)
