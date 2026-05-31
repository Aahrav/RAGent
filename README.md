# RAGent — Production-Grade RAG Assistant with Agentic Capabilities

A comprehensive AI system that combines Retrieval-Augmented Generation (RAG), tool calling, hallucination control, and full observability — designed to showcase deep MLOps and system design skills.

---

## Overview

RAGent is an end-to-end AI assistant that:

- **Answers questions** over your private documents (PDF, text, markdown)
- **Decides autonomously** when to use internal knowledge vs. external tools (search, calculator, etc.)
- **Provides citations** and verifies answers to reduce hallucinations
- **Includes caching, rate limiting, authentication**, and full observability (logs, metrics, traces)
- **Containerized** and ready for cloud deployment

Built with FastAPI, LangChain/LangGraph, Qdrant, and Redis.

---

## System Architecture

```
Clients (Web UI, CLI, Slack Bot)
        │
        ▼
┌──────────────────────┐
│   API Gateway (FastAPI)  │
│  Rate Limiting → Auth    │
│  → Request ID → Routing  │
└──────────┬───────────────┘
           │
     ┌─────┼──────┐
     ▼     ▼      ▼
   RAG   Tool   Direct
  Pipeline Agent  Response
     │     │
     ▼     ▼
  Qdrant  Web Search,
  Vector  Calculator,
  DB      Custom APIs
     │
     ▼
  Redis Cache
  Observability
  (Logs, Metrics, Traces)
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **Vector DB (Qdrant)** | Stores document embeddings; similarity search |
| **Redis Cache** | Caches frequent queries/responses; rate-limit data |
| **LangGraph Agent** | Orchestrates tool use, reasoning, and state |
| **Hallucination Guardrails** | Verifies answers against sources, scores confidence |
| **Router/Classifier** | Routes queries: RAG vs Tool vs Direct answer |
| **Observability** | Structured logging, Prometheus metrics, distributed traces |

---

## Project Structure

```
ragent/
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
│
├── src/
│   ├── main.py                    # FastAPI app entry
│   ├── config.py                  # All configurations
│   │
│   ├── api/                       # API layer
│   │   ├── routes/
│   │   │   ├── chat.py             # /chat endpoint
│   │   │   ├── ingest.py           # /ingest endpoint
│   │   │   └── health.py           # /health endpoint
│   │   └── middleware.py          # Rate limiting, logging
│   │
│   ├── core/                      # Core business logic
│   │   ├── router.py              # Query classification
│   │   ├── cache.py               # Redis cache layer
│   │   └── security.py            # Auth, rate limiting
│   │
│   ├── services/
│   │   ├── rag/                   # RAG pipeline
│   │   │   ├── pipeline.py
│   │   │   ├── document_loader.py
│   │   │   ├── text_splitter.py
│   │   │   ├── retriever.py
│   │   │   ├── generator.py
│   │   │   └── evaluator.py
│   │   ├── agent/                 # LangGraph agent
│   │   │   ├── agent.py
│   │   │   ├── tools.py
│   │   │   └── state.py
│   │   └── guardrails/            # Hallucination & safety
│   │       ├── hallucination.py
│   │       └── safety.py
│   │
│   ├── storage/                   # Storage layer
│   │   ├── vector_db.py
│   │   ├── document_store.py
│   │   └── cache.py
│   │
│   ├── ml/                        # ML components
│   │   ├── embedding.py
│   │   └── llm.py
│   │
│   └── utils/                     # Utilities
│       ├── logger.py
│       ├── metrics.py
│       └── tracing.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── notebooks/
│   ├── evaluation.ipynb
│   └── experiments.ipynb
│
├── docs/
│   ├── system_design.md
│   ├── api_docs.md
│   └── deployment.md
│
└── scripts/
    ├── ingest_docs.py
    ├── evaluate.py
    └── benchmark.py
```

---

## Features (Progressive Phases)

### Phase 1 — Core RAG
- [ ] Document loader (PDF, txt, markdown)
- [ ] Text chunking (recursive, semantic)
- [ ] Embedding + indexing to Qdrant
- [ ] Retrieval + generation pipeline (basic Q&A)

### Phase 2 — Hallucination Control
- [ ] Return source citations with answers
- [ ] Implement groundedness scoring
- [ ] Fallback for low-confidence responses

### Phase 3 — Tool Calling (Agentic)
- [ ] Add built-in tools: web search, calculator, date/time
- [ ] Implement LangGraph agent that decides tool usage
- [ ] Enable multi-step reasoning (e.g., search → calculate → answer)

### Phase 4 — Production Polish
- [ ] Streaming responses (token-by-token)
- [ ] Query preprocessing (rewriting, expansion)
- [ ] Evaluation harness (RAGAS metrics: faithfulness, answer relevance)
- [ ] Dockerize all services; docker-compose for local dev
- [ ] Basic auth + rate limiting middleware
- [ ] Structured logging + Prometheus metrics endpoint

### Phase 5 — Observability & Reliability
- [ ] Distributed tracing (OpenTelemetry) → Jaeger/Weave
- [ ] Alerting on latency/error spikes
- [ ] Chaos testing (e.g., latency injection)
- [ ] Performance benchmarking scripts

---

## API Contracts

### POST `/chat`

**Request:**
```json
{
  "query": "What was our Q3 revenue according to the latest report?",
  "session_id": "optional-uuid",
  "use_tools": true
}
```

**Response:**
```json
{
  "answer": "According to the Q3 report, revenue was $12.4M, up 8% YoY.",
  "citations": [
    {
      "document": "Q3_Report.pdf",
      "page": 4,
      "text": "Revenue for Q3 2024 reached $12.4 million..."
    }
  ],
  "tools_used": [],
  "latency_ms": 1120,
  "confidence": 0.93
}
```

### POST `/ingest`

**Request:**
```json
{
  "sources": ["data/manuals/", "data/reports/Q3_Report.pdf"]
}
```

**Response:**
```json
{
  "ingested_documents": 12,
  "total_chunks": 345,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "duration_sec": 18.4
}
```

---

## Getting Started

1. Clone the repo
2. Copy `.env.example` → `.env` and fill in:
   - `OPENAI_API_KEY` (or your preferred LLM provider)
   - `QDRANT_URL` & `QDRANT_API_KEY` (or use local Docker)
   - `REDIS_URL`
3. Run:
   ```bash
   docker compose up -d   # starts Qdrant + Redis
   pip install -r requirements.txt
   python src/main.py     # starts FastAPI on http://localhost:8000
   ```
4. Test:
   ```bash
   curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"query":"Explain photosynthesis in simple terms."}'
   ```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI |
| LLM Orchestration | LangChain + LangGraph |
| Vector DB | Qdrant |
| Cache | Redis |
| Embeddings | sentence-transformers |
| Observability | Prometheus, Grafana, OpenTelemetry |
| Deployment | Docker, Docker Compose |

---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Blog](https://blog.langchain.dev/langgraph/)
- [Qdrant Vector DB](https://qdrant.tech/)
- [RAGAS Evaluation Metrics](https://github.com/explodinggradients/ragas)
- [Prometheus + Grafana Tutorial](https://grafana.com/docs/grafana/latest/)
- [OpenTelemetry for Python](https://opentelemetry.io/docs/instrumentation/python/)

---

*This project is currently in the design/planning phase. See `AI_System_Design.md` for the full system design document.*
