# TRACE: Timeline Reasoning Agent for Continuity & Evidence - System Context and Architecture

## 1. The Core Concept
You're building a system that ingests a huge fictional universe (chapters, character sheets, story bible) and, when the author describes a new scene, checks whether it logically contradicts anything already established — wrong location, wrong timeline, dead character reappearing, impossible travel time, etc. 

**Key Design Decision:** Facts get extracted into structured, queryable timelines, not just embedded as text. Similarity search alone can't reliably catch contradictions because the contradicting sentence often doesn't share vocabulary with the new scene.

Everything below either supports getting facts in accurately (ingestion), retrieving the right facts reliably (retrieval), reasoning about them correctly (reasoning/agentic layer), proving it works (evaluation), or running it like a real service (infra/ops).

## 2. Ingestion & Extraction Layer

*   **Structured Fact Extraction (Planned):** Every chapter/scene gets run through an LLM extraction step that pulls out: characters present, location, status/state changes, and time markers, output as structured JSON (character, location, status, chapter, time_marker). This is the foundation — without it, you only have prose, and prose can't be queried precisely.
*   **Time-Expression Normalization (Planned):** A dedicated step converts fuzzy expressions ("three weeks later") into a comparable, sortable value per storyline thread, so two events from different chapters can be placed on the same timeline.
*   **Entity Resolution & Disambiguation (Planned):** Decides which specific entity a mention actually refers to (resolving doppelgangers, reused names, etc.), so timelines don't merge different characters into one.
*   **Async/Queue-Based Ingestion Pipeline (Planned):** Ingesting a massive story bible is slow. Running it as background jobs in a queue (rather than a blocking script) means the author can keep working while ingestion happens, and workers can scale horizontally.
*   **Versioned Knowledge Base (Planned):** Authors edit chapters. When a chapter gets rewritten, old extracted facts become stale. Requires supporting re-ingestion that invalidates/supersedes old timeline entries.

## 3. Storage Layer

*   **Two Parallel Stores (Planned):**
    *   *Vector Store:* Embeddings of raw prose, for fuzzy/thematic questions.
    *   *Structured Timeline/Graph Store:* The actual character-location-status-time table, queried directly (not by similarity) for contradiction-checking.
*   **Vector DB Choice Justification:** (Currently Qdrant). Must be able to explain the tradeoff — filtering support, hybrid search compatibility, scaling behavior.
*   **Embedding Model Versioning (Planned):** Having a plan (re-embed everything, or maintain a mapping) to prevent silent retrieval degradation if the embedding model is upgraded.

## 4. Retrieval Layer

*   **Hybrid Retrieval (Already Implemented):** Combines sparse keyword-style search (BM25) with dense semantic search.
*   **Semantic Caching (Already Implemented):** Serves cached results for semantically similar questions, saving cost and latency.
*   **Query Rewriting (Already Implemented):** Turns underspecified author questions into clearer, retrieval-friendly queries before search runs.
*   **Re-ranking (Planned):** A cross-encoder re-ranks the hybrid candidate set for actual relevance to the query to tighten precision right before generation.
*   **Contextual/Parent-Child Chunking (Planned):** Retrieve small, precise chunks for matching, but expand each match to its full surrounding scene before handing it to the LLM.
*   **Multi-Hop / Iterative Retrieval (Planned):** The agent can issue follow-up retrievals based on what the first pass found (e.g., retrieve X's status -> retrieve travel-time -> retrieve escape scene).
*   **Self-RAG / Retrieval-Necessity Check (Planned):** The agent decides whether retrieval is even needed before searching, saving cost and context pollution.
*   **GraphRAG / Structured Knowledge Graph (Planned):** Formalization of the timeline store as a proper graph (character → location → time → status edges), enabling direct traversal queries.

## 5. Reasoning / Agentic Layer

*   **Agentic RAG (Already Implemented):** The agent decides what to do next — retrieve more, check the timeline store, verify against another source.
*   **LangGraph Orchestration (Already Implemented):** Chosen for deterministic pipelines with conditional branches, giving explicit, inspectable control flow.
*   **LLM-as-Judge (Already Implemented):** An LLM evaluates the quality/correctness of the system's own output at runtime and offline.
*   **Short-Term Memory (Already Implemented):** Keeps context across a multi-turn conversation with the author without re-establishing everything from scratch.
*   **Hallucination/Groundedness Detection (Planned):** Verifies each specific claim in the output actually traces back to a retrieved fact.
*   **Confidence Scoring / Abstention (Planned):** When timeline data is incomplete or ambiguous, the system should say "not enough information to confirm" rather than force a confident answer.

## 6. Model Strategy

*   **Fine-Tuned Small Models (Planned):** Extraction and time-normalization are narrow, repetitive tasks perfect for a fine-tuned small model instead of paying frontier-model prices.
*   **Model Routing (Planned):** Simple fact lookups route to a cheap model; complex multi-step contradiction reasoning goes to a larger model.
*   **Speculative Decoding (Planned, minor):** A small draft model proposes tokens that a larger model verifies in bulk, speeding up generation.

## 7. Serving & Inference Infra

*   **vLLM (Planned):** For self-hosted models, vLLM gives continuous batching and efficient KV-cache memory management.
*   **Quantization (Planned):** Reduces memory footprint for self-hosted models.
*   **Containerization + Orchestration (Planned):** Docker/Kubernetes for deployment (Docker already heavily in use).
*   **Load Balancing + Autoscaling (Planned):** Multiple model replicas scaling based on queue depth or request volume.
*   **Circuit Breakers / Fallback Chains (Planned):** If the primary LLM API fails, fall back to a secondary model or cached response.

## 8. Observability, Security, Ops

*   **Streaming (Already Implemented):** Responses stream token-by-token to the author.
*   **Telemetry + Prometheus (Already Implemented):** Metrics on latency, request volume, error rates exposed for monitoring.
*   **Distributed Tracing (Planned Upgrade):** Extends observability to full trace spans across the agent's multi-step chain (currently basic tracing exists, planned to deepen).
*   **Token/Cost Tracking (Planned):** Extends Prometheus to track dollars-per-query.
*   **Auth and Rate Limiting (Already Implemented):** API protection.
*   **Input Validation / Schema Enforcement (Planned):** Structured extraction outputs get validated against a schema; malformed JSON triggers a retry.
*   **Prompt-Injection Defenses (Planned):** Guards against embedded instructions in uploaded content.

## 9. Evaluation

*   **RAGAS-Based Evaluation (Already Implemented):** Automated metrics (faithfulness, answer relevance, context precision/recall).
*   **Golden Dataset / Regression Test Set (Planned):** A curated set of deliberately planted real contradictions plus near-misses to measure precision and recall.
*   **CI/CD with Eval Wired In (Planned):** The golden dataset eval runs automatically on every code change.
*   **Human-in-the-Loop Feedback (Planned):** When an author marks a flagged contradiction as a false positive, it feeds back into eval/fine-tuning data.
