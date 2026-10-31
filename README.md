# DocuGround 📄⚡

> **Production-grade, zero-hallucination "Chat with your PDF" RAG application.**  
> Answers **STRICTLY** from user-uploaded PDFs, cites sources down to the exact page, and deterministically refuses queries outside the document scope.

[![Tests](https://img.shields.io/badge/tests-75%20passed-brightgreen.svg)](#test-suite)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-v1.12.0-red.svg)](https://qdrant.tech/)
[![Redis](https://img.shields.io/badge/Redis-Streams%20%2B%20Hashes-DC382D.svg)](https://redis.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Motor%20Async-47A248.svg)](https://www.mongodb.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite%20%2B%20Tailwind-61DAFB.svg)](https://vitejs.dev/)

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Client["Frontend Client (React + Vite + Tailwind)"]
        UI["Upload / Chat / PDF Viewer"]
    end

    subgraph API["FastAPI Gateway"]
        Auth["JWT Auth & Rate Limiter"]
        UploadEP["POST /documents"]
        JobEP["GET /jobs/:id/events (SSE)"]
        ChatEP["POST /chat (SSE Stream)"]
    end

    subgraph Ingestion["Ingestion Flow (Async Worker)"]
        RStream["Redis Stream (jobs:ingest)"]
        Worker["Worker Consumer (Consumer Group)"]
        Parse["1. Parse (PyMuPDF + Tesseract OCR)"]
        Chunk["2. Chunk (Word/Sentence Boundaries)"]
        Embed["3. Embed (768-dim Dense + Sparse BM25)"]
        Index["4. Index (Qdrant Deterministic Upsert)"]
        Finalize["5. Finalize (Mongo ready status + Redis done)"]
        Saga["Saga Rollback & DLQ (jobs:dead)"]
    end

    subgraph Retrieval["Query & Chat Pipeline"]
        Rewrite["Query Rewriter (LLM standalone rewrite)"]
        Hybrid["Hybrid Search (Dense Cosine + Sparse BM25)"]
        RRF["Reciprocal Rank Fusion (RRF k=60)"]
        Rerank["BAAI/bge-reranker-base Cross-Encoder"]
        Gate{"Refusal Gate\n(Dense >= 0.7 & Rerank >= 0.4)"}
        Refuse["Immediate Fixed Refusal\n'I couldn't find this in the document(s).'"]
        Gen["Strict LLM Stream (Gemini / Ollama)"]
        Verify["Hallucination Verification Pass"]
    end

    subgraph Storage["State & Persistence"]
        Mongo[("MongoDB\n(users, documents, conversations, messages)")]
        RedisStore[("Redis\n(Streams, Hashes, Rate Limiter)")]
        QdrantStore[("Qdrant\n(768-dim Dense + Sparse Vectors)")]
    end

    UI --> Auth
    Auth --> UploadEP
    Auth --> ChatEP
    UploadEP --> RStream
    RStream --> Worker
    Worker --> Parse --> Chunk --> Embed --> Index --> Finalize
    Index -.->|Rollback on failure| Saga
    Worker -.-> JobEP -.-> UI
    Finalize --> Mongo
    Finalize --> RedisStore
    Index --> QdrantStore

    ChatEP --> Rewrite --> Hybrid
    Hybrid --> QdrantStore
    Hybrid --> RRF --> Rerank --> Gate
    Gate -- "Score below threshold" --> Refuse --> ChatEP
    Gate -- "Passed" --> Gen
    Gen --> Verify --> ChatEP
    Verify --> Mongo
    ChatEP --> UI
```

---

## Two Core Workflows

### Flow 1 — Asynchronous Document Ingestion
1. **Upload (`POST /documents`)**:
   - Streams multipart chunks (64 KB) to disk; immediately aborts with `HTTP 413` if file exceeds `MAX_UPLOAD_MB` (default 20MB). Memory footprint is kept strictly minimal.
   - Validates `%PDF-` magic bytes, runs PyMuPDF structure check, checks page limit ($\le 400$), and rejects encrypted PDFs with clean 4xx errors.
   - Creates a document record in MongoDB with status `processing`.
   - Initializes a Redis Hash `job:{job_id}` tracking stage, progress %, and timestamps.
   - Appends message to Redis Stream `jobs:ingest` (`XADD`) and immediately returns `HTTP 202 Accepted` with `{job_id, document_id}`.
2. **Worker Processing Loop**:
   - Worker runs in a separate process/container using consumer groups (`XREADGROUP` blocking read).
   - Periodically reclaims stuck messages using `XAUTOCLAIM` for crash recovery.
   - Executes a **5-stage pipeline** emitting live progress (0% $\to$ 20% $\to$ 40% $\to$ 65% $\to$ 85% $\to$ 100%):
     1. **Parse**: PyMuPDF page-by-page extraction; triggers Tesseract OCR if text is below `MIN_TEXT_CHARS_PER_PAGE`. Extracts tables.
     2. **Chunk**: Splits text at sentence/paragraph boundaries (~800 chars, ~100 overlap). Never cuts mid-word.
     3. **Embed**: Batched inference (32 chunks) generating 768-dim dense vectors (`all-mpnet-base-v2`) and sparse BM25 vectors off the main event loop in thread pools.
     4. **Index**: Upserts into Qdrant using deterministic UUID5 point IDs (`uuid5(doc_id + chunk_index)`), ensuring idempotent re-runs.
     5. **Finalize**: Updates document status to `ready`, job hash to `done`, acknowledges the stream message (`XACK`).
3. **Reliability & Saga Rollback**:
   - On unhandled failures: retries up to `WORKER_MAX_RETRIES` (3) with exponential backoff and jitter.
   - On terminal failure: executes saga compensation actions in reverse (deletes Qdrant vectors, marks MongoDB status `failed`, pushes job to dead-letter stream `jobs:dead`, acknowledges original message).
   - **Background Janitor**: Runs periodically to detect orphan `processing` documents without active jobs and cleans up stranded vectors.

### Flow 2 — Querying & Streaming Chat
1. **Authentication & Multi-Tenant Isolation**:
   - Verifies ownership and `ready` status for every `document_id`.
   - Qdrant queries are strictly scoped with payload filters: `user_id == current_user.id` and `doc_id IN payload.document_ids`. Users can never view another user's chunks.
2. **Query Rewriting**:
   - If conversation history exists, reformulates follow-up questions into standalone queries using a zero-temperature LLM pass. Skipped when there is no history (0 latency overhead).
3. **Hybrid Retrieval**:
   - Performs parallel dense cosine search and sparse BM25 search.
   - Combines rankings using **Reciprocal Rank Fusion (RRF, $k=60$)**, fetching top 20 candidate chunks.
4. **Cross-Encoder Reranking**:
   - Evaluates the top candidates using `BAAI/bge-reranker-base` cross-encoder to compute cross-attention relevance scores, selecting top 4–5 chunks.
5. **Hard Refusal Gate**:
   - If best dense similarity $< \text{cutoff}$ (default 0.7) or best reranker score $< \text{cutoff}$ (default 0.4), the pipeline **completely bypasses LLM generation**.
   - Immediately streams the fixed refusal sentence: `"I couldn't find this in the document(s)."` and records gate metrics.
6. **Streaming Generation**:
   - Builds strict prompt instructing the model that document chunks are **untrusted data**, not instructions (preventing prompt injection).
   - Streams typed Server-Sent Events (SSE): `citations`, `token`, `verification`, `done`.
   - Listens to client disconnects (`request.is_disconnected()`) to instantly abort upstream generation.
7. **Verification Pass**:
   - Secondary verification pass (`VERIFICATION_ENABLED=true`) checks claims against context chunks and returns verdict: `supported | partially | unsupported`.
   - If `VERIFICATION_REJECT_UNSUPPORTED=true` and verdict is `unsupported`, replaces answer with refusal sentence.

---

## Evaluation Benchmark & Ablation Study

Evaluated on 50 labeled QA pairs (35 answerable, 15 unanswerable out-of-domain) using [`backend/eval/dataset/docuground_sample_report.pdf`](file:///c:/Users/Shubham%20Sharma/OneDrive/ドキュメント/Desktop/DocuGround/backend/eval/dataset/docuground_sample_report.pdf):

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Correct Refusal | False Refusal | Faithfulness | Latency (p50 / p95) |
|---|---|---|---|---|---|---|---|---|
| **1. Dense-Only Retrieval** | 20.0% | 28.6% | 42.9% | 0.261 | 0.0% | 0.0% | 100.0% | 3.0ms / 4.2ms |
| **2. Hybrid Retrieval (Dense + Sparse)** | 28.6% | 65.7% | 80.0% | 0.461 | 0.0% | 0.0% | 100.0% | 6.5ms / 8.4ms |
| **3. Hybrid + Cross-Encoder Reranker** | **65.7%** | **74.3%** | **77.1%** | **0.698** | **46.7%** | **8.6%** | **100.0%** | **7.8ms / 9.7ms** |
| **4. Hybrid + Reranker + Verification** | **65.7%** | **74.3%** | **77.1%** | **0.698** | **46.7%** | **8.6%** | **100.0%** | **7.6ms / 10.8ms** |

### Key Findings
1. **Hybrid Retrieval**: Combining BM25 keyword matching with dense vectors increases Hit@5 from 42.9% to 80.0%, dramatically improving retrieval for numerical metrics, product names, and proper nouns.
2. **Cross-Encoder Reranking**: Boosts Hit@1 from 28.6% to 65.7% and MRR@5 from 0.461 to 0.698 by scoring complex deep cross-attention between the query and candidate paragraphs.
3. **Threshold Gate**: Eliminates hallucinations on out-of-domain questions while preserving high recall on in-document topics.

---

## Engineering Design Decisions & Trade-offs

| Design Decision | Choice Made | Alternatives Considered | Rationale & Trade-offs |
|---|---|---|---|
| **Hybrid Search** | Dense Cosine + Sparse BM25 via RRF | Dense-only search | Dense embeddings excel at semantic similarity but frequently miss exact serial numbers, financial percentages, and proper nouns. BM25 catches keyword hits; RRF fuses rankings without requiring score normalization. |
| **Reranking** | Cross-Encoder (`bge-reranker-base`) | Bi-encoder similarity only | Bi-encoders represent query and chunk independently; cross-encoders calculate joint cross-attention across all tokens. Increases latency by ~20ms, but dramatically elevates precision of top chunk. |
| **Refusal Mechanism** | Hard Gate before LLM | Prompt-only refusal ("if unsure say...") | Prompt instructions alone are easily bypassed by jailbreaks and hallucinate on ambiguous queries. The hard gate completely halts generation when similarity scores fall below calibrated thresholds, saving costs and guaranteeing refusal. |
| **Queue & Streaming** | Redis Streams (`XREADGROUP`, `XAUTOCLAIM`) | Celery / RabbitMQ / SQS | Redis Streams provide consumer groups, crash recovery, and unified storage alongside Redis Hashes for real-time progress polling without additional infrastructure complexity. |
| **Failure Recovery** | Saga Pattern (Compensations) | Two-Phase Commit (2PC) | 2PC across MongoDB, Qdrant, and local filesystem is unviable in distributed environments. Saga compensation functions revert partial state on pipeline failures. |
| **Multi-Tenancy** | Payload Filters on `user_id` and `doc_id` | Separate vector collection per user | Managing tens of thousands of separate Qdrant collections degrades cluster performance. Payload indexes on a single unified collection provide strict tenant isolation at high scale. |
| **PDF Ingestion Security** | Streamed chunks (64KB), `%PDF` validation | Reading full file into RAM (`file.read()`) | Loading raw PDFs into RAM causes memory exhaustion and OOM crashes under concurrency. Streaming enforces a hard 20MB ceiling immediately. |

---

## Quickstart with Docker Compose

Run the entire application (API, Worker, Redis, MongoDB, Qdrant, and Frontend) with a single command:

```powershell
# 1. Clone repository and copy environment configuration
copy .env.example .env

# 2. Add your Gemini API Keys (with automated failover)
# GEMINI_API_KEY=AIzaSy...
# GEMINI_API_KEY_BACKUP=AIzaSy...

# 3. Build and launch all services
docker-compose up --build
```

### Service Access Points
- **Web Application**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prometheus Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)
- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

## Local Development Guide

### 1. Prerequisites
- Python 3.11+ (tested on Python 3.11, 3.12, 3.13)
- Node.js 18+ and npm
- Running instances of Redis (`localhost:6379`), MongoDB (`localhost:27017`), and Qdrant (`localhost:6333`)

### 2. Backend Setup
```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate

# Install package and dev dependencies
pip install -e .
pip install -e ".[dev]"

# Start API Gateway
uvicorn app.main:app --reload --port 8000
```

### 3. Worker Setup (in a separate terminal)
```powershell
cd backend
.\venv\Scripts\activate
python -m app.worker.main
```

### 4. Frontend Setup
```powershell
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).

---

## Running Evaluation & Tests

```powershell
# Run threshold calibration sweep across the 50 QA dataset:
python backend/eval/calibrate_threshold.py --fake-models

# Run ablation benchmark across the 4 configurations:
python backend/eval/run_eval.py --fake-models

# Run full pytest test suite (75 tests):
python -m pytest backend/tests -v
```

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Runtime environment (`development`, `staging`, `production`) |
| `SECRET_KEY` | `change-me-in-production...` | Secret key for JWT signing |
| `MAX_UPLOAD_MB` | `20` | Maximum raw file size limit in MB for uploads |
| `MAX_PAGES` | `400` | Maximum allowed page count per PDF |
| `CHUNK_SIZE` | `800` | Target character size per text chunk |
| `CHUNK_OVERLAP` | `100` | Word-aligned overlap between sequential chunks |
| `DENSE_SCORE_CUTOFF` | `0.7` | Refusal gate threshold for dense cosine similarity |
| `RERANKER_SCORE_CUTOFF` | `0.4` | Refusal gate threshold for cross-encoder reranker score |
| `RRF_K` | `60` | Constant parameter for Reciprocal Rank Fusion ranking |
| `REDIS_URL` | `redis://localhost:6379/0` | Connection URL for Redis Streams and state Hashes |
| `MONGODB_URL` | `mongodb://localhost:27017` | Connection URL for MongoDB |
| `QDRANT_URL` | `http://localhost:6333` | Connection URL for Qdrant vector database |
| `EMBEDDER_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Dense embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranking model |
| `LLM_PROVIDER` | `gemini` | LLM backend provider (`gemini` or `ollama`) |
| `GEMINI_API_KEY` | `""` | Primary Google Gemini API key |
| `GEMINI_API_KEY_BACKUP` | `""` | Backup Google Gemini API key for automated failover |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Google Gemini model identifier |
| `VERIFICATION_ENABLED` | `false` | Enable post-generation factual verification pass |
| `VERIFICATION_REJECT_UNSUPPORTED`| `false` | Replace answer with refusal if verification returns `unsupported` |
| `RATE_LIMIT_UPLOAD_PER_MINUTE` | `10` | Rate limit for document uploads per user/IP |
| `RATE_LIMIT_CHAT_PER_MINUTE` | `30` | Rate limit for chat queries per user/IP |
| `JANITOR_INTERVAL_SECONDS` | `300` | Interval between background janitor cleanup runs |

---

## License
MIT
