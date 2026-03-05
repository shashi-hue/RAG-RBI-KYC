---
title: RBI KYC Compliance Assistant
emoji: ⚖️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

<h1 align="center">RAG System for RBI KYC Master Direction</h1>

<p align="center">
  A production-grade Retrieval-Augmented Generation system for querying the
  RBI Master Direction on Know Your Customer (KYC), 2016 — updated through 2025.
  Every answer is grounded strictly in the regulatory document with inline paragraph citations.

  The system preserves the full document structure — chapters, parts, paragraphs, amendment history, and footnote metadata
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/shashi-hue/RAG-RBI-KYC">🟢 Live Demo (HuggingFace Spaces)</a> &nbsp;|&nbsp;
  <a href="https://www.rbi.org.in/CommonPerson/english/scripts/notification.aspx?id=2607">📄 Source Document</a>
</p>

---


## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## What This Project Does

Regulatory documents like the KYC Master Direction are dense, heavily amended, and cross-referenced. Standard RAG approaches lose critical structure — chapter boundaries, amendment status, deleted provisions, and table relationships. This system addresses that by:

- **Preserving document hierarchy** in every chunk (chapter, part, paragraph, footnotes)
- **Tracking amendment status** — each chunk is tagged as `active`, `amended`, `inserted`, or `deleted`
- **Retaining historical text** for deleted provisions, enabling audit and compliance history queries
- **Parsing structured tables** (Annex IV FPI document matrix) as row-level chunks with category columns
- **Extracting repealed circulars** from the Appendix for cross-reference lookups

The result is a system that can answer not just "what does the regulation say" but also "what did it used to say", "which provision was amended by which circular", and "what documents does a Category III FPI need" — all with inline citations back to the source text.

---

## Architecture

```
PDF (KYC Master Direction 2025)
  │
  ├─ parser.py ──────────► chunks.jsonl (144 chunks with chapter/para/status/footnotes)
  ├─ definitions.py ─────► per-term definition chunks (split from Para 3)
  ├─ annex_iv.py ────────► annex_iv.jsonl (18 FPI document table rows)
  └─ appendix.py ────────► repealed_circulars.json (255 repealed circulars)
        │
        ▼
  embed.py ──────────────► Qdrant (hybrid: dense BGE-large + sparse BM25)
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  FastAPI                                    │
  │                                             │
  │  query ─► router.py (regex + LLM fallback)  │
  │            │                                │
  │            ├─ FPI_DOCS → annex_iv sources   │
  │            ├─ HISTORICAL → include deleted  │
  │            ├─ CHAPTER → scoped retrieval    │
  │            └─ GENERAL → full hybrid search  │
  │            │                                │
  │            ▼                                │
  │  retriever.py (hybrid search + RRF + rerank)│
  │            │                                │
  │            ▼                                │
  │  chain.py (Groq LLM + citation extraction)  │
  │            │                                │
  │            ▼                                │
  │  response with inline [N] citations         │
  └─────────────────────────────────────────────┘
```

---

## Project Structure

```
├── src/
│   ├── ingestion/          # PDF parsing, chunking, metadata extraction
│   │   ├── parser.py       # Main document parser (PyMuPDF)
│   │   ├── definitions.py  # Splits definition section into per-term chunks
│   │   ├── annex_iv.py     # FPI document table extraction (pdfplumber)
│   │   ├── appendix.py     # Repealed circulars extraction
│   │   ├── models.py       # Dataclasses: KYCChunk, TableChunk, Footnote
│   │   └── run_ingestion.py
│   ├── embedding/          # Dense + sparse embedding, Qdrant upsert
│   │   └── embed.py
│   ├── retrieval/          # Hybrid search, RRF fusion, cross-encoder reranking
│   │   └── retriever.py
│   ├── llm/                # LLM chain, prompt engineering, intent routing
│   │   ├── chain.py        # Query → retrieve → generate → cite
│   │   ├── prompts.py      # System and human prompt templates
│   │   ├── router.py       # Intent classifier (regex-first, LLM fallback)
│   │   └── response.py     # Pydantic response models
│   └── api/                # FastAPI application
│       ├── app.py          # Lifespan, CORS, routes
│       ├── routes/         # /query, /query/route, /query/stream
│       ├── middleware.py
│       └── templates/      # Single-file chat frontend
├── evaluation/             # Evaluation framework
│   ├── datasets.py         # Hand-written QA pairs (deep, multihop, intent, negative)
│   ├── metrics.py          # hit_rate, MRR, recall@k, precision@k, refusal detection
│   ├── judge.py            # LLM-as-judge (faithfulness + answer relevance)
│   └── run_eval.py         # Full evaluation pipeline with token budget management
├── scripts/                # Data pipeline and eval dataset generation scripts
├── tests/                  # Unit and integration tests
├── data/
│   ├── raw/                # Source PDFs (DVC-tracked)
│   ├── processed/          # Parsed chunks, embeddings manifest (DVC-tracked)
│   └── eval/               # Evaluation dataset and results
├── params.yaml             # Single config file (Hydra)
├── dvc.yaml                # Pipeline stages: ingest → embed
├── Dockerfile              # Multi-stage build with model warm-up
├── docker-compose.yaml     # API + Qdrant for local development
└── requirements/
    ├── prod.txt            # Pinned production dependencies (with hashes)
    └── dev.txt             # Development dependencies (DVC, MLflow, testing)
```

---

## Setup

### Prerequisites

- Python 3.12
- Docker and Docker Compose (for local Qdrant)
- A Groq API key (for LLM inference)
- A Qdrant instance (local via Docker or Qdrant Cloud free tier)

### Installation

```bash
git clone https://github.com/shashi-hue/RAG-RBI-KYC.git
cd RAG-RBI-KYC

python -m venv .venv && source .venv/bin/activate
pip install -r requirements/prod.txt
```

### Configuration

All configuration lives in `params.yaml`. Set the required environment variables:

```bash
export GROQ_API_KEY="your-groq-api-key"
export QDRANT_URL="http://localhost:6333"   # or your Qdrant Cloud URL
export QDRANT_API_KEY=""                     # leave empty for local Qdrant
```

### Running the Pipeline

```bash
# Start Qdrant
docker compose up qdrant -d

# Run the full ingestion + embedding pipeline
dvc repro

# Start the API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Or use Docker Compose to run everything:

```bash
docker compose up --build
```

---

## Usage

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat frontend |
| `/query` | POST | Standard query → answer with citations |
| `/query/route` | POST | Router-dispatched query (intent-aware retrieval) |
| `/query/stream` | POST | Streaming response via Server-Sent Events |

### Example Request

```bash
curl -X POST http://localhost:8000/query/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What documents does a Category III FPI need to submit for entity level KYC?"}'
```

The response includes the answer with inline `[N]` citations, the list of source chunks with their amendment status, and metadata (elapsed time, chunks used).

---

## Evaluation

The system is evaluated on a custom dataset of 84 questions across four categories:

| Category | Count | What It Tests |
|----------|-------|---------------|
| Retrieval (FAQ + deep + multihop) | 59 | Chunk retrieval accuracy and answer quality |
| Intent classification | 15 | Router correctness across all 4 intents |
| Negative / out-of-scope | 10 | System refuses to answer off-topic questions |

The FAQ subset is drawn from RBI's own official FAQ document [FAQ's_KYC_Directions_2025](https://www.rbi.org.in/commonman/english/Scripts/FAQs.aspx?Id=3782), making the evaluation grounded in questions the regulator considers important.

### Results

| Metric | v1 Baseline | v2 Current | Change |
|--------|-------------|------------|--------|
| Hit Rate | 0.586 | **0.864** | +47.5% |
| MRR | 0.431 | **0.692** | +60.7% |
| Recall@5 | 0.569 | **0.831** | +46.0% |
| Precision@5 | 0.135 | **0.190** | +41.1% |
| Faithfulness | 0.483 | **0.658** | +36.2% |
| Answer Relevance | 0.698 | **0.814** | +16.5% |
| Intent Accuracy | 0.800 | **0.933** | +16.7% |
| Refusal Rate | 0.000 | **1.000** |  fixed (Counted "no relevent chunks" as refusal) |

**Note on faithfulness score:** The evaluation dataset includes RBI's official FAQ answers, which are written in plain language and sometimes paraphrase or interpret the regulatory text rather than quoting it verbatim. Since the faithfulness metric measures strict grounding in retrieved context, FAQ-derived answers score lower even when the RAG system's response is factually correct. Manual inspection confirms the system produces accurate, well-cited answers in practice.

### Key Fixes (v1 → v2)

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Definition queries failing | All 14 definitions in a single chunk | `definitions.py` splits each term into its own chunk |
| V-CIP queries misrouted | "V-CIP" matched Chapter V regex instead of Chapter VI | Fixed regex precedence in `router.py` |
| LLM hallucinating list items | Model completing lists beyond what context contained | Added prompt rule: no list completion beyond context |
| Refusal rate at 0% | `is_refusal()` regex patterns not matching | Fixed detection logic in `metrics.py` |

---

## Deployment

This project uses a dual deployment strategy:

| Target | Platform | Purpose |
|--------|----------|---------|
| Production image | AWS ECR + EC2 | Portfolio — demonstrates MLOps and cloud deployment skills |
| Live demo | Hugging Face Spaces | Free, always-accessible showcase |

### CI/CD

GitHub Actions automatically:

- Runs tests (`pytest`) and linting (`ruff`) on every push
- Builds the Docker image and pushes to AWS ECR on `main`
- Deploys to AWS EC2 (auto-restart, public HTTPS URL)
- Pushes to Hugging Face Spaces in parallel (free hosting)

### Infrastructure

| Component | Service | Notes |
|-----------|---------|-------|
| Vector Store | Qdrant Cloud | Always-on |
| LLM | Groq API | Sub-second inference |
| Embeddings | Loaded at startup via `warmup_models.py` | Cached in the Docker image layer to avoid download on cold start |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Document Parsing | PyMuPDF (fitz), pdfplumber |
| Dense Embeddings | BAAI/bge-large-en-v1.5 (1024d) |
| Sparse Embeddings | BM25 via FastEmbed (Qdrant/bm25) |
| Vector Store | Qdrant (hybrid collection, RRF fusion) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-12-v2 |
| LLM | Groq API (LLaMA 3.3 70B) |
| Framework | FastAPI, LangChain |
| Configuration | Hydra + params.yaml |
| Data Pipeline | DVC |
| Experiment Tracking | MLflow (local), LangSmith (LLM traces) |
| Containerization | Docker (multi-stage build), Docker Compose |
| Cloud Deployment | AWS ECR + AWS EC2 |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.