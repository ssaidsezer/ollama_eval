# UYSM Edge LLM Evaluator

This project is a local RAG and model evaluation workspace.
It includes a FastAPI backend, a React frontend, and Python utilities for PDF indexing, retrieval, and answer evaluation.

## What This Project Does

- Index PDF documents into Qdrant
- Retrieve context with vector search or BM25
- Generate answers with Ollama models
- Evaluate answers with DeepSeek, OpenAI, or a local Ollama model
- Manage model profiles from the web UI

## Tech Stack

- Backend: FastAPI, Python
- Frontend: React, TypeScript, Vite, MUI
- Retrieval: Qdrant, BM25
- Model runtime: Ollama
- Evaluation: DeepSeek/OpenAI API or local Ollama

## Requirements

Before running the project, make sure these are available:

- Python 3
- Node.js and npm
- Qdrant
- Ollama

## Environment

The backend reads values from a root `.env` file if you use one.
Useful variables:

- `OLLAMA_BASE_URL`
- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_EVAL_MODEL_NAME`
- `OPENAI_API_KEY`
- `OPENAI_EVAL_MODEL_NAME`
- `CORS_ORIGINS`

`DEEPSEEK_API_KEY` is only needed if you use DeepSeek evaluation.
`OPENAI_API_KEY` is only needed if you use OpenAI evaluation.

## Install Backend Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

## Run The Backend

From the project root:

```bash
python -m uvicorn web_backend.main:app --reload --port 8000
```

The backend will run on `http://127.0.0.1:8000`.

## Run The Frontend

Open a second terminal:

```bash
cd web_frontend
npm install
npm run dev
```

The frontend will usually run on `http://localhost:5173`.

## Run Order

1. Start Qdrant
2. Start Ollama
3. Run the backend with `uvicorn`
4. Run the frontend with `npm run dev`

## Basic Usage

1. Open the frontend in your browser
2. Check connection status for Ollama and Qdrant
3. Upload and index PDFs
4. Run chat or CSV evaluation
5. Review results in the UI

## RAG Retrieval Contracts

The evaluator supports four retrieval variants:

- `classic_vector`: vector search over fixed-size chunks
- `classic_bm25`: BM25 over fixed-size chunks
- `smart_vector`: vector search over child chunks, followed by full parent retrieval
- `smart_bm25`: BM25 over child chunks, followed by full parent retrieval

The `k` parameter has a mode-specific meaning:

- Classic modes: maximum number of chunks returned.
- Smart modes: maximum number of unique parents returned. Child hits that point to the same parent are deduplicated, and vector retrieval continues to subsequent result pages when necessary.

BM25 results with `score <= 0` are not supplied to the answer model. If every BM25 score is non-positive, retrieval is empty and the exported row uses `rag_context_status=empty` with `retrieved_chunks=[]`.

BM25 corpora and indexes are cached in the backend process, with at most eight collection entries. Successful indexing invalidates the affected cache entry. If Qdrant is changed by another process, restart the backend or explicitly call `invalidate_bm25_cache()` before relying on BM25 results.

## Smart Index Integrity

Smart RAG requires every child to reference an existing parent and every parent to have at least one child. Check the current collections without modifying Qdrant:

```bash
.venv312/bin/python scripts/rag/check_index_integrity.py \
  --base-collection documents_embeddinggemma_300m_3000p_500c_100ov

.venv312/bin/python scripts/rag/check_index_integrity.py \
  --base-collection documents_qwen3-embedding_latest_3000p_500c_100ov
```

Use `--json` for machine-readable output. The command exits with status `0` when the collection is consistent and status `1` when it finds orphan children, parents without children, or mismatched source sets.

`--rebuild-source PATH` is a mutating operation. Use it only after selecting and verifying the exact source PDF and confirming the embedding model, boundary model, Ollama URL, Qdrant URL, and target base collection. The default command above is read-only.

## RAG Ablation Notes

Run a small retrieval-only gate before a full answer-generation ablation. For identical questions and `k`, verify that BM25 has no non-positive returned scores, smart results contain no duplicate parent IDs, integrity checks pass, and empty retrievals are explicitly marked.

`response_time_seconds` measures answer-model generation, not retrieval or end-to-end runtime. Report retrieval latency separately, and derive end-to-end per-question time from the ablation run duration and completed row count.

## Project Structure

```text
.
├── web_backend/      FastAPI application
├── web_frontend/     React frontend
├── data/             Sample data and PDFs
├── scripts/          Helper scripts
├── pipeline.py       Main evaluation pipeline
├── rag/              PDF indexing and retrieval package
└── prompts.py        Prompt templates
```

## Notes

- The frontend uses `VITE_API_BASE` if you want to override the API base URL.
- By default, the frontend can work with the FastAPI backend through the Vite setup.
- Make sure Ollama and Qdrant are reachable before starting evaluations or indexing.
