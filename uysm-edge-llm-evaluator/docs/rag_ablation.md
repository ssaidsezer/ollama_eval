# RAG Ablation Runner

This script runs the same QA CSV against four RAG retrieval variants:

- `classic_vector`
- `smart_vector`
- `classic_bm25`
- `smart_bm25`

It uses `data/csv/question_answers_100.csv` by default.

The legacy 100-question default is intentionally unchanged. Fixed-150 runs must
pass `--csv data/qa/final_qa_150.csv` explicitly.

## Required Qdrant Collections

Use `rag_ablation` as the base collection name. The following collections must exist and be populated:

```text
rag_ablation
rag_ablation_children
rag_ablation_parents
```

The mapping is:

```text
classic_vector -> rag_ablation
classic_bm25   -> rag_ablation
smart_vector   -> rag_ablation_children + rag_ablation_parents
smart_bm25     -> rag_ablation_children + rag_ablation_parents
```

## Required Environment

Set these in `.env` or in the shell:

```bash
OLLAMA_BASE_URL=http://<ollama-host>:<port>
QDRANT_URL=http://<qdrant-host>:<port>
OLLAMA_EMBED_MODEL=<embedding-model-name>
```

For evaluation, also set one of:

```bash
DEEPSEEK_API_KEY=<key>
OPENAI_API_KEY=<key>
```

Local evaluation does not need an API key, but it needs an Ollama eval model.

## Run One Or More QA Models

If classic and smart indexing produced different collection base names, pass both explicitly:

```bash
python scripts/evaluation/run_rag_ablation.py \
  --classic-collection documents_embeddinggemma_300m_1000c_200ov \
  --smart-collection documents_embeddinggemma_300m_3000p_500c_100ov \
  --models '["model_a","model_b"]' \
  --embed-model embeddinggemma:300m
```

Comma-separated:

```bash
python scripts/evaluation/run_rag_ablation.py \
  --models "model_a,model_b" \
  --collection rag_ablation
```

JSON list:

```bash
python scripts/evaluation/run_rag_ablation.py \
  --models '["model_a","model_b","model_c"]' \
  --collection rag_ablation
```

All installed non-embedding Ollama models:

```bash
python scripts/evaluation/run_rag_ablation.py \
  --all \
  --collection rag_ablation
```

## Reliability And Resume

The runner checkpoints after every question. A transient retrieval/inference or
evaluation failure is retried once; a final question failure is recorded in the
checkpoint and the next question continues. If Ollama itself fails its health
check, the manifest is saved with `aborted_service_unavailable` and the run exits.

Benchmark defaults are deterministic:

```text
timeout=300
embedding_timeout=300
num_predict=512
temperature=0
seed=42
```

To continue an interrupted output directory, reuse the same command and add:

```bash
--out-dir data/results/rag_ablation_<timestamp> --resume
```

Successful rows are skipped. Inference or evaluation error rows are retried and
replaced atomically. A non-empty output directory is rejected unless `--resume`
is explicit.

## Optional Evaluation

DeepSeek:

```bash
python scripts/evaluation/run_rag_ablation.py \
  --models "model_a" \
  --collection rag_ablation \
  --eval \
  --eval-backend deepseek
```

OpenAI:

```bash
python scripts/evaluation/run_rag_ablation.py \
  --models "model_a" \
  --collection rag_ablation \
  --eval \
  --eval-backend openai \
  --eval-model gpt-5.4-mini
```

Local Ollama eval:

```bash
python scripts/evaluation/run_rag_ablation.py \
  --models "model_a" \
  --collection rag_ablation \
  --eval \
  --eval-backend local \
  --eval-local-model "eval_model_name"
```

## Outputs

Each run writes a timestamped output directory:

```text
data/results/rag_ablation_<timestamp>/
  checkpoints/
  merged_results.csv
  manifest.csv
  run.log
  errors.log
```

Checkpoint rows include `inference_status`, `evaluation_status`, error stage/type,
attempt count, generated token count, and prompt token count. `manifest.csv` is
updated after every question with attempted, successful, and error totals.

Send back the whole `data/results/rag_ablation_<timestamp>/` directory.
