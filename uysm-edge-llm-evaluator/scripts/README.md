# Evaluator scripts

Scripts are grouped by responsibility. The only public interface change in this
refactor is the file path: command-line options, defaults, schemas, output
locations, resume behavior, prompts, metrics, and experiment logic are retained.

## Setup

Install the normal project dependencies first. PDF report generation additionally
requires:

```bash
python -m pip install -r requirements-report.txt
python -m playwright install chromium
```

## Active commands

| Area | Script | Purpose | Example | Mutates |
|---|---|---|---|---|
| Evaluation | `evaluation/batch_eval.py` | Batch model evaluation | `python scripts/evaluation/batch_eval.py --help` | result directory |
| Evaluation | `evaluation/evaluate_balanced_v2.py` | Balanced-v2 answer scoring | `python scripts/evaluation/evaluate_balanced_v2.py --help` | evaluated CSV/JSON |
| Evaluation | `evaluation/run_llamacpp_inference_eval.py` | llama.cpp inference and evaluation | `python scripts/evaluation/run_llamacpp_inference_eval.py --help` | result directory |
| Evaluation | `evaluation/run_rag_ablation.py` | RAG ablation runner | `python scripts/evaluation/run_rag_ablation.py --help` | result directory |
| QA | `qa/build_balanced_qa.py` | Sample a reproducible balanced QA set | `python scripts/qa/build_balanced_qa.py --help` | output CSV |
| QA | `qa/cull_generated_qa.py` | Flag and report generated QA candidates | `python scripts/qa/cull_generated_qa.py --help` | annotated CSV/report |
| QA | `qa/gen_qa_from_images.py` | Generate QA candidates from rendered pages | `python scripts/qa/gen_qa_from_images.py --help` | generation cache/results |
| QA | `qa/gen_qa_render.py` | Render source PDFs into page chunks | `python scripts/qa/gen_qa_render.py --help` | image chunks/manifest |
| QA | `qa/validate_rag_blind_test.py` | Validate/project the blind-test dataset | `python scripts/qa/validate_rag_blind_test.py --help` | outputs unless `--check-only` |
| RAG | `rag/check_index_integrity.py` | Read-only Qdrant index integrity check | `python scripts/rag/check_index_integrity.py --help` | no |
| Experiments | `experiments/run_speculative_ablation.py` | Remote speculative-decoding ablation | `python scripts/experiments/run_speculative_ablation.py --help` | remote service and result directory |
| Ops | `ops/verify_web_api.py` | Web API smoke verification | `python scripts/ops/verify_web_api.py` | no repository files |
| Reporting | `reporting/generate_consolidated_report.py` | Generate the consolidated evaluation PDF | `python scripts/reporting/generate_consolidated_report.py` | `output/pdf/uysm_consolidated_evaluation_report.pdf` |

The llama.cpp and RAG runners retain their legacy 100-question default. Fixed-150
runs must explicitly pass `--csv data/qa/final_qa_150.csv`.

`build_balanced_qa.py` retains the legacy defaults (`--per-book 45`, seed `42`,
and `data/qa/final_qa_100.csv`). The canonical 150-row set was produced with:

```bash
python scripts/qa/build_balanced_qa.py \
  --per-book 70 \
  --out data/qa/final_qa_150.csv \
  --seed 42
```

## Archive

The hardcoded July 2026 experiment scripts and their provenance are retained in
[`archive/run_2026_07/`](archive/run_2026_07/README.md). They are not generalized
operational entry points.
