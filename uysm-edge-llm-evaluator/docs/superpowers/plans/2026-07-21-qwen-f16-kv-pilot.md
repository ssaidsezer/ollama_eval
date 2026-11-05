# Qwen F16 KV Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete only the Qwen3.6 35B-A3B symmetric `f16/f16` KV-cache condition over the frozen 120-question set, preserve raw inference evidence, and stop before any other condition.

**Architecture:** The Mac keeps the frozen CSV, benchmark client, checkpoints, and summaries. A benchmark-owned llama.cpp server runs on the Jetson bound to loopback, and an SSH control-socket tunnel exposes it at `127.0.0.1:8084`. The existing standalone inference runner sends sequential no-RAG requests and writes atomically resumable results.

**Tech Stack:** Python 3.12, pytest, llama.cpp server build `571d0d5`, SSH control socket, CSV/JSON logs.

## Global Constraints

- Run only Qwen3.6 35B-A3B `Q4_K_M` with K cache `f16` and V cache `f16`.
- Use context size `32768`, `--parallel 1`, Flash Attention on, and `--no-cache-prompt`.
- Use temperature `0`, maximum completion length `512`, thinking disabled, and the repository no-RAG prompt.
- Use the frozen 120 questions in their existing order.
- Do not start Gemma, `q8_0`, `q4_0`, OpenAI evaluation, or report generation in this pilot.
- Do not update llama.cpp, download models, modify system packages, or reboot the Jetson.
- Stop only the benchmark-owned server PID.

---

### Task 1: Materialize and Validate the Frozen Input

**Files:**
- Create: `data/csv/rag_blind_test_120_gold.csv`
- Read: `.worktrees/rag-blind-test-set/data/csv/rag_blind_test_120_gold.csv`

**Interfaces:**
- Consumes: frozen CSV with `question` and `reference_answer` columns.
- Produces: canonical main-worktree CSV accepted by `load_questions(path, "question", "reference_answer")`.

- [ ] **Step 1: Copy the frozen artifact without transforming rows**

Run:

```bash
cp .worktrees/rag-blind-test-set/data/csv/rag_blind_test_120_gold.csv \
  data/csv/rag_blind_test_120_gold.csv
```

- [ ] **Step 2: Verify identity, schema, and row count**

Run:

```bash
shasum -a 256 \
  .worktrees/rag-blind-test-set/data/csv/rag_blind_test_120_gold.csv \
  data/csv/rag_blind_test_120_gold.csv
.venv312/bin/python -c "import csv; p='data/csv/rag_blind_test_120_gold.csv'; rows=list(csv.DictReader(open(p, encoding='utf-8-sig'))); assert len(rows)==120; assert all(r['question'] and r['reference_answer'] for r in rows); print(len(rows))"
```

Expected: both SHA-256 values are identical and Python prints `120`.

- [ ] **Step 3: Commit only the canonical input**

```bash
git add data/csv/rag_blind_test_120_gold.csv
git commit -m "data: add frozen 120-question blind set"
```

### Task 2: Verify the Existing Benchmark Client

**Files:**
- Read: `scripts/evaluation/run_llamacpp_inference_eval.py`
- Test: `tests/test_llamacpp_inference_eval.py`

**Interfaces:**
- Consumes: `--csv`, `--question-col question`, `--answer-col reference_answer`, and llama.cpp OpenAI-compatible endpoint.
- Produces: atomic `raw_results.csv` and `summary.json` under the selected output directory.

- [ ] **Step 1: Run the focused offline test suite**

Run:

```bash
.venv312/bin/python -m pytest tests/test_llamacpp_inference_eval.py -q
```

Expected: all tests pass without Jetson or OpenAI network calls.

- [ ] **Step 2: Validate the 120-row input through the runner loader**

Run:

```bash
.venv312/bin/python -c "from pathlib import Path; from scripts.evaluation.run_llamacpp_inference_eval import load_questions; rows=load_questions(Path('data/csv/rag_blind_test_120_gold.csv'), 'question', 'reference_answer'); assert len(rows)==120; print(len(rows))"
```

Expected: `120`.

### Task 3: Start and Smoke-Test the Qwen F16 Server

**Files:**
- Create remotely: `/home/odine/kv_ablation/qwen_f16/server_command.txt`
- Create remotely: `/home/odine/kv_ablation/qwen_f16/server.log`
- Create remotely: `/home/odine/kv_ablation/qwen_f16/server.pid`

**Interfaces:**
- Consumes: Qwen GGUF at `/home/odine/models/llama.cpp/models--lmstudio-community--Qwen3.6-35B-A3B-GGUF/snapshots/68a34855558af61cbef0324d31f411be8a506b08/Qwen3.6-35B-A3B-Q4_K_M.gguf`.
- Produces: llama.cpp endpoint reachable through `http://127.0.0.1:8084/v1`.

- [ ] **Step 1: Refuse to overwrite an active service**

Run remotely:

```bash
test -z "$(pgrep -f '/home/odine/llama.cpp/build/bin/llama-server')"
```

Expected: exit status `0`.

- [ ] **Step 2: Start the benchmark-owned server**

Use these exact arguments:

```bash
/home/odine/llama.cpp/build/bin/llama-server \
  --model /home/odine/models/llama.cpp/models--lmstudio-community--Qwen3.6-35B-A3B-GGUF/snapshots/68a34855558af61cbef0324d31f411be8a506b08/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --alias qwen3.6:35b-a3b \
  --host 127.0.0.1 \
  --port 8084 \
  --ctx-size 32768 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --no-cache-prompt
```

Persist the command, redirect stdout/stderr to `server.log`, and save the PID.

- [ ] **Step 3: Open the control-socket tunnel and wait for health**

Run the SSH tunnel with `/tmp/odine-codex.sock`, then poll:

```bash
curl --fail --silent http://127.0.0.1:8084/health
```

Expected: HTTP success and a ready/ok health payload.

- [ ] **Step 4: Verify the log configuration**

The server log must identify the Qwen model and show `f16` K/V allocation,
context `32768`, one slot, Flash Attention enabled, and prompt caching disabled.
If any invariant is absent or contradicted, stop the saved PID and fail the pilot.

- [ ] **Step 5: Run a three-question smoke test**

Create a temporary three-row projection from the canonical CSV and run:

```bash
.venv312/bin/python scripts/evaluation/run_llamacpp_inference_eval.py \
  --csv /tmp/qwen_f16_smoke.csv \
  --answer-col reference_answer \
  --base-url http://127.0.0.1:8084/v1 \
  --model qwen3.6:35b-a3b \
  --run-label qwen3.6-35b-a3b_q4km_kv-f16_ctx32768_fa-on_prompt-cache-off_smoke \
  --out-dir /tmp/qwen_f16_smoke_results \
  --temperature 0 \
  --max-tokens 512 \
  --skip-eval
```

Expected: three successful rows, non-empty answers, and `cached_tokens=0` for all rows.

### Task 4: Run the Full Qwen F16 Condition and Stop

**Files:**
- Create: `data/results/kv_ablation_20260721/qwen_f16/raw_results.csv`
- Create: `data/results/kv_ablation_20260721/qwen_f16/summary.json`
- Preserve remotely: `/home/odine/kv_ablation/qwen_f16/server.log`

**Interfaces:**
- Consumes: verified endpoint and canonical 120-row CSV.
- Produces: complete Qwen `f16/f16` raw checkpoint and inference summary.

- [ ] **Step 1: Restart to remove smoke-test state**

Stop the PID saved by Task 3, verify it exited, start the same exact server
command again, wait for health, and perform the runner's one unmeasured warmup.

- [ ] **Step 2: Run all 120 questions without OpenAI evaluation**

Run:

```bash
.venv312/bin/python scripts/evaluation/run_llamacpp_inference_eval.py \
  --csv data/csv/rag_blind_test_120_gold.csv \
  --answer-col reference_answer \
  --base-url http://127.0.0.1:8084/v1 \
  --model qwen3.6:35b-a3b \
  --run-label qwen3.6-35b-a3b_q4km_kv-f16_ctx32768_fa-on_prompt-cache-off \
  --out-dir data/results/kv_ablation_20260721/qwen_f16 \
  --temperature 0 \
  --max-tokens 512 \
  --timeout 300 \
  --skip-eval
```

Expected: `raw_results.csv` contains 120 rows and `summary.json` reports 120 successful inference rows and zero cached tokens.

- [ ] **Step 3: Validate outputs before reporting completion**

Run:

```bash
.venv312/bin/python -c "import csv,json; p='data/results/kv_ablation_20260721/qwen_f16'; rows=list(csv.DictReader(open(p+'/raw_results.csv',encoding='utf-8-sig'))); s=json.load(open(p+'/summary.json')); assert len(rows)==120; assert all(r['inference_status']=='success' for r in rows); assert all(int(r['cached_tokens'])==0 for r in rows); assert s['counts']['successful']==120; print(s['counts'])"
```

Expected: `{'total': 120, 'successful': 120, 'failed': 0}`.

- [ ] **Step 4: Stop at the user checkpoint**

Stop only the benchmark-owned server PID, close the benchmark tunnel, preserve
all local results and remote logs, report the Qwen F16 metrics, and do not start
Gemma or another KV type.
