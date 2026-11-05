# Balanced Answer Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a balanced-v2 evaluator with explicit groundedness and high-reasoning OpenAI judging, then re-evaluate all six saved 150-row conditions without rerunning inference.

**Architecture:** Extend the existing standalone evaluator with a versioned rubric and artifact paths. Strict-v1 CSV/JSON files remain unchanged; balanced-v2 reads `raw_results.csv`, writes separate checkpoints and summaries, and passes a configurable reasoning effort to the same OpenAI judge.

**Tech Stack:** Python 3.12, pytest, OpenAI Chat Completions, CSV/JSON atomic checkpoints.

## Global Constraints

- Preserve `raw_results.csv`, `evaluated_results.csv`, and `summary.json`.
- Use `gpt-5.4-mini` with `reasoning_effort=high` for the requested rerun.
- Do not contact or restart the Jetson during evaluation-only resume.
- Keep answerable and unanswerable metrics separate.

---

### Task 1: Balanced-v2 Rubric and Parsing

**Files:**
- Modify: `tests/test_llamacpp_inference_eval.py`
- Modify: `scripts/evaluation/run_llamacpp_inference_eval.py`

**Interfaces:**
- Produces: `BALANCED_V2_EVAL_SYSTEM_PROMPT`, `parse_balanced_v2_evaluation(content)`, and a v2 evaluation row containing `groundedness_score`.

- [ ] **Step 1: Write failing tests**

Add tests asserting that the v2 prompt says harmless accurate context lowers groundedness without substantially lowering correctness, and that parsing requires a numeric `groundedness_score` in `[0, 10]`.

- [ ] **Step 2: Verify RED**

Run: `.venv312/bin/python -m pytest tests/test_llamacpp_inference_eval.py -q`

Expected: failures because the v2 prompt and parser do not exist.

- [ ] **Step 3: Implement the rubric and parser**

Add anchored correctness/completeness rules, separate groundedness, strict handling for `cevaplanabilir=hayir`, and high-risk penalties only for false, contradictory, unsafe, or action-changing claims.

- [ ] **Step 4: Verify GREEN**

Run the focused evaluator tests and confirm they pass.

### Task 2: Versioned Artifacts and High Reasoning

**Files:**
- Modify: `tests/test_llamacpp_inference_eval.py`
- Modify: `scripts/evaluation/run_llamacpp_inference_eval.py`

**Interfaces:**
- Adds CLI: `--eval-rubric {strict-v1,balanced-v2}` and `--eval-reasoning-effort {none,low,medium,high}`.
- Balanced-v2 outputs: `evaluated_results_balanced_v2.csv` and `summary_balanced_v2.json`.

- [ ] **Step 1: Write failing tests**

Test that balanced-v2 chooses versioned paths, preserves strict-v1 files, sends `reasoning_effort="high"`, resumes successful v2 rows, and reports groundedness aggregates.

- [ ] **Step 2: Verify RED**

Run the focused tests and confirm failures are caused by missing v2 behavior.

- [ ] **Step 3: Implement minimal v2 execution support**

Route the selected prompt/parser, add v2 CSV fields and summary metrics, pass reasoning effort only when configured, and bypass inference health/warmup only through an explicit `--evaluation-only` flag that requires a complete successful raw checkpoint.

- [ ] **Step 4: Verify GREEN and regression suite**

Run: `.venv312/bin/python -m pytest tests/test_llamacpp_inference_eval.py -q`

Expected: all evaluator tests pass.

### Task 3: Re-evaluate and Verify Six Conditions

**Files:**
- Create in each condition directory: `evaluated_results_balanced_v2.csv`
- Create in each condition directory: `summary_balanced_v2.json`

**Interfaces:**
- Consumes the six existing `raw_results.csv` files under `data/results/kv_ablation_20260721_150/`.
- Produces 900 successful balanced-v2 evaluation rows.

- [ ] **Step 1: Record strict-v1 artifact checksums**

Hash all six `evaluated_results.csv` and `summary.json` files before rerunning.

- [ ] **Step 2: Run balanced-v2 evaluation**

For each condition, run the evaluator with `--evaluation-only --resume --eval-rubric balanced-v2 --eval-model gpt-5.4-mini --eval-reasoning-effort high`.

- [ ] **Step 3: Verify outputs**

Assert 150 inference successes, 150 v2 evaluation successes, zero evaluation failures, valid groundedness scores, and 900 total evaluated rows.

- [ ] **Step 4: Verify strict-v1 preservation**

Recompute hashes and require exact equality with Step 1.

- [ ] **Step 5: Report calibrated ranking**

Compare correctness, completeness, groundedness, refusal accuracy, hallucination risk, mean latency, and generation throughput. Select the RAG condition using the combined evidence.
