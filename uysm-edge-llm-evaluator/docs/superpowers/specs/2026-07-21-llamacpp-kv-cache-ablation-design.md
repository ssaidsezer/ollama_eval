# llama.cpp KV-Cache Ablation Design

## Goal

Measure the speed, memory, and answer-quality effects of symmetric KV-cache
quantization for the two existing edge models on the Jetson AGX Orin 64 GB.
Speculative decoding and RAG are out of scope. The only experimental variable
within each model is the KV-cache data type.

## Checked Environment

The live Jetson inventory on 2026-07-21 confirmed:

- host: `odine@192.168.1.29`;
- llama.cpp server: `/home/odine/llama.cpp/build/bin/llama-server`;
- llama.cpp build: `571d0d5`;
- Qwen weights: `Qwen3.6-35B-A3B-Q4_K_M.gguf`, approximately 21 GB;
- Gemma weights: `gemma-4-26B-A4B-it-Q4_K_M.gguf`, approximately 16 GB;
- supported K/V cache types include `f16`, `q8_0`, `q4_0`, and `q4_1`;
- no llama.cpp or Ollama model server was active during inventory.

## Experiment Matrix

Use symmetric K/V types only:

| Model | K cache | V cache |
| --- | --- | --- |
| Qwen3.6 35B-A3B Q4_K_M | `f16` | `f16` |
| Qwen3.6 35B-A3B Q4_K_M | `q8_0` | `q8_0` |
| Qwen3.6 35B-A3B Q4_K_M | `q4_0` | `q4_0` |
| Gemma4 26B-A4B Q4_K_M | `f16` | `f16` |
| Gemma4 26B-A4B Q4_K_M | `q8_0` | `q8_0` |
| Gemma4 26B-A4B Q4_K_M | `q4_0` | `q4_0` |

This produces six full conditions and 720 measured answers. `q4_1` and
asymmetric K/V combinations are excluded from the first experiment to keep the
matrix interpretable and bounded.

Run conditions in an interleaved order to reduce model- or time-correlated
thermal drift:

1. Qwen `f16`
2. Gemma `f16`
3. Qwen `q8_0`
4. Gemma `q8_0`
5. Qwen `q4_0`
6. Gemma `q4_0`

## Frozen Inputs and Controls

Use the frozen 120-question blind gold set currently stored at
`.worktrees/rag-blind-test-set/data/csv/rag_blind_test_120_gold.csv`. The
benchmark reads `question` as input and joins `reference_answer` for evaluation.
The canonical CSV must be brought into the main worktree before execution so a
result does not depend on another worktree remaining present.

All six conditions use:

- no RAG or retrieval context;
- the same repository no-RAG system prompt;
- context size `32768`;
- `--parallel 1`;
- Flash Attention enabled;
- prompt cache explicitly disabled with `--no-cache-prompt`;
- model weights fixed at `Q4_K_M`;
- temperature `0`;
- maximum completion length `512` tokens;
- model thinking disabled;
- identical question order;
- one server process and one measured request at a time.

Prompt caching is disabled because repeated system-prompt prefixes would add a
second reuse mechanism to the KV-type comparison. Each request must process its
prompt independently.

## Server and Run Lifecycle

Each condition has an isolated server lifecycle:

1. Confirm that no previous benchmark server is active.
2. Start `llama-server` with the selected model, `--cache-type-k`, and
   `--cache-type-v`, plus the frozen controls above.
3. Save the complete server command and server log in the condition directory.
4. Wait for the health endpoint.
5. Send one unmeasured warmup request.
6. Run a three-question smoke test and verify non-empty responses, timing
   fields, configured model alias, and no server error.
7. Restart the server to remove smoke-test state.
8. Warm up once again and run all 120 questions sequentially.
9. Capture final telemetry, stop only the benchmark-owned server, and verify
   that it exited before starting the next condition.

A failure in one condition must not erase completed conditions. Per-question
checkpointing and resume semantics remain active.

## Measurements

Record per question:

- client-observed wall time;
- server-reported prompt and generation time;
- prompt and completion token counts;
- prompt and generation tokens per second;
- finish reason, retry count, and structured failure details;
- model answer and optional reasoning content.

Record per condition:

- exact server command, llama.cpp build, model path, and KV types;
- idle-before-start, post-load, post-warmup, and peak system memory;
- llama.cpp-reported model and KV-cache allocations from the server log;
- total runtime, success/failure count, and truncation count;
- latency and throughput mean, median, p95, and p99;
- OpenAI correctness, completeness, hallucination risk, and verdict aggregates.

OpenAI evaluation runs on the Mac after local inference checkpoints are
complete. Evaluator latency is stored separately and never included in Jetson
inference timing.

## Outputs

Create one timestamped parent directory with one subdirectory per condition.
Each condition contains:

- `server_command.txt`;
- `server.log`;
- `system_metrics.csv`;
- `raw_results.csv`;
- `evaluated_results.csv`;
- `summary.json`.

The parent directory also contains a manifest and a comparison report covering
all six conditions. Run labels encode model, weight type, symmetric KV type,
context size, Flash Attention state, and prompt-cache state.

## Failure Handling and Safety

- Refuse to start a condition if port `8084` is owned by an unrecognized
  process.
- Track the PID of the benchmark-owned server and stop only that PID.
- Treat health-check failure, model-load failure, CUDA allocation failure, and
  unsupported cache-type errors as condition-level failures with preserved
  logs.
- Treat request timeouts, malformed responses, and empty answers as row-level
  failures eligible for resume.
- Do not update llama.cpp, download models, alter system packages, or reboot the
  Jetson as part of this experiment.

## Verification and Acceptance Criteria

Before the full remote run:

- unit tests cover matrix generation, command construction, frozen controls,
  run labels, prompt-cache disabling, checkpoint/resume behavior, and summary
  aggregation;
- existing llama.cpp benchmark and RAG tests remain green;
- every condition passes its three-question smoke test.

The experiment is complete when all six conditions have 120 successful or
explicitly failed rows, all configurations and logs are preserved, OpenAI
evaluation is joined without contaminating inference timing, and the final
report compares memory, speed, and quality without selecting a single metric as
the sole winner.
