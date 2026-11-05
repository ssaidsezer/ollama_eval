# llama.cpp Inference Evaluation

This benchmark sends the repository's 100 questions directly to an already-running
llama.cpp server. It records inference telemetry first, then evaluates the saved
answers with OpenAI. RAG is not used.

## Prerequisites

- llama-server is already running with the intended model and runtime flags.
- The configured model alias matches the server alias.
- `OPENAI_API_KEY` is exported for quality evaluation.
- Run the command from the repository root with `.venv312`.

The script does not start, stop, load, or unload the model. Model load time is
therefore excluded. It performs one unrecorded warmup request by default, then
measures the 100 questions sequentially.

The legacy 100-question CSV remains the default. For a fixed-150 experiment,
pass `--csv data/qa/final_qa_150.csv` explicitly.

## Qwen Q4_K_M / KV Q8 run

For the current server configuration (`Q4_K_M`, KV cache `q8_0`, Flash Attention
on, prompt cache on, thinking off):

```bash
export OPENAI_API_KEY='...'

.venv312/bin/python scripts/evaluation/run_llamacpp_inference_eval.py \
  --base-url http://192.168.1.29:8084/v1 \
  --model qwen3.6:35b-a3b \
  --eval-model gpt-5.4-mini \
  --run-label qwen3.6-35b-a3b_q4km_kv-q8_prompt-cache-on
```

The server configuration in the run label is descriptive: the client cannot
change or independently verify the server's quantization, KV-cache type, Flash
Attention, or prompt-cache flags. Confirm those values from the llama-server
startup command/log before starting a run.

## Outputs

By default each invocation creates a timestamped directory under `data/results/`:

- `raw_results.csv`: answer and per-question inference telemetry
- `evaluated_results.csv`: inference fields plus OpenAI quality scores
- `summary.json`: counts, latency/throughput percentiles, cache ratio, and quality means
- `run.log`: progress and row-level statuses

Client wall time, prompt processing time, generation time, and generation tokens
per second are reported separately. `cached_tokens` and newly processed prompt
tokens remain visible; cached prefix reuse is not mistaken for uncached prompt
speed. OpenAI evaluation time is separate and never contributes to inference
latency.

## Resume and inference-only modes

Use a fixed output directory to make resume deterministic:

```bash
.venv312/bin/python scripts/evaluation/run_llamacpp_inference_eval.py \
  --out-dir data/results/qwen_q4km_kvq8_run1 \
  --resume
```

Successful inference and evaluation rows are skipped; failed or missing rows are
retried. To collect local inference results without calling OpenAI, add
`--skip-eval`. To intentionally disable the unmeasured warmup, add `--no-warmup`.

Do not unload/reload the model between questions. The target measurement is steady
state inference on the edge device; model startup belongs in a separate cold-start
benchmark.
