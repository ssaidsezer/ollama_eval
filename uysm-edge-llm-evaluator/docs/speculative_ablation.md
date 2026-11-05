# Speculative Decoding Ablation

`scripts/experiments/run_speculative_ablation.py` compares three llama.cpp speculative-decoding
conditions with the completed Qwen and Gemma Q8 KV-cache baselines:

1. Qwen3.6 35B-A3B with `ngram-mod`;
2. Gemma4 26B-A4B with `ngram-mod`;
3. Gemma4 26B-A4B with a target-specific EAGLE-3 draft model.

The script controls a benchmark-owned remote `llama-server` over SSH. It refuses
to run when another llama-server or another listener on remote port `8084` is
present. It records the PID it starts and stops only that PID after verifying the
process executable.

## Prerequisites

- SSH key access to `odine@192.168.1.29`;
- the existing target GGUF files and llama.cpp binary at the default Jetson paths;
- a converted, target-specific Gemma4-26B-A4B EAGLE-3 GGUF;
- `data/qa/final_qa_150.csv` and the completed Q8 baseline directories;
- `OPENAI_API_KEY` unless `--skip-eval` is used.

The script does not download models, update llama.cpp, install packages, or
reboot the Jetson.

## Run

```bash
export OPENAI_API_KEY='...'

.venv312/bin/python scripts/experiments/run_speculative_ablation.py \
  --gemma-eagle-model /home/odine/models/gemma-4-26B-A4B-it-speculator-eagle3.gguf \
  --out-dir data/results/speculative_ablation_qa150
```

Run only a subset by passing a comma-separated list:

```bash
.venv312/bin/python scripts/experiments/run_speculative_ablation.py \
  --conditions qwen_ngram,gemma_ngram \
  --skip-eval \
  --out-dir data/results/speculative_ablation_ngram_smoke
```

Resume the same output directory with `--resume`. A condition is skipped only
when it already has 150 successful inference rows and a parsed speculative
statistics file. Otherwise the existing inference runner resumes its atomic
per-question checkpoints.

## Frozen controls

All three conditions use target weights `Q4_K_M`, K/V cache `q8_0`, context
`32768`, Flash Attention, one slot, prompt cache disabled, temperature `0`,
thinking disabled, and at most `512` completion tokens. Each server gets a
three-question smoke run followed by a clean restart and the full 150-question
run.

The two `ngram-mod` conditions use match/min/max lengths `24/48/64`. EAGLE-3
uses `--spec-draft-n-max 3`. The client does not invent per-question acceptance
values: aggregate accepted/generated token counts are parsed from the complete
server log after shutdown.

## Outputs

Each condition directory contains the exact server command, local copy of the
remote PID, smoke/full server logs, raw/evaluated inference CSVs, `summary.json`,
and `speculative_stats.json`. The parent directory contains `manifest.json` and
`comparison.csv`.

`comparison.csv` reports baseline-relative median latency and generation tok/s,
quality means, truncation counts, and draft acceptance. A condition is marked
successful only when latency or generation tok/s improves by at least 10%, both
quality means do not decrease, and truncations do not increase.
