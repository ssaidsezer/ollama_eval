# Speculative ablation result status

This directory is retained as partial experiment evidence, not as a completed
three-condition benchmark.

| Condition | Status | Evidence boundary |
|---|---|---|
| `qwen_ngram` | Partial | 150/150 inference rows and 150/150 balanced-v2 evaluations completed. The run manifest remains failed because the required aggregate speculative accepted/generated counts were not available. |
| `gemma_ngram` | Incomplete | 55/150 inference rows succeeded; 95 failed after the remote connection was lost (`URLError`/`RemoteDisconnected`, followed by SSH exit 255). |
| `gemma_eagle3` | Not run | No target-specific EAGLE-3 draft GGUF was available. |

Do not use `comparison.csv` to declare a speculative-decoding winner. The Qwen
answer-quality artifacts may be inspected independently, but a controlled
performance conclusion requires complete aggregate acceptance statistics and
successful runs for all intended conditions.
