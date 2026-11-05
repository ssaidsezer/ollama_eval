# Balanced Answer Evaluation Design

## Goal

Replace the overly strict reference-only judge with a balanced rubric that
separates factual correctness, reference coverage, document groundedness, and
safety. Preserve the existing strict-v1 evaluation artifacts unchanged.

## Scope

The change applies to `scripts/evaluation/run_llamacpp_inference_eval.py` and its tests.
It does not rerun model inference, alter the fixed 150-question benchmark, or
change Jetson timing data.

## Rubric

The evaluator returns:

- `correctness_score` from 0 to 10: whether the answer is factually and
  operationally correct;
- `completeness_score` from 0 to 10: coverage of the reference answer's key
  requirements;
- `groundedness_score` from 0 to 10: how fully factual claims are supported by
  the supplied reference;
- `hallucination_risk` as `low`, `medium`, or `high`: reserved for false,
  contradictory, unsafe, or action-changing unsupported claims;
- `verdict` as `pass`, `partial`, or `fail`;
- a concise `reason` that explains the main scoring decision.

Accurate and harmless context absent from the reference may lower groundedness,
but must not substantially lower correctness. False, contradictory, unsafe, or
action-changing additions lower correctness and raise hallucination risk.

Scoring anchors:

- 9-10: correct and covers nearly all required points;
- 7-8: mostly correct with minor omissions;
- 4-6: partially correct with material omissions;
- 1-3: little usable correct content or a major misunderstanding;
- 0: entirely wrong, irrelevant, or directly dangerous.

For `cevaplanabilir=hayir`, a pass requires an explicit statement that the
provided documents do not contain the answer and no invented factual answer.

## Artifact Versioning

Existing `evaluated_results.csv` and its strict-v1 scores remain untouched.
Balanced evaluation writes:

- `evaluated_results_balanced_v2.csv`;
- `summary_balanced_v2.json`;
- v2 progress entries in the existing run log.

Resume skips only successful rows in the balanced-v2 artifact. Inference rows
continue to come from the existing `raw_results.csv` files.

## Summary Metrics

The v2 summary reports correctness, completeness, and groundedness means for:

- all successfully evaluated rows;
- answerable rows;
- unanswerable rows.

It also reports unanswerable grounded-refusal accuracy, hallucination rate, and
verdict counts. Inference timing is copied from the existing raw results and is
never recomputed from evaluator latency.

## Testing

Tests must prove that:

1. the prompt distinguishes harmless extra context from harmful unsupported
   claims;
2. v2 JSON parsing validates `groundedness_score` and existing fields;
3. v2 files do not overwrite strict-v1 artifacts;
4. resume skips successful v2 evaluations while retrying failed ones;
5. summaries split answerable and unanswerable metrics correctly;
6. all existing inference and evaluation tests remain green.

## Acceptance Criteria

The implementation is complete when the focused tests and full evaluator test
suite pass, all six saved conditions produce 150 successful balanced-v2 rows,
strict-v1 artifacts remain byte-for-byte present, and the selected RAG
condition is based on the balanced quality and speed results.
