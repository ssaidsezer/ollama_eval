from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments import run_speculative_ablation as ablation


QWEN_MODEL = "/models/Qwen3.6-35B-A3B-Q4_K_M.gguf"
GEMMA_MODEL = "/models/gemma-4-26B-A4B-it-Q4_K_M.gguf"
EAGLE_MODEL = "/models/gemma-4-26B-A4B-eagle3.gguf"


def test_build_conditions_creates_the_fixed_three_condition_matrix() -> None:
    conditions = ablation.build_conditions(
        ["qwen_ngram", "gemma_ngram", "gemma_eagle3"],
        qwen_model=QWEN_MODEL,
        gemma_model=GEMMA_MODEL,
        gemma_eagle_model=EAGLE_MODEL,
    )

    assert [condition.name for condition in conditions] == [
        "qwen_ngram",
        "gemma_ngram",
        "gemma_eagle3",
    ]
    assert conditions[0].model_alias == "qwen3.6:35b-a3b"
    assert conditions[0].spec_args == (
        "--spec-type",
        "ngram-mod",
        "--spec-ngram-mod-n-match",
        "24",
        "--spec-ngram-mod-n-min",
        "48",
        "--spec-ngram-mod-n-max",
        "64",
    )
    assert conditions[2].draft_model == EAGLE_MODEL
    assert conditions[2].spec_args == (
        "--spec-type",
        "draft-eagle3",
        "--spec-draft-n-max",
        "3",
    )


def test_build_conditions_requires_eagle_model_only_when_eagle_is_selected() -> None:
    assert ablation.build_conditions(
        ["qwen_ngram"],
        qwen_model=QWEN_MODEL,
        gemma_model=GEMMA_MODEL,
        gemma_eagle_model="",
    )

    with pytest.raises(ValueError, match="gemma-eagle-model"):
        ablation.build_conditions(
            ["gemma_eagle3"],
            qwen_model=QWEN_MODEL,
            gemma_model=GEMMA_MODEL,
            gemma_eagle_model="",
        )


def test_build_server_command_freezes_benchmark_controls() -> None:
    condition = ablation.build_conditions(
        ["gemma_eagle3"],
        qwen_model=QWEN_MODEL,
        gemma_model=GEMMA_MODEL,
        gemma_eagle_model=EAGLE_MODEL,
    )[0]

    command = ablation.build_server_command(
        condition,
        server_binary="/opt/llama-server",
        remote_port=8084,
    )

    assert command[:3] == ["/opt/llama-server", "--model", GEMMA_MODEL]
    assert command[command.index("--model-draft") + 1] == EAGLE_MODEL
    for pair in (
        ("--ctx-size", "32768"),
        ("--parallel", "1"),
        ("--flash-attn", "on"),
        ("--cache-type-k", "q8_0"),
        ("--cache-type-v", "q8_0"),
        ("--host", "127.0.0.1"),
        ("--port", "8084"),
    ):
        assert command[command.index(pair[0]) + 1] == pair[1]
    assert "--no-cache-prompt" in command


def test_parse_speculative_stats_uses_latest_cumulative_totals() -> None:
    log = """
draft acceptance rate = 0.50000 (  5 accepted /  10 generated)
statistics ngram_mod: #calls = 20, #gen drafts = 3, #acc drafts = 2, #gen tokens = 12, #acc tokens = 7
draft acceptance rate = 0.62500 (  25 accepted /  40 generated)
statistics ngram_mod: #calls = 80, #gen drafts = 9, #acc drafts = 8, #gen tokens = 40, #acc tokens = 25
"""

    stats = ablation.parse_speculative_stats(log, expected_type="ngram-mod")

    assert stats == {
        "spec_type": "ngram-mod",
        "draft_generated_tokens": 40,
        "draft_accepted_tokens": 25,
        "draft_acceptance_rate": 0.625,
        "statistics_found": True,
    }


def test_parse_speculative_stats_reads_per_slot_print_timing_format() -> None:
    # Real llama-server print_timing output: no "rate" keyword, trailing "mean len".
    log = (
        "2.58.954.941 I slot print_timing: id  0 | task 4955 | "
        "draft acceptance = 0.03125 (    2 accepted /    64 generated), mean len =  3.00\n"
    )

    stats = ablation.parse_speculative_stats(log, expected_type="ngram-mod")

    assert stats == {
        "spec_type": "ngram-mod",
        "draft_generated_tokens": 64,
        "draft_accepted_tokens": 2,
        "draft_acceptance_rate": 0.03125,
        "statistics_found": True,
    }


def test_parse_speculative_stats_reports_missing_without_inventing_zeroes() -> None:
    stats = ablation.parse_speculative_stats("server started", expected_type="draft-eagle3")

    assert stats["statistics_found"] is False
    assert stats["draft_generated_tokens"] is None
    assert stats["draft_accepted_tokens"] is None
    assert stats["draft_acceptance_rate"] is None


def test_validate_baseline_requires_same_frozen_inference_controls(tmp_path: Path) -> None:
    baseline = tmp_path / "summary.json"
    baseline.write_text(
        json.dumps(
            {
                "config": {
                    "model": "qwen3.6:35b-a3b",
                    "csv": "data/qa/final_qa_150.csv",
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                "counts": {"total": 150, "successful": 150, "failed": 0},
            }
        ),
        encoding="utf-8",
    )

    loaded = ablation.validate_baseline_summary(
        baseline,
        expected_model="qwen3.6:35b-a3b",
        expected_csv=Path("data/qa/final_qa_150.csv"),
    )
    assert loaded["counts"]["successful"] == 150

    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["config"]["temperature"] = 0.2
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="temperature"):
        ablation.validate_baseline_summary(
            baseline,
            expected_model="qwen3.6:35b-a3b",
            expected_csv=Path("data/qa/final_qa_150.csv"),
        )


def test_validate_baseline_quality_requires_all_150_evaluations() -> None:
    with pytest.raises(ValueError, match="150 successful quality evaluations"):
        ablation.validate_baseline_quality({"evaluation": {"successful": 0, "skipped": 150}})

    ablation.validate_baseline_quality({"evaluation": {"successful": 150, "skipped": 0}})


def test_validate_baseline_server_command_requires_q8_and_frozen_server_controls() -> None:
    valid = (
        "/opt/llama-server --model /models/Qwen-Q4_K_M.gguf "
        "--ctx-size 32768 --parallel 1 --flash-attn on "
        "--cache-type-k q8_0 --cache-type-v q8_0 --no-cache-prompt"
    )
    ablation.validate_baseline_server_command(valid)

    with pytest.raises(ValueError, match="--cache-type-v q8_0"):
        ablation.validate_baseline_server_command(valid.replace("--cache-type-v q8_0", "--cache-type-v f16"))


def test_condition_is_resumable_only_with_complete_summary_and_stats(tmp_path: Path) -> None:
    condition_dir = tmp_path / "qwen_ngram"
    condition_dir.mkdir()
    (condition_dir / "summary.json").write_text(
        json.dumps({"counts": {"total": 150, "successful": 150}}), encoding="utf-8"
    )
    assert ablation.condition_is_resumable(condition_dir) is False

    (condition_dir / "speculative_stats.json").write_text(
        json.dumps({"statistics_found": True}), encoding="utf-8"
    )
    assert ablation.condition_is_resumable(condition_dir) is True


def test_comparison_requires_speed_quality_and_no_new_truncation() -> None:
    condition = ablation.build_conditions(
        ["qwen_ngram"],
        qwen_model=QWEN_MODEL,
        gemma_model=GEMMA_MODEL,
        gemma_eagle_model="",
    )[0]

    def summary(latency: float, tps: float, correctness: float, truncations: int) -> dict:
        return {
            "inference": {
                "wall_time_seconds": {"median": latency},
                "generation_tokens_per_second": {"median": tps},
            },
            "evaluation": {
                "correctness_score_mean": correctness,
                "completeness_score_mean": correctness,
            },
            "counts": {"truncated": truncations},
        }

    row = ablation.build_comparison_row(
        condition,
        baseline=summary(10, 20, 8, 0),
        result=summary(8, 25, 8, 1),
        stats={"draft_acceptance_rate": 0.7},
        status="completed",
    )

    assert row["speed_threshold_met"] is True
    assert row["quality_not_degraded"] is True
    assert row["truncation_not_degraded"] is False
    assert row["successful"] is False


def test_owned_pid_stop_command_checks_pid_and_expected_binary() -> None:
    command = ablation.build_owned_pid_stop_script(
        pid_file="/tmp/spec/gemma.pid",
        expected_binary="/home/odine/llama.cpp/build/bin/llama-server",
    )

    assert "kill --" in command
    assert "gemma.pid" in command
    assert "/home/odine/llama.cpp/build/bin/llama-server" in command
    assert "pkill" not in command


@pytest.mark.parametrize(
    ("log_text", "expected"),
    [
        ("CUDA error: out of memory", "cuda_allocation_failed"),
        ("error loading model: failed to load model", "model_load_failed"),
        ("unknown argument: --spec-type", "unsupported_speculative_flag"),
        ("server is loading", "health_timeout"),
    ],
)
def test_classify_server_start_failure(log_text: str, expected: str) -> None:
    assert ablation.classify_server_start_failure(log_text) == expected
