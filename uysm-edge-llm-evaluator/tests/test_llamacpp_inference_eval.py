from __future__ import annotations

import json
import csv
import time
from pathlib import Path

import pytest

from prompts import build_no_rag_prompt
from scripts.evaluation import run_llamacpp_inference_eval as benchmark


def test_build_inference_payload_uses_repo_no_rag_prompt_and_disables_thinking() -> None:
    payload = benchmark.build_inference_payload(
        "qwen3.6:35b-a3b",
        "Soru?",
        max_tokens=512,
        temperature=0.0,
    )

    assert payload == {
        "model": "qwen3.6:35b-a3b",
        "messages": [{"role": "user", "content": build_no_rag_prompt("Soru?")}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
        "max_tokens": 512,
    }


def test_parse_inference_response_keeps_cache_and_server_timings() -> None:
    parsed = benchmark.parse_inference_response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Ankara", "reasoning_content": ""},
                }
            ],
            "usage": {
                "prompt_tokens": 31,
                "completion_tokens": 13,
                "total_tokens": 44,
                "prompt_tokens_details": {"cached_tokens": 26},
            },
            "timings": {
                "cache_n": 26,
                "prompt_n": 5,
                "prompt_ms": 262.95,
                "prompt_per_second": 19.01,
                "predicted_n": 13,
                "predicted_ms": 439.76,
                "predicted_per_second": 29.56,
            },
        },
        wall_time_seconds=0.75,
    )

    assert parsed["model_answer"] == "Ankara"
    assert parsed["cached_tokens"] == 26
    assert parsed["processed_prompt_tokens"] == 5
    assert parsed["generation_tokens_per_second"] == 29.56
    assert parsed["wall_time_seconds"] == 0.75
    assert parsed["finish_reason"] == "stop"


def test_parse_inference_response_rejects_empty_answers() -> None:
    with pytest.raises(ValueError, match="empty model answer"):
        benchmark.parse_inference_response(
            {
                "choices": [{"finish_reason": "length", "message": {"content": ""}}],
                "usage": {},
                "timings": {},
            },
            wall_time_seconds=1.0,
        )


def test_run_inference_row_returns_structured_failure() -> None:
    def failing_post(*args, **kwargs):
        raise TimeoutError("timed out")

    row = benchmark.run_inference_row(
        question_index=7,
        question="Soru?",
        reference_answer="Cevap",
        run_label="run",
        model="model",
        base_url="http://host/v1",
        max_tokens=32,
        temperature=0.0,
        timeout=2.0,
        attempt_count=1,
        post_fn=failing_post,
    )

    assert row["inference_status"] == "failed"
    assert row["error_type"] == "TimeoutError"
    assert row["error_message"] == "timed out"
    assert row["question_index"] == 7
    assert row["attempt_count"] == 1


def test_run_inference_row_posts_to_chat_completions() -> None:
    seen = {}

    def fake_post(url, payload, timeout):
        seen.update(url=url, payload=payload, timeout=timeout)
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": "Yanıt"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            "timings": {
                "cache_n": 4,
                "prompt_n": 6,
                "prompt_ms": 100.0,
                "prompt_per_second": 60.0,
                "predicted_n": 2,
                "predicted_ms": 50.0,
                "predicted_per_second": 40.0,
            },
        }

    row = benchmark.run_inference_row(
        question_index=0,
        question="Soru?",
        reference_answer="Yanıt",
        run_label="run",
        model="model",
        base_url="http://host/v1/",
        max_tokens=32,
        temperature=0.0,
        timeout=2.0,
        attempt_count=1,
        post_fn=fake_post,
    )

    assert seen["url"] == "http://host/v1/chat/completions"
    assert seen["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert row["inference_status"] == "success"
    assert row["reference_answer"] == "Yanıt"
    assert row["cached_tokens"] == 4


def test_atomic_checkpoint_round_trip_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "raw_results.csv"
    fieldnames = ("question_index", "inference_status", "model_answer")
    rows = [
        {"question_index": 0, "inference_status": "success", "model_answer": "A"},
        {"question_index": 1, "inference_status": "failed", "model_answer": ""},
    ]

    benchmark.write_csv_atomic(path, rows, fieldnames)
    loaded = benchmark.read_csv_rows(path)

    assert benchmark.successful_indices(loaded, "inference_status") == {0}
    assert loaded[0]["model_answer"] == "A"
    assert not list(path.parent.glob("*.tmp"))


def test_nearest_rank_handles_empty_and_small_samples() -> None:
    assert benchmark.nearest_rank([], 0.95) is None
    assert benchmark.nearest_rank([3.0], 0.99) == 3.0
    assert benchmark.nearest_rank([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert benchmark.nearest_rank([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_summary_reports_latency_throughput_cache_and_failures() -> None:
    rows = [
        {
            "inference_status": "success",
            "wall_time_seconds": 1.0,
            "prompt_ms": 200,
            "generation_ms": 800,
            "prompt_tokens_per_second": 50,
            "generation_tokens_per_second": 25,
            "prompt_tokens": 20,
            "cached_tokens": 10,
            "completion_tokens": 20,
        },
        {
            "inference_status": "success",
            "wall_time_seconds": 2.0,
            "prompt_ms": 400,
            "generation_ms": 1600,
            "prompt_tokens_per_second": 40,
            "generation_tokens_per_second": 20,
            "prompt_tokens": 20,
            "cached_tokens": 0,
            "completion_tokens": 40,
        },
        {"inference_status": "failed"},
    ]

    summary = benchmark.build_summary(rows, {"run_label": "run"})

    assert summary["counts"] == {"total": 3, "successful": 2, "failed": 1}
    assert summary["inference"]["wall_time_seconds"]["p50"] == 1.0
    assert summary["inference"]["wall_time_seconds"]["mean"] == 1.5
    assert summary["inference"]["measured_wall_time_seconds_total"] == 3.0
    assert summary["cache"]["cached_token_ratio"] == 0.25
    assert summary["tokens"]["completion_total"] == 60


def test_parse_evaluation_validates_ranges_and_enums() -> None:
    parsed = benchmark.parse_evaluation(
        json.dumps(
            {
                "correctness_score": 9,
                "completeness_score": 8,
                "hallucination_risk": "low",
                "verdict": "pass",
                "reason": "Referansla uyumlu.",
            }
        )
    )

    assert parsed["correctness_score"] == 9.0
    assert parsed["verdict"] == "pass"

    with pytest.raises(ValueError, match="correctness_score"):
        benchmark.parse_evaluation(
            json.dumps(
                {
                    "correctness_score": 11,
                    "completeness_score": 8,
                    "hallucination_risk": "low",
                    "verdict": "pass",
                    "reason": "Hatalı skor.",
                }
            )
        )


def test_evaluate_row_uses_gpt_model_and_keeps_inference_telemetry() -> None:
    seen = {}

    class FakeCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)
            message = type(
                "Message",
                (),
                {
                    "content": json.dumps(
                        {
                            "correctness_score": 10,
                            "completeness_score": 9,
                            "hallucination_risk": "low",
                            "verdict": "pass",
                            "reason": "Doğru.",
                        }
                    )
                },
            )()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
    )()
    raw = {
        "question_index": 0,
        "question": "Başkent?",
        "reference_answer": "Ankara",
        "model_answer": "Ankara'dır.",
        "inference_status": "success",
        "wall_time_seconds": 0.7,
    }

    evaluated = benchmark.evaluate_row(raw, client=client, eval_model="gpt-5.4-mini")

    assert seen["model"] == "gpt-5.4-mini"
    assert seen["response_format"] == {"type": "json_object"}
    assert evaluated["evaluation_status"] == "success"
    assert evaluated["correctness_score"] == 10.0
    assert evaluated["wall_time_seconds"] == 0.7
    assert evaluated["evaluation_duration_seconds"] >= 0


def test_evaluator_receives_unanswerable_grounded_refusal_rubric() -> None:
    messages = benchmark.build_eval_messages(
        "Bu doz belgelerde var mı?",
        "Sağlanan belgelerde bu bilgi yer almıyor.",
        "Verilen belgelerde bu bilgi yer almıyor.",
        "hayir",
    )

    assert "cevaplanamaz" in messages[0]["content"]
    assert "desteksiz bir doz vermeyi fail say" in messages[0]["content"]
    assert "Cevaplanabilir: hayir" in messages[1]["content"]


def test_evaluation_summary_splits_answerable_and_unanswerable_metrics() -> None:
    summary = benchmark._evaluation_summary(
        [
            {
                "evaluation_status": "success",
                "answerability": "evet",
                "correctness_score": 8,
                "completeness_score": 7,
                "verdict": "pass",
                "hallucination_risk": "low",
            },
            {
                "evaluation_status": "success",
                "answerability": "hayir",
                "correctness_score": 10,
                "completeness_score": 10,
                "verdict": "pass",
                "hallucination_risk": "low",
            },
            {
                "evaluation_status": "success",
                "answerability": "hayir",
                "correctness_score": 0,
                "completeness_score": 0,
                "verdict": "fail",
                "hallucination_risk": "high",
            },
        ]
    )

    assert summary["answerable"]["successful"] == 1
    assert summary["answerable"]["correctness_score_mean"] == 8.0
    assert summary["unanswerable"]["successful"] == 2
    assert summary["unanswerable"]["grounded_refusal_accuracy"] == 0.5
    assert summary["unanswerable"]["hallucination_rate"] == 0.5


def test_evaluate_row_skips_failed_inference() -> None:
    row = benchmark.evaluate_row(
        {
            "question_index": 1,
            "question": "Soru?",
            "reference_answer": "Cevap",
            "model_answer": "",
            "inference_status": "failed",
        },
        client=object(),
        eval_model="gpt-5.4-mini",
    )

    assert row["evaluation_status"] == "skipped"
    assert row["evaluation_error_type"] == "InferenceUnavailable"


def test_parse_args_uses_qwen_speed_eval_defaults() -> None:
    args = benchmark.parse_args([])

    assert args.csv.endswith("data/qa/old/question_answers_100.csv")
    assert args.base_url == "http://192.168.1.29:8084/v1"
    assert args.model == "qwen3.6:35b-a3b"
    assert args.eval_model == "gpt-5.4-mini"
    assert args.run_label == "qwen3.6-35b-a3b_q4km_kv-q8_prompt-cache-on"
    assert args.max_tokens == 512
    assert args.temperature == 0.0


def test_load_questions_requires_configured_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("foo,bar\na,b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="question.*answer"):
        benchmark.load_questions(csv_path, "question", "answer")


def test_execute_run_keeps_warmup_outside_results_and_evaluates_after_inference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "questions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(
            [
                {"question": "Soru 1?", "answer": "Cevap 1"},
                {"question": "Soru 2?", "answer": "Cevap 2"},
            ]
        )

    events: list[str] = []
    inference_counter = 0

    def fake_post(url, payload, timeout):
        nonlocal inference_counter
        inference_counter += 1
        events.append(f"inference:{payload['messages'][0]['content']}")
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": f"Yanıt {inference_counter}"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            "timings": {
                "cache_n": 4,
                "prompt_n": 6,
                "prompt_ms": 100,
                "prompt_per_second": 60,
                "predicted_n": 2,
                "predicted_ms": 50,
                "predicted_per_second": 40,
            },
        }

    class FakeCompletions:
        def create(self, **kwargs):
            events.append("evaluation")
            message = type(
                "Message",
                (),
                {
                    "content": json.dumps(
                        {
                            "correctness_score": 8,
                            "completeness_score": 8,
                            "hallucination_risk": "low",
                            "verdict": "pass",
                            "reason": "Uyumlu.",
                        }
                    )
                },
            )()
            return type(
                "Response",
                (),
                {"choices": [type("Choice", (), {"message": message})()]},
            )()

    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
    )()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    out_dir = tmp_path / "run"
    args = benchmark.parse_args(
        [
            "--csv",
            str(csv_path),
            "--out-dir",
            str(out_dir),
            "--base-url",
            "http://jetson:8084/v1",
        ]
    )

    exit_code = benchmark.execute_run(
        args,
        post_fn=fake_post,
        health_fn=lambda *_: None,
        openai_client_factory=lambda: client,
    )

    assert exit_code == 0
    raw_rows = benchmark.read_csv_rows(out_dir / "raw_results.csv")
    evaluated_rows = benchmark.read_csv_rows(out_dir / "evaluated_results.csv")
    assert len(raw_rows) == 2
    assert len(evaluated_rows) == 2
    assert inference_counter == 3  # one discarded warmup plus two measured requests
    assert events[:3] == [
        f"inference:{build_no_rag_prompt('Kısa bir yanıt ver: hazır mısın?')}",
        f"inference:{build_no_rag_prompt('Soru 1?')}",
        f"inference:{build_no_rag_prompt('Soru 2?')}",
    ]
    assert events[3:] == ["evaluation", "evaluation"]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"] == {"total": 2, "successful": 2, "failed": 0}
    assert summary["evaluation"]["successful"] == 2


def _questions_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "questions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerow({"question": "Soru 1?", "answer": "Cevap 1"})
    return csv_path


def _ok_post(url, payload, timeout):
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": "Yanıt"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        "timings": {"prompt_n": 10, "prompt_ms": 100, "predicted_n": 2, "predicted_ms": 50},
    }


def _failing_client(exc: Exception):
    class FakeCompletions:
        def create(self, **kwargs):
            raise exc

    return type(
        "Client", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()}
    )()


def test_execute_run_fails_when_every_evaluation_fails(tmp_path: Path, monkeypatch) -> None:
    """Inference succeeding is not enough: an all-failed judge leaves the summary empty."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    args = benchmark.parse_args(
        ["--csv", str(_questions_csv(tmp_path)), "--out-dir", str(tmp_path / "run"),
         "--base-url", "http://jetson:8084/v1"]
    )

    exit_code = benchmark.execute_run(
        args,
        post_fn=_ok_post,
        health_fn=lambda *_: None,
        openai_client_factory=lambda: _failing_client(RuntimeError("judge down")),
    )

    assert exit_code == 3
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["successful"] == 1
    assert summary["evaluation"]["successful"] == 0
    assert summary["evaluation"]["correctness_score_mean"] is None


def test_execute_run_still_succeeds_when_evaluation_is_skipped(tmp_path: Path) -> None:
    args = benchmark.parse_args(
        ["--csv", str(_questions_csv(tmp_path)), "--out-dir", str(tmp_path / "run"),
         "--base-url", "http://jetson:8084/v1", "--skip-eval"]
    )

    exit_code = benchmark.execute_run(args, post_fn=_ok_post, health_fn=lambda *_: None)

    assert exit_code == 0


def test_evaluate_row_retries_transient_failures_once() -> None:
    calls = {"n": 0}

    class FlakyCompletions:
        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("rate limited")
            message = type("Message", (), {"content": json.dumps({
                "correctness_score": 8, "completeness_score": 7,
                "hallucination_risk": "low", "verdict": "pass", "reason": "iyi",
            })})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": FlakyCompletions()})()})()
    row = {"inference_status": "success", "model_answer": "cevap", "question": "soru",
           "reference_answer": "ref", "answerability": "evet", "question_index": 0}

    result = benchmark.evaluate_row(row, client=client, eval_model="m", sleep_fn=lambda _: None)

    assert calls["n"] == 2
    assert result["evaluation_status"] == "success"
    assert result["evaluation_attempt_count"] == 2


def test_failed_evaluation_records_how_many_attempts_it_burned() -> None:
    """A retried-then-failed row must be distinguishable from a single-attempt failure."""
    class DownCompletions:
        def create(self, **kwargs):
            raise TimeoutError("upstream timeout")

    client = type("Client", (), {"chat": type("Chat", (), {"completions": DownCompletions()})()})()
    row = {"inference_status": "success", "model_answer": "cevap", "question": "soru",
           "reference_answer": "ref", "answerability": "evet", "question_index": 0}

    result = benchmark.evaluate_row(row, client=client, eval_model="m", sleep_fn=lambda _: None)

    assert result["evaluation_status"] == "failed"
    assert result["evaluation_attempt_count"] == 2


def test_skipped_evaluation_records_zero_attempts() -> None:
    result = benchmark.evaluate_row(
        {"inference_status": "error", "model_answer": "", "question_index": 0},
        client=object(),
        eval_model="m",
    )

    assert result["evaluation_status"] == "skipped"
    assert result["evaluation_attempt_count"] == 0


def test_evaluation_duration_excludes_retry_backoff() -> None:
    """Duration is judge latency; counting the backoff would inflate the benchmark."""
    calls = {"n": 0}

    class FlakyCompletions:
        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("upstream timeout")
            message = type("Message", (), {"content": json.dumps({
                "correctness_score": 9, "completeness_score": 9,
                "hallucination_risk": "low", "verdict": "pass", "reason": "iyi",
            })})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": FlakyCompletions()})()})()
    row = {"inference_status": "success", "model_answer": "cevap", "question": "soru",
           "reference_answer": "ref", "answerability": "evet", "question_index": 0}

    slept: list[float] = []

    def sleep_fn(seconds: float) -> None:
        slept.append(seconds)
        time.sleep(0.05)

    result = benchmark.evaluate_row(row, client=client, eval_model="m", sleep_fn=sleep_fn)

    assert slept == [1]
    assert result["evaluation_duration_seconds"] < 0.05


def test_evaluate_row_does_not_retry_a_permanent_failure() -> None:
    calls = {"n": 0}

    class BadCompletions:
        def create(self, **kwargs):
            calls["n"] += 1
            raise ValueError("model not found")

    client = type("Client", (), {"chat": type("Chat", (), {"completions": BadCompletions()})()})()
    row = {"inference_status": "success", "model_answer": "cevap", "question": "soru",
           "reference_answer": "ref", "answerability": "evet", "question_index": 0}

    result = benchmark.evaluate_row(row, client=client, eval_model="m", sleep_fn=lambda _: None)

    assert calls["n"] == 1
    assert result["evaluation_status"] == "failed"
    assert result["evaluation_error_type"] == "ValueError"
