from __future__ import annotations

from types import SimpleNamespace
import json
import uuid

import requests
import pandas as pd
import pytest
import httpx
import openai

import pipeline
from scripts.evaluation import run_rag_ablation
from rag import bm25, indexing, qdrant_store, retrieval, smart_indexing
from rag.chunking import (
    _child_block_id,
    _chunk_id,
    _chunk_text,
    _normalize_text_for_dedup,
    _parent_block_id,
    _split_children_from_parent,
)
from rag.pdf_text import _clean_pdf_text
from web_backend.chat_service import run_chat_eval
from web_backend import analysis_service
from web_backend import model_profiles_store
from web_backend.schemas import ChatEvalRequest


@pytest.fixture(autouse=True)
def clear_bm25_cache_between_tests():
    bm25._BM25_CACHE.clear()
    yield
    bm25._BM25_CACHE.clear()


def test_normalize_text_for_dedup_collapses_whitespace() -> None:
    assert _normalize_text_for_dedup("  alfa\n\n beta\tgamma  ") == "alfa beta gamma"


def test_chunk_id_is_stable_uuid_for_same_source_page_and_index() -> None:
    first = _chunk_id("docs/example.pdf", 2, 7)
    second = _chunk_id("docs/example.pdf", 2, 7)

    assert first == second
    assert uuid.UUID(first).version == 5


def test_chunk_text_uses_overlap_without_splitting_words() -> None:
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."

    chunks = _chunk_text(text, chunk_size=28, chunk_overlap=6)

    assert chunks == [
        "Alpha beta gamma.",
        "gamma. Delta epsilon zeta.",
        "zeta. Eta theta iota.",
    ]


def test_clean_pdf_text_removes_common_pdf_noise() -> None:
    noisy = "\x00Başlık\n---\nSayfa 12\n••\nResim 1-12\nİçerik satırı\n\n\nDevam\n19"

    assert _clean_pdf_text(noisy) == "Başlık\n\nİçerik satırı\n\nDevam"


def test_external_eval_provider_helpers_select_keys_and_clients(monkeypatch) -> None:
    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(pipeline, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")

    assert pipeline.is_external_eval_backend("DeepSeek") is True
    assert pipeline.is_external_eval_backend("OpenAI") is True
    assert pipeline.is_external_eval_backend("Yerel (Ollama)") is False
    assert pipeline.get_eval_api_key("DeepSeek") == "deepseek-env-key"
    assert pipeline.get_eval_api_key("OpenAI") == "openai-env-key"
    assert pipeline.get_eval_api_key("OpenAI", request_key="request-key") == "request-key"

    pipeline.get_external_eval_client("DeepSeek")
    pipeline.get_external_eval_client("OpenAI")

    assert calls == [
        {"api_key": "deepseek-env-key", "base_url": "https://api.deepseek.com"},
        {"api_key": "openai-env-key"},
    ]


def test_openai_eval_model_default_is_gpt_5_4_mini(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_EVAL_MODEL_NAME", raising=False)
    monkeypatch.setattr(pipeline, "OPENAI_EVAL_MODEL_NAME", None)
    monkeypatch.setattr(pipeline, "EVAL_MODEL_NAME", "deepseek-v4-pro")

    assert pipeline.get_default_eval_model("OpenAI") == "gpt-5.4-mini"


def test_chat_eval_request_accepts_openai_backend() -> None:
    req = ChatEvalRequest(
        question="Soru?",
        rag_mode="no_rag",
        eval_backend="OpenAI",
        qa_models_selected=["qwen3:1.7b"],
    )

    assert req.eval_backend == "OpenAI"


def test_rag_ablation_eval_backend_choices_map_to_pipeline_backends() -> None:
    assert run_rag_ablation.resolve_eval_backend("deepseek") == "deepseek"
    assert run_rag_ablation.resolve_eval_backend("openai") == "openai"
    assert run_rag_ablation.resolve_eval_backend("local") == "ollama"


def test_ablation_cli_defaults_bound_ollama_generation(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_rag_ablation.py"])

    args = run_rag_ablation.parse_args()

    assert args.timeout == 300
    assert args.num_predict == 512
    assert args.embedding_timeout == 300
    assert args.temperature == 0
    assert args.seed == 42
    assert args.qa_backend == "ollama"
    assert args.qa_base_url == ""


def test_ablation_cli_accepts_llamacpp_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rag_ablation.py",
            "--qa-backend",
            "llamacpp",
            "--qa-base-url",
            "http://127.0.0.1:8084/v1",
        ],
    )

    args = run_rag_ablation.parse_args()

    assert args.qa_backend == "llamacpp"
    assert args.qa_base_url == "http://127.0.0.1:8084/v1"


def test_ablation_llamacpp_requires_base_url_and_explicit_models() -> None:
    missing_url = SimpleNamespace(
        qa_backend="llamacpp", qa_base_url="", use_all=False, list_models=False
    )
    with pytest.raises(ValueError, match="qa-base-url"):
        run_rag_ablation.validate_backend_args(missing_url)

    discovery = SimpleNamespace(
        qa_backend="llamacpp",
        qa_base_url="http://127.0.0.1:8084/v1",
        use_all=True,
        list_models=False,
    )
    with pytest.raises(ValueError, match="Ollama"):
        run_rag_ablation.validate_backend_args(discovery)


def test_generate_rag_answer_llamacpp_posts_openai_chat_and_parses_metrics(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Doğru cevap."},
                    }
                ],
                "usage": {
                    "prompt_tokens": 31,
                    "completion_tokens": 15,
                    "total_tokens": 46,
                    "prompt_tokens_details": {"cached_tokens": 7},
                },
                "timings": {
                    "prompt_ms": 300.0,
                    "predicted_ms": 500.0,
                    "predicted_per_second": 30.0,
                },
            }

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse()

    monkeypatch.setattr(pipeline.requests, "post", fake_post)

    result = pipeline.generate_rag_answer_llamacpp(
        question="Soru?",
        context="Kaynak bağlam",
        model="Qwen3.6-35B-A3B",
        base_url="http://127.0.0.1:8084/v1",
        timeout=45,
        options={"num_predict": 128, "temperature": 0.0, "seed": 42},
    )

    assert calls[0][0] == "http://127.0.0.1:8084/v1/chat/completions"
    assert calls[0][2] == 45
    payload = calls[0][1]
    assert payload["model"] == "Qwen3.6-35B-A3B"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 42
    assert "Kaynak bağlam" in payload["messages"][-1]["content"]
    assert result["answer"] == "Doğru cevap."
    assert result["eval_count"] == 15
    assert result["prompt_eval_count"] == 31
    assert result["tokens_per_second"] == 30.0
    assert result["eval_duration_seconds"] == 0.5


def test_generate_rag_answer_llamacpp_rejects_empty_content(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "", "reasoning_content": "thinking"},
                    }
                ]
            }

    monkeypatch.setattr(pipeline.requests, "post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(ValueError, match="empty model answer"):
        pipeline.generate_rag_answer_llamacpp(
            question="Soru?",
            context="Bağlam",
            model="model",
            base_url="http://127.0.0.1:8084/v1",
        )


def test_pipeline_llamacpp_backend_skips_ollama_warmup_and_routes_generation(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        pipeline,
        "load_questions",
        lambda *args, **kwargs: [
            {"question_index": 0, "question": "Soru?", "observation_idea": "Beklenen"}
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "warmup_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Ollama warmup called")),
    )
    monkeypatch.setattr(
        pipeline,
        "retrieve_chunks",
        lambda **kwargs: [{"text": "Kaynak bağlam", "source": "manual.pdf"}],
    )

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return {
            "answer": "Üretilen",
            "response_time_seconds": 0.2,
            "eval_count": 12,
        }

    monkeypatch.setattr(pipeline, "generate_rag_answer_llamacpp", fake_generate)

    rows = pipeline.run_full_pipeline(
        csv_path="input.csv",
        qa_model="Qwen3.6-35B-A3B",
        qa_backend="llamacpp",
        qa_base_url="http://127.0.0.1:8084/v1",
        rag_mode="rag",
        eval_enabled=False,
    )

    assert rows[0]["model_answer"] == "Üretilen"
    assert calls == [
        {
            "question": "Soru?",
            "context": "Kaynak bağlam",
            "model": "Qwen3.6-35B-A3B",
            "base_url": "http://127.0.0.1:8084/v1",
            "system_prompt": None,
            "options": None,
            "timeout": 120,
        }
    ]


def test_rag_ablation_help_distinguishes_classic_and_smart_k(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["run_rag_ablation.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        run_rag_ablation.parse_args()

    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "classic: maximum chunks" in help_text
    assert "smart: maximum unique parents" in help_text


def test_resilient_pipeline_retries_timeout_then_continues(monkeypatch) -> None:
    questions = [
        {"question_index": 0, "question": "ilk", "observation_idea": "a"},
        {"question_index": 1, "question": "ikinci", "observation_idea": "b"},
    ]
    attempts = {0: 0, 1: 0}
    emitted = []

    monkeypatch.setattr(pipeline, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(pipeline, "warmup_model", lambda model, timeout: None)
    monkeypatch.setattr(pipeline, "retrieve_chunks", lambda **kwargs: [])

    def fake_generate(*, question, **kwargs):
        index = 0 if question == "ilk" else 1
        attempts[index] += 1
        if index == 0:
            raise requests.Timeout("slow")
        return {"answer": "ok", "response_time_seconds": 0.1, "eval_count": 7}

    monkeypatch.setattr(pipeline, "generate_rag_answer_ollama", fake_generate)

    rows = pipeline.run_full_pipeline(
        csv_path="input.csv",
        qa_model="model",
        rag_mode="rag",
        eval_enabled=False,
        continue_on_question_error=True,
        health_check=lambda: True,
        retry_backoff_seconds=0,
        row_callback=emitted.append,
    )

    assert attempts == {0: 2, 1: 1}
    assert [row["inference_status"] for row in rows] == ["error", "ok"]
    assert rows[0]["attempt_count"] == 2
    assert rows[0]["error_type"] == "Timeout"
    assert rows[1]["eval_count"] == 7
    assert emitted == rows


def test_resilient_pipeline_preserves_answer_when_evaluation_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "load_questions",
        lambda *args, **kwargs: [
            {"question_index": 0, "question": "soru", "observation_idea": "beklenen"}
        ],
    )
    monkeypatch.setattr(pipeline, "warmup_model", lambda model, timeout: None)
    monkeypatch.setattr(pipeline, "retrieve_chunks", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "generate_rag_answer_ollama",
        lambda **kwargs: {"answer": "üretilen", "response_time_seconds": 0.1},
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_answer_any",
        lambda **kwargs: (_ for _ in ()).throw(requests.Timeout("eval slow")),
    )

    rows = pipeline.run_full_pipeline(
        csv_path="input.csv",
        qa_model="model",
        rag_mode="rag",
        eval_enabled=True,
        openai_client=object(),
        continue_on_question_error=True,
        health_check=lambda: True,
        retry_backoff_seconds=0,
    )

    assert len(rows) == 1
    assert rows[0]["model_answer"] == "üretilen"
    assert rows[0]["inference_status"] == "ok"
    assert rows[0]["evaluation_status"] == "error"
    assert rows[0]["error_stage"] == "evaluation"
    # The inference succeeded first try; only the evaluation was retried.
    assert rows[0]["attempt_count"] == 1
    assert rows[0]["inference_attempt_count"] == 1
    assert rows[0]["evaluation_attempt_count"] == 2


def test_resilient_pipeline_emits_current_error_before_service_abort(monkeypatch) -> None:
    emitted = []
    monkeypatch.setattr(
        pipeline,
        "load_questions",
        lambda *args, **kwargs: [
            {"question_index": 0, "question": "soru", "observation_idea": "cevap"}
        ],
    )
    monkeypatch.setattr(pipeline, "warmup_model", lambda model, timeout: None)
    monkeypatch.setattr(pipeline, "retrieve_chunks", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "generate_rag_answer_ollama",
        lambda **kwargs: (_ for _ in ()).throw(requests.ConnectionError("down")),
    )

    with pytest.raises(pipeline.OllamaServiceUnavailable):
        pipeline.run_full_pipeline(
            csv_path="input.csv",
            qa_model="model",
            rag_mode="rag",
            eval_enabled=False,
            continue_on_question_error=True,
            health_check=lambda: False,
            row_callback=emitted.append,
        )

    assert len(emitted) == 1
    assert emitted[0]["inference_status"] == "error"
    assert emitted[0]["question_index"] == 0


def test_analysis_filters_inference_errors_and_quality_errors() -> None:
    df = pd.DataFrame(
        [
            {"inference_status": "ok", "evaluation_status": "ok", "ai_score": 8},
            {"inference_status": "ok", "evaluation_status": "error", "ai_score": None},
            {"inference_status": "error", "evaluation_status": "skipped", "ai_score": None},
        ]
    )

    inference_rows = analysis_service._successful_inference_rows(df)
    quality_rows = analysis_service._successful_evaluation_rows(inference_rows)

    assert len(inference_rows) == 2
    assert quality_rows["ai_score"].tolist() == [8.0]


def _single_question_pipeline(monkeypatch, captured: dict) -> None:
    monkeypatch.setattr(
        pipeline,
        "load_questions",
        lambda *args, **kwargs: [
            {"question_index": 0, "question": "soru", "observation_idea": "beklenen"}
        ],
    )
    monkeypatch.setattr(pipeline, "warmup_model", lambda model, timeout: None)
    monkeypatch.setattr(pipeline, "retrieve_chunks", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "generate_rag_answer_ollama",
        lambda **kwargs: {"answer": "üretilen", "response_time_seconds": 0.1},
    )

    def _capture(**kwargs):
        captured["client"] = kwargs.get("client")
        return {"ai_score": 8, "ai_verdict": "correct"}

    monkeypatch.setattr(pipeline, "evaluate_answer_any", _capture)


@pytest.mark.parametrize("backend", ["deepseek", "openai"])
def test_external_eval_client_reaches_the_judge_for_every_external_backend(
    monkeypatch, backend: str
) -> None:
    """DeepSeek is the default backend; its client must not be dropped on the way to the judge."""
    captured: dict = {}
    _single_question_pipeline(monkeypatch, captured)
    sentinel = object()

    pipeline.run_full_pipeline(
        csv_path="input.csv",
        qa_model="model",
        rag_mode="rag",
        eval_enabled=True,
        eval_backend=backend,
        openai_client=sentinel,
    )

    assert captured["client"] is sentinel


def test_local_eval_backend_does_not_receive_an_external_client(monkeypatch) -> None:
    captured: dict = {}
    _single_question_pipeline(monkeypatch, captured)

    pipeline.run_full_pipeline(
        csv_path="input.csv",
        qa_model="model",
        rag_mode="rag",
        eval_enabled=True,
        eval_backend="ollama",
        eval_local_model="qwen3:1.7b",
        openai_client=object(),
    )

    assert captured["client"] is None


def test_analysis_response_never_contains_nan(tmp_path) -> None:
    """NaN is not a valid JSON literal; it breaks JSON.parse in the browser."""
    csv_text = (
        "model;rag_type;ai_score;ai_verdict;ai_hallucination_risk;tokens_per_second;"
        "response_time_seconds;question;model_answer;answer;inference_status;evaluation_status\n"
        "m1;RAG'li;;;;;1.5;soru;cevap;ref;ok;error\n"
        "m1;RAG'siz;;;;;1.7;soru;cevap;ref;ok;error\n"
    )
    response = analysis_service.analyze_export_csv(csv_text.encode("utf-8"))

    payload = json.dumps(response.model_dump())
    assert "NaN" not in payload and "Infinity" not in payload

    def _reject(constant: str) -> None:
        raise AssertionError(f"non-JSON constant in response: {constant}")

    json.loads(payload, parse_constant=_reject)


def test_judge_output_cannot_overwrite_measured_columns() -> None:
    clean = pipeline._sanitize_eval_fields(
        {"model": "evil", "response_time_seconds": 0.0, "ai_score": 8, "ai_reason": "iyi"}
    )

    assert "model" not in clean and "response_time_seconds" not in clean
    assert clean["ai_score"] == 8.0
    assert "dropped reserved field" in clean["evaluation_warning"]


def test_judge_output_keeps_custom_rubric_fields() -> None:
    """Eval system prompts are user-customizable, so unknown fields must survive."""
    clean = pipeline._sanitize_eval_fields({"custom_metric": 3, "ai_reason": "satır\nsonu"})

    assert clean["custom_metric"] == 3
    assert clean["ai_reason"] == "satır sonu"
    assert "evaluation_warning" not in clean


def test_judge_out_of_range_score_is_nulled_not_recorded() -> None:
    clean = pipeline._sanitize_eval_fields({"ai_score": 99, "ai_verdict": "CORRECT"})

    assert clean["ai_score"] is None
    assert clean["ai_verdict"] == "correct"
    assert "out of range" in clean["evaluation_warning"]


def test_malformed_judge_response_is_not_recorded_as_a_scored_evaluation() -> None:
    """A judge that returns an array has produced no score, so the row must not read "ok"."""
    clean = pipeline._sanitize_eval_fields([{"ai_score": 9}])

    assert clean["evaluation_status"] == "warning"
    assert "expected object" in clean["evaluation_warning"]


def test_judge_output_cannot_overwrite_retry_counts_or_status() -> None:
    clean = pipeline._sanitize_eval_fields(
        {
            "evaluation_attempt_count": 99,
            "inference_attempt_count": 99,
            "evaluation_status": "ok",
            "ai_score": 8,
        }
    )

    assert "evaluation_attempt_count" not in clean
    assert "inference_attempt_count" not in clean
    assert "evaluation_status" not in clean


def test_local_eval_normalizes_a_scheme_less_base_url(monkeypatch) -> None:
    captured: dict = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": '{"ai_score": 7}'}}

    def _post(url, **kwargs):
        captured["url"] = url
        return _Response()

    monkeypatch.setattr(pipeline.requests, "post", _post)

    pipeline._evaluate_answer_local(
        {"question": "soru", "model_answer": "cevap"},
        model="qwen3:1.7b",
        base_url="192.168.1.151:11434",
    )

    assert captured["url"] == "http://192.168.1.151:11434/api/chat"


def test_checkpoint_keeps_both_rag_and_no_rag_rows_for_one_question() -> None:
    rows: list[dict] = []
    run_rag_ablation.upsert_question_row(rows, {"question_index": 0, "rag_type": "RAG'li", "inference_status": "ok"})
    run_rag_ablation.upsert_question_row(rows, {"question_index": 0, "rag_type": "RAG'siz", "inference_status": "ok"})

    assert len(rows) == 2
    assert {row["rag_type"] for row in rows} == {"RAG'li", "RAG'siz"}

    run_rag_ablation.upsert_question_row(
        rows, {"question_index": 0, "rag_type": "RAG'li", "inference_status": "error"}
    )
    assert len(rows) == 2
    by_type = {row["rag_type"]: row["inference_status"] for row in rows}
    assert by_type == {"RAG'li": "error", "RAG'siz": "ok"}


def test_resume_requires_every_row_of_a_question_to_have_succeeded() -> None:
    rows = [
        {"question_index": 0, "rag_type": "RAG'li", "inference_status": "ok"},
        {"question_index": 0, "rag_type": "RAG'siz", "inference_status": "error"},
        {"question_index": 1, "rag_type": "RAG'li", "inference_status": "ok"},
    ]

    assert run_rag_ablation.successful_question_indices(rows) == {1}


def test_analysis_reads_comma_delimited_result_csvs() -> None:
    """Not every result CSV in this repo is semicolon-delimited."""
    csv_text = (
        "model,rag_type,ai_score,ai_verdict,ai_hallucination_risk,tokens_per_second,"
        "response_time_seconds,question,model_answer,answer,inference_status,evaluation_status\n"
        "m1,RAG'li,8,correct,low,30.0,1.5,soru,cevap,ref,ok,ok\n"
        "m1,RAG'siz,4,partial,high,31.0,1.2,soru,cevap,ref,ok,ok\n"
    )
    response = analysis_service.analyze_export_csv(csv_text.encode("utf-8"))

    assert response.warnings == []
    assert response.global_summary["rag_mean_score"] == 8.0
    assert response.global_summary["no_rag_mean_score"] == 4.0


def test_model_memory_requires_exact_model_name(monkeypatch) -> None:
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "models": [
                    {
                        "name": "qwen3-embedding:latest",
                        "size": 100 * 1024 * 1024,
                        "size_vram": 100 * 1024 * 1024,
                    },
                    {
                        "name": "qwen3:8b",
                        "size": 200 * 1024 * 1024,
                        "size_vram": 150 * 1024 * 1024,
                    },
                ]
            }

    monkeypatch.setattr(pipeline.requests, "get", lambda *args, **kwargs: Response())

    result = pipeline._get_ollama_model_memory("qwen3:8b", "http://ollama")

    assert result == {"vram_mb": 150.0, "ram_mb": 50.0}


def test_atomic_checkpoint_round_trip_and_resume_selection(tmp_path) -> None:
    path = tmp_path / "checkpoint.csv"
    rows = [
        {"question_index": 0, "inference_status": "ok", "model_answer": "a"},
        {"question_index": 1, "inference_status": "error", "model_answer": ""},
    ]

    run_rag_ablation.write_rows_atomic(path, rows)
    loaded = run_rag_ablation.read_checkpoint_rows(path)

    assert loaded == rows
    assert run_rag_ablation.successful_question_indices(loaded) == {0}


def test_resume_retries_evaluation_errors_when_eval_is_required() -> None:
    rows = [
        {"question_index": 0, "inference_status": "ok", "evaluation_status": "ok"},
        {"question_index": 1, "inference_status": "ok", "evaluation_status": "error"},
    ]

    assert run_rag_ablation.successful_question_indices(
        rows, require_evaluation=True
    ) == {0}


def test_resilient_pipeline_does_not_warm_up_when_all_questions_are_skipped(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "load_questions",
        lambda *args, **kwargs: [
            {"question_index": 0, "question": "done", "observation_idea": "answer"}
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "warmup_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("warmup called")),
    )

    rows = pipeline.run_full_pipeline(
        csv_path="input.csv",
        qa_model="model",
        eval_enabled=False,
        continue_on_question_error=True,
        skip_question_indices={0},
    )

    assert rows == []


def test_existing_output_dir_requires_explicit_resume(tmp_path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "run.log").write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="--resume"):
        run_rag_ablation.ensure_output_dir_available(out_dir, resume=False)

    run_rag_ablation.ensure_output_dir_available(out_dir, resume=True)


def test_ablation_positive_cli_validation_rejects_zero_timeout() -> None:
    args = SimpleNamespace(timeout=0, embedding_timeout=300, k=10, num_predict=512)

    with pytest.raises(ValueError, match="--timeout"):
        run_rag_ablation.validate_positive_args(args)


def test_openai_timeout_is_transient_for_evaluation_retry() -> None:
    exc = openai.APITimeoutError(request=httpx.Request("POST", "https://api.test"))

    assert pipeline.is_transient_error(exc) is True


def test_vector_retrieval_forwards_embedding_timeout(monkeypatch) -> None:
    captured = {}

    def fake_embeddings(texts, model, timeout):
        captured.update(texts=texts, model=model, timeout=timeout)
        return [[0.1, 0.2]]

    class Client:
        def query_points(self, **kwargs):
            return SimpleNamespace(points=[])

    monkeypatch.setattr(retrieval, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(retrieval, "get_qdrant_client", lambda url: Client())

    assert retrieval.retrieve_chunks(
        "soru",
        collection_name="docs",
        qdrant_url="http://qdrant",
        embed_model="embedding-model",
        embedding_timeout=321,
    ) == []
    assert captured == {
        "texts": ["soru"],
        "model": "embedding-model",
        "timeout": 321,
    }


def test_rag_ablation_eval_backend_requirements() -> None:
    assert run_rag_ablation.required_eval_env_var("deepseek") == "DEEPSEEK_API_KEY"
    assert run_rag_ablation.required_eval_env_var("openai") == "OPENAI_API_KEY"
    assert run_rag_ablation.required_eval_env_var("local") == ""


def test_rag_ablation_all_models_discovers_available_qa_models() -> None:
    calls = []

    def fake_list_models():
        calls.append("called")
        return ["llama3.1:8b", "qwen3:1.7b"], "", 0.1, 2

    models = run_rag_ablation.resolve_models(
        raw_models="",
        use_all=True,
        list_models_fn=fake_list_models,
    )

    assert calls == ["called"]
    assert models == ["llama3.1:8b", "qwen3:1.7b"]


def test_rag_ablation_models_accepts_json_list_string() -> None:
    models = run_rag_ablation.parse_models('["llama3.1:8b", "qwen3:1.7b", "gemma3:4b"]')

    assert models == ["llama3.1:8b", "qwen3:1.7b", "gemma3:4b"]


def test_rag_ablation_uses_separate_classic_and_smart_collections() -> None:
    classic = {"name": "classic_vector", "smart_rag": False}
    smart = {"name": "smart_vector", "smart_rag": True}

    assert (
        run_rag_ablation.collection_for_variant(
            classic,
            fallback_collection="fallback",
            classic_collection="classic_docs",
            smart_collection="smart_docs",
        )
        == "classic_docs"
    )
    assert (
        run_rag_ablation.collection_for_variant(
            smart,
            fallback_collection="fallback",
            classic_collection="classic_docs",
            smart_collection="smart_docs",
        )
        == "smart_docs"
    )


def test_rag_ablation_ensures_output_parent_before_writing(tmp_path) -> None:
    output_path = tmp_path / "missing" / "nested" / "result.csv"

    run_rag_ablation.ensure_output_parent(output_path)

    assert output_path.parent.is_dir()


def test_chat_eval_openai_backend_requires_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    req = ChatEvalRequest(
        question="Soru?",
        rag_mode="no_rag",
        eval_backend="OpenAI",
        qa_models_selected=["qwen3:1.7b"],
    )

    resp = run_chat_eval(req, all_models=["qwen3:1.7b"])

    assert resp.errors == [
        "OpenAI değerlendirme motoru seçili. OPENAI_API_KEY gerekli "
        "(istek gövdesinde veya OPENAI_API_KEY ortam değişkeni)."
    ]


def test_resolve_eval_profile_uses_external_eval_model_for_deepseek_and_openai(
    monkeypatch,
) -> None:
    seen_models = []

    def fake_default_profile(model_name):
        seen_models.append(model_name)
        return {
            "id": f"profile-{model_name}",
            "version": 3,
            "model_name": model_name,
            "system_prompt": f"system for {model_name}",
            "params": {"temperature": 0.2},
        }

    monkeypatch.setattr(model_profiles_store, "get_default_profile", fake_default_profile)

    deepseek_system, _, _, deepseek_extra, deepseek_trace = model_profiles_store.resolve_eval_profile(
        eval_backend="DeepSeek",
        eval_model_name="deepseek-v4-pro",
        local_eval_model_name="local-eval",
    )
    openai_system, _, _, openai_extra, openai_trace = model_profiles_store.resolve_eval_profile(
        eval_backend="OpenAI",
        eval_model_name="gpt-4o-mini",
        local_eval_model_name="local-eval",
    )

    assert seen_models == ["deepseek-v4-pro", "gpt-4o-mini"]
    assert deepseek_system == "system for deepseek-v4-pro"
    assert openai_system == "system for gpt-4o-mini"
    assert deepseek_extra == {"temperature": 0.2}
    assert openai_extra == {"temperature": 0.2}
    assert deepseek_trace["eval_profile_id"] == "profile-deepseek-v4-pro"
    assert openai_trace["eval_profile_id"] == "profile-gpt-4o-mini"


def test_split_children_from_parent_produces_stable_metadata_and_current_overlap() -> None:
    parent_text = "Birinci cümle. İkinci cümle. Üçüncü cümle."
    parent_id = _parent_block_id("doc.pdf", 3)

    children = _split_children_from_parent(
        parent_text=parent_text,
        parent_id=parent_id,
        source="doc.pdf",
        block_idx=3,
        child_size=24,
        child_overlap=8,
    )

    assert [child["text"] for child in children] == [
        "Birinci cümle.",
        "i cümle. İkinci cümle.",
        "i cümle. Üçüncü cümle.",
    ]
    assert children[0]["id"] == _child_block_id(parent_id, 0)
    assert children[0]["parent_id"] == parent_id
    assert children[0]["metadata"] == {"source": "doc.pdf", "block_idx": 3, "child_idx": 0}


def test_index_pdfs_uses_job_specific_ollama_base_url(monkeypatch) -> None:
    calls = {}

    monkeypatch.setattr(
        indexing,
        "_extract_pdf_chunks",
        lambda *args, **kwargs: [
            {
                "id": "chunk-1",
                "text": "Alpha",
                "metadata": {"source": "doc.pdf", "page": 1, "chunk_index": 0},
            }
        ],
    )

    def fake_get_embeddings(texts, model, base_url):
        calls["embeddings"] = {"texts": texts, "model": model, "base_url": base_url}
        return [[0.1, 0.2]]

    class FakeClient:
        def delete(self, **kwargs):
            calls["delete"] = kwargs

        def upsert(self, **kwargs):
            calls["upsert"] = kwargs

    monkeypatch.setattr(indexing, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(indexing, "get_qdrant_client", lambda url: FakeClient())
    monkeypatch.setattr(indexing, "ensure_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        indexing,
        "invalidate_bm25_cache",
        lambda qdrant_url, collection_name: calls.setdefault(
            "invalidated", (qdrant_url, collection_name)
        ),
        raising=False,
    )

    indexing.index_pdfs(
        ["doc.pdf"],
        collection_name="docs",
        qdrant_url="http://qdrant",
        embed_model="embeddinggemma:300m",
        ollama_base_url="http://192.168.1.151:11434",
    )

    assert calls["embeddings"] == {
        "texts": ["Alpha"],
        "model": "embeddinggemma:300m",
        "base_url": "http://192.168.1.151:11434",
    }
    assert calls["invalidated"] == ("http://qdrant", "docs")


def test_smart_index_pdfs_uses_job_specific_ollama_base_url_for_boundary_and_embeddings(
    monkeypatch,
) -> None:
    calls = {}

    monkeypatch.setattr(smart_indexing, "_extract_full_pdf_text", lambda path: "Alpha. Beta.")
    monkeypatch.setattr(smart_indexing, "_split_into_parent_blocks", lambda text, parent_size: [text])
    monkeypatch.setattr(
        smart_indexing,
        "_refine_boundaries_with_llm",
        lambda initial_blocks, base_url, model, parent_size, timeout, progress_callback: (
            calls.setdefault(
                "boundary",
                {"base_url": base_url, "model": model, "blocks": initial_blocks},
            )
            and initial_blocks,
            [],
        ),
    )
    monkeypatch.setattr(
        smart_indexing,
        "_split_children_from_parent",
        lambda **kwargs: [
            {
                "id": "child-1",
                "text": "Alpha",
                "parent_id": kwargs["parent_id"],
                "metadata": {"source": kwargs["source"], "block_idx": 0, "child_idx": 0},
            }
        ],
    )

    def fake_get_embeddings(texts, model, base_url):
        calls["embeddings"] = {"texts": texts, "model": model, "base_url": base_url}
        return [[0.1, 0.2]]

    class FakeClient:
        def delete(self, **kwargs):
            pass

        def upsert(self, **kwargs):
            calls.setdefault("upserts", []).append(kwargs)

    monkeypatch.setattr(smart_indexing, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(smart_indexing, "get_qdrant_client", lambda url: FakeClient())
    monkeypatch.setattr(smart_indexing, "_ensure_parent_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(smart_indexing, "ensure_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        smart_indexing,
        "invalidate_bm25_cache",
        lambda qdrant_url, collection_name: calls.setdefault(
            "invalidated", (qdrant_url, collection_name)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        smart_indexing,
        "inspect_parent_child_integrity",
        lambda *args, **kwargs: {"ok": True},
    )

    smart_indexing.index_pdfs_smart(
        ["doc.pdf"],
        base_collection="docs",
        boundary_llm_model="qwen3:1.7b",
        embed_model="embeddinggemma:300m",
        qdrant_url="http://qdrant",
        ollama_base_url="http://192.168.1.151:11434",
    )

    assert calls["boundary"] == {
        "base_url": "http://192.168.1.151:11434",
        "model": "qwen3:1.7b",
        "blocks": ["Alpha. Beta."],
    }
    assert calls["embeddings"] == {
        "texts": ["Alpha"],
        "model": "embeddinggemma:300m",
        "base_url": "http://192.168.1.151:11434",
    }
    assert calls["upserts"]
    assert all(call.get("wait") is True for call in calls["upserts"])
    assert calls["invalidated"] == ("http://qdrant", "docs_children")


def test_inspect_parent_child_integrity_reports_both_directions() -> None:
    class FakeScrollClient:
        def scroll(self, **kwargs):
            if kwargs["collection_name"] == "docs_parents":
                return (
                    [
                        SimpleNamespace(id="p1", payload={"source": "a.pdf"}),
                        SimpleNamespace(id="p2", payload={"source": "a.pdf"}),
                    ],
                    None,
                )
            return (
                [
                    SimpleNamespace(
                        id="child-ok", payload={"parent_id": "p1", "source": "a.pdf"}
                    ),
                    SimpleNamespace(
                        id="child-orphan",
                        payload={"parent_id": "missing-parent", "source": "a.pdf"},
                    ),
                ],
                None,
            )

    report = qdrant_store.inspect_parent_child_integrity(
        FakeScrollClient(), base_collection="docs"
    )

    assert report["parents"] == 2
    assert report["children"] == 2
    assert report["parent_ids_without_children"] == ["p2"]
    assert report["orphan_child_ids"] == ["child-orphan"]
    assert report["source_sets_equal"] is True
    assert report["ok"] is False


def test_inspect_parent_child_integrity_filters_source_and_reports_clean() -> None:
    calls = []

    class FakeScrollClient:
        def scroll(self, **kwargs):
            calls.append(kwargs)
            source = kwargs["scroll_filter"].must[0].match.value
            assert source == "a.pdf"
            if kwargs["collection_name"] == "docs_parents":
                return ([SimpleNamespace(id="p1", payload={"source": source})], None)
            return (
                [SimpleNamespace(id="c1", payload={"parent_id": "p1", "source": source})],
                None,
            )

    report = qdrant_store.inspect_parent_child_integrity(
        FakeScrollClient(), base_collection="docs", source="a.pdf"
    )

    assert report == {
        "parents": 1,
        "children": 1,
        "orphan_child_ids": [],
        "parent_ids_without_children": [],
        "source_sets_equal": True,
        "ok": True,
    }
    assert len(calls) == 2


def test_smart_index_preparation_failure_does_not_delete_or_upsert(monkeypatch) -> None:
    class RecordingClient:
        def __init__(self):
            self.delete_calls = []
            self.upsert_calls = []

        def delete(self, **kwargs):
            self.delete_calls.append(kwargs)

        def upsert(self, **kwargs):
            self.upsert_calls.append(kwargs)

    client = RecordingClient()
    monkeypatch.setattr(smart_indexing, "_extract_full_pdf_text", lambda path: "Alpha beta")
    monkeypatch.setattr(smart_indexing, "_split_into_parent_blocks", lambda *args, **kwargs: ["Alpha beta"])
    monkeypatch.setattr(smart_indexing, "_ensure_parent_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(smart_indexing, "get_qdrant_client", lambda url: client)
    monkeypatch.setattr(
        smart_indexing,
        "get_embeddings",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embedding failed")),
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        smart_indexing.index_pdfs_smart(
            ["doc.pdf"],
            base_collection="docs",
            qdrant_url="http://qdrant",
            ollama_base_url="",
        )

    assert client.delete_calls == []
    assert client.upsert_calls == []


def test_qdrant_query_points_compat_prefers_query_points_response_points() -> None:
    class QueryPointsClient:
        def __init__(self) -> None:
            self.calls = []

        def query_points(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(points=["first", "second"])

    client = QueryPointsClient()

    results = qdrant_store._qdrant_query_points_compat(
        client=client,
        collection_name="docs",
        query_vector=[0.1, 0.2],
        limit=2,
        score_threshold=0.55,
        run_id="test",
        location="test",
    )

    assert results == ["first", "second"]
    assert client.calls == [
        {
            "collection_name": "docs",
            "query": [0.1, 0.2],
            "limit": 2,
            "score_threshold": 0.55,
        }
    ]


def test_qdrant_query_points_compat_passes_offset_when_provided() -> None:
    class QueryPointsClient:
        def __init__(self) -> None:
            self.calls = []

        def query_points(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(points=[])

    client = QueryPointsClient()

    qdrant_store._qdrant_query_points_compat(
        client=client,
        collection_name="docs_children",
        query_vector=[0.1, 0.2],
        limit=16,
        score_threshold=0.55,
        run_id="test",
        location="test",
        offset=16,
    )

    assert client.calls == [
        {
            "collection_name": "docs_children",
            "query": [0.1, 0.2],
            "limit": 16,
            "score_threshold": 0.55,
            "offset": 16,
        }
    ]


def test_qdrant_query_points_compat_falls_back_to_search() -> None:
    class SearchClient:
        def __init__(self) -> None:
            self.calls = []

        def search(self, **kwargs):
            self.calls.append(kwargs)
            return [{"payload": {"text": "fallback"}, "score": 0.9}]

    client = SearchClient()

    results = qdrant_store._qdrant_query_points_compat(
        client=client,
        collection_name="docs",
        query_vector=[0.3],
        limit=1,
        score_threshold=0.5,
        run_id="test",
        location="test",
    )

    assert results == [{"payload": {"text": "fallback"}, "score": 0.9}]
    assert client.calls == [
        {
            "collection_name": "docs",
            "query_vector": [0.3],
            "limit": 1,
            "score_threshold": 0.5,
        }
    ]


def test_select_unique_parent_hits_skips_duplicate_parents_until_k() -> None:
    candidates = [
        ({"parent_id": "p1", "text": "p1-a", "child_idx": 0}, 0.99),
        ({"parent_id": "p1", "text": "p1-b", "child_idx": 1}, 0.98),
        ({"parent_id": "p2", "text": "p2-a", "child_idx": 0}, 0.90),
        ({"parent_id": "p3", "text": "p3-a", "child_idx": 0}, 0.80),
    ]

    ids, info = qdrant_store._select_unique_parent_hits(candidates, k=3)

    assert ids == ["p1", "p2", "p3"]
    assert info["p1"]["child_text"] == "p1-a"
    assert info["p1"]["score"] == 0.99
    assert info["p1"]["child_idx"] == 0


def test_select_unique_parent_hits_returns_empty_for_non_positive_k() -> None:
    candidates = [({"parent_id": "p1", "text": "p1-a"}, 0.99)]

    assert qdrant_store._select_unique_parent_hits(candidates, k=0) == ([], {})
    assert qdrant_store._select_unique_parent_hits(candidates, k=-1) == ([], {})


def test_retrieve_chunks_smart_pages_until_k_unique_parents(monkeypatch) -> None:
    class ParentClient:
        def retrieve(self, **kwargs):
            assert kwargs["ids"] == ["p1", "p2", "p3"]
            return [
                SimpleNamespace(id="p1", payload={"text": "Parent one"}),
                SimpleNamespace(id="p2", payload={"text": "Parent two"}),
                SimpleNamespace(id="p3", payload={"text": "Parent three"}),
            ]

    first_page = [
        SimpleNamespace(
            payload={"text": f"p1 child {idx}", "parent_id": "p1", "child_idx": idx},
            score=1.0 - idx / 100,
        )
        for idx in range(15)
    ] + [
        SimpleNamespace(
            payload={"text": "p2 child", "parent_id": "p2", "child_idx": 0},
            score=0.80,
        )
    ]
    second_page = [
        SimpleNamespace(
            payload={"text": "p3 child", "parent_id": "p3", "child_idx": 0},
            score=0.70,
        )
    ]
    offsets = []

    def fake_query(**kwargs):
        offsets.append(kwargs.get("offset"))
        return first_page if kwargs.get("offset") == 0 else second_page

    monkeypatch.setattr(retrieval, "get_embeddings", lambda *args, **kwargs: [[0.1, 0.2]])
    monkeypatch.setattr(retrieval, "get_qdrant_client", lambda url: ParentClient())
    monkeypatch.setattr(retrieval, "_qdrant_query_points_compat", fake_query)

    chunks = retrieval.retrieve_chunks_smart(
        "soru",
        base_collection="docs",
        k=3,
        qdrant_url="http://qdrant",
        embed_model="model-a",
    )

    assert [chunk["parent_id"] for chunk in chunks] == ["p1", "p2", "p3"]
    assert chunks[0]["child_text"] == "p1 child 0"
    assert offsets == [0, 16]


def test_retrieve_chunks_smart_skips_embedding_and_query_for_non_positive_k(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("retrieval dependency must not be called")

    monkeypatch.setattr(retrieval, "get_embeddings", fail_if_called)
    monkeypatch.setattr(retrieval, "get_qdrant_client", fail_if_called)

    assert retrieval.retrieve_chunks_smart("soru", base_collection="docs", k=0) == []


def test_retrieve_chunks_deduplicates_results_and_preserves_scores(monkeypatch) -> None:
    calls = {}
    fake_client = object()

    def fake_get_embeddings(texts, model, timeout):
        calls["embeddings"] = {"texts": texts, "model": model, "timeout": timeout}
        return [[0.1, 0.2, 0.3]]

    def fake_get_qdrant_client(url):
        calls["client_url"] = url
        return fake_client

    def fake_query(**kwargs):
        calls["query"] = kwargs
        return [
            {"payload": {"text": "Alpha beta"}, "score": 0.91},
            SimpleNamespace(payload={"text": " Alpha\nbeta "}, score=0.89),
            SimpleNamespace(payload={"text": "Gamma"}, score=0.77),
            {"payload": {"text": "   "}, "score": 0.1},
        ]

    monkeypatch.setattr(retrieval, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(retrieval, "get_qdrant_client", fake_get_qdrant_client)
    monkeypatch.setattr(retrieval, "_qdrant_query_points_compat", fake_query)

    chunks = retrieval.retrieve_chunks(
        "soru",
        collection_name="docs",
        k=4,
        qdrant_url="http://qdrant",
        score_threshold=0.6,
        embed_model="model-a",
    )

    assert chunks == [
        {"text": "Alpha beta", "score": 0.91},
        {"text": "Gamma", "score": 0.77},
    ]
    assert calls["embeddings"] == {
        "texts": ["soru"],
        "model": "model-a",
        "timeout": 300,
    }
    assert calls["client_url"] == "http://qdrant"
    assert calls["query"] == {
        "client": fake_client,
        "collection_name": "docs",
        "query_vector": [0.1, 0.2, 0.3],
        "limit": 4,
        "score_threshold": 0.6,
        "run_id": calls["query"]["run_id"],
        "location": "rag/retrieval.py:retrieve_chunks:query_dispatch",
    }


def test_retrieve_chunks_preserves_payload_metadata(monkeypatch) -> None:
    fake_client = object()

    monkeypatch.setattr(
        retrieval, "get_embeddings", lambda texts, model, timeout: [[0.1, 0.2]]
    )
    monkeypatch.setattr(retrieval, "get_qdrant_client", lambda url: fake_client)
    monkeypatch.setattr(
        retrieval,
        "_qdrant_query_points_compat",
        lambda **kwargs: [
            {
                "payload": {
                    "text": "Alpha beta",
                    "source": "/docs/manual.pdf",
                    "page": 4,
                    "chunk": 2,
                },
                "score": 0.92,
            }
        ],
    )

    chunks = retrieval.retrieve_chunks(
        "soru",
        collection_name="docs",
        qdrant_url="http://qdrant",
        embed_model="model-a",
    )

    assert chunks == [
        {
            "text": "Alpha beta",
            "score": 0.92,
            "source": "/docs/manual.pdf",
            "page": 4,
            "chunk": 2,
        }
    ]


def test_retrieve_parent_docs_preserves_hit_order_and_deduplicates_text() -> None:
    class ParentClient:
        def retrieve(self, **kwargs):
            assert kwargs == {
                "collection_name": "docs_parents",
                "ids": ["p2", "p1", "p3"],
                "with_payload": True,
            }
            return [
                SimpleNamespace(id="p1", payload={"text": "Same text"}),
                SimpleNamespace(id="p2", payload={"text": "First text"}),
                {"id": "p3", "payload": {"text": " Same   text "}},
            ]

    docs = qdrant_store._retrieve_parent_docs(
        client=ParentClient(),
        collection_name="docs_parents",
        parent_ids_ordered=["p2", "p1", "p3"],
        child_info={
            "p2": {"child_text": "child two", "score": 0.8},
            "p1": {"child_text": "child one", "score": 0.7},
            "p3": {"child_text": "child three", "score": 0.6},
        },
    )

    assert docs == [
        {"text": "First text", "child_text": "child two", "score": 0.8, "parent_id": "p2"},
        {"text": "Same text", "child_text": "child one", "score": 0.7, "parent_id": "p1"},
    ]


def test_retrieve_parent_docs_preserves_parent_and_child_metadata() -> None:
    class ParentClient:
        def retrieve(self, **kwargs):
            return [
                SimpleNamespace(
                    id="p1",
                    payload={
                        "text": "Parent text",
                        "source": "/docs/manual.pdf",
                        "block_idx": 9,
                    },
                )
            ]

    docs = qdrant_store._retrieve_parent_docs(
        client=ParentClient(),
        collection_name="docs_parents",
        parent_ids_ordered=["p1"],
        child_info={
            "p1": {
                "child_text": "child hit",
                "score": 0.7,
                "child_idx": 3,
            }
        },
    )

    assert docs == [
        {
            "text": "Parent text",
            "child_text": "child hit",
            "score": 0.7,
            "parent_id": "p1",
            "source": "/docs/manual.pdf",
            "block_idx": 9,
            "child_idx": 3,
        }
    ]


def test_bm25_retrieve_chunks_excludes_non_positive_scores(monkeypatch) -> None:
    class FakeBM25:
        def __init__(self, corpus):
            self.corpus = corpus

        def get_scores(self, query):
            return [1.5, 0.0, -0.2]

    monkeypatch.setattr(bm25, "_BM25_AVAILABLE", True)
    monkeypatch.setattr(bm25, "_BM25Okapi", FakeBM25)
    monkeypatch.setattr(bm25, "get_qdrant_client", lambda url: object())
    monkeypatch.setattr(
        bm25,
        "_scroll_all_payloads",
        lambda client, collection_name: [
            {
                "text": "Alpha beta",
                "source": "/docs/manual.pdf",
                "page": 4,
                "chunk": 2,
            },
            {
                "text": "Gamma delta",
                "source": "/docs/other.pdf",
                "page": 5,
                "chunk": 1,
            },
            {
                "text": "Epsilon zeta",
                "source": "/docs/third.pdf",
                "page": 6,
                "chunk": 0,
            },
        ],
    )

    chunks = bm25.retrieve_chunks_bm25(
        "alpha",
        collection_name="docs",
        k=3,
        qdrant_url="http://qdrant",
    )

    assert chunks == [
        {
            "text": "Alpha beta",
            "score": 1.5,
            "bm25_score": 1.5,
            "bm25_zero_match": False,
            "source": "/docs/manual.pdf",
            "page": 4,
            "chunk": 2,
        },
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("UÇAK, eğitim!", ["uçak", "eğitim"]),
        ("T-129 / ATAK", ["t", "129", "atak"]),
        ("  çok\tboşluk ", ["çok", "boşluk"]),
    ],
)
def test_bm25_tokenize_normalizes_unicode_words(text, expected) -> None:
    assert bm25._bm25_tokenize(text) == expected


def test_bm25_reuses_index_for_same_collection(monkeypatch) -> None:
    calls = {"scroll": 0, "build": 0}

    class FakeBM25:
        def __init__(self, corpus):
            calls["build"] += 1

        def get_scores(self, query):
            return [1.0]

    def fake_scroll(client, collection_name):
        calls["scroll"] += 1
        return [{"text": "Alpha", "source": "doc.pdf"}]

    bm25._BM25_CACHE.clear()
    monkeypatch.setattr(bm25, "_BM25_AVAILABLE", True)
    monkeypatch.setattr(bm25, "_BM25Okapi", FakeBM25)
    monkeypatch.setattr(bm25, "get_qdrant_client", lambda url: object())
    monkeypatch.setattr(bm25, "_scroll_all_payloads", fake_scroll)

    bm25.retrieve_chunks_bm25("alpha", collection_name="docs", qdrant_url="http://qdrant")
    bm25.retrieve_chunks_bm25("alpha", collection_name="docs", qdrant_url="http://qdrant")

    assert calls == {"scroll": 1, "build": 1}


def test_bm25_cache_isolated_by_collection_and_explicitly_invalidated(monkeypatch) -> None:
    calls = {"scroll": 0, "build": 0}

    class FakeBM25:
        def __init__(self, corpus):
            calls["build"] += 1

        def get_scores(self, query):
            return [1.0]

    def fake_scroll(client, collection_name):
        calls["scroll"] += 1
        return [{"text": collection_name, "source": "doc.pdf"}]

    bm25._BM25_CACHE.clear()
    monkeypatch.setattr(bm25, "_BM25_AVAILABLE", True)
    monkeypatch.setattr(bm25, "_BM25Okapi", FakeBM25)
    monkeypatch.setattr(bm25, "get_qdrant_client", lambda url: object())
    monkeypatch.setattr(bm25, "_scroll_all_payloads", fake_scroll)

    bm25.retrieve_chunks_bm25("query", collection_name="docs-a", qdrant_url="http://qdrant")
    bm25.retrieve_chunks_bm25("query", collection_name="docs-b", qdrant_url="http://qdrant")
    bm25.invalidate_bm25_cache("http://qdrant", "docs-a")
    bm25.retrieve_chunks_bm25("query", collection_name="docs-a", qdrant_url="http://qdrant")

    assert calls == {"scroll": 3, "build": 3}


def test_bm25_retrieve_chunks_returns_empty_when_all_scores_are_zero(monkeypatch) -> None:
    class FakeBM25:
        def __init__(self, corpus):
            self.corpus = corpus

        def get_scores(self, query):
            return [0.0, 0.0]

    monkeypatch.setattr(bm25, "_BM25_AVAILABLE", True)
    monkeypatch.setattr(bm25, "_BM25Okapi", FakeBM25)
    monkeypatch.setattr(bm25, "get_qdrant_client", lambda url: object())
    monkeypatch.setattr(
        bm25,
        "_scroll_all_payloads",
        lambda client, collection_name: [
            {"text": "Alpha beta", "source": "/docs/manual.pdf"},
            {"text": "Gamma delta", "source": "/docs/other.pdf"},
        ],
    )

    chunks = bm25.retrieve_chunks_bm25(
        "missing",
        collection_name="docs",
        k=2,
        qdrant_url="http://qdrant",
    )

    assert chunks == []


def test_bm25_smart_returns_empty_when_all_child_scores_are_zero(monkeypatch) -> None:
    class FakeBM25:
        def __init__(self, corpus):
            self.corpus = corpus

        def get_scores(self, query):
            return [0.0, 0.0]

    class FakeClient:
        def retrieve(self, **kwargs):
            return [
                SimpleNamespace(id="p1", payload={"text": "Parent one"}),
                SimpleNamespace(id="p2", payload={"text": "Parent two"}),
            ]

    monkeypatch.setattr(bm25, "_BM25_AVAILABLE", True)
    monkeypatch.setattr(bm25, "_BM25Okapi", FakeBM25)
    monkeypatch.setattr(bm25, "get_qdrant_client", lambda url: FakeClient())
    monkeypatch.setattr(
        bm25,
        "_scroll_all_payloads",
        lambda client, collection_name: [
            {"text": "Alpha beta", "parent_id": "p1"},
            {"text": "Gamma delta", "parent_id": "p2"},
        ],
    )

    chunks = bm25.retrieve_chunks_bm25_smart(
        "missing",
        base_collection="docs",
        k=2,
        qdrant_url="http://qdrant",
    )

    assert chunks == []


def test_bm25_smart_scans_past_duplicate_children_to_fill_unique_parent_k(monkeypatch) -> None:
    class FakeBM25:
        def __init__(self, corpus):
            self.corpus = corpus

        def get_scores(self, query):
            return [10.0, 9.0, 8.0, 7.0]

    class FakeClient:
        def retrieve(self, **kwargs):
            assert kwargs["ids"] == ["p1", "p2", "p3"]
            return [
                SimpleNamespace(id="p1", payload={"text": "Parent one"}),
                SimpleNamespace(id="p2", payload={"text": "Parent two"}),
                SimpleNamespace(id="p3", payload={"text": "Parent three"}),
            ]

    monkeypatch.setattr(bm25, "_BM25_AVAILABLE", True)
    monkeypatch.setattr(bm25, "_BM25Okapi", FakeBM25)
    monkeypatch.setattr(bm25, "get_qdrant_client", lambda url: FakeClient())
    monkeypatch.setattr(
        bm25,
        "_scroll_all_payloads",
        lambda client, collection_name: [
            {"text": "p1 first", "parent_id": "p1", "child_idx": 0},
            {"text": "p1 second", "parent_id": "p1", "child_idx": 1},
            {"text": "p2 first", "parent_id": "p2", "child_idx": 0},
            {"text": "p3 first", "parent_id": "p3", "child_idx": 0},
        ],
    )

    chunks = bm25.retrieve_chunks_bm25_smart(
        "query",
        base_collection="docs",
        k=3,
        qdrant_url="http://qdrant",
    )

    assert [chunk["parent_id"] for chunk in chunks] == ["p1", "p2", "p3"]
    assert chunks[0]["child_text"] == "p1 first"
    assert chunks[0]["bm25_score"] == 10.0


def test_rag_variant_and_context_status_helpers() -> None:
    from pipeline import build_context_status, build_rag_variant

    assert build_rag_variant("vector", False) == "classic_vector"
    assert build_rag_variant("vector", True) == "smart_vector"
    assert build_rag_variant("bm25", False) == "classic_bm25"
    assert build_rag_variant("bm25", True) == "smart_bm25"
    assert build_rag_variant("vector", False, rag_mode="no_rag") == "no_rag"
    assert build_context_status("rag", []) == "empty"
    assert build_context_status("rag", [{"text": "ctx"}]) == "retrieved"
    assert build_context_status("no_rag", []) == "not_applicable"


def test_run_full_pipeline_marks_empty_rag_context_and_variant(monkeypatch) -> None:
    import pipeline

    monkeypatch.setattr(
        pipeline,
        "load_questions",
        lambda *args, **kwargs: [
            {"question_index": 0, "question": "Soru?", "observation_idea": "Cevap"}
        ],
    )
    monkeypatch.setattr(pipeline, "warmup_model", lambda model, timeout: None)
    monkeypatch.setattr(pipeline, "retrieve_chunks", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "generate_rag_answer_ollama",
        lambda **kwargs: {"answer": "fallback", "response_time_seconds": 0.1},
    )

    rows = pipeline.run_full_pipeline(
        csv_path="input.csv",
        collection_name="docs",
        qdrant_url="http://qdrant",
        k=3,
        qa_model="qwen3:1.7b",
        rag_mode="rag",
        eval_enabled=False,
        retrieval_mode="vector",
        smart_chunking=False,
        embed_model="embeddinggemma:300m",
        score_threshold=0.4,
    )

    assert rows[0]["rag_type"] == "RAG'li"
    assert rows[0]["rag_variant"] == "classic_vector"
    assert rows[0]["retrieval_mode"] == "vector"
    assert rows[0]["smart_rag"] is False
    assert rows[0]["collection_name"] == "docs"
    assert rows[0]["embed_model"] == "embeddinggemma:300m"
    assert rows[0]["k"] == 3
    assert rows[0]["score_threshold"] == 0.4
    assert rows[0]["rag_context_status"] == "empty"
    assert rows[0]["retrieved_chunk_size"] == 0


def test_run_full_pipeline_marks_empty_bm25_context(monkeypatch) -> None:
    import pipeline

    monkeypatch.setattr(
        pipeline,
        "load_questions",
        lambda *args, **kwargs: [
            {"question_index": 0, "question": "Eşleşmeyen soru?", "observation_idea": "Cevap"}
        ],
    )
    monkeypatch.setattr(pipeline, "warmup_model", lambda model, timeout: None)
    monkeypatch.setattr(pipeline, "retrieve_chunks_bm25", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "generate_rag_answer_ollama",
        lambda **kwargs: {"answer": "fallback", "response_time_seconds": 0.1},
    )

    rows = pipeline.run_full_pipeline(
        csv_path="input.csv",
        collection_name="docs",
        qdrant_url="http://qdrant",
        k=3,
        qa_model="qwen3:1.7b",
        rag_mode="rag",
        eval_enabled=False,
        retrieval_mode="bm25",
        smart_chunking=False,
        embed_model="embeddinggemma:300m",
    )

    assert rows[0]["rag_variant"] == "classic_bm25"
    assert rows[0]["rag_context_status"] == "empty"
    assert rows[0]["retrieved_chunk_size"] == 0
    assert rows[0]["retrieved_chunks"] == "[]"
