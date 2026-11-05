#!/usr/bin/env python3
"""Benchmark a resident llama.cpp model and evaluate answers separately."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompts import build_no_rag_prompt

JsonDict = dict[str, Any]
PostJson = Callable[[str, JsonDict, float], JsonDict]

RAW_FIELDNAMES = (
    "run_label",
    "model",
    "question_index",
    "question",
    "reference_answer",
    "answerability",
    "model_answer",
    "reasoning_content",
    "finish_reason",
    "wall_time_seconds",
    "prompt_tokens",
    "cached_tokens",
    "processed_prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_ms",
    "prompt_tokens_per_second",
    "generation_ms",
    "generation_tokens_per_second",
    "inference_status",
    "attempt_count",
    "error_type",
    "error_message",
)
EVALUATION_FIELDNAMES = (
    "correctness_score",
    "completeness_score",
    "hallucination_risk",
    "verdict",
    "evaluation_reason",
    "evaluation_status",
    "evaluation_duration_seconds",
    "evaluation_error_type",
    "evaluation_error_message",
    "evaluation_attempt_count",
)


def build_inference_payload(
    model: str,
    question: str,
    *,
    max_tokens: int,
    temperature: float,
) -> JsonDict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": build_no_rag_prompt(question)}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _number(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def _integer(mapping: dict[str, Any], key: str, default: int = 0) -> int:
    value = mapping.get(key, default)
    if value in (None, ""):
        return default
    return int(value)


def parse_inference_response(payload: JsonDict, *, wall_time_seconds: float) -> JsonDict:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("llama.cpp response has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("llama.cpp choice is not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("llama.cpp choice has no message")
    answer = str(message.get("content") or "").strip()
    if not answer:
        raise ValueError("empty model answer")

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    prompt_tokens = _integer(usage, "prompt_tokens")
    cached_tokens = _integer(prompt_details, "cached_tokens", _integer(timings, "cache_n"))
    processed_prompt_tokens = _integer(timings, "prompt_n", max(prompt_tokens - cached_tokens, 0))
    completion_tokens = _integer(usage, "completion_tokens", _integer(timings, "predicted_n"))

    return {
        "model_answer": answer,
        "reasoning_content": str(message.get("reasoning_content") or "").strip(),
        "finish_reason": str(choice.get("finish_reason") or ""),
        "wall_time_seconds": round(float(wall_time_seconds), 6),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "processed_prompt_tokens": processed_prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": _integer(usage, "total_tokens", prompt_tokens + completion_tokens),
        "prompt_ms": _number(timings, "prompt_ms"),
        "prompt_tokens_per_second": _number(timings, "prompt_per_second"),
        "generation_ms": _number(timings, "predicted_ms"),
        "generation_tokens_per_second": _number(timings, "predicted_per_second"),
    }


def post_json(url: str, payload: JsonDict, timeout: float) -> JsonDict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("HTTP response is not a JSON object")
    return parsed


def run_inference_row(
    *,
    question_index: int,
    question: str,
    reference_answer: str,
    answerability: str = "evet",
    run_label: str,
    model: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    attempt_count: int,
    post_fn: PostJson = post_json,
) -> JsonDict:
    base = {
        "run_label": run_label,
        "model": model,
        "question_index": question_index,
        "question": question,
        "reference_answer": reference_answer,
        "answerability": answerability,
        "attempt_count": attempt_count,
    }
    request_payload = build_inference_payload(
        model,
        question,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    started = time.perf_counter()
    try:
        response = post_fn(
            f"{base_url.rstrip('/')}/chat/completions",
            request_payload,
            timeout,
        )
        parsed = parse_inference_response(
            response,
            wall_time_seconds=time.perf_counter() - started,
        )
        return {
            **base,
            **parsed,
            "inference_status": "success",
            "error_type": "",
            "error_message": "",
        }
    except Exception as exc:
        return {
            **base,
            "model_answer": "",
            "reasoning_content": "",
            "finish_reason": "",
            "wall_time_seconds": round(time.perf_counter() - started, 6),
            "inference_status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8-sig",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def successful_indices(
    rows: Iterable[Mapping[str, object]],
    status_field: str,
) -> set[int]:
    result: set[int] = set()
    for row in rows:
        if str(row.get(status_field, "")) != "success":
            continue
        try:
            result.add(int(str(row.get("question_index", ""))))
        except ValueError:
            continue
    return result


def nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    index = max(math.ceil(percentile * len(ordered)) - 1, 0)
    return ordered[index]


def _as_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_summary(rows: Sequence[Mapping[str, object]], field: str) -> JsonDict:
    values = [number for row in rows if (number := _as_float(row.get(field))) is not None]
    if not values:
        return {"mean": None, "median": None, "p50": None, "p95": None, "p99": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p50": nearest_rank(values, 0.50),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
    }


def build_summary(
    rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> JsonDict:
    successful = [row for row in rows if str(row.get("inference_status", "")) == "success"]
    prompt_total = sum(int(_as_float(row.get("prompt_tokens")) or 0) for row in successful)
    cached_total = sum(int(_as_float(row.get("cached_tokens")) or 0) for row in successful)
    completion_total = sum(int(_as_float(row.get("completion_tokens")) or 0) for row in successful)
    measured_wall_time_total = sum(
        _as_float(row.get("wall_time_seconds")) or 0.0 for row in successful
    )
    inference_summary = {
        field: _metric_summary(successful, field)
        for field in (
            "wall_time_seconds",
            "prompt_ms",
            "generation_ms",
            "prompt_tokens_per_second",
            "generation_tokens_per_second",
        )
    }
    inference_summary["measured_wall_time_seconds_total"] = measured_wall_time_total
    return {
        "config": dict(config),
        "counts": {
            "total": len(rows),
            "successful": len(successful),
            "failed": len(rows) - len(successful),
        },
        "inference": inference_summary,
        "cache": {
            "cached_tokens_total": cached_total,
            "prompt_tokens_total": prompt_total,
            "cached_token_ratio": (cached_total / prompt_total) if prompt_total else 0.0,
        },
        "tokens": {"completion_total": completion_total},
    }


EVAL_SYSTEM_PROMPT = """Sen sıkı ve tutarlı bir Türkçe soru-cevap değerlendiricisisin.
Model cevabını yalnızca verilen soru ve referans cevaba göre değerlendir.
Gereksiz uzunluğu ödüllendirme. Referansla çelişen veya desteksiz bilgiyi halüsinasyon say.
Cevaplanabilir hayir ise soru cevaplanamaz; yalnızca verilen belgelerde bilginin bulunmadığını açıkça söyleyen
ve yeni bir olgusal bilgi uydurmayan yanıt pass olabilir. Belirsiz bir "bilmiyorum" yanıtını
partial say; desteksiz bir doz vermeyi fail say ve hallucination_risk high ver.
Yalnızca şu alanları içeren bir JSON nesnesi döndür:
correctness_score (0-10), completeness_score (0-10),
hallucination_risk (low|medium|high), verdict (pass|partial|fail), reason (kısa açıklama).
"""


def build_eval_messages(
    question: str,
    reference_answer: str,
    model_answer: str,
    answerability: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": (
                f"Soru: {question}\n"
                f"Referans cevap: {reference_answer}\n"
                f"Cevaplanabilir: {answerability}\n"
                f"Model cevabı: {model_answer}"
            ),
        },
    ]


def parse_evaluation(content: str) -> JsonDict:
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("evaluation output must be a JSON object")

    result: JsonDict = {}
    for field in ("correctness_score", "completeness_score"):
        value = _as_float(parsed.get(field))
        if value is None or not 0 <= value <= 10:
            raise ValueError(f"{field} must be a number in [0, 10]")
        result[field] = value

    risk = str(parsed.get("hallucination_risk", "")).strip().lower()
    if risk not in {"low", "medium", "high"}:
        raise ValueError("hallucination_risk must be low, medium, or high")
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in {"pass", "partial", "fail"}:
        raise ValueError("verdict must be pass, partial, or fail")
    reason = " ".join(str(parsed.get("reason", "")).split())
    if not reason:
        raise ValueError("reason must be non-empty")

    result.update(
        hallucination_risk=risk,
        verdict=verdict,
        evaluation_reason=reason,
    )
    return result


def _is_retryable(exc: Exception) -> bool:
    """Transient-error test that keeps this script free of provider imports."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    name = type(exc).__name__
    return any(token in name for token in ("Timeout", "Connection", "RateLimit", "InternalServer"))


def evaluate_row(
    row: Mapping[str, object],
    *,
    client: object,
    eval_model: str,
    attempts: int = 2,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    result = dict(row)
    if str(row.get("inference_status", "")) != "success" or not str(
        row.get("model_answer", "")
    ).strip():
        result.update(
            evaluation_status="skipped",
            evaluation_duration_seconds=0.0,
            evaluation_error_type="InferenceUnavailable",
            evaluation_error_message="Successful non-empty inference is required.",
            evaluation_attempt_count=0,
        )
        return result

    last_exc: Exception | None = None
    attempt = 0
    started = time.perf_counter()
    for attempt in range(1, max(attempts, 1) + 1):
        # Restarted per attempt so the recorded duration is judge latency, not retry backoff.
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=eval_model,
                messages=build_eval_messages(
                    str(row.get("question", "")),
                    str(row.get("reference_answer", "")),
                    str(row.get("model_answer", "")),
                    str(row.get("answerability", "evet")),
                ),
                response_format={"type": "json_object"},
            )
            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError("evaluator response has no choices")
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if not content:
                raise ValueError("evaluator response has empty content")
            parsed = parse_evaluation(str(content))
            result.update(
                parsed,
                evaluation_status="success",
                evaluation_duration_seconds=round(time.perf_counter() - started, 6),
                evaluation_error_type="",
                evaluation_error_message="",
                evaluation_attempt_count=attempt,
            )
            return result
        except Exception as exc:
            last_exc = exc
            if attempt >= max(attempts, 1) or not _is_retryable(exc):
                break
            sleep_fn(2 ** (attempt - 1))

    result.update(
        evaluation_status="failed",
        evaluation_duration_seconds=round(time.perf_counter() - started, 6),
        evaluation_error_type=type(last_exc).__name__ if last_exc else "UnknownError",
        evaluation_error_message=str(last_exc or "unknown error"),
        evaluation_attempt_count=attempt,
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "data/qa/old/question_answers_100.csv"),
        help="Question/reference-answer CSV path.",
    )
    parser.add_argument("--base-url", default="http://192.168.1.29:8084/v1")
    parser.add_argument("--model", default="qwen3.6:35b-a3b")
    parser.add_argument("--eval-model", default="gpt-5.4-mini")
    parser.add_argument(
        "--run-label",
        default="qwen3.6-35b-a3b_q4km_kv-q8_prompt-cache-on",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--question-col", default="question")
    parser.add_argument("--answer-col", default="answer")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args(argv)


def load_questions(
    path: Path,
    question_col: str,
    answer_col: str,
) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = [name for name in (question_col, answer_col) if name not in fields]
        if missing:
            raise ValueError(
                f"CSV must contain configured question and answer columns; missing: {', '.join(missing)}"
            )
        result = []
        for index, row in enumerate(reader):
            question = str(row.get(question_col, "")).strip()
            answer = str(row.get(answer_col, "")).strip()
            if not question:
                raise ValueError(f"CSV row {index} has an empty question")
            result.append(
                {
                    "question_index": index,
                    "question": question,
                    "reference_answer": answer,
                    "answerability": str(row.get("cevaplanabilir", "evet")).strip().lower() or "evet",
                }
            )
    if not result:
        raise ValueError("CSV contains no question rows")
    return result


def _health_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return f"{normalized}/health"


def check_health(base_url: str, timeout: float) -> None:
    request = urllib.request.Request(_health_url(base_url), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"llama.cpp health returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"llama.cpp health returned HTTP {exc.code}") from exc


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _row_key(row: Mapping[str, object]) -> int:
    """Checkpoint identity. This script emits one row per question, and the resume paths
    (`raw_by_index`, `successful_indices`) key on question_index too."""
    return int(str(row.get("question_index", -1)))


def _upsert_row(rows: list[JsonDict], replacement: JsonDict) -> None:
    key = _row_key(replacement)
    for position, row in enumerate(rows):
        if _row_key(row) == key:
            rows[position] = replacement
            return
    rows.append(replacement)
    rows.sort(key=_row_key)


def _evaluation_summary(rows: Sequence[Mapping[str, object]]) -> JsonDict:
    successful = [row for row in rows if str(row.get("evaluation_status", "")) == "success"]
    summary: JsonDict = {
        "successful": len(successful),
        "failed": sum(str(row.get("evaluation_status", "")) == "failed" for row in rows),
        "skipped": sum(str(row.get("evaluation_status", "")) == "skipped" for row in rows),
        "correctness_score_mean": _metric_summary(successful, "correctness_score")["mean"],
        "completeness_score_mean": _metric_summary(successful, "completeness_score")["mean"],
    }
    answerable = [row for row in successful if str(row.get("answerability", "evet")) != "hayir"]
    unanswerable = [row for row in successful if str(row.get("answerability", "evet")) == "hayir"]
    summary["answerable"] = {
        "successful": len(answerable),
        "correctness_score_mean": _metric_summary(answerable, "correctness_score")["mean"],
        "completeness_score_mean": _metric_summary(answerable, "completeness_score")["mean"],
    }
    summary["unanswerable"] = {
        "successful": len(unanswerable),
        "grounded_refusal_accuracy": (
            sum(str(row.get("verdict", "")) == "pass" for row in unanswerable) / len(unanswerable)
            if unanswerable else None
        ),
        "hallucination_rate": (
            sum(str(row.get("hallucination_risk", "")) == "high" for row in unanswerable) / len(unanswerable)
            if unanswerable else None
        ),
    }
    return summary


def _default_openai_client_factory() -> object:
    from openai import OpenAI

    return OpenAI()


def execute_run(
    args: argparse.Namespace,
    *,
    post_fn: PostJson = post_json,
    health_fn: Callable[[str, float], None] = check_health,
    openai_client_factory: Callable[[], object] = _default_openai_client_factory,
) -> int:
    if args.max_tokens <= 0 or args.timeout <= 0:
        raise ValueError("--max-tokens and --timeout must be positive")
    if not args.skip_eval and not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required unless --skip-eval is used")

    questions = load_questions(Path(args.csv), args.question_col, args.answer_col)
    health_fn(args.base_url, args.timeout)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / "data/results" / f"llamacpp_inference_{timestamp}"
    )
    raw_path = out_dir / "raw_results.csv"
    evaluated_path = out_dir / "evaluated_results.csv"
    summary_path = out_dir / "summary.json"
    log_path = out_dir / "run.log"
    if out_dir.exists() and not args.resume and any(
        path.exists() for path in (raw_path, evaluated_path, summary_path)
    ):
        raise FileExistsError(f"Output already contains a benchmark run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    config = {
        "run_label": args.run_label,
        "model": args.model,
        "base_url": args.base_url,
        "csv": str(Path(args.csv)),
        "eval_model": args.eval_model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "prompt_cache": "server-enabled; measured via cached_tokens",
        "model_load_included": False,
        "warmup_included": False,
    }
    log(f"Starting {args.run_label} with {len(questions)} questions")

    if not args.no_warmup:
        warmup_payload = build_inference_payload(
            args.model,
            "Kısa bir yanıt ver: hazır mısın?",
            max_tokens=min(args.max_tokens, 32),
            temperature=args.temperature,
        )
        post_fn(
            f"{args.base_url.rstrip('/')}/chat/completions",
            warmup_payload,
            args.timeout,
        )
        log("Warmup completed and excluded from measurements")

    raw_rows: list[JsonDict] = [dict(row) for row in read_csv_rows(raw_path)] if args.resume else []
    completed = successful_indices(raw_rows, "inference_status")
    for question in questions:
        question_index = int(question["question_index"])
        if question_index in completed:
            continue
        previous = next(
            (row for row in raw_rows if int(str(row.get("question_index", -1))) == question_index),
            None,
        )
        attempt_count = int(str(previous.get("attempt_count", 0))) + 1 if previous else 1
        result = run_inference_row(
            question_index=question_index,
            question=str(question["question"]),
            reference_answer=str(question["reference_answer"]),
            answerability=str(question["answerability"]),
            run_label=args.run_label,
            model=args.model,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            attempt_count=attempt_count,
            post_fn=post_fn,
        )
        _upsert_row(raw_rows, result)
        write_csv_atomic(raw_path, raw_rows, RAW_FIELDNAMES)
        log(f"Inference {question_index + 1}/{len(questions)}: {result['inference_status']}")

    if args.skip_eval:
        evaluated_rows = [
            {
                **row,
                "evaluation_status": "skipped",
                "evaluation_duration_seconds": 0.0,
                "evaluation_error_type": "EvaluationDisabled",
                "evaluation_error_message": "--skip-eval was used",
                "evaluation_attempt_count": 0,
            }
            for row in raw_rows
        ]
    else:
        client = openai_client_factory()
        evaluated_rows = [dict(row) for row in read_csv_rows(evaluated_path)] if args.resume else []
        evaluated = successful_indices(evaluated_rows, "evaluation_status")
        raw_by_index = {int(str(row["question_index"])): row for row in raw_rows}
        for question_index in sorted(raw_by_index):
            if question_index in evaluated:
                continue
            result = evaluate_row(
                raw_by_index[question_index],
                client=client,
                eval_model=args.eval_model,
            )
            _upsert_row(evaluated_rows, result)
            write_csv_atomic(
                evaluated_path,
                evaluated_rows,
                (*RAW_FIELDNAMES, *EVALUATION_FIELDNAMES),
            )
            log(
                f"Evaluation {question_index + 1}/{len(questions)}: "
                f"{result['evaluation_status']}"
            )

    if args.skip_eval:
        write_csv_atomic(
            evaluated_path,
            evaluated_rows,
            (*RAW_FIELDNAMES, *EVALUATION_FIELDNAMES),
        )
    summary = build_summary(raw_rows, config)
    summary["evaluation"] = _evaluation_summary(evaluated_rows)
    _write_json_atomic(summary_path, summary)

    evaluation = summary["evaluation"]
    if not summary["counts"]["successful"]:
        log("FAILED no successful inference rows")
        return 2
    if not args.skip_eval:
        if evaluation["failed"]:
            log(f"WARNING {evaluation['failed']} evaluation(s) failed")
        if not evaluation["successful"]:
            log("FAILED every evaluation failed; summary quality metrics are empty")
            return 3
    log("Run completed")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return execute_run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
