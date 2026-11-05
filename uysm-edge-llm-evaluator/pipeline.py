from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, TextIO, Union

import requests
import openai
from openai import OpenAI

from prompts import (
    DEFAULT_EVAL_SYSTEM_PROMPT,
    build_no_rag_prompt,
    build_rag_prompt,
)
from rag.bm25 import retrieve_chunks_bm25, retrieve_chunks_bm25_smart
from rag.config import DEFAULT_COLLECTION_NAME, QDRANT_URL
from rag.retrieval import (
    retrieve_chunks,
    retrieve_chunks_smart,
)


QA_OLLAMA_MODEL = os.getenv("QA_OLLAMA_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_EVAL_MODEL_NAME = os.getenv("DEEPSEEK_EVAL_MODEL_NAME")
OPENAI_EVAL_MODEL_NAME = os.getenv("OPENAI_EVAL_MODEL_NAME")
_CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp1254", "cp1252", "latin-1"]
_DEEPSEEK_API_BASE_URL = "https://api.deepseek.com"


class OllamaServiceUnavailable(RuntimeError):
    """Raised when a transient question failure is followed by a failed Ollama health check."""


def is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        return response is not None and response.status_code >= 500
    return False


def check_ollama_health(base_url: str = OLLAMA_BASE_URL, timeout: int = 10) -> bool:
    try:
        api_base = _require_ollama_base_url(base_url)
        response = requests.get(f"{api_base}/api/tags", timeout=timeout)
        response.raise_for_status()
        return True
    except Exception:
        return False


def build_rag_variant(retrieval_mode: str, smart_rag: bool, rag_mode: str = "rag") -> str:
    if rag_mode == "no_rag":
        return "no_rag"
    mode = (retrieval_mode or "vector").lower()
    prefix = "smart" if smart_rag else "classic"
    if mode == "bm25":
        return f"{prefix}_bm25"
    return f"{prefix}_vector"


def build_context_status(rag_mode: str, chunks: list[dict]) -> str:
    if rag_mode == "no_rag":
        return "not_applicable"
    return "retrieved" if chunks else "empty"


def _chunk_source_label(chunk: dict) -> str:
    src = os.path.basename(str(chunk.get("source") or "")) or str(chunk.get("source") or "")
    page = chunk.get("page", "")
    block_idx = chunk.get("block_idx", "")
    chunk_idx = chunk.get("chunk", chunk.get("child_idx", ""))
    if not src:
        return ""
    suffixes = []
    if page != "":
        suffixes.append(f"page {page}")
    if block_idx != "":
        suffixes.append(f"block {block_idx}")
    if chunk_idx != "":
        suffixes.append(f"chunk {chunk_idx}")
    return f"{src} - {', '.join(suffixes)}" if suffixes else src


def _chunk_sources(chunks: list[dict]) -> str:
    labels = [_chunk_source_label(c) for c in chunks]
    return " | ".join(dict.fromkeys(label for label in labels if label))


def _rag_trace_fields(
    *,
    rag_mode: str,
    retrieval_mode: str,
    smart_rag: bool,
    collection_name: str,
    embed_model: str,
    k: int,
    score_threshold: float,
    chunks: list[dict],
) -> dict:
    return {
        "rag_variant": build_rag_variant(retrieval_mode, smart_rag, rag_mode=rag_mode),
        "retrieval_mode": retrieval_mode if rag_mode != "no_rag" else "",
        "smart_rag": bool(smart_rag) if rag_mode != "no_rag" else False,
        "collection_name": collection_name if rag_mode != "no_rag" else "",
        "embed_model": embed_model if rag_mode != "no_rag" else "",
        "k": k if rag_mode != "no_rag" else 0,
        "score_threshold": score_threshold if rag_mode != "no_rag" else None,
        "rag_context_status": build_context_status(rag_mode, chunks),
    }


def _resolve_ollama_base_url(base_url: str | None = None) -> str:
    resolved = (base_url or "").strip()
    if not resolved:
        resolved = os.getenv("OLLAMA_BASE_URL", "").strip()
    if not resolved:
        return ""
    if not resolved.startswith("http"):
        resolved = f"http://{resolved}"
    return resolved.rstrip("/")


def load_questions(
    csv_path: str,
    question_col: str = "question",
    answer_col: str = "answer",
) -> List[Dict]:
    """
    Load questions and optional observation_idea from a CSV.
    Supports configurable headers and falls back to legacy "Questions"/"Answers".
    """
    items: List[Dict] = []
    with _open_csv_safe(csv_path) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question = (row.get(question_col) or row.get("Questions") or "").strip()
            observation = (row.get(answer_col) or row.get("Answers") or "").strip()
            if not question:
                continue
            items.append(
                {
                    "question_index": idx,
                    "question": question,
                    "observation_idea": observation,
                }
            )
    return items


def _open_csv_safe(path: str):
    """UTF-8 → cp1254 → latin-1 sırasıyla dener; Türkçe karakterleri doğru okur."""
    for enc in _CSV_ENCODINGS:
        try:
            f = open(path, newline="", encoding=enc)
            f.read()
            f.seek(0)
            return f
        except (UnicodeDecodeError, UnicodeError):
            try:
                f.close()
            except Exception:
                pass
    return open(path, newline="", encoding="utf-8", errors="replace")


def warmup_model(
    model: str = QA_OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
) -> None:
    """
    Ollama'ya modeli RAM'e yüklemesi için boş bir istek gönderir.
    Bu çağrı response_time ölçümüne dahil edilmez.
    """
    api_base = _resolve_ollama_base_url(base_url)
    if not api_base:
        raise ValueError(
            "OLLAMA_BASE_URL tanımlı değil; warmup yapılamadı."
        )
    resp = requests.post(
        f"{api_base}/api/generate",
        json={"model": model, "keep_alive": "5m"},
        timeout=timeout,
    )
    resp.raise_for_status()


def unload_model(
    model: str,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 30,
) -> None:
    """
    keep_alive=0 göndererek modeli Ollama'nın VRAM/RAM'inden boşaltır.
    """
    api_base = _resolve_ollama_base_url(base_url)
    if not api_base:
        raise ValueError(
            "OLLAMA_BASE_URL tanımlı değil; unload yapılamadı."
        )
    resp = requests.post(
        f"{api_base}/api/generate",
        json={"model": model, "keep_alive": 0},
        timeout=timeout,
    )
    resp.raise_for_status()


def _require_ollama_base_url(base_url: str) -> str:
    resolved = _resolve_ollama_base_url(base_url)
    if not resolved:
        raise ValueError(
            "OLLAMA_BASE_URL ortam değişkeni tanımlı değil. "
            "Lütfen .env dosyasına uzak sunucu adresini ekleyin "
            "(örn: OLLAMA_BASE_URL=http://192.168.1.151:11434)."
        )
    return resolved


def _get_ollama_model_memory(model: str, base_url: str) -> Dict[str, Optional[float]]:
    """Query /api/ps and return ram_mb + vram_mb for the given model. Returns None values on failure."""
    try:
        resp = requests.get(f"{base_url}/api/ps", timeout=5)
        resp.raise_for_status()
        for entry in (resp.json() or {}).get("models", []):
            name = entry.get("name", "") or entry.get("model", "")
            if name == model:
                size_total = entry.get("size", 0) or 0
                size_vram = entry.get("size_vram", 0) or 0
                return {
                    "vram_mb": round(size_vram / 1024 / 1024, 1),
                    "ram_mb": round((size_total - size_vram) / 1024 / 1024, 1),
                }
    except Exception:
        pass
    return {"vram_mb": None, "ram_mb": None}


def _collect_ollama_chat_response(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    base_url: str,
    timeout: int,
    options: Optional[Dict[str, Any]] = None,
    think: bool = False,
) -> Dict:
    api_base = _require_ollama_base_url(base_url)
    t0 = time.time()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    payload["think"] = bool(think)
    if options:
        payload["options"] = options
    resp = requests.post(
        f"{api_base}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()

    chunks: list[str] = []
    last_chunk = resp.json() or {}
    message = last_chunk.get("message") or {}
    content = message.get("content") or ""
    if content:
        chunks.append(content)

    answer = "".join(chunks).replace("\r\n", "\n").strip()
    result = {
        "answer": answer,
        "response_time_seconds": time.time() - t0,
        "eval_count": None,
        "eval_duration_seconds": None,
        "tokens_per_second": None,
    }
    raw_eval_count = last_chunk.get("eval_count")
    raw_prompt_eval_count = last_chunk.get("prompt_eval_count")
    raw_eval_duration = last_chunk.get("eval_duration")
    if isinstance(raw_eval_count, (int, float)) and isinstance(raw_eval_duration, (int, float)):
        result["eval_count"] = int(raw_eval_count)
        result["eval_duration_seconds"] = float(raw_eval_duration) / 1e9
        if result["eval_duration_seconds"] > 0:
            result["tokens_per_second"] = result["eval_count"] / result["eval_duration_seconds"]
    result["prompt_eval_count"] = (
        int(raw_prompt_eval_count) if isinstance(raw_prompt_eval_count, (int, float)) else None
    )
    result.update(_get_ollama_model_memory(model, api_base))
    return result


def generate_rag_answer_ollama(
    question: str,
    context: str,
    model: str = QA_OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
    *,
    system_prompt: Optional[str] = None,
    ollama_options: Optional[Dict[str, Any]] = None,
    think: bool = False,
) -> Dict:
    """
    Call a local Ollama model (e.g. Qwen3 1.7B) with question + context.

    Returns a dict with:
      - answer: str
      - response_time_seconds: float
    """
    user_content = build_rag_prompt(question, context)
    o_msgs: List[Dict[str, str]] = []
    if system_prompt and str(system_prompt).strip():
        o_msgs.append({"role": "system", "content": str(system_prompt).strip()})
    o_msgs.append({"role": "user", "content": user_content})
    return _collect_ollama_chat_response(
        messages=o_msgs,
        model=model,
        base_url=base_url,
        timeout=timeout,
        options=ollama_options,
        think=think,
    )


def generate_no_rag_answer_ollama(
    question: str,
    model: str = QA_OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
    *,
    system_prompt: Optional[str] = None,
    ollama_options: Optional[Dict[str, Any]] = None,
    think: bool = False,
) -> Dict:
    """
    Call a local Ollama model WITHOUT any retrieved context (no RAG).

    Returns a dict with:
      - answer: str
      - response_time_seconds: float
    """
    user_content = build_no_rag_prompt(question)
    o_msgs: List[Dict[str, str]] = []
    if system_prompt and str(system_prompt).strip():
        o_msgs.append({"role": "system", "content": str(system_prompt).strip()})
    o_msgs.append({"role": "user", "content": user_content})
    return _collect_ollama_chat_response(
        messages=o_msgs,
        model=model,
        base_url=base_url,
        timeout=timeout,
        options=ollama_options,
        think=think,
    )


def _require_llamacpp_base_url(base_url: str) -> str:
    resolved = (base_url or "").strip()
    if not resolved:
        raise ValueError("llama.cpp QA backend için qa_base_url gerekli.")
    if not resolved.startswith("http"):
        resolved = f"http://{resolved}"
    return resolved.rstrip("/")


def check_llamacpp_health(base_url: str, timeout: int = 10) -> bool:
    try:
        api_base = _require_llamacpp_base_url(base_url)
        health_base = api_base[:-3] if api_base.endswith("/v1") else api_base
        response = requests.get(f"{health_base}/health", timeout=timeout)
        response.raise_for_status()
        return True
    except Exception:
        return False


def _generate_answer_llamacpp(
    *,
    question: str,
    context: Optional[str],
    model: str,
    base_url: str,
    timeout: int,
    system_prompt: Optional[str],
    options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    api_base = _require_llamacpp_base_url(base_url)
    user_content = (
        build_rag_prompt(question, context or "")
        if context is not None
        else build_no_rag_prompt(question)
    )
    messages: List[Dict[str, str]] = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": user_content})

    request_options = dict(options or {})
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": int(request_options.pop("num_predict", 512)),
        "temperature": float(request_options.pop("temperature", 0.0)),
    }
    if "seed" in request_options:
        payload["seed"] = int(request_options.pop("seed"))
    payload.update(request_options)

    started = time.time()
    response = requests.post(
        f"{api_base}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("llama.cpp response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("llama.cpp choice has no message")
    answer = str(message.get("content") or "").strip()
    if not answer:
        raise ValueError("empty model answer")

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
    completion_tokens = usage.get("completion_tokens", timings.get("predicted_n"))
    prompt_tokens = usage.get("prompt_tokens", timings.get("prompt_n"))
    predicted_ms = timings.get("predicted_ms")
    eval_duration = (
        float(predicted_ms) / 1000
        if isinstance(predicted_ms, (int, float))
        else None
    )
    tokens_per_second = timings.get("predicted_per_second")
    if tokens_per_second is None and isinstance(completion_tokens, (int, float)) and eval_duration:
        tokens_per_second = float(completion_tokens) / eval_duration

    return {
        "answer": answer,
        "response_time_seconds": time.time() - started,
        "eval_count": int(completion_tokens) if isinstance(completion_tokens, (int, float)) else None,
        "prompt_eval_count": int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None,
        "eval_duration_seconds": eval_duration,
        "tokens_per_second": (
            float(tokens_per_second) if isinstance(tokens_per_second, (int, float)) else None
        ),
    }


def generate_rag_answer_llamacpp(
    question: str,
    context: str,
    model: str,
    base_url: str,
    timeout: int = 120,
    *,
    system_prompt: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _generate_answer_llamacpp(
        question=question,
        context=context,
        model=model,
        base_url=base_url,
        timeout=timeout,
        system_prompt=system_prompt,
        options=options,
    )


def generate_no_rag_answer_llamacpp(
    question: str,
    model: str,
    base_url: str,
    timeout: int = 120,
    *,
    system_prompt: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _generate_answer_llamacpp(
        question=question,
        context=None,
        model=model,
        base_url=base_url,
        timeout=timeout,
        system_prompt=system_prompt,
        options=options,
    )


def _normalize_eval_backend(eval_backend: str | None) -> str:
    return (eval_backend or "").strip().lower()


def is_external_eval_backend(eval_backend: str | None) -> bool:
    return _normalize_eval_backend(eval_backend) in {"deepseek", "openai"}


def get_eval_api_key_env_var(eval_backend: str | None) -> str:
    backend = _normalize_eval_backend(eval_backend)
    if backend == "openai":
        return "OPENAI_API_KEY"
    return "DEEPSEEK_API_KEY"


def get_eval_api_key(eval_backend: str | None, request_key: Optional[str] = None) -> str:
    key = (request_key or "").strip()
    if key:
        return key
    return os.environ.get(get_eval_api_key_env_var(eval_backend), "").strip()


def get_default_eval_model(eval_backend: str | None) -> str:
    backend = _normalize_eval_backend(eval_backend)
    if backend == "openai":
        return (os.environ.get("OPENAI_EVAL_MODEL_NAME") or OPENAI_EVAL_MODEL_NAME or "gpt-5.4-mini")
    if backend == "deepseek":
        return (os.environ.get("DEEPSEEK_EVAL_MODEL_NAME") or DEEPSEEK_EVAL_MODEL_NAME or EVAL_MODEL_NAME or "")
    return EVAL_MODEL_NAME or ""


def get_external_eval_client(eval_backend: str, api_key: Optional[str] = None) -> OpenAI:
    backend = _normalize_eval_backend(eval_backend)
    key = get_eval_api_key(eval_backend, api_key)
    if backend == "deepseek":
        kwargs: Dict[str, Any] = {"base_url": _DEEPSEEK_API_BASE_URL}
        if key:
            kwargs["api_key"] = key
        return OpenAI(**kwargs)
    if backend == "openai":
        if key:
            return OpenAI(api_key=key)
        return OpenAI()
    raise ValueError(f"Desteklenmeyen harici değerlendirme motoru: {eval_backend}")


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Compatibility wrapper for the current default external provider, DeepSeek."""
    return get_external_eval_client("DeepSeek", api_key=api_key)


# Columns the pipeline owns. A judge that emits one of these names must not be able to
# overwrite measured data, so they are dropped from evaluation output before merging.
_RESERVED_ROW_KEYS = frozenset({
    "model", "question", "question_index", "model_answer", "answer", "observation_idea",
    "response_time_seconds", "eval_count", "prompt_eval_count", "eval_duration_seconds",
    "tokens_per_second", "ram_mb", "vram_mb", "rag_type", "rag_variant", "retrieval_mode",
    "smart_rag", "collection_name", "embed_model", "k", "score_threshold", "rag_context_status",
    "retrieved_chunk_size", "chunk_sources", "retrieved_chunks", "inference_status",
    "evaluation_status", "error_stage", "error_type", "error_message", "attempt_count",
    "evaluation_warning", "inference_attempt_count", "evaluation_attempt_count",
    "timestamp", "mode", "expected_answer",
})

_AI_VERDICTS = frozenset({"correct", "partial", "incorrect"})
_AI_RISKS = frozenset({"low", "medium", "high"})


def _sanitize_eval_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Make judge output safe to merge into a result row.

    Eval system prompts are user-customizable (model profiles), so unknown fields are kept
    as-is. Only the pipeline's own column names are dropped, and the standard ai_* fields are
    validated when present. Problems are reported in `evaluation_warning` rather than raised,
    so one malformed judge response cannot abort a whole batch run.
    """
    if not isinstance(parsed, dict):
        # Structurally unusable: there are no fields to merge, so the row must not be counted
        # as a scored evaluation. `evaluation_status` is reserved, so only we can set it here.
        return {
            "evaluation_status": "warning",
            "evaluation_warning": f"judge returned {type(parsed).__name__}, expected object",
        }

    warnings: List[str] = []
    clean: Dict[str, Any] = {}
    for key, value in parsed.items():
        name = str(key)
        if name in _RESERVED_ROW_KEYS:
            warnings.append(f"dropped reserved field {name!r}")
            continue
        clean[name] = value.replace("\r\n", " ").replace("\n", " ").strip() if isinstance(value, str) else value

    if "ai_score" in clean:
        try:
            score = float(clean["ai_score"])
        except (TypeError, ValueError):
            warnings.append(f"ai_score not numeric: {clean['ai_score']!r}")
            clean["ai_score"] = None
        else:
            if 0 <= score <= 10:
                clean["ai_score"] = score
            else:
                warnings.append(f"ai_score out of range: {score!r}")
                clean["ai_score"] = None

    for field, allowed in (("ai_verdict", _AI_VERDICTS), ("ai_hallucination_risk", _AI_RISKS)):
        if field not in clean:
            continue
        normalized = str(clean[field]).strip().lower()
        if normalized in allowed:
            clean[field] = normalized
        else:
            # Keep the raw value visible in the CSV instead of silently rewriting it.
            warnings.append(f"{field} not one of {sorted(allowed)}: {clean[field]!r}")

    if warnings:
        clean["evaluation_warning"] = "; ".join(warnings)
    return clean


def evaluate_answer(
    record: Dict,
    eval_model: str = EVAL_MODEL_NAME,
    client: Optional[OpenAI] = None,
    *,
    system_prompt: Optional[str] = None,
    extra_create_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Evaluate a single QA record with an OpenAI-compatible client, returning a flat dict suitable for CSV.

    record MUST contain:
      - model
      - question_index
      - question
      - observation_idea
      - model_answer
      - response_time_seconds
    """
    if client is None:
        client = get_openai_client()

    system = (system_prompt or "").strip() or DEFAULT_EVAL_SYSTEM_PROMPT

    user = (
        f'Soru: "{record["question"]}"\n'
        f'Beklenen cevap: "{record.get("observation_idea", "")}"\n'
        f'Modelin cevabı: "{record.get("model_answer", "")}"\n'
    )

    create_kwargs: Dict[str, Any] = {
        "model": eval_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    if extra_create_kwargs:
        for k, v in extra_create_kwargs.items():
            if v is not None:
                create_kwargs[k] = v
    response = client.chat.completions.create(**create_kwargs)

    msg = response.choices[0].message

    parsed: Dict
    content = msg.content or "{}"
    parsed = json.loads(content)

    return _sanitize_eval_fields(parsed)


def _extract_json_from_text(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    raise RuntimeError(
        f"Modelin çıktısında JSON nesnesi bulunamadı. Ham çıktı: {text[:300]!r}"
    )


def _evaluate_answer_local(
    record: Dict,
    model: str,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
    *,
    system_prompt: Optional[str] = None,
    ollama_options: Optional[Dict[str, Any]] = None,
    think: bool = False,
) -> Dict:
    """
    OpenAI yerine yerel bir modeli (örn. Ollama) eval için kullan.
    """
    system = (system_prompt or "").strip() or DEFAULT_EVAL_SYSTEM_PROMPT

    user = (
        f'Soru: "{record["question"]}"\n'
        f'Beklenen cevap: "{record.get("observation_idea", "")}"\n'
        f'Modelin cevabı: "{record.get("model_answer", "")}"\n'
    )

    user_tail = (
        user + "\n\nYukarıdaki talimatlara göre SADECE geçerli bir JSON nesnesi üret. Başka metin ekleme."
    )

    api_base = _require_ollama_base_url(base_url)

    msg_list: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_tail},
    ]
    payload: Dict[str, Any] = {
        "model": model,
        "messages": msg_list,
        "stream": False,
    }
    payload["think"] = bool(think)
    if ollama_options:
        payload["options"] = ollama_options

    resp = requests.post(
        f"{api_base}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    content = ""
    if isinstance(data, dict):
        message = data.get("message") or {}
        if isinstance(message, dict):
            content = str(message.get("content") or "")
        elif "choices" in data:
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = str(msg.get("content") or "")

    content = content.replace("\r\n", "\n").strip()
    try:
        json_text = _extract_json_from_text(content)
        parsed = json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Yerel eval modeli ({model}) geçerli JSON döndürmedi. Ham içerik: {content[:300]!r}"
        ) from exc

    return _sanitize_eval_fields(parsed)


def evaluate_answer_any(
    record: Dict,
    eval_model: str = EVAL_MODEL_NAME,
    client: Optional[OpenAI] = None,
    backend: str = "openai",
    local_model: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
    *,
    eval_system_prompt: Optional[str] = None,
    eval_ollama_options: Optional[Dict[str, Any]] = None,
    eval_think: bool = False,
    eval_openai_extras: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Bir QA kaydını seçilen backend ile değerlendir.

    backend:
      - "openai": OpenAI API ile değerlendir (varsayılan).
      - "ollama": Yerel Ollama benzeri HTTP API ile değerlendir.
    """
    backend = (backend or "openai").lower()

    if backend == "ollama":
        model_name = local_model or QA_OLLAMA_MODEL
        return _evaluate_answer_local(
            record=record,
            model=model_name,
            base_url=base_url,
            timeout=timeout,
            system_prompt=eval_system_prompt,
            ollama_options=eval_ollama_options,
            think=eval_think,
        )

    # Varsayılan: mevcut OpenAI-compatible evaluate_answer fonksiyonunu kullan
    return evaluate_answer(
        record=record,
        eval_model=eval_model,
        client=client,
        system_prompt=eval_system_prompt,
        extra_create_kwargs=eval_openai_extras,
    )


def run_full_pipeline(
    csv_path: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    qdrant_url: str = QDRANT_URL,
    eval_model: str = EVAL_MODEL_NAME,
    k: int = 5,
    openai_client: Optional[OpenAI] = None,
    eval_backend: str = "openai",
    eval_local_model: Optional[str] = None,
    qa_model: str = QA_OLLAMA_MODEL,
    rag_mode: str = "rag",
    eval_enabled: bool = True,
    question_col: str = "question",
    answer_col: str = "answer",
    smart_chunking: bool = False,
    score_threshold: float = 0.55,
    retrieval_mode: str = "vector",
    embed_model: str = "",
    *,
    qa_backend: str = "ollama",
    qa_base_url: str = "",
    qa_system_prompt: Optional[str] = None,
    qa_ollama_options: Optional[Dict[str, Any]] = None,
    qa_think: bool = False,
    qa_timeout: int = 120,
    embedding_timeout: int = 300,
    eval_system_prompt: Optional[str] = None,
    eval_ollama_options: Optional[Dict[str, Any]] = None,
    eval_think: bool = False,
    eval_openai_extras: Optional[Dict[str, Any]] = None,
    profile_trace: Optional[Dict[str, Any]] = None,
    continue_on_question_error: bool = False,
    retry_backoff_seconds: float = 2.0,
    row_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    health_check: Optional[Callable[[], bool]] = None,
    skip_question_indices: Optional[Set[int]] = None,
    _questions_override: Optional[List[Dict[str, Any]]] = None,
    _skip_warmup: bool = False,
) -> List[Dict]:
    """
    High-level helper:
      - Load questions from CSV.
      - For each question, retrieve context from Qdrant (via Ollama embeddings).
      - Ask Ollama for an answer.
      - Evaluate the answer and return flat dict rows.

    rag_mode: "rag" | "no_rag" | "both"
    smart_chunking: True ise child chunk eşleşmesi → parent bağlam ile çalışır.
    retrieval_mode: "vector" | "bm25"
    """
    questions = _questions_override or load_questions(
        csv_path, question_col=question_col, answer_col=answer_col
    )
    if not questions:
        return []

    eval_backend = (eval_backend or "openai").lower()
    qa_backend = (qa_backend or "ollama").strip().lower()
    if qa_backend not in {"ollama", "llamacpp"}:
        raise ValueError(f"Unsupported QA backend: {qa_backend}")
    if qa_backend == "llamacpp":
        _require_llamacpp_base_url(qa_base_url)
    if eval_enabled and is_external_eval_backend(eval_backend) and openai_client is None:
        openai_client = get_external_eval_client(eval_backend)

    _REMOVE_COLS = {"observation_idea", "ai_strengths", "ai_issues", "ai_suggested_fix"}

    if continue_on_question_error:
        resilient_rows: List[Dict[str, Any]] = []
        completed_indices = skip_question_indices or set()
        pending_questions = [
            item for item in questions if int(item["question_index"]) not in completed_indices
        ]
        if not pending_questions:
            return resilient_rows
        if not _skip_warmup and qa_backend == "ollama":
            warmup_model(model=qa_model, timeout=qa_timeout)

        def emit(row: Dict[str, Any]) -> None:
            resilient_rows.append(row)
            if row_callback:
                row_callback(row)

        for item in pending_questions:
            question_index = int(item["question_index"])

            generated_rows: List[Dict[str, Any]] = []
            inference_exc: Optional[Exception] = None
            service_unavailable_exc: Optional[OllamaServiceUnavailable] = None
            inference_attempts = 0
            for inference_attempts in (1, 2):
                try:
                    generated_rows = run_full_pipeline(
                        csv_path=csv_path,
                        collection_name=collection_name,
                        qdrant_url=qdrant_url,
                        eval_model=eval_model,
                        k=k,
                        openai_client=openai_client,
                        eval_backend=eval_backend,
                        eval_local_model=eval_local_model,
                        qa_model=qa_model,
                        rag_mode=rag_mode,
                        eval_enabled=False,
                        question_col=question_col,
                        answer_col=answer_col,
                        smart_chunking=smart_chunking,
                        score_threshold=score_threshold,
                        retrieval_mode=retrieval_mode,
                        embed_model=embed_model,
                        qa_backend=qa_backend,
                        qa_base_url=qa_base_url,
                        qa_system_prompt=qa_system_prompt,
                        qa_ollama_options=qa_ollama_options,
                        qa_think=qa_think,
                        qa_timeout=qa_timeout,
                        embedding_timeout=embedding_timeout,
                        profile_trace=profile_trace,
                        _questions_override=[item],
                        _skip_warmup=True,
                    )
                    inference_exc = None
                    break
                except Exception as exc:
                    inference_exc = exc
                    if not is_transient_error(exc) or inference_attempts == 2:
                        break
                    checker = health_check or check_ollama_health
                    if not checker():
                        service_unavailable_exc = OllamaServiceUnavailable(
                            f"Ollama health check failed after {type(exc).__name__}: {exc}"
                        )
                        break
                    time.sleep(retry_backoff_seconds)

            if inference_exc is not None:
                error_row = {
                        "model": qa_model,
                        "question_index": question_index,
                        "question": item["question"],
                        "answer": item.get("observation_idea", ""),
                        "model_answer": "",
                        "rag_type": "RAG'li" if rag_mode != "no_rag" else "RAG'siz",
                        "rag_variant": build_rag_variant(retrieval_mode, smart_chunking, rag_mode),
                        "inference_status": "error",
                        "evaluation_status": "skipped",
                        "error_stage": "inference",
                        "error_type": type(inference_exc).__name__,
                        "error_message": str(inference_exc),
                        "attempt_count": inference_attempts,
                        "inference_attempt_count": inference_attempts,
                        "evaluation_attempt_count": 0,
                    }
                emit(error_row)
                if service_unavailable_exc is not None:
                    raise service_unavailable_exc from inference_exc
                continue

            for generated_row in generated_rows:
                generated_row.update(
                    {
                        "inference_status": "ok",
                        "evaluation_status": "skipped" if not eval_enabled else "pending",
                        "error_stage": "",
                        "error_type": "",
                        "error_message": "",
                        # attempt_count stays the inference count for CSV back-compatibility.
                        "attempt_count": inference_attempts,
                        "inference_attempt_count": inference_attempts,
                        "evaluation_attempt_count": 0,
                    }
                )
                if eval_enabled:
                    eval_exc: Optional[Exception] = None
                    eval_fields: Dict[str, Any] = {}
                    eval_attempts = 0
                    eval_record = {
                        **generated_row,
                        "observation_idea": generated_row.get("answer", ""),
                    }
                    for eval_attempts in (1, 2):
                        try:
                            eval_fields = evaluate_answer_any(
                                record=eval_record,
                                eval_model=eval_model,
                                client=openai_client if is_external_eval_backend(eval_backend) else None,
                                backend=eval_backend,
                                local_model=eval_local_model,
                                eval_system_prompt=eval_system_prompt,
                                eval_ollama_options=eval_ollama_options,
                                eval_think=eval_think,
                                eval_openai_extras=eval_openai_extras,
                            )
                            eval_exc = None
                            break
                        except Exception as exc:
                            eval_exc = exc
                            if not is_transient_error(exc) or eval_attempts == 2:
                                break
                            time.sleep(retry_backoff_seconds)
                    generated_row["evaluation_attempt_count"] = eval_attempts
                    if eval_exc is None:
                        # "ok" first so a sanitizer-set status (malformed judge output) wins.
                        generated_row["evaluation_status"] = "ok"
                        generated_row.update(eval_fields)
                    else:
                        generated_row.update(
                            {
                                "evaluation_status": "error",
                                "error_stage": "evaluation",
                                "error_type": type(eval_exc).__name__,
                                "error_message": str(eval_exc),
                            }
                        )
                emit(generated_row)
        return resilient_rows

    rows: List[Dict] = []

    if not _skip_warmup and qa_backend == "ollama":
        warmup_model(model=qa_model, timeout=qa_timeout)

    for item in questions:
        question = item["question"]
        observation_idea = item.get("observation_idea", "")

        # --- RAG'li ---
        if rag_mode in ("rag", "both"):
            if retrieval_mode == "bm25":
                if smart_chunking:
                    chunks = retrieve_chunks_bm25_smart(
                        question=question,
                        base_collection=collection_name,
                        k=k,
                        qdrant_url=qdrant_url,
                    )
                else:
                    chunks = retrieve_chunks_bm25(
                        question=question,
                        collection_name=collection_name,
                        k=k,
                        qdrant_url=qdrant_url,
                    )
            elif smart_chunking:
                chunks = retrieve_chunks_smart(
                    question=question,
                    base_collection=collection_name,
                    k=k,
                    qdrant_url=qdrant_url,
                    score_threshold=score_threshold,
                    embed_model=embed_model or None,
                    embedding_timeout=embedding_timeout,
                )
            else:
                chunks = retrieve_chunks(
                    question=question,
                    collection_name=collection_name,
                    k=k,
                    qdrant_url=qdrant_url,
                    score_threshold=score_threshold,
                    embed_model=embed_model or None,
                    embedding_timeout=embedding_timeout,
                )
            context = "\n\n".join(c["text"] for c in chunks)
            if qa_backend == "llamacpp":
                rag_result = generate_rag_answer_llamacpp(
                    question=question,
                    context=context,
                    model=qa_model,
                    base_url=qa_base_url,
                    system_prompt=qa_system_prompt,
                    options=qa_ollama_options,
                    timeout=qa_timeout,
                )
            else:
                rag_result = generate_rag_answer_ollama(
                    question=question,
                    context=context,
                    model=qa_model,
                    system_prompt=qa_system_prompt,
                    ollama_options=qa_ollama_options,
                    think=qa_think,
                    timeout=qa_timeout,
                )
            record = {
                "model": qa_model,
                "question_index": item["question_index"],
                "question": question,
                "observation_idea": observation_idea,
                "model_answer": rag_result.get("answer", ""),
                "response_time_seconds": rag_result.get("response_time_seconds", 0.0),
                "eval_count": rag_result.get("eval_count"),
                "prompt_eval_count": rag_result.get("prompt_eval_count"),
                "eval_duration_seconds": rag_result.get("eval_duration_seconds"),
                "tokens_per_second": rag_result.get("tokens_per_second"),
                "ram_mb": rag_result.get("ram_mb"),
                "vram_mb": rag_result.get("vram_mb"),
            }
            if eval_enabled:
                eval_fields = evaluate_answer_any(
                    record=record,
                    eval_model=eval_model,
                    client=openai_client if is_external_eval_backend(eval_backend) else None,
                    backend=eval_backend,
                    local_model=eval_local_model,
                    eval_system_prompt=eval_system_prompt,
                    eval_ollama_options=eval_ollama_options,
                    eval_think=eval_think,
                    eval_openai_extras=eval_openai_extras,
                )
                eval_row = {**record, **eval_fields}
            else:
                eval_row = {**record}
            for col in _REMOVE_COLS:
                eval_row.pop(col, None)
            eval_row["answer"] = observation_idea
            eval_row["rag_type"] = "RAG'li"
            eval_row.update(
                _rag_trace_fields(
                    rag_mode="rag",
                    retrieval_mode=retrieval_mode,
                    smart_rag=smart_chunking,
                    collection_name=collection_name,
                    embed_model=embed_model,
                    k=k,
                    score_threshold=score_threshold,
                    chunks=chunks,
                )
            )
            eval_row["retrieved_chunk_size"] = len(chunks)
            eval_row["chunk_sources"] = _chunk_sources(chunks)
            eval_row["retrieved_chunks"] = json.dumps([c["text"] for c in chunks], ensure_ascii=False)
            if profile_trace:
                eval_row.update({k: v for k, v in profile_trace.items() if v is not None})
            rows.append(eval_row)

        # --- RAG'siz ---
        if rag_mode in ("no_rag", "both"):
            if qa_backend == "llamacpp":
                no_rag_result = generate_no_rag_answer_llamacpp(
                    question=question,
                    model=qa_model,
                    base_url=qa_base_url,
                    system_prompt=qa_system_prompt,
                    options=qa_ollama_options,
                    timeout=qa_timeout,
                )
            else:
                no_rag_result = generate_no_rag_answer_ollama(
                    question=question,
                    model=qa_model,
                    system_prompt=qa_system_prompt,
                    ollama_options=qa_ollama_options,
                    think=qa_think,
                    timeout=qa_timeout,
                )
            record = {
                "model": qa_model,
                "question_index": item["question_index"],
                "question": question,
                "observation_idea": observation_idea,
                "model_answer": no_rag_result.get("answer", ""),
                "response_time_seconds": no_rag_result.get("response_time_seconds", 0.0),
                "eval_count": no_rag_result.get("eval_count"),
                "prompt_eval_count": no_rag_result.get("prompt_eval_count"),
                "eval_duration_seconds": no_rag_result.get("eval_duration_seconds"),
                "tokens_per_second": no_rag_result.get("tokens_per_second"),
                "ram_mb": no_rag_result.get("ram_mb"),
                "vram_mb": no_rag_result.get("vram_mb"),
            }
            if eval_enabled:
                eval_fields = evaluate_answer_any(
                    record=record,
                    eval_model=eval_model,
                    client=openai_client if is_external_eval_backend(eval_backend) else None,
                    backend=eval_backend,
                    local_model=eval_local_model,
                    eval_system_prompt=eval_system_prompt,
                    eval_ollama_options=eval_ollama_options,
                    eval_think=eval_think,
                    eval_openai_extras=eval_openai_extras,
                )
                eval_row = {**record, **eval_fields}
            else:
                eval_row = {**record}
            for col in _REMOVE_COLS:
                eval_row.pop(col, None)
            eval_row["answer"] = observation_idea
            eval_row["rag_type"] = "RAG'siz"
            eval_row.update(
                _rag_trace_fields(
                    rag_mode="no_rag",
                    retrieval_mode=retrieval_mode,
                    smart_rag=smart_chunking,
                    collection_name=collection_name,
                    embed_model=embed_model,
                    k=k,
                    score_threshold=score_threshold,
                    chunks=[],
                )
            )
            eval_row["retrieved_chunk_size"] = 0
            eval_row["chunk_sources"] = ""
            eval_row["retrieved_chunks"] = "[]"
            if profile_trace:
                eval_row.update({k: v for k, v in profile_trace.items() if v is not None})
            rows.append(eval_row)

    return rows


def write_results_to_csv(
    rows: Sequence[Dict],
    output_path: Union[str, TextIO],
    delimiter: str = ";",
) -> int:
    """
    Write evaluation rows to a CSV file (one row per dict).
    Returns number of rows written.
    """
    if not rows:
        if isinstance(output_path, str):
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write("")
        else:
            output_path.write("")
        return 0

    seen: dict = {}
    for row in rows:
        for k in row:
            if k not in seen:
                seen[k] = None
    fieldnames = list(seen)

    if isinstance(output_path, str):
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        writer = csv.DictWriter(output_path, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return len(rows)


if __name__ == "__main__":
    # Simple CLI usage example:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RAG + evaluation pipeline over a CSV file."
    )
    parser.add_argument("csv_path", help="Input CSV path (e.g. sample_rag_input.csv)")
    parser.add_argument(
        "--out",
        dest="output_path",
        default="output.csv",
        help="Output CSV path (default: output.csv)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve from Chroma.",
    )
    args = parser.parse_args()

    client = get_openai_client()
    rows_ = run_full_pipeline(
        csv_path=args.csv_path,
        k=args.k,
        openai_client=client,
        qdrant_url=QDRANT_URL,
    )
    written = write_results_to_csv(rows_, args.output_path)
    print(f"Wrote {written} rows to {args.output_path}")
