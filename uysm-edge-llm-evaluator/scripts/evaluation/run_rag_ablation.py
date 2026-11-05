#!/usr/bin/env python3
"""Run the four supported RAG retrieval variants over one QA CSV.

Default target:
  CSV: data/qa/old/question_answers_100.csv
  base Qdrant collection: rag_ablation

The script expects these Qdrant collections to exist and be populated:
  rag_ablation
  rag_ablation_children
  rag_ablation_parents
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=True)
except Exception:
    pass

from pipeline import (  # noqa: E402
    EVAL_MODEL_NAME,
    OllamaServiceUnavailable,
    check_llamacpp_health,
    check_ollama_health,
    get_default_eval_model,
    get_external_eval_client,
    load_questions,
    run_full_pipeline,
    unload_model,
    write_results_to_csv,
)
from web_backend.compat import list_ollama_models, list_qdrant_collections  # noqa: E402


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

RAG_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "name": "classic_vector",
        "retrieval_mode": "vector",
        "smart_rag": False,
        "required_collections": ("{base}",),
    },
    {
        "name": "smart_vector",
        "retrieval_mode": "vector",
        "smart_rag": True,
        "required_collections": ("{base}_children", "{base}_parents"),
    },
    {
        "name": "classic_bm25",
        "retrieval_mode": "bm25",
        "smart_rag": False,
        "required_collections": ("{base}",),
    },
    {
        "name": "smart_bm25",
        "retrieval_mode": "bm25",
        "smart_rag": True,
        "required_collections": ("{base}_children", "{base}_parents"),
    },
)


def safe_filename(name: str) -> str:
    return SAFE_NAME_RE.sub("_", name).strip("_") or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "data/qa/old/question_answers_100.csv"),
        help="Soru/cevap CSV path.",
    )
    parser.add_argument(
        "--collection",
        default="rag_ablation",
        help="Fallback base Qdrant collection. Used when --classic-collection/--smart-collection are not set.",
    )
    parser.add_argument(
        "--classic-collection",
        default="",
        help="Classic collection for classic_vector and classic_bm25.",
    )
    parser.add_argument(
        "--smart-collection",
        default="",
        help="Smart base collection for smart_vector and smart_bm25; script uses <name>_children/<name>_parents.",
    )
    parser.add_argument(
        "--models",
        default=os.getenv("QA_OLLAMA_MODEL", ""),
        help='QA model listesi: "m1,m2" veya JSON list \'["m1","m2"]\'.',
    )
    parser.add_argument(
        "--all",
        dest="use_all",
        action="store_true",
        help="Ollama'daki embedding olmayan tüm QA modellerini çalıştır.",
    )
    parser.add_argument("--list-models", action="store_true", help="Ollama'daki QA modellerini listele ve çık.")
    parser.add_argument(
        "--qa-backend",
        choices=["ollama", "llamacpp"],
        default="ollama",
        help="Answer generation backend (default: ollama).",
    )
    parser.add_argument(
        "--qa-base-url",
        default="",
        help="QA backend base URL. llama.cpp example: http://127.0.0.1:8084/v1",
    )
    parser.add_argument("--out-dir", default="", help="Output dir. Default: data/results/rag_ablation_<timestamp>.")
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Retrieval limit (classic: maximum chunks; smart: maximum unique parents).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Ollama warmup/chat timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=512,
        help="Maximum Ollama output tokens per answer (default: 512).",
    )
    parser.add_argument("--embedding-timeout", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing --out-dir; successful inference rows are skipped.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.55, help="Vector search score threshold.")
    parser.add_argument("--embed-model", default=os.getenv("OLLAMA_EMBED_MODEL", ""), help="Embedding model name.")
    parser.add_argument("--question-col", default="question")
    parser.add_argument("--answer-col", default="answer")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enable answer evaluation. Default is off; retrieval/answer ablation only.",
    )
    parser.add_argument(
        "--eval-backend",
        choices=["deepseek", "openai", "local"],
        default="deepseek",
        help="Evaluation backend. 'local' uses Ollama.",
    )
    parser.add_argument(
        "--eval-model",
        default="",
        help="DeepSeek/OpenAI eval model. Default is backend-specific env/default.",
    )
    parser.add_argument("--eval-local-model", default="", help="Local/Ollama eval backend için model adı.")
    parser.add_argument(
        "--variants",
        default=",".join(v["name"] for v in RAG_VARIANTS),
        help="Virgülle ayrılmış varyant listesi. Default: all.",
    )
    parser.add_argument(
        "--skip-collection-check",
        action="store_true",
        help="Qdrant collection existence check'i atla.",
    )
    parser.add_argument(
        "--keep-model-loaded",
        action="store_true",
        help="Run sonunda Ollama modelini unload etme.",
    )
    return parser.parse_args()


def log_line(log_fp, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    log_fp.write(line + "\n")
    log_fp.flush()


def parse_models(raw: str) -> list[str]:
    value = (raw or "").strip()
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("--models list formatı string elemanlardan oluşmalı.")
        return [item.strip() for item in parsed if item.strip()]
    return [m.strip() for m in value.split(",") if m.strip()]


def resolve_models(raw_models: str, use_all: bool, list_models_fn=list_ollama_models) -> list[str]:
    if not use_all:
        return parse_models(raw_models)
    models, error, _, _ = list_models_fn()
    if error:
        raise RuntimeError(error)
    return [model for model in models if model]


def selected_variants(raw: str) -> list[dict[str, Any]]:
    requested = [v.strip() for v in raw.split(",") if v.strip()]
    by_name = {v["name"]: v for v in RAG_VARIANTS}
    unknown = [v for v in requested if v not in by_name]
    if unknown:
        allowed = ", ".join(by_name)
        raise ValueError(f"Bilinmeyen variant: {', '.join(unknown)}. Allowed: {allowed}")
    return [by_name[name] for name in requested]


def resolve_eval_backend(eval_backend: str) -> str:
    backend = (eval_backend or "").strip().lower()
    if backend == "local":
        return "ollama"
    return backend


def required_eval_env_var(eval_backend: str) -> str:
    backend = (eval_backend or "").strip().lower()
    if backend == "deepseek":
        return "DEEPSEEK_API_KEY"
    if backend == "openai":
        return "OPENAI_API_KEY"
    return ""


def collection_for_variant(
    variant: dict[str, Any],
    *,
    fallback_collection: str,
    classic_collection: str = "",
    smart_collection: str = "",
) -> str:
    if variant.get("smart_rag"):
        return (smart_collection or fallback_collection).strip()
    return (classic_collection or fallback_collection).strip()


def required_collections_for_run(
    variants: list[dict[str, Any]],
    *,
    fallback_collection: str,
    classic_collection: str = "",
    smart_collection: str = "",
) -> set[str]:
    required: set[str] = set()
    for variant in variants:
        base_collection = collection_for_variant(
            variant,
            fallback_collection=fallback_collection,
            classic_collection=classic_collection,
            smart_collection=smart_collection,
        )
        for template in variant["required_collections"]:
            required.add(template.format(base=base_collection))
    return required


def require_collections(
    variants: list[dict[str, Any]],
    *,
    fallback_collection: str,
    classic_collection: str = "",
    smart_collection: str = "",
) -> None:
    collections, error = list_qdrant_collections()
    if error:
        raise RuntimeError(error)
    existing = set(collections)
    required = required_collections_for_run(
        variants,
        fallback_collection=fallback_collection,
        classic_collection=classic_collection,
        smart_collection=smart_collection,
    )
    missing = sorted(required - existing)
    if missing:
        raise RuntimeError(
            "Eksik Qdrant koleksiyonları: "
            + ", ".join(missing)
            + f". Mevcut koleksiyon sayısı: {len(existing)}."
        )


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_output_parent(path)
    fieldnames = [
        "variant",
        "model",
        "collection",
        "status",
        "total_questions",
        "attempted_questions",
        "successful_questions",
        "inference_errors",
        "evaluation_errors",
        "rows",
        "duration_seconds",
        "output_csv",
        "error",
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def ensure_output_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_output_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_results_to_csv(rows, str(tmp_path))
    os.replace(tmp_path, path)


def read_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp, delimiter=";"))
    for row in rows:
        raw_index = row.get("question_index")
        if raw_index not in (None, ""):
            try:
                row["question_index"] = int(raw_index)
            except (TypeError, ValueError):
                pass
    return rows


def _row_key(row: dict[str, Any]) -> tuple[int, str]:
    """Checkpoint identity. rag_mode="both" emits one RAG and one no-RAG row per question,
    so question_index alone is not unique."""
    return int(row.get("question_index", -1)), str(row.get("rag_type", "") or "")


def successful_question_indices(
    rows: list[dict[str, Any]], *, require_evaluation: bool = False
) -> set[int]:
    """Question indices whose every checkpointed row succeeded."""
    by_index: dict[int, bool] = {}
    for row in rows:
        try:
            index = int(row["question_index"])
        except (KeyError, TypeError, ValueError):
            continue
        ok = row.get("inference_status") == "ok"
        if require_evaluation and row.get("evaluation_status") != "ok":
            ok = False
        by_index[index] = by_index.get(index, True) and ok
    return {index for index, ok in by_index.items() if ok}


def upsert_question_row(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> None:
    key = _row_key(new_row)
    rows[:] = [row for row in rows if _row_key(row) != key]
    rows.append(new_row)
    rows.sort(key=_row_key)


def validate_positive_args(args: argparse.Namespace) -> None:
    for name in ("timeout", "embedding_timeout", "k", "num_predict"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} pozitif olmalı.")


def validate_backend_args(args: argparse.Namespace) -> None:
    if args.qa_backend != "llamacpp":
        return
    if not (args.qa_base_url or "").strip():
        raise ValueError("llama.cpp backend için --qa-base-url gerekli.")
    if args.use_all or args.list_models:
        raise ValueError("--all ve --list-models yalnızca Ollama backend ile kullanılabilir.")


def ensure_output_dir_available(out_dir: Path, *, resume: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not resume:
        raise ValueError(
            f"Output dizini boş değil: {out_dir}. Devam etmek için --resume kullanın."
        )


def main() -> int:
    args = parse_args()

    try:
        validate_positive_args(args)
        validate_backend_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.list_models:
        models, error, _, filtered_embedding_count = list_ollama_models()
        if error:
            print(error, file=sys.stderr)
            return 1
        for model in models:
            print(model)
        if filtered_embedding_count:
            print(f"# filtered embedding models: {filtered_embedding_count}", file=sys.stderr)
        return 0

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV bulunamadı: {csv_path}", file=sys.stderr)
        return 1

    try:
        variants = selected_variants(args.variants)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        models = resolve_models(args.models, args.use_all)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not models:
        print(
            "QA modeli bulunamadı. Örnek: --models 'llama3.1:8b' "
            "veya tüm QA modelleri için --all kullanın.",
            file=sys.stderr,
        )
        return 1

    pipeline_eval_backend = resolve_eval_backend(args.eval_backend)
    eval_model = args.eval_model or get_default_eval_model(pipeline_eval_backend)

    if args.eval:
        required_env = required_eval_env_var(args.eval_backend)
        if required_env and not os.getenv(required_env):
            print(
                f"{args.eval_backend} eval için {required_env} gerekli. "
                "Alternatif: --eval-backend local --eval-local-model MODEL_ADI.",
                file=sys.stderr,
            )
            return 1
        if pipeline_eval_backend == "ollama" and not (args.eval_local_model or "").strip():
            print(
                "Local eval için --eval-local-model gerekli. "
                "Örnek: --eval --eval-backend local --eval-local-model qwen3:1.7b",
                file=sys.stderr,
            )
            return 1

    if not args.skip_collection_check:
        try:
            require_collections(
                variants,
                fallback_collection=args.collection,
                classic_collection=args.classic_collection,
                smart_collection=args.smart_collection,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / f"data/results/rag_ablation_{ts}"
    try:
        ensure_output_dir_available(out_dir, resume=args.resume)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    errors_path = out_dir / "errors.log"
    manifest_path = out_dir / "manifest.csv"
    merged_path = out_dir / "merged_results.csv"

    external_eval_client = (
        get_external_eval_client(pipeline_eval_backend)
        if args.eval and pipeline_eval_backend in {"deepseek", "openai"}
        else None
    )
    manifest_rows: list[dict[str, Any]] = []
    total_questions = len(
        load_questions(str(csv_path), question_col=args.question_col, answer_col=args.answer_col)
    )
    service_aborted = False

    ensure_output_parent(log_path)
    log_mode = "a" if args.resume and log_path.exists() else "w"
    with log_path.open(log_mode, encoding="utf-8", buffering=1) as log_fp:
        log_line(log_fp, f"CSV: {csv_path}")
        log_line(log_fp, f"Fallback collection: {args.collection}")
        log_line(log_fp, f"Classic collection: {args.classic_collection or args.collection}")
        log_line(log_fp, f"Smart collection: {args.smart_collection or args.collection}")
        log_line(log_fp, f"Models: {', '.join(models)}")
        log_line(log_fp, f"QA backend: {args.qa_backend} base_url={args.qa_base_url or '-'}")
        log_line(log_fp, f"Variants: {', '.join(v['name'] for v in variants)}")
        log_line(log_fp, f"Ollama limits: timeout={args.timeout}s num_predict={args.num_predict}")
        log_line(
            log_fp,
            f"Sampling: temperature={args.temperature} seed={args.seed} "
            f"embedding_timeout={args.embedding_timeout}s resume={args.resume}",
        )
        log_line(
            log_fp,
            "Eval: "
            + (
                f"enabled backend={args.eval_backend} pipeline_backend={pipeline_eval_backend} "
                f"eval_model={eval_model or '-'} local_eval_model={args.eval_local_model or '-'}"
                if args.eval
                else "disabled"
            ),
        )
        log_line(log_fp, f"Output dir: {out_dir}")

        for model in models:
            model_had_work = False
            for variant in variants:
                variant_name = variant["name"]
                collection_name = collection_for_variant(
                    variant,
                    fallback_collection=args.collection,
                    classic_collection=args.classic_collection,
                    smart_collection=args.smart_collection,
                )
                output_csv = checkpoint_dir / f"{safe_filename(model)}__{variant_name}.csv"
                checkpoint_rows = read_checkpoint_rows(output_csv) if args.resume else []
                completed_indices = successful_question_indices(
                    checkpoint_rows, require_evaluation=bool(args.eval)
                )
                started = time.time()
                log_line(log_fp, f"START model={model} variant={variant_name} collection={collection_name}")
                manifest_row = {
                    "variant": variant_name,
                    "model": model,
                    "collection": collection_name,
                    "status": "running",
                    "total_questions": total_questions,
                    "attempted_questions": len(checkpoint_rows),
                    "successful_questions": sum(
                        row.get("inference_status") == "ok" for row in checkpoint_rows
                    ),
                    "inference_errors": sum(
                        row.get("inference_status") == "error" for row in checkpoint_rows
                    ),
                    "evaluation_errors": sum(
                        row.get("evaluation_status") == "error" for row in checkpoint_rows
                    ),
                    "rows": len(checkpoint_rows),
                    "duration_seconds": 0,
                    "output_csv": str(output_csv.relative_to(out_dir)),
                    "error": "",
                }
                manifest_rows.append(manifest_row)
                write_manifest(manifest_path, manifest_rows)

                def persist_row(row: Dict[str, Any]) -> None:
                    row["ablation_variant"] = variant_name
                    row["ablation_collection"] = collection_name
                    upsert_question_row(checkpoint_rows, row)
                    write_rows_atomic(output_csv, checkpoint_rows)
                    manifest_row.update(
                        {
                            "attempted_questions": len(checkpoint_rows),
                            "successful_questions": sum(
                                item.get("inference_status") == "ok" for item in checkpoint_rows
                            ),
                            "inference_errors": sum(
                                item.get("inference_status") == "error" for item in checkpoint_rows
                            ),
                            "evaluation_errors": sum(
                                item.get("evaluation_status") == "error" for item in checkpoint_rows
                            ),
                            "rows": len(checkpoint_rows),
                            "duration_seconds": round(time.time() - started, 2),
                        }
                    )
                    write_manifest(manifest_path, manifest_rows)

                try:
                    processed_rows = run_full_pipeline(
                        csv_path=str(csv_path),
                        collection_name=collection_name,
                        eval_model=eval_model or "",
                        k=args.k,
                        openai_client=external_eval_client,
                        eval_backend=pipeline_eval_backend,
                        eval_local_model=args.eval_local_model or None,
                        qa_model=model,
                        rag_mode="rag",
                        eval_enabled=bool(args.eval),
                        question_col=args.question_col,
                        answer_col=args.answer_col,
                        smart_chunking=bool(variant["smart_rag"]),
                        score_threshold=args.score_threshold,
                        retrieval_mode=str(variant["retrieval_mode"]),
                        embed_model=args.embed_model,
                        qa_backend=args.qa_backend,
                        qa_base_url=args.qa_base_url,
                        qa_timeout=args.timeout,
                        embedding_timeout=args.embedding_timeout,
                        qa_ollama_options={
                            "num_predict": args.num_predict,
                            "temperature": args.temperature,
                            "seed": args.seed,
                        },
                        continue_on_question_error=True,
                        row_callback=persist_row,
                        health_check=(
                            (lambda: check_llamacpp_health(args.qa_base_url, timeout=10))
                            if args.qa_backend == "llamacpp"
                            else (lambda: check_ollama_health(timeout=10))
                        ),
                        skip_question_indices=completed_indices,
                    )
                    model_had_work = model_had_work or bool(processed_rows)
                    duration = round(time.time() - started, 2)
                    manifest_row.update(status="ok", duration_seconds=duration)
                    write_manifest(manifest_path, manifest_rows)
                    log_line(
                        log_fp,
                        f"OK model={model} variant={variant_name} rows={len(checkpoint_rows)} duration={duration}s",
                    )
                except OllamaServiceUnavailable as exc:
                    duration = round(time.time() - started, 2)
                    manifest_row.update(
                        status="aborted_service_unavailable",
                        duration_seconds=duration,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    write_manifest(manifest_path, manifest_rows)
                    log_line(log_fp, f"ABORT model={model} variant={variant_name}: {exc}")
                    service_aborted = True
                    break
                except Exception as exc:
                    duration = round(time.time() - started, 2)
                    manifest_row.update(
                        status="error",
                        duration_seconds=duration,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    write_manifest(manifest_path, manifest_rows)
                    ensure_output_parent(errors_path)
                    with errors_path.open("a", encoding="utf-8") as error_fp:
                        error_fp.write(
                            f"\n\n=== {datetime.now().isoformat()} | model={model} | variant={variant_name} ===\n"
                        )
                        error_fp.write(traceback.format_exc())
                    log_line(log_fp, f"ERR model={model} variant={variant_name}: {type(exc).__name__}: {exc}")

            if service_aborted:
                break

            if model_had_work and not args.keep_model_loaded and args.qa_backend == "ollama":
                try:
                    unload_model(model, base_url=os.getenv("OLLAMA_BASE_URL", ""))
                except Exception as exc:
                    log_line(log_fp, f"WARN unload_model({model}) başarısız: {exc}")
                    if not check_ollama_health(timeout=10):
                        service_aborted = True
                        log_line(log_fp, "ABORT Ollama unload sonrası health check başarısız.")
                        break

        all_rows = []
        for checkpoint_path in sorted(checkpoint_dir.glob("*.csv")):
            all_rows.extend(read_checkpoint_rows(checkpoint_path))
        ensure_output_parent(merged_path)
        write_rows_atomic(merged_path, all_rows)
        merged_count = len(all_rows)
        write_manifest(manifest_path, manifest_rows)
        ok_count = sum(1 for row in manifest_rows if row["status"] == "ok")
        err_count = sum(1 for row in manifest_rows if row["status"] == "error")
        log_line(log_fp, f"DONE ok={ok_count} errors={err_count} merged_rows={merged_count}")
        log_line(log_fp, f"Merged CSV: {merged_path}")
        log_line(log_fp, f"Manifest: {manifest_path}")

    return 1 if service_aborted or any(row["status"] == "error" for row in manifest_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
