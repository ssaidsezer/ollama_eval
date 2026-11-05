#!/usr/bin/env python3
"""Batch CSV evaluation across many Ollama QA models.

- Per-model `try/except` and optional timeout: a failing/hanging model
  is logged and the loop continues with the next one.
- Per-model checkpoint CSV is flushed to disk immediately, so partial
  results survive disconnects/crashes.
- A final merged CSV with all rows from successful models.
- Plain-text log of every model's start/end, duration, status, error.
- Designed to run inside tmux/screen over SSH.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from pipeline import (  # noqa: E402
    EVAL_MODEL_NAME,
    get_openai_client,
    run_full_pipeline,
    unload_model,
    write_results_to_csv,
)
from rag.config import DEFAULT_COLLECTION_NAME, QDRANT_URL  # noqa: E402
from web_backend.compat import list_ollama_models  # noqa: E402


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    return SAFE_NAME_RE.sub("_", name).strip("_") or "model"


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise _TimeoutError("per-model timeout exceeded")


def run_with_timeout(seconds: int, fn, *args, **kwargs):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return fn(*args, **kwargs)
    prev = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        default=str(REPO_ROOT / "data/qa/old/question_answers_100.csv"),
        help="Soru/cevap CSV path",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Çıktı dizini (default: data/results/batch_<timestamp>)",
    )
    p.add_argument(
        "--models",
        default="",
        help="Virgülle ayrılmış model listesi. Boşsa Ollama'da kurulu tüm modeller.",
    )
    p.add_argument("--exclude", default="", help="Virgülle ayrılmış hariç tutulan modeller.")
    p.add_argument("--list-models", action="store_true", help="Sadece kurulu modelleri listele.")

    p.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME))
    p.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", QDRANT_URL))
    p.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", ""))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--score-threshold", type=float, default=0.55)
    p.add_argument("--rag-mode", choices=["rag", "no_rag", "both"], default="rag")
    p.add_argument("--retrieval-mode", choices=["vector", "bm25"], default="vector")
    p.add_argument("--smart-rag", action="store_true")
    p.add_argument("--question-col", default="question")
    p.add_argument("--answer-col", default="answer")

    p.add_argument("--no-eval", action="store_true", help="Değerlendirmeyi kapat (sadece QA çıktısı).")
    p.add_argument("--eval-backend", choices=["openai", "ollama"], default="openai")
    p.add_argument("--eval-model", default=os.getenv("EVAL_MODEL_NAME") or EVAL_MODEL_NAME or "")
    p.add_argument("--eval-local-model", default=None, help="Ollama eval için model adı.")

    p.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Model başına saniye cinsinden timeout (0 = kapalı). Default 1800.",
    )
    return p.parse_args()


def discover_models(args: argparse.Namespace, log) -> List[str]:
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    excluded = {m.strip() for m in args.exclude.split(",") if m.strip()}

    if requested:
        models = requested
    else:
        installed, _, _, _ = list_ollama_models()
        models = sorted({m for m in installed if m})
        log(f"Ollama'da kurulu {len(models)} model bulundu.")

    models = [m for m in models if m not in excluded]
    return models


def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / f"data/results/batch_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log_path = out_dir / f"batch_{ts}.log"
    final_csv = out_dir / f"output_all_{ts}.csv"
    errors_log = out_dir / "errors.log"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        return _run(args, out_dir, ckpt_dir, final_csv, errors_log, log_fp)


def _run(args, out_dir, ckpt_dir, final_csv, errors_log, log_fp) -> int:
    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")

    log(f"Output dir: {out_dir}")
    log(f"CSV: {args.csv}")
    log(f"Collection: {args.collection} | k={args.k} | rag_mode={args.rag_mode} | retrieval={args.retrieval_mode}")

    if args.list_models:
        installed, _, _, _ = list_ollama_models()
        for m in sorted({m for m in installed if m}):
            print(m)
        return 0

    models = discover_models(args, log)
    if not models:
        log("HATA: Çalıştırılacak model yok.")
        return 1
    log(f"Çalışacak modeller ({len(models)}): {', '.join(models)}")

    if not Path(args.csv).exists():
        log(f"HATA: CSV bulunamadı: {args.csv}")
        return 1

    if args.rag_mode != "no_rag" and not (args.collection or "").strip():
        log("HATA: rag_mode != no_rag için --collection gerekli.")
        return 1

    openai_client = None
    eval_enabled = not args.no_eval
    if eval_enabled and args.eval_backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            log("HATA: OpenAI eval için OPENAI_API_KEY gerekli (veya --eval-backend ollama).")
            return 1
        openai_client = get_openai_client()

    all_rows: list[dict] = []
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    overall_start = time.time()
    for idx, model in enumerate(models, start=1):
        slug = safe_filename(model)
        ckpt_path = ckpt_dir / f"output_{slug}.csv"
        log(f"--- [{idx}/{len(models)}] {model} başlıyor")
        t0 = time.time()
        try:
            rows = run_with_timeout(
                args.timeout,
                run_full_pipeline,
                csv_path=args.csv,
                collection_name=args.collection,
                qdrant_url=args.qdrant_url,
                eval_model=args.eval_model or "",
                k=args.k,
                openai_client=openai_client,
                eval_backend=args.eval_backend,
                eval_local_model=args.eval_local_model,
                qa_model=model,
                rag_mode=args.rag_mode,
                eval_enabled=eval_enabled,
                question_col=args.question_col,
                answer_col=args.answer_col,
                embed_model=args.embed_model,
                smart_chunking=args.smart_rag,
                score_threshold=args.score_threshold,
                retrieval_mode=args.retrieval_mode,
            )
            dur = time.time() - t0
            n = write_results_to_csv(rows, str(ckpt_path))
            all_rows.extend(rows)
            succeeded.append(model)
            log(f"OK  {model}: {n} satır → {ckpt_path.name} ({dur:.1f}s)")
        except _TimeoutError:
            dur = time.time() - t0
            msg = f"TIMEOUT after {dur:.1f}s (limit {args.timeout}s)"
            failed.append((model, msg))
            log(f"TMO {model}: {msg}")
        except Exception as exc:
            dur = time.time() - t0
            tb = traceback.format_exc()
            failed.append((model, f"{type(exc).__name__}: {exc}"))
            log(f"ERR {model} ({dur:.1f}s): {type(exc).__name__}: {exc}")
            with errors_log.open("a", encoding="utf-8") as fh:
                fh.write(f"\n\n=== {datetime.now().isoformat()} | {model} ===\n{tb}")
        finally:
            try:
                unload_model(model)
            except Exception as exc:
                log(f"WARN unload_model({model}) başarısız: {exc}")

    total_dur = time.time() - overall_start

    if all_rows:
        n_total = write_results_to_csv(all_rows, str(final_csv))
        log(f"FINAL {final_csv} → {n_total} satır")
    else:
        log("UYARI: Hiç başarılı satır yok, final CSV yazılmadı.")

    log(f"Toplam süre: {total_dur/60:.1f} dk | başarılı: {len(succeeded)} | başarısız: {len(failed)}")
    for m, err in failed:
        log(f"  FAILED {m}: {err}")

    return 0 if all_rows else 2


if __name__ == "__main__":
    sys.exit(main())
