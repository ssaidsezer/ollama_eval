"""ASR benchmark report reader and pipeline runner."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
ASR_DIR = WORKSPACE_DIR / "asr"
PIPELINE_DIR = ASR_DIR / "ATBK_Pipeline"
REPORTS_DIR = PIPELINE_DIR / "reports"
TRANSCRIPTS_DIR = PIPELINE_DIR / "transcripts"
AUDIO_DIR = PIPELINE_DIR / "data" / "audio"


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def get_reports() -> Dict[str, Any]:
    """Return all available ASR benchmark artifacts."""
    return {
        "whisper_benchmark": _read_json(REPORTS_DIR / "whisper_benchmark.json"),
        "qwen_benchmark": _read_json(REPORTS_DIR / "qwen_benchmark.json"),
        "benchmark_results": _read_json(REPORTS_DIR / "benchmark_results.json"),
        "all_qa_results": _read_json(REPORTS_DIR / "all_qa_results.json"),
        "document_analysis": _read_json(REPORTS_DIR / "document_analysis.json"),
        "all_transcriptions": _read_json(TRANSCRIPTS_DIR / "all_transcriptions.json"),
        "benchmark_report_text": _read_text(REPORTS_DIR / "benchmark_report.txt"),
        "exists": PIPELINE_DIR.exists(),
    }


def resolve_audio_path(rel_path: str) -> Optional[Path]:
    """Safely resolve an audio path under the pipeline audio directory."""
    if not rel_path:
        return None

    p = Path(rel_path)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((WORKSPACE_DIR / rel_path).resolve())
        candidates.append((ASR_DIR / rel_path).resolve())
        candidates.append((PIPELINE_DIR / rel_path).resolve())
        candidates.append((AUDIO_DIR / rel_path).resolve())

    audio_root = AUDIO_DIR.resolve()
    for cand in candidates:
        try:
            cand.relative_to(audio_root)
        except ValueError:
            continue
        if cand.exists() and cand.is_file():
            return cand
    return None


def run_pipeline_job(
    progress_cb: Callable[[str, int, int, float], None],
    *,
    steps: Optional[List[int]] = None,
    skip_benchmark: bool = False,
) -> Dict[str, Any]:
    """Run atbk_asr_llm_local.py as a subprocess. Streams stdout into progress phase text."""
    script = ASR_DIR / "atbk_asr_llm_local.py"
    if not script.exists():
        raise FileNotFoundError(f"ASR scripti bulunamadı: {script}")

    cmd: List[str] = [sys.executable, str(script)]
    if steps:
        cmd += ["--steps", ",".join(str(s) for s in steps)]
    if skip_benchmark:
        cmd.append("--skip-benchmark")

    started = time.time()
    progress_cb("Başlatılıyor", 0, 0, 0.0)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ASR_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_line = ""
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        last_line = line[-200:]
        progress_cb(last_line, 0, 0, time.time() - started)

    rc = proc.wait()
    elapsed = time.time() - started
    if rc != 0:
        raise RuntimeError(f"ASR pipeline başarısız (rc={rc}). Son satır: {last_line}")

    return {
        "exit_code": rc,
        "elapsed_sec": elapsed,
        "command": " ".join(cmd),
    }
