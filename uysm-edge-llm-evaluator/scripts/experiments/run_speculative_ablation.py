#!/usr/bin/env python3
"""Run the fixed llama.cpp speculative-decoding ablation on a remote Jetson."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation import run_llamacpp_inference_eval as inference_eval

JsonDict = dict[str, Any]

DEFAULT_QWEN_MODEL = (
    "/home/odine/models/llama.cpp/"
    "models--lmstudio-community--Qwen3.6-35B-A3B-GGUF/"
    "snapshots/68a34855558af61cbef0324d31f411be8a506b08/"
    "Qwen3.6-35B-A3B-Q4_K_M.gguf"
)
DEFAULT_GEMMA_MODEL = (
    "/home/odine/models/llama.cpp/"
    "models--lmstudio-community--gemma-4-26B-A4B-it-GGUF/"
    "snapshots/f11c185920b2c7b202e07e4a9768bf1bea0111d1/"
    "gemma-4-26B-A4B-it-Q4_K_M.gguf"
)
DEFAULT_SERVER_BINARY = "/home/odine/llama.cpp/build/bin/llama-server"
DEFAULT_CONDITIONS = "qwen_ngram,gemma_ngram,gemma_eagle3"


class ConditionFailure(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class Condition:
    name: str
    model_alias: str
    model_path: str
    spec_type: str
    spec_args: tuple[str, ...]
    baseline_subdir: str
    draft_model: str = ""


def _split_conditions(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--conditions must select at least one condition")
    return names


def build_conditions(
    names: Sequence[str],
    *,
    qwen_model: str,
    gemma_model: str,
    gemma_eagle_model: str,
) -> list[Condition]:
    definitions = {
        "qwen_ngram": Condition(
            name="qwen_ngram",
            model_alias="qwen3.6:35b-a3b",
            model_path=qwen_model,
            spec_type="ngram-mod",
            spec_args=(
                "--spec-type", "ngram-mod",
                "--spec-ngram-mod-n-match", "24",
                "--spec-ngram-mod-n-min", "48",
                "--spec-ngram-mod-n-max", "64",
            ),
            baseline_subdir="qwen_q8_0",
        ),
        "gemma_ngram": Condition(
            name="gemma_ngram",
            model_alias="gemma4:26b-a4b",
            model_path=gemma_model,
            spec_type="ngram-mod",
            spec_args=(
                "--spec-type", "ngram-mod",
                "--spec-ngram-mod-n-match", "24",
                "--spec-ngram-mod-n-min", "48",
                "--spec-ngram-mod-n-max", "64",
            ),
            baseline_subdir="gemma_q8_0",
        ),
        "gemma_eagle3": Condition(
            name="gemma_eagle3",
            model_alias="gemma4:26b-a4b",
            model_path=gemma_model,
            spec_type="draft-eagle3",
            spec_args=("--spec-type", "draft-eagle3", "--spec-draft-n-max", "3"),
            baseline_subdir="gemma_q8_0",
            draft_model=gemma_eagle_model,
        ),
    }
    unknown = [name for name in names if name not in definitions]
    if unknown:
        raise ValueError(f"Unknown condition(s): {', '.join(unknown)}")
    if "gemma_eagle3" in names and not gemma_eagle_model.strip():
        raise ValueError("--gemma-eagle-model is required for gemma_eagle3")
    return [definitions[name] for name in names]


def build_server_command(
    condition: Condition,
    *,
    server_binary: str,
    remote_port: int,
) -> list[str]:
    command = [
        server_binary,
        "--model", condition.model_path,
        "--alias", condition.model_alias,
        "--host", "127.0.0.1",
        "--port", str(remote_port),
        "--ctx-size", "32768",
        "--parallel", "1",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--no-cache-prompt",
    ]
    if condition.draft_model:
        command.extend(("--model-draft", condition.draft_model))
    command.extend(condition.spec_args)
    return command


_STATISTICS_RE = re.compile(
    r"statistics\s+(?P<kind>[\w-]+):.*?"
    r"#gen tokens\s*=\s*(?P<generated>\d+),\s*"
    r"#acc tokens\s*=\s*(?P<accepted>\d+)",
    re.IGNORECASE,
)
# Matches both wordings seen from llama.cpp:
#   "draft acceptance rate = 0.625 (25 accepted / 40 generated)"        (aggregate line)
#   "draft acceptance = 0.03125 (2 accepted / 64 generated), mean len.." (per-slot print_timing)
_ACCEPTANCE_RE = re.compile(
    r"draft acceptance(?:\s+rate)?\s*=\s*(?P<rate>[0-9.]+)\s*"
    r"\(\s*(?P<accepted>\d+) accepted\s*/\s*(?P<generated>\d+) generated\s*\)",
    re.IGNORECASE,
)


def parse_speculative_stats(log_text: str, *, expected_type: str) -> JsonDict:
    statistics = list(_STATISTICS_RE.finditer(log_text))
    acceptance = list(_ACCEPTANCE_RE.finditer(log_text))
    if statistics:
        match = statistics[-1]
        generated = int(match.group("generated"))
        accepted = int(match.group("accepted"))
        return {
            "spec_type": expected_type,
            "draft_generated_tokens": generated,
            "draft_accepted_tokens": accepted,
            "draft_acceptance_rate": accepted / generated if generated else None,
            "statistics_found": True,
        }
    if acceptance:
        match = acceptance[-1]
        return {
            "spec_type": expected_type,
            "draft_generated_tokens": int(match.group("generated")),
            "draft_accepted_tokens": int(match.group("accepted")),
            "draft_acceptance_rate": float(match.group("rate")),
            "statistics_found": True,
        }
    return {
        "spec_type": expected_type,
        "draft_generated_tokens": None,
        "draft_accepted_tokens": None,
        "draft_acceptance_rate": None,
        "statistics_found": False,
    }


def classify_server_start_failure(log_text: str) -> str:
    lowered = log_text.lower()
    if "cuda" in lowered and any(term in lowered for term in ("out of memory", "allocation", "alloc failed")):
        return "cuda_allocation_failed"
    if any(term in lowered for term in ("error loading model", "failed to load model", "model_load")):
        return "model_load_failed"
    if "unknown argument" in lowered or "unrecognized option" in lowered:
        return "unsupported_speculative_flag"
    return "health_timeout"


def validate_baseline_summary(
    path: Path,
    *,
    expected_model: str,
    expected_csv: Path,
) -> JsonDict:
    if not path.exists():
        raise FileNotFoundError(f"Baseline summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    expected = {
        "model": expected_model,
        "max_tokens": 512,
        "temperature": 0.0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Baseline {path} has incompatible {key}: {config.get(key)!r}; expected {value!r}"
            )
    configured_csv = Path(str(config.get("csv", "")))
    if configured_csv.name != expected_csv.name:
        raise ValueError(
            f"Baseline {path} has incompatible csv: {configured_csv}; expected {expected_csv}"
        )
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    if counts.get("total") != 150 or counts.get("successful") != 150:
        raise ValueError(f"Baseline {path} must contain 150 successful rows")
    return payload


def validate_baseline_server_command(command_text: str) -> None:
    tokens = shlex.split(command_text)
    required_pairs = (
        ("--ctx-size", "32768"),
        ("--parallel", "1"),
        ("--flash-attn", "on"),
        ("--cache-type-k", "q8_0"),
        ("--cache-type-v", "q8_0"),
    )
    for flag, expected in required_pairs:
        if flag not in tokens or tokens.index(flag) + 1 >= len(tokens) or tokens[tokens.index(flag) + 1] != expected:
            raise ValueError(f"Baseline server command must contain {flag} {expected}")
    for flag in ("--no-cache-prompt",):
        if flag not in tokens:
            raise ValueError(f"Baseline server command must contain {flag}")
    if not any("Q4_K_M.gguf" in token for token in tokens):
        raise ValueError("Baseline server command must use Q4_K_M.gguf target weights")
    if "--spec-type" in tokens or "--model-draft" in tokens:
        raise ValueError("Baseline server command must not enable speculative decoding")


def validate_baseline_quality(summary: Mapping[str, Any]) -> None:
    evaluation = summary.get("evaluation")
    successful = evaluation.get("successful") if isinstance(evaluation, dict) else None
    if successful != 150:
        raise ValueError(
            "Baseline must contain 150 successful quality evaluations; "
            f"found {successful!r}"
        )


def build_owned_pid_stop_script(*, pid_file: str, expected_binary: str) -> str:
    quoted_pid = shlex.quote(pid_file)
    quoted_binary = shlex.quote(expected_binary)
    return (
        f"test -f {quoted_pid} || exit 0\n"
        f"pid=$(tr -dc '0-9' < {quoted_pid})\n"
        "test -n \"$pid\" || exit 41\n"
        "test -d \"/proc/$pid\" || exit 0\n"
        "actual=$(readlink -f \"/proc/$pid/exe\")\n"
        f"expected=$(readlink -f {quoted_binary})\n"
        "test \"$actual\" = \"$expected\" || exit 42\n"
        "kill -- \"$pid\"\n"
        "i=0; while kill -0 \"$pid\" 2>/dev/null && test \"$i\" -lt 30; do "
        "i=$((i+1)); sleep 1; done\n"
        "kill -0 \"$pid\" 2>/dev/null && exit 43 || true\n"
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class RemoteJetson:
    def __init__(self, host: str, *, local_port: int, remote_port: int, control_path: str = "") -> None:
        self.host = host
        self.local_port = local_port
        self.remote_port = remote_port
        self.control_path = control_path
        self.tunnel: subprocess.Popen[bytes] | None = None
        self.control_forward_active = False

    def _ssh_base(self) -> list[str]:
        command = ["ssh"]
        if self.control_path:
            command.extend(("-S", self.control_path))
        return command

    def run(self, script: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                *self._ssh_base(), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                self.host, "bash", "-s",
            ],
            input=script,
            text=True,
            capture_output=True,
            check=check,
        )

    def start_tunnel(self) -> None:
        if self.control_path:
            result = subprocess.run(
                [*self._ssh_base(), "-O", "forward", "-L", f"{self.local_port}:127.0.0.1:{self.remote_port}", self.host],
                capture_output=True,
            )
            if result.returncode:
                raise RuntimeError(f"SSH tunnel failed: {result.stderr.decode(errors='replace').strip()}")
            self.control_forward_active = True
            return
        self.tunnel = subprocess.Popen(
            [
                *self._ssh_base(), "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                "-L", f"{self.local_port}:127.0.0.1:{self.remote_port}", self.host,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.5)
        if self.tunnel.poll() is not None:
            detail = (self.tunnel.stderr.read() if self.tunnel.stderr else b"").decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(f"SSH tunnel failed: {detail.strip()}")

    def stop_tunnel(self) -> None:
        if self.control_path and self.control_forward_active:
            subprocess.run(
                [*self._ssh_base(), "-O", "cancel", "-L", f"{self.local_port}:127.0.0.1:{self.remote_port}", self.host],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            self.control_forward_active = False
            return
        if self.tunnel is None or self.tunnel.poll() is not None:
            self.tunnel = None
            return
        self.tunnel.send_signal(signal.SIGTERM)
        try:
            self.tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.tunnel.kill()
            self.tunnel.wait(timeout=5)
        self.tunnel = None


def _remote_preflight(
    remote: RemoteJetson,
    condition: Condition,
    *,
    server_binary: str,
    remote_port: int,
) -> None:
    if remote.run(f"test -x {shlex.quote(server_binary)}\n", check=False).returncode:
        raise ConditionFailure("server_binary_missing", f"Server binary is not executable: {server_binary}")
    for path in (condition.model_path, condition.draft_model):
        if path and remote.run(f"test -r {shlex.quote(path)}\n", check=False).returncode:
            raise ConditionFailure("model_missing", f"Remote model is not readable: {path}")
    if remote.run(
        f"! pgrep -f -- {shlex.quote(server_binary)} >/dev/null\n", check=False
    ).returncode:
        raise ConditionFailure("unrecognized_server_active", "A llama-server process is already active")
    if remote.run(
        f"! ss -ltnH 'sport = :{remote_port}' | grep -q .\n", check=False
    ).returncode:
        raise ConditionFailure("remote_port_busy", f"Remote port {remote_port} is already in use")


def _remote_start_server(
    remote: RemoteJetson,
    command: Sequence[str],
    *,
    remote_dir: str,
    log_name: str,
) -> str:
    pid_file = f"{remote_dir}/server.pid"
    log_file = f"{remote_dir}/{log_name}"
    command_text = shlex.join(command)
    script = (
        f"mkdir -p {shlex.quote(remote_dir)}\n"
        f"printf '%s\\n' {shlex.quote(command_text)} > {shlex.quote(remote_dir + '/server_command.txt')}\n"
        f"nohup {command_text} > {shlex.quote(log_file)} 2>&1 &\n"
        "pid=$!\n"
        f"printf '%s\\n' \"$pid\" > {shlex.quote(pid_file)}\n"
        "printf '%s\\n' \"$pid\"\n"
    )
    result = remote.run(script)
    pid = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not pid.isdigit():
        raise RuntimeError(f"Remote server did not return a PID: {result.stdout!r}")
    return pid


def _remote_stop_server(
    remote: RemoteJetson, *, remote_dir: str, server_binary: str
) -> None:
    remote.run(
        build_owned_pid_stop_script(
            pid_file=f"{remote_dir}/server.pid", expected_binary=server_binary
        )
    )


def _fetch_remote_text(remote: RemoteJetson, remote_path: str) -> str:
    result = remote.run(f"test -f {shlex.quote(remote_path)}\ncat {shlex.quote(remote_path)}\n")
    return result.stdout


def _wait_for_health(base_url: str, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            inference_eval.check_health(base_url, 2.0)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"llama.cpp health timeout after {timeout_seconds}s: {last_error}")


def _make_smoke_csv(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [row for _, row in zip(range(3), reader)]
    if len(rows) != 3:
        raise ValueError("Input CSV must contain at least three rows")
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_inference_client(
    *,
    csv_path: Path,
    answer_col: str,
    base_url: str,
    condition: Condition,
    out_dir: Path,
    resume: bool,
    skip_eval: bool,
    eval_model: str,
) -> None:
    argv = [
        "--csv", str(csv_path),
        "--answer-col", answer_col,
        "--base-url", base_url,
        "--model", condition.model_alias,
        "--eval-model", eval_model,
        "--run-label", (
            f"{condition.name}_q4km_kv-q8_0_ctx32768_fa-on_"
            "prompt-cache-off_qa150"
        ),
        "--out-dir", str(out_dir),
        "--temperature", "0",
        "--max-tokens", "512",
        "--timeout", "300",
    ]
    if resume:
        argv.append("--resume")
    if skip_eval:
        argv.append("--skip-eval")
    exit_code = inference_eval.execute_run(inference_eval.parse_args(argv))
    if exit_code != 0:
        raise RuntimeError(f"Inference client failed with exit code {exit_code}")


def _summary_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    return counts.get("total") == 150 and counts.get("successful") == 150


def condition_is_resumable(condition_dir: Path) -> bool:
    stats_path = condition_dir / "speculative_stats.json"
    if not _summary_is_complete(condition_dir / "summary.json") or not stats_path.exists():
        return False
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return stats.get("statistics_found") is True


def _metric(summary: Mapping[str, Any], field: str, statistic: str) -> float | None:
    section = summary.get("inference")
    if not isinstance(section, dict) or not isinstance(section.get(field), dict):
        return None
    value = section[field].get(statistic)
    return float(value) if value is not None else None


def _quality(summary: Mapping[str, Any], field: str) -> float | None:
    evaluation = summary.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    value = evaluation.get(field)
    return float(value) if value is not None else None


def _relative_improvement(value: float | None, baseline: float | None, *, lower_is_better: bool) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return ((baseline - value) / baseline if lower_is_better else (value - baseline) / baseline) * 100


COMPARISON_FIELDS = (
    "condition", "baseline_subdir", "spec_type", "status",
    "baseline_latency_median", "spec_latency_median", "latency_improvement_percent",
    "baseline_generation_tps_median", "spec_generation_tps_median", "tps_improvement_percent",
    "baseline_correctness_mean", "spec_correctness_mean",
    "baseline_completeness_mean", "spec_completeness_mean",
    "draft_generated_tokens", "draft_accepted_tokens", "draft_acceptance_rate",
    "baseline_truncated", "spec_truncated", "truncation_not_degraded",
    "speed_threshold_met", "quality_not_degraded", "successful",
)


def build_comparison_row(
    condition: Condition,
    *,
    baseline: Mapping[str, Any],
    result: Mapping[str, Any],
    stats: Mapping[str, Any],
    status: str,
) -> JsonDict:
    baseline_latency = _metric(baseline, "wall_time_seconds", "median")
    result_latency = _metric(result, "wall_time_seconds", "median")
    baseline_tps = _metric(baseline, "generation_tokens_per_second", "median")
    result_tps = _metric(result, "generation_tokens_per_second", "median")
    latency_gain = _relative_improvement(result_latency, baseline_latency, lower_is_better=True)
    tps_gain = _relative_improvement(result_tps, baseline_tps, lower_is_better=False)
    baseline_correctness = _quality(baseline, "correctness_score_mean")
    result_correctness = _quality(result, "correctness_score_mean")
    baseline_completeness = _quality(baseline, "completeness_score_mean")
    result_completeness = _quality(result, "completeness_score_mean")
    quality_available = None not in (
        baseline_correctness, result_correctness, baseline_completeness, result_completeness
    )
    quality_ok = (
        result_correctness >= baseline_correctness and result_completeness >= baseline_completeness
        if quality_available else None
    )
    baseline_counts = baseline.get("counts") if isinstance(baseline.get("counts"), dict) else {}
    result_counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
    baseline_truncated = int(baseline_counts.get("truncated", 0))
    result_truncated = int(result_counts.get("truncated", 0))
    truncation_ok = result_truncated <= baseline_truncated
    speed_ok = any(gain is not None and gain >= 10.0 for gain in (latency_gain, tps_gain))
    return {
        "condition": condition.name,
        "baseline_subdir": condition.baseline_subdir,
        "spec_type": condition.spec_type,
        "status": status,
        "baseline_latency_median": baseline_latency,
        "spec_latency_median": result_latency,
        "latency_improvement_percent": latency_gain,
        "baseline_generation_tps_median": baseline_tps,
        "spec_generation_tps_median": result_tps,
        "tps_improvement_percent": tps_gain,
        "baseline_correctness_mean": baseline_correctness,
        "spec_correctness_mean": result_correctness,
        "baseline_completeness_mean": baseline_completeness,
        "spec_completeness_mean": result_completeness,
        "draft_generated_tokens": stats.get("draft_generated_tokens"),
        "draft_accepted_tokens": stats.get("draft_accepted_tokens"),
        "draft_acceptance_rate": stats.get("draft_acceptance_rate"),
        "baseline_truncated": baseline_truncated,
        "spec_truncated": result_truncated,
        "truncation_not_degraded": truncation_ok,
        "speed_threshold_met": speed_ok,
        "quality_not_degraded": quality_ok,
        "successful": bool(speed_ok and quality_ok is True and truncation_ok),
    }


def _with_truncation_count(summary: JsonDict, csv_path: Path) -> JsonDict:
    rows = inference_eval.read_csv_rows(csv_path)
    result = dict(summary)
    counts = dict(result.get("counts") or {})
    counts["truncated"] = sum(
        str(row.get("finish_reason", "")).strip().lower() in {"length", "max_tokens"}
        for row in rows
    )
    result["counts"] = counts
    return result


def _write_comparison(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="odine@192.168.1.29")
    parser.add_argument("--ssh-control-path", default="")
    parser.add_argument("--csv", default=str(REPO_ROOT / "data/qa/final_qa_150.csv"))
    parser.add_argument("--answer-col", default="answer")
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--gemma-eagle-model", default="")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--server-binary", default=DEFAULT_SERVER_BINARY)
    parser.add_argument(
        "--baseline-root",
        default=str(REPO_ROOT / "data/results/kv_ablation_20260721_150"),
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--remote-root", default="/home/odine/speculative_ablation")
    parser.add_argument("--local-port", type=int, default=18084)
    parser.add_argument("--remote-port", type=int, default=8084)
    parser.add_argument("--health-timeout", type=float, default=300.0)
    parser.add_argument("--eval-model", default="gpt-5.4-mini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    questions = inference_eval.load_questions(csv_path, "question", args.answer_col)
    if len(questions) != 150:
        raise ValueError(f"Speculative ablation requires exactly 150 questions; found {len(questions)}")
    if not args.skip_eval and not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required unless --skip-eval is used")

    conditions = build_conditions(
        _split_conditions(args.conditions),
        qwen_model=args.qwen_model,
        gemma_model=args.gemma_model,
        gemma_eagle_model=args.gemma_eagle_model,
    )
    baseline_root = Path(args.baseline_root)
    baselines: dict[str, JsonDict] = {}
    for condition in conditions:
        baseline_dir = baseline_root / condition.baseline_subdir
        baseline = validate_baseline_summary(
            baseline_dir / "summary.json", expected_model=condition.model_alias,
            expected_csv=csv_path,
        )
        command_path = baseline_dir / "server_command.txt"
        if not command_path.exists():
            raise FileNotFoundError(f"Baseline server command not found: {command_path}")
        validate_baseline_server_command(command_path.read_text(encoding="utf-8"))
        if not args.skip_eval:
            validate_baseline_quality(baseline)
        baselines[condition.name] = _with_truncation_count(
            baseline, baseline_dir / "raw_results.csv"
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "data/results" / f"speculative_ablation_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest: JsonDict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "csv": str(csv_path),
        "baseline_root": str(baseline_root),
        "conditions": {condition.name: {"status": "pending"} for condition in conditions},
    }
    if args.resume and manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            manifest = loaded
    _write_json(manifest_path, manifest)

    remote = RemoteJetson(
        args.ssh_host, local_port=args.local_port, remote_port=args.remote_port,
        control_path=args.ssh_control_path,
    )
    base_url = f"http://127.0.0.1:{args.local_port}/v1"
    comparison_rows: list[JsonDict] = []

    for condition in conditions:
        condition_dir = out_dir / condition.name
        condition_dir.mkdir(parents=True, exist_ok=True)
        result_summary_path = condition_dir / "summary.json"
        stats_path = condition_dir / "speculative_stats.json"
        if args.resume and condition_is_resumable(condition_dir):
            result_summary = _with_truncation_count(
                json.loads(result_summary_path.read_text(encoding="utf-8")),
                condition_dir / "raw_results.csv",
            )
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            comparison_rows.append(
                build_comparison_row(
                    condition, baseline=baselines[condition.name], result=result_summary,
                    stats=stats, status="completed",
                )
            )
            continue

        remote_dir = f"{args.remote_root.rstrip('/')}/{condition.name}"
        command = build_server_command(
            condition, server_binary=args.server_binary, remote_port=args.remote_port
        )
        (condition_dir / "server_command.txt").write_text(
            shlex.join(command) + "\n", encoding="utf-8"
        )
        manifest["conditions"][condition.name] = {
            "status": "running", "config": asdict(condition)
        }
        _write_json(manifest_path, manifest)
        server_started = False
        active_log_name = "server_smoke.log"
        try:
            _remote_preflight(
                remote, condition, server_binary=args.server_binary, remote_port=args.remote_port
            )
            _remote_start_server(
                remote, command, remote_dir=remote_dir, log_name="server_smoke.log"
            )
            server_started = True
            remote.start_tunnel()
            _wait_for_health(base_url, timeout_seconds=args.health_timeout)
            with tempfile.TemporaryDirectory(prefix="speculative-smoke-") as tmp:
                smoke_csv = Path(tmp) / "smoke.csv"
                smoke_out = Path(tmp) / "results"
                _make_smoke_csv(csv_path, smoke_csv)
                _run_inference_client(
                    csv_path=smoke_csv, answer_col=args.answer_col, base_url=base_url,
                    condition=condition, out_dir=smoke_out, resume=False,
                    skip_eval=True, eval_model=args.eval_model,
                )
                if not _summary_is_complete_for_count(smoke_out / "summary.json", 3):
                    raise RuntimeError(f"Smoke test failed for {condition.name}")
            remote.stop_tunnel()
            _remote_stop_server(remote, remote_dir=remote_dir, server_binary=args.server_binary)
            server_started = False
            smoke_log = _fetch_remote_text(remote, f"{remote_dir}/server_smoke.log")
            (condition_dir / "server_smoke.log").write_text(smoke_log, encoding="utf-8")

            full_pid = _remote_start_server(
                remote, command, remote_dir=remote_dir, log_name="server.log"
            )
            active_log_name = "server.log"
            (condition_dir / "server.pid").write_text(full_pid + "\n", encoding="utf-8")
            server_started = True
            remote.start_tunnel()
            _wait_for_health(base_url, timeout_seconds=args.health_timeout)
            _run_inference_client(
                csv_path=csv_path, answer_col=args.answer_col, base_url=base_url,
                condition=condition, out_dir=condition_dir, resume=args.resume,
                skip_eval=args.skip_eval, eval_model=args.eval_model,
            )
            remote.stop_tunnel()
            _remote_stop_server(remote, remote_dir=remote_dir, server_binary=args.server_binary)
            server_started = False
            server_log = _fetch_remote_text(remote, f"{remote_dir}/server.log")
            (condition_dir / "server.log").write_text(server_log, encoding="utf-8")
            stats = parse_speculative_stats(server_log, expected_type=condition.spec_type)
            _write_json(stats_path, stats)
            if not stats["statistics_found"]:
                raise RuntimeError(f"Speculative statistics missing for {condition.name}")
            result_summary = _with_truncation_count(
                json.loads(result_summary_path.read_text(encoding="utf-8")),
                condition_dir / "raw_results.csv",
            )
            manifest["conditions"][condition.name]["status"] = "completed"
            comparison_rows.append(
                build_comparison_row(
                    condition, baseline=baselines[condition.name], result=result_summary,
                    stats=stats, status="completed",
                )
            )
        except Exception as exc:
            category = exc.category if isinstance(exc, ConditionFailure) else type(exc).__name__
            if isinstance(exc, TimeoutError) and server_started:
                try:
                    startup_log = _fetch_remote_text(remote, f"{remote_dir}/{active_log_name}")
                    (condition_dir / active_log_name).write_text(startup_log, encoding="utf-8")
                    category = classify_server_start_failure(startup_log)
                except Exception:
                    category = "health_timeout"
            manifest["conditions"][condition.name] = {
                **manifest["conditions"].get(condition.name, {}),
                "status": "failed",
                "error_type": category,
                "error_message": str(exc),
            }
            result_summary: JsonDict = {}
            stats: JsonDict = {}
            if result_summary_path.exists():
                result_summary = _with_truncation_count(
                    json.loads(result_summary_path.read_text(encoding="utf-8")),
                    condition_dir / "raw_results.csv",
                )
            if stats_path.exists():
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
            comparison_rows.append(
                build_comparison_row(
                    condition, baseline=baselines[condition.name], result=result_summary,
                    stats=stats, status="failed",
                )
            )
        finally:
            remote.stop_tunnel()
            if server_started:
                try:
                    _remote_stop_server(
                        remote, remote_dir=remote_dir, server_binary=args.server_binary
                    )
                except Exception as stop_exc:
                    manifest["conditions"][condition.name]["stop_error"] = str(stop_exc)
            _write_json(manifest_path, manifest)

    _write_comparison(out_dir / "comparison.csv", comparison_rows)
    failed = sum(
        manifest["conditions"].get(condition.name, {}).get("status") == "failed"
        for condition in conditions
    )
    return 1 if failed else 0


def _summary_is_complete_for_count(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    return counts.get("total") == expected and counts.get("successful") == expected


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return execute(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
