"""Shared result-loading helpers for evaluator report generators."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def sniff_delimiter(sample: str) -> str:
    """Pick the delimiter from a header line: semicolon if it dominates, else comma."""
    return ";" if sample.count(";") > sample.count(",") else ","


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a result CSV, detecting comma or semicolon delimiters."""
    with path.open(encoding="utf-8-sig") as handle:
        delimiter = sniff_delimiter(handle.readline())
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def p95(values: list[float]) -> float | None:
    """Return the interpolated 95th percentile used by the original reports."""
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        # pos would be 0 and both interpolation weights would collapse to zero.
        return values[0]
    pos = (len(values) - 1) * 0.95
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


# A stray unparseable cell is tolerable; a systematic one (decimal commas, a delimiter
# sniffed wrong) would otherwise average the few cells that happened to parse and report
# the result as fact.
_MAX_UNPARSEABLE_FRACTION = 0.1


def _numbers(rows: list[dict[str, str]], key: str, *, source: object = "results") -> list[float]:
    """Parse one column to float, skipping blanks.

    Unparseable non-blank cells are skipped too, but raise once they exceed
    `_MAX_UNPARSEABLE_FRACTION` of the populated cells.
    """
    values: list[float] = []
    skipped: list[str] = []
    for row in rows:
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            skipped.append(str(raw))
    populated = len(values) + len(skipped)
    if skipped and len(skipped) > _MAX_UNPARSEABLE_FRACTION * populated:
        raise ValueError(
            f"{len(skipped)}/{populated} values in column {key!r} of {source} are not numeric "
            f"(e.g. {skipped[0]!r}); refusing to report a statistic over the remainder"
        )
    return values


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def condition(path: Path) -> dict[str, object]:
    """Load one experiment condition without changing the legacy schema."""
    results_name = "merged_results.csv" if (path / "merged_results.csv").exists() else "raw_results.csv"
    results_path = path / results_name
    rows = read_rows(results_path)
    if not rows:
        raise ValueError(f"No result rows in {results_path}")
    successful_rows = [row for row in rows if row.get("inference_status") in {"success", "ok"}]
    summary_path = path / "summary_balanced_v2.json"
    quality = json.loads(summary_path.read_text(encoding="utf-8"))["evaluation"] if summary_path.exists() else None
    latency_key = "response_time_seconds" if "response_time_seconds" in rows[0] else "wall_time_seconds"
    tps_key = "tokens_per_second" if "tokens_per_second" in rows[0] else "generation_tokens_per_second"
    latencies = _numbers(successful_rows, latency_key, source=results_path)
    return {
        "rows": rows,
        "n": len(successful_rows),
        "total": len(rows),
        # "latency" is the MEAN and is labelled "Ort. sn" by existing reports; do not repurpose it.
        "latency": _mean(latencies),
        "latency_median": statistics.median(latencies) if latencies else None,
        "p95": p95(latencies),
        "tps": _mean(_numbers(successful_rows, tps_key, source=results_path)),
        "quality": quality,
        "chunks": _mean(_numbers(successful_rows, "retrieved_chunk_size", source=results_path))
        if "retrieved_chunk_size" in rows[0]
        else None,
    }
