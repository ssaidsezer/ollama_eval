import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "generate_consolidated_report", ROOT / "scripts/reporting/generate_consolidated_report.py"
)
ccr = importlib.util.module_from_spec(spec)
sys.modules["generate_consolidated_report"] = ccr
spec.loader.exec_module(ccr)

from scripts.reporting import report_utils  # noqa: E402


def test_unified_quality_averages_three_subscores():
    q = {"correctness_score_mean": 6.0, "completeness_score_mean": 3.0, "groundedness_score_mean": 3.0}
    assert ccr.unified_quality(q) == 4.0


def test_pass_rate_from_verdicts():
    q = {"verdicts": {"pass": 25, "partial": 50, "fail": 25}}
    assert ccr.pass_rate(q) == 0.25


def test_pass_rate_empty_verdicts_returns_zero():
    assert ccr.pass_rate({}) == 0.0


def _rows():
    return [
        {"name": "Gemma f16", "quality_score": 3.71, "pass_rate": 0.1867, "latency": 3.79,
         "latency_median": 3.41, "p95": 7.1, "tps": 34.4},
        {"name": "Qwen f16", "quality_score": 3.08, "pass_rate": 0.1133, "latency": 3.28,
         "latency_median": 2.95, "p95": 5.0, "tps": 37.3},
    ]


def test_svg_quality_bars_is_valid_svg_with_all_conditions():
    svg = ccr.svg_quality_bars(_rows(), "Kalite")
    assert svg.strip().startswith("<svg")
    assert svg.strip().endswith("</svg>")
    assert "Gemma f16" in svg and "Qwen f16" in svg
    assert "Kalite" in svg


def test_svg_latency_is_valid_svg_with_all_conditions():
    svg = ccr.svg_latency(_rows(), "Süre")
    assert svg.strip().startswith("<svg") and svg.strip().endswith("</svg>")
    assert "Gemma f16" in svg and "Qwen f16" in svg


def test_render_html_contains_all_sections():
    kv = _rows(); cv = _rows(); rag = _rows()
    retrieval = {"classic_vector": {"median_seconds": 0.64}}
    html = ccr.render_html(kv, cv, rag, retrieval)
    assert "<html" in html.lower()
    for needle in ["KV Cache", "RAG", "Yönetici Özeti", "Sabit Protokol", "<svg"]:
        assert needle in html, needle


def test_svg_latency_plots_the_median_not_the_mean():
    """Sections 1b/2e are titled "Medyan Yanıt Süresi" — the bar label must be the median."""
    svg = ccr.svg_latency(_rows(), "Medyan Yanıt Süresi")
    assert "3.41s" in svg and "2.95s" in svg
    assert "3.79s" not in svg and "3.28s" not in svg


def test_quality_table_reports_both_median_and_mean():
    table = ccr._quality_table(_rows())
    assert "Medyan sn" in table and "Ort. sn" in table
    assert "<td>3.41</td><td>3.79</td>" in table


def test_p95_handles_degenerate_inputs():
    assert report_utils.p95([4.2]) == 4.2
    assert report_utils.p95([]) is None


def test_p95_interpolation_is_unchanged_for_published_runs():
    assert report_utils.p95([1.0, 2.0]) == 1.95
    assert report_utils.p95([1.0, 2.0, 3.0]) == pytest.approx(2.9)


def test_condition_reports_mean_and_median_separately(tmp_path):
    """"latency" stays the MEAN — the archived report labels it "Ort. sn"."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "raw_results.csv").write_text(
        "question_index;inference_status;response_time_seconds;tokens_per_second\n"
        "0;ok;1.0;10\n1;ok;2.0;20\n2;ok;30.0;30\n",
        encoding="utf-8",
    )
    data = report_utils.condition(run)
    assert data["latency"] == pytest.approx(11.0)
    assert data["latency_median"] == pytest.approx(2.0)


def test_condition_skips_blank_numeric_cells(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "raw_results.csv").write_text(
        "question_index;inference_status;response_time_seconds;tokens_per_second\n"
        "0;ok;1.0;10\n1;ok;3.0;\n",
        encoding="utf-8",
    )
    data = report_utils.condition(run)
    assert data["tps"] == pytest.approx(10.0)
    assert data["latency_median"] == pytest.approx(2.0)


def test_condition_rejects_a_systematically_unparseable_column(tmp_path):
    """Decimal commas must not yield a mean over the few cells that happened to parse."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "raw_results.csv").write_text(
        "question_index;inference_status;response_time_seconds;tokens_per_second\n"
        "0;ok;1,0;10\n1;ok;2,5;20\n2;ok;3.0;30\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="are not numeric"):
        report_utils.condition(run)


def test_condition_tolerates_one_stray_unparseable_cell(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = "".join(f"{i};ok;1.0;10\n" for i in range(19))
    (run / "raw_results.csv").write_text(
        "question_index;inference_status;response_time_seconds;tokens_per_second\n"
        + rows
        + "19;ok;n/a;10\n",
        encoding="utf-8",
    )
    data = report_utils.condition(run)
    assert data["latency"] == pytest.approx(1.0)


def test_condition_rejects_an_empty_results_csv(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "raw_results.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No result rows"):
        report_utils.condition(run)
