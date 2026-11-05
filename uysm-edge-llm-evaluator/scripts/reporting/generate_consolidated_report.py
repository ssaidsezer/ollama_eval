"""Konsolide UYSM değerlendirme raporu: KV cache + RAG ablasyonu, tek PDF."""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reporting.report_utils import condition

# ── veri kaynakları ────────────────────────────────────────────────────────
KV_ABLATION = ROOT / "data/results/kv_ablation_20260721_150"
KV_RAG_CV = ROOT / "data/results/kv_rag_classic_vector_20260722_150"
RAG_METHODS = ROOT / "data/results/rag_methods_gemma_q4_20260722_150"
CLASSIC_BM25 = ROOT / "data/results/kv_rag_classic_bm25_20260722_150"
RETRIEVAL_TIMING = RAG_METHODS / "retrieval_timing/summary.json"

KV_LABELS = [
    ("Gemma f16", "gemma_f16"), ("Gemma q8_0", "gemma_q8_0"), ("Gemma q4_0", "gemma_q4_0"),
    ("Qwen f16", "qwen_f16"), ("Qwen q8_0", "qwen_q8_0"), ("Qwen q4_0", "qwen_q4_0"),
]
RAG_METHOD_PATHS = {
    "Classic Vector": KV_RAG_CV / "gemma_q4_0",
    "Classic BM25": CLASSIC_BM25 / "gemma_q4_0",
    "Smart Vector": RAG_METHODS / "smart_vector",
    "Smart BM25": RAG_METHODS / "smart_bm25",
}

BLUE, GREEN, AMBER = "#426b8f", "#2f8f68", "#d99120"
INK, MUTE, GRID = "#24292f", "#6b7681", "#e4e8ec"


# ── skor / normalizasyon ───────────────────────────────────────────────────
def unified_quality(quality: dict) -> float:
    return (
        quality["correctness_score_mean"]
        + quality["completeness_score_mean"]
        + quality["groundedness_score_mean"]
    ) / 3


def pass_rate(quality: dict) -> float:
    verdicts = quality.get("verdicts") or {}
    total = sum(verdicts.values())
    return verdicts.get("pass", 0) / total if total else 0.0


def load_row(name: str, path: Path) -> dict:
    data = condition(path)
    q = data["quality"] or {}
    return {
        "name": name,
        "quality_score": unified_quality(q) if q else None,
        "correctness": q.get("correctness_score_mean"),
        "completeness": q.get("completeness_score_mean"),
        "groundedness": q.get("groundedness_score_mean"),
        "pass_rate": pass_rate(q),
        "latency": data["latency"],
        "latency_median": data["latency_median"],
        "p95": data["p95"],
        "tps": data["tps"],
        "chunks": data["chunks"],
    }


# ── inline SVG grafikler ───────────────────────────────────────────────────
def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="-apple-system,Segoe UI,Arial,sans-serif">',
        f'<text x="0" y="22" font-size="15" font-weight="700" fill="{INK}">{escape(title)}</text>',
    ]


def svg_quality_bars(rows: list[dict], title: str, width: int = 720) -> str:
    row_h, top, left, maxv = 34, 44, 150, 10.0
    plot_w = width - left - 60
    height = top + row_h * len(rows) + 20
    out = _svg_header(width, height, title)
    for i, r in enumerate(rows):
        y = top + i * row_h
        val = r["quality_score"] or 0.0
        bar = plot_w * (val / maxv)
        out.append(f'<text x="{left-10}" y="{y+18}" text-anchor="end" font-size="12" fill="{INK}">{escape(r["name"])}</text>')
        out.append(f'<rect x="{left}" y="{y+6}" width="{plot_w}" height="16" rx="3" fill="{GRID}"/>')
        out.append(f'<rect x="{left}" y="{y+6}" width="{bar:.1f}" height="16" rx="3" fill="{BLUE}"/>')
        out.append(f'<text x="{left+bar+8:.1f}" y="{y+18}" font-size="11" font-weight="600" fill="{INK}">{val:.2f}</text>')
        out.append(f'<text x="{width-4}" y="{y+18}" text-anchor="end" font-size="10" fill="{MUTE}">pass {r["pass_rate"]*100:.0f}%</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_latency(rows: list[dict], title: str, width: int = 720) -> str:
    row_h, top, left = 34, 44, 150
    plot_w = width - left - 90
    maxv = max((r["p95"] or r.get("latency_median") or r["latency"] or 0) for r in rows) * 1.1 or 1.0
    height = top + row_h * len(rows) + 20
    out = _svg_header(width, height, title)
    for i, r in enumerate(rows):
        y = top + i * row_h
        # The bar is the MEDIAN, matching the chart titles; the mean is in the reference table.
        lat = r.get("latency_median") or 0.0
        bar = plot_w * (lat / maxv)
        p95x = left + plot_w * ((r["p95"] or 0.0) / maxv)
        out.append(f'<text x="{left-10}" y="{y+18}" text-anchor="end" font-size="12" fill="{INK}">{escape(r["name"])}</text>')
        out.append(f'<rect x="{left}" y="{y+6}" width="{plot_w}" height="16" rx="3" fill="{GRID}"/>')
        out.append(f'<rect x="{left}" y="{y+6}" width="{bar:.1f}" height="16" rx="3" fill="{GREEN}"/>')
        out.append(f'<circle cx="{p95x:.1f}" cy="{y+14}" r="4" fill="{AMBER}"/>')
        out.append(f'<text x="{left+bar+8:.1f}" y="{y+18}" font-size="11" fill="{INK}">{lat:.2f}s</text>')
        out.append(f'<text x="{width-4}" y="{y+18}" text-anchor="end" font-size="10" fill="{MUTE}">{(r["tps"] or 0.0):.1f} tok/s</text>')
    out.append("</svg>")
    return "\n".join(out)


# ── HTML + PDF ─────────────────────────────────────────────────────────────
CSS = """
@page { size: A4 landscape; margin: 14mm 16mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', Arial, sans-serif; color: #24292f; font-size: 12px; line-height: 1.5; }
h1 { font-size: 21px; margin: 0 0 4px; }
h2 { font-size: 15px; color: #426b8f; margin: 22px 0 8px; border-bottom: 1px solid #e4e8ec; padding-bottom: 4px; }
.sub { color: #6b7681; font-size: 12px; margin-bottom: 18px; }
.callout { background: #f3f6f9; border-left: 3px solid #426b8f; padding: 10px 14px; border-radius: 4px; margin: 10px 0; }
.section { page-break-inside: avoid; margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 11px; margin: 8px 0; }
th, td { border: 1px solid #d4dae0; padding: 5px 8px; text-align: left; }
th { background: #dfe9f2; font-weight: 700; }
tr:nth-child(even) td { background: #f6f8fa; }
.cover { text-align: center; padding-top: 60mm; page-break-after: always; }
.cover h1 { font-size: 30px; }
svg { width: 100%; height: auto; }
.pb { page-break-before: always; }
"""


def _quality_table(rows: list[dict]) -> str:
    head = (
        "<tr><th>Koşul</th><th>Kalite (0-10)</th><th>Doğruluk</th><th>Eksiksizlik</th>"
        "<th>Dayanaklılık</th><th>Pass %</th><th>Medyan sn</th><th>Ort. sn</th></tr>"
    )
    body = "".join(
        f"<tr><td>{escape(r['name'])}</td><td>{(r['quality_score'] or 0):.2f}</td>"
        f"<td>{(r.get('correctness') or 0):.2f}</td><td>{(r.get('completeness') or 0):.2f}</td>"
        f"<td>{(r.get('groundedness') or 0):.2f}</td><td>{r['pass_rate']*100:.0f}%</td>"
        f"<td>{(r.get('latency_median') or 0):.2f}</td><td>{(r.get('latency') or 0):.2f}</td></tr>"
        for r in rows
    )
    return f"<table>{head}{body}</table>"


def render_html(kv_rows, cv_rows, rag_rows, retrieval) -> str:
    ret_rows = "".join(
        f"<tr><td>{escape(k)}</td><td>{v.get('median_seconds', 0):.3f}s</td></tr>"
        for k, v in retrieval.items()
    )
    return f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<div class="cover"><h1>UYSM Değerlendirme Raporu</h1>
<p class="sub">KV Cache Ablasyonu &amp; RAG Ablasyonu · 150 soru · gpt-5.4-mini balanced-v2 jüri</p></div>

<h1>Yönetici Özeti</h1>
<p class="callout">Bu rapor iki deney bloğunu tek birleşik kalite skoru (doğruluk, eksiksizlik ve
dayanaklılığın 0-10 ortalaması) üzerinden karşılaştırır. Süre ve kalite ayrı görsellerde sunulur.
KV cache tipleri parallel=1 protokolünde hız avantajı üretmedi; RAG bağlamı kaliteyi belirgin
biçimde yükseltti. Speculative decoding bu rapora dahil değildir (veri eksik).</p>

<h2>Sabit Protokol</h2>
<table><tr><th>Bileşen</th><th>Değer</th></tr>
<tr><td>Modeller</td><td>Gemma4 26B-A4B ve Qwen3.6 35B-A3B; Q4_K_M</td></tr>
<tr><td>KV cache</td><td>Simetrik K/V: f16, q8_0, q4_0</td></tr>
<tr><td>RAG</td><td>Classic/Smart × Vector/BM25; embeddinggemma:300m; k=10; threshold=0.55</td></tr>
<tr><td>Üretim</td><td>ctx 32768; parallel 1; Flash Attention açık; prompt cache kapalı; temperature 0; max 512</td></tr>
<tr><td>Jüri</td><td>gpt-5.4-mini; reasoning_effort=high; balanced-v2</td></tr>
<tr><td>Veri</td><td>final_qa_150.csv — sabit 150 soru</td></tr></table>

<div class="pb"></div>
<h1>1. KV Cache Ablasyonu <span class="sub">(RAG'sız, 6 koşul)</span></h1>
<div class="section"><h2>1a. Kalite Karşılaştırması</h2>{svg_quality_bars(kv_rows, "Birleşik Kalite Skoru — KV Cache (RAG'sız)")}</div>
<div class="section"><h2>1b. Süre Karşılaştırması</h2>{svg_latency(kv_rows, "Medyan Yanıt Süresi (nokta = p95, sağ = token/s)")}</div>
<div class="section"><h2>1c. Alt-Skor Referans Tablosu</h2>{_quality_table(kv_rows)}</div>

<div class="pb"></div>
<h1>2. RAG Ablasyonu</h1>
<div class="section"><h2>2a. RAG Yöntemleri — Kalite (Gemma-q4)</h2>{svg_quality_bars(rag_rows, "Birleşik Kalite Skoru — 4 RAG Yöntemi")}</div>
<div class="section"><h2>2b. RAG Yöntemleri — Üretim Süresi</h2>{svg_latency(rag_rows, "Üretim Süresi (retrieval hariç)")}</div>
<div class="section"><h2>2c. Retrieval Süresi (ayrı)</h2>
<table><tr><th>Yöntem</th><th>Medyan retrieval</th></tr>{ret_rows}</table></div>

<div class="pb"></div>
<h2>2d. Classic Vector RAG × KV Cache — Kalite</h2>{svg_quality_bars(cv_rows, "Birleşik Kalite Skoru — Classic Vector RAG")}
<h2>2e. Classic Vector RAG × KV Cache — Süre</h2>{svg_latency(cv_rows, "Medyan Yanıt Süresi")}
{_quality_table(cv_rows)}

<div class="pb"></div>
<h1>3. Sonuç &amp; Sınırlılıklar</h1>
<p class="callout">RAG'sız KV ablasyonunda kalite skorları düşüktür (~3-4); RAG bağlamı altında
belirgin yükselir (~7-8). Bu fark bağlam sağlamanın etkisidir, model kapasitesinin değil.
KV kuantizasyonu bu tek-istek protokolünde hız kazandırmadı; faydası bellek tasarrufu
potansiyelindedir (bu rapor tepe bellek ölçmez). Sonuçlar bu donanım, sabit 150 soru ve
tek-istek protokolüne aittir; held-out genelleme iddiası taşımaz.</p>
</body></html>"""


def html_to_pdf(html: str, out_path: Path) -> None:
    from playwright.sync_api import sync_playwright
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        page.pdf(path=str(out_path), format="A4", landscape=True, print_background=True)
        browser.close()


def main() -> None:
    kv_rows = [load_row(label, KV_ABLATION / sub) for label, sub in KV_LABELS]
    cv_rows = [load_row(label, KV_RAG_CV / sub) for label, sub in KV_LABELS]
    rag_rows = [load_row(name, path) for name, path in RAG_METHOD_PATHS.items()]
    retrieval = json.loads(RETRIEVAL_TIMING.read_text(encoding="utf-8"))["methods"]
    html = render_html(kv_rows, cv_rows, rag_rows, retrieval)
    out = ROOT / "output/pdf/uysm_consolidated_evaluation_report.pdf"
    html_to_pdf(html, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
