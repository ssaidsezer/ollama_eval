# Konsolide Değerlendirme Raporu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** KV cache ve RAG ablasyon deneylerini tek modern-minimal PDF raporunda birleştiren `scripts/reporting/generate_consolidated_report.py` üretmek.

**Architecture:** Sonuç dizinlerini mevcut `condition()` yükleyicisiyle oku → birleşik kalite skorunu (3 alt-skor ortalaması) hesapla → inline SVG grafikleri (yatay bar kalite + latency) ve gömülü-CSS HTML üret → Playwright/Chromium ile tek PDF'e bas.

**Tech Stack:** Python 3.12, Playwright (chromium, PDF render), stdlib (json/csv/statistics), inline SVG (bağımlılıksız grafik).

## Global Constraints

- Çıktı: tek PDF → `output/pdf/uysm_consolidated_evaluation_report.pdf`
- Dil: Türkçe (etiketler, başlıklar, açıklamalar)
- Birleşik kalite skoru: `(correctness_score_mean + completeness_score_mean + groundedness_score_mean) / 3`, ölçek 0-10
- Grafiklerde YALNIZ birleşik skor; alt-skorlar sadece referans tablosunda
- Süre ve kalite AYRI görsellerde
- Kapsam: KV cache + RAG (speculative HARİÇ)
- Veri yükleme: mevcut `scripts/archive/run_2026_07/generate_final_experiment_reports.py` içindeki `condition(path)` fonksiyonunu yeniden kullan (delimiter/key auto-detect'i zaten var)
- Render motoru: Playwright chromium (weasyprint bu makinede bozuk)
- Tüm renkler yumuşak palet: `#426b8f` (mavi), `#2f8f68` (yeşil), `#d99120` (amber), nötr griler

## Veri kaynak haritası (verbatim, generate_final_experiment_reports.py:130-131'den)

```python
ROOT = Path(__file__).resolve().parents[1]
KV_ABLATION = ROOT / "data/results/kv_ablation_20260721_150"          # 6: {gemma,qwen}_{f16,q8_0,q4_0}
KV_RAG_CV   = ROOT / "data/results/kv_rag_classic_vector_20260722_150" # 6: aynı 6 koşul
RAG_METHODS = ROOT / "data/results/rag_methods_gemma_q4_20260722_150"  # smart_vector, smart_bm25, retrieval_timing
CLASSIC_BM25 = ROOT / "data/results/kv_rag_classic_bm25_20260722_150"  # gemma_q4_0

# RAG yöntemleri (4) → gemma_q4_0 bazlı:
RAG_METHOD_PATHS = {
    "Classic Vector": KV_RAG_CV / "gemma_q4_0",
    "Classic BM25":   CLASSIC_BM25 / "gemma_q4_0",
    "Smart Vector":   RAG_METHODS / "smart_vector",
    "Smart BM25":     RAG_METHODS / "smart_bm25",
}
RETRIEVAL_TIMING = RAG_METHODS / "retrieval_timing/summary.json"  # {"methods": {"classic_vector": {median_seconds,...}}}
```

`condition(path)` döner: `{rows, n, total, latency (mean), p95, tps (mean), quality (dict|None), chunks}`.
`quality` = summary_balanced_v2.json'un `evaluation` bölümü: `correctness_score_mean`,
`completeness_score_mean`, `groundedness_score_mean`, `verdicts.{pass,partial,fail}`.

---

### Task 1: Birleşik kalite skoru + koşul normalizasyonu

**Files:**
- Create: `scripts/reporting/generate_consolidated_report.py`
- Test: `tests/test_consolidated_report.py`

**Interfaces:**
- Consumes: `condition(path)` from `scripts/archive/run_2026_07/generate_final_experiment_reports.py`
- Produces:
  - `unified_quality(quality: dict) -> float` — 3 alt-skorun ortalaması (0-10)
  - `pass_rate(quality: dict) -> float` — `verdicts.pass / (pass+partial+fail)`, hiç verdict yoksa 0.0
  - `load_row(name: str, path: Path) -> dict` — `{name, quality_score, correctness, completeness, groundedness, pass_rate, latency, p95, tps, chunks}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_consolidated_report.py
import importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "generate_consolidated_report", ROOT / "scripts/reporting/generate_consolidated_report.py"
)
ccr = importlib.util.module_from_spec(spec)
sys.modules["generate_consolidated_report"] = ccr
spec.loader.exec_module(ccr)


def test_unified_quality_averages_three_subscores():
    q = {"correctness_score_mean": 6.0, "completeness_score_mean": 3.0, "groundedness_score_mean": 3.0}
    assert ccr.unified_quality(q) == 4.0


def test_pass_rate_from_verdicts():
    q = {"verdicts": {"pass": 25, "partial": 50, "fail": 25}}
    assert ccr.pass_rate(q) == 0.25


def test_pass_rate_empty_verdicts_returns_zero():
    assert ccr.pass_rate({}) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_consolidated_report.py -q`
Expected: FAIL — module/file not found or attribute error.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/reporting/generate_consolidated_report.py
"""Konsolide UYSM değerlendirme raporu: KV cache + RAG ablasyonu, tek PDF."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# generate_final_experiment_reports.condition() yeniden kullanımı (delimiter/key auto-detect)
_spec = importlib.util.spec_from_file_location(
    "generate_final_experiment_reports", ROOT / "scripts/archive/run_2026_07/generate_final_experiment_reports.py"
)
_gfer = importlib.util.module_from_spec(_spec)
sys.modules["generate_final_experiment_reports"] = _gfer
_spec.loader.exec_module(_gfer)
condition = _gfer.condition


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
        "p95": data["p95"],
        "tps": data["tps"],
        "chunks": data["chunks"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_consolidated_report.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/reporting/generate_consolidated_report.py tests/test_consolidated_report.py
git commit -m "feat: unified quality score + condition normalization for consolidated report"
```

---

### Task 2: Inline SVG grafik üreticileri

**Files:**
- Modify: `scripts/reporting/generate_consolidated_report.py`
- Test: `tests/test_consolidated_report.py`

**Interfaces:**
- Consumes: `load_row` output dicts (list)
- Produces:
  - `svg_quality_bars(rows: list[dict], title: str, width: int = 720) -> str` — yatay bar, birleşik `quality_score`, her barın ucunda pass-rate etiketi. Geçerli `<svg>` string.
  - `svg_latency(rows: list[dict], title: str, width: int = 720) -> str` — her koşul için medyan latency bar + p95 nokta işareti + tps ikincil etiket. Geçerli `<svg>` string.

- [ ] **Step 1: Write the failing test**

```python
def _rows():
    return [
        {"name": "Gemma f16", "quality_score": 3.71, "pass_rate": 0.1867, "latency": 3.79, "p95": 7.1, "tps": 34.4},
        {"name": "Qwen f16", "quality_score": 3.08, "pass_rate": 0.1133, "latency": 3.28, "p95": 5.0, "tps": 37.3},
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_consolidated_report.py -k svg -q`
Expected: FAIL — `svg_quality_bars`/`svg_latency` not defined.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/reporting/generate_consolidated_report.py`:

```python
from html import escape

BLUE, GREEN, AMBER = "#426b8f", "#2f8f68", "#d99120"
INK, MUTE, GRID = "#24292f", "#6b7681", "#e4e8ec"


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
    maxv = max((r["p95"] or r["latency"] or 0) for r in rows) * 1.1 or 1.0
    height = top + row_h * len(rows) + 20
    out = _svg_header(width, height, title)
    for i, r in enumerate(rows):
        y = top + i * row_h
        lat = r["latency"] or 0.0
        bar = plot_w * (lat / maxv)
        p95x = left + plot_w * ((r["p95"] or 0.0) / maxv)
        out.append(f'<text x="{left-10}" y="{y+18}" text-anchor="end" font-size="12" fill="{INK}">{escape(r["name"])}</text>')
        out.append(f'<rect x="{left}" y="{y+6}" width="{plot_w}" height="16" rx="3" fill="{GRID}"/>')
        out.append(f'<rect x="{left}" y="{y+6}" width="{bar:.1f}" height="16" rx="3" fill="{GREEN}"/>')
        out.append(f'<circle cx="{p95x:.1f}" cy="{y+14}" r="4" fill="{AMBER}"/>')
        out.append(f'<text x="{left+bar+8:.1f}" y="{y+18}" font-size="11" fill="{INK}">{lat:.2f}s</text>')
        out.append(f'<text x="{width-4}" y="{y+18}" text-anchor="end" font-size="10" fill="{MUTE}">{r["tps"]:.1f} tok/s</text>')
    out.append("</svg>")
    return "\n".join(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_consolidated_report.py -k svg -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/reporting/generate_consolidated_report.py tests/test_consolidated_report.py
git commit -m "feat: inline SVG quality and latency chart generators"
```

---

### Task 3: HTML birleştirme + PDF render + main()

**Files:**
- Modify: `scripts/reporting/generate_consolidated_report.py`
- Test: `tests/test_consolidated_report.py`

**Interfaces:**
- Consumes: `load_row`, `svg_quality_bars`, `svg_latency`, `RAG_METHOD_PATHS`, `RETRIEVAL_TIMING`
- Produces:
  - `render_html(kv_rows, cv_rows, rag_rows, retrieval) -> str` — tam HTML (gömülü CSS), tüm bölümler
  - `html_to_pdf(html: str, out_path: Path) -> None` — Playwright chromium ile A4 landscape PDF
  - `main() -> None` — dizinleri oku, HTML kur, PDF yaz

- [ ] **Step 1: Write the failing test**

```python
def test_render_html_contains_all_sections():
    kv = _rows(); cv = _rows(); rag = _rows()
    retrieval = {"classic_vector": {"median_seconds": 0.64}}
    html = ccr.render_html(kv, cv, rag, retrieval)
    assert "<html" in html.lower()
    for needle in ["KV Cache", "RAG", "Yönetici Özeti", "Sabit Protokol", "<svg"]:
        assert needle in html, needle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_consolidated_report.py -k render_html -q`
Expected: FAIL — `render_html` not defined.

- [ ] **Step 3: Write minimal implementation**

Append data-source constants + HTML/PDF functions. Constants verbatim from the Veri kaynak haritası section above (ROOT, KV_ABLATION, KV_RAG_CV, RAG_METHODS, CLASSIC_BM25, RAG_METHOD_PATHS, RETRIEVAL_TIMING).

```python
import json

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
    head = "<tr><th>Koşul</th><th>Kalite (0-10)</th><th>Doğruluk</th><th>Eksiksizlik</th><th>Dayanaklılık</th><th>Pass %</th></tr>"
    body = "".join(
        f"<tr><td>{escape(r['name'])}</td><td>{(r['quality_score'] or 0):.2f}</td>"
        f"<td>{(r['correctness'] or 0):.2f}</td><td>{(r['completeness'] or 0):.2f}</td>"
        f"<td>{(r['groundedness'] or 0):.2f}</td><td>{r['pass_rate']*100:.0f}%</td></tr>"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_consolidated_report.py -q`
Expected: PASS (all).

- [ ] **Step 5: Generate the real PDF end-to-end**

Run: `python3 scripts/reporting/generate_consolidated_report.py`
Expected: `Wrote .../output/pdf/uysm_consolidated_evaluation_report.pdf`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/reporting/generate_consolidated_report.py tests/test_consolidated_report.py
git commit -m "feat: consolidated KV+RAG evaluation report (HTML->PDF via chromium)"
```

---

### Task 4: Görsel doğrulama

**Files:** yok (inceleme)

- [ ] **Step 1:** PDF sayfalarını pymupdf ile PNG'ye çevir (scratchpad'e), her sayfayı gözle incele.
- [ ] **Step 2:** Kontrol listesi: (a) grafiklerde yalnız birleşik skor var mı, (b) süre ve kalite ayrı görsel mi, (c) alt-skorlar sadece tabloda mı, (d) tüm 6+4+6 koşul görünüyor mu, (e) tipografi/hizalama temiz mi.
- [ ] **Step 3:** Sorun varsa ilgili task'a dön ve düzelt. Temizse kullanıcıya PDF'i sun.

## Self-Review

**Spec coverage:** Tek birleşik skor (T1 `unified_quality`) ✓; süre/kalite ayrı (T2 iki ayrı SVG, T3 ayrı bölümler) ✓; alt-skorlar sadece tabloda (T3 `_quality_table`) ✓; KV+RAG kapsam (T3 `main`) ✓; speculative hariç ✓; modern-minimal (CSS + SVG palet) ✓; tek PDF (T3 `html_to_pdf`) ✓; render motoru chromium ✓.

**Placeholder scan:** Tüm kod blokları tam; TBD/TODO yok.

**Type consistency:** `condition()` dönüş alanları (`quality`, `latency`, `p95`, `tps`, `chunks`) T1'de tüketiliyor; `load_row` çıktı anahtarları (`quality_score`, `pass_rate`, `name`, `latency`, `p95`, `tps`) T2 SVG'lerde ve T3 tabloda birebir kullanılıyor.
