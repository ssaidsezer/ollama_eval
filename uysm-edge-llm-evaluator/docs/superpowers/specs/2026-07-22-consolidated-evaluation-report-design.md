# Konsolide UYSM Değerlendirme Raporu — Tasarım

**Tarih:** 2026-07-22
**Durum:** Onaylandı, implementasyon bekliyor

## Amaç

KV cache ablasyonu ve RAG ablasyonu deneylerini tek, modern-minimal bir PDF raporunda
birleştirmek. Mevcut ayrı raporların iki sorununu gidermek:

1. **Kalite ayrık metriklerle gösteriliyor** (doğruluk/eksiksizlik/dayanaklılık ayrı bar) —
   kullanıcı tek birleşik kalite skoru istiyor.
2. **Grafikler zarafetsiz** — default matplotlib stili, çok yer kaplayan tek boyutlu barlar.

## Kararlar (kullanıcı onaylı)

- **Birleşik kalite skoru:** 3 alt-skorun ortalaması `(correctness + completeness + groundedness) / 3`,
  ölçek 0-10. Alt-skorlar yalnız küçük referans tablosunda kalır, grafiklerde birleşik skor.
- **Kapsam:** KV cache + RAG. Speculative decoding **hariç** (gemma rerun host'a bağlı, veri eksik).
- **Görsel dil:** Modern & minimal — temiz tipografi, ince ızgara, yumuşak palet, yatay bar +
  nokta grafikleri.
- **Süre ve kalite ayrı görsellerde** her deney bloğunda.
- **Render motoru:** HTML/CSS + inline SVG → Playwright/Chromium ile PDF.
  (weasyprint bu makinede native-lib hatası veriyor; Chromium tam CSS/SVG desteği sağlıyor.)

## Veri kaynakları

| Blok | Dizin | Koşullar |
|------|-------|----------|
| KV cache ablasyonu (no-RAG) | `data/results/kv_ablation_20260721_150/*/summary_balanced_v2.json` | Gemma/Qwen × f16/q8_0/q4_0 (6) |
| RAG yöntemleri | `data/results/rag_methods_gemma_q4_20260722_150/` (+ classic bm25/vector dizinleri) | Gemma-q4: Classic/Smart × Vector/BM25 (4) |
| Classic Vector RAG × KV | `data/results/kv_rag_classic_vector_20260722_150/*/summary_balanced_v2.json` | Gemma+Qwen × f16/q8_0/q4_0 (6) |

Her `summary_balanced_v2.json` içinde `evaluation` bölümü:
`correctness_score_mean`, `completeness_score_mean`, `groundedness_score_mean` (0-10),
`verdicts.{pass,partial,fail}`, ve `inference` bölümünde latency/tps metrikleri.

## Rapor yapısı

1. **Kapak + Yönetici Özeti** — kapsam, model/donanım, tek paragraf ana bulgular.
2. **Sabit Protokol** — modeller, KV tipleri, RAG ayarları, jüri (gpt-5.4-mini balanced-v2), 150 soru.
3. **KV Cache Ablasyonu** (no-RAG, 6 koşul)
   - 3a. Kalite karşılaştırması — yatay bar, birleşik Kalite Skoru (0-10), pass-rate etiketi.
   - 3b. Süre karşılaştırması — ayrı görsel: medyan latency + p95, token/s.
   - 3c. Alt-skor referans tablosu (doğruluk/eksiksizlik/dayanaklılık).
4. **RAG Ablasyonu**
   - 4a. RAG yöntemleri (Gemma-q4) — kalite (birleşik) + süre ayrı. Retrieval süresi ayrı.
   - 4b. Classic Vector RAG × KV cache — kalite + süre ayrı.
5. **Sonuç & Sınırlılıklar** — ölçek farkı (no-RAG ~3-4 vs RAG ~7-8, bağlam nedeniyle),
   held-out olmadığı notu, speculative'in ayrı/beklemede olduğu.

## Bileşenler

`scripts/reporting/generate_consolidated_report.py`:

- **`load_condition(dir)`** — bir koşul dizininden summary_balanced_v2.json okur, normalize edilmiş
  dict döner: `{name, quality_score, correctness, completeness, groundedness, pass_rate,
  latency_median, latency_p95, tps_median, ...}`.
- **`unified_quality(ev)`** — 3 alt-skorun ortalaması. Tek sorumluluk, ayrı test edilir.
- **`svg_hbar(rows, ...)` / `svg_latency(rows, ...)`** — inline SVG grafik üreticileri. Her biri
  saf fonksiyon: veri girer, SVG string çıkar.
- **`render_html(sections)`** — bölümleri tek HTML string'e birleştirir (gömülü CSS).
- **`html_to_pdf(html, out_path)`** — Playwright/Chromium ile PDF'e basar.
- **`main()`** — dizinleri okur, bölümleri kurar, PDF yazar.

## Test stratejisi

- `unified_quality` — bilinen alt-skorlardan doğru ortalama.
- `load_condition` — örnek summary'den doğru alanları çıkarır, eksik alanları güvenli işler.
- SVG üreticileri — çıktının geçerli SVG olduğunu ve tüm veri noktalarını içerdiğini doğrula
  (smoke düzeyi; piksel testi değil).

## Sınırlılıklar / kapsam dışı

- Speculative decoding bloğu bu rapora dahil değil (veri eksik).
- Grafikler statik SVG; interaktivite yok (PDF hedefi).
- Held-out genelleme iddiası yok; sonuçlar bu donanım/soru setine ait.
