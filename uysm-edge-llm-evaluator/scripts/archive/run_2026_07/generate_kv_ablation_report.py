#!/usr/bin/env python3
"""Generate the Turkish PDF report for the fixed-150 KV-cache ablation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "data/results/kv_ablation_20260721_150"
OUT = ROOT / "output/pdf/uysm_kv_cache_ablation_150_balanced_v2_report.pdf"
TMP = ROOT / "tmp/pdfs/kv_report"
CONDITIONS = ["qwen_f16", "qwen_q8_0", "qwen_q4_0", "gemma_f16", "gemma_q8_0", "gemma_q4_0"]
DISPLAY = {
    "qwen_f16": "Qwen - f16", "qwen_q8_0": "Qwen - q8_0", "qwen_q4_0": "Qwen - q4_0",
    "gemma_f16": "Gemma - f16", "gemma_q8_0": "Gemma - q8_0", "gemma_q4_0": "Gemma - q4_0",
}
GREEN, AMBER, RED, BLUE, NAVY = "#2f8f68", "#d99120", "#c64f4f", "#426b8f", "#273746"


def load_data():
    rows, frames = [], []
    for condition in CONDITIONS:
        with open(RESULTS / condition / "summary_balanced_v2.json", encoding="utf-8") as f:
            s = json.load(f)
        e, inf = s["evaluation"], s["inference"]
        row = {
            "condition": condition, "label": DISPLAY[condition],
            "model": "Qwen3.6 35B-A3B" if condition.startswith("qwen") else "Gemma4 26B-A4B",
            "kv": condition.split("_", 1)[1], "correctness": e["correctness_score_mean"],
            "completeness": e["completeness_score_mean"], "groundedness": e["groundedness_score_mean"],
            "pass": e["verdicts"].get("pass", 0), "partial": e["verdicts"].get("partial", 0),
            "fail": e["verdicts"].get("fail", 0), "high": e["hallucination_risk"].get("high", 0),
            "medium": e["hallucination_risk"].get("medium", 0), "low": e["hallucination_risk"].get("low", 0),
            "mean_s": inf["wall_time_seconds"]["mean"], "p95_s": inf["wall_time_seconds"]["p95"],
            "tps": inf["generation_tokens_per_second"]["mean"],
            "refusal": e["unanswerable"]["grounded_refusal_accuracy"],
            "unans_hall": e["unanswerable"]["hallucination_rate"],
        }
        rows.append(row)
        d = pd.read_csv(RESULTS / condition / "evaluated_results_balanced_v2.csv")
        d["condition"] = condition
        frames.append(d)
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)


def charts(summary, all_rows):
    TMP.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 16, "axes.titleweight": "bold"})
    ordered = summary.sort_values("correctness")

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    y = np.arange(len(ordered))
    ax.barh(y, ordered.correctness, color=[GREEN if x.startswith("gemma") else BLUE for x in ordered.condition])
    ax.set_yticks(y, ordered.label); ax.set_xlim(0, 10); ax.set_xlabel("Ortalama skor (0-10)")
    ax.set_title("Dengeli-v2 Doğruluk Skoru"); ax.grid(axis="x", alpha=.2)
    for i, v in enumerate(ordered.correctness): ax.text(v + .08, i, f"{v:.2f}", va="center", fontweight="bold")
    fig.tight_layout(); fig.savefig(TMP / "correctness.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    x = np.arange(len(summary)); ax.bar(x, summary["pass"], label="pass", color=GREEN)
    ax.bar(x, summary.partial, bottom=summary["pass"], label="partial", color=AMBER)
    ax.bar(x, summary.fail, bottom=summary["pass"] + summary.partial, label="fail", color=RED)
    ax.set_xticks(x, summary.label, rotation=18, ha="right"); ax.set_ylim(0, 150); ax.set_ylabel("Cevap sayısı")
    ax.set_title("Karar Dağılımı - Her Koşulda 150 Cevap"); ax.legend(ncol=3); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(TMP / "verdicts.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.bar(x, summary.low / 1.5, label="low", color=GREEN)
    ax.bar(x, summary.medium / 1.5, bottom=summary.low / 1.5, label="medium", color=AMBER)
    ax.bar(x, summary.high / 1.5, bottom=(summary.low + summary.medium) / 1.5, label="high", color=RED)
    ax.set_xticks(x, summary.label, rotation=18, ha="right"); ax.set_ylim(0, 100); ax.set_ylabel("Oran (%)")
    ax.set_title("Halüsinasyon Riski Dağılımı"); ax.legend(ncol=3); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(TMP / "risk.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    width=.37; ax.bar(x-width/2, summary.mean_s, width, label="Ortalama", color=BLUE)
    ax.bar(x+width/2, summary.p95_s, width, label="p95", color=AMBER)
    ax.set_xticks(x, summary.label, rotation=18, ha="right"); ax.set_ylabel("Saniye")
    ax.set_title("Uçtan Uca Yanıt Süresi"); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(TMP / "latency.png", dpi=180); plt.close(fig)

    q = all_rows.groupby("question_index").agg(
        correctness=("correctness_score", "mean"), pass_count=("verdict", lambda s: (s == "pass").sum()),
        high_count=("hallucination_risk", lambda s: (s == "high").sum()), question=("question", "first"),
    ).reset_index()
    easy = q.nlargest(5, "correctness"); hard = q.nsmallest(5, "correctness")
    return easy, hard


def build_pdf(summary, all_rows, easy, hard):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    font_dir = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("Arial", str(font_dir / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(font_dir / "Arial Bold.ttf")))
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Arial", fontSize=10, leading=14, textColor=colors.HexColor("#24292f"), spaceAfter=7)
    h1 = ParagraphStyle("H1", parent=body, fontName="Arial-Bold", fontSize=20, leading=24, spaceBefore=5, spaceAfter=14)
    h2 = ParagraphStyle("H2", parent=body, fontName="Arial-Bold", fontSize=14, leading=18, textColor=colors.HexColor(BLUE), spaceBefore=8, spaceAfter=8)
    small = ParagraphStyle("Small", parent=body, fontSize=8.4, leading=11)
    center = ParagraphStyle("Center", parent=body, alignment=TA_CENTER)
    title = ParagraphStyle("Title", parent=center, fontName="Arial-Bold", fontSize=27, leading=32)
    subtitle = ParagraphStyle("Subtitle", parent=center, fontSize=12, textColor=colors.HexColor("#555555"))
    callout = ParagraphStyle("Callout", parent=body, backColor=colors.HexColor("#eef3f7"), borderPadding=10, leading=15)

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Arial", 7.5); canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(16*mm, 9*mm, "UYSM KV-Cache Ablasyon Değerlendirme Raporu")
        canvas.drawRightString(281*mm, 9*mm, f"Sayfa {doc.page}"); canvas.restoreState()

    page_w, page_h = landscape(A4)
    doc = BaseDocTemplate(str(OUT), pagesize=(page_w, page_h), leftMargin=18*mm, rightMargin=18*mm, topMargin=15*mm, bottomMargin=15*mm,
                          title="UYSM KV-Cache Ablasyon Değerlendirme Raporu", author="UYSM")
    doc.addPageTemplates(PageTemplate(id="main", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")], onPage=footer))
    story=[]
    def p(text, style=body): story.append(Paragraph(text, style))
    def heading(text): p(text, h1)
    def subheading(text): p(text, h2)
    def page(): story.append(PageBreak())
    def img(name, h=125*mm): story.append(Image(str(TMP/name), width=240*mm, height=h))
    def table(data, widths=None, font=8.3):
        t=Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dfe9f2")), ("FONTNAME",(0,0),(-1,0),"Arial-Bold"),
            ("FONTNAME",(0,1),(-1,-1),"Arial"), ("FONTSIZE",(0,0),(-1,-1),font), ("LEADING",(0,0),(-1,-1),font+3),
            ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#aab4be")), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f7f9")]), ("LEFTPADDING",(0,0),(-1,-1),5),
        ])); story.append(t)

    story += [Spacer(1, 36*mm), Paragraph("UYSM KV-Cache Ablasyon<br/>Değerlendirme Raporu", title), Spacer(1, 6*mm),
              Paragraph("150 soruluk sabit benchmark | 2 model | 3 KV tipi | 6 koşul | 900 cevap", subtitle), Spacer(1, 18*mm),
              Paragraph("Bu rapor, Qwen3.6 ve Gemma4 modellerinin f16, q8_0 ve q4_0 simetrik KV-cache koşullarındaki kalite ve hız sonuçlarını özetler. Dengeli-v2 değerlendirmesi OpenAI gpt-5.4-mini ve yüksek akıl yürütme düzeyiyle yapılmıştır.", callout)]
    page(); heading("1. Amaç ve Kapsam")
    p("Amaç, model ailesi ve KV-cache veri tipinin aynı 150 soruluk mühendislik benchmark'ında cevap kalitesi ile çıkarım performansına etkisini kontrollü biçimde karşılaştırmaktır. Bu çalışma RAG içermez; sonuçlar daha sonra yapılacak RAG deneyinde kullanılacak tek model/KV koşulunu seçmek için üretilmiştir.")
    subheading("2. Deney Tasarımı")
    design=[["Bileşen","Sabit / Değer"],["Soru seti","final_qa_150.csv - 150 soru (140 cevaplanabilir, 10 cevaplanamaz)"],["Model aileleri","Qwen3.6 35B-A3B ve Gemma4 26B-A4B"],["Ağırlık kuantizasyonu","Q4_K_M"],["Deney değişkeni","Simetrik K/V cache: f16, q8_0, q4_0"],["llama.cpp ayarları","ctx 32768, parallel 1, Flash Attention açık, prompt cache kapalı"],["Üretim","temperature 0, max_tokens 512, düşünme kapalı"],["Değerlendirici","gpt-5.4-mini, reasoning_effort=high, balanced-v2"],["Başarı","900/900 inference ve 900/900 değerlendirme satırı başarılı"]]
    table(design,[48*mm,190*mm],8.7)
    p("<b>Kapsam notu:</b> Bu sabit benchmark, held-out genelleme kanıtı değildir. KV-cache sonuçları tek istekli ve kısa/orta bağlamlı bu protokole aittir.", callout)

    page(); heading("3. Değerlendirme Metrikleri")
    metrics=[["Metrik","Yorum"],["correctness_score (0-10)","Yanıtın ana iddialarının referansa göre doğruluğu"],["completeness_score (0-10)","Referanstaki gerekli noktaların kapsanma düzeyi"],["groundedness_score (0-10)","Yanıttaki iddiaların referansla desteklenme düzeyi"],["verdict","pass, partial veya fail"],["hallucination_risk","low, medium veya high"],["mean / p95 latency","Ortalama ve yüzde 95 uçtan uca yanıt süresi"],["generation tokens/s","Üretim aşaması token hızı"],["grounded refusal accuracy","Cevaplanamaz sorularda doğru biçimde yanıt vermeme oranı"]]
    table(metrics,[62*mm,176*mm],9)
    subheading("Dengeli-v2 Rubriğinin Yorumu")
    p("Doğruluk, eksiksizlik ve dayanaklılık ayrı puanlanır. Zararsız ve doğru ek bağlam, doğruluğu otomatik olarak düşürmez; referansla desteklenmeyen ayrıntılar öncelikle dayanaklılık puanını etkiler. Yanlış, çelişkili, güvensiz veya eylemi değiştiren eklemeler ise doğruluğu düşürür ve halüsinasyon riskini artırır.")
    p("Karar etiketi model tarafından üretildiği için eşik-temelli deterministik bir sınıf değildir. Bu nedenle koşul seçiminin ana dayanağı sürekli skorlar, risk dağılımı ve hız metrikleridir.", callout)

    page(); heading("4. Genel Sonuçlar")
    aggregate=[["Metrik","900 kayıt geneli"],["Ortalama doğruluk",f"{all_rows.correctness_score.mean():.2f}/10"],["Ortalama eksiksizlik",f"{all_rows.completeness_score.mean():.2f}/10"],["Ortalama dayanaklılık",f"{all_rows.groundedness_score.mean():.2f}/10"],["Pass / Partial / Fail",f"{(all_rows.verdict=='pass').sum()} / {(all_rows.verdict=='partial').sum()} / {(all_rows.verdict=='fail').sum()}"],["Low / Medium / High risk",f"{(all_rows.hallucination_risk=='low').sum()} / {(all_rows.hallucination_risk=='medium').sum()} / {(all_rows.hallucination_risk=='high').sum()}"],["Ortalama yanıt süresi",f"{summary.mean_s.mean():.2f} sn (koşul ortalamalarının ortalaması)"],["Başarılı kayıt","900 / 900"]]
    table(aggregate,[72*mm,92*mm],9)
    p("Gemma ailesi üç KV tipinde de Qwen'den daha yüksek kalite puanları üretmiştir. Qwen daha hızlıdır; ancak bu benchmark'ta hız farkı kalite farkını telafi etmemektedir. KV tipi kendi model ailesi içinde küçük kalite dalgalanmaları yaratmış, belirgin bir hız kazancı sağlamamıştır.", callout)
    subheading("En Güçlü Koşul")
    p("<b>Gemma - q4_0</b>, en yüksek ortalama doğruluk (4.56) ve dayanaklı koşullardan biri (3.67) olmuştur. Ortalama yanıt süresi 3.90 sn'dir; Gemma - q8_0'a göre fark yaklaşık 0.07 sn'dir. Bu nedenle kontrollü RAG aşaması için önerilen koşuldur.")

    page(); heading("5. Görselleştirmeler"); subheading("5.1 Ortalama doğruluk sıralaması"); img("correctness.png",111*mm)
    p("Gemma'nın üç koşulu ilk üç sıradadır. Aile içindeki farklar küçük olduğundan KV tipinin kalite etkisi, model ailesi etkisinden çok daha zayıf görünmektedir.")
    page(); subheading("5.2 Pass / partial / fail dağılımı"); img("verdicts.png",111*mm)
    p("Gemma koşullarında fail sayısı Qwen koşullarına göre daha düşüktür. Bununla birlikte en iyi koşulda bile 150 cevabın 67'si fail etiketlidir; no-RAG cevap kalitesi üretim kullanımı için yeterli değildir.")
    page(); subheading("5.3 Halüsinasyon riski"); img("risk.png",111*mm)
    p("Qwen koşullarında high-risk oranı yaklaşık %56-57, Gemma koşullarında %35-42 aralığındadır. Risk dağılımı model ailesine güçlü biçimde bağlıdır; KV tipi bu tabloyu temelden değiştirmemiştir.")
    page(); subheading("5.4 Yanıt süresi"); img("latency.png",111*mm)
    p("Qwen koşulları ortalamada yaklaşık 3.25-3.44 sn, Gemma koşulları 3.83-3.90 sn aralığındadır. KV kuantizasyonu bu tek istekli protokolde istikrarlı bir hız avantajı sağlamamıştır.")

    page(); heading("6. Koşul Bazında Sıralama")
    ranked=summary.sort_values("correctness",ascending=False)
    data=[["Koşul","Doğr.","Eksik.","Day.","Pass","Partial","Fail","High %","Ort. sn","p95 sn","Tok/sn"]]
    for _,r in ranked.iterrows(): data.append([r.label,f"{r.correctness:.2f}",f"{r.completeness:.2f}",f"{r.groundedness:.2f}",int(r['pass']),int(r.partial),int(r.fail),f"{r.high/1.5:.1f}",f"{r.mean_s:.2f}",f"{r.p95_s:.2f}",f"{r.tps:.2f}"])
    table(data,[35*mm]+[18*mm]*10,7.8)
    subheading("Aile İçindeki KV Etkisi")
    p("Qwen'de en yüksek doğruluk q8_0 (3.63), Gemma'da q4_0 (4.56) ile ölçülmüştür. Ancak farklar küçük ve tek bir sabit benchmark'a aittir. Bu nedenle sonuç, KV kuantizasyonunun doğrudan kaliteyi artırdığı şeklinde genellenmemelidir.")
    subheading("Hız - Kalite Dengesi")
    trade=[["Hedef","Öneri","Gerekçe"],["En yüksek kalite","Gemma - q4_0","En yüksek doğruluk; en yüksek eksiksizlik; güçlü dayanaklılık"],["Gemma içinde en düşük ortalama süre","Gemma - f16","3.83 sn; q8_0 ile pratik olarak aynı"],["En düşük ortalama süre","Qwen - f16","3.25 sn; fakat kalite belirgin biçimde düşük"],["RAG için kontrollü aday","Gemma - q4_0","Kalite lideri; Gemma q8_0'a göre yalnızca ~0.07 sn yavaş"]]
    table(trade,[55*mm,52*mm,131*mm],8.5)

    page(); heading("7. Soru Bazında Gözlemler")
    p("Aşağıdaki tablolar, her soru için altı koşulun ortalama doğruluk skorunu gösterir. Konu etiketi üretilmemiştir; sorular raporda kısa metinle verilmiştir.")
    def qtable(df,title_text):
        subheading(title_text); dat=[["Soru","Ort. doğr.","Pass","High risk","Kısa soru metni"]]
        for _,r in df.iterrows():
            text=str(r.question); text=(text[:105]+"...") if len(text)>108 else text
            dat.append([int(r.question_index),f"{r.correctness:.2f}",int(r.pass_count),int(r.high_count),Paragraph(text,small)])
        table(dat,[16*mm,24*mm,16*mm,21*mm,161*mm],7.7)
    qtable(easy,"7.1 En yüksek ortalamalı beş soru")
    qtable(hard,"7.2 En düşük ortalamalı beş soru")

    page(); heading("8. Ana Bulgular")
    findings=[
        ("8.1 Model ailesi ana etkendir","Gemma, her KV tipinde Qwen'den daha yüksek doğruluk, eksiksizlik ve dayanaklılık üretmiştir."),
        ("8.2 KV kuantizasyonu hız kazandırmadı","q8_0 ve q4_0, bu parallel=1 ve kısa/orta bağlam protokolünde f16'ya karşı istikrarlı bir gecikme avantajı göstermemiştir. Olası fayda bellek ve eşzamanlılık başlığında ayrıca ölçülmelidir."),
        ("8.3 No-RAG kalite sınırlıdır","En iyi koşulun ortalama doğruluğu 4.56/10'dur. Bu sonuçlar bilgi erişimi olmadan model bilgisinin görev için yetersiz kaldığını gösterir; ancak RAG katkısı henüz bu raporda ölçülmemiştir."),
        ("8.4 Cevaplanamaz sorular kritik sorundur","Altı koşulun tamamında grounded refusal accuracy %0'dır. 10 cevaplanamaz soruda halüsinasyon oranı beş koşulda %100, Qwen-q8_0'da %90'dır."),
        ("8.5 Kontrollü RAG adayı","Gemma - q4_0, kalite liderliği ve küçük gecikme cezası nedeniyle sonraki dört RAG varyantı için seçilmelidir."),
    ]
    for a,b in findings: subheading(a); p(b)

    page(); heading("9. Sınırlılıklar")
    limits=["Benchmark 150 sabit sorudan oluşur; bağımsız held-out genelleme testi değildir.","Çalışma no-RAG koşuludur; retrieval kalitesi veya RAG katkısı hakkında doğrudan sonuç vermez.","Yalnızca iki model ailesi, Q4_K_M ağırlıklar ve simetrik K/V cache tipleri test edilmiştir.","parallel=1 ve mevcut soru uzunlukları, uzun bağlam ve çoklu eşzamanlı istek davranışını temsil etmez.","VRAM tepe kullanımı bu sonuç paketinde koşullar arasında karşılaştırılabilir bir rapor metriği olarak kaydedilmemiştir.","Kalite değerlendirmesi tek bir otomatik jüri modeline dayanır; dengeli-v2 ilk satırlarda elle kontrol edilmiş olsa da geniş insan doğrulaması yapılmamıştır.","LLM tarafından üretilen verdict etiketi, sürekli skorlarla her zaman deterministik biçimde uyumlu değildir."]
    for i,x in enumerate(limits,1): p(f"<b>{i}.</b> {x}")
    subheading("10. Sonuç")
    p("Bu kontrollü 900-cevaplık deneyde Gemma4 26B-A4B, Qwen3.6 35B-A3B'den daha yüksek kalite üretmiş; Qwen ise daha hızlı çalışmıştır. KV-cache tipinin hız ve kalite üzerindeki etkisi model ailesi etkisine göre küçüktür. Gemma - q4_0, 4.56 doğruluk, 3.32 eksiksizlik ve 3.67 dayanaklılık ortalamalarıyla en güçlü koşuldur.")
    p("<b>Güvenli özet:</b> Sabit 150 soruluk no-RAG mühendislik benchmark'ında Gemma-q4_0 en yüksek dengeli-v2 kalite skorlarını verdi. KV kuantizasyonu parallel=1 koşulunda belirgin hız kazancı sağlamadı. Tüm koşullar cevaplanamaz sorularda yetersiz kaldığı için sonraki RAG testinde kaynak-yetersizliği durumunda açık ret talimatı zorunludur.", callout)

    page(); heading("11. Önerilen Sonraki Adımlar")
    steps=["Gemma - q4_0 ile classic_bm25, classic_vector, smart_bm25 ve smart_vector koşullarını aynı 150 soru üzerinde çalıştırın.","Her soru/varyant için retrieve k=10 bağlamlarını önce dondurun; generation karşılaştırmasında aynı bağlamı kullanın.","Retrieval kalitesini generation kalitesinden ayrı raporlayın; configured k ile gerçek retrieved_chunk_size değerlerini ayırın.","RAG prompt'una bağlamda yanıt yoksa açıkça yanıt vermeme talimatı ekleyin ve 10 cevaplanamaz soruyu ayrı güvenlik metriği olarak izleyin.","Karar etiketi gerekiyorsa balanced-v2 sürekli skorlarından deterministik eşikler türetin; aynı cevapları yeniden LLM'e puanlatmadan raporlayın.","Uzun bağlam veya concurrency iddiası için ayrı bir stres testi planlayın; bu çalışmanın latency sonucunu o senaryolara genellemeyin."]
    for i,x in enumerate(steps,1): p(f"<b>{i}.</b> {x}")
    subheading("Kaynak Artifaktlar")
    p("Ham ve değerlendirilmiş sonuçlar: <b>data/results/kv_ablation_20260721_150/</b><br/>Her koşul: raw_results.csv, evaluated_results_balanced_v2.csv ve summary_balanced_v2.json.<br/>Değerlendirme sürümü: <b>balanced-v2</b>; jüri: <b>gpt-5.4-mini</b>; reasoning effort: <b>high</b>.")
    p("Rapor oluşturma tarihi: 22 Temmuz 2026", small)
    doc.build(story)


def main():
    summary, all_rows = load_data()
    easy, hard = charts(summary, all_rows)
    build_pdf(summary, all_rows, easy, hard)
    print(OUT)


if __name__ == "__main__":
    main()
