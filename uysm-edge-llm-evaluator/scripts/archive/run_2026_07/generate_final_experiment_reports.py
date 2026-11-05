#!/usr/bin/env python3
"""Generate separate Turkish PDF reports for KV, RAG, and speculative runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reporting.report_utils import condition


OUT = ROOT / "output/pdf"
TMP = ROOT / "tmp/pdfs/final_experiments"
KV_BASE = ROOT / "data/results/kv_rag_classic_vector_20260722_150"
RAG_BASE = ROOT / "data/results/rag_methods_gemma_q4_20260722_150"
SPEC_BASE = ROOT / "data/results/speculative_ablation_20260722_150"
BLUE, GREEN, AMBER, RED = "#426b8f", "#2f8f68", "#d99120", "#c64f4f"


def styles():
    font_dir = Path("/System/Library/Fonts/Supplemental")
    if "Arial" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Arial", str(font_dir / "Arial.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(font_dir / "Arial Bold.ttf")))
    base = getSampleStyleSheet()
    body = ParagraphStyle("BodyX", parent=base["BodyText"], fontName="Arial", fontSize=10, leading=14, textColor=colors.HexColor("#24292f"), spaceAfter=7)
    return {
        "body": body,
        "title": ParagraphStyle("TitleX", parent=body, fontName="Arial-Bold", fontSize=25, leading=30, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("SubtitleX", parent=body, fontSize=12, textColor=colors.HexColor("#555555"), alignment=TA_CENTER),
        "h1": ParagraphStyle("H1X", parent=body, fontName="Arial-Bold", fontSize=19, leading=23, spaceAfter=12),
        "h2": ParagraphStyle("H2X", parent=body, fontName="Arial-Bold", fontSize=13, leading=17, textColor=colors.HexColor(BLUE), spaceBefore=7, spaceAfter=7),
        "callout": ParagraphStyle("CalloutX", parent=body, backColor=colors.HexColor("#eef3f7"), borderPadding=9, leading=15),
        "small": ParagraphStyle("SmallX", parent=body, fontSize=8.2, leading=10.5),
    }


def document(path: Path, title_text: str):
    page_w, page_h = landscape(A4)
    doc = BaseDocTemplate(str(path), pagesize=(page_w, page_h), leftMargin=17*mm, rightMargin=17*mm, topMargin=15*mm, bottomMargin=15*mm, title=title_text, author="UYSM")
    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Arial", 7.5); canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(16*mm, 9*mm, title_text); canvas.drawRightString(281*mm, 9*mm, f"Sayfa {doc.page}"); canvas.restoreState()
    doc.addPageTemplates(PageTemplate(id="main", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="frame")], onPage=footer))
    return doc


def add_table(story, data, widths, font=8.5):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe9f2")), ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Arial"), ("FONTSIZE", (0, 0), (-1, -1), font), ("LEADING", (0, 0), (-1, -1), font + 3),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#aab4be")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f9")]), ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ])); story.append(table)


def title_page(story, st, title, subtitle, intro):
    story += [Spacer(1, 35*mm), Paragraph(title, st["title"]), Spacer(1, 6*mm), Paragraph(subtitle, st["subtitle"]), Spacer(1, 17*mm), Paragraph(intro, st["callout"]), PageBreak()]


def chart(path: Path, labels: list[str], series: list[tuple[str, list[float], str]], ylabel: str, title: str):
    fig, ax = plt.subplots(figsize=(11.2, 4.8)); x = range(len(labels)); width = .8 / len(series)
    for index, (name, values, color) in enumerate(series):
        positions = [value - .4 + width / 2 + index * width for value in x]
        ax.bar(positions, values, width, label=name, color=color)
    ax.set_xticks(list(x), labels, rotation=15, ha="right"); ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold"); ax.grid(axis="y", alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def generate_kv():
    st = styles(); TMP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    names = ["gemma_f16", "gemma_q8_0", "gemma_q4_0", "qwen_f16", "qwen_q8_0", "qwen_q4_0"]
    labels = ["Gemma f16", "Gemma q8", "Gemma q4", "Qwen f16", "Qwen q8", "Qwen q4"]
    values = {name: condition(KV_BASE / name) for name in names}
    chart(TMP / "kv_quality.png", labels, [("Doğruluk", [values[n]["quality"]["correctness_score_mean"] for n in names], BLUE), ("Dayanaklılık", [values[n]["quality"]["groundedness_score_mean"] for n in names], GREEN)], "Skor (0-10)", "Classic Vector RAG Altında Kalite")
    chart(TMP / "kv_speed.png", labels, [("Yanıt süresi (sn)", [values[n]["latency"] for n in names], AMBER), ("Token/sn / 10", [values[n]["tps"] / 10 for n in names], BLUE)], "Ölçeklenmiş değer", "Yanıt Süresi ve Üretim Hızı")
    path = OUT / "uysm_classic_vector_rag_kv_cache_report.pdf"; story=[]
    title_page(story, st, "Classic Vector RAG<br/>KV-Cache Karşılaştırması", "150 soru | Gemma + Qwen | f16, q8_0, q4_0 | 900 değerlendirme", "Bu rapor aynı Classic Vector RAG bağlamları altında iki model ve üç simetrik KV-cache tipini karşılaştırır. Tüm koşullar aynı 150 soru, dondurulmuş retrieval bağlamları ve üretim ayarlarını kullanır.")
    story += [Paragraph("1. Deney Tasarımı", st["h1"])]
    add_table(story, [["Bileşen", "Değer"], ["Soru seti", "final_qa_150.csv - sabit 150 soruluk mühendislik benchmark'ı"], ["RAG", "Classic Vector; embeddinggemma:300m; k=10; threshold=0.55"], ["Modeller", "Gemma4 26B-A4B ve Qwen3.6 35B-A3B; Q4_K_M ağırlık"], ["Değişken", "Simetrik K/V cache: f16, q8_0, q4_0"], ["Sabit üretim", "ctx 32768; parallel 1; Flash Attention açık; prompt cache kapalı; temperature 0; max 512"], ["Jüri", "gpt-5.4-mini; reasoning_effort=high; balanced-v2"], ["Başarı", "900/900 inference ve 900/900 değerlendirme başarılı"]], [55*mm, 182*mm])
    story += [Paragraph("Dondurulmuş bağlam doğrulaması", st["h2"]), Paragraph("Altı koşulun retrieved_chunks alanları frozen_contexts.jsonl ile birebir doğrulandı. Bu nedenle model ve KV farkları retrieval değişiminden kaynaklanmaz. Classic Vector 116 soruda en az bir chunk döndürdü; 34 soruda threshold nedeniyle bağlam boştu.", st["body"]), Paragraph("Bu sabit benchmark held-out genelleme kanıtı değildir; sonuçlar bu donanım, soru seti ve tek-istek protokolüne aittir.", st["callout"]), PageBreak(), Paragraph("2. Sonuç Tablosu", st["h1"])]
    data=[["Koşul","Doğr.","Eksik.","Day.","Pass","Fail","High risk","Ort. sn","p95 sn","Tok/sn"]]
    for name,label in zip(names,labels):
        v=values[name]; q=v["quality"]; data.append([label,f"{q['correctness_score_mean']:.2f}",f"{q['completeness_score_mean']:.2f}",f"{q['groundedness_score_mean']:.2f}",q["verdicts"].get("pass",0),q["verdicts"].get("fail",0),q["hallucination_risk"].get("high",0),f"{v['latency']:.2f}",f"{v['p95']:.2f}",f"{v['tps']:.2f}"])
    add_table(story,data,[34*mm]+[20*mm]*9,8.1)
    story += [Paragraph("Ana okuma", st["h2"]), Paragraph("Gemma-f16 en yüksek doğruluğu (7.44), Gemma-q4_0 ise en yüksek dayanaklılığı (6.99) verdi. Farklar küçüktür. Qwen-q4_0 doğrulukta Gemma-q4_0'a çok yaklaştı (7.39 ve 7.41), ancak daha yüksek high-risk sayısına sahipti (33 ve 23).", st["body"]), Paragraph("Qwen'in üretim token hızı daha yüksektir; buna karşın toplam yanıt süresi biraz daha uzundur. Bunun nedeni yanıt uzunluğu ve toplam üretilen token miktarının token/sn değerinden ayrı olmasıdır.", st["callout"]), PageBreak(), Paragraph("3. Grafikler", st["h1"]), Image(str(TMP / "kv_quality.png"), width=238*mm, height=102*mm), Spacer(1,5*mm), Image(str(TMP / "kv_speed.png"), width=238*mm, height=102*mm), PageBreak(), Paragraph("4. Sonuç ve Öneri", st["h1"]), Paragraph("KV kuantizasyonu bu parallel=1 deneyinde hız avantajı üretmedi: her iki modelde de f16 en yüksek token/sn değerini verdi. q4_0'ın temel faydası burada hızdan çok bellek tasarrufu potansiyelidir; bu rapor bellek tepe kullanımını karşılaştırmadığı için ayrıca ölçülmelidir.", st["body"]), Paragraph("Kalite-hız dengesi için Gemma-q4_0 güçlü ve güvenli varsayılan koşuldur: 7.41 doğruluk, 6.99 dayanaklılık ve en düşük high-risk sayısı. Yalnız ham doğruluk hedeflenirse Gemma-f16 çok küçük bir farkla liderdir.", st["callout"]), Paragraph("Kaynak artifaktlar", st["h2"]), Paragraph("data/results/kv_rag_classic_vector_20260722_150/<br/>Her koşulda merged_results.csv, retrieved_chunks, server logları, evaluated_results_balanced_v2.csv ve summary_balanced_v2.json bulunur.", st["small"])]
    document(path, "Classic Vector RAG KV-Cache Raporu").build(story); return path


def generate_rag():
    st=styles(); timing=json.loads((RAG_BASE / "retrieval_timing/summary.json").read_text())["methods"]
    paths={"Classic Vector":KV_BASE/"gemma_q4_0","Classic BM25":ROOT/"data/results/kv_rag_classic_bm25_20260722_150/gemma_q4_0","Smart Vector":RAG_BASE/"smart_vector","Smart BM25":RAG_BASE/"smart_bm25"}
    keys={"Classic Vector":"classic_vector","Classic BM25":"classic_bm25","Smart Vector":"smart_vector","Smart BM25":"smart_bm25"}; values={n:condition(p) for n,p in paths.items()}; labels=list(paths)
    chart(TMP/"rag_quality.png",labels,[("Doğruluk",[values[n]["quality"]["correctness_score_mean"] for n in labels],BLUE),("Eksiksizlik",[values[n]["quality"]["completeness_score_mean"] for n in labels],GREEN),("Dayanaklılık",[values[n]["quality"]["groundedness_score_mean"] for n in labels],AMBER)],"Skor (0-10)","Dört RAG Yönteminde Yanıt Kalitesi")
    path=OUT/"uysm_gemma_q4_rag_methods_report.pdf"; story=[]
    title_page(story,st,"Gemma-q4_0<br/>Dört RAG Yöntemi Karşılaştırması","150 soru | Classic/Smart x Vector/BM25 | 600 değerlendirme","Bu rapor yalnız Gemma4 26B-A4B ve q4_0 KV-cache ile dört retrieval yöntemini karşılaştırır. Retrieval süresi, üretim yanıt süresi ve kalite ayrı ölçülmüştür.")
    story += [Paragraph("1. Yöntemler",st["h1"])]
    add_table(story,[["Yöntem","Arama birimi","Gerçek ort. chunk","Önemli davranış"],["Classic Vector","1000 karakterlik chunk",f"{values['Classic Vector']['chunks']:.2f}","Embedding benzerliği; 0.55 threshold nedeniyle bazı sorular boş"],["Classic BM25","1000 karakterlik chunk",f"{values['Classic BM25']['chunks']:.2f}","Sözcüksel eşleşme; her soruda 10 pozitif eşleşme"],["Smart Vector","Child arama, parent dönüş",f"{values['Smart Vector']['chunks']:.2f}","Kısa child ile eşleştirir, geniş parent bağlamı üretir"],["Smart BM25","Child BM25, parent dönüş",f"{values['Smart BM25']['chunks']:.2f}","Sözcüksel child seçimi ve geniş parent bağlamı"]],[43*mm,48*mm,31*mm,115*mm],8.4)
    story += [Paragraph("Kontroller",st["h2"]),Paragraph("Tüm yöntemler k=10 kullandı; vector yöntemlerinde threshold=0.55 ve embeddinggemma:300m kullanıldı. Model, KV-cache, prompt, üretim ayarları ve 150 soru sabittir. Configured k ile gerçek retrieved_chunk_size ayrı raporlanmıştır.",st["body"]),PageBreak(),Paragraph("2. Birleşik Sonuçlar",st["h1"])]
    data=[["Yöntem","Doğr.","Eksik.","Day.","Pass","Fail","Ret. ort. sn","Ret. p95","Gen. ort. sn","Tok/sn"]]
    for n in labels:
        v=values[n];q=v["quality"];t=timing[keys[n]];data.append([n,f"{q['correctness_score_mean']:.2f}",f"{q['completeness_score_mean']:.2f}",f"{q['groundedness_score_mean']:.2f}",q["verdicts"].get("pass",0),q["verdicts"].get("fail",0),f"{t['mean_seconds']:.3f}",f"{t['p95_seconds']:.3f}",f"{v['latency']:.2f}",f"{v['tps']:.2f}"])
    add_table(story,data,[35*mm]+[20*mm]*9,7.8)
    story += [Paragraph("En iyi kalite",st["h2"]),Paragraph("Classic BM25 en yüksek doğruluk (8.19), eksiksizlik (7.74) ve dayanaklılık (7.73) skorlarını verdi. Ayrıca 102 pass ve yalnız 19 fail ile en güçlü genel kalite profilini oluşturdu.",st["callout"]),Paragraph("Smart BM25 103 pass ile bir adet daha fazla pass üretti; fakat ortalama skorları Classic BM25'ten düşük ve üretim süresi yaklaşık üç kat daha uzundu. Parent bağlamlarının büyüklüğü, daha düşük token/sn ve daha uzun üretim süresiyle birlikte maliyeti artırdı.",st["body"]),PageBreak(),Paragraph("3. Kalite Grafiği",st["h1"]),Image(str(TMP/"rag_quality.png"),width=238*mm,height=102*mm),Paragraph("Vector yöntemleri daha az gerçek chunk döndürdü. Smart Vector, Classic Vector'a göre kaliteyi yükseltti; bu, parent genişletmenin bazı sorularda bağlam kapsamını iyileştirdiğini gösterir. Ancak Smart BM25, Classic BM25'i geçemedi.",st["body"]),Paragraph("Retrieval süreleri ayrı bir sıcak-cache taramasıdır. İlk istek maliyetleri max değerlerde görünür; üretim response_time_seconds alanına retrieval dahil değildir.",st["callout"]),PageBreak(),Paragraph("4. Sonuç ve Öneri",st["h1"]),Paragraph("Bu sabit 150 soruluk benchmark için önerilen yöntem <b>Classic BM25</b>'tir. En yüksek üç kalite ortalamasını verirken üretim gecikmesi Smart yöntemlerden belirgin biçimde düşüktür. Classic Vector en düşük üretim gecikmesine sahiptir ancak 34 soruda bağlam boş kalması kaliteyi sınırlar.",st["callout"]),Paragraph("Smart yöntemler ancak geniş parent bağlamının gerekli olduğu özel soru tiplerinde seçici olarak değerlendirilebilir. Genel varsayılan olarak kullanmak, özellikle Smart BM25'te yüksek gecikme cezası getirir.",st["body"]),Paragraph("Sınırlılıklar",st["h2"]),Paragraph("Bu çalışma retrieval recall için ayrı gold-chunk etiketi içermez; burada ölçülen kalite son yanıta aittir. BM25'in sıcak-cache zamanları sürekli servis durumunu temsil eder. Sonuçlar tek koleksiyon, tek embedding modeli ve sabit 150 soru ile sınırlıdır; held-out genelleme iddiası taşımaz.",st["body"]),Paragraph("Kaynak artifaktlar",st["h2"]),Paragraph("Classic Vector: data/results/kv_rag_classic_vector_20260722_150/gemma_q4_0/<br/>Classic BM25: data/results/kv_rag_classic_bm25_20260722_150/gemma_q4_0/<br/>Smart yöntemler ve retrieval timing: data/results/rag_methods_gemma_q4_20260722_150/",st["small"])]
    document(path,"Gemma-q4 RAG Yöntemleri Raporu").build(story); return path


def generate_speculative():
    st=styles(); st["body"].fontSize=9.2; st["body"].leading=12.5; st["callout"].fontSize=9.2; st["callout"].leading=12.5
    manifest=json.loads((SPEC_BASE/"manifest.json").read_text()) if (SPEC_BASE/"manifest.json").exists() else {"conditions":{}}
    path=OUT/"uysm_speculative_decoding_report.pdf"; story=[]
    title_page(story,st,"Speculative Decoding<br/>Deney Raporu","Qwen n-gram | Gemma n-gram | Gemma EAGLE-3 durum raporu","Bu rapor speculative decoding koşullarının çalıştırılma durumunu, performans çıktısını ve doğrulanmış sınırlılıkları açıkça ayırır. Eksik metrikler tahmin edilmemiştir.")
    story += [Paragraph("1. Sabit Protokol",st["h1"])]
    add_table(story,[["Bileşen","Değer"],["Hedef modeller","Qwen3.6 35B-A3B ve Gemma4 26B-A4B; Q4_K_M"],["KV-cache","q8_0 K/V"],["Sunucu","ctx 32768; parallel 1; Flash Attention açık; prompt cache kapalı"],["Üretim","temperature 0; max 512; thinking kapalı"],["n-gram","ngram-mod; match/min/max = 24/48/64"],["EAGLE-3","spec-draft-n-max=3; hedefe özgü draft GGUF gerekir"],["Veri","150 soru; bu speculative script RAG kullanmaz"]],[55*mm,182*mm])
    data=[["Koşul","Durum","Satır","Ort. sn","p95 sn","Tok/sn","Acceptance","Balanced-v2"]]
    for key,label in [("qwen_ngram","Qwen n-gram"),("gemma_ngram","Gemma n-gram")]:
        directory=SPEC_BASE/key
        if (directory/"raw_results.csv").exists():
            v=condition(directory); stats=json.loads((directory/"speculative_stats.json").read_text()) if (directory/"speculative_stats.json").exists() else {}
            acceptance=stats.get("draft_acceptance_rate"); q="Tamamlandı" if (directory/"summary_balanced_v2.json").exists() else "Bekliyor - API limiti"
            data.append([label,manifest.get("conditions",{}).get(key,{}).get("status","unknown"),v["n"],f"{v['latency']:.2f}",f"{v['p95']:.2f}",f"{v['tps']:.2f}",f"{100*acceptance:.1f}%" if acceptance is not None else "Logda yok",q])
        else:data.append([label,manifest.get("conditions",{}).get(key,{}).get("status","yok"),0,"-","-","-","-","-"])
    data.append(["Gemma EAGLE-3","Bloke - draft GGUF yok",0,"-","-","-","-","-"])
    add_table(story,data,[34*mm,38*mm,14*mm,20*mm,20*mm,20*mm,28*mm,52*mm],7.8)
    story += [Paragraph("2. Bulguların Yorumu",st["h1"]),Paragraph("n-gram inference satırları ve sunucu logları kaydedildi. Ancak mevcut llama.cpp sunucu logu aggregate accepted/generated token istatistiğini üretmedi; bu nedenle acceptance oranı raporlanamaz. Script bu alanı zorunlu tuttuğu için koşulu teknik olarak failed işaretleyebilir; bu, 150 inference satırının başarısız olduğu anlamına gelmez.",st["callout"]),Paragraph("Gemma EAGLE-3 çalıştırılmadı çünkü /home/odine/models altında hedefe özgü Gemma draft GGUF bulunamadı. Başka bir draft modelle ikame yapılmadı; böyle bir ikame kontrollü deney tasarımını bozar.",st["body"]),Paragraph("Kalite değerlendirmesi",st["h2"]),Paragraph("OpenAI kullanım limiti yeni speculative cevapları gpt-5.4-mini ile değerlendirme aşamasında doldu. KV ve RAG değerlendirmeleri tamamlandı; yalnız speculative kalite skorları bekliyor. Limit yenilendiğinde evaluate_balanced_v2.py ham CSV'ler üzerinde resume-safe biçimde çalıştırılabilir.",st["body"]),Paragraph("3. Sonuç",st["h1"]),Paragraph("Speculative decoding için performans üretimi kısmen tamamlandı; n-gram koşullarında ham cevaplar ve zamanlama verisi mevcuttur. EAGLE-3 ve balanced-v2 kalite değerlendirmesi tamamlanmadan kesin bir kazanan ilan edilmemelidir. Ayrıca acceptance metriği için llama.cpp log formatı veya ayrı istemci/sunucu ölçümü düzeltilmelidir. Kaynaklar data/results/speculative_ablation_20260722_150/ altında saklanmıştır.",st["callout"])]
    document(path,"Speculative Decoding Deney Raporu").build(story); return path


def main():
    for path in (generate_kv(),generate_rag(),generate_speculative()): print(path)


if __name__ == "__main__": main()
