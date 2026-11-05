#!/usr/bin/env python3
"""Generate Turkish QA pairs from rendered PDF chunk images via OpenAI vision.

Reads the manifest produced by gen_qa_render.py, sends each chunk's labeled page
images to the model with a fixed instruction prompt, and collects the returned
question/answer pairs into a working CSV.

Key facts:
- One vision call per chunk (model default gpt-5.4, temperature 0.2).
- Answers are synthesized (not verbatim); each row carries a verbatim source
  excerpt (kaynak_alinti) and the load-bearing facts (dogrulanan_bilgi) so the
  citation can be reviewed against the page without a string-match verifier.
- `sayfa` is the printed page number (legacy-compatible); `pdf_page_index` is the
  physical PDF page. Both come from the manifest / the image label band.
- Resumable: per-chunk raw JSON is cached; re-running skips completed chunks.

The API key is read from the OPENAI_API_KEY environment variable. Pass it inline
so it is never written to disk:
    OPENAI_API_KEY=sk-... python3 scripts/qa/gen_qa_from_images.py
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

CSV_FIELDS = (
    "kaynak", "question", "answer", "sayfa", "pdf_page_index",
    "kaynak_alinti", "dogrulanan_bilgi", "zorluk", "soru_tipi",
    "sayfa_uyarisi", "chunk",
)

ALLOWED_ZORLUK = {"kolay", "orta", "zor"}
ALLOWED_TIPI = {"dogrudan", "senaryo", "prosedur", "karsilastirma", "cause_effect"}

SYSTEM_PROMPT = """Sen, Türkçe bir Askeri/Taktik Sağlık RAG değerlendirme veri seti hazırlayan uzman bir soru üreticisisin. Görevin, sana görsel olarak verilen kitap sayfalarından, bir RAG (retrieval-augmented generation) sistemini test etmek için gerçekçi soru-cevap çiftleri üretmektir.

## GİRDİ
Sana bir kitaptan ardışık sayfaların görüntüleri verilecek. Her görüntünün en üstünde bir etiket bandı vardır:
"<dosya_adı>  —  SAYFA <basılı_no>  (pdf_page_index <fiziksel_no>)"
- SAYFA = kitabın basılı sayfa numarası (kaynak gösteriminde BUNU kullan).
- pdf_page_index = PDF'teki fiziksel sayfa sırası (ayrıca kaydedeceğiz).
Sayfanın kendi alt/üst kısmındaki basılı numara ile etiketteki SAYFA numarası aynı olmalıdır; farklıysa etiketteki SAYFA'yı esas al ve bunu "sayfa_uyarisi" alanında bildir.

## NE ÜRETECEKSİN
Bu sayfalardaki bilgiye dayanan 0-3 arası soru-cevap çifti. Sayfa bilgi yönünden zenginse (dozlar, mesafeler, prosedür adımları, sayısal değerler, kurallar) daha çok; kapak/içindekiler/şekil-ağırlıklı/boş sayfaysa 0 soru üret. Zorlama soru üretme — kalitesiz bir sorudansa hiç soru üretme.

## SORULARIN NİTELİĞİ (en önemli bölüm)
1. DİL: Doğal, konuşma diline yakın Türkçe. Sorular sahada sesli sorulacakmış gibi KISA olmalı. Birinci ağızdan, senaryo çerçeveli sor. Örnek ton:
   "Arkadaşım şokta ve şiddetli ağrısı var ama fentanil veremiyorum, hangi ağrı kesiciyi ne dozda yapmalıyım?"
   "Elimde Makarof var, 45 metredeki hedefe öldürücü atış yapabilir miyim?"
   Yapay, sınav-tarzı ("Aşağıdakilerden hangisi...") veya kitabı doğrudan tarayan ("X sayfasına göre...") sorular ÜRETME.

2. CEVAP: Kaynaktaki bilgiyi kullanan ama kelimesi kelimesine KOPYALAMAYAN, kendi cümlelerinle kurulmuş bir cevap yaz. Cevap doğru, eksiksiz ve klinik/taktik olarak güvenli olmalı. Kritik sayısal değerleri (doz, mesafe, süre, yüzde) aynen koru. Sayfada olmayan hiçbir bilgi ekleme.

3. DAYANAK: Cevabın dayandığı ifadeyi sayfadan KELİMESİ KELİMESİNE (birebir) "kaynak_alinti" olarak ver. Bu alıntı, verdiğin sayfada gerçekten geçen metin olmalı — uydurma/parafraz DEĞİL. Alıntının içindeki "(bkz sayfa 8)" gibi parantez içi çapraz sayfa referanslarını alıntıdan ÇIKAR; yalnızca olguyu içeren metni ver.

4. KRİTİK BİLGİLER: Cevabın doğruluğunu belirleyen 1-3 anahtar olguyu "dogrulanan_bilgi" listesine yaz (örn. "800 mikrogram", "5-7.5 cm", "her 30 dakikada bir"). Bu olguların HEM kaynak_alinti içinde HEM cevabın içinde geçtiğinden emin ol.

## ETİKETLEME
Her soru için:
- "zorluk": "kolay" | "orta" | "zor"
    kolay = tek bir açık olgu, cevabı sayfada doğrudan bulunur.
    orta = senaryoyu doğru olguya bağlamak gerekir.
    zor = aynı sayfadaki birden fazla olguyu birleştirmek, bir durumu çıkarımla yorumlamak, veya kaynakla farklı kelimelerle (düşük sözcük örtüşmesi) ifade edilmiş olması gerekir. "zor", cevabın belirsiz olması demek DEĞİLDİR; erişim/akıl yürütme açısından zor demektir.
- "soru_tipi": "dogrudan" | "senaryo" | "prosedur" | "karsilastirma" | "cause_effect"

## KAPSAM DIŞI
Sadece kaynakta açıkça yazan bilgiden soru üret. Kendi tıbbi/askeri bilgini EKLEME. Sayfadaki bir şekle/tabloya bakılması gereken ama metinle ifade edilmeyen soruları üretme (görsele bağımlı sorular RAG metin erişimini test etmez).

ÖNEMLİ — TABLO KURALI: Bir tablonun satır/sütun konumundan anlam çıkaran soru ÜRETME. Bir değerin (örn. "3000 g", "100 m") hangi silaha/nesneye ait olduğu yalnızca tablodaki sütun hizasından anlaşılıyorsa ve etiket ile değer aynı cümle/ifade içinde birlikte geçmiyorsa, o sorudan vazgeç. Yalnızca kaynak_alinti'nin kendisi HEM etiketi (neyin) HEM değeri birlikte içeriyorsa tablo verisinden soru üret. Örnek KÖTÜ alıntı: "Tetik Kuvveti 2000 g 2000 g 3000 g" (hangi değerin hangi silaha ait olduğu belirsiz). Örnek İYİ alıntı: "Zigana T Etkili Menzil 100 m" (etiket ve değer birlikte).

## ÇIKTI BİÇİMİ
Yalnızca şu şemada, başka hiçbir açıklama olmadan bir JSON nesnesi döndür:
{
  "sorular": [
    {
      "question": "<kısa, doğal Türkçe soru>",
      "answer": "<parafraz edilmiş, doğru, kendi cümlelerinle cevap>",
      "sayfa": <cevabın alındığı basılı SAYFA numarası (tam sayı)>,
      "pdf_page_index": <o sayfanın etiketteki fiziksel numarası (tam sayı)>,
      "kaynak_alinti": "<sayfadan birebir alıntı>",
      "dogrulanan_bilgi": ["<olgu 1>", "<olgu 2>"],
      "zorluk": "kolay|orta|zor",
      "soru_tipi": "dogrudan|senaryo|prosedur|karsilastirma|cause_effect",
      "sayfa_uyarisi": ""
    }
  ]
}
Hiç uygun soru yoksa {"sorular": []} döndür.
"""

DIFFICULTY_HINTS = {
    "easy": "Mümkünse bu parçadan en az bir 'kolay' soru üret.",
    "hard": "Mümkünse bu parçadan en az bir 'zor' soru üret.",
}


def build_user_prompt(chunk: dict[str, Any], difficulty_hint: str) -> str:
    pages = ", ".join(
        f"SAYFA {p['sayfa']} (pdf_page_index {p['pdf_page_index']})" for p in chunk["pages"]
    )
    lines = [
        f"Kitap: {chunk['kaynak']}",
        f"Bu parçadaki sayfalar: {pages}.",
        "Aşağıdaki sayfa görüntülerini incele ve talimatlara uygun soru-cevap çiftlerini üret.",
    ]
    if difficulty_hint and difficulty_hint in DIFFICULTY_HINTS:
        lines.append(DIFFICULTY_HINTS[difficulty_hint])
    return "\n".join(lines)


def encode_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_messages(chunk: dict[str, Any], difficulty_hint: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": build_user_prompt(chunk, difficulty_hint)}]
    for image_rel in chunk["images"]:
        content.append(
            {"type": "image_url", "image_url": {"url": encode_image(REPO_ROOT / image_rel)}}
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def normalize_questions(raw: dict[str, Any], chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Coerce one API response into clean CSV rows; drop malformed entries."""
    valid_pages = {p["sayfa"]: p["pdf_page_index"] for p in chunk["pages"]}
    rows: list[dict[str, Any]] = []
    for item in raw.get("sorular", []) or []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        try:
            sayfa = int(item.get("sayfa"))
        except (TypeError, ValueError):
            sayfa = 0
        # Prefer the manifest's physical index for the cited printed page.
        pdf_page_index = valid_pages.get(sayfa)
        if pdf_page_index is None:
            try:
                pdf_page_index = int(item.get("pdf_page_index"))
            except (TypeError, ValueError):
                pdf_page_index = 0
        facts = item.get("dogrulanan_bilgi")
        if isinstance(facts, list):
            facts_str = " | ".join(str(f).strip() for f in facts if str(f).strip())
        else:
            facts_str = str(facts or "").strip()
        zorluk = str(item.get("zorluk", "")).strip().lower()
        tipi = str(item.get("soru_tipi", "")).strip().lower()
        rows.append(
            {
                "kaynak": chunk["kaynak"],
                "question": question,
                "answer": answer,
                "sayfa": sayfa,
                "pdf_page_index": pdf_page_index,
                "kaynak_alinti": str(item.get("kaynak_alinti", "")).strip(),
                "dogrulanan_bilgi": facts_str,
                "zorluk": zorluk if zorluk in ALLOWED_ZORLUK else "",
                "soru_tipi": tipi if tipi in ALLOWED_TIPI else "",
                "sayfa_uyarisi": str(item.get("sayfa_uyarisi", "")).strip(),
                "chunk": chunk["chunk"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/qa_chunks/manifest.json")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/results/qa_generation")
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--only-kaynak", default="", help="Limit to one source PDF filename.")
    p.add_argument("--limit", type=int, default=0, help="Process at most N chunks (0 = all).")
    p.add_argument(
        "--difficulty-hint",
        choices=["", "easy", "hard"],
        default="",
        help="Optional per-chunk nudge toward an easy or hard question. Default: emergent.",
    )
    p.add_argument("--dry-run", action="store_true", help="Assemble+print one payload; no API call.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    chunks = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.only_kaynak:
        chunks = [c for c in chunks if c["kaynak"] == args.only_kaynak]
    if args.limit > 0:
        chunks = chunks[: args.limit]

    if args.dry_run:
        chunk = chunks[0]
        messages = build_messages(chunk, args.difficulty_hint)
        preview = {
            "model": args.model,
            "temperature": args.temperature,
            "chunk": chunk["chunk"],
            "system_prompt_chars": len(SYSTEM_PROMPT),
            "user_prompt": messages[1]["content"][0]["text"],
            "image_count": len(chunk["images"]),
            "first_image_prefix": messages[1]["content"][1]["image_url"]["url"][:60] + "...",
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is not set. Run: OPENAI_API_KEY=sk-... python3 scripts/qa/gen_qa_from_images.py", file=sys.stderr)
        return 1
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "generated_qa.csv"
    log_path = args.out_dir / "run.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    all_rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        slug = chunk["chunk"].replace("/", "__")
        raw_path = raw_dir / f"{slug}.json"
        if raw_path.exists():  # resume: reuse cached response
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            all_rows.extend(normalize_questions(raw, chunk))
            log(f"[{idx}/{len(chunks)}] {chunk['chunk']}: cached")
            continue
        try:
            response = client.chat.completions.create(
                model=args.model,
                temperature=args.temperature,
                response_format={"type": "json_object"},
                messages=build_messages(chunk, args.difficulty_hint),
            )
            content = response.choices[0].message.content or "{}"
            raw = json.loads(content)
            raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            rows = normalize_questions(raw, chunk)
            all_rows.extend(rows)
            log(f"[{idx}/{len(chunks)}] {chunk['chunk']}: {len(rows)} soru")
        except Exception as exc:  # keep going; a failed chunk can be re-run later
            log(f"[{idx}/{len(chunks)}] {chunk['chunk']}: ERROR {type(exc).__name__}: {exc}")
        write_csv(out_csv, all_rows)  # flush after every chunk

    log(f"DONE: {len(all_rows)} soru -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
