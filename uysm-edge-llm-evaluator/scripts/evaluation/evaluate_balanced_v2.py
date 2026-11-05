#!/usr/bin/env python3
"""Resume-safe balanced-v2 evaluation for inference CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reporting.report_utils import sniff_delimiter  # noqa: E402


PROMPT = """Sen tutarlı ve dengeli bir Türkçe soru-cevap değerlendiricisisin.
Soruyu, referans cevabı, cevaplanabilirlik etiketini ve model cevabını değerlendir.

Birbirinden bağımsız üç puan ver:
1. correctness_score (0-10): Model cevabındaki olgular ve önerilen eylemler doğru mu?
2. completeness_score (0-10): Referans cevaptaki gerekli ana noktaların ne kadarı karşılanmış?
3. groundedness_score (0-10): Model cevabındaki olgusal iddiaların ne kadarı verilen referans tarafından destekleniyor?

Doğru ve zararsız ek bağlam referansta yazmıyor diye correctness_score'u düşürme. Böyle ek bilgi
groundedness_score'u bir miktar düşürebilir. Yanlış, referansla çelişen, güvensiz veya uygulanacak
eylemi değiştiren desteksiz bilgi correctness_score'u düşürür ve hallucination_risk'i yükseltir.
Bir iddianın yanlış olduğunu belirleyemiyorsan onu sırf referansta yok diye halüsinasyon sayma;
groundedness puanına yansıt.

Puan çıpaları:
- 9-10: doğru ve gerekli noktaların neredeyse tamamını kapsıyor.
- 7-8: çoğunlukla doğru, küçük eksikler var.
- 4-6: kısmen doğru, önemli eksikler var.
- 1-3: az miktarda kullanılabilir doğru içerik veya büyük yanlış anlama var.
- 0: tamamen yanlış, ilgisiz veya doğrudan tehlikeli.

Cevaplanabilir=hayir ise yalnızca belgelerde bilginin bulunmadığını açıkça belirten ve olgusal cevap
uydurmayan yanıt pass olabilir. Belirsiz bir bilmiyorum yanıtı partial olabilir. Uydurulan doz veya
özgül olgusal cevap fail ve hallucination_risk=high olmalıdır.

Yalnızca şu alanları içeren JSON döndür:
correctness_score, completeness_score, groundedness_score,
hallucination_risk (low|medium|high), verdict (pass|partial|fail), reason.
"""

EVAL_FIELDS = [
    "correctness_score", "completeness_score", "groundedness_score",
    "hallucination_risk", "verdict", "evaluation_reason",
    "evaluation_status_balanced_v2", "evaluation_duration_seconds_balanced_v2",
    "evaluation_error_type_balanced_v2", "evaluation_error_message_balanced_v2",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    with path.open(encoding="utf-8-sig") as handle:
        delimiter = sniff_delimiter(handle.readline())
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter)), delimiter


def atomic_csv(path: Path, rows: list[dict[str, object]], fields: list[str], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["question_index"])))
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, path)


def atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, path)


def parse_response(content: str) -> dict[str, object]:
    value = json.loads(content)
    for key in ("correctness_score", "completeness_score", "groundedness_score"):
        value[key] = float(value[key])
        if not 0 <= value[key] <= 10:
            raise ValueError(f"invalid {key}")
    value["hallucination_risk"] = str(value["hallucination_risk"]).lower()
    value["verdict"] = str(value["verdict"]).lower()
    if value["hallucination_risk"] not in {"low", "medium", "high"}:
        raise ValueError("invalid hallucination_risk")
    if value["verdict"] not in {"pass", "partial", "fail"}:
        raise ValueError("invalid verdict")
    value["evaluation_reason"] = " ".join(str(value.pop("reason")).split())
    return value


def normalize_answerability(value: str) -> str:
    normalized = value.strip().lower().replace("ı", "i")
    return "hayir" if normalized in {"hayir", "no", "false", "0"} else "evet"


def average(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--qa-csv", default="data/qa/final_qa_150.csv")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=150,
        help="Required row count for both the inference CSV and the QA source (default: 150).",
    )
    args = parser.parse_args()

    load_dotenv(Path.cwd() / ".env")
    source = Path(args.input_csv)
    raw, delimiter = read_csv(source)
    expected = args.expected_rows
    if len(raw) != expected or any(row.get("inference_status") not in {"success", "ok"} for row in raw):
        raise RuntimeError(f"requires {expected} successful inference rows: {source}")
    qa, _ = read_csv(Path(args.qa_csv))
    if len(qa) != expected:
        raise RuntimeError(f"QA source must contain exactly {expected} rows")

    output = Path(args.output_csv) if args.output_csv else source.with_name("evaluated_results_balanced_v2.csv")
    summary_path = Path(args.summary) if args.summary else source.with_name("summary_balanced_v2.json")
    existing, _ = read_csv(output) if output.exists() else ([], delimiter)
    completed = {
        int(row["question_index"]): row for row in existing
        if row.get("evaluation_status_balanced_v2") == "success"
    }
    fields = list(raw[0]) + [
        field for field in ("answerability_effective", *EVAL_FIELDS) if field not in raw[0]
    ]
    client = OpenAI()

    for row in raw:
        index = int(row["question_index"])
        if index in completed:
            continue
        reference = row.get("reference_answer") or row.get("answer") or qa[index]["answer"]
        answerability = normalize_answerability(row.get("answerability") or qa[index].get("cevaplanabilir", "evet"))
        row = {**row, "answerability_effective": answerability}
        started = time.perf_counter()
        error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    reasoning_effort="high",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": (
                            f"Soru: {row['question']}\nReferans cevap: {reference}\n"
                            f"Cevaplanabilir: {answerability}\nModel cevabı: {row['model_answer']}"
                        )},
                    ],
                )
                parsed = parse_response(response.choices[0].message.content or "")
                result = {
                    **row, **parsed,
                    "evaluation_status_balanced_v2": "success",
                    "evaluation_duration_seconds_balanced_v2": round(time.perf_counter() - started, 6),
                    "evaluation_error_type_balanced_v2": "",
                    "evaluation_error_message_balanced_v2": "",
                }
                break
            except Exception as exc:
                error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        else:
            result = {
                **row,
                "evaluation_status_balanced_v2": "failed",
                "evaluation_duration_seconds_balanced_v2": round(time.perf_counter() - started, 6),
                "evaluation_error_type_balanced_v2": type(error).__name__ if error else "UnknownError",
                "evaluation_error_message_balanced_v2": str(error or "unknown error"),
            }
        completed[index] = result
        atomic_csv(output, list(completed.values()), fields, delimiter)
        print(f"{source.parent.name} {index + 1}/{expected} {result['evaluation_status_balanced_v2']}", flush=True)

    rows = [completed[index] for index in sorted(completed)]
    successful = [row for row in rows if row.get("evaluation_status_balanced_v2") == "success"]
    answerable: list[dict[str, object]] = []
    unanswerable: list[dict[str, object]] = []
    for row in successful:
        # Bucket on what the judge was actually told, not on a second lookup that may disagree.
        effective = row.get("answerability_effective") or normalize_answerability(
            str(qa[int(row["question_index"])].get("cevaplanabilir", "evet"))
        )
        (answerable if effective == "evet" else unanswerable).append(row)
    summary = {
        "rubric": "balanced-v2",
        "eval_model": args.model,
        "reasoning_effort": "high",
        "source_csv": str(source),
        "counts": {"total": len(rows), "successful": len(successful), "failed": len(rows) - len(successful)},
        "evaluation": {
            "correctness_score_mean": average(successful, "correctness_score"),
            "completeness_score_mean": average(successful, "completeness_score"),
            "groundedness_score_mean": average(successful, "groundedness_score"),
            "verdicts": dict(Counter(str(row["verdict"]) for row in successful)),
            "hallucination_risk": dict(Counter(str(row["hallucination_risk"]) for row in successful)),
            "answerable": {
                "count": len(answerable),
                "correctness_score_mean": average(answerable, "correctness_score"),
                "completeness_score_mean": average(answerable, "completeness_score"),
                "groundedness_score_mean": average(answerable, "groundedness_score"),
            },
            "unanswerable": {
                "count": len(unanswerable),
                "grounded_refusal_accuracy": (sum(row["verdict"] == "pass" for row in unanswerable) / len(unanswerable)) if unanswerable else None,
                "hallucination_rate": (sum(row["hallucination_risk"] == "high" for row in unanswerable) / len(unanswerable)) if unanswerable else None,
            },
        },
    }
    atomic_json(summary_path, summary)
    print(f"complete {source}: {len(successful)}/{expected}", flush=True)


if __name__ == "__main__":
    main()
