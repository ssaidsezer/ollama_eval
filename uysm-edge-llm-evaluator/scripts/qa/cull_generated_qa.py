#!/usr/bin/env python3
"""Report on, dedup, and flag the generated QA pool before manual culling.

Reads every cached chunk response from gen_qa_from_images.py, then:
- prints distributions (per source / zorluk / soru_tipi),
- flags near-duplicate questions (summary-vs-detail overlap),
- flags likely-extractive answers (answer ~= kaynak_alinti),
- flags rows whose dogrulanan_bilgi facts are absent from the answer,
- flags rows with a non-empty sayfa_uyarisi (page-number mismatch),
- writes an annotated CSV with a `flags` column and a Markdown report.

This does NOT auto-delete anything. It surfaces candidates for review; final
dataset sizing and source balance remain explicit downstream decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

CSV_FIELDS = (
    "kaynak", "question", "answer", "sayfa", "pdf_page_index",
    "kaynak_alinti", "dogrulanan_bilgi", "zorluk", "soru_tipi",
    "sayfa_uyarisi", "chunk", "flags",
)

_TOKEN_RE = re.compile(r"[^\wçğıöşüÇĞİÖŞÜ]+", re.UNICODE)
_STOP = frozenset((
    "acaba", "ama", "ancak", "bir", "bu", "da", "de", "gibi", "hangi", "için",
    "ile", "ise", "mi", "mı", "mu", "mü", "nasıl", "ne", "neden", "ve", "veya",
    "ya", "zaman", "yer", "olarak", "olan", "göre", "kadar", "var", "ben", "elimde",
))


def norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


def tokens(value: str) -> set[str]:
    return {t for t in _TOKEN_RE.sub(" ", norm(value)).split() if t and t not in _STOP}


def question_similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    return max(jaccard, SequenceMatcher(None, norm(a), norm(b)).ratio())


def load_pool(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        chunk = path.stem.replace("__", "/")
        kaynak = (
            "taktik_muharebe_yarali_bakimi_el_kitabi.pdf"
            if "taktik" in path.stem
            else "askeri_egitim_kitabi.pdf"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("sorular", []) or []:
            if not isinstance(item, dict):
                continue
            facts = item.get("dogrulanan_bilgi")
            facts_str = (
                " | ".join(str(f).strip() for f in facts if str(f).strip())
                if isinstance(facts, list)
                else str(facts or "").strip()
            )
            rows.append(
                {
                    "kaynak": kaynak,
                    "question": str(item.get("question", "")).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "sayfa": item.get("sayfa", ""),
                    "pdf_page_index": item.get("pdf_page_index", ""),
                    "kaynak_alinti": str(item.get("kaynak_alinti", "")).strip(),
                    "dogrulanan_bilgi": facts_str,
                    "zorluk": str(item.get("zorluk", "")).strip(),
                    "soru_tipi": str(item.get("soru_tipi", "")).strip(),
                    "sayfa_uyarisi": str(item.get("sayfa_uyarisi", "")).strip(),
                    "chunk": chunk,
                }
            )
    return rows


def flag_rows(rows: list[dict[str, Any]], *, dup_threshold: float) -> None:
    """Attach a `flags` field to each row (semicolon-separated issue codes)."""
    # Extractive answers + missing facts + page warnings (per-row).
    for row in rows:
        flags: list[str] = []
        answer, alinti = norm(row["answer"]), norm(row["kaynak_alinti"])
        if answer and alinti and SequenceMatcher(None, answer, alinti).ratio() > 0.9:
            flags.append("extractive_answer")
        # A fact is "present" if most of its content tokens appear in the answer.
        # Substring matching is wrong here: synthesized answers paraphrase the
        # fact (imperative -> plain, reordered words), so we check token overlap.
        answer_tokens = tokens(row["answer"])
        for fact in row["dogrulanan_bilgi"].split(" | "):
            fact_tokens = tokens(fact)
            if not fact_tokens:
                continue
            coverage = len(fact_tokens & answer_tokens) / len(fact_tokens)
            if coverage < 0.5:
                flags.append("fact_weak_in_answer")
                break
        if row["sayfa_uyarisi"]:
            flags.append("page_warning")
        if not row["kaynak_alinti"]:
            flags.append("no_excerpt")
        row["_flags"] = flags

    # Near-duplicate questions (within each source book).
    by_source: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_source[row["kaynak"]].append(idx)
    for indices in by_source.values():
        for i_pos, i in enumerate(indices):
            for j in indices[i_pos + 1 :]:
                if question_similarity(rows[i]["question"], rows[j]["question"]) >= dup_threshold:
                    rows[i]["_flags"].append(f"dup~{rows[j]['chunk'].split('/')[-1]}")
                    rows[j]["_flags"].append(f"dup~{rows[i]['chunk'].split('/')[-1]}")

    for row in rows:
        row["flags"] = ";".join(dict.fromkeys(row.pop("_flags")))  # dedupe, keep order


def build_report(rows: list[dict[str, Any]]) -> str:
    # Display-only English labels for the Turkish tag values in the data.
    # The CSV keeps the original Turkish; only the report reads in English.
    field_titles = {"zorluk": "Difficulty", "soru_tipi": "Question type"}
    value_labels = {
        "kolay": "easy", "orta": "medium", "zor": "hard",
        "dogrudan": "direct", "senaryo": "scenario", "prosedur": "procedure",
        "karsilastirma": "comparison", "cause_effect": "cause_effect",
    }
    lines = ["# Generated QA pool report", "", f"Total questions: {len(rows)}", ""]
    per_source = Counter(r["kaynak"] for r in rows)
    lines += ["## Per source", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(per_source.items())]
    for field in ("zorluk", "soru_tipi"):
        lines += ["", f"## {field_titles[field]}", ""]
        for k, v in sorted(Counter(r[field] for r in rows).items()):
            label = value_labels.get(k, k) if k else "(empty)"
            lines += [f"- {label}: {v}"]
    flag_counter: Counter[str] = Counter()
    for row in rows:
        for f in row["flags"].split(";"):
            base = f.split("~")[0]
            if base:
                flag_counter[base] += 1
    lines += ["", "## Flags (rows affected)", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(flag_counter.items())] or ["- none"]
    clean = sum(1 for r in rows if not r["flags"])
    lines += ["", f"## Clean rows (no flags): {clean}", ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/results/qa_generation/raw")
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "data/results/qa_generation/pool_annotated.csv")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "data/results/qa_generation/pool_report.md")
    parser.add_argument("--dup-threshold", type=float, default=0.6)
    args = parser.parse_args()

    rows = load_pool(args.raw_dir)
    if not rows:
        print(f"No cached responses in {args.raw_dir}")
        return 1
    flag_rows(rows, dup_threshold=args.dup_threshold)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    args.report.write_text(build_report(rows), encoding="utf-8")

    print(build_report(rows))
    print(f"Annotated pool: {args.out_csv}")
    print(f"Report:         {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
