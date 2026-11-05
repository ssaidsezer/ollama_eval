#!/usr/bin/env python3
"""Sample a balanced final QA set from the clean pool + the unanswerable set.

Selects N answerable questions per source book from the flag-free rows of
pool_annotated.csv, spread across difficulty (keeping all scarce 'zor'), then
appends the verified unanswerable questions. Output is one CSV under data/qa/.

Default target: 45 TCCC + 45 Askeri + 10 unanswerable = 100.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_FIELDS = (
    "kaynak", "question", "answer", "sayfa", "pdf_page_index",
    "kaynak_alinti", "dogrulanan_bilgi", "zorluk", "soru_tipi",
    "cevaplanabilir", "chunk",
)


def book_of(row: dict[str, str]) -> str:
    return "tccc" if "taktik" in row["kaynak"] else "askeri"


def load(path: Path, delimiter: str = ";") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp, delimiter=delimiter))


def sample_book(rows: list[dict[str, str]], n: int, rng: random.Random) -> list[dict[str, str]]:
    """Pick n rows spread across difficulty; take all 'zor', then fill proportionally."""
    by_diff: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_diff[row.get("zorluk", "")].append(row)
    for bucket in by_diff.values():
        rng.shuffle(bucket)

    selected: list[dict[str, str]] = []
    # Take all 'zor' first (scarce), then medium, then easy, proportional to what's left.
    selected.extend(by_diff.get("zor", []))
    remaining = n - len(selected)
    filler = by_diff.get("orta", []) + by_diff.get("kolay", []) + by_diff.get("", [])
    rng.shuffle(filler)
    selected.extend(filler[: max(remaining, 0)])
    return selected[:n]


def to_out_row(row: dict[str, str], cevaplanabilir: str) -> dict[str, Any]:
    return {
        "kaynak": row.get("kaynak", ""),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "sayfa": row.get("sayfa", ""),
        "pdf_page_index": row.get("pdf_page_index", ""),
        "kaynak_alinti": row.get("kaynak_alinti", ""),
        "dogrulanan_bilgi": row.get("dogrulanan_bilgi", ""),
        "zorluk": row.get("zorluk", ""),
        "soru_tipi": row.get("soru_tipi", ""),
        "cevaplanabilir": cevaplanabilir,
        "chunk": row.get("chunk", ""),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=REPO_ROOT / "data/qa/generated_312/pool_annotated.csv")
    parser.add_argument("--unanswerable", type=Path, default=REPO_ROOT / "data/qa/generated_312/unanswerable_10.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/qa/final_qa_100.csv")
    parser.add_argument("--per-book", type=int, default=45, help="Answerable questions per source book.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for reproducibility.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    rng = random.Random(args.seed)
    pool = load(args.pool)
    clean = [r for r in pool if not (r.get("flags") or "").strip()]

    selected: list[dict[str, Any]] = []
    for book in ("tccc", "askeri"):
        book_rows = [r for r in clean if book_of(r) == book]
        if len(book_rows) < args.per_book:
            raise SystemExit(
                f"Not enough clean {book} rows: have {len(book_rows)}, need {args.per_book}."
            )
        selected.extend(to_out_row(r, "evet") for r in sample_book(book_rows, args.per_book, rng))

    for row in load(args.unanswerable):
        selected.append(to_out_row(row, "hayir"))

    # Comma-delimited + UTF-8 (no BOM) to match the legacy question_answers_*.csv
    # format that pipeline.load_questions reads with a default csv.DictReader.
    # Fields containing commas are auto-quoted by csv.writer.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    # Summary
    ans = [r for r in selected if r["cevaplanabilir"] == "evet"]
    print(f"Final set: {len(selected)} rows -> {args.out}")
    print(f"  answerable:   {len(ans)}  ({dict(Counter(book_of(r) for r in ans))})")
    print(f"  unanswerable: {len(selected) - len(ans)}")
    print(f"  difficulty:   {dict(Counter(r['zorluk'] for r in ans))}")
    print(f"  type:         {dict(Counter(r['soru_tipi'] for r in ans))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
