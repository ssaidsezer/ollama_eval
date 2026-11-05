#!/usr/bin/env python3
"""Measure retrieval-only latency for the four frozen RAG methods."""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.bm25 import retrieve_chunks_bm25, retrieve_chunks_bm25_smart
from rag.retrieval import retrieve_chunks, retrieve_chunks_smart


QA = Path("data/qa/final_qa_150.csv")
OUT = Path("data/results/rag_methods_gemma_q4_20260722_150/retrieval_timing")
QDRANT = "http://192.168.1.149:8873"
CLASSIC = "documents_embeddinggemma_300m_1000c_200ov"
SMART = "documents_embeddinggemma_300m_3000p_500c_100ov"
EMBED = "embeddinggemma:300m"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    with QA.open(encoding="utf-8-sig", newline="") as handle:
        questions = [row["question"] for row in csv.DictReader(handle)]
    methods = {
        "classic_vector": lambda q: retrieve_chunks(q, collection_name=CLASSIC, k=10, qdrant_url=QDRANT, score_threshold=0.55, embed_model=EMBED),
        "classic_bm25": lambda q: retrieve_chunks_bm25(q, collection_name=CLASSIC, k=10, qdrant_url=QDRANT),
        "smart_vector": lambda q: retrieve_chunks_smart(q, base_collection=SMART, k=10, qdrant_url=QDRANT, score_threshold=0.55, embed_model=EMBED),
        "smart_bm25": lambda q: retrieve_chunks_bm25_smart(q, base_collection=SMART, k=10, qdrant_url=QDRANT),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"config": {"questions": len(questions), "k": 10, "score_threshold": 0.55, "embed_model": EMBED, "qdrant_url": QDRANT}, "methods": {}}
    for name, retrieve in methods.items():
        output_path = OUT / f"{name}.csv"
        rows = []
        if output_path.exists():
            with output_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        completed = {int(row["question_index"]) for row in rows}
        for index, question in enumerate(questions):
            if index in completed:
                continue
            for attempt in range(3):
                started = time.perf_counter()
                try:
                    chunks = retrieve(question)
                    elapsed = time.perf_counter() - started
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            rows.append({"question_index": index, "retrieval_time_seconds": round(elapsed, 6), "retrieved_chunk_size": len(chunks)})
            with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(sorted(rows, key=lambda row: int(row["question_index"])))
            print(f"{name} {index + 1}/150 {elapsed:.3f}s chunks={len(chunks)}", flush=True)
        values = [float(row["retrieval_time_seconds"]) for row in rows]
        summary["methods"][name] = {
            "mean_seconds": statistics.fmean(values), "median_seconds": statistics.median(values),
            "p95_seconds": percentile(values, 0.95), "max_seconds": max(values),
            "mean_chunks": statistics.fmean(int(row["retrieved_chunk_size"]) for row in rows),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
