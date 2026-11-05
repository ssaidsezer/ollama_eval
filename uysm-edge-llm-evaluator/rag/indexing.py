from __future__ import annotations

import time
from typing import List, Sequence

from qdrant_client.http import models as qmodels

from .chunking import _extract_pdf_chunks
from .config import (
    DEFAULT_COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    QDRANT_BATCH_SIZE,
    QDRANT_URL,
)
from .embeddings import get_embeddings
from .bm25 import invalidate_bm25_cache
from .qdrant_store import ensure_collection, get_qdrant_client


def index_pdfs(
    pdf_paths: Sequence[str],
    collection_name: str = DEFAULT_COLLECTION_NAME or "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    qdrant_url: str = QDRANT_URL or "",
    embed_model: str = OLLAMA_EMBED_MODEL,
    ollama_base_url: str | None = None,
    progress_callback=None,
) -> dict:
    """
    PDF'leri Ollama embedding ile embed edip Qdrant'a indeksler.

    progress_callback(phase, current, total, elapsed_sec):
        phase: "pdf_extract" | "ollama_embed" | "qdrant_upsert"
        current, total: ilerleme sayaçları
        elapsed_sec: o adımın süresi
    """
    timings: dict = {
        "total_chunks": 0,
        "pdf_extract_sec": 0.0,
        "ollama_embed_sec": 0.0,
        "qdrant_upsert_sec": 0.0,
        "total_sec": 0.0,
    }
    total_t0 = time.time()

    t0 = time.time()
    all_chunks: List[dict] = []
    for i, path in enumerate(pdf_paths):
        all_chunks.extend(
            _extract_pdf_chunks(
                path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
        elapsed = time.time() - t0
        if progress_callback:
            progress_callback("pdf_extract", i + 1, len(pdf_paths), elapsed)

    timings["pdf_extract_sec"] = round(time.time() - t0, 2)

    if not all_chunks:
        timings["total_sec"] = round(time.time() - total_t0, 2)
        return timings

    texts = [c["text"] for c in all_chunks]
    total_texts = len(texts)
    timings["total_chunks"] = total_texts

    t0 = time.time()
    all_embeddings: List[List[float]] = []
    for start in range(0, total_texts, EMBED_BATCH_SIZE):
        batch_texts = texts[start: start + EMBED_BATCH_SIZE]
        batch_embeddings = get_embeddings(
            batch_texts,
            model=embed_model,
            base_url=ollama_base_url or OLLAMA_BASE_URL,
        )
        all_embeddings.extend(batch_embeddings)
        elapsed = time.time() - t0
        done = min(start + EMBED_BATCH_SIZE, total_texts)
        if progress_callback:
            progress_callback("ollama_embed", done, total_texts, elapsed)

    timings["ollama_embed_sec"] = round(time.time() - t0, 2)

    vector_size = len(all_embeddings[0])

    client = get_qdrant_client(url=qdrant_url)
    ensure_collection(client, collection_name, vector_size=vector_size)

    unique_sources = sorted({str(path) for path in pdf_paths if str(path).strip()})
    if unique_sources:
        source_filter = qmodels.Filter(
            should=[
                qmodels.FieldCondition(
                    key="source",
                    match=qmodels.MatchValue(value=source),
                )
                for source in unique_sources
            ]
        )
        client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(filter=source_filter),
            wait=True,
        )

    points = [
        qmodels.PointStruct(
            id=all_chunks[i]["id"],
            vector=all_embeddings[i],
            payload={
                **all_chunks[i]["metadata"],
                "text": all_chunks[i]["text"],
            },
        )
        for i in range(len(all_chunks))
    ]

    t0 = time.time()
    for start in range(0, len(points), QDRANT_BATCH_SIZE):
        batch = points[start: start + QDRANT_BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        elapsed = time.time() - t0
        done = min(start + QDRANT_BATCH_SIZE, len(points))
        if progress_callback:
            progress_callback("qdrant_upsert", done, len(points), elapsed)

    timings["qdrant_upsert_sec"] = round(time.time() - t0, 2)
    invalidate_bm25_cache(qdrant_url, collection_name)
    timings["total_sec"] = round(time.time() - total_t0, 2)

    return timings
