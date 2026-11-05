from __future__ import annotations

import time
from typing import List

import qdrant_client as _qdrant_pkg

from .chunking import _normalize_text_for_dedup
from .config import DEFAULT_COLLECTION_NAME, OLLAMA_EMBED_MODEL, QDRANT_URL, _debug_log
from .embeddings import get_embeddings
from .qdrant_store import (
    _hit_payload_and_score,
    _qdrant_query_points_compat,
    _retrieve_parent_docs,
    _select_unique_parent_hits,
    get_qdrant_client,
)


def _with_payload_metadata(base: dict, payload: dict, fields: tuple[str, ...]) -> dict:
    for field in fields:
        value = payload.get(field)
        if value is not None:
            base[field] = value
    return base


def retrieve_context(
    question: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
    score_threshold: float = 0.55,
    embedding_timeout: int = 300,
) -> str:
    """
    Soruyu Ollama ile embed edip Qdrant'tan en yakın k chunk'ı döndürür.
    """
    run_id = f"retrieve_context_{int(time.time() * 1000)}"
    question_embedding = get_embeddings([question], timeout=embedding_timeout)[0]

    client = get_qdrant_client(url=qdrant_url)
    _debug_log(
        run_id,
        "H1",
        "rag/retrieval.py:retrieve_context:before_query",
        "qdrant_client_capabilities",
        {
            "qdrant_version": getattr(_qdrant_pkg, "__version__", "unknown"),
            "client_type": type(client).__name__,
            "has_query_points": hasattr(client, "query_points"),
            "has_search": hasattr(client, "search"),
            "embedding_dim": len(question_embedding) if isinstance(question_embedding, list) else -1,
            "collection_name": collection_name,
        },
    )

    results = _qdrant_query_points_compat(
        client=client,
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=k,
        score_threshold=score_threshold,
        run_id=run_id,
        location="rag/retrieval.py:retrieve_context:query_dispatch",
    )

    docs: List[str] = []
    seen: set[str] = set()
    for hit in results:
        payload, _score = _hit_payload_and_score(hit)
        text = payload.get("text", "")
        if isinstance(text, str) and text.strip():
            key = _normalize_text_for_dedup(text)
            if key and key not in seen:
                seen.add(key)
                docs.append(text)

    if not docs:
        return ""

    return "\n\n".join(docs)


def retrieve_chunks(
    question: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
    score_threshold: float = 0.55,
    embed_model: str | None = None,
    embedding_timeout: int = 300,
) -> List[dict]:
    """
    Soruyu Ollama ile embed edip Qdrant'tan en yakın k chunk'ı döndürür.
    Her eleman {"text": str, "score": float, ...metadata} dict'idir.
    """
    run_id = f"retrieve_chunks_{int(time.time() * 1000)}"
    question_embedding = get_embeddings(
        [question], model=embed_model or OLLAMA_EMBED_MODEL, timeout=embedding_timeout
    )[0]

    client = get_qdrant_client(url=qdrant_url)
    _debug_log(
        run_id,
        "H1",
        "rag/retrieval.py:retrieve_chunks:before_query",
        "qdrant_client_capabilities",
        {
            "qdrant_version": getattr(_qdrant_pkg, "__version__", "unknown"),
            "client_type": type(client).__name__,
            "has_query_points": hasattr(client, "query_points"),
            "has_search": hasattr(client, "search"),
            "embedding_dim": len(question_embedding) if isinstance(question_embedding, list) else -1,
            "collection_name": collection_name,
            "embed_model": embed_model or OLLAMA_EMBED_MODEL,
        },
    )

    results = _qdrant_query_points_compat(
        client=client,
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=k,
        score_threshold=score_threshold,
        run_id=run_id,
        location="rag/retrieval.py:retrieve_chunks:query_dispatch",
    )

    docs: List[dict] = []
    seen: set[str] = set()
    for hit in results:
        payload, score = _hit_payload_and_score(hit)
        text = payload.get("text", "")
        if isinstance(text, str) and text.strip():
            key = _normalize_text_for_dedup(text)
            if key and key not in seen:
                seen.add(key)
                docs.append(
                    _with_payload_metadata(
                        {"text": text, "score": score},
                        payload,
                        ("source", "page", "chunk"),
                    )
                )

    return docs


def retrieve_chunks_smart(
    question: str,
    base_collection: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
    score_threshold: float = 0.55,
    embed_model: str | None = None,
    embedding_timeout: int = 300,
) -> List[dict]:
    """
    Smart retrieval: children koleksiyonunda vektör araması yapar,
    her eşleşen child chunk için parent_id ile tam parent bloğu Qdrant'tan çeker.
    """
    if k <= 0:
        return []

    children_col = f"{base_collection}_children"
    parents_col = f"{base_collection}_parents"

    run_id = f"retrieve_chunks_smart_{int(time.time() * 1000)}"
    question_embedding = get_embeddings(
        [question], model=embed_model or OLLAMA_EMBED_MODEL, timeout=embedding_timeout
    )[0]
    client = get_qdrant_client(url=qdrant_url)

    batch_size = max(k * 4, 16)
    offset = 0
    candidates: list[tuple[dict, float]] = []
    parent_ids_ordered: List[str] = []
    child_info: dict = {}

    while len(parent_ids_ordered) < k:
        results = _qdrant_query_points_compat(
            client=client,
            collection_name=children_col,
            query_vector=question_embedding,
            limit=batch_size,
            score_threshold=score_threshold,
            run_id=run_id,
            location="rag/retrieval.py:retrieve_chunks_smart:query_dispatch",
            offset=offset,
        )
        if not results:
            break

        candidates.extend(_hit_payload_and_score(hit) for hit in results)
        parent_ids_ordered, child_info = _select_unique_parent_hits(candidates, k)
        if len(parent_ids_ordered) >= k or len(results) < batch_size:
            break
        offset += len(results)

    if not parent_ids_ordered:
        return []

    return _retrieve_parent_docs(client, parents_col, parent_ids_ordered, child_info)
