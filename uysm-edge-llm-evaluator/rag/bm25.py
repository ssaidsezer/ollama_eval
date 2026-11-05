from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

from .chunking import _normalize_text_for_dedup
from .config import DEFAULT_COLLECTION_NAME, QDRANT_URL
from .qdrant_store import (
    _retrieve_parent_docs,
    _scroll_all_payloads,
    _select_unique_parent_hits,
    get_qdrant_client,
)


_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_BM25_CACHE_MAX_ENTRIES = 8


@dataclass(frozen=True)
class _CachedBM25Index:
    payloads: tuple[dict, ...]
    corpus_texts: tuple[str, ...]
    index: object | None


_BM25_CACHE: OrderedDict[tuple[str, str], _CachedBM25Index] = OrderedDict()


def _bm25_tokenize(text: str) -> List[str]:
    """Split Unicode words while normalizing case and punctuation."""
    return _TOKEN_RE.findall(text.casefold())


def _bm25_cache_key(qdrant_url: str, collection_name: str) -> tuple[str, str]:
    return ((qdrant_url or QDRANT_URL).rstrip("/"), collection_name)


def _get_bm25_index(qdrant_url: str, collection_name: str) -> _CachedBM25Index:
    key = _bm25_cache_key(qdrant_url, collection_name)
    cached = _BM25_CACHE.get(key)
    if cached is not None:
        _BM25_CACHE.move_to_end(key)
        return cached

    client = get_qdrant_client(url=qdrant_url)
    payloads = tuple(dict(payload) for payload in _scroll_all_payloads(client, collection_name))
    corpus_texts = tuple(str(payload.get("text", "")) for payload in payloads)
    index = _BM25Okapi([_bm25_tokenize(text) for text in corpus_texts]) if payloads else None
    cached = _CachedBM25Index(payloads=payloads, corpus_texts=corpus_texts, index=index)
    _BM25_CACHE[key] = cached
    _BM25_CACHE.move_to_end(key)
    while len(_BM25_CACHE) > _BM25_CACHE_MAX_ENTRIES:
        _BM25_CACHE.popitem(last=False)
    return cached


def invalidate_bm25_cache(qdrant_url: str, collection_name: str) -> None:
    _BM25_CACHE.pop(_bm25_cache_key(qdrant_url, collection_name), None)


def _with_payload_metadata(base: dict, payload: dict, fields: tuple[str, ...]) -> dict:
    for field in fields:
        value = payload.get(field)
        if value is not None:
            base[field] = value
    return base


def _bm25_doc(text: str, score: float, payload: dict) -> dict:
    numeric_score = float(score)
    return _with_payload_metadata(
        {
            "text": text,
            "score": numeric_score,
            "bm25_score": numeric_score,
            "bm25_zero_match": numeric_score <= 0.0,
        },
        payload,
        ("source", "page", "chunk"),
    )


def retrieve_chunks_bm25(
    question: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
) -> List[dict]:
    """
    Klasik koleksiyondan tüm chunk'ları çekip BM25 ile sıralar.
    Dönüş: [{"text": str, "score": float, ...metadata}]
    """
    if not _BM25_AVAILABLE:
        raise RuntimeError("rank-bm25 paketi yüklü değil. `pip install rank-bm25` komutunu çalıştırın.")

    cached = _get_bm25_index(qdrant_url, collection_name)
    payloads = cached.payloads
    if not payloads:
        return []

    corpus_texts = cached.corpus_texts
    tokenized_query = _bm25_tokenize(question)
    scores = cached.index.get_scores(tokenized_query)

    indexed = sorted(
        ((idx, float(score)) for idx, score in enumerate(scores) if float(score) > 0.0),
        key=lambda x: x[1],
        reverse=True,
    )[:k]

    docs: List[dict] = []
    seen: set = set()
    for idx, score in indexed:
        text = corpus_texts[idx].strip()
        if not text:
            continue
        key = _normalize_text_for_dedup(text)
        if key in seen:
            continue
        seen.add(key)
        docs.append(_bm25_doc(text, score, payloads[idx]))
    return docs


def retrieve_chunks_bm25_smart(
    question: str,
    base_collection: str = DEFAULT_COLLECTION_NAME,
    k: int = 5,
    qdrant_url: str = QDRANT_URL,
) -> List[dict]:
    """
    Smart koleksiyonundan BM25 ile children içinde arama yapar,
    eşleşen child'ın parent_id'si ile parents koleksiyonundan tam parent bloğu çeker.
    """
    if not _BM25_AVAILABLE:
        raise RuntimeError("rank-bm25 paketi yüklü değil. `pip install rank-bm25` komutunu çalıştırın.")

    children_col = f"{base_collection}_children"
    parents_col = f"{base_collection}_parents"

    cached = _get_bm25_index(qdrant_url, children_col)
    child_payloads = cached.payloads
    if not child_payloads:
        return []

    tokenized_query = _bm25_tokenize(question)
    scores = cached.index.get_scores(tokenized_query)

    indexed = sorted(
        ((idx, float(score)) for idx, score in enumerate(scores) if float(score) > 0.0),
        key=lambda x: x[1],
        reverse=True,
    )

    candidates = []
    for idx, score in indexed:
        payload = {
            **child_payloads[idx],
            "bm25_score": score,
            "bm25_zero_match": False,
        }
        candidates.append((payload, score))

    parent_ids_ordered, child_info = _select_unique_parent_hits(candidates, k)

    if not parent_ids_ordered:
        return []

    client = get_qdrant_client(url=qdrant_url)
    return _retrieve_parent_docs(client, parents_col, parent_ids_ordered, child_info)
