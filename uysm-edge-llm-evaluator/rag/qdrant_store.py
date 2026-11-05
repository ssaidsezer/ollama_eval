from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .chunking import _normalize_text_for_dedup
from .config import QDRANT_URL, _debug_log


_qdrant_client: Optional[QdrantClient] = None
_qdrant_client_url: Optional[str] = None


def get_qdrant_client(url: str = QDRANT_URL) -> QdrantClient:
    """Lazy-init Qdrant bağlantısı."""
    global _qdrant_client, _qdrant_client_url
    normalized_url = (url or QDRANT_URL).rstrip("/")
    if _qdrant_client is None or _qdrant_client_url != normalized_url:
        _qdrant_client = QdrantClient(url=normalized_url, timeout=600)
        _qdrant_client_url = normalized_url
    return _qdrant_client


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
) -> None:
    """Qdrant koleksiyonunu yoksa oluştur, dimension değiştiyse sil ve yeniden oluştur."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        info = client.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            existing_size = next(iter(vectors_config.values())).size
        else:
            existing_size = vectors_config.size if vectors_config else None
        if existing_size == vector_size:
            return
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )


def _qdrant_query_points_compat(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int,
    score_threshold: float,
    run_id: str,
    location: str,
    offset: int | None = None,
) -> list:
    _debug_log(
        run_id,
        "H2",
        location,
        "qdrant_query_dispatch",
        {
            "has_query_points": hasattr(client, "query_points"),
            "has_search": hasattr(client, "search"),
            "collection_name": collection_name,
            "limit": limit,
            "score_threshold": score_threshold,
            "query_dim": len(query_vector),
        },
    )
    if hasattr(client, "query_points"):
        query_kwargs = dict(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        if offset is not None:
            query_kwargs["offset"] = offset
        query_result = client.query_points(**query_kwargs)
        return getattr(query_result, "points", query_result)

    if hasattr(client, "search"):
        search_kwargs = dict(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        if offset is not None:
            search_kwargs["offset"] = offset
        return client.search(**search_kwargs)

    raise AttributeError("Qdrant client has neither query_points nor search methods.")


def _record_payload_and_id(record) -> tuple[str, dict]:
    if isinstance(record, dict):
        point_id = str(record.get("id", ""))
        payload = record.get("payload") or {}
    else:
        point_id = str(getattr(record, "id", ""))
        payload = getattr(record, "payload", None) or {}
    return point_id, payload


def _hit_payload_and_score(hit) -> tuple[dict, float]:
    if isinstance(hit, dict):
        return hit.get("payload") or {}, hit.get("score", 0.0)
    return getattr(hit, "payload", None) or {}, getattr(hit, "score", 0.0)


def _select_unique_parent_hits(candidates, k: int) -> tuple[List[str], dict]:
    """Select the highest-ranked child hit for up to ``k`` unique parents."""
    if k <= 0:
        return [], {}

    parent_ids: List[str] = []
    child_info: dict = {}
    for payload, score in candidates:
        parent_id = str(payload.get("parent_id") or "")
        if not parent_id or parent_id in child_info:
            continue

        info = {
            "child_text": payload.get("text", ""),
            "score": float(score),
        }
        for field in (
            "source",
            "block_idx",
            "child_idx",
            "bm25_score",
            "bm25_zero_match",
        ):
            value = payload.get(field)
            if value is not None:
                info[field] = value

        parent_ids.append(parent_id)
        child_info[parent_id] = info
        if len(parent_ids) == k:
            break

    return parent_ids, child_info


def _retrieve_parent_docs(
    client: QdrantClient,
    collection_name: str,
    parent_ids_ordered: List[str],
    child_info: dict,
) -> List[dict]:
    parent_records = client.retrieve(
        collection_name=collection_name,
        ids=parent_ids_ordered,
        with_payload=True,
    )

    parent_payload_map: dict = {}
    for rec in parent_records:
        point_id, payload = _record_payload_and_id(rec)
        if point_id:
            parent_payload_map[point_id] = payload

    docs: List[dict] = []
    seen_text: set = set()

    for parent_id in parent_ids_ordered:
        parent_payload = parent_payload_map.get(parent_id, {})
        parent_text = parent_payload.get("text", "")
        info = child_info[parent_id]
        if parent_text.strip():
            key = _normalize_text_for_dedup(parent_text)
            if key not in seen_text:
                seen_text.add(key)
                doc = {
                    "text": parent_text,
                    "child_text": info["child_text"],
                    "score": info["score"],
                    "parent_id": parent_id,
                }
                for field in ("source", "block_idx"):
                    value = parent_payload.get(field)
                    if value is not None:
                        doc[field] = value
                for field in ("child_idx", "bm25_score", "bm25_zero_match"):
                    value = info.get(field)
                    if value is not None:
                        doc[field] = value
                docs.append(doc)

    return docs


def _ensure_parent_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Parent blokları için Qdrant koleksiyonu oluşturur.
    Boyutu 1 olan dummy vektör kullanılır; parent'lar ID ile doğrudan çekilir.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=1,
                distance=qmodels.Distance.COSINE,
            ),
        )


def _scroll_all_records(
    client: QdrantClient,
    collection_name: str,
    source: str | None = None,
) -> List[tuple[str, dict]]:
    """Scroll a collection while preserving point IDs and optional source scope."""
    records: List[tuple[str, dict]] = []
    scroll_filter = None
    if source is not None:
        scroll_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="source",
                    match=qmodels.MatchValue(value=source),
                )
            ]
        )
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for rec in result:
            pid, payload = _record_payload_and_id(rec)
            records.append((pid, payload))
        if next_offset is None:
            break
        offset = next_offset
    return records


def _scroll_all_payloads(client: QdrantClient, collection_name: str) -> List[dict]:
    """Bir koleksiyondaki tüm payload'ları sayfalayarak döndürür."""
    payloads: List[dict] = []
    for pid, payload in _scroll_all_records(client, collection_name):
        item = dict(payload)
        item["_point_id"] = pid
        payloads.append(item)
    return payloads


def inspect_parent_child_integrity(
    client: QdrantClient,
    base_collection: str,
    source: str | None = None,
) -> dict:
    """Report broken references between smart RAG parent and child collections."""
    parent_records = _scroll_all_records(client, f"{base_collection}_parents", source)
    child_records = _scroll_all_records(client, f"{base_collection}_children", source)

    parent_ids = {point_id for point_id, _payload in parent_records if point_id}
    referenced_parent_ids = {
        str(payload.get("parent_id") or "")
        for _point_id, payload in child_records
        if payload.get("parent_id")
    }
    orphan_child_ids = sorted(
        point_id
        for point_id, payload in child_records
        if str(payload.get("parent_id") or "") not in parent_ids
    )
    parent_ids_without_children = sorted(parent_ids - referenced_parent_ids)
    parent_sources = {payload.get("source") for _pid, payload in parent_records}
    child_sources = {payload.get("source") for _pid, payload in child_records}
    source_sets_equal = parent_sources == child_sources

    return {
        "parents": len(parent_records),
        "children": len(child_records),
        "orphan_child_ids": orphan_child_ids,
        "parent_ids_without_children": parent_ids_without_children,
        "source_sets_equal": source_sets_equal,
        "ok": not orphan_child_ids and not parent_ids_without_children and source_sets_equal,
    }
