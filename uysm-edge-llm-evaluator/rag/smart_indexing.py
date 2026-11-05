from __future__ import annotations

import json as _json
import os
import re as _re
import time
from typing import List, Optional, Sequence

import requests
from qdrant_client.http import models as qmodels

from .chunking import (
    _parent_block_id,
    _split_children_from_parent,
    _split_into_parent_blocks,
)
from .config import (
    DEFAULT_COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    QDRANT_BATCH_SIZE,
    QDRANT_URL,
    SMART_BOUNDARY_LLM_MODEL,
    SMART_CHILD_OVERLAP,
    SMART_CHILD_SIZE,
    SMART_PARENT_BLOCK_SIZE,
    _debug_log,
)
from .embeddings import get_embeddings
from .bm25 import invalidate_bm25_cache
from .pdf_text import _extract_full_pdf_text
from .qdrant_store import (
    _ensure_parent_collection,
    ensure_collection,
    get_qdrant_client,
    inspect_parent_child_integrity,
)


def _line_index_to_char_offset(
    block_a: str,
    block_b: str,
    line_index: int,
) -> Optional[int]:
    """
    Global line_index'i (block_a + '\\n' + block_b içinde) karakter offset'ine çevirir.
    Döndürür: char offset veya None (güvenli sınırlar dışında ise).
    """
    combined = block_a + "\n" + block_b
    lines = combined.split("\n")
    if line_index < 1 or line_index >= len(lines):
        return None
    offset = sum(len(lines[i]) + 1 for i in range(line_index))
    if offset < 50 or offset > len(combined) - 50:
        return None
    return offset


def _call_llm_for_boundary(
    block_a: str,
    block_b: str,
    base_url: str,
    model: str,
    timeout: int = 120,
) -> tuple[Optional[int], bool]:
    """
    block_a ve block_b'yi satır bazlı JSON olarak LLM'e gönderir.
    Döndürür: (global_line_index veya None, timed_out)
    """
    combined = block_a + "\n" + block_b
    raw_lines = combined.split("\n")
    total_lines = len(raw_lines)
    last_line_index = total_lines - 1

    line_entries = [{"index": i, "text": raw_lines[i]} for i in range(total_lines)]
    input_payload = {"lines": line_entries}

    system_prompt = (
        "You are a document segmentation assistant.\n"
        "You will receive a JSON object with a 'lines' array (index + text pairs).\n"
        "These lines come from two consecutive PDF blocks.\n\n"
        "If there is a clear NEW TOPIC or NEW SECTION: set indeks to the first line "
        "of the new topic, status=\"boundary_found\".\n"
        "If single topic throughout: indeks=-1, status=\"no_boundary\".\n"
        "If unsure: indeks=-1, status=\"uncertain\".\n\n"
        f"indeks must be between 1 and {last_line_index}.\n"
        "Prefer paragraph breaks or section headings as boundaries."
    )

    output_schema = {
        "type": "object",
        "properties": {
            "indeks": {"type": "integer"},
            "status": {"type": "string"},
        },
        "required": ["indeks", "status"],
        "additionalProperties": False,
    }

    call_id = f"boundary_{int(time.time() * 1000)}"
    full_prompt = f"{system_prompt}\n\n{_json.dumps(input_payload, ensure_ascii=False)}"

    _debug_log(
        call_id, "boundary_call",
        "rag/smart_indexing.py:_call_llm_for_boundary:input",
        "llm_input",
        {
            "model": model,
            "total_lines": total_lines,
            "block_a_preview": block_a[:120].replace("\n", "\\n"),
            "block_b_preview": block_b[:120].replace("\n", "\\n"),
            "prompt": full_prompt,
            "input_payload": input_payload,
        },
    )

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.0},
                "format": output_schema,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.Timeout, requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout):
        _debug_log(call_id, "boundary_call", "rag/smart_indexing.py:_call_llm_for_boundary:timeout", "llm_timeout", {})
        return None, True
    except Exception as e:
        _debug_log(call_id, "boundary_call", "rag/smart_indexing.py:_call_llm_for_boundary:error", "llm_error", {"error": str(e)})
        return None, False

    content = ""
    if isinstance(data, dict):
        content = str(data.get("response") or "")

    _debug_log(
        call_id, "boundary_call",
        "rag/smart_indexing.py:_call_llm_for_boundary:raw_response",
        "llm_raw_output",
        {"raw": content},
    )

    clean = _re.sub(r'<think>.*?</think>', '', content, flags=_re.DOTALL).strip()

    try:
        parsed = _json.loads(clean)
    except (_json.JSONDecodeError, ValueError):
        _debug_log(call_id, "boundary_call", "rag/smart_indexing.py:_call_llm_for_boundary:parse_fail", "llm_parse_error", {"clean": clean})
        return None, False

    indeks = parsed.get("indeks", -1)
    status = parsed.get("status", "")

    _debug_log(
        call_id, "boundary_call",
        "rag/smart_indexing.py:_call_llm_for_boundary:result",
        "llm_decision",
        {
            "indeks": indeks,
            "status": status,
            "boundary_found": status == "boundary_found" and isinstance(indeks, int) and 1 <= indeks < total_lines,
        },
    )

    if status in ("no_boundary", "uncertain"):
        return None, False

    if not isinstance(indeks, int) or indeks == -1:
        return None, False

    if indeks < 1 or indeks >= total_lines:
        return None, False

    return indeks, False


def _refine_boundaries_with_llm(
    initial_blocks: List[str],
    base_url: str,
    model: str,
    parent_size: int = SMART_PARENT_BLOCK_SIZE,
    timeout: int = 120,
    progress_callback=None,
) -> tuple[List[str], List[dict]]:
    """Strict 2-block pairwise semantik sınır tespiti."""
    if not initial_blocks:
        return [], []

    refined: List[str] = []
    timeout_records: List[dict] = []
    total_initial = len(initial_blocks)
    hard_cap_chars = max(parent_size * 2 + 200, parent_size + 500)

    def _append_with_fallback_split(text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        if len(clean) <= hard_cap_chars:
            refined.append(clean)
            return
        parts = _split_into_parent_blocks(clean, parent_size=parent_size)
        refined.extend(parts if parts else [clean])

    pending = ""
    block_ptr = 0

    while block_ptr < total_initial or pending:
        if pending:
            block_a = pending
            pending = ""
        else:
            block_a = initial_blocks[block_ptr]
            block_ptr += 1

        if block_ptr >= total_initial:
            _append_with_fallback_split(block_a)
            if progress_callback:
                progress_callback("llm_boundary", total_initial, total_initial, 0)
            break

        block_b = initial_blocks[block_ptr]
        block_ptr += 1

        line_idx, timed_out = _call_llm_for_boundary(
            block_a, block_b, base_url, model, timeout
        )

        if timed_out:
            combined_text = block_a + "\n" + block_b
            timeout_records.append({
                "block_idx": block_ptr - 1,
                "preview": combined_text[:120].replace("\n", " "),
                "full_text": combined_text.strip(),
            })

        if line_idx is not None:
            char_offset = _line_index_to_char_offset(block_a, block_b, line_idx)
            if char_offset is not None:
                combined = block_a + "\n" + block_b
                first = combined[:char_offset]
                second = combined[char_offset:]
                _append_with_fallback_split(first)
                pending = second.strip()
            else:
                _append_with_fallback_split(block_a + "\n" + block_b)
        else:
            _append_with_fallback_split(block_a + "\n" + block_b)

        if progress_callback:
            progress_callback("llm_boundary", min(block_ptr, total_initial), total_initial, 0)

    return refined, timeout_records


def _prepare_source_points(
    *,
    source: str,
    refined_blocks: List[str],
    child_size: int,
    child_overlap: int,
    embed_model: str,
    base_url: str,
) -> tuple[List[qmodels.PointStruct], List[qmodels.PointStruct]]:
    """Prepare all parent and child points without mutating Qdrant."""
    parent_points: List[qmodels.PointStruct] = []
    child_specs: List[dict] = []

    for block_idx, parent_text in enumerate(refined_blocks):
        clean_text = parent_text.strip()
        if not clean_text:
            continue
        parent_id = _parent_block_id(source, block_idx)
        parent_points.append(
            qmodels.PointStruct(
                id=parent_id,
                vector=[0.0],
                payload={"text": clean_text, "source": source, "block_idx": block_idx},
            )
        )
        child_specs.extend(
            _split_children_from_parent(
                parent_text=clean_text,
                parent_id=parent_id,
                source=source,
                block_idx=block_idx,
                child_size=child_size,
                child_overlap=child_overlap,
            )
        )

    child_embeddings: List[List[float]] = []
    child_texts = [child["text"] for child in child_specs]
    for start in range(0, len(child_texts), EMBED_BATCH_SIZE):
        child_embeddings.extend(
            get_embeddings(
                child_texts[start: start + EMBED_BATCH_SIZE],
                model=embed_model,
                base_url=base_url,
            )
        )
    if len(child_embeddings) != len(child_specs):
        raise RuntimeError(
            f"Child embedding count mismatch: {len(child_embeddings)} != {len(child_specs)}"
        )

    child_points = [
        qmodels.PointStruct(
            id=child["id"],
            vector=child_embeddings[idx],
            payload={
                **child["metadata"],
                "text": child["text"],
                "parent_id": child["parent_id"],
            },
        )
        for idx, child in enumerate(child_specs)
    ]
    return parent_points, child_points


def index_pdfs_smart(
    pdf_paths: Sequence[str],
    base_collection: str = DEFAULT_COLLECTION_NAME,
    parent_size: int = SMART_PARENT_BLOCK_SIZE,
    child_size: int = SMART_CHILD_SIZE,
    child_overlap: int = SMART_CHILD_OVERLAP,
    boundary_llm_model: str = SMART_BOUNDARY_LLM_MODEL,
    embed_model: str = OLLAMA_EMBED_MODEL,
    qdrant_url: str = QDRANT_URL,
    ollama_base_url: str | None = None,
    progress_callback=None,
) -> dict:
    """
    Smart chunking pipeline:
        PDF → büyük parent bloklar → LLM semantik sınır tespiti
        → parent bloklar Qdrant'a kaydedilir
        → child chunk'lar embed edilip ayrı koleksiyona yazılır
    """
    parents_col = f"{base_collection}_parents"
    children_col = f"{base_collection}_children"

    timings: dict = {
        "total_parents": 0,
        "total_children": 0,
        "pdf_extract_sec": 0.0,
        "llm_boundary_sec": 0.0,
        "parent_store_sec": 0.0,
        "child_embed_sec": 0.0,
        "child_upsert_sec": 0.0,
        "total_sec": 0.0,
        "timeout_count": 0,
        "timeout_blocks": [],
    }
    total_t0 = time.time()
    client = get_qdrant_client(url=qdrant_url)
    run_id = f"smart_{int(time.time() * 1000)}"
    _debug_log(
        run_id,
        "H1",
        "rag/smart_indexing.py:index_pdfs_smart:start",
        "smart_index_start",
        {
            "pdf_count": len(pdf_paths),
            "base_collection": base_collection,
            "parent_size": parent_size,
            "child_size": child_size,
            "child_overlap": child_overlap,
            "embed_model": embed_model,
            "boundary_llm_model": boundary_llm_model,
        },
    )

    t0 = time.time()
    source_blocks: dict = {}
    for i, path in enumerate(pdf_paths):
        full_text = _extract_full_pdf_text(path)
        source_blocks[path] = _split_into_parent_blocks(full_text, parent_size=parent_size)
        if progress_callback:
            progress_callback("pdf_extract", i + 1, len(pdf_paths), time.time() - t0)
    timings["pdf_extract_sec"] = round(time.time() - t0, 2)
    _debug_log(
        run_id,
        "H2",
        "rag/smart_indexing.py:index_pdfs_smart:post_extract",
        "source_blocks_ready",
        {
            "sources": len(source_blocks),
            "total_initial_blocks": sum(len(v) for v in source_blocks.values()),
            "empty_source_count": sum(1 for v in source_blocks.values() if not v),
        },
    )

    if not any(source_blocks.values()):
        timings["total_sec"] = round(time.time() - total_t0, 2)
        return timings

    _ensure_parent_collection(client, parents_col)
    children_col_ready = False

    total_all_blocks = sum(len(v) for v in source_blocks.values())
    _block_offset = [0]

    llm_t0 = time.time()
    parent_store_acc = 0.0
    child_embed_acc = 0.0
    child_upsert_acc = 0.0

    for source, blocks in source_blocks.items():
        if not blocks:
            _block_offset[0] += len(blocks)
            continue
        source_parent_written = 0
        source_children_written = 0
        source_empty_parent_count = 0
        _debug_log(
            run_id,
            "H3",
            "rag/smart_indexing.py:index_pdfs_smart:source_begin",
            "source_processing_start",
            {
                "source": os.path.basename(source),
                "block_count": len(blocks),
                "children_col_ready": children_col_ready,
            },
        )

        base_url = ollama_base_url or OLLAMA_BASE_URL

        def _make_inner_cb(off: int, glob_total: int, t_ref: float):
            def _inner(phase, current, total, _elapsed):
                if progress_callback:
                    progress_callback(
                        "llm_boundary",
                        off + current,
                        glob_total,
                        time.time() - t_ref,
                    )
            return _inner

        if base_url:
            inner_cb = _make_inner_cb(_block_offset[0], total_all_blocks, llm_t0)
            refined, src_timeouts = _refine_boundaries_with_llm(
                initial_blocks=blocks,
                base_url=base_url,
                model=boundary_llm_model,
                parent_size=parent_size,
                timeout=120,
                progress_callback=inner_cb,
            )
            for rec in src_timeouts:
                rec["source"] = source
            timings["timeout_blocks"].extend(src_timeouts)
        else:
            refined = blocks
            if progress_callback:
                progress_callback(
                    "llm_boundary",
                    _block_offset[0] + len(blocks),
                    total_all_blocks,
                    time.time() - llm_t0,
                )
        _debug_log(
            run_id,
            "H2",
            "rag/smart_indexing.py:index_pdfs_smart:post_refine",
            "refine_completed",
            {
                "source": os.path.basename(source),
                "input_blocks": len(blocks),
                "refined_blocks": len(refined),
                "timeouts_for_source": len([r for r in timings["timeout_blocks"] if r.get("source") == source]),
            },
        )

        _block_offset[0] += len(blocks)

        t_ce = time.time()
        parent_points, child_points = _prepare_source_points(
            source=source,
            refined_blocks=refined,
            child_size=child_size,
            child_overlap=child_overlap,
            embed_model=embed_model,
            base_url=base_url,
        )
        child_embed_acc += time.time() - t_ce
        source_empty_parent_count = len(refined) - len(parent_points)

        if not child_points:
            raise RuntimeError(f"Smart indexing produced no child points for source: {source}")
        if not children_col_ready:
            ensure_collection(client, children_col, vector_size=len(child_points[0].vector))
            children_col_ready = True

        src_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source))]
        )
        client.delete(
            collection_name=children_col,
            points_selector=qmodels.FilterSelector(filter=src_filter),
            wait=True,
        )
        client.delete(
            collection_name=parents_col,
            points_selector=qmodels.FilterSelector(filter=src_filter),
            wait=True,
        )

        t_cu = time.time()
        for start in range(0, len(child_points), QDRANT_BATCH_SIZE):
            client.upsert(
                collection_name=children_col,
                points=child_points[start: start + QDRANT_BATCH_SIZE],
                wait=True,
            )
        child_upsert_acc += time.time() - t_cu

        t_ps = time.time()
        for start in range(0, len(parent_points), QDRANT_BATCH_SIZE):
            client.upsert(
                collection_name=parents_col,
                points=parent_points[start: start + QDRANT_BATCH_SIZE],
                wait=True,
            )
        parent_store_acc += time.time() - t_ps

        source_parent_written = len(parent_points)
        source_children_written = len(child_points)
        timings["total_parents"] += source_parent_written
        timings["total_children"] += source_children_written

        integrity = inspect_parent_child_integrity(client, base_collection, source=source)
        if not integrity["ok"]:
            raise RuntimeError(f"Smart index integrity check failed for {source}: {integrity}")
        invalidate_bm25_cache(qdrant_url, children_col)
        _debug_log(
            run_id,
            "H3",
            "rag/smart_indexing.py:index_pdfs_smart:source_end",
            "source_processing_end",
            {
                "source": os.path.basename(source),
                "parents_written": source_parent_written,
                "children_written": source_children_written,
                "empty_parent_count": source_empty_parent_count,
                "children_col_ready": children_col_ready,
            },
        )

    timings["llm_boundary_sec"] = round(time.time() - llm_t0, 2)
    timings["parent_store_sec"] = round(parent_store_acc, 2)
    timings["child_embed_sec"] = round(child_embed_acc, 2)
    timings["child_upsert_sec"] = round(child_upsert_acc, 2)
    timings["timeout_count"] = len(timings["timeout_blocks"])
    timings["total_sec"] = round(time.time() - total_t0, 2)
    return timings
