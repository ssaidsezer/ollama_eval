from __future__ import annotations

import hashlib
import uuid
from typing import List

from .config import SMART_CHILD_OVERLAP, SMART_CHILD_SIZE, SMART_PARENT_BLOCK_SIZE
from .pdf_text import _decode_pdf_text, _extract_pages_text


def _normalize_text_for_dedup(text: str) -> str:
    """Whitespace farklarını normalize ederek metin karşılaştır."""
    return " ".join(text.split()).strip()


def _chunk_id(source: str, page: int, chunk_index: int) -> str:
    """
    Aynı PDF tekrar indekslendiğinde point ID sabit kalsın.
    Böylece Qdrant upsert duplicate eklemek yerine güncelleme yapar.
    """
    raw = f"{source}|{page}|{chunk_index}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _parent_block_id(source: str, block_idx: int) -> str:
    raw = f"parent|{source}|{block_idx}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _child_block_id(parent_id: str, child_idx: int) -> str:
    raw = f"child|{parent_id}|{child_idx}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _snap_to_word_boundary(text: str, pos: int) -> int:
    """
    pos karakterinden geriye doğru en yakın boşluk/satır sonunu bulur ve
    kesim noktasını o boşluğun hemen sonrasına taşır.
    """
    if pos <= 0 or pos >= len(text):
        return pos
    if text[pos] in (" ", "\n", "\t"):
        return pos
    for i in range(pos, max(pos - 200, 0), -1):
        if text[i] in (" ", "\n", "\t"):
            return i + 1
    return pos


def _snap_to_sentence_boundary(text: str, pos: int, max_lookback: int = 400) -> int:
    """
    pos'tan geriye doğru cümle sonu noktalama işareti arar.
    Bulamazsa _snap_to_word_boundary'ye düşer.
    """
    if pos <= 0 or pos >= len(text):
        return pos
    limit = max(pos - max_lookback, 0)
    for i in range(pos - 1, limit, -1):
        if text[i] in ".!?" and i + 1 < len(text) and text[i + 1] in (" ", "\n", "\t"):
            return i + 1
    return _snap_to_word_boundary(text, pos)


def _snap_to_best_boundary(text: str, pos: int, max_lookback: int = 400) -> int:
    """
    Önce paragraf sonu (\\n\\n), sonra cümle sonu, en son kelime sınırını dener.
    Hiyerarşik tercih: paragraph > sentence > word.
    """
    if pos <= 0 or pos >= len(text):
        return pos
    limit = max(pos - max_lookback, 0)
    for i in range(pos - 1, limit, -1):
        if text[i] == "\n" and i > 0 and text[i - 1] == "\n":
            return i + 1
    return _snap_to_sentence_boundary(text, pos, max_lookback)


def _chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """Basit karakter tabanlı chunking."""
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = _snap_to_best_boundary(text, end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def _extract_pdf_chunks(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[dict]:
    """PDF'den text chunk'larını çıkar."""
    chunks: List[dict] = []

    pages = _extract_pages_text(pdf_path)
    for page_index, page_text in enumerate(pages):
        try:
            page_text = _decode_pdf_text(page_text)
        except Exception:
            page_text = ""

        for chunk_index, chunk_text_str in enumerate(
            _chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                {
                    "id": _chunk_id(pdf_path, page_index, chunk_index),
                    "text": chunk_text_str,
                    "metadata": {
                        "source": pdf_path,
                        "page": page_index,
                        "chunk": chunk_index,
                    },
                }
            )

    return chunks


def _split_into_parent_blocks(
    text: str,
    parent_size: int = SMART_PARENT_BLOCK_SIZE,
) -> List[str]:
    """Tüm metni örtüşmesiz sabit boyutlu büyük bloklara böler."""
    text = text.strip()
    if not text:
        return []
    blocks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + parent_size, len(text))
        if end < len(text):
            end = _snap_to_best_boundary(text, end)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
        if end >= len(text):
            break
        start = end
    return blocks


def _split_children_from_parent(
    parent_text: str,
    parent_id: str,
    source: str,
    block_idx: int,
    child_size: int = SMART_CHILD_SIZE,
    child_overlap: int = SMART_CHILD_OVERLAP,
) -> List[dict]:
    """Semantik parent bloğu sabit boyutlu child chunk'lara böler."""
    children: List[dict] = []
    child_idx = 0
    start = 0
    text = parent_text.strip()

    while start < len(text):
        end = min(start + child_size, len(text))
        if end < len(text):
            end = _snap_to_sentence_boundary(text, end)
        chunk = text[start:end].strip()
        if chunk:
            children.append(
                {
                    "id": _child_block_id(parent_id, child_idx),
                    "text": chunk,
                    "parent_id": parent_id,
                    "metadata": {
                        "source": source,
                        "block_idx": block_idx,
                        "child_idx": child_idx,
                    },
                }
            )
            child_idx += 1
        if end >= len(text):
            break
        start = max(end - child_overlap, 0)

    return children
