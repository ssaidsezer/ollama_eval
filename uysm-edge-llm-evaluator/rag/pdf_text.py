from __future__ import annotations

import re as _re
from typing import List

import pdfplumber

try:
    import fitz as _fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False


_RE_SEPARATOR = _re.compile(r'^[\-=_\.~\*#]{3,}\s*$', _re.MULTILINE)
_RE_PAGE_NUM = _re.compile(r'^\s*(?:Sayfa\s*)?\d+\s*$|^\s*-\s*\d+\s*-\s*$', _re.MULTILINE | _re.IGNORECASE)
_RE_SHORT_LINE = _re.compile(r'^\s{0,2}[^\w\s]{0,2}\s*$', _re.MULTILINE)
_RE_CTRL_CHAR = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_RE_MULTI_NL = _re.compile(r'\n{3,}')
_RE_BULLETS = _re.compile(r'^\s*\*?[•○●◦▪▸►]{2,}\s*', _re.MULTILINE)
_RE_TRAILING_NUM = _re.compile(r'\n\d{1,4}\s*$')
_RE_IMAGE_REF = _re.compile(r'\(?\bResim\s+\d+[-–]\d+\b\.?\)?', _re.IGNORECASE)


def _extract_pages_text(pdf_path: str) -> list[str]:
    """Her PDF sayfasının ham metnini döndürür."""
    if _PYMUPDF_AVAILABLE:
        try:
            doc = _fitz.open(pdf_path)
            pages = [page.get_text() for page in doc]
            doc.close()
            return pages
        except Exception:
            pass
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def _decode_pdf_text(raw: str) -> str:
    """PDF'den gelen metindeki encoding bozukluklarını temizler."""
    if not raw:
        return ""
    try:
        fixed = raw.encode("latin-1").decode("utf-8")
        if any(c in fixed for c in "çğıöşüÇĞİÖŞÜ"):
            return fixed
        if any(c in raw for c in "çğıöşüÇĞİÖŞÜ"):
            return raw
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def _clean_pdf_text(text: str) -> str:
    """PDF metnindeki gürültüyü temizler."""
    text = _RE_CTRL_CHAR.sub('', text)
    text = _RE_SEPARATOR.sub('', text)
    text = _RE_PAGE_NUM.sub('', text)
    text = _RE_SHORT_LINE.sub('', text)
    text = _RE_BULLETS.sub('', text)
    text = _RE_TRAILING_NUM.sub('', text)
    text = _RE_IMAGE_REF.sub('', text)
    text = _RE_MULTI_NL.sub('\n\n', text)
    return text.strip()


def _extract_full_pdf_text(pdf_path: str) -> str:
    """PDF'teki tüm sayfaların metnini tek bir string olarak döndürür."""
    parts: List[str] = []
    for text in _extract_pages_text(pdf_path):
        try:
            text = _decode_pdf_text(text)
        except Exception:
            text = ""
        if text.strip():
            parts.append(text)
    return _clean_pdf_text("\n".join(parts))
