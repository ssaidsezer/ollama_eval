#!/usr/bin/env python3
"""Render PDF content pages into labeled per-chunk images for QA generation.

Each PDF's content window (front/back matter skipped) is split into
non-overlapping N-page chunks. Every page is rendered to a PNG with a page
label stamped in the top margin, so the downstream vision model can attribute
each fact to an exact source page.

Output layout:
  <out-dir>/<pdf_stem>/chunk_<NNN>/page_<PPP>.png
  <out-dir>/manifest.json   # chunk -> (pdf, [page numbers], image paths)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

REPO_ROOT = Path(__file__).resolve().parents[2]

# Content windows verified by inspecting front/back matter (1-indexed, inclusive).
# offset = physical_page - printed_footer_number (verified constant across each book).
DEFAULT_SOURCES = {
    "taktik_muharebe_yarali_bakimi_el_kitabi.pdf": {"window": (7, 129), "offset": 6},
    "askeri_egitim_kitabi.pdf": {"window": (5, 205), "offset": 0},
}


def printed_page_number(page: "fitz.Page", offset: int) -> int:
    """Best-effort printed page number: read the footer, fall back to physical-offset."""
    height = page.rect.height
    footer_numbers: list[int] = []
    for _x0, _y0, _x1, y1, text, *_ in page.get_text("blocks"):
        if y1 > height * 0.90:
            footer_numbers.extend(int(tok) for tok in re.findall(r"\b\d{1,3}\b", text))
    if footer_numbers:
        return footer_numbers[-1]
    return page.number + 1 - offset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf-dir", type=Path, default=REPO_ROOT / "data/pdf")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/qa_chunks")
    p.add_argument("--pages-per-chunk", type=int, default=3)
    p.add_argument("--dpi", type=int, default=150, help="Render resolution (150 is legible and cheap).")
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated PDF filenames to limit rendering (default: all configured).",
    )
    return p.parse_args()


def render(args: argparse.Namespace) -> None:
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    args.out_dir = args.out_dir.resolve()
    manifest: list[dict] = []
    for filename, spec in DEFAULT_SOURCES.items():
        if only and filename not in only:
            continue
        first, last = spec["window"]
        offset = spec["offset"]
        pdf_path = args.pdf_dir / filename
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        doc = fitz.open(pdf_path)
        last = min(last, doc.page_count)
        stem = pdf_path.stem
        pages = list(range(first, last + 1))
        chunk_id = 0
        for start in range(0, len(pages), args.pages_per_chunk):
            chunk_pages = pages[start : start + args.pages_per_chunk]
            chunk_id += 1
            chunk_dir = args.out_dir / stem / f"chunk_{chunk_id:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            image_paths = []
            page_entries = []
            for physical_page in chunk_pages:
                page = doc[physical_page - 1]
                printed = printed_page_number(page, offset)
                pix = page.get_pixmap(dpi=args.dpi)
                out_path = chunk_dir / f"page_{physical_page:03d}.png"
                label = f"{filename}  —  SAYFA {printed}  (pdf_page_index {physical_page})"
                out_path.write_bytes(_stamp(pix, label))
                image_paths.append(str(out_path.relative_to(REPO_ROOT)))
                page_entries.append({"pdf_page_index": physical_page, "sayfa": printed})
            manifest.append(
                {
                    "chunk": f"{stem}/chunk_{chunk_id:03d}",
                    "kaynak": filename,
                    "pages": page_entries,
                    "images": image_paths,
                }
            )
        doc.close()
        print(f"{filename}: pages {first}-{last} (offset {offset}) -> {chunk_id} chunks")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {args.out_dir / 'manifest.json'} ({len(manifest)} chunks)")


def _stamp(pix: "fitz.Pixmap", label: str) -> bytes:
    """Draw a labeled white band above the rendered page, return PNG bytes.

    Works in image (pixel) space: the page is already rasterized at `dpi`, so we
    build a canvas that is `band_px` taller, paint a white label band, and insert
    the page image below it. Fontsize is in the same pixel space.
    """
    png_bytes = pix.tobytes("png")
    band_px = max(34, int(pix.height * 0.045))
    doc = fitz.open()
    page = doc.new_page(width=pix.width, height=pix.height + band_px)
    page.draw_rect(fitz.Rect(0, 0, pix.width, band_px), color=(0, 0, 0), fill=(1, 1, 1))
    page.insert_text((10, band_px * 0.7), label, fontsize=max(14, band_px * 0.5), color=(0, 0, 0))
    page.insert_image(fitz.Rect(0, band_px, pix.width, pix.height + band_px), stream=png_bytes)
    # Page is already at target pixel size; render 1:1 (no extra dpi scaling).
    out = page.get_pixmap().tobytes("png")
    doc.close()
    return out


if __name__ == "__main__":
    render(parse_args())
