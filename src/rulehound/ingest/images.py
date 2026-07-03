"""Page rendering and per-chunk crops (design doc §4.4)."""

from __future__ import annotations

from pathlib import Path

import fitz

from ..models import RuleChunk


def render_pages(pdf_path: str | Path, pages_dir: str | Path, dpi: int = 150) -> list[Path]:
    """Render every page to PNG at the configured DPI."""
    pages_dir = Path(pages_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    doc = fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc):
            path = pages_dir / f"page_{i + 1:04d}.png"
            page.get_pixmap(matrix=matrix).save(str(path))
            out.append(path)
    finally:
        doc.close()
    return out


def crop_chunks(
    pdf_path: str | Path,
    chunks: list[RuleChunk],
    crops_dir: str | Path,
    dpi: int = 150,
    padding_px: int = 12,
) -> None:
    """Crop the union of each chunk's bboxes (+padding) per page.

    Sets `crop_paths` on each chunk to paths relative to crops_dir's parent
    style: just the filename; the API mounts crops_dir at /crops.
    """
    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pad_pts = padding_px * 72 / dpi

    doc = fitz.open(str(pdf_path))
    try:
        for chunk in chunks:
            chunk.crop_paths = []
            for page_num, boxes in sorted(chunk.bboxes.items()):
                if not boxes:
                    continue
                page = doc[page_num - 1]
                union = fitz.Rect(boxes[0])
                for b in boxes[1:]:
                    union |= fitz.Rect(b)
                union = fitz.Rect(
                    union.x0 - pad_pts, union.y0 - pad_pts,
                    union.x1 + pad_pts, union.y1 + pad_pts,
                ) & page.rect
                if union.is_empty:
                    continue
                path = crops_dir / f"{chunk.rule_id}_{page_num}.png"
                page.get_pixmap(matrix=matrix, clip=union).save(str(path))
                chunk.crop_paths.append(path.name)
    finally:
        doc.close()
