"""Text + layout extraction (design doc §4.1).

Extracts text blocks with bounding boxes, font size and weight via PyMuPDF's
get_text("dict"), detects heading candidates heuristically, and handles the
two-column layout of the Core Rules by ordering blocks column-major when a
page clusters into two columns (§11 layout note).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF

from ..models import BBox

BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold


@dataclass
class TextBlock:
    page: int  # 1-based
    bbox: BBox
    text: str
    font_size: float
    is_bold: bool
    in_box: bool = False  # inside a drawn/filled rectangle (sidebar/commentary)
    is_heading: bool = False
    heading_level: int | None = None
    heading_confidence: float = 0.0


def _span_is_bold(span: dict) -> bool:
    if span.get("flags", 0) & BOLD_FLAG:
        return True
    return "bold" in span.get("font", "").lower()


def _boxed_regions(page: fitz.Page) -> list[fitz.Rect]:
    """Filled/stroked rectangles that plausibly frame sidebar text."""
    regions: list[fitz.Rect] = []
    page_area = abs(page.rect)
    try:
        drawings = page.get_drawings()
    except Exception:
        return regions
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        r = fitz.Rect(rect)
        area = abs(r)
        # Ignore hairlines and near-full-page decoration
        if area < 2000 or area > 0.6 * page_area:
            continue
        if d.get("fill") is not None or d.get("type") in ("f", "fs", "s"):
            regions.append(r)
    return regions


def _block_from_dict(page_num: int, block: dict, boxed: list[fitz.Rect]) -> TextBlock | None:
    parts: list[str] = []
    sizes: Counter[float] = Counter()
    bold_chars = 0
    total_chars = 0
    for line in block.get("lines", []):
        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
        parts.append(line_text)
        for span in line.get("spans", []):
            n = len(span.get("text", ""))
            if not n:
                continue
            sizes[round(span.get("size", 0.0), 1)] += n
            total_chars += n
            if _span_is_bold(span):
                bold_chars += n
    text = "\n".join(parts).strip()
    if not text:
        return None
    font_size = sizes.most_common(1)[0][0] if sizes else 0.0
    bbox = tuple(block["bbox"])  # type: ignore[arg-type]
    rect = fitz.Rect(bbox)
    in_box = any(r.contains(rect) or (r & rect and abs(r & rect) > 0.8 * abs(rect)) for r in boxed)
    return TextBlock(
        page=page_num,
        bbox=bbox,  # type: ignore[arg-type]
        text=text,
        font_size=font_size,
        is_bold=total_chars > 0 and bold_chars / total_chars > 0.6,
        in_box=in_box,
    )


def _order_page_blocks(blocks: list[TextBlock], page_width: float) -> list[TextBlock]:
    """Reading order. Detect two-column pages and order column-major (§11)."""
    if len(blocks) < 4:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    mid = page_width / 2
    full_width = [b for b in blocks if (b.bbox[2] - b.bbox[0]) > 0.6 * page_width]
    narrow = [b for b in blocks if b not in full_width]
    left = [b for b in narrow if (b.bbox[0] + b.bbox[2]) / 2 < mid]
    right = [b for b in narrow if (b.bbox[0] + b.bbox[2]) / 2 >= mid]

    two_col = len(left) >= 2 and len(right) >= 2
    if not two_col:
        return sorted(blocks, key=lambda b: (round(b.bbox[1]), b.bbox[0]))

    # Full-width blocks act as horizontal separators splitting the page into
    # bands; within each band read left column top-to-bottom then right column.
    separators = sorted(full_width, key=lambda b: b.bbox[1])
    bands: list[tuple[float, float]] = []
    prev = -1.0
    for sep in separators:
        bands.append((prev, sep.bbox[1]))
        prev = sep.bbox[1]
    bands.append((prev, float("inf")))

    ordered: list[TextBlock] = []
    for lo, hi in bands:
        def in_band(b: TextBlock) -> bool:
            return lo <= b.bbox[1] < hi

        ordered.extend(sorted((b for b in full_width if in_band(b)), key=lambda b: b.bbox[1]))
        ordered.extend(sorted((b for b in left if in_band(b)), key=lambda b: b.bbox[1]))
        ordered.extend(sorted((b for b in right if in_band(b)), key=lambda b: b.bbox[1]))
    return ordered


def _detect_headings(blocks: list[TextBlock]) -> None:
    """Mark heading candidates and assign levels by font-size tier (§4.1)."""
    size_weight: Counter[float] = Counter()
    for b in blocks:
        size_weight[b.font_size] += len(b.text)
    if not size_weight:
        return
    body_size = size_weight.most_common(1)[0][0]

    candidates: list[TextBlock] = []
    for b in blocks:
        if b.in_box:
            continue
        first_line = b.text.split("\n", 1)[0].strip()
        short = len(first_line) <= 80 and len(b.text) <= 120
        looks_like_title = short and not first_line.rstrip().endswith((".", ",", ";", ":"))
        if b.font_size > body_size * 1.15 and looks_like_title:
            b.is_heading = True
            b.heading_confidence = min(1.0, 0.5 + (b.font_size / body_size - 1.0))
            candidates.append(b)
        elif b.is_bold and b.font_size >= body_size * 0.95 and looks_like_title:
            b.is_heading = True
            b.heading_confidence = 0.5
            candidates.append(b)

    if not candidates:
        return
    tiers = sorted({round(b.font_size, 1) for b in candidates}, reverse=True)
    tier_level = {size: i + 1 for i, size in enumerate(tiers)}
    for b in candidates:
        b.heading_level = tier_level[round(b.font_size, 1)]


def extract_blocks(pdf_path: str | Path) -> list[TextBlock]:
    """Extract ordered text blocks with heading candidates from the PDF."""
    doc = fitz.open(str(pdf_path))
    all_blocks: list[TextBlock] = []
    try:
        for page_index, page in enumerate(doc):
            boxed = _boxed_regions(page)
            page_blocks: list[TextBlock] = []
            for raw in page.get_text("dict")["blocks"]:
                if raw.get("type") != 0:
                    continue
                tb = _block_from_dict(page_index + 1, raw, boxed)
                if tb is not None:
                    page_blocks.append(tb)
            all_blocks.extend(_order_page_blocks(page_blocks, page.rect.width))
    finally:
        doc.close()
    _detect_headings(all_blocks)
    return all_blocks


def write_blocks_jsonl(blocks: list[TextBlock], path: str | Path) -> None:
    """Intermediate JSONL: one record per text block (§4.1)."""
    with open(path, "w") as f:
        for b in blocks:
            f.write(json.dumps(asdict(b)) + "\n")


def read_blocks_jsonl(path: str | Path) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["bbox"] = tuple(d["bbox"])
            blocks.append(TextBlock(**d))
    return blocks
