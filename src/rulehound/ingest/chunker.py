"""Rule-block chunking (design doc §4.2).

A chunk = one named rule: heading + all body text until the next
same-or-higher-level heading. Boxed sidebar text becomes `commentary`.
Chunks over the token limit split on paragraph boundaries into parts
sharing the same rule_id prefix.
"""

from __future__ import annotations

import re

from ..models import BBox, RuleChunk
from .pdf_extract import TextBlock

SECTION_SEP = " > "


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def rule_id_for(section_titles: list[str]) -> str:
    return ".".join(slugify(t) for t in section_titles)


def _estimate_tokens(text: str) -> int:
    # ~1.3 tokens per word is a safe over-estimate for English prose.
    return int(len(text.split()) * 1.3)


def _split_paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_oversized(chunk: RuleChunk, max_tokens: int) -> list[RuleChunk]:
    if _estimate_tokens(chunk.text) <= max_tokens:
        return [chunk]
    paras = _split_paragraphs(chunk.text)
    if len(paras) < 2:
        return [chunk]  # cannot split below paragraph granularity
    parts: list[list[str]] = [[]]
    count = 0
    for p in paras:
        t = _estimate_tokens(p)
        if parts[-1] and count + t > max_tokens:
            parts.append([])
            count = 0
        parts[-1].append(p)
        count += t
    out: list[RuleChunk] = []
    for i, group in enumerate(parts, start=1):
        out.append(
            RuleChunk(
                rule_id=f"{chunk.rule_id}--part-{i}",
                title=f"{chunk.title} (part {i}/{len(parts)})",
                section_path=chunk.section_path,
                text="\n\n".join(group),
                commentary=chunk.commentary if i == 1 else None,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                bboxes=chunk.bboxes,
                crop_paths=list(chunk.crop_paths),
                doc_hash=chunk.doc_hash,
            )
        )
    return out


class _ChunkBuilder:
    def __init__(self, section_titles: list[str]) -> None:
        self.section_titles = list(section_titles)
        self.text_parts: list[str] = []
        self.commentary_parts: list[str] = []
        self.pages: set[int] = set()
        self.bboxes: dict[int, list[BBox]] = {}

    def add(self, block: TextBlock) -> None:
        if block.in_box:
            self.commentary_parts.append(block.text)
        else:
            self.text_parts.append(block.text)
        self.pages.add(block.page)
        self.bboxes.setdefault(block.page, []).append(block.bbox)

    def build(self, doc_hash: str) -> RuleChunk | None:
        if not self.section_titles:
            return None
        text = "\n\n".join(self.text_parts).strip()
        commentary = "\n\n".join(self.commentary_parts).strip() or None
        if not text and not commentary:
            return None
        return RuleChunk(
            rule_id=rule_id_for(self.section_titles),
            title=self.section_titles[-1],
            section_path=SECTION_SEP.join(self.section_titles),
            text=text,
            commentary=commentary,
            page_start=min(self.pages) if self.pages else 0,
            page_end=max(self.pages) if self.pages else 0,
            bboxes=self.bboxes,
            doc_hash=doc_hash,
        )


def chunk_blocks(
    blocks: list[TextBlock], doc_hash: str = "", max_tokens: int = 1200
) -> list[RuleChunk]:
    """Walk ordered blocks, opening a new chunk at every heading."""
    chunks: list[RuleChunk] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    builder: _ChunkBuilder | None = None

    def flush() -> None:
        nonlocal builder
        if builder is not None:
            built = builder.build(doc_hash)
            if built is not None:
                chunks.append(built)
        builder = None

    for block in blocks:
        if block.is_heading and block.heading_level is not None and not block.in_box:
            flush()
            level = block.heading_level
            title = " ".join(block.text.split())
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            builder = _ChunkBuilder([t for _, t in stack])
            builder.pages.add(block.page)
            builder.bboxes.setdefault(block.page, []).append(block.bbox)
        elif builder is not None:
            builder.add(block)
        # body text before the first heading (cover page etc.) is dropped
    flush()

    # Deduplicate rule_ids (repeated headings) then apply size limits.
    seen: dict[str, int] = {}
    for c in chunks:
        n = seen.get(c.rule_id, 0)
        seen[c.rule_id] = n + 1
        if n:
            c.rule_id = f"{c.rule_id}-{n + 1}"

    out: list[RuleChunk] = []
    for c in chunks:
        out.extend(_split_oversized(c, max_tokens))
    return out
