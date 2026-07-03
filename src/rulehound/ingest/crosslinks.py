"""Cross-reference edge extraction (design doc §4.3).

Emits (from_rule_id, to_rule_id, kind) edges where kind is one of
explicit_see | title_mention | page_ref. Longest title matched first so
"Move" never swallows "Normal Move"; matches are case-insensitive and
word-boundary anchored.
"""

from __future__ import annotations

import re

from ..models import RuleChunk, RuleRef

_SEE_RE = re.compile(r"\bsee\s+(?:the\s+)?([A-Z][A-Za-z'\- ]{2,60})")
_PAGE_RE = re.compile(r"\((?:pg|page)\.?\s*(\d+)\)", re.IGNORECASE)


def _base_rule_id(rule_id: str) -> str:
    return rule_id.split("--part-")[0]


def _title_map(chunks: list[RuleChunk]) -> dict[str, str]:
    """Normalized title -> rule_id. Part chunks map their base title once."""
    mapping: dict[str, str] = {}
    for c in chunks:
        title = re.sub(r"\s*\(part \d+/\d+\)$", "", c.title).strip().lower()
        if len(title) < 3:
            continue
        mapping.setdefault(title, _base_rule_id(c.rule_id))
    return mapping


def extract_crosslinks(chunks: list[RuleChunk]) -> list[RuleRef]:
    titles = _title_map(chunks)
    if not titles:
        return []

    # Longest-first alternation: at equal start positions the regex engine
    # tries alternatives in order, so longer titles win.
    ordered = sorted(titles.keys(), key=len, reverse=True)
    mention_re = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in ordered) + r")\b", re.IGNORECASE
    )

    page_index: dict[int, list[str]] = {}
    for c in chunks:
        page_index.setdefault(c.page_start, []).append(_base_rule_id(c.rule_id))

    edges: set[tuple[str, str, str]] = set()
    for c in chunks:
        from_id = _base_rule_id(c.rule_id)
        text = c.text
        own_title = re.sub(r"\s*\(part \d+/\d+\)$", "", c.title).strip().lower()

        # Explicit "see X" references
        see_spans: list[tuple[int, int]] = []
        for m in _SEE_RE.finditer(text):
            captured = m.group(1).strip().lower()
            target = None
            for t in ordered:  # longest known title that prefixes the capture
                if captured == t or captured.startswith(t + " "):
                    target = titles[t]
                    break
            if target and target != from_id:
                edges.add((from_id, target, "explicit_see"))
                see_spans.append(m.span())

        # Title mentions (skip regions already claimed by explicit_see)
        for m in mention_re.finditer(text):
            if any(lo <= m.start() < hi for lo, hi in see_spans):
                continue
            t = m.group(1).lower()
            if t == own_title:
                continue
            target = titles[t]
            if target != from_id:
                edges.add((from_id, target, "title_mention"))

        # "(pg N)" references -> rules starting on that page
        for m in _PAGE_RE.finditer(text):
            page = int(m.group(1))
            for target in page_index.get(page, [])[:5]:
                if target != from_id:
                    edges.add((from_id, target, "page_ref"))

    return [RuleRef(f, t, k) for f, t, k in sorted(edges)]
