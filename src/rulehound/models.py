"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points


@dataclass
class RuleChunk:
    """One named rule: heading + body until the next same-or-higher heading (§4.2)."""

    rule_id: str
    title: str
    section_path: str  # "Movement Phase > Transports > Disembark"
    text: str
    commentary: str | None = None
    page_start: int = 0
    page_end: int = 0
    # per-page bounding boxes covering the rule's text: {page_number: [bbox, ...]}
    bboxes: dict[int, list[BBox]] = field(default_factory=dict)
    crop_paths: list[str] = field(default_factory=list)
    doc_hash: str = ""


@dataclass
class RuleRef:
    from_rule: str
    to_rule: str
    kind: str  # explicit_see | title_mention | page_ref


@dataclass
class ScoredRule:
    rule_id: str
    title: str
    score: float
