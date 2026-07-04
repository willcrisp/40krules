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


@dataclass
class WeaponProfile:
    """One weapon row from a datasheet's ranged/melee weapons table.

    Stat fields stay strings — GW prints "D6+3", "3+", "Melee", and search
    is text-only by design, so nothing is parsed into numbers.
    """

    name: str
    range: str = ""
    attacks: str = ""
    skill: str = ""  # BS or WS, e.g. "3+"
    strength: str = ""
    ap: str = ""
    damage: str = ""
    keywords: list[str] = field(default_factory=list)
    # deterministic id ("{unit_id}--{weapon-slug}"), assigned by the store
    weapon_id: str = ""


@dataclass
class UnitProfile:
    """One unit datasheet: stat block + weapons + abilities prose.

    `raw_text` always holds the full extracted text for the unit's region so
    a unit whose stat table failed structured parsing (parse_confidence =
    "fallback") is still stored, cropped, and text-searchable — never dropped.
    """

    unit_id: str  # "{faction-slug}--{unit-slug}"
    faction: str
    name: str
    movement: str = ""
    toughness: str = ""
    save: str = ""
    wounds: str = ""
    leadership: str = ""
    oc: str = ""
    keywords: list[str] = field(default_factory=list)
    abilities_text: str = ""
    points: str = ""
    weapons: list[WeaponProfile] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    bboxes: dict[int, list[BBox]] = field(default_factory=dict)
    crop_paths: list[str] = field(default_factory=list)
    doc_hash: str = ""
    parse_confidence: str = "ok"  # "ok" | "fallback"
    raw_text: str = ""
