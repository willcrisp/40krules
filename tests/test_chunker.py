"""Phase 1 acceptance (§4.2): each known rule is its own chunk with correct
boundaries; hierarchy is tracked; sidebars land in commentary; oversized
chunks split on paragraphs."""

from __future__ import annotations

from pathlib import Path

import pytest

from rulehound.ingest.chunker import chunk_blocks, slugify
from rulehound.ingest.pdf_extract import extract_blocks
from rulehound.models import RuleChunk

EXPECTED_RULES = [
    "Engagement Range",
    "Objective Control",
    "Normal Move",
    "Advance",
    "Fall Back",
    "Embark",
    "Disembark",
    "Reserves",
    "Deep Strike",
]


@pytest.fixture(scope="module")
def chunks(fixture_pdf: Path) -> list[RuleChunk]:
    return chunk_blocks(extract_blocks(fixture_pdf), doc_hash="test")


def by_title(chunks: list[RuleChunk], title: str) -> RuleChunk:
    matches = [c for c in chunks if c.title == title]
    assert matches, f"no chunk titled {title!r}; got {[c.title for c in chunks]}"
    return matches[0]


def test_every_known_rule_is_its_own_chunk(chunks: list[RuleChunk]) -> None:
    titles = {c.title for c in chunks}
    missing = [t for t in EXPECTED_RULES if t not in titles]
    assert not missing, f"missing chunks: {missing}"


def test_section_path_tracks_hierarchy(chunks: list[RuleChunk]) -> None:
    disembark = by_title(chunks, "Disembark")
    assert disembark.section_path == "Movement Phase > Transports > Disembark"
    assert disembark.rule_id == "movement-phase.transports.disembark"

    deep_strike = by_title(chunks, "Deep Strike")
    assert deep_strike.section_path == "Reinforcements > Deep Strike"


def test_text_boundaries(chunks: list[RuleChunk]) -> None:
    disembark = by_title(chunks, "Disembark")
    assert "getting out of a transport" in disembark.text
    # next rule's text must not bleed in
    assert "Reserves start the battle" not in disembark.text
    embark = by_title(chunks, "Embark")
    assert "boarding the vehicle" in embark.text
    assert "getting out" not in embark.text


def test_sidebar_becomes_commentary(chunks: list[RuleChunk]) -> None:
    disembark = by_title(chunks, "Disembark")
    assert disembark.commentary and "Designer's note" in disembark.commentary
    assert "Designer's note" not in disembark.text


def test_pages_and_bboxes_recorded(chunks: list[RuleChunk]) -> None:
    for c in chunks:
        assert c.page_start >= 1
        assert c.page_end >= c.page_start
        assert c.bboxes, f"{c.rule_id} has no bboxes"


def test_oversized_chunk_splits_on_paragraphs() -> None:
    from rulehound.ingest.pdf_extract import TextBlock

    para = "words " * 300  # ~390 estimated tokens per paragraph
    blocks = [
        TextBlock(page=1, bbox=(0, 0, 100, 10), text="Big Rule", font_size=20,
                  is_bold=True, is_heading=True, heading_level=1),
        *[
            TextBlock(page=1, bbox=(0, 20 + i, 100, 30 + i), text=para.strip(),
                      font_size=10, is_bold=False)
            for i in range(5)
        ],
    ]
    chunks = chunk_blocks(blocks, max_tokens=1200)
    assert len(chunks) > 1
    assert all(c.rule_id.startswith("big-rule--part-") for c in chunks)
    assert all(c.section_path == "Big Rule" for c in chunks)


def test_slugify() -> None:
    assert slugify("Movement Phase") == "movement-phase"
    assert slugify("Designer's Note!") == "designer-s-note"
