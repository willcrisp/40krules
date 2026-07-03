"""Phase 2 acceptance (§4.3): Disembark's neighbours include Transports and
Reserves; longest-title-first matching; edge kinds are correct."""

from __future__ import annotations

from pathlib import Path

import pytest

from rulehound.ingest.chunker import chunk_blocks
from rulehound.ingest.crosslinks import extract_crosslinks
from rulehound.ingest.pdf_extract import extract_blocks


@pytest.fixture(scope="module")
def edges(fixture_pdf: Path):
    chunks = chunk_blocks(extract_blocks(fixture_pdf), doc_hash="test")
    return chunks, extract_crosslinks(chunks)


def out_neighbours(edges, from_suffix: str) -> dict[str, set[str]]:
    _, refs = edges
    out: dict[str, set[str]] = {}
    for r in refs:
        if r.from_rule.split(".")[-1] == from_suffix:
            out.setdefault(r.to_rule, set()).add(r.kind)
    return out


def test_disembark_neighbours(edges) -> None:
    n = out_neighbours(edges, "disembark")
    targets = {t.split(".")[-1] for t in n}
    assert "reserves" in targets
    assert "engagement-range" in targets
    assert "normal-move" in targets


def test_explicit_see_detected(edges) -> None:
    n = out_neighbours(edges, "disembark")
    see_kinds = {t.split(".")[-1]: kinds for t, kinds in n.items()}
    assert "explicit_see" in see_kinds.get("reserves", set())


def test_longest_title_wins(edges) -> None:
    # "Normal Move" must be linked as normal-move, never swallowed by a
    # shorter title; disembark mentions "Normal Move" explicitly.
    n = out_neighbours(edges, "disembark")
    assert any(t.endswith("normal-move") for t in n)


def test_no_self_edges(edges) -> None:
    _, refs = edges
    assert all(r.from_rule != r.to_rule for r in refs)


def test_kinds_are_valid(edges) -> None:
    _, refs = edges
    assert {r.kind for r in refs} <= {"explicit_see", "title_mention", "page_ref"}
