"""Phase 3 acceptance (§6): hybrid retrieval puts the right rule at rank 1
for title queries; adversarial near-misses don't swap; expansion attaches
related rules to the top hit."""

from __future__ import annotations

from rulehound.config import SearchConfig
from rulehound.ingest.embed import HashingEmbedder
from rulehound.models import ScoredRule
from rulehound.search.expand import related_for
from rulehound.search.hybrid import hybrid_search, rrf_fuse

CFG = SearchConfig()


def top1(store, query: str) -> str:
    results, _ = hybrid_search(store, HashingEmbedder(384), query, CFG)
    assert results, f"no results for {query!r}"
    return results[0].rule_id.split(".")[-1]


def test_direct_title_queries(store) -> None:
    assert top1(store, "disembark") == "disembark"
    assert top1(store, "deep strike") == "deep-strike"
    assert top1(store, "fall back") == "fall-back"
    assert top1(store, "engagement range") == "engagement-range"


def test_paraphrase_queries(store) -> None:
    assert top1(store, "getting out of a transport") == "disembark"
    assert top1(store, "arriving from reserves") == "reserves"


def test_symptom_queries(store) -> None:
    assert top1(store, "who controls an objective marker") == "objective-control"


def test_adversarial_embark_vs_disembark(store) -> None:
    assert top1(store, "embark") == "embark"
    assert top1(store, "disembark") == "disembark"


def test_keyword_only_mode_still_works(store) -> None:
    results, timings = hybrid_search(store, None, "disembark", CFG)
    assert results[0].rule_id.endswith("disembark")
    assert timings.embed_ms == 0.0
    assert timings.vector_ms == 0.0


def test_rrf_fusion_combines_sources() -> None:
    kw = [ScoredRule("a", "A", -1.0), ScoredRule("b", "B", -2.0)]
    vec = [ScoredRule("b", "B", 0.9), ScoredRule("c", "C", 0.8)]
    fused = rrf_fuse({"keyword": kw, "vector": vec}, query="zzz", cfg=CFG)
    ids = [f.rule_id for f in fused]
    assert ids[0] == "b"  # appears in both lists
    b = fused[0]
    assert b.score == (1 / (CFG.rrf_k + 2)) + (1 / (CFG.rrf_k + 1))


def test_title_boost_dominates_rrf() -> None:
    kw = [ScoredRule("a", "Advance", -1.0), ScoredRule("b", "Fall Back", -2.0)]
    fused = rrf_fuse({"keyword": kw}, query="fall back", cfg=CFG)
    assert fused[0].rule_id == "b"


def test_expansion_on_top_result(store) -> None:
    results, _ = hybrid_search(store, HashingEmbedder(384), "disembark", CFG)
    related = related_for(store, results[0].rule_id, CFG.related_max)
    assert 0 < len(related) <= CFG.related_max
    for r in related:
        assert set(r) == {"rule_id", "title"}  # titles + ids only, no text
    titles = {r["title"] for r in related}
    assert "Reserves" in titles
