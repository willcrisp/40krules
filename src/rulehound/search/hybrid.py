"""Hybrid retrieval: BM25 + vector with RRF fusion (design doc §6, §8)."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..config import SearchConfig
from ..ingest.embed import Embedder
from ..models import ScoredRule
from ..store.base import SearchStore

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rulehound-search")


@dataclass
class Timings:
    embed_ms: float = 0.0
    keyword_ms: float = 0.0
    vector_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "embed": round(self.embed_ms, 2),
            "keyword": round(self.keyword_ms, 2),
            "vector": round(self.vector_ms, 2),
            "total": round(self.total_ms, 2),
        }


@dataclass
class FusedResult:
    rule_id: str
    title: str
    score: float
    sources: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def rrf_fuse(
    ranked_lists: dict[str, list[ScoredRule]],
    query: str,
    cfg: SearchConfig,
) -> list[FusedResult]:
    """Reciprocal Rank Fusion (k_rrf from config) + title boost.

    Title exact/prefix boosts sit above the maximum attainable RRF score so
    title hits dominate — most queries name the rule (§6).
    """
    fused: dict[str, FusedResult] = {}
    for source, results in ranked_lists.items():
        for rank, r in enumerate(results, start=1):
            entry = fused.setdefault(r.rule_id, FusedResult(r.rule_id, r.title, 0.0))
            entry.score += 1.0 / (cfg.rrf_k + rank)
            entry.sources.append(source)

    q = _norm(query)
    for entry in fused.values():
        t = _norm(entry.title)
        if not t:
            continue
        if q == t:
            entry.score += cfg.title_exact_boost
        elif t.startswith(q) or q.startswith(t) or t in q:
            entry.score += cfg.title_prefix_boost

    return sorted(fused.values(), key=lambda e: e.score, reverse=True)


def hybrid_search(
    store: SearchStore,
    embedder: Embedder | None,
    query: str,
    cfg: SearchConfig,
) -> tuple[list[FusedResult], Timings]:
    """Embed, run FTS5 and vector search in parallel, fuse, return top_k."""
    t = Timings()
    t0 = time.perf_counter()

    embedding: list[float] | None = None
    if embedder is not None:
        te = time.perf_counter()
        embedding = embedder.encode([query])[0]
        t.embed_ms = (time.perf_counter() - te) * 1000

    def run_keyword() -> list[ScoredRule]:
        tk = time.perf_counter()
        out = store.keyword_search(query, cfg.candidate_k)
        t.keyword_ms = (time.perf_counter() - tk) * 1000
        return out

    def run_vector() -> list[ScoredRule]:
        if embedding is None:
            return []
        tv = time.perf_counter()
        out = store.vector_search(embedding, cfg.candidate_k)
        t.vector_ms = (time.perf_counter() - tv) * 1000
        return out

    kw_future = _pool.submit(run_keyword)
    vec_future = _pool.submit(run_vector)
    ranked = {"keyword": kw_future.result(), "vector": vec_future.result()}

    results = rrf_fuse(ranked, query, cfg)[: cfg.top_k]
    t.total_ms = (time.perf_counter() - t0) * 1000
    return results, t
