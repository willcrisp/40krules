"""Golden-set evaluation (design doc §9).

Usage: uv run python eval/run_eval.py [--golden eval/golden_queries.yaml]

Reports Recall@1, Recall@3, MRR for hybrid retrieval plus keyword-only and
vector-only ablations, and writes eval/eval_report.md.
Gate: hybrid Recall@3 >= 0.90.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rulehound.config import load_config  # noqa: E402
from rulehound.ingest.embed import get_embedder  # noqa: E402
from rulehound.search.hybrid import hybrid_search, rrf_fuse  # noqa: E402
from rulehound.store.sqlite_store import SqliteStore  # noqa: E402

GATE_RECALL_AT_3 = 0.90


def matches(rule_id: str, expect: str) -> bool:
    tail = rule_id.split("--part-")[0].split(".")[-1]
    return tail == expect


def evaluate(rank_fn, queries: list[dict]) -> dict:
    hits1 = hits3 = 0
    rr_sum = 0.0
    failures: list[tuple[str, str, list[str]]] = []
    for item in queries:
        q, expect = item["q"], item["expect"]
        ranked = rank_fn(q)
        rank = next((i for i, rid in enumerate(ranked, 1) if matches(rid, expect)), None)
        if rank == 1:
            hits1 += 1
        if rank is not None and rank <= 3:
            hits3 += 1
        rr_sum += 1.0 / rank if rank else 0.0
        if rank is None or rank > 3:
            failures.append((q, expect, ranked[:3]))
    n = len(queries)
    return {
        "recall@1": hits1 / n,
        "recall@3": hits3 / n,
        "mrr": rr_sum / n,
        "n": n,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(Path(__file__).parent / "golden_queries.yaml"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--report", default=str(Path(__file__).parent / "eval_report.md"))
    args = parser.parse_args()

    queries = yaml.safe_load(Path(args.golden).read_text())["queries"]
    cfg = load_config(args.config)
    store = SqliteStore(cfg.paths.db_path, vector_dim=cfg.embedding.dimension)
    if store.rule_count() == 0:
        print("DB is empty — run ingest first: python -m rulehound.ingest data/raw/core_rules.pdf")
        return 2
    embedder = get_embedder(cfg.embedding, log=print)

    def hybrid(q: str) -> list[str]:
        results, _ = hybrid_search(store, embedder, q, cfg.search)
        return [r.rule_id for r in results]

    def keyword_only(q: str) -> list[str]:
        ranked = store.keyword_search(q, cfg.search.candidate_k)
        fused = rrf_fuse({"keyword": ranked}, q, cfg.search)
        return [r.rule_id for r in fused[: cfg.search.top_k]]

    def vector_only(q: str) -> list[str]:
        if embedder is None:
            return []
        ranked = store.vector_search(embedder.encode([q])[0], cfg.search.candidate_k)
        return [r.rule_id for r in ranked[: cfg.search.top_k]]

    runs = {
        "hybrid": evaluate(hybrid, queries),
        "keyword-only": evaluate(keyword_only, queries),
        "vector-only": evaluate(vector_only, queries),
    }

    lines = ["# Rulehound eval report", "", f"Golden set: {runs['hybrid']['n']} queries", ""]
    lines += ["| run | Recall@1 | Recall@3 | MRR |", "|---|---|---|---|"]
    for name, r in runs.items():
        lines.append(f"| {name} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['mrr']:.3f} |")
    gate = runs["hybrid"]["recall@3"] >= GATE_RECALL_AT_3
    lines += ["", f"**Gate (hybrid Recall@3 >= {GATE_RECALL_AT_3}): {'PASS' if gate else 'FAIL'}**"]
    if runs["hybrid"]["failures"]:
        lines += ["", "## Hybrid failures (expected not in top 3)", ""]
        for q, expect, top3 in runs["hybrid"]["failures"]:
            lines.append(f"- `{q}` expected **{expect}**, got: {', '.join(top3) or '(none)'}")
    report = "\n".join(lines) + "\n"
    Path(args.report).write_text(report)
    print(report)
    store.close()
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
