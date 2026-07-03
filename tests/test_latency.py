"""Phase 4 acceptance (§8): p95 server-side latency <= 150 ms, warm process.

Uses the hash embedder (no ML deps); the real-model embed budget (60 ms) is
measured on the dev box against the real corpus.
"""

from __future__ import annotations

import statistics
import time

from fastapi.testclient import TestClient

from rulehound.api.app import create_app

QUERIES = [
    "disembark",
    "getting out of a transport",
    "deep strike",
    "arriving from reserves",
    "engagement range",
    "who controls an objective",
    "fall back and shoot",
    "advance",
    "embark",
    "normal move",
]

BUDGET_MS = 150


def test_p95_latency_within_budget(ingested) -> None:
    cfg, _ = ingested
    app = create_app(cfg)
    with TestClient(app) as client:
        client.get("/search", params={"q": "warmup"})
        samples: list[float] = []
        for _ in range(3):
            for q in QUERIES:
                t0 = time.perf_counter()
                res = client.get("/search", params={"q": q})
                elapsed = (time.perf_counter() - t0) * 1000
                assert res.status_code == 200
                samples.append(elapsed)
                # server-side timing must also be reported
                assert res.json()["latency_ms"]["total"] <= BUDGET_MS

    samples.sort()
    p50 = statistics.median(samples)
    p95 = samples[int(len(samples) * 0.95) - 1]
    print(f"\nlatency p50={p50:.1f}ms p95={p95:.1f}ms over {len(samples)} requests")
    assert p95 <= BUDGET_MS, f"p95 {p95:.1f}ms exceeds {BUDGET_MS}ms budget"
