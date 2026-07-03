"""Ingest orchestration (§4): idempotency keyed on content hash, phase
resume, and derived artifacts on disk."""

from __future__ import annotations

from rulehound.ingest.pipeline import file_hash, run_ingest
from rulehound.store.sqlite_store import SqliteStore

from .conftest import make_config


def test_summary_counts(ingested) -> None:
    _, summary = ingested
    assert summary["chunks"] >= 9
    assert summary["refs"] > 0
    assert summary["embedded"] == summary["chunks"]
    assert summary["vector_enabled"] is True


def test_artifacts_on_disk(ingested) -> None:
    cfg, summary = ingested
    short = summary["doc_hash"][:12]
    assert (cfg.paths.data_dir / f"blocks_{short}.jsonl").is_file()
    assert (cfg.paths.data_dir / f"chunks_{short}.json").is_file()
    assert list(cfg.paths.pages_dir.glob("page_*.png"))
    assert list(cfg.paths.crops_dir.glob("*.png"))
    assert cfg.paths.db_path.is_file()


def test_reingest_is_idempotent(fixture_pdf, tmp_path) -> None:
    cfg = make_config(tmp_path)
    logs: list[str] = []
    first = run_ingest(fixture_pdf, cfg, log=logs.append)
    logs.clear()
    second = run_ingest(fixture_pdf, cfg, log=logs.append)
    assert first["doc_hash"] == second["doc_hash"] == file_hash(fixture_pdf)
    assert second["chunks"] == first["chunks"]
    # second run reuses completed phases rather than re-extracting
    assert any("reused" in m for m in logs)

    store = SqliteStore(cfg.paths.db_path, vector_dim=cfg.embedding.dimension)
    assert store.rule_count() == first["chunks"]
    store.close()


def test_model_mismatch_disables_vectors(fixture_pdf, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from rulehound.api.app import create_app

    cfg = make_config(tmp_path)
    run_ingest(fixture_pdf, cfg, log=lambda m: None)

    # simulate a config that names a different model than the DB was built with
    cfg.embedding.model = "some-other-model"
    app = create_app(cfg)
    with TestClient(app) as c:
        health = c.get("/health").json()
        assert health["embedder"] is None
        assert "mismatch" in (health["embedder_error"] or "")
        # keyword-only search still functions (§8: vector is additive)
        res = c.get("/search", params={"q": "disembark"})
        assert res.json()["results"][0]["rule_id"].endswith("disembark")
