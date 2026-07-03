"""Phase 0 acceptance: config loads, schema applies, /health returns ok."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rulehound.config import load_config
from rulehound.api.app import create_app
from rulehound.store.sqlite_store import SqliteStore

from .conftest import make_config


def test_config_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.search.top_k == 5
    assert cfg.embedding.dimension == 384


def test_config_from_file(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[embedding]\nmodel = "hash"\ndimension = 128\n[search]\ntop_k = 3\n'
    )
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.embedding.model == "hash"
    assert cfg.embedding.dimension == 128
    assert cfg.search.top_k == 3
    assert cfg.paths.db_path == tmp_path / "data/rulehound.db"


def test_schema_applies(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "test.db", vector_dim=8)
    assert store.rule_count() == 0
    assert store.keyword_search("anything", 5) == []
    store.set_meta("k", "v")
    assert store.get_meta("k") == "v"
    store.close()


def test_health_on_empty_db(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["rules"] == 0
    assert body["embedder"] == "hash"


def test_index_serves_ui(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        res = client.get("/")
    assert res.status_code == 200
    assert "RULEHOUND" in res.text
