"""Shared fixtures. Everything runs against the synthetic fixture PDF —
CI never touches GW content (design doc §10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rulehound.config import Config, EmbeddingConfig, PathsConfig
from rulehound.ingest.pipeline import run_ingest
from rulehound.store.sqlite_store import SqliteStore

from .fixtures.build_fixture import build_fixture_pdf


@pytest.fixture(scope="session")
def fixture_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_fixture_pdf(tmp_path_factory.mktemp("pdf") / "fixture.pdf")


def make_config(root: Path) -> Config:
    cfg = Config()
    cfg.paths = PathsConfig(
        data_dir=root / "data",
        raw_dir=root / "data/raw",
        pages_dir=root / "data/pages",
        crops_dir=root / "data/crops",
        db_path=root / "data/rulehound.db",
    )
    # deterministic, no-ML embedder so the vector path is exercised in CI
    cfg.embedding = EmbeddingConfig(model="hash", dimension=384)
    return cfg


@pytest.fixture(scope="session")
def ingested(fixture_pdf: Path, tmp_path_factory: pytest.TempPathFactory):
    """(config, summary) for a fully ingested fixture corpus."""
    root = tmp_path_factory.mktemp("rulehound")
    cfg = make_config(root)
    summary = run_ingest(fixture_pdf, cfg, log=lambda m: None)
    return cfg, summary


@pytest.fixture()
def store(ingested) -> SqliteStore:
    cfg, _ = ingested
    s = SqliteStore(cfg.paths.db_path, vector_dim=cfg.embedding.dimension)
    yield s
    s.close()
