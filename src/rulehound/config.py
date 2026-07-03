"""Config loading from config.toml (design doc §3)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    pages_dir: Path = Path("data/pages")
    crops_dir: Path = Path("data/crops")
    db_path: Path = Path("data/rulehound.db")

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.raw_dir, self.pages_dir, self.crops_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384


@dataclass
class SearchConfig:
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    title_exact_boost: float = 0.05
    title_prefix_boost: float = 0.02
    related_max: int = 4


@dataclass
class ImagesConfig:
    dpi: int = 150
    crop_padding_px: int = 12


@dataclass
class ChunkingConfig:
    max_tokens: int = 1200


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    images: ImagesConfig = field(default_factory=ImagesConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load config.toml. Search order: explicit arg, $RULEHOUND_CONFIG, ./config.toml.

    Missing file or missing keys fall back to defaults. Relative paths in
    [paths] are resolved against the config file's directory.
    """
    if path is None:
        path = os.environ.get("RULEHOUND_CONFIG") or "config.toml"
    path = Path(path)

    raw: dict = {}
    base = Path.cwd()
    if path.is_file():
        raw = tomllib.loads(path.read_text())
        base = path.resolve().parent

    cfg = Config()

    p = raw.get("paths", {})
    cfg.paths = PathsConfig(
        data_dir=base / p.get("data_dir", "data"),
        raw_dir=base / p.get("raw_dir", "data/raw"),
        pages_dir=base / p.get("pages_dir", "data/pages"),
        crops_dir=base / p.get("crops_dir", "data/crops"),
        db_path=base / p.get("db_path", "data/rulehound.db"),
    )

    e = raw.get("embedding", {})
    cfg.embedding = EmbeddingConfig(
        model=e.get("model", cfg.embedding.model),
        dimension=int(e.get("dimension", cfg.embedding.dimension)),
    )

    s = raw.get("search", {})
    cfg.search = SearchConfig(
        top_k=int(s.get("top_k", cfg.search.top_k)),
        candidate_k=int(s.get("candidate_k", cfg.search.candidate_k)),
        rrf_k=int(s.get("rrf_k", cfg.search.rrf_k)),
        title_exact_boost=float(s.get("title_exact_boost", cfg.search.title_exact_boost)),
        title_prefix_boost=float(s.get("title_prefix_boost", cfg.search.title_prefix_boost)),
        related_max=int(s.get("related_max", cfg.search.related_max)),
    )

    i = raw.get("images", {})
    cfg.images = ImagesConfig(
        dpi=int(i.get("dpi", cfg.images.dpi)),
        crop_padding_px=int(i.get("crop_padding_px", cfg.images.crop_padding_px)),
    )

    c = raw.get("chunking", {})
    cfg.chunking = ChunkingConfig(max_tokens=int(c.get("max_tokens", cfg.chunking.max_tokens)))

    return cfg
