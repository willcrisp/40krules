"""Embedding generation (design doc §4.5).

Default model is BAAI/bge-small-en-v1.5 via sentence-transformers (the
`embeddings` extra). A deterministic "hash" embedder is provided for tests
and as a no-ML fallback. If no embedder can be constructed the system runs
keyword-only — vector search is additive, never load-bearing (§8).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from ..config import EmbeddingConfig


class Embedder(Protocol):
    name: str
    dimension: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic bag-of-token-ngrams hashed into a fixed dim, L2-normalised.

    Purely lexical (no semantics) but exercises the full vector path in tests
    without pulling in torch.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.name = "hash"
        self.dimension = dimension

    def _features(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        feats = list(tokens)
        feats += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return feats

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            for feat in self._features(text):
                h = hashlib.md5(feat.encode()).digest()
                idx = int.from_bytes(h[:4], "little") % self.dimension
                sign = 1.0 if h[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dimension: int) -> None:
        from sentence_transformers import SentenceTransformer  # lazy, heavy import

        self.name = model_name
        self.dimension = dimension
        self._model = SentenceTransformer(model_name)
        actual = self._model.get_sentence_embedding_dimension()
        if actual != dimension:
            raise ValueError(
                f"Model {model_name} produces {actual}-dim vectors; config says {dimension}"
            )

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


def get_embedder(cfg: EmbeddingConfig, log=None) -> Embedder | None:
    """Build the configured embedder, or None (keyword-only mode) on failure."""
    if cfg.model == "hash":
        return HashingEmbedder(cfg.dimension)
    try:
        return SentenceTransformerEmbedder(cfg.model, cfg.dimension)
    except Exception as exc:  # missing extra, no network for model download, ...
        if log:
            log(f"embedder '{cfg.model}' unavailable ({exc}); running keyword-only")
        return None


def embedding_text(title: str, text: str) -> str:
    """What gets embedded: title + text, never commentary (§4.5)."""
    return f"{title}\n{text}"
