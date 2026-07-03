"""SearchStore protocol (design doc §6).

The SQLite implementation is the prototype store; a turbopuffer adapter can
implement the same protocol later without changing anything above it.
"""

from __future__ import annotations

from typing import Protocol

from ..models import RuleChunk, RuleRef, ScoredRule


class SearchStore(Protocol):
    # --- query side (§6) ---
    def keyword_search(
        self, query: str, k: int, extra_terms: list[str] | None = None
    ) -> list[ScoredRule]: ...

    def vector_search(self, embedding: list[float], k: int) -> list[ScoredRule]: ...

    def get_rule(self, rule_id: str) -> RuleChunk | None: ...

    def neighbours(self, rule_id: str, direction: str = "out") -> list[RuleRef]: ...

    # --- ingest side ---
    def replace_document(
        self, chunks: list[RuleChunk], refs: list[RuleRef], doc_hash: str
    ) -> None: ...

    def store_vectors(
        self, vectors: dict[str, list[float]], model_name: str, dimension: int
    ) -> None: ...

    def load_vocab(self) -> dict[str, int]: ...

    def get_meta(self, key: str) -> str | None: ...

    def set_meta(self, key: str, value: str) -> None: ...

    def rule_count(self) -> int: ...
