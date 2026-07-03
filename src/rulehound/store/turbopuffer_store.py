"""Turbopuffer adapter — production swap, stub only in the prototype (§2, §6).

When written for real this implements SearchStore with a single native
hybrid (BM25 + vector) query; fusion happens server-side and hybrid.py's
RRF is bypassed. Nothing above the protocol changes.
"""

from __future__ import annotations


class TurbopufferStore:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "TurbopufferStore is a post-prototype adapter; use SqliteStore."
        )
