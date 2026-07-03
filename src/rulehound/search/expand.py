"""1-hop cross-reference expansion for the top result (design doc §6 step 4)."""

from __future__ import annotations

from ..store.base import SearchStore

_KINDS = ("explicit_see", "title_mention")
_KIND_PRIORITY = {k: i for i, k in enumerate(_KINDS)}


def related_for(store: SearchStore, rule_id: str, max_n: int = 4) -> list[dict[str, str]]:
    """Outgoing explicit_see/title_mention neighbours: titles + rule_ids only."""
    base_id = rule_id.split("--part-")[0]
    refs = [r for r in store.neighbours(base_id, direction="out") if r.kind in _KINDS]
    refs.sort(key=lambda r: _KIND_PRIORITY[r.kind])

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.to_rule in seen:
            continue
        seen.add(ref.to_rule)
        # Split rules store parts as {id}--part-N with no base row.
        rule = store.get_rule(ref.to_rule) or store.get_rule(f"{ref.to_rule}--part-1")
        title = rule.title.split(" (part ")[0] if rule else ref.to_rule
        out.append({"rule_id": ref.to_rule, "title": title})
        if len(out) >= max_n:
            break
    return out
