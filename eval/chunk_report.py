"""Phase 1 acceptance report (design doc §4.2, §10).

Usage: uv run python eval/chunk_report.py
Checks the 15-rule spot-check checklist against the ingested DB and writes
eval/chunk_report.md listing each rule, its detected boundaries, pass/fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rulehound.config import load_config  # noqa: E402
from rulehound.store.sqlite_store import SqliteStore  # noqa: E402

# The 15 known rules to spot-check (§4.2 acceptance)
CHECKLIST = [
    "Disembark",
    "Embark",
    "Deep Strike",
    "Engagement Range",
    "Fall Back",
    "Advance",
    "Normal Move",
    "Reserves",
    "Objective Markers",
    "Battle-shock Tests",
    "Fire Overwatch",
    "Pile In",
    "Consolidation",
    "Mortal Wounds",
    "Invulnerable Saves",
]


def main() -> int:
    cfg = load_config()
    store = SqliteStore(cfg.paths.db_path, vector_dim=cfg.embedding.dimension)
    if store.rule_count() == 0:
        print("DB is empty — run ingest first.")
        return 2

    rows = store.db.execute(
        "SELECT rule_id, title, section_path, page_start, page_end,"
        " length(text) AS text_len, commentary IS NOT NULL AS has_commentary"
        " FROM rules ORDER BY rule_id"
    ).fetchall()
    by_title = {}
    for r in rows:
        by_title.setdefault(r["title"].split(" (part ")[0].lower(), []).append(r)

    lines = [
        "# Chunk report (Phase 1 acceptance)",
        "",
        f"Total chunks: {len(rows)}",
        "",
        "| rule | found | rule_id | section path | pages | text chars | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    all_pass = True
    for name in CHECKLIST:
        found = by_title.get(name.lower(), [])
        if found:
            r = found[0]
            ok = r["text_len"] > 50
            all_pass &= ok
            lines.append(
                f"| {name} | yes | `{r['rule_id']}` | {r['section_path']} "
                f"| {r['page_start']}–{r['page_end']} | {r['text_len']} | {'PASS' if ok else 'FAIL (too short)'} |"
            )
        else:
            all_pass = False
            lines.append(f"| {name} | **no** | — | — | — | — | FAIL |")

    lines += ["", f"**Overall: {'PASS' if all_pass else 'FAIL'}**", ""]
    lines += ["## All chunk titles", ""]
    lines += [f"- `{r['rule_id']}` — {r['title']} (p{r['page_start']})" for r in rows]

    report = "\n".join(lines) + "\n"
    out = Path(__file__).parent / "chunk_report.md"
    out.write_text(report)
    print(report[:2000])
    print(f"\nwrote {out}")
    store.close()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
