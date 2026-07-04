"""SQLite store for the faction-datasheet corpus.

A deliberately separate DB file from the rules corpus (`sqlite_store.py`) so
a datasheet ingest can never wipe or corrupt the rules index. Unlike
`replace_document` there, replacement is scoped **per faction** — uploading
Faction B leaves Faction A's units intact.

Search is exposed as two surfaces (units, weapons) shaped like the
`SearchStore` query side, so `hybrid_search` runs against either unchanged.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections import Counter
from importlib import resources
from pathlib import Path

from ..models import ScoredRule, UnitProfile, WeaponProfile
from .sqlite_store import META_DIM, META_MODEL, _fts_query, _serialize_f32


def _slugify(text: str) -> str:
    # same rules as ingest.chunker.slugify, duplicated to keep the store
    # import-light (chunker pulls in PyMuPDF)
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def faction_hash_key(faction: str) -> str:
    return f"faction_doc_hash:{faction}"


class _SearchSurface:
    """Adapter matching the SearchStore query shape hybrid_search expects."""

    def __init__(self, keyword, vector) -> None:
        self.keyword_search = keyword
        self.vector_search = vector


class DatasheetStore:
    """One file: units + weapon profiles + FTS5 + vectors. Thread-safe via per-call lock."""

    def __init__(self, db_path: str | Path, vector_dim: int = 384) -> None:
        self.db_path = Path(db_path)
        self.vector_dim = vector_dim
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.vector_enabled = self._load_vec_extension()
        self._apply_schema()
        self.units = _SearchSurface(self.keyword_search_units, self.vector_search_units)
        self.weapons = _SearchSurface(self.keyword_search_weapons, self.vector_search_weapons)

    def _load_vec_extension(self) -> bool:
        try:
            import sqlite_vec

            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
            return True
        except Exception:
            return False

    def _apply_schema(self) -> None:
        schema = resources.files("rulehound.store").joinpath("datasheet_schema.sql").read_text(encoding="utf-8")
        with self._lock:
            self.db.executescript(schema)
            if self.vector_enabled:
                self.db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS units_vec USING vec0("
                    "  unit_rowid INTEGER PRIMARY KEY,"
                    f"  embedding FLOAT[{self.vector_dim}]"
                    ")"
                )
            self.db.commit()

    def close(self) -> None:
        self.db.close()

    # --- ingest side -------------------------------------------------------

    def replace_faction(self, units: list[UnitProfile], faction: str, doc_hash: str) -> None:
        """Replace one faction's units; every other faction is untouched."""
        with self._lock:
            cur = self.db.cursor()
            old_rowids = [
                row["rowid"]
                for row in cur.execute("SELECT rowid FROM units WHERE faction = ?", (faction,))
            ]
            if old_rowids:
                marks = ",".join("?" * len(old_rowids))
                if self.vector_enabled:
                    # rowids get reused by SQLite after DELETE — stale vectors
                    # would silently attach to the wrong unit.
                    cur.execute(f"DELETE FROM units_vec WHERE unit_rowid IN ({marks})", old_rowids)
                cur.execute(
                    "DELETE FROM weapon_profiles WHERE unit_id IN"
                    " (SELECT unit_id FROM units WHERE faction = ?)",
                    (faction,),
                )
                cur.execute("DELETE FROM units WHERE faction = ?", (faction,))

            cur.executemany(
                "INSERT INTO units (unit_id, faction, name, movement, toughness, save,"
                " wounds, leadership, oc, keywords, abilities_text, points, raw_text,"
                " parse_confidence, page_start, page_end, crop_paths, doc_hash)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        u.unit_id, faction, u.name, u.movement, u.toughness, u.save,
                        u.wounds, u.leadership, u.oc, json.dumps(u.keywords),
                        u.abilities_text, u.points, u.raw_text, u.parse_confidence,
                        u.page_start, u.page_end, json.dumps(u.crop_paths), doc_hash,
                    )
                    for u in units
                ],
            )
            weapon_rows = []
            for u in units:
                seen: Counter[str] = Counter()
                for w in u.weapons:
                    slug = _slugify(w.name)
                    seen[slug] += 1
                    w.weapon_id = f"{u.unit_id}--{slug}" + (
                        f"-{seen[slug]}" if seen[slug] > 1 else ""
                    )
                    weapon_rows.append(
                        (
                            w.weapon_id, u.unit_id, w.name, w.range, w.attacks,
                            w.skill, w.strength, w.ap, w.damage, json.dumps(w.keywords),
                        )
                    )
            cur.executemany(
                "INSERT INTO weapon_profiles (weapon_id, unit_id, name, \"range\","
                " attacks, skill, strength, ap, damage, keywords)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                weapon_rows,
            )
            # External-content FTS tables can't track our scoped deletes; a
            # rebuild from the content tables is always consistent.
            cur.execute("INSERT INTO units_fts(units_fts) VALUES('rebuild')")
            cur.execute("INSERT INTO weapons_fts(weapons_fts) VALUES('rebuild')")

            # Corpus vocabulary for spell correction — rebuilt over the whole
            # corpus (all factions), matching exactly what FTS indexes.
            from ..search.spell import tokenize

            counts: Counter[str] = Counter()
            for row in cur.execute(
                "SELECT name, keywords, abilities_text, raw_text FROM units"
            ).fetchall():
                counts.update(
                    tokenize(f"{row['name']} {row['keywords']} {row['abilities_text']} {row['raw_text']}")
                )
            for row in cur.execute("SELECT name, keywords FROM weapon_profiles").fetchall():
                counts.update(tokenize(f"{row['name']} {row['keywords']}"))
            cur.execute("DELETE FROM vocab")
            cur.executemany("INSERT INTO vocab (term, freq) VALUES (?,?)", counts.items())

            cur.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                (faction_hash_key(faction), doc_hash),
            )
            self.db.commit()

    def store_vectors(
        self, vectors: dict[str, list[float]], model_name: str, dimension: int
    ) -> None:
        """Upsert vectors for the given unit_ids only — other factions' vectors stay."""
        if not self.vector_enabled:
            raise RuntimeError("sqlite-vec extension not available")
        if dimension != self.vector_dim:
            raise ValueError(f"vector dim {dimension} != store dim {self.vector_dim}")
        with self._lock:
            cur = self.db.cursor()
            rowids = {
                row["unit_id"]: row["rowid"]
                for row in cur.execute("SELECT rowid, unit_id FROM units")
            }
            target = [rowids[uid] for uid in vectors if uid in rowids]
            if target:
                marks = ",".join("?" * len(target))
                cur.execute(f"DELETE FROM units_vec WHERE unit_rowid IN ({marks})", target)
            cur.executemany(
                "INSERT INTO units_vec (unit_rowid, embedding) VALUES (?,?)",
                [
                    (rowids[uid], _serialize_f32(vec))
                    for uid, vec in vectors.items()
                    if uid in rowids
                ],
            )
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (META_MODEL, model_name))
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (META_DIM, str(dimension)))
            self.db.commit()

    # --- query side --------------------------------------------------------

    def keyword_search_units(
        self, query: str, k: int, extra_terms: list[str] | None = None
    ) -> list[ScoredRule]:
        fts = _fts_query(query, extra_terms)
        with self._lock:
            rows = self.db.execute(
                "SELECT u.unit_id, u.name, bm25(units_fts) AS rank"
                " FROM units_fts JOIN units u ON u.rowid = units_fts.rowid"
                " WHERE units_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts, k),
            ).fetchall()
        return [ScoredRule(row["unit_id"], row["name"], -row["rank"]) for row in rows]

    def vector_search_units(self, embedding: list[float], k: int) -> list[ScoredRule]:
        if not self.vector_enabled:
            return []
        with self._lock:
            has_vecs = self.db.execute("SELECT count(*) c FROM units_vec").fetchone()["c"]
            if not has_vecs:
                return []
            rows = self.db.execute(
                "SELECT u.unit_id, u.name, v.distance"
                " FROM units_vec v JOIN units u ON u.rowid = v.unit_rowid"
                " WHERE v.embedding MATCH ? AND k = ?"
                " ORDER BY v.distance",
                (_serialize_f32(embedding), k),
            ).fetchall()
        return [ScoredRule(row["unit_id"], row["name"], -row["distance"]) for row in rows]

    def keyword_search_weapons(
        self, query: str, k: int, extra_terms: list[str] | None = None
    ) -> list[ScoredRule]:
        fts = _fts_query(query, extra_terms)
        with self._lock:
            rows = self.db.execute(
                "SELECT w.weapon_id, w.name, bm25(weapons_fts) AS rank"
                " FROM weapons_fts JOIN weapon_profiles w ON w.rowid = weapons_fts.rowid"
                " WHERE weapons_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts, k),
            ).fetchall()
        return [ScoredRule(row["weapon_id"], row["name"], -row["rank"]) for row in rows]

    def vector_search_weapons(self, embedding: list[float], k: int) -> list[ScoredRule]:
        # Weapons are short structured rows — keyword search carries them;
        # only units get embeddings. hybrid_search handles the empty list.
        return []

    def get_unit(self, unit_id: str) -> UnitProfile | None:
        with self._lock:
            row = self.db.execute("SELECT * FROM units WHERE unit_id = ?", (unit_id,)).fetchone()
            if row is None:
                return None
            weapon_rows = self.db.execute(
                "SELECT * FROM weapon_profiles WHERE unit_id = ? ORDER BY rowid", (unit_id,)
            ).fetchall()
        return UnitProfile(
            unit_id=row["unit_id"],
            faction=row["faction"],
            name=row["name"],
            movement=row["movement"] or "",
            toughness=row["toughness"] or "",
            save=row["save"] or "",
            wounds=row["wounds"] or "",
            leadership=row["leadership"] or "",
            oc=row["oc"] or "",
            keywords=json.loads(row["keywords"] or "[]"),
            abilities_text=row["abilities_text"] or "",
            points=row["points"] or "",
            weapons=[self._weapon_from_row(w) for w in weapon_rows],
            page_start=row["page_start"] or 0,
            page_end=row["page_end"] or 0,
            crop_paths=json.loads(row["crop_paths"] or "[]"),
            doc_hash=row["doc_hash"],
            parse_confidence=row["parse_confidence"],
            raw_text=row["raw_text"],
        )

    def get_weapon(self, weapon_id: str) -> tuple[WeaponProfile, str] | None:
        """Returns (weapon, unit_id) so callers can show the owning unit."""
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM weapon_profiles WHERE weapon_id = ?", (weapon_id,)
            ).fetchone()
        if row is None:
            return None
        return self._weapon_from_row(row), row["unit_id"]

    @staticmethod
    def _weapon_from_row(row: sqlite3.Row) -> WeaponProfile:
        return WeaponProfile(
            name=row["name"],
            range=row["range"] or "",
            attacks=row["attacks"] or "",
            skill=row["skill"] or "",
            strength=row["strength"] or "",
            ap=row["ap"] or "",
            damage=row["damage"] or "",
            keywords=json.loads(row["keywords"] or "[]"),
            weapon_id=row["weapon_id"],
        )

    def factions(self) -> list[str]:
        with self._lock:
            rows = self.db.execute("SELECT DISTINCT faction FROM units ORDER BY faction").fetchall()
        return [r["faction"] for r in rows]

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))
            self.db.commit()

    def load_vocab(self) -> dict[str, int]:
        with self._lock:
            rows = self.db.execute("SELECT term, freq FROM vocab").fetchall()
        return {r["term"]: r["freq"] for r in rows}

    def unit_count(self) -> int:
        with self._lock:
            return self.db.execute("SELECT count(*) c FROM units").fetchone()["c"]

    def weapon_count(self) -> int:
        with self._lock:
            return self.db.execute("SELECT count(*) c FROM weapon_profiles").fetchone()["c"]
