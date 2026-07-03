"""SQLite SearchStore: FTS5 (BM25) + sqlite-vec (design doc §2, §5, §6)."""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import threading
from importlib import resources
from pathlib import Path

from ..models import RuleChunk, RuleRef, ScoredRule

META_MODEL = "embedding_model"
META_DIM = "embedding_dimension"
META_DOC_HASH = "doc_hash"


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _fts_query(query: str) -> str:
    """Sanitise a natural-language query for FTS5: OR of quoted tokens."""
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


class SqliteStore:
    """One file: metadata + FTS5 + vectors. Thread-safe via per-call lock."""

    def __init__(self, db_path: str | Path, vector_dim: int = 384) -> None:
        self.db_path = Path(db_path)
        self.vector_dim = vector_dim
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.vector_enabled = self._load_vec_extension()
        self._apply_schema()

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
        schema = resources.files("rulehound.store").joinpath("schema.sql").read_text()
        with self._lock:
            self.db.executescript(schema)
            if self.vector_enabled:
                self.db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS rules_vec USING vec0("
                    "  rule_rowid INTEGER PRIMARY KEY,"
                    f"  embedding FLOAT[{self.vector_dim}]"
                    ")"
                )
            self.db.commit()

    def close(self) -> None:
        self.db.close()

    # --- ingest side -------------------------------------------------------

    def replace_document(
        self, chunks: list[RuleChunk], refs: list[RuleRef], doc_hash: str
    ) -> None:
        with self._lock:
            cur = self.db.cursor()
            cur.execute("INSERT INTO rules_fts(rules_fts) VALUES('delete-all')")
            cur.execute("DELETE FROM rule_refs")
            cur.execute("DELETE FROM rules")
            if self.vector_enabled:
                cur.execute("DELETE FROM rules_vec")
            cur.executemany(
                "INSERT INTO rules (rule_id, title, section_path, text, commentary,"
                " page_start, page_end, crop_paths, doc_hash)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        c.rule_id, c.title, c.section_path, c.text, c.commentary,
                        c.page_start, c.page_end, json.dumps(c.crop_paths), doc_hash,
                    )
                    for c in chunks
                ],
            )
            cur.execute(
                "INSERT INTO rules_fts(rowid, title, text)"
                " SELECT rowid, title, text FROM rules"
            )
            known = {c.rule_id for c in chunks} | {
                c.rule_id.split("--part-")[0] for c in chunks
            }
            cur.executemany(
                "INSERT OR IGNORE INTO rule_refs (from_rule, to_rule, kind) VALUES (?,?,?)",
                [
                    (r.from_rule, r.to_rule, r.kind)
                    for r in refs
                    if r.from_rule in known and r.to_rule in known
                ],
            )
            cur.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                (META_DOC_HASH, doc_hash),
            )
            self.db.commit()

    def store_vectors(
        self, vectors: dict[str, list[float]], model_name: str, dimension: int
    ) -> None:
        if not self.vector_enabled:
            raise RuntimeError("sqlite-vec extension not available")
        if dimension != self.vector_dim:
            raise ValueError(
                f"vector dim {dimension} != store dim {self.vector_dim}"
            )
        with self._lock:
            cur = self.db.cursor()
            rowids = {
                row["rule_id"]: row["rowid"]
                for row in cur.execute("SELECT rowid, rule_id FROM rules")
            }
            cur.execute("DELETE FROM rules_vec")
            cur.executemany(
                "INSERT INTO rules_vec (rule_rowid, embedding) VALUES (?,?)",
                [
                    (rowids[rid], _serialize_f32(vec))
                    for rid, vec in vectors.items()
                    if rid in rowids
                ],
            )
            cur.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (META_MODEL, model_name)
            )
            cur.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (META_DIM, str(dimension))
            )
            self.db.commit()

    # --- query side --------------------------------------------------------

    def keyword_search(self, query: str, k: int) -> list[ScoredRule]:
        fts = _fts_query(query)
        with self._lock:
            rows = self.db.execute(
                "SELECT r.rule_id, r.title, bm25(rules_fts) AS rank"
                " FROM rules_fts JOIN rules r ON r.rowid = rules_fts.rowid"
                " WHERE rules_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts, k),
            ).fetchall()
        # bm25() is lower-is-better; negate so higher = better everywhere.
        return [ScoredRule(row["rule_id"], row["title"], -row["rank"]) for row in rows]

    def vector_search(self, embedding: list[float], k: int) -> list[ScoredRule]:
        if not self.vector_enabled:
            return []
        with self._lock:
            has_vecs = self.db.execute("SELECT count(*) c FROM rules_vec").fetchone()["c"]
            if not has_vecs:
                return []
            rows = self.db.execute(
                "SELECT r.rule_id, r.title, v.distance"
                " FROM rules_vec v JOIN rules r ON r.rowid = v.rule_rowid"
                " WHERE v.embedding MATCH ? AND k = ?"
                " ORDER BY v.distance",
                (_serialize_f32(embedding), k),
            ).fetchall()
        return [ScoredRule(row["rule_id"], row["title"], -row["distance"]) for row in rows]

    def get_rule(self, rule_id: str) -> RuleChunk | None:
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM rules WHERE rule_id = ?", (rule_id,)
            ).fetchone()
        if row is None:
            return None
        return RuleChunk(
            rule_id=row["rule_id"],
            title=row["title"],
            section_path=row["section_path"],
            text=row["text"],
            commentary=row["commentary"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            crop_paths=json.loads(row["crop_paths"] or "[]"),
            doc_hash=row["doc_hash"],
        )

    def neighbours(self, rule_id: str, direction: str = "out") -> list[RuleRef]:
        clauses = {"out": "from_rule = ?", "in": "to_rule = ?", "both": "from_rule = ? OR to_rule = ?"}
        params = (rule_id,) if direction != "both" else (rule_id, rule_id)
        with self._lock:
            rows = self.db.execute(
                f"SELECT from_rule, to_rule, kind FROM rule_refs WHERE {clauses[direction]}",
                params,
            ).fetchall()
        return [RuleRef(r["from_rule"], r["to_rule"], r["kind"]) for r in rows]

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))
            self.db.commit()

    def rule_count(self) -> int:
        with self._lock:
            return self.db.execute("SELECT count(*) c FROM rules").fetchone()["c"]
