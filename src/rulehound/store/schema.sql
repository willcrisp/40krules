-- Rulehound schema (design doc §5). The vec0 table is created separately in
-- sqlite_store.py because its dimension comes from config and the extension
-- may be unavailable (keyword-only mode).

CREATE TABLE IF NOT EXISTS rules (
  rule_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  section_path TEXT NOT NULL,
  text TEXT NOT NULL,
  commentary TEXT,
  page_start INTEGER, page_end INTEGER,
  crop_paths TEXT,           -- JSON array
  doc_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rule_refs (
  from_rule TEXT REFERENCES rules(rule_id),
  to_rule   TEXT REFERENCES rules(rule_id),
  kind TEXT CHECK (kind IN ('explicit_see','title_mention','page_ref')),
  PRIMARY KEY (from_rule, to_rule, kind)
);

CREATE VIRTUAL TABLE IF NOT EXISTS rules_fts USING fts5(
  title, text, content='rules', content_rowid='rowid',
  tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
