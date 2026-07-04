-- Datasheet corpus schema: a separate DB file from the rules corpus so a
-- datasheet ingest can never touch the rules index. The units_vec vec0 table
-- is created in datasheet_store.py (dimension comes from config and the
-- extension may be unavailable), mirroring schema.sql / sqlite_store.py.

CREATE TABLE IF NOT EXISTS units (
  unit_id TEXT PRIMARY KEY,
  faction TEXT NOT NULL,
  name TEXT NOT NULL,
  movement TEXT, toughness TEXT, save TEXT, wounds TEXT,
  leadership TEXT, oc TEXT,
  keywords TEXT,              -- JSON array
  abilities_text TEXT,
  points TEXT,
  raw_text TEXT NOT NULL,
  parse_confidence TEXT NOT NULL DEFAULT 'ok',
  page_start INTEGER, page_end INTEGER,
  crop_paths TEXT,            -- JSON array
  doc_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_units_faction ON units(faction);

CREATE TABLE IF NOT EXISTS weapon_profiles (
  weapon_id TEXT PRIMARY KEY, -- "{unit_id}--{weapon-slug}", stable across re-ingest
  unit_id TEXT NOT NULL REFERENCES units(unit_id),
  name TEXT NOT NULL,
  "range" TEXT, attacks TEXT, skill TEXT, strength TEXT, ap TEXT, damage TEXT,
  keywords TEXT               -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_weapons_unit ON weapon_profiles(unit_id);

CREATE VIRTUAL TABLE IF NOT EXISTS units_fts USING fts5(
  name, keywords, abilities_text, raw_text,
  content='units', content_rowid='rowid',
  tokenize='porter unicode61'
);

-- Weapon names/keywords are searchable in their own right ("bolt rifle",
-- "lethal hits") even though numeric stat filtering is out of scope.
CREATE VIRTUAL TABLE IF NOT EXISTS weapons_fts USING fts5(
  name, keywords,
  content='weapon_profiles', content_rowid='rowid',
  tokenize='porter unicode61'
);

-- Datasheet-corpus vocabulary for query spell correction, kept separate from
-- the rules corpus vocab so unit-name corrections never mix with rules prose.
CREATE TABLE IF NOT EXISTS vocab (term TEXT PRIMARY KEY, freq INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
