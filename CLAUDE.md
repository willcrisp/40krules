# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Rulehound: a local-first retrieval prototype for the Warhammer 40k Core Rules.
Given a natural-language query, it returns **verbatim rule text + a cropped
page image**, hybrid-retrieved (BM25 + vector) in under 300ms — no LLM
anywhere in the hot path. The full design rationale (storage decision,
schema, phased execution plan, deliberate deferrals) lives in
[`DESIGN.md`](DESIGN.md) — read it before making architectural changes;
this file only covers what's needed to work in the code day to day.

## Commands

```bash
uv sync                        # core deps (keyword + hash-embedder modes)
uv sync --extra embeddings     # + sentence-transformers for the real bge model

uv run pytest                        # full suite — entirely against a synthetic fixture PDF
uv run pytest tests/test_hybrid.py    # one file
uv run pytest tests/test_hybrid.py::test_direct_title_queries  # one test

uv run python -m rulehound.ingest data/raw/core_rules.pdf   # ingest CLI (idempotent)
uv run python -m rulehound.api --port 8000                  # serve API + UI

uv run python eval/chunk_report.py    # Phase 1 acceptance: 15-rule spot check
uv run python eval/run_eval.py        # Recall@1/3 + MRR, hybrid vs keyword-only vs vector-only
```

There is no lint/format command configured yet.

**Never commit `data/`** (raw PDF, page renders, crops, the SQLite DB) — it's
Games Workshop IP and is gitignored/dockerignored. Every test builds its own
hand-made synthetic fixture PDF (`tests/fixtures/build_fixture.py`); CI must
never touch real GW content.

## Architecture

### Ingest pipeline (`src/rulehound/ingest/`)

`pipeline.py` orchestrates six phases in order — extract → chunk → crosslinks
→ images → store → embed — keyed on a **content hash of the input PDF**.
Re-running with the same file is a no-op per phase (tracked via
`meta` table rows `ingest_phase:{doc_hash}:{phase}`); `--force` re-runs
everything. Read `pipeline.py` first when touching any ingest phase — it's
the map of how the phases feed each other (blocks → chunks → refs/images
computed from chunks → DB rows → vectors).

- `pdf_extract.py` — PyMuPDF text+layout extraction, including two-column
  reading-order detection (clusters blocks by x-midpoint into bands when a
  page looks two-column).
- `chunker.py` — one chunk = one heading + body until the next
  same-or-higher heading; tracks the full heading hierarchy as
  `section_path`, boxed/filled-rect text becomes `commentary` (not embedded
  into the main chunk text), oversized chunks split on paragraph boundaries
  into `--part-N` chunks sharing a rule_id prefix.
- `crosslinks.py` — regex + longest-title-first matching to avoid "Move"
  swallowing "Normal Move"; emits `explicit_see` / `title_mention` /
  `page_ref` edges.
- `embed.py` — `Embedder` protocol with three implementations:
  `SentenceTransformerEmbedder` (real model), `HashingEmbedder`
  (deterministic bag-of-ngrams, no ML deps — used by every test and as a
  no-network fallback), and `get_embedder()` which returns `None` on any
  failure so the system degrades to keyword-only rather than erroring.

### Storage (`src/rulehound/store/`)

Everything above the `SearchStore` Protocol (`base.py`) is store-agnostic.
`sqlite_store.py` is the only real implementation (SQLite FTS5 for BM25 +
the `sqlite-vec` extension for vectors, one file). `turbopuffer_store.py` is
a deliberate stub (`raise NotImplementedError`) — the intended production
swap when this stops being single-user/local; don't implement it unless
asked. The DB also stores an embedding-model fingerprint in `meta`; if the
configured model doesn't match what the DB was embedded with, vector search
is refused and keyword search keeps working (`app.py::AppState.refresh_embedder`).

### Search (`src/rulehound/search/`)

`hybrid.py::hybrid_search` runs FTS5 keyword search and vector search in
parallel (thread pool), fuses with **Reciprocal Rank Fusion**, then adds a
title exact/prefix boost. The boost constants in `config.toml`
(`title_exact_boost`/`title_prefix_boost`) are deliberately sized **larger**
than the max attainable RRF score so a title match always wins — most
queries name the rule directly. `expand.py` attaches 1-hop cross-reference
neighbours (titles + ids only, no text) to the top result only.

`spell.py` is a corpus-driven typo layer, not a dictionary spell checker —
the vocabulary is built from the ingested rules text at ingest time
(`vocab` table), so real rule words are never "corrected" into each other
(e.g. "embark" can never become "disembark"). It's wired into
`hybrid_search` as: corrected tokens are OR'd in *additively* on the keyword
side (originals are kept so stem matches still work) but *substitute* the
query for embedding and the title boost. Separately, the final token of any
query is FTS5 prefix-matched so search-as-you-type finds results mid-word.

### API (`src/rulehound/api/`)

`app.py::AppState` holds the long-lived store/embedder/corrector, loaded
once at startup and refreshed after ingest (`IngestManager` runs ingest in a
background thread so `POST /ingest` returns immediately; poll
`/ingest/status`). `routes.py` is intentionally thin — verbatim text +
image paths only, never any generation. The UI (`static/index.html`) is a
single page: search-as-you-type plus an upload control that hits the same
`/ingest` endpoint the CLI uses.

### Config (`config.py`)

Load order: explicit path → `$RULEHOUND_CONFIG` → `./config.toml` →
dataclass defaults. `$RULEHOUND_DATA_DIR`, if set, overrides the entire
`[paths]` section at once — this is how the Railway deploy points all
derived data (DB, pages, crops, uploaded PDF) at a mounted volume instead of
config.toml paths. `__main__.py` reads `$PORT` (the convention Railway/most
PaaS inject) for the bind port.

### Deployment

`Dockerfile` + `railway.json` build/host the app as a single container.
**`pyproject.toml` pins `torch` to the CPU-only wheel index**
(`[tool.uv.sources]` → `download.pytorch.org/whl/cpu`) — without this, the
default PyPI linux wheel for `torch` pulls the full CUDA toolchain (~5GB of
`nvidia-*` packages) that a CPU-only container never uses. If you touch the
`embeddings` extra or the CPU pin, keep them in sync (torch must stay a
direct dependency there for the `tool.uv.sources` override to apply to it —
it doesn't reliably apply to torch as a bare transitive dep of
sentence-transformers).

### Eval harness (`eval/`)

`golden_queries.yaml` (direct-naming, paraphrase, symptom, adversarial
near-miss, and typo queries) maps queries to expected rule_id, matched
against the **final segment** of the returned rule_id so it survives
re-chunking. Gate: hybrid Recall@3 ≥ 0.90 (`run_eval.py`, which also reports
keyword-only/vector-only ablations). `chunk_report.py` is the Phase 1
acceptance check — a 15-rule spot-check against whatever's currently
ingested.
