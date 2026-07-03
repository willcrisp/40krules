# Rulehound — 40k Rules Retrieval Prototype

Given a natural-language query like *"what are the rules for disembarking"*,
return the **verbatim rule text** and **cropped page image** for the matching
rule(s) in under 300 ms, with cross-referenced rules surfaced alongside.
Retrieval-first: no LLM anywhere in the hot path. See
`rulehound design doc` (Draft v1) for the full design this implements.

## Quickstart

```bash
uv sync                        # core deps (keyword + hash-embedder modes)
uv sync --extra embeddings     # + sentence-transformers for the real model

# Put your Core Rules PDF in place (never committed — data/ is gitignored):
cp ~/Downloads/core_rules.pdf data/raw/

# Ingest: extract -> chunk -> crosslink -> render/crop -> embed
uv run python -m rulehound.ingest data/raw/core_rules.pdf

# Serve the API + minimal UI
uv run python -m rulehound.api --port 8000
# open http://127.0.0.1:8000
```

The UI is deliberately minimal: a query box (results as you type), and an
upload control to build/rebuild the index and embeddings from a PDF —
uploading via the UI is equivalent to running the ingest CLI.

## Endpoints

- `GET /search?q=...&k=5` — hybrid BM25 + vector retrieval, RRF-fused,
  verbatim text + crop image paths + 1-hop related rules on the top hit,
  with per-stage `latency_ms`.
- `GET /rule/{rule_id}` — full rule incl. neighbours both directions.
- `GET /health` — status, rule count, embedding model/meta.
- `POST /ingest` (multipart PDF) / `POST /ingest/rebuild` / `GET /ingest/status`.

## Embeddings

Configured in `config.toml` (`[embedding]`). Default is
`BAAI/bge-small-en-v1.5` (384-dim, local CPU via sentence-transformers).
`model = "hash"` selects a deterministic no-ML lexical embedder (used by
tests). If the configured model can't be loaded, or the DB was embedded with
a different model than configured, vector search is disabled and keyword
search keeps working — vector is additive, never load-bearing.

## Typo tolerance

Two layers keep search-as-you-type robust, both inside the latency budget:
the final query token is FTS5 prefix-matched (`"disemb"*` finds Disembark
mid-word), and out-of-vocabulary tokens are spell-corrected against a
vocabulary built from the ingested rules text at ingest time (SymSpell-style
deletion index, sub-millisecond). Corrections are additive on the keyword
side and never rewrite words that exist in the corpus, so "embark" can never
be "corrected" into "disembark". When a correction fires, `/search` returns
`corrected_query` and the UI shows it.

## Tests & eval

```bash
uv run pytest                        # runs entirely against a synthetic fixture PDF
uv run python eval/chunk_report.py   # Phase 1 acceptance: 15-rule spot check
uv run python eval/run_eval.py       # Recall@1/3 + MRR, hybrid vs ablations
```

The golden set (`eval/golden_queries.yaml`) matches expectations against the
final segment of the rule_id; adjust slugs after the first chunk report
against the real PDF. Gate: hybrid Recall@3 ≥ 0.90.

CI never touches GW content: tests build their own hand-made fixture PDF.
Never commit the PDF, page renders, crops, or the DB (`data/` is gitignored).

## Layout

Matches the design doc §3: `src/rulehound/{ingest,store,search,api}`,
SQLite (FTS5 + sqlite-vec) behind a `SearchStore` protocol with a
turbopuffer adapter as the future production swap (stub raises
`NotImplementedError`).
