# Design Document: 40k Rules Retrieval Prototype ("Rulehound")

**Status:** Draft v1 — for agent execution
**Scope:** Single-user local prototype. Core Rules document only. No faction data, no points, no LLM synthesis.
**Non-goals (this prototype):** Multi-document precedence/errata layering, datasheet/points lookup, deployment, auth, mobile UI, LLM answer generation.

---

## 1. Objective

Given a natural-language query like *"what are the rules for disembarking"*, return the **verbatim rule text** and the **cropped page image** for the matching rule(s) in **< 300ms end-to-end** on local hardware, with cross-referenced rules surfaced alongside.

The prototype exists to validate three things:

1. Rule-block chunking of the Core Rules PDF produces clean, individually retrievable units.
2. Hybrid retrieval (BM25 + dense vectors) reliably puts the correct rule at rank 1 for common query phrasings.
3. Retrieval-first (no LLM in the hot path) feels faster than opening the rulebook.

---

## 2. Storage decision: local-first, turbopuffer-compatible

**Decision: SQLite (FTS5 + sqlite-vec) for the prototype, behind a thin `SearchStore` interface, with turbopuffer as a drop-in production swap.**

Rationale:

- The corpus is tiny by vector-DB standards. The Core Rules yield roughly 300–600 rule chunks; the *entire* game corpus (all codexes, dataslates, FAQs) is plausibly < 20k chunks. This is well inside embedded-database territory.
- Local SQLite removes the network round trip entirely. Turbopuffer's query latency is good, but any hosted store adds 20–100ms of RTT that dominates the latency budget at this corpus size. For a prototype whose primary success metric is "faster than the book," local wins.
- SQLite is already the system-of-record for chunk metadata and cross-references (Section 5), so co-locating FTS5 and vectors means one file, zero infra, zero cost.
- Turbopuffer *can* honestly handle this — it natively supports hybrid BM25 + vector queries and the corpus is trivially small for it. It becomes the right choice when the app is multi-user/hosted. The `SearchStore` interface (Section 6) is designed so that swap is an adapter, not a rewrite.

**Embeddings:** local model via `sentence-transformers`, default `BAAI/bge-small-en-v1.5` (384-dim). Free, ~30ms/query on CPU, no API key. Interface must allow swapping to Voyage later (config-driven model name + dimension).

---

## 3. Repository layout

```
rulehound/
  pyproject.toml            # uv-managed; python >= 3.12
  config.toml               # paths, model name, top_k, fusion weights
  data/
    raw/                    # input PDF (gitignored)
    pages/                  # rendered page PNGs (gitignored)
    crops/                  # per-chunk cropped images (gitignored)
    rulehound.db            # SQLite: metadata + FTS5 + vectors (gitignored)
  src/rulehound/
    ingest/
      pdf_extract.py        # text + layout extraction
      chunker.py             # rule-block chunking
      crosslinks.py          # cross-reference edge extraction
      images.py             # page render + crop region computation
      embed.py               # embedding generation
      pipeline.py            # orchestrates full ingest, idempotent
    store/
      schema.sql
      base.py               # SearchStore protocol
      sqlite_store.py       # FTS5 + sqlite-vec implementation
      turbopuffer_store.py  # stub only in prototype; raises NotImplementedError
    search/
      hybrid.py             # BM25 + vector query, RRF fusion
      expand.py             # 1-hop cross-reference expansion
    api/
      app.py                # FastAPI app
      routes.py             # /search, /rule/{id}, /health
  tests/
    fixtures/               # 3–5 page PDF excerpt for CI (hand-made, not GW content)
    test_chunker.py
    test_hybrid.py
    test_latency.py
  eval/
    golden_queries.yaml     # query -> expected rule_id pairs
    run_eval.py
```

Stack constraints: Python 3.12+, FastAPI, `uv` for deps, `pymupdf` (fitz) for PDF work, `sqlite-vec` extension, `sentence-transformers`. No Docker required for the prototype.

---

## 4. Ingestion pipeline

Ingest is a CLI: `python -m rulehound.ingest data/raw/core_rules.pdf`. It must be **idempotent** (re-running replaces derived data keyed on a content hash of the PDF) and **resumable per phase**.

### 4.1 Text + layout extraction (`pdf_extract.py`)

- Use PyMuPDF `get_text("dict")` to extract text spans with bounding boxes, font size, and font weight per page.
- Emit an intermediate JSONL: one record per text block with `{page, bbox, text, font_size, is_bold}`.
- Detect headings heuristically: font size above body-text modal size, or bold spans that start a block. Emit heading candidates with confidence.

### 4.2 Rule-block chunking (`chunker.py`)

- A **chunk = one named rule**: heading + all body text until the next same-or-higher-level heading.
- Track heading hierarchy (e.g. *Movement Phase → Transports → Disembark*) and store the full path as `section_path`.
- Attach designer's-commentary / example sidebars (visually boxed text) to the rule chunk they sit under, in a separate `commentary` field — retrievable but not embedded into the main chunk text.
- Hard limits: if a chunk exceeds 1,200 tokens, split on paragraph boundaries into `part 1/2` chunks sharing the same `rule_id` prefix. Do not split below heading granularity otherwise.
- Output fields per chunk: `rule_id` (slug of section_path), `title`, `section_path`, `text`, `commentary`, `page_start`, `page_end`, `bboxes` (list of per-page bounding boxes covering the rule's text).

**Acceptance for 4.2:** manual spot-check checklist of 15 known rules (Disembark, Deep Strike, Engagement Range, Fall Back, Objective Control, etc.) — each must exist as its own chunk with correct text boundaries.

### 4.3 Cross-reference extraction (`crosslinks.py`)

- Regex + title-matching pass over chunk text for phrases matching other chunk titles ("Reserves", "Normal move", "Engagement Range") and explicit "see X" / "(pg N)" references.
- Emit edges `(from_rule_id, to_rule_id, kind)` where kind ∈ {`explicit_see`, `title_mention`, `page_ref`}.
- Match longest title first to avoid "Move" swallowing "Normal Move". Case-insensitive, word-boundary anchored.

### 4.4 Images (`images.py`)

- Render every page to PNG at 150 DPI into `data/pages/`.
- For each chunk, compute a crop per page from the union of its bboxes plus 12px padding; save to `data/crops/{rule_id}_{page}.png`.
- Store crop file paths on the chunk record. Serve as static files.

### 4.5 Embedding (`embed.py`)

- Embed `title + "\n" + text` (not commentary). Batch, normalise, write vectors into `sqlite-vec` table.
- Store model name + dimension in a `meta` table; refuse to query if the query-time model config mismatches.

---

## 5. Schema (`schema.sql`)

```sql
CREATE TABLE rules (
  rule_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  section_path TEXT NOT NULL,
  text TEXT NOT NULL,
  commentary TEXT,
  page_start INTEGER, page_end INTEGER,
  crop_paths TEXT,           -- JSON array
  doc_hash TEXT NOT NULL
);

CREATE TABLE rule_refs (
  from_rule TEXT REFERENCES rules(rule_id),
  to_rule   TEXT REFERENCES rules(rule_id),
  kind TEXT CHECK (kind IN ('explicit_see','title_mention','page_ref')),
  PRIMARY KEY (from_rule, to_rule, kind)
);

CREATE VIRTUAL TABLE rules_fts USING fts5(
  title, text, content='rules', content_rowid='rowid',
  tokenize='porter unicode61'
);

-- sqlite-vec virtual table, dimension from config
CREATE VIRTUAL TABLE rules_vec USING vec0(
  rule_rowid INTEGER PRIMARY KEY,
  embedding FLOAT[384]
);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
```

---

## 6. Search (`SearchStore` protocol + hybrid)

```python
class SearchStore(Protocol):
    def keyword_search(self, query: str, k: int) -> list[ScoredRule]: ...
    def vector_search(self, embedding: list[float], k: int) -> list[ScoredRule]: ...
    def get_rule(self, rule_id: str) -> Rule | None: ...
    def neighbours(self, rule_id: str) -> list[RuleRef]: ...
```

Hybrid flow (`hybrid.py`):

1. Run FTS5 (BM25) and vector search in parallel, `k=20` each.
2. Fuse with Reciprocal Rank Fusion, `k_rrf=60`. Title exact/prefix match adds a fixed boost (title hits should dominate — most queries name the rule).
3. Take top 5 fused results.
4. `expand.py`: for the top result only, attach its 1-hop `rule_refs` neighbours (`explicit_see` and `title_mention` kinds), max 4, as `related` — titles + rule_ids only, not full text.

The turbopuffer adapter, when written later, implements the same protocol with a single hybrid query (it supports BM25 + vector natively); RRF then happens server-side or is replaced by turbopuffer's fusion. Nothing above the protocol changes.

---

## 7. API

FastAPI, two endpoints plus static file serving for crops:

```
GET /search?q=...&k=5
  -> {
       "query": "...",
       "latency_ms": {"embed": .., "keyword": .., "vector": .., "total": ..},
       "results": [
         {
           "rule_id": "movement-phase.transports.disembark",
           "title": "Disembark",
           "section_path": "Movement Phase > Transports > Disembark",
           "text": "...verbatim...",
           "commentary": "...",
           "pages": [18],
           "crops": ["/crops/....png"],
           "score": 0.87,
           "related": [{"rule_id": "...", "title": "Reserves"}]
         }
       ]
     }

GET /rule/{rule_id}   -> full rule record incl. neighbours both directions
GET /health           -> ok + model/meta info
```

The response is **verbatim text + image only**. No generation anywhere in the prototype.

---

## 8. Performance budget

Measured on the dev box, warm process, via `tests/test_latency.py` (p50/p95 over the golden query set):

| Stage | Budget (p95) |
|---|---|
| Query embedding (CPU, bge-small) | 60 ms |
| FTS5 query | 10 ms |
| Vector query | 20 ms |
| Fusion + expansion + serialise | 10 ms |
| **Total server-side** | **≤ 150 ms** |

Model is loaded once at startup. If embedding blows the budget on target hardware, fallback is `all-MiniLM-L6-v2` or ONNX-quantised bge-small; failing that, keyword-only mode remains functional (vector search is additive, never load-bearing for exact title queries).

---

## 9. Evaluation

`eval/golden_queries.yaml`: ≥ 40 query → expected `rule_id` pairs, covering:

- Direct naming: "disembark rules", "deep strike"
- Paraphrase: "getting out of a transport", "arriving from reserves"
- Symptom phrasing: "can I shoot after falling back", "who controls an objective"
- Adversarial near-misses: "embark" vs "disembark" must not swap ranks.

`run_eval.py` reports **Recall@1, Recall@3, MRR**. **Gate: Recall@3 ≥ 0.90** on the golden set before the prototype is considered done. Keyword-only and vector-only ablations must also be reported so hybrid's contribution is visible.

---

## 10. Execution phases (agent handoff)

Each phase ends with its stated acceptance check passing; do not proceed on failure.

- **Phase 0 — Scaffold.** Repo layout, `pyproject.toml`, config loading, empty schema applied, `/health` returns. *Accept: `pytest` green on scaffold tests, server boots.*
- **Phase 1 — Extraction + chunking.** 4.1–4.2 against the real PDF placed at `data/raw/`. *Accept: 15-rule spot-check checklist emitted as a report (`eval/chunk_report.md`) listing each rule, its detected boundaries, and pass/fail; all pass.*
- **Phase 2 — Crosslinks + images.** 4.3–4.4. *Accept: Disembark's neighbours include Transports and Reserves; crop images open and visually contain the rule text (report with embedded thumbnails).*
- **Phase 3 — Embeddings + hybrid search.** 4.5, Section 6, sqlite store complete. *Accept: golden-set eval runs; report numbers even if below gate.*
- **Phase 4 — API + latency.** Section 7 endpoints, latency instrumentation. *Accept: latency test p95 ≤ 150 ms server-side.*
- **Phase 5 — Tuning to gate.** Adjust fusion weights, title boost, chunking fixes surfaced by eval failures. *Accept: Recall@3 ≥ 0.90 and latency budget still met.*

Agent constraints: never commit PDF, page renders, crops, or the DB (gitignore enforced in Phase 0); all tests must run against the synthetic fixture PDF so CI never touches GW content; every phase produces a short markdown report in `eval/`.

---

## 11. Known limitations & deliberate deferrals

- **Copyright:** the ingested text and images are Games Workshop IP. This design is for a personal, local tool. Distribution of the app *with data* is out of scope and legally fraught; distribution of the *pipeline* (code only, bring-your-own-PDF) is the only shippable shape.
- **Single document:** no precedence/errata layering. The versioning model (errata > dataslate > codex > core) is the first post-prototype design task.
- **Two-column / layout edge cases:** PyMuPDF block ordering can interleave columns; the chunker report in Phase 1 exists specifically to catch this. If ordering is broken, add a column-detection pass (cluster blocks by x-midpoint) before chunking.
- **Diagrams:** crops capture whatever is inside the rule's bbox union; free-floating diagrams referenced by a rule but positioned elsewhere on the page will be missed. Deferred.
- **No answer synthesis:** multi-rule reasoning questions return multiple rules side by side, not a conclusion. LLM synthesis is a post-prototype layer and must cite rule_ids when added.
