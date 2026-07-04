"""API routes: /search, /rule/{id}, /health, plus minimal UI + ingest (design doc §7).

Responses are verbatim text + images only — no generation anywhere.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from .. import __version__
from ..models import RuleChunk
from ..search.expand import related_for
from ..search.hybrid import hybrid_search
from ..store import sqlite_store

router = APIRouter()

_INDEX_HTML = Path(__file__).parent / "static" / "index.html"


def _state(request: Request):
    return request.app.state.rulehound


def _rule_payload(rule: RuleChunk, related: list[dict] | None = None, score: float | None = None) -> dict:
    payload = {
        "rule_id": rule.rule_id,
        "title": rule.title,
        "section_path": rule.section_path,
        "text": rule.text,
        "commentary": rule.commentary,
        "pages": list(range(rule.page_start, rule.page_end + 1)) if rule.page_start else [],
        "crops": [f"/crops/{p}" for p in rule.crop_paths],
    }
    if score is not None:
        payload["score"] = round(score, 4)
    if related is not None:
        payload["related"] = related
    return payload


@router.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML.read_text(encoding="utf-8")


@router.get("/search")
def search(request: Request, q: str, k: int = 0) -> dict:
    state = _state(request)
    cfg = state.cfg.search
    t0 = time.perf_counter()
    fused, timings, correction = hybrid_search(
        state.store, state.embedder, q, cfg, corrector=state.corrector
    )
    top_n = fused[: k or cfg.top_k]

    results = []
    for i, item in enumerate(top_n):
        rule = state.store.get_rule(item.rule_id)
        if rule is None:
            continue
        related = related_for(state.store, item.rule_id, cfg.related_max) if i == 0 else None
        results.append(_rule_payload(rule, related=related, score=item.score))

    timings.total_ms = (time.perf_counter() - t0) * 1000
    payload = {"query": q, "latency_ms": timings.as_dict(), "results": results}
    if correction and correction.changed:
        payload["corrected_query"] = correction.corrected
    return payload


@router.get("/rule/{rule_id}")
def get_rule(request: Request, rule_id: str) -> dict:
    state = _state(request)
    rule = state.store.get_rule(rule_id) or state.store.get_rule(f"{rule_id}--part-1")
    if rule is None:
        raise HTTPException(status_code=404, detail=f"unknown rule_id: {rule_id}")

    base_id = rule.rule_id.split("--part-")[0]
    refs = state.store.neighbours(base_id, direction="both")
    outgoing, incoming = [], []
    for ref in refs:
        other_id = ref.to_rule if ref.from_rule == base_id else ref.from_rule
        other = state.store.get_rule(other_id) or state.store.get_rule(f"{other_id}--part-1")
        entry = {"rule_id": other_id, "title": other.title if other else other_id, "kind": ref.kind}
        (outgoing if ref.from_rule == base_id else incoming).append(entry)

    payload = _rule_payload(rule)
    payload["neighbours"] = {"outgoing": outgoing, "incoming": incoming}
    return payload


@router.get("/health")
def health(request: Request) -> dict:
    state = _state(request)
    store = state.store
    return {
        "status": "ok",
        "version": __version__,
        "rules": store.rule_count(),
        "vector_enabled": store.vector_enabled,
        "embedder": getattr(state.embedder, "name", None),
        "embedder_error": state.embedder_error,
        "db_meta": {
            "doc_hash": store.get_meta(sqlite_store.META_DOC_HASH),
            "embedding_model": store.get_meta(sqlite_store.META_MODEL),
            "embedding_dimension": store.get_meta(sqlite_store.META_DIM),
        },
    }


@router.post("/ingest")
async def ingest(request: Request, file: UploadFile) -> dict:
    """Upload a rules PDF and (re)build chunks + embeddings from it."""
    state = _state(request)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="upload a .pdf file")
    dest = state.cfg.paths.raw_dir / Path(file.filename).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    if not state.ingest.start(dest):
        raise HTTPException(status_code=409, detail="an ingest is already running")
    return {"status": "started", "file": dest.name}


@router.post("/ingest/rebuild")
def rebuild(request: Request) -> dict:
    """Re-run ingest (including embeddings) on the most recent uploaded PDF."""
    state = _state(request)
    pdfs = sorted(state.cfg.paths.raw_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime)
    if not pdfs:
        raise HTTPException(status_code=404, detail="no PDF in data/raw — upload one first")
    if not state.ingest.start(pdfs[-1], force=True):
        raise HTTPException(status_code=409, detail="an ingest is already running")
    return {"status": "started", "file": pdfs[-1].name}


@router.get("/ingest/status")
def ingest_status(request: Request) -> dict:
    return _state(request).ingest.status
