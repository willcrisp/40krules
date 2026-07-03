"""Full ingest orchestration (design doc §4).

Idempotent: derived data is keyed on a content hash of the PDF; re-running
with the same file and completed phases is a no-op unless --force.
Resumable per phase: each completed phase is recorded in the meta table and
intermediates (blocks.jsonl, chunks.json) live in data_dir.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ..config import Config
from ..models import RuleChunk, RuleRef
from ..store.sqlite_store import SqliteStore
from . import chunker, crosslinks, embed, images, pdf_extract

PHASES = ["extract", "chunk", "crosslinks", "images", "store", "embed"]

Log = Callable[[str], None]


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _phase_key(doc_hash: str, phase: str) -> str:
    return f"ingest_phase:{doc_hash}:{phase}"


def _chunks_to_json(chunks: list[RuleChunk], path: Path) -> None:
    payload = []
    for c in chunks:
        d = asdict(c)
        d["bboxes"] = {str(k): v for k, v in c.bboxes.items()}
        payload.append(d)
    path.write_text(json.dumps(payload))


def _chunks_from_json(path: Path) -> list[RuleChunk]:
    out = []
    for d in json.loads(path.read_text()):
        d["bboxes"] = {int(k): [tuple(b) for b in v] for k, v in d["bboxes"].items()}
        out.append(RuleChunk(**d))
    return out


def run_ingest(
    pdf_path: str | Path,
    cfg: Config,
    store: SqliteStore | None = None,
    force: bool = False,
    log: Log = print,
) -> dict:
    """Run all ingest phases. Returns a summary dict."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    cfg.paths.ensure_dirs()

    doc_hash = file_hash(pdf_path)
    own_store = store is None
    if store is None:
        store = SqliteStore(cfg.paths.db_path, vector_dim=cfg.embedding.dimension)

    blocks_path = cfg.paths.data_dir / f"blocks_{doc_hash[:12]}.jsonl"
    chunks_path = cfg.paths.data_dir / f"chunks_{doc_hash[:12]}.json"

    def done(phase: str) -> bool:
        return not force and store.get_meta(_phase_key(doc_hash, phase)) == "done"

    def mark(phase: str) -> None:
        store.set_meta(_phase_key(doc_hash, phase), "done")
        log(f"[ingest] phase '{phase}' done")

    try:
        # 4.1 extract
        if done("extract") and blocks_path.is_file():
            blocks = pdf_extract.read_blocks_jsonl(blocks_path)
            log(f"[ingest] extract: reused {len(blocks)} blocks")
        else:
            blocks = pdf_extract.extract_blocks(pdf_path)
            pdf_extract.write_blocks_jsonl(blocks, blocks_path)
            mark("extract")
            log(f"[ingest] extract: {len(blocks)} blocks")

        # 4.2 chunk
        if done("chunk") and chunks_path.is_file():
            chunks = _chunks_from_json(chunks_path)
            log(f"[ingest] chunk: reused {len(chunks)} chunks")
        else:
            chunks = chunker.chunk_blocks(blocks, doc_hash, cfg.chunking.max_tokens)
            _chunks_to_json(chunks, chunks_path)
            mark("chunk")
            log(f"[ingest] chunk: {len(chunks)} chunks")

        # 4.3 crosslinks
        refs: list[RuleRef] = crosslinks.extract_crosslinks(chunks)
        if not done("crosslinks"):
            mark("crosslinks")
        log(f"[ingest] crosslinks: {len(refs)} edges")

        # 4.4 images
        if not done("images"):
            images.render_pages(pdf_path, cfg.paths.pages_dir, cfg.images.dpi)
            images.crop_chunks(
                pdf_path, chunks, cfg.paths.crops_dir,
                cfg.images.dpi, cfg.images.crop_padding_px,
            )
            _chunks_to_json(chunks, chunks_path)  # persist crop_paths
            mark("images")
        log("[ingest] images: pages rendered, crops written")

        # store rows (rerun whenever upstream reran)
        if not done("store"):
            store.replace_document(chunks, refs, doc_hash)
            mark("store")
        log(f"[ingest] store: {store.rule_count()} rules in DB")

        # 4.5 embed
        embedded = 0
        if not done("embed"):
            embedder = embed.get_embedder(cfg.embedding, log=log)
            if embedder is not None and store.vector_enabled:
                texts = [embed.embedding_text(c.title, c.text) for c in chunks]
                vectors: dict[str, list[float]] = {}
                batch = 64
                for i in range(0, len(chunks), batch):
                    encoded = embedder.encode(texts[i : i + batch])
                    for c, v in zip(chunks[i : i + batch], encoded):
                        vectors[c.rule_id] = v
                store.store_vectors(vectors, embedder.name, embedder.dimension)
                embedded = len(vectors)
                mark("embed")
                log(f"[ingest] embed: {embedded} vectors ({embedder.name})")
            else:
                log("[ingest] embed: skipped (no embedder / no sqlite-vec); keyword-only mode")

        return {
            "doc_hash": doc_hash,
            "blocks": len(blocks),
            "chunks": len(chunks),
            "refs": len(refs),
            "embedded": embedded,
            "vector_enabled": store.vector_enabled,
        }
    finally:
        if own_store:
            store.close()


def reset_derived(cfg: Config) -> None:
    """Delete all derived data (DB, pages, crops, intermediates)."""
    for d in (cfg.paths.pages_dir, cfg.paths.crops_dir):
        shutil.rmtree(d, ignore_errors=True)
    cfg.paths.db_path.unlink(missing_ok=True)
    for f in cfg.paths.data_dir.glob("blocks_*.jsonl"):
        f.unlink()
    for f in cfg.paths.data_dir.glob("chunks_*.json"):
        f.unlink()
