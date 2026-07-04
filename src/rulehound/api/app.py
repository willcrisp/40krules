"""FastAPI app (design doc §7). Model + store load once at startup (§8)."""

from __future__ import annotations

import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..config import Config, load_config
from ..ingest import embed
from ..ingest.pipeline import run_ingest
from ..search.spell import SpellCorrector
from ..store import sqlite_store
from ..store.sqlite_store import SqliteStore


class IngestManager:
    """Runs ingest in a background thread; one job at a time."""

    def __init__(self, app_state: "AppState") -> None:
        self.state = app_state
        self._lock = threading.Lock()
        self.status: dict = {"state": "idle", "message": "", "log": []}

    def start(self, pdf_path: Path, force: bool = False) -> bool:
        with self._lock:
            if self.status["state"] == "running":
                return False
            self.status = {"state": "running", "message": f"ingesting {pdf_path.name}", "log": []}
        threading.Thread(target=self._run, args=(pdf_path, force), daemon=True).start()
        return True

    def _run(self, pdf_path: Path, force: bool) -> None:
        def log(msg: str) -> None:
            print(msg)
            self.status["log"].append(msg)
            self.status["message"] = msg

        try:
            summary = run_ingest(pdf_path, self.state.cfg, store=self.state.store, force=force, log=log)
            self.state.refresh_embedder()
            self.state.refresh_corrector()
            self.status.update(state="done", summary=summary, message="ingest complete")
        except Exception as exc:
            traceback.print_exc()
            self.status.update(state="error", message=f"{type(exc).__name__}: {exc}")


class AppState:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cfg.paths.ensure_dirs()
        self.store = SqliteStore(cfg.paths.db_path, vector_dim=cfg.embedding.dimension)
        self.embedder = None
        self.embedder_error: str | None = None
        self.corrector: SpellCorrector | None = None
        self.ingest = IngestManager(self)
        self.refresh_embedder()
        self.refresh_corrector()

    def refresh_embedder(self) -> None:
        """Load the embedder once; refuse vector queries on model mismatch (§4.5)."""
        stored_model = self.store.get_meta(sqlite_store.META_MODEL)
        if stored_model is not None and stored_model != self.cfg.embedding.model:
            self.embedder = None
            self.embedder_error = (
                f"model mismatch: DB embedded with '{stored_model}', "
                f"config says '{self.cfg.embedding.model}' — vector search disabled"
            )
            return
        if self.embedder is not None:
            return
        errors: list[str] = []
        self.embedder = embed.get_embedder(self.cfg.embedding, log=errors.append)
        self.embedder_error = errors[0] if errors else None

    def refresh_corrector(self) -> None:
        """(Re)build the query spell corrector from the corpus vocabulary."""
        vocab = self.store.load_vocab()
        self.corrector = SpellCorrector(vocab) if vocab else None

    def close(self) -> None:
        self.store.close()


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rulehound = AppState(cfg)
        yield
        app.state.rulehound.close()

    app = FastAPI(title="Rulehound", lifespan=lifespan)

    from .routes import router

    app.include_router(router)

    cfg.paths.ensure_dirs()
    app.mount("/crops", StaticFiles(directory=str(cfg.paths.crops_dir)), name="crops")
    app.mount("/pages", StaticFiles(directory=str(cfg.paths.pages_dir)), name="pages")
    return app
