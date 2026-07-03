"""Run the server: python -m rulehound.api [--port 8000]."""

from __future__ import annotations

import argparse

import uvicorn

from ..config import load_config
from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="rulehound.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    app = create_app(load_config(args.config))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
