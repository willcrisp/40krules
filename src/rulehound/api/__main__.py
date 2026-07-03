"""Run the server: python -m rulehound.api [--port 8000].

Defaults are read from the environment so a container can run this with no
args: $PORT (the convention Railway and most PaaS platforms inject) or
$RULEHOUND_PORT for the port, $RULEHOUND_HOST for the bind address.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from ..config import load_config
from .app import create_app


def main() -> None:
    default_port = int(os.environ.get("PORT") or os.environ.get("RULEHOUND_PORT") or 8000)
    default_host = os.environ.get("RULEHOUND_HOST", "127.0.0.1")

    parser = argparse.ArgumentParser(prog="rulehound.api")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    app = create_app(load_config(args.config))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
