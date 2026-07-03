"""CLI: python -m rulehound.ingest data/raw/core_rules.pdf (design doc §4)."""

from __future__ import annotations

import argparse
import json
import sys

from ..config import load_config
from .pipeline import run_ingest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rulehound.ingest")
    parser.add_argument("pdf", help="Path to the Core Rules PDF")
    parser.add_argument("--config", default=None, help="Path to config.toml")
    parser.add_argument("--force", action="store_true", help="Re-run all phases")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    summary = run_ingest(args.pdf, cfg, force=args.force)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
