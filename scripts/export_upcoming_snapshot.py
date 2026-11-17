#!/usr/bin/env python3
"""Export GET /v1/predictions/upcoming shape into the Astro build-time snapshot.

Writes web/src/data/upcoming-snapshot.json from the local predictions store so
a static site (or Cursor publish-app) can ship real fixtures without a live API
at build time. Does not invent rows — empty store yields an empty matches list.

Usage (repo root, after pipeline + publish):

    python scripts/export_upcoming_snapshot.py
    # optional: --out path/to/file.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import service  # noqa: E402
from src.config import PREDICTIONS_DB  # noqa: E402
from src.store.predictions import connect  # noqa: E402
from src.store.schema import init_db  # noqa: E402

DEFAULT_OUT = ROOT / "web" / "src" / "data" / "upcoming-snapshot.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSON path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=PREDICTIONS_DB,
        help=f"Predictions SQLite path (default: {PREDICTIONS_DB})",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"error: predictions DB missing: {args.db}", file=sys.stderr)
        print("Run: python -m src.jobs.precompute (with FOOTBALL_DATA_ORG_TOKEN)", file=sys.stderr)
        return 1

    init_db(args.db)
    conn = connect(args.db)
    try:
        try:
            payload = service.get_upcoming(conn, None)
        except service.NotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    finally:
        conn.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    n = len(payload.get("matches") or [])
    print(f"Wrote {n} upcoming matches ({payload.get('model_version')}) → {args.out}")
    return 0 if n > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
