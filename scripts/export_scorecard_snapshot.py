#!/usr/bin/env python3
"""Export settled ISO weeks into a scorecard snapshot.

Writes the shape of GET /v1/predictions/weeks plus each week's matches, from
the local predictions store. Does not invent rows. An empty settled ledger
still writes the file and exits 2.

The Astro past pages do not read this file yet. They keep using the live API,
then the labelled sample.

Usage (repo root, after settle on this machine):

    python scripts/export_scorecard_snapshot.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import service  # noqa: E402
from src.config import PREDICTIONS_DB  # noqa: E402
from src.store.predictions import connect  # noqa: E402
from src.store.schema import init_db  # noqa: E402

DEFAULT_OUT = ROOT / "web" / "src" / "data" / "scorecard-snapshot.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--db", type=Path, default=PREDICTIONS_DB)
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()
    try:
        code = export_scorecard(args.db, args.out, model_version=args.model_version)
    except service.NotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    weeks = _week_count(args.out) if args.out.exists() else 0
    print(f"Wrote {weeks} settled weeks -> {args.out}")
    return code


def export_scorecard(db: Path, out: Path, *, model_version: str | None = None) -> int:
    """Write one scorecard JSON file. Return 0 when a week exists, else 2."""
    if not db.exists():
        print(f"error: predictions DB missing: {db}", file=sys.stderr)
        print("Run publish on this machine, then python -m src.jobs.settle", file=sys.stderr)
        return 1

    init_db(db)
    conn = connect(db)
    try:
        index = service.list_weeks(conn, model_version)
        weeks = []
        for summary in index["weeks"]:
            detail = service.get_week(conn, summary["iso_week"], index["model_version"])
            weeks.append(
                {
                    "iso_week": summary["iso_week"],
                    "match_count": summary["match_count"],
                    "matches": detail["matches"],
                }
            )
    finally:
        conn.close()

    payload = {
        "model_version": index["model_version"],
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weeks": weeks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if weeks else 2


def _week_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weeks = payload.get("weeks") or []
    return len(weeks)


if __name__ == "__main__":
    raise SystemExit(main())
