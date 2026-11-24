#!/usr/bin/env python3
"""Manual slate refresh. Cron stays off.

Modes
  freshness  Read the committed upcoming snapshot. No token.
  publish    pipeline, precompute, export the upcoming snapshot. Needs a token.
  settle     settle, then export the scorecard. Needs a token and a local DB.

A GitHub Actions runner starts without data/processed/predictions.sqlite.
Settle stops before calling football-data.org when that file is absent.
Nothing here commits or pushes.

    python scripts/refresh_slate.py --mode freshness
    python scripts/refresh_slate.py --mode publish
    python scripts/refresh_slate.py --mode settle
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freshness", "publish", "settle"), required=True)
    parser.add_argument("--now", default=None, help="UTC timestamp for freshness only")
    args = parser.parse_args()

    if args.mode == "freshness":
        command = [sys.executable, str(ROOT / "scripts" / "check_snapshot_freshness.py")]
        if args.now:
            command.extend(["--now", args.now])
        return _run(command)

    # Load .env for local runs. An already-set variable wins, including an
    # empty Actions secret. Never print the value. Freshness stays on the
    # stdlib path above so it does not need this import.
    from dotenv import load_dotenv

    sys.path.insert(0, str(ROOT))
    load_dotenv(ROOT / ".env")
    from src.config import PREDICTIONS_DB

    token = os.environ.get("FOOTBALL_DATA_ORG_TOKEN", "")
    missing = require_token(token)
    if missing:
        return missing

    if args.mode == "publish":
        code = _run([sys.executable, "-m", "src.pipeline"])
        if code:
            return code
        code = _run([sys.executable, "-m", "src.jobs.precompute"])
        if code:
            return code
        return _run([sys.executable, str(ROOT / "scripts" / "export_upcoming_snapshot.py")])

    missing_db = require_predictions_db(PREDICTIONS_DB)
    if missing_db:
        return missing_db
    code = _run([sys.executable, "-m", "src.jobs.settle"])
    if code:
        return code
    return _run([sys.executable, str(ROOT / "scripts" / "export_scorecard_snapshot.py")])


def require_token(token: str) -> int:
    """Return 1 when the fixture token is missing. The value is not printed."""
    if token.strip():
        print("TOKEN_PRESENT=yes")
        return 0
    print("error: FOOTBALL_DATA_ORG_TOKEN is missing or empty.", file=sys.stderr)
    print(
        "Use the same name in .env, the Cursor environment secret, "
        "and the GitHub Actions secret.",
        file=sys.stderr,
    )
    print("This step does not print the secret.", file=sys.stderr)
    return 1


def require_predictions_db(path: Path) -> int:
    """Return 2 when settle has no local ledger to update."""
    if path.is_file():
        return 0
    print(f"error: predictions DB missing: {path}", file=sys.stderr)
    print(
        "Publish on this machine first (python scripts/refresh_slate.py --mode publish).",
        file=sys.stderr,
    )
    print(
        "GitHub Actions does not keep data/processed/predictions.sqlite between runs.",
        file=sys.stderr,
    )
    return 2


def _run(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
