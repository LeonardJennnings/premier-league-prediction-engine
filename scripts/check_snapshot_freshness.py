#!/usr/bin/env python3
"""Report whether the committed upcoming snapshot still has a future kickoff.

Reads web/src/data/upcoming-snapshot.json. Does not call football-data.org and
does not read FOOTBALL_DATA_ORG_TOKEN.

Exit 0: at least one kickoff is still at or after --now.
Exit 2: the file loaded, and every kickoff is before --now (or there are no matches).
Exit 1: the file is missing or is not an upcoming payload.

    python scripts/check_snapshot_freshness.py
    python scripts/check_snapshot_freshness.py --now 2026-11-24T12:00:00Z
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "web" / "src" / "data" / "upcoming-snapshot.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--now",
        default=None,
        help="UTC timestamp to compare with kickoffs (default: current UTC)",
    )
    args = parser.parse_args()
    try:
        now = _parse_utc(args.now) if args.now else datetime.now(timezone.utc)
        report = assess_snapshot(args.snapshot, now)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for key in (
        "model_version",
        "matches",
        "earliest_kickoff",
        "latest_kickoff",
        "still_upcoming",
        "kicked_off",
        "scheduled_after_kickoff",
        "status",
    ):
        print(f"{key}={report[key]}")
    return 0 if report["status"] == "fresh" else 2


def assess_snapshot(path: Path, now: datetime) -> dict:
    """Summarise one upcoming snapshot against an UTC instant."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        raise ValueError(f"upcoming snapshot has no matches list: {path}")
    now_utc = _as_utc(now)
    kickoffs: list[datetime] = []
    scheduled_after = 0
    for match in payload["matches"]:
        if not isinstance(match, dict) or "kickoff_utc" not in match:
            raise ValueError(f"match missing kickoff_utc in {path}")
        kick = _parse_utc(str(match["kickoff_utc"]))
        kickoffs.append(kick)
        if kick < now_utc and match.get("status") == "scheduled":
            scheduled_after += 1
    still = sum(1 for kick in kickoffs if kick >= now_utc)
    if not kickoffs:
        status = "empty"
    elif still:
        status = "fresh"
    else:
        status = "stale"
    version = payload.get("model_version") or "unknown"
    return {
        "model_version": version,
        "matches": len(kickoffs),
        "earliest_kickoff": _fmt(min(kickoffs)) if kickoffs else "",
        "latest_kickoff": _fmt(max(kickoffs)) if kickoffs else "",
        "still_upcoming": still,
        "kicked_off": len(kickoffs) - still,
        "scheduled_after_kickoff": scheduled_after,
        "status": status,
    }


def _parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _fmt(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
