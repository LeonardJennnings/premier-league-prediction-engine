"""Filter fixtures to the rolling publish window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import PUBLISH_WINDOW_DAYS


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def matches_in_publish_window(
    matches: list[dict],
    *,
    now_utc: datetime,
    window_days: int = PUBLISH_WINDOW_DAYS,
) -> list[dict]:
    end = now_utc + timedelta(days=window_days)
    kept: list[dict] = []
    for match in matches:
        kickoff = _parse_utc(match.get("utcDate") or match["kickoff_utc"])
        if now_utc <= kickoff < end:
            kept.append(match)
    return kept
