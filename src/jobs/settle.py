"""Settle published rows from football-data.org match updates.

Scores, result, kickoff, and status may change. Probability and book columns
do not. Status values are ``scheduled``, ``postponed``, ``cancelled``, and
``settled``. ``POSTPONED`` and ``SUSPENDED`` map to ``postponed``.
``CANCELLED`` maps to ``cancelled``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from src.config import PREDICTIONS_DB
from src.fixtures.client import fetch_pl_matches
from src.logging_setup import setup_logging
from src.store.predictions import connect, settle_prediction, update_prediction_state
from src.store.schema import init_db

logger = logging.getLogger(__name__)

# After a row is postponed, look this far past its stored kickoff for a new date.
POSTPONED_LOOKAHEAD_DAYS = 90

_STATUS_FROM_API = {
    "POSTPONED": "postponed",
    "SUSPENDED": "postponed",
    "CANCELLED": "cancelled",
    "CANCELED": "cancelled",
}
_LISTED_API = {"TIMED", "SCHEDULED"}


def settle_from_api(*, token: str | None = None, model_version: str | None = None) -> dict:
    """Refresh status and scores for tracked rows from one fixture pull.

    The request spans stored kickoffs. football-data.org ``dateTo`` is
    exclusive, so the end date is one day after the latest kickoff. Postponed
    rows also look ahead for a rescheduled ``utcDate``. The cache is not read;
    a successful fetch still overwrites the cache file.

    ``FINISHED`` writes the full-time score, including a correction when an
    already settled row has different goals. ``POSTPONED`` and ``SUSPENDED``
    set ``postponed``. ``CANCELLED`` sets ``cancelled``. A listed match can
    move a postponed row back to ``scheduled`` and store a new kickoff.
    A match with no prediction row is logged and skipped.
    """
    init_db(PREDICTIONS_DB)
    conn = connect(PREDICTIONS_DB)
    try:
        rows = _tracked(conn, model_version)
        if not rows:
            counts = _counts(0, 0, 0, 0)
            logger.info("Settle %s: %s", model_version or "all", counts)
            return counts

        unsettled = sum(1 for row in rows if row["status"] != "settled")
        date_from, date_to = _query_dates(rows)
        # Skip the cache read. Postponements and corrected scores must be visible
        # on the next run. A successful fetch still overwrites the cache file.
        matches = fetch_pl_matches(
            date_from,
            date_to,
            token=token,
            use_cache=False,
        )
        existing_ids = {
            int(row["match_id"])
            for row in conn.execute("SELECT match_id FROM predictions")
        }
        by_id = {int(match["match_id"]): match for match in matches}

        settled = 0
        for row in rows:
            match = by_id.get(int(row["match_id"]))
            if match is None:
                continue
            if _apply_match(conn, row, match):
                settled += 1

        skipped_unknown = _skip_unknown(matches, existing_ids)
        counts = _counts(unsettled, len(matches), settled, skipped_unknown)
    finally:
        conn.close()

    logger.info("Settle %s: %s", model_version or "all", counts)
    return counts


def main() -> None:
    setup_logging()
    counts = settle_from_api()
    logger.info(
        "Settled unsettled=%s fetched=%s settled=%s skipped_unknown=%s",
        counts["unsettled"],
        counts["fetched"],
        counts["settled"],
        counts["skipped_unknown"],
    )


def _tracked(conn, model_version: str | None) -> list[dict]:
    """Rows that can still change. Cancelled rows are left alone."""
    sql = "SELECT * FROM predictions WHERE status != 'cancelled'"
    params: list[str] = []
    if model_version is not None:
        sql += " AND model_version = ?"
        params.append(model_version)
    sql += " ORDER BY kickoff_utc, match_id, model_version"
    return [dict(row) for row in conn.execute(sql, params)]


def _query_dates(rows: list[dict]) -> tuple[str, str]:
    dates = [_kickoff_date(row["kickoff_utc"]) for row in rows]
    start = min(dates)
    # dateTo is exclusive, so add a day to include the latest kickoff date.
    end = max(dates) + timedelta(days=1)
    postponed = [
        _kickoff_date(row["kickoff_utc"])
        for row in rows
        if row["status"] == "postponed"
    ]
    if postponed:
        horizon = max(postponed) + timedelta(days=POSTPONED_LOOKAHEAD_DAYS)
        end = max(end, horizon + timedelta(days=1))
    return start.isoformat(), end.isoformat()


def _kickoff_date(value: str):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _apply_match(conn, row: dict, match: dict) -> bool:
    """Apply one API match. True only when settlement scores were written."""
    if match.get("status") == "FINISHED":
        return _apply_finished(conn, row, match)
    if row["status"] == "settled":
        return False
    _apply_open(conn, row, match)
    return False


def _apply_finished(conn, row: dict, match: dict) -> bool:
    score = _full_time_score(match)
    if score is None:
        return False
    home_goals, away_goals = score
    if (
        row["status"] == "settled"
        and row["home_goals"] == home_goals
        and row["away_goals"] == away_goals
    ):
        return False
    return settle_prediction(
        conn,
        int(row["match_id"]),
        row["model_version"],
        home_goals,
        away_goals,
        _result_code(home_goals, away_goals),
    )


def _apply_open(conn, row: dict, match: dict) -> None:
    api_status = match.get("status")
    kickoff = match.get("kickoff_utc")
    fields: dict = {}
    mapped = _STATUS_FROM_API.get(api_status)
    if mapped is not None:
        if row["status"] != mapped:
            fields["status"] = mapped
    elif api_status in _LISTED_API and row["status"] == "postponed":
        fields["status"] = "scheduled"
    elif api_status not in _LISTED_API:
        return
    if kickoff and kickoff != row["kickoff_utc"]:
        if mapped is not None or api_status in _LISTED_API:
            fields["kickoff_utc"] = kickoff
    if not fields:
        return
    update_prediction_state(
        conn, int(row["match_id"]), row["model_version"], **fields
    )
    logger.info(
        "Updated match %s %s %s", row["match_id"], row["model_version"], fields
    )


def _full_time_score(match: dict) -> tuple[int, int] | None:
    if match.get("status") != "FINISHED":
        return None
    home = match.get("home_goals")
    away = match.get("away_goals")
    if home is None or away is None:
        return None
    return int(home), int(away)


def _result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _skip_unknown(matches: list[dict], existing_ids: set[int]) -> int:
    skipped = 0
    seen: set[int] = set()
    for match in matches:
        match_id = int(match["match_id"])
        if match_id in existing_ids or match_id in seen:
            continue
        seen.add(match_id)
        logger.info("Skipping match_id %s; no prediction row", match_id)
        skipped += 1
    return skipped


def _counts(unsettled: int, fetched: int, settled: int, skipped_unknown: int) -> dict:
    return {
        "unsettled": unsettled,
        "fetched": fetched,
        "settled": settled,
        "skipped_unknown": skipped_unknown,
    }


if __name__ == "__main__":
    main()
