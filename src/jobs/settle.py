"""Settle published rows from football-data.org finished matches.

Scores, result, and status are written. Probability columns are not.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from src.config import PREDICTIONS_DB
from src.fixtures.client import fetch_pl_matches
from src.logging_setup import setup_logging
from src.store.predictions import connect, settle_prediction
from src.store.schema import init_db

logger = logging.getLogger(__name__)


def settle_from_api(*, token: str | None = None, model_version: str | None = None) -> dict:
    """Fill settlement fields for unsettled rows covered by one finished-match pull.

    The request spans the kickoff dates of those rows. football-data.org
    ``dateTo`` is exclusive, so the end date is one day after the latest kickoff.
    A finished match with no prediction row is logged and skipped.
    """
    init_db(PREDICTIONS_DB)
    conn = connect(PREDICTIONS_DB)
    try:
        candidates = _unsettled(conn, model_version)
        if not candidates:
            counts = _counts(0, 0, 0, 0)
            logger.info("Settle %s: %s", model_version or "all", counts)
            return counts

        date_from, date_to = _query_dates(candidates)
        # Skip the cache read. A finished set for these dates can grow between runs.
        # A successful fetch still overwrites the cache file.
        matches = fetch_pl_matches(
            date_from,
            date_to,
            token=token,
            status="FINISHED",
            use_cache=False,
        )
        existing_ids = {
            int(row["match_id"])
            for row in conn.execute("SELECT match_id FROM predictions")
        }
        scores: dict[int, tuple[int, int]] = {}
        for match in matches:
            score = _full_time_score(match)
            if score is None:
                continue
            scores[int(match["match_id"])] = score

        settled = 0
        for row in candidates:
            score = scores.get(int(row["match_id"]))
            if score is None:
                continue
            home_goals, away_goals = score
            updated = settle_prediction(
                conn,
                int(row["match_id"]),
                row["model_version"],
                home_goals,
                away_goals,
                _result_code(home_goals, away_goals),
            )
            if updated:
                settled += 1

        skipped_unknown = _skip_unknown(matches, existing_ids)
        counts = _counts(len(candidates), len(matches), settled, skipped_unknown)
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


def _unsettled(conn, model_version: str | None) -> list[dict]:
    sql = "SELECT * FROM predictions WHERE status != 'settled'"
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
    return start.isoformat(), end.isoformat()


def _kickoff_date(value: str):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


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
