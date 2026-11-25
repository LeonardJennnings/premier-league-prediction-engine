"""Publish Premier League fixtures into the append-only ledger.

Home, draw, and away on a new row are the plain sum of the stored score
table. The artefact's temperature is not applied. The public version name
ends in ``-grid`` so that change is visible next to the 17 Nov rows.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.config import LEDGER_DIR, MODELS_DIR, PREDICTIONS_DB, PUBLISH_WINDOW_DAYS
from src.fixtures.books import load_book_index, lookup_book
from src.fixtures.client import fetch_pl_matches
from src.fixtures.window import matches_in_publish_window
from src.ledger.lines import FORECAST_LINE_MODEL, build_prediction_line
from src.ledger.log import append_book_lines, append_predictions
from src.logging_setup import setup_logging
from src.models.artefacts import (
    _resolve_version,
    load_artefact,
    published_model_version,
)
from src.store.predictions import connect, insert_predictions
from src.store.schema import init_db
from src.teams.mapping import load_team_name_map, resolve_team_name

logger = logging.getLogger(__name__)

_PROB_FIELDS = (
    "p_home",
    "p_draw",
    "p_away",
    "p_btts_yes",
    "p_btts_no",
    "p_over_25",
    "p_under_25",
)


def publish_slate(
    model_version: str | None = None,
    *,
    now_utc: datetime | None = None,
    token: str | None = None,
) -> dict:
    """Load an artefact and append one ledger line per new in-window match.

    An unmapped name aborts the batch before anything is written. A mapped
    club with no fitted history is published at league-average rates and
    flagged. A match that has already kicked off is refused and nothing from
    that call is written. A match that already has a ``model`` line is
    skipped, including when this artefact is newer. Fixtures are refetched;
    the on-disk cache is only written.
    """
    now = _as_utc(now_utc)
    version = _resolve_version(model_version, Path(MODELS_DIR))
    # Temperature stays on the artefact for the lab. It is not applied here.
    model, _temperature, _hyperparams = load_artefact(version, models_dir=MODELS_DIR)
    public_version = published_model_version(version)

    date_from = now.date()
    # football-data.org dateTo is exclusive, so add one day to include the last day.
    date_to = (now + timedelta(days=PUBLISH_WINDOW_DAYS)).date() + timedelta(days=1)
    # Skip the cache read. A same-day postponement or a newly listed fixture
    # must show up on the next publish. A successful fetch still writes the cache.
    fetched = fetch_pl_matches(
        date_from.isoformat(),
        date_to.isoformat(),
        token=token,
        use_cache=False,
    )
    in_window = matches_in_publish_window(fetched, now_utc=now)

    counts = {
        "fetched": len(fetched),
        "in_window": len(in_window),
        "inserted": 0,
        "skipped_duplicate": 0,
    }
    if not in_window:
        logger.info("Publish %s: %s", public_version, counts)
        return counts

    mapped = _mapped_fixtures(in_window, load_team_name_map())
    published_at = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    as_of_date = _as_of_date(model.as_of)
    books = load_book_index()

    lines = []
    sqlite_rows = []
    book_lines = []
    for fixture, home, away in mapped:
        line = build_prediction_line(
            model,
            fixture,
            home,
            away,
            model_version=public_version,
            artefact_version=version,
            published_at=published_at,
            as_of_date=as_of_date,
            forecast_line=FORECAST_LINE_MODEL,
        )
        quote = lookup_book(books, home, away, fixture["kickoff_utc"])
        lines.append(line)
        sqlite_rows.append(_sqlite_row(line, quote))
        if quote:
            book_lines.append(_book_line(line, quote, captured_at=published_at))

    # Ledger first. A started match raises here and the store is left untouched.
    written = append_predictions(lines, now_utc=now, path=LEDGER_DIR)
    inserted_ids = set(written["inserted_ids"])
    if inserted_ids:
        append_book_lines(
            [row for row in book_lines if int(row["match_id"]) in inserted_ids],
            path=LEDGER_DIR,
        )
        init_db(PREDICTIONS_DB)
        conn = connect(PREDICTIONS_DB)
        try:
            insert_predictions(
                conn,
                [row for row in sqlite_rows if int(row["match_id"]) in inserted_ids],
            )
        finally:
            conn.close()
    counts["inserted"] = written["inserted"]
    counts["skipped_duplicate"] = written["skipped_duplicate"]

    logger.info("Publish %s: %s", public_version, counts)
    return counts


def main() -> None:
    setup_logging()
    counts = publish_slate()
    logger.info(
        "Published slate fetched=%s in_window=%s inserted=%s skipped_duplicate=%s",
        counts["fetched"],
        counts["in_window"],
        counts["inserted"],
        counts["skipped_duplicate"],
    )


def _sqlite_row(line: dict, quote: dict | None) -> dict:
    detail = {
        "artefact_version": line["artefact_version"],
        "forecast_line": line["forecast_line"],
        "lambda_home": line["lambda_home"],
        "lambda_away": line["lambda_away"],
        "rho": line["rho"],
        "score_table": line["score_table"],
        "most_likely_score": line["most_likely_score"],
        "next_scores": line["next_scores"],
        "totals": line["totals"],
        "league_average_teams": line["league_average_teams"],
        "uses_league_average": line["uses_league_average"],
        "score_omitted_reason": line["score_omitted_reason"],
    }
    row = {
        "match_id": line["match_id"],
        "model_version": line["model_version"],
        "kickoff_utc": line["kickoff_utc"],
        "home_team": line["home_team"],
        "away_team": line["away_team"],
        "matchday": line["matchday"],
        "competition": line["competition"],
        "published_at": line["published_at"],
        "as_of_date": line["as_of_date"],
        "status": "scheduled",
        "detail_json": json.dumps(detail, sort_keys=True),
    }
    for field in _PROB_FIELDS:
        row[field] = line[field]
    row.update(_book_fields(quote))
    return row


def _book_line(line: dict, quote: dict, *, captured_at: str) -> dict:
    return {
        "match_id": line["match_id"],
        "forecast_line": line["forecast_line"],
        "kind": "opening",
        "captured_at": captured_at,
        "kickoff_utc": line["kickoff_utc"],
        "source": quote.get("book_source"),
        "p_home": quote.get("book_p_home"),
        "p_draw": quote.get("book_p_draw"),
        "p_away": quote.get("book_p_away"),
        "p_over_25": quote.get("book_p_over_25"),
        "p_under_25": quote.get("book_p_under_25"),
    }


def _book_fields(quote: dict | None) -> dict:
    if not quote:
        return {
            "book_p_home": None,
            "book_p_draw": None,
            "book_p_away": None,
            "book_p_over_25": None,
            "book_p_under_25": None,
            "book_source": None,
        }
    return {
        "book_p_home": quote.get("book_p_home"),
        "book_p_draw": quote.get("book_p_draw"),
        "book_p_away": quote.get("book_p_away"),
        "book_p_over_25": quote.get("book_p_over_25"),
        "book_p_under_25": quote.get("book_p_under_25"),
        "book_source": quote.get("book_source"),
    }


def _as_utc(now_utc: datetime | None) -> datetime:
    if now_utc is None:
        return datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=timezone.utc)
    return now_utc


def _as_of_date(as_of) -> str:
    if as_of is None:
        return ""
    ts = pd.Timestamp(as_of)
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d")


def _mapped_fixtures(
    fixtures: list[dict], mapping: dict[str, str]
) -> list[tuple[dict, str, str]]:
    """Map every fixture. An unmapped name aborts before any line is written.

    A name that is on the map but missing from the fitted model is kept.
    ``build_prediction_line`` gives that club league-average rates and flags it.
    """
    unmapped: list[str] = []
    mapped: list[tuple[dict, str, str]] = []
    for fixture in fixtures:
        home = _canonical(fixture["home_team"], mapping, unmapped)
        away = _canonical(fixture["away_team"], mapping, unmapped)
        if home is None or away is None:
            continue
        mapped.append((fixture, home, away))
    if unmapped:
        listed = ", ".join(repr(name) for name in _unique(unmapped))
        logger.error("Aborting publish; unmapped teams: %s", listed)
        raise KeyError(f"Unmapped team: {listed}")
    return mapped


def _canonical(name: str, mapping: dict[str, str], unmapped: list[str]) -> str | None:
    try:
        return resolve_team_name(name, mapping)
    except KeyError:
        unmapped.append(name)
        return None


def _unique(names: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


if __name__ == "__main__":
    main()
