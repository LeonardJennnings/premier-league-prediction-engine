"""Publish a 10-day Premier League slate from a saved Dixon–Coles artefact."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR, PREDICTIONS_DB, PUBLISH_WINDOW_DAYS
from src.fixtures.client import fetch_pl_matches
from src.fixtures.window import matches_in_publish_window
from src.logging_setup import setup_logging
from src.models.artefacts import _resolve_version, load_artefact, teams_in_model
from src.models.predict import predict_matches
from src.store.predictions import connect, insert_prediction
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
    """Load an artefact, predict the in-window slate, and insert each row once.

    Every in-window fixture is mapped and checked against the model before any
    insert. An unmapped name or a team missing from attack or defence aborts
    the batch.
    """
    now = _as_utc(now_utc)
    version = _resolve_version(model_version, Path(MODELS_DIR))
    model, temperature, _hyperparams = load_artefact(version, models_dir=MODELS_DIR)

    date_from = now.date()
    # football-data.org dateTo is exclusive, so add one day to include the 10th day.
    date_to = (now + timedelta(days=PUBLISH_WINDOW_DAYS)).date() + timedelta(days=1)
    fetched = fetch_pl_matches(date_from.isoformat(), date_to.isoformat(), token=token)
    in_window = matches_in_publish_window(fetched, now_utc=now)

    counts = {
        "fetched": len(fetched),
        "in_window": len(in_window),
        "inserted": 0,
        "skipped_duplicate": 0,
    }
    if not in_window:
        logger.info("Publish %s: %s", version, counts)
        return counts

    mapped = _validated_fixtures(in_window, load_team_name_map(), model)
    frame = pd.DataFrame(
        {
            "match_id": [fixture["match_id"] for fixture, _home, _away in mapped],
            "home_team": [home for _fixture, home, _away in mapped],
            "away_team": [away for _fixture, _home, away in mapped],
        }
    )
    predicted = predict_matches(model, frame, temperature=temperature)
    published_at = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    as_of_date = _as_of_date(model.as_of)

    init_db(PREDICTIONS_DB)
    conn = connect(PREDICTIONS_DB)
    try:
        for (fixture, home, away), pred in zip(
            mapped, predicted.itertuples(index=False), strict=True
        ):
            row = {
                "match_id": fixture["match_id"],
                "model_version": version,
                "kickoff_utc": fixture["kickoff_utc"],
                "home_team": home,
                "away_team": away,
                "matchday": fixture.get("matchday"),
                "competition": "PL",
                "published_at": published_at,
                "as_of_date": as_of_date,
                "status": "scheduled",
                "book_p_home": None,
                "book_p_draw": None,
                "book_p_away": None,
                "book_p_over_25": None,
                "book_p_under_25": None,
                "book_source": None,
            }
            for field in _PROB_FIELDS:
                row[field] = float(getattr(pred, field))
            if insert_prediction(conn, row):
                counts["inserted"] += 1
            else:
                counts["skipped_duplicate"] += 1
    finally:
        conn.close()

    logger.info("Publish %s: %s", version, counts)
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


def _validated_fixtures(
    fixtures: list[dict], mapping: dict[str, str], model
) -> list[tuple[dict, str, str]]:
    """Map every fixture, then reject teams absent from attack or defence."""
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

    known = teams_in_model(model)
    missing: list[str] = []
    for _fixture, home, away in mapped:
        for team in (home, away):
            if team not in known or team not in model.defence:
                missing.append(team)
    if missing:
        listed = ", ".join(repr(name) for name in _unique(missing))
        logger.error("Aborting publish; teams missing from model: %s", listed)
        raise KeyError(f"Team missing from model: {listed}")
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
