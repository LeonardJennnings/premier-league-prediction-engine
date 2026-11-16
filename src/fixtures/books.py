"""Fill nullable book probabilities from the football-data.co.uk fixture list.

The free football-data.org API has no odds. This module downloads the public
upcoming-fixtures CSV, de-vigs Bet365 1X2 (and over/under 2.5 when both
prices are present), and matches rows on canonical team names plus date.
A failed download or an unmatched fixture leaves the columns null.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.config import FD_UK_FIXTURES_CSV, FD_UK_FIXTURES_URL
from src.features import add_implied_odds

logger = logging.getLogger(__name__)

BOOK_SOURCE = "football-data.co.uk fixtures.csv (B365)"
REQUEST_TIMEOUT = 30

# Opening prices are filled on the upcoming file. Closing columns are the
# fallback when the opening trio is blank.
_ONE_X_TWO = (
    ("B365H", "B365D", "B365A"),
    ("B365CH", "B365CD", "B365CA"),
)
_OVER_UNDER = (
    ("B365>2.5", "B365<2.5"),
    ("B365C>2.5", "B365C<2.5"),
)


def download_fixtures_csv(
    dest: Path | None = None,
    *,
    session=None,
    timeout: int = REQUEST_TIMEOUT,
) -> Path | None:
    """Download the fixtures CSV. Return None instead of raising on failure."""
    path = FD_UK_FIXTURES_CSV if dest is None else dest
    try:
        getter = session.get if session is not None else requests.get
        response = getter(FD_UK_FIXTURES_URL, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning(
            "Book fixture CSV unavailable (%s); book columns left null", exc
        )
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def load_book_index(dest: Path | None = None, *, session=None) -> dict:
    """Download and index book probabilities. Empty when that cannot be done."""
    path = download_fixtures_csv(dest, session=session)
    if path is None:
        return {}
    try:
        return index_fixtures_csv(path)
    except Exception as exc:
        logger.warning(
            "Book fixture CSV could not be read (%s); book columns left null", exc
        )
        return {}


def index_fixtures_csv(path: Path) -> dict[tuple[str, str, str], dict]:
    """Index Premier League rows by ``(home, away, YYYY-MM-DD)``.

    When a ``Div`` column exists, only ``E0`` is kept. Rows without a usable
    Bet365 1X2 price are omitted. Over/under is set only when both sides exist.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"HomeTeam", "AwayTeam", "Date"}
    if not required <= set(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"Book fixture CSV missing columns: {missing}")
    if "Div" in frame.columns:
        frame = frame.loc[frame["Div"].astype(str).str.strip() == "E0"].copy()
    if frame.empty:
        return {}

    dates = pd.to_datetime(frame["Date"], dayfirst=True, format="mixed", errors="coerce")
    indexed: dict[tuple[str, str, str], dict] = {}
    for idx, row in frame.iterrows():
        day = dates.loc[idx]
        if pd.isna(day):
            continue
        home = _name(row["HomeTeam"])
        away = _name(row["AwayTeam"])
        if home is None or away is None:
            continue
        books = _books_from_row(row)
        if books is None:
            continue
        indexed[(home, away, day.strftime("%Y-%m-%d"))] = books
    return indexed


def lookup_book(
    index: dict[tuple[str, str, str], dict],
    home_team: str,
    away_team: str,
    kickoff_utc: str,
) -> dict | None:
    """Return a copy of the book fields, or None when nothing matches."""
    home = home_team.strip()
    away = away_team.strip()
    for day in kickoff_dates(kickoff_utc):
        found = index.get((home, away, day))
        if found is not None:
            return dict(found)
    return None


def kickoff_dates(kickoff_utc: str) -> list[str]:
    """Calendar dates that may match a date-only fixtures row.

    Europe/London is tried first, then the UTC date when it differs, so a
    kickoff just after midnight UTC still matches the UK listing.
    """
    text = str(kickoff_utc).strip()
    if len(text) == 10 and text[4] == "-" and "T" not in text:
        return [text]
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    utc_day = parsed.astimezone(timezone.utc).date().isoformat()
    try:
        london_day = parsed.astimezone(ZoneInfo("Europe/London")).date().isoformat()
    except Exception:
        london_day = utc_day
    if london_day == utc_day:
        return [london_day]
    return [london_day, utc_day]


def _books_from_row(row: pd.Series) -> dict | None:
    prices = _first_prices(row, _ONE_X_TWO)
    if prices is None:
        return None
    home, draw, away = prices
    payload = {
        "home_odds_close": home,
        "draw_odds_close": draw,
        "away_odds_close": away,
    }
    totals = _first_prices(row, _OVER_UNDER)
    if totals is not None:
        payload["over_25_odds_close"] = totals[0]
        payload["under_25_odds_close"] = totals[1]
    implied = add_implied_odds(pd.DataFrame([payload])).iloc[0]
    if "implied_home" not in implied or pd.isna(implied["implied_home"]):
        return None
    books = {
        "book_p_home": float(implied["implied_home"]),
        "book_p_draw": float(implied["implied_draw"]),
        "book_p_away": float(implied["implied_away"]),
        "book_p_over_25": None,
        "book_p_under_25": None,
        "book_source": BOOK_SOURCE,
    }
    if "implied_over_25" in implied and pd.notna(implied["implied_over_25"]):
        books["book_p_over_25"] = float(implied["implied_over_25"])
        books["book_p_under_25"] = float(implied["implied_under_25"])
    return books


def _first_prices(row: pd.Series, groups: tuple[tuple[str, ...], ...]) -> list[float] | None:
    for names in groups:
        if not all(name in row.index for name in names):
            continue
        odds = [_decimal_odds(row[name]) for name in names]
        if all(price is not None for price in odds):
            return odds
    return None


def _decimal_odds(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 1.0:
        return None
    return number


def _name(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text
