"""Elo rating comparator for 1X2 probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.config import ELO_HOME_BONUS, ELO_INITIAL, ELO_K

_HOME_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}


@dataclass
class EloModel:
    """Fitted Elo ratings plus the home bonus and K used to update them."""

    ratings: dict[str, float]
    home_bonus: float
    k: float


def _expected_home_score(r_home: float, r_away: float, home_bonus: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_home + home_bonus - r_away) / 400.0))


def fit_elo(
    matches: pd.DataFrame,
    k: float | None = None,
    home_bonus: float | None = None,
    initial: float | None = None,
) -> EloModel:
    """Walk train matches in date order and update Elo ratings after each result."""
    k = ELO_K if k is None else float(k)
    home_bonus = ELO_HOME_BONUS if home_bonus is None else float(home_bonus)
    initial = ELO_INITIAL if initial is None else float(initial)

    df = matches.dropna(subset=["date", "home_team", "away_team", "result"]).copy()
    df = df[df["result"].isin(_HOME_SCORE)]
    df = df.sort_values("date", kind="mergesort")

    ratings: dict[str, float] = {}
    for row in df.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team
        r_home = ratings.get(home_team, initial)
        r_away = ratings.get(away_team, initial)
        expected = _expected_home_score(r_home, r_away, home_bonus)
        actual = _HOME_SCORE[row.result]
        ratings[home_team] = r_home + k * (actual - expected)
        ratings[away_team] = r_away + k * ((1.0 - actual) - (1.0 - expected))

    return EloModel(ratings=ratings, home_bonus=home_bonus, k=k)


def predict_match(model: EloModel, home_team: str, away_team: str) -> dict[str, float]:
    """Return 1X2 probabilities from current Elo ratings (not BTTS / O/U)."""
    r_home = model.ratings.get(home_team, ELO_INITIAL)
    r_away = model.ratings.get(away_team, ELO_INITIAL)
    e = _expected_home_score(r_home, r_away, model.home_bonus)
    gap = abs((r_home + model.home_bonus) - r_away)
    p_draw = 0.28 * math.exp(-gap / 400.0)
    remain = 1.0 - p_draw
    return {
        "home": e * remain,
        "draw": p_draw,
        "away": (1.0 - e) * remain,
    }
