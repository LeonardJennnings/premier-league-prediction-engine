"""Poisson goals model for match outcome probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
import pandas as pd


@dataclass
class PoissonModel:
    """Team attack/defence strengths plus home advantage.

    ``away_base`` is the average away goals in the fit. The away rate uses
    that base. An earlier version used ``league_avg`` (the average of home
    and away scoring) for the away rate, then multiplied the home rate by
    home advantage as well. Average teams were then predicted to score too
    many away goals, and the comparison flattered Dixon-Coles.
    """

    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    league_avg: float
    away_base: float


def fit_poisson(matches: pd.DataFrame) -> PoissonModel:
    """Estimate attack, defence, and home advantage from past matches."""
    df = matches.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()

    # Average goals scored by one side in one match.
    league_avg = float((df["home_goals"].mean() + df["away_goals"].mean()) / 2)
    if league_avg <= 0:
        raise ValueError("league_avg must be positive")

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    scored: dict[str, list[float]] = {t: [] for t in teams}
    conceded: dict[str, list[float]] = {t: [] for t in teams}

    for row in df.itertuples(index=False):
        scored[row.home_team].append(float(row.home_goals))
        conceded[row.home_team].append(float(row.away_goals))
        scored[row.away_team].append(float(row.away_goals))
        conceded[row.away_team].append(float(row.home_goals))

    # A team with no goals yet would otherwise get a rate of exactly 0, and
    # the log loss of any goal they then score would be infinite. The floor
    # matches the Dixon-Coles moment start.
    floor = 1e-3
    attack = {
        t: max((sum(scored[t]) / len(scored[t]) / league_avg) if scored[t] else 1.0, floor)
        for t in teams
    }
    defence = {
        t: max((sum(conceded[t]) / len(conceded[t]) / league_avg) if conceded[t] else 1.0, floor)
        for t in teams
    }

    # Home teams score more than the overall one-side average → advantage > 1.
    avg_home = float(df["home_goals"].mean())
    avg_away = float(df["away_goals"].mean())
    home_advantage = float(avg_home / league_avg)

    return PoissonModel(
        attack=attack,
        defence=defence,
        home_advantage=home_advantage,
        league_avg=league_avg,
        away_base=avg_away,
    )


def match_rates(model: PoissonModel, home_team: str, away_team: str) -> tuple[float, float]:
    """Home and away means for one fixture.

    Same shape as Dixon-Coles: home is scaled by home advantage, away is not
    scaled by the combined average. Attack and defence stay ratios to the
    one-side league average, so an average team predicts the sample home
    mean and the sample away mean.
    """
    home_attack = model.attack.get(home_team, 1.0)
    away_attack = model.attack.get(away_team, 1.0)
    home_defence = model.defence.get(home_team, 1.0)
    away_defence = model.defence.get(away_team, 1.0)
    lambda_home = home_attack * away_defence * model.league_avg * model.home_advantage
    lambda_away = away_attack * home_defence * model.away_base
    # Keep every rate positive so an exact score still has a probability.
    return float(max(lambda_home, 1e-6)), float(max(lambda_away, 1e-6))


def predict_match(
    model: PoissonModel,
    home_team: str,
    away_team: str,
    max_goals: int = 8,
) -> dict[str, float]:
    """Return home/draw/away probabilities that sum to 1."""
    lambda_home, lambda_away = match_rates(model, home_team, away_team)

    # Probability of exactly k goals under a Poisson with mean λ.
    goals = np.arange(0, max_goals + 1)
    # Manual Poisson pmf: e^{-λ} * λ^k / k!
    home_goal_probs = np.exp(-lambda_home) * np.power(lambda_home, goals) / np.array(
        [math.factorial(int(k)) for k in goals]
    )
    away_goal_probs = np.exp(-lambda_away) * np.power(lambda_away, goals) / np.array(
        [math.factorial(int(k)) for k in goals]
    )

    # Independent scoreline grid: P(i, j) = P(home=i) * P(away=j)
    grid = np.outer(home_goal_probs, away_goal_probs)

    p_home = float(np.tril(grid, k=-1).sum())  # home goals > away goals
    p_draw = float(np.trace(grid))            # equal
    p_away = float(np.triu(grid, k=1).sum())  # away goals > home goals

    total = p_home + p_draw + p_away
    if total <= 0:
        return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}

    return {
        "home": p_home / total,
        "draw": p_draw / total,
        "away": p_away / total,
    }