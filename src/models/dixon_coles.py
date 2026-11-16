"""Dixon–Coles goals model with time decay."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.models.markets import markets_from_grid


@dataclass
class DixonColesModel:
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float
    xi: float
    as_of: pd.Timestamp | None = None


def tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam * mu * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + mu * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def scoreline_grid_from_rates(
    lam: float,
    mu: float,
    rho: float,
    max_goals: int = 8,
) -> np.ndarray:
    goals = np.arange(0, max_goals + 1)
    home_p = np.exp(-lam) * np.power(lam, goals) / np.array(
        [math.factorial(int(k)) for k in goals]
    )
    away_p = np.exp(-mu) * np.power(mu, goals) / np.array(
        [math.factorial(int(k)) for k in goals]
    )
    grid = np.outer(home_p, away_p)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            grid[i, j] *= tau(i, j, lam, mu, rho)
    total = grid.sum()
    if total <= 0:
        raise ValueError("non-positive scoreline grid; check rho/lambda/mu")
    return grid / total
