"""Lab-only specialised over/under 2.5 comparators.

Product O/U stays on the Dixon–Coles scoreline grid. These helpers score
totals-oriented alternatives in the walk-forward harness only — never publish
or UI defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from src.models.dixon_coles import DixonColesModel, expected_goals

_EPS = 1e-15
_T_LO = 0.05
_T_HI = 10.0
_MIN_LOGISTIC_ROWS = 50


def over_25_from_total_rate(lam_total: float) -> float:
    """P(total goals >= 3) under an independent Poisson total with mean lam_total.

    Closed form: 1 - e^{-λ}(1 + λ + λ²/2). Clipped away from {0, 1} for scoring.
    """
    lam = float(lam_total)
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("lam_total must be a finite non-negative float")
    # P(X <= 2) for X ~ Poisson(λ)
    under = math.exp(-lam) * (1.0 + lam + 0.5 * lam * lam)
    over = 1.0 - under
    return float(np.clip(over, _EPS, 1.0 - _EPS))


def predict_poisson_total_over_25(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
) -> float:
    """Independent-Poisson totals line from Dixon–Coles λ and μ (ignores ρ).

    Same attack/defence/home rates as the product grid, but totals mass is
    Poisson(λ+μ) rather than the bivariate grid sum. Useful as a lab read on
    whether low-score dependence is hurting or helping O/U 2.5.
    """
    lam, mu = expected_goals(model, home_team, away_team)
    return over_25_from_total_rate(lam + mu)


def apply_binary_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale a binary probability: sigmoid(logit(p) / T)."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    prob = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    logit = np.log(prob) - np.log(1.0 - prob)
    scaled = logit / float(temperature)
    return 1.0 / (1.0 + np.exp(-scaled))


def fit_binary_temperature(y_true, p: np.ndarray) -> float:
    """Fit T > 0 that minimises binary log loss of temperature-scaled p."""
    y = np.asarray(y_true, dtype=float)
    arr = np.asarray(p, dtype=float)

    def nll(t: float) -> float:
        prob = np.clip(apply_binary_temperature(arr, t), _EPS, 1.0 - _EPS)
        return float(-(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)).mean())

    result = minimize_scalar(nll, bounds=(_T_LO, _T_HI), method="bounded")
    return float(result.x)


@dataclass
class TotalsLogisticModel:
    """Logistic O/U 2.5 on pre-match λ features from a goals model."""

    clf: LogisticRegression
    feature_names: tuple[str, ...] = ("lam", "mu", "lam_total")


def _design_matrix(lams: np.ndarray, mus: np.ndarray) -> np.ndarray:
    lam = np.asarray(lams, dtype=float).reshape(-1)
    mu = np.asarray(mus, dtype=float).reshape(-1)
    return np.column_stack([lam, mu, lam + mu])


def _rate_rows(
    matches: pd.DataFrame, model: DixonColesModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lams: list[float] = []
    mus: list[float] = []
    y: list[float] = []
    for row in matches.itertuples(index=False):
        lam, mu = expected_goals(model, row.home_team, row.away_team)
        lams.append(lam)
        mus.append(mu)
        total = float(row.home_goals) + float(row.away_goals)
        y.append(1.0 if total >= 3 else 0.0)
    return np.asarray(lams), np.asarray(mus), np.asarray(y, dtype=float)


def fit_totals_logistic(
    matches: pd.DataFrame,
    model: DixonColesModel,
) -> TotalsLogisticModel | None:
    """Fit a lab logistic on train-fold DC rates. Returns None if too thin."""
    if matches.empty or "home_goals" not in matches.columns:
        return None
    lams, mus, y = _rate_rows(matches, model)
    if len(y) < _MIN_LOGISTIC_ROWS or len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        C=1.0,
    )
    clf.fit(_design_matrix(lams, mus), y.astype(int))
    return TotalsLogisticModel(clf=clf)


def predict_totals_logistic_over_25(
    lab_model: TotalsLogisticModel,
    goals_model: DixonColesModel,
    home_team: str,
    away_team: str,
) -> float:
    """P(over 2.5) from the lab logistic given DC rates for one fixture."""
    lam, mu = expected_goals(goals_model, home_team, away_team)
    X = _design_matrix(np.array([lam]), np.array([mu]))
    p = float(lab_model.clf.predict_proba(X)[0, 1])
    return float(np.clip(p, _EPS, 1.0 - _EPS))
