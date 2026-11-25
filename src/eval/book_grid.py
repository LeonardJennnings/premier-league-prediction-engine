"""Score table rebuilt from the book's prices.

There is no free correct-score price. Two goal rates are chosen so a
Poisson score table (the low-score correction left at zero) matches the
de-vigged home/draw/away prices and the de-vigged over/under 2.5 price.
The table is our reconstruction. It is not the bookmaker's correct-score odds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.config import DIXON_COLES_MAX_GOALS
from src.models.dixon_coles import scoreline_grid_from_rates
from src.models.markets import markets_from_grid

# The plan's label. Use this sentence wherever the table is shown.
BOOK_SCORE_TABLE_LABEL = (
    "our reconstruction of the book's score table from its prices, "
    "not the bookmaker's correct-score odds"
)

# The slice stops if either derived market is further than this from the raw prices.
PRICE_LOG_LOSS_TOLERANCE = 0.005

_LOG_FLOOR = 1e-15
_RATE_LO = 0.05
_RATE_HI = 6.0


class BookGridUnfaithful(RuntimeError):
    """The rebuilt table does not reproduce the prices. Do not publish it."""


@dataclass(frozen=True)
class BookGrid:
    """One match: the two rates, the 0..8 grid, and the markets summed from it."""

    lam: float
    mu: float
    grid: np.ndarray
    home: float
    draw: float
    away: float
    over_25: float
    under_25: float
    squared_error: float


def poisson_probabilities(rate: float, max_goals: int) -> np.ndarray:
    """P(0), ..., P(max_goals) for a Poisson, not renormalised."""
    if rate < 0.0 or not math.isfinite(rate):
        raise ValueError("rate must be finite and non-negative")
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")
    if rate == 0.0:
        out = np.zeros(max_goals + 1, dtype=float)
        out[0] = 1.0
        return out
    goals = np.arange(0, max_goals + 1, dtype=float)
    log_p = -rate + goals * math.log(rate) - gammaln(goals + 1.0)
    return np.exp(log_p)


def markets_from_rates(lam: float, mu: float, max_goals: int = DIXON_COLES_MAX_GOALS) -> dict[str, float]:
    """1X2 and over/under 2.5 from an independent Poisson grid, rho = 0."""
    home_p = poisson_probabilities(lam, max_goals)
    away_p = poisson_probabilities(mu, max_goals)
    grid = np.outer(home_p, away_p)
    total = float(grid.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Poisson grid has no mass")
    grid = grid / total
    index = np.arange(max_goals + 1)
    home_goals = index[:, None]
    away_goals = index[None, :]
    return {
        "home": float(grid[home_goals > away_goals].sum()),
        "draw": float(np.trace(grid)),
        "away": float(grid[home_goals < away_goals].sum()),
        "over_25": float(grid[home_goals + away_goals >= 3].sum()),
        "under_25": float(grid[home_goals + away_goals <= 2].sum()),
    }


def _clip_prob(probability: float) -> float:
    if not math.isfinite(probability):
        raise ValueError("probability must be finite")
    return min(1.0 - 1e-12, max(1e-12, float(probability)))


def _total_rate_for_over(p_over: float) -> float:
    """Poisson mean whose P(3 or more) is near ``p_over``."""
    target = _clip_prob(p_over)
    lo, hi = 0.05, 8.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        at_most_two = math.exp(-mid) * (1.0 + mid + mid * mid / 2.0)
        if 1.0 - at_most_two < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _squared_error(lam: float, mu: float, target: np.ndarray, max_goals: int) -> float:
    markets = markets_from_rates(lam, mu, max_goals)
    predicted = np.array(
        [markets["home"], markets["draw"], markets["away"], markets["over_25"]],
        dtype=float,
    )
    return float(np.sum((predicted - target) ** 2))


def fit_book_grid(
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over: float,
    *,
    max_goals: int = DIXON_COLES_MAX_GOALS,
) -> BookGrid:
    """Find home and away goal rates that match these four prices. Rho stays 0.

    ``p_home``, ``p_draw`` and ``p_away`` are the de-vigged result prices.
    ``p_over`` is the de-vigged probability of three or more goals.
    """
    target = np.array(
        [_clip_prob(p_home), _clip_prob(p_draw), _clip_prob(p_away), _clip_prob(p_over)],
        dtype=float,
    )
    total = _total_rate_for_over(target[3])
    edge = target[0] / max(target[0] + target[2], 1e-9)
    starts = [
        (max(total * edge, _RATE_LO), max(total * (1.0 - edge), _RATE_LO)),
        (1.5, 1.2),
        (1.8, 0.9),
        (0.9, 1.6),
    ]
    log_lo = math.log(_RATE_LO)
    log_hi = math.log(_RATE_HI)

    def objective(theta: np.ndarray) -> float:
        lam = float(np.exp(theta[0]))
        mu = float(np.exp(theta[1]))
        return _squared_error(lam, mu, target, max_goals)

    best_error = float("inf")
    best_theta = np.log(np.array(starts[0], dtype=float))
    for lam0, mu0 in starts:
        start = np.log(np.array([lam0, mu0], dtype=float))
        fitted = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(log_lo, log_hi), (log_lo, log_hi)],
        )
        error = float(fitted.fun) if fitted.success else objective(fitted.x)
        if error < best_error:
            best_error = error
            best_theta = np.asarray(fitted.x, dtype=float)

    lam = float(np.exp(best_theta[0]))
    mu = float(np.exp(best_theta[1]))
    grid = scoreline_grid_from_rates(lam, mu, 0.0, max_goals=max_goals)
    markets = markets_from_grid(grid)
    predicted = np.array(
        [markets["home"], markets["draw"], markets["away"], markets["over_25"]],
        dtype=float,
    )
    return BookGrid(
        lam=lam,
        mu=mu,
        grid=grid,
        home=float(markets["home"]),
        draw=float(markets["draw"]),
        away=float(markets["away"]),
        over_25=float(markets["over_25"]),
        under_25=float(markets["under_25"]),
        squared_error=float(np.sum((predicted - target) ** 2)),
    )


def _picked_log_loss(probability: float) -> float:
    p = min(1.0, max(float(probability), _LOG_FLOOR))
    return float(-math.log(p))


def market_log_losses(
    *,
    result: str,
    home_goals: int,
    away_goals: int,
    price_home: float,
    price_draw: float,
    price_away: float,
    price_over: float,
    price_under: float,
    grid_home: float,
    grid_draw: float,
    grid_away: float,
    grid_over: float,
    grid_under: float,
) -> dict[str, float]:
    """Log loss of the rebuilt markets and of the raw prices, on one match."""
    price = {"H": price_home, "D": price_draw, "A": price_away}
    grid = {"H": grid_home, "D": grid_draw, "A": grid_away}
    if result not in price:
        raise ValueError(f"unknown result {result}")
    over = int(home_goals) + int(away_goals) >= 3
    return {
        "loss_1x2_grid": _picked_log_loss(grid[result]),
        "loss_1x2_price": _picked_log_loss(price[result]),
        "loss_ou_grid": _picked_log_loss(grid_over if over else grid_under),
        "loss_ou_price": _picked_log_loss(price_over if over else price_under),
    }


def fidelity_gap(rows: list[dict]) -> dict[str, float]:
    """Mean log loss of the rebuilt markets minus the raw prices.

    A positive gap means the rebuilt table put less probability on what
    happened than the price it was fitted to.
    """
    if not rows:
        raise ValueError("no reconstructed matches")
    keys = ("loss_1x2_grid", "loss_1x2_price", "loss_ou_grid", "loss_ou_price")
    means = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    means["gap_1x2"] = means["loss_1x2_grid"] - means["loss_1x2_price"]
    means["gap_ou"] = means["loss_ou_grid"] - means["loss_ou_price"]
    means["n_matches"] = float(len(rows))
    return means


# Closing prices only. An earlier over/under price is not a closing target.
CLOSING_PRICE_COLUMNS = (
    "implied_home",
    "implied_draw",
    "implied_away",
    "implied_over_25",
    "implied_under_25",
)
OPENING_PRICE_COLUMNS = (
    "implied_home_open",
    "implied_draw_open",
    "implied_away_open",
    "implied_over_25_open",
    "implied_under_25_open",
)

BOOK_RECONSTRUCTIONS = (
    ("book_score_reconstruction", "", CLOSING_PRICE_COLUMNS),
    ("book_score_reconstruction_open", "_open", OPENING_PRICE_COLUMNS),
)


def prices_complete(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    """True when every row has a finite price in each named column."""
    if any(column not in frame.columns for column in columns):
        return False
    values = frame.loc[:, list(columns)]
    if values.isna().any().any():
        return False
    numeric = values.apply(pd.to_numeric, errors="coerce")
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())


def assert_prices_reproduced(rows: list[dict], *, tolerance: float = PRICE_LOG_LOSS_TOLERANCE) -> dict[str, float]:
    """Stop the slice when the rebuilt markets miss the raw prices by more than the tolerance."""
    gap = fidelity_gap(rows)
    missed = []
    if abs(gap["gap_1x2"]) > tolerance:
        missed.append(
            f"home/draw/away log loss {gap['loss_1x2_grid']:.4f} versus "
            f"raw prices {gap['loss_1x2_price']:.4f} "
            f"(difference {gap['gap_1x2']:.4f}, {int(gap['n_matches'])} matches)"
        )
    if abs(gap["gap_ou"]) > tolerance:
        missed.append(
            f"over/under 2.5 log loss {gap['loss_ou_grid']:.4f} versus "
            f"raw prices {gap['loss_ou_price']:.4f} "
            f"(difference {gap['gap_ou']:.4f}, {int(gap['n_matches'])} matches)"
        )
    if missed:
        raise BookGridUnfaithful(
            "The rebuilt score table does not reproduce the prices within "
            f"{tolerance}. " + " ".join(missed)
        )
    return gap
