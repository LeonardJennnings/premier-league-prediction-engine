"""Dixon–Coles goals model with time decay."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.config import DIXON_COLES_L2, DIXON_COLES_MAX_GOALS, DIXON_COLES_XI
from src.models.markets import markets_from_grid

logger = logging.getLogger(__name__)

# Conservative ρ box. The likelihood also rejects any ρ that makes τ ≤ 0.
_RHO_LO = -0.2
_RHO_HI = 0.2
# Worse than any realistic weighted NLL, so a τ ≤ 0 step is never accepted.
_PENALTY = 1.0e12


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


def expected_goals(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
) -> tuple[float, float]:
    """Home and away means: λ = α_home β_away γ, μ = α_away β_home.

    Teams missing from the fit use league-average attack and defence (1.0).
    """
    attack_home = model.attack.get(home_team, 1.0)
    attack_away = model.attack.get(away_team, 1.0)
    defence_home = model.defence.get(home_team, 1.0)
    defence_away = model.defence.get(away_team, 1.0)
    lam = attack_home * defence_away * model.home_advantage
    mu = attack_away * defence_home
    return float(lam), float(mu)


def scoreline_grid(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    max_goals: int | None = None,
) -> np.ndarray:
    """Renormalized scoreline grid for one fixture.

    ``max_goals`` defaults to ``DIXON_COLES_MAX_GOALS``. ρ is projected into the
    admissible interval for these rates so τ stays positive.
    """
    if max_goals is None:
        max_goals = DIXON_COLES_MAX_GOALS
    lam, mu = expected_goals(model, home_team, away_team)
    rho = _project_rho(model.rho, np.array([lam]), np.array([mu]))
    try:
        return scoreline_grid_from_rates(lam, mu, rho, max_goals=max_goals)
    except ValueError:
        logger.warning(
            "Non-positive Dixon–Coles grid for %s vs %s; using rho=0",
            home_team,
            away_team,
        )
        return scoreline_grid_from_rates(lam, mu, 0.0, max_goals=max_goals)


def predict_match(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    max_goals: int | None = None,
) -> dict[str, float]:
    """1X2, BTTS, and over/under 2.5 from one Dixon–Coles grid."""
    return markets_from_grid(
        scoreline_grid(model, home_team, away_team, max_goals=max_goals)
    )


def fit_dixon_coles(
    matches: pd.DataFrame,
    as_of=None,
    xi: float | None = None,
    l2: float | None = None,
    max_goals: int | None = None,
) -> DixonColesModel:
    """Weighted Dixon–Coles MLE.

    ξ is fixed (config default ``DIXON_COLES_XI``), not jointly optimized.
    Attack is reparameterized so its mean is 1. Defence and attack are pulled
    toward 1 with an L2 term scaled by ``1 / (team_matches + 1)``. If the
    optimizer raises, returns a non-finite point, or stays in the τ ≤ 0 penalty
    region, the model falls back to the Poisson moment init with ρ = 0. A finite
    but unsuccessful L-BFGS-B status keeps that point and logs a warning.

    ``max_goals`` is accepted for callers that share the predict signature.
    The likelihood uses the observed scores, so the grid cap does not enter the fit.
    """
    if max_goals is not None and int(max_goals) < 1:
        raise ValueError("max_goals must be at least 1")

    xi = float(DIXON_COLES_XI if xi is None else xi)
    l2 = float(DIXON_COLES_L2 if l2 is None else l2)
    frame, as_of_ts = _prepare_matches(matches, as_of)
    if frame.empty:
        logger.warning("Dixon–Coles fit has no matches; returning a neutral model")
        return _neutral_model(xi, as_of_ts)

    teams = sorted(set(frame["home_team"]) | set(frame["away_team"]))
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    home_ix = frame["home_team"].map(index).to_numpy(dtype=int)
    away_ix = frame["away_team"].map(index).to_numpy(dtype=int)
    home_goals = frame["home_goals"].to_numpy(dtype=int)
    away_goals = frame["away_goals"].to_numpy(dtype=int)
    days_ago = (as_of_ts - frame["date"]).dt.total_seconds().to_numpy(dtype=float)
    days_ago = np.maximum(days_ago / 86400.0, 0.0)
    weights = np.exp(-xi * days_ago)

    hg = home_goals.astype(float)
    ag = away_goals.astype(float)
    attack0, defence0, gamma0 = _poisson_init(n, home_ix, away_ix, hg, ag, weights)
    counts = np.bincount(home_ix, minlength=n) + np.bincount(away_ix, minlength=n)
    shrink = 1.0 / (counts.astype(float) + 1.0)
    log_fact = gammaln(hg + 1.0) + gammaln(ag + 1.0)
    low_score = _low_score_masks(home_goals, away_goals)
    fallback = _arrays_to_model(teams, attack0, defence0, gamma0, 0.0, xi, as_of_ts)

    def objective(theta: np.ndarray) -> float:
        attack, defence, gamma, rho = _unpack_theta(theta, n)
        if (
            not np.isfinite(attack).all()
            or not np.isfinite(defence).all()
            or not np.isfinite(gamma)
            or not np.isfinite(rho)
            or gamma <= 0.0
        ):
            return _PENALTY
        lam = np.maximum(attack[home_ix] * defence[away_ix] * gamma, 1e-8)
        mu = np.maximum(attack[away_ix] * defence[home_ix], 1e-8)
        rho_hat = _project_rho(rho, lam, mu)
        tau_values = _tau_array(rho_hat, lam, mu, low_score)
        if np.any(~np.isfinite(tau_values)) or np.any(tau_values <= 0.0):
            return _PENALTY
        log_p = (
            np.log(tau_values)
            - lam
            + hg * np.log(lam)
            - mu
            + ag * np.log(mu)
            - log_fact
        )
        if not np.all(np.isfinite(log_p)):
            return _PENALTY
        l2_penalty = l2 * float(
            np.sum(shrink * ((attack - 1.0) ** 2 + (defence - 1.0) ** 2))
        )
        # Keep the free ρ parameter on the admissible side of the clip.
        pull = (rho - rho_hat) ** 2
        return -float(np.dot(weights, log_p)) + l2_penalty + pull

    try:
        result = minimize(
            objective,
            _pack_init(attack0, defence0, gamma0, n),
            method="L-BFGS-B",
            bounds=_theta_bounds(n),
            options={"maxiter": 500, "ftol": 1e-12, "maxfun": 50_000},
        )
    except Exception as exc:
        logger.warning(
            "Dixon–Coles optimizer failed (%s); falling back to Poisson init with rho=0",
            exc,
        )
        return fallback

    if (
        not np.all(np.isfinite(result.x))
        or not np.isfinite(result.fun)
        or float(result.fun) >= _PENALTY / 10.0
    ):
        logger.warning(
            "Dixon–Coles optimizer failed (%s); falling back to Poisson init with rho=0",
            getattr(result, "message", "non-finite result"),
        )
        return fallback

    if not result.success:
        logger.warning(
            "Dixon–Coles optimizer reported %s; keeping the finite solution",
            result.message,
        )

    attack, defence, gamma, rho = _unpack_theta(result.x, n)
    lam = np.maximum(attack[home_ix] * defence[away_ix] * gamma, 1e-8)
    mu = np.maximum(attack[away_ix] * defence[home_ix], 1e-8)
    rho = _project_rho(rho, lam, mu)
    tau_values = _tau_array(rho, lam, mu, low_score)
    if np.any(~np.isfinite(tau_values)) or np.any(tau_values <= 0.0):
        logger.warning(
            "Fitted rho made tau non-positive; falling back to rho=0 with fitted strengths"
        )
        rho = 0.0
    return _arrays_to_model(teams, attack, defence, gamma, rho, xi, as_of_ts)


def _prepare_matches(
    matches: pd.DataFrame,
    as_of,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Drop incomplete rows and matches after ``as_of`` (default: latest date)."""
    required = ["home_team", "away_team", "home_goals", "away_goals", "date"]
    missing = [column for column in required if column not in matches.columns]
    if missing:
        raise ValueError(f"matches missing columns: {missing}")

    frame = matches.loc[:, required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
    frame = frame.dropna(subset=required)
    frame = frame[(frame["home_goals"] >= 0) & (frame["away_goals"] >= 0)]
    if frame.empty:
        as_of_ts = None if as_of is None else pd.Timestamp(as_of)
        return frame, as_of_ts

    as_of_ts = pd.Timestamp(frame["date"].max()) if as_of is None else pd.Timestamp(as_of)
    frame = frame.loc[frame["date"] <= as_of_ts].copy()
    if frame.empty:
        return frame, as_of_ts
    frame["home_goals"] = frame["home_goals"].round().astype(int)
    frame["away_goals"] = frame["away_goals"].round().astype(int)
    return frame, as_of_ts


def _neutral_model(xi: float, as_of: pd.Timestamp | None) -> DixonColesModel:
    return DixonColesModel(
        attack={},
        defence={},
        home_advantage=1.3,
        rho=0.0,
        xi=xi,
        as_of=as_of,
    )


def _poisson_init(
    n: int,
    home_ix: np.ndarray,
    away_ix: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Moment attack/defence (mean attack 1) and a home-advantage start near the data."""
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(len(home_goals), dtype=float)
    weight_sum = float(weights.sum())
    avg_home = float(np.dot(weights, home_goals) / weight_sum)
    avg_away = float(np.dot(weights, away_goals) / weight_sum)
    league = (avg_home + avg_away) / 2.0
    if league <= 0.0:
        league = 1.0

    scored = np.zeros(n)
    conceded = np.zeros(n)
    mass = np.zeros(n)
    np.add.at(scored, home_ix, weights * home_goals)
    np.add.at(conceded, home_ix, weights * away_goals)
    np.add.at(mass, home_ix, weights)
    np.add.at(scored, away_ix, weights * away_goals)
    np.add.at(conceded, away_ix, weights * home_goals)
    np.add.at(mass, away_ix, weights)

    attack = np.ones(n)
    defence = np.ones(n)
    seen = mass > 0.0
    attack[seen] = np.maximum(scored[seen] / mass[seen] / league, 1e-3)
    defence[seen] = np.maximum(conceded[seen] / mass[seen] / league, 1e-3)
    attack = attack / attack.mean()

    base = attack[home_ix] * defence[away_ix]
    base_mean = float(np.dot(weights, base) / weight_sum)
    if avg_home > 0.0 and base_mean > 0.0 and np.isfinite(base_mean):
        gamma = avg_home / base_mean
    else:
        gamma = 1.3
    if not np.isfinite(gamma) or gamma <= 0.0:
        gamma = 1.3
    gamma = float(np.clip(gamma, math.exp(-1.5), math.exp(2.5)))
    defence = np.clip(defence, math.exp(-5.0), math.exp(5.0))
    return attack, defence, gamma


def _pack_init(
    attack: np.ndarray,
    defence: np.ndarray,
    gamma: float,
    n: int,
) -> np.ndarray:
    """Free parameters: n−1 log-attacks, log-defence, log home advantage, ρ."""
    log_attack = np.log(np.maximum(attack, 1e-8))
    log_attack = log_attack - log_attack.mean()
    parts: list[np.ndarray] = []
    if n > 1:
        parts.append(log_attack[1:])
    parts.append(np.log(np.maximum(defence, 1e-8)))
    parts.append(np.array([math.log(max(gamma, 1e-8)), -0.03], dtype=float))
    return np.concatenate(parts)


def _theta_bounds(n: int) -> list[tuple[float | None, float | None]]:
    """Box ρ. Log-defence and log-γ are wide numerical bounds, not a prior."""
    n_attack = max(n - 1, 0)
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * n_attack
    bounds += [(-5.0, 5.0)] * n
    bounds.append((-1.5, 2.5))
    bounds.append((_RHO_LO, _RHO_HI))
    return bounds


def _unpack_theta(theta: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Map free parameters to strengths. Mean attack is exactly 1."""
    if n == 1:
        log_attack = np.zeros(1, dtype=float)
        offset = 0
    else:
        log_attack = np.empty(n, dtype=float)
        log_attack[1:] = theta[: n - 1]
        log_attack[0] = -float(np.sum(log_attack[1:]))
        offset = n - 1
    # Subtract the max so exp() does not overflow. Subtracting a constant
    # cancels once attack is divided by its mean.
    log_attack = log_attack - np.max(log_attack)
    attack = np.exp(log_attack)
    attack = attack / attack.mean()
    defence = np.exp(theta[offset : offset + n])
    gamma = float(math.exp(float(theta[offset + n])))
    rho = float(theta[offset + n + 1])
    return attack, defence, gamma, rho


def _low_score_masks(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        (home_goals == 0) & (away_goals == 0),
        (home_goals == 1) & (away_goals == 0),
        (home_goals == 0) & (away_goals == 1),
        (home_goals == 1) & (away_goals == 1),
    )


def _tau_array(
    rho: float,
    lam: np.ndarray,
    mu: np.ndarray,
    low_score: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Dixon–Coles τ on observed scores. Every other cell is 1."""
    m00, m10, m01, m11 = low_score
    tau_values = np.ones(lam.shape, dtype=float)
    tau_values[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau_values[m10] = 1.0 + mu[m10] * rho
    tau_values[m01] = 1.0 + lam[m01] * rho
    tau_values[m11] = 1.0 - rho
    return tau_values


def _admissible_rho_interval(lam: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """ρ interval on which τ > 0 for 0–0, 1–0, 0–1, and 1–1 at these rates.

    Intersected with the conservative box (−0.2, 0.2).
    """
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-8)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-8)
    lower = max(_RHO_LO, float(np.max(-1.0 / lam)), float(np.max(-1.0 / mu)))
    upper = min(_RHO_HI, 1.0, float(np.min(1.0 / (lam * mu))))
    return lower, upper


def _project_rho(rho: float, lam: np.ndarray, mu: np.ndarray) -> float:
    """Clip ρ into the open admissible interval. 0 is always feasible for finite rates."""
    if not np.isfinite(rho):
        return 0.0
    lower, upper = _admissible_rho_interval(lam, mu)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return 0.0
    pad = min(1e-8, 0.25 * (upper - lower))
    return float(np.clip(rho, lower + pad, upper - pad))


def _arrays_to_model(
    teams: list[str],
    attack: np.ndarray,
    defence: np.ndarray,
    gamma: float,
    rho: float,
    xi: float,
    as_of: pd.Timestamp | None,
) -> DixonColesModel:
    attack = np.asarray(attack, dtype=float)
    mean_attack = float(attack.mean()) if len(attack) else 1.0
    if not np.isfinite(mean_attack) or mean_attack <= 0.0:
        attack = np.ones(len(teams), dtype=float)
    else:
        attack = attack / mean_attack
    defence = np.asarray(defence, dtype=float)
    return DixonColesModel(
        attack={team: float(attack[i]) for i, team in enumerate(teams)},
        defence={team: float(defence[i]) for i, team in enumerate(teams)},
        home_advantage=float(gamma),
        rho=float(rho),
        xi=float(xi),
        as_of=as_of,
    )
