"""Dixon–Coles goals model with time decay."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.config import (
    DIXON_COLES_L2,
    DIXON_COLES_MAX_GOALS,
    DIXON_COLES_PRIOR_KAPPA,
    DIXON_COLES_XI,
    THIN_PRIOR_PROMOTED_FACTOR,
)
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
    # 0 means the small L2 penalty is the only pull toward the average.
    prior_kappa: float = 0.0
    # Used for a club the fit has not seen, and only when prior_kappa > 0.
    new_club_attack: float = 1.0
    new_club_defence: float = 1.0


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


def _poisson_pmf(rate: float, max_goals: int) -> np.ndarray:
    goals = np.arange(0, max_goals + 1)
    return np.exp(-rate) * np.power(rate, goals) / np.array(
        [math.factorial(int(k)) for k in goals]
    )


def score_table_from_rates(
    lam: float,
    mu: float,
    rho: float,
    max_goals: int = 8,
) -> dict:
    """Score probabilities for 0..max_goals, plus one bucket for every other score.

    The low-score correction touches only cells inside the grid. The ``other``
    bucket is the independent-Poisson tail (any score with a side above
    ``max_goals``), then the grid and that bucket are scaled so they sum to 1.
    A result such as 9-0 is scored from ``other``, which stays positive for
    ordinary goal rates.
    """
    if int(max_goals) < 0:
        raise ValueError("max_goals must be at least 0")
    if lam < 0 or mu < 0 or not np.isfinite(lam) or not np.isfinite(mu):
        raise ValueError("rates must be finite and non-negative")
    home_p = _poisson_pmf(float(lam), int(max_goals))
    away_p = _poisson_pmf(float(mu), int(max_goals))
    grid = np.outer(home_p, away_p)
    # Tail before the low-score correction. That correction does not create
    # scores outside the grid, so the tail stays in the other bucket.
    other = max(0.0, 1.0 - float(home_p.sum() * away_p.sum()))
    limit = min(2, int(max_goals) + 1)
    for i in range(limit):
        for j in range(limit):
            grid[i, j] *= tau(i, j, float(lam), float(mu), float(rho))
    total = float(grid.sum()) + other
    if total <= 0 or not np.isfinite(total):
        raise ValueError("non-positive score table; check rho/lambda/mu")
    grid = grid / total
    other = float(other / total)
    size = int(max_goals) + 1
    return {
        "max_goals": int(max_goals),
        "cells": [[float(grid[i, j]) for j in range(size)] for i in range(size)],
        "other": other,
    }


def _stored_rate(
    model: DixonColesModel,
    rates: dict[str, float],
    team: str,
    new_club_rate: float,
) -> float:
    """Rate for a club in the fit, or a starting rate when the fit has not seen them.

    With the prior off, an unseen club stays at 1. With the prior on, an unseen
    club uses the new-club start stored on the model.
    """
    if team in rates:
        return float(rates[team])
    if float(model.prior_kappa) > 0.0:
        return float(new_club_rate)
    return 1.0


def expected_goals(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
) -> tuple[float, float]:
    """Home and away means: λ = α_home β_away γ, μ = α_away β_home.

    Teams missing from the fit use the new-club start when the prior is on,
    and 1 otherwise.
    """
    attack_home = _stored_rate(model, model.attack, home_team, model.new_club_attack)
    attack_away = _stored_rate(model, model.attack, away_team, model.new_club_attack)
    defence_home = _stored_rate(model, model.defence, home_team, model.new_club_defence)
    defence_away = _stored_rate(model, model.defence, away_team, model.new_club_defence)
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


def score_table(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    max_goals: int | None = None,
) -> dict:
    """Stored score table for one fixture, including the tail bucket.

    Teams missing from the fit use the same rates as ``expected_goals``.
    ``rho`` is the value actually used after it is projected into the
    admissible interval.
    """
    if max_goals is None:
        max_goals = DIXON_COLES_MAX_GOALS
    lam, mu = expected_goals(model, home_team, away_team)
    rho = _project_rho(model.rho, np.array([lam]), np.array([mu]))
    try:
        table = score_table_from_rates(lam, mu, rho, max_goals=max_goals)
    except ValueError:
        logger.warning(
            "Non-positive score table for %s vs %s; using rho=0",
            home_team,
            away_team,
        )
        rho = 0.0
        table = score_table_from_rates(lam, mu, rho, max_goals=max_goals)
    table["lambda_home"] = float(lam)
    table["lambda_away"] = float(mu)
    table["rho"] = float(rho)
    return table


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
    prior_kappa: float | None = None,
    promoted_factor: float | None = None,
) -> DixonColesModel:
    """Weighted Dixon–Coles MLE.

    ξ is fixed (config default ``DIXON_COLES_XI``), not jointly optimized.
    Attack is reparameterized so its mean is 1. With ``prior_kappa`` at 0,
    defence and attack are pulled toward 1 with an L2 term scaled by
    ``1 / (team_matches + 1)``. A positive ``prior_kappa`` replaces that term.
    It is the curvature of a penalty on log attack and log defence, so a club
    with few matches moves toward the pack and a club with a long history
    barely moves. A club that plays in the latest season in the fit and has
    no match in the season before that is pulled toward a lower attack and a
    higher concession rate. ``promoted_factor`` is that fixed shift. It is not
    searched inside the fit. If the optimizer raises, returns ``success=False``,
    returns a non-finite point, or stays in the τ ≤ 0 penalty region, the model
    logs a warning and falls back to the Poisson moment init with ρ = 0.

    ``max_goals`` is accepted for callers that share the predict signature.
    The likelihood uses the observed scores, so the grid cap does not enter the fit.
    """
    if max_goals is not None and int(max_goals) < 1:
        raise ValueError("max_goals must be at least 1")

    xi = float(DIXON_COLES_XI if xi is None else xi)
    l2 = float(DIXON_COLES_L2 if l2 is None else l2)
    if prior_kappa is None:
        prior_kappa = float(DIXON_COLES_PRIOR_KAPPA)
    else:
        prior_kappa = float(prior_kappa)
    if promoted_factor is None:
        promoted_factor = float(THIN_PRIOR_PROMOTED_FACTOR)
    else:
        promoted_factor = float(promoted_factor)
    if not np.isfinite(prior_kappa) or prior_kappa < 0.0:
        raise ValueError("prior_kappa must be finite and >= 0")
    if (
        not np.isfinite(promoted_factor)
        or promoted_factor <= 0.0
        or promoted_factor > 1.0
    ):
        raise ValueError("promoted_factor must be in (0, 1]")
    # The real prior replaces the small L2 term. Stacking both would be a second pull.
    if prior_kappa > 0.0:
        l2 = 0.0
    frame, as_of_ts = _prepare_matches(matches, as_of)
    if frame.empty:
        logger.warning("Dixon–Coles fit has no matches; returning a neutral model")
        return _neutral_model(xi, as_of_ts, prior_kappa=prior_kappa)

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
    attack0, defence0, gamma0, mass = _poisson_init(
        n, home_ix, away_ix, hg, ag, weights
    )
    is_new = _new_club_flags(frame, teams)
    attack0, defence0 = _shrink_start(
        attack0, defence0, mass, is_new, prior_kappa, promoted_factor
    )
    counts = np.bincount(home_ix, minlength=n) + np.bincount(away_ix, minlength=n)
    shrink = 1.0 / (counts.astype(float) + 1.0)
    log_fact = gammaln(hg + 1.0) + gammaln(ag + 1.0)
    low_score = _low_score_masks(home_goals, away_goals)
    log_factor_a, log_factor_d = _log_factors(is_new, promoted_factor)
    fallback = _model_from_fit(
        teams,
        attack0,
        defence0,
        gamma0,
        0.0,
        xi,
        as_of_ts,
        prior_kappa,
        is_new,
        promoted_factor,
    )

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
        prior_penalty = _profile_prior(
            attack, defence, log_factor_a, log_factor_d, prior_kappa
        )
        # Keep the free ρ parameter on the admissible side of the clip.
        pull = (rho - rho_hat) ** 2
        return -float(np.dot(weights, log_p)) + l2_penalty + prior_penalty + pull

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
        not result.success
        or not np.all(np.isfinite(result.x))
        or not np.isfinite(result.fun)
        or float(result.fun) >= _PENALTY / 10.0
    ):
        logger.warning(
            "Dixon–Coles optimizer failed (%s); falling back to Poisson init with rho=0",
            getattr(result, "message", "non-finite result"),
        )
        return fallback

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
    return _model_from_fit(
        teams,
        attack,
        defence,
        gamma,
        rho,
        xi,
        as_of_ts,
        prior_kappa,
        is_new,
        promoted_factor,
    )


def _new_club_flags(frame: pd.DataFrame, teams: list[str]) -> np.ndarray:
    """True for a club in the latest season of the fit and absent from the one before.

    One season in the fit is not enough to tell a new club from anyone else, so
    every club is treated as already having a history. The season labels are
    the training rows only.
    """
    flags = np.zeros(len(teams), dtype=bool)
    if "season_code" not in frame.columns or frame.empty:
        return flags
    seasons = sorted({str(code) for code in frame["season_code"].dropna().unique()})
    if len(seasons) < 2:
        return flags
    latest = seasons[-1]
    previous = seasons[-2]
    codes = frame["season_code"].astype(str)
    played_latest = set(frame.loc[codes == latest, "home_team"]) | set(
        frame.loc[codes == latest, "away_team"]
    )
    played_previous = set(frame.loc[codes == previous, "home_team"]) | set(
        frame.loc[codes == previous, "away_team"]
    )
    for i, team in enumerate(teams):
        flags[i] = team in played_latest and team not in played_previous
    return flags


def _log_factors(is_new: np.ndarray, promoted_factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Log shift for a new club: lower attack, higher concession rate, one fraction."""
    shift = float(math.log(promoted_factor))
    log_factor_a = np.where(is_new, shift, 0.0)
    log_factor_d = np.where(is_new, -shift, 0.0)
    return log_factor_a.astype(float), log_factor_d.astype(float)


def _shrink_start(
    attack: np.ndarray,
    defence: np.ndarray,
    mass: np.ndarray,
    is_new: np.ndarray,
    kappa: float,
    promoted_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Start the optimizer nearer the prior. The fit itself still applies the penalty."""
    if kappa <= 0.0:
        return attack, defence
    factor_a = np.where(is_new, promoted_factor, 1.0)
    factor_d = np.where(is_new, 1.0 / promoted_factor, 1.0)
    n_eff = np.maximum(np.asarray(mass, dtype=float), 0.0)
    attack = (n_eff * attack + kappa * factor_a) / (n_eff + kappa)
    defence = (n_eff * defence + kappa * factor_d) / (n_eff + kappa)
    mean_attack = float(attack.mean())
    if np.isfinite(mean_attack) and mean_attack > 0.0:
        attack = attack / mean_attack
    defence = np.clip(defence, math.exp(-5.0), math.exp(5.0))
    return attack, defence


def _profile_prior(
    attack: np.ndarray,
    defence: np.ndarray,
    log_factor_a: np.ndarray,
    log_factor_d: np.ndarray,
    kappa: float,
) -> float:
    """Penalty that shrinks log-rates toward each other, with a free common level.

    Curvature is about ``kappa`` per club, the same scale as that many matches.
    The common level is the mean of the current rates, so the penalty does not
    drag the whole league toward 1. A new club's target sits ``promoted_factor``
    away from that level.
    """
    if kappa <= 0.0:
        return 0.0
    rel_a = np.log(np.maximum(attack, 1e-12)) - log_factor_a
    rel_d = np.log(np.maximum(defence, 1e-12)) - log_factor_d
    penalty = float(
        np.sum((rel_a - rel_a.mean()) ** 2) + np.sum((rel_d - rel_d.mean()) ** 2)
    )
    if not np.isfinite(penalty):
        return _PENALTY
    return 0.5 * float(kappa) * penalty


def _new_club_levels(
    attack: np.ndarray,
    defence: np.ndarray,
    is_new: np.ndarray,
    promoted_factor: float,
) -> tuple[float, float]:
    """Starting attack and defence for a club the fit has not seen."""
    log_factor_a, log_factor_d = _log_factors(is_new, promoted_factor)
    center_a = float(np.mean(np.log(np.maximum(attack, 1e-12)) - log_factor_a))
    center_d = float(np.mean(np.log(np.maximum(defence, 1e-12)) - log_factor_d))
    new_attack = math.exp(center_a + math.log(promoted_factor))
    new_defence = math.exp(center_d - math.log(promoted_factor))
    if not np.isfinite(new_attack) or new_attack <= 0.0:
        new_attack = float(promoted_factor)
    if not np.isfinite(new_defence) or new_defence <= 0.0:
        new_defence = 1.0 / float(promoted_factor)
    return float(new_attack), float(new_defence)


def _model_from_fit(
    teams: list[str],
    attack: np.ndarray,
    defence: np.ndarray,
    gamma: float,
    rho: float,
    xi: float,
    as_of: pd.Timestamp | None,
    prior_kappa: float,
    is_new: np.ndarray,
    promoted_factor: float,
) -> DixonColesModel:
    if prior_kappa > 0.0:
        new_attack, new_defence = _new_club_levels(
            attack, defence, is_new, promoted_factor
        )
    else:
        new_attack, new_defence = 1.0, 1.0
    return _arrays_to_model(
        teams,
        attack,
        defence,
        gamma,
        rho,
        xi,
        as_of,
        prior_kappa=prior_kappa,
        new_club_attack=new_attack,
        new_club_defence=new_defence,
    )


def _prepare_matches(
    matches: pd.DataFrame,
    as_of,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Drop incomplete rows and matches after ``as_of`` (default: latest date)."""
    required = ["home_team", "away_team", "home_goals", "away_goals", "date"]
    missing = [column for column in required if column not in matches.columns]
    if missing:
        raise ValueError(f"matches missing columns: {missing}")

    columns = list(required)
    if "season_code" in matches.columns:
        columns.append("season_code")
    frame = matches.loc[:, columns].copy()
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


def _neutral_model(
    xi: float,
    as_of: pd.Timestamp | None,
    prior_kappa: float = 0.0,
) -> DixonColesModel:
    return DixonColesModel(
        attack={},
        defence={},
        home_advantage=1.3,
        rho=0.0,
        xi=xi,
        as_of=as_of,
        prior_kappa=float(prior_kappa),
    )


def _poisson_init(
    n: int,
    home_ix: np.ndarray,
    away_ix: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Moment attack/defence (mean attack 1), a home-advantage start, and weight mass."""
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
    return attack, defence, gamma, mass


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
    prior_kappa: float = 0.0,
    new_club_attack: float = 1.0,
    new_club_defence: float = 1.0,
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
        prior_kappa=float(prior_kappa),
        new_club_attack=float(new_club_attack),
        new_club_defence=float(new_club_defence),
    )
