import math

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import (
    fit_dixon_coles,
    predict_match,
    scoreline_grid,
    scoreline_grid_from_rates,
    tau,
)
from src.models.markets import markets_from_grid


def test_tau_rho_zero_is_one():
    assert tau(0, 0, 1.2, 1.1, 0.0) == 1.0
    assert tau(1, 1, 1.2, 1.1, 0.0) == 1.0


def test_tau_formulas():
    lam, mu, rho = 1.5, 1.2, -0.1
    assert abs(tau(0, 0, lam, mu, rho) - (1 - lam * mu * rho)) < 1e-12
    assert abs(tau(1, 0, lam, mu, rho) - (1 + mu * rho)) < 1e-12
    assert abs(tau(0, 1, lam, mu, rho) - (1 + lam * rho)) < 1e-12
    assert abs(tau(1, 1, lam, mu, rho) - (1 - rho)) < 1e-12
    assert tau(2, 1, lam, mu, rho) == 1.0


def test_rho_zero_grid_matches_independent_poisson_1x2():
    lam, mu = 1.4, 1.1
    max_goals = 8
    grid = scoreline_grid_from_rates(lam, mu, rho=0.0, max_goals=max_goals)
    m = markets_from_grid(grid)

    # Independent Poisson reference
    goals = np.arange(0, max_goals + 1)
    ph = np.exp(-lam) * np.power(lam, goals) / np.array([math.factorial(int(k)) for k in goals])
    pa = np.exp(-mu) * np.power(mu, goals) / np.array([math.factorial(int(k)) for k in goals])
    ref = np.outer(ph, pa)
    ref = ref / ref.sum()
    home = float(np.tril(ref, k=-1).sum())
    draw = float(np.trace(ref))
    away = float(np.triu(ref, k=1).sum())
    assert abs(m["home"] - home) < 1e-9
    assert abs(m["draw"] - draw) < 1e-9
    assert abs(m["away"] - away) < 1e-9


def _toy_matches() -> pd.DataFrame:
    rows = []
    teams = ["Alpha", "Beta", "Gamma"]
    start = pd.Timestamp("2024-01-01")
    for i in range(30):
        h = teams[i % 3]
        a = teams[(i + 1) % 3]
        rows.append(
            {
                "date": start + pd.Timedelta(days=i * 3),
                "home_team": h,
                "away_team": a,
                "home_goals": 1 + (i % 2),
                "away_goals": i % 2,
            }
        )
    return pd.DataFrame(rows)


def test_fit_predict_probs_sum_to_one():
    model = fit_dixon_coles(_toy_matches())
    probs = predict_match(model, "Alpha", "Beta")
    assert abs(probs["home"] + probs["draw"] + probs["away"] - 1.0) < 1e-8


def test_scoreline_grid_renormalized():
    model = fit_dixon_coles(_toy_matches())
    grid = scoreline_grid(model, "Alpha", "Beta")
    assert abs(grid.sum() - 1.0) < 1e-8


def test_unseen_team_does_not_crash():
    model = fit_dixon_coles(_toy_matches())
    probs = predict_match(model, "NewTeam", "Alpha")
    assert abs(sum(probs[k] for k in ("home", "draw", "away")) - 1.0) < 1e-8


def test_mean_attack_near_one():
    model = fit_dixon_coles(_toy_matches())
    mean_att = sum(model.attack.values()) / len(model.attack)
    assert abs(mean_att - 1.0) < 1e-5


def _two_season_matches() -> pd.DataFrame:
    """A long-history club and a new club with the same short, low-scoring run."""
    rows = []
    start = pd.Timestamp("2023-08-01")
    regulars = ["Alpha", "Beta", "Gamma"]
    for i in range(36):
        home = regulars[i % 3]
        away = regulars[(i + 1) % 3]
        rows.append(
            {
                "season_code": "2324",
                "date": start + pd.Timedelta(days=i * 4),
                "home_team": home,
                "away_team": away,
                "home_goals": 2,
                "away_goals": 1,
            }
        )
    later = pd.Timestamp("2024-08-10")
    # NewClub and the regulars. NewClub has not been seen before and scores nothing.
    for i, opponent in enumerate(regulars):
        rows.append(
            {
                "season_code": "2425",
                "date": later + pd.Timedelta(days=i * 7),
                "home_team": "NewClub",
                "away_team": opponent,
                "home_goals": 0,
                "away_goals": 3,
            }
        )
        rows.append(
            {
                "season_code": "2425",
                "date": later + pd.Timedelta(days=i * 7 + 1),
                "home_team": opponent,
                "away_team": "NewClub",
                "home_goals": 3,
                "away_goals": 0,
            }
        )
    return pd.DataFrame(rows)


def test_short_history_moves_more_than_a_long_one():
    frame = _two_season_matches()
    base = fit_dixon_coles(frame, prior_kappa=0.0)
    pulled = fit_dixon_coles(frame, prior_kappa=32.0, promoted_factor=0.75)

    def move(team: str) -> float:
        return abs(math.log(base.attack[team]) - math.log(pulled.attack[team]))

    assert move("NewClub") > move("Alpha")
    mean_att = sum(pulled.attack.values()) / len(pulled.attack)
    assert abs(mean_att - 1.0) < 1e-5


def test_new_club_starts_lower_when_the_fixed_factor_is_lower():
    frame = _two_season_matches()
    level = fit_dixon_coles(frame, prior_kappa=80.0, promoted_factor=1.0)
    lower = fit_dixon_coles(frame, prior_kappa=80.0, promoted_factor=0.75)
    assert lower.attack["NewClub"] < level.attack["NewClub"]
    assert "NewClub" in lower.attack


def test_unseen_club_uses_the_new_club_start_only_when_the_prior_is_on():
    from src.models.dixon_coles import expected_goals

    frame = _toy_matches()
    off = fit_dixon_coles(frame, prior_kappa=0.0)
    on = fit_dixon_coles(frame, prior_kappa=20.0, promoted_factor=0.75)
    assert off.prior_kappa == 0.0
    assert off.new_club_attack == 1.0
    assert on.prior_kappa == 20.0
    assert on.new_club_attack < 1.0
    # The new-club defence sits above the fitted clubs' typical defence,
    # and the new-club attack sits below an average attack.
    typical_defence = math.exp(
        sum(math.log(rate) for rate in on.defence.values()) / len(on.defence)
    )
    assert on.new_club_defence > typical_defence
    lam_off, _ = expected_goals(off, "Nobody", "Alpha")
    lam_on, _ = expected_goals(on, "Nobody", "Alpha")
    assert lam_off == pytest.approx(off.defence["Alpha"] * off.home_advantage)
    assert lam_on == pytest.approx(
        on.new_club_attack * on.defence["Alpha"] * on.home_advantage
    )


def test_optimizer_success_false_falls_back_to_poisson_init_rho_zero(monkeypatch):
    """When minimize reports success=False, return Poisson-init strengths with ρ=0."""
    from types import SimpleNamespace

    import src.models.dixon_coles as dc

    def _fake_minimize(objective, x0, method=None, bounds=None, options=None):
        bad = np.asarray(x0, dtype=float).copy()
        # Finite but non-success — the honest-failure path, not an exception.
        return SimpleNamespace(
            success=False,
            x=bad,
            fun=float(objective(bad)),
            message="forced failure for unit test",
        )

    monkeypatch.setattr(dc, "minimize", _fake_minimize)
    model = fit_dixon_coles(_toy_matches())
    assert model.rho == 0.0
    assert model.attack
    assert model.defence
    assert abs(sum(model.attack.values()) / len(model.attack) - 1.0) < 1e-5
    # Fallback is the Poisson moment init, so rates stay positive and markets sum.
    probs = predict_match(model, "Alpha", "Beta")
    assert abs(probs["home"] + probs["draw"] + probs["away"] - 1.0) < 1e-8
