import math

import numpy as np
import pandas as pd

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
