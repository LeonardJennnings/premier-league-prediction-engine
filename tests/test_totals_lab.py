"""Unit tests for lab-only O/U 2.5 specialised comparators."""

import math

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import DixonColesModel, expected_goals, scoreline_grid
from src.models.markets import markets_from_grid
from src.models.totals_lab import (
    apply_binary_temperature,
    fit_binary_temperature,
    fit_totals_logistic,
    over_25_from_total_rate,
    predict_poisson_total_over_25,
    predict_totals_logistic_over_25,
)


def _neutral_dc() -> DixonColesModel:
    return DixonColesModel(
        attack={"Alpha": 1.0, "Beta": 1.0},
        defence={"Alpha": 1.0, "Beta": 1.0},
        home_advantage=1.2,
        rho=0.0,
        xi=0.0019,
    )


def test_over_25_from_total_rate_closed_form():
    lam = 2.5
    expected = 1.0 - math.exp(-lam) * (1.0 + lam + 0.5 * lam * lam)
    assert abs(over_25_from_total_rate(lam) - expected) < 1e-12


def test_over_25_rejects_negative_rate():
    with pytest.raises(ValueError):
        over_25_from_total_rate(-0.1)


def test_poisson_total_matches_sum_of_independent_rates():
    model = _neutral_dc()
    lam, mu = expected_goals(model, "Alpha", "Beta")
    assert abs(
        predict_poisson_total_over_25(model, "Alpha", "Beta")
        - over_25_from_total_rate(lam + mu)
    ) < 1e-12


def test_poisson_total_ignores_rho():
    """Lab totals depend only on λ+μ — ρ is intentionally unused."""
    base = dict(
        attack={"Alpha": 1.2, "Beta": 0.85},
        defence={"Alpha": 0.9, "Beta": 1.1},
        home_advantage=1.35,
        xi=0.0019,
    )
    m0 = DixonColesModel(**base, rho=0.0)
    m_rho = DixonColesModel(**base, rho=-0.18)
    assert abs(
        predict_poisson_total_over_25(m0, "Alpha", "Beta")
        - predict_poisson_total_over_25(m_rho, "Alpha", "Beta")
    ) < 1e-12


def test_dc_rho_preserves_over_under_mass():
    """Dixon–Coles τ reshuffles 0–0/1–0/0–1/1–1 but leaves O/U 2.5 mass unchanged.

    So poisson_total and grid O/U agree up to finite-grid truncation — the
    specialised logistic (not ρ) is the line that can actually diverge.
    """
    from src.models.dixon_coles import scoreline_grid_from_rates

    lam, mu = 2.4, 1.1
    g0 = markets_from_grid(scoreline_grid_from_rates(lam, mu, 0.0, max_goals=15))
    g_rho = markets_from_grid(scoreline_grid_from_rates(lam, mu, -0.15, max_goals=15))
    assert abs(g0["over_25"] - g_rho["over_25"]) < 1e-12
    lab = over_25_from_total_rate(lam + mu)
    assert abs(g0["over_25"] - lab) < 1e-6


def test_poisson_total_close_to_grid_when_rho_zero():
    """With ρ=0, grid O/U ≈ independent Poisson total (finite-grid truncation only)."""
    model = _neutral_dc()
    grid_over = markets_from_grid(scoreline_grid(model, "Alpha", "Beta", max_goals=12))[
        "over_25"
    ]
    lab_over = predict_poisson_total_over_25(model, "Alpha", "Beta")
    assert abs(grid_over - lab_over) < 1e-3


def test_binary_temperature_identity_at_one():
    p = np.array([0.2, 0.5, 0.8])
    out = apply_binary_temperature(p, 1.0)
    assert np.allclose(out, p, atol=1e-12)


def test_fit_binary_temperature_pulls_toward_better_loss():
    # Confident wrong → temperature > 1 softens.
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.9, 0.85, 0.15, 0.2])
    t = fit_binary_temperature(y, p)
    assert t > 1.0


def test_totals_logistic_fit_and_predict_roundtrip():
    model = _neutral_dc()
    n = 80
    matches = pd.DataFrame(
        {
            "home_team": ["Alpha"] * n,
            "away_team": ["Beta"] * n,
            "home_goals": [2 if i % 2 == 0 else 0 for i in range(n)],
            "away_goals": [1 if i % 2 == 0 else 0 for i in range(n)],
        }
    )
    lab = fit_totals_logistic(matches, model)
    assert lab is not None
    p = predict_totals_logistic_over_25(lab, model, "Alpha", "Beta")
    assert 0.0 < p < 1.0


def test_totals_logistic_returns_none_when_too_thin():
    model = _neutral_dc()
    matches = pd.DataFrame(
        {
            "home_team": ["Alpha", "Alpha"],
            "away_team": ["Beta", "Beta"],
            "home_goals": [1, 0],
            "away_goals": [0, 1],
        }
    )
    assert fit_totals_logistic(matches, model) is None
