import math

import numpy as np

from src.models.dixon_coles import scoreline_grid_from_rates, tau
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
