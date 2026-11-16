import numpy as np

from src.models.markets import markets_from_grid, normalize_grid


def test_normalize_grid_sums_to_one():
    grid = np.ones((3, 3), dtype=float)
    out = normalize_grid(grid)
    assert abs(out.sum() - 1.0) < 1e-12


def test_markets_from_hand_grid():
    # 2x2 toy: mass on 1-0 and 0-1 only after normalize
    grid = np.zeros((2, 2), dtype=float)
    grid[1, 0] = 0.6  # home win 1-0
    grid[0, 1] = 0.4  # away win 0-1
    m = markets_from_grid(grid)
    assert abs(m["home"] - 0.6) < 1e-12
    assert abs(m["away"] - 0.4) < 1e-12
    assert abs(m["draw"] - 0.0) < 1e-12
    assert abs(m["btts_yes"] - 0.0) < 1e-12
    assert abs(m["btts_no"] - 1.0) < 1e-12
    assert abs(m["under_25"] - 1.0) < 1e-12
    assert abs(m["over_25"] - 0.0) < 1e-12


def test_btts_and_over_on_larger_grid():
    grid = np.zeros((4, 4), dtype=float)
    grid[2, 2] = 1.0  # 2-2: BTTS yes, over 2.5
    m = markets_from_grid(grid)
    assert abs(m["btts_yes"] - 1.0) < 1e-12
    assert abs(m["over_25"] - 1.0) < 1e-12
    assert abs(m["draw"] - 1.0) < 1e-12
