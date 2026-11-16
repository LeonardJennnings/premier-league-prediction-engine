"""Derive match markets by summing cells of a scoreline probability grid."""

from __future__ import annotations

import numpy as np


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=float)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("grid must be a square 2D array")
    total = g.sum()
    if total <= 0:
        raise ValueError("grid must have positive total mass")
    return g / total


def markets_from_grid(grid: np.ndarray) -> dict[str, float]:
    """Return 1X2, BTTS, and O/U 2.5 probabilities from a scoreline grid."""
    g = normalize_grid(grid)
    n = g.shape[0]
    home = draw = away = 0.0
    btts_yes = btts_no = 0.0
    over_25 = under_25 = 0.0
    for i in range(n):
        for j in range(n):
            p = float(g[i, j])
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
            if i >= 1 and j >= 1:
                btts_yes += p
            else:
                btts_no += p
            if i + j >= 3:
                over_25 += p
            else:
                under_25 += p
    return {
        "home": home,
        "draw": draw,
        "away": away,
        "btts_yes": btts_yes,
        "btts_no": btts_no,
        "over_25": over_25,
        "under_25": under_25,
    }
