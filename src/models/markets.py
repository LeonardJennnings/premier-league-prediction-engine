"""Derive match markets by summing cells of a scoreline probability grid."""

from __future__ import annotations

import math

import numpy as np

# Lines stored on a published row. 2.5 stays the public over/under column.
TOTAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)


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


def _line_key(line: float) -> str:
    return str(line).replace(".", "")


def markets_from_score_table(table: dict) -> dict[str, float]:
    """Sum a stored score table.

    Home, draw, away, and both-teams-to-score sum the 0..max cells only.
    The ``other`` bucket is any score outside that grid, so it cannot be split
    into a result or into both-teams-to-score. Every such score has more goals
    than the highest stored line, so it is added to each over.
    """
    cells = table["cells"]
    other = float(table.get("other") or 0.0)
    n = len(cells)
    home = draw = away = 0.0
    btts_yes = btts_no = 0.0
    over = {line: 0.0 for line in TOTAL_LINES}
    under = {line: 0.0 for line in TOTAL_LINES}
    for i in range(n):
        row = cells[i]
        if len(row) != n:
            raise ValueError("score table cells must be square")
        for j in range(n):
            p = float(row[j])
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
            for line in TOTAL_LINES:
                if i + j > line:
                    over[line] += p
                else:
                    under[line] += p
    out = {
        "home": home,
        "draw": draw,
        "away": away,
        "btts_yes": btts_yes,
        "btts_no": btts_no,
    }
    for line in TOTAL_LINES:
        key = _line_key(line)
        out[f"over_{key}"] = over[line] + other
        out[f"under_{key}"] = under[line]
    return out


def ranked_scores(table: dict) -> list[dict]:
    """Exact scores inside the grid, highest probability first."""
    ranked: list[dict] = []
    for i, row in enumerate(table["cells"]):
        for j, probability in enumerate(row):
            ranked.append(
                {"home": i, "away": j, "probability": float(probability)}
            )
    ranked.sort(key=lambda item: (-item["probability"], item["home"], item["away"]))
    return ranked


def scoreline_probability(table: dict, home_goals: int, away_goals: int) -> float:
    """Probability of one final score. Scores outside the grid use ``other``."""
    if home_goals < 0 or away_goals < 0:
        raise ValueError("goals must be non-negative")
    max_goals = int(table["max_goals"])
    if home_goals <= max_goals and away_goals <= max_goals:
        return float(table["cells"][home_goals][away_goals])
    return float(table["other"])


def scoreline_log_loss(table: dict, home_goals: int, away_goals: int) -> float:
    """Natural-log score of one final score. Infinite only when the probability is 0."""
    probability = scoreline_probability(table, home_goals, away_goals)
    if probability <= 0.0:
        return math.inf
    return -math.log(probability)
