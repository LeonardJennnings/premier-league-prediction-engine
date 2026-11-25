"""Tests for the Poisson goals model."""

import pandas as pd
import pytest

from src.models.poisson import fit_poisson, predict_match


def _toy_matches() -> pd.DataFrame:
    """Small made-up sample so tests don't need real CSVs."""
    return pd.DataFrame(
        [
            {"home_team": "Alpha", "away_team": "Beta", "home_goals": 2, "away_goals": 0},
            {"home_team": "Beta", "away_team": "Alpha", "home_goals": 1, "away_goals": 1},
            {"home_team": "Alpha", "away_team": "Gamma", "home_goals": 3, "away_goals": 1},
            {"home_team": "Gamma", "away_team": "Beta", "home_goals": 0, "away_goals": 2},
            {"home_team": "Beta", "away_team": "Gamma", "home_goals": 2, "away_goals": 2},
            {"home_team": "Gamma", "away_team": "Alpha", "home_goals": 0, "away_goals": 2},
        ]
    )


def test_predict_match_probs_sum_to_one():
    model = fit_poisson(_toy_matches())
    probs = predict_match(model, "Alpha", "Beta")

    assert set(probs) == {"home", "draw", "away"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_stronger_home_side_gets_higher_home_prob():
    model = fit_poisson(_toy_matches())
    # Alpha scored more in the toy data than Gamma did.
    alpha_home = predict_match(model, "Alpha", "Beta")["home"]
    gamma_home = predict_match(model, "Gamma", "Beta")["home"]
    assert alpha_home > gamma_home


def test_team_with_no_goals_still_has_a_positive_rate():
    from src.models.poisson import match_rates

    df = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0},
            {"home_team": "B", "away_team": "A", "home_goals": 1, "away_goals": 0},
            {"home_team": "A", "away_team": "C", "home_goals": 2, "away_goals": 0},
            {"home_team": "C", "away_team": "A", "home_goals": 0, "away_goals": 1},
        ]
    )
    model = fit_poisson(df)
    lam, mu = match_rates(model, "C", "A")
    assert lam > 0.0
    assert mu > 0.0


def test_average_team_rates_match_each_sides_mean():
    """An even schedule of 2-1 results predicts 2 and 1, not the blended average."""
    from src.models.poisson import match_rates

    teams = ["A", "B", "C"]
    rows = []
    for home in teams:
        for away in teams:
            if home == away:
                continue
            rows.append(
                {"home_team": home, "away_team": away, "home_goals": 2, "away_goals": 1}
            )
    model = fit_poisson(pd.DataFrame(rows))
    lam, mu = match_rates(model, "A", "B")
    assert lam == pytest.approx(2.0)
    assert mu == pytest.approx(1.0)
    # The old away rate used the blended one-side average, 1.5, which is not 1.
    assert mu != pytest.approx(1.5)
    