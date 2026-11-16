import math

import pandas as pd

from src.config import ELO_HOME_BONUS, ELO_INITIAL, ELO_K
from src.models.elo import EloModel, fit_elo, predict_match


def test_elo_probs_sum_to_one():
    matches = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "home_team": ["A", "B"] * 5,
            "away_team": ["B", "A"] * 5,
            "result": ["H", "A", "H", "D", "H", "A", "H", "H", "D", "H"],
        }
    )
    model = fit_elo(matches)
    p = predict_match(model, "A", "B")
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9


def test_elo_winner_gains_rating():
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "result": ["H", "H"],
        }
    )
    model = fit_elo(matches)
    assert model.ratings["A"] > model.ratings["B"]


def test_fit_uses_config_defaults():
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "result": ["D"],
        }
    )
    model = fit_elo(matches)
    assert model.k == ELO_K
    assert model.home_bonus == ELO_HOME_BONUS


def test_single_home_win_updates_ratings():
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "result": ["H"],
        }
    )
    model = fit_elo(matches, k=20.0, home_bonus=60.0, initial=1500.0)
    expected_home = 1 / (1 + 10 ** (-(1500.0 + 60.0 - 1500.0) / 400.0))
    assert abs(model.ratings["A"] - (1500.0 + 20.0 * (1.0 - expected_home))) < 1e-9
    assert abs(model.ratings["B"] - (1500.0 + 20.0 * (0.0 - (1.0 - expected_home)))) < 1e-9


def test_elo_updates_in_date_order():
    chronological = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "result": ["A", "H"],
        }
    )
    reversed_dates = chronological.iloc[::-1].reset_index(drop=True)
    model_sorted = fit_elo(chronological)
    model_reversed = fit_elo(reversed_dates)
    assert model_sorted.ratings["A"] == model_reversed.ratings["A"]
    assert model_sorted.ratings["B"] == model_reversed.ratings["B"]


def test_1x2_mapping_matches_locked_formula():
    model = EloModel(ratings={"A": 1500.0, "B": 1500.0}, home_bonus=60.0, k=20.0)
    p = predict_match(model, "A", "B")
    r_home, r_away = 1500.0, 1500.0
    e = 1 / (1 + 10 ** (-(r_home + 60.0 - r_away) / 400.0))
    gap = abs((r_home + 60.0) - r_away)
    p_draw = 0.28 * math.exp(-gap / 400.0)
    assert abs(p["home"] - e * (1 - p_draw)) < 1e-12
    assert abs(p["draw"] - p_draw) < 1e-12
    assert abs(p["away"] - (1 - e) * (1 - p_draw)) < 1e-12


def test_unknown_team_starts_at_initial():
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "result": ["H"],
        }
    )
    model = fit_elo(matches)
    p = predict_match(model, "X", "Y")
    e = 1 / (1 + 10 ** (-(ELO_INITIAL + model.home_bonus - ELO_INITIAL) / 400.0))
    gap = abs((ELO_INITIAL + model.home_bonus) - ELO_INITIAL)
    p_draw = 0.28 * math.exp(-gap / 400.0)
    assert abs(p["home"] - e * (1 - p_draw)) < 1e-12
    assert abs(p["draw"] - p_draw) < 1e-12
    assert abs(p["away"] - (1 - e) * (1 - p_draw)) < 1e-12
    assert "X" not in model.ratings
    assert "Y" not in model.ratings
