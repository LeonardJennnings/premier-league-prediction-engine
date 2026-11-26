"""Tests for feature helpers."""

import pandas as pd
import pytest

from src.features import (
    add_features,
    add_history_dependent_features,
    add_implied_odds,
    add_rest_days,
    assert_history_before_cutoff,
    features_for_fold,
)


def test_implied_odds_normalise_and_favourite():
    df = pd.DataFrame(
        [
            {
                "home_odds_close": 2.0,
                "draw_odds_close": 4.0,
                "away_odds_close": 4.0,
            }
        ]
    )
    out = add_implied_odds(df)

    total = out.loc[0, "implied_home"] + out.loc[0, "implied_draw"] + out.loc[0, "implied_away"]
    assert abs(total - 1.0) < 1e-9
    # Lowest odds (2.0) should be the highest probability
    assert out.loc[0, "implied_home"] > out.loc[0, "implied_draw"]
    assert out.loc[0, "implied_home"] > out.loc[0, "implied_away"]


def test_implied_odds_missing_columns_leaves_frame():
    df = pd.DataFrame([{"home_team": "Liverpool"}])
    out = add_implied_odds(df)
    assert "implied_home" not in out.columns
    assert list(out.columns) == ["home_team"]


def test_implied_opening_odds_normalise():
    df = pd.DataFrame(
        [
            {
                "home_odds_open": 2.0,
                "draw_odds_open": 4.0,
                "away_odds_open": 4.0,
                "home_odds_close": 1.8,
                "draw_odds_close": 3.5,
                "away_odds_close": 4.5,
            }
        ]
    )
    out = add_implied_odds(df)

    total_open = (
        out.loc[0, "implied_home_open"]
        + out.loc[0, "implied_draw_open"]
        + out.loc[0, "implied_away_open"]
    )
    total_close = (
        out.loc[0, "implied_home"]
        + out.loc[0, "implied_draw"]
        + out.loc[0, "implied_away"]
    )
    assert abs(total_open - 1.0) < 1e-9
    assert abs(total_close - 1.0) < 1e-9
    assert out.loc[0, "implied_home_open"] != out.loc[0, "implied_home"]


def test_implied_over_under_25_devig_when_both_present():
    df = pd.DataFrame(
        [
            {
                "over_25_odds_close": 1.80,
                "under_25_odds_close": 2.10,
            }
        ]
    )
    out = add_implied_odds(df)

    total = out.loc[0, "implied_over_25"] + out.loc[0, "implied_under_25"]
    assert abs(total - 1.0) < 1e-9
    # Lower over odds should map to the higher de-vig probability
    assert out.loc[0, "implied_over_25"] > out.loc[0, "implied_under_25"]
    raw_over = 1 / 1.80
    raw_under = 1 / 2.10
    expected_over = raw_over / (raw_over + raw_under)
    assert abs(out.loc[0, "implied_over_25"] - expected_over) < 1e-12


def test_implied_over_under_25_missing_columns_leaves_frame():
    df = pd.DataFrame([{"home_odds_close": 2.0, "draw_odds_close": 4.0, "away_odds_close": 4.0}])
    out = add_implied_odds(df)
    assert "implied_over_25" not in out.columns
    assert "implied_under_25" not in out.columns
    assert "implied_home" in out.columns


def test_implied_over_under_25_requires_both_columns():
    df = pd.DataFrame([{"over_25_odds_close": 1.80}])
    out = add_implied_odds(df)
    assert "implied_over_25" not in out.columns
    assert "implied_under_25" not in out.columns
    assert list(out.columns) == ["over_25_odds_close"]


def test_assert_history_before_cutoff_passes_on_past_only():
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-10"]),
            "home_team": ["A", "B"],
            "away_team": ["C", "D"],
        }
    )
    assert_history_before_cutoff(history, "2024-08-15")


def test_assert_history_before_cutoff_catches_future_looking_join():
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-20"]),
            "home_team": ["Arsenal", "Future FC"],
            "away_team": ["Chelsea", "Leak United"],
        }
    )
    with pytest.raises(ValueError, match="on or after cutoff"):
        assert_history_before_cutoff(history, "2024-08-15")


def test_add_history_dependent_features_rejects_leaky_history():
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Home"],
            "away_team": ["Away"],
            "result": ["H"],
        }
    )
    leaky = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-16"]),
            "home_team": ["Past", "Future"],
            "away_team": ["X", "Y"],
        }
    )
    with pytest.raises(ValueError, match="on or after cutoff"):
        add_history_dependent_features(targets, leaky, cutoff="2024-08-15")


def test_features_for_fold_point_in_time_and_cutoff():
    train = pd.DataFrame(
        {
            "season_code": ["2425", "2425"],
            "date": pd.to_datetime(["2024-08-01", "2024-08-08"]),
            "home_team": ["Arsenal", "Liverpool"],
            "away_team": ["Chelsea", "Everton"],
            "result": ["H", "D"],
            "home_odds_close": [2.0, 2.5],
            "draw_odds_close": [3.5, 3.4],
            "away_odds_close": [3.5, 2.8],
        }
    )
    test = pd.DataFrame(
        {
            "season_code": ["2425"],
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Spurs"],
            "away_team": ["Wolves"],
            "result": ["A"],
            "home_odds_close": [1.8],
            "draw_odds_close": [3.6],
            "away_odds_close": [4.5],
        }
    )
    train_f, test_f = features_for_fold(train, test)
    assert "match_id" in train_f.columns and "match_id" in test_f.columns
    assert "implied_home" in test_f.columns
    total = (
        test_f.loc[0, "implied_home"]
        + test_f.loc[0, "implied_draw"]
        + test_f.loc[0, "implied_away"]
    )
    assert abs(total - 1.0) < 1e-9


def test_add_features_with_explicit_history_filters_to_cutoff():
    rows = pd.DataFrame(
        {
            "season_code": ["2425"] * 3,
            "date": pd.to_datetime(["2024-08-01", "2024-08-10", "2024-08-20"]),
            "home_team": ["A", "B", "C"],
            "away_team": ["X", "Y", "Z"],
            "result": ["H", "D", "A"],
            "home_odds_close": [2.0, 2.0, 2.0],
            "draw_odds_close": [3.5, 3.5, 3.5],
            "away_odds_close": [3.5, 3.5, 3.5],
        }
    )
    # Targets are the last row; history is the full table — add_features must
    # slice history to date < cutoff before the assert runs.
    targets = rows.iloc[[2]].copy()
    out = add_features(targets, history=rows, cutoff="2024-08-20")
    assert len(out) == 1
    assert abs(out.loc[out.index[0], "implied_home"] + out.loc[out.index[0], "implied_draw"]
               + out.loc[out.index[0], "implied_away"] - 1.0) < 1e-9


def test_later_match_does_not_change_earlier_rest_days():
    frame = pd.DataFrame(
        {
            "date": ["2024-08-01", "2024-08-08", "2024-08-11"],
            "home_team": ["Arsenal", "Chelsea", "Arsenal"],
            "away_team": ["Chelsea", "Liverpool", "Liverpool"],
        }
    )
    before = add_rest_days(frame)
    later = pd.concat(
        [
            frame,
            pd.DataFrame(
                {
                    "date": ["2024-08-20"],
                    "home_team": ["Arsenal"],
                    "away_team": ["Spurs"],
                }
            ),
        ],
        ignore_index=True,
    )
    after = add_rest_days(later)
    assert before["home_rest_days"].equals(after["home_rest_days"].iloc[:3].reset_index(drop=True))
    assert before["away_rest_days"].equals(after["away_rest_days"].iloc[:3].reset_index(drop=True))
    assert pd.isna(after.loc[0, "home_rest_days"])
    assert after.loc[2, "home_rest_days"] == 10
    assert after.loc[3, "home_rest_days"] == 9


def test_same_date_matches_are_not_each_others_previous_match():
    frame = pd.DataFrame(
        {
            "date": ["2024-08-10", "2024-08-10", "2024-08-14"],
            "home_team": ["Arsenal", "Liverpool", "Arsenal"],
            "away_team": ["Chelsea", "Arsenal", "Spurs"],
        }
    )
    out = add_rest_days(frame)
    assert pd.isna(out.loc[0, "home_rest_days"])
    assert pd.isna(out.loc[1, "away_rest_days"])
    assert out.loc[2, "home_rest_days"] == 4
