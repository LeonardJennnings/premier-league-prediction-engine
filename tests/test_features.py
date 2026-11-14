"""Tests for feature helpers."""

import pandas as pd

from src.features import add_implied_odds


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