"""Build model-ready features from the cleaned matches table."""

import pandas as pd


def add_match_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple unique id per match."""
    out = df.copy()
    out["match_id"] = (
        out["season_code"].astype(str)
        + "_"
        + out["date"].astype(str)
        + "_"
        + out["home_team"].astype(str)
        + "_"
        + out["away_team"].astype(str)
    )
    return out


def add_implied_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Convert closing decimal odds to normalised implied probabilities."""
    out = df.copy()
    needed = ["home_odds_close", "draw_odds_close", "away_odds_close"]
    if not all(c in out.columns for c in needed):
        return out

    raw_h = 1 / out["home_odds_close"]
    raw_d = 1 / out["draw_odds_close"]
    raw_a = 1 / out["away_odds_close"]
    total = raw_h + raw_d + raw_a

    out["implied_home"] = raw_h / total
    out["implied_draw"] = raw_d / total
    out["implied_away"] = raw_a / total
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all Phase 5 feature steps."""
    out = add_match_id(df)
    out = add_implied_odds(out)
    return out