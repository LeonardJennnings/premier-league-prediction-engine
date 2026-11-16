"""Assign synthetic gameweek ids for walk-forward folds."""

from __future__ import annotations

import pandas as pd


def assign_gameweeks(matches: pd.DataFrame, matches_per_gw: int = 10) -> pd.DataFrame:
    """Return a copy with a 1-based gameweek column within each season_code.

    Matches are sorted by date within each season, then chunked into rounds of
    at most ``matches_per_gw`` rows.
    """
    if matches_per_gw < 1:
        raise ValueError("matches_per_gw must be >= 1")

    out = matches.copy()
    out["date"] = pd.to_datetime(out["date"])

    sort_cols = ["season_code", "date"]
    for col in ("home_team", "away_team"):
        if col in out.columns:
            sort_cols.append(col)
    out = out.sort_values(sort_cols).reset_index(drop=True)

    weeks: list[int] = []
    for _, season_df in out.groupby("season_code", sort=False):
        n = len(season_df)
        gw = [i // matches_per_gw + 1 for i in range(n)]
        weeks.extend(gw)

    out["gameweek"] = weeks
    return out