"""Clean and tidy the raw match tables."""

# Cleaned match schema (target columns):
# season_code      - e.g. "2526"
# date             - real date (not a string)
# home_team        - home team name
# away_team        - away team name
# home_goals       - full-time home goals (int)
# away_goals       - full-time away goals (int)
# result           - "H", "D", or "A"
# home_odds_close  - closing decimal odds for home win (if present)
# draw_odds_close  - closing decimal odds for draw (if present)
# away_odds_close  - closing decimal odds for away win (if present)

from pathlib import Path
from src.config import DATA_PROCESSED, SEASON_CODES, raw_path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    """Read one raw football-data.co.uk CSV."""
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_frame(df: pd.DataFrame, season_code: str) -> pd.DataFrame:
    """Turn one raw season table into the cleaned schema."""
    # Prefer Bet365 closing odds; skip them if a season doesn't have them.
    odds_map = {
        "B365CH": "home_odds_close",
        "B365CD": "draw_odds_close",
        "B365CA": "away_odds_close",
    }
    keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    keep += [c for c in odds_map if c in df.columns]

    out = df.loc[:, keep].copy()
    out = out.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "FTR": "result",
            **{raw: clean for raw, clean in odds_map.items() if raw in out.columns},
        }
    )

    out["date"] = pd.to_datetime(out["date"], dayfirst=True, format="mixed")
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals", "result"])
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    out["season_code"] = season_code

    cols = [
        "season_code",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "home_odds_close",
        "draw_odds_close",
        "away_odds_close",
    ]
    cols = [c for c in cols if c in out.columns]
    return out.loc[:, cols].sort_values("date").reset_index(drop=True)


def build_matches_table(season_codes=None) -> pd.DataFrame:
    """Load and clean each season, then stack into one table."""
    if season_codes is None:
        season_codes = SEASON_CODES
    frames = []
    for code in season_codes:
        path = raw_path(code)
        raw = load_raw(path)
        frames.append(clean_frame(raw, code))
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def save_matches(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the cleaned matches table to data/processed/matches.csv."""
    if path is None:
        path = DATA_PROCESSED / "matches.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

