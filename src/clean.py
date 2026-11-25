"""Clean and tidy the raw match tables."""

# Cleaned match schema (target columns):
# season_code      - e.g. "2526"
# date             - real date (not a string)
# home_team        - home team name
# away_team        - away team name
# home_goals       - full-time home goals (int)
# away_goals       - full-time away goals (int)
# result           - "H", "D", or "A"
# home_odds_open   - Bet365 pre-closing (opening) decimal odds for home win
# draw_odds_open   - Bet365 pre-closing draw odds
# away_odds_open   - Bet365 pre-closing away odds
# home_odds_close  - closing decimal odds for home win (if present)
# draw_odds_close  - closing decimal odds for draw (if present)
# away_odds_close  - closing decimal odds for away win (if present)
# over_25_odds_open   - pre-closing over 2.5 (Bet365, else market average)
# under_25_odds_open  - pre-closing under 2.5
# over_25_odds_close  - closing over 2.5 (Bet365 C, else the market-average close)
# under_25_odds_close - closing under 2.5
# A pre-close price is never copied into a closing column.

from pathlib import Path
from src.config import DATA_PROCESSED, SEASON_CODES, raw_path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    """Read one raw football-data.co.uk CSV."""
    return pd.read_csv(path, encoding="utf-8-sig")


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name that exists on df, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _copy_odds(df: pd.DataFrame, out: pd.DataFrame, target: str, candidates: list[str]) -> None:
    """Copy the first present odds column onto ``out[target]`` as numeric."""
    raw = _first_present(df, candidates)
    if raw is None:
        return
    out[target] = pd.to_numeric(df[raw], errors="coerce")


def clean_frame(df: pd.DataFrame, season_code: str) -> pd.DataFrame:
    """Turn one raw season table into the cleaned schema."""
    keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"raw frame missing required columns: {missing}")

    out = df.loc[:, keep].copy()
    out = out.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "FTR": "result",
        }
    )

    # football-data.co.uk: columns without C are pre-closing; *C* are closing.
    _copy_odds(df, out, "home_odds_open", ["B365H"])
    _copy_odds(df, out, "draw_odds_open", ["B365D"])
    _copy_odds(df, out, "away_odds_open", ["B365A"])
    _copy_odds(df, out, "home_odds_close", ["B365CH"])
    _copy_odds(df, out, "draw_odds_close", ["B365CD"])
    _copy_odds(df, out, "away_odds_close", ["B365CA"])

    # Opening totals use the non-C series. Closing totals use the C series
    # only. An earlier price is not a substitute when the close is missing.
    _copy_odds(df, out, "over_25_odds_open", ["B365>2.5", "Avg>2.5"])
    _copy_odds(df, out, "under_25_odds_open", ["B365<2.5", "Avg<2.5"])
    _copy_odds(df, out, "over_25_odds_close", ["B365C>2.5", "AvgC>2.5"])
    _copy_odds(df, out, "under_25_odds_close", ["B365C<2.5", "AvgC<2.5"])

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
        "home_odds_open",
        "draw_odds_open",
        "away_odds_open",
        "home_odds_close",
        "draw_odds_close",
        "away_odds_close",
        "over_25_odds_open",
        "under_25_odds_open",
        "over_25_odds_close",
        "under_25_odds_close",
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
