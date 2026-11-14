"""Tests for the cleaning functions."""

import pandas as pd

from src.clean import clean_frame


def test_clean_frame_parses_date_and_keeps_result():
    raw = pd.DataFrame(
        [
            {
                "Date": "15/08/2025",
                "HomeTeam": "Liverpool",
                "AwayTeam": "Bournemouth",
                "FTHG": 4,
                "FTAG": 2,
                "FTR": "H",
                "B365CH": 1.5,
                "B365CD": 4.0,
                "B365CA": 6.0,
            }
        ]
    )
    out = clean_frame(raw, "2526")

    assert len(out) == 1
    assert out.loc[0, "season_code"] == "2526"
    assert str(out.loc[0, "date"].date()) == "2025-08-15"
    assert out.loc[0, "home_team"] == "Liverpool"
    assert out.loc[0, "home_goals"] == 4
    assert out.loc[0, "result"] == "H"


def test_clean_frame_drops_blank_row():
    raw = pd.DataFrame(
        [
            {
                "Date": "15/08/2025",
                "HomeTeam": "Liverpool",
                "AwayTeam": "Bournemouth",
                "FTHG": 4,
                "FTAG": 2,
                "FTR": "H",
            },
            {
                "Date": None,
                "HomeTeam": None,
                "AwayTeam": None,
                "FTHG": None,
                "FTAG": None,
                "FTR": None,
            },
        ]
    )
    out = clean_frame(raw, "2526")
    assert len(out) == 1


def test_clean_frame_keeps_nottingham_name():
    raw = pd.DataFrame(
        [
            {
                "Date": "01/09/2025",
                "HomeTeam": "Nott'm Forest",
                "AwayTeam": "West Ham",
                "FTHG": 1,
                "FTAG": 1,
                "FTR": "D",
            }
        ]
    )
    out = clean_frame(raw, "2526")
    assert out.loc[0, "home_team"] == "Nott'm Forest"