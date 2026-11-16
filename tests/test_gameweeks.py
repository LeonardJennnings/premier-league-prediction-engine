import pandas as pd

from src.eval.gameweeks import assign_gameweeks


def test_assign_gameweeks_chunks_of_ten_within_season():
    dates = pd.date_range("2024-08-01", periods=25, freq="D")
    matches = pd.DataFrame(
        {
            "season_code": ["2425"] * 25,
            "date": dates,
            "home_team": [f"H{i}" for i in range(25)],
            "away_team": [f"A{i}" for i in range(25)],
        }
    )
    out = assign_gameweeks(matches, matches_per_gw=10)
    assert list(out["gameweek"][:10]) == [1] * 10
    assert list(out["gameweek"][10:20]) == [2] * 10
    assert list(out["gameweek"][20:]) == [3] * 5


def test_assign_gameweeks_restarts_each_season():
    matches = pd.DataFrame(
        {
            "season_code": ["2324"] * 3 + ["2425"] * 3,
            "date": pd.to_datetime(
                [
                    "2023-08-01",
                    "2023-08-02",
                    "2023-08-03",
                    "2024-08-01",
                    "2024-08-02",
                    "2024-08-03",
                ]
            ),
        }
    )
    out = assign_gameweeks(matches, matches_per_gw=2)
    assert out.loc[out["season_code"] == "2324", "gameweek"].tolist() == [1, 1, 2]
    assert out.loc[out["season_code"] == "2425", "gameweek"].tolist() == [1, 1, 2]
    