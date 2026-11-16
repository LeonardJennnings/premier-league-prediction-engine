import pandas as pd

from src.eval.gameweeks import assign_gameweeks
from src.eval.walkforward import iter_folds


def _synthetic_matches(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-08-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "season_code": ["2425"] * n,
            "date": dates,
            "home_team": ["Arsenal"] * n,
            "away_team": ["Chelsea"] * n,
            "home_goals": [1] * n,
            "away_goals": [0] * n,
            "result": ["H"] * n,
            "implied_home": [0.5] * n,
            "implied_draw": [0.25] * n,
            "implied_away": [0.25] * n,
        }
    )


def test_iter_folds_train_strictly_before_test():
    matches = assign_gameweeks(_synthetic_matches(25), matches_per_gw=5)
    folds = list(iter_folds(matches, min_train_rows=0))
    assert len(folds) == 5
    for train, test, season, gw in folds:
        assert season == "2425"
        assert (train["date"] < test["date"].min()).all()
        assert set(test["gameweek"]) == {gw}


def test_iter_folds_skips_until_min_train():
    matches = assign_gameweeks(_synthetic_matches(25), matches_per_gw=5)
    folds = list(iter_folds(matches, min_train_rows=12))
    assert all(len(train) >= 12 for train, *_ in folds)
    assert len(folds) == 2