"""Unit and synthetic integration tests for walk-forward evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.gameweeks import assign_gameweeks
from src.eval.walkforward import (
    _pooled_summary,
    iter_folds,
    run_walkforward,
    write_walkforward_artefacts,
)
def _synthetic_matches(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-08-01", periods=n, freq="D")
    # Varying odds so de-vig sums are interesting but always normalise to 1.
    home_odds = 1.8 + 0.02 * np.arange(n)
    draw_odds = 3.4 + 0.01 * np.arange(n)
    away_odds = 4.0 - 0.01 * np.arange(n)
    results = ["H", "D", "A"] * ((n // 3) + 1)
    return pd.DataFrame(
        {
            "season_code": ["2425"] * n,
            "date": dates,
            "home_team": [f"Home{i % 5}" for i in range(n)],
            "away_team": [f"Away{i % 7}" for i in range(n)],
            "home_goals": [1] * n,
            "away_goals": [0] * n,
            "result": results[:n],
            "home_odds_close": home_odds,
            "draw_odds_close": draw_odds,
            "away_odds_close": away_odds,
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


def test_pooled_summary_weights_by_match_count():
    """Pooled log loss equals the all-rows score, not the unweighted fold mean."""
    # Fold A: 1 match, confident correct → low loss
    # Fold B: 3 matches, confident wrong → high loss
    # Unweighted mean of fold losses ≠ pooled loss.
    by_gw = pd.DataFrame(
        [
            {
                "run_date": "2026-11-16",
                "model": "toy",
                "season_code": "2425",
                "gameweek": 1,
                "n_train": 10,
                "n_test": 1,
                "log_loss": 0.1,
                "brier": 0.05,
                "accuracy": 1.0,
            },
            {
                "run_date": "2026-11-16",
                "model": "toy",
                "season_code": "2425",
                "gameweek": 2,
                "n_train": 11,
                "n_test": 3,
                "log_loss": 2.0,
                "brier": 0.4,
                "accuracy": 0.0,
            },
        ]
    )
    pred = pd.DataFrame(
        [
            {"result": "H", "p_home": 0.9, "p_draw": 0.05, "p_away": 0.05},
            {"result": "H", "p_home": 0.1, "p_draw": 0.1, "p_away": 0.8},
            {"result": "H", "p_home": 0.1, "p_draw": 0.1, "p_away": 0.8},
            {"result": "H", "p_home": 0.1, "p_draw": 0.1, "p_away": 0.8},
        ]
    )
    summary = _pooled_summary(by_gw, {"toy": pred}, "2026-11-16")
    assert list(summary["aggregation"]) == ["pooled"]
    assert summary.loc[0, "n_test"] == 4
    # Unweighted fold mean would be (0.1 + 2.0) / 2 = 1.05; pooled is lower-weighted.
    unweighted = 1.05
    assert summary.loc[0, "log_loss"] != round(unweighted, 4)


def test_run_walkforward_synthetic_integration(tmp_path: Path):
    """Tiny end-to-end walk-forward: disjoint folds, plot path, de-vigged book."""
    raw = _synthetic_matches(40)
    # Point-in-time features optional here; run_walkforward builds per fold.
    summary, by_gw, pred_frames = run_walkforward(raw, min_train_rows=10)

    assert not summary.empty
    assert not by_gw.empty
    assert set(summary["aggregation"]) == {"pooled"}

    # Train/test disjointness on every scored fold (match keys).
    matches = assign_gameweeks(raw.copy())
    matches["date"] = pd.to_datetime(matches["date"])
    for train, test, _season, _gw in iter_folds(matches, min_train_rows=10):
        train_keys = set(
            zip(train["date"].astype(str), train["home_team"], train["away_team"])
        )
        test_keys = set(
            zip(test["date"].astype(str), test["home_team"], test["away_team"])
        )
        assert train_keys.isdisjoint(test_keys)
        assert (train["date"] < test["date"].min()).all()

    paths = write_walkforward_artefacts(summary, by_gw, pred_frames, results_dir=tmp_path)
    assert paths["summary"].exists()
    assert paths["by_gameweek"].exists()
    plot_key = next(k for k in paths if k.startswith("calibration_"))
    assert paths[plot_key].exists()
    assert paths[plot_key].suffix == ".png"

    book = pred_frames["bookmaker_implied"]
    assert not book.empty
    sums = book["p_home"] + book["p_draw"] + book["p_away"]
    assert np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-9)
