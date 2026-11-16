"""Tests for nested Dixon–Coles ξ walk-forward tune (no peeking)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.gameweeks import assign_gameweeks
from src.eval.xi_tune import (
    half_life_days,
    run_xi_tune,
    score_dixon_coles_folds,
    write_xi_tune_artefacts,
)


def _matches(n: int = 60) -> pd.DataFrame:
    """Synthetic multi-season matches with closing implied odds."""
    dates = pd.date_range("2023-08-01", periods=n, freq="D")
    results = [["H", "D", "A"][i % 3] for i in range(n)]
    goals = {"H": (2, 0), "D": (1, 1), "A": (0, 2)}
    # Split into older (tune) and recent (holdout) seasons by index.
    seasons = ["2223"] * (n // 3) + ["2324"] * (n // 3) + ["2526"] * (n - 2 * (n // 3))
    homes = ["Arsenal", "Chelsea", "Liverpool", "Spurs", "Everton"]
    aways = ["Chelsea", "Arsenal", "Spurs", "Liverpool", "Fulham"]
    implied_home = [0.45 + (i % 5) * 0.02 for i in range(n)]
    implied_draw = [0.28] * n
    implied_away = [1.0 - h - d for h, d in zip(implied_home, implied_draw)]
    return pd.DataFrame(
        {
            "season_code": seasons,
            "date": dates,
            "home_team": [homes[i % 5] for i in range(n)],
            "away_team": [aways[i % 5] for i in range(n)],
            "home_goals": [goals[r][0] for r in results],
            "away_goals": [goals[r][1] for r in results],
            "result": results,
            "implied_home": implied_home,
            "implied_draw": implied_draw,
            "implied_away": implied_away,
        }
    )


def test_half_life_days_one_year():
    assert half_life_days(0.0019) == pytest.approx(np.log(2) / 0.0019, rel=1e-6)
    assert half_life_days(0.0) == float("inf")


def test_score_folds_respects_season_filter_and_date_cutoff():
    matches = assign_gameweeks(_matches(45), matches_per_gw=5)
    dc_by_gw, book_by_gw, preds = score_dixon_coles_folds(
        matches, xi=0.0019, seasons={"2223", "2324"}, min_train_rows=5
    )
    assert not dc_by_gw.empty
    assert set(dc_by_gw["season_code"].astype(str)) <= {"2223", "2324"}
    assert "2526" not in set(dc_by_gw["season_code"].astype(str))
    assert not book_by_gw.empty
    assert len(preds) == int(dc_by_gw["n_test"].sum())


def test_run_xi_tune_selects_on_tune_only_and_reports_holdout(tmp_path: Path):
    matches = _matches(60)
    result = run_xi_tune(
        matches,
        candidates=(0.0010, 0.0019, 0.0050),
        tune_seasons=("2223", "2324"),
        holdout_seasons=("2526",),
        baseline_xi=0.0019,
        min_train_rows=5,
    )
    assert result.chosen_xi in (0.0010, 0.0019, 0.0050)
    assert set(result.candidate_tune["era"]) == {"tune"}
    assert "2526" not in ",".join(result.candidate_tune["seasons"].unique())
    assert set(result.holdout_comparison["era"]) == {"holdout"}
    lines = set(result.holdout_comparison["line"])
    assert "dixon_coles_chosen" in lines
    assert "bookmaker_implied" in lines
    # Selection used only tune rows — holdout DC LL must not appear in candidate grid.
    assert "holdout" not in set(result.candidate_tune["era"])

    paths = write_xi_tune_artefacts(result, results_dir=tmp_path)
    assert paths["candidates"].exists()
    assert paths["holdout"].exists()
    text = paths["readme"].read_text(encoding="utf-8")
    assert "tune" in text.lower()
    assert "holdout" in text.lower()
    assert "Frozen ξ" in text or "frozen" in text.lower()
    assert "book" in text.lower()
    assert str(result.chosen_xi) in text or f"{result.chosen_xi}" in text


def test_run_xi_tune_rejects_overlapping_seasons():
    with pytest.raises(ValueError, match="disjoint"):
        run_xi_tune(
            _matches(30),
            candidates=(0.0019,),
            tune_seasons=("2526",),
            holdout_seasons=("2526",),
            min_train_rows=5,
        )


def test_tie_breaks_toward_baseline(monkeypatch):
    """When tune LL ties after rounding, prefer the baseline ξ."""
    import src.eval.xi_tune as xi_tune

    real_score = xi_tune.score_dixon_coles_folds

    def _constant_ll(matches, *, xi, seasons=None, min_train_rows=200, run_at=None):
        dc_by_gw, book_by_gw, preds = real_score(
            matches,
            xi=xi,
            seasons=seasons,
            min_train_rows=min_train_rows,
            run_at=run_at,
        )
        if not preds.empty:
            # Force identical rounded pooled metrics for every candidate.
            n = len(preds)
            preds = preds.copy()
            preds["p_home"] = 0.5
            preds["p_draw"] = 0.25
            preds["p_away"] = 0.25
            assert n == len(preds)
        return dc_by_gw, book_by_gw, preds

    monkeypatch.setattr(xi_tune, "score_dixon_coles_folds", _constant_ll)
    result = run_xi_tune(
        _matches(48),
        candidates=(0.0010, 0.0019, 0.0050),
        tune_seasons=("2223", "2324"),
        holdout_seasons=("2526",),
        baseline_xi=0.0019,
        min_train_rows=5,
    )
    assert np.isclose(result.chosen_xi, 0.0019)
