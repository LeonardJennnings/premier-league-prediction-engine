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
    # Opening slightly softer than closing so the two book lines differ.
    home_open = home_odds + 0.05
    draw_open = draw_odds + 0.05
    away_open = away_odds - 0.05
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
            "home_odds_open": home_open,
            "draw_odds_open": draw_open,
            "away_odds_open": away_open,
            "home_odds_close": home_odds,
            "draw_odds_close": draw_odds,
            "away_odds_close": away_odds,
        }
    )


def _mixed_matches(n: int = 24) -> pd.DataFrame:
    """H/D/A cycle so the sklearn baseline has more than one class."""
    dates = pd.date_range("2024-08-01", periods=n, freq="D")
    results = [["H", "D", "A"][i % 3] for i in range(n)]
    goals = {"H": (2, 0), "D": (1, 1), "A": (0, 2)}
    homes = ["Arsenal", "Chelsea", "Liverpool", "Spurs"]
    aways = ["Chelsea", "Arsenal", "Spurs", "Liverpool"]
    implied_home = [0.45 + (i % 5) * 0.02 for i in range(n)]
    implied_away = [0.27 - (i % 5) * 0.02 for i in range(n)]
    return pd.DataFrame(
        {
            "season_code": ["2425"] * n,
            "date": dates,
            "home_team": [homes[i % 4] for i in range(n)],
            "away_team": [aways[i % 4] for i in range(n)],
            "home_goals": [goals[r][0] for r in results],
            "away_goals": [goals[r][1] for r in results],
            "result": results,
            "implied_home": implied_home,
            "implied_draw": [0.28] * n,
            "implied_away": implied_away,
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
                "market": "1x2",
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
                "market": "1x2",
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
    summary, by_gw, pred_frames, _scores = run_walkforward(raw, min_train_rows=10)

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

    opening = pred_frames["bookmaker_opening"]
    assert not opening.empty
    open_sums = opening["p_home"] + opening["p_draw"] + opening["p_away"]
    assert np.allclose(open_sums.to_numpy(dtype=float), 1.0, atol=1e-9)
    assert "bookmaker_opening" in set(summary["model"])


def test_new_models_import_and_fit_on_synthetic():
    from src.models.dixon_coles import fit_dixon_coles, predict_match as dc_predict
    from src.models.elo import fit_elo, predict_match as elo_predict

    matches = _synthetic_matches(40)
    matches["home_goals"] = 1
    matches["away_goals"] = 0
    dc = fit_dixon_coles(matches)
    elo = fit_elo(matches.assign(result=["H"] * 40))
    assert "home" in dc_predict(dc, "Arsenal", "Chelsea")
    assert "home" in elo_predict(elo, "Arsenal", "Chelsea")


def test_walkforward_includes_dixon_coles_and_elo():
    matches = assign_gameweeks(_mixed_matches(20), matches_per_gw=5)
    summary, _, preds, _scores = run_walkforward(matches, min_train_rows=5)
    models = set(summary["model"])
    assert "dixon_coles" in models
    assert "elo" in models
    assert "poisson" in models
    assert "dixon_coles_over_25" in models
    assert "poisson_total_over_25" in models
    assert "dixon_coles_btts" in models
    assert "dixon_coles_calibrated" not in models
    assert "bookmaker_over_25" not in models
    dc = preds["dixon_coles"]
    assert {"p_over_25", "p_under_25", "p_btts_yes", "p_btts_no"} <= set(dc.columns)
    assert ((dc["p_over_25"] >= 0) & (dc["p_over_25"] <= 1)).all()
    assert ((dc["p_under_25"] >= 0) & (dc["p_under_25"] <= 1)).all()
    ou = summary.loc[summary["model"] == "dixon_coles_over_25"].iloc[0]
    pt = summary.loc[summary["model"] == "poisson_total_over_25"].iloc[0]
    btts = summary.loc[summary["model"] == "dixon_coles_btts"].iloc[0]
    assert ou["market"] == "over_under_25"
    assert pt["market"] == "over_under_25"
    assert btts["market"] == "btts"
    assert ou["aggregation"] == "pooled"
    assert not preds["dixon_coles_over_25"].empty
    assert not preds["poisson_total_over_25"].empty
    assert not preds["dixon_coles_btts"].empty
    assert summary.loc[summary["model"] == "dixon_coles", "market"].iloc[0] == "1x2"


def _mixed_matches_with_totals(n: int = 80) -> pd.DataFrame:
    """Like _mixed_matches but goal totals span under and over 2.5 for logistic fit."""
    frame = _mixed_matches(n)
    # Cycle totals 1, 2, 3, 4 so both O/U classes appear.
    totals = [1, 2, 3, 4]
    homes = []
    aways = []
    for i in range(n):
        t = totals[i % 4]
        h = t // 2
        a = t - h
        homes.append(h)
        aways.append(a)
    frame["home_goals"] = homes
    frame["away_goals"] = aways
    # Keep 1X2 labels consistent with the new scores.
    results = []
    for h, a in zip(homes, aways):
        if h > a:
            results.append("H")
        elif h == a:
            results.append("D")
        else:
            results.append("A")
    frame["result"] = results
    return frame


def test_walkforward_lab_ou_lines_and_prior_fold_calibration():
    """Lab O/U comparators score every fold; binary temperature uses prior folds only."""
    # Need enough expanding-window train rows for totals_logistic (>=50).
    matches = assign_gameweeks(_mixed_matches_with_totals(80), matches_per_gw=5)
    summary, by_gw, preds, _scores = run_walkforward(
        matches, min_train_rows=5, calibration_min_rows=10
    )
    ou_models = set(summary.loc[summary["market"] == "over_under_25", "model"])
    assert "dixon_coles_over_25" in ou_models
    assert "poisson_total_over_25" in ou_models
    assert "totals_logistic_over_25" in ou_models

    raw = by_gw[by_gw["model"] == "poisson_total_over_25"]
    cal = by_gw[by_gw["model"] == "poisson_total_over_25_calibrated"]
    prior = 0
    expected_gws: list[int] = []
    for row in raw.itertuples(index=False):
        if prior >= 10:
            expected_gws.append(int(row.gameweek))
        prior += int(row.n_test)
    assert cal["gameweek"].astype(int).tolist() == expected_gws
    assert len(preds["poisson_total_over_25_calibrated"]) == int(cal["n_test"].sum())
    assert len(expected_gws) > 0
    assert len(expected_gws) < len(raw)


def test_calibrated_line_uses_only_prior_folds(monkeypatch):
    from src.eval.calibration import fit_temperature

    seen_lengths: list[int] = []

    def _spy(y_true, P):
        seen_lengths.append(len(list(y_true)))
        return fit_temperature(y_true, P)

    monkeypatch.setattr("src.eval.walkforward.fit_temperature", _spy)

    matches = assign_gameweeks(_mixed_matches(24), matches_per_gw=4)
    _, by_gw, preds, _scores = run_walkforward(
        matches, min_train_rows=4, calibration_min_rows=8
    )
    raw = by_gw[(by_gw["model"] == "dixon_coles") & (by_gw["market"] == "1x2")]
    cal = by_gw[by_gw["model"] == "dixon_coles_calibrated"]
    prior = 0
    expected_gws: list[int] = []
    expected_prior: list[int] = []
    for row in raw.itertuples(index=False):
        if prior >= 8:
            expected_gws.append(int(row.gameweek))
            expected_prior.append(prior)
        prior += int(row.n_test)

    assert cal["gameweek"].astype(int).tolist() == expected_gws
    assert seen_lengths == expected_prior
    assert len(preds["dixon_coles_calibrated"]) == int(cal["n_test"].sum())
    assert len(expected_gws) > 0
    assert len(expected_gws) < len(raw)


def test_benter_blend_uses_only_prior_folds(monkeypatch):
    from src.eval.blend import fit_benter_blend

    seen_lengths: list[int] = []

    def _spy(y_true, P_model, P_book):
        seen_lengths.append(len(list(y_true)))
        return fit_benter_blend(y_true, P_model, P_book)

    monkeypatch.setattr("src.eval.walkforward.fit_benter_blend", _spy)

    matches = assign_gameweeks(_mixed_matches(24), matches_per_gw=4)
    summary, by_gw, preds, _scores = run_walkforward(
        matches, min_train_rows=4, blend_min_rows=8
    )
    book = by_gw[by_gw["model"] == "bookmaker_implied"]
    blend = by_gw[by_gw["model"] == "dixon_coles_book_blend"]
    prior = 0
    expected_gws: list[int] = []
    expected_prior: list[int] = []
    for row in book.itertuples(index=False):
        if prior >= 8:
            expected_gws.append(int(row.gameweek))
            expected_prior.append(prior)
        prior += int(row.n_test)

    assert blend["gameweek"].astype(int).tolist() == expected_gws
    assert seen_lengths == expected_prior
    assert len(expected_gws) > 0
    assert "blend_w_model" in blend.columns
    assert blend["blend_w_model"].notna().all()
    assert "dixon_coles_book_blend" in set(summary["model"])
    blend_sum = summary.loc[summary["model"] == "dixon_coles_book_blend"].iloc[0]
    assert blend_sum["market"] == "1x2"
    assert pd.notna(blend_sum["blend_w_model"])
    assert len(preds["dixon_coles_book_blend"]) == int(blend["n_test"].sum())


def test_benter_ou_blend_row_when_book_over_present():
    frame = _mixed_matches(24)
    frame["implied_over_25"] = 0.55
    matches = assign_gameweeks(frame, matches_per_gw=4)
    summary, by_gw, _, _scores = run_walkforward(
        matches, min_train_rows=4, blend_min_rows=8
    )
    assert "dixon_coles_book_blend_over_25" in set(summary["model"])
    ou_blend = by_gw[by_gw["model"] == "dixon_coles_book_blend_over_25"]
    assert not ou_blend.empty
    assert (ou_blend["market"] == "over_under_25").all()
    assert ou_blend["blend_w_model"].notna().all()
    row = summary.loc[summary["model"] == "dixon_coles_book_blend_over_25"].iloc[0]
    assert row["market"] == "over_under_25"
    assert pd.notna(row["blend_w_model"])


def test_over_under_book_row_when_implied_over_present():
    frame = _mixed_matches(20)
    frame["implied_over_25"] = 0.55
    frame["implied_over_25_open"] = 0.52
    matches = assign_gameweeks(frame, matches_per_gw=5)
    summary, _, _, _scores = run_walkforward(matches, min_train_rows=5)
    assert "bookmaker_over_25" in set(summary["model"])
    assert "bookmaker_over_25_open" in set(summary["model"])
    row = summary.loc[summary["model"] == "bookmaker_over_25"].iloc[0]
    assert row["n_test"] > 0
    assert row["log_loss"] > 0
    assert row["aggregation"] == "pooled"
    assert row["market"] == "over_under_25"
    # Binary O/U and 1X2 share a CSV but must stay labelled separately.
    assert set(summary.loc[summary["market"] == "1x2", "model"]) >= {
        "dixon_coles",
        "elo",
        "poisson",
    }
    assert set(summary.loc[summary["market"] == "over_under_25", "model"]) >= {
        "dixon_coles_over_25",
        "poisson_total_over_25",
        "bookmaker_over_25",
        "bookmaker_over_25_open",
    }


def test_walkforward_scores_opening_book_when_odds_present():
    matches = assign_gameweeks(_synthetic_matches(25), matches_per_gw=5)
    summary, _, preds, _scores = run_walkforward(matches, min_train_rows=5)
    assert "bookmaker_opening" in set(summary["model"])
    assert "bookmaker_implied" in set(summary["model"])
    open_row = summary.loc[summary["model"] == "bookmaker_opening"].iloc[0]
    close_row = summary.loc[summary["model"] == "bookmaker_implied"].iloc[0]
    assert open_row["market"] == "1x2"
    assert close_row["market"] == "1x2"
    assert open_row["n_test"] > 0
    # Opening and closing should not be identical on this synthetic set.
    assert not np.allclose(
        preds["bookmaker_opening"]["p_home"].to_numpy(dtype=float),
        preds["bookmaker_implied"]["p_home"].to_numpy(dtype=float),
    )


def test_walkforward_readme_names_new_models(tmp_path):
    summary = pd.DataFrame(
        {
            "model": ["poisson"],
            "market": ["1x2"],
            "run_date": ["2026-11-16"],
            "n_folds": [1],
            "n_test": [1],
            "log_loss": [1.0],
            "brier": [0.2],
            "accuracy": [0.5],
            "aggregation": ["pooled"],
        }
    )
    by_gw = summary.copy()
    by_gw["season_code"] = "2425"
    by_gw["gameweek"] = 1
    by_gw["n_train"] = 10
    preds = {
        "poisson": pd.DataFrame(
            {
                "result": ["H"],
                "p_home": [0.5],
                "p_draw": [0.3],
                "p_away": [0.2],
            }
        )
    }
    paths = write_walkforward_artefacts(summary, by_gw, preds, results_dir=tmp_path)
    text = paths["readme"].read_text(encoding="utf-8")
    assert "dixon_coles" in text
    assert "dixon_coles_calibrated" in text
    assert "dixon_coles_over_25" in text
    assert "poisson_total_over_25" in text
    assert "totals_logistic_over_25" in text
    assert "dixon_coles_btts" in text
    assert "elo" in text
    assert "bookmaker_opening" in text
    assert "bookmaker_over_25_open" in text
    assert "dixon_coles_book_blend" in text
    assert "w_model" in text
    assert "fusion" in text.lower()
    assert "prior" in text.lower()
    assert "pooled" in text.lower()
    assert "do not rank" in text.lower()
    assert "lab" in text.lower()


def test_season_gate_reports_dixon_coles_and_elo_not_calibrated():
    from src.evaluate import run_backtest

    train = _mixed_matches(18)
    train["season_code"] = "2425"
    test = _mixed_matches(6)
    test["season_code"] = "2526"
    test["date"] = pd.date_range("2025-08-01", periods=6, freq="D")
    # Opening implied probs for season-gate bookmaker_opening row.
    test["implied_home_open"] = test["implied_home"] + 0.01
    test["implied_draw_open"] = test["implied_draw"] - 0.005
    test["implied_away_open"] = 1.0 - test["implied_home_open"] - test["implied_draw_open"]
    table = run_backtest(matches=pd.concat([train, test], ignore_index=True))
    models = set(table["model"])
    assert "dixon_coles" in models
    assert "elo" in models
    assert "dixon_coles_calibrated" not in models
    assert {
        "naive_frequencies",
        "poisson",
        "sklearn_baseline",
        "bookmaker_implied",
        "bookmaker_opening",
    } <= models
