"""Evaluate model predictions against results and bookmaker odds."""

# Eval rule (do not change just to make scores look nicer):
# - Test set = all matches in season_code "2526"
# - Training data = all configured seasons strictly before the test match date
#   (for season-level fit: train on seasons in SEASON_CODES that are before 2526)
# - Bookmaker comparison uses implied_home/draw/away on the same test rows;
#   drop rows where those odds are missing
# - Metrics: log loss, multiclass Brier, accuracy (most likely outcome)

from __future__ import annotations

import logging

import numpy as np

OUTCOME_ORDER = ("H", "D", "A")  # home, draw, away


def _as_proba_matrix(probas) -> np.ndarray:
    """Shape (n, 3) in H/D/A order."""
    arr = np.asarray(probas, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 3:
        raise ValueError("probas must have 3 columns: H, D, A")
    return arr


def _one_hot(y_true) -> np.ndarray:
    """Map H/D/A labels to one-hot rows."""
    mapping = {label: i for i, label in enumerate(OUTCOME_ORDER)}
    idx = np.array([mapping[y] for y in y_true], dtype=int)
    out = np.zeros((len(idx), 3), dtype=float)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def log_loss(y_true, probas, eps: float = 1e-15) -> float:
    """Mean negative log probability of the true outcome (lower is better)."""
    P = np.clip(_as_proba_matrix(probas), eps, 1.0)
    Y = _one_hot(y_true)
    # Only the true class contributes: -sum Y * log P
    return float(-np.sum(Y * np.log(P), axis=1).mean())


def brier_score(y_true, probas) -> float:
    """Mean squared error over the three outcomes (lower is better)."""
    P = _as_proba_matrix(probas)
    Y = _one_hot(y_true)
    return float(np.mean(np.sum((P - Y) ** 2, axis=1) / 3.0))


def accuracy(y_true, probas) -> float:
    """Share of matches where the top predicted outcome is correct."""
    P = _as_proba_matrix(probas)
    pred_idx = np.argmax(P, axis=1)
    pred = [OUTCOME_ORDER[i] for i in pred_idx]
    y_list = list(y_true)
    return float(np.mean([p == y for p, y in zip(pred, y_list)]))

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.clean import build_matches_table
from src.config import RESULTS, SEASON_CODES
from src.features import add_features
from src.models.baseline import fit_baseline, predict_match_features
from src.models.dixon_coles import fit_dixon_coles
from src.models.dixon_coles import predict_match as predict_dixon_coles
from src.models.elo import fit_elo
from src.models.elo import predict_match as predict_elo
from src.models.poisson import fit_poisson, predict_match

TEST_SEASON = "2526"


def _train_seasons(test_season: str = TEST_SEASON) -> list[str]:
    return [s for s in SEASON_CODES if s < test_season]


def _probas_to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.array([[r["home"], r["draw"], r["away"]] for r in rows], dtype=float)


def _naive_probas(train: pd.DataFrame, n: int) -> np.ndarray:
    """Always predict the historical H/D/A frequencies from train."""
    counts = train["result"].value_counts(normalize=True)
    p = {
        "home": float(counts.get("H", 0.0)),
        "draw": float(counts.get("D", 0.0)),
        "away": float(counts.get("A", 0.0)),
    }
    # Normalise in case of missing labels
    total = p["home"] + p["draw"] + p["away"]
    p = {k: v / total for k, v in p.items()}
    return np.tile([p["home"], p["draw"], p["away"]], (n, 1))


def run_backtest(
    test_season: str = TEST_SEASON,
    matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit on earlier seasons, score on the fixed test season, return a results table.

    Dixon–Coles and Elo are raw 1X2 lines. Temperature calibration stays on the
    walk-forward path so this gate does not fit a scaler on the test season.
    """
    if matches is None:
        matches = add_features(build_matches_table())
    else:
        matches = matches.copy()
    train = matches[matches["season_code"].isin(_train_seasons(test_season))].copy()
    test = matches[matches["season_code"] == test_season].copy()
    test = test.dropna(subset=["implied_home", "implied_draw", "implied_away", "result"])

    if train.empty or test.empty:
        raise ValueError("Train or test set is empty — check season codes and data.")

    y_true = test["result"].tolist()

    # --- predictors ---
    naive_P = _naive_probas(train, len(test))

    poisson_model = fit_poisson(train)
    poisson_rows = [
        predict_match(poisson_model, row.home_team, row.away_team)
        for row in test.itertuples(index=False)
    ]
    poisson_P = _probas_to_matrix(poisson_rows)

    dc_model = fit_dixon_coles(train)
    dc_rows = [
        predict_dixon_coles(dc_model, row.home_team, row.away_team)
        for row in test.itertuples(index=False)
    ]
    dc_P = _probas_to_matrix(dc_rows)

    elo_model = fit_elo(train)
    elo_rows = [
        predict_elo(elo_model, row.home_team, row.away_team)
        for row in test.itertuples(index=False)
    ]
    elo_P = _probas_to_matrix(elo_rows)

    baseline_model = fit_baseline(train)
    baseline_rows = [predict_match_features(baseline_model, row) for _, row in test.iterrows()]
    baseline_P = _probas_to_matrix(baseline_rows)

    bookie_P = test[["implied_home", "implied_draw", "implied_away"]].to_numpy(dtype=float)

    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = []
    for name, P in [
        ("naive_frequencies", naive_P),
        ("poisson", poisson_P),
        ("dixon_coles", dc_P),
        ("elo", elo_P),
        ("sklearn_baseline", baseline_P),
        ("bookmaker_implied", bookie_P),
    ]:
        rows.append(
            {
                "run_date": run_at,
                "model": name,
                "test_set": test_season,
                "n_train": len(train),
                "n_test": len(test),
                "log_loss": round(log_loss(y_true, P), 4),
                "brier": round(brier_score(y_true, P), 4),
                "accuracy": round(accuracy(y_true, P), 4),
            }
        )

    open_cols = ["implied_home_open", "implied_draw_open", "implied_away_open"]
    if set(open_cols) <= set(test.columns):
        open_test = test.dropna(subset=open_cols + ["result"])
        if not open_test.empty:
            open_P = open_test[open_cols].to_numpy(dtype=float)
            y_open = open_test["result"].tolist()
            rows.append(
                {
                    "run_date": run_at,
                    "model": "bookmaker_opening",
                    "test_set": test_season,
                    "n_train": len(train),
                    "n_test": len(open_test),
                    "log_loss": round(log_loss(y_open, open_P), 4),
                    "brier": round(brier_score(y_open, open_P), 4),
                    "accuracy": round(accuracy(y_open, open_P), 4),
                }
            )
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Append/write eval results under results/ (committed on purpose)."""
    if path is None:
        path = RESULTS / "eval_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import argparse

    from src.config import LOG_LEVEL
    from src.eval.walkforward import run_walkforward, write_walkforward_artefacts
    from src.logging_setup import setup_logging

    setup_logging(LOG_LEVEL)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Premier League evaluation")
    parser.add_argument(
        "--mode",
        choices=(
            "season",
            "walkforward",
            "xi_tune",
            "thin_prior",
            "grid_calibration",
            "market_blend",
            "gap_split",
            "two_speed",
        ),
        default="season",
        help=(
            "season = fixed 2526 gate; walkforward = gameweek walk-forward; "
            "xi_tune = nested ξ selection on older folds + holdout report; "
            "thin_prior = one strength for thin clubs, chosen on earlier seasons; "
            "grid_calibration = one scale and one steady low-score correction; "
            "market_blend = one weight on the price available before kickoff; "
            "gap_split = split the saved scoreline gap into rates and shape, no refit; "
            "two_speed = one weight between a slow fit and a fast fit"
        ),
    )
    args = parser.parse_args()

    if args.mode == "season":
        table = run_backtest()
        out = save_results(table)
        logger.info("\n%s", table.to_string(index=False))
        logger.info("Saved %s", out)
    elif args.mode == "thin_prior":
        from src.eval.thin_prior import run_thin_prior, write_thin_prior_artefacts

        result = run_thin_prior()
        paths = write_thin_prior_artefacts(result)
        logger.info(
            "Strength %.4g dropped=%s (%s)",
            result.chosen_kappa,
            result.dropped,
            result.reason,
        )
        for label, path in paths.items():
            logger.info("Wrote %s -> %s", label, path)
    elif args.mode == "grid_calibration":
        from src.eval.grid_calibration import (
            run_grid_calibration,
            write_grid_calibration_artefacts,
        )

        result = run_grid_calibration()
        paths = write_grid_calibration_artefacts(result)
        logger.info(
            "Scale %.2f correction %.2f dropped=%s (%s)",
            result.chosen_scale,
            result.chosen_rho,
            result.dropped,
            result.reason,
        )
        for label, path in paths.items():
            logger.info("Wrote %s -> %s", label, path)
    elif args.mode == "gap_split":
        from src.clean import build_matches_table
        from src.eval.gap_split import run_gap_split, write_gap_split_artefacts

        result = run_gap_split(matches=build_matches_table())
        paths = write_gap_split_artefacts(result)
        logger.info("Gap sits in the %s", result.location)
        for label, path in paths.items():
            logger.info("Wrote %s -> %s", label, path)
    elif args.mode == "two_speed":
        from src.eval.two_speed import run_two_speed, write_two_speed_artefacts

        result = run_two_speed()
        paths = write_two_speed_artefacts(result)
        logger.info(
            "Weight %.2f dropped=%s edge=%s replaces=%s (%s)",
            result.chosen_weight,
            result.dropped,
            result.edge,
            result.replaces_pure,
            result.reason,
        )
        for label, path in paths.items():
            logger.info("Wrote %s -> %s", label, path)
    elif args.mode == "market_blend":
        from src.eval.market_blend import run_market_blend, write_market_blend_artefacts

        result = run_market_blend()
        paths = write_market_blend_artefacts(result)
        logger.info(
            "Weight %.2f dropped=%s (%s)",
            result.chosen_weight,
            result.dropped,
            result.reason,
        )
        for label, path in paths.items():
            logger.info("Wrote %s -> %s", label, path)
    elif args.mode == "xi_tune":
        from src.eval.xi_tune import run_xi_tune, write_xi_tune_artefacts

        result = run_xi_tune()
        paths = write_xi_tune_artefacts(result)
        logger.info("Chosen ξ=%.6f (baseline=%.6f)", result.chosen_xi, result.baseline_xi)
        logger.info("\nTune grid:\n%s", result.candidate_tune.to_string(index=False))
        logger.info(
            "\nHoldout comparison:\n%s",
            result.holdout_comparison.to_string(index=False),
        )
        for label, path in paths.items():
            logger.info("Wrote %s -> %s", label, path)
    else:
        summary, by_gw, pred_frames, match_scores = run_walkforward()
        paths = write_walkforward_artefacts(
            summary, by_gw, pred_frames, match_scores=match_scores
        )
        logger.info("\n%s", summary.to_string(index=False))
        for label, path in paths.items():
            logger.info("Wrote %s -> %s", label, path)
