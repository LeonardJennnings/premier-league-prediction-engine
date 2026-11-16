"""Gameweek walk-forward evaluation runner."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BLEND_MIN_ROWS, CALIBRATION_MIN_ROWS, RESULTS
from src.eval.blend import (
    apply_benter_blend,
    apply_binary_logodds_blend,
    fit_benter_blend,
    fit_binary_logodds_blend,
)
from src.eval.calibration import (
    apply_temperature,
    fit_temperature,
    plot_reliability,
    reliability_table,
)
from src.eval.gameweeks import assign_gameweeks
from src.evaluate import accuracy, brier_score, log_loss
from src.features import features_for_fold
from src.models.baseline import fit_baseline, predict_match_features
from src.models.dixon_coles import fit_dixon_coles
from src.models.dixon_coles import predict_match as predict_dixon_coles
from src.models.elo import fit_elo
from src.models.elo import predict_match as predict_elo
from src.models.poisson import fit_poisson, predict_match

logger = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 200


def iter_folds(
    matches: pd.DataFrame,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str, int]]:
    """Yield (train, test, season_code, gameweek) with a strict date cutoff.

    Train = every match with date strictly before this fold's earliest kickoff.
    Folds with fewer than min_train_rows in train are skipped.
    """
    if "gameweek" not in matches.columns:
        raise ValueError("matches must include gameweek; call assign_gameweeks first")

    for (season, gw), test in matches.groupby(["season_code", "gameweek"], sort=True):
        cutoff = test["date"].min()
        train = matches[matches["date"] < cutoff]
        if len(train) < min_train_rows:
            logger.warning(
                "Skipping %s GW%s: train_rows=%s < min_train_rows=%s",
                season,
                gw,
                len(train),
                min_train_rows,
            )
            continue
        yield train.copy(), test.copy(), str(season), int(gw)


def _probas_to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.array([[r["home"], r["draw"], r["away"]] for r in rows], dtype=float)


def _naive_probas(train: pd.DataFrame, n: int) -> np.ndarray:
    counts = train["result"].value_counts(normalize=True)
    p = np.array(
        [
            float(counts.get("H", 0.0)),
            float(counts.get("D", 0.0)),
            float(counts.get("A", 0.0)),
        ],
        dtype=float,
    )
    total = p.sum()
    if total <= 0:
        p = np.array([1 / 3, 1 / 3, 1 / 3])
    else:
        p = p / total
    return np.tile(p, (n, 1))


def _metric_row(
    run_at: str,
    model: str,
    season: str,
    gw: int,
    n_train: int,
    y_true,
    P,
    *,
    market: str = "1x2",
    **extra,
) -> dict:
    row = {
        "run_date": run_at,
        "model": model,
        "market": market,
        "season_code": season,
        "gameweek": gw,
        "n_train": n_train,
        "n_test": len(y_true),
        "log_loss": round(log_loss(y_true, P), 4),
        "brier": round(brier_score(y_true, P), 4),
        "accuracy": round(accuracy(y_true, P), 4),
    }
    row.update(extra)
    return row


def _binary_log_loss(y_true, p, eps: float = 1e-15) -> float:
    """Mean binary log loss for an over/under probability."""
    y = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)).mean())


def _binary_metric_row(
    run_at: str,
    model: str,
    season: str,
    gw: int,
    n_train: int,
    y_true,
    p,
    *,
    market: str,
    **extra,
) -> dict:
    """Score a binary market line. Brier is mean squared error, not the 1X2 Brier."""
    y = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    row = {
        "run_date": run_at,
        "model": model,
        "market": market,
        "season_code": season,
        "gameweek": gw,
        "n_train": n_train,
        "n_test": len(y),
        "log_loss": round(_binary_log_loss(y, prob), 4),
        "brier": round(float(np.mean((prob - y) ** 2)), 4),
        "accuracy": round(float(np.mean((prob >= 0.5) == (y >= 0.5))), 4),
    }
    row.update(extra)
    return row


def _infer_market(model: str, fold_rows: pd.DataFrame, frame: pd.DataFrame) -> str:
    """Label the market family so binary lines are not ranked with 1X2."""
    if not fold_rows.empty and "market" in fold_rows.columns:
        values = fold_rows["market"].dropna().unique()
        if len(values) == 1:
            return str(values[0])
    if model.endswith("_over_25"):
        return "over_under_25"
    if model.endswith("_btts"):
        return "btts"
    if {"y_over", "p_over"} <= set(frame.columns):
        return "over_under_25"
    if {"y_btts", "p_btts"} <= set(frame.columns):
        return "btts"
    return "1x2"


def _pooled_summary(by_gw: pd.DataFrame, pred_frames: dict[str, pd.DataFrame], run_at: str) -> pd.DataFrame:
    """Research SoT row: pooled proper scores over all scored matches.

    Per-fold means can skew the headline when fold sizes differ (partial
    gameweeks, odds dropouts). Pooled log loss / Brier / accuracy use every
    prediction once. Per-fold tables remain in ``by_gw`` for diagnosis.

    1X2 frames use ``p_home`` / ``p_draw`` / ``p_away``. Binary frames use
    ``y_over`` / ``p_over`` (totals) or ``y_btts`` / ``p_btts``. The ``market``
    column labels the family; binary log loss is not comparable to 1X2.
    """
    rows = []
    models = sorted(by_gw["model"].unique()) if not by_gw.empty else list(pred_frames)
    for model in models:
        frame = pred_frames.get(model, pd.DataFrame())
        fold_rows = by_gw[by_gw["model"] == model] if not by_gw.empty else pd.DataFrame()
        n_folds = int(len(fold_rows))
        n_test_folds = int(fold_rows["n_test"].sum()) if not fold_rows.empty else 0
        market = _infer_market(model, fold_rows, frame)

        if frame.empty:
            rows.append(
                {
                    "run_date": run_at,
                    "model": model,
                    "market": market,
                    "n_folds": n_folds,
                    "n_test": n_test_folds,
                    "log_loss": float("nan"),
                    "brier": float("nan"),
                    "accuracy": float("nan"),
                    "aggregation": "pooled",
                }
            )
            continue

        def _with_blend_meta(row: dict) -> dict:
            """Attach mean prior-fold Benter coeffs when present on by-gw rows."""
            for col in ("blend_alpha", "blend_beta", "blend_w_model"):
                if not fold_rows.empty and col in fold_rows.columns:
                    values = fold_rows[col].dropna()
                    row[col] = round(float(values.mean()), 4) if not values.empty else float("nan")
            return row

        if {"p_home", "p_draw", "p_away", "result"} <= set(frame.columns):
            y_true = frame["result"].tolist()
            P = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
            rows.append(
                _with_blend_meta(
                    {
                        "run_date": run_at,
                        "model": model,
                        "market": market,
                        "n_folds": n_folds,
                        "n_test": len(y_true),
                        "log_loss": round(log_loss(y_true, P), 4),
                        "brier": round(brier_score(y_true, P), 4),
                        "accuracy": round(accuracy(y_true, P), 4),
                        "aggregation": "pooled",
                    }
                )
            )
            continue

        binary_cols = None
        if {"y_over", "p_over"} <= set(frame.columns):
            binary_cols = ("y_over", "p_over")
        elif {"y_btts", "p_btts"} <= set(frame.columns):
            binary_cols = ("y_btts", "p_btts")
        if binary_cols is not None:
            y = frame[binary_cols[0]].to_numpy(dtype=float)
            p = frame[binary_cols[1]].to_numpy(dtype=float)
            rows.append(
                _with_blend_meta(
                    {
                        "run_date": run_at,
                        "model": model,
                        "market": market,
                        "n_folds": n_folds,
                        "n_test": len(y),
                        "log_loss": round(_binary_log_loss(y, p), 4),
                        "brier": round(float(np.mean((np.clip(p, 0.0, 1.0) - y) ** 2)), 4),
                        "accuracy": round(
                            float(np.mean((np.clip(p, 0.0, 1.0) >= 0.5) == (y >= 0.5))), 4
                        ),
                        "aggregation": "pooled",
                    }
                )
            )
            continue

        rows.append(
            _with_blend_meta(
                {
                    "run_date": run_at,
                    "model": model,
                    "market": market,
                    "n_folds": n_folds,
                    "n_test": n_test_folds,
                    "log_loss": float("nan"),
                    "brier": float("nan"),
                    "accuracy": float("nan"),
                    "aggregation": "pooled",
                }
            )
        )
    return pd.DataFrame(rows)


def run_walkforward(
    matches: pd.DataFrame | None = None,
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
    calibration_min_rows: int = CALIBRATION_MIN_ROWS,
    blend_min_rows: int = BLEND_MIN_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run full walk-forward; return summary, by-gameweek, and per-model prediction frames.

    Features are built per fold via :func:`src.features.features_for_fold` so
    history-dependent columns cannot see the test window.
    """
    from src.clean import build_matches_table

    if matches is None:
        # Raw cleaned table — fold loop applies as-of features.
        matches = build_matches_table()
    matches = assign_gameweeks(matches)
    matches["date"] = pd.to_datetime(matches["date"])

    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    by_gw_rows: list[dict] = []
    pred_records: dict[str, list[dict]] = {
        "naive_frequencies": [],
        "poisson": [],
        "dixon_coles": [],
        "dixon_coles_calibrated": [],
        "dixon_coles_book_blend": [],
        "dixon_coles_over_25": [],
        "dixon_coles_book_blend_over_25": [],
        "dixon_coles_btts": [],
        "elo": [],
        "sklearn_baseline": [],
        "bookmaker_implied": [],
        "bookmaker_opening": [],
        "bookmaker_over_25": [],
        "bookmaker_over_25_open": [],
    }

    prior_dc_y: list[str] = []
    prior_dc_P: list[np.ndarray] = []
    # Prior-fold OOS pairs for Benter blend (closing book only).
    prior_blend_y: list[str] = []
    prior_blend_dc: list[np.ndarray] = []
    prior_blend_book: list[np.ndarray] = []
    prior_ou_y: list[float] = []
    prior_ou_dc: list[np.ndarray] = []
    prior_ou_book: list[np.ndarray] = []

    book_1x2_cols = ["implied_home", "implied_draw", "implied_away"]

    for train_raw, test_raw, season, gw in iter_folds(matches, min_train_rows=min_train_rows):
        train, test = features_for_fold(train_raw, test_raw)
        y_all = test["result"].tolist()

        naive_P = _naive_probas(train, len(test))
        by_gw_rows.append(
            _metric_row(run_at, "naive_frequencies", season, gw, len(train), y_all, naive_P)
        )
        for i, (_, row) in enumerate(test.iterrows()):
            pred_records["naive_frequencies"].append(
                {
                    "result": row["result"],
                    "p_home": naive_P[i, 0],
                    "p_draw": naive_P[i, 1],
                    "p_away": naive_P[i, 2],
                }
            )

        poisson_model = fit_poisson(train)
        poisson_rows = [
            predict_match(poisson_model, row.home_team, row.away_team)
            for row in test.itertuples(index=False)
        ]
        poisson_P = _probas_to_matrix(poisson_rows)
        by_gw_rows.append(
            _metric_row(run_at, "poisson", season, gw, len(train), y_all, poisson_P)
        )
        for i, (_, row) in enumerate(test.iterrows()):
            pred_records["poisson"].append(
                {
                    "result": row["result"],
                    "p_home": poisson_P[i, 0],
                    "p_draw": poisson_P[i, 1],
                    "p_away": poisson_P[i, 2],
                }
            )

        dc_model = fit_dixon_coles(train)
        dc_rows = [
            predict_dixon_coles(dc_model, row.home_team, row.away_team)
            for row in test.itertuples(index=False)
        ]
        dc_P = _probas_to_matrix(dc_rows)
        by_gw_rows.append(
            _metric_row(run_at, "dixon_coles", season, gw, len(train), y_all, dc_P)
        )
        for i, row in enumerate(test.itertuples(index=False)):
            pred_records["dixon_coles"].append(
                {
                    "result": row.result,
                    "p_home": dc_P[i, 0],
                    "p_draw": dc_P[i, 1],
                    "p_away": dc_P[i, 2],
                    "p_over_25": dc_rows[i]["over_25"],
                    "p_under_25": dc_rows[i]["under_25"],
                    "p_btts_yes": dc_rows[i]["btts_yes"],
                    "p_btts_no": dc_rows[i]["btts_no"],
                }
            )

        # Model O/U and BTTS from the same raw grid (not temperature-scaled).
        y_over = (
            (test["home_goals"].to_numpy(dtype=float) + test["away_goals"].to_numpy(dtype=float))
            >= 3
        ).astype(float)
        p_over = np.array([row["over_25"] for row in dc_rows], dtype=float)
        by_gw_rows.append(
            _binary_metric_row(
                run_at,
                "dixon_coles_over_25",
                season,
                gw,
                len(train),
                y_over,
                p_over,
                market="over_under_25",
            )
        )
        for y_i, p_i in zip(y_over, p_over):
            pred_records["dixon_coles_over_25"].append(
                {"y_over": float(y_i), "p_over": float(p_i)}
            )

        y_btts = (
            (test["home_goals"].to_numpy(dtype=float) >= 1)
            & (test["away_goals"].to_numpy(dtype=float) >= 1)
        ).astype(float)
        p_btts = np.array([row["btts_yes"] for row in dc_rows], dtype=float)
        by_gw_rows.append(
            _binary_metric_row(
                run_at,
                "dixon_coles_btts",
                season,
                gw,
                len(train),
                y_btts,
                p_btts,
                market="btts",
            )
        )
        for y_i, p_i in zip(y_btts, p_btts):
            pred_records["dixon_coles_btts"].append(
                {"y_btts": float(y_i), "p_btts": float(p_i)}
            )

        # Temperature uses earlier folds only. Append this fold after the fit.
        if prior_dc_y and len(prior_dc_y) >= calibration_min_rows:
            temperature = fit_temperature(prior_dc_y, np.vstack(prior_dc_P))
            cal_P = apply_temperature(dc_P, temperature)
            by_gw_rows.append(
                _metric_row(
                    run_at,
                    "dixon_coles_calibrated",
                    season,
                    gw,
                    len(train),
                    y_all,
                    cal_P,
                )
            )
            for i, row in enumerate(test.itertuples(index=False)):
                pred_records["dixon_coles_calibrated"].append(
                    {
                        "result": row.result,
                        "p_home": cal_P[i, 0],
                        "p_draw": cal_P[i, 1],
                        "p_away": cal_P[i, 2],
                        "p_over_25": dc_rows[i]["over_25"],
                        "p_under_25": dc_rows[i]["under_25"],
                    }
                )
        prior_dc_y.extend(y_all)
        prior_dc_P.append(dc_P)

        # Benter DC + closing-book blend on prior-fold OOS pairs only.
        has_book_1x2 = (
            test[book_1x2_cols].notna().all(axis=1).to_numpy()
            if set(book_1x2_cols) <= set(test.columns)
            else np.zeros(len(test), dtype=bool)
        )
        if (
            prior_blend_y
            and len(prior_blend_y) >= blend_min_rows
            and has_book_1x2.any()
        ):
            weights = fit_benter_blend(
                prior_blend_y,
                np.vstack(prior_blend_dc),
                np.vstack(prior_blend_book),
            )
            book_P_fold = test.loc[has_book_1x2, book_1x2_cols].to_numpy(dtype=float)
            y_blend = test.loc[has_book_1x2, "result"].tolist()
            blend_P = apply_benter_blend(
                dc_P[has_book_1x2], book_P_fold, weights.alpha, weights.beta
            )
            by_gw_rows.append(
                _metric_row(
                    run_at,
                    "dixon_coles_book_blend",
                    season,
                    gw,
                    len(train),
                    y_blend,
                    blend_P,
                    blend_alpha=round(weights.alpha, 4),
                    blend_beta=round(weights.beta, 4),
                    blend_w_model=round(weights.w_model, 4),
                )
            )
            for i, result in enumerate(y_blend):
                pred_records["dixon_coles_book_blend"].append(
                    {
                        "result": result,
                        "p_home": blend_P[i, 0],
                        "p_draw": blend_P[i, 1],
                        "p_away": blend_P[i, 2],
                        "blend_alpha": weights.alpha,
                        "blend_beta": weights.beta,
                        "blend_w_model": weights.w_model,
                    }
                )
        if has_book_1x2.any():
            prior_blend_y.extend(test.loc[has_book_1x2, "result"].tolist())
            prior_blend_dc.append(dc_P[has_book_1x2])
            prior_blend_book.append(
                test.loc[has_book_1x2, book_1x2_cols].to_numpy(dtype=float)
            )

        elo_model = fit_elo(train)
        elo_rows = [
            predict_elo(elo_model, row.home_team, row.away_team)
            for row in test.itertuples(index=False)
        ]
        elo_P = _probas_to_matrix(elo_rows)
        by_gw_rows.append(
            _metric_row(run_at, "elo", season, gw, len(train), y_all, elo_P)
        )
        for i, row in enumerate(test.itertuples(index=False)):
            pred_records["elo"].append(
                {
                    "result": row.result,
                    "p_home": elo_P[i, 0],
                    "p_draw": elo_P[i, 1],
                    "p_away": elo_P[i, 2],
                }
            )

        base_train = train.dropna(
            subset=["implied_home", "implied_draw", "implied_away", "result"]
        )
        base_test = test.dropna(
            subset=["implied_home", "implied_draw", "implied_away", "result"]
        )
        if len(base_train) >= min_train_rows and not base_test.empty:
            baseline_model = fit_baseline(base_train)
            baseline_rows = [
                predict_match_features(baseline_model, row)
                for _, row in base_test.iterrows()
            ]
            baseline_P = _probas_to_matrix(baseline_rows)
            y_base = base_test["result"].tolist()
            by_gw_rows.append(
                _metric_row(
                    run_at, "sklearn_baseline", season, gw, len(base_train), y_base, baseline_P
                )
            )
            for i, (_, row) in enumerate(base_test.iterrows()):
                pred_records["sklearn_baseline"].append(
                    {
                        "result": row["result"],
                        "p_home": baseline_P[i, 0],
                        "p_draw": baseline_P[i, 1],
                        "p_away": baseline_P[i, 2],
                    }
                )

            bookie_P = base_test[
                ["implied_home", "implied_draw", "implied_away"]
            ].to_numpy(dtype=float)
            by_gw_rows.append(
                _metric_row(
                    run_at, "bookmaker_implied", season, gw, len(base_train), y_base, bookie_P
                )
            )
            for i, (_, row) in enumerate(base_test.iterrows()):
                pred_records["bookmaker_implied"].append(
                    {
                        "result": row["result"],
                        "p_home": bookie_P[i, 0],
                        "p_draw": bookie_P[i, 1],
                        "p_away": bookie_P[i, 2],
                    }
                )

        open_cols = [
            "implied_home_open",
            "implied_draw_open",
            "implied_away_open",
        ]
        if set(open_cols) <= set(test.columns):
            open_test = test.dropna(subset=open_cols + ["result"])
            if not open_test.empty:
                open_P = open_test[open_cols].to_numpy(dtype=float)
                y_open = open_test["result"].tolist()
                n_train_open = (
                    int(train.dropna(subset=open_cols).shape[0])
                    if set(open_cols) <= set(train.columns)
                    else 0
                )
                by_gw_rows.append(
                    _metric_row(
                        run_at,
                        "bookmaker_opening",
                        season,
                        gw,
                        n_train_open,
                        y_open,
                        open_P,
                    )
                )
                for i, (_, row) in enumerate(open_test.iterrows()):
                    pred_records["bookmaker_opening"].append(
                        {
                            "result": row["result"],
                            "p_home": open_P[i, 0],
                            "p_draw": open_P[i, 1],
                            "p_away": open_P[i, 2],
                        }
                    )

        has_ou_book = (
            test["implied_over_25"].notna().to_numpy()
            if "implied_over_25" in test.columns
            else np.zeros(len(test), dtype=bool)
        )
        if has_ou_book.any():
            ou_test = test.loc[has_ou_book]
            y_over_book = (
                (ou_test["home_goals"].to_numpy(dtype=float)
                 + ou_test["away_goals"].to_numpy(dtype=float))
                >= 3
            ).astype(float)
            p_over_book = ou_test["implied_over_25"].to_numpy(dtype=float)
            p_over_dc_fold = np.array(
                [row["over_25"] for row, keep in zip(dc_rows, has_ou_book) if keep],
                dtype=float,
            )
            n_train_ou = (
                int(train["implied_over_25"].notna().sum())
                if "implied_over_25" in train.columns
                else 0
            )
            by_gw_rows.append(
                _binary_metric_row(
                    run_at,
                    "bookmaker_over_25",
                    season,
                    gw,
                    n_train_ou,
                    y_over_book,
                    p_over_book,
                    market="over_under_25",
                )
            )
            for y_i, p_i in zip(y_over_book, p_over_book):
                pred_records["bookmaker_over_25"].append(
                    {"y_over": float(y_i), "p_over": float(p_i)}
                )

            if prior_ou_y and len(prior_ou_y) >= blend_min_rows:
                ou_weights = fit_binary_logodds_blend(
                    prior_ou_y,
                    np.concatenate(prior_ou_dc),
                    np.concatenate(prior_ou_book),
                )
                blend_over = apply_binary_logodds_blend(
                    p_over_dc_fold,
                    p_over_book,
                    ou_weights.alpha,
                    ou_weights.beta,
                )
                by_gw_rows.append(
                    _binary_metric_row(
                        run_at,
                        "dixon_coles_book_blend_over_25",
                        season,
                        gw,
                        len(train),
                        y_over_book,
                        blend_over,
                        market="over_under_25",
                        blend_alpha=round(ou_weights.alpha, 4),
                        blend_beta=round(ou_weights.beta, 4),
                        blend_w_model=round(ou_weights.w_model, 4),
                    )
                )
                for y_i, p_i in zip(y_over_book, blend_over):
                    pred_records["dixon_coles_book_blend_over_25"].append(
                        {
                            "y_over": float(y_i),
                            "p_over": float(p_i),
                            "blend_alpha": ou_weights.alpha,
                            "blend_beta": ou_weights.beta,
                            "blend_w_model": ou_weights.w_model,
                        }
                    )

            prior_ou_y.extend(float(v) for v in y_over_book)
            prior_ou_dc.append(p_over_dc_fold)
            prior_ou_book.append(p_over_book)

        if "implied_over_25_open" in test.columns:
            ou_open = test.dropna(
                subset=["implied_over_25_open", "home_goals", "away_goals"]
            )
            if not ou_open.empty:
                y_over_o = (
                    (ou_open["home_goals"] + ou_open["away_goals"]) >= 3
                ).astype(float)
                p_over_o = ou_open["implied_over_25_open"].to_numpy(dtype=float)
                n_train_ou_o = (
                    int(train["implied_over_25_open"].notna().sum())
                    if "implied_over_25_open" in train.columns
                    else 0
                )
                by_gw_rows.append(
                    _binary_metric_row(
                        run_at,
                        "bookmaker_over_25_open",
                        season,
                        gw,
                        n_train_ou_o,
                        y_over_o,
                        p_over_o,
                        market="over_under_25",
                    )
                )
                for y_i, p_i in zip(y_over_o.to_numpy(), p_over_o):
                    pred_records["bookmaker_over_25_open"].append(
                        {"y_over": float(y_i), "p_over": float(p_i)}
                    )

    by_gw = pd.DataFrame(by_gw_rows)
    pred_frames = {k: pd.DataFrame(v) for k, v in pred_records.items()}
    summary = _pooled_summary(by_gw, pred_frames, run_at)
    return summary, by_gw, pred_frames


def write_walkforward_artefacts(
    summary: pd.DataFrame,
    by_gw: pd.DataFrame,
    pred_frames: dict[str, pd.DataFrame],
    results_dir: Path | None = None,
) -> dict[str, Path]:
    """Write CSVs, home-win reliability plots, and WALKFORWARD.md under results/."""
    results_dir = Path(results_dir or RESULTS)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    summary_path = results_dir / "walkforward_summary.csv"
    by_gw_path = results_dir / "walkforward_by_gameweek.csv"
    summary.to_csv(summary_path, index=False)
    by_gw.to_csv(by_gw_path, index=False)
    paths["summary"] = summary_path
    paths["by_gameweek"] = by_gw_path

    for model_name, frame in pred_frames.items():
        if frame.empty or "p_home" not in frame.columns or "result" not in frame.columns:
            continue
        y_home = (frame["result"] == "H").astype(float).to_numpy()
        p_home = frame["p_home"].to_numpy(dtype=float)
        table = reliability_table(y_home, p_home, n_bins=10)
        plot_path = results_dir / f"calibration_{model_name}_home.png"
        plot_reliability(table, plot_path, title=f"{model_name} — home win reliability")
        paths[f"calibration_{model_name}"] = plot_path

    readme = results_dir / "WALKFORWARD.md"
    readme.write_text(
        "# Walk-forward artefacts\n"
        "\n"
        "## Protocol\n"
        "\n"
        "- Folds: synthetic gameweeks (<=10 matches, date-ordered within each season)\n"
        "- Train: all matches with date strictly before the fold\n"
        "- Features: built per fold; history-dependent columns use date < fold cutoff only\n"
        "- Minimum train rows: 200\n"
        "- Models (1X2, market=1x2): naive_frequencies, poisson, dixon_coles, "
        "dixon_coles_calibrated, dixon_coles_book_blend, elo, sklearn_baseline, "
        "bookmaker_implied (Bet365 closing), bookmaker_opening (Bet365 pre-closing)\n"
        "- dixon_coles_calibrated: temperature scaling fit on prior-fold "
        "Dixon–Coles 1X2 predictions only. The line is skipped until that "
        f"prior store has at least {CALIBRATION_MIN_ROWS} rows "
        "(the raw line is not copied in). Summary fold counts can therefore "
        "differ from raw Dixon–Coles; compare them on shared gameweeks.\n"
        "- dixon_coles_book_blend: Benter log-odds fusion of raw Dixon–Coles "
        "with de-vigged Bet365 **closing** 1X2. Coefficients (α, β) and "
        "w_model = α/(α+β) are fit on prior-fold OOS pairs only (never "
        "in-sample DC fitted probs). Skipped until "
        f"{BLEND_MIN_ROWS} prior rows. Publish w_model even if ≈ 0 — that "
        "means the market dominates. Labelled model+market fusion, not pure "
        "model skill. sklearn_baseline remains a market-echo comparator "
        "(odds-as-features), not an edge claim.\n"
        "- Binary markets (do not rank against 1X2): dixon_coles_over_25 "
        "(market=over_under_25) and dixon_coles_btts (market=btts) are scored "
        "from the raw scoreline grid every fold. bookmaker_over_25 uses "
        "closing totals (`implied_over_25`); bookmaker_over_25_open uses "
        "pre-closing totals when present. dixon_coles_book_blend_over_25 is "
        "an optional prior-fold log-odds blend of grid over vs closing book "
        "over when both exist. Missing O/U odds drop a row from that book "
        "comparison only. Temperature scaling does not change O/U or BTTS.\n"
        "- ξ (`DIXON_COLES_XI`) is a frozen config constant from nested tune "
        "(`python -m src.evaluate --mode xi_tune`); not jointly estimated. "
        "See `results/XI_TUNE.md`.\n"
        "- Summary metrics: pooled over all scored matches (per-fold CSV for diagnosis). "
        "Blend rows also carry mean blend_alpha / blend_beta / blend_w_model.\n"
        "- Calibration plots: home-win reliability for 1X2 lines\n"
        "\n"
        "## Regenerate\n"
        "\n"
        "```bash\n"
        "source .venv/bin/activate\n"
        "python -m src.evaluate --mode walkforward\n"
        "```\n"
        "\n"
        "Season gate: `python -m src.evaluate --mode season`\n"
        "\n"
        "See docs/EVAL_PROTOCOL.md for interpretation.\n",
        encoding="utf-8",
    )
    paths["readme"] = readme
    return paths
