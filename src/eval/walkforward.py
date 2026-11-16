"""Gameweek walk-forward evaluation runner."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS
from src.eval.calibration import plot_reliability, reliability_table
from src.eval.gameweeks import assign_gameweeks
from src.evaluate import accuracy, brier_score, log_loss
from src.features import features_for_fold
from src.models.baseline import fit_baseline, predict_match_features
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
) -> dict:
    return {
        "run_date": run_at,
        "model": model,
        "season_code": season,
        "gameweek": gw,
        "n_train": n_train,
        "n_test": len(y_true),
        "log_loss": round(log_loss(y_true, P), 4),
        "brier": round(brier_score(y_true, P), 4),
        "accuracy": round(accuracy(y_true, P), 4),
    }


def _pooled_summary(by_gw: pd.DataFrame, pred_frames: dict[str, pd.DataFrame], run_at: str) -> pd.DataFrame:
    """Research SoT row: pooled proper scores over all scored matches.

    Per-fold means can skew the headline when fold sizes differ (partial
    gameweeks, odds dropouts). Pooled log loss / Brier / accuracy use every
    prediction once. Per-fold tables remain in ``by_gw`` for diagnosis.
    """
    rows = []
    models = sorted(by_gw["model"].unique()) if not by_gw.empty else list(pred_frames)
    for model in models:
        frame = pred_frames.get(model, pd.DataFrame())
        fold_rows = by_gw[by_gw["model"] == model] if not by_gw.empty else pd.DataFrame()
        n_folds = int(len(fold_rows))
        n_test_folds = int(fold_rows["n_test"].sum()) if not fold_rows.empty else 0

        if frame.empty:
            rows.append(
                {
                    "run_date": run_at,
                    "model": model,
                    "n_folds": n_folds,
                    "n_test": n_test_folds,
                    "log_loss": float("nan"),
                    "brier": float("nan"),
                    "accuracy": float("nan"),
                    "aggregation": "pooled",
                }
            )
            continue

        y_true = frame["result"].tolist()
        P = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        rows.append(
            {
                "run_date": run_at,
                "model": model,
                "n_folds": n_folds,
                "n_test": len(y_true),
                "log_loss": round(log_loss(y_true, P), 4),
                "brier": round(brier_score(y_true, P), 4),
                "accuracy": round(accuracy(y_true, P), 4),
                "aggregation": "pooled",
            }
        )
    return pd.DataFrame(rows)


def run_walkforward(
    matches: pd.DataFrame | None = None,
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
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
        "sklearn_baseline": [],
        "bookmaker_implied": [],
    }

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
        if frame.empty:
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
        "- Models: naive_frequencies, poisson, sklearn_baseline, bookmaker_implied\n"
        "- Summary metrics: pooled over all scored matches (per-fold CSV for diagnosis)\n"
        "- Calibration plots: home-win reliability\n"
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
