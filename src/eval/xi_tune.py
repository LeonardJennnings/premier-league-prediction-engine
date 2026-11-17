"""Nested walk-forward tune for Dixon–Coles time-decay ξ.

Protocol (single holdout recent era — no peeking):

1. Candidate ξ values are scored with a Dixon–Coles-only gameweek walk-forward.
2. Selection uses **tune** folds only (older seasons).
3. The winning ξ is frozen and scored on **holdout** folds (recent seasons).
4. Bookmaker closing 1X2 is reported on the same holdout matches for comparison.
5. ξ is never chosen by joint MLE inside the match likelihood, and never
   re-selected on the holdout / headline folds after the fact.

See ``docs/EVAL_PROTOCOL.md`` (§ Nested ξ tune) and ``results/XI_TUNE.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DIXON_COLES_XI_BASELINE,
    RESULTS,
    XI_TUNE_CANDIDATES,
    XI_TUNE_HOLDOUT_SEASONS,
    XI_TUNE_SEASONS,
)
from src.eval.gameweeks import assign_gameweeks
from src.eval.walkforward import MIN_TRAIN_ROWS, _metric_row, _probas_to_matrix, iter_folds
from src.evaluate import accuracy, brier_score, log_loss
from src.features import add_implied_odds
from src.models.dixon_coles import fit_dixon_coles
from src.models.dixon_coles import predict_match as predict_dixon_coles

logger = logging.getLogger(__name__)


def _holdout_honesty(beats_book: bool) -> str:
    if beats_book:
        return (
            "Frozen ξ log loss is below the closing book on this holdout — "
            "treat as provisional; do not advertise without nested / paired tests.\n"
        )
    return (
        "Frozen ξ **does not** beat the closing book on this holdout "
        "(expected for a goals-only model).\n"
    )


@dataclass(frozen=True)
class XiTuneResult:
    """Frozen ξ plus tune/holdout tables for the nested protocol."""

    chosen_xi: float
    baseline_xi: float
    candidate_tune: pd.DataFrame
    holdout_comparison: pd.DataFrame
    tune_seasons: tuple[str, ...]
    holdout_seasons: tuple[str, ...]
    run_date: str


def half_life_days(xi: float) -> float:
    """Days until a match weight halves under ``e^{-ξ t}``."""
    if xi <= 0:
        return float("inf")
    return float(np.log(2.0) / xi)


def _fold_season_set(seasons: tuple[str, ...] | list[str]) -> set[str]:
    return {str(s) for s in seasons}


def score_dixon_coles_folds(
    matches: pd.DataFrame,
    *,
    xi: float,
    seasons: set[str] | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
    run_at: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Walk-forward Dixon–Coles (+ closing book) on selected seasons' folds.

    Returns ``(dc_by_gw, book_by_gw, dc_preds)`` where ``dc_preds`` has one row
    per scored match with result and 1X2 probabilities. Book rows only cover
    matches with closing implied odds.
    """
    if "gameweek" not in matches.columns:
        matches = assign_gameweeks(matches)
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    # Point-in-time de-vig; safe to materialise once (no history join).
    if "implied_home" not in frame.columns:
        frame = add_implied_odds(frame)
    run_at = run_at or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    dc_rows: list[dict] = []
    book_rows: list[dict] = []
    pred_rows: list[dict] = []

    for train, test, season, gw in iter_folds(frame, min_train_rows=min_train_rows):
        if seasons is not None and str(season) not in seasons:
            continue

        y_all = test["result"].tolist()
        model = fit_dixon_coles(train, xi=xi)
        dc_match_rows = [
            predict_dixon_coles(model, row.home_team, row.away_team)
            for row in test.itertuples(index=False)
        ]
        dc_P = _probas_to_matrix(dc_match_rows)
        dc_rows.append(
            _metric_row(run_at, "dixon_coles", season, gw, len(train), y_all, dc_P)
        )
        for i, row in enumerate(test.itertuples(index=False)):
            pred_rows.append(
                {
                    "season_code": season,
                    "gameweek": gw,
                    "result": row.result,
                    "p_home": dc_P[i, 0],
                    "p_draw": dc_P[i, 1],
                    "p_away": dc_P[i, 2],
                }
            )

        book_cols = ["implied_home", "implied_draw", "implied_away", "result"]
        if not all(c in test.columns for c in book_cols):
            continue
        book_test = test.dropna(subset=book_cols)
        if not book_test.empty:
            book_P = book_test[
                ["implied_home", "implied_draw", "implied_away"]
            ].to_numpy(dtype=float)
            y_book = book_test["result"].tolist()
            book_rows.append(
                _metric_row(
                    run_at,
                    "bookmaker_implied",
                    season,
                    gw,
                    len(train),
                    y_book,
                    book_P,
                )
            )

    return pd.DataFrame(dc_rows), pd.DataFrame(book_rows), pd.DataFrame(pred_rows)


def _pooled_1x2(preds: pd.DataFrame) -> dict[str, float | int]:
    """Pooled log loss / Brier / accuracy over prediction rows."""
    if preds.empty:
        return {
            "n_test": 0,
            "log_loss": float("nan"),
            "brier": float("nan"),
            "accuracy": float("nan"),
        }
    y_true = preds["result"].tolist()
    P = preds[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    return {
        "n_test": int(len(preds)),
        "log_loss": round(log_loss(y_true, P), 4),
        "brier": round(brier_score(y_true, P), 4),
        "accuracy": round(accuracy(y_true, P), 4),
    }


def _pooled_from_by_gw(by_gw: pd.DataFrame) -> dict[str, float | int]:
    """Reconstruct pooled metrics from per-fold rows via prediction counts.

    Prefer :func:`_pooled_1x2` on raw prediction frames when available. This
    helper exists for book rows where we only keep fold metrics.
    """
    if by_gw.empty:
        return {
            "n_folds": 0,
            "n_test": 0,
            "log_loss": float("nan"),
            "brier": float("nan"),
            "accuracy": float("nan"),
        }
    n_test = int(by_gw["n_test"].sum())
    # n-weighted means match pooled averages for these proper scores.
    w = by_gw["n_test"].to_numpy(dtype=float)
    return {
        "n_folds": int(len(by_gw)),
        "n_test": n_test,
        "log_loss": round(float(np.average(by_gw["log_loss"], weights=w)), 4),
        "brier": round(float(np.average(by_gw["brier"], weights=w)), 4),
        "accuracy": round(float(np.average(by_gw["accuracy"], weights=w)), 4),
    }


def run_xi_tune(
    matches: pd.DataFrame | None = None,
    *,
    candidates: list[float] | tuple[float, ...] | None = None,
    tune_seasons: list[str] | tuple[str, ...] | None = None,
    holdout_seasons: list[str] | tuple[str, ...] | None = None,
    baseline_xi: float | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> XiTuneResult:
    """Select ξ on tune folds; freeze; score holdout (DC vs book).

    Selection criterion: lowest pooled 1X2 log loss on the tune era.
    Ties break toward the candidate closest to ``baseline_xi``, then lower ξ.
    """
    from src.clean import build_matches_table

    if matches is None:
        matches = build_matches_table()
    matches = assign_gameweeks(matches)
    matches["date"] = pd.to_datetime(matches["date"])
    if "implied_home" not in matches.columns:
        matches = add_implied_odds(matches)

    candidates = tuple(
        float(x) for x in (candidates if candidates is not None else XI_TUNE_CANDIDATES)
    )
    if not candidates:
        raise ValueError("candidates must be non-empty")
    for xi in candidates:
        if not np.isfinite(xi) or xi < 0:
            raise ValueError(f"xi candidates must be finite and >= 0; got {xi}")

    tune_seasons_t = tuple(
        str(s) for s in (tune_seasons if tune_seasons is not None else XI_TUNE_SEASONS)
    )
    holdout_seasons_t = tuple(
        str(s)
        for s in (
            holdout_seasons if holdout_seasons is not None else XI_TUNE_HOLDOUT_SEASONS
        )
    )
    if set(tune_seasons_t) & set(holdout_seasons_t):
        raise ValueError(
            f"tune and holdout seasons must be disjoint; overlap="
            f"{sorted(set(tune_seasons_t) & set(holdout_seasons_t))}"
        )

    # Before/after uses the pre-tune baseline, not the already-frozen config ξ.
    baseline = float(
        DIXON_COLES_XI_BASELINE if baseline_xi is None else baseline_xi
    )
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tune_set = _fold_season_set(tune_seasons_t)
    holdout_set = _fold_season_set(holdout_seasons_t)

    tune_rows: list[dict] = []
    for xi in candidates:
        logger.info("Scoring ξ=%.6f on tune seasons %s", xi, tune_seasons_t)
        _dc_by_gw, _book_by_gw, preds = score_dixon_coles_folds(
            matches,
            xi=xi,
            seasons=tune_set,
            min_train_rows=min_train_rows,
            run_at=run_at,
        )
        pooled = _pooled_1x2(preds)
        tune_rows.append(
            {
                "run_date": run_at,
                "xi": xi,
                "half_life_days": round(half_life_days(xi), 1),
                "era": "tune",
                "seasons": ",".join(tune_seasons_t),
                "n_folds": int(_dc_by_gw["gameweek"].count()) if not _dc_by_gw.empty else 0,
                "n_test": pooled["n_test"],
                "log_loss": pooled["log_loss"],
                "brier": pooled["brier"],
                "accuracy": pooled["accuracy"],
                "is_baseline": bool(np.isclose(xi, baseline)),
            }
        )

    candidate_tune = pd.DataFrame(tune_rows)
    if candidate_tune.empty or candidate_tune["n_test"].sum() == 0:
        raise ValueError("no tune folds scored — check seasons / min_train_rows / data")

    # Lowest tune log loss; tie-break toward baseline, then smaller ξ.
    ranked = candidate_tune.sort_values(
        by=["log_loss", "is_baseline", "xi"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    chosen_xi = float(ranked.loc[0, "xi"])
    logger.info("Frozen ξ=%.6f (tune LL=%.4f)", chosen_xi, ranked.loc[0, "log_loss"])

    holdout_lines: list[dict] = []
    # Score baseline and chosen (dedupe if equal) plus book on holdout.
    holdout_xis = []
    for xi in (baseline, chosen_xi):
        if not any(np.isclose(xi, existing) for existing in holdout_xis):
            holdout_xis.append(xi)

    book_pooled: dict[str, float | int] | None = None
    for xi in holdout_xis:
        logger.info("Scoring ξ=%.6f on holdout seasons %s", xi, holdout_seasons_t)
        dc_by_gw, book_by_gw, preds = score_dixon_coles_folds(
            matches,
            xi=xi,
            seasons=holdout_set,
            min_train_rows=min_train_rows,
            run_at=run_at,
        )
        pooled = _pooled_1x2(preds)
        label = "dixon_coles_chosen" if np.isclose(xi, chosen_xi) else "dixon_coles_baseline"
        if np.isclose(xi, chosen_xi) and np.isclose(xi, baseline):
            label = "dixon_coles_chosen"
        holdout_lines.append(
            {
                "run_date": run_at,
                "line": label,
                "xi": xi,
                "half_life_days": round(half_life_days(xi), 1),
                "era": "holdout",
                "seasons": ",".join(holdout_seasons_t),
                "n_folds": int(len(dc_by_gw)),
                "n_test": pooled["n_test"],
                "log_loss": pooled["log_loss"],
                "brier": pooled["brier"],
                "accuracy": pooled["accuracy"],
            }
        )
        if book_pooled is None and not book_by_gw.empty:
            book_pooled = _pooled_from_by_gw(book_by_gw)

    if book_pooled is not None:
        holdout_lines.append(
            {
                "run_date": run_at,
                "line": "bookmaker_implied",
                "xi": float("nan"),
                "half_life_days": float("nan"),
                "era": "holdout",
                "seasons": ",".join(holdout_seasons_t),
                "n_folds": book_pooled["n_folds"],
                "n_test": book_pooled["n_test"],
                "log_loss": book_pooled["log_loss"],
                "brier": book_pooled["brier"],
                "accuracy": book_pooled["accuracy"],
            }
        )

    holdout_comparison = pd.DataFrame(holdout_lines)
    return XiTuneResult(
        chosen_xi=chosen_xi,
        baseline_xi=baseline,
        candidate_tune=candidate_tune,
        holdout_comparison=holdout_comparison,
        tune_seasons=tune_seasons_t,
        holdout_seasons=holdout_seasons_t,
        run_date=run_at,
    )


def write_xi_tune_artefacts(
    result: XiTuneResult,
    results_dir: Path | None = None,
) -> dict[str, Path]:
    """Write tune grid, holdout comparison, and protocol note under results/."""
    results_dir = Path(results_dir or RESULTS)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    tune_path = results_dir / "xi_tune_candidates.csv"
    holdout_path = results_dir / "xi_tune_holdout.csv"
    result.candidate_tune.to_csv(tune_path, index=False)
    result.holdout_comparison.to_csv(holdout_path, index=False)
    paths["candidates"] = tune_path
    paths["holdout"] = holdout_path

    chosen_row = result.candidate_tune.loc[
        np.isclose(result.candidate_tune["xi"], result.chosen_xi)
    ].iloc[0]
    book_hold = result.holdout_comparison[
        result.holdout_comparison["line"] == "bookmaker_implied"
    ]

    def _fmt_row(row: pd.Series | None, label: str) -> str:
        if row is None or (isinstance(row, pd.DataFrame) and row.empty):
            return f"| {label} | — | — | — | — | — |"
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        xi_s = "—" if pd.isna(row.get("xi")) else f"{float(row['xi']):.4f}"
        return (
            f"| {label} | {xi_s} | {row['n_test']} | "
            f"{row['log_loss']:.4f} | {row['brier']:.4f} | {row['accuracy']:.4f} |"
        )

    chosen_hold = result.holdout_comparison[
        result.holdout_comparison["line"] == "dixon_coles_chosen"
    ]
    base_hold = result.holdout_comparison[
        result.holdout_comparison["line"] == "dixon_coles_baseline"
    ]

    beats_book = False
    if not chosen_hold.empty and not book_hold.empty:
        beats_book = float(chosen_hold.iloc[0]["log_loss"]) < float(
            book_hold.iloc[0]["log_loss"]
        )

    note = results_dir / "XI_TUNE.md"
    note.write_text(
        "# Nested ξ tune artefacts\n"
        "\n"
        "## Protocol\n"
        "\n"
        "- **Selection:** gameweek walk-forward Dixon–Coles 1X2 log loss on "
        f"**tune** seasons `{', '.join(result.tune_seasons)}` only.\n"
        "- **Freeze:** lowest tune pooled log loss; ties → nearer baseline ξ, "
        "then smaller ξ.\n"
        f"- **Holdout report:** seasons `{', '.join(result.holdout_seasons)}` "
        "(not used for selection).\n"
        "- **Not used:** joint MLE of ξ inside the match likelihood; re-tuning "
        "on holdout / headline folds after the fact.\n"
        f"- **Baseline ξ (pre-tune):** `{result.baseline_xi}` "
        f"(~{half_life_days(result.baseline_xi):.0f}-day half-life).\n"
        f"- **Frozen ξ:** `{result.chosen_xi}` "
        f"(~{half_life_days(result.chosen_xi):.0f}-day half-life); "
        f"tune LL `{chosen_row['log_loss']}` on `{int(chosen_row['n_test'])}` matches.\n"
        "- Book line on holdout is **Bet365 closing** de-vigged 1X2 "
        "(`bookmaker_implied`). Opening odds are a separate comparator "
        "(see walk-forward artefacts).\n"
        "\n"
        "## Holdout comparison (1X2)\n"
        "\n"
        "| Line | ξ | n | Log loss | Brier | Acc |\n"
        "|---|---|---|---|---|---|\n"
        f"{_fmt_row(chosen_hold, 'DC frozen ξ')}\n"
        f"{_fmt_row(base_hold if not base_hold.empty else None, 'DC baseline ξ')}\n"
        f"{_fmt_row(book_hold, 'Closing book')}\n"
        "\n"
        f"{_holdout_honesty(beats_book)}"
        "\n"
        "## Regenerate\n"
        "\n"
        "```bash\n"
        "source .venv/bin/activate\n"
        "python -m src.evaluate --mode xi_tune\n"
        "```\n"
        "\n"
        "Candidate grid and season split live in `src/config.py` "
        "(`XI_TUNE_*`). Freeze the tune winner as `DIXON_COLES_XI`; keep the "
        "pre-tune value in `DIXON_COLES_XI_BASELINE` for before/after tables. "
        "Do not re-select using holdout metrics.\n"
        "\n"
        "See `docs/EVAL_PROTOCOL.md` (§ Nested ξ tune).\n",
        encoding="utf-8",
    )
    paths["readme"] = note
    return paths
