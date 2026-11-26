"""Split the saved scoreline gap into rates and table shape.

The walk-forward file already scored the pure model and the closing
reconstruction. This module does not fit a new model. It rebuilds two
tables per match from those saved rows:

- the reconstruction's home and away rates, with the model's low-score
  correction
- the same rates, with the correction at zero

The gap that disappears when the reconstruction's rates are swapped in is
the rate piece. The gap that remains is the shape piece.

Rate-miss cuts are fixed at 0.15 and 0.40. They are not moved after the
grouped losses are seen. Groups that use different matches are reported
separately and are not ranked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from src.config import DIXON_COLES_MAX_GOALS, RESULTS
from src.eval.scoreline import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    block_bootstrap_mean_diff,
    score_scoreline,
)
from src.models.dixon_coles import calibrated_score_grid

# Fixed before the grouped losses were read. Do not move these after looking.
RATE_MISS_LOW = 0.15
RATE_MISS_HIGH = 0.40

# The plan's line for a gap that is too small to be the tenth of a nat.
SMALL_GAP = 0.01

_BIN_UNDER = "under 0.15"
_BIN_MID = "from 0.15 to 0.40"
_BIN_OVER = "over 0.40"
_BIN_ORDER = (_BIN_UNDER, _BIN_MID, _BIN_OVER)

_MODEL = "dixon_coles"
_MODEL_RHO0 = "dixon_coles_rho0"
_BOOK = "book_score_reconstruction"
_RHO_SEARCH = (-0.199, 0.199)


@dataclass(frozen=True)
class QuotedGap:
    """One mean, with the round-bootstrap range and the match count."""

    family: str
    group: str
    mean_diff: float
    ci_low: float
    ci_high: float
    n_matches: int
    n_rounds: int
    n_bootstrap: int
    seed: int
    note: str


@dataclass(frozen=True)
class GapSplitResult:
    """Quoted gaps from one saved walk-forward file. No new fit."""

    gaps: tuple[QuotedGap, ...]
    location: str
    run_date: str
    max_model_replay_error: float
    max_book_replay_error: float


def rate_miss_bin(abs_diff: float) -> str:
    """Map an absolute rate miss onto the three fixed cuts.

    0.15 and 0.40 belong to the middle group. Under is strict. Over is strict.
    """
    if not np.isfinite(abs_diff) or abs_diff < 0.0:
        raise ValueError("rate miss must be finite and non-negative")
    if abs_diff < RATE_MISS_LOW:
        return _BIN_UNDER
    if abs_diff <= RATE_MISS_HIGH:
        return _BIN_MID
    return _BIN_OVER


def locate_tenth(rate_mean: float, shape_mean: float) -> str:
    """Say whether the tenth sits in the rates, the shape, or both.

    A piece under 0.01 is not the tenth. That line is the plan's small-gap
    rule. It is not a cut chosen from these losses.
    """
    rate_small = abs(float(rate_mean)) < SMALL_GAP
    shape_small = abs(float(shape_mean)) < SMALL_GAP
    if shape_small and not rate_small:
        return "rates"
    if rate_small and not shape_small:
        return "shape"
    return "both"


def _truncated_mean(rate: float, max_goals: int) -> float:
    """Mean of a Poisson conditional on a score of at most ``max_goals``."""
    if rate < 0.0 or not np.isfinite(rate):
        raise ValueError("rate must be finite and non-negative")
    if rate == 0.0:
        return 0.0
    goals = np.arange(0, max_goals + 1, dtype=float)
    log_p = -rate + goals * np.log(rate) - gammaln(goals + 1.0)
    log_p -= log_p.max()
    weights = np.exp(log_p)
    weights /= weights.sum()
    return float(goals @ weights)


def rate_from_expected_goals(expected: float, max_goals: int = DIXON_COLES_MAX_GOALS) -> float:
    """Invert the 0-to-max Poisson mean.

    On an independent Poisson table the home mean depends only on the home
    rate, so the saved expected goals are enough to recover that rate.
    """
    if not np.isfinite(expected) or expected < 0.0:
        raise ValueError("expected goals must be finite and non-negative")
    if expected == 0.0:
        return 0.0
    ceiling = float(max_goals) - 1e-9
    if expected >= ceiling:
        raise ValueError("expected goals are too close to the grid cap to invert")
    lo, hi = 1e-8, 20.0
    if _truncated_mean(hi, max_goals) <= expected:
        raise ValueError("expected goals are above the searchable rate range")
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if _truncated_mean(mid, max_goals) < expected:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _cell_loss(lam: float, mu: float, rho: float, home_goals: int, away_goals: int) -> float:
    """Scoreline log loss of one table. The correction is projected so it stays admissible."""
    grid, used_lam, used_mu, used_rho = calibrated_score_grid(
        float(lam), float(mu), float(rho), rho=float(rho)
    )
    scored = score_scoreline(
        grid,
        int(home_goals),
        int(away_goals),
        lam=used_lam,
        mu=used_mu,
        rho=used_rho,
    )
    return float(scored["cell_log_loss"])


def _loss_errors(
    rho: float,
    lam: np.ndarray,
    mu: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    target_loss: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            _cell_loss(lam_i, mu_i, rho, int(home_i), int(away_i)) - float(target_i)
            for lam_i, mu_i, home_i, away_i, target_i in zip(
                lam, mu, home_goals, away_goals, target_loss, strict=True
            )
        ]
    )


def _recover_round_rho(
    lam: np.ndarray,
    mu: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    target_loss: np.ndarray,
) -> float:
    """One low-score correction for the round, chosen to rebuild the saved model loss.

    This is not a new fit of attack and defence. The rates are already fixed.
    When every correction in the search rebuilds the saved loss, the rows do
    not identify one, and the correction stays at zero.
    """
    zero_error = float(np.max(np.abs(_loss_errors(0.0, lam, mu, home_goals, away_goals, target_loss))))
    if zero_error < 1e-6:
        return 0.0

    def objective(rho: float) -> float:
        errors = _loss_errors(rho, lam, mu, home_goals, away_goals, target_loss)
        return float(np.dot(errors, errors))

    fitted = minimize_scalar(objective, bounds=_RHO_SEARCH, method="bounded", options={"xatol": 1e-10})
    rho = float(fitted.x)
    if abs(rho - _RHO_SEARCH[0]) < 1e-4 or abs(rho - _RHO_SEARCH[1]) < 1e-4:
        raise ValueError("low-score correction sat on the search boundary")
    worst = float(np.max(np.abs(_loss_errors(rho, lam, mu, home_goals, away_goals, target_loss))))
    if worst > 1e-6:
        raise ValueError("no low-score correction rebuilds the saved model loss")
    return rho


def _scoreline(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    part = frame.loc[(frame["market"] == "scoreline") & (frame["model"] == model)].copy()
    if part.empty:
        raise ValueError(f"no scoreline rows for {model}")
    if part["match_id"].duplicated().any():
        raise ValueError(f"duplicate matches for {model}")
    return part


def build_match_table(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per match: saved losses, recovered rates, and the two rebuilt tables."""
    model = _scoreline(scores, _MODEL)
    rho0 = _scoreline(scores, _MODEL_RHO0)
    book = _scoreline(scores, _BOOK)
    checks = scores.loc[
        (scores["market"] == "book_grid_check") & (scores["model"] == _BOOK),
        ["match_id", "lam", "mu"],
    ].rename(columns={"lam": "book_lam", "mu": "book_mu"})
    if checks.empty:
        raise ValueError("closing reconstruction rates are missing")
    if checks["match_id"].duplicated().any():
        raise ValueError("duplicate closing reconstruction rates")

    model_ids = set(model["match_id"])
    if model_ids != set(book["match_id"]) or model_ids != set(rho0["match_id"]):
        raise ValueError("model and closing reconstruction were scored on different matches")
    if model_ids != set(checks["match_id"]):
        raise ValueError("reconstruction rates do not cover the scored matches")

    joined = model.merge(rho0, on="match_id", suffixes=("", "_rho0"))
    joined = joined.merge(
        book[["match_id", "loss", "round_id", "home_goals", "away_goals"]],
        on="match_id",
        suffixes=("", "_book"),
    )
    joined = joined.merge(checks, on="match_id")
    if not joined["round_id"].equals(joined["round_id_rho0"]) or not joined["round_id"].equals(
        joined["round_id_book"]
    ):
        raise ValueError("round ids disagree across the saved lines")
    if not np.allclose(joined["home_goals"], joined["home_goals_book"]) or not np.allclose(
        joined["away_goals"], joined["away_goals_book"]
    ):
        raise ValueError("goals disagree across the saved lines")
    if not np.allclose(joined["exp_home"], joined["exp_home_rho0"], atol=1e-8) or not np.allclose(
        joined["exp_away"], joined["exp_away_rho0"], atol=1e-8
    ):
        raise ValueError("model expected goals disagree between the two saved model lines")

    model_lam = np.array(
        [rate_from_expected_goals(float(value)) for value in joined["exp_home_rho0"].to_numpy()],
        dtype=float,
    )
    model_mu = np.array(
        [rate_from_expected_goals(float(value)) for value in joined["exp_away_rho0"].to_numpy()],
        dtype=float,
    )
    book_lam = joined["book_lam"].to_numpy(dtype=float)
    book_mu = joined["book_mu"].to_numpy(dtype=float)
    home_goals = joined["home_goals"].to_numpy(dtype=int)
    away_goals = joined["away_goals"].to_numpy(dtype=int)
    model_loss = joined["loss"].to_numpy(dtype=float)
    book_loss = joined["loss_book"].to_numpy(dtype=float)

    book_replay = np.array(
        [
            _cell_loss(lam, mu, 0.0, home, away)
            for lam, mu, home, away in zip(book_lam, book_mu, home_goals, away_goals, strict=True)
        ]
    )
    book_error = float(np.max(np.abs(book_replay - book_loss)))
    if book_error > 1e-8:
        raise ValueError(
            "rebuilding the closing table from its saved rates does not match the saved loss"
        )

    joined = joined.reset_index(drop=True)
    rho = np.empty(len(joined), dtype=float)
    for _, index in joined.groupby("round_id", sort=False).groups.items():
        positions = np.asarray(index, dtype=int)
        rho[positions] = _recover_round_rho(
            model_lam[positions],
            model_mu[positions],
            home_goals[positions],
            away_goals[positions],
            model_loss[positions],
        )

    model_replay = np.array(
        [
            _cell_loss(lam, mu, rho_i, home, away)
            for lam, mu, rho_i, home, away in zip(
                model_lam, model_mu, rho, home_goals, away_goals, strict=True
            )
        ]
    )
    model_error = float(np.max(np.abs(model_replay - model_loss)))
    if model_error > 1e-6:
        raise ValueError("recovered low-score correction does not rebuild the saved model loss")

    hybrid = np.array(
        [
            _cell_loss(lam, mu, rho_i, home, away)
            for lam, mu, rho_i, home, away in zip(
                book_lam, book_mu, rho, home_goals, away_goals, strict=True
            )
        ]
    )
    total_gap = model_loss - book_loss
    rate_gap = model_loss - hybrid
    shape_gap = hybrid - book_loss
    if float(np.max(np.abs(rate_gap + shape_gap - total_gap))) > 1e-9:
        raise ValueError("rate piece and shape piece do not add to the saved gap")

    abs_total = np.abs((model_lam + model_mu) - (book_lam + book_mu))
    abs_split = np.abs((model_lam - model_mu) - (book_lam - book_mu))
    table = pd.DataFrame(
        {
            "match_id": joined["match_id"].to_numpy(),
            "round_id": joined["round_id"].to_numpy(),
            "season_code": joined["season_code"].astype(str).to_numpy(),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "model_lam": model_lam,
            "model_mu": model_mu,
            "book_lam": book_lam,
            "book_mu": book_mu,
            "rho": rho,
            "model_loss": model_loss,
            "book_loss": book_loss,
            "hybrid_loss": hybrid,
            "total_gap": total_gap,
            "rate_gap": rate_gap,
            "shape_gap": shape_gap,
            "abs_total_rate": abs_total,
            "abs_rate_split": abs_split,
            "total_rate_bin": [rate_miss_bin(float(value)) for value in abs_total],
            "rate_split_bin": [rate_miss_bin(float(value)) for value in abs_split],
            "model_replay_error": np.abs(model_replay - model_loss),
            "book_replay_error": np.abs(book_replay - book_loss),
        }
    )
    return table


def quote_gap(
    family: str,
    group: str,
    diff: np.ndarray,
    round_ids: np.ndarray,
    *,
    note: str,
    n_boot: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> QuotedGap:
    """Mean of ``diff`` with a 95% range from resampling whole rounds."""
    values = np.asarray(diff, dtype=float)
    rounds = np.asarray(round_ids)
    if values.size == 0:
        return QuotedGap(
            family=family,
            group=group,
            mean_diff=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_matches=0,
            n_rounds=0,
            n_bootstrap=0,
            seed=int(seed),
            note=note,
        )
    mean, low, high, n_rounds = block_bootstrap_mean_diff(
        values, rounds, n_boot=n_boot, seed=seed
    )
    return QuotedGap(
        family=family,
        group=group,
        mean_diff=round(mean, 4),
        ci_low=round(low, 4),
        ci_high=round(high, 4),
        n_matches=int(values.size),
        n_rounds=int(n_rounds),
        n_bootstrap=int(n_boot),
        seed=int(seed),
        note=note,
    )


_LOSS_NOTE = "A positive number means the model has the higher scoreline log loss."
_HOME_NOTE = "A positive number means the model home rate is higher than the reconstruction home rate."


def _subset_gap(
    table: pd.DataFrame,
    mask: np.ndarray,
    column: str,
    family: str,
    group: str,
    note: str,
    *,
    n_boot: int,
    seed: int,
) -> QuotedGap:
    part = table.loc[mask]
    return quote_gap(
        family,
        group,
        part[column].to_numpy(dtype=float),
        part["round_id"].to_numpy(),
        note=note,
        n_boot=n_boot,
        seed=seed,
    )


def summarise_match_table(
    table: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    run_date: str | None = None,
) -> GapSplitResult:
    """Quote the unsplit gap, the rate and shape pieces, and the fixed groups."""
    if table.empty:
        raise ValueError("match table is empty")
    gaps: list[QuotedGap] = [
        quote_gap(
            "same_matches",
            "model minus closing reconstruction",
            table["total_gap"].to_numpy(dtype=float),
            table["round_id"].to_numpy(),
            note=_LOSS_NOTE,
            n_boot=n_boot,
            seed=seed,
        ),
        quote_gap(
            "same_matches",
            "rate piece",
            table["rate_gap"].to_numpy(dtype=float),
            table["round_id"].to_numpy(),
            note=_LOSS_NOTE,
            n_boot=n_boot,
            seed=seed,
        ),
        quote_gap(
            "same_matches",
            "shape piece",
            table["shape_gap"].to_numpy(dtype=float),
            table["round_id"].to_numpy(),
            note=_LOSS_NOTE,
            n_boot=n_boot,
            seed=seed,
        ),
    ]
    for season in sorted(table["season_code"].astype(str).unique()):
        mask = table["season_code"].astype(str).to_numpy() == season
        gaps.append(
            _subset_gap(
                table,
                mask,
                "total_gap",
                "season",
                season,
                _LOSS_NOTE,
                n_boot=n_boot,
                seed=seed,
            )
        )
    for family, column in (
        ("total_rate", "total_rate_bin"),
        ("rate_split", "rate_split_bin"),
    ):
        labels = table[column].to_numpy()
        for group in _BIN_ORDER:
            gaps.append(
                _subset_gap(
                    table,
                    labels == group,
                    "total_gap",
                    family,
                    group,
                    _LOSS_NOTE,
                    n_boot=n_boot,
                    seed=seed,
                )
            )
    home_error = table["model_lam"].to_numpy(dtype=float) - table["book_lam"].to_numpy(dtype=float)
    for season in sorted(table["season_code"].astype(str).unique()):
        mask = table["season_code"].astype(str).to_numpy() == season
        gaps.append(
            quote_gap(
                "home_rate",
                season,
                home_error[mask],
                table.loc[mask, "round_id"].to_numpy(),
                note=_HOME_NOTE,
                n_boot=n_boot,
                seed=seed,
            )
        )
    rate = gaps[1]
    shape = gaps[2]
    if run_date is None:
        run_date = datetime.now(ZoneInfo("Europe/London")).strftime("%d %b %Y").lstrip("0")
    return GapSplitResult(
        gaps=tuple(gaps),
        location=locate_tenth(rate.mean_diff, shape.mean_diff),
        run_date=run_date,
        max_model_replay_error=float(table["model_replay_error"].max()),
        max_book_replay_error=float(table["book_replay_error"].max()),
    )


def _lookup(result: GapSplitResult, family: str, group: str) -> QuotedGap:
    for gap in result.gaps:
        if gap.family == family and gap.group == group:
            return gap
    raise KeyError(f"missing gap {family}/{group}")


def _fmt(gap: QuotedGap) -> str:
    if gap.n_matches == 0 or not np.isfinite(gap.mean_diff):
        return "no matches"
    return (
        f"{gap.mean_diff:.4f} "
        f"(95% range {gap.ci_low:.4f} to {gap.ci_high:.4f}, "
        f"{gap.n_matches} matches, {gap.n_rounds} rounds)"
    )


def _range_clears_zero(gap: QuotedGap) -> bool:
    if gap.n_matches == 0 or not np.isfinite(gap.ci_low) or not np.isfinite(gap.ci_high):
        return False
    return gap.ci_low > 0.0 or gap.ci_high < 0.0


def _table(gaps: list[QuotedGap]) -> str:
    lines = [
        "| Group | Mean | 95% range | Matches | Rounds |",
        "|---|---|---|---|---|",
    ]
    for gap in gaps:
        if gap.n_matches == 0 or not np.isfinite(gap.mean_diff):
            lines.append(f"| {gap.group} |  |  | 0 | 0 |")
            continue
        lines.append(
            f"| {gap.group} | {gap.mean_diff:.4f} | "
            f"{gap.ci_low:.4f} to {gap.ci_high:.4f} | "
            f"{gap.n_matches} | {gap.n_rounds} |"
        )
    return "\n".join(lines)


def _season_list(names: list[str]) -> str:
    if len(names) <= 1:
        return names[0] if names else ""
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + " and " + names[-1]


def _home_sentence(result: GapSplitResult) -> str:
    home = [gap for gap in result.gaps if gap.family == "home_rate"]
    below = [gap.group for gap in home if gap.ci_high < 0.0]
    above = [gap.group for gap in home if gap.ci_low > 0.0]
    unclear = [gap.group for gap in home if not _range_clears_zero(gap)]
    parts: list[str] = []
    if len(below) >= 2:
        parts.append(
            "The model home rate is lower than the reconstruction home rate, "
            f"with the range clear of zero, in seasons {_season_list(below)}."
        )
    elif len(above) >= 2:
        parts.append(
            "The model home rate is higher than the reconstruction home rate, "
            f"with the range clear of zero, in seasons {_season_list(above)}."
        )
    else:
        parts.append(
            "No two seasons show a home-rate difference clear of zero in the same direction."
        )
    if unclear:
        parts.append(f"In seasons {_season_list(unclear)} that range includes zero.")
    return " ".join(parts)


def _decision_paragraph(result: GapSplitResult) -> str:
    total = _lookup(result, "same_matches", "model minus closing reconstruction")
    rate = _lookup(result, "same_matches", "rate piece")
    shape = _lookup(result, "same_matches", "shape piece")
    if result.location == "rates":
        opening = (
            "The tenth is in the rates. "
            "The piece that remains after the reconstruction's rates are swapped in is under 0.01, "
            "so it is not the tenth. "
            "The piece that disappears with that swap is the tenth. "
            "A later change to the goal rates is aimed at this gap. "
            "A later change to the low-score correction is aimed at a piece under 0.01."
        )
    elif result.location == "shape":
        opening = (
            "The tenth is in the shape of the table. "
            "It remains after the reconstruction's rates are swapped in. "
            "The piece that disappears with that swap is under 0.01. "
            "A later change to the goal rates is aimed at a piece under 0.01. "
            "A later change to the low-score correction is aimed at this gap."
        )
    else:
        opening = (
            "The tenth is in both the rates and the shape. "
            "Neither piece is under 0.01. "
            "A later change to the goal rates and a later change to the low-score correction "
            "are both aimed at a piece of this gap."
        )
    return (
        f"{opening} "
        f"On the same matches, model minus closing reconstruction is {_fmt(total)}. "
        f"The rate piece is {_fmt(rate)}. "
        f"The shape piece is {_fmt(shape)}. "
        f"{_home_sentence(result)} "
        "The closing reconstruction has the lower scoreline loss on these matches. "
        "Every coefficient stays as it is."
    )


def render_gap_split(result: GapSplitResult) -> str:
    """Plain-language note. Every quoted gap has a range and a match count."""
    total = _lookup(result, "same_matches", "model minus closing reconstruction")
    rate = _lookup(result, "same_matches", "rate piece")
    shape = _lookup(result, "same_matches", "shape piece")
    seasons = [gap for gap in result.gaps if gap.family == "season"]
    total_rate = [gap for gap in result.gaps if gap.family == "total_rate"]
    rate_split = [gap for gap in result.gaps if gap.family == "rate_split"]
    home = [gap for gap in result.gaps if gap.family == "home_rate"]
    lines = [
        "# Where the scoreline gap sits",
        "",
        f"**Date:** {result.run_date} (London)",
        "",
        (
            "This note reads the saved walk-forward rows in "
            "`results/walkforward_match_scores.csv`. The fitted model stays as it is. "
            "The gap is Dixon-Coles scoreline "
            "log loss minus the loss of the closing reconstruction. A positive number "
            "means Dixon-Coles put less probability on the score that happened. "
            "The closing reconstruction is our reconstruction of the book's score table "
            "from its prices, not the bookmaker's correct-score odds."
        ),
        "",
        "## How the split is built",
        "",
        (
            "The saved file has the scoreline loss and the expected goals for Dixon-Coles, "
            "and the home and away rates for the closing reconstruction. "
            "The model rates are the Poisson rates that produce those expected goals on a "
            "0-to-8 table with the low-score correction at zero. "
            "The low-score correction is the one value per round that rebuilds the saved "
            "Dixon-Coles loss from those rates. Attack and defence are not fit again. "
            f"Rebuilding the closing table from its saved rates matches the saved loss "
            f"(largest absolute difference {result.max_book_replay_error:.3g}). "
            f"Rebuilding Dixon-Coles from the recovered rates and correction matches the "
            f"saved loss (largest absolute difference {result.max_model_replay_error:.3g})."
        ),
        "",
        (
            "The rate piece is the saved model loss minus the loss of a table that uses "
            "the reconstruction's rates and the model's low-score correction. "
            "That is the part of the gap that disappears when the book's rates are swapped in. "
            "The shape piece is that swapped table minus the closing reconstruction, "
            "which keeps the correction at zero. That is the part that remains. "
            "On each match the two pieces add to the saved gap."
        ),
        "",
        (
            f"The range is a 95% percentile interval from {total.n_bootstrap} resamples of "
            f"whole rounds, seed {total.seed}. "
            "A range that contains zero has not separated the two lines on that set of matches."
        ),
        "",
        "## Same matches",
        "",
        (
            f"These three numbers use the same {total.n_matches} matches and "
            f"{total.n_rounds} rounds. They are parts of one gap, not a ranking."
        ),
        "",
        _table([total, rate, shape]),
        "",
        "## Where the tenth sits",
        "",
        _decision_paragraph(result),
        "",
        "## By season",
        "",
        (
            "Each season is its own set of matches. The rows stay in season order. "
            "They are not ranked."
        ),
        "",
        _table(seasons),
        "",
        "## By the size of the total-rate miss",
        "",
        (
            "The total rate is the home rate plus the away rate. "
            "The miss is the absolute difference between Dixon-Coles and the closing reconstruction. "
            f"The cuts are under {RATE_MISS_LOW:.2f}, from {RATE_MISS_LOW:.2f} to {RATE_MISS_HIGH:.2f}, "
            f"and over {RATE_MISS_HIGH:.2f}. "
            "A miss of 0.15 or 0.40 sits in the middle group. "
            "The cuts were fixed before these losses were read. "
            "The three groups are different matches. They are not ranked."
        ),
        "",
        _table(total_rate),
        "",
        "## By the size of the rate-split miss",
        "",
        (
            "The rate split is the home rate minus the away rate. "
            "The miss is the absolute difference between Dixon-Coles and the closing reconstruction. "
            "The cuts are the same 0.15 and 0.40. "
            "The three groups are different matches. They are not ranked."
        ),
        "",
        _table(rate_split),
        "",
        "## Home rate by season",
        "",
        (
            "This block is not a scoreline gap. It is the model home rate minus the "
            "reconstruction home rate, on the matches of that season. "
            "A positive number means the model home rate is higher. "
            "Seasons are different matches and are not ranked against each other, "
            "or against the scoreline tables above."
        ),
        "",
        _table(home),
        "",
        "## Regenerate",
        "",
        "```bash",
        "python -m src.evaluate --mode gap_split",
        "```",
        "",
        "The grouped table is `results/gap_split.csv`.",
        "",
    ]
    text = "\n".join(lines)
    if "·" in text:
        raise ValueError("gap split note contains a mid-dot")
    return text


def gaps_frame(result: GapSplitResult) -> pd.DataFrame:
    """One row per quoted gap, in reading order. Not sorted by the mean."""
    return pd.DataFrame([gap.__dict__ for gap in result.gaps])


def _assert_published(result: GapSplitResult, paired_path: Path) -> None:
    """The unsplit gap has to reprint the committed walk-forward row."""
    paired = pd.read_csv(paired_path)
    row = paired.loc[
        (paired["metric"] == "scoreline_log_loss")
        & (paired["model_a"] == _MODEL)
        & (paired["model_b"] == _BOOK)
        & (paired["sample"] == "all")
    ]
    if len(row) != 1:
        raise ValueError("committed paired file has no single closing-reconstruction gap")
    published = row.iloc[0]
    gap = _lookup(result, "same_matches", "model minus closing reconstruction")
    checks = {
        "mean_diff": gap.mean_diff,
        "ci_low": gap.ci_low,
        "ci_high": gap.ci_high,
        "n_matches": gap.n_matches,
        "n_rounds": gap.n_rounds,
    }
    for key, value in checks.items():
        if value != published[key] and float(value) != float(published[key]):
            raise ValueError(
                "unsplit gap does not match the committed row "
                f"({key} {value} versus {published[key]}). The join is wrong."
            )
    if int(published["seed"]) != BOOTSTRAP_SEED or int(published["n_bootstrap"]) != BOOTSTRAP_DRAWS:
        raise ValueError("committed paired row does not use the published bootstrap")


def run_gap_split(
    path: Path | None = None,
    *,
    n_boot: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    check_published: bool = True,
) -> GapSplitResult:
    """Read the saved walk-forward file and split the gap. No refit."""
    scores_path = RESULTS / "walkforward_match_scores.csv" if path is None else Path(path)
    scores = pd.read_csv(scores_path)
    table = build_match_table(scores)
    result = summarise_match_table(table, n_boot=n_boot, seed=seed)
    if (
        check_published
        and n_boot == BOOTSTRAP_DRAWS
        and seed == BOOTSTRAP_SEED
        and scores_path.name == "walkforward_match_scores.csv"
    ):
        _assert_published(result, scores_path.parent / "walkforward_paired.csv")
    return result


def write_gap_split_artefacts(
    result: GapSplitResult,
    results_dir: Path | None = None,
) -> dict[str, str]:
    """Write the note and the grouped table. Does not change the model."""
    root = RESULTS if results_dir is None else Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    note = root / "GAP_SPLIT.md"
    table = root / "gap_split.csv"
    note.write_text(render_gap_split(result), encoding="utf-8")
    gaps_frame(result).to_csv(table, index=False)
    return {"note": str(note), "table": str(table)}
