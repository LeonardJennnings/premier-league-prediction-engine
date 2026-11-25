"""Scoreline metrics for the walk-forward backtest.

Cell log loss is the proper score of the full score table: minus the log of
the probability on the score that happened. A single picked score is only a
hit rate. Always saying 1-1 has no probability table, so it is a hit rate
only.

Goal-difference RPS is the ranked probability score on home goals minus away
goals, from -8 to +8. It is the sum of squared errors of the cumulative
probabilities, leaving out the last bin (that bin is always 1). Lower is
better. It is not divided by the number of bins.

Comparisons resample whole rounds, not single matches. The published range
is the 2.5 and 97.5 percentiles of that resampled mean.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import DIXON_COLES_MAX_GOALS, XI_TUNE_HOLDOUT_SEASONS, XI_TUNE_SEASONS
from src.eval.book_grid import BOOK_SCORE_TABLE_LABEL, fidelity_gap
from src.models.dixon_coles import scoreline_grid_from_rates, tau

# Resamples of whole rounds. Fixed seed so a committed range can be rebuilt.
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20261124

_LOG_LOSS_FLOOR = 1e-15


def poisson_pmf(goals: int, mean: float) -> float:
    """P(exactly ``goals``) for a Poisson with this mean."""
    if goals < 0:
        raise ValueError("goals must be non-negative")
    if mean < 0 or not math.isfinite(mean):
        raise ValueError("Poisson mean must be finite and non-negative")
    if mean == 0.0:
        return 1.0 if goals == 0 else 0.0
    return float(math.exp(-mean) * (mean**goals) / math.factorial(goals))


def exact_cell_probability(
    lam: float,
    mu: float,
    rho: float,
    home_goals: int,
    away_goals: int,
) -> float:
    """Untruncated probability of one exact score, including the low-score correction."""
    p = poisson_pmf(home_goals, lam) * poisson_pmf(away_goals, mu)
    p *= tau(int(home_goals), int(away_goals), float(lam), float(mu), float(rho))
    return float(p)


def empirical_frequency_grid(
    home_goals: Sequence[int] | np.ndarray,
    away_goals: Sequence[int] | np.ndarray,
    max_goals: int = DIXON_COLES_MAX_GOALS,
) -> np.ndarray:
    """Share of earlier matches at each score from 0 to ``max_goals``.

    A score with either side above ``max_goals`` is not a cell. Its count is
    left out of the grid, so the cells sum to less than 1 when that happens.
    A cell that has not occurred is 0. Callers score those with an explicit
    rule rather than inventing a probability.
    """
    home = np.asarray(home_goals, dtype=int)
    away = np.asarray(away_goals, dtype=int)
    if home.shape != away.shape:
        raise ValueError("home and away goal arrays must match")
    n = int(home.size)
    if n == 0:
        raise ValueError("frequency table needs at least one earlier match")
    grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    inside = (home >= 0) & (away >= 0) & (home <= max_goals) & (away <= max_goals)
    np.add.at(grid, (home[inside], away[inside]), 1.0)
    grid /= n
    return grid


def constant_poisson_grid(
    home_goals: Sequence[int] | np.ndarray,
    away_goals: Sequence[int] | np.ndarray,
    max_goals: int = DIXON_COLES_MAX_GOALS,
) -> np.ndarray:
    """One Poisson table from the training mean home and away rates. No team strength."""
    home = np.asarray(home_goals, dtype=float)
    away = np.asarray(away_goals, dtype=float)
    if home.size == 0 or away.size == 0:
        raise ValueError("constant Poisson needs at least one earlier match")
    lam = float(home.mean())
    mu = float(away.mean())
    if lam < 0.0 or mu < 0.0 or not math.isfinite(lam) or not math.isfinite(mu):
        raise ValueError("constant Poisson rates must be finite and non-negative")
    return scoreline_grid_from_rates(lam, mu, 0.0, max_goals=max_goals)


def goal_difference_distribution(grid: np.ndarray) -> np.ndarray:
    """P(home - away = d) for d = -max_goals .. +max_goals."""
    g = np.asarray(grid, dtype=float)
    max_goals = g.shape[0] - 1
    dist = np.zeros(2 * max_goals + 1, dtype=float)
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            dist[home - away + max_goals] += g[home, away]
    total = float(dist.sum())
    if total <= 0.0:
        raise ValueError("goal-difference distribution has no mass")
    return dist / total


def goal_difference_rps(grid: np.ndarray, home_goals: int, away_goals: int) -> float:
    """Ranked probability score of the goal difference. Lower is better.

    Differences outside -max..+max are clipped to the nearest end bin. The
    grid has no cell there, so the score uses the most extreme difference it
    can represent.
    """
    dist = goal_difference_distribution(grid)
    max_goals = grid.shape[0] - 1
    diff = int(home_goals) - int(away_goals)
    diff = max(-max_goals, min(max_goals, diff))
    outcome = diff + max_goals
    cdf = np.cumsum(dist)
    observed = np.zeros_like(dist)
    observed[outcome] = 1.0
    observed_cdf = np.cumsum(observed)
    return float(np.sum((cdf[:-1] - observed_cdf[:-1]) ** 2))


def _featured_scores(grid: np.ndarray, k: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Modal score, then the ``k`` highest cells. Ties keep the earlier cell."""
    flat = np.asarray(grid, dtype=float).ravel()
    if flat.size == 0:
        raise ValueError("empty score grid")
    order = np.argsort(-flat, kind="mergesort")
    width = grid.shape[1]
    top = [(int(idx // width), int(idx % width)) for idx in order[:k]]
    mode_idx = int(order[0])
    return (int(mode_idx // width), int(mode_idx % width)), top


def score_scoreline(
    grid: np.ndarray,
    home_goals: int,
    away_goals: int,
    *,
    lam: float | None = None,
    mu: float | None = None,
    rho: float = 0.0,
    top_k: int = 3,
    unseen_probability: float | None = None,
) -> dict[str, float]:
    """Score one match against a 0..max grid.

    A result inside the grid uses that cell. A result outside it (a 9-0, for
    example) uses the untruncated Poisson probability of that exact score when
    rates are supplied. That probability is finite. It is not moved into a
    neighbouring cell.

    A league-frequency cell that has never occurred is 0 on the grid. Pass
    ``unseen_probability`` (1 over one more than the number of earlier matches)
    so that new score still has a stated probability. Seen scores keep their
    raw share of earlier matches.
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("grid must be square")
    max_goals = g.shape[0] - 1
    home_goals = int(home_goals)
    away_goals = int(away_goals)
    inside = 0 <= home_goals <= max_goals and 0 <= away_goals <= max_goals
    if inside:
        p_cell = float(g[home_goals, away_goals])
        if p_cell <= 0.0 and unseen_probability is not None:
            p_cell = float(unseen_probability)
    else:
        if lam is not None and mu is not None:
            p_cell = exact_cell_probability(lam, mu, rho, home_goals, away_goals)
        elif unseen_probability is not None:
            p_cell = float(unseen_probability)
        else:
            raise ValueError("rates are required to score a result outside the grid")
    if not math.isfinite(p_cell) or p_cell < 0.0:
        raise ValueError("score probability must be finite and non-negative")
    if p_cell <= 0.0:
        cell_log_loss = float("inf")
    else:
        cell_log_loss = float(-math.log(max(p_cell, _LOG_LOSS_FLOOR)))

    total = float(g.sum())
    g_norm = g / total if total > 0.0 else g
    mode, top = _featured_scores(g_norm, top_k)
    actual = (home_goals, away_goals)
    goals = np.arange(g_norm.shape[0], dtype=float)
    exp_home = float(g_norm.sum(axis=1) @ goals)
    exp_away = float(g_norm.sum(axis=0) @ goals)
    err_home = exp_home - home_goals
    err_away = exp_away - away_goals
    return {
        "p_cell": p_cell,
        "cell_log_loss": cell_log_loss,
        "modal_hit": float(actual == mode),
        "top3_hit": float(actual in top),
        "exp_home": exp_home,
        "exp_away": exp_away,
        "abs_err_home": abs(err_home),
        "abs_err_away": abs(err_away),
        "sq_err_home": err_home**2,
        "sq_err_away": err_away**2,
        "abs_err_total": abs((exp_home + exp_away) - (home_goals + away_goals)),
        "rps": goal_difference_rps(g_norm, home_goals, away_goals),
    }


def assert_one_match_set(rows: pd.DataFrame, *, n_col: str = "n_test") -> int:
    """Raise if a scoreline table mixes lines scored on different matches."""
    if rows.empty:
        raise ValueError("scoreline table is empty")
    counts = {int(n) for n in rows[n_col].tolist()}
    if len(counts) != 1:
        raise ValueError(
            "scoreline rows were scored on different matches: "
            + ", ".join(str(n) for n in sorted(counts))
        )
    return counts.pop()


def summarize_scoreline(match_scores: pd.DataFrame, *, run_date: str) -> pd.DataFrame:
    """One pooled row per scoreline model. Every row uses the same matches."""
    frame = match_scores.loc[match_scores["market"] == "scoreline"].copy()
    if frame.empty:
        raise ValueError("no scoreline rows to summarize")
    rows: list[dict] = []
    for model, part in frame.groupby("model", sort=True):
        hit = float(part["hit"].mean())
        is_point_mass = model == "always_1_1"
        losses = part["loss"].to_numpy(dtype=float)
        if is_point_mass:
            log_loss = float("nan")
        elif not np.isfinite(losses).all():
            log_loss = float("inf")
        else:
            log_loss = round(float(losses.mean()), 4)
        finite_sq_home = part["sq_err_home"].to_numpy(dtype=float)
        finite_sq_away = part["sq_err_away"].to_numpy(dtype=float)

        def _rmse(values: np.ndarray) -> float:
            if not np.isfinite(values).any():
                return float("nan")
            return round(float(np.sqrt(np.nanmean(values))), 4)

        rows.append(
            {
                "run_date": run_date,
                "model": model,
                "market": "scoreline",
                "n_folds": int(part["round_id"].nunique()),
                "n_test": int(len(part)),
                "log_loss": log_loss,
                "brier": float("nan"),
                "accuracy": round(hit, 4),
                "aggregation": "pooled",
                "modal_hit": round(hit, 4),
                "top3_hit": (
                    float("nan")
                    if part["top3_hit"].isna().all()
                    else round(float(part["top3_hit"].mean()), 4)
                ),
                "mae_home": round(float(part["abs_err_home"].mean()), 4),
                "mae_away": round(float(part["abs_err_away"].mean()), 4),
                "rmse_home": _rmse(finite_sq_home),
                "rmse_away": _rmse(finite_sq_away),
                "mae_total": round(float(part["abs_err_total"].mean()), 4),
                "rps": (
                    float("nan")
                    if part["rps"].isna().all()
                    else round(float(part["rps"].mean()), 4)
                ),
            }
        )
    summary = pd.DataFrame(rows)
    assert_one_match_set(summary)
    return summary


def _align_pair(left: pd.DataFrame, right: pd.DataFrame, model_a: str, model_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the shared matches. Raise if each model has matches the other lacks."""
    if left["match_id"].duplicated().any() or right["match_id"].duplicated().any():
        raise ValueError(f"duplicate matches in {model_a} or {model_b}")
    ids_a = set(left["match_id"])
    ids_b = set(right["match_id"])
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a
    if only_a and only_b:
        raise ValueError(
            f"{model_a} and {model_b} were scored on different matches "
            f"({len(only_a)} only in {model_a}, {len(only_b)} only in {model_b})"
        )
    common = ids_a & ids_b
    if not common:
        raise ValueError(f"{model_a} and {model_b} have no matches in common")
    left = left.loc[left["match_id"].isin(common)].sort_values("match_id")
    right = right.loc[right["match_id"].isin(common)].sort_values("match_id")
    if not left["round_id"].reset_index(drop=True).equals(right["round_id"].reset_index(drop=True)):
        raise ValueError(f"round ids disagree for {model_a} vs {model_b}")
    return left.reset_index(drop=True), right.reset_index(drop=True)


def block_bootstrap_mean_diff(
    diff: np.ndarray,
    round_ids: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, int]:
    """Mean of ``diff``, and a 95% range from resampling whole rounds.

    Returns mean, low, high, and the number of rounds.
    """
    values = np.asarray(diff, dtype=float)
    rounds = np.asarray(round_ids)
    if values.shape[0] != rounds.shape[0] or values.size == 0:
        raise ValueError("diff and round_ids must be non-empty and the same length")
    if not np.isfinite(values).all():
        raise ValueError("bootstrap diff has a non-finite value")
    uniq, inverse = np.unique(rounds, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n_rounds = len(groups)
    for draw_i in range(n_boot):
        chosen = rng.integers(0, n_rounds, size=n_rounds)
        sample = np.concatenate([values[groups[i]] for i in chosen])
        means[draw_i] = float(sample.mean())
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high), n_rounds


def _subset_for_sample(frame: pd.DataFrame, sample: str) -> pd.DataFrame:
    if sample == "all":
        return frame
    if sample == "holdout":
        seasons = set(XI_TUNE_HOLDOUT_SEASONS)
    elif sample == "tune":
        seasons = set(XI_TUNE_SEASONS)
    else:
        raise ValueError(f"unknown sample {sample}")
    return frame.loc[frame["season_code"].astype(str).isin(seasons)].copy()


def paired_comparisons(
    match_scores: pd.DataFrame,
    pairs: Sequence[tuple[str, str, str, str, str]],
    *,
    n_boot: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Paired gaps on identical matches.

    Each spec is (market, value, model_a, model_b, sample). ``value`` is
    ``loss`` or ``hit``. The difference is A minus B. For a loss, positive
    means A put less probability on what happened. For a hit, positive means
    A was right more often.
    """
    rows: list[dict] = []
    for market, value, model_a, model_b, sample in pairs:
        if value not in {"loss", "hit"}:
            raise ValueError(f"unknown paired value {value}")
        base = match_scores.loc[match_scores["market"] == market]
        left = _subset_for_sample(base.loc[base["model"] == model_a], sample)
        right = _subset_for_sample(base.loc[base["model"] == model_b], sample)
        if left.empty or right.empty:
            raise ValueError(f"missing rows for {model_a} vs {model_b} ({sample})")
        left, right = _align_pair(left, right, model_a, model_b)
        if value == "loss":
            if not np.isfinite(left["loss"]).all() or not np.isfinite(right["loss"]).all():
                raise ValueError(
                    f"{model_a} vs {model_b} has a non-finite loss, so the gap is not a number"
                )
            diff = left["loss"].to_numpy(dtype=float) - right["loss"].to_numpy(dtype=float)
            metric = f"{market}_log_loss"
            note = "A positive difference means the first line has the higher log loss"
        else:
            diff = left["hit"].to_numpy(dtype=float) - right["hit"].to_numpy(dtype=float)
            metric = f"{market}_hit"
            note = "A positive difference means the first line was right more often"
        mean, low, high, n_rounds = block_bootstrap_mean_diff(
            diff, left["round_id"].to_numpy(), n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "sample": sample,
                "n_matches": int(len(left)),
                "n_rounds": int(n_rounds),
                "mean_diff": round(mean, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "n_bootstrap": int(n_boot),
                "seed": int(seed),
                "note": note,
            }
        )
    return pd.DataFrame(rows)


# Gaps the walk-forward publishes. Each one is A minus B on one match set.
PUBLISHED_PAIRS: tuple[tuple[str, str, str, str, str], ...] = (
    ("scoreline", "loss", "dixon_coles", "league_score_frequencies", "all"),
    ("scoreline", "loss", "dixon_coles", "constant_poisson", "all"),
    ("scoreline", "loss", "dixon_coles", "poisson", "all"),
    ("scoreline", "loss", "dixon_coles", "dixon_coles_rho0", "all"),
    ("scoreline", "hit", "dixon_coles", "always_1_1", "all"),
    ("scoreline", "hit", "dixon_coles", "league_score_frequencies", "all"),
    ("scoreline", "loss", "dixon_coles", "book_score_reconstruction", "all"),
    ("scoreline", "hit", "dixon_coles", "book_score_reconstruction", "all"),
    ("scoreline", "loss", "book_score_reconstruction_open", "book_score_reconstruction", "all"),
    ("scoreline", "loss", "dixon_coles", "league_score_frequencies", "holdout"),
    ("scoreline", "loss", "dixon_coles", "constant_poisson", "holdout"),
    ("scoreline", "loss", "dixon_coles", "book_score_reconstruction", "holdout"),
    ("1x2", "loss", "dixon_coles", "bookmaker_implied", "all"),
    ("1x2", "loss", "dixon_coles", "bookmaker_implied", "holdout"),
    ("1x2", "loss", "dixon_coles", "poisson", "all"),
    ("1x2", "loss", "dixon_coles", "naive_frequencies", "all"),
    ("1x2", "loss", "dixon_coles_calibrated", "dixon_coles", "all"),
    ("1x2", "loss", "dixon_coles_book_blend", "bookmaker_implied", "all"),
    ("1x2", "loss", "bookmaker_opening", "bookmaker_implied", "all"),
)


def render_measurement_notes(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    match_scores: pd.DataFrame,
) -> str:
    """Plain-language notes for the committed walk-forward readme.

    Every gap names the difference, the 95% range, and the match count.
    Levels that are not differences name the match count and are not ranked
    against a line with a different count.
    """
    score = summary.loc[summary["market"] == "scoreline"].copy()
    n_score = assert_one_match_set(score)
    scored = match_scores.loc[match_scores["market"] == "scoreline"]
    dc_matches = scored.loc[scored["model"] == "dixon_coles"]
    tune_n = int(dc_matches["season_code"].astype(str).isin(XI_TUNE_SEASONS).sum())
    holdout_n = int(dc_matches["season_code"].astype(str).isin(XI_TUNE_HOLDOUT_SEASONS).sum())

    def level(model: str, column: str) -> str:
        row = score.loc[score["model"] == model]
        if row.empty or pd.isna(row.iloc[0][column]):
            return "n/a"
        return f"{float(row.iloc[0][column]):.4f}"

    lines = [
        "## Scoreline",
        "",
        "The scoreline is the full table of exact scores, not the single most likely score.",
        "Log loss here is minus the log of the probability on the score that happened. Lower is better.",
        "A hit rate is how often the single most likely score was the score that happened.",
        "",
        (
            f"The pooled figures below include seasons {', '.join(XI_TUNE_SEASONS)} "
            f"({tune_n} matches), which were used to choose the time-decay setting. "
            "They are not a clean test of that choice. The clean figure is the holdout, "
            f"seasons {', '.join(XI_TUNE_HOLDOUT_SEASONS)}. When the setting was chosen, "
            "that holdout was 480 matches (see results/XI_TUNE.md). "
            f"This run scores {holdout_n} holdout matches. Do not use the holdout to choose the next setting."
        ),
        "",
        (
            f"Every scoreline row below is the same {n_score} matches. "
            "Always saying 1-1 is a hit rate only. It has no probability table, so it has no log loss."
        ),
        "",
        "| Line | What it is | Scoreline log loss | Most likely score hit | Top 3 hit | Home goal error | Away goal error | Goal-difference RPS | Matches |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    descriptions = {
        "always_1_1": "Always say 1-1 (hit rate only)",
        "league_score_frequencies": "League score frequencies from earlier matches, no team information",
        "constant_poisson": "Poisson with the earlier matches' average home and away rates, no team strength",
        "poisson": "Team-strength Poisson (attack, defence, each side's own average rate)",
        "dixon_coles_rho0": "Dixon-Coles goal rates with the low-score correction turned off",
        "dixon_coles": "Dixon-Coles",
        "book_score_reconstruction": "Closing prices: " + BOOK_SCORE_TABLE_LABEL,
        "book_score_reconstruction_open": "Opening prices: " + BOOK_SCORE_TABLE_LABEL,
    }
    order = list(descriptions)
    for model in order:
        row = score.loc[score["model"] == model]
        if row.empty:
            continue
        item = row.iloc[0]
        log_loss = "not a probability forecast" if model == "always_1_1" else f"{float(item['log_loss']):.4f}"
        top3 = "n/a" if pd.isna(item["top3_hit"]) else f"{float(item['top3_hit']):.4f}"
        mae_h = "n/a" if pd.isna(item["mae_home"]) else f"{float(item['mae_home']):.4f}"
        mae_a = "n/a" if pd.isna(item["mae_away"]) else f"{float(item['mae_away']):.4f}"
        rps = "n/a" if pd.isna(item["rps"]) else f"{float(item['rps']):.4f}"
        lines.append(
            f"| {descriptions[model]} | `{model}` | {log_loss} | {float(item['modal_hit']):.4f} | {top3} | {mae_h} | {mae_a} | {rps} | {int(item['n_test'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Home and away goal error are the average absolute difference between "
                "the goals the table expected and the goals that were scored. "
                "Goal-difference RPS scores the ordered margin (home goals minus away goals). Lower is better."
            ),
            "",
            "### Gaps on the same matches",
            "",
            (
                "A gap is the mean of model A minus model B on one shared set of matches. "
                f"The range is a 95% percentile interval from {BOOTSTRAP_DRAWS} resamples of whole rounds, "
                f"seed {BOOTSTRAP_SEED}. A range that contains zero has not separated the two lines on this sample. "
                "The closing book is the competitor on the result markets. A smaller log loss than the book "
                "is not described as a win over the book."
            ),
            "",
        ]
    )
    metric_names = {
        "scoreline_log_loss": "scoreline log loss",
        "scoreline_hit": "most-likely-score hit rate",
        "1x2_log_loss": "home/draw/away log loss",
        "1x2_hit": "home/draw/away hit rate",
    }
    sample_names = {
        "all": "all scored matches",
        "holdout": "the holdout seasons",
        "tune": "the seasons used to choose the time-decay setting",
    }
    for _, gap in paired.iterrows():
        metric_name = metric_names.get(str(gap["metric"]), str(gap["metric"]))
        sample_name = sample_names.get(str(gap["sample"]), str(gap["sample"]))
        lines.append(
            (
                f"- {gap['model_a']} minus {gap['model_b']}, {metric_name}, "
                f"{sample_name}: {float(gap['mean_diff']):.4f} "
                f"(95% range {float(gap['ci_low']):.4f} to {float(gap['ci_high']):.4f}, "
                f"{int(gap['n_matches'])} matches, {int(gap['n_rounds'])} rounds). "
                f"{gap['note']}."
            )
        )
    lines.extend(
        [
            "",
            (
                "Per-match scores are in `results/walkforward_match_scores.csv`. "
                "The paired ranges are in `results/walkforward_paired.csv`. "
                "Holdout scoreline levels, which use fewer matches than the headline table, "
                "are in `results/walkforward_scoreline_holdout.csv` and are not ranked against the headline rows."
            ),
            "",
            (
                f"Dixon-Coles scoreline log loss on the headline set is {level('dixon_coles', 'log_loss')} "
                f"on {n_score} matches. "
                f"League frequencies are {level('league_score_frequencies', 'log_loss')} on those same matches."
            ),
            "",
        ]
    )
    if "market" in match_scores.columns:
        checks = match_scores.loc[match_scores["market"] == "book_grid_check"]
    else:
        checks = match_scores.iloc[0:0]
    if not checks.empty:
        lines.extend(
            [
                "### Rebuilt book score table",
                "",
                (
                    f"{BOOK_SCORE_TABLE_LABEL}. "
                    "Two goal rates are chosen so a Poisson table, with the low-score correction left at zero, "
                    "matches the de-vigged home, draw and away prices and the de-vigged over/under 2.5 price. "
                    "The lines below are on the same matches as the scoreline table. "
                    "A smaller scoreline log loss than Dixon-Coles is not a win over the book."
                ),
                "",
            ]
        )
        for model_name, part in checks.groupby("model", sort=True):
            gap = fidelity_gap(part.to_dict("records"))
            which = "Closing" if model_name == "book_score_reconstruction" else "Opening"
            lines.append(
                (
                    f"- {which} prices, {int(gap['n_matches'])} matches. "
                    f"Rebuilt home/draw/away log loss {gap['loss_1x2_grid']:.4f}, "
                    f"raw prices {gap['loss_1x2_price']:.4f}, "
                    f"difference {gap['gap_1x2']:.4f}. "
                    f"Rebuilt over/under 2.5 log loss {gap['loss_ou_grid']:.4f}, "
                    f"raw prices {gap['loss_ou_price']:.4f}, "
                    f"difference {gap['gap_ou']:.4f}."
                )
            )
        lines.append("")
    return "\n".join(lines)
