"""One weight between the model goal rates and the price before kickoff.

The weight is chosen on seasons 2223, 2324 and 2425 only, by scoreline log
loss. Seasons 2526 and 2627 are scored after the weight is frozen. They are
not used to pick it. Each round's model rates are fit only on earlier
matches. The market rates for a match come from that match's opening prices.
The closing price is not an input. It is scored only as the reconstructed
table already used in this repo.

The stop rule is the one in docs/experiments/2026-11-25-model-plus-market.md.
This module does not read the live record.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import (
    DIXON_COLES_MAX_GOALS,
    MARKET_BLEND_LATER_SEASONS,
    MARKET_BLEND_MIN_GAIN,
    MARKET_BLEND_MIN_WEIGHT,
    MARKET_BLEND_SETTING_SEASONS,
    MARKET_BLEND_WEIGHTS,
    RESULTS,
)
from src.eval.book_grid import (
    BOOK_SCORE_TABLE_LABEL,
    assert_prices_reproduced,
    fit_book_grid,
    market_log_losses,
)
from src.eval.scoreline import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    block_bootstrap_mean_diff,
    score_scoreline,
)
from src.models.dixon_coles import (
    expected_goals,
    fit_dixon_coles,
    score_table_from_rates,
    scoreline_grid,
    scoreline_grid_from_rates,
)
from src.models.markets import markets_from_grid

logger = logging.getLogger(__name__)

_PURE = "dixon_coles"
_BLEND = "model_plus_market"
_BOOK = "book_closing_table"


@dataclass(frozen=True)
class MarketBlendResult:
    """Chosen weight, the stop decision, and the tables behind them."""

    chosen_weight: float
    dropped: bool
    reason: str
    candidates: pd.DataFrame
    paired: pd.DataFrame
    matches: pd.DataFrame
    elapsed_seconds: float
    run_date: str
    skipped_matches: int


def weight_name(weight: float) -> str:
    """Stable label for one weight."""
    return f"weight_{float(weight):.2f}"


def blended_rates(
    model_lam: float,
    model_mu: float,
    price_lam: float,
    price_mu: float,
    weight: float,
) -> tuple[float, float]:
    """Mix the two goal rates with one weight on the model."""
    w = float(weight)
    if not math.isfinite(w) or w < 0.0 or w > 1.0:
        raise ValueError("weight must be between 0 and 1")
    rates = (model_lam, model_mu, price_lam, price_mu)
    if any((not math.isfinite(rate)) or rate < 0.0 for rate in rates):
        raise ValueError("rates must be finite and non-negative")
    return (
        w * float(model_lam) + (1.0 - w) * float(price_lam),
        w * float(model_mu) + (1.0 - w) * float(price_mu),
    )


def published_blend_table(
    model,
    home_team: str,
    away_team: str,
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over: float,
    weight: float,
) -> dict:
    """One score table from the model rates and the price available before kickoff.

    The low-score correction on this table is zero. The closing price is not
    an input. The four probabilities are the de-vigged home, draw, away and
    over 2.5 prices for this match.
    """
    model_lam, model_mu = expected_goals(model, home_team, away_team)
    book = fit_book_grid(float(p_home), float(p_draw), float(p_away), float(p_over))
    lam, mu = blended_rates(model_lam, model_mu, book.lam, book.mu, weight)
    table = score_table_from_rates(lam, mu, 0.0, max_goals=DIXON_COLES_MAX_GOALS)
    table["lambda_home"] = float(lam)
    table["lambda_away"] = float(mu)
    table["rho"] = 0.0
    table["blend_weight"] = float(weight)
    return table


def choose_weight(
    scores: pd.DataFrame,
    weights: tuple[float, ...] | list[float],
    setting_seasons: tuple[str, ...] | list[str],
) -> float:
    """Lowest setting-season scoreline log loss. Ties keep the smaller weight.

    Later seasons in ``scores`` are ignored. The pure model and the closing
    table are ignored.
    """
    seasons = {str(season) for season in setting_seasons}
    setting = scores.loc[scores["season_code"].astype(str).isin(seasons)]
    best: tuple | None = None
    match_count: int | None = None
    for weight in weights:
        part = setting.loc[setting["model"] == weight_name(weight)]
        if part.empty:
            raise ValueError(f"no setting-season rows for {weight_name(weight)}")
        if match_count is None:
            match_count = int(len(part))
        elif len(part) != match_count:
            raise ValueError("weights were scored on different matches")
        loss = float(part["scoreline_loss"].mean())
        key = (loss, float(weight))
        if best is None or key < best[0]:
            best = (key, float(weight))
    if best is None:
        raise ValueError("no candidate weight")
    return best[1]


def stop_decision(
    weight: float,
    scoreline_gain: float,
    *,
    min_weight: float = MARKET_BLEND_MIN_WEIGHT,
    min_gain: float = MARKET_BLEND_MIN_GAIN,
) -> tuple[bool, str]:
    """Drop the second line only when the weight is low and the gain is short.

    ``scoreline_gain`` is the reconstructed closing table minus the blend.
    Positive means the blend has the lower scoreline log loss.
    """
    gain = round(float(scoreline_gain), 4)
    low_weight = float(weight) < float(min_weight)
    short_gain = gain < float(min_gain)
    if low_weight and short_gain:
        return True, (
            f"the weight on the model is {float(weight):.2f}, under {float(min_weight):.2f}, "
            f"and the scoreline gain against the reconstructed closing table is {gain:.4f}, "
            f"under {float(min_gain):.2f}"
        )
    if low_weight:
        return False, (
            f"the weight on the model is {float(weight):.2f}, under {float(min_weight):.2f}, "
            f"but the scoreline gain against the reconstructed closing table is {gain:.4f}, "
            f"which is not under {float(min_gain):.2f}"
        )
    return False, (
        f"the weight on the model is {float(weight):.2f}, which is not under "
        f"{float(min_weight):.2f}. The scoreline gain against the reconstructed "
        f"closing table is {gain:.4f}"
    )


def run_market_blend(
    matches: pd.DataFrame | None = None,
    *,
    weights: tuple[float, ...] | list[float] | None = None,
    setting_seasons: tuple[str, ...] | list[str] | None = None,
    later_seasons: tuple[str, ...] | list[str] | None = None,
) -> MarketBlendResult:
    """Choose the weight on earlier seasons, then score the later ones."""
    from src.clean import build_matches_table
    from src.eval.gameweeks import assign_gameweeks
    from src.eval.walkforward import MIN_TRAIN_ROWS
    from src.features import add_implied_odds

    started = time.perf_counter()
    weights_t = tuple(float(item) for item in (MARKET_BLEND_WEIGHTS if weights is None else weights))
    setting_t = tuple(
        str(item)
        for item in (
            MARKET_BLEND_SETTING_SEASONS if setting_seasons is None else setting_seasons
        )
    )
    later_t = tuple(
        str(item)
        for item in (MARKET_BLEND_LATER_SEASONS if later_seasons is None else later_seasons)
    )
    _check_windows(weights_t, setting_t, later_t)

    if matches is None:
        matches = build_matches_table()
    frame = assign_gameweeks(matches)
    frame["date"] = pd.to_datetime(frame["date"])
    if "implied_home_open" not in frame.columns or "implied_home" not in frame.columns:
        frame = add_implied_odds(frame)

    setting_rows, setting_checks, setting_skipped = _walk_seasons(
        frame,
        seasons=set(setting_t),
        weights=weights_t,
        min_train_rows=MIN_TRAIN_ROWS,
    )
    chosen = choose_weight(setting_rows, weights_t, setting_t)
    logger.info("Chosen weight %.2f from the setting seasons", chosen)
    later_rows, later_checks, later_skipped = _walk_seasons(
        frame,
        seasons=set(later_t),
        weights=(chosen,),
        min_train_rows=MIN_TRAIN_ROWS,
    )
    assert_prices_reproduced(setting_checks + later_checks)
    scores = _tag_lines(pd.concat([setting_rows, later_rows], ignore_index=True), chosen)
    paired = pd.DataFrame(_paired_rows(scores, setting_t, later_t))
    dropped, reason = _decision_from_paired(paired, chosen)
    return MarketBlendResult(
        chosen_weight=float(chosen),
        dropped=dropped,
        reason=reason,
        candidates=_candidate_table(scores, setting_t, weights_t, chosen),
        paired=paired,
        matches=scores,
        elapsed_seconds=time.perf_counter() - started,
        run_date=datetime.now(ZoneInfo("Europe/London")).strftime("%d %b %Y"),
        skipped_matches=int(setting_skipped + later_skipped),
    )


def write_market_blend_artefacts(result: MarketBlendResult, results_dir=None) -> dict[str, str]:
    """Write the note and the tables. Does not change the published model."""
    root = RESULTS if results_dir is None else results_dir
    root.mkdir(parents=True, exist_ok=True)
    note = root / "MODEL_PLUS_MARKET.md"
    candidates = root / "market_blend_candidates.csv"
    paired = root / "market_blend_paired.csv"
    matches = root / "market_blend_matches.csv"
    note.write_text(render_market_blend_note(result), encoding="utf-8")
    result.candidates.to_csv(candidates, index=False)
    result.paired.to_csv(paired, index=False)
    result.matches.to_csv(matches, index=False)
    return {
        "note": str(note),
        "candidates": str(candidates),
        "paired": str(paired),
        "matches": str(matches),
    }


def render_market_blend_note(result: MarketBlendResult) -> str:
    """Plain-language result. Every gap has a range and a match count."""
    later_book = _lookup(result.paired, "later_book_minus_blend_scoreline")
    later_pure = _lookup(result.paired, "later_pure_minus_blend_scoreline")
    later_result = _lookup(result.paired, "later_pure_minus_blend_result")
    later_price = _lookup(result.paired, "later_blend_minus_closing_result")
    setting_pure = _lookup(result.paired, "setting_pure_minus_blend_scoreline")
    lines = [
        "# One weight between the model and the price before kickoff",
        "",
        f"**Date:** {result.run_date} (London)",
        (
            "**Question:** does one weight, mixing the model's goal rates with the "
            "rates from the price available before kickoff, earn a second score table "
            "beside the pure model?"
        ),
        "",
        "## What was fit",
        "",
        (
            "The pure model is the current Dixon-Coles score table. The time-decay "
            "setting stays at 0.0035 and the thin-data strength stays at 32. "
            "Its low-score correction is the value fitted on each round."
        ),
        "",
        (
            "The second line mixes that model's goal rates with the goal rates "
            "implied by the opening home, draw, away and over/under 2.5 prices. "
            "The closing price is not an input. The weights tried were "
            "0.00, 0.05, 0.10, 0.25, 0.50 and 1.00. Nothing else was tried after looking. "
            "The blend and the reconstructed closing table both leave the low-score "
            "correction at zero."
        ),
        "",
        (
            "The reconstructed table is "
            f"{BOOK_SCORE_TABLE_LABEL}."
        ),
        "",
        "## How the weight was chosen",
        "",
        (
            "Each round's model rates are fit only on matches dated before that round. "
            "The weight is the one with the lowest scoreline log loss on seasons "
            "2223, 2324 and 2425. A tie keeps the smaller weight. Those seasons were "
            "already used to choose other settings, so this table is where the weight "
            "was picked. It is not a clean test."
        ),
        "",
        _candidate_markdown(result.candidates),
        "",
        f"The chosen weight on the model is {_fmt_weight(result.chosen_weight)}.",
        "",
        _edge_sentence(result),
        "",
        (
            "The gap below is on the setting seasons. Those matches were used to "
            "pick the weight, so this gap is not a clean test."
        ),
        "",
        _gain_sentence(
            "On the setting seasons, scoreline log loss",
            "the pure model minus the blend",
            setting_pure,
        ),
        "",
        (
            "Seasons "
            f"{', '.join(MARKET_BLEND_LATER_SEASONS)} were not used to pick the weight."
        ),
        "",
        "## Later seasons",
        "",
        (
            "The model rates are refit every round on earlier matches. The weight "
            "stays at the chosen value. The opening prices used for a match are "
            "that match's own prices. A range is a 95% percentile interval from "
            f"{BOOTSTRAP_DRAWS} resamples of whole rounds, seed {BOOTSTRAP_SEED}. "
            "A range that contains zero has not separated the two lines on this sample."
        ),
        "",
        "### Against the reconstructed closing table",
        "",
        (
            "This is the stop comparison. The difference is the reconstructed closing "
            "table minus the blend. A positive number means the blend has the lower "
            "scoreline log loss."
        ),
        "",
        _gain_sentence(
            "On every later-season match, scoreline log loss",
            "the reconstructed closing table minus the blend",
            later_book,
        ),
        "",
        "### Against the pure model",
        "",
        (
            "The difference is the pure model minus the blend. A positive number "
            "means the blend has the lower log loss."
        ),
        "",
        _gain_sentence(
            "On those same matches, scoreline log loss",
            "the pure model minus the blend",
            later_pure,
        ),
        "",
        _gain_sentence(
            "On those same matches, home/draw/away log loss",
            "the pure model minus the blend",
            later_result,
        ),
        "",
        "### Closing price",
        "",
        _price_sentence(later_price),
        "",
        "## Decision",
        "",
        _decision_paragraph(result, later_book),
        "",
        _coverage_sentence(result),
        "",
        "## Regenerate",
        "",
        "```bash",
        "python -m src.evaluate --mode market_blend",
        "```",
        "",
        (
            f"The run took {result.elapsed_seconds / 60.0:.1f} minutes. "
            "The per-match rows are `results/market_blend_matches.csv`. "
            "The ranges are `results/market_blend_paired.csv`."
        ),
        "",
    ]
    text = "\n".join(line for line in lines if line is not None)
    _reject_bad_language(text)
    return text


def _check_windows(weights, setting, later) -> None:
    if not weights:
        raise ValueError("weights must be non-empty")
    if any((not math.isfinite(weight)) or weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("weights must be between 0 and 1")
    overlap = set(setting) & set(later)
    if overlap:
        raise ValueError(f"setting and later seasons overlap: {sorted(overlap)}")
    if not setting or not later:
        raise ValueError("setting and later seasons must both be non-empty")


def _price(row, name: str) -> float:
    try:
        value = float(getattr(row, name))
    except (TypeError, ValueError, AttributeError):
        return float("nan")
    return value


def _prices_ok(row) -> bool:
    names = (
        "implied_home_open",
        "implied_draw_open",
        "implied_away_open",
        "implied_over_25_open",
        "implied_under_25_open",
        "implied_home",
        "implied_draw",
        "implied_away",
        "implied_over_25",
        "implied_under_25",
    )
    return all(math.isfinite(_price(row, name)) and _price(row, name) > 0.0 for name in names)


def _walk_seasons(
    frame: pd.DataFrame,
    *,
    seasons: set[str],
    weights: tuple[float, ...],
    min_train_rows: int,
) -> tuple[pd.DataFrame, list[dict], int]:
    from src.eval.walkforward import iter_folds

    rows: list[dict] = []
    checks: list[dict] = []
    skipped = 0
    seen = 0
    for train, test, season, gw in iter_folds(frame, min_train_rows=min_train_rows):
        if str(season) not in seasons:
            continue
        seen += 1
        if seen == 1 or seen % 10 == 0:
            logger.info("Scoring %s round %s", season, gw)
        model = fit_dixon_coles(train)
        for row in test.itertuples(index=False):
            if not _prices_ok(row):
                skipped += 1
                continue
            identity = _identity(row, season, gw)
            lam_m, mu_m = expected_goals(model, row.home_team, row.away_team)
            opening = fit_book_grid(
                _price(row, "implied_home_open"),
                _price(row, "implied_draw_open"),
                _price(row, "implied_away_open"),
                _price(row, "implied_over_25_open"),
            )
            closing = fit_book_grid(
                _price(row, "implied_home"),
                _price(row, "implied_draw"),
                _price(row, "implied_away"),
                _price(row, "implied_over_25"),
            )
            checks.append(
                market_log_losses(
                    result=str(row.result),
                    home_goals=int(row.home_goals),
                    away_goals=int(row.away_goals),
                    price_home=_price(row, "implied_home_open"),
                    price_draw=_price(row, "implied_draw_open"),
                    price_away=_price(row, "implied_away_open"),
                    price_over=_price(row, "implied_over_25_open"),
                    price_under=_price(row, "implied_under_25_open"),
                    grid_home=opening.home,
                    grid_draw=opening.draw,
                    grid_away=opening.away,
                    grid_over=opening.over_25,
                    grid_under=opening.under_25,
                )
            )
            checks.append(
                market_log_losses(
                    result=str(row.result),
                    home_goals=int(row.home_goals),
                    away_goals=int(row.away_goals),
                    price_home=_price(row, "implied_home"),
                    price_draw=_price(row, "implied_draw"),
                    price_away=_price(row, "implied_away"),
                    price_over=_price(row, "implied_over_25"),
                    price_under=_price(row, "implied_under_25"),
                    grid_home=closing.home,
                    grid_draw=closing.draw,
                    grid_away=closing.away,
                    grid_over=closing.over_25,
                    grid_under=closing.under_25,
                )
            )
            closing_result = _result_loss(
                str(row.result),
                _price(row, "implied_home"),
                _price(row, "implied_draw"),
                _price(row, "implied_away"),
            )
            rows.append(
                {
                    **identity,
                    **_grid_scores(
                        scoreline_grid(model, row.home_team, row.away_team),
                        int(row.home_goals),
                        int(row.away_goals),
                        lam_m,
                        mu_m,
                        float(model.rho),
                        str(row.result),
                    ),
                    "model": _PURE,
                    "weight": float("nan"),
                    "closing_result_loss": closing_result,
                }
            )
            rows.append(
                {
                    **identity,
                    **_grid_scores(
                        closing.grid,
                        int(row.home_goals),
                        int(row.away_goals),
                        closing.lam,
                        closing.mu,
                        0.0,
                        str(row.result),
                    ),
                    "model": _BOOK,
                    "weight": float("nan"),
                    "closing_result_loss": closing_result,
                }
            )
            for weight in weights:
                lam, mu = blended_rates(lam_m, mu_m, opening.lam, opening.mu, weight)
                grid = scoreline_grid_from_rates(lam, mu, 0.0, max_goals=DIXON_COLES_MAX_GOALS)
                rows.append(
                    {
                        **identity,
                        **_grid_scores(
                            grid,
                            int(row.home_goals),
                            int(row.away_goals),
                            lam,
                            mu,
                            0.0,
                            str(row.result),
                        ),
                        "model": weight_name(weight),
                        "weight": float(weight),
                        "closing_result_loss": closing_result,
                    }
                )
    if not rows:
        raise ValueError(f"no scored rounds in seasons {sorted(seasons)}")
    return pd.DataFrame(rows), checks, skipped


def _grid_scores(grid, home_goals, away_goals, lam, mu, rho, result) -> dict[str, float]:
    scored = score_scoreline(
        grid, home_goals, away_goals, lam=lam, mu=mu, rho=rho
    )
    markets = markets_from_grid(grid)
    return {
        "scoreline_loss": float(scored["cell_log_loss"]),
        "result_loss": _result_loss(result, markets["home"], markets["draw"], markets["away"]),
    }


def _identity(row, season: str, gw: int) -> dict:
    date = pd.Timestamp(row.date)
    date_s = "" if pd.isna(date) else date.date().isoformat()
    return {
        "match_id": f"{season}_{date_s}_{row.home_team}_{row.away_team}",
        "round_id": f"{season}_{int(gw)}",
        "season_code": str(season),
        "gameweek": int(gw),
        "date": date_s,
        "home_team": str(row.home_team),
        "away_team": str(row.away_team),
        "home_goals": int(row.home_goals),
        "away_goals": int(row.away_goals),
        "result": str(row.result),
    }


def _result_loss(result: str, home: float, draw: float, away: float) -> float:
    probs = {"H": home, "D": draw, "A": away}
    if result not in probs:
        raise ValueError(f"unknown result {result}")
    probability = min(max(float(probs[result]), 1e-15), 1.0)
    return float(-math.log(probability))


def _tag_lines(scores: pd.DataFrame, weight: float) -> pd.DataFrame:
    chosen = scores.loc[scores["model"] == weight_name(weight)].copy()
    chosen["model"] = _BLEND
    return pd.concat([scores, chosen], ignore_index=True)


def _candidate_table(
    scores: pd.DataFrame,
    setting_seasons: tuple[str, ...],
    weights: tuple[float, ...],
    chosen: float,
) -> pd.DataFrame:
    seasons = {str(season) for season in setting_seasons}
    setting = scores.loc[scores["season_code"].astype(str).isin(seasons)]
    rows = []
    for weight in weights:
        part = setting.loc[setting["model"] == weight_name(weight)]
        rows.append(
            {
                "weight": float(weight),
                "scoreline_log_loss": round(float(part["scoreline_loss"].mean()), 4),
                "matches": int(len(part)),
                "result_log_loss": round(float(part["result_loss"].mean()), 4),
                "chosen": bool(float(weight) == float(chosen)),
            }
        )
    return pd.DataFrame(rows)


def _paired_rows(
    scores: pd.DataFrame,
    setting_seasons: tuple[str, ...],
    later_seasons: tuple[str, ...],
) -> list[dict]:
    return [
        _diff_row(
            "setting_pure_minus_blend_scoreline",
            _slice(scores, setting_seasons),
            _PURE,
            _BLEND,
            "scoreline_loss",
            "Setting seasons chose the weight. This gap is not a clean test.",
        ),
        _diff_row(
            "later_book_minus_blend_scoreline",
            _slice(scores, later_seasons),
            _BOOK,
            _BLEND,
            "scoreline_loss",
            "Positive means the blend has the lower scoreline log loss than the reconstructed closing table.",
        ),
        _diff_row(
            "later_pure_minus_blend_scoreline",
            _slice(scores, later_seasons),
            _PURE,
            _BLEND,
            "scoreline_loss",
            "Positive means the blend has the lower scoreline log loss than the pure model.",
        ),
        _diff_row(
            "later_pure_minus_blend_result",
            _slice(scores, later_seasons),
            _PURE,
            _BLEND,
            "result_loss",
            "Positive means the blend has the lower home/draw/away log loss than the pure model. Not the stop rule.",
        ),
        _closing_price_row(_slice(scores, later_seasons)),
    ]


def _slice(scores: pd.DataFrame, seasons: tuple[str, ...]) -> pd.DataFrame:
    wanted = {str(season) for season in seasons}
    return scores.loc[scores["season_code"].astype(str).isin(wanted)].copy()


def _diff_row(name, scores, left_name, right_name, column, note) -> dict:
    left = scores.loc[scores["model"] == left_name]
    right = scores.loc[scores["model"] == right_name]
    return _gap_row(name, left_name, right_name, _align_diff(left, right, column), note)


def _closing_price_row(scores: pd.DataFrame) -> dict:
    blend = scores.loc[scores["model"] == _BLEND].copy()
    price = blend.copy()
    price["model"] = "closing_price"
    price["result_loss"] = price["closing_result_loss"]
    note = (
        "A positive difference means the blend has the higher home/draw/away log loss. "
        "The closing price is the competitor. This is not a claim about the bookmaker."
    )
    return _gap_row(
        "later_blend_minus_closing_result",
        _BLEND,
        "closing_price",
        _align_diff(blend, price, "result_loss"),
        note,
    )


def _align_diff(left: pd.DataFrame, right: pd.DataFrame, column: str) -> dict:
    if left.empty or right.empty:
        raise ValueError("a paired gap needs both lines")
    merged = left.merge(right, on="match_id", suffixes=("_a", "_b"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"lines were scored on different matches ({len(left)} and {len(right)}, "
            f"{len(merged)} shared)"
        )
    if not merged["round_id_a"].equals(merged["round_id_b"]):
        raise ValueError("round ids disagree on a paired gap")
    diff = merged[f"{column}_a"].to_numpy(dtype=float) - merged[f"{column}_b"].to_numpy(
        dtype=float
    )
    mean, low, high, n_rounds = block_bootstrap_mean_diff(
        diff, merged["round_id_a"].to_numpy()
    )
    return {
        "mean_diff": float(mean),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_matches": int(len(merged)),
        "n_rounds": int(n_rounds),
    }


def _gap_row(name: str, model_a: str, model_b: str, gap: dict, note: str) -> dict:
    if gap["n_matches"] < 1 or gap["n_rounds"] < 1:
        raise ValueError(f"{name} is missing a match count")
    return {
        "comparison": name,
        "model_a": model_a,
        "model_b": model_b,
        "mean_diff": round(gap["mean_diff"], 4),
        "ci_low": round(gap["ci_low"], 4),
        "ci_high": round(gap["ci_high"], 4),
        "n_matches": int(gap["n_matches"]),
        "n_rounds": int(gap["n_rounds"]),
        "n_bootstrap": int(BOOTSTRAP_DRAWS),
        "seed": int(BOOTSTRAP_SEED),
        "note": note,
        "mean_diff_exact": float(gap["mean_diff"]),
    }


def _decision_from_paired(paired: pd.DataFrame, weight: float) -> tuple[bool, str]:
    return stop_decision(
        weight, float(_lookup(paired, "later_book_minus_blend_scoreline")["mean_diff"])
    )


def _lookup(paired: pd.DataFrame, name: str) -> pd.Series:
    rows = paired.loc[paired["comparison"] == name]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {name}, found {len(rows)}")
    return rows.iloc[0]


def _candidate_markdown(candidates: pd.DataFrame) -> str:
    header = "| Weight on the model | Scoreline log loss | Matches | Home/draw/away log loss |"
    rule = "|---|---|---|---|"
    body = []
    for row in candidates.itertuples(index=False):
        mark = " (chosen)" if bool(row.chosen) else ""
        body.append(
            f"| {_fmt_weight(row.weight)}{mark} | {float(row.scoreline_log_loss):.4f} | "
            f"{int(row.matches)} | {float(row.result_log_loss):.4f} |"
        )
    return "\n".join([header, rule, *body])


def _gain_sentence(prefix: str, difference: str, row: pd.Series) -> str:
    text = (
        f"{prefix}: {difference} is {float(row['mean_diff']):.4f} "
        f"(95% range {float(row['ci_low']):.4f} to {float(row['ci_high']):.4f}, "
        f"{int(row['n_matches'])} matches, {int(row['n_rounds'])} rounds)."
    )
    if float(row["ci_low"]) <= 0.0 <= float(row["ci_high"]):
        text += " The range includes zero."
    return text


def _price_sentence(row: pd.Series) -> str:
    return (
        "On the later-season matches, home/draw/away log loss: the blend minus "
        f"the closing price is {float(row['mean_diff']):.4f} "
        f"(95% range {float(row['ci_low']):.4f} to {float(row['ci_high']):.4f}, "
        f"{int(row['n_matches'])} matches, {int(row['n_rounds'])} rounds). "
        "The closing price is the competitor. This is not a claim about the bookmaker. "
        "A positive difference means the blend has the higher log loss."
        + (
            " The range includes zero."
            if float(row["ci_low"]) <= 0.0 <= float(row["ci_high"])
            else ""
        )
    )


def _decision_paragraph(result: MarketBlendResult, later_book: pd.Series) -> str:
    gain = float(later_book["mean_diff"])
    if result.dropped:
        return (
            f"The stop rule drops the second line: {result.reason}. "
            f"The weight on the model is {_fmt_weight(result.chosen_weight)}. "
            "That weight is the result. The pure model stays the published probability."
        )
    text = (
        f"The stop rule keeps the second line: {result.reason}. "
        f"The published second line uses weight {_fmt_weight(result.chosen_weight)} "
        "and is labelled model plus market. The pure model stays beside it. "
        "The rule uses the point estimate, rounded to 4 decimal places. "
        "The second line is written for a match only when the price available "
        "before kickoff has home, draw, away and over/under 2.5. A match without "
        "that price keeps the pure model only. The closing price is not an input "
        "to that line."
    )
    if abs(gain) < 0.01:
        text += (
            " The gap against the reconstructed closing table is under 0.01. "
            "That is a point estimate inside the range above. "
            "It is not a success, and it is not a claim about the bookmaker."
        )
    else:
        text += " It is not a claim about the bookmaker."
    return text


def _edge_sentence(result: MarketBlendResult) -> str:
    weights = [float(value) for value in result.candidates["weight"]]
    weight = float(result.chosen_weight)
    if len(weights) > 1 and weight in (min(weights), max(weights)):
        return (
            f"Weight {_fmt_weight(weight)} is at the end of the list that was written down. "
            "No further value was added after looking."
        )
    return ""


def _coverage_sentence(result: MarketBlendResult) -> str:
    if result.skipped_matches == 0:
        return (
            "Every scored match had an opening and a closing home, draw, away "
            "and over/under 2.5 price. Rounds with fewer than 200 earlier matches "
            "were not scored."
        )
    return (
        f"{int(result.skipped_matches)} matches were left out because a price was missing. "
        "The gaps above use only the matches that had every required price."
    )


def _fmt_weight(value: float) -> str:
    return f"{float(value):.2f}"


def _reject_bad_language(text: str) -> None:
    lowered = text.lower()
    banned = (
        "beat the book",
        "beats the book",
        "beaten the book",
        "ahead of the book",
        "won the week",
        "tipster",
    )
    for phrase in banned:
        if phrase in lowered:
            raise ValueError(f"result note uses banned wording: {phrase}")
    if "·" in text:
        raise ValueError("result note uses a mid-dot")
