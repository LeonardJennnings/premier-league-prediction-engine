"""One weight between a slow Dixon-Coles fit and a fast Dixon-Coles fit.

The weight is chosen on seasons 2223, 2324 and 2425 only, by scoreline log
loss. Seasons 2526 and 2627 are scored after the weight is frozen. They are
not used to pick it. Each round's two fits use only earlier matches. The
weight mixes the two home rates and the two away rates. It does not read a
price. The closing price is scored only as the reconstructed table already
used in this repo.

The slow decay stays at the published time-decay setting. The fast decay is
the fastest value already in XI_TUNE_CANDIDATES. Neither is searched here.
This is not a state-space model and it is not a Kalman filter.

The stop rule is the one in docs/experiments/2026-11-26-two-speed-memory.md.
This module does not read the live record.
"""

from __future__ import annotations

import ast
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import (
    DIXON_COLES_MAX_GOALS,
    RESULTS,
    TWO_SPEED_FAST_XI,
    TWO_SPEED_LATER_SEASONS,
    TWO_SPEED_MIN_GAIN,
    TWO_SPEED_MIN_WEIGHT,
    TWO_SPEED_SETTING_SEASONS,
    TWO_SPEED_SLOW_XI,
    TWO_SPEED_WEIGHTS,
    XI_TUNE_CANDIDATES,
)
from src.eval.book_grid import (
    BOOK_SCORE_TABLE_LABEL,
    CLOSING_PRICE_COLUMNS,
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
    scoreline_grid,
    scoreline_grid_from_rates,
)
from src.models.markets import markets_from_grid

logger = logging.getLogger(__name__)

_PURE = "dixon_coles"
_LINE = "two_speed"
_BOOK = "book_closing_table"
_LINE_FUNCTIONS = ("mixed_rates", "two_speed_grid")


@dataclass(frozen=True)
class TwoSpeedResult:
    """Chosen weight, the stop decision, and the tables behind them."""

    chosen_weight: float
    slow_xi: float
    fast_xi: float
    dropped: bool
    edge: bool
    replaces_pure: bool
    reason: str
    candidates: pd.DataFrame
    paired: pd.DataFrame
    matches: pd.DataFrame
    elapsed_seconds: float
    run_date: str
    skipped_book_matches: int


def weight_name(weight: float) -> str:
    """Stable label for one weight on the fast fit."""
    return f"weight_{float(weight):.2f}"


def mixed_rates(
    slow_lam: float,
    slow_mu: float,
    fast_lam: float,
    fast_mu: float,
    weight: float,
) -> tuple[float, float]:
    """Mix the two goal rates. The weight is on the fast fit.

    The same weight is used for the home rate and the away rate. No price
    enters this mix.
    """
    w = float(weight)
    if not math.isfinite(w) or w < 0.0 or w > 1.0:
        raise ValueError("weight must be between 0 and 1")
    rates = (slow_lam, slow_mu, fast_lam, fast_mu)
    if any((not math.isfinite(rate)) or rate < 0.0 for rate in rates):
        raise ValueError("rates must be finite and non-negative")
    return (
        (1.0 - w) * float(slow_lam) + w * float(fast_lam),
        (1.0 - w) * float(slow_mu) + w * float(fast_mu),
    )


def two_speed_grid(slow_model, fast_model, home_team: str, away_team: str, weight: float):
    """One score table from the slow fit and the fast fit.

    Weight 0 is the slow fit, which is the current pure model. Any other
    weight mixes the home rates and the away rates and keeps the slow fit's
    low-score correction. The closing price is not an input.
    """
    if float(weight) == 0.0:
        lam, mu = expected_goals(slow_model, home_team, away_team)
        grid = scoreline_grid(slow_model, home_team, away_team)
        return grid, lam, mu, float(slow_model.rho)
    slow_lam, slow_mu = expected_goals(slow_model, home_team, away_team)
    fast_lam, fast_mu = expected_goals(fast_model, home_team, away_team)
    lam, mu = mixed_rates(slow_lam, slow_mu, fast_lam, fast_mu, weight)
    rho = float(slow_model.rho)
    try:
        grid = scoreline_grid_from_rates(lam, mu, rho, max_goals=DIXON_COLES_MAX_GOALS)
    except ValueError:
        rho = 0.0
        grid = scoreline_grid_from_rates(lam, mu, 0.0, max_goals=DIXON_COLES_MAX_GOALS)
    return grid, lam, mu, rho


def line_function_sources() -> dict[str, str]:
    """Source of the functions that build the two-speed line.

    Those functions are not allowed to name a price column. The closing
    table is scored elsewhere, as a competitor.
    """
    from pathlib import Path

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _LINE_FUNCTIONS:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise ValueError(f"could not read the source of {node.name}")
            found[node.name] = segment
    missing = [name for name in _LINE_FUNCTIONS if name not in found]
    if missing:
        raise ValueError(f"missing line functions: {missing}")
    return found


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


def range_includes_zero(ci_low: float, ci_high: float) -> bool:
    """True when the printed 4-decimal range covers zero."""
    low = round(float(ci_low), 4)
    high = round(float(ci_high), 4)
    return low <= 0.0 <= high


def stop_decision(
    weight: float,
    scoreline_gain: float,
    ci_low: float,
    ci_high: float,
    *,
    min_weight: float = TWO_SPEED_MIN_WEIGHT,
    min_gain: float = TWO_SPEED_MIN_GAIN,
) -> tuple[bool, str]:
    """Drop the line if the weight is low, the gain is short, or the range covers zero.

    ``scoreline_gain`` is the current pure model minus the two-speed line.
    Positive means the two-speed line has the lower scoreline log loss.
    """
    gain = round(float(scoreline_gain), 4)
    low_weight = float(weight) < float(min_weight)
    short_gain = gain < float(min_gain)
    covers_zero = range_includes_zero(ci_low, ci_high)
    low = round(float(ci_low), 4)
    high = round(float(ci_high), 4)
    reasons: list[str] = []
    if low_weight:
        reasons.append(
            f"the weight on the fast fit is {float(weight):.2f}, under {float(min_weight):.2f}"
        )
    if short_gain:
        reasons.append(
            "the paired scoreline gain versus the current pure model is "
            f"{gain:.4f}, under {float(min_gain):.3f}"
        )
    if covers_zero:
        reasons.append(
            f"the 95% range of that gain is {low:.4f} to {high:.4f}, which includes zero"
        )
    if reasons:
        return True, "; ".join(reasons)
    return False, (
        f"the weight on the fast fit is {float(weight):.2f}, the paired scoreline gain "
        f"versus the current pure model is {gain:.4f}, and the 95% range "
        f"{low:.4f} to {high:.4f} excludes zero"
    )


def edge_met(book_minus_line: float, ci_low: float, ci_high: float) -> bool:
    """True when the two-speed line clears the reconstructed closing table.

    The gap is that table's scoreline log loss minus the line's. Positive
    means the line was closer to the score that happened. The printed range
    has to sit entirely above zero.
    """
    gain = round(float(book_minus_line), 4)
    low = round(float(ci_low), 4)
    return gain > 0.0 and low > 0.0 and not range_includes_zero(ci_low, ci_high)


def run_two_speed(
    matches: pd.DataFrame | None = None,
    *,
    weights: tuple[float, ...] | list[float] | None = None,
    slow_xi: float | None = None,
    fast_xi: float | None = None,
    setting_seasons: tuple[str, ...] | list[str] | None = None,
    later_seasons: tuple[str, ...] | list[str] | None = None,
    min_train_rows: int | None = None,
) -> TwoSpeedResult:
    """Choose the weight on earlier seasons, then score the later ones."""
    from src.clean import build_matches_table
    from src.eval.gameweeks import assign_gameweeks
    from src.eval.walkforward import MIN_TRAIN_ROWS
    from src.features import add_implied_odds

    started = time.perf_counter()
    weights_t = tuple(float(item) for item in (TWO_SPEED_WEIGHTS if weights is None else weights))
    slow = float(TWO_SPEED_SLOW_XI if slow_xi is None else slow_xi)
    fast = float(TWO_SPEED_FAST_XI if fast_xi is None else fast_xi)
    setting_t = tuple(
        str(item)
        for item in (TWO_SPEED_SETTING_SEASONS if setting_seasons is None else setting_seasons)
    )
    later_t = tuple(
        str(item)
        for item in (TWO_SPEED_LATER_SEASONS if later_seasons is None else later_seasons)
    )
    train_floor = int(MIN_TRAIN_ROWS if min_train_rows is None else min_train_rows)
    _check_windows(weights_t, slow, fast, setting_t, later_t)

    if matches is None:
        matches = build_matches_table()
    frame = assign_gameweeks(matches)
    frame["date"] = pd.to_datetime(frame["date"])
    if "implied_home" not in frame.columns:
        frame = add_implied_odds(frame)

    setting_rows, setting_checks, setting_skipped = _walk_seasons(
        frame,
        seasons=set(setting_t),
        weights=weights_t,
        slow_xi=slow,
        fast_xi=fast,
        min_train_rows=train_floor,
    )
    chosen = choose_weight(setting_rows, weights_t, setting_t)
    logger.info("Chosen weight on the fast fit %.2f", chosen)
    later_rows, later_checks, later_skipped = _walk_seasons(
        frame,
        seasons=set(later_t),
        weights=(chosen,),
        slow_xi=slow,
        fast_xi=fast,
        min_train_rows=train_floor,
    )
    assert_prices_reproduced(setting_checks + later_checks)
    scores = _tag_lines(pd.concat([setting_rows, later_rows], ignore_index=True), chosen)
    _assert_weight_zero_matches_pure(scores)
    paired = pd.DataFrame(_paired_rows(scores, setting_t, later_t))
    dropped, edge, replaces, reason = _decision_from_paired(paired, chosen)
    return TwoSpeedResult(
        chosen_weight=float(chosen),
        slow_xi=slow,
        fast_xi=fast,
        dropped=dropped,
        edge=edge,
        replaces_pure=replaces,
        reason=reason,
        candidates=_candidate_table(scores, setting_t, weights_t, chosen),
        paired=paired,
        matches=scores,
        elapsed_seconds=time.perf_counter() - started,
        run_date=datetime.now(ZoneInfo("Europe/London")).strftime("%d %b %Y"),
        skipped_book_matches=int(setting_skipped + later_skipped),
    )


def write_two_speed_artefacts(result: TwoSpeedResult, results_dir=None) -> dict[str, str]:
    """Write the note and the tables. Does not change the published model."""
    root = RESULTS if results_dir is None else results_dir
    root.mkdir(parents=True, exist_ok=True)
    note = root / "TWO_SPEED.md"
    candidates = root / "two_speed_candidates.csv"
    paired = root / "two_speed_paired.csv"
    matches = root / "two_speed_matches.csv"
    note.write_text(render_two_speed_note(result), encoding="utf-8")
    result.candidates.to_csv(candidates, index=False)
    result.paired.to_csv(paired, index=False)
    result.matches.to_csv(matches, index=False)
    return {
        "note": str(note),
        "candidates": str(candidates),
        "paired": str(paired),
        "matches": str(matches),
    }


def render_two_speed_note(result: TwoSpeedResult) -> str:
    """Plain-language result. Every gap has a range and a match count."""
    later_pure = _lookup(result.paired, "later_pure_minus_two_speed_scoreline")
    later_book = _lookup(result.paired, "later_book_minus_two_speed_scoreline")
    later_result = _lookup(result.paired, "later_pure_minus_two_speed_result")
    later_book_result = _lookup(result.paired, "later_book_minus_two_speed_result")
    later_ou = _lookup(result.paired, "later_pure_minus_two_speed_ou")
    setting_pure = _lookup(result.paired, "setting_pure_minus_two_speed_scoreline")
    lines = [
        "# Two speeds of team memory, one weight",
        "",
        f"**Date:** {result.run_date} (London)",
        (
            "**Question:** does one weight, mixing a slow Dixon-Coles fit with a "
            "fast Dixon-Coles fit, put more probability on the score that happened "
            "than the current pure model does, on matches the weight was not chosen on?"
        ),
        "",
        "## What was fit",
        "",
        (
            "The pure model is the current Dixon-Coles score table. Its time-decay "
            f"setting stays at {result.slow_xi:.4f} and the thin-data strength stays at 32. "
            "Its low-score correction is the value fitted on each round. The "
            "model-plus-market weight stays at 0.25. The stopped score-table "
            "calibration is unchanged."
        ),
        "",
        (
            "The two-speed line fits that same model twice. The slow fit uses "
            f"{result.slow_xi:.4f}. The fast fit uses {result.fast_xi:.4f}, the fastest "
            "decay already listed for the time-decay search, about a 58-day half-life. "
            "No other decay was tried. One weight mixes the fast fit's home rate and "
            "away rate with the slow fit's rates. The same weight is used for both. "
            "The low-score correction stays the slow fit's correction. The weights "
            "tried are the rows in the table below. Nothing else was tried "
            "after looking. This is not a state-space model and it is not a Kalman filter. "
            "The closing price is not an input."
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
            "Each round's two fits use only matches dated before that round. "
            "The weight is the one with the lowest scoreline log loss on seasons "
            "2223, 2324 and 2425. A tie keeps the smaller weight. Those seasons were "
            "already used to choose other settings, so this table is where the weight "
            "was picked. It is not a clean test."
        ),
        "",
        _candidate_markdown(result.candidates),
        "",
        f"The chosen weight on the fast fit is {_fmt_weight(result.chosen_weight)}.",
        "",
        _edge_of_list_sentence(result),
        "",
        (
            "The gap below is on the setting seasons. Those matches were used to "
            "pick the weight, so this gap is not a clean test."
        ),
        "",
        _gain_sentence(
            "On the setting seasons, scoreline log loss",
            "the pure model minus the two-speed line",
            setting_pure,
        ),
        "",
        (
            "Seasons "
            f"{', '.join(TWO_SPEED_LATER_SEASONS)} were not used to pick the weight."
        ),
        "",
        "## Later seasons",
        "",
        (
            "Both fits are redone every round on earlier matches. The weight stays "
            "at the chosen value. A range is a 95% percentile interval from "
            f"{BOOTSTRAP_DRAWS} resamples of whole rounds, seed {BOOTSTRAP_SEED}. "
            "A range that contains zero has not separated the two lines on this sample."
        ),
        "",
        "### Against the current pure model",
        "",
        (
            "This is the stop comparison. The difference is the pure model minus "
            "the two-speed line. A positive number means the two-speed line has "
            "the lower scoreline log loss."
        ),
        "",
        _gain_sentence(
            "On every later-season match, scoreline log loss",
            "the pure model minus the two-speed line",
            later_pure,
        ),
        "",
        _gain_sentence(
            "On those same matches, home/draw/away log loss",
            "the pure model minus the two-speed line",
            later_result,
        ),
        "",
        _gain_sentence(
            "On those same matches, over/under 2.5 log loss",
            "the pure model minus the two-speed line",
            later_ou,
        ),
        "",
        (
            "Home, draw or away log loss and over/under 2.5 log loss do not choose "
            "the weight, and they do not meet the edge definition."
        ),
        "",
        "### Against the reconstructed closing table",
        "",
        (
            "This is the edge comparison. The difference is the reconstructed closing "
            "table minus the two-speed line. A positive number means the two-speed "
            "line has the lower scoreline log loss. The closing price is the "
            "competitor. It is not an input."
        ),
        "",
        _gain_sentence(
            "On the later-season matches with a closing table, scoreline log loss",
            "the reconstructed closing table minus the two-speed line",
            later_book,
        ),
        "",
        _gain_sentence(
            "On those same matches, home/draw/away log loss",
            "the reconstructed closing table minus the two-speed line",
            later_book_result,
        ),
        "",
        "## Decision",
        "",
        _decision_paragraph(result, later_pure, later_book),
        "",
        _coverage_sentence(result),
        "",
        "## Regenerate",
        "",
        "```bash",
        "python -m src.evaluate --mode two_speed",
        "```",
        "",
        (
            f"The run took {result.elapsed_seconds / 60.0:.1f} minutes. "
            "The per-match rows are `results/two_speed_matches.csv`. "
            "The ranges are `results/two_speed_paired.csv`."
        ),
        "",
    ]
    text = "\n".join(line for line in lines if line is not None)
    _reject_bad_language(text)
    return text


def _check_windows(weights, slow_xi: float, fast_xi: float, setting, later) -> None:
    if not weights:
        raise ValueError("weights must be non-empty")
    if any((not math.isfinite(weight)) or weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("weights must be between 0 and 1")
    if not math.isfinite(slow_xi) or slow_xi <= 0.0:
        raise ValueError("slow decay must be finite and positive")
    if not math.isfinite(fast_xi) or fast_xi <= slow_xi:
        raise ValueError("fast decay must be faster than the slow decay")
    if not any(math.isclose(fast_xi, candidate) for candidate in XI_TUNE_CANDIDATES):
        raise ValueError("fast decay must be one of the decays already listed")
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
    return all(
        math.isfinite(_price(row, name)) and _price(row, name) > 0.0
        for name in CLOSING_PRICE_COLUMNS
    )


def _walk_seasons(
    frame: pd.DataFrame,
    *,
    seasons: set[str],
    weights: tuple[float, ...],
    slow_xi: float,
    fast_xi: float,
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
        slow = fit_dixon_coles(train, xi=slow_xi)
        fast = fit_dixon_coles(train, xi=fast_xi)
        for row in test.itertuples(index=False):
            identity = _identity(row, season, gw)
            pure_grid = scoreline_grid(slow, row.home_team, row.away_team)
            pure_lam, pure_mu = expected_goals(slow, row.home_team, row.away_team)
            rows.append(
                {
                    **identity,
                    **_grid_scores(
                        pure_grid,
                        int(row.home_goals),
                        int(row.away_goals),
                        pure_lam,
                        pure_mu,
                        float(slow.rho),
                        str(row.result),
                    ),
                    "model": _PURE,
                    "weight": 0.0,
                }
            )
            for weight in weights:
                grid, lam, mu, rho = two_speed_grid(
                    slow, fast, row.home_team, row.away_team, weight
                )
                rows.append(
                    {
                        **identity,
                        **_grid_scores(
                            grid,
                            int(row.home_goals),
                            int(row.away_goals),
                            lam,
                            mu,
                            rho,
                            str(row.result),
                        ),
                        "model": weight_name(weight),
                        "weight": float(weight),
                    }
                )
            if not _prices_ok(row):
                skipped += 1
                continue
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
                }
            )
    if not rows:
        raise ValueError(f"no scored rounds in seasons {sorted(seasons)}")
    return pd.DataFrame(rows), checks, skipped


def _grid_scores(grid, home_goals, away_goals, lam, mu, rho, result) -> dict[str, float]:
    scored = score_scoreline(grid, home_goals, away_goals, lam=lam, mu=mu, rho=rho)
    markets = markets_from_grid(grid)
    return {
        "scoreline_loss": float(scored["cell_log_loss"]),
        "result_loss": _result_loss(result, markets["home"], markets["draw"], markets["away"]),
        "ou_loss": _ou_loss(home_goals, away_goals, markets["over_25"], markets["under_25"]),
        "lambda_home": float(lam),
        "lambda_away": float(mu),
        "rho": float(rho),
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


def _ou_loss(home_goals: int, away_goals: int, over: float, under: float) -> float:
    probability = float(over) if int(home_goals) + int(away_goals) >= 3 else float(under)
    probability = min(max(probability, 1e-15), 1.0)
    return float(-math.log(probability))


def _tag_lines(scores: pd.DataFrame, weight: float) -> pd.DataFrame:
    chosen = scores.loc[scores["model"] == weight_name(weight)].copy()
    chosen["model"] = _LINE
    return pd.concat([scores, chosen], ignore_index=True)


def _assert_weight_zero_matches_pure(scores: pd.DataFrame) -> None:
    zero = scores.loc[scores["model"] == weight_name(0.0)]
    if zero.empty:
        return
    pure = scores.loc[scores["model"] == _PURE, ["match_id", "scoreline_loss"]]
    merged = zero.merge(pure, on="match_id", suffixes=("_zero", "_pure"))
    if len(merged) != len(zero):
        raise ValueError("weight 0 and the pure model were scored on different matches")
    delta = (merged["scoreline_loss_zero"] - merged["scoreline_loss_pure"]).abs().max()
    if float(delta) > 1e-9:
        raise ValueError(f"weight 0 does not match the pure model (max abs diff {delta})")


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
                "chosen": bool(math.isclose(float(weight), float(chosen))),
            }
        )
    return pd.DataFrame(rows)


def _paired_rows(
    scores: pd.DataFrame,
    setting_seasons: tuple[str, ...],
    later_seasons: tuple[str, ...],
) -> list[dict]:
    later = _slice(scores, later_seasons)
    return [
        _diff_row(
            "setting_pure_minus_two_speed_scoreline",
            _slice(scores, setting_seasons),
            _PURE,
            _LINE,
            "scoreline_loss",
            "Setting seasons chose the weight. This gap is not a clean test.",
        ),
        _diff_row(
            "later_pure_minus_two_speed_scoreline",
            later,
            _PURE,
            _LINE,
            "scoreline_loss",
            "Positive means the two-speed line has the lower scoreline log loss than the pure model.",
        ),
        _diff_row(
            "later_pure_minus_two_speed_result",
            later,
            _PURE,
            _LINE,
            "result_loss",
            "Positive means the two-speed line has the lower home/draw/away log loss. Not the stop rule.",
        ),
        _diff_row(
            "later_pure_minus_two_speed_ou",
            later,
            _PURE,
            _LINE,
            "ou_loss",
            "Positive means the two-speed line has the lower over/under 2.5 log loss. Not the stop rule.",
        ),
        _diff_row(
            "later_book_minus_two_speed_scoreline",
            later,
            _BOOK,
            _LINE,
            "scoreline_loss",
            "Positive means the two-speed line has the lower scoreline log loss than the reconstructed closing table.",
        ),
        _diff_row(
            "later_book_minus_two_speed_result",
            later,
            _BOOK,
            _LINE,
            "result_loss",
            "Positive means the two-speed line has the lower home/draw/away log loss than the reconstructed closing table. Not the edge definition.",
        ),
    ]


def _slice(scores: pd.DataFrame, seasons: tuple[str, ...]) -> pd.DataFrame:
    wanted = {str(season) for season in seasons}
    return scores.loc[scores["season_code"].astype(str).isin(wanted)].copy()


def _diff_row(name, scores, left_name, right_name, column, note) -> dict:
    left = scores.loc[scores["model"] == left_name]
    right = scores.loc[scores["model"] == right_name]
    shared = set(left["match_id"]) & set(right["match_id"])
    left = left.loc[left["match_id"].isin(shared)]
    right = right.loc[right["match_id"].isin(shared)]
    return _gap_row(name, left_name, right_name, _align_diff(left, right, column), note)


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


def _decision_from_paired(paired: pd.DataFrame, weight: float) -> tuple[bool, bool, bool, str]:
    pure = _lookup(paired, "later_pure_minus_two_speed_scoreline")
    book = _lookup(paired, "later_book_minus_two_speed_scoreline")
    dropped, reason = stop_decision(
        weight,
        float(pure["mean_diff"]),
        float(pure["ci_low"]),
        float(pure["ci_high"]),
    )
    edge = edge_met(float(book["mean_diff"]), float(book["ci_low"]), float(book["ci_high"]))
    replaces = (not dropped) and edge
    return dropped, edge, replaces, reason


def _lookup(paired: pd.DataFrame, name: str) -> pd.Series:
    rows = paired.loc[paired["comparison"] == name]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {name}, found {len(rows)}")
    return rows.iloc[0]


def _candidate_markdown(candidates: pd.DataFrame) -> str:
    header = "| Weight on the fast fit | Scoreline log loss | Matches | Home/draw/away log loss |"
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
    if range_includes_zero(float(row["ci_low"]), float(row["ci_high"])):
        text += " The range includes zero."
    return text


def _decision_paragraph(result: TwoSpeedResult, later_pure: pd.Series, later_book: pd.Series) -> str:
    gain = float(later_pure["mean_diff"])
    band = (
        "The plan's expected shrink against the current pure model was 0.000 to 0.015. "
        "That size is not enough, on its own, to clear the reconstructed closing table."
    )
    if math.isclose(float(result.chosen_weight), 0.0):
        band += (
            " The chosen weight is 0.00, so on the later seasons the two-speed line "
            "is the current pure model. The paired gain is zero because the two lines "
            "are the same."
        )
        ordered = result.candidates.sort_values("weight")
        losses = [float(value) for value in ordered["scoreline_log_loss"]]
        if len(losses) > 1 and all(losses[i] <= losses[i + 1] + 1e-12 for i in range(len(losses) - 1)):
            band += (
                " On the setting seasons the scoreline log loss rose as the weight "
                "on the fast fit rose."
            )
    elif 0.0 <= gain <= 0.015:
        band += " The later-season gain versus the pure model is inside that band."
    if result.dropped:
        return (
            f"The stop rule drops the two-speed line: {result.reason}. "
            f"The weight on the fast fit is {_fmt_weight(result.chosen_weight)}. "
            "The published model stays the current pure model. "
            f"{band} {_edge_words(result, later_book)} {_level_sentence(result)}"
        )
    if result.replaces_pure:
        return (
            f"The stop rule does not fire: {result.reason}. "
            "The edge definition is met, so the two-speed line replaces the pure model. "
            f"The published weight on the fast fit is {_fmt_weight(result.chosen_weight)}. "
            f"{_edge_words(result, later_book)}"
        )
    return (
        f"The stop rule does not fire: {result.reason}. "
        "The edge definition is not met, so the two-speed line is a labelled alternative. "
        "It does not replace the pure model. "
        f"The weight on the fast fit is {_fmt_weight(result.chosen_weight)}. "
        f"{band} {_edge_words(result, later_book)}"
    )


def _edge_words(result: TwoSpeedResult, later_book: pd.Series) -> str:
    gap = float(later_book["mean_diff"])
    if result.edge:
        return (
            "On the later seasons the two-speed line has the lower scoreline log loss "
            f"than the reconstructed closing table, by {gap:.4f}, and that range excludes zero. "
            f"The comparison uses {int(later_book['n_matches'])} matches and "
            f"{int(later_book['n_rounds'])} rounds."
        )
    return (
        "The edge definition is not met. The reconstructed closing table is the competitor. "
        "This result is not a claim about the bookmaker. "
        f"The scoreline gap against that table is {gap:.4f} "
        f"(95% range {float(later_book['ci_low']):.4f} to {float(later_book['ci_high']):.4f}, "
        f"{int(later_book['n_matches'])} matches, {int(later_book['n_rounds'])} rounds)."
    )


def _level_sentence(result: TwoSpeedResult) -> str:
    """Absolute losses, so the gap is not read off an older walk-forward file."""
    if result.matches.empty:
        return ""
    later = result.matches.loc[
        result.matches["season_code"].astype(str).isin(set(TWO_SPEED_LATER_SEASONS))
    ]
    pure = later.loc[later["model"] == _PURE, "scoreline_loss"]
    book = later.loc[later["model"] == _BOOK, "scoreline_loss"]
    if pure.empty or book.empty or len(pure) != len(book):
        return ""
    return (
        f"On those {int(len(pure))} matches the pure model's scoreline log loss is "
        f"{float(pure.mean()):.4f} and the reconstructed closing table's is "
        f"{float(book.mean()):.4f}. The saved walk-forward gap of about 0.10 is the "
        "model from before strength 32. This run uses strength 32 and does not retune it."
    )


def _edge_of_list_sentence(result: TwoSpeedResult) -> str:
    weights = [float(value) for value in result.candidates["weight"]]
    weight = float(result.chosen_weight)
    if len(weights) > 1 and any(math.isclose(weight, end) for end in (min(weights), max(weights))):
        return (
            f"Weight {_fmt_weight(weight)} is at the end of the list that was written down. "
            "No further value was added after looking."
        )
    return ""


def _coverage_sentence(result: TwoSpeedResult) -> str:
    if result.skipped_book_matches == 0:
        return (
            "Every scored match had a closing home, draw, away and over/under 2.5 price, "
            "so the pure-model comparison and the closing-table comparison use the same matches. "
            "Rounds with fewer than 200 earlier matches were not scored."
        )
    return (
        f"{int(result.skipped_book_matches)} matches were left out of the closing-table "
        "comparison because a closing price was missing. The pure-model comparison uses "
        "every scored match. Rounds with fewer than 200 earlier matches were not scored."
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
