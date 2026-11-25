"""One scale and one steady low-score correction for the score table.

The pair is chosen on seasons 2223, 2324 and 2425 only, by scoreline log
loss. Seasons 2526 and 2627 are scored after the pair is frozen. They are
not used to pick it. Each scored round is fit only on earlier matches.
The scale and the correction are applied after that fit, to both goal rates
and to the score table. They are not a second fit.

The stop rule is the one in
docs/experiments/2026-11-25-score-table-calibration.md.
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
    GRID_CALIBRATION_LATER_SEASONS,
    GRID_CALIBRATION_MIN_GAIN,
    GRID_CALIBRATION_RHOS,
    GRID_CALIBRATION_SCALES,
    GRID_CALIBRATION_SETTING_SEASONS,
    RESULTS,
)
from src.eval.scoreline import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    block_bootstrap_mean_diff,
    score_scoreline,
)
from src.models.dixon_coles import (
    calibrated_score_grid,
    expected_goals,
    fit_dixon_coles,
)
from src.models.markets import markets_from_grid

logger = logging.getLogger(__name__)

_BASELINE = "dixon_coles"
_CHOSEN = "grid_calibration"
_BOOK = "bookmaker_implied"


@dataclass(frozen=True)
class GridCalibrationResult:
    """Chosen pair, the stop decision, and the tables behind them."""

    chosen_scale: float
    chosen_rho: float
    dropped: bool
    reason: str
    candidates: pd.DataFrame
    paired: pd.DataFrame
    matches: pd.DataFrame
    elapsed_seconds: float
    run_date: str


def pair_name(scale: float, rho: float) -> str:
    """Stable label for one pair. The numbers are the ones written down."""
    return f"scale_{float(scale):.2f}_rho_{float(rho):.2f}"


def choose_pair(
    scores: pd.DataFrame,
    scales: tuple[float, ...] | list[float],
    rhos: tuple[float, ...] | list[float],
    setting_seasons: tuple[str, ...] | list[str],
) -> tuple[float, float]:
    """Lowest setting-season scoreline log loss. Ties keep the smaller change.

    Later seasons in ``scores`` are ignored. The previous line is ignored.
    """
    seasons = {str(season) for season in setting_seasons}
    setting = scores.loc[scores["season_code"].astype(str).isin(seasons)]
    best: tuple | None = None
    match_count: int | None = None
    for scale in scales:
        for rho in rhos:
            part = setting.loc[setting["model"] == pair_name(scale, rho)]
            if part.empty:
                raise ValueError(f"no setting-season rows for {pair_name(scale, rho)}")
            if match_count is None:
                match_count = int(len(part))
            elif len(part) != match_count:
                raise ValueError("candidates were scored on different matches")
            loss = float(part["scoreline_loss"].mean())
            key = (loss, abs(float(scale) - 1.0), abs(float(rho)), float(scale), float(rho))
            if best is None or key < best[0]:
                best = (key, float(scale), float(rho))
    if best is None:
        raise ValueError("no candidate pair")
    return best[1], best[2]


def stop_decision(scoreline_gain: float, min_gain: float = GRID_CALIBRATION_MIN_GAIN) -> tuple[bool, str]:
    """Drop when the later-season scoreline gain, at 4 d.p., is under ``min_gain``.

    The gain is the previous line minus the new line. Positive means the new
    line has the lower log loss.
    """
    gain = round(float(scoreline_gain), 4)
    if gain < float(min_gain):
        return True, (
            f"the paired scoreline gain on the later seasons is {gain:.4f}, "
            f"which is under {float(min_gain):.3f}"
        )
    return False, (
        f"the paired scoreline gain on the later seasons is {gain:.4f}, "
        f"which is not under {float(min_gain):.3f}"
    )


def run_grid_calibration(
    matches: pd.DataFrame | None = None,
    *,
    scales: tuple[float, ...] | list[float] | None = None,
    rhos: tuple[float, ...] | list[float] | None = None,
    setting_seasons: tuple[str, ...] | list[str] | None = None,
    later_seasons: tuple[str, ...] | list[str] | None = None,
) -> GridCalibrationResult:
    """Choose the pair on earlier seasons, then score the later ones."""
    from src.clean import build_matches_table
    from src.eval.gameweeks import assign_gameweeks
    from src.eval.walkforward import MIN_TRAIN_ROWS
    from src.features import add_implied_odds

    started = time.perf_counter()
    scales_t = tuple(float(item) for item in (GRID_CALIBRATION_SCALES if scales is None else scales))
    rhos_t = tuple(float(item) for item in (GRID_CALIBRATION_RHOS if rhos is None else rhos))
    setting_t = tuple(
        str(item)
        for item in (
            GRID_CALIBRATION_SETTING_SEASONS if setting_seasons is None else setting_seasons
        )
    )
    later_t = tuple(
        str(item)
        for item in (GRID_CALIBRATION_LATER_SEASONS if later_seasons is None else later_seasons)
    )
    _check_windows(scales_t, rhos_t, setting_t, later_t)

    if matches is None:
        matches = build_matches_table()
    frame = assign_gameweeks(matches)
    frame["date"] = pd.to_datetime(frame["date"])
    if "implied_home" not in frame.columns:
        frame = add_implied_odds(frame)

    pairs = tuple((scale, rho) for scale in scales_t for rho in rhos_t)
    setting_rows = _walk_seasons(
        frame,
        seasons=set(setting_t),
        pairs=pairs,
        min_train_rows=MIN_TRAIN_ROWS,
    )
    chosen_scale, chosen_rho = choose_pair(setting_rows, scales_t, rhos_t, setting_t)
    logger.info(
        "Chosen scale %.2f and correction %.2f from the setting seasons",
        chosen_scale,
        chosen_rho,
    )
    later_rows = _walk_seasons(
        frame,
        seasons=set(later_t),
        pairs=((chosen_scale, chosen_rho),),
        min_train_rows=MIN_TRAIN_ROWS,
    )
    scores = _tag_lines(
        pd.concat([setting_rows, later_rows], ignore_index=True),
        chosen_scale,
        chosen_rho,
    )
    paired = pd.DataFrame(_paired_rows(scores, setting_t, later_t))
    dropped, reason = _decision_from_paired(paired)
    return GridCalibrationResult(
        chosen_scale=float(chosen_scale),
        chosen_rho=float(chosen_rho),
        dropped=dropped,
        reason=reason,
        candidates=_candidate_table(scores, setting_t, scales_t, rhos_t, chosen_scale, chosen_rho),
        paired=paired,
        matches=scores,
        elapsed_seconds=time.perf_counter() - started,
        run_date=datetime.now(ZoneInfo("Europe/London")).strftime("%d %b %Y"),
    )


def write_grid_calibration_artefacts(
    result: GridCalibrationResult,
    results_dir=None,
) -> dict[str, str]:
    """Write the note and the tables. Does not change the published model."""
    root = RESULTS if results_dir is None else results_dir
    root.mkdir(parents=True, exist_ok=True)
    note = root / "GRID_CALIBRATION.md"
    candidates = root / "grid_calibration_candidates.csv"
    paired = root / "grid_calibration_paired.csv"
    matches = root / "grid_calibration_matches.csv"
    note.write_text(render_grid_calibration_note(result), encoding="utf-8")
    result.candidates.to_csv(candidates, index=False)
    result.paired.to_csv(paired, index=False)
    result.matches.to_csv(matches, index=False)
    return {
        "note": str(note),
        "candidates": str(candidates),
        "paired": str(paired),
        "matches": str(matches),
    }


def render_grid_calibration_note(result: GridCalibrationResult) -> str:
    """Plain-language result. Every gap has a range and a match count."""
    later_score = _lookup(result.paired, "later_scoreline")
    later_result = _lookup(result.paired, "later_result")
    setting_score = _lookup(result.paired, "setting_scoreline")
    lines = [
        "# A scale and a steady low-score correction",
        "",
        f"**Date:** {result.run_date} (London)",
        (
            "**Question:** does one scale on both goal rates, with one constant "
            "low-score correction, lower scoreline log loss on later seasons?"
        ),
        "",
        "## What was fit",
        "",
        (
            "The previous line is the current Dixon-Coles score table. The time-decay "
            "setting stays at 0.0035 and the thin-data strength stays at 32. "
            "The low-score correction on that line is the value fitted on each round."
        ),
        "",
        (
            "The new line takes those same fitted rates and multiplies both by one "
            "scale, then uses one constant low-score correction. The scales tried "
            "were 0.95, 1.00 and 1.05. The corrections tried were -0.10, -0.05 and "
            "0.00. Nothing else was tried after looking."
        ),
        "",
        "## How the pair was chosen",
        "",
        (
            "Each round is fit only on matches dated before that round. The pair is "
            "the one with the lowest scoreline log loss on seasons 2223, 2324 and "
            "2425. A tie keeps the scale closer to 1, then the correction closer to "
            "0. Those seasons were already used to choose other settings, so this "
            "table is where the pair was picked. It is not a clean test."
        ),
        "",
        _candidate_markdown(result.candidates),
        "",
        (
            f"The chosen pair is scale {_fmt_scale(result.chosen_scale)} and "
            f"correction {_fmt_rho(result.chosen_rho)}."
        ),
        "",
        _edge_sentence(result),
        "",
        (
            "The gap below is on the setting seasons. Those matches were used to "
            "pick the pair, so this gap is not a clean test."
        ),
        "",
        _gain_sentence("On the setting seasons, scoreline log loss", setting_score),
        "",
        (
            "Seasons "
            f"{', '.join(GRID_CALIBRATION_LATER_SEASONS)} were not used to pick the pair."
        ),
        "",
        "## Later seasons",
        "",
        (
            "The team rates are refit every round on earlier matches. The scale and "
            "the correction stay at the chosen pair. A gain is the previous line "
            "minus the new line on one shared set of matches. The range is a 95% "
            f"percentile interval from {BOOTSTRAP_DRAWS} resamples of whole rounds, "
            f"seed {BOOTSTRAP_SEED}. A positive gain means the new line has the "
            "lower log loss. A range that contains zero has not separated the two "
            "lines on this sample."
        ),
        "",
        "### Scoreline",
        "",
        _gain_sentence("On every later-season match, scoreline log loss", later_score),
        "",
        "### Home, draw or away",
        "",
        _gain_sentence(
            "On those same matches, home/draw/away log loss",
            later_result,
        ),
        "",
        "### Closing line",
        "",
        _book_paragraph(result.paired),
        "",
        "## Decision",
        "",
        _decision_paragraph(result, later_score),
        "",
        "## Regenerate",
        "",
        "```bash",
        "python -m src.evaluate --mode grid_calibration",
        "```",
        "",
        (
            f"The run took {result.elapsed_seconds / 60.0:.1f} minutes. "
            "The per-match rows are `results/grid_calibration_matches.csv`. "
            "The ranges are `results/grid_calibration_paired.csv`."
        ),
        "",
    ]
    text = "\n".join(line for line in lines if line is not None)
    _reject_bad_language(text)
    return text


def _check_windows(scales, rhos, setting, later) -> None:
    if not scales or not rhos:
        raise ValueError("scales and corrections must both be non-empty")
    if any((not math.isfinite(scale)) or scale <= 0.0 for scale in scales):
        raise ValueError("scales must be finite and positive")
    if any(not math.isfinite(rho) for rho in rhos):
        raise ValueError("corrections must be finite")
    overlap = set(setting) & set(later)
    if overlap:
        raise ValueError(f"setting and later seasons overlap: {sorted(overlap)}")
    if not setting or not later:
        raise ValueError("setting and later seasons must both be non-empty")


def _walk_seasons(
    frame: pd.DataFrame,
    *,
    seasons: set[str],
    pairs: tuple[tuple[float, float], ...],
    min_train_rows: int,
) -> pd.DataFrame:
    from src.eval.walkforward import iter_folds

    rows: list[dict] = []
    seen = 0
    for train, test, season, gw in iter_folds(frame, min_train_rows=min_train_rows):
        if str(season) not in seasons:
            continue
        seen += 1
        if seen == 1 or seen % 10 == 0:
            logger.info("Scoring %s round %s", season, gw)
        model = fit_dixon_coles(train)
        for row in test.itertuples(index=False):
            identity = _identity(row, season, gw)
            book_loss = _book_result_loss(row)
            lam, mu = expected_goals(model, row.home_team, row.away_team)
            rows.append(
                {
                    **identity,
                    **_scored_row(lam, mu, model.rho, row, scale=1.0, rho=None),
                    "model": _BASELINE,
                    "scale": 1.0,
                    "rho": float(model.rho),
                    "book_result_loss": book_loss,
                }
            )
            for scale, rho in pairs:
                rows.append(
                    {
                        **identity,
                        **_scored_row(lam, mu, model.rho, row, scale=scale, rho=rho),
                        "model": pair_name(scale, rho),
                        "scale": float(scale),
                        "rho": float(rho),
                        "book_result_loss": book_loss,
                    }
                )
    if not rows:
        raise ValueError(f"no scored rounds in seasons {sorted(seasons)}")
    return pd.DataFrame(rows)


def _scored_row(lam, mu, fitted_rho, row, *, scale: float, rho: float | None) -> dict:
    grid, used_lam, used_mu, used_rho = calibrated_score_grid(
        lam, mu, fitted_rho, scale=scale, rho=rho
    )
    scored = score_scoreline(
        grid,
        int(row.home_goals),
        int(row.away_goals),
        lam=used_lam,
        mu=used_mu,
        rho=used_rho,
    )
    markets = markets_from_grid(grid)
    return {
        "scoreline_loss": float(scored["cell_log_loss"]),
        "result_loss": _result_loss(
            str(row.result), markets["home"], markets["draw"], markets["away"]
        ),
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


def _book_result_loss(row) -> float:
    try:
        home = float(row.implied_home)
        draw = float(row.implied_draw)
        away = float(row.implied_away)
    except (TypeError, ValueError, AttributeError):
        return float("nan")
    if not all(math.isfinite(p) and p > 0.0 for p in (home, draw, away)):
        return float("nan")
    if abs((home + draw + away) - 1.0) > 1e-6:
        return float("nan")
    return _result_loss(str(row.result), home, draw, away)


def _tag_lines(scores: pd.DataFrame, scale: float, rho: float) -> pd.DataFrame:
    chosen = scores.loc[scores["model"] == pair_name(scale, rho)].copy()
    chosen["model"] = _CHOSEN
    return pd.concat([scores, chosen], ignore_index=True)


def _candidate_table(
    scores: pd.DataFrame,
    setting_seasons: tuple[str, ...],
    scales: tuple[float, ...],
    rhos: tuple[float, ...],
    chosen_scale: float,
    chosen_rho: float,
) -> pd.DataFrame:
    seasons = {str(season) for season in setting_seasons}
    setting = scores.loc[scores["season_code"].astype(str).isin(seasons)]
    rows = []
    for scale in scales:
        for rho in rhos:
            part = setting.loc[setting["model"] == pair_name(scale, rho)]
            rows.append(
                {
                    "scale": float(scale),
                    "rho": float(rho),
                    "scoreline_log_loss": round(float(part["scoreline_loss"].mean()), 4),
                    "matches": int(len(part)),
                    "result_log_loss": round(float(part["result_loss"].mean()), 4),
                    "chosen": bool(
                        float(scale) == float(chosen_scale) and float(rho) == float(chosen_rho)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_rows(
    scores: pd.DataFrame,
    setting_seasons: tuple[str, ...],
    later_seasons: tuple[str, ...],
) -> list[dict]:
    rows = [
        _gain_row(
            "setting_scoreline",
            _slice(scores, setting_seasons),
            "scoreline_loss",
            "Setting seasons chose the pair. This gap is not a clean test.",
        ),
        _gain_row(
            "later_scoreline",
            _slice(scores, later_seasons),
            "scoreline_loss",
            "Positive means the new line has the lower scoreline log loss.",
        ),
        _gain_row(
            "later_result",
            _slice(scores, later_seasons),
            "result_loss",
            "Positive means the new line has the lower home/draw/away log loss. Not the stop rule.",
        ),
    ]
    rows.extend(_book_rows(_slice(scores, later_seasons)))
    return rows


def _slice(scores: pd.DataFrame, seasons: tuple[str, ...]) -> pd.DataFrame:
    wanted = {str(season) for season in seasons}
    return scores.loc[scores["season_code"].astype(str).isin(wanted)].copy()


def _gain_row(name: str, scores: pd.DataFrame, column: str, note: str) -> dict:
    previous = scores.loc[scores["model"] == _BASELINE]
    new = scores.loc[scores["model"] == _CHOSEN]
    return _gap_row(name, _BASELINE, _CHOSEN, _align_gain(previous, new, column), note)


def _book_rows(scores: pd.DataFrame) -> list[dict]:
    base = scores.loc[scores["model"] == _BASELINE].copy()
    usable = base["book_result_loss"].map(lambda value: math.isfinite(float(value)))
    book_ids = set(base.loc[usable, "match_id"])
    if not book_ids:
        return []
    previous = base.loc[base["match_id"].isin(book_ids)]
    book = previous.copy()
    book["model"] = _BOOK
    book["result_loss"] = book["book_result_loss"]
    new = scores.loc[(scores["model"] == _CHOSEN) & scores["match_id"].isin(book_ids)]
    note = (
        "A positive difference means the first line has the higher log loss. "
        "The closing line is the competitor. This is not a claim about the bookmaker."
    )
    return [
        _gap_row(
            "later_book_new",
            _CHOSEN,
            _BOOK,
            _align_diff(new, book, "result_loss"),
            note,
        ),
        _gap_row(
            "later_book_previous",
            _BASELINE,
            _BOOK,
            _align_diff(previous, book, "result_loss"),
            note,
        ),
    ]


def _align_gain(previous: pd.DataFrame, new: pd.DataFrame, column: str) -> dict:
    """Previous minus new. Positive means the new line has the lower log loss."""
    gap = _align_diff(previous, new, column)
    return gap


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


def _decision_from_paired(paired: pd.DataFrame) -> tuple[bool, str]:
    return stop_decision(float(_lookup(paired, "later_scoreline")["mean_diff"]))


def _lookup(paired: pd.DataFrame, name: str) -> pd.Series:
    rows = paired.loc[paired["comparison"] == name]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {name}, found {len(rows)}")
    return rows.iloc[0]


def _candidate_markdown(candidates: pd.DataFrame) -> str:
    header = (
        "| Scale | Correction | Scoreline log loss | Matches | Home/draw/away log loss |"
    )
    rule = "|---|---|---|---|---|"
    body = []
    for row in candidates.itertuples(index=False):
        mark = " (chosen)" if bool(row.chosen) else ""
        body.append(
            f"| {_fmt_scale(row.scale)}{mark} | {_fmt_rho(row.rho)} | "
            f"{float(row.scoreline_log_loss):.4f} | {int(row.matches)} | "
            f"{float(row.result_log_loss):.4f} |"
        )
    return "\n".join([header, rule, *body])


def _gain_sentence(prefix: str, row: pd.Series) -> str:
    text = (
        f"{prefix}: the previous line minus the new line is {float(row['mean_diff']):.4f} "
        f"(95% range {float(row['ci_low']):.4f} to {float(row['ci_high']):.4f}, "
        f"{int(row['n_matches'])} matches, {int(row['n_rounds'])} rounds)."
    )
    if float(row["ci_low"]) <= 0.0 <= float(row["ci_high"]):
        text += " The range includes zero."
    return text


def _book_paragraph(paired: pd.DataFrame) -> str:
    if "later_book_new" not in set(paired["comparison"]):
        return "No closing home/draw/away line was available on the later-season matches."
    new = _lookup(paired, "later_book_new")
    previous = _lookup(paired, "later_book_previous")
    return (
        "On the later-season matches that have a closing home/draw/away line, "
        f"the new line minus the closing line is {float(new['mean_diff']):.4f} "
        f"(95% range {float(new['ci_low']):.4f} to {float(new['ci_high']):.4f}, "
        f"{int(new['n_matches'])} matches, {int(new['n_rounds'])} rounds). "
        "The previous line minus the closing line on those same matches is "
        f"{float(previous['mean_diff']):.4f} "
        f"(95% range {float(previous['ci_low']):.4f} to {float(previous['ci_high']):.4f}, "
        f"{int(previous['n_matches'])} matches, {int(previous['n_rounds'])} rounds). "
        "The closing line is the competitor. These gaps are not a claim about the "
        "bookmaker. A positive difference means that line has the higher log loss. "
        "This run does not rebuild a score table from book prices."
    )


def _decision_paragraph(result: GridCalibrationResult, later_score: pd.Series) -> str:
    gain = float(later_score["mean_diff"])
    if result.dropped:
        return (
            f"The stop rule drops the change: {result.reason}. "
            "The published score table stays the previous one: no extra scale, "
            "and the low-score correction fitted on each round."
        )
    text = (
        f"The stop rule keeps the change: {result.reason}. "
        f"The published score table uses scale {_fmt_scale(result.chosen_scale)} "
        f"and correction {_fmt_rho(result.chosen_rho)}. "
        "The rule uses the point estimate, rounded to 4 decimal places."
    )
    if abs(gain) < 0.01:
        text += (
            " The gain is under 0.01. That is a point estimate inside the range above. "
            "It is not a success, and it is not a claim about the bookmaker."
        )
    else:
        text += " It is not a claim about the bookmaker."
    return text


def _edge_sentence(result: GridCalibrationResult) -> str:
    scales = [float(value) for value in result.candidates["scale"].unique()]
    rhos = [float(value) for value in result.candidates["rho"].unique()]
    bits = []
    scale = float(result.chosen_scale)
    rho = float(result.chosen_rho)
    if len(scales) > 1 and scale in (min(scales), max(scales)) and scale != 1.0:
        bits.append(
            f"Scale {_fmt_scale(scale)} is at the end of the list that was written down."
        )
    if len(rhos) > 1 and rho in (min(rhos), max(rhos)):
        bits.append(
            f"Correction {_fmt_rho(rho)} is at the end of the list that was written down."
        )
    if not bits:
        return ""
    bits.append("No further value was added after looking.")
    return " ".join(bits)


def _fmt_scale(value: float) -> str:
    return f"{float(value):.2f}"


def _fmt_rho(value: float) -> str:
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
