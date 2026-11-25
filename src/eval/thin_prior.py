"""One strength for clubs with few matches.

The strength is chosen on seasons 2223, 2324 and 2425 only, by the scoreline
log loss of early matches. Early means a club's first eight matches in that
season. Seasons 2526 and 2627 are scored after the strength is frozen. They
are not used to pick it. Each scored round is fit only on earlier matches.

The stop rule is the one in docs/experiments/2026-11-25-thin-data-prior.md.
This module does not read the live record.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import (
    RESULTS,
    THIN_PRIOR_CANDIDATES,
    THIN_PRIOR_EARLY_MATCHES,
    THIN_PRIOR_LATER_SEASONS,
    THIN_PRIOR_PROMOTED_FACTOR,
    THIN_PRIOR_SETTING_SEASONS,
)
from src.eval.scoreline import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    block_bootstrap_mean_diff,
    score_scoreline,
)
from src.models.dixon_coles import expected_goals, fit_dixon_coles, scoreline_grid
from src.models.markets import markets_from_grid

logger = logging.getLogger(__name__)

_BASELINE = "dixon_coles"
_CHOSEN = "thin_prior"
_BOOK = "bookmaker_implied"


@dataclass(frozen=True)
class ThinPriorResult:
    """Chosen strength, the stop decision, and the tables behind them."""

    chosen_kappa: float
    dropped: bool
    reason: str
    candidates: pd.DataFrame
    paired: pd.DataFrame
    matches: pd.DataFrame
    elapsed_seconds: float
    run_date: str


def kappa_name(kappa: float) -> str:
    """Stable label for one strength. ``0`` is the previous fit."""
    value = float(kappa)
    if value.is_integer():
        return f"prior_{int(value)}"
    return f"prior_{value:g}"


def prior_appearance_counts(
    season: pd.Series,
    team: pd.Series,
    date: pd.Series,
) -> np.ndarray:
    """How many earlier matches that club has already played in that season.

    Matches on the same date do not count as earlier.
    """
    frame = pd.DataFrame(
        {
            "season": pd.Series(season).astype(str).to_numpy(),
            "team": pd.Series(team).astype(str).to_numpy(),
            "date": pd.to_datetime(pd.Series(date)).to_numpy(),
        }
    )
    frame["_ord"] = np.arange(len(frame))
    ordered = frame.sort_values(["season", "team", "date", "_ord"], kind="mergesort")
    counts = np.empty(len(frame), dtype=int)
    for _, group in ordered.groupby(["season", "team"], sort=False):
        dates = group["date"].to_numpy()
        counts[group["_ord"].to_numpy()] = np.searchsorted(dates, dates, side="left")
    return counts


def annotate_matches(
    matches: pd.DataFrame,
    early_limit: int = THIN_PRIOR_EARLY_MATCHES,
) -> pd.DataFrame:
    """Add ``early`` and ``new_club_early`` from the fixture list, not the score.

    ``early`` is the pre-registered slice: either club is inside its first
    ``early_limit`` matches of the season. ``new_club_early`` is the smaller
    slice where that club also missed the previous season. The stop rule uses
    ``early`` only.
    """
    if early_limit < 1:
        raise ValueError("early_limit must be at least 1")
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    home_prior = prior_appearance_counts(
        frame["season_code"], frame["home_team"], frame["date"]
    )
    away_prior = prior_appearance_counts(
        frame["season_code"], frame["away_team"], frame["date"]
    )
    frame["early"] = (home_prior < early_limit) | (away_prior < early_limit)
    frame["new_club_early"] = _new_club_early(
        frame, home_prior, away_prior, early_limit
    )
    return frame


def choose_kappa(
    scores: pd.DataFrame,
    candidates: tuple[float, ...] | list[float],
    setting_seasons: tuple[str, ...] | list[str],
) -> float:
    """Lowest early-match scoreline log loss on the setting seasons.

    Later seasons in ``scores`` are ignored. Ties keep the smaller strength,
    which is closer to the previous fit.
    """
    seasons = {str(season) for season in setting_seasons}
    setting = scores.loc[
        scores["season_code"].astype(str).isin(seasons) & scores["early"].astype(bool)
    ]
    best_mean: float | None = None
    best_kappa: float | None = None
    for kappa in sorted(float(item) for item in candidates):
        part = setting.loc[setting["model"] == kappa_name(kappa), "scoreline_loss"]
        if part.empty:
            continue
        values = part.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{kappa_name(kappa)} has a non-finite scoreline loss")
        mean = float(values.mean())
        if best_mean is None or mean < best_mean:
            best_mean = mean
            best_kappa = kappa
    if best_kappa is None:
        raise ValueError("no setting-era early matches to choose a strength from")
    return float(best_kappa)


def stop_decision(
    early_scoreline_diff: float,
    overall_scoreline_diff: float,
    overall_result_diff: float,
) -> tuple[bool, str]:
    """Whether to drop the change. Differences are prior minus previous.

    A negative difference means the prior line has the lower log loss.
    Rounded figures are what the note quotes, so the decision uses those.
    """
    early = round(float(early_scoreline_diff), 4)
    overall_score = round(float(overall_scoreline_diff), 4)
    overall_result = round(float(overall_result_diff), 4)
    reasons: list[str] = []
    if not early < 0.0:
        reasons.append(
            "early-match scoreline log loss is not lower on the later seasons"
        )
    if overall_score > 0.0:
        reasons.append("overall scoreline log loss is higher on the later seasons")
    if overall_result > 0.0:
        reasons.append(
            "overall home/draw/away log loss is higher on the later seasons"
        )
    if reasons:
        return True, "; ".join(reasons)
    return False, (
        "early-match scoreline log loss is lower on the later seasons, "
        "and the overall scores are not higher"
    )


def run_thin_prior(
    matches: pd.DataFrame | None = None,
    *,
    candidates: tuple[float, ...] | list[float] | None = None,
    setting_seasons: tuple[str, ...] | list[str] | None = None,
    later_seasons: tuple[str, ...] | list[str] | None = None,
    early_limit: int = THIN_PRIOR_EARLY_MATCHES,
    promoted_factor: float = THIN_PRIOR_PROMOTED_FACTOR,
) -> ThinPriorResult:
    """Choose the strength on earlier seasons, then score the later ones."""
    from src.clean import build_matches_table
    from src.eval.gameweeks import assign_gameweeks
    from src.eval.walkforward import MIN_TRAIN_ROWS, iter_folds
    from src.features import add_implied_odds

    started = time.perf_counter()
    candidates_t = tuple(
        float(item) for item in (THIN_PRIOR_CANDIDATES if candidates is None else candidates)
    )
    setting_t = tuple(
        str(item)
        for item in (
            THIN_PRIOR_SETTING_SEASONS if setting_seasons is None else setting_seasons
        )
    )
    later_t = tuple(
        str(item)
        for item in (THIN_PRIOR_LATER_SEASONS if later_seasons is None else later_seasons)
    )
    _check_windows(candidates_t, setting_t, later_t)

    if matches is None:
        matches = build_matches_table()
    frame = assign_gameweeks(matches)
    frame["date"] = pd.to_datetime(frame["date"])
    if "implied_home" not in frame.columns:
        frame = add_implied_odds(frame)
    frame = annotate_matches(frame, early_limit=early_limit)

    setting_rows = _walk_seasons(
        frame,
        seasons=set(setting_t),
        kappas=candidates_t,
        promoted_factor=promoted_factor,
        min_train_rows=MIN_TRAIN_ROWS,
    )
    chosen = choose_kappa(setting_rows, candidates_t, setting_t)
    logger.info("Chosen strength %.4g from the setting seasons", chosen)
    later_rows = _walk_seasons(
        frame,
        seasons=set(later_t),
        kappas=(0.0, chosen) if chosen != 0.0 else (0.0,),
        promoted_factor=promoted_factor,
        min_train_rows=MIN_TRAIN_ROWS,
    )
    tagged = _tag_lines(pd.concat([setting_rows, later_rows], ignore_index=True), chosen)
    candidates_table = _candidate_table(tagged, setting_t, candidates_t)
    paired = _paired_table(tagged, setting_t, later_t, chosen)
    dropped, reason = _decision_from_paired(paired)
    elapsed = time.perf_counter() - started
    run_date = datetime.now(ZoneInfo("Europe/London")).strftime("%d %b %Y").lstrip("0")
    return ThinPriorResult(
        chosen_kappa=float(chosen),
        dropped=dropped,
        reason=reason,
        candidates=candidates_table,
        paired=paired,
        matches=tagged,
        elapsed_seconds=float(elapsed),
        run_date=run_date,
    )


def write_thin_prior_artefacts(
    result: ThinPriorResult,
    results_dir=None,
) -> dict[str, str]:
    """Write the paired table, the match rows, and the plain-language note."""
    root = RESULTS if results_dir is None else results_dir
    root.mkdir(parents=True, exist_ok=True)
    candidates_path = root / "thin_prior_candidates.csv"
    paired_path = root / "thin_prior_paired.csv"
    matches_path = root / "thin_prior_matches.csv"
    note_path = root / "THIN_DATA_PRIOR.md"
    result.candidates.to_csv(candidates_path, index=False)
    result.paired.to_csv(paired_path, index=False)
    result.matches.to_csv(matches_path, index=False)
    note_path.write_text(render_thin_prior_note(result), encoding="utf-8")
    return {
        "note": str(note_path),
        "candidates": str(candidates_path),
        "paired": str(paired_path),
        "matches": str(matches_path),
    }


def render_thin_prior_note(result: ThinPriorResult) -> str:
    """Plain-language result. Every gap has a range and a match count."""
    lines = [
        "# A starting strength for clubs with few matches",
        "",
        f"**Date:** {result.run_date} (London)",
        "**Question:** does one strength, pulling a short history toward a starting attack and defence, improve early matches without making the overall score worse?",
        "",
        "## What was fit",
        "",
        "The previous model is Dixon-Coles with the time-decay setting frozen at 0.0035 and the small existing penalty. The low-score correction was not retuned.",
        "",
        (
            f"The new line replaces that small penalty with one strength. "
            f"A club with no match in the previous season starts at "
            f"{THIN_PRIOR_PROMOTED_FACTOR:.2f} times an average club's attack, "
            "and its concession rate is higher by the same fraction. "
            "That fraction was fixed before the run. It was not searched. "
            f"The strengths tried were {', '.join(_fmt_kappa(k) for k in result.candidates['kappa'])}. "
            "Nothing else was tried after looking at the scores."
        ),
        "",
        "## How the strength was chosen",
        "",
        (
            "Each round is fit only on matches dated before that round. "
            f"The strength is the one with the lowest scoreline log loss on early matches "
            f"in seasons {', '.join(THIN_PRIOR_SETTING_SEASONS)}. "
            "Early means a club's first eight matches in the season. "
            "A tie keeps the smaller strength. "
            "Those three seasons were already used to choose the time-decay setting, "
            "so this table is where the strength was picked. It is not a clean test."
        ),
        "",
        _candidate_markdown(result.candidates),
        "",
        (
            f"The chosen strength is {_fmt_kappa(result.chosen_kappa)}. "
            "The gap below is on the early matches in the setting seasons. "
            "Those matches were used to pick the strength, so this gap is not a clean test."
        ),
        "",
        _gap_paragraph(
            result.paired,
            "setting_early_scoreline",
            "On the setting-season early matches, scoreline log loss",
        ),
        "",
        _grid_edge_paragraph(result.candidates),
        "",
        (
            "Seasons "
            f"{', '.join(THIN_PRIOR_LATER_SEASONS)} were not used to pick the strength."
        ),
        "",
        "## Later seasons",
        "",
        (
            "The model is refit every round on earlier matches, with the strength held fixed. "
            "A gap is the prior line minus the previous line on one shared set of matches. "
            f"The range is a 95% percentile interval from {BOOTSTRAP_DRAWS} resamples of whole rounds, "
            f"seed {BOOTSTRAP_SEED}. "
            "A negative difference means the prior line has the lower log loss. "
            "A range that contains zero has not separated the two lines on this sample."
        ),
        "",
        "### Early matches",
        "",
        _gap_paragraph(
            result.paired,
            "later_early_scoreline",
            "On early matches, scoreline log loss",
        ),
        "",
        _gap_paragraph(
            result.paired,
            "later_early_1x2",
            "On those same early matches, home/draw/away log loss",
        ),
        "",
        "### Every later-season match",
        "",
        _gap_paragraph(
            result.paired,
            "later_overall_scoreline",
            "On every later-season match, scoreline log loss",
        ),
        "",
        _gap_paragraph(
            result.paired,
            "later_overall_1x2",
            "On those same matches, home/draw/away log loss",
        ),
        "",
        "### Closing line",
        "",
        _book_paragraph(result.paired),
        "",
        "### Clubs that missed the previous season",
        "",
        (
            "This is not the stop rule. The stop rule uses every club's first eight matches, "
            "which is the definition written down before the run. "
            "The rows below are only the matches in the later seasons where a club "
            "with no match in the previous season is still inside those eight."
        ),
        "",
        _gap_paragraph(
            result.paired,
            "later_new_early_scoreline",
            "On that smaller slice, scoreline log loss",
        ),
        "",
        "## Decision",
        "",
        _decision_paragraph(result),
        "",
        "## Regenerate",
        "",
        "```bash",
        "python -m src.evaluate --mode thin_prior",
        "```",
        "",
        (
            f"The run took {result.elapsed_seconds / 60.0:.1f} minutes. "
            "The per-match rows are `results/thin_prior_matches.csv`. "
            "The ranges are `results/thin_prior_paired.csv`."
        ),
        "",
    ]
    text = "\n".join(lines)
    _reject_bad_language(text)
    return text


def _check_windows(candidates, setting, later) -> None:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if any((not math.isfinite(kappa)) or kappa < 0.0 for kappa in candidates):
        raise ValueError("strengths must be finite and >= 0")
    overlap = set(setting) & set(later)
    if overlap:
        raise ValueError(f"setting and later seasons overlap: {sorted(overlap)}")
    if not setting or not later:
        raise ValueError("setting and later seasons must both be non-empty")


def _walk_seasons(
    frame: pd.DataFrame,
    *,
    seasons: set[str],
    kappas: tuple[float, ...],
    promoted_factor: float,
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
        fitted = {
            float(kappa): fit_dixon_coles(
                train,
                prior_kappa=float(kappa),
                promoted_factor=promoted_factor,
            )
            for kappa in kappas
        }
        for row in test.itertuples(index=False):
            identity = _identity(row, season, gw)
            book_loss = _book_result_loss(row)
            for kappa, model in fitted.items():
                scored = _score_match(model, row)
                rows.append(
                    {
                        **identity,
                        "model": kappa_name(kappa),
                        "kappa": float(kappa),
                        "scoreline_loss": scored["scoreline_loss"],
                        "result_loss": scored["result_loss"],
                        "book_result_loss": book_loss,
                    }
                )
    if not rows:
        raise ValueError(f"no scored rounds in seasons {sorted(seasons)}")
    return pd.DataFrame(rows)


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
        "early": bool(row.early),
        "new_club_early": bool(row.new_club_early),
    }


def _score_match(model, row) -> dict[str, float]:
    grid = scoreline_grid(model, row.home_team, row.away_team)
    lam, mu = expected_goals(model, row.home_team, row.away_team)
    scored = score_scoreline(
        grid,
        int(row.home_goals),
        int(row.away_goals),
        lam=lam,
        mu=mu,
        rho=float(model.rho),
    )
    markets = markets_from_grid(grid)
    return {
        "scoreline_loss": float(scored["cell_log_loss"]),
        "result_loss": _result_loss(
            str(row.result), markets["home"], markets["draw"], markets["away"]
        ),
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


def _tag_lines(scores: pd.DataFrame, chosen: float) -> pd.DataFrame:
    """Copy the previous fit and the chosen strength onto stable line names.

    Later-season rows for strengths that were not chosen are not in ``scores``.
    The setting-era rows keep every strength, under ``prior_<k>``.
    """
    pieces = [scores]
    baseline = scores.loc[scores["model"] == kappa_name(0.0)].copy()
    baseline["model"] = _BASELINE
    pieces.append(baseline)
    chosen_rows = scores.loc[scores["model"] == kappa_name(chosen)].copy()
    chosen_rows["model"] = _CHOSEN
    pieces.append(chosen_rows)
    return pd.concat(pieces, ignore_index=True)


def _candidate_table(
    scores: pd.DataFrame,
    setting_seasons: tuple[str, ...],
    candidates: tuple[float, ...],
) -> pd.DataFrame:
    seasons = {str(season) for season in setting_seasons}
    setting = scores.loc[scores["season_code"].astype(str).isin(seasons)]
    rows: list[dict] = []
    early_n: set[int] = set()
    overall_n: set[int] = set()
    for kappa in candidates:
        part = setting.loc[setting["model"] == kappa_name(kappa)]
        early = part.loc[part["early"].astype(bool)]
        early_n.add(len(early))
        overall_n.add(len(part))
        rows.append(
            {
                "kappa": float(kappa),
                "early_scoreline_log_loss": round(float(early["scoreline_loss"].mean()), 4),
                "early_matches": int(len(early)),
                "overall_scoreline_log_loss": round(float(part["scoreline_loss"].mean()), 4),
                "overall_matches": int(len(part)),
                "early_result_log_loss": round(float(early["result_loss"].mean()), 4),
                "overall_result_log_loss": round(float(part["result_loss"].mean()), 4),
                "chosen": bool(float(kappa) == float(scores.loc[scores["model"] == _CHOSEN, "kappa"].iloc[0])),
            }
        )
    if len(early_n) != 1 or len(overall_n) != 1:
        raise ValueError(
            "candidate rows were not scored on the same matches: "
            f"early {sorted(early_n)}, overall {sorted(overall_n)}"
        )
    return pd.DataFrame(rows)


def _paired_table(
    scores: pd.DataFrame,
    setting_seasons: tuple[str, ...],
    later_seasons: tuple[str, ...],
    chosen: float,
) -> pd.DataFrame:
    later = scores.loc[scores["season_code"].astype(str).isin(set(later_seasons))]
    setting = scores.loc[scores["season_code"].astype(str).isin(set(setting_seasons))]
    specs = [
        ("later_early_scoreline", "scoreline_loss", later, "early", _CHOSEN, _BASELINE),
        ("later_early_1x2", "result_loss", later, "early", _CHOSEN, _BASELINE),
        ("later_overall_scoreline", "scoreline_loss", later, None, _CHOSEN, _BASELINE),
        ("later_overall_1x2", "result_loss", later, None, _CHOSEN, _BASELINE),
        (
            "later_new_early_scoreline",
            "scoreline_loss",
            later,
            "new_club_early",
            _CHOSEN,
            _BASELINE,
        ),
        (
            "setting_early_scoreline",
            "scoreline_loss",
            setting,
            "early",
            kappa_name(chosen),
            _BASELINE,
        ),
    ]
    rows = [_one_gap(name, column, frame, mask, left, right) for name, column, frame, mask, left, right in specs]
    rows.extend(_book_gaps(later))
    return pd.DataFrame(rows)


def _one_gap(name, column, frame, mask, left_name, right_name) -> dict:
    part = frame if mask is None else frame.loc[frame[mask].astype(bool)]
    left = part.loc[part["model"] == left_name]
    right = part.loc[part["model"] == right_name]
    gap = _align_gap(left, right, column)
    return _gap_row(
        name,
        left_name,
        right_name,
        gap,
        "A negative difference means the first line has the lower log loss.",
    )


def _book_gaps(later: pd.DataFrame) -> list[dict]:
    book_ids = set(
        later.loc[np.isfinite(later["book_result_loss"].to_numpy(dtype=float)), "match_id"]
    )
    # The book loss is stored on every model row. Build one book row per match.
    base = later.loc[
        (later["model"] == _BASELINE) & later["match_id"].isin(book_ids)
    ].copy()
    if base.empty:
        return []
    book = base.copy()
    book["model"] = _BOOK
    book["result_loss"] = book["book_result_loss"]
    prior = later.loc[(later["model"] == _CHOSEN) & later["match_id"].isin(book_ids)]
    previous = base
    note = (
        "A negative difference means the first line has the lower log loss. "
        "The closing line is the competitor. This is not a claim about the bookmaker."
    )
    return [
        _gap_row(
            "later_book_prior",
            _CHOSEN,
            _BOOK,
            _align_gap(prior, book, "result_loss"),
            note,
        ),
        _gap_row(
            "later_book_previous",
            _BASELINE,
            _BOOK,
            _align_gap(previous, book, "result_loss"),
            note,
        ),
    ]


def _align_gap(left: pd.DataFrame, right: pd.DataFrame, column: str) -> dict:
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
        # Unrounded mean, so a decision test can see the figure before display rounding.
        "mean_diff_exact": float(gap["mean_diff"]),
    }


def _decision_from_paired(paired: pd.DataFrame) -> tuple[bool, str]:
    return stop_decision(
        _lookup(paired, "later_early_scoreline")["mean_diff"],
        _lookup(paired, "later_overall_scoreline")["mean_diff"],
        _lookup(paired, "later_overall_1x2")["mean_diff"],
    )


def _lookup(paired: pd.DataFrame, name: str) -> pd.Series:
    rows = paired.loc[paired["comparison"] == name]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {name}")
    return rows.iloc[0]


def _fmt_kappa(kappa: float) -> str:
    value = float(kappa)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _candidate_markdown(candidates: pd.DataFrame) -> str:
    header = (
        "| Strength | Early scoreline log loss | Early matches | "
        "Overall scoreline log loss | Overall matches | "
        "Early home/draw/away log loss | Overall home/draw/away log loss |"
    )
    rule = "|---|---|---|---|---|---|---|"
    body = []
    for row in candidates.itertuples(index=False):
        mark = " (chosen)" if bool(row.chosen) else ""
        body.append(
            f"| {_fmt_kappa(row.kappa)}{mark} | {row.early_scoreline_log_loss:.4f} | "
            f"{int(row.early_matches)} | {row.overall_scoreline_log_loss:.4f} | "
            f"{int(row.overall_matches)} | {row.early_result_log_loss:.4f} | "
            f"{row.overall_result_log_loss:.4f} |"
        )
    return "\n".join([header, rule, *body])


def _gap_paragraph(paired: pd.DataFrame, name: str, label: str) -> str:
    row = _lookup(paired, name)
    sentence = (
        f"{label}: the prior line minus the previous line is {row['mean_diff']:.4f} "
        f"(95% range {row['ci_low']:.4f} to {row['ci_high']:.4f}, "
        f"{int(row['n_matches'])} matches, {int(row['n_rounds'])} rounds)."
    )
    if float(row["ci_low"]) <= 0.0 <= float(row["ci_high"]):
        sentence += " The range includes zero."
    return sentence


def _book_paragraph(paired: pd.DataFrame) -> str:
    if "later_book_prior" not in set(paired["comparison"]):
        return "No closing home/draw/away line was available on the later-season matches."
    prior = _lookup(paired, "later_book_prior")
    previous = _lookup(paired, "later_book_previous")
    if int(prior["n_matches"]) != int(previous["n_matches"]):
        raise ValueError("book gaps are not on the same matches")
    return (
        "On the later-season matches that have a closing home/draw/away line, "
        f"the prior line minus the closing line is {prior['mean_diff']:.4f} "
        f"(95% range {prior['ci_low']:.4f} to {prior['ci_high']:.4f}, "
        f"{int(prior['n_matches'])} matches, {int(prior['n_rounds'])} rounds). "
        f"The previous line minus the closing line on those same matches is {previous['mean_diff']:.4f} "
        f"(95% range {previous['ci_low']:.4f} to {previous['ci_high']:.4f}, "
        f"{int(previous['n_matches'])} matches, {int(previous['n_rounds'])} rounds). "
        "The closing line is the competitor. These gaps are not a claim about the bookmaker. "
        "A positive difference means that line has the higher log loss."
    )


def _decision_paragraph(result: ThinPriorResult) -> str:
    overall = _lookup(result.paired, "later_overall_scoreline")
    if result.dropped:
        return (
            f"The stop rule drops the change: {result.reason}. "
            "The published fit stays the previous model. "
            "The paired gaps above are the result, including the early-match slice."
        )
    early = _lookup(result.paired, "later_early_scoreline")
    new_clubs = _lookup(result.paired, "later_new_early_scoreline")
    text = (
        f"The stop rule keeps the change: {result.reason}. "
        f"The published fit uses strength {_fmt_kappa(result.chosen_kappa)}."
    )
    if float(early["ci_low"]) <= 0.0 <= float(early["ci_high"]):
        text += (
            " The early-match range includes zero, so that slice has not separated the two lines. "
            "The rule uses the point estimates."
        )
    text += (
        " The overall movement is concentrated on clubs that missed the previous season: "
        f"that smaller slice is {float(new_clubs['mean_diff']):.4f} "
        f"(95% range {float(new_clubs['ci_low']):.4f} to {float(new_clubs['ci_high']):.4f}, "
        f"{int(new_clubs['n_matches'])} matches, {int(new_clubs['n_rounds'])} rounds)."
    )
    if abs(float(overall["mean_diff"])) < 0.01:
        text += (
            " The overall scoreline change is under 0.01. "
            "That is a point estimate inside the range above. "
            "It is not a success, and it is not a claim about the bookmaker."
        )
    else:
        text += " It is not a claim about the bookmaker."
    return text


def _grid_edge_paragraph(candidates: pd.DataFrame) -> str:
    """Say when the chosen strength is the largest one that was tried."""
    ordered = candidates.sort_values("kappa")
    if len(ordered) < 2 or not bool(ordered.iloc[-1]["chosen"]):
        return ""
    top = ordered.iloc[-1]
    return (
        f"Strength {_fmt_kappa(top['kappa'])} is the largest one that was tried. "
        "No larger strength was added after looking. The table above is the whole grid."
    )


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


def _new_club_early(
    frame: pd.DataFrame,
    home_prior: np.ndarray,
    away_prior: np.ndarray,
    limit: int,
) -> np.ndarray:
    seasons = sorted({str(code) for code in frame["season_code"].dropna().unique()})
    previous = {seasons[i]: seasons[i - 1] for i in range(1, len(seasons))}
    teams_in: dict[str, set[str]] = {}
    for season, part in frame.groupby(frame["season_code"].astype(str), sort=False):
        teams_in[str(season)] = set(part["home_team"].astype(str)) | set(
            part["away_team"].astype(str)
        )
    flags = np.zeros(len(frame), dtype=bool)
    season_arr = frame["season_code"].astype(str).to_numpy()
    home = frame["home_team"].astype(str).to_numpy()
    away = frame["away_team"].astype(str).to_numpy()
    for i in range(len(frame)):
        prev = previous.get(season_arr[i])
        if prev is None:
            continue
        known = teams_in[prev]
        home_new = home[i] not in known and int(home_prior[i]) < limit
        away_new = away[i] not in known and int(away_prior[i]) < limit
        flags[i] = bool(home_new or away_new)
    return flags
