"""One match: the pure model beside the book's reconstructed score table.

Written once at publish. The score that happened is not known yet, so both
scoreline log losses stay null. Settlement appends a later line. The
prediction line is never edited.

The table is our reconstruction. It is not the bookmaker's correct-score odds.
"""

from __future__ import annotations

import math

from src.eval.book_grid import BOOK_SCORE_TABLE_LABEL, fit_book_grid
from src.eval.scoreline import exact_cell_probability
from src.ledger.lines import FORECAST_LINE_MODEL
from src.models.dixon_coles import scoreline_grid_from_rates
from src.models.markets import scoreline_log_loss

_PRICE_FIELDS = (
    ("home", "book_p_home", "price_home"),
    ("draw", "book_p_draw", "price_draw"),
    ("away", "book_p_away", "price_away"),
    ("over/under 2.5", "book_p_over_25", "price_over_25"),
)


def publish_comparison(line: dict, quote: dict | None, *, recorded_at: str) -> dict:
    """The row stored when a model line is first published.

    A missing price stays null, with a reason. No odds are filled in.
    """
    prices, reason = _prices(quote)
    row = {
        "match_id": int(line["match_id"]),
        "forecast_line": FORECAST_LINE_MODEL,
        "kind": "publish",
        "label": BOOK_SCORE_TABLE_LABEL,
        "recorded_at": recorded_at,
        "missing_reason": reason,
        "price_home": prices["price_home"],
        "price_draw": prices["price_draw"],
        "price_away": prices["price_away"],
        "price_over_25": prices["price_over_25"],
        "price_under_25": _optional_under(quote),
        "book_lambda": None,
        "book_mu": None,
        "book_most_likely_score": None,
        "book_p_home": None,
        "book_p_draw": None,
        "book_p_away": None,
        "book_p_over_25": None,
        "book_p_under_25": None,
        "model_lambda_home": _finite(line.get("lambda_home")),
        "model_lambda_away": _finite(line.get("lambda_away")),
        "model_most_likely_score": line.get("most_likely_score"),
        "model_p_home": _finite(line.get("p_home")),
        "model_p_draw": _finite(line.get("p_draw")),
        "model_p_away": _finite(line.get("p_away")),
        "model_p_over_25": _finite(line.get("p_over_25")),
        "model_p_under_25": _finite(line.get("p_under_25")),
        "gap_home": None,
        "gap_draw": None,
        "gap_away": None,
        "gap_over_25": None,
        "gap_under_25": None,
        "model_scoreline_log_loss": None,
        "book_scoreline_log_loss": None,
        "scoreline_log_loss_gap": None,
    }
    if reason is not None:
        return row
    try:
        fitted = fit_book_grid(
            prices["price_home"],
            prices["price_draw"],
            prices["price_away"],
            prices["price_over_25"],
        )
    except (ValueError, RuntimeError):
        row["missing_reason"] = "reconstruction did not fit"
        return row
    score = _most_likely(fitted.grid)
    row["book_lambda"] = float(fitted.lam)
    row["book_mu"] = float(fitted.mu)
    row["book_most_likely_score"] = score
    row["book_p_home"] = float(fitted.home)
    row["book_p_draw"] = float(fitted.draw)
    row["book_p_away"] = float(fitted.away)
    row["book_p_over_25"] = float(fitted.over_25)
    row["book_p_under_25"] = float(fitted.under_25)
    row["gap_home"] = _gap(row["model_p_home"], row["book_p_home"])
    row["gap_draw"] = _gap(row["model_p_draw"], row["book_p_draw"])
    row["gap_away"] = _gap(row["model_p_away"], row["book_p_away"])
    row["gap_over_25"] = _gap(row["model_p_over_25"], row["book_p_over_25"])
    row["gap_under_25"] = _gap(row["model_p_under_25"], row["book_p_under_25"])
    return row


def scoreline_record(publish_row: dict, prediction: dict, home_goals: int, away_goals: int) -> dict:
    """Append-only score of the final score. Does not change the publish row."""
    home_goals = int(home_goals)
    away_goals = int(away_goals)
    model_loss, model_reason = _model_loss(prediction, home_goals, away_goals)
    book_loss, book_reason = _book_loss(publish_row, home_goals, away_goals)
    gap = None
    if model_loss is not None and book_loss is not None:
        gap = model_loss - book_loss
    reasons = [reason for reason in (book_reason, model_reason) if reason]
    return {
        "match_id": int(publish_row["match_id"]),
        "forecast_line": FORECAST_LINE_MODEL,
        "kind": "scoreline",
        "label": BOOK_SCORE_TABLE_LABEL,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "model_scoreline_log_loss": model_loss,
        "book_scoreline_log_loss": book_loss,
        "scoreline_log_loss_gap": gap,
        "model_most_likely_hit": _hit(publish_row.get("model_most_likely_score"), home_goals, away_goals),
        "book_most_likely_hit": _hit(publish_row.get("book_most_likely_score"), home_goals, away_goals),
        "missing_reason": "; ".join(reasons) if reasons else None,
    }


def scoreline_lines_for_settled(
    predictions: list[dict],
    settlements: list[dict],
    comparisons: list[dict],
) -> list[dict]:
    """One scoreline line per settled score that does not already have one.

    A match with no publish row is left alone. The book's table is not
    rebuilt after the fact.
    """
    from src.ledger.export_site import resolve_match

    published = {
        (int(row["match_id"]), str(row.get("forecast_line") or FORECAST_LINE_MODEL)): row
        for row in comparisons
        if row.get("kind") == "publish"
    }
    have = {
        (
            int(row["match_id"]),
            str(row.get("forecast_line") or FORECAST_LINE_MODEL),
            int(row["home_goals"]),
            int(row["away_goals"]),
        )
        for row in comparisons
        if row.get("kind") == "scoreline"
    }
    fresh: list[dict] = []
    for prediction in predictions:
        if str(prediction.get("forecast_line") or FORECAST_LINE_MODEL) != FORECAST_LINE_MODEL:
            continue
        state = resolve_match(prediction, settlements)
        if state["status"] != "settled" or state["home_goals"] is None or state["away_goals"] is None:
            continue
        key = (int(prediction["match_id"]), FORECAST_LINE_MODEL)
        publish_row = published.get(key)
        if publish_row is None:
            continue
        score_key = (key[0], key[1], int(state["home_goals"]), int(state["away_goals"]))
        if score_key in have:
            continue
        fresh.append(scoreline_record(publish_row, prediction, state["home_goals"], state["away_goals"]))
        have.add(score_key)
    return fresh


def _prices(quote: dict | None) -> tuple[dict[str, float | None], str | None]:
    if not quote:
        return {stored: None for _label, _key, stored in _PRICE_FIELDS}, "no pre-kickoff price"
    stored: dict[str, float | None] = {}
    missing: list[str] = []
    unusable: list[str] = []
    for label, key, name in _PRICE_FIELDS:
        status, number = _read_price(quote, key)
        stored[name] = number
        if status == "missing":
            missing.append(label)
        elif status == "unusable":
            unusable.append(label)
    if missing:
        listed = ", ".join(missing)
        return stored, f"missing pre-kickoff price: {listed}"
    if unusable:
        listed = ", ".join(unusable)
        return stored, f"pre-kickoff price is not usable: {listed}"
    return stored, None


def _optional_under(quote: dict | None) -> float | None:
    if not quote:
        return None
    status, number = _read_price(quote, "book_p_under_25")
    if status != "ok":
        return None
    return number


def _read_price(quote: dict, key: str) -> tuple[str, float | None]:
    if key not in quote or quote.get(key) is None:
        return "missing", None
    try:
        number = float(quote.get(key))
    except (TypeError, ValueError):
        return "unusable", None
    if not math.isfinite(number) or number <= 0.0 or number >= 1.0:
        return "unusable", None
    return "ok", number


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _gap(model: float | None, book: float | None) -> float | None:
    if model is None or book is None:
        return None
    return float(model) - float(book)


def _most_likely(grid) -> dict:
    best_key = None
    best = None
    for home in range(grid.shape[0]):
        for away in range(grid.shape[1]):
            probability = float(grid[home, away])
            key = (-probability, home, away)
            if best_key is None or key < best_key:
                best_key = key
                best = {"home": home, "away": away, "probability": probability}
    if best is None:
        raise ValueError("empty score grid")
    return best


def _hit(score, home_goals: int, away_goals: int) -> bool | None:
    if not isinstance(score, dict) or score.get("home") is None or score.get("away") is None:
        return None
    return int(score["home"]) == int(home_goals) and int(score["away"]) == int(away_goals)


def _model_loss(prediction: dict, home_goals: int, away_goals: int) -> tuple[float | None, str | None]:
    table = prediction.get("score_table")
    if not table:
        return None, "model score table was not stored"
    loss = scoreline_log_loss(table, home_goals, away_goals)
    if not math.isfinite(loss):
        return None, "model score probability is zero"
    return float(loss), None


def _book_loss(publish_row: dict, home_goals: int, away_goals: int) -> tuple[float | None, str | None]:
    if publish_row.get("missing_reason"):
        return None, str(publish_row["missing_reason"])
    lam = _finite(publish_row.get("book_lambda"))
    mu = _finite(publish_row.get("book_mu"))
    if lam is None or mu is None:
        return None, "reconstructed book rates were not stored"
    grid = scoreline_grid_from_rates(lam, mu, 0.0)
    max_goals = grid.shape[0] - 1
    if 0 <= home_goals <= max_goals and 0 <= away_goals <= max_goals:
        probability = float(grid[home_goals, away_goals])
    else:
        probability = exact_cell_probability(lam, mu, 0.0, home_goals, away_goals)
    if not math.isfinite(probability) or probability <= 0.0:
        return None, "reconstructed score probability is zero"
    return float(-math.log(probability)), None
