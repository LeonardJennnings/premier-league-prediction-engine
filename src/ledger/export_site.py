"""Site files built from the ledger. Grouped by matchday, not by calendar week.

A Monday match stays with the rest of its round. Postponed and cancelled
matches stay in the list and are left out of every average. Nothing here is
written back onto a prediction line.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from src.eval.book_grid import BOOK_SCORE_TABLE_LABEL

UPCOMING_NAME = "ledger-upcoming.json"
SETTLED_NAME = "ledger-settled.json"
SCORECARD_NAME = "ledger-scorecard.json"

SMALL_SAMPLE_NOTE = "This is a small number of matches, and a week cannot show skill."


def resolve_match(prediction: dict, settlements: list[dict]) -> dict:
    """Latest public state of one prediction. The prediction line is not edited."""
    rows = [
        row
        for row in settlements
        if int(row["match_id"]) == int(prediction["match_id"])
        and str(row.get("forecast_line")) == str(prediction.get("forecast_line"))
    ]
    result = None
    corrections: list[dict] = []
    statuses: list[dict] = []
    for row in rows:
        kind = row.get("kind")
        if kind == "result":
            result = row
        elif kind == "correction":
            corrections.append(row)
        elif kind == "status":
            statuses.append(row)
    score = corrections[-1] if corrections else result
    kickoff = str(prediction["kickoff_utc"])
    if score is not None and score.get("home_goals") is not None and score.get("away_goals") is not None:
        home_goals = int(score["home_goals"])
        away_goals = int(score["away_goals"])
        return {
            "status": "settled",
            "kickoff_utc": kickoff,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": score.get("result") or _result_code(home_goals, away_goals),
            "provider_updated_at": score.get("provider_updated_at"),
        }
    if statuses:
        latest = statuses[-1]
        return {
            "status": str(latest.get("status") or "scheduled"),
            "kickoff_utc": str(latest.get("kickoff_utc") or kickoff),
            "home_goals": None,
            "away_goals": None,
            "result": None,
            "provider_updated_at": latest.get("provider_updated_at"),
        }
    return {
        "status": "scheduled",
        "kickoff_utc": kickoff,
        "home_goals": None,
        "away_goals": None,
        "result": None,
        "provider_updated_at": None,
    }


def build_site_files(
    predictions: list[dict],
    settlements: list[dict],
    book_lines: list[dict],
    *,
    now_utc: datetime,
    comparisons: list[dict] | None = None,
) -> dict[str, dict]:
    """Return the three site documents. Bytes are a pure function of these inputs."""
    now = _as_utc(now_utc)
    by_match = _publish_comparisons(comparisons)
    blends = {
        int(row["match_id"]): row
        for row in predictions
        if str(row.get("forecast_line") or "") == "model_plus_market"
    }
    model_rows = [
        row
        for row in predictions
        if str(row.get("forecast_line") or "model") == "model"
    ]
    upcoming_rows: list[dict] = []
    listed_rows: list[dict] = []
    for prediction in model_rows:
        state = resolve_match(prediction, settlements)
        kickoff = _parse_utc(state["kickoff_utc"])
        card = _card(prediction, state, book_lines, now=now)
        blend = blends.get(int(prediction["match_id"]))
        if blend is not None:
            card["model_plus_market"] = _blend_view(blend)
        status = state["status"]
        if status in {"postponed", "cancelled"} or status == "settled" or kickoff <= now:
            listed_rows.append(card)
        else:
            score = _reconstructed_score(by_match.get(int(prediction["match_id"])))
            if score is not None:
                card["book_reconstructed_score"] = score
            upcoming_rows.append(card)
    return {
        UPCOMING_NAME: {
            "note": "Matches still to kick off. Grouped by matchday.",
            "matchdays": _by_matchday(upcoming_rows),
        },
        SETTLED_NAME: {
            "note": (
                "Finished, postponed, cancelled, and matches waiting for a score. "
                "Grouped by matchday."
            ),
            "matchdays": _by_matchday(listed_rows),
        },
        SCORECARD_NAME: _scorecard(listed_rows, comparisons),
    }


def write_site_files(documents: dict[str, dict], export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in documents.items():
        target = export_dir / name
        target.write_bytes(_dumps(payload))


def _card(prediction: dict, state: dict, book_lines: list[dict], *, now: datetime) -> dict:
    kind = "closing" if state["status"] == "settled" else "opening"
    book = _book(book_lines, int(prediction["match_id"]), kind)
    if book is None and kind == "closing":
        book = _book(book_lines, int(prediction["match_id"]), "opening")
    status = state["status"]
    if status == "scheduled" and _parse_utc(state["kickoff_utc"]) <= now:
        status = "pending"
    return {
        "match_id": int(prediction["match_id"]),
        "matchday": prediction.get("matchday"),
        "kickoff_utc": state["kickoff_utc"],
        "home_team": prediction["home_team"],
        "away_team": prediction["away_team"],
        "model_version": prediction["model_version"],
        "status": status,
        "home_goals": state["home_goals"],
        "away_goals": state["away_goals"],
        "result": state["result"],
        "provider_updated_at": state["provider_updated_at"],
        "most_likely_score": prediction.get("most_likely_score"),
        "next_scores": prediction.get("next_scores"),
        "p_home": prediction.get("p_home"),
        "p_draw": prediction.get("p_draw"),
        "p_away": prediction.get("p_away"),
        "p_btts_yes": prediction.get("p_btts_yes"),
        "p_btts_no": prediction.get("p_btts_no"),
        "p_over_25": prediction.get("p_over_25"),
        "p_under_25": prediction.get("p_under_25"),
        "totals": prediction.get("totals"),
        "lambda_home": prediction.get("lambda_home"),
        "lambda_away": prediction.get("lambda_away"),
        "uses_league_average": prediction.get("uses_league_average"),
        "league_average_teams": prediction.get("league_average_teams"),
        "score_omitted_reason": prediction.get("score_omitted_reason"),
        "book": book,
    }


def _blend_view(prediction: dict) -> dict:
    """The second line, labelled, sitting on the pure model's card."""
    return {
        "label": "model plus market",
        "blend_weight": prediction.get("blend_weight"),
        "p_home": prediction.get("p_home"),
        "p_draw": prediction.get("p_draw"),
        "p_away": prediction.get("p_away"),
        "p_over_25": prediction.get("p_over_25"),
        "p_under_25": prediction.get("p_under_25"),
        "most_likely_score": prediction.get("most_likely_score"),
        "lambda_home": prediction.get("lambda_home"),
        "lambda_away": prediction.get("lambda_away"),
    }


def _scorecard(listed: list[dict], comparisons: list[dict] | None) -> dict:
    counts = {"settled": 0, "pending": 0, "postponed": 0, "void": 0}
    by_matchday: dict = {}
    for row in listed:
        label = _count_label(row["status"])
        counts[label] = counts.get(label, 0) + 1
        bucket = by_matchday.setdefault(
            row["matchday"],
            {"matchday": row["matchday"], "settled": 0, "pending": 0, "postponed": 0, "void": 0},
        )
        bucket[label] = bucket.get(label, 0) + 1
    settled = [row for row in listed if row["status"] == "settled" and row["home_goals"] is not None]
    return {
        "note": SMALL_SAMPLE_NOTE,
        "counts": counts,
        "matchdays": [by_matchday[key] for key in sorted(by_matchday, key=_matchday_key)],
        "home_draw_away": _result_average(settled),
        "book_comparison": _book_comparison(settled),
        "book_reconstruction": _reconstruction_summary(settled, comparisons),
        "exact_score": _exact_score(settled),
        "goal_error": _goal_error(settled),
    }


def _publish_comparisons(comparisons: list[dict] | None) -> dict[int, dict]:
    found: dict[int, dict] = {}
    for row in comparisons or []:
        if row.get("kind") != "publish" or str(row.get("forecast_line") or "model") != "model":
            continue
        found[int(row["match_id"])] = row
    return found


def _reconstructed_score(row: dict | None) -> dict | None:
    """The book's most likely score, only when a price was reconstructed."""
    if not row or row.get("missing_reason"):
        return None
    score = row.get("book_most_likely_score")
    if not isinstance(score, dict):
        return None
    if score.get("home") is None or score.get("away") is None or score.get("probability") is None:
        return None
    return {
        "home": int(score["home"]),
        "away": int(score["away"]),
        "probability": float(score["probability"]),
        "label": str(row.get("label") or BOOK_SCORE_TABLE_LABEL),
    }


def _reconstruction_summary(settled: list[dict], comparisons: list[dict] | None) -> dict:
    """Mean scoreline gap on settled matches that have both log losses.

    A missing price stays out. It is not entered as a gap of zero.
    """
    scored = {}
    for row in comparisons or []:
        if row.get("kind") != "scoreline" or str(row.get("forecast_line") or "model") != "model":
            continue
        if row.get("home_goals") is None or row.get("away_goals") is None:
            continue
        scored[(int(row["match_id"]), int(row["home_goals"]), int(row["away_goals"]))] = row
    gaps = []
    model_losses = []
    book_losses = []
    model_hits = 0
    book_hits = 0
    for match in settled:
        row = scored.get((int(match["match_id"]), int(match["home_goals"]), int(match["away_goals"])))
        if row is None or row.get("scoreline_log_loss_gap") is None:
            continue
        gaps.append(float(row["scoreline_log_loss_gap"]))
        model_losses.append(float(row["model_scoreline_log_loss"]))
        book_losses.append(float(row["book_scoreline_log_loss"]))
        if row.get("model_most_likely_hit") is True:
            model_hits += 1
        if row.get("book_most_likely_hit") is True:
            book_hits += 1
    return {
        "label": BOOK_SCORE_TABLE_LABEL,
        "match_count": len(gaps),
        "mean_gap": _mean(gaps),
        "model_log_loss": _mean(model_losses),
        "reconstruction_log_loss": _mean(book_losses),
        "model_most_likely_hits": model_hits,
        "reconstruction_most_likely_hits": book_hits,
    }


def _result_average(settled: list[dict]) -> dict:
    losses = []
    for row in settled:
        probability = _result_probability(row)
        if probability is None:
            continue
        losses.append(_log_loss(probability))
    return {
        "match_count": len(losses),
        "model_log_loss": _mean(losses),
    }


def _book_comparison(settled: list[dict]) -> dict:
    gaps = []
    model_losses = []
    book_losses = []
    for row in settled:
        book = row.get("book") or {}
        if book.get("kind") != "closing":
            continue
        model_p = _result_probability(row)
        book_p = _book_result_probability(book, row.get("result"))
        if model_p is None or book_p is None:
            continue
        model_loss = _log_loss(model_p)
        book_loss = _log_loss(book_p)
        model_losses.append(model_loss)
        book_losses.append(book_loss)
        gaps.append(model_loss - book_loss)
    gap = _mean(gaps)
    return {
        "match_count": len(gaps),
        "model_log_loss": _mean(model_losses),
        "closing_book_log_loss": _mean(book_losses),
        "gap_model_minus_closing": gap,
        "range_95": _gap_range(gaps),
    }


def _exact_score(settled: list[dict]) -> dict:
    picked = [row for row in settled if isinstance(row.get("most_likely_score"), dict)]
    hits = 0
    always = 0
    printed = []
    for row in picked:
        score = row["most_likely_score"]
        if int(score["home"]) == int(row["home_goals"]) and int(score["away"]) == int(row["away_goals"]):
            hits += 1
        if int(row["home_goals"]) == 1 and int(row["away_goals"]) == 1:
            always += 1
        if score.get("probability") is not None:
            printed.append(float(score["probability"]))
    return {
        "match_count": len(picked),
        "hits": hits,
        "always_1_1_hits": always,
        "mean_printed_probability": _mean(printed),
    }


def _goal_error(settled: list[dict]) -> dict:
    errors = []
    for row in settled:
        if row.get("lambda_home") is None or row.get("lambda_away") is None:
            continue
        errors.append(
            abs(int(row["home_goals"]) - float(row["lambda_home"]))
            + abs(int(row["away_goals"]) - float(row["lambda_away"]))
        )
    return {"match_count": len(errors), "mean_absolute_error": _mean(errors)}


def _gap_range(gaps: list[float]) -> list[float] | None:
    """A 95 percent range for the mean gap. Null until there are two matches."""
    if len(gaps) < 2:
        return None
    rng = random.Random(0)
    means = []
    size = len(gaps)
    for _ in range(1000):
        sample = [gaps[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()
    return [means[24], means[974]]


def _book(book_lines: list[dict], match_id: int, kind: str) -> dict | None:
    rows = [
        row
        for row in book_lines
        if int(row["match_id"]) == match_id
        and str(row.get("forecast_line")) == "model"
        and str(row.get("kind")) == kind
    ]
    if not rows:
        return None
    row = rows[-1]
    return {
        "kind": kind,
        "captured_at": row.get("captured_at"),
        "source": row.get("source"),
        "p_home": row.get("p_home"),
        "p_draw": row.get("p_draw"),
        "p_away": row.get("p_away"),
        "p_over_25": row.get("p_over_25"),
        "p_under_25": row.get("p_under_25"),
    }


def _by_matchday(rows: list[dict]) -> list[dict]:
    groups: dict = {}
    for row in rows:
        groups.setdefault(row["matchday"], []).append(row)
    out = []
    for matchday in sorted(groups, key=_matchday_key):
        matches = sorted(groups[matchday], key=lambda row: (row["kickoff_utc"], row["match_id"]))
        out.append({"matchday": matchday, "matches": matches})
    return out


def _count_label(status: str) -> str:
    if status == "cancelled":
        return "void"
    if status in {"settled", "pending", "postponed"}:
        return status
    return "pending"


def _result_probability(row: dict) -> float | None:
    result = row.get("result")
    column = {"H": "p_home", "D": "p_draw", "A": "p_away"}.get(result)
    if column is None or row.get(column) is None:
        return None
    return float(row[column])


def _book_result_probability(book: dict, result: str | None) -> float | None:
    column = {"H": "p_home", "D": "p_draw", "A": "p_away"}.get(result or "")
    if column is None or book.get(column) is None:
        return None
    return float(book[column])


def _result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _log_loss(probability: float) -> float:
    return -math.log(min(max(probability, 1e-15), 1.0))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _matchday_key(matchday) -> tuple:
    if matchday is None:
        return (True, 0)
    return (False, int(matchday))


def _dumps(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_utc(now_utc: datetime) -> datetime:
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=timezone.utc)
    return now_utc.astimezone(timezone.utc)
