"""Copy the 17 Nov 2026 publish into the ledger and settle those ten matches.

The probabilities are the committed snapshot. The score table is left empty
on purpose: that publish did not store one, and it is not invented later.
"""

from __future__ import annotations

from src.ledger.log import (
    append_historical_predictions,
    append_settlements,
    ensure_ledger,
)

# Full-time scores read from football-data.org on 25 Nov 2026.
# provider_updated_at is that API's lastUpdated. Do not edit these to match
# a later change. A change is a new settlement line.
MATCHDAY_11_RESULTS = (
    {"match_id": 560643, "home_goals": 1, "away_goals": 1, "provider_updated_at": "2026-11-24T00:20:27Z"},
    {"match_id": 560642, "home_goals": 1, "away_goals": 4, "provider_updated_at": "2026-11-24T00:20:27Z"},
    {"match_id": 560644, "home_goals": 0, "away_goals": 0, "provider_updated_at": "2026-11-24T00:20:27Z"},
    {"match_id": 560647, "home_goals": 2, "away_goals": 0, "provider_updated_at": "2026-11-24T00:20:27Z"},
    {"match_id": 560648, "home_goals": 2, "away_goals": 0, "provider_updated_at": "2026-11-24T00:20:27Z"},
    {"match_id": 560650, "home_goals": 3, "away_goals": 2, "provider_updated_at": "2026-11-24T00:20:27Z"},
    {"match_id": 560651, "home_goals": 2, "away_goals": 2, "provider_updated_at": "2026-11-24T00:20:27Z"},
    {"match_id": 560646, "home_goals": 2, "away_goals": 0, "provider_updated_at": "2026-11-24T00:20:27Z"},
    {"match_id": 560645, "home_goals": 4, "away_goals": 0, "provider_updated_at": "2026-11-24T00:20:27Z"},
    {"match_id": 560649, "home_goals": 4, "away_goals": 0, "provider_updated_at": "2026-11-24T00:20:27Z"},
)

SCORE_OMITTED_REASON = (
    "Predicted score was not stored on 17 Nov 2026, so these rows are copied "
    "from that publish and the score is left empty."
)

_COPIED_FIELDS = (
    "match_id",
    "matchday",
    "kickoff_utc",
    "home_team",
    "away_team",
    "competition",
    "model_version",
    "published_at",
    "as_of_date",
    "p_home",
    "p_draw",
    "p_away",
    "p_btts_yes",
    "p_btts_no",
    "p_over_25",
    "p_under_25",
    "book_p_home",
    "book_p_draw",
    "book_p_away",
    "book_p_over_25",
    "book_p_under_25",
    "book_source",
)


def prediction_line_from_nov17(original: dict) -> dict:
    """One ledger line with the published numbers unchanged and no score."""
    line = {field: original.get(field) for field in _COPIED_FIELDS}
    line["forecast_line"] = "model"
    line["artefact_version"] = original.get("model_version")
    line["lambda_home"] = None
    line["lambda_away"] = None
    line["rho"] = None
    line["score_table"] = None
    line["most_likely_score"] = None
    line["next_scores"] = None
    line["totals"] = None
    line["league_average_teams"] = []
    line["uses_league_average"] = False
    line["score_omitted_reason"] = SCORE_OMITTED_REASON
    return line


def settlement_line(result: dict, *, recorded_at: str, forecast_line: str = "model") -> dict:
    home_goals = int(result["home_goals"])
    away_goals = int(result["away_goals"])
    return {
        "match_id": int(result["match_id"]),
        "forecast_line": forecast_line,
        "kind": "result",
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": _result_code(home_goals, away_goals),
        "status": "settled",
        "provider_updated_at": result["provider_updated_at"],
        "recorded_at": recorded_at,
        "source": "football-data.org",
    }


def backfill_matchday_11(
    snapshot_matches: list[dict],
    *,
    recorded_at: str,
    ledger_path=None,
) -> dict:
    """Append the ten copied rows and their results. A second call adds nothing.

    Kickoff is already in the past. These lines were published on 17 Nov, before
    those kickoffs, so the "do not write after kickoff" rule does not apply.
    The copy still checks that each stored publish time is before its kickoff.
    """
    root = ensure_ledger(ledger_path)
    lines = [
        prediction_line_from_nov17(original)
        for original in snapshot_matches
        if int(original.get("matchday") or 0) == 11
    ]
    written = append_historical_predictions(lines, path=root)
    settlements = [
        settlement_line(result, recorded_at=recorded_at) for result in MATCHDAY_11_RESULTS
    ]
    settled = append_settlements(settlements, path=root)
    return {
        "predictions_inserted": written["inserted"],
        "predictions_skipped": written["skipped_duplicate"],
        "settlements_inserted": settled["inserted"],
        "settlements_skipped": settled["skipped_duplicate"],
    }


def _result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"
