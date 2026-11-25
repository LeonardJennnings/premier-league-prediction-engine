"""Rebuild the local SQLite copy from the ledger. The JSONL files stay the record."""

from __future__ import annotations

import json
from pathlib import Path

from src.ledger.export_site import resolve_match
from src.store.predictions import (
    connect,
    get_model_artefact,
    insert_prediction,
    settle_prediction,
    update_prediction_state,
    upsert_model_artefact,
)
from src.store.schema import init_db

_PROBABILITIES = (
    "p_home",
    "p_draw",
    "p_away",
    "p_btts_yes",
    "p_btts_no",
    "p_over_25",
    "p_under_25",
)


def rebuild_store(
    predictions: list[dict],
    settlements: list[dict],
    book_lines: list[dict],
    *,
    db_path: Path,
    models_dir: Path,
) -> None:
    """Insert ledger rows the store does not have, then apply scores and status.

    An existing probability is left alone. Settlement columns are updated only
    when the ledger's latest score or status differs.
    """
    init_db(db_path)
    conn = connect(db_path)
    try:
        _register_models(conn, models_dir)
        for prediction in predictions:
            row = _sqlite_row(prediction, book_lines)
            insert_prediction(conn, row)
            _apply_state(conn, prediction, settlements)
    finally:
        conn.close()


def _register_models(conn, models_dir: Path) -> None:
    if not models_dir.is_dir():
        return
    for meta_path in sorted(models_dir.glob("dc-*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        version = str(meta["model_version"])
        if get_model_artefact(conn, version) is not None:
            continue
        model_path = meta_path.parent / "model.json"
        as_of = ""
        if model_path.is_file():
            payload = json.loads(model_path.read_text(encoding="utf-8"))
            as_of = str(payload.get("as_of") or "")[:10]
        upsert_model_artefact(
            conn,
            version,
            as_of,
            str(meta_path.parent),
            None,
            created_at=str(meta.get("created_at") or ""),
        )


def _sqlite_row(prediction: dict, book_lines: list[dict]) -> dict:
    opening = [
        line
        for line in book_lines
        if int(line["match_id"]) == int(prediction["match_id"])
        and str(line.get("forecast_line")) == str(prediction.get("forecast_line"))
        and line.get("kind") == "opening"
    ]
    book = opening[-1] if opening else None
    detail = {
        "artefact_version": prediction.get("artefact_version"),
        "forecast_line": prediction.get("forecast_line"),
        "lambda_home": prediction.get("lambda_home"),
        "lambda_away": prediction.get("lambda_away"),
        "rho": prediction.get("rho"),
        "score_table": prediction.get("score_table"),
        "most_likely_score": prediction.get("most_likely_score"),
        "next_scores": prediction.get("next_scores"),
        "totals": prediction.get("totals"),
        "league_average_teams": prediction.get("league_average_teams"),
        "uses_league_average": prediction.get("uses_league_average"),
        "score_omitted_reason": prediction.get("score_omitted_reason"),
    }
    row = {
        "match_id": int(prediction["match_id"]),
        "model_version": prediction["model_version"],
        "kickoff_utc": prediction["kickoff_utc"],
        "home_team": prediction["home_team"],
        "away_team": prediction["away_team"],
        "matchday": prediction.get("matchday"),
        "competition": prediction.get("competition") or "PL",
        "published_at": prediction["published_at"],
        "as_of_date": prediction.get("as_of_date") or "",
        "status": "scheduled",
        "detail_json": json.dumps(detail, sort_keys=True),
        "book_p_home": None if book is None else book.get("p_home"),
        "book_p_draw": None if book is None else book.get("p_draw"),
        "book_p_away": None if book is None else book.get("p_away"),
        "book_p_over_25": None if book is None else book.get("p_over_25"),
        "book_p_under_25": None if book is None else book.get("p_under_25"),
        "book_source": None if book is None else book.get("source"),
    }
    for name in _PROBABILITIES:
        row[name] = prediction[name]
    return row


def _apply_state(conn, prediction: dict, settlements: list[dict]) -> None:
    state = resolve_match(prediction, settlements)
    match_id = int(prediction["match_id"])
    version = prediction["model_version"]
    current = conn.execute(
        "SELECT * FROM predictions WHERE match_id = ? AND model_version = ?",
        (match_id, version),
    ).fetchone()
    if current is None:
        return
    if state["status"] == "settled":
        if (
            current["status"] == "settled"
            and current["home_goals"] == state["home_goals"]
            and current["away_goals"] == state["away_goals"]
        ):
            return
        settle_prediction(
            conn,
            match_id,
            version,
            int(state["home_goals"]),
            int(state["away_goals"]),
            str(state["result"]),
        )
        return
    fields = {}
    if state["status"] in {"postponed", "cancelled"} and current["status"] != state["status"]:
        fields["status"] = state["status"]
    if state["status"] == "scheduled" and current["status"] == "postponed":
        fields["status"] = "scheduled"
    if state["kickoff_utc"] and state["kickoff_utc"] != current["kickoff_utc"]:
        if state["status"] in {"postponed", "cancelled", "scheduled"}:
            fields["kickoff_utc"] = state["kickoff_utc"]
    if fields:
        update_prediction_state(conn, match_id, version, **fields)
