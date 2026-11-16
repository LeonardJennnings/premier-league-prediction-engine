from pathlib import Path

from src.store.predictions import (
    connect,
    get_latest_model_version,
    get_model_artefact,
    insert_prediction,
    list_settled_predictions_for_week,
    list_settled_weeks,
    list_upcoming_predictions,
    settle_prediction,
    upsert_model_artefact,
)
from src.store.schema import init_db


def _row(**overrides):
    base = {
        "match_id": 1001,
        "model_version": "v-old",
        "kickoff_utc": "2026-11-22T15:00:00Z",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "matchday": 12,
        "competition": "PL",
        "p_home": 0.45,
        "p_draw": 0.28,
        "p_away": 0.27,
        "p_btts_yes": 0.55,
        "p_btts_no": 0.45,
        "p_over_25": 0.52,
        "p_under_25": 0.48,
        "book_p_home": None,
        "book_p_draw": None,
        "book_p_away": None,
        "book_p_over_25": None,
        "book_p_under_25": None,
        "book_source": None,
        "published_at": "2026-11-16T12:00:00Z",
        "as_of_date": "2026-11-16",
        "status": "scheduled",
    }
    base.update(overrides)
    return base


def test_latest_model_version_by_created_at(tmp_path: Path):
    db = init_db(tmp_path / "p.sqlite")
    conn = connect(db)
    upsert_model_artefact(conn, "v-old", "2026-11-01", "models/v-old", {}, created_at="2026-11-01T00:00:00Z")
    upsert_model_artefact(conn, "v-new", "2026-11-10", "models/v-new", {}, created_at="2026-11-10T00:00:00Z")
    assert get_latest_model_version(conn) == "v-new"
    assert get_model_artefact(conn, "v-old")["as_of_date"] == "2026-11-01"
    conn.close()


def test_list_upcoming_respects_window_and_status(tmp_path: Path):
    db = init_db(tmp_path / "p.sqlite")
    conn = connect(db)
    now = "2026-11-16T12:00:00Z"
    insert_prediction(conn, _row(match_id=1, kickoff_utc="2026-11-17T15:00:00Z", status="scheduled"))
    insert_prediction(conn, _row(match_id=2, kickoff_utc="2026-11-18T15:00:00Z", status="postponed"))
    insert_prediction(conn, _row(match_id=3, kickoff_utc="2026-11-30T15:00:00Z", status="scheduled"))  # outside 10d
    insert_prediction(conn, _row(match_id=4, kickoff_utc="2026-11-10T15:00:00Z", status="scheduled"))  # past
    insert_prediction(conn, _row(match_id=5, kickoff_utc="2026-11-17T12:00:00Z", status="settled"))
    rows = list_upcoming_predictions(conn, "v-old", now_utc=now, window_days=10)
    assert [r["match_id"] for r in rows] == [1, 2]
    conn.close()


def test_settled_weeks_iso_and_detail(tmp_path: Path):
    db = init_db(tmp_path / "p.sqlite")
    conn = connect(db)
    # 2026-11-14 is Saturday of ISO week 2026-W46
    insert_prediction(conn, _row(match_id=10, kickoff_utc="2026-11-14T15:00:00Z"))
    settle_prediction(conn, 10, "v-old", 2, 1, "H")
    insert_prediction(conn, _row(match_id=11, kickoff_utc="2026-11-07T15:00:00Z"))
    settle_prediction(conn, 11, "v-old", 0, 0, "D")
    insert_prediction(conn, _row(match_id=12, kickoff_utc="2026-11-14T15:00:00Z", status="scheduled"))
    weeks = list_settled_weeks(conn, "v-old")
    assert weeks[0]["iso_week"] == "2026-W46"
    assert weeks[0]["match_count"] == 1
    assert weeks[1]["iso_week"] == "2026-W45"
    detail = list_settled_predictions_for_week(conn, "v-old", "2026-W46")
    assert [r["match_id"] for r in detail] == [10]
    conn.close()
