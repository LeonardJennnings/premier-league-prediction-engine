import sqlite3
from pathlib import Path

import pytest

from src.store.predictions import (
    connect,
    get_prediction,
    insert_prediction,
    insert_predictions,
    settle_prediction,
    update_prediction_state,
)
from src.store.schema import init_db


def _row(**overrides):
    base = {
        "match_id": 1001,
        "model_version": "dc-test-1",
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


def test_insert_once_and_settle(tmp_path: Path):
    db = tmp_path / "predictions.sqlite"
    init_db(db)
    conn = connect(db)
    assert insert_prediction(conn, _row()) is True
    assert insert_prediction(conn, _row()) is False  # duplicate
    got = get_prediction(conn, 1001, "dc-test-1")
    assert got is not None
    assert got["p_home"] == pytest.approx(0.45)
    assert got["status"] == "scheduled"
    assert settle_prediction(conn, 1001, "dc-test-1", 2, 1, "H") is True
    got2 = get_prediction(conn, 1001, "dc-test-1")
    assert got2["status"] == "settled"
    assert got2["home_goals"] == 2
    assert got2["away_goals"] == 1
    assert got2["result"] == "H"
    assert got2["p_home"] == pytest.approx(0.45)  # probs untouched
    conn.close()


def test_settle_unknown_match_returns_false(tmp_path: Path):
    db = tmp_path / "predictions.sqlite"
    init_db(db)
    conn = connect(db)
    assert settle_prediction(conn, 999, "dc-test-1", 1, 0, "H") is False
    conn.close()


def test_insert_predictions_skips_duplicates_without_rewriting(tmp_path: Path):
    db = tmp_path / "predictions.sqlite"
    init_db(db)
    conn = connect(db)
    first = insert_predictions(conn, [_row(match_id=1), _row(match_id=2)])
    assert first == {"inserted": 2, "skipped_duplicate": 0}
    second = insert_predictions(
        conn, [_row(match_id=2, p_home=0.99), _row(match_id=3, p_home=0.11)]
    )
    assert second == {"inserted": 1, "skipped_duplicate": 1}
    kept = get_prediction(conn, 2, "dc-test-1")
    added = get_prediction(conn, 3, "dc-test-1")
    assert kept["p_home"] == pytest.approx(0.45)
    assert added["p_home"] == pytest.approx(0.11)
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 3
    conn.close()


def test_insert_predictions_rolls_back_when_a_later_row_fails(tmp_path: Path):
    db = tmp_path / "predictions.sqlite"
    init_db(db)
    conn = connect(db)
    bad = _row(match_id=2)
    del bad["p_home"]
    with pytest.raises(KeyError):
        insert_predictions(conn, [_row(match_id=1), bad])
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    assert insert_prediction(conn, _row(match_id=1)) is True
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
    conn.close()


def test_update_state_changes_kickoff_and_status_only(tmp_path: Path):
    db = tmp_path / "predictions.sqlite"
    init_db(db)
    conn = connect(db)
    assert insert_prediction(conn, _row(book_p_home=0.4)) is True
    assert update_prediction_state(
        conn,
        1001,
        "dc-test-1",
        kickoff_utc="2026-11-23T12:00:00Z",
        status="postponed",
    )
    got = get_prediction(conn, 1001, "dc-test-1")
    assert got["kickoff_utc"] == "2026-11-23T12:00:00Z"
    assert got["status"] == "postponed"
    assert got["p_home"] == pytest.approx(0.45)
    assert got["book_p_home"] == pytest.approx(0.4)
    assert got["home_goals"] is None
    with pytest.raises(ValueError, match="Refusing to update"):
        update_prediction_state(conn, 1001, "dc-test-1", p_home=0.01)
    with pytest.raises(ValueError, match="Refusing to update"):
        update_prediction_state(conn, 1001, "dc-test-1", book_p_home=0.2)
    kept = get_prediction(conn, 1001, "dc-test-1")
    assert kept["p_home"] == pytest.approx(0.45)
    assert kept["book_p_home"] == pytest.approx(0.4)
    assert update_prediction_state(conn, 999, "dc-test-1", status="cancelled") is False
    conn.close()
