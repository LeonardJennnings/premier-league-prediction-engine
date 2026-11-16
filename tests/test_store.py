import sqlite3
from pathlib import Path

import pytest

from src.store.predictions import connect, get_prediction, insert_prediction, settle_prediction
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
