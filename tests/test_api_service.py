from pathlib import Path

import pytest

from src.api.service import (
    NotFoundError,
    get_latest_model,
    get_model,
    get_upcoming,
    get_week,
    list_weeks,
    resolve_model_version,
)
from src.store.predictions import connect, insert_prediction, settle_prediction, upsert_model_artefact
from src.store.schema import init_db


def _seed(tmp_path: Path):
    db = init_db(tmp_path / "p.sqlite")
    conn = connect(db)
    upsert_model_artefact(conn, "v-old", "2026-11-01", "models/v-old", {"xi": 0.001}, created_at="2026-11-01T00:00:00Z")
    upsert_model_artefact(conn, "v-new", "2026-11-10", "models/v-new", {"xi": 0.002}, created_at="2026-11-10T00:00:00Z")
    return conn


def _row(**overrides):
    base = {
        "match_id": 1001,
        "model_version": "v-new",
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


def test_resolve_defaults_to_latest(tmp_path: Path):
    conn = _seed(tmp_path)
    assert resolve_model_version(conn, None) == "v-new"
    assert resolve_model_version(conn, "v-old") == "v-old"
    with pytest.raises(NotFoundError):
        resolve_model_version(conn, "missing")
    conn.close()


def test_get_week_404_when_empty(tmp_path: Path):
    conn = _seed(tmp_path)
    with pytest.raises(NotFoundError):
        get_week(conn, "2026-W01", None)
    conn.close()


def test_get_upcoming_happy_path(tmp_path: Path):
    conn = _seed(tmp_path)
    now = "2026-11-16T12:00:00Z"
    empty = get_upcoming(conn, None, now_utc=now)
    assert empty == {"model_version": "public", "matches": []}
    insert_prediction(conn, _row(match_id=1, kickoff_utc="2026-11-17T15:00:00Z", status="scheduled"))
    insert_prediction(conn, _row(match_id=2, kickoff_utc="2026-11-18T15:00:00Z", status="postponed"))
    insert_prediction(conn, _row(match_id=3, kickoff_utc="2026-11-30T15:00:00Z", status="scheduled"))
    insert_prediction(conn, _row(match_id=4, model_version="v-old", kickoff_utc="2026-11-17T15:00:00Z"))
    got = get_upcoming(conn, None, now_utc=now)
    assert got["model_version"] == "public"
    assert [r["match_id"] for r in got["matches"]] == [1, 4, 2]
    pinned = get_upcoming(conn, "v-old", now_utc=now)
    assert pinned["model_version"] == "v-old"
    assert [r["match_id"] for r in pinned["matches"]] == [4]
    conn.close()


def test_list_weeks_and_get_week_happy_path(tmp_path: Path):
    conn = _seed(tmp_path)
    insert_prediction(conn, _row(match_id=10, kickoff_utc="2026-11-14T15:00:00Z"))
    settle_prediction(conn, 10, "v-new", 2, 1, "H")
    insert_prediction(conn, _row(match_id=11, kickoff_utc="2026-11-07T15:00:00Z"))
    settle_prediction(conn, 11, "v-new", 0, 0, "D")
    insert_prediction(conn, _row(match_id=12, kickoff_utc="2026-11-14T15:00:00Z", status="scheduled"))
    weeks = list_weeks(conn, None)
    assert weeks["model_version"] == "public"
    assert weeks["weeks"][0] == {"iso_week": "2026-W46", "match_count": 1}
    assert weeks["weeks"][1] == {"iso_week": "2026-W45", "match_count": 1}
    detail = get_week(conn, "2026-W46", None)
    assert detail["model_version"] == "public"
    assert detail["iso_week"] == "2026-W46"
    assert [r["match_id"] for r in detail["matches"]] == [10]
    assert detail["matches"][0]["home_goals"] == 2
    assert detail["matches"][0]["result"] == "H"
    conn.close()


def test_unpinned_read_keeps_the_first_published_row(tmp_path: Path):
    conn = _seed(tmp_path)
    insert_prediction(
        conn,
        _row(
            match_id=1,
            model_version="v-old",
            published_at="2026-11-01T00:00:00Z",
            p_home=0.11,
            kickoff_utc="2026-11-17T15:00:00Z",
        ),
    )
    insert_prediction(
        conn,
        _row(
            match_id=1,
            model_version="v-new",
            published_at="2026-11-16T12:00:00Z",
            p_home=0.45,
            kickoff_utc="2026-11-17T15:00:00Z",
        ),
    )
    got = get_upcoming(conn, None, now_utc="2026-11-16T12:00:00Z")
    row = next(match for match in got["matches"] if match["match_id"] == 1)
    assert row["model_version"] == "v-old"
    assert row["p_home"] == 0.11
    pinned = get_upcoming(conn, "v-new", now_utc="2026-11-16T12:00:00Z")
    assert [match["p_home"] for match in pinned["matches"]] == [0.45]
    conn.close()


def test_latest_model_ignores_sample_local_when_a_real_version_exists(tmp_path: Path):
    conn = _seed(tmp_path)
    upsert_model_artefact(
        conn,
        "sample-local",
        "2026-11-25",
        "fixtures/sample-local",
        {},
        created_at="2026-11-25T00:00:00Z",
    )
    assert get_latest_model(conn)["model_version"] == "v-new"
    conn.close()


def test_get_model_parses_hyperparams(tmp_path: Path):
    conn = _seed(tmp_path)
    latest = get_latest_model(conn)
    assert latest["model_version"] == "v-new"
    assert latest["as_of_date"] == "2026-11-10"
    assert latest["created_at"] == "2026-11-10T00:00:00Z"
    assert latest["artefact_path"] == "models/v-new"
    assert latest["hyperparams"] == {"xi": 0.002}
    old = get_model(conn, "v-old")
    assert old["hyperparams"] == {"xi": 0.001}
    with pytest.raises(NotFoundError):
        get_model(conn, "missing")
    conn.close()


def test_get_model_corrupt_hyperparams_are_null(tmp_path: Path):
    db = init_db(tmp_path / "p.sqlite")
    conn = connect(db)
    upsert_model_artefact(
        conn,
        "v-bad",
        "2026-11-10",
        "models/v-bad",
        "{not-json",
        created_at="2026-11-10T00:00:00Z",
    )
    payload = get_model(conn, "v-bad")
    assert payload["hyperparams"] is None
    conn.close()
