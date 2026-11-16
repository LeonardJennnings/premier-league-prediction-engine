"""Settlement writes scores only. Published probabilities stay as inserted."""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.jobs.settle import POSTPONED_LOOKAHEAD_DAYS, main, settle_from_api
from src.store.predictions import connect, get_prediction, insert_prediction
from src.store.schema import init_db

_PROBS = (
    "p_home",
    "p_draw",
    "p_away",
    "p_btts_yes",
    "p_btts_no",
    "p_over_25",
    "p_under_25",
)


def _row(**overrides):
    base = {
        "match_id": 1001,
        "model_version": "dc-test-1",
        "kickoff_utc": "2026-11-18T15:00:00Z",
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


def _finished(match_id, kickoff, home_goals, away_goals, winner, status="FINISHED"):
    """One football-data.org match object, including the full-time score."""
    return {
        "id": match_id,
        "utcDate": kickoff,
        "status": status,
        "matchday": 12,
        "homeTeam": {"name": "Arsenal FC"},
        "awayTeam": {"name": "Chelsea FC"},
        "score": {
            "winner": winner,
            "duration": "REGULAR",
            "fullTime": {"home": home_goals, "away": away_goals},
            "halfTime": {"home": 0, "away": 0},
        },
    }


def _patch_db(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "predictions.sqlite"
    init_db(db)
    monkeypatch.setattr("src.jobs.settle.PREDICTIONS_DB", db)
    cache = tmp_path / "fixtures"
    cache.mkdir()
    monkeypatch.setattr("src.fixtures.client.FIXTURES_CACHE_DIR", cache)
    return db


def _patch_payload(monkeypatch, payload: dict) -> dict:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        captured["calls"] = captured.get("calls", 0) + 1
        return FakeResponse()

    monkeypatch.setattr("src.fixtures.client.requests.get", fake_get)
    return captured


def _probs(row: dict) -> dict:
    return {name: row[name] for name in _PROBS}


def test_settle_from_finished_payload_keeps_probabilities(tmp_path, monkeypatch):
    db = _patch_db(monkeypatch, tmp_path)
    conn = connect(db)
    scheduled = [
        _row(match_id=1001, kickoff_utc="2026-11-18T15:00:00Z", p_home=0.45),
        _row(match_id=1002, kickoff_utc="2026-11-22T17:30:00Z", p_home=0.33),
        _row(match_id=1003, kickoff_utc="2026-11-19T20:00:00Z", p_home=0.40),
        _row(match_id=1004, kickoff_utc="2026-11-21T12:30:00Z", p_home=0.20),
        _row(match_id=1005, kickoff_utc="2026-11-20T15:00:00Z", p_home=0.51),
    ]
    for row in scheduled:
        assert insert_prediction(conn, row) is True
    before = {
        row["match_id"]: _probs(get_prediction(conn, row["match_id"], "dc-test-1"))
        for row in scheduled
    }
    conn.close()

    payload = {
        "matches": [
            _finished(1001, "2026-11-18T15:00:00Z", 2, 1, "HOME_TEAM"),
            _finished(1003, "2026-11-19T20:00:00Z", 0, 0, "DRAW"),
            _finished(1004, "2026-11-21T12:30:00Z", 1, 3, "AWAY_TEAM"),
            _finished(9999, "2026-11-18T19:00:00Z", 1, 1, "DRAW"),
            _finished(1005, "2026-11-20T15:00:00Z", 4, 0, "HOME_TEAM", status="TIMED"),
        ]
    }
    captured = _patch_payload(monkeypatch, payload)

    counts = settle_from_api(token="secret-token")

    assert counts == {
        "unsettled": 5,
        "fetched": 5,
        "settled": 3,
        "skipped_unknown": 1,
    }
    url = captured["url"]
    assert "competitions/PL/matches" in url
    assert "status=" not in url
    assert "dateFrom=2026-11-18" in url
    assert "dateTo=2026-11-23" in url
    assert captured["headers"]["X-Auth-Token"] == "secret-token"
    assert captured["calls"] == 1

    conn = connect(db)
    try:
        home = get_prediction(conn, 1001, "dc-test-1")
        draw = get_prediction(conn, 1003, "dc-test-1")
        away = get_prediction(conn, 1004, "dc-test-1")
        still_open = get_prediction(conn, 1002, "dc-test-1")
        not_finished = get_prediction(conn, 1005, "dc-test-1")
        unknown = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE match_id = 9999"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        conn.close()

    assert home["status"] == "settled"
    assert home["home_goals"] == 2
    assert home["away_goals"] == 1
    assert home["result"] == "H"
    assert home["settled_at"]
    assert draw["status"] == "settled"
    assert draw["home_goals"] == 0
    assert draw["away_goals"] == 0
    assert draw["result"] == "D"
    assert away["status"] == "settled"
    assert away["home_goals"] == 1
    assert away["away_goals"] == 3
    assert away["result"] == "A"
    assert still_open["status"] == "scheduled"
    assert still_open["home_goals"] is None
    assert still_open["result"] is None
    assert not_finished["status"] == "scheduled"
    assert not_finished["home_goals"] is None
    assert unknown == 0
    assert total == 5
    for match_id, row in (
        (1001, home),
        (1002, still_open),
        (1003, draw),
        (1004, away),
        (1005, not_finished),
    ):
        assert _probs(row) == before[match_id]


def test_settle_limits_to_requested_model_version(tmp_path, monkeypatch):
    db = _patch_db(monkeypatch, tmp_path)
    conn = connect(db)
    insert_prediction(conn, _row(model_version="dc-a", p_home=0.61))
    insert_prediction(conn, _row(model_version="dc-b", p_home=0.22))
    conn.close()
    _patch_payload(
        monkeypatch,
        {"matches": [_finished(1001, "2026-11-18T15:00:00Z", 1, 0, "HOME_TEAM")]},
    )

    counts = settle_from_api(token="secret-token", model_version="dc-a")

    assert counts["unsettled"] == 1
    assert counts["settled"] == 1
    assert counts["skipped_unknown"] == 0
    conn = connect(db)
    try:
        settled = get_prediction(conn, 1001, "dc-a")
        other = get_prediction(conn, 1001, "dc-b")
    finally:
        conn.close()
    assert settled["status"] == "settled"
    assert settled["result"] == "H"
    assert settled["p_home"] == pytest.approx(0.61)
    assert other["status"] == "scheduled"
    assert other["home_goals"] is None
    assert other["p_home"] == pytest.approx(0.22)


def test_settle_refetches_same_dates_after_cache_write(tmp_path, monkeypatch):
    """A later FINISHED match on the same date span must be fetched again.

    The cache directory is fixed. The first pull writes it. Without skipping
    the cache read, the second settle would reuse that file and leave the new
    row unsettled.
    """
    db = _patch_db(monkeypatch, tmp_path)
    cache_dir = tmp_path / "fixtures"
    conn = connect(db)
    insert_prediction(conn, _row(match_id=1001, kickoff_utc="2026-11-18T15:00:00Z"))
    insert_prediction(conn, _row(match_id=1002, kickoff_utc="2026-11-18T17:30:00Z"))
    conn.close()

    first = {"matches": [_finished(1001, "2026-11-18T15:00:00Z", 2, 1, "HOME_TEAM")]}
    second = {
        "matches": [
            _finished(1001, "2026-11-18T15:00:00Z", 2, 1, "HOME_TEAM"),
            _finished(1002, "2026-11-18T17:30:00Z", 0, 1, "AWAY_TEAM"),
        ]
    }
    payloads = [first, second]
    captured = {"calls": 0, "urls": []}

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured["calls"] += 1
        captured["urls"].append(url)
        return FakeResponse(payloads[captured["calls"] - 1])

    monkeypatch.setattr("src.fixtures.client.requests.get", fake_get)

    first_counts = settle_from_api(token="secret-token")
    cache_file = cache_dir / "PL_2026-11-18_2026-11-19.json"
    assert first_counts["settled"] == 1
    assert captured["calls"] == 1
    assert cache_file.is_file()

    second_counts = settle_from_api(token="secret-token")

    assert captured["calls"] == 2
    assert second_counts == {
        "unsettled": 1,
        "fetched": 2,
        "settled": 1,
        "skipped_unknown": 0,
    }
    for url in captured["urls"]:
        assert "dateFrom=2026-11-18" in url
        assert "dateTo=2026-11-19" in url
        assert "status=" not in url

    conn = connect(db)
    try:
        already = get_prediction(conn, 1001, "dc-test-1")
        newly = get_prediction(conn, 1002, "dc-test-1")
    finally:
        conn.close()
    assert already["status"] == "settled"
    assert already["result"] == "H"
    assert newly["status"] == "settled"
    assert newly["home_goals"] == 0
    assert newly["away_goals"] == 1
    assert newly["result"] == "A"
    written = json.loads(cache_file.read_text(encoding="utf-8"))
    assert [match["id"] for match in written["matches"]] == [1001, 1002]


def test_settle_with_no_unsettled_rows_does_not_fetch(tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("network should not be called")

    monkeypatch.setattr("src.fixtures.client.requests.get", boom)

    assert settle_from_api(token="secret-token") == {
        "unsettled": 0,
        "fetched": 0,
        "settled": 0,
        "skipped_unknown": 0,
    }


def _listed(match_id, kickoff, status):
    return {
        "id": match_id,
        "utcDate": kickoff,
        "status": status,
        "matchday": 12,
        "homeTeam": {"name": "Arsenal FC"},
        "awayTeam": {"name": "Chelsea FC"},
        "score": {"fullTime": {"home": None, "away": None}},
    }


def _sequential_payloads(monkeypatch, payloads: list[dict]) -> dict:
    captured = {"calls": 0, "urls": []}

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured["calls"] += 1
        captured["urls"].append(url)
        return FakeResponse(payloads[captured["calls"] - 1])

    monkeypatch.setattr("src.fixtures.client.requests.get", fake_get)
    return captured


@pytest.mark.parametrize("api_status", ["POSTPONED", "SUSPENDED"])
def test_settle_postponed_updates_status_and_kickoff(tmp_path, monkeypatch, api_status):
    db = _patch_db(monkeypatch, tmp_path)
    conn = connect(db)
    original = _row(book_p_home=0.41, p_home=0.45)
    assert insert_prediction(conn, original) is True
    before = _probs(get_prediction(conn, 1001, "dc-test-1"))
    conn.close()
    _patch_payload(
        monkeypatch,
        {
            "matches": [
                _listed(1001, "2026-11-18T17:30:00Z", api_status),
                _listed(9999, "2026-11-18T19:00:00Z", api_status),
            ]
        },
    )

    counts = settle_from_api(token="secret-token")

    assert counts["unsettled"] == 1
    assert counts["settled"] == 0
    assert counts["skipped_unknown"] == 1
    conn = connect(db)
    try:
        row = get_prediction(conn, 1001, "dc-test-1")
        unknown = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE match_id = 9999"
        ).fetchone()[0]
    finally:
        conn.close()
    assert row["status"] == "postponed"
    assert row["kickoff_utc"] == "2026-11-18T17:30:00Z"
    assert row["home_goals"] is None
    assert row["result"] is None
    assert row["book_p_home"] == pytest.approx(0.41)
    assert _probs(row) == before
    assert unknown == 0


def test_settle_cancelled_updates_status_and_stops_polling(tmp_path, monkeypatch):
    db = _patch_db(monkeypatch, tmp_path)
    conn = connect(db)
    assert insert_prediction(conn, _row(p_home=0.37)) is True
    conn.close()
    _patch_payload(
        monkeypatch,
        {"matches": [_listed(1001, "2026-11-18T15:00:00Z", "CANCELLED")]},
    )

    counts = settle_from_api(token="secret-token")

    assert counts["settled"] == 0
    conn = connect(db)
    try:
        row = get_prediction(conn, 1001, "dc-test-1")
    finally:
        conn.close()
    assert row["status"] == "cancelled"
    assert row["p_home"] == pytest.approx(0.37)
    assert row["home_goals"] is None

    def boom(*args, **kwargs):
        raise AssertionError("cancelled rows should not be fetched")

    monkeypatch.setattr("src.fixtures.client.requests.get", boom)
    assert settle_from_api(token="secret-token") == {
        "unsettled": 0,
        "fetched": 0,
        "settled": 0,
        "skipped_unknown": 0,
    }


def test_settle_reschedule_after_postpone_updates_kickoff(tmp_path, monkeypatch):
    db = _patch_db(monkeypatch, tmp_path)
    conn = connect(db)
    assert insert_prediction(conn, _row(p_home=0.45, book_p_away=0.22)) is True
    before_probs = _probs(get_prediction(conn, 1001, "dc-test-1"))
    conn.close()
    payloads = [
        {"matches": [_listed(1001, "2026-11-18T15:00:00Z", "POSTPONED")]},
        {"matches": [_listed(1001, "2026-12-02T15:00:00Z", "TIMED")]},
    ]
    captured = _sequential_payloads(monkeypatch, payloads)

    settle_from_api(token="secret-token")
    settle_from_api(token="secret-token")

    assert captured["calls"] == 2
    assert "dateFrom=2026-11-18" in captured["urls"][0]
    assert "dateTo=2026-11-19" in captured["urls"][0]
    horizon = date(2026, 11, 18) + timedelta(days=POSTPONED_LOOKAHEAD_DAYS + 1)
    assert "dateFrom=2026-11-18" in captured["urls"][1]
    assert f"dateTo={horizon.isoformat()}" in captured["urls"][1]
    conn = connect(db)
    try:
        row = get_prediction(conn, 1001, "dc-test-1")
    finally:
        conn.close()
    assert row["status"] == "scheduled"
    assert row["kickoff_utc"] == "2026-12-02T15:00:00Z"
    assert row["book_p_away"] == pytest.approx(0.22)
    assert _probs(row) == {name: before_probs[name] for name in before_probs}


def test_settle_corrects_finished_score_without_touching_probabilities(
    tmp_path, monkeypatch
):
    db = _patch_db(monkeypatch, tmp_path)
    conn = connect(db)
    assert insert_prediction(conn, _row(p_home=0.45, book_p_home=0.4)) is True
    conn.close()
    payloads = [
        {"matches": [_finished(1001, "2026-11-18T15:00:00Z", 2, 1, "HOME_TEAM")]},
        {"matches": [_finished(1001, "2026-11-18T15:00:00Z", 1, 1, "DRAW")]},
        {"matches": [_finished(1001, "2026-11-18T15:00:00Z", 1, 1, "DRAW")]},
    ]
    captured = _sequential_payloads(monkeypatch, payloads)

    first = settle_from_api(token="secret-token")
    assert first["settled"] == 1

    second = settle_from_api(token="secret-token")
    conn = connect(db)
    try:
        settled_at = get_prediction(conn, 1001, "dc-test-1")["settled_at"]
    finally:
        conn.close()
    third = settle_from_api(token="secret-token")

    assert captured["calls"] == 3
    assert second == {
        "unsettled": 0,
        "fetched": 1,
        "settled": 1,
        "skipped_unknown": 0,
    }
    assert third["settled"] == 0
    assert third["unsettled"] == 0
    conn = connect(db)
    try:
        row = get_prediction(conn, 1001, "dc-test-1")
    finally:
        conn.close()
    assert row["status"] == "settled"
    assert row["home_goals"] == 1
    assert row["away_goals"] == 1
    assert row["result"] == "D"
    assert row["settled_at"] == settled_at
    assert row["p_home"] == pytest.approx(0.45)
    assert row["book_p_home"] == pytest.approx(0.4)
    assert row["home_team"] == "Arsenal"


def test_settle_main_logs_counts(monkeypatch, caplog):
    monkeypatch.setattr(
        "src.jobs.settle.settle_from_api",
        lambda **kwargs: {
            "unsettled": 2,
            "fetched": 1,
            "settled": 1,
            "skipped_unknown": 0,
        },
    )
    with caplog.at_level("INFO", logger="src.jobs.settle"):
        main()
    assert "settled=1" in caplog.text
    assert "unsettled=2" in caplog.text
