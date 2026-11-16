import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.fixtures.client import fetch_pl_matches, normalize_match
from src.fixtures.window import matches_in_publish_window


def test_normalize_match():
    raw = {
        "id": 42,
        "utcDate": "2026-11-22T15:00:00Z",
        "matchday": 12,
        "status": "TIMED",
        "homeTeam": {"name": "Arsenal FC"},
        "awayTeam": {"name": "Chelsea FC"},
    }
    m = normalize_match(raw)
    assert m["match_id"] == 42
    assert m["home_team"] == "Arsenal FC"
    assert m["away_team"] == "Chelsea FC"
    assert m["kickoff_utc"] == "2026-11-22T15:00:00Z"
    assert m["matchday"] == 12
    assert m["status"] == "TIMED"


def test_window_filters(tmp_path: Path):
    now = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    matches = [
        {"utcDate": "2026-11-15T15:00:00Z"},  # past
        {"utcDate": "2026-11-20T15:00:00Z"},  # inside
        {"utcDate": "2026-11-30T15:00:00Z"},  # outside 10d
    ]
    out = matches_in_publish_window(matches, now_utc=now, window_days=10)
    assert len(out) == 1
    assert out[0]["utcDate"].startswith("2026-11-20")


def test_fetch_uses_cache(tmp_path: Path, monkeypatch):
    cache = tmp_path / "fixtures"
    cache.mkdir()
    payload = {"matches": [{"id": 1, "utcDate": "2026-11-20T15:00:00Z",
                            "matchday": 1, "status": "TIMED",
                            "homeTeam": {"name": "Arsenal FC"},
                            "awayTeam": {"name": "Chelsea FC"}}]}
    (cache / "PL_2026-11-16_2026-11-26.json").write_text(
        __import__("json").dumps(payload), encoding="utf-8"
    )

    def boom(*args, **kwargs):
        raise AssertionError("network should not be called")

    monkeypatch.setattr("src.fixtures.client.requests.get", boom)
    rows = fetch_pl_matches(
        "2026-11-16", "2026-11-26", token="dummy", cache_dir=cache
    )
    assert len(rows) == 1
    assert rows[0]["match_id"] == 1


def test_window_filters_normalized_kickoff_utc():
    now = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    matches = [
        {
            "match_id": 1,
            "kickoff_utc": "2026-11-15T15:00:00Z",
            "home_team": "Past FC",
            "away_team": "Gone FC",
            "matchday": 11,
            "status": "FINISHED",
        },
        {
            "match_id": 2,
            "kickoff_utc": "2026-11-20T15:00:00Z",
            "home_team": "Arsenal FC",
            "away_team": "Chelsea FC",
            "matchday": 12,
            "status": "TIMED",
        },
        {
            "match_id": 3,
            "kickoff_utc": "2026-11-30T15:00:00Z",
            "home_team": "Far FC",
            "away_team": "Away FC",
            "matchday": 14,
            "status": "TIMED",
        },
    ]
    out = matches_in_publish_window(matches, now_utc=now, window_days=10)
    assert len(out) == 1
    assert out[0]["match_id"] == 2
    assert "utcDate" not in out[0]


def test_window_is_half_open():
    now = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    at_start = {"utcDate": "2026-11-16T12:00:00Z"}
    just_before_end = {"utcDate": "2026-11-26T11:59:59Z"}
    at_end = {"utcDate": "2026-11-26T12:00:00Z"}
    out = matches_in_publish_window(
        [at_start, just_before_end, at_end], now_utc=now, window_days=10
    )
    dates = [m["utcDate"] for m in out]
    assert dates == [at_start["utcDate"], just_before_end["utcDate"]]


def test_fetch_requires_token_on_cache_miss(tmp_path: Path):
    cache = tmp_path / "fixtures"
    cache.mkdir()
    with pytest.raises(ValueError, match="FOOTBALL_DATA_ORG_TOKEN"):
        fetch_pl_matches(
            "2026-11-16", "2026-11-26", token="", cache_dir=cache
        )


def test_fetch_cache_miss_writes_cache(tmp_path: Path, monkeypatch):
    cache = tmp_path / "fixtures"
    cache.mkdir()
    payload = {
        "matches": [
            {
                "id": 7,
                "utcDate": "2026-11-20T15:00:00Z",
                "matchday": 2,
                "status": "TIMED",
                "homeTeam": {"name": "Arsenal FC"},
                "awayTeam": {"name": "Chelsea FC"},
            }
        ]
    }
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
        return FakeResponse()

    monkeypatch.setattr("src.fixtures.client.requests.get", fake_get)
    rows = fetch_pl_matches(
        "2026-11-16", "2026-11-26", token="secret", cache_dir=cache
    )
    assert len(rows) == 1
    assert rows[0]["match_id"] == 7
    cache_file = cache / "PL_2026-11-16_2026-11-26.json"
    assert json.loads(cache_file.read_text(encoding="utf-8")) == payload
    assert not list(cache.glob("*.tmp"))
    assert "competitions/PL/matches" in captured["url"]
    assert "dateFrom=2026-11-16" in captured["url"]
    assert "dateTo=2026-11-26" in captured["url"]
    assert captured["headers"]["X-Auth-Token"] == "secret"
    assert captured["timeout"] is not None


def test_fetch_uses_session_when_provided(tmp_path: Path):
    cache = tmp_path / "fixtures"
    cache.mkdir()
    payload = {
        "matches": [
            {
                "id": 9,
                "utcDate": "2026-11-21T15:00:00Z",
                "matchday": 3,
                "status": "TIMED",
                "homeTeam": {"name": "Liverpool FC"},
                "awayTeam": {"name": "Everton FC"},
            }
        ]
    }

    class FakeSession:
        def get(self, url, headers=None, timeout=None, **kwargs):
            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return payload

            return FakeResponse()

    rows = fetch_pl_matches(
        "2026-11-16",
        "2026-11-26",
        token="secret",
        cache_dir=cache,
        session=FakeSession(),
    )
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Liverpool FC"

