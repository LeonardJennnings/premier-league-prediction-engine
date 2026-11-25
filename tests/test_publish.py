import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.eval.market_blend import published_blend_table
from src.fixtures.books import BOOK_SOURCE, index_fixtures_csv
from src.jobs.publish import main, publish_slate
from src.ledger.export_site import build_site_files
from src.ledger.lines import FORECAST_LINE_MARKET, MARKET_LINE_LABEL
from src.ledger.log import LedgerRefused, read_jsonl
from src.models.artefacts import published_model_version, save_artefact
from src.models.dixon_coles import DixonColesModel, score_table
from src.models.markets import markets_from_score_table
from src.models.predict import predict_matches
from src.store.predictions import connect, get_prediction
from src.store.schema import init_db


def _model(**overrides) -> DixonColesModel:
    attack = overrides.pop("attack", {"Arsenal": 1.2, "Chelsea": 0.9})
    defence = overrides.pop("defence", {"Arsenal": 0.8, "Chelsea": 1.1})
    return DixonColesModel(
        attack=attack,
        defence=defence,
        home_advantage=1.3,
        rho=0.0,
        xi=0.0019,
        as_of=pd.Timestamp("2026-11-16"),
    )


def _patch_paths(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    db = tmp_path / "db.sqlite"
    ledger = tmp_path / "ledger"
    init_db(db)
    monkeypatch.setattr("src.jobs.publish.MODELS_DIR", models_dir)
    monkeypatch.setattr("src.jobs.publish.PREDICTIONS_DB", db)
    monkeypatch.setattr("src.jobs.publish.LEDGER_DIR", ledger)
    # Keep publish tests off the live fixtures.csv. Book tests override this.
    monkeypatch.setattr("src.jobs.publish.load_book_index", lambda: {})
    return models_dir, db, ledger


def _prediction_count(db) -> int:
    conn = connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        conn.close()


def test_publish_aborts_on_unmapped_team(tmp_path, monkeypatch):
    models_dir, db, ledger = _patch_paths(monkeypatch, tmp_path)
    model = DixonColesModel(
        attack={"Arsenal": 1.0, "Chelsea": 1.0},
        defence={"Arsenal": 1.0, "Chelsea": 1.0},
        home_advantage=1.3,
        rho=0.0,
        xi=0.0019,
        as_of=pd.Timestamp("2026-11-16"),
    )
    version, _ = save_artefact(model, 1.0, {"xi": 0.0019}, models_dir)

    monkeypatch.setattr(
        "src.jobs.publish.fetch_pl_matches",
        lambda *a, **k: [
            {
                "match_id": 1,
                "kickoff_utc": "2026-11-20T15:00:00Z",
                "home_team": "Not A Real Team FC",
                "away_team": "Chelsea FC",
                "matchday": 1,
                "status": "TIMED",
            }
        ],
    )
    monkeypatch.setattr(
        "src.jobs.publish.load_team_name_map",
        lambda: {"Chelsea FC": "Chelsea"},
    )

    with pytest.raises(KeyError, match="Unmapped team"):
        publish_slate(
            version,
            now_utc=datetime(2026, 11, 16, tzinfo=timezone.utc),
            token="x",
        )
    assert _prediction_count(db) == 0
    assert read_jsonl(ledger / "predictions.jsonl") == []


def test_publish_inserts_and_refuses_duplicate(tmp_path, monkeypatch):
    models_dir, db, ledger = _patch_paths(monkeypatch, tmp_path)
    model = _model()
    version, _ = save_artefact(model, 1.4, {"xi": 0.0019}, models_dir)
    fixtures = [
        {
            "match_id": 7,
            "kickoff_utc": "2026-11-20T15:00:00Z",
            "home_team": "Arsenal FC",
            "away_team": "Chelsea FC",
            "matchday": 12,
            "status": "TIMED",
        },
        {
            "match_id": 8,
            "kickoff_utc": "2026-12-01T15:00:00Z",
            "home_team": "Arsenal FC",
            "away_team": "Chelsea FC",
            "matchday": 13,
            "status": "TIMED",
        },
    ]
    monkeypatch.setattr("src.jobs.publish.fetch_pl_matches", lambda *a, **k: fixtures)
    monkeypatch.setattr(
        "src.jobs.publish.load_team_name_map",
        lambda: {"Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea"},
    )
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)

    first = publish_slate(version, now_utc=now, token="x")
    assert first == {
        "fetched": 2,
        "in_window": 1,
        "inserted": 1,
        "skipped_duplicate": 0,
    }

    second = publish_slate(version, now_utc=now, token="x")
    assert second["inserted"] == 0
    assert second["skipped_duplicate"] == 1
    assert _prediction_count(db) == 1

    public = published_model_version(version)
    conn = connect(db)
    try:
        row = get_prediction(conn, 7, public)
    finally:
        conn.close()
    assert row is not None
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Chelsea"
    assert row["kickoff_utc"] == "2026-11-20T15:00:00Z"
    assert row["matchday"] == 12
    assert row["competition"] == "PL"
    assert row["status"] == "scheduled"
    assert row["as_of_date"] == "2026-11-16"
    assert row["book_p_home"] is None
    assert row["book_source"] is None

    table_markets = markets_from_score_table(score_table(model, "Arsenal", "Chelsea"))
    assert row["p_home"] == pytest.approx(table_markets["home"])
    assert row["p_draw"] == pytest.approx(table_markets["draw"])
    assert row["p_away"] == pytest.approx(table_markets["away"])
    assert row["p_btts_yes"] == pytest.approx(table_markets["btts_yes"])
    assert row["p_over_25"] == pytest.approx(table_markets["over_25"])
    frame = pd.DataFrame(
        [{"match_id": 7, "home_team": "Arsenal", "away_team": "Chelsea"}]
    )
    scaled = predict_matches(model, frame, temperature=1.4).iloc[0]
    assert row["p_home"] != pytest.approx(float(scaled["p_home"]))
    lines = read_jsonl(ledger / "predictions.jsonl")
    assert len(lines) == 1
    assert lines[0]["model_version"] == public
    assert lines[0]["score_omitted_reason"] is None


def test_publish_aborts_before_insert_when_any_team_is_unmapped(tmp_path, monkeypatch):
    models_dir, db, _ledger = _patch_paths(monkeypatch, tmp_path)
    version, _ = save_artefact(_model(), 1.0, {"xi": 0.0019}, models_dir)
    monkeypatch.setattr(
        "src.jobs.publish.fetch_pl_matches",
        lambda *a, **k: [
            {
                "match_id": 1,
                "kickoff_utc": "2026-11-20T15:00:00Z",
                "home_team": "Arsenal FC",
                "away_team": "Chelsea FC",
                "matchday": 1,
                "status": "TIMED",
            },
            {
                "match_id": 2,
                "kickoff_utc": "2026-11-21T15:00:00Z",
                "home_team": "Not A Real Team FC",
                "away_team": "Chelsea FC",
                "matchday": 1,
                "status": "TIMED",
            },
        ],
    )
    monkeypatch.setattr(
        "src.jobs.publish.load_team_name_map",
        lambda: {"Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea"},
    )

    with pytest.raises(KeyError, match="Unmapped team"):
        publish_slate(
            version,
            now_utc=datetime(2026, 11, 16, tzinfo=timezone.utc),
            token="x",
        )
    assert _prediction_count(db) == 0


def test_publish_flags_league_average_when_history_is_missing(tmp_path, monkeypatch):
    models_dir, db, ledger = _patch_paths(monkeypatch, tmp_path)
    model = _model(
        attack={"Arsenal": 1.2, "Chelsea": 0.9, "Liverpool": 1.1},
        defence={"Arsenal": 0.8, "Liverpool": 1.0},
    )
    version, _ = save_artefact(model, 1.0, {"xi": 0.0019}, models_dir)
    monkeypatch.setattr(
        "src.jobs.publish.fetch_pl_matches",
        lambda *a, **k: [
            {
                "match_id": 1,
                "kickoff_utc": "2026-11-20T15:00:00Z",
                "home_team": "Liverpool FC",
                "away_team": "Arsenal FC",
                "matchday": 1,
                "status": "TIMED",
            },
            {
                "match_id": 2,
                "kickoff_utc": "2026-11-21T15:00:00Z",
                "home_team": "Chelsea FC",
                "away_team": "Arsenal FC",
                "matchday": 1,
                "status": "TIMED",
            },
        ],
    )
    monkeypatch.setattr(
        "src.jobs.publish.load_team_name_map",
        lambda: {
            "Arsenal FC": "Arsenal",
            "Chelsea FC": "Chelsea",
            "Liverpool FC": "Liverpool",
        },
    )

    counts = publish_slate(
        version,
        now_utc=datetime(2026, 11, 16, tzinfo=timezone.utc),
        token="x",
    )
    assert counts["inserted"] == 2
    assert _prediction_count(db) == 2
    lines = {row["match_id"]: row for row in read_jsonl(ledger / "predictions.jsonl")}
    assert lines[1]["uses_league_average"] is False
    assert lines[1]["league_average_teams"] == []
    assert lines[2]["uses_league_average"] is True
    assert lines[2]["league_average_teams"] == ["Chelsea"]
    # Chelsea has an attack rate and no defence rate, so only the away mean uses 1.0.
    assert lines[2]["lambda_home"] == pytest.approx(0.9 * 0.8 * 1.3)
    assert lines[2]["lambda_away"] == pytest.approx(1.2)


def test_publish_requests_exclusive_date_to(tmp_path, monkeypatch):
    models_dir, _db, _ledger = _patch_paths(monkeypatch, tmp_path)
    version, _ = save_artefact(_model(), 1.0, {"xi": 0.0019}, models_dir)
    captured = {}

    def fake_fetch(date_from, date_to, **kwargs):
        captured["date_from"] = date_from
        captured["date_to"] = date_to
        captured["token"] = kwargs.get("token")
        captured["use_cache"] = kwargs.get("use_cache")
        return []

    monkeypatch.setattr("src.jobs.publish.fetch_pl_matches", fake_fetch)
    now = datetime(2026, 11, 16, 18, tzinfo=timezone.utc)
    counts = publish_slate(version, now_utc=now, token="tok")

    assert captured == {
        "date_from": "2026-11-16",
        "date_to": "2026-11-24",
        "token": "tok",
        "use_cache": False,
    }
    assert counts == {
        "fetched": 0,
        "in_window": 0,
        "inserted": 0,
        "skipped_duplicate": 0,
    }


def _raw_match(match_id, kickoff):
    return {
        "id": match_id,
        "utcDate": kickoff,
        "status": "TIMED",
        "matchday": 12,
        "homeTeam": {"name": "Arsenal FC"},
        "awayTeam": {"name": "Chelsea FC"},
    }


def test_publish_refetches_same_window_after_cache_write(tmp_path, monkeypatch):
    """A fixture added after the first publish must be fetched again.

    The cache directory is fixed. The first pull writes it. Without skipping
    the cache read, the second publish would reuse that file and miss the new
    match.
    """
    models_dir, db, _ledger = _patch_paths(monkeypatch, tmp_path)
    version, _ = save_artefact(_model(), 1.0, {"xi": 0.0019}, models_dir)
    cache_dir = tmp_path / "fixtures"
    cache_dir.mkdir()
    monkeypatch.setattr("src.fixtures.client.FIXTURES_CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        "src.jobs.publish.load_team_name_map",
        lambda: {"Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea"},
    )
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    first_body = {"matches": [_raw_match(7, "2026-11-20T15:00:00Z")]}
    second_body = {
        "matches": [
            _raw_match(7, "2026-11-20T15:00:00Z"),
            _raw_match(8, "2026-11-21T17:30:00Z"),
        ]
    }
    payloads = [first_body, second_body]
    captured = {"calls": 0}

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured["calls"] += 1
        return FakeResponse(payloads[captured["calls"] - 1])

    monkeypatch.setattr("src.fixtures.client.requests.get", fake_get)

    first = publish_slate(version, now_utc=now, token="secret-token")
    cache_file = cache_dir / "PL_2026-11-16_2026-11-24.json"
    assert first["inserted"] == 1
    assert captured["calls"] == 1
    assert cache_file.is_file()

    second = publish_slate(version, now_utc=now, token="secret-token")

    assert captured["calls"] == 2
    assert second == {
        "fetched": 2,
        "in_window": 2,
        "inserted": 1,
        "skipped_duplicate": 1,
    }
    assert _prediction_count(db) == 2
    public = published_model_version(version)
    conn = connect(db)
    try:
        original = get_prediction(conn, 7, public)
        added = get_prediction(conn, 8, public)
    finally:
        conn.close()
    assert original["status"] == "scheduled"
    assert added is not None
    assert added["kickoff_utc"] == "2026-11-21T17:30:00Z"
    assert added["home_team"] == "Arsenal"
    written = json.loads(cache_file.read_text(encoding="utf-8"))
    assert [match["id"] for match in written["matches"]] == [7, 8]


def test_publish_fills_books_from_fixture_csv(tmp_path, monkeypatch):
    models_dir, db, ledger = _patch_paths(monkeypatch, tmp_path)
    model = _model()
    version, _ = save_artefact(model, 1.4, {"xi": 0.0019}, models_dir)
    csv_path = Path(__file__).parent / "fixtures" / "fdco_upcoming.csv"
    monkeypatch.setattr(
        "src.jobs.publish.load_book_index",
        lambda: index_fixtures_csv(csv_path),
    )
    monkeypatch.setattr(
        "src.jobs.publish.fetch_pl_matches",
        lambda *a, **k: [
            {
                "match_id": 7,
                "kickoff_utc": "2026-11-20T15:00:00Z",
                "home_team": "Arsenal FC",
                "away_team": "Chelsea FC",
                "matchday": 12,
                "status": "TIMED",
            }
        ],
    )
    monkeypatch.setattr(
        "src.jobs.publish.load_team_name_map",
        lambda: {"Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea"},
    )

    counts = publish_slate(
        version,
        now_utc=datetime(2026, 11, 16, tzinfo=timezone.utc),
        token="x",
    )
    assert counts["inserted"] == 2

    public = published_model_version(version)
    conn = connect(db)
    try:
        row = get_prediction(conn, 7, public)
    finally:
        conn.close()
    raw_home, raw_draw, raw_away = 1 / 2.0, 1 / 3.4, 1 / 4.0
    total = raw_home + raw_draw + raw_away
    raw_over, raw_under = 1 / 1.80, 1 / 2.10
    ou_total = raw_over + raw_under
    assert row["book_p_home"] == pytest.approx(raw_home / total)
    assert row["book_p_draw"] == pytest.approx(raw_draw / total)
    assert row["book_p_away"] == pytest.approx(raw_away / total)
    assert row["book_p_over_25"] == pytest.approx(raw_over / ou_total)
    assert row["book_p_under_25"] == pytest.approx(raw_under / ou_total)
    assert row["book_source"] == BOOK_SOURCE
    table_markets = markets_from_score_table(score_table(model, "Arsenal", "Chelsea"))
    assert row["p_home"] == pytest.approx(table_markets["home"])
    frame = pd.DataFrame(
        [{"match_id": 7, "home_team": "Arsenal", "away_team": "Chelsea"}]
    )
    scaled = predict_matches(model, frame, temperature=1.4).iloc[0]
    assert row["p_home"] != pytest.approx(float(scaled["p_home"]))
    assert row["p_home"] != pytest.approx(row["book_p_home"])
    books = read_jsonl(ledger / "book_lines.jsonl")
    assert len(books) == 1
    assert books[0]["kind"] == "opening"
    assert books[0]["forecast_line"] == "model"
    assert books[0]["p_home"] == pytest.approx(row["book_p_home"])

    predictions = read_jsonl(ledger / "predictions.jsonl")
    assert [item["forecast_line"] for item in predictions] == ["model", FORECAST_LINE_MARKET]
    blend = predictions[1]
    table = published_blend_table(
        model,
        "Arsenal",
        "Chelsea",
        row["book_p_home"],
        row["book_p_draw"],
        row["book_p_away"],
        row["book_p_over_25"],
        0.25,
    )
    blend_markets = markets_from_score_table(table)
    assert blend["p_home"] == pytest.approx(blend_markets["home"])
    assert blend["p_draw"] == pytest.approx(blend_markets["draw"])
    assert blend["p_away"] == pytest.approx(blend_markets["away"])
    assert blend["p_home"] != pytest.approx(row["p_home"])
    assert blend["rho"] == pytest.approx(0.0)
    assert blend["blend_weight"] == pytest.approx(0.25)
    assert blend["line_label"] == MARKET_LINE_LABEL
    assert blend["model_version"] == f"{public}+market"
    again = publish_slate(
        version,
        now_utc=datetime(2026, 11, 16, tzinfo=timezone.utc),
        token="x",
    )
    assert again["inserted"] == 0
    assert again["skipped_duplicate"] == 2


def test_publish_omits_the_blend_when_the_over_price_is_missing(tmp_path, monkeypatch):
    models_dir, _db, ledger = _patch_paths(monkeypatch, tmp_path)
    version, _ = save_artefact(_model(), 1.4, {"xi": 0.0019}, models_dir)
    monkeypatch.setattr(
        "src.jobs.publish.lookup_book",
        lambda *a, **k: {
            "book_p_home": 0.45,
            "book_p_draw": 0.28,
            "book_p_away": 0.27,
            "book_p_over_25": None,
            "book_p_under_25": None,
            "book_source": "test",
        },
    )
    monkeypatch.setattr(
        "src.jobs.publish.fetch_pl_matches",
        lambda *a, **k: [
            {
                "match_id": 7,
                "kickoff_utc": "2026-11-20T15:00:00Z",
                "home_team": "Arsenal FC",
                "away_team": "Chelsea FC",
                "matchday": 12,
                "status": "TIMED",
            }
        ],
    )
    monkeypatch.setattr(
        "src.jobs.publish.load_team_name_map",
        lambda: {"Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea"},
    )
    counts = publish_slate(
        version,
        now_utc=datetime(2026, 11, 16, tzinfo=timezone.utc),
        token="x",
    )
    assert counts["inserted"] == 1
    lines = read_jsonl(ledger / "predictions.jsonl")
    assert [row["forecast_line"] for row in lines] == ["model"]


def test_site_export_folds_the_blend_onto_the_model_card():
    kickoff = "2026-11-20T15:00:00Z"
    model = {
        "match_id": 7,
        "forecast_line": "model",
        "matchday": 12,
        "kickoff_utc": kickoff,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "model_version": "dc-grid",
        "most_likely_score": {"home": 1, "away": 1, "probability": 0.1},
        "next_scores": [],
        "p_home": 0.4,
        "p_draw": 0.3,
        "p_away": 0.3,
        "p_btts_yes": 0.5,
        "p_btts_no": 0.5,
        "p_over_25": 0.5,
        "p_under_25": 0.5,
        "totals": {},
        "lambda_home": 1.2,
        "lambda_away": 1.1,
        "uses_league_average": False,
        "league_average_teams": [],
        "score_omitted_reason": None,
    }
    blend = {
        **model,
        "forecast_line": "model_plus_market",
        "model_version": "dc-grid+market",
        "p_home": 0.46,
        "p_draw": 0.27,
        "p_away": 0.27,
        "blend_weight": 0.25,
    }
    settlement = {
        "match_id": 7,
        "forecast_line": "model",
        "kind": "result",
        "home_goals": 1,
        "away_goals": 0,
        "result": "H",
        "kickoff_utc": kickoff,
    }
    documents = build_site_files(
        [model, blend],
        [settlement],
        [],
        now_utc=datetime(2026, 11, 21, tzinfo=timezone.utc),
    )
    assert documents["ledger-upcoming.json"]["matchdays"] == []
    settled = documents["ledger-settled.json"]["matchdays"][0]["matches"]
    assert len(settled) == 1
    assert settled[0]["p_home"] == 0.4
    assert settled[0]["model_plus_market"]["label"] == "model plus market"
    assert settled[0]["model_plus_market"]["p_home"] == 0.46
    assert settled[0]["model_plus_market"]["blend_weight"] == 0.25
    assert documents["ledger-scorecard.json"]["counts"]["settled"] == 1


def test_main_logs_publish_counts(monkeypatch, caplog):
    monkeypatch.setattr(
        "src.jobs.publish.publish_slate",
        lambda model_version=None, **kwargs: {
            "fetched": 3,
            "in_window": 2,
            "inserted": 2,
            "skipped_duplicate": 0,
        },
    )
    with caplog.at_level("INFO", logger="src.jobs.publish"):
        main()
    assert "inserted=2" in caplog.text
    assert "in_window=2" in caplog.text
