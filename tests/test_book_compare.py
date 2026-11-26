"""The published model line stores the reconstructed book, once."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.eval.book_grid import BOOK_SCORE_TABLE_LABEL, fit_book_grid
from src.fixtures.books import index_fixtures_csv
from src.jobs.publish import publish_slate
from src.ledger.book_compare import publish_comparison, scoreline_lines_for_settled
from src.ledger.export_site import build_site_files
from src.ledger.log import COMPARISONS_NAME, append_book_comparisons, read_jsonl
from src.models.artefacts import save_artefact
from src.models.dixon_coles import scoreline_grid_from_rates
from src.models.markets import markets_from_grid
from tests.test_daily import _match, _prediction, _prepare, _run
from tests.test_publish import _model, _patch_paths

_WHEN = datetime(2026, 11, 16, tzinfo=timezone.utc)
_FIXTURE = {
    "match_id": 7,
    "kickoff_utc": "2026-11-20T15:00:00Z",
    "home_team": "Arsenal FC",
    "away_team": "Chelsea FC",
    "matchday": 12,
    "status": "TIMED",
}
_NAMES = {"Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea"}


def _publish(tmp_path, monkeypatch, quote):
    models_dir, _db, ledger = _patch_paths(monkeypatch, tmp_path)
    version, _ = save_artefact(_model(), 1.4, {"xi": 0.0019}, models_dir)
    monkeypatch.setattr("src.jobs.publish.lookup_book", lambda *a, **k: quote)
    monkeypatch.setattr("src.jobs.publish.fetch_pl_matches", lambda *a, **k: [_FIXTURE])
    monkeypatch.setattr("src.jobs.publish.load_team_name_map", lambda: _NAMES)
    publish_slate(version, now_utc=_WHEN, token="x")
    return ledger


def _quote():
    return {
        "book_p_home": 0.46,
        "book_p_draw": 0.27,
        "book_p_away": 0.27,
        "book_p_over_25": 0.54,
        "book_p_under_25": 0.46,
        "book_source": "test",
    }


def test_stored_markets_match_the_fitter(tmp_path, monkeypatch):
    ledger = _publish(tmp_path, monkeypatch, _quote())
    rows = read_jsonl(ledger / COMPARISONS_NAME)
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == BOOK_SCORE_TABLE_LABEL
    assert row["kind"] == "publish"
    assert row["forecast_line"] == "model"
    assert row["missing_reason"] is None
    assert row["model_scoreline_log_loss"] is None
    assert row["book_scoreline_log_loss"] is None
    assert row["scoreline_log_loss_gap"] is None
    fitted = fit_book_grid(row["price_home"], row["price_draw"], row["price_away"], row["price_over_25"])
    assert row["book_p_home"] == pytest.approx(fitted.home, abs=1e-6)
    assert row["book_p_draw"] == pytest.approx(fitted.draw, abs=1e-6)
    assert row["book_p_away"] == pytest.approx(fitted.away, abs=1e-6)
    assert row["book_p_over_25"] == pytest.approx(fitted.over_25, abs=1e-6)
    assert row["book_p_under_25"] == pytest.approx(fitted.under_25, abs=1e-6)
    assert row["book_lambda"] == pytest.approx(fitted.lam, abs=1e-6)
    assert row["book_mu"] == pytest.approx(fitted.mu, abs=1e-6)
    rebuilt = markets_from_grid(scoreline_grid_from_rates(row["book_lambda"], row["book_mu"], 0.0))
    assert rebuilt["home"] == pytest.approx(fitted.home, abs=1e-6)
    assert rebuilt["over_25"] == pytest.approx(fitted.over_25, abs=1e-6)
    assert row["gap_home"] == pytest.approx(row["model_p_home"] - row["book_p_home"], abs=1e-12)
    assert isinstance(row["gap_home"], float)


def test_missing_price_does_not_create_a_gap(tmp_path, monkeypatch):
    ledger = _publish(
        tmp_path,
        monkeypatch,
        {
            "book_p_home": 0.45,
            "book_p_draw": 0.28,
            "book_p_away": 0.27,
            "book_p_over_25": None,
            "book_p_under_25": None,
            "book_source": "test",
        },
    )
    rows = read_jsonl(ledger / COMPARISONS_NAME)
    assert len(rows) == 1
    row = rows[0]
    assert row["missing_reason"] == "missing pre-kickoff price: over/under 2.5"
    for key in (
        "book_lambda",
        "book_mu",
        "book_most_likely_score",
        "book_p_home",
        "book_p_draw",
        "book_p_away",
        "book_p_over_25",
        "book_p_under_25",
        "gap_home",
        "gap_draw",
        "gap_away",
        "gap_over_25",
        "gap_under_25",
        "scoreline_log_loss_gap",
    ):
        assert row[key] is None
    predictions = read_jsonl(ledger / "predictions.jsonl")
    documents = build_site_files(
        predictions,
        [
            {
                "match_id": 7,
                "forecast_line": "model",
                "kind": "result",
                "home_goals": 1,
                "away_goals": 0,
                "result": "H",
                "kickoff_utc": _FIXTURE["kickoff_utc"],
            }
        ],
        [],
        now_utc=datetime(2026, 11, 21, tzinfo=timezone.utc),
        comparisons=rows,
    )
    summary = documents["ledger-scorecard.json"]["book_reconstruction"]
    assert summary["match_count"] == 0
    assert summary["mean_gap"] is None
    settled = documents["ledger-settled.json"]["matchdays"][0]["matches"][0]
    assert "book_reconstructed_score" not in settled


def test_publish_twice_does_not_duplicate_the_row(tmp_path, monkeypatch):
    models_dir, _db, ledger = _patch_paths(monkeypatch, tmp_path)
    version, _ = save_artefact(_model(), 1.4, {"xi": 0.0019}, models_dir)
    csv_path = Path(__file__).parent / "fixtures" / "fdco_upcoming.csv"
    monkeypatch.setattr("src.jobs.publish.load_book_index", lambda: index_fixtures_csv(csv_path))
    monkeypatch.setattr("src.jobs.publish.fetch_pl_matches", lambda *a, **k: [_FIXTURE])
    monkeypatch.setattr("src.jobs.publish.load_team_name_map", lambda: _NAMES)
    first = publish_slate(version, now_utc=_WHEN, token="x")
    assert first["inserted"] == 2
    stored = (ledger / COMPARISONS_NAME).read_bytes()
    predictions = (ledger / "predictions.jsonl").read_bytes()
    second = publish_slate(version, now_utc=_WHEN, token="x")
    assert second["inserted"] == 0
    assert second["skipped_duplicate"] == 2
    assert read_jsonl(ledger / COMPARISONS_NAME) == read_jsonl_bytes(stored)
    assert (ledger / COMPARISONS_NAME).read_bytes() == stored
    assert (ledger / "predictions.jsonl").read_bytes() == predictions
    assert len(read_jsonl(ledger / COMPARISONS_NAME)) == 1


def test_settlement_appends_scoreline_loss_and_leaves_the_prediction(tmp_path, monkeypatch):
    ledger = _publish(tmp_path, monkeypatch, _quote())
    predictions_bytes = (ledger / "predictions.jsonl").read_bytes()
    publish_bytes = (ledger / COMPARISONS_NAME).read_bytes()
    predictions = read_jsonl(ledger / "predictions.jsonl")
    model_line = next(row for row in predictions if row["forecast_line"] == "model")
    settlement = {
        "match_id": 7,
        "forecast_line": "model",
        "kind": "result",
        "home_goals": 2,
        "away_goals": 1,
        "result": "H",
        "kickoff_utc": _FIXTURE["kickoff_utc"],
    }
    fresh = scoreline_lines_for_settled(predictions, [settlement], read_jsonl(ledger / COMPARISONS_NAME))
    assert len(fresh) == 1
    assert fresh[0]["model_scoreline_log_loss"] is not None
    assert fresh[0]["book_scoreline_log_loss"] is not None
    assert fresh[0]["scoreline_log_loss_gap"] == pytest.approx(
        fresh[0]["model_scoreline_log_loss"] - fresh[0]["book_scoreline_log_loss"]
    )
    assert fresh[0]["label"] == BOOK_SCORE_TABLE_LABEL
    written = append_book_comparisons(fresh, path=ledger)
    assert written["inserted"] == 1
    again = append_book_comparisons(fresh, path=ledger)
    assert again["inserted"] == 0
    assert again["skipped_duplicate"] == 1
    assert (ledger / "predictions.jsonl").read_bytes() == predictions_bytes
    assert model_line == next(row for row in read_jsonl(ledger / "predictions.jsonl") if row["forecast_line"] == "model")
    compared = (ledger / COMPARISONS_NAME).read_bytes()
    assert compared.startswith(publish_bytes)
    assert len(read_jsonl(ledger / COMPARISONS_NAME)) == 2

    documents = build_site_files(
        predictions,
        [settlement],
        [],
        now_utc=datetime(2026, 11, 21, tzinfo=timezone.utc),
        comparisons=read_jsonl(ledger / COMPARISONS_NAME),
    )
    summary = documents["ledger-scorecard.json"]["book_reconstruction"]
    assert summary["match_count"] == 1
    assert summary["mean_gap"] == pytest.approx(fresh[0]["scoreline_log_loss_gap"])
    assert summary["label"] == BOOK_SCORE_TABLE_LABEL
    upcoming = documents["ledger-upcoming.json"]["matchdays"]
    assert upcoming == []


def test_upcoming_card_shows_the_reconstructed_score_when_a_price_exists(tmp_path, monkeypatch):
    ledger = _publish(tmp_path, monkeypatch, _quote())
    documents = build_site_files(
        read_jsonl(ledger / "predictions.jsonl"),
        [],
        read_jsonl(ledger / "book_lines.jsonl"),
        now_utc=_WHEN,
        comparisons=read_jsonl(ledger / COMPARISONS_NAME),
    )
    match = documents["ledger-upcoming.json"]["matchdays"][0]["matches"][0]
    score = match["book_reconstructed_score"]
    assert score["label"] == BOOK_SCORE_TABLE_LABEL
    assert isinstance(score["home"], int) and isinstance(score["away"], int)
    assert 0.0 < score["probability"] < 1.0


def test_daily_settle_appends_one_scoreline_line(tmp_path):
    line = _prediction(
        score_table={
            "max_goals": 1,
            "cells": [[0.2, 0.2], [0.4, 0.2]],
            "other": 0.0,
        }
    )
    repo = _prepare(tmp_path, [line])
    comparison = publish_comparison(line, _quote(), recorded_at="2026-11-16T09:00:00Z")
    path = repo / "data" / "ledger" / COMPARISONS_NAME
    path.write_text(json.dumps(comparison, sort_keys=True) + "\n", encoding="utf-8")
    before = (repo / "data" / "ledger" / "predictions.jsonl").read_bytes()
    _run(repo, fixtures=[_match(1, "FINISHED", "2026-11-20T15:00:00Z", 1, 0)])
    rows = read_jsonl(path)
    assert [row["kind"] for row in rows] == ["publish", "scoreline"]
    assert rows[0] == comparison
    assert rows[1]["home_goals"] == 1 and rows[1]["away_goals"] == 0
    assert rows[1]["scoreline_log_loss_gap"] is not None
    assert (repo / "data" / "ledger" / "predictions.jsonl").read_bytes() == before
    card = json.loads((repo / "web" / "src" / "data" / "ledger-scorecard.json").read_text())
    assert card["book_reconstruction"]["match_count"] == 1
    _run(repo, fixtures=[_match(1, "FINISHED", "2026-11-20T15:00:00Z", 1, 0)])
    assert [row["kind"] for row in read_jsonl(path)] == ["publish", "scoreline"]


def read_jsonl_bytes(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def test_no_price_at_all_names_the_reason(tmp_path, monkeypatch):
    ledger = _publish(tmp_path, monkeypatch, None)
    row = read_jsonl(ledger / COMPARISONS_NAME)[0]
    assert row["missing_reason"] == "no pre-kickoff price"
    assert row["gap_home"] is None
    assert row["price_home"] is None
