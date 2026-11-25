"""Acceptance checks for the daily ledger command."""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.jobs.daily import main, run_daily
from src.ledger.log import read_jsonl
from src.store.predictions import connect, upsert_model_artefact
from src.store.schema import init_db

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _prediction(**overrides) -> dict:
    row = {
        "match_id": 1,
        "matchday": 12,
        "kickoff_utc": "2026-11-20T15:00:00Z",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "competition": "PL",
        "forecast_line": "model",
        "model_version": "dc-test-grid",
        "artefact_version": "dc-test",
        "published_at": "2026-11-16T09:00:00Z",
        "as_of_date": "2026-11-16",
        "lambda_home": 1.4,
        "lambda_away": 1.1,
        "rho": -0.03,
        "score_table": None,
        "most_likely_score": {"home": 1, "away": 1, "probability": 0.12},
        "next_scores": [],
        "p_home": 0.45,
        "p_draw": 0.28,
        "p_away": 0.27,
        "p_btts_yes": 0.5,
        "p_btts_no": 0.5,
        "p_over_25": 0.52,
        "p_under_25": 0.48,
        "totals": None,
        "league_average_teams": [],
        "uses_league_average": False,
        "score_omitted_reason": None,
    }
    row.update(overrides)
    return row


def _match(match_id, status, kickoff, home=None, away=None):
    return {
        "match_id": match_id,
        "status": status,
        "kickoff_utc": kickoff,
        "home_team": "Arsenal FC",
        "away_team": "Chelsea FC",
        "home_goals": home,
        "away_goals": away,
        "provider_updated_at": "2026-11-24T00:20:27Z",
        "matchday": 12,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _prepare(tmp_path: Path, predictions: list[dict]) -> Path:
    ledger = tmp_path / "data" / "ledger"
    _write_jsonl(ledger / "predictions.jsonl", predictions)
    (tmp_path / "web" / "src" / "data").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    return tmp_path


def _run(repo: Path, **kwargs):
    kwargs.setdefault("book_index", {})
    kwargs.setdefault("season_frame", pd.DataFrame())
    return run_daily(
        now_utc=NOW,
        ledger_dir=repo / "data" / "ledger",
        export_dir=repo / "web" / "src" / "data",
        models_dir=repo / "models",
        db_path=repo / "data" / "processed" / "predictions.sqlite",
        repo_root=repo,
        **kwargs,
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    found = {}
    for path in root.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            found[path.relative_to(root).as_posix()] = path.read_bytes()
    return found


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def test_dry_run_prints_counts_and_changes_no_file(tmp_path: Path, capsys):
    repo = _prepare(tmp_path, [_prediction()])
    before = _snapshot(repo)
    counts = _run(
        repo,
        dry_run=True,
        fixtures=[_match(1, "FINISHED", "2026-11-20T15:00:00Z", 2, 1)],
    )
    assert counts["settled"] == 1
    assert counts["added"] == 0
    assert _snapshot(repo) == before
    printed = capsys.readouterr().out
    assert "dry-run" in printed
    assert "settled=1" in printed
    assert "added=0" in printed
    assert "price-filled=0" in printed


def test_two_runs_with_no_new_data_do_not_commit(tmp_path: Path):
    repo = _prepare(tmp_path, [_prediction()])
    _git(repo, "init")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")
    predictions = (repo / "data" / "ledger" / "predictions.jsonl").read_bytes()
    fixture = [_match(1, "FINISHED", "2026-11-20T15:00:00Z", 2, 1)]
    first = _run(repo, commit=True, fixtures=fixture)
    assert first["committed"] is True
    assert first["settled"] == 1
    committed = _head(repo)
    message = _git(repo, "log", "-1", "--format=%s")
    assert message == "Ledger: added 0, settled 1, price-filled 0"
    assert (repo / "data" / "ledger" / "predictions.jsonl").read_bytes() == predictions
    second = _run(repo, commit=True, fixtures=fixture)
    assert second["settled"] == 0
    assert second["committed"] is False
    assert _head(repo) == committed
    assert (repo / "data" / "ledger" / "predictions.jsonl").read_bytes() == predictions


def test_one_settlement_line_and_a_later_correction(tmp_path: Path):
    repo = _prepare(tmp_path, [_prediction()])
    before = (repo / "data" / "ledger" / "predictions.jsonl").read_bytes()
    _run(repo, fixtures=[_match(1, "FINISHED", "2026-11-20T15:00:00Z", 2, 1)])
    rows = read_jsonl(repo / "data" / "ledger" / "settlements.jsonl")
    assert len(rows) == 1
    assert rows[0]["kind"] == "result"
    assert rows[0]["home_goals"] == 2
    assert rows[0]["provider_updated_at"] == "2026-11-24T00:20:27Z"
    original = (repo / "data" / "ledger" / "settlements.jsonl").read_bytes().splitlines()[0]
    _run(repo, fixtures=[_match(1, "FINISHED", "2026-11-20T15:00:00Z", 2, 1)])
    assert len(read_jsonl(repo / "data" / "ledger" / "settlements.jsonl")) == 1
    _run(repo, fixtures=[_match(1, "FINISHED", "2026-11-20T15:00:00Z", 3, 1)])
    updated = read_jsonl(repo / "data" / "ledger" / "settlements.jsonl")
    assert [row["kind"] for row in updated] == ["result", "correction"]
    assert updated[0]["home_goals"] == 2
    assert updated[1]["home_goals"] == 3
    assert (repo / "data" / "ledger" / "settlements.jsonl").read_bytes().splitlines()[0] == original
    assert (repo / "data" / "ledger" / "predictions.jsonl").read_bytes() == before


def test_awarded_result_records_the_provider_update_time(tmp_path: Path):
    repo = _prepare(tmp_path, [_prediction()])
    _run(repo, fixtures=[_match(1, "AWARDED", "2026-11-20T15:00:00Z", 3, 0)])
    row = read_jsonl(repo / "data" / "ledger" / "settlements.jsonl")[0]
    assert row["kind"] == "result"
    assert row["provider_status"] == "AWARDED"
    assert row["provider_updated_at"] == "2026-11-24T00:20:27Z"
    assert row["home_goals"] == 3
    assert row["result"] == "H"


def test_postponed_match_stays_listed_and_out_of_averages(tmp_path: Path):
    finished = _prediction(match_id=1)
    postponed = _prediction(match_id=2, kickoff_utc="2026-11-28T15:00:00Z")
    repo = _prepare(tmp_path, [finished, postponed])
    before = (repo / "data" / "ledger" / "predictions.jsonl").read_bytes()
    _run(
        repo,
        fixtures=[
            _match(1, "FINISHED", "2026-11-20T15:00:00Z", 2, 1),
            _match(2, "POSTPONED", "2026-11-28T15:00:00Z"),
        ],
    )
    assert (repo / "data" / "ledger" / "predictions.jsonl").read_bytes() == before
    listed = json.loads((repo / "web" / "src" / "data" / "ledger-settled.json").read_text())
    listed_ids = [match["match_id"] for day in listed["matchdays"] for match in day["matches"]]
    assert 2 in listed_ids
    upcoming = json.loads((repo / "web" / "src" / "data" / "ledger-upcoming.json").read_text())
    upcoming_ids = [match["match_id"] for day in upcoming["matchdays"] for match in day["matches"]]
    assert 2 not in upcoming_ids
    card = json.loads((repo / "web" / "src" / "data" / "ledger-scorecard.json").read_text())
    assert card["counts"]["postponed"] == 1
    assert card["counts"]["settled"] == 1
    assert card["home_draw_away"]["match_count"] == 1
    assert card["exact_score"]["match_count"] == 1
    assert card["goal_error"]["match_count"] == 1
    assert card["book_comparison"]["match_count"] == 0


def test_monday_match_stays_with_its_matchday(tmp_path: Path):
    rows = [
        _prediction(match_id=10, kickoff_utc="2026-11-28T15:00:00Z"),
        _prediction(match_id=11, kickoff_utc="2026-11-30T20:00:00Z"),
    ]
    repo = _prepare(tmp_path, rows)
    _run(repo, fixtures=[])
    upcoming = json.loads((repo / "web" / "src" / "data" / "ledger-upcoming.json").read_text())
    assert [day["matchday"] for day in upcoming["matchdays"]] == [12]
    assert [match["match_id"] for match in upcoming["matchdays"][0]["matches"]] == [10, 11]
    assert "iso_week" not in upcoming


def test_unknown_team_writes_nothing(tmp_path: Path):
    repo = _prepare(tmp_path, [_prediction()])
    before = _snapshot(repo)
    fixture = _match(99, "TIMED", "2026-11-28T15:00:00Z")
    fixture["home_team"] = "Nope FC"
    with pytest.raises(KeyError, match="Unmapped team"):
        _run(repo, fixtures=[fixture])
    assert _snapshot(repo) == before


def test_api_error_writes_nothing(tmp_path: Path, monkeypatch):
    repo = _prepare(tmp_path, [_prediction()])
    before = _snapshot(repo)

    def boom(*args, **kwargs):
        raise requests.HTTPError("fixtures failed")

    monkeypatch.setattr("src.jobs.daily.fetch_pl_matches", boom)
    with pytest.raises(requests.HTTPError):
        _run(repo)
    assert _snapshot(repo) == before


def test_main_logs_an_alert(monkeypatch, caplog):
    def boom(**kwargs):
        raise KeyError("Unmapped team: 'Nope FC'")

    monkeypatch.setattr("src.jobs.daily.run_daily", boom)
    with caplog.at_level(logging.ERROR, logger="src.jobs.daily"):
        with pytest.raises(KeyError):
            main([])
    assert "ALERT: daily ledger failed" in caplog.text


def test_refit_runs_only_after_a_newer_result(tmp_path: Path, monkeypatch):
    repo = _prepare(tmp_path, [_prediction()])
    model_dir = repo / "models" / "dc-old"
    model_dir.mkdir()
    called = []

    def fake_fit(*args, **kwargs):
        called.append(True)
        return "dc-new", model_dir

    monkeypatch.setattr("src.jobs.daily.fit_and_save", fake_fit)
    monkeypatch.setattr("src.jobs.daily.load_training_matches", lambda: pd.DataFrame())
    (model_dir / "model.json").write_text(json.dumps({"as_of": "2026-11-23T00:00:00"}), encoding="utf-8")
    _run(repo, fixtures=[_match(1, "FINISHED", "2026-11-20T15:00:00Z", 1, 0)])
    assert called == []

    (model_dir / "model.json").write_text(json.dumps({"as_of": "2026-11-01T00:00:00"}), encoding="utf-8")
    _run(repo, fixtures=[_match(1, "FINISHED", "2026-11-20T15:00:00Z", 1, 0)])
    assert called == [True]


def test_season_file_writes_labelled_opening_and_closing_prices(tmp_path: Path):
    repo = _prepare(tmp_path, [_prediction()])
    season = pd.DataFrame(
        [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "date": "2026-11-20",
                "implied_home_open": 0.50,
                "implied_draw_open": 0.30,
                "implied_away_open": 0.20,
                "implied_home": 0.40,
                "implied_draw": 0.30,
                "implied_away": 0.30,
            }
        ]
    )
    counts = _run(
        repo,
        season_frame=season,
        fixtures=[_match(1, "FINISHED", "2026-11-20T15:00:00Z", 2, 1)],
    )
    assert counts["price_filled"] == 2
    lines = read_jsonl(repo / "data" / "ledger" / "book_lines.jsonl")
    assert {line["kind"] for line in lines} == {"opening", "closing"}
    assert all("season file" in line["source"] for line in lines)
    again = _run(
        repo,
        season_frame=season,
        fixtures=[_match(1, "FINISHED", "2026-11-20T15:00:00Z", 2, 1)],
    )
    assert again["price_filled"] == 0
    assert len(read_jsonl(repo / "data" / "ledger" / "book_lines.jsonl")) == 2


def test_prematch_book_is_captured_once(tmp_path: Path):
    repo = _prepare(tmp_path, [_prediction(kickoff_utc="2026-11-28T15:00:00Z")])
    index = {
        ("Arsenal", "Chelsea", "2026-11-28"): {
            "book_p_home": 0.5,
            "book_p_draw": 0.3,
            "book_p_away": 0.2,
            "book_p_over_25": None,
            "book_p_under_25": None,
            "book_source": "football-data.co.uk fixtures.csv (B365)",
        }
    }
    first = _run(repo, book_index=index, fixtures=[])
    second = _run(repo, book_index=index, fixtures=[])
    lines = read_jsonl(repo / "data" / "ledger" / "book_lines.jsonl")
    assert first["price_filled"] == 1
    assert second["price_filled"] == 0
    assert len(lines) == 1
    assert lines[0]["kind"] == "opening"


def test_new_fixture_is_handed_to_publish(tmp_path: Path, monkeypatch):
    repo = _prepare(tmp_path, [])
    model_dir = repo / "models" / "dc-old"
    model_dir.mkdir()
    (model_dir / "model.json").write_text(json.dumps({"as_of": "2026-11-23T00:00:00"}), encoding="utf-8")
    seen = []

    def fake_publish(**kwargs):
        seen.append(kwargs["matches"])
        return {"inserted": 0, "skipped_duplicate": 0, "fetched": 1, "in_window": 1}

    monkeypatch.setattr("src.jobs.daily.publish_slate", fake_publish)
    _run(repo, fixtures=[_match(50, "TIMED", "2026-11-28T15:00:00Z")])
    assert [match["match_id"] for match in seen[0]] == [50]


def test_workflow_is_scheduled_manual_and_alerts():
    text = (ROOT / ".github" / "workflows" / "daily-ledger.yml").read_text(encoding="utf-8")
    head, _jobs = text.split("jobs:", 1)
    assert "push:" not in head
    assert 'cron: "0 6 * * *"' in text
    assert "workflow_dispatch" in text
    assert "dry_run:" in text
    assert "contents: write" in text
    assert "issues: write" in text
    assert "group: ledger" in text
    assert "cancel-in-progress: false" in text
    assert "python -m src.jobs.daily --dry-run" in text
    assert "python -m src.jobs.daily --commit" in text
    assert "Daily ledger failed" in text


def test_seed_refuses_a_real_store(tmp_path: Path, monkeypatch, capsys):
    import scripts.seed_sample_predictions as seed

    db = tmp_path / "predictions.sqlite"
    monkeypatch.setattr("src.store.schema.PREDICTIONS_DB", db)
    init_db(db)
    conn = connect(db)
    upsert_model_artefact(
        conn,
        "dc-real",
        "2026-11-23",
        "models/dc-real",
        {},
        created_at="2026-11-23T00:00:00Z",
    )
    conn.close()
    with pytest.raises(SystemExit):
        seed.main()
    conn = connect(db)
    versions = [row[0] for row in conn.execute("SELECT model_version FROM model_artefacts")]
    predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    assert versions == ["dc-real"]
    assert predictions == 0
    assert "refusing to seed" in capsys.readouterr().err
