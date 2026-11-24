"""Ops cadence scripts: freshness, scorecard export, token and ledger gates."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.store.predictions import connect, insert_prediction, settle_prediction, upsert_model_artefact
from src.store.schema import init_db

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script: str, args: list[str], env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )


def _write_snapshot(path: Path, matches: list[dict]) -> None:
    path.write_text(
        json.dumps({"model_version": "dc-test", "matches": matches}),
        encoding="utf-8",
    )


def _match(kickoff: str, status: str = "scheduled") -> dict:
    return {"kickoff_utc": kickoff, "status": status, "match_id": 1}


def test_freshness_counts_remaining_and_aged_rows(tmp_path: Path):
    path = tmp_path / "upcoming-snapshot.json"
    _write_snapshot(
        path,
        [
            _match("2026-11-23T20:00:00Z"),
            _match("2026-11-28T15:00:00Z"),
        ],
    )
    proc = _run(
        "check_snapshot_freshness.py",
        ["--snapshot", str(path), "--now", "2026-11-24T12:00:00Z"],
    )
    assert proc.returncode == 0, proc.stderr
    assert "status=fresh" in proc.stdout
    assert "still_upcoming=1" in proc.stdout
    assert "kicked_off=1" in proc.stdout
    assert "scheduled_after_kickoff=1" in proc.stdout
    assert "latest_kickoff=2026-11-28T15:00:00Z" in proc.stdout


def test_freshness_exits_when_every_kickoff_has_passed(tmp_path: Path):
    path = tmp_path / "upcoming-snapshot.json"
    _write_snapshot(path, [_match("2026-11-23T20:00:00Z")])
    proc = _run(
        "check_snapshot_freshness.py",
        ["--snapshot", str(path), "--now", "2026-11-24T00:00:00Z"],
    )
    assert proc.returncode == 2, proc.stderr
    assert "status=stale" in proc.stdout
    assert "scheduled_after_kickoff=1" in proc.stdout


def test_freshness_empty_and_missing(tmp_path: Path):
    empty = tmp_path / "empty.json"
    _write_snapshot(empty, [])
    proc = _run(
        "check_snapshot_freshness.py",
        ["--snapshot", str(empty), "--now", "2026-11-24T00:00:00Z"],
    )
    assert proc.returncode == 2
    assert "status=empty" in proc.stdout

    missing = _run(
        "check_snapshot_freshness.py",
        ["--snapshot", str(tmp_path / "nope.json")],
    )
    assert missing.returncode == 1
    assert "error:" in missing.stderr


def test_scorecard_export_writes_settled_weeks_only(tmp_path: Path):
    db = init_db(tmp_path / "predictions.sqlite")
    conn = connect(db)
    upsert_model_artefact(
        conn,
        "v-new",
        "2026-11-16",
        "models/v-new",
        {"xi": 0.0035},
        created_at="2026-11-16T00:00:00Z",
    )
    insert_prediction(conn, _prediction(match_id=1, kickoff_utc="2026-11-22T15:00:00Z"))
    insert_prediction(
        conn,
        _prediction(match_id=2, kickoff_utc="2026-11-23T20:00:00Z", status="scheduled"),
    )
    assert settle_prediction(conn, 1, "v-new", 2, 1, "H")
    conn.close()

    out = tmp_path / "scorecard-snapshot.json"
    code = _load("export_scorecard_snapshot").export_scorecard(db, out, model_version="v-new")
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["model_version"] == "v-new"
    assert len(payload["weeks"]) == 1
    week = payload["weeks"][0]
    kick = datetime.fromisoformat("2026-11-22T15:00:00+00:00")
    year, iso_week, _day = kick.isocalendar()
    assert week["iso_week"] == f"{year}-W{iso_week:02d}"
    assert week["match_count"] == 1
    assert len(week["matches"]) == 1
    assert week["matches"][0]["result"] == "H"
    assert week["matches"][0]["home_goals"] == 2
    assert "exported_at" in payload


def test_scorecard_export_empty_ledger_exits_2(tmp_path: Path):
    db = init_db(tmp_path / "predictions.sqlite")
    conn = connect(db)
    upsert_model_artefact(
        conn,
        "v-new",
        "2026-11-16",
        "models/v-new",
        None,
        created_at="2026-11-16T00:00:00Z",
    )
    conn.close()
    out = tmp_path / "scorecard-snapshot.json"
    code = _load("export_scorecard_snapshot").export_scorecard(db, out)
    assert code == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["weeks"] == []


def test_scorecard_export_missing_db(tmp_path: Path):
    code = _load("export_scorecard_snapshot").export_scorecard(
        tmp_path / "missing.sqlite",
        tmp_path / "out.json",
    )
    assert code == 1
    assert not (tmp_path / "out.json").exists()


def test_refresh_gates_do_not_print_a_token(tmp_path: Path, capsys):
    slate = _load("refresh_slate")
    assert slate.require_token("") == 1
    assert slate.require_token("   ") == 1
    assert slate.require_token("test-token-not-a-secret") == 0
    captured = capsys.readouterr()
    assert "TOKEN_PRESENT=yes" in captured.out
    assert "missing or empty" in captured.err
    assert "test-token-not-a-secret" not in captured.out + captured.err
    assert slate.require_predictions_db(tmp_path / "missing.sqlite") == 2
    assert slate.require_predictions_db(_touch(tmp_path / "predictions.sqlite")) == 0


def test_refresh_freshness_reads_committed_snapshot():
    env = os.environ.copy()
    env.pop("FOOTBALL_DATA_ORG_TOKEN", None)
    proc = _run(
        "refresh_slate.py",
        ["--mode", "freshness", "--now", "2020-01-01T00:00:00Z"],
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "status=fresh" in proc.stdout
    assert "TOKEN_PRESENT" not in proc.stdout


def test_refresh_cli_stops_before_network():
    env = os.environ.copy()
    env["FOOTBALL_DATA_ORG_TOKEN"] = ""
    publish = _run("refresh_slate.py", ["--mode", "publish"], env=env)
    assert publish.returncode == 1
    assert "missing or empty" in publish.stderr
    assert "TOKEN_PRESENT" not in publish.stdout

    env["FOOTBALL_DATA_ORG_TOKEN"] = "test-token-not-a-secret"
    db = ROOT / "data" / "processed" / "predictions.sqlite"
    if db.exists():
        return
    settle = _run("refresh_slate.py", ["--mode", "settle"], env=env)
    assert settle.returncode == 2, settle.stderr
    combined = settle.stdout + settle.stderr
    assert "predictions DB missing" in combined
    assert "test-token-not-a-secret" not in combined


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _prediction(**overrides):
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
