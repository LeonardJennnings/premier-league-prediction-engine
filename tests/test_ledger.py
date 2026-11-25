"""Ledger rules: one row per match, no write after kickoff, matchday 11 copied."""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.config import LEDGER_DIR, ROOT
from src.jobs.publish import publish_slate
from src.ledger.backfill import SCORE_OMITTED_REASON, backfill_matchday_11
from src.ledger.log import LedgerRefused, read_jsonl
from src.models.artefacts import published_model_version, save_artefact
from src.models.dixon_coles import DixonColesModel
from src.models.markets import markets_from_score_table
from tests.test_publish import _model, _patch_paths

FIXTURE = ROOT / "tests" / "fixtures" / "upcoming-2026-11-17.json"
_COPIED = (
    "match_id",
    "matchday",
    "kickoff_utc",
    "home_team",
    "away_team",
    "competition",
    "model_version",
    "published_at",
    "as_of_date",
    "p_home",
    "p_draw",
    "p_away",
    "p_btts_yes",
    "p_btts_no",
    "p_over_25",
    "p_under_25",
)


def test_backfill_copies_matchday_11_and_settles_it(tmp_path):
    snapshot = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ledger = tmp_path / "ledger"
    first = backfill_matchday_11(
        snapshot["matches"],
        recorded_at="2026-11-25T01:00:00Z",
        ledger_path=ledger,
    )
    assert first["predictions_inserted"] == 10
    assert first["settlements_inserted"] == 10
    lines = read_jsonl(ledger / "predictions.jsonl")
    assert len(lines) == 10
    by_id = {row["match_id"]: row for row in lines}
    for original in snapshot["matches"]:
        got = by_id[original["match_id"]]
        for field in _COPIED:
            assert got[field] == original[field]
        assert got["forecast_line"] == "model"
        assert got["score_table"] is None
        assert got["most_likely_score"] is None
        assert got["next_scores"] is None
        assert got["lambda_home"] is None
        assert got["lambda_away"] is None
        assert got["rho"] is None
        assert got["totals"] is None
        assert got["score_omitted_reason"] == SCORE_OMITTED_REASON
    settlements = read_jsonl(ledger / "settlements.jsonl")
    assert len(settlements) == 10
    by_result = {row["match_id"]: row for row in settlements}
    assert by_result[560643]["home_goals"] == 1
    assert by_result[560643]["away_goals"] == 1
    assert by_result[560643]["result"] == "D"
    assert by_result[560643]["status"] == "settled"
    assert by_result[560645]["home_goals"] == 4
    assert by_result[560645]["away_goals"] == 0
    second = backfill_matchday_11(
        snapshot["matches"],
        recorded_at="2026-11-25T02:00:00Z",
        ledger_path=ledger,
    )
    assert second["predictions_inserted"] == 0
    assert second["settlements_inserted"] == 0
    assert len(read_jsonl(ledger / "predictions.jsonl")) == 10
    prefix = (ledger / "predictions.jsonl").read_bytes()
    backfill_matchday_11(
        snapshot["matches"],
        recorded_at="2026-11-25T03:00:00Z",
        ledger_path=ledger,
    )
    assert (ledger / "predictions.jsonl").read_bytes() == prefix


def test_publish_twice_with_a_newer_model_keeps_one_row(tmp_path, monkeypatch):
    models_dir, db, ledger = _patch_paths(monkeypatch, tmp_path)
    first_model = _model()
    version_a, _ = save_artefact(first_model, 1.1, {"xi": 0.0019}, models_dir)
    newer = DixonColesModel(
        attack={"Arsenal": 1.5, "Chelsea": 0.7},
        defence={"Arsenal": 0.7, "Chelsea": 1.4},
        home_advantage=1.1,
        rho=-0.04,
        xi=0.0035,
        as_of=pd.Timestamp("2026-11-20"),
    )
    version_b, _ = save_artefact(newer, 1.2, {"xi": 0.0035}, models_dir)
    assert version_a != version_b
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
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    first = publish_slate(version_a, now_utc=now, token="x")
    second = publish_slate(version_b, now_utc=now, token="x")
    assert first["inserted"] == 1
    assert second["inserted"] == 0
    assert second["skipped_duplicate"] == 1
    lines = read_jsonl(ledger / "predictions.jsonl")
    assert len(lines) == 1
    assert lines[0]["model_version"] == published_model_version(version_a)
    assert lines[0]["artefact_version"] == version_a
    assert not lines[0]["model_version"].endswith(version_b)
    from src.store.predictions import connect

    conn = connect(db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_publish_refuses_a_match_that_has_kicked_off(tmp_path, monkeypatch):
    models_dir, db, ledger = _patch_paths(monkeypatch, tmp_path)
    version, _ = save_artefact(_model(), 1.0, {"xi": 0.0019}, models_dir)
    started = {
        "match_id": 7,
        "kickoff_utc": "2026-11-15T15:00:00Z",
        "home_team": "Arsenal FC",
        "away_team": "Chelsea FC",
        "matchday": 12,
        "status": "TIMED",
    }
    monkeypatch.setattr("src.jobs.publish.fetch_pl_matches", lambda *a, **k: [started])
    monkeypatch.setattr(
        "src.jobs.publish.matches_in_publish_window",
        lambda matches, **kwargs: matches,
    )
    monkeypatch.setattr(
        "src.jobs.publish.load_team_name_map",
        lambda: {"Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea"},
    )
    with pytest.raises(LedgerRefused, match="after kickoff"):
        publish_slate(
            version,
            now_utc=datetime(2026, 11, 16, tzinfo=timezone.utc),
            token="x",
        )
    from src.store.predictions import connect

    conn = connect(db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        conn.close()
    assert count == 0
    assert read_jsonl(ledger / "predictions.jsonl") == []


def test_published_markets_match_the_stored_table(tmp_path, monkeypatch):
    models_dir, _db, ledger = _patch_paths(monkeypatch, tmp_path)
    model = _model()
    version, _ = save_artefact(model, 1.4, {"xi": 0.0019}, models_dir)
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
    publish_slate(
        version,
        now_utc=datetime(2026, 11, 16, tzinfo=timezone.utc),
        token="x",
    )
    line = read_jsonl(ledger / "predictions.jsonl")[0]
    table = line["score_table"]
    markets = markets_from_score_table(table)
    assert line["forecast_line"] == "model"
    assert line["model_version"].endswith("-grid")
    assert line["lambda_home"] > 0
    assert line["lambda_away"] > 0
    assert line["most_likely_score"]["probability"] > 0
    assert len(line["next_scores"]) == 2
    for key in ("0.5", "1.5", "2.5", "3.5", "4.5"):
        assert set(line["totals"][key]) == {"over", "under"}
    assert line["p_home"] == pytest.approx(markets["home"])
    assert line["p_draw"] == pytest.approx(markets["draw"])
    assert line["p_away"] == pytest.approx(markets["away"])
    assert line["p_btts_yes"] == pytest.approx(markets["btts_yes"])
    assert line["p_btts_no"] == pytest.approx(markets["btts_no"])
    assert line["p_over_25"] == pytest.approx(markets["over_25"])
    assert line["p_under_25"] == pytest.approx(markets["under_25"])
    assert line["totals"]["2.5"]["over"] == pytest.approx(line["p_over_25"])
    cell_mass = sum(sum(row) for row in table["cells"])
    assert cell_mass + table["other"] == pytest.approx(1.0)
    home = 0.0
    over = table["other"]
    for i, row in enumerate(table["cells"]):
        for j, probability in enumerate(row):
            if i > j:
                home += probability
            if i + j >= 3:
                over += probability
    assert line["p_home"] == pytest.approx(home)
    assert line["p_over_25"] == pytest.approx(over)


def test_committed_ledger_holds_matchdays_11_and_12():
    """The files on disk are the record that has to reach main."""
    predictions = read_jsonl(LEDGER_DIR / "predictions.jsonl")
    settlements = read_jsonl(LEDGER_DIR / "settlements.jsonl")
    snapshot = json.loads(FIXTURE.read_text(encoding="utf-8"))
    md11 = [row for row in predictions if row.get("matchday") == 11]
    md12 = [row for row in predictions if row.get("matchday") == 12]
    assert len(md11) == 10
    by_id = {row["match_id"]: row for row in md11}
    for original in snapshot["matches"]:
        got = by_id[original["match_id"]]
        for field in _COPIED:
            assert got[field] == original[field]
        assert got["score_table"] is None
        assert got["score_omitted_reason"] == SCORE_OMITTED_REASON
    settled_ids = {
        row["match_id"]
        for row in settlements
        if row.get("kind") == "result" and row.get("status") == "settled"
    }
    assert {row["match_id"] for row in md11} <= settled_ids
    assert len(md12) == 10
    forest = [
        row for row in md12 if row["kickoff_utc"] == "2026-11-27T20:00:00Z"
    ]
    assert len(forest) == 1
    assert forest[0]["home_team"] == "Nott'm Forest"
    assert forest[0]["away_team"] == "Chelsea"
    for row in md12:
        assert row["score_table"] is not None
        assert row["score_table"]["other"] > 0
        assert row["model_version"].endswith("-grid")
        assert row["most_likely_score"]["probability"] > 0
        assert len(row["next_scores"]) == 2
        assert row["score_omitted_reason"] is None
        assert row["forecast_line"] == "model"
    assert (LEDGER_DIR / "manifest.json").is_file()
    assert (LEDGER_DIR / "book_lines.jsonl").is_file()
