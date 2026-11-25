import ast
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.store.predictions import (
    connect,
    connect_readonly,
    insert_prediction,
    settle_prediction,
    upsert_model_artefact,
)
from src.store.schema import init_db

NOW = "2026-11-16T12:00:00Z"

_HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
_WRITE_METHODS = {"put", "post", "delete", "patch", "trace"}
_V1_PATHS = [
    "/v1/predictions/upcoming",
    "/v1/predictions/weeks",
    "/v1/predictions/weeks/{iso_week}",
    "/v1/models/latest",
    "/v1/models/{model_version}",
]
_FORBIDDEN_IMPORT_PREFIXES = (
    "src.jobs",
    "src.models",
    "src.eval",
    "src.pipeline",
    "src.fixtures",
)


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


def _seed(tmp_path: Path, *, predictions: bool) -> Path:
    db = init_db(tmp_path / "p.sqlite")
    conn = connect(db)
    upsert_model_artefact(
        conn, "v-old", "2026-11-01", "models/v-old", {"xi": 0.001}, created_at="2026-11-01T00:00:00Z"
    )
    upsert_model_artefact(
        conn, "v-new", "2026-11-10", "models/v-new", {"xi": 0.002}, created_at="2026-11-10T00:00:00Z"
    )
    if predictions:
        insert_prediction(
            conn,
            _row(match_id=1, kickoff_utc="2026-11-17T15:00:00Z", status="scheduled"),
        )
        insert_prediction(
            conn,
            _row(match_id=4, model_version="v-old", kickoff_utc="2026-11-17T15:00:00Z"),
        )
        insert_prediction(conn, _row(match_id=10, kickoff_utc="2026-11-14T15:00:00Z"))
        settle_prediction(conn, 10, "v-new", 2, 1, "H")
        insert_prediction(conn, _row(match_id=11, kickoff_utc="2026-11-07T15:00:00Z"))
        settle_prediction(conn, 11, "v-new", 0, 0, "D")
        insert_prediction(
            conn, _row(match_id=12, kickoff_utc="2026-11-14T15:00:00Z", status="scheduled")
        )
    conn.close()
    return db


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.api.service._now_utc_iso", lambda: NOW)
    return TestClient(create_app(db_path=_seed(tmp_path, predictions=True)))


@pytest.fixture
def empty_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.api.service._now_utc_iso", lambda: NOW)
    return TestClient(create_app(db_path=_seed(tmp_path, predictions=False)))


def test_health_ok():
    client = TestClient(create_app(db_path=Path("/nonexistent/no.sqlite")))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_upcoming_503_when_db_missing():
    client = TestClient(create_app(db_path=Path("/nonexistent/no.sqlite")))
    r = client.get("/v1/predictions/upcoming")
    assert r.status_code == 503
    assert r.json()["detail"] == "predictions database unavailable"


def test_upcoming_latest_preserves_null_books(client: TestClient):
    r = client.get("/v1/predictions/upcoming")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "public"
    assert [m["match_id"] for m in body["matches"]] == [1, 4]
    match = body["matches"][0]
    assert match["home_team"] == "Arsenal"
    assert match["away_team"] == "Chelsea"
    assert match["status"] == "scheduled"
    assert match["p_home"] == 0.45
    assert match["book_p_home"] is None
    assert match["book_p_draw"] is None
    assert match["book_p_away"] is None
    assert match["book_p_over_25"] is None
    assert match["book_p_under_25"] is None
    assert match["book_source"] is None
    assert match["home_goals"] is None
    assert match["away_goals"] is None
    assert match["result"] is None
    assert match["settled_at"] is None


def test_upcoming_pins_model_version(client: TestClient):
    r = client.get("/v1/predictions/upcoming", params={"model_version": "v-old"})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "v-old"
    assert [m["match_id"] for m in body["matches"]] == [4]


def test_upcoming_unknown_version_404(client: TestClient):
    r = client.get("/v1/predictions/upcoming", params={"model_version": "missing"})
    assert r.status_code == 404
    assert r.json()["detail"] == "model_version not found: missing"


def test_upcoming_empty_is_200(empty_client: TestClient):
    r = empty_client.get("/v1/predictions/upcoming")
    assert r.status_code == 200
    assert r.json() == {"model_version": "public", "matches": []}


def test_weeks_listed_newest_first(client: TestClient):
    r = client.get("/v1/predictions/weeks")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "public"
    assert body["weeks"] == [
        {"iso_week": "2026-W46", "match_count": 1},
        {"iso_week": "2026-W45", "match_count": 1},
    ]


def test_week_detail_returns_settled_matches(client: TestClient):
    r = client.get("/v1/predictions/weeks/2026-W46")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "public"
    assert body["iso_week"] == "2026-W46"
    assert [m["match_id"] for m in body["matches"]] == [10]
    match = body["matches"][0]
    assert match["status"] == "settled"
    assert match["home_goals"] == 2
    assert match["away_goals"] == 1
    assert match["result"] == "H"


def test_unknown_week_404(client: TestClient):
    r = client.get("/v1/predictions/weeks/2026-W01")
    assert r.status_code == 404
    assert r.json()["detail"] == "no settled matches for iso_week=2026-W01"


def test_malformed_week_422(client: TestClient):
    r = client.get("/v1/predictions/weeks/2026-W4")
    assert r.status_code == 422


def test_models_latest_and_by_version(client: TestClient):
    latest = client.get("/v1/models/latest")
    assert latest.status_code == 200
    body = latest.json()
    assert body["model_version"] == "v-new"
    assert body["as_of_date"] == "2026-11-10"
    assert body["created_at"] == "2026-11-10T00:00:00Z"
    assert body["artefact_path"] == "models/v-new"
    assert body["hyperparams"] == {"xi": 0.002}

    old = client.get("/v1/models/v-old")
    assert old.status_code == 200
    assert old.json()["model_version"] == "v-old"
    assert old.json()["hyperparams"] == {"xi": 0.001}

    missing = client.get("/v1/models/missing")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "model_version not found: missing"


def test_openapi_routes_are_get_only(client: TestClient):
    schema = client.get("/openapi.json").json()
    for path in _V1_PATHS:
        assert path in schema["paths"]
        methods = set(schema["paths"][path]) & _HTTP_METHODS
        assert methods == {"get"}
    for path, ops in schema["paths"].items():
        segments = [seg for seg in path.split("/") if seg]
        assert "predict" not in segments
        assert (set(ops) & _HTTP_METHODS & _WRITE_METHODS) == set()


def test_api_package_forbids_fit_and_model_imports():
    api_root = Path(__file__).resolve().parents[1] / "src" / "api"
    for path in sorted(api_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert not module.startswith(_FORBIDDEN_IMPORT_PREFIXES), (
                    f"{path.name} imports {module}"
                )


def test_upcoming_without_artefacts_is_an_empty_public_slate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.api.service._now_utc_iso", lambda: NOW)
    db = init_db(tmp_path / "empty.sqlite")
    client = TestClient(create_app(db_path=db))
    upcoming = client.get("/v1/predictions/upcoming")
    assert upcoming.status_code == 200
    assert upcoming.json() == {"model_version": "public", "matches": []}
    weeks = client.get("/v1/predictions/weeks")
    assert weeks.status_code == 200
    assert weeks.json() == {"model_version": "public", "weeks": []}
    latest = client.get("/v1/models/latest")
    assert latest.status_code == 404
    assert latest.json()["detail"] == "no model artefacts published"


def test_api_db_connection_is_read_only(tmp_path: Path):
    db = _seed(tmp_path, predictions=True)
    conn = connect_readonly(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "UPDATE predictions SET status = 'settled' WHERE match_id = 1"
        )
    conn.close()


def test_models_latest_tolerates_corrupt_hyperparams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("src.api.service._now_utc_iso", lambda: NOW)
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
    conn.close()
    client = TestClient(create_app(db_path=db))
    r = client.get("/v1/models/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "v-bad"
    assert body["hyperparams"] is None
