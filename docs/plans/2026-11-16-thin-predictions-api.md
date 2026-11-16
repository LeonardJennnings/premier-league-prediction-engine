# Thin read-only predictions API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.  
> **Commits:** Skip automatic git commits unless explicitly asked.

**Goal:** Ship a FastAPI read-only HTTP API that serves stored upcoming and settled predictions (plus model metadata) for Epic 5, without fitting or predicting on the request path.

**Architecture:** Routes → `src/api/service.py` → new store read helpers on SQLite. Default `model_version` is latest artefact; optional pin. Upcoming uses the Epic 3 rolling window; past scorecard groups settled rows by ISO week of `kickoff_utc`.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn (already in requirements), sqlite3, pytest, httpx (pulled in by FastAPI TestClient).

**Spec:** [`docs/specs/thin-predictions-api.md`](../specs/thin-predictions-api.md)

## Global Constraints

- Read-only: no fit/publish/settle/predict on the request path; no write HTTP routes
- Serve stored rows from `PREDICTIONS_DB` only
- Default latest `model_artefacts` by `created_at` desc, then `model_version` desc; optional `?model_version=`
- Upcoming: `scheduled`/`postponed`, `kickoff_utc` in `[now, now + PUBLISH_WINDOW_DAYS)`, same version
- Past: ISO week `YYYY-Www` from UTC kickoff; settled rows only; empty/unknown week → `404`
- Empty upcoming → `200` + `matches: []`
- Missing DB / unreadable → `503`; unknown version → `404`
- CORS for local Next.js (`http://localhost:3000`)
- Docs: professional portfolio voice; no tipster language
- Skip automatic commits unless asked

## File map

| Path | Responsibility |
|---|---|
| `requirements.txt` | Add pinned `fastapi` |
| `src/config.py` | Optional `API_CORS_ORIGINS` (default localhost:3000) |
| `src/store/predictions.py` | Read helpers: latest artefact, get artefact, list upcoming, list settled by week, list weeks |
| `src/api/__init__.py` | Package marker |
| `src/api/schemas.py` | Pydantic response models |
| `src/api/service.py` | Version resolve, upcoming/weeks/models orchestration; maps store rows → schemas; raises domain errors |
| `src/api/deps.py` | DB connection / path dependency; `503` if missing |
| `src/api/routes/health.py` | `GET /health` |
| `src/api/routes/predictions.py` | Upcoming + weeks |
| `src/api/routes/models.py` | Latest + by version |
| `src/api/app.py` | FastAPI app, CORS, routers |
| `tests/test_store_reads.py` | Store read helper unit tests |
| `tests/test_api_service.py` | Service unit tests (temp DB) |
| `tests/test_api.py` | FastAPI TestClient integration |
| `README.md` | How to run the API |

---

### Task 1: Store read helpers

**Files:**
- Modify: `src/store/predictions.py`
- Create: `tests/test_store_reads.py`

**Interfaces:**
- Consumes: existing `connect`, schema tables
- Produces:
  - `get_latest_model_version(conn) -> str | None`
  - `get_model_artefact(conn, model_version: str) -> dict | None` — keys: model_version, as_of_date, created_at, artefact_path, hyperparams (raw text or None)
  - `list_upcoming_predictions(conn, model_version: str, *, now_utc: str, window_days: int) -> list[dict]`
  - `list_settled_weeks(conn, model_version: str) -> list[dict]` — each `{"iso_week": "YYYY-Www", "match_count": int}`, newest first
  - `list_settled_predictions_for_week(conn, model_version: str, iso_week: str) -> list[dict]`

ISO week in SQL (SQLite): `printf('%s-W%02d', strftime('%G', replace(kickoff_utc, 'T', ' ')), cast(strftime('%V', replace(kickoff_utc, 'T', ' ')) as integer))`  
If `printf`/`%G` proves awkward on the CI SQLite, compute the week key in Python from parsed `kickoff_utc` inside the helper instead — prefer Python `datetime.fromisoformat(...).isocalendar()` for portability and test clarity.

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timezone
from pathlib import Path

from src.store.predictions import (
    connect,
    get_latest_model_version,
    get_model_artefact,
    insert_prediction,
    list_settled_predictions_for_week,
    list_settled_weeks,
    list_upcoming_predictions,
    upsert_model_artefact,
)
from src.store.schema import init_db


def _row(**overrides):
    base = {
        "match_id": 1001,
        "model_version": "v-old",
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


def test_latest_model_version_by_created_at(tmp_path: Path):
    db = init_db(tmp_path / "p.sqlite")
    conn = connect(db)
    upsert_model_artefact(conn, "v-old", "2026-11-01", "models/v-old", {}, created_at="2026-11-01T00:00:00Z")
    upsert_model_artefact(conn, "v-new", "2026-11-10", "models/v-new", {}, created_at="2026-11-10T00:00:00Z")
    assert get_latest_model_version(conn) == "v-new"
    assert get_model_artefact(conn, "v-old")["as_of_date"] == "2026-11-01"
    conn.close()


def test_list_upcoming_respects_window_and_status(tmp_path: Path):
    db = init_db(tmp_path / "p.sqlite")
    conn = connect(db)
    now = "2026-11-16T12:00:00Z"
    insert_prediction(conn, _row(match_id=1, kickoff_utc="2026-11-17T15:00:00Z", status="scheduled"))
    insert_prediction(conn, _row(match_id=2, kickoff_utc="2026-11-18T15:00:00Z", status="postponed"))
    insert_prediction(conn, _row(match_id=3, kickoff_utc="2026-11-30T15:00:00Z", status="scheduled"))  # outside 10d
    insert_prediction(conn, _row(match_id=4, kickoff_utc="2026-11-10T15:00:00Z", status="scheduled"))  # past
    insert_prediction(conn, _row(match_id=5, kickoff_utc="2026-11-17T12:00:00Z", status="settled"))
    rows = list_upcoming_predictions(conn, "v-old", now_utc=now, window_days=10)
    assert [r["match_id"] for r in rows] == [2, 1]
    conn.close()


def test_settled_weeks_iso_and_detail(tmp_path: Path):
    db = init_db(tmp_path / "p.sqlite")
    conn = connect(db)
    # 2026-11-14 is Saturday of ISO week 2026-W46
    insert_prediction(
        conn,
        _row(
            match_id=10,
            kickoff_utc="2026-11-14T15:00:00Z",
            status="settled",
            home_goals=2,
            away_goals=1,
            result="H",
            settled_at="2026-11-14T17:00:00Z",
        ),
    )
    insert_prediction(
        conn,
        _row(
            match_id=11,
            kickoff_utc="2026-11-07T15:00:00Z",
            status="settled",
            home_goals=0,
            away_goals=0,
            result="D",
            settled_at="2026-11-07T17:00:00Z",
        ),
    )
    insert_prediction(conn, _row(match_id=12, kickoff_utc="2026-11-14T15:00:00Z", status="scheduled"))
    weeks = list_settled_weeks(conn, "v-old")
    assert weeks[0]["iso_week"] == "2026-W46"
    assert weeks[0]["match_count"] == 1
    assert weeks[1]["iso_week"] == "2026-W45"
    detail = list_settled_predictions_for_week(conn, "v-old", "2026-W46")
    assert [r["match_id"] for r in detail] == [10]
    conn.close()
```

Note: `insert_prediction` only writes `INSERT_COLUMNS` — settlement fields must be set via `settle_prediction` or `update_prediction_state` after insert. Adjust the test to insert scheduled then `settle_prediction(...)`.

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_store_reads.py -v`  
Expected: FAIL (functions not defined) or import error.

- [ ] **Step 3: Implement helpers in `src/store/predictions.py`**

```python
from datetime import datetime, timedelta, timezone


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso_week_key(kickoff_utc: str) -> str:
    dt = _parse_utc(kickoff_utc)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def get_latest_model_version(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT model_version FROM model_artefacts
        ORDER BY created_at DESC, model_version DESC
        LIMIT 1
        """
    ).fetchone()
    return None if row is None else row["model_version"]


def get_model_artefact(conn: sqlite3.Connection, model_version: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM model_artefacts WHERE model_version = ?",
        (model_version,),
    ).fetchone()
    return None if row is None else dict(row)


def list_upcoming_predictions(
    conn: sqlite3.Connection,
    model_version: str,
    *,
    now_utc: str,
    window_days: int,
) -> list[dict]:
    now = _parse_utc(now_utc)
    end = now + timedelta(days=window_days)
    rows = conn.execute(
        """
        SELECT * FROM predictions
        WHERE model_version = ?
          AND status IN ('scheduled', 'postponed')
        ORDER BY kickoff_utc, match_id
        """,
        (model_version,),
    ).fetchall()
    out = []
    for row in rows:
        kick = _parse_utc(row["kickoff_utc"])
        if now <= kick < end:
            out.append(dict(row))
    return out


def list_settled_weeks(conn: sqlite3.Connection, model_version: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT kickoff_utc FROM predictions
        WHERE model_version = ? AND status = 'settled'
        """,
        (model_version,),
    ).fetchall()
    counts: dict[str, int] = {}
    for row in rows:
        key = _iso_week_key(row["kickoff_utc"])
        counts[key] = counts.get(key, 0) + 1
    return [
        {"iso_week": week, "match_count": counts[week]}
        for week in sorted(counts.keys(), reverse=True)
    ]


def list_settled_predictions_for_week(
    conn: sqlite3.Connection, model_version: str, iso_week: str
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT * FROM predictions
        WHERE model_version = ? AND status = 'settled'
        ORDER BY kickoff_utc, match_id
        """,
        (model_version,),
    ).fetchall()
    return [dict(r) for r in rows if _iso_week_key(r["kickoff_utc"]) == iso_week]
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_store_reads.py -v`  
Expected: PASS

---

### Task 2: Service layer + domain errors

**Files:**
- Create: `src/api/__init__.py` (empty)
- Create: `src/api/service.py`
- Create: `tests/test_api_service.py`

**Interfaces:**
- Consumes: store read helpers; `PUBLISH_WINDOW_DAYS` from config
- Produces:
  - `class NotFoundError(Exception)` with message
  - `resolve_model_version(conn, model_version: str | None) -> str`
  - `get_upcoming(conn, model_version: str | None, *, now_utc: str | None = None) -> dict` → `{"model_version", "matches"}`
  - `list_weeks(conn, model_version: str | None) -> dict` → `{"model_version", "weeks"}`
  - `get_week(conn, iso_week: str, model_version: str | None) -> dict` → `{"model_version", "iso_week", "matches"}` or raise `NotFoundError`
  - `get_latest_model(conn) -> dict` / `get_model(conn, model_version: str) -> dict` — parse `hyperparams` JSON to dict|None; raise `NotFoundError` when missing

- [ ] **Step 1: Write failing service tests**

```python
import json
from pathlib import Path

import pytest

from src.api.service import (
    NotFoundError,
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
    # reuse _row helper pattern from store tests — scheduled upcoming under v-new
    return conn


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
```

Expand with one happy-path upcoming + settled week using the same seed style as Task 1 (insert + settle).

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_api_service.py -v`

- [ ] **Step 3: Implement `src/api/service.py`**

```python
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config import PUBLISH_WINDOW_DAYS
from src.store import predictions as store


class NotFoundError(Exception):
    pass


def resolve_model_version(conn, model_version: str | None) -> str:
    if model_version is not None:
        if store.get_model_artefact(conn, model_version) is None:
            raise NotFoundError(f"model_version not found: {model_version}")
        return model_version
    latest = store.get_latest_model_version(conn)
    if latest is None:
        raise NotFoundError("no model artefacts published")
    return latest


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_upcoming(conn, model_version: str | None, *, now_utc: str | None = None) -> dict:
    version = resolve_model_version(conn, model_version)
    rows = store.list_upcoming_predictions(
        conn,
        version,
        now_utc=now_utc or _now_utc_iso(),
        window_days=PUBLISH_WINDOW_DAYS,
    )
    return {"model_version": version, "matches": rows}


def list_weeks(conn, model_version: str | None) -> dict:
    version = resolve_model_version(conn, model_version)
    return {"model_version": version, "weeks": store.list_settled_weeks(conn, version)}


def get_week(conn, iso_week: str, model_version: str | None) -> dict:
    version = resolve_model_version(conn, model_version)
    rows = store.list_settled_predictions_for_week(conn, version, iso_week)
    if not rows:
        raise NotFoundError(f"no settled matches for iso_week={iso_week}")
    return {"model_version": version, "iso_week": iso_week, "matches": rows}


def _artefact_payload(row: dict) -> dict:
    raw = row.get("hyperparams")
    if raw is None or raw == "":
        hyper = None
    elif isinstance(raw, str):
        hyper = json.loads(raw)
    else:
        hyper = raw
    return {
        "model_version": row["model_version"],
        "as_of_date": row["as_of_date"],
        "created_at": row["created_at"],
        "artefact_path": row["artefact_path"],
        "hyperparams": hyper,
    }


def get_latest_model(conn) -> dict:
    version = resolve_model_version(conn, None)
    row = store.get_model_artefact(conn, version)
    assert row is not None
    return _artefact_payload(row)


def get_model(conn, model_version: str) -> dict:
    row = store.get_model_artefact(conn, model_version)
    if row is None:
        raise NotFoundError(f"model_version not found: {model_version}")
    return _artefact_payload(row)
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_api_service.py tests/test_store_reads.py -v`

---

### Task 3: FastAPI app, schemas, deps, health

**Files:**
- Modify: `requirements.txt` — add `fastapi` (pin current stable compatible with existing `starlette`/`uvicorn`; if versions conflict, upgrade starlette/uvicorn as needed so `pip install` resolves)
- Modify: `src/config.py` — add CORS origins
- Create: `src/api/schemas.py`
- Create: `src/api/deps.py`
- Create: `src/api/routes/__init__.py`
- Create: `src/api/routes/health.py`
- Create: `src/api/app.py`
- Create: `tests/test_api.py` (start with health + missing DB)

**Interfaces:**
- `API_CORS_ORIGINS` from env comma-separated; default `http://localhost:3000`
- `get_connection` dependency: open `PREDICTIONS_DB` or override via app `state.db_path`; missing file → raise `HTTPException(503)`
- App: `create_app(db_path: Path | None = None) -> FastAPI` for tests; module-level `app = create_app()`

- [ ] **Step 1: Install fastapi**

```bash
python -m pip install "fastapi>=0.115"
python -m pip freeze | grep -iE 'fastapi|starlette|uvicorn|httpx'
# update requirements.txt pins to match the working set
```

- [ ] **Step 2: Write failing health / 503 tests**

```python
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app


def test_health_ok():
    client = TestClient(create_app(db_path=Path("/nonexistent/no.sqlite")))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_upcoming_503_when_db_missing():
    client = TestClient(create_app(db_path=Path("/nonexistent/no.sqlite")))
    r = client.get("/v1/predictions/upcoming")
    assert r.status_code == 503
```

- [ ] **Step 3: Implement schemas, deps, health, app**

`schemas.py` — `MatchOut` with all fields from the spec (floats, optional books/settlement); envelopes `UpcomingOut`, `WeekOut`, `WeeksOut`, `WeekSummary`, `ModelOut`.

`deps.py`:

```python
from pathlib import Path
import sqlite3
from fastapi import HTTPException, Request
from src.store.predictions import connect
from src.config import PREDICTIONS_DB


def get_db(request: Request):
    path: Path = getattr(request.app.state, "db_path", None) or PREDICTIONS_DB
    if not path.is_file():
        raise HTTPException(status_code=503, detail="predictions database unavailable")
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()
```

`app.py`: CORSMiddleware with `API_CORS_ORIGINS`; include health router; mount prediction/model routers in Task 4 (stubs ok if tests only hit health/503 — for 503 upcoming, register a minimal upcoming route that depends on `get_db`, or finish Task 4 in the same PR slice).

Prefer finishing Task 3 with health only, and move the `503` upcoming assertion into Task 4 once the route exists. If so, Task 3 test is only `test_health_ok`.

- [ ] **Step 4: Run health test PASS**

Run: `pytest tests/test_api.py::test_health_ok -v`

---

### Task 4: Prediction and model routes

**Files:**
- Create: `src/api/routes/predictions.py`
- Create: `src/api/routes/models.py`
- Modify: `src/api/app.py` — include routers under `/v1`
- Modify: `tests/test_api.py` — full integration suite

**Interfaces:**
- Routes call service; map `NotFoundError` → `HTTPException(404)`
- Optional query `model_version: str | None = None`
- `iso_week` path param: validate with regex `^\d{4}-W\d{2}$` (FastAPI/`Annotated` pattern) → `422` on malformed

- [ ] **Step 1: Write integration tests** (temp DB seeded with two versions, upcoming + settled week)

Cover:
- `GET /v1/predictions/upcoming` → latest version, 200, null books preserved
- `?model_version=v-old` pins
- unknown version → 404
- empty upcoming → 200 `matches: []`
- `GET /v1/predictions/weeks` lists weeks newest first
- `GET /v1/predictions/weeks/2026-W46` returns settled matches
- unknown week → 404
- malformed week `2026-W4` → 422
- `GET /v1/models/latest` and `/v1/models/{version}`
- OpenAPI paths are GET-only for these resources (no POST)
- missing DB → 503 on a prediction route

Use `freezegun` only if already a dependency; otherwise pass controllable kickoff times relative to “now” by seeding far-future kickoffs and monkeypatching `src.api.service._now_utc_iso` to a fixed string.

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_api.py -v`

- [ ] **Step 3: Implement routes**

Example upcoming handler:

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from src.api.deps import get_db
from src.api import service
from src.api.schemas import UpcomingOut

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/upcoming", response_model=UpcomingOut)
def upcoming(model_version: str | None = Query(default=None), conn=Depends(get_db)):
    try:
        return service.get_upcoming(conn, model_version)
    except service.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
```

Wire models similarly. Ensure route modules do **not** import `src.jobs` or `src.models.dixon_coles`.

- [ ] **Step 4: Run full API tests PASS**

Run: `pytest tests/test_api.py tests/test_api_service.py tests/test_store_reads.py -v`  
Expected: PASS

- [ ] **Step 5: Sanity — no model import on API package**

```bash
python -c "import src.api.app; import sys; assert 'src.models.dixon_coles' not in sys.modules"
```

---

### Task 5: README + config polish

**Files:**
- Modify: `README.md` — add “Predictions API” section after fixtures/settlement
- Modify: `.env.example` — optional `API_CORS_ORIGINS=http://localhost:3000`
- Modify: `src/config.py` if not done in Task 3

- [ ] **Step 1: Document run command**

```markdown
## Predictions API

Read-only FastAPI surface over the SQLite prediction store (no fit/predict on the request path).

```bash
uvicorn src.api.app:app --reload
```

Open http://127.0.0.1:8000/docs for OpenAPI. Endpoints: `GET /health`, `GET /v1/predictions/upcoming`, `GET /v1/predictions/weeks`, `GET /v1/predictions/weeks/{iso_week}`, `GET /v1/models/latest`, `GET /v1/models/{model_version}`. Optional `?model_version=` pin. Requires a populated `data/processed/predictions.sqlite` from `fit` / `publish` / `settle`.
```

- [ ] **Step 2: Run full suite**

Run: `pytest -q`  
Expected: all green (Epics 1–4).

- [ ] **Step 3: Manual smoke (optional if a real DB exists)**

```bash
uvicorn src.api.app:app --reload
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/predictions/upcoming | head
```

---

## Self-review checklist

1. **Spec coverage:** upcoming, weeks list/detail, models latest/by id, health, version default+pin, 404/422/503, empty upcoming 200, CORS, OpenAPI/README, null books, no write routes, no model on request path.
2. **Placeholders:** none — helpers and route sketches are concrete.
3. **Types:** `NotFoundError` → 404; envelopes include `model_version`; ISO week `YYYY-Www`.

## Execution handoff

Plan saved to `docs/plans/2026-11-16-thin-predictions-api.md`.
