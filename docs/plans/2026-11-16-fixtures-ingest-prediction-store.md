# Fixtures ingest + immutable prediction store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.  
> **Commits:** Skip automatic git commits unless explicitly asked.

**Goal:** Ingest Premier League fixtures from football-data.org, fit versioned Dixon–Coles artefacts, publish calibrated probabilities into an append-only SQLite store, and settle finished matches without rewriting published probs.

**Architecture:** Modular pipeline — fixtures client (+ cache) → team-name map (fail closed) → `fit` (artefact on disk) → `publish` (load artefact, never refit, insert-once) → `settle` (FD.org FINISHED → settlement fields only). Epic 4 will read the store only.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, requests, sqlite3 (stdlib), pytest. Existing Dixon–Coles + `predict_matches`.

**Spec:** [`docs/specs/fixtures-ingest-prediction-store.md`](../specs/fixtures-ingest-prediction-store.md)

## Global Constraints

- Premier League only (`competitions/PL`)
- SQLite at `data/processed/predictions.sqlite`; portable types; thin store module
- `fit` writes artefacts; `publish` loads them and never refits
- Probabilities immutable at insert; settlement updates score fields only
- Primary key `(match_id, model_version)`; refuse silent overwrite
- Rolling **10-day** publish window (weekend + midweek)
- Nullable `book_*` columns; no live odds fetch; no paper-return math; no affiliate fields
- Unmapped team or team missing from artefact → **abort publish batch**
- Note: `expected_goals` currently falls back to attack/defence `1.0` for unknown teams — publish must check `team in model.attack` (and defence) explicitly before predicting
- Docs: professional portfolio voice; no tipster language; no third-person builder references
- Skip automatic commits unless asked

## File map

| Path | Responsibility |
|---|---|
| `src/config.py` | Paths: sqlite, models dir, fixtures cache, publish window days, FD.org base URL |
| `data/reference/team_name_map.json` | Canonical FD.org → co.uk name map (committed) |
| `src/teams/mapping.py` | Load map; resolve; raise on unknown |
| `src/store/schema.py` | DDL + `init_db` |
| `src/store/predictions.py` | insert-once, settle, lookups |
| `src/models/artefacts.py` | Save/load Dixon–Coles + temperature + metadata |
| `src/fixtures/client.py` | FD.org HTTP + file cache |
| `src/fixtures/window.py` | Filter matches to next N days |
| `src/jobs/fit.py` | Fit artefact CLI |
| `src/jobs/publish.py` | Publish slate CLI |
| `src/jobs/settle.py` | Settlement CLI |
| `src/jobs/precompute.py` | fit then publish |
| `.gitignore` | Ignore `models/*` artefact dirs except keep path; sqlite already under `data/processed/` |
| `README.md` | Document fit / publish / settle |
| `tests/test_team_mapping.py` | Map resolve / fail closed |
| `tests/test_store.py` | Insert-once + settlement immutability |
| `tests/test_artefacts.py` | Round-trip artefact |
| `tests/test_fixtures_client.py` | Cache + window filter (HTTP mocked) |
| `tests/test_publish.py` | Abort on bad map / missing team; window |

---

### Task 1: Config paths + team-name map

**Files:**
- Modify: `src/config.py`
- Create: `data/reference/team_name_map.json`
- Create: `src/teams/__init__.py`
- Create: `src/teams/mapping.py`
- Create: `tests/test_team_mapping.py`
- Modify: `.gitignore` (ensure `data/reference/` is **not** ignored; keep ignoring `data/raw/` and `data/processed/`)

**Interfaces:**
- Consumes: JSON file of `{ "fd_org_name": "canonical_name", ... }`
- Produces:
  - Config: `PREDICTIONS_DB`, `MODELS_DIR`, `FIXTURES_CACHE_DIR`, `TEAM_NAME_MAP_PATH`, `PUBLISH_WINDOW_DAYS = 10`, `FD_ORG_BASE = "https://api.football-data.org/v4"`
  - `load_team_name_map(path: Path | None = None) -> dict[str, str]`
  - `resolve_team_name(fd_org_name: str, mapping: dict[str, str]) -> str` — raises `KeyError` if missing
  - `resolve_fixture_teams(home: str, away: str, mapping: dict[str, str]) -> tuple[str, str]`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from src.teams.mapping import load_team_name_map, resolve_team_name


def test_resolve_known_alias():
    mapping = load_team_name_map()
    assert resolve_team_name("Manchester United FC", mapping) == "Man United"
    assert resolve_team_name("Manchester City FC", mapping) == "Man City"
    assert resolve_team_name("Nottingham Forest FC", mapping) == "Nott'm Forest"


def test_unknown_name_fails_closed():
    mapping = {"Arsenal FC": "Arsenal"}
    with pytest.raises(KeyError, match="Unmapped team"):
        resolve_team_name("Unknown United FC", mapping)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_team_mapping.py -v`  
Expected: FAIL (module not found)

- [ ] **Step 3: Add config + JSON map + mapping module**

Append to `src/config.py` (keep existing constants):

```python
PREDICTIONS_DB = DATA_PROCESSED / "predictions.sqlite"
MODELS_DIR = ROOT / "models"
FIXTURES_CACHE_DIR = DATA_RAW / "fixtures"
TEAM_NAME_MAP_PATH = ROOT / "data" / "reference" / "team_name_map.json"
PUBLISH_WINDOW_DAYS = 10
FD_ORG_BASE = "https://api.football-data.org/v4"
```

Create `data/reference/team_name_map.json` covering current PL-relevant co.uk names (extend as needed). Minimum keys (FD.org short/long forms commonly seen):

```json
{
  "Arsenal FC": "Arsenal",
  "Arsenal": "Arsenal",
  "Aston Villa FC": "Aston Villa",
  "Aston Villa": "Aston Villa",
  "AFC Bournemouth": "Bournemouth",
  "Bournemouth": "Bournemouth",
  "Brentford FC": "Brentford",
  "Brentford": "Brentford",
  "Brighton & Hove Albion FC": "Brighton",
  "Brighton & Hove Albion": "Brighton",
  "Burnley FC": "Burnley",
  "Burnley": "Burnley",
  "Chelsea FC": "Chelsea",
  "Chelsea": "Chelsea",
  "Crystal Palace FC": "Crystal Palace",
  "Crystal Palace": "Crystal Palace",
  "Everton FC": "Everton",
  "Everton": "Everton",
  "Fulham FC": "Fulham",
  "Fulham": "Fulham",
  "Ipswich Town FC": "Ipswich",
  "Ipswich Town": "Ipswich",
  "Leeds United FC": "Leeds",
  "Leeds United": "Leeds",
  "Leicester City FC": "Leicester",
  "Leicester City": "Leicester",
  "Liverpool FC": "Liverpool",
  "Liverpool": "Liverpool",
  "Luton Town FC": "Luton",
  "Luton Town": "Luton",
  "Manchester City FC": "Man City",
  "Manchester City": "Man City",
  "Manchester United FC": "Man United",
  "Manchester United": "Man United",
  "Newcastle United FC": "Newcastle",
  "Newcastle United": "Newcastle",
  "Nottingham Forest FC": "Nott'm Forest",
  "Nottingham Forest": "Nott'm Forest",
  "Sheffield United FC": "Sheffield United",
  "Sheffield United": "Sheffield United",
  "Southampton FC": "Southampton",
  "Southampton": "Southampton",
  "Sunderland AFC": "Sunderland",
  "Sunderland": "Sunderland",
  "Tottenham Hotspur FC": "Tottenham",
  "Tottenham Hotspur": "Tottenham",
  "West Ham United FC": "West Ham",
  "West Ham United": "West Ham",
  "Wolverhampton Wanderers FC": "Wolves",
  "Wolverhampton Wanderers": "Wolves"
}
```

`src/teams/mapping.py`:

```python
"""Map football-data.org team names to football-data.co.uk canonical names."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import TEAM_NAME_MAP_PATH


def load_team_name_map(path: Path | None = None) -> dict[str, str]:
    map_path = TEAM_NAME_MAP_PATH if path is None else path
    with map_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"team name map empty or invalid: {map_path}")
    return {str(k): str(v) for k, v in data.items()}


def resolve_team_name(fd_org_name: str, mapping: dict[str, str]) -> str:
    name = fd_org_name.strip()
    if name in mapping:
        return mapping[name]
    raise KeyError(f"Unmapped team: {fd_org_name!r}")


def resolve_fixture_teams(
    home: str, away: str, mapping: dict[str, str]
) -> tuple[str, str]:
    return resolve_team_name(home, mapping), resolve_team_name(away, mapping)
```

Empty `src/teams/__init__.py`. Ensure `.gitignore` does **not** ignore `data/reference/`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_team_mapping.py -v`  
Expected: PASS

- [ ] **Step 5: Commit** (only if asked)

---

### Task 2: SQLite store (schema, insert-once, settle)

**Files:**
- Create: `src/store/__init__.py`
- Create: `src/store/schema.py`
- Create: `src/store/predictions.py`
- Create: `tests/test_store.py`

**Interfaces:**
- Consumes: `PREDICTIONS_DB` path
- Produces:
  - `init_db(path: Path | None = None) -> Path`
  - `insert_prediction(conn, row: dict) -> bool` — `True` if inserted; `False` if duplicate key (no overwrite)
  - `settle_prediction(conn, match_id: int, model_version: str, home_goals: int, away_goals: int, result: str) -> bool`
  - `get_prediction(conn, match_id: int, model_version: str) -> dict | None`
  - Helpers: `connect(path) -> sqlite3.Connection`

Row dict keys for insert:  
`match_id, model_version, kickoff_utc, home_team, away_team, matchday, competition, p_home, p_draw, p_away, p_btts_yes, p_btts_no, p_over_25, p_under_25, book_p_home, book_p_draw, book_p_away, book_p_over_25, book_p_under_25, book_source, published_at, as_of_date, status`

- [ ] **Step 1: Write the failing test**

```python
import sqlite3
from pathlib import Path

import pytest

from src.store.predictions import connect, get_prediction, insert_prediction, settle_prediction
from src.store.schema import init_db


def _row(**overrides):
    base = {
        "match_id": 1001,
        "model_version": "dc-test-1",
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


def test_insert_once_and_settle(tmp_path: Path):
    db = tmp_path / "predictions.sqlite"
    init_db(db)
    conn = connect(db)
    assert insert_prediction(conn, _row()) is True
    assert insert_prediction(conn, _row()) is False  # duplicate
    got = get_prediction(conn, 1001, "dc-test-1")
    assert got is not None
    assert got["p_home"] == pytest.approx(0.45)
    assert got["status"] == "scheduled"
    assert settle_prediction(conn, 1001, "dc-test-1", 2, 1, "H") is True
    got2 = get_prediction(conn, 1001, "dc-test-1")
    assert got2["status"] == "settled"
    assert got2["home_goals"] == 2
    assert got2["away_goals"] == 1
    assert got2["result"] == "H"
    assert got2["p_home"] == pytest.approx(0.45)  # probs untouched
    conn.close()


def test_settle_unknown_match_returns_false(tmp_path: Path):
    db = tmp_path / "predictions.sqlite"
    init_db(db)
    conn = connect(db)
    assert settle_prediction(conn, 999, "dc-test-1", 1, 0, "H") is False
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store.py -v`  
Expected: FAIL (module not found)

- [ ] **Step 3: Implement schema + store**

`src/store/schema.py` — create tables `team_name_map` (optional mirror; primary map remains JSON), `model_artefacts`, `predictions` per spec. Use:

```sql
CREATE TABLE IF NOT EXISTS predictions (
  match_id INTEGER NOT NULL,
  model_version TEXT NOT NULL,
  kickoff_utc TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  matchday INTEGER,
  competition TEXT NOT NULL DEFAULT 'PL',
  p_home REAL NOT NULL,
  p_draw REAL NOT NULL,
  p_away REAL NOT NULL,
  p_btts_yes REAL NOT NULL,
  p_btts_no REAL NOT NULL,
  p_over_25 REAL NOT NULL,
  p_under_25 REAL NOT NULL,
  book_p_home REAL,
  book_p_draw REAL,
  book_p_away REAL,
  book_p_over_25 REAL,
  book_p_under_25 REAL,
  book_source TEXT,
  published_at TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  status TEXT NOT NULL,
  home_goals INTEGER,
  away_goals INTEGER,
  result TEXT,
  settled_at TEXT,
  PRIMARY KEY (match_id, model_version)
);
```

Plus `model_artefacts` as in the spec. `insert_prediction` uses `INSERT OR IGNORE` / check `cursor.rowcount`, or `INSERT` catching integrity error — prefer explicit: `SELECT` then insert only if missing, return bool. `settle_prediction` runs `UPDATE predictions SET home_goals=?, away_goals=?, result=?, settled_at=?, status='settled' WHERE match_id=? AND model_version=?` and does **not** touch `p_*` columns.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_store.py -v`  
Expected: PASS

- [ ] **Step 5: Commit** (only if asked)

---

### Task 3: Model artefacts (save / load)

**Files:**
- Create: `src/models/artefacts.py`
- Create: `tests/test_artefacts.py`
- Modify: `.gitignore` — ignore `models/**` contents but allow directory; keep sqlite under processed ignored already. Add:

```gitignore
models/*/
!models/.gitkeep
```

Create empty `models/.gitkeep`.

**Interfaces:**
- Consumes: `DixonColesModel`, `temperature: float`, hyperparams dict
- Produces:
  - `build_model_version(as_of: pd.Timestamp, content_hash: str) -> str` e.g. `dc-20261116-<8char>`
  - `save_artefact(model, temperature, hyperparams, models_dir) -> tuple[str, Path]`  # version, path
  - `load_artefact(model_version: str | None, models_dir) -> tuple[DixonColesModel, float, dict]`  
    — `None` version → latest by `created_at` / directory mtime
  - `teams_in_model(model: DixonColesModel) -> set[str]`

Persist JSON (not pickle) for portfolio clarity: `model.json` with attack/defence dicts, home_advantage, rho, xi, as_of, temperature, hyperparams. Optional `meta.json` with version + hash.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pandas as pd

from src.models.artefacts import load_artefact, save_artefact, teams_in_model
from src.models.dixon_coles import DixonColesModel


def test_artefact_round_trip(tmp_path: Path):
    model = DixonColesModel(
        attack={"Arsenal": 1.1, "Chelsea": 0.9},
        defence={"Arsenal": 0.95, "Chelsea": 1.05},
        home_advantage=1.3,
        rho=-0.05,
        xi=0.0019,
        as_of=pd.Timestamp("2026-11-16"),
    )
    version, path = save_artefact(
        model,
        temperature=1.2,
        hyperparams={"xi": 0.0019, "max_goals": 8},
        models_dir=tmp_path,
    )
    assert path.exists()
    loaded, temperature, hyperparams = load_artefact(version, models_dir=tmp_path)
    assert temperature == pytest.approx(1.2)
    assert loaded.attack["Arsenal"] == pytest.approx(1.1)
    assert "Arsenal" in teams_in_model(loaded)
    assert "Spurs" not in teams_in_model(loaded)
```

(Add `import pytest`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_artefacts.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `src/models/artefacts.py`**

Use `json` + `hashlib.sha256` of canonical JSON for content hash. `load_artefact(None, ...)` picks lexicographically latest version directory matching `dc-*` or reads an `latest` pointer file written by `save_artefact`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_artefacts.py -v`  
Expected: PASS

- [ ] **Step 5: Commit** (only if asked)

---

### Task 4: Fixtures client + 10-day window

**Files:**
- Create: `src/fixtures/__init__.py`
- Create: `src/fixtures/client.py`
- Create: `src/fixtures/window.py`
- Create: `tests/test_fixtures_client.py`

**Interfaces:**
- Consumes: `FOOTBALL_DATA_ORG_TOKEN`, `FD_ORG_BASE`, `FIXTURES_CACHE_DIR`
- Produces:
  - `fetch_pl_matches(date_from: str, date_to: str, *, token: str | None = None, cache_dir: Path | None = None, session=None) -> list[dict]`  
    GET `/competitions/PL/matches?dateFrom=&dateTo=` with header `X-Auth-Token`. Cache JSON to `cache_dir / f"PL_{date_from}_{date_to}.json"`. If cache file exists, return parsed cache (no network).
  - `matches_in_publish_window(matches: list[dict], *, now_utc: datetime, window_days: int = 10) -> list[dict]`  
    Keep matches whose `utcDate` is in `[now_utc, now_utc + window_days days)`.
  - `normalize_match(raw: dict) -> dict` with keys `match_id`, `kickoff_utc`, `home_team`, `away_team`, `matchday`, `status` from FD.org payload (`id`, `utcDate`, `homeTeam.name`, `awayTeam.name`, `matchday`, `status`).

- [ ] **Step 1: Write the failing test**

```python
from datetime import datetime, timezone
from pathlib import Path

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fixtures_client.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement client + window**

On cache miss, `requests.get` with timeout; raise clear error if token missing/empty when network needed. Write cache atomically (write temp then rename).

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fixtures_client.py -v`  
Expected: PASS

- [ ] **Step 5: Commit** (only if asked)

---

### Task 5: `fit` job

**Files:**
- Create: `src/jobs/__init__.py`
- Create: `src/jobs/fit.py`
- Create: `tests/test_fit_job.py`
- Modify: `src/store/predictions.py` — add `upsert_model_artefact(conn, ...)` if not already in Task 2

**Interfaces:**
- Consumes: cleaned matches via `build_matches_table` + `add_features` (or load `data/processed/matches.csv` if present), `fit_dixon_coles`, `save_artefact`, `init_db`
- Produces: CLI `python -m src.jobs.fit` writing artefact + `model_artefacts` row
- Temperature: chronological split — fit DC on all matches with `date < cutoff` where cutoff is the date of the first of the last `CALIBRATION_MIN_ROWS` matches (or temperature `1.0` if fewer than `CALIBRATION_MIN_ROWS` prior rows available for a proper holdout). Holdout: predict raw 1X2 on the last `CALIBRATION_MIN_ROWS` matches after fitting on strictly earlier matches; `fit_temperature` on that holdout; store T on artefact.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pandas as pd
import pytest

from src.jobs.fit import fit_and_save


def _tiny_matches() -> pd.DataFrame:
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City"]
    rows = []
    start = pd.Timestamp("2024-01-01")
    for i in range(40):
        home = teams[i % 4]
        away = teams[(i + 1) % 4]
        rows.append(
            {
                "date": start + pd.Timedelta(days=i),
                "home_team": home,
                "away_team": away,
                "home_goals": i % 3,
                "away_goals": (i + 1) % 3,
                "result": "H" if i % 3 > (i + 1) % 3 else ("A" if i % 3 < (i + 1) % 3 else "D"),
            }
        )
    return pd.DataFrame(rows)


def test_fit_and_save_writes_artefact(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.jobs.fit.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("src.jobs.fit.PREDICTIONS_DB", tmp_path / "predictions.sqlite")
    version, path = fit_and_save(_tiny_matches(), as_of=pd.Timestamp("2024-02-20"))
    assert path.exists()
    assert version.startswith("dc-")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_job.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `fit_and_save` + `__main__`**

`fit_and_save(matches, as_of=None, models_dir=None, db_path=None) -> tuple[str, Path]`.  
CLI loads processed CSV if present else builds via pipeline helpers; calls `fit_and_save`; logs version path.

Register artefact in sqlite `model_artefacts`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fit_job.py -v`  
Expected: PASS

- [ ] **Step 5: Commit** (only if asked)

---

### Task 6: `publish` job

**Files:**
- Create: `src/jobs/publish.py`
- Create: `tests/test_publish.py`

**Interfaces:**
- Consumes: `load_artefact`, `fetch_pl_matches`, `matches_in_publish_window`, `normalize_match`, `load_team_name_map`, `resolve_fixture_teams`, `predict_matches`, `insert_prediction`, `teams_in_model`
- Produces: `publish_slate(model_version: str | None = None, *, now_utc=None, token=None) -> dict` with counts `{fetched, in_window, inserted, skipped_duplicate}`  
  Abort with raised error listing unmapped / missing-from-model teams (no partial insert).  
  Transaction: validate **all** fixtures first; only then insert.

**Critical check:** before predict, for every mapped team assert `team in teams_in_model(model)` — do not rely on Dixon–Coles `1.0` fallback.

- [ ] **Step 1: Write the failing test**

```python
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.models.artefacts import save_artefact
from src.models.dixon_coles import DixonColesModel
from src.jobs.publish import publish_slate
from src.store.schema import init_db


def test_publish_aborts_on_unmapped_team(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    db = tmp_path / "db.sqlite"
    init_db(db)
    model = DixonColesModel(
        attack={"Arsenal": 1.0, "Chelsea": 1.0},
        defence={"Arsenal": 1.0, "Chelsea": 1.0},
        home_advantage=1.3,
        rho=0.0,
        xi=0.0019,
        as_of=pd.Timestamp("2026-11-16"),
    )
    version, _ = save_artefact(model, 1.0, {"xi": 0.0019}, models_dir)

    monkeypatch.setattr("src.jobs.publish.MODELS_DIR", models_dir)
    monkeypatch.setattr("src.jobs.publish.PREDICTIONS_DB", db)
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
```

Also add `test_publish_inserts_and_refuses_duplicate` with fully mapped fixture + model containing both teams; second publish → `inserted==0`, `skipped_duplicate==1`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_publish.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement publish**

Date range: `date_from = now.date()`, `date_to = (now + timedelta(days=PUBLISH_WINDOW_DAYS)).date()` — note FD.org `dateTo` is exclusive per their docs; pass end+1 day if needed so the 10th day is included. Document the chosen convention in a one-line comment.

Apply temperature from artefact via `predict_matches(..., temperature=temperature)`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_publish.py -v`  
Expected: PASS

- [ ] **Step 5: Commit** (only if asked)

---

### Task 7: `settle` + optional `precompute` + docs

**Files:**
- Create: `src/jobs/settle.py`
- Create: `src/jobs/precompute.py`
- Create: `tests/test_settle.py`
- Modify: `README.md` — Fit / publish / settle section (portfolio voice)
- Modify: `.env.example` if needed (token already present)

**Interfaces:**
- `settle_from_api(*, token=None, model_version=None) -> dict` — load unsettled rows from DB (optionally filter version); fetch FINISHED matches by id or by date range covering those kickoffs; `settle_prediction` for each; skip unknown ids.
- `precompute` main: call `fit_and_save` then `publish_slate(latest)`.

- [ ] **Step 1: Write failing settle test** (mock FD.org finished payload; assert probs unchanged)

- [ ] **Step 2: Implement settle + precompute**

- [ ] **Step 3: README** — document:

```bash
# requires FOOTBALL_DATA_ORG_TOKEN in .env
python -m src.jobs.fit
python -m src.jobs.publish
python -m src.jobs.settle
# or
python -m src.jobs.precompute
```

Mention 10-day window, immutable rows, team map path `data/reference/team_name_map.json`.

- [ ] **Step 4: Full pytest**

Run: `pytest -v`  
Expected: all green

- [ ] **Step 5: Commit** (only if asked)

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| FD.org fixtures client + cache | 4 |
| Team-name map fail closed / abort batch | 1, 6 |
| Versioned artefacts; fit ≠ publish | 3, 5, 6 |
| SQLite insert-once + settlement-only updates | 2, 7 |
| 10-day rolling window | 4, 6 |
| Nullable book columns | 2 |
| No paper returns / affiliates / live odds | Global + docs |
| Future fields note (already in spec) | — |
| Kickoff / T−N freeze deferred | — |
| README / portfolio docs | 7 |
| Tests listed in spec | 1–7 |

## Plan self-review notes

- Publish validates the full slate before any insert (no partial silent publish).
- Explicit `teams_in_model` check counters Dixon–Coles’ average-strength fallback for unknown teams.
- FD.org free tier: no odds fetch; book columns remain null.
- `dateTo` exclusivity called out in Task 6 — implementer must verify inclusive 10-day coverage.
- Commits are opt-in per project preference.
