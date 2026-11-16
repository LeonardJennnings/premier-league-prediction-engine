# Thin read-only predictions API

**Date:** 16 Nov 2026  
**Status:** Ready for implementation  
**Related:** [PRD](../prd.md) · [Project context](../project-context.md) · [Fixtures ingest and prediction store](fixtures-ingest-prediction-store.md)

## Summary

Expose a small **FastAPI** HTTP surface that **only reads** the append-only SQLite prediction store. Next.js (Epic 5) and any tools consume stored rows — never fit, never `POST /predict`, never call Dixon–Coles on the request path.

## Motivation

Epics 1–3 deliver evaluation, a goals model with derived markets, and an immutable publish/settle ledger. Without a thin read API, the public UI would either talk to lab code (Streamlit / model process) or reach into SQLite directly. The PRD architecture is precompute → store → API → UI; this epic is the missing middle.

## Scope

**In scope**

- FastAPI app under `src/api/` with a thin service layer over store reads
- Endpoints for upcoming slate, past ISO weeks (scorecard), week index, and model metadata
- Default to the latest `model_version`, with optional `?model_version=` pin
- Clear HTTP errors for missing week / unknown version / unreadable DB
- CORS suitable for a local Next.js app
- OpenAPI-backed contract documented for Epic 5
- pytest coverage via FastAPI `TestClient` and a temp seeded DB

**Out of scope**

- Any write / fit / publish / settle HTTP
- Auth, rate limiting, API keys
- Postgres, CDN, response caching layers
- Paper EV / value metrics, affiliates
- Streamlit changes
- Next.js UI (Epic 5)
- Kickoff / T−N freeze policies (still deferred from Epic 3)

## Design decisions

| Topic | Decision | Rationale |
|---|---|---|
| Framework | FastAPI (+ existing uvicorn) | OpenAPI for the UI contract; Starlette is already a transitive dep |
| Layering | Routes → service → store | Service owns version resolve, upcoming window, ISO-week buckets; routes stay dumb |
| Slate grouping | Rolling upcoming + past by ISO week of kickoff | Matches Epic 3’s 10-day publish window; midweek games are not orphans |
| Default version | Latest `model_artefacts` row; optional pin | Public UI needs zero params; reviewers can inspect history |
| Empty upcoming | `200` + `matches: []` | “Nothing published yet” is not a client error |
| Missing week / version | `404` | Clear for UI navigation |
| Unreadable / missing DB | `503` | Ops signal; never looks like a model failure mid-request |
| Auth | None for v1 | Public entertainment surface |
| Books | Pass through nullable `book_*` | Epic 3 may leave books null until co.uk has E0 rows |

## Architecture

```text
Next.js (Epic 5)  →  FastAPI (read-only)  →  predictions read service  →  SQLite store
                                                                     ↗ model_artefacts
Jobs / Streamlit remain on write and lab paths; this app never imports fit/publish.
```

| Module | Owns | Does not |
|---|---|---|
| `src/api/app.py` | App factory, CORS, router mount, lifespan DB path | SQL; slate rules |
| `src/api/routes/*.py` | HTTP status codes, query params, response models | SQL; version/week logic |
| `src/api/schemas.py` | Pydantic request/response shapes (OpenAPI) | Persistence |
| `src/api/service.py` | Resolve latest version; upcoming filter; ISO-week list/detail | HTTP; writes |
| `src/store/` | Read helpers for queries used by the service | HTTP; business grouping |

Existing insert-once / settlement rules in the store are unchanged. Epic 4 only **adds** read helpers the service needs.

## Endpoints

Base path prefix: `/v1`.

| Method | Path | Behaviour |
|---|---|---|
| `GET` | `/health` | Liveness; does not require a populated DB |
| `GET` | `/v1/predictions/upcoming` | Upcoming slate for resolved `model_version` |
| `GET` | `/v1/predictions/weeks` | List ISO weeks that have ≥1 settled row for the version |
| `GET` | `/v1/predictions/weeks/{iso_week}` | Settled rows for that week + version; `404` if none |
| `GET` | `/v1/models/latest` | Metadata for the latest artefact row |
| `GET` | `/v1/models/{model_version}` | Metadata for one version; `404` if unknown |

**Query params (where noted):**

- `model_version` (optional) on upcoming, weeks list, and week detail — pin; omit → latest.

No other routes in v1. In particular: no `POST /predict`, no job triggers.

## Slate and week rules

### Version resolve

1. If `model_version` is provided, require a matching `model_artefacts` row. Missing → `404`.
2. If omitted, take the latest artefact by `created_at` descending, with `model_version` as tie-break (lexicographic descending). No artefacts → `404` on prediction endpoints that need a version (health still `200`).

Every prediction response echoes the resolved `model_version`.

### Upcoming

- `kickoff_utc >= now` (UTC)
- `status IN ('scheduled', 'postponed')`
- Kickoff within `PUBLISH_WINDOW_DAYS` (config; currently `10`) from now
- Same `model_version`
- Order by `kickoff_utc`, then `match_id`
- Empty → `200` with `matches: []`

### Past weeks

- `iso_week` format: `YYYY-Www` (ISO-8601 week), derived from `kickoff_utc` in UTC
- Week detail (v1): rows with `status = 'settled'` for that week and version
- Week list: distinct ISO weeks that have at least one settled row for the version, newest first
- Unknown or empty week id for that version → `404`

Cancelled / postponed rows are not required on the past scorecard in v1 (upcoming already surfaces relevant postponements).

## Response shapes

### Match object (shared)

| Field group | Fields |
|---|---|
| Identity | `match_id`, `home_team`, `away_team`, `kickoff_utc`, `matchday`, `competition`, `status` |
| Model | `p_home`, `p_draw`, `p_away`, `p_btts_yes`, `p_btts_no`, `p_over_25`, `p_under_25` |
| Book (nullable) | `book_p_home`, `book_p_draw`, `book_p_away`, `book_p_over_25`, `book_p_under_25`, `book_source` |
| Audit | `published_at`, `as_of_date`, `model_version` |
| Settlement (nullable) | `home_goals`, `away_goals`, `result`, `settled_at` |

### Envelope examples

**Upcoming / week detail**

```json
{
  "model_version": "2026-11-16T12:00:00Z-abcd1234",
  "matches": [ { "...": "Match object" } ]
}
```

Week detail also includes `"iso_week": "2026-W46"`.

**Weeks list**

```json
{
  "model_version": "…",
  "weeks": [
    { "iso_week": "2026-W46", "match_count": 10 }
  ]
}
```

**Model metadata**

```json
{
  "model_version": "…",
  "as_of_date": "2026-11-15",
  "created_at": "…",
  "artefact_path": "models/…",
  "hyperparams": { }
}
```

`hyperparams` is parsed JSON when stored as text; `null` if absent.

## Error behaviour

| Condition | Status | Notes |
|---|---|---|
| Unknown `model_version` pin | `404` | Body explains version not found |
| No artefacts when defaulting latest | `404` | Nothing to serve yet |
| Unknown / empty `iso_week` for version | `404` | Clear for UI nav |
| Malformed `iso_week` | `422` | FastAPI validation |
| DB file missing or unreadable | `503` | Ops; not a model failure |
| Empty upcoming slate | `200` | `matches: []` |

Error bodies use a small consistent shape, e.g. `{"detail": "…"}` (FastAPI default is fine).

## Run and config

- Entry: `uvicorn src.api.app:app --reload` (from repo root, venv active)
- DB path: existing `PREDICTIONS_DB` (`data/processed/predictions.sqlite`)
- CORS: allow localhost Next.js origins (e.g. `http://localhost:3000`) via config/env; no credentials required for v1
- Add `fastapi` to `requirements.txt` (pin a current stable); uvicorn already present

## Testing strategy

- Temp SQLite seeded with two `model_version`s, a mix of upcoming `scheduled`/`postponed` and settled rows across two ISO weeks
- Default version resolves to latest; pin returns the older set
- Unknown version → `404`; unknown week → `404`; empty upcoming → `200 []`
- Response includes null `book_*` when seeded null
- OpenAPI lists only the documented GET routes (no write methods)
- Existing Epic 1–3 tests remain green

## Success criteria

1. API returns stored upcoming predictions for a published slate without importing model/fit code on the request path.
2. API returns historical settled rows for at least one ISO week.
3. Missing week and unknown `model_version` produce clear `404`s; empty upcoming is `200`.
4. Contract (this spec + generated OpenAPI) is enough for Epic 5 to build against.
5. README documents how to run the API locally.

## References

- Product epic: [`prd.md`](../prd.md) Epic 4
- Store schema and row lifecycle: [`fixtures-ingest-prediction-store.md`](fixtures-ingest-prediction-store.md)
- Epic 3 book null behaviour: [`notes/2026-11-16-epic3-known-limitations.md`](../notes/2026-11-16-epic3-known-limitations.md)
