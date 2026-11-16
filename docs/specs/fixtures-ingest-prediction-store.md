# Fixtures ingest and immutable prediction store

**Date:** 16 Nov 2026  
**Status:** Ready for implementation  
**Related:** [PRD](../prd.md) · [Project context](../project-context.md) · [Goals model and derived markets](goals-model-derived-markets.md) · [Evaluation protocol](../EVAL_PROTOCOL.md)

## Summary

Pull upcoming Premier League fixtures from the **football-data.org** free API, fit a versioned Dixon–Coles artefact from historical football-data.co.uk data, publish calibrated market probabilities into an **append-only SQLite** store, and settle finished matches without rewriting published probabilities.

Public past history never silently changes when the model is retrained. A later thin API (Epic 4) and Next.js UI (Epic 5) read stored rows only.

## Motivation

Epics 1–2 deliver walk-forward evaluation and a Dixon–Coles scoreline grid with derived 1X2, BTTS, and over/under 2.5. The product still lacks a production-shaped write path: upcoming fixtures, a durable prediction ledger, and settlement. Without that path, there is no honest weekend (or midweek) slate and no immutable scorecard.

## Scope

**In scope**

- football-data.org fixtures client with response caching and rate-limit-aware use of `FOOTBALL_DATA_ORG_TOKEN`
- Canonical team-name mapping between football-data.org and football-data.co.uk names (fail closed)
- Versioned model artefacts from an explicit `fit` step
- `publish` that loads an artefact and never refits
- SQLite prediction store with insert-once probability rows
- Settlement from football-data.org finished matches (batch / manual; not a live score service)
- Rolling **10-day** upcoming fixture window (weekend and midweek)
- Nullable bookmaker probability columns for a future odds source
- Documentation in a professional portfolio voice

**Out of scope**

- Thin HTTP API and Next.js UI
- Live odds ingestion and paid football-data.org Odds Add-On wiring
- Paper-return / expected-value calculations
- Affiliate fields or monetization CTAs
- Postgres (SQLite for v1; schema kept portable)
- Kickoff or T−N publish-freeze policies (deferred; see Future fields)

## Design decisions

| Topic | Decision | Rationale |
|---|---|---|
| Store | SQLite file under `data/processed/` | Zero ops for v1; thin store module keeps a later Postgres move small |
| Train vs publish | Clean split: `fit` writes artefact; `publish` loads it | Reproducible public rows; retrain is an explicit version bump |
| Combined CLI | Optional `precompute` runs fit then publish as two steps | Convenience without blurring boundaries |
| Immutability | Probabilities frozen at insert | Simplest honesty story for v1 |
| Book columns | Nullable `book_*` + `book_source` | Free football-data.org has no odds; schema stays ready |
| Settlement source | football-data.org `FINISHED` matches | Same `match_id` as fixtures; ~minutes of free-tier delay is acceptable for a scorecard |
| Slate window | Rolling next 10 days | Covers weekend and midweek without cluster heuristics |
| Team names | Explicit map; unknown names abort the publish batch | Wrong names are worse than a failed job |
| Architecture | Modular client → map → fit/publish/settle → store | Testable boundaries; Epic 4 only needs the store |

## Architecture

```text
football-data.org  →  fixtures client (+ cache)
                   →  team-name map (org → co.uk canonical)
                   →  predict_matches(loaded artefact)
                   →  SQLite predictions (insert-once)

historical E0 CSV  →  fit  →  models/<model_version>/
settle job         →  FINISHED matches  →  settlement fields only
```

| Module | Owns | Does not |
|---|---|---|
| Fixtures client | Auth, caching, PL match payloads | Predictions, DB writes |
| Team map | Canonical table + lookup | Fuzzy guessing |
| `fit` | Artefact on disk + `model_artefacts` row | Publishing prediction rows |
| `publish` | Load artefact → 10-day slate → insert-once | Refitting |
| Store | Schema, immutability, settlement updates | HTTP serving |
| `settle` | Goals, result, `settled_at`, status | Changing probabilities |

Existing `predict_matches` from Epic 2 remains the batch predict entrypoint. Temperature applies to 1X2 only; BTTS and over/under stay on the raw scoreline grid.

## Schema

**File:** `data/processed/predictions.sqlite` (gitignored; path via config).

### `team_name_map`

| Column | Notes |
|---|---|
| `fd_org_name` | PK — name/string key from football-data.org |
| `canonical_name` | football-data.co.uk spelling used by the model |

### `model_artefacts`

| Column | Notes |
|---|---|
| `model_version` | PK |
| `as_of_date` | Training cutoff date |
| `created_at` | UTC timestamp |
| `artefact_path` | Path under `models/` |
| `hyperparams` | JSON text (ξ, max goals, L2, temperature, etc.) |

### `predictions`

| Group | Columns |
|---|---|
| Identity | `match_id`, `model_version`, `kickoff_utc`, `home_team`, `away_team`, `matchday` (nullable), `competition` |
| Model | `p_home`, `p_draw`, `p_away`, `p_btts_yes`, `p_btts_no`, `p_over_25`, `p_under_25` |
| Book (nullable) | `book_p_home`, `book_p_draw`, `book_p_away`, `book_p_over_25`, `book_p_under_25`, `book_source` |
| Audit | `published_at`, `as_of_date`, `status` |
| Settlement | `home_goals`, `away_goals`, `result`, `settled_at` |

**Primary key:** `(match_id, model_version)`.

**Status values:** `scheduled` at publish; `settled` after settlement.

### Artefact layout

`models/<model_version>/` holds team strengths, ρ, temperature, as-of metadata, hyperparams, and a content hash. Publish loads this directory only.

## Row lifecycle

1. **Publish** inserts a `scheduled` row; probabilities are immutable thereafter for that `(match_id, model_version)`.
2. **Settle** updates goals, result, `settled_at`, and `status` only.
3. Re-publish of the same key is refused (no silent overwrite).
4. Intentional corrections use a **new** `model_version` row; old rows are never deleted or probability-rewritten.
5. Upcoming slate reads: kickoffs in the future (or unsettled within the window). Scorecard reads: prefer `status=settled`.

## Jobs

| Command | Behaviour |
|---|---|
| `fit` | Fit Dixon–Coles (+ temperature per existing policy) on historical E0; write artefact + `model_artefacts` row |
| `publish` | Load named or `latest` artefact; fetch PL fixtures for the next 10 days; map names; predict; insert-once |
| `settle` | For published unsettled `match_id`s, pull finished matches; write settlement fields only |
| `precompute` | Optional: run `fit` then `publish` as two explicit steps |

Fixture API responses are cached under `data/raw/fixtures/` so reruns can avoid unnecessary network calls within rate limits (free tier: 10 requests/minute; scores delayed on the order of minutes — acceptable for settlement, not used as a live feed).

## Failure rules

- Missing `FOOTBALL_DATA_ORG_TOKEN` → fail fast with a clear message.
- Any fixture with an unmapped team name → **abort the publish batch** and log the football-data.org name(s).
- Team present in fixtures but absent from the loaded artefact → **abort the publish batch**; do not invent strengths or publish a partial slate.
- Duplicate `(match_id, model_version)` → refuse overwrite; leave the existing row.
- Settlement for an unknown `match_id` → log and skip; never insert a settlement-only row.
- No public API path updates probabilities after insert.

## Testing strategy

- Team map resolves known aliases; unknown names fail closed.
- Store insert-once; second insert refused; settlement cannot change probability columns.
- Publish window includes fixtures inside 10 days and excludes those outside.
- Fit → load round-trip preserves `model_version` metadata and predictability for a sample match.
- Fixtures client uses cache on repeat calls (HTTP mocked in unit tests).
- Existing Epic 1–2 tests remain green.

## Success criteria

1. A manual or scheduled `publish` writes a PL slate to SQLite that survives process restart.
2. Retraining and publishing under a new `model_version` leaves prior rows unchanged.
3. `settle` marks finished matches with scores without altering stored probabilities.
4. Unmapped team names cannot produce a partial silent publish.
5. Spec and README describe the pipeline clearly for portfolio reviewers.

## Future fields and deferred policies

Documented for later work; **not** added as schema columns or jobs in this epic.

| Topic | Direction |
|---|---|
| Paper returns | Derive on read from model probs + book odds + settled result (UI or eval helper); entertainment / paper metrics framing only |
| Live or reference book odds | Fill nullable `book_*` when a source is chosen (paid football-data.org Odds Add-On or another vendor) |
| Affiliates | Separate product change when monetization is in scope |
| Freeze at kickoff | Allow refresh until kickoff, then lock — revisit once book timing matters |
| Freeze at T−N hours | Publish refresh window ending N hours before kickoff — same deferral |

## References

- Product epic and ship bar: [`prd.md`](../prd.md)
- Model and markets: [`goals-model-derived-markets.md`](goals-model-derived-markets.md)
- football-data.org pricing (free fixtures; odds as paid add-on): https://www.football-data.org/pricing
- football-data.org API docs: https://docs.football-data.org/general/v4/index.html
