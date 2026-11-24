# Ops cadence

Lab ledger for Premier League probabilities. Publish freezes a model version. Settle writes scores and status onto those rows. Probabilities stay as first inserted. Book lines stay null when the free football-data.co.uk file has no Premier League rows. This page is the operator checklist.

`FOOTBALL_DATA_ORG_TOKEN` is the fixture token. Historical `pipeline` and `fit` do not need it. `publish`, `settle`, and the publish half of `precompute` do.

## Where the token lives

Use the same name in each place. It matches `.env.example`.

| Place | Who reads it |
|---|---|
| Local `.env` (gitignored) | Laptop `publish` / `settle` |
| Cursor Cloud Agent environment secret | Cloud agents on a VM that booted after the secret was saved |
| GitHub Actions secret | [Slate refresh](../.github/workflows/slate-refresh.yml) only |

[CI](../.github/workflows/ci.yml) runs pytest and the Astro build. The token is read by **Slate refresh** (`publish` and `settle` only).

Cursor published-app secrets are a different store. The static site reads the committed upcoming snapshot (or a live API). Leave the token off the published app.

Never paste the token into chat, a commit, or a log. Local and Actions checks print `TOKEN_PRESENT=yes` or a missing-secret error.

## When to refresh

| When | Command | What you commit |
|---|---|---|
| Before a matchday | `python scripts/refresh_slate.py --mode publish` | `web/src/data/upcoming-snapshot.json` after you review it |
| After a matchday, on the machine that holds the ledger | `python scripts/refresh_slate.py --mode settle` | `web/src/data/scorecard-snapshot.json` when you want that export kept |
| Any time | `python scripts/refresh_slate.py --mode freshness` | Nothing. Exit 2 means every committed kickoff is already in the past |

`publish` runs `python -m src.pipeline`, then `python -m src.jobs.precompute`, then `scripts/export_upcoming_snapshot.py`.

`settle` runs `python -m src.jobs.settle`, then `scripts/export_scorecard_snapshot.py`. A revised probability is a new model version.

## Snapshot freshness

`scripts/check_snapshot_freshness.py` reads `web/src/data/upcoming-snapshot.json`.

| Exit | Meaning |
|---|---|
| 0 | `status=fresh`. At least one kickoff is still at or after `--now` |
| 2 | `status=stale` or `status=empty`. Re-publish before treating the file as this week's slate |
| 1 | The file is missing or is not an upcoming payload |

`scheduled_after_kickoff` counts rows still marked `scheduled` after their kickoff. That count is the committed file ageing in place.

The snapshot exported on 2026-11-17 (`dc-20261108-a7b8d765`) runs from `2026-11-21T12:30:00Z` through `2026-11-23T20:00:00Z`. From 2026-11-24 onwards that file is stale until a new publish is exported. CI stays green when the snapshot is stale. Freshness is an ops signal.

## Scorecard export

`web/src/data/scorecard-snapshot.json` is the settled-weeks payload: model version, `exported_at`, and each ISO week with its matches. Exit 2 writes the file with zero weeks.

Past pages still read the live API, then the labelled sample in `web/src/lib/sample.ts`. Wiring this file into those pages is a separate change. An empty ledger writes zero weeks.

## GitHub Actions stub

Workflow: **Slate refresh** (`.github/workflows/slate-refresh.yml`).

Run it by hand from the Actions tab. Choose `freshness`, `publish`, or `settle`.

| Mode | Secret | Runner behaviour |
|---|---|---|
| freshness | none | Same check as the local script |
| publish | `FOOTBALL_DATA_ORG_TOKEN` | Refit, publish, export. Uploads the snapshot as an artifact |
| settle | `FOOTBALL_DATA_ORG_TOKEN` | Stops before the fixture call when `data/processed/predictions.sqlite` is absent |

Limits, so this stays a stub:

- The workflow uploads artifacts and leaves `main` unchanged. Review the artifact, then commit the JSON you want kept.
- Each Actions run starts without `predictions.sqlite` (`models/` is gitignored too). Settle on a fresh runner exits 2.
- Cron stays off. Add a schedule after a manual `publish` run has succeeded and the SQLite file has a home between runs.

## Related commands

```bash
python scripts/check_snapshot_freshness.py
python scripts/check_snapshot_freshness.py --now 2026-11-24T12:00:00Z
python scripts/export_scorecard_snapshot.py
python -m src.jobs.publish
python -m src.jobs.settle
```
