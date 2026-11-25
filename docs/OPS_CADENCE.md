# Ops cadence

Lab ledger for Premier League probabilities. Publish freezes a model version. Settle writes scores and status onto those rows. Probabilities stay as first inserted. Book lines stay null when the free football-data.co.uk file has no Premier League rows. This page is the operator checklist.

`FOOTBALL_DATA_ORG_TOKEN` is the fixture token. Historical `pipeline` and `fit` do not need it. `publish`, `settle`, and the publish half of `precompute` do.

## Where the token lives

Use the same name in each place. It matches `.env.example`.

| Place | Who reads it |
|---|---|
| Local `.env` (gitignored) | Laptop `publish` / `settle` |
| Cursor Cloud Agent environment secret | Cloud agents on a VM that booted after the secret was saved |
| GitHub Actions secret | [Daily ledger](../.github/workflows/daily-ledger.yml) and [Slate refresh](../.github/workflows/slate-refresh.yml) |

[CI](../.github/workflows/ci.yml) runs pytest and the Astro build. It does not read the token. The daily job and the manual slate refresh do.

Cursor published-app secrets are a different store. The static site reads the committed upcoming snapshot (or a live API). Leave the token off the published app.

Never paste the token into chat, a commit, or a log. Local and Actions checks print `TOKEN_PRESENT=yes` or a missing-secret error.

## When to refresh

| When | Command | What you commit |
|---|---|---|
| Before a matchday | `python scripts/refresh_slate.py --mode publish` | `web/src/data/upcoming-snapshot.json` after you review it |
| After a matchday, on the machine that holds the ledger | `python scripts/refresh_slate.py --mode settle` | `web/src/data/scorecard-snapshot.json` when you want that export kept |
| Any time | `python scripts/refresh_slate.py --mode freshness` | Nothing. Exit 2 means every committed kickoff is already in the past |

`publish` runs `python -m src.pipeline`, then `python -m src.jobs.precompute`, then `scripts/export_upcoming_snapshot.py`.

`settle` runs `python -m src.jobs.settle`, then `scripts/export_scorecard_snapshot.py`. The public ledger keeps the first line for a match. A newer fit does not replace it.

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

## Daily ledger

One command does the day's work. It settles finished matches, refits only when a new result is later than the current model, publishes fixtures in the next 7 days that do not already have a line, captures book prices, and rebuilds the site files.

```bash
python -m src.jobs.daily --dry-run
python -m src.jobs.daily
python -m src.jobs.daily --commit
```

`--dry-run` prints how many rows it would add, settle, and price-fill. It does not write a file. A second real run with no new scores does not create a commit. An unknown team name or a failed fixtures call writes nothing and logs `ALERT: daily ledger failed`.

Workflow: **Daily ledger** (`.github/workflows/daily-ledger.yml`). It runs at 06:00 UTC and can be started by hand. The manual run has a dry-run switch. Only one run is allowed at a time. On failure it opens an issue titled `Daily ledger failed`.

The schedule needs a repository Actions secret named `FOOTBALL_DATA_ORG_TOKEN`. The workflow asks for permission to write contents and issues. If a push to `main` is rejected, allow GitHub Actions to push, or the failed run will open that issue.

The same command runs on a machine that already has the token in its environment. The site files it writes are `web/src/data/ledger-upcoming.json`, `web/src/data/ledger-settled.json`, and `web/src/data/ledger-scorecard.json`. They are grouped by matchday. A postponed match stays in the list and is left out of the averages.

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
- Each Actions run starts without `predictions.sqlite`. Fitted `models/<version>/model.json` and `data/ledger/` are committed. Settle on a fresh runner still exits 2 until that local store exists.
- The daily ledger workflow is the scheduled job. This stub is only for a one-off artifact.

## Related commands

```bash
python scripts/check_snapshot_freshness.py
python scripts/check_snapshot_freshness.py --now 2026-11-24T12:00:00Z
python scripts/export_scorecard_snapshot.py
python -m src.jobs.publish
python -m src.jobs.settle
```
