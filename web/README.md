# Public weekend site (Epic 5)

Astro static site for **weekend + past + how-it-works**. Reads the Epic 4 Predictions API only — never fits or publishes.

## Pages

| Route | Job |
|---|---|
| `/` | This weekend’s model vs book 1X2 strips (+ BTTS / O/U; book O2.5 when present) |
| `/past` | Settled week index, latest scorecard, hit/miss waffle peeks |
| `/past/[iso_week]` | One immutable ISO week |
| `/how-it-works` | Short method slides + store-derived peeks |

## Env

Copy `.env.example` → `.env`:

```bash
PUBLIC_PREDICTIONS_API_BASE=http://127.0.0.1:8000
```

On the API side (repo root `.env`), allow the Astro origin **and** whatever origin serves the static `dist/` (needed for client refresh):

```bash
API_CORS_ORIGINS=http://localhost:4321,http://127.0.0.1:4321
```

## Freshness (SSG + light client refresh)

Pages fetch the API **at build time**. Weekend preference order:

1. **Live API** — `GET /v1/predictions/upcoming` when reachable and non-empty
2. **Committed snapshot** — `src/data/upcoming-snapshot.json` (real publish export; not sample)
3. **Labelled sample** — `src/lib/sample.ts` only when both of the above are empty/missing

Past peeks still fall back to sample settled rows when the store has no settled weeks.

To keep `dist/` off the sample weekend slate when the API is not running at build (Cursor publish-app, static host):

1. After `python -m src.jobs.precompute` (or `publish`), run:
   ```bash
   python scripts/export_upcoming_snapshot.py
   ```
   Commit the updated `web/src/data/upcoming-snapshot.json`.
2. **Client refresh** — Weekend (`LiveWeekend`), Past peeks (`LivePastPeeks`), and Method peeks (`MethodPeeks`) re-fetch `PUBLIC_PREDICTIONS_API_BASE` after hydrate when a live API exists. CORS must allow the site origin.
3. **Rebuild on publish** — Still rebuild (or re-deploy) after refreshing the snapshot / settle so:
   - `/past/[iso_week]` static paths include new ISO weeks (`getStaticPaths`)
   - First paint / no-JS viewers see real rows without waiting on the island
4. **Ideal recipe with live API** — API up → export snapshot (optional but durable) → `cd web && npm run build` with `PUBLIC_PREDICTIONS_API_BASE` → deploy `dist/` → keep API reachable for client refresh when you have one.

## Run locally

```bash
# Terminal A — API (repo root, venv active)
uvicorn src.api.app:app --reload

# Terminal B — site
cd web
npm install
npm run dev
```

Open http://localhost:4321. If the API is down or the slate is empty, pages fall back to a labelled sample slate (`src/lib/sample.ts` + `sample-settled.json`).

### Empty store → live API sample

From the **repo root** (Python deps installed):

```bash
python scripts/seed_sample_predictions.py
uvicorn src.api.app:app --reload
```

Then `cd web && npm run dev`. Peeks use every settled week from the API; strips use `/v1/predictions/upcoming`.

```bash
npm run build    # static export → dist/
npm run preview
npm run check:metrics
```

`check:metrics` imports `src/lib/metrics.ts` directly (no duplicated formulae).

## Ownership

| Area | Owner |
|---|---|
| Routes, IA, API client, match row shells | UI/UX (this package) |
| Tokens, HeroUI theme, Motion count-ups | Designer — see `TODO(Designer)` |
| Waffle binding, Brier/log-loss, bucket math, calibration curve | Data Scientist — `src/lib/metrics.ts` + Past peeks |

Design direction: Context `docs/epic5-design-system-direction.md` (Version C). Anti-slop gate: `docs/epic5-anti-slop-checklist.md`.
