# Public weekend site (Epic 5)

Astro static site for **weekend + past + how-it-works**. Reads the Epic 4 Predictions API only — never fits or publishes.

## Pages

| Route | Job |
|---|---|
| `/` | This weekend’s model vs book 1X2 strips (+ BTTS / O/U) |
| `/past` | Settled week index, latest scorecard, hit/miss waffle peeks |
| `/past/[iso_week]` | One immutable ISO week |
| `/how-it-works` | Short method slides |

## Env

Copy `.env.example` → `.env`:

```bash
PUBLIC_PREDICTIONS_API_BASE=http://127.0.0.1:8000
```

On the API side (repo root `.env`), allow the Astro origin:

```bash
API_CORS_ORIGINS=http://localhost:4321,http://127.0.0.1:4321
```

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

## Ownership

| Area | Owner |
|---|---|
| Routes, IA, API client, match row shells | UI/UX (this package) |
| Tokens, HeroUI theme, Motion count-ups | Designer — see `TODO(Designer)` |
| Waffle binding, Brier/log-loss, bucket math, calibration curve | Data Scientist — `src/lib/metrics.ts` + Past peeks |

Design direction: Context `docs/epic5-design-system-direction.md` (Version C). Anti-slop gate: `docs/epic5-anti-slop-checklist.md`.
