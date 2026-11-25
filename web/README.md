# Public weekend site (Epic 5)

Astro static site for the weekend slate, the past scorecard, and a short method page. A production build reads the committed ledger files and does not call an API.

| File | What the page uses it for |
|---|---|
| `src/data/ledger-upcoming.json` | Matches still to kick off, grouped by matchday |
| `src/data/ledger-settled.json` | Finished, postponed, and pending matches |
| `src/data/ledger-scorecard.json` | Counts and averages, each with its match count |

`python -m src.jobs.daily` rewrites those three files. Rebuild the site after that commit.

## Pages

| Route | Job |
|---|---|
| `/` | This matchday's most likely score, most likely result, expected goals, and the book beside the model |
| `/past` | Every settled match, predicted against actual, plus the scorecard |
| `/past/[matchday]` | One matchday, including a Monday game that belongs to that round |
| `/how-it-works` | Short method notes |

Sample rows appear only when `PUBLIC_SHOW_SAMPLE=1`. Leave that unset for a production build.

The site does not call `127.0.0.1` unless `PUBLIC_PREDICTIONS_API_BASE` is set. The public pages do not need that variable.

## Hosting

The build is a static `dist/` directory. No server and no fixture token. On Vercel, set the project root to `web`, the build command to `npm run build`, and the output directory to `dist`. Do not add `FOOTBALL_DATA_ORG_TOKEN` or an API base.

## Older snapshot export

`src/data/upcoming-snapshot.json` is the 17 Nov export kept for the freshness check. The pages above do not read it. Cadence: [`docs/OPS_CADENCE.md`](../docs/OPS_CADENCE.md).

## Run locally

```bash
cd web
npm install
npm run dev
```

Open http://localhost:4321. The pages show the committed ledger. They do not fall back to sample rows unless `PUBLIC_SHOW_SAMPLE=1`.

```bash
npm run build    # static export → dist/
npm run preview
npm run check:metrics
```

`check:metrics` checks the shared metric maths, the slate rules, and the ledger pages.

## Ownership

| Area | Owner |
|---|---|
| Routes, IA, API client, match row shells | UI/UX (this package) |
| Tokens, HeroUI theme, Motion count-ups | Designer — see `TODO(Designer)` |
| Waffle binding, Brier/log-loss, bucket math, calibration curve | Data Scientist — `src/lib/metrics.ts` + Past peeks |

Design direction: Context `docs/epic5-design-system-direction.md` (Version C). Anti-slop gate: `docs/epic5-anti-slop-checklist.md`.
