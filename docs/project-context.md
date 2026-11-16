# Prediction Engine — Project Context

Last updated: 16 Nov 2026 (Europe/Amsterdam)

## What this is

A public Premier League prediction project: a free, model-first weekend slate with an honest published track record, built as a small production-shaped ML system (eval, precompute, API, UI).

Methodological inspiration: probability estimation versus the market, with humility about variance (the same family of thinking often associated with quantitative sports modelling work such as Bill Benter’s).

**Primary audience for these docs:** engineers and hiring managers reviewing the repository. The live product is secondary until a public surface ships. Build detail for the current window lives in [`prd.md`](prd.md).

## Goals by horizon

| Horizon | Priority |
|---|---|
| Short term (~4–6 weeks) | Ship a credible public surface with honest evaluation |
| Medium term | Stronger match-level quality and a durable public site |
| Longer term | Maintain the free product; UK/Ireland affiliates only when product quality and compliance are ready |

If priorities collide in the current window: shipping a trustworthy eval + public slate comes first. Monetization packaging comes after something real is live.

## Users

- Casual Premier League fans who want calm “what’s likely this weekend?” framing
- Recreational bettors who want probabilities and market context, not tipster pressure
- Repository reviewers assessing problem framing, evaluation honesty, and shipping judgment

## Product shape

**Free core.** Bookmaker affiliates are a later product note for UK/Ireland when geography and programs allow — not part of the current build. Live odds and deeper value tooling can wait; v1 uses free/historical odds for backtests.

**Core outputs (current window):**

- Match winner probabilities (1X2)
- Goals markets (BTTS, over/under 2.5), preferably derived from the same scoreline grid
- Model vs book implied probabilities, with careful claims

**Later:** ACCAs / bet builders only after a singles track record exists; always with expected hit-rate framing and disclaimers — never as sure things.

**Public trust:** publish predictions before kickoff, settle after, never delete misses. Entertainment / not financial advice.

## Architecture direction

- Python engine is the source of truth
- Predictions are precomputed on a schedule
- A thin API serves stored rows
- Next.js is the public UI (Streamlit remains a lab surface)
- The model does not run in the browser

**Current-window done definition:** engine + API + thin UI with weekend page, past scorecard, and how-it-works; markets include 1X2 plus BTTS and O/U 2.5. Ship bar detail in [`prd.md`](prd.md).

## Scope for v1

- League: Premier League only
- Brand: “Premier League Prediction Engine” until a consumer brand is needed
- Geography: build product and traffic first; UK/Ireland affiliates later if warranted
- Odds: historical/free for backtests first; live odds later
- Quality order: calibration → 1X2 quality → BTTS/O/U coverage → value vs book → ACCAs after singles

## Documentation map

1. This file — durable product and architecture decisions
2. [`prd.md`](prd.md) — current build window, epics, ship bar
3. [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) — how models are scored
4. [`specs/`](specs/) — technical designs for in-flight work

## Current build decisions (16 Nov 2026)

Detail and epics: [`prd.md`](prd.md).

- Lab-honest calibrated weekend probabilities; model and book shown side by side; no tipster/ACCA product
- Evaluation: walk-forward + calibration plots; beating the book is stretch, not a ship blocker
- Fixtures: football-data.org free API; predictions saved once — past weekends never silently rewrite history
- Markets: 1X2 and BTTS + O/U 2.5 (prefer scoreline-grid derivation)
- Site: weekend page + past scorecard + how-it-works; affiliates off
- Architecture: Python precompute → store → thin API → Next.js; Streamlit lab only

## Explicitly later

- Consumer brand and heavy go-to-market
- Soft launch / social timing tied to fixtures
- Broader launch aimed at a following season
- Live odds API, premium tier, ACCA productization
- Affiliate wiring once traffic, trust, and compliance are in place
