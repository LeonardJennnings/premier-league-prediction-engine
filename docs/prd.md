# Product PRD — Premier League Prediction Engine (4–6 week build)

**Audience:** Leonard (builder) + reviewers of a portfolio project (hiring managers, engineers)  
**Date:** 16 Nov 2026  
**Horizon:** one focused build window (~4–6 weeks)  
**Sources:** [`project-context.md`](project-context.md), [`engine-deep-research.md`](engine-deep-research.md), [`overnight-research-pack.md`](overnight-research-pack.md)  
**Not this doc:** app code, pull requests, legal advice.

---

## 1. Problem and goal

### Problem

Premier League fans and recreational bettors see a lot of tipster noise (sure-thing parlays, ACCAs, rewritten histories) and almost no calm, model-first probability literacy. FiveThirtyEight’s SPI is gone. Tip farms win SEO; honest public track records are rare.

Meanwhile the existing repo is a teaching stack: historical CSVs, a simple goals model, Streamlit demo. It does **not** yet publish a weekend slate, keep an immutable scorecard, or show model vs bookmaker side by side.

For a portfolio reader: the gap is between a notebook demo and a small production ML product — eval, serving, and a public surface that does not hide misses.

### Goal (this build)

Ship a **lab-honest** free Premier League project that is credible as both a public site and a body of engineering work:

- Every weekend: **home / draw / away** probabilities that are **calibrated** (when we say ~40%, those outcomes happen about that often over many matches).
- Show **model numbers and bookmaker-implied numbers side by side** — no tipster voice, no ACCA product.
- Also show **both-teams-to-score (BTTS)** and **over/under 2.5 goals** in the same window, preferably **derived from the same scoreline grid** (the table of possible final scores the goals model produces), not separate tipster-style nets.
- Public **past-weekends scorecard** that **never rewrites history**.
- Short **how-it-works** page so a non-expert can understand the method.
- Architecture that could later support UK/Ireland affiliates — **affiliates stay off for this build**.

**One-line promise:** *Calibrated weekend probabilities for the Premier League, with the model and the book shown side by side — honest ledger, not tipster energy.*

---

## 2. Users

| Who | What they need from v1 |
|---|---|
| **Casual PL fans** | “What’s likely this weekend?” in plain language; model vs book as context, not a tip. |
| **Recreational bettors** | Confidence framing before they place; entertainment / not financial advice; no pressure CTAs. |
| **Leonard (builder)** | A learning loop: data pipeline → eval → ship → public mistakes stay visible. FDE-shaped skills in the background; not interview theater this month. |
| **Hiring managers / engineers** | Clear problem framing, honest eval, a shippable architecture (precompute → API → UI), and judgment about what not to build. |

---

## 3. Non-goals (this window)

- Tipster / ACCA / bet-builder product or copy (guaranteed parlays, “sure thing” language).
- Beating the closing book as a **ship blocker** (nice stretch; honesty ships without it).
- Bookmaker affiliate links, geo-gated CTAs, CPA/revshare wiring.
- Multi-league coverage; live odds APIs; premium tiers.
- Consumer brand rename / heavy GTM.
- Running the model in the browser; replacing Streamlit for lab work (Streamlit stays the lab).
- Separate neural nets per market when the scoreline grid can derive BTTS / O/U 2.5.
- Ireland/UK compliance solicitor review for affiliate ads (comes when affiliates turn on).

---

## 4. Success metrics

Explain once, then use:

- **Log loss / Brier score:** proper scores for probability quality (lower is better). Prefer these over “% of winners picked.”
- **Calibration / reliability plot:** buckets of predicted probability vs how often that outcome actually happened. Goal: curves that track the diagonal, not just a flashy accuracy %.
- **Walk-forward eval:** train only on data available *before* each gameweek (or expanding window), then score that week — closer to real life than one fixed holdout season alone.
- **De-vigged book:** bookmaker odds with the overround (book margin) removed so implied probabilities sum to ~100%; our default comparison baseline.

### Ship bar (must hit to call the window done)

| Metric | Target |
|---|---|
| Public surface | Weekend page + past scorecard + short how-it-works live on stored predictions |
| Markets on weekend page | Home/draw/away **and** BTTS + over/under 2.5 |
| Numbers shown | **Model and book side by side** (declare which closing/reference book) |
| Eval | Walk-forward protocol + calibration plots committed / documented |
| Trust | Predictions saved once for public history; past page never silently rewrites |
| Affiliates | Explicitly off (no CTAs) |
| Architecture | Python precompute → store → thin API → Next.js; Streamlit lab only |
| League | Premier League only |

### Stretch (celebrate if hit; do not block ship)

- Model proper scores better than the de-vigged closing book on walk-forward folds.
- Optional second-stage **market blend** (combine model + market out of sample — Benter-style fusion) with clear labelling.
- Historical **value / expected-return** backtest paper metrics vs a declared closing book — still no tipster UI.

### North-star before monetization

Returning readers checking the weekend slate + settled scorecard. Year-1 affiliate revenue expectation for this build: **£0 by design**.

---

## 5. Scope in / out

### In scope

| Area | Decision |
|---|---|
| **Promise** | Lab-honest product: calibrated 1X2 + model vs book; not tipster/ACCA energy |
| **Public numbers** | Always show **model and book side by side** |
| **Eval** | Walk-forward + calibration plots; beating the book is stretch |
| **Historical data** | football-data.co.uk E0 CSVs (results + closing odds already in the teaching stack) |
| **Fixtures** | football-data.org **free** API for upcoming PL matches |
| **Predictions** | Save once; public past-weekends never rewrites history (corrections = new superseding row if ever needed — never quiet overwrite after kickoff) |
| **Markets** | 1X2 + BTTS + O/U 2.5 in the same window; prefer derive from scoreline grid |
| **Site** | Weekend page + past scorecard + short how-it-works |
| **Affiliates** | **Off** this build |
| **Architecture** | Python engine precompute → store → thin read-only API → Next.js public UI |
| **Lab** | Streamlit remains for exploration; not the public face |
| **Geography / league** | PL only; UK/IE affiliates later as a product note |
| **Docs** | Learning-friendly: explain terms; epics note transferable skills |

### Out of scope (see also Non-goals)

Live odds productization; ACCAs; multi-league; affiliate GTM; Opta-grade SPI rebuild; browser-side inference.

---

## 6. Epics (ordered for ~4–6 weeks)

Do these in order. Later epics assume earlier ones exist. Each epic lists **learning notes** (transferable FDE / product-engineering skills).

### Epic 1 — Honest eval harness (walk-forward + calibration)

**Outcome:** A repeatable research protocol that scores models the way the product will live — time-aware folds, proper scores, and calibration plots — always vs the de-vigged book.

**Includes:**
- Gameweek (or expanding-window) walk-forward over historical E0.
- Log loss, Brier, reliability / calibration plots.
- Keep a fixed season gate if useful for regression, but walk-forward is the research source of truth.
- Document leakage rules (no fitting on matches being scored).

**Done when:** One command (or clearly documented script path) produces walk-forward metrics + plots; README / how-it-works draft can describe the protocol in plain language.

**Learning notes:** Evaluation design, train/serve leakage, proper scoring rules, communicating uncertainty with plots — core FDE “prove it before you ship it” muscle.

---

### Epic 2 — Goals model + derived markets (1X2, BTTS, O/U 2.5)

**Outcome:** Upgrade the fundamental goals model in the same family as today’s Poisson (prefer **Dixon–Coles** with time decay: a goals model that corrects low-score / draw mass and down-weights old matches). Freeze a **scoreline grid**; derive home/draw/away, BTTS, and over/under 2.5 as sums of cells — do not invent separate tipster nets.

**Includes:**
- Model upgrade + fit pipeline on historical PL.
- Derived market probabilities from the grid.
- Where football-data.co.uk has O/U 2.5 closing columns, evaluate derived O/U against them; BTTS may lack market odds on E0 — still ship model BTTS from the grid.
- Optional Elo (simple rating system) as a **comparator line**, not the sole engine.
- Calibration step (e.g. temperature / vector scaling) so public 1X2 is not raw overconfident.

**Done when:** Offline pipeline outputs calibrated 1X2 + BTTS + O/U 2.5 for historical and (later) upcoming fixtures; eval harness from Epic 1 scores them.

**Learning notes:** Generative vs discriminative models, feature/time decay, multi-market consistency from one latent structure, calibration vs sharpness.

---

### Epic 3 — Fixtures ingest + immutable prediction store

**Outcome:** Pull upcoming Premier League fixtures from the **football-data.org free API**, run the precompute job, and **persist predictions once** with enough metadata to reproduce a weekend (`match_id`, kickoff, teams, probs, `model_version`, `published_at` / as-of time, status).

**Includes:**
- Team-name mapping between football-data.co.uk history and football-data.org fixtures (known failure mode — treat as first-class).
- Append-only / refuse silent overwrite after kickoff; public past page reads settled rows only.
- Settlement path: after results land, mark rows settled; never delete misses.

**Done when:** A scheduled or manually triggered job writes a weekend slate that survives restart; past weekends remain unchanged when the model is retrained.

**Learning notes:** Batch ETL, idempotency, schema for auditability, “train/serve” reproducibility, rate limits / vendor constraints.

---

### Epic 4 — Thin read-only predictions API

**Outcome:** A small HTTP API that **only serves stored rows** — no `POST /predict`, no fitting in request path. Next.js (and any tools) read from here.

**Includes:**
- Endpoints for current weekend slate, past weekends / scorecard, optional model-version metadata.
- Clear error behaviour when a weekend is missing.
- Streamlit continues to talk to lab code paths; production UI does not call the model process.

**Done when:** API returns stored predictions for at least one published weekend and historical settled rows; contract documented for the UI epic.

**Learning notes:** Separation of concerns, read-vs-write surfaces, API contracts, caching of precomputed artefacts — classic production ML serving pattern without MLOps theater.

---

### Epic 5 — Next.js public site (weekend + past + how-it-works)

**Outcome:** The public face of the project.

**Pages:**
1. **Weekend** — upcoming (or current) PL slate with **model and book side by side** for 1X2, plus BTTS and O/U 2.5; calm copy; entertainment / not advice.
2. **Past scorecard** — settled weekends; never rewritten; misses stay visible.
3. **How it works** — short, plain language: data sources, walk-forward idea, calibration, scoreline grid → derived markets, what we do *not* claim.

**Done when:** Someone can open the site, understand this weekend’s probabilities, compare to the book, and check last weekend’s track record without reading the repo.

**Learning notes:** Product narrative, trust UX, consuming a thin API from a modern frontend, shipping a real surface vs a notebook demo.

---

### Stretch (inside the window only if Epics 1–5 are solid)

- **Market blend + historical value paper metrics** vs a declared closing book (Bet365 / Avg / Pinnacle — pick one and label it). Still no tipster UI; beating the book remains non-blocking.
- Public polish: publish freeze timing (T−N hours), clearer CLV (closing-line value) language on the scorecard.

---

## 7. Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Team-name mapping** FD.co.uk ↔ FD.org | Wrong fixtures → wrong published probs | Canonical mapping table + tests; fail closed rather than guess |
| **Overconfidence / tipster drift** | Copy or UI slides into “picks” | Side-by-side model+book; banned language list; how-it-works disclaimers |
| **Calibration looks bad after walk-forward** | Public trust | Ship honesty + plots; improve model; do not hide misses |
| **Beating the book takes longer than 6 weeks** | Morale / scope creep | Explicit stretch; ship bar does not require CLV edge |
| **BTTS without market odds on E0** | Harder to show “book” side for BTTS | Show model BTTS; book column for BTTS only when a source exists; always show book for 1X2 (and O/U when columns exist) |
| **Scope creep (affiliates, ACCAs, live odds)** | Misses the learning + trust wedge | Non-goals above; affiliates off |
| **football-data.org rate limits / free-tier change** | Fixtures job fails | Cache responses; document fallback; don’t hard-depend on live calls in the request path |
| **Streamlit leakage habits** | Lab refits on full history including “past” matches | Production path uses stored snapshots only |

---

## 8. Open questions

Resolve during build where possible; none should block starting Epic 1.

1. **Which closing book is the public “book” column?** Bet365 close, market Avg, or Pinnacle when present — pick one default and label it everywhere.
2. **Publish freeze:** exact T−N hours before kickoff when a row becomes immutable for the public past page.
3. **Prediction store technology:** SQLite file vs Postgres for v1 (either fine if append-only semantics hold).
4. **Model artefacts:** retrain every precompute vs versioned saved fits — how do we reproduce a published weekend exactly?
5. **BTTS book column:** omit until an odds source exists, or leave blank with explanation?
6. **Soft launch geography:** global-readable content now, UK/IE monetization later (preferred by research) — confirm for GTM when affiliates near.
7. **Calibration gate:** any hard ECE / reliability threshold that blocks a `model_version` from publish, or human review for v1?
8. **Elo on the public page:** comparator line in v1 UI, or lab-only until 1X2 calibration is stable?

---

## 9. Decisions encoded from the grill (plain summary)

These are the product choices this PRD implements — written in normal language, not ceremony:

- We are building a **lab-honest** product: calibrated weekend home/draw/away probabilities and **model vs book side by side**. Tipster / ACCA energy is out.
- **Public numbers always show model and book together.**
- **Eval** means walk-forward + calibration plots; **beating the book is a stretch**, not a ship requirement.
- **Upcoming fixtures** come from the **football-data.org free API**.
- Predictions are **saved once**; the **past-weekends page never rewrites history**.
- **Affiliates are off** for this build (UK/IE later).
- The **site** is weekend page + past scorecard + short how-it-works.
- **Markets** in this window: 1X2 **and** BTTS + over/under 2.5; derive from the scoreline grid when possible.
- **Architecture:** Python engine precomputes → store → thin API → Next.js; Streamlit is lab only.
- **Premier League only**; docs stay **learning-friendly**.

Durable “why” still lives in [`project-context.md`](project-context.md); this PRD owns epics and the 4–6 week ship bar.

---

*Entertainment / probability literacy framing. Not financial advice. No app or PR work in this document.*
