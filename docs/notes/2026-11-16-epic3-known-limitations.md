# Epic 3 — known limitations after PRD follow-ups

**Date:** 16 Nov 2026  
**Branch:** `epic-3-fixtures-ingest-prediction-store`  
**Related:** [`fixtures-ingest-prediction-store` spec](../specs/fixtures-ingest-prediction-store.md) · commits `c34fcd2`, `817321d`

Epic 3 ships an append-only prediction ledger (fit → publish → settle) that matches the PRD “Done when” bar. A PRD review and follow-up pass closed the important gaps (stale publish cache, postpone/cancel/score corrections, transactional insert, optional book fill). What remains are **known limitations**, not merge blockers. This note records them so Epic 4–5 and the next live weekend are not surprised.

---

## 1. Book columns stay null until co.uk lists Premier League rows

### The story

Publish can enrich nullable `book_*` from [football-data.co.uk `fixtures.csv`](https://www.football-data.co.uk/fixtures.csv) (Bet365 1X2 and over/under when both sides exist). On 16 Nov 2026 that file returned HTTP 200 with Bet365 columns, but **no `E0` (Premier League) rows** — only lower leagues. So a live publish that day would correctly leave `book_*` and `book_source` null.

### Why that is acceptable for now

- Model probabilities still publish; the ledger is honest.
- Free football-data.org has no odds; co.uk fixtures is the free side-by-side path without a paid Odds Add-On.
- The fill path is tested against a local CSV fixture; failure to download or match fails soft.

### What to expect in production

Book side-by-side appears when the co.uk fixtures file includes matching `E0` home/away/date rows (typically after their Fri/Tue fixture scrapes). Until then, UI/API should treat null book columns as “no book source yet,” not as a failed publish.

### Follow-ups (optional)

- A books-only backfill for already-published keys once E0 lines appear (today, duplicate publish does not rewrite `book_*`).
- Or a declared paid/live odds source later; keep `book_source` labelled.

---

## 2. Settle request window grows with history

`settle_from_api` includes every non-cancelled kickoff (including already settled rows) so late score corrections can still be applied. At Premier League volume this is fine for v1. After long retention or very frequent polling, cap the span (e.g. unsettled/postponed + last N days of settled) to keep FD.org calls small.

---

## 3. First postponement depends on the stored kickoff date

If football-data.org stops returning a match on the original kickoff date before settle runs, the first `POSTPONED` signal can be missed. Once the row is `postponed`, the next settle looks ahead 90 days for a reschedule. Mitigate with a settle run after each matchday (or soon after known rearrangements).

---

## 4. Intentional freeze rules

| Behaviour | Rationale |
|---|---|
| Team names never rewrite after insert | Public row identity stays stable |
| Duplicate publish does not backfill books | Probabilities and books are not silently rewritten |
| Cancelled rows are not re-polled | Avoids churn; a rare reversal would need a manual path |
| `SUSPENDED` maps to `postponed` | One status is enough for v1 scorecard logic |

---

## Pointers

| Item | Location |
|---|---|
| Book enrichment | `src/fixtures/books.py` |
| Publish (no fixture cache read) | `src/jobs/publish.py` |
| Settle (postpone / cancel / correct) | `src/jobs/settle.py` |
| Spec (statuses, book fill, batch insert) | `docs/specs/fixtures-ingest-prediction-store.md` |
| Follow-up engineering report | `.superpowers/sdd/task-prd-followups-report.md` (local SDD scratch; not required in Git) |
