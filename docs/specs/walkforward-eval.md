# Walk-forward evaluation design

**Date:** 16 Nov 2026  
**Status:** Approved for implementation  
**Related:** [Evaluation protocol](../EVAL_PROTOCOL.md) · [PRD](../prd.md) · [Project context](../project-context.md)

## Problem

A single fixed holdout season is a useful regression check, but it does not match how predictions are produced in production. Weekend forecasts must use only information available before kickoff. This design adds a time-aware evaluation harness: gameweek walk-forward folds, proper scoring rules, calibration plots, and a consistent comparison to de-vigged bookmaker probabilities.

## Design decisions

| Topic | Decision | Rationale |
|---|---|---|
| Primary protocol | Gameweek walk-forward | Mirrors production timing; reduces optimism from a single lucky/unlucky season |
| Secondary protocol | Fixed season gate (`2526`) | Fast, stable smoke test after refactors |
| Gameweek construction | Date-ordered rounds of ≤10 matches within each season | Source CSVs have no official gameweek field; 10 matches approximates a Premier League round |
| Presentation vs evaluation | Folds need not match UI grouping | Public surfaces may use calendar/weekend views without changing the research protocol |
| Model refit | Refit every fold on all matches with a strictly earlier date | Avoids within-season leakage from freezing strengths too early |
| Execution model | Offline batch job | Evaluation and precompute are not request-path work; scale later with a worker/VPS if needed |
| Package layout | `src/eval/` with a thin CLI in `src/evaluate.py` | Separates fold logic, calibration, and metrics behind clear module boundaries |
| Artefacts | Summary metrics, per-gameweek metrics, calibration plots, short regenerate notes | Supports both high-level reporting and fold-level diagnosis |

**Out of scope for this change:** public Next.js UI, fixtures API, immutable prediction store, Dixon–Coles model upgrade, affiliate surfaces.

## Behaviour

1. Load the cleaned matches table.
2. Assign `gameweek` within each `season_code` by sorting on match date and chunking into rounds of up to 10.
3. For each fold, define train as all matches with date strictly before the fold, and test as the fold’s matches.
4. Fit naive frequencies, Poisson, and the sklearn baseline on train only; score the fold; record bookmaker implied probabilities on overlapping rows.
5. Aggregate log loss, Brier score, and accuracy overall and per gameweek.
6. Compute reliability tables (predicted probability buckets vs observed frequency) and write plots.
7. Persist artefacts under `results/`; leave the existing season-gate path intact.

**Leakage rule:** do not train on matches in the current test fold; do not use post-kickoff information in features for that fold.

## Module layout

```text
src/eval/
  __init__.py
  gameweeks.py      # assign_gameweeks(matches) -> DataFrame
  walkforward.py    # fold iteration + walk-forward runner
  calibration.py    # reliability tables + plot helpers
src/evaluate.py     # CLI entry for season gate and walk-forward
tests/
  test_gameweeks.py
  test_calibration.py
  test_evaluate.py
results/
  eval_results.csv
  walkforward_summary.csv
  walkforward_by_gameweek.csv
  calibration_*.png
  WALKFORWARD.md
```

Shared metric helpers remain in `evaluate.py` unless imports force a move to `src/eval/metrics.py`. Plotting uses `matplotlib` (added to dependencies if missing).

## Data flow

```text
matches
  → assign gameweeks
  → for each fold: fit on past → predict fold → metric rows
  → summary + per-gameweek tables
  → calibration buckets → PNG
```

## Failure modes

- Insufficient train history for an early fold: skip with a logged warning; document any minimum train size in `results/WALKFORWARD.md`.
- Missing closing odds: exclude those rows from the bookmaker line only; document differing sample sizes if needed.
- Unseen teams at predict time: use the existing league-average strength fallback in the Poisson model.

## Testing

- Gameweek assignment on a small deterministic fixture table
- Calibration bucketing on a hand-checked example
- Existing metric unit tests unchanged
- Optional synthetic smoke test for one short walk-forward (no network I/O)

## Acceptance criteria

- A documented command regenerates walk-forward summary, per-gameweek metrics, and calibration plots
- Season-gate evaluation continues to run and write `results/eval_results.csv`
- Protocol is described in `docs/EVAL_PROTOCOL.md`
- `pytest` passes

## Implementation follow-ups

- Final CLI shape (`--mode` vs subcommands)
- Minimum train size per scored fold
- Whether v1 reliability plots cover home win only or all three 1X2 outcomes
