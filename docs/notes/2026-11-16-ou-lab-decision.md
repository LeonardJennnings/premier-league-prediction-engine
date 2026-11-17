# O/U lab comparator decision (V2-4)

**Date:** 2026-11-17 (artefacts regenerated under frozen ξ=0.0035 after #10+#8)  
**Scope:** Optional specialised over/under 2.5 lines in the walk-forward harness only.  
**Product rule:** Scoreline-grid O/U (`dixon_coles_over_25` via `markets_from_grid`) stays the publish / weekend default.

## What was scored

On the same gameweek walk-forward folds as the headline O/U book lines:

| Line | Role |
|---|---|
| `dixon_coles_over_25` | Product grid O/U (baseline) |
| `poisson_total_over_25` | Independent Poisson on λ+μ from the same DC rates (ignores ρ) |
| `poisson_total_over_25_calibrated` | Prior-fold binary temperature on the lab Poisson-total line |
| `totals_logistic_over_25` | Logistic on (λ, μ, λ+μ) fit on the train fold only |
| `bookmaker_over_25` | De-vigged Bet365 **closing** totals |
| `bookmaker_over_25_open` | De-vigged Bet365 **opening** totals |

## Theory note (why poisson_total ≈ grid)

Dixon–Coles τ reshuffles probability among 0–0, 1–0, 0–1, and 1–1, but the **total mass** of that block is unchanged. Cells with total goals ≥ 3 are untouched by τ. Therefore O/U 2.5 from the bivariate grid matches independent Poisson(λ+μ) up to finite-grid truncation. `poisson_total_over_25` is a hygiene / teaching comparator, not a promotion candidate.

## Walk-forward result (frozen ξ=0.0035)

| Line | n_folds | n_test | Log loss | Brier | Acc |
|---|---:|---:|---:|---:|---:|
| bookmaker_over_25 (close) | 142 | 1420 | **0.6707** | 0.2390 | 0.5866 |
| bookmaker_over_25_open | 142 | 1420 | 0.6735 | 0.2404 | 0.5697 |
| dixon_coles_book_blend_over_25 | 122 | 1220 | 0.6734 | 0.2401 | 0.5885 |
| dixon_coles_over_25 (grid) | 142 | 1420 | 0.6900 | 0.2475 | 0.5732 |
| poisson_total_over_25 | 142 | 1420 | 0.6901 | 0.2475 | 0.5732 |
| poisson_total_over_25_calibrated | 122 | 1220 | 0.6853 | 0.2460 | 0.5623 |
| totals_logistic_over_25 | 142 | 1420 | 0.6889 | 0.2472 | 0.5761 |

Regenerate: `python -m src.evaluate --mode walkforward` → `results/walkforward_summary.csv`.

## Decision

**Keep grid-only on the public surface.** Lab lines remain harness citizens.

Rationale:

1. Grid coherence (1X2 + BTTS + O/U from one scoreline table) is a product feature.
2. `poisson_total_over_25` matches grid O/U to three decimals (τ mass conservation) — not a promotion candidate.
3. `totals_logistic_over_25` is within noise of grid (0.6889 vs 0.6900) and still trails closing book (0.6707) by a wide margin — not enough for a public alternative; keep scoring it in the harness.
4. Calibrated poisson-total looks a touch sharper (0.6853) but on fewer folds (1,220 vs 1,420) and still trails closing book. Not material enough for a dual-labelled UI swap.
5. Combo markets (if ever shown) stay sums of scoreline cells, never products of separate nets.

**Status:** keep-grid-only (lab harness retained; no dual-label promotion).
