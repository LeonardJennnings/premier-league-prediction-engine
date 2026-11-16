# Evaluation protocol

How model quality is measured in this project, and why those choices were made.

Product context: [`project-context.md`](project-context.md) · [`prd.md`](prd.md)  
Implementation design: [`specs/walkforward-eval.md`](specs/walkforward-eval.md)  
Season-gate artefact notes: [`../results/EVAL_README.md`](../results/EVAL_README.md)

## Objective

The system outputs **probabilities**, not tips. Quality means:

1. **Proper scoring** — log loss and Brier score reward calibrated probability vectors, not only top-1 picks.
2. **Calibration** — when the model assigns ~40% to an outcome, that outcome should occur near 40% of the time across a large sample.
3. **Market comparison** — scores are reported next to **de-vigged** bookmaker implied probabilities (overround removed so home/draw/away sum to ~1).

Consistently beating the closing book is uncommon and is **not** a release requirement. Beating a naive frequency baseline and remaining competitive with the market is a credible outcome for a transparent, simple stack.

## Protocols

### Season gate (regression)

| | |
|---|---|
| Train | Configured seasons before `2526` |
| Test | Season `2526` matches with closing odds present |
| Role | Stable smoke test after code changes |

```bash
python -m src.evaluate --mode season
```

Output: `results/eval_results.csv`

### Gameweek walk-forward (primary research protocol)

Walk-forward evaluation scores each round using only matches dated **strictly earlier** than that round. That aligns evaluation with production constraints: no peeking at the fixtures being scored.

**This is the research source of truth.** The fixed season gate above is a regression check only — do not treat the `2526` table alone as the headline research result.

```bash
python -m src.evaluate --mode walkforward
```

Outputs: `results/walkforward_summary.csv`, `results/walkforward_by_gameweek.csv`, `results/calibration_*_home.png`, `results/WALKFORWARD.md`

**Gameweek construction.** Historical CSVs from football-data.co.uk do not include an official gameweek field. Matches are sorted by date within each season and grouped into sequential rounds of up to **10** games (approximately one Premier League round). This definition is for evaluation only. Product UI may group fixtures by calendar weekend without changing the research protocol.

**Refit policy.** Models are refit at every fold on the expanding history available before that fold. Strengths are not frozen for an entire season. The job is an offline batch process; it is not executed on HTTP request paths.

**Feature as-of rule.** Point-in-time columns (match id, closing implied odds) come from the match row. History-dependent features (form, Elo, decay — when added) are built per fold via `features_for_fold` / `add_features(..., history=..., cutoff=...)` and must use only matches with `date <` the fold cutoff. A helper assert rejects future-looking history joins.

**Comparators**

| Line | Description |
|---|---|
| Naive frequencies | Historical home/draw/away rates from the train window |
| Poisson | Goals model with attack, defence, and home advantage |
| Sklearn baseline | Logistic regression on pre-match features available at train time |
| Bookmaker implied | De-vigged closing probabilities on the scored rows |

**Metrics**

| Metric | Better | Role |
|---|---|---|
| Log loss | Lower | Penalises confident errors |
| Brier score | Lower | Squared error of the probability vector |
| Accuracy | Higher | Secondary; top-1 hit rate only |

**Summary aggregation.** `walkforward_summary.csv` reports **pooled** log loss, Brier, and accuracy over all scored matches (equivalent to an n-weighted fold mean for these averages). `walkforward_by_gameweek.csv` keeps per-fold rows for diagnosis when fold sizes or odds coverage differ.

**Calibration.** Reliability plots bucket predicted probabilities against observed frequencies (home-win only today). Diagonal-tracking curves matter more than a headline accuracy figure. Expected calibration error (ECE) and full 1X2 (draw/away) reliability plots are natural follow-ups when richer models need a fuller honesty read.

**Target artefacts**

| Path | Contents |
|---|---|
| `results/walkforward_summary.csv` | Pooled aggregate metrics by model |
| `results/walkforward_by_gameweek.csv` | Metrics by round |
| `results/calibration_*.png` | Reliability plots |
| `results/WALKFORWARD.md` | Reproduce steps beside the artefacts |

## Leakage policy

Matches in a test fold are excluded from fitting for that fold. Features must not use information that would be unavailable before kickoff. Large apparent edges versus closing odds are treated as defects until proven otherwise.

## Interpretation guide

- Naive should be the weakest line on proper scores
- Models should beat naive
- Sitting near the bookmaker line is a strong result for this architecture
- Calibration plots belong in any serious read of the numbers, including weak weeks
- Unexplained large wins over closing odds warrant investigation before celebration

## Scope boundaries

Walk-forward gameweeks are the research source of truth; the fixed season remains a regression gate. Affiliate links, tipster language, and parlays are out of scope for the current build window.
