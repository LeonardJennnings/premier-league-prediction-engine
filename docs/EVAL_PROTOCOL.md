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

**Feature as-of rule.** Point-in-time columns (match id, opening/closing implied odds) come from the match row. History-dependent features (form, Elo, decay — when added) are built per fold via `features_for_fold` / `add_features(..., history=..., cutoff=...)` and must use only matches with `date <` the fold cutoff. A helper assert rejects future-looking history joins.

**Opening vs closing book.** football-data.co.uk stores Bet365 pre-closing odds (`B365H/D/A`, `B365>2.5`) and closing odds (`B365CH/CD/CA`, `B365C>2.5`). Closing remains the harder skill benchmark and the default `bookmaker_implied` line. Opening is a separate comparator for “could we have matched the book at publish time?” — do not mix the two when claiming an edge. Closing O/U prefers `B365C>2.5` / `AvgC>2.5` and only falls back to the non-C series when closing totals are absent (older seasons).

**Comparators**

| Line | Description |
|---|---|
| Naive frequencies | Historical home/draw/away rates from the train window |
| Poisson | Independent Poisson goals model (attack, defence, home advantage) |
| Dixon–Coles (raw) | Time-decayed Dixon–Coles 1X2 from the scoreline grid |
| Dixon–Coles (calibrated) | Temperature-scaled 1X2; walk-forward only, after a prior-fold warm-start |
| Elo | Rating comparator (1X2 only) |
| Sklearn baseline | Logistic regression on pre-match features available at train time (**market-echo** when odds are features — not an independent edge claim) |
| Bookmaker implied | De-vigged Bet365 **closing** 1X2 (`B365CH/CD/CA`) |
| Bookmaker opening | De-vigged Bet365 **pre-closing** 1X2 (`B365H/D/A`) when present |
| Dixon–Coles + book blend | `dixon_coles_book_blend`: Benter log-odds fusion of raw DC with **closing** book; prior-fold OOS only |
| Dixon–Coles O/U 2.5 | Binary line `dixon_coles_over_25` from the raw grid (`market=over_under_25`) |
| Dixon–Coles BTTS | Binary line `dixon_coles_btts` from the raw grid (`market=btts`); model-only until a book series exists |
| Bookmaker O/U 2.5 | Binary `bookmaker_over_25` when closing totals columns exist (`market=over_under_25`) |
| Bookmaker O/U 2.5 open | Binary `bookmaker_over_25_open` when pre-closing totals exist |
| DC + book O/U blend | Optional `dixon_coles_book_blend_over_25`: prior-fold log-odds blend of grid over vs closing book over |

**Derived markets.** 1X2, both-teams-to-score, and over/under 2.5 are read from one renormalised Dixon–Coles scoreline grid (and the `predict_matches` helper). There are no separate BTTS or totals classifiers. Temperature scaling is applied to 1X2 only; BTTS and O/U stay on the raw grid. BTTS is model-only until a reliable bookmaker series is on the historical feed.

**Over/under and BTTS scoring.** Walk-forward publishes pooled binary metrics for `dixon_coles_over_25` and `dixon_coles_btts` every fold, plus `bookmaker_over_25` / `bookmaker_over_25_open` when the matching implied totals column is present. Each summary / by-gameweek row carries a `market` label (`1x2`, `over_under_25`, or `btts`). Missing over/under odds drop a row from that book comparison only.

**Do not rank binary markets against 1X2.** Binary O/U and BTTS log loss are two-class scores. 1X2 log loss is a three-class score. Those numbers are not comparable; filter on `market` (or the model name suffix) and never sort the full CSV as one league table.

**Time decay (ξ).** Dixon–Coles match weights are \(e^{-\xi t}\) with \(t\) in days. ξ is a **frozen config constant** (`DIXON_COLES_XI`), not jointly estimated with attack, defence, home advantage, or ρ. The value is chosen by the nested protocol below (not by in-likelihood MLE).

### Nested ξ tune (hyperparameter selection)

ξ is selected on **older** walk-forward folds and frozen before scoring a **held-out recent** era. This avoids picking the decay rate on the same folds advertised as the headline scorecard.

| | |
|---|---|
| Command | `python -m src.evaluate --mode xi_tune` |
| Candidates | `XI_TUNE_CANDIDATES` in `src/config.py` |
| Tune era | Folds with `season_code` in `XI_TUNE_SEASONS` (default `2223`, `2324`, `2425`) |
| Holdout era | Folds with `season_code` in `XI_TUNE_HOLDOUT_SEASONS` (default `2526`, `2627`) — **not** used for selection |
| Selection rule | Lowest pooled 1X2 log loss on the tune era; ties → nearer baseline ξ, then smaller ξ |
| Freeze | Winner written to `DIXON_COLES_XI` with a comment pointing at `results/XI_TUNE.md` |
| Holdout report | DC at baseline ξ, DC at frozen ξ, and closing `bookmaker_implied` on the holdout matches |
| Artefacts | `results/xi_tune_candidates.csv`, `results/xi_tune_holdout.csv`, `results/XI_TUNE.md` |

**Leakage rules for ξ.** Do not retune on holdout or on the full walk-forward headline after freezing. Do not claim the model beats the closing book unless the holdout numbers show it (and even then prefer nested / paired evidence before public claims).

**Calibrated vs raw coverage.** The calibrated Dixon–Coles line is skipped until the prior-fold prediction store has at least 200 rows (`CALIBRATION_MIN_ROWS`). Summary tables can therefore show fewer folds for `dixon_coles_calibrated` than for `dixon_coles`. Compare those two lines on shared gameweeks, not on the headline means. Season gate reports raw Dixon–Coles and Elo only; calibrated 1X2 is walk-forward-only in this version.

**OOS Benter blend (DC + closing book).** For each gameweek fold, raw Dixon–Coles 1X2 probabilities on matches with closing odds are stored as out-of-sample predictions. A second-stage Benter blend \(c \propto \exp(\alpha \log f + \beta \log \pi)\) is fit on **prior folds only**, then applied to the current fold. The summary reports mean `blend_alpha`, `blend_beta`, and `blend_w_model = α/(α+β)`. Publish the weight even when \(w_{\text{model}} \approx 0\) (market dominates) — that is a valid scientific result, not a harness failure. The blend fuses against **closing** Bet365 for the “does the model add anything after a sharp prior?” question; opening remains a separate publish-time comparator and is not mixed into this line. Skipped until `BLEND_MIN_ROWS` (200) prior OOS pairs exist. Optional O/U blend uses the same nested protocol on grid-derived over vs `implied_over_25`.

**Blend leakage checklist**

- [x] Base DC probs for blend pairs are fold OOS (`date < cutoff` train only) — never in-sample fitted probs on the scored matches.
- [x] Blend coefficients fit on prior-fold OOS pairs only; current fold is scored after the fit and appended afterward.
- [x] Closing vs opening book lines stay separate; this blend declares closing.
- [x] Sklearn / odds-as-features lines stay labelled market-echo, not edge.
- [x] No tipster claim that blend “beats the closing book” without nested / paired evidence (stretch only).

**Offline predictions.** `src.models.predict.predict_matches` scores a fitted Dixon–Coles model on a match table. Optional temperature is applied to 1X2 only, matching the rule above. The walk-forward blend row is a research/harness citizen; weekend UI must label it “model + market fusion” if ever shown.

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
