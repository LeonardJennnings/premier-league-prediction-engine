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

Outputs: `results/walkforward_summary.csv`, `results/walkforward_by_gameweek.csv`, `results/walkforward_match_scores.csv`, `results/walkforward_paired.csv`, `results/walkforward_scoreline_holdout.csv`, `results/walkforward_book_grid_check.csv`, `results/calibration_*_home.png`, `results/WALKFORWARD.md`

**Gameweek construction.** Historical CSVs from football-data.co.uk do not include an official gameweek field. Matches are sorted by date within each season and grouped into sequential rounds of up to **10** games (approximately one Premier League round). This definition is for evaluation only. Product UI may group fixtures by calendar weekend without changing the research protocol.

**Refit policy.** Models are refit at every fold on the expanding history available before that fold. Strengths are not frozen for an entire season. The job is an offline batch process; it is not executed on HTTP request paths.

**Feature as-of rule.** Point-in-time columns (match id, opening/closing implied odds) come from the match row. History-dependent features (form, Elo, decay — when added) are built per fold via `features_for_fold` / `add_features(..., history=..., cutoff=...)` and must use only matches with `date <` the fold cutoff. A helper assert rejects future-looking history joins.

**Opening vs closing book.** football-data.co.uk stores Bet365 pre-closing odds (`B365H/D/A`, `B365>2.5`) and closing odds (`B365CH/CD/CA`, `B365C>2.5`). Closing remains the harder skill benchmark and the default `bookmaker_implied` line. Opening is a separate comparator for the price that could have been seen at publish time. Do not mix the two when comparing lines. Closing over/under uses `B365C>2.5` or `AvgC>2.5` only. A pre-close total is not copied into the closing column, and it is not scored as `bookmaker_over_25`. Publish reads the non-closing columns on the upcoming file and does not substitute a closing price when those are blank.

**Live record.** Two looks are written down before the settled ledger is used to judge the model: 380 settled matches, and 1,140 settled matches. Between those, a range on the site is descriptive. It is not a reason to change a setting or to declare a winner. The note is `docs/experiments/live-record-looks.md`. The ledger is not a tuning set.

**Comparators**

| Line | Description |
|---|---|
| Naive frequencies | Historical home/draw/away rates from the train window |
| Poisson | Independent Poisson goals model (attack, defence, home advantage). The away rate uses the average away goals, not the blended home-and-away average |
| Dixon–Coles (raw) | Time-decayed Dixon–Coles 1X2 from the scoreline grid |
| Dixon–Coles (calibrated) | Temperature-scaled 1X2; walk-forward only, after a prior-fold warm-start |
| Elo | Rating comparator (1X2 only) |
| Sklearn baseline | Logistic regression on pre-match features available at train time (**market-echo** when odds are features — not an independent edge claim) |
| Bookmaker implied | De-vigged Bet365 **closing** 1X2 (`B365CH/CD/CA`) |
| Bookmaker opening | De-vigged Bet365 **pre-closing** 1X2 (`B365H/D/A`) when present |
| Dixon–Coles + book blend | `dixon_coles_book_blend`: Benter log-odds fusion of raw DC with **closing** book; prior-fold OOS only |
| Dixon–Coles O/U 2.5 | Binary line `dixon_coles_over_25` from the raw grid (`market=over_under_25`) — **product default** |
| Poisson-total O/U 2.5 (lab) | `poisson_total_over_25`: independent Poisson on λ+μ from the same DC rates (ignores ρ) |
| Poisson-total O/U calibrated (lab) | `poisson_total_over_25_calibrated`: prior-fold binary temperature on the lab line |
| Totals logistic O/U 2.5 (lab) | `totals_logistic_over_25`: logistic on (λ, μ, λ+μ) fit on the train fold only |
| Dixon–Coles BTTS | Binary line `dixon_coles_btts` from the raw grid (`market=btts`); model-only until a book series exists |
| Bookmaker O/U 2.5 | Binary `bookmaker_over_25` when closing totals columns exist (`market=over_under_25`) |
| Bookmaker O/U 2.5 open | Binary `bookmaker_over_25_open` when pre-closing totals exist |
| DC + book O/U blend | Optional `dixon_coles_book_blend_over_25`: prior-fold log-odds blend of grid over vs closing book over |

**Derived markets.** 1X2, both-teams-to-score, and over/under 2.5 are read from one renormalised Dixon–Coles scoreline grid (and the `predict_matches` helper). Publish and the weekend UI keep that grid spine. Lab-only specialised O/U lines (`poisson_total_*`, `totals_logistic_over_25`) are scored in walk-forward for research; they must not silently replace grid O/U without a new dual-labelled `model_version`. Temperature scaling is applied to 1X2 only on the product path; BTTS and grid O/U stay on the raw grid. BTTS is model-only until a reliable bookmaker series is on the historical feed.

**Over/under and BTTS scoring.** Walk-forward publishes pooled binary metrics for `dixon_coles_over_25`, lab O/U comparators, and `dixon_coles_btts` every fold, plus `bookmaker_over_25` / `bookmaker_over_25_open` when the matching implied totals column is present. Each summary / by-gameweek row carries a `market` label (`1x2`, `over_under_25`, or `btts`). Missing over/under odds drop a row from that book comparison only.

**Lab O/U note.** Dixon–Coles τ reshuffles mass among 0–0 / 1–0 / 0–1 / 1–1 but preserves O/U 2.5 mass at fixed λ, μ. So `poisson_total_over_25` ≈ grid O/U up to finite-grid truncation — useful as a hygiene check, not a promotion candidate. The line that can diverge is `totals_logistic_over_25`. See `docs/notes/2026-11-16-ou-lab-decision.md` for the walk-forward decision.

**Do not rank lines that were not scored on the same matches.** Binary O/U and BTTS log loss are two-class scores. 1X2 log loss is a three-class score. Scoreline log loss is the log loss of one cell in the score table. Those numbers are not comparable. Filter on `market` and never sort the full CSV as one league table. Calibrated and blend rows also cover fewer matches than the raw line. Compare those on the paired table, which keeps one match set and prints the count.

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

### Thin-data prior (one strength)

A club with few matches is pulled toward a starting attack and defence. A club with no match in the previous season starts lower than a club that already has a history. The fraction is fixed (`THIN_PRIOR_PROMOTED_FACTOR`). It is not searched. The only searched number is the strength (`THIN_PRIOR_CANDIDATES`).

| | |
|---|---|
| Command | `python -m src.evaluate --mode thin_prior` |
| Candidates | `THIN_PRIOR_CANDIDATES` in `src/config.py` |
| Setting era | Seasons `2223`, `2324`, `2425`. These already chose the time-decay setting, so they pick this strength and are not a clean test of it |
| Later era | Seasons `2526`, `2627`. Scored after the strength is frozen. Not used to pick it |
| Selection rule | Lowest scoreline log loss on early matches (a club's first eight of the season). Ties keep the smaller strength |
| Stop rule | `docs/experiments/2026-11-25-thin-data-prior.md`. Drop if early-match scoreline log loss is not lower on the later seasons, or if overall scoreline log loss is higher, or if overall home/draw/away log loss is higher |
| Artefacts | `results/THIN_DATA_PRIOR.md`, `results/thin_prior_candidates.csv`, `results/thin_prior_paired.csv`, `results/thin_prior_matches.csv` |

Each scored round is fit only on earlier matches. The published fit keeps the previous penalty when the stop rule drops the change. The live record is not a tuning set.

The run on 25 Nov 2026 kept strength 32. The committed `results/walkforward_*` tables were scored before that change, with the previous penalty. `results/THIN_DATA_PRIOR.md` is the measurement of the change. A later walk-forward run will score Dixon-Coles at strength 32. The model file already committed under `models/` is the previous fit. The next fit uses strength 32.

### Score-table calibration (one scale, one correction)

```bash
python -m src.evaluate --mode grid_calibration
```

| Piece | Rule |
|---|---|
| Question | Does one scale on both goal rates, plus one constant low-score correction, lower scoreline log loss on later seasons? |
| Candidates | Scales 0.95, 1.00, 1.05 and corrections -0.10, -0.05, 0.00. Nine pairs. Written down before the run |
| Setting era | Seasons `2223`, `2324`, `2425`. Chooses the pair. Not a clean test |
| Later era | Seasons `2526`, `2627`. Scored after the pair is frozen. Not used to pick it |
| Selection rule | Lowest scoreline log loss. Ties keep the scale closer to 1, then the correction closer to 0 |
| Stop rule | `docs/experiments/2026-11-25-score-table-calibration.md`. Drop if the later-season scoreline gain, previous minus new, is under 0.005 at 4 decimal places |
| Artefacts | `results/GRID_CALIBRATION.md`, `results/grid_calibration_candidates.csv`, `results/grid_calibration_paired.csv`, `results/grid_calibration_matches.csv` |

Each scored round is fit only on earlier matches. The scale and the correction are applied after the fit. The published table stays the previous one when the stop rule drops the change. The live record is not a tuning set. This run does not rebuild a score table from book prices.

The run on 25 Nov 2026 dropped the change. On the later seasons the previous line minus the new line was -0.0002 (95% range -0.0047 to 0.0046, 490 matches, 49 rounds). The published score table is unchanged.

### Model plus market (one weight)

```bash
python -m src.evaluate --mode market_blend
```

| Piece | Rule |
|---|---|
| Question | Does one weight, mixing the model's goal rates with the rates from the price available before kickoff, earn a second score table? |
| Candidates | Weights 0.00, 0.05, 0.10, 0.25, 0.50, 1.00 on the model's rates. Written down before the run |
| Input | Opening home, draw, away and over/under 2.5 prices for that match. The closing price is not an input |
| Setting era | Seasons `2223`, `2324`, `2425`. Chooses the weight. Not a clean test |
| Later era | Seasons `2526`, `2627`. Scored after the weight is frozen. Not used to pick it |
| Selection rule | Lowest scoreline log loss. Ties keep the smaller weight |
| Stop rule | `docs/experiments/2026-11-25-model-plus-market.md`. Drop the second line if the weight is under 0.10 and the later-season scoreline gain against the reconstructed closing table is under 0.01 |
| Artefacts | `results/MODEL_PLUS_MARKET.md`, `results/market_blend_candidates.csv`, `results/market_blend_paired.csv`, `results/market_blend_matches.csv` |

Each scored round fits the model only on earlier matches. The pure model stays the published probability when the stop rule drops the second line. The live record is not a tuning set. The reconstructed closing table is the one already in the repo.

The run on 25 Nov 2026 kept weight 0.25 as a second line labelled model plus market. On the later seasons the reconstructed closing table minus the blend was -0.0073 (95% range -0.0162 to 0.0018, 490 matches, 49 rounds). The range includes zero. The gap is under 0.01, so it is not a success. The pure model stays beside it. The second line is written only when the price available before kickoff has home, draw, away and over/under 2.5. The headline walk-forward tables were not regenerated.

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

## Scoreline

The walk-forward also scores the exact score, on the same matches as the raw Dixon-Coles 1X2 line. The market label is `scoreline`, so those rows are not sorted against 1X2 log loss.

| Line | Meaning |
|---|---|
| Dixon-Coles | Log loss of the cell for the score that happened, from the same grid as the 1X2 line |
| Dixon-Coles with the low-score correction off | The same home and away rates, with the low-score correction set to zero |
| Team-strength Poisson | Attack and defence, with each side's own average goal rate. This is the fixed Poisson baseline |
| Constant Poisson | One home rate and one away rate, the averages of earlier matches. No team strength |
| League score frequencies | The share of earlier matches that finished with that score. A score that has not occurred yet is given probability 1/(n+1), where n is the number of earlier matches. Scores that have occurred keep their raw share |
| Always 1-1 | Hit rate only. A single picked score has no probability table, so it has no log loss |

Also recorded, on the same matches: whether the most likely score happened, whether the score was one of the three most likely, the average absolute error of home goals, away goals and the total, and a ranked score of the goal difference (home goals minus away goals, from -8 to +8). Lower is better on the ranked score. A side that scores 9 or more is not a cell of the 0-to-8 grid. Its probability is the untruncated Poisson probability of that exact score, which is finite. It is not moved into the 8-goal cell.

The headline pool includes seasons `2223`, `2324` and `2425`. Those seasons were used to choose the time-decay setting, so the headline is not a clean test of that choice. The clean figure is the holdout, seasons `2526` and `2627`. When the setting was chosen, that holdout was 480 matches (`results/XI_TUNE.md`). Later matches in those seasons were not used to choose the setting either. The holdout file reports its own match count. Do not use the holdout to choose the next setting, and do not rank the holdout rows against the headline rows.

## Paired gaps

`results/walkforward_paired.csv` is the comparison table. Each row is model A minus model B on one shared set of matches. For log loss, a positive number means A put less probability on what happened. For a hit rate, a positive number means A was right more often.

The range is a 95% percentile interval from 4000 resamples of whole rounds, seed `20261124`. Resampling whole rounds keeps the ten matches in a round together. A range that contains zero has not separated the two lines on this sample. Every gap in the notes names that range and the match count.

Per-match scores are in `results/walkforward_match_scores.csv`, so a range can be rebuilt without fitting the models again.

The closing book is the competitor on the result markets. A smaller log loss than the book is not a win over the book.

There is no free correct-score price. When the de-vigged home, draw, away and over/under 2.5 prices are all present, the walk-forward also scores our reconstruction of the book's score table from its prices, not the bookmaker's correct-score odds. Two goal rates are chosen so a Poisson table, with the low-score correction left at zero, matches those four prices. Closing prices and opening prices are rebuilt separately, and each row is kept only when it covers the same matches as the other scoreline rows. The run stops if the rebuilt home/draw/away log loss, or the rebuilt over/under 2.5 log loss, is more than 0.005 away from the raw prices. That check is in `results/walkforward_book_grid_check.csv`.

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
