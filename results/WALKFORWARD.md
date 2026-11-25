# Walk-forward artefacts

## Protocol

- Folds: synthetic gameweeks (<=10 matches, date-ordered within each season)
- Train: all matches with date strictly before the fold
- Features: built per fold; history-dependent columns use date < fold cutoff only
- Minimum train rows: 200
- Models (1X2, market=1x2): naive_frequencies, poisson, dixon_coles, dixon_coles_calibrated, dixon_coles_book_blend, elo, sklearn_baseline, bookmaker_implied (Bet365 closing), bookmaker_opening (Bet365 pre-closing)
- dixon_coles_calibrated: temperature scaling fit on prior-fold Dixon–Coles 1X2 predictions only. The line is skipped until that prior store has at least 200 rows (the raw line is not copied in). Summary fold counts can therefore differ from raw Dixon–Coles; compare them on shared gameweeks.
- dixon_coles_book_blend: Benter log-odds fusion of raw Dixon–Coles with de-vigged Bet365 **closing** 1X2. Coefficients (α, β) and w_model = α/(α+β) are fit on prior-fold OOS pairs only (never in-sample DC fitted probs). Skipped until 200 prior rows. Publish w_model even if ≈ 0 — that means the market dominates. Labelled model+market fusion, not pure model skill. sklearn_baseline remains a market-echo comparator (odds-as-features), not an edge claim.
- Do not rank lines that were scored on different matches. Calibrated and blend rows cover fewer matches than the raw line. The paired table keeps one match set per comparison and prints a 95% range from resampling whole rounds.
- Scoreline rows (market=scoreline) score the exact score: cell log loss, most-likely-score hits, top-three hits, goal errors, and a ranked score on goal difference. Baselines on the same matches: always 1-1 (hit rate only), league score frequencies, a no-team-strength Poisson, the team-strength Poisson, and Dixon-Coles goal rates with the low-score correction turned off. The headline pool includes the seasons used to choose the time-decay setting. The holdout seasons are the clean figure. See the measurement notes when present. When the price columns are present, the same matches also carry our reconstruction of the book's score table from its prices, not the bookmaker's correct-score odds. Opening and closing reconstructions are separate rows. A rebuilt table that misses the raw prices by more than 0.005 log loss stops the run.
- Binary markets (do not rank against 1X2): dixon_coles_over_25 (market=over_under_25) and dixon_coles_btts (market=btts) are scored from the raw scoreline grid every fold. Lab-only O/U comparators: poisson_total_over_25 (independent Poisson on λ+μ from the same DC rates), optional poisson_total_over_25_calibrated (prior-fold binary temperature), and totals_logistic_over_25 (logistic on λ, μ, λ+μ fit on the train fold). These lab lines never replace grid O/U on publish or the weekend page. bookmaker_over_25 uses closing totals (`implied_over_25`); bookmaker_over_25_open uses pre-closing totals when present. dixon_coles_book_blend_over_25 is an optional prior-fold log-odds blend of grid over vs closing book over when both exist. Missing O/U odds drop a row from that book comparison only. 1X2 temperature scaling does not change grid O/U or BTTS.
- ξ (`DIXON_COLES_XI`) is a frozen config constant from nested tune (`python -m src.evaluate --mode xi_tune`); not jointly estimated. See `results/XI_TUNE.md`.
- Summary metrics: pooled over all scored matches (per-fold CSV for diagnosis). Blend rows also carry mean blend_alpha / blend_beta / blend_w_model.
- Calibration plots: home-win reliability for 1X2 lines

## Regenerate

```bash
source .venv/bin/activate
python -m src.evaluate --mode walkforward
```

Season gate: `python -m src.evaluate --mode season`

See docs/EVAL_PROTOCOL.md for interpretation.

## Scoreline

The scoreline is the full table of exact scores, not the single most likely score.
Log loss here is minus the log of the probability on the score that happened. Lower is better.
A hit rate is how often the single most likely score was the score that happened.

The pooled figures below include seasons 2223, 2324, 2425 (940 matches), which were used to choose the time-decay setting. They are not a clean test of that choice. The clean figure is the holdout, seasons 2526, 2627. When the setting was chosen, that holdout was 480 matches (see results/XI_TUNE.md). This run scores 490 holdout matches. Do not use the holdout to choose the next setting.

Every scoreline row below is the same 1430 matches. Always saying 1-1 is a hit rate only. It has no probability table, so it has no log loss.

| Line | What it is | Scoreline log loss | Most likely score hit | Top 3 hit | Home goal error | Away goal error | Goal-difference RPS | Matches |
|---|---|---|---|---|---|---|---|---|
| Always say 1-1 (hit rate only) | `always_1_1` | not a probability forecast | 0.1126 | n/a | n/a | n/a | n/a | 1430 |
| League score frequencies from earlier matches, no team information | `league_score_frequencies` | 3.0926 | 0.1084 | 0.2706 | 1.0622 | 0.9488 | 1.0263 | 1430 |
| Poisson with the earlier matches' average home and away rates, no team strength | `constant_poisson` | 3.0936 | 0.1126 | 0.2734 | 1.0625 | 0.9483 | 1.0267 | 1430 |
| Team-strength Poisson (attack, defence, each side's own average rate) | `poisson` | 3.0530 | 0.1070 | 0.2958 | 0.9799 | 0.9054 | 0.9455 | 1430 |
| Dixon-Coles goal rates with the low-score correction turned off | `dixon_coles_rho0` | 3.0577 | 0.1105 | 0.2888 | 0.9889 | 0.9026 | 0.9393 | 1430 |
| Dixon-Coles | `dixon_coles` | 3.0585 | 0.1070 | 0.2902 | 0.9889 | 0.9026 | 0.9392 | 1430 |
| Closing prices: our reconstruction of the book's score table from its prices, not the bookmaker's correct-score odds | `book_score_reconstruction` | 2.9590 | 0.1252 | 0.3112 | 0.9594 | 0.8769 | 0.9080 | 1430 |
| Opening prices: our reconstruction of the book's score table from its prices, not the bookmaker's correct-score odds | `book_score_reconstruction_open` | 2.9689 | 0.1217 | 0.3035 | 0.9674 | 0.8794 | 0.9141 | 1430 |

Home and away goal error are the average absolute difference between the goals the table expected and the goals that were scored. Goal-difference RPS scores the ordered margin (home goals minus away goals). Lower is better.

### Gaps on the same matches

A gap is the mean of model A minus model B on one shared set of matches. The range is a 95% percentile interval from 4000 resamples of whole rounds, seed 20261124. A range that contains zero has not separated the two lines on this sample. The closing book is the competitor on the result markets. A smaller log loss than the book is not described as a win over the book.

- dixon_coles minus league_score_frequencies, scoreline log loss, all scored matches: -0.0341 (95% range -0.0819 to 0.0172, 1430 matches, 143 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus constant_poisson, scoreline log loss, all scored matches: -0.0351 (95% range -0.0841 to 0.0178, 1430 matches, 143 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus poisson, scoreline log loss, all scored matches: 0.0055 (95% range -0.0135 to 0.0262, 1430 matches, 143 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus dixon_coles_rho0, scoreline log loss, all scored matches: 0.0008 (95% range -0.0020 to 0.0036, 1430 matches, 143 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus always_1_1, most-likely-score hit rate, all scored matches: -0.0056 (95% range -0.0238 to 0.0119, 1430 matches, 143 rounds). A positive difference means the first line was right more often.
- dixon_coles minus league_score_frequencies, most-likely-score hit rate, all scored matches: -0.0014 (95% range -0.0168 to 0.0140, 1430 matches, 143 rounds). A positive difference means the first line was right more often.
- dixon_coles minus book_score_reconstruction, scoreline log loss, all scored matches: 0.0995 (95% range 0.0608 to 0.1419, 1430 matches, 143 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus book_score_reconstruction, most-likely-score hit rate, all scored matches: -0.0182 (95% range -0.0343 to -0.0021, 1430 matches, 143 rounds). A positive difference means the first line was right more often.
- book_score_reconstruction_open minus book_score_reconstruction, scoreline log loss, all scored matches: 0.0098 (95% range 0.0043 to 0.0151, 1430 matches, 143 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus league_score_frequencies, scoreline log loss, the holdout seasons: 0.0141 (95% range -0.0702 to 0.1160, 490 matches, 49 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus constant_poisson, scoreline log loss, the holdout seasons: 0.0219 (95% range -0.0596 to 0.1210, 490 matches, 49 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus book_score_reconstruction, scoreline log loss, the holdout seasons: 0.1027 (95% range 0.0297 to 0.1934, 490 matches, 49 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus bookmaker_implied, home/draw/away log loss, all scored matches: 0.0338 (95% range 0.0128 to 0.0606, 1430 matches, 143 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus bookmaker_implied, home/draw/away log loss, the holdout seasons: 0.0513 (95% range -0.0021 to 0.1225, 490 matches, 49 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus poisson, home/draw/away log loss, all scored matches: -0.0040 (95% range -0.0144 to 0.0084, 1430 matches, 143 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles minus naive_frequencies, home/draw/away log loss, all scored matches: -0.0733 (95% range -0.1044 to -0.0378, 1430 matches, 143 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles_calibrated minus dixon_coles, home/draw/away log loss, all scored matches: -0.0021 (95% range -0.0062 to 0.0015, 1230 matches, 123 rounds). A positive difference means the first line has the higher log loss.
- dixon_coles_book_blend minus bookmaker_implied, home/draw/away log loss, all scored matches: 0.0017 (95% range -0.0025 to 0.0063, 1230 matches, 123 rounds). A positive difference means the first line has the higher log loss.
- bookmaker_opening minus bookmaker_implied, home/draw/away log loss, all scored matches: 0.0058 (95% range 0.0021 to 0.0094, 1430 matches, 143 rounds). A positive difference means the first line has the higher log loss.

Per-match scores are in `results/walkforward_match_scores.csv`. The paired ranges are in `results/walkforward_paired.csv`. Holdout scoreline levels, which use fewer matches than the headline table, are in `results/walkforward_scoreline_holdout.csv` and are not ranked against the headline rows.

Dixon-Coles scoreline log loss on the headline set is 3.0585 on 1430 matches. League frequencies are 3.0926 on those same matches.

### Rebuilt book score table

our reconstruction of the book's score table from its prices, not the bookmaker's correct-score odds. Two goal rates are chosen so a Poisson table, with the low-score correction left at zero, matches the de-vigged home, draw and away prices and the de-vigged over/under 2.5 price. The lines below are on the same matches as the scoreline table. A smaller scoreline log loss than Dixon-Coles is not a win over the book.

- Closing prices, 1430 matches. Rebuilt home/draw/away log loss 0.9645, raw prices 0.9635, difference 0.0010. Rebuilt over/under 2.5 log loss 0.6711, raw prices 0.6711, difference 0.0000.
- Opening prices, 1430 matches. Rebuilt home/draw/away log loss 0.9695, raw prices 0.9693, difference 0.0002. Rebuilt over/under 2.5 log loss 0.6739, raw prices 0.6738, difference 0.0001.
