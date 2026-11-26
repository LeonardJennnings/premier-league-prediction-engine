# Two speeds of team memory, one weight

**Date:** 26 Nov 2026 (London)
**Question:** does one weight, mixing a slow Dixon-Coles fit with a fast Dixon-Coles fit, put more probability on the score that happened than the current pure model does, on matches the weight was not chosen on?

## What was fit

The pure model is the current Dixon-Coles score table. Its time-decay setting stays at 0.0035 and the thin-data strength stays at 32. Its low-score correction is the value fitted on each round. The model-plus-market weight stays at 0.25. The stopped score-table calibration is unchanged.

The two-speed line fits that same model twice. The slow fit uses 0.0035. The fast fit uses 0.0120, the fastest decay already listed for the time-decay search, about a 58-day half-life. No other decay was tried. One weight mixes the fast fit's home rate and away rate with the slow fit's rates. The same weight is used for both. The low-score correction stays the slow fit's correction. The weights tried are the rows in the table below. Nothing else was tried after looking. This is not a state-space model and it is not a Kalman filter. The closing price is not an input.

The reconstructed table is our reconstruction of the book's score table from its prices, not the bookmaker's correct-score odds.

## How the weight was chosen

Each round's two fits use only matches dated before that round. The weight is the one with the lowest scoreline log loss on seasons 2223, 2324 and 2425. A tie keeps the smaller weight. Those seasons were already used to choose other settings, so this table is where the weight was picked. It is not a clean test.

| Weight on the fast fit | Scoreline log loss | Matches | Home/draw/away log loss |
|---|---|---|---|
| 0.00 (chosen) | 3.0266 | 940 | 0.9635 |
| 0.05 | 3.0271 | 940 | 0.9641 |
| 0.10 | 3.0277 | 940 | 0.9648 |
| 0.25 | 3.0300 | 940 | 0.9671 |
| 0.50 | 3.0352 | 940 | 0.9717 |
| 1.00 | 3.0510 | 940 | 0.9832 |

The chosen weight on the fast fit is 0.00.

Weight 0.00 is at the end of the list that was written down. No further value was added after looking.

The gap below is on the setting seasons. Those matches were used to pick the weight, so this gap is not a clean test.

On the setting seasons, scoreline log loss: the pure model minus the two-speed line is 0.0000 (95% range 0.0000 to 0.0000, 940 matches, 94 rounds). The range includes zero.

Seasons 2526, 2627 were not used to pick the weight.

## Later seasons

Both fits are redone every round on earlier matches. The weight stays at the chosen value. A range is a 95% percentile interval from 4000 resamples of whole rounds, seed 20261124. A range that contains zero has not separated the two lines on this sample.

### Against the current pure model

This is the stop comparison. The difference is the pure model minus the two-speed line. A positive number means the two-speed line has the lower scoreline log loss.

On every later-season match, scoreline log loss: the pure model minus the two-speed line is 0.0000 (95% range 0.0000 to 0.0000, 490 matches, 49 rounds). The range includes zero.

On those same matches, home/draw/away log loss: the pure model minus the two-speed line is 0.0000 (95% range 0.0000 to 0.0000, 490 matches, 49 rounds). The range includes zero.

On those same matches, over/under 2.5 log loss: the pure model minus the two-speed line is 0.0000 (95% range 0.0000 to 0.0000, 490 matches, 49 rounds). The range includes zero.

Home, draw or away log loss and over/under 2.5 log loss do not choose the weight, and they do not meet the edge definition.

### Against the reconstructed closing table

This is the edge comparison. The difference is the reconstructed closing table minus the two-speed line. A positive number means the two-speed line has the lower scoreline log loss. The closing price is the competitor. It is not an input.

On the later-season matches with a closing table, scoreline log loss: the reconstructed closing table minus the two-speed line is -0.0287 (95% range -0.0502 to -0.0079, 490 matches, 49 rounds).

On those same matches, home/draw/away log loss: the reconstructed closing table minus the two-speed line is -0.0196 (95% range -0.0348 to -0.0047, 490 matches, 49 rounds).

## Decision

The stop rule drops the two-speed line: the weight on the fast fit is 0.00, under 0.10; the paired scoreline gain versus the current pure model is 0.0000, under 0.005; the 95% range of that gain is 0.0000 to 0.0000, which includes zero. The weight on the fast fit is 0.00. The published model stays the current pure model. The plan's expected shrink against the current pure model was 0.000 to 0.015. That size is not enough, on its own, to clear the reconstructed closing table. The chosen weight is 0.00, so on the later seasons the two-speed line is the current pure model. The paired gain is zero because the two lines are the same. On the setting seasons the scoreline log loss rose as the weight on the fast fit rose. The edge definition is not met. The reconstructed closing table is the competitor. This result is not a claim about the bookmaker. The scoreline gap against that table is -0.0287 (95% range -0.0502 to -0.0079, 490 matches, 49 rounds). On those 490 matches the pure model's scoreline log loss is 2.9188 and the reconstructed closing table's is 2.8900. The saved walk-forward gap of about 0.10 is the model from before strength 32. This run uses strength 32 and does not retune it.

Every scored match had a closing home, draw, away and over/under 2.5 price, so the pure-model comparison and the closing-table comparison use the same matches. Rounds with fewer than 200 earlier matches were not scored.

## Regenerate

```bash
python -m src.evaluate --mode two_speed
```

The run took 0.7 minutes. The per-match rows are `results/two_speed_matches.csv`. The ranges are `results/two_speed_paired.csv`.
