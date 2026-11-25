# One weight between the model and the price before kickoff

**Date:** 25 Nov 2026 (London)
**Question:** does one weight, mixing the model's goal rates with the rates from the price available before kickoff, earn a second score table beside the pure model?

## What was fit

The pure model is the current Dixon-Coles score table. The time-decay setting stays at 0.0035 and the thin-data strength stays at 32. Its low-score correction is the value fitted on each round.

The second line mixes that model's goal rates with the goal rates implied by the opening home, draw, away and over/under 2.5 prices. The closing price is not an input. The weights tried were 0.00, 0.05, 0.10, 0.25, 0.50 and 1.00. Nothing else was tried after looking. The blend and the reconstructed closing table both leave the low-score correction at zero.

The reconstructed table is our reconstruction of the book's score table from its prices, not the bookmaker's correct-score odds.

## How the weight was chosen

Each round's model rates are fit only on matches dated before that round. The weight is the one with the lowest scoreline log loss on seasons 2223, 2324 and 2425. A tie keeps the smaller weight. Those seasons were already used to choose other settings, so this table is where the weight was picked. It is not a clean test.

| Weight on the model | Scoreline log loss | Matches | Home/draw/away log loss |
|---|---|---|---|
| 0.00 | 3.0060 | 940 | 0.9431 |
| 0.05 | 3.0051 | 940 | 0.9432 |
| 0.10 | 3.0043 | 940 | 0.9433 |
| 0.25 (chosen) | 3.0033 | 940 | 0.9443 |
| 0.50 | 3.0056 | 940 | 0.9480 |
| 1.00 | 3.0251 | 940 | 0.9633 |

The chosen weight on the model is 0.25.



The gap below is on the setting seasons. Those matches were used to pick the weight, so this gap is not a clean test.

On the setting seasons, scoreline log loss: the pure model minus the blend is 0.0233 (95% range 0.0105 to 0.0364, 940 matches, 94 rounds).

Seasons 2526, 2627 were not used to pick the weight.

## Later seasons

The model rates are refit every round on earlier matches. The weight stays at the chosen value. The opening prices used for a match are that match's own prices. A range is a 95% percentile interval from 4000 resamples of whole rounds, seed 20261124. A range that contains zero has not separated the two lines on this sample.

### Against the reconstructed closing table

This is the stop comparison. The difference is the reconstructed closing table minus the blend. A positive number means the blend has the lower scoreline log loss.

On every later-season match, scoreline log loss: the reconstructed closing table minus the blend is -0.0073 (95% range -0.0162 to 0.0018, 490 matches, 49 rounds). The range includes zero.

### Against the pure model

The difference is the pure model minus the blend. A positive number means the blend has the lower log loss.

On those same matches, scoreline log loss: the pure model minus the blend is 0.0215 (95% range 0.0059 to 0.0377, 490 matches, 49 rounds).

On those same matches, home/draw/away log loss: the pure model minus the blend is 0.0151 (95% range 0.0039 to 0.0269, 490 matches, 49 rounds).

### Closing price

On the later-season matches, home/draw/away log loss: the blend minus the closing price is 0.0062 (95% range -0.0011 to 0.0133, 490 matches, 49 rounds). The closing price is the competitor. This is not a claim about the bookmaker. A positive difference means the blend has the higher log loss. The range includes zero.

## Decision

The stop rule keeps the second line: the weight on the model is 0.25, which is not under 0.10. The scoreline gain against the reconstructed closing table is -0.0073. The published second line uses weight 0.25 and is labelled model plus market. The pure model stays beside it. The rule uses the point estimate, rounded to 4 decimal places. The second line is written for a match only when the price available before kickoff has home, draw, away and over/under 2.5. A match without that price keeps the pure model only. The closing price is not an input to that line. The gap against the reconstructed closing table is under 0.01. That is a point estimate inside the range above. It is not a success, and it is not a claim about the bookmaker.

Every scored match had an opening and a closing home, draw, away and over/under 2.5 price. Rounds with fewer than 200 earlier matches were not scored.

## Regenerate

```bash
python -m src.evaluate --mode market_blend
```

The run took 0.6 minutes. The per-match rows are `results/market_blend_matches.csv`. The ranges are `results/market_blend_paired.csv`.
