# A scale and a steady low-score correction

**Date:** 25 Nov 2026 (London)
**Question:** does one scale on both goal rates, with one constant low-score correction, lower scoreline log loss on later seasons?

## What was fit

The previous line is the current Dixon-Coles score table. The time-decay setting stays at 0.0035 and the thin-data strength stays at 32. The low-score correction on that line is the value fitted on each round.

The new line takes those same fitted rates and multiplies both by one scale, then uses one constant low-score correction. The scales tried were 0.95, 1.00 and 1.05. The corrections tried were -0.10, -0.05 and 0.00. Nothing else was tried after looking.

## How the pair was chosen

Each round is fit only on matches dated before that round. The pair is the one with the lowest scoreline log loss on seasons 2223, 2324 and 2425. A tie keeps the scale closer to 1, then the correction closer to 0. Those seasons were already used to choose other settings, so this table is where the pair was picked. It is not a clean test.

| Scale | Correction | Scoreline log loss | Matches | Home/draw/away log loss |
|---|---|---|---|---|
| 0.95 | -0.10 | 3.0361 | 940 | 0.9652 |
| 0.95 | -0.05 | 3.0337 | 940 | 0.9645 |
| 0.95 | 0.00 | 3.0326 | 940 | 0.9644 |
| 1.00 | -0.10 | 3.0292 | 940 | 0.9633 |
| 1.00 | -0.05 | 3.0265 | 940 | 0.9630 |
| 1.00 (chosen) | 0.00 | 3.0251 | 940 | 0.9633 |
| 1.05 | -0.10 | 3.0299 | 940 | 0.9618 |
| 1.05 | -0.05 | 3.0269 | 940 | 0.9619 |
| 1.05 | 0.00 | 3.0254 | 940 | 0.9627 |

The chosen pair is scale 1.00 and correction 0.00.

Correction 0.00 is at the end of the list that was written down. No further value was added after looking.

The gap below is on the setting seasons. Those matches were used to pick the pair, so this gap is not a clean test.

On the setting seasons, scoreline log loss: the previous line minus the new line is 0.0015 (95% range -0.0012 to 0.0042, 940 matches, 94 rounds). The range includes zero.

Seasons 2526, 2627 were not used to pick the pair.

## Later seasons

The team rates are refit every round on earlier matches. The scale and the correction stay at the chosen pair. A gain is the previous line minus the new line on one shared set of matches. The range is a 95% percentile interval from 4000 resamples of whole rounds, seed 20261124. A positive gain means the new line has the lower log loss. A range that contains zero has not separated the two lines on this sample.

### Scoreline

On every later-season match, scoreline log loss: the previous line minus the new line is -0.0002 (95% range -0.0047 to 0.0046, 490 matches, 49 rounds). The range includes zero.

### Home, draw or away

On those same matches, home/draw/away log loss: the previous line minus the new line is -0.0014 (95% range -0.0047 to 0.0016, 490 matches, 49 rounds). The range includes zero.

### Closing line

On the later-season matches that have a closing home/draw/away line, the new line minus the closing line is 0.0227 (95% range 0.0070 to 0.0384, 490 matches, 49 rounds). The previous line minus the closing line on those same matches is 0.0213 (95% range 0.0056 to 0.0368, 490 matches, 49 rounds). The closing line is the competitor. These gaps are not a claim about the bookmaker. A positive difference means that line has the higher log loss. This run does not rebuild a score table from book prices.

## Decision

The stop rule drops the change: the paired scoreline gain on the later seasons is -0.0002, which is under 0.005. The published score table stays the previous one: no extra scale, and the low-score correction fitted on each round.

## Regenerate

```bash
python -m src.evaluate --mode grid_calibration
```

The run took 0.3 minutes. The per-match rows are `results/grid_calibration_matches.csv`. The ranges are `results/grid_calibration_paired.csv`.
