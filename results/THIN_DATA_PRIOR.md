# A starting strength for clubs with few matches

**Date:** 25 Nov 2026 (London)
**Question:** does one strength, pulling a short history toward a starting attack and defence, improve early matches without making the overall score worse?

## What was fit

The previous model is Dixon-Coles with the time-decay setting frozen at 0.0035 and the small existing penalty. The low-score correction was not retuned.

The new line replaces that small penalty with one strength. A club with no match in the previous season starts at 0.75 times an average club's attack, and its concession rate is higher by the same fraction. That fraction was fixed before the run. It was not searched. The strengths tried were 0, 2, 4, 8, 16, 32. Nothing else was tried after looking at the scores.

## How the strength was chosen

Each round is fit only on matches dated before that round. The strength is the one with the lowest scoreline log loss on early matches in seasons 2223, 2324, 2425. Early means a club's first eight matches in the season. A tie keeps the smaller strength. Those three seasons were already used to choose the time-decay setting, so this table is where the strength was picked. It is not a clean test.

| Strength | Early scoreline log loss | Early matches | Overall scoreline log loss | Overall matches | Early home/draw/away log loss | Overall home/draw/away log loss |
|---|---|---|---|---|---|---|
| 0 | 3.1269 | 328 | 3.0928 | 940 | 0.9605 | 0.9612 |
| 2 | 3.0058 | 328 | 3.0450 | 940 | 0.9557 | 0.9580 |
| 4 | 2.9974 | 328 | 3.0383 | 940 | 0.9557 | 0.9572 |
| 8 | 2.9877 | 328 | 3.0305 | 940 | 0.9558 | 0.9566 |
| 16 | 2.9801 | 328 | 3.0249 | 940 | 0.9574 | 0.9580 |
| 32 (chosen) | 2.9797 | 328 | 3.0266 | 940 | 0.9624 | 0.9635 |

The chosen strength is 32. The gap below is on the early matches in the setting seasons. Those matches were used to pick the strength, so this gap is not a clean test.

On the setting-season early matches, scoreline log loss: the prior line minus the previous line is -0.1473 (95% range -0.2622 to -0.0546, 328 matches, 34 rounds).

Strength 32 is the largest one that was tried. No larger strength was added after looking. The table above is the whole grid.

Seasons 2526, 2627 were not used to pick the strength.

## Later seasons

The model is refit every round on earlier matches, with the strength held fixed. A gap is the prior line minus the previous line on one shared set of matches. The range is a 95% percentile interval from 4000 resamples of whole rounds, seed 20261124. A negative difference means the prior line has the lower log loss. A range that contains zero has not separated the two lines on this sample.

### Early matches

On early matches, scoreline log loss: the prior line minus the previous line is -0.1213 (95% range -0.2816 to 0.0078, 270 matches, 27 rounds). The range includes zero.

On those same early matches, home/draw/away log loss: the prior line minus the previous line is -0.0601 (95% range -0.1925 to 0.0330, 270 matches, 27 rounds). The range includes zero.

### Every later-season match

On every later-season match, scoreline log loss: the prior line minus the previous line is -0.0739 (95% range -0.1635 to -0.0020, 490 matches, 49 rounds).

On those same matches, home/draw/away log loss: the prior line minus the previous line is -0.0300 (95% range -0.1042 to 0.0231, 490 matches, 49 rounds). The range includes zero.

### Closing line

On the later-season matches that have a closing home/draw/away line, the prior line minus the closing line is 0.0213 (95% range 0.0056 to 0.0368, 490 matches, 49 rounds). The previous line minus the closing line on those same matches is 0.0513 (95% range -0.0021 to 0.1225, 490 matches, 49 rounds). The closing line is the competitor. These gaps are not a claim about the bookmaker. A positive difference means that line has the higher log loss.

### Clubs that missed the previous season

This is not the stop rule. The stop rule uses every club's first eight matches, which is the definition written down before the run. The rows below are only the matches in the later seasons where a club with no match in the previous season is still inside those eight.

On that smaller slice, scoreline log loss: the prior line minus the previous line is -0.4493 (95% range -0.9791 to -0.0213, 77 matches, 27 rounds).

## Decision

The stop rule keeps the change: early-match scoreline log loss is lower on the later seasons, and the overall scores are not higher. The published fit uses strength 32. The early-match range includes zero, so that slice has not separated the two lines. The rule uses the point estimates. The overall movement is concentrated on clubs that missed the previous season: that smaller slice is -0.4493 (95% range -0.9791 to -0.0213, 77 matches, 27 rounds). It is not a claim about the bookmaker.

## Regenerate

```bash
python -m src.evaluate --mode thin_prior
```

The run took 1.9 minutes. The per-match rows are `results/thin_prior_matches.csv`. The ranges are `results/thin_prior_paired.csv`.
