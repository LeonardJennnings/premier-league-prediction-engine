# Where the scoreline gap sits

**Date:** 26 Nov 2026 (London)

This note reads the saved walk-forward rows in `results/walkforward_match_scores.csv`. The fitted model stays as it is. The gap is Dixon-Coles scoreline log loss minus the loss of the closing reconstruction. A positive number means Dixon-Coles put less probability on the score that happened. The closing reconstruction is our reconstruction of the book's score table from its prices, not the bookmaker's correct-score odds.

## How the split is built

The saved file has the scoreline loss and the expected goals for Dixon-Coles, and the home and away rates for the closing reconstruction. The model rates are the Poisson rates that produce those expected goals on a 0-to-8 table with the low-score correction at zero. The low-score correction is the one value per round that rebuilds the saved Dixon-Coles loss from those rates. Attack and defence are not fit again. Rebuilding the closing table from its saved rates matches the saved loss (largest absolute difference 0). Rebuilding Dixon-Coles from the recovered rates and correction matches the saved loss (largest absolute difference 1.87e-09).

The rate piece is the saved model loss minus the loss of a table that uses the reconstruction's rates and the model's low-score correction. That is the part of the gap that disappears when the book's rates are swapped in. The shape piece is that swapped table minus the closing reconstruction, which keeps the correction at zero. That is the part that remains. On each match the two pieces add to the saved gap.

The range is a 95% percentile interval from 4000 resamples of whole rounds, seed 20261124. A range that contains zero has not separated the two lines on that set of matches.

## Same matches

These three numbers use the same 1430 matches and 143 rounds. They are parts of one gap, not a ranking.

| Group | Mean | 95% range | Matches | Rounds |
|---|---|---|---|---|
| model minus closing reconstruction | 0.0995 | 0.0608 to 0.1419 | 1430 | 143 |
| rate piece | 0.0986 | 0.0602 to 0.1409 | 1430 | 143 |
| shape piece | 0.0009 | -0.0019 to 0.0036 | 1430 | 143 |

## Where the tenth sits

The tenth is in the rates. The piece that remains after the reconstruction's rates are swapped in is under 0.01, so it is not the tenth. The piece that disappears with that swap is the tenth. A later change to the goal rates is aimed at this gap. A later change to the low-score correction is aimed at a piece under 0.01. On the same matches, model minus closing reconstruction is 0.0995 (95% range 0.0608 to 0.1419, 1430 matches, 143 rounds). The rate piece is 0.0986 (95% range 0.0602 to 0.1409, 1430 matches, 143 rounds). The shape piece is 0.0009 (95% range -0.0019 to 0.0036, 1430 matches, 143 rounds). The model home rate is lower than the reconstruction home rate, with the range clear of zero, in seasons 2425, 2526 and 2627. In seasons 2223 and 2324 that range includes zero. The closing reconstruction has the lower scoreline loss on these matches. Every coefficient stays as it is.

## By season

Each season is its own set of matches. The rows stay in season order. They are not ranked.

| Group | Mean | 95% range | Matches | Rounds |
|---|---|---|---|---|
| 2223 | 0.0755 | 0.0178 to 0.1351 | 180 | 18 |
| 2324 | 0.1338 | 0.0628 to 0.2228 | 380 | 38 |
| 2425 | 0.0724 | 0.0221 to 0.1465 | 380 | 38 |
| 2526 | 0.0691 | 0.0166 to 0.1500 | 380 | 38 |
| 2627 | 0.2185 | -0.0176 to 0.5110 | 110 | 11 |

## By the size of the total-rate miss

The total rate is the home rate plus the away rate. The miss is the absolute difference between Dixon-Coles and the closing reconstruction. The cuts are under 0.15, from 0.15 to 0.40, and over 0.40. A miss of 0.15 or 0.40 sits in the middle group. The cuts were fixed before these losses were read. The three groups are different matches. They are not ranked.

| Group | Mean | 95% range | Matches | Rounds |
|---|---|---|---|---|
| under 0.15 | 0.0250 | 0.0050 to 0.0453 | 411 | 140 |
| from 0.15 to 0.40 | 0.0474 | 0.0252 to 0.0699 | 621 | 142 |
| over 0.40 | 0.2577 | 0.1341 to 0.4045 | 398 | 132 |

## By the size of the rate-split miss

The rate split is the home rate minus the away rate. The miss is the absolute difference between Dixon-Coles and the closing reconstruction. The cuts are the same 0.15 and 0.40. The three groups are different matches. They are not ranked.

| Group | Mean | 95% range | Matches | Rounds |
|---|---|---|---|---|
| under 0.15 | 0.0227 | 0.0013 to 0.0434 | 441 | 137 |
| from 0.15 to 0.40 | 0.0646 | 0.0285 to 0.1190 | 547 | 143 |
| over 0.40 | 0.2192 | 0.1116 to 0.3440 | 442 | 140 |

## Home rate by season

This block is not a scoreline gap. It is the model home rate minus the reconstruction home rate, on the matches of that season. A positive number means the model home rate is higher. Seasons are different matches and are not ranked against each other, or against the scoreline tables above.

| Group | Mean | 95% range | Matches | Rounds |
|---|---|---|---|---|
| 2223 | 0.0451 | -0.0178 to 0.0995 | 180 | 18 |
| 2324 | 0.0111 | -0.0306 to 0.0517 | 380 | 38 |
| 2425 | -0.0591 | -0.0879 to -0.0316 | 380 | 38 |
| 2526 | -0.0358 | -0.0613 to -0.0103 | 380 | 38 |
| 2627 | -0.1210 | -0.2047 to -0.0419 | 110 | 11 |

## Rest days

Rest is the number of days since that club's previous match in the cleaned match list. A match on the same date does not count as earlier. Short rest means either club last played 3 days ago or fewer. That cut was fixed before these losses were read. A club's first match in the file has no rest day. 7 scored matches are missing a rest day for at least one club and are left out. The rest comparison uses 1423 matches. Short rest and not short rest are different matches. They are not ranked. No coefficient is fit.

| Group | Mean | 95% range | Matches | Rounds |
|---|---|---|---|---|
| short rest | 0.0558 | 0.0143 to 0.1008 | 221 | 55 |
| not short rest | 0.1083 | 0.0631 to 0.1570 | 1202 | 143 |

The short-rest gap sits with the overall gap. The overall mean 0.0995 lies inside the short-rest range. A later rest-day coefficient is not started.

## First eight matches of the season

Early means either club is inside its first eight matches of the season. The count is the one already used for the thin-data prior. A match on the same date does not count as earlier. Early and not early are different matches. They are not ranked. No coefficient is fit.

| Group | Mean | 95% range | Matches | Rounds |
|---|---|---|---|---|
| early | 0.1636 | 0.0783 to 0.2622 | 598 | 61 |
| not early | 0.0534 | 0.0319 to 0.0767 | 832 | 85 |

## Regenerate

```bash
python -m src.evaluate --mode gap_split
```

The grouped table is `results/gap_split.csv`.
