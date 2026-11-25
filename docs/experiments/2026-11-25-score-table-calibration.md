# A scale and a steady low-score correction

**Status:** written before the run. This is not a result.
**Date:** 25 Nov 2026
**Slice:** calibrate the score table (slice 8). Do not start that run until this note is committed.

## Question

Does one scale on both goal rates, together with one constant low-score correction, lower scoreline log loss on later seasons by at least 0.005 compared with the current score table?

## Candidates

Two numbers, and no others.

The scale is applied to both goal rates after the fit, so the two rates stay in proportion. The values are 0.95, 1.00 and 1.05. A scale of 1.00 leaves the rates alone.

The constant low-score correction is used on the score table in place of the value fitted on that round. The values are -0.10, -0.05 and 0.00. The fitted value is still estimated, because that is the previous model. It is not one of these three.

Every pair from those two lists is tried. That is nine pairs. Nothing else is added after looking. The time-decay setting stays at 0.0035. The thin-data strength stays at 32. The promoted-club shift stays at 0.75.

A tie keeps the scale closer to 1, then the correction closer to 0, then the smaller scale, then the smaller correction.

## Metric

Scoreline log loss chooses the pair and decides the stop rule. Home, draw or away log loss is reported on the same matches and does not choose the pair. Two lines are compared only on the same matches. The gap is the mean difference, with a 95% range from resampling whole rounds, and the match count.

The previous line is the current score table: scale 1, and the low-score correction fitted on that round. The new line is the chosen pair, applied to the same fitted rates.

The closing home, draw or away line is the competitor. This run does not rebuild a score table from book prices.

## Data windows

Each round is fit only on matches dated before that round. The pair is chosen by scoreline log loss on seasons 2223, 2324 and 2425. Those seasons already chose the time-decay setting and the thin-data strength, so they are not a clean test. Seasons 2526 and 2627 have been looked at before. They are not used to pick the pair. They are the later rounds in the stop rule. The live ledger is not a tuning set. No code in this run imports the ledger.

## Stop rule

The gain is the previous line's scoreline log loss minus the new line's, on seasons 2526 and 2627, same matches. Drop the change if that gain, rounded to 4 decimal places, is under 0.005. A range that includes smaller numbers does not by itself drop a gain that clears 0.005. A gain under about 0.01 is a point estimate inside its range. It is not a success, and it is not a claim about the bookmaker.

## If it fails

The published score table stays as it is: no extra scale, and the low-score correction fitted each round. The note in `results/` records the paired gap, the range, and the match count. Nothing in that note describes the model as ahead of the bookmaker.

## If it is kept

The published score table uses the chosen scale and the chosen constant correction. Home, draw, away, both teams to score, and over/under are sums of that one table. The separate home, draw or away temperature is not what the public row shows.

## Expected

The review that prompted this note expected a scoreline change of 0.000 to 0.015. It also saw the fitted low-score correction change sign from round to round, from about -0.12 to about 0.11, with a full-sample value near -0.07. Those figures had no range. They are the motive for the candidate list, not a settled gap.
