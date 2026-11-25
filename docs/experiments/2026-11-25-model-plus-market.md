# One weight between the model and the price before kickoff

**Status:** written before the run. This is not a result.
**Date:** 25 Nov 2026
**Slice:** one blend with the publish-time price (slice 9). Do not start that run until this note is committed.

## Question

Does one weight, mixing the model's goal rates with the goal rates implied by the price available before kickoff, produce a second score table worth publishing beside the pure model?

## Candidates

One number: the weight on the model's goal rates. The values are 0.00, 0.05, 0.10, 0.25, 0.50 and 1.00. A weight of 0 uses only the rates from the price available before kickoff. A weight of 1 uses only the model's rates. The two rates are mixed with the same weight. Nothing else is tried after looking.

The price available before kickoff is the opening home, draw, away and over/under 2.5 line. The closing price is not an input. It is only the competitor, through the reconstructed score table. That reconstruction is the one already in the repo. This run does not fit a new one.

The low-score correction on the blend, and on that reconstructed table, stays at zero. It is not searched. The time-decay setting stays at 0.0035. The thin-data strength stays at 32. The pure model is unchanged.

A tie keeps the smaller weight.

## Metric

Scoreline log loss chooses the weight. The same score, on the later seasons, is what the stop rule compares with the reconstructed closing table. Home, draw or away log loss is reported on the same matches and does not choose the weight. Two lines are compared only on the same matches. The gap is the mean difference, with a 95% range from resampling whole rounds, and the match count.

The pure model is the current score table, including its own fitted low-score correction. The blend is a separate line. Its label is model plus market.

## Data windows

Each round's model rates are fit only on matches dated before that round. The opening prices used for a match are that match's own prices from before kickoff. The weight is chosen by scoreline log loss on seasons 2223, 2324 and 2425. Those seasons already chose other settings, so they are not a clean test. Seasons 2526 and 2627 have been looked at before. They are not used to pick the weight. They are the later rounds in the stop rule. The live ledger is not a tuning set. No code in this run imports the ledger.

## Stop rule

On the later seasons, drop the second line if both of these are true. The chosen weight on the model, as written in the candidate list, is under 0.10. And the blend's scoreline log loss is not at least 0.01 lower than the reconstructed closing table. That gap is the closing table's scoreline log loss minus the blend's, rounded to 4 decimal places. A range that includes smaller numbers does not by itself change the point estimate. If the second line is dropped, the weight is still reported, and the pure model stays the published probability. A gap under about 0.01 is a point estimate inside its range. It is not a success, and it is not a claim about the bookmaker.

## If it fails

The pure model stays the published one. The note in `results/` records the weight, the paired gap against the reconstructed closing table, the range, and the match count. Nothing in that note describes the model as ahead of the bookmaker.

## If it is kept

A second forecast line is published beside the pure model, labelled model plus market. It is one score table from the blended rates. It is not a replacement for the pure model, and it is not described as a win over the bookmaker.

## Expected

The review that prompted this note expected the blend to close most of a gap of about 0.10 on the scoreline, and expected whatever is left against the reconstructed closing table to be somewhere from 0.00 to 0.02, possibly zero. Those figures had no range. They are the motive, not a settled gap.

## After the run

The sections above were written before the run. They stay the rules. The paired gaps, the ranges, and the match counts are in `results/MODEL_PLUS_MARKET.md`.
