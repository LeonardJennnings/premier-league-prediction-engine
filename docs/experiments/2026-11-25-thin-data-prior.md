# A starting strength for clubs with few matches

**Status:** written before the run. This is not a result.
**Date:** 25 Nov 2026
**Slice:** the thin-data prior (slice 7). Do not start that run until this note is on `main`.

## Question

Does one new strength number, which pulls a club with few matches toward a starting attack and defence, improve the score on that club's early matches without making the overall score worse?

## Candidates

One number: how hard a short history is pulled toward the starting rates. A promoted club starts lower than a club that already has a history. No second number in this run. The time-decay setting stays at the frozen value 0.0035. The low-score correction is not retuned here.

## Metric

Home, draw or away log loss, and scoreline log loss. Two lines are compared only on the same matches. The gap is the mean difference, with a 95% range from resampling whole rounds, and the match count. Early matches means a club's first eight matches in a season.

## Data windows

Any choice of the new number uses only rounds earlier than the matches being scored. The seasons 2223, 2324 and 2425 were already used to choose the time-decay setting, so they are not a clean holdout for this note. The holdout that was set aside for that choice (480 matches at the time, 490 in the current file) has been looked at, so it is not used to pick this number either. The live ledger is not a tuning set. No code in this run imports the ledger.

## Stop rule

Drop the change if the early-match score is no better in a season later than the one used to set the number, or if the overall score gets worse. A gain under about 0.01 nats is reported as a point estimate inside its range. It is not a reason to call the change a success.

## If it fails

The previous model stays the published one. The note in `results/` records the paired gap, the range, and the match count, including the early-match slice. Nothing in that note describes the model as ahead of the bookmaker.

## Expected

The review that prompted this note measured the current season as 1.142 for Dixon-Coles and 1.035 for Elo, on the same 110 matches, with the bookmaker at 1.023. That review did not attach a range, so those three figures are the motive, not a settled gap. The hope is a clear improvement on a club's first eight or so matches, and an overall change of 0.000 to 0.005 nats. That range is an expectation, not a target to hit.

## After the run

The sections above were written before the run. They stay the rules. The paired gaps, the ranges, and the match counts are in `results/THIN_DATA_PRIOR.md`.
