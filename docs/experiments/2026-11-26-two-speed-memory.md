# Two speeds of team memory, one weight

**Status:** written before the run. This is not a result.
**Date:** 26 Nov 2026
**Bet:** two speeds of memory, one weight. Do not start the run until this note is committed.

## Question

Does one weight, mixing a slow Dixon-Coles fit with a fast Dixon-Coles fit, put more probability on the score that happened than the current pure model does, on matches the weight was not chosen on?

## Candidates

One number: the weight on the fast fit's goal rates. The values are 0.00, 0.05, 0.10, 0.25, 0.50 and 1.00. A weight of 0.00 uses only the slow fit. A weight of 1.00 uses only the fast fit's home rate and away rate. The same weight is used for the home rate and the away rate. Nothing else is tried after looking.

The slow decay stays at 0.0035, about a 198-day half-life. That is the published time-decay setting. It is not searched again.

The fast decay is 0.0120, about a 58-day half-life. That value is already in `XI_TUNE_CANDIDATES`. It is the fastest one on that list. No new decay is added, and this run does not choose among the list.

The low-score correction on the mixture is the slow fit's own fitted correction, projected onto the mixed rates. It is not a candidate. The thin-data strength stays at 32. The promoted-club shift stays at 0.75. The model-plus-market weight stays at 0.25. The stopped score-table calibration stays as it is: no extra scale, and no constant correction in place of the fitted one.

This is two fits of the current model. It is not a state-space model and it is not a Kalman filter. No rest coefficient is added. No shared home-strength term is added. The closing price is not an input. A tie keeps the smaller weight.

## Metric

Scoreline log loss chooses the weight. Lower is better. Two lines are compared only on the same matches. The gap is the mean difference, with a 95% range from resampling whole rounds (4000 draws, seed 20261124), and the match count and the round count.

The pure model is the current Dixon-Coles score table: the slow fit, including its fitted low-score correction. The two-speed line is a separate line. Home, draw or away log loss, and over/under 2.5 log loss, are reported on the same matches. They do not choose the weight.

The competitor for the edge definition is the reconstructed closing table already in the repo. Two goal rates are chosen so a Poisson table, with the low-score correction at zero, matches the de-vigged closing home, draw, away and over/under 2.5 prices. That reconstruction is our reconstruction of the book's score table from its prices, not the bookmaker's correct-score odds. The closing price is not an input to either model line.

An edge is all of the following, together, on seasons 2526 and 2627. The two-speed line has a lower scoreline log loss than that reconstructed table. The matches are ones the weight was not chosen on. The gap is that table's scoreline log loss minus the two-speed line's. Positive means the two-speed line was closer to the score that happened. The 95% range excludes zero. The match count and the round count are printed with the range. The gap versus the current pure model is reported beside it. Home, draw or away log loss does not meet this definition.

## Data windows

Each round's two fits use only matches dated before that round. The weight is chosen by scoreline log loss on seasons 2223, 2324 and 2425. Those seasons already chose the time decay, the thin-data strength, and the blend weight, so they are not a clean test. Seasons 2526 and 2627 have been looked at before. They are not used to pick the weight. They are scored after the weight is frozen. The weight on a later-season match was chosen only on earlier seasons. The live ledger is not a tuning set. No code in this run imports the ledger.

The rate-versus-shape split already placed the scoreline gap in the rates, which is why this bet starts. This run does not repeat that split.

## Stop rule

On seasons 2526 and 2627, drop the two-speed line if any of these is true. The chosen weight on the fast fit is under 0.10. The paired scoreline gain versus the current pure model, pure loss minus two-speed loss, rounded to 4 decimal places, is under 0.005. The 95% range of that gain includes zero.

## If it fails

The published model stays the current pure model. The note in `results/` records the weight, the paired gain against the pure model, the paired gap against the reconstructed closing table, both ranges, and the match counts. A small gap is a point estimate inside a range. It is not a win. The book is not described as beaten unless the edge definition above is met.

## If it is kept

If the stop rule does not fire, and the edge definition is not met, the two-speed line is published only as a labelled alternative. It does not replace the pure model, and it is not a silent swap.

The two-speed line replaces the pure model only when the stop rule does not fire and the edge definition is met. That is the only case in which the published probability changes. Home, draw, away, both teams to score, and over/under would then be sums of that one table.

## Expected

The plan's expected shrink against the current pure model is 0.000 to 0.015. That size is not enough, on its own, to clear the reconstructed closing table. On the later seasons the current pure model sits about 0.10 behind that table, and the range on that gap runs from about 0.03 to about 0.19. A shrink of a few thousandths leaves that range covering zero. A result inside the expected band is a smaller gap with the range still including zero. It is worth writing down. It is not an edge, and it is not enough to say the book was beaten.

## After the run

The sections above were written before the run. They stay the rules. The paired gaps, the ranges, and the match counts go in `results/TWO_SPEED.md`. They are not added to this file.
