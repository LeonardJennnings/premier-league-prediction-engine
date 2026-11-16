# Evaluation notes

## Test set rule

- Test set: all Premier League matches in season 2526 with closing odds present
- Train set: configured seasons strictly before 2526 (2223, 2324, 2425)
- Do not change this rule just to make scores look nicer

## Metrics

- Log loss (lower better): punishes confident wrong probabilities
- Brier score (lower better): mean squared error of the three outcome probabilities
- Accuracy (higher better): share of matches where the top predicted outcome is correct — useful supporting number, not the only score

## Models compared

- naive_frequencies — always predict historical H/D/A rates from train
- poisson — goals model fit on train only
- sklearn_baseline — logistic regression on implied odds, fit on train only
- bookmaker_implied — normalised closing Bet365 probabilities on the test rows
- bookmaker_opening — normalised pre-closing Bet365 probabilities when `B365H/D/A` are present (separate line; do not mix with closing when claiming an edge)

## What good looks like

- Naive should be worst
- Our models should beat naive
- Getting close to bookmakers is a strong result
- A huge wosing odds is more likely a bug or leakage than a breakthrough — investigate first

## Howo regenerate

With the venv active, from the project root run:

    python -m src.evaluate

Results are saved to results/eval_results.csv (committed on purpose; raw match CSVs are not).
