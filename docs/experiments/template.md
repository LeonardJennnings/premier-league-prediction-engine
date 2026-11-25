# Experiment note

Write this before the run. Commit it before the code that tries the idea. Fill every section. If a section does not apply, say so in one sentence. Do not add a result to this file after the run. The result goes in `results/`, next to the paired table.

## Question

One sentence. What would have to be true for this change to be worth keeping?

## Candidates

The numbers that are allowed to change. Name each one. A run that adds a number not listed here does not count.

## Metric

The score, the matches it is computed on, and how two lines are compared. A comparison uses the same matches, a mean difference, a 95% range from resampling whole rounds, and a match count.

## Data windows

Which matches may be used to choose a number, and which matches are only scored. A number sees only earlier rounds. The seasons already used to choose the time-decay setting are not a fresh holdout. The live ledger is not a tuning set.

## Stop rule

The result that means the change is dropped. Write it before looking.

## If it fails

What is published anyway. Usually that is the paired gap, the range, and the match count, with the previous model left in place. A small gap is a point estimate inside a range. It is not a win, and it is not a claim about the bookmaker.

## Expected

A range, not a promise. Say where the change should show up, and how large a gain would be unsurprising.
