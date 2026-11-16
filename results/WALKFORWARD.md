# Walk-forward artefacts

## Protocol

- Folds: synthetic gameweeks (<=10 matches, date-ordered within each season)
- Train: all matches with date strictly before the fold
- Features: built per fold; history-dependent columns use date < fold cutoff only
- Minimum train rows: 200
- Models: naive_frequencies, poisson, sklearn_baseline, bookmaker_implied
- Summary metrics: pooled over all scored matches (per-fold CSV for diagnosis)
- Calibration plots: home-win reliability

## Regenerate

```bash
source .venv/bin/activate
python -m src.evaluate --mode walkforward
```

Season gate: `python -m src.evaluate --mode season`

See docs/EVAL_PROTOCOL.md for interpretation.
