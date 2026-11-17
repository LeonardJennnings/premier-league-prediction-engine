# Walk-forward artefacts

## Protocol

- Folds: synthetic gameweeks (<=10 matches, date-ordered within each season)
- Train: all matches with date strictly before the fold
- Features: built per fold; history-dependent columns use date < fold cutoff only
- Minimum train rows: 200
- Models (1X2, market=1x2): naive_frequencies, poisson, dixon_coles, dixon_coles_calibrated, elo, sklearn_baseline, bookmaker_implied (Bet365 closing), bookmaker_opening (Bet365 pre-closing)
- dixon_coles_calibrated: temperature scaling fit on prior-fold Dixon–Coles 1X2 predictions only. The line is skipped until that prior store has at least 200 rows (the raw line is not copied in). Summary fold counts can therefore differ from raw Dixon–Coles; compare them on shared gameweeks.
- Binary markets (do not rank against 1X2): dixon_coles_over_25 (market=over_under_25) and dixon_coles_btts (market=btts) are scored from the raw scoreline grid every fold. bookmaker_over_25 uses closing totals (`implied_over_25`); bookmaker_over_25_open uses pre-closing totals when present. Missing O/U odds drop a row from that book comparison only. Temperature scaling does not change O/U or BTTS.
- ξ (`DIXON_COLES_XI`) is a frozen config constant from nested tune (`python -m src.evaluate --mode xi_tune`); not jointly estimated. See `results/XI_TUNE.md`.
- Summary metrics: pooled over all scored matches (per-fold CSV for diagnosis)
- Calibration plots: home-win reliability for 1X2 lines

## Regenerate

```bash
source .venv/bin/activate
python -m src.evaluate --mode walkforward
```

Season gate: `python -m src.evaluate --mode season`

See docs/EVAL_PROTOCOL.md for interpretation.
