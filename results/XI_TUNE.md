# Nested ξ tune artefacts

## Protocol

- **Selection:** gameweek walk-forward Dixon–Coles 1X2 log loss on **tune** seasons `2223, 2324, 2425` only.
- **Freeze:** lowest tune pooled log loss; ties → nearer baseline ξ, then smaller ξ.
- **Holdout report:** seasons `2526, 2627` (not used for selection).
- **Not used:** joint MLE of ξ inside the match likelihood; re-tuning on holdout / headline folds after the fact.
- **Baseline ξ (pre-tune):** `0.0019` (~365-day half-life).
- **Frozen ξ:** `0.0035` (~198-day half-life); tune LL `0.9612` on `940` matches.
- Book line on holdout is **Bet365 closing** de-vigged 1X2 (`bookmaker_implied`). Opening odds are a separate comparator (see walk-forward artefacts).

## Holdout comparison (1X2)

| Line | ξ | n | Log loss | Brier | Acc |
|---|---|---|---|---|---|
| DC frozen ξ | 0.0035 | 480 | 1.0670 | 0.2061 | 0.4688 |
| DC baseline ξ | 0.0019 | 480 | 1.0628 | 0.2056 | 0.4813 |
| Closing book | — | 480 | 1.0125 | 0.2025 | 0.4938 |

Frozen ξ **does not** beat the closing book on this holdout (expected for a goals-only model).

## Regenerate

```bash
source .venv/bin/activate
python -m src.evaluate --mode xi_tune
```

Candidate grid and season split live in `src/config.py` (`XI_TUNE_*`).
`DIXON_COLES_XI` is frozen at **0.0035** from this run; `DIXON_COLES_XI_BASELINE`
keeps the pre-tune **0.0019** for before/after tables. Holdout slightly preferred
the baseline (1.0628 vs 1.0670) — expected noise; the protocol still freezes the
tune winner and does **not** peek at holdout to re-select.

See `docs/EVAL_PROTOCOL.md` (§ Nested ξ tune).
