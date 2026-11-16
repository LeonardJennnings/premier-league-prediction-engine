# Goals model and derived markets

**Date:** 16 Nov 2026  
**Status:** Ready for implementation  
**Related:** [PRD](../prd.md) · [Project context](../project-context.md) · [Evaluation protocol](../EVAL_PROTOCOL.md) · [Walk-forward evaluation](walkforward-eval.md)

## Summary

Upgrade the match model from independent Poisson home/draw/away probabilities to a **Dixon–Coles** goals model with time decay. One scoreline probability grid becomes the single source for **1X2**, **both-teams-to-score (BTTS)**, and **over/under 2.5**. Public-facing 1X2 probabilities are **temperature-calibrated**. An **Elo** rating line and the existing independent Poisson remain as comparators in the walk-forward harness — not as the product engine.

## Motivation

Independent Poisson models are a clear teaching baseline, but they systematically mis-state low-score outcomes (especially 0–0 and 1–1) and treat a result from three seasons ago as equal to last weekend. For a public probability product, that shows up as overconfident or poorly calibrated 1X2 numbers, and as inconsistent BTTS / totals if those markets are fitted separately.

The design goal is generative consistency: estimate how matches produce scorelines, then **read markets off the same grid**. That is easier to explain, harder to game with tipster-style nets, and aligns with how serious football modelling has treated this problem since Dixon and Coles (1997).

## Scope

**In scope**

- Dixon–Coles maximum-likelihood fit with exponential time decay
- Scoreline grid → 1X2, BTTS, over/under 2.5
- Temperature scaling for published 1X2 (vector scaling retained as a research alternative)
- Elo and independent Poisson as evaluation comparators
- Walk-forward and season-gate scoring of the new lines
- Over/under bookmaker comparison where football-data.co.uk closing columns exist
- Historical offline outputs suitable for later fixture publishing

**Out of scope**

- Live fixture ingest and immutable prediction store (later delivery)
- Public API and Next.js UI
- Affiliates, tipster copy, or market-blend / value products
- Joint optimisation of the decay rate inside the likelihood
- Expected-goals (xG) inputs

## Approach

### Why Dixon–Coles

The model keeps Maher’s attack / defence / home-advantage structure and adds two ideas from Dixon and Coles:

1. **Low-score dependence** — a single parameter ρ adjusts only the joint probabilities of 0–0, 1–0, 0–1, and 1–1 via a correction τ. When ρ = 0, the model reduces to independent Poisson.
2. **Time decay** — older matches receive exponentially smaller weight so recent form dominates team strength estimates.

Expected goals for home \(i\) vs away \(j\):

\[
\lambda = \alpha_i\,\beta_j\,\gamma, \qquad
\mu = \alpha_j\,\beta_i
\]

Joint scoreline probability:

\[
P(X=x,Y=y)=\tau_{\lambda,\mu}(x,y)\,\mathrm{Pois}(x;\lambda)\,\mathrm{Pois}(y;\mu)
\]

with

\[
\tau_{\lambda,\mu}(x,y)=
\begin{cases}
1-\lambda\mu\rho & (0,0)\\
1+\mu\rho & (1,0)\\
1+\lambda\rho & (0,1)\\
1-\rho & (1,1)\\
1 & \text{otherwise}
\end{cases}
\]

τ is not a free parameter; it is determined by λ, μ, and ρ. Attack strengths are normalised so their mean equals 1 (the classic identifiability constraint).

### Time decay

Match age \(t\) in **days** before the as-of date is weighted by \(\phi(t)=e^{-\xi t}\). The decay rate ξ is a **frozen configuration constant** (`DIXON_COLES_XI`), chosen by nested walk-forward on older seasons and reported on a held-out recent era (`python -m src.evaluate --mode xi_tune`; see `results/XI_TUNE.md` and `docs/EVAL_PROTOCOL.md`). Dixon and Coles select ξ by a separate predictive criterion rather than joint maximum likelihood; this repo follows that rule (no in-likelihood MLE of ξ).

### Numerical safeguards grounded in the theory

| Concern | Handling |
|---|---|
| Invalid ρ (τ ≤ 0) | Optimiser bounds keep ρ inside the admissible interval implied by every training match, so the weighted log-likelihood never hits −∞ |
| Relative attack/defence drift | Mean attack fixed at 1 |
| Promoted / thin-history teams | L2 shrinkage toward the league mean, stronger when a team has few matches |
| Teams absent from the fit | League-average attack and defence at predict time |
| Finite scoreline grid | Truncate at a maximum goals value (default 8), apply τ, then renormalise |
| Optimiser failure | Rare path: log and fall back to the nested independent-Poisson grid (ρ = 0) so evaluation continues |

### Derived markets

From one renormalised grid:

| Market | Aggregation |
|---|---|
| Home / draw / away | Cells with home goals >, =, or < away goals |
| BTTS | Both teams ≥ 1 goal vs otherwise |
| Over / under 2.5 | Total goals ≥ 3 vs ≤ 2 |

There are no separate classifiers for BTTS or totals. That keeps markets coherent: changing team strength moves all three surfaces together.

### Calibration

Raw generative probabilities are often sharp but imperfectly calibrated. Before public 1X2 use, apply **temperature scaling** fitted only on predictions from **earlier** walk-forward folds (no leakage into the fold being scored). Early folds without enough prior predictions publish raw Dixon–Coles 1X2 until a documented sample threshold is reached.

**Vector scaling** (separate adjustments per outcome) lives in the same module as a research alternative. Temperature remains the default published path: one parameter, lower overfit risk, easier to explain on a how-it-works page.

Calibration is a post-hoc probability adjustment. It is not part of the 1997 Dixon–Coles paper; it is standard practice when generative forecasts are shown to people.

### Comparators

| Line | Purpose |
|---|---|
| Naive frequencies | Floor |
| Independent Poisson | Shows the lift from Dixon–Coles on the same family |
| Dixon–Coles (raw) | Generative model before temperature |
| Dixon–Coles (calibrated) | Default research and product 1X2 line |
| Elo | Simple rating comparator (1X2 only) |
| Sklearn baseline | Existing feature-based line |
| Bookmaker implied | De-vigged closing 1X2; over/under when columns exist |

BTTS is model-only until a reliable bookmaker series is available on the historical feed. Missing over/under odds exclude a row from the book comparison only; model totals are still scored against results.

## System shape

```text
historical matches
        │
        ├──────────────────────────────┐
        ▼                              ▼
 Dixon–Coles (MLE + decay)           Elo (comparator)
        │
        ▼
 scoreline grid
        │
        ▼
 markets (1X2, BTTS, O/U 2.5)
        │
        ▼
 temperature calibration (1X2)
        │
        ▼
 walk-forward / season-gate evaluation
 (+ independent Poisson, sklearn, book)
```

### Module boundaries

| Module | Responsibility |
|---|---|
| `src/models/dixon_coles.py` | Fit, scoreline grid, thin 1X2 predict helper |
| `src/models/markets.py` | Pure grid → market aggregations |
| `src/models/elo.py` | Elo fit / 1X2 predict |
| `src/models/poisson.py` | Existing independent Poisson (unchanged) |
| `src/eval/calibration.py` | Reliability plots; temperature and vector scaling |
| `src/eval/walkforward.py` | Extended model lines plus binary O/U and BTTS metrics (`market` labelled) |
| `src/clean.py` / `src/features.py` | Retain and de-vig over/under closing columns when present |

Primary interfaces:

- `fit_dixon_coles(matches, as_of=None, xi=...) -> DixonColesModel`
- `scoreline_grid(model, home, away, max_goals=8) -> np.ndarray`
- market helpers over that grid
- `fit_temperature` / `apply_temperature` (and vector-scaling equivalents)

## Evaluation

The Epic 1 harness remains the source of truth. This work extends it; it does not replace it.

- Refit Dixon–Coles (and Elo) on each walk-forward train window with strictly earlier matches only
- Report log loss, Brier, and accuracy for 1X2 lines as today
- Score model over/under against outcomes; compare to the book where closing totals odds exist
- Emit reliability plots for calibrated Dixon–Coles (and optionally raw)
- Keep the fixed season gate as a fast regression check

Offline artefacts should make it obvious how to regenerate numbers. Protocol documentation in `EVAL_PROTOCOL.md` will name the new lines and the over/under rule.

## Testing strategy

- ρ = 0 recovers independent Poisson probabilities on a fixed λ, μ
- Invalid ρ is rejected by bounds / likelihood guards
- Market probabilities sum to one on hand-built grids; BTTS and totals cell logic is checked explicitly
- Shrinkage pulls sparse teams toward the league mean
- Temperature fit/apply round-trips; vector scaling remains callable
- Elo ratings move in the expected direction on a tiny match sequence
- A short synthetic walk-forward includes the new lines without network I/O
- Existing Poisson, metrics, and gameweek tests stay green

## Success criteria

1. Historical pipeline produces calibrated 1X2 plus model BTTS and over/under 2.5 from one Dixon–Coles grid
2. Walk-forward and season-gate evaluation include Dixon–Coles (raw and calibrated), Elo, and over/under scoring where data allows
3. Evaluation protocol documentation describes the model family, decay, calibration, and market derivation in plain language
4. Automated tests cover the critical mathematical and leakage edges above

## References

- Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C*.
- Project evaluation design: [`walkforward-eval.md`](walkforward-eval.md)
- Product constraints for this build window: [`prd.md`](../prd.md)
