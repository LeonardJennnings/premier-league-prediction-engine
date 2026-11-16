# Epic 2 review fixes — what broke the “done” call

**Date:** 16 Nov 2026  
**Branch:** `epic-2-goals-model-derived-markets`  
**Related:** [`goals-model-derived-markets` spec](../specs/goals-model-derived-markets.md) · commits `199ec4d`, `f2e994d`

Epic 2 (Dixon–Coles + derived markets) passed task-by-task implementation, then two review gates found issues that had to be fixed before calling the branch merge-ready. This note is the engineering story: what looked wrong, why it mattered, and what we changed.

---

## 1. Optimizer `success=False` kept a bad fit

### The story

Dixon–Coles is fitted with `scipy.optimize.minimize` (L-BFGS-B) on a weighted negative log-likelihood. The design already said: if the optimizer fails, do not invent strengths — fall back to a simple Poisson moment init with ρ = 0 (the nested independent-Poisson model) and log a warning so walk-forward can continue.

The first implementation treated some failures correctly (exceptions, non-finite `x` / `fun`, objective stuck in the τ ≤ 0 penalty basin). It treated another case differently:

> SciPy returns `success=False` but a **finite** parameter vector.

That path logged a warning and **kept** the unsuccessful point. The docstring even documented that choice.

### Why that is wrong

`success=False` on L-BFGS-B is not “almost converged, ship it.” It covers iteration limits, abnormal line searches, and other stops where SciPy refuses to claim a solution. Keeping that point means:

- Walk-forward folds can silently publish attack / defence / ρ that never cleared the optimizer’s success flag.
- The “fallback” story in the spec becomes two stories: one honest nested-Poisson path, and one “keep whatever finite junk we had.”
- Reviewers (and future you) cannot tell from metrics alone whether a weird fold used a real MLE or a half-finished vector.

The tighter failure rule was already in the brief. The code drifted toward “finite is good enough.”

### The fix

Commit `199ec4d` folded `not result.success` into the same branch as the other hard failures:

1. Build the Poisson-init fallback once (ρ = 0).
2. If the optimizer raises, returns `success=False`, returns non-finite values, or sits in the τ penalty region → log and **return the fallback**.
3. Only unpack and accept `result.x` when `success` is true and the objective looks sane.

Net change was small (~15 lines in `src/models/dixon_coles.py`). Behaviour change is large: unsuccessful fits no longer leak into evaluation as if they were MLE solutions.

### What we did not do (yet)

There is still no unit test that mocks `minimize` with `success=False` and asserts ρ = 0 + init strengths. The path is covered by code review and the live fits we spot-checked (full-history and large train windows both reported true convergence). A forced-failure test remains a good follow-up.

---

## 2. Protocol and README ahead of the harness

### The story

Whole-branch review compared three surfaces that should agree:

| Surface | Role |
|---|---|
| Code / walk-forward harness | What is actually scored |
| `docs/EVAL_PROTOCOL.md` | What we claim is scored |
| `README.md` Results table | What a visitor sees for the season gate |

They did not agree.

**Over/under.** The protocol said (in effect) that model totals are still scored against results when book odds are missing. The harness only:

- stores Dixon–Coles `p_over_25` / `p_under_25` on in-memory prediction frames,
- scores a **`bookmaker_over_25`** binary line when implied O/U columns exist,
- never writes a model O/U summary metrics row.

So a careful reader of the protocol would look for a model O/U log-loss line that does not exist. Worse, `bookmaker_over_25` sits in the same CSV `log_loss` column as 1X2 models even though binary and three-way log loss are not comparable.

**Season-gate README.** `results/eval_results.csv` already had Dixon–Coles and Elo rows from the wired season gate. The README Results table still listed only naive / Poisson / sklearn / book — the pre–Epic 2 set.

### Why that is wrong

For this project, honesty is the product. A protocol that over-claims scoring is the same class of bug as a tipster rewriting history: the numbers look more complete than the system is. A README that omits new season-gate lines hides the work and understates what `eval_results.csv` already contains.

### The fix

Commit `f2e994d` (docs + artefact generator text; no new metrics):

1. **EVAL_PROTOCOL** — state clearly that model O/U probs are produced and stored, but are **not** a published walk-forward summary line; only `bookmaker_over_25` is scored when columns exist; do not rank that binary score against 1X2.
2. **README** — add Dixon–Coles and Elo to the season-gate table from the committed CSV (DC log loss 1.033, Elo 1.042 on the fixed `2526` gate) and update the reading paragraph without overselling.
3. **WALKFORWARD.md / generator string** — same O/U wording so regenerate notes match the protocol.

We deliberately did **not** rush a model O/U metrics line into the summary CSV just to make the old protocol true. Prefer aligning claims to the harness; add scoring later if we want that artefact.

---

## Takeaways

1. **Finite ≠ successful.** Optimizer APIs return several failure modes; map all of them to the designed fallback, not only the dramatic ones.
2. **Docs are part of the eval surface.** If the CSV, the protocol, and the README disagree, the review failed even when pytest is green.
3. **Prefer under-claiming.** Shipping grid-derived O/U probabilities without a summary metric is fine; claiming they are scored when they are not is not.

## Pointers

| Item | Location |
|---|---|
| Fallback behaviour | `src/models/dixon_coles.py` (`fit_dixon_coles`) |
| Optimizer fix | commit `199ec4d` |
| Protocol / README honesty | commit `f2e994d` |
| Spec (intended failure rule) | `docs/specs/goals-model-derived-markets.md` |
