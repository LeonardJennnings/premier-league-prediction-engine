# Goals model + derived markets — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.  
> **For this repo’s preferred path:** work the plan in tutor mode (explain → learner types → verify). Skip automatic commits unless Leonard asks to commit.

**Goal:** Ship Dixon–Coles (MLE + time decay) with a frozen scoreline grid that yields calibrated 1X2, BTTS, and O/U 2.5, scored by the existing walk-forward harness alongside Elo and independent Poisson comparators.

**Architecture:** Layered modules — `dixon_coles.py` fits and builds grids; `markets.py` aggregates cells; `elo.py` is a 1X2 comparator; `calibration.py` gains temperature (default) and vector scaling (research). Walk-forward / season gate import the new lines without changing leakage rules.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy (already in requirements), scikit-learn, matplotlib, pytest.

**Spec:** [`docs/specs/goals-model-derived-markets.md`](../specs/goals-model-derived-markets.md) · Protocol: [`docs/EVAL_PROTOCOL.md`](../EVAL_PROTOCOL.md)

## Global Constraints

- Premier League only; historical E0 via existing clean/features pipeline
- One scoreline grid → 1X2 / BTTS / O/U 2.5 (no separate market nets)
- Leakage: train only on matches with `date` strictly before the test fold; temperature fit uses only prior-fold predictions
- ξ fixed in config (days); do not jointly MLE ξ
- Mean attack = 1; ρ kept inside τ > 0 bounds; L2 shrinkage for sparse teams
- Keep independent Poisson; add Elo as comparator
- Temperature is the default calibrated 1X2 line; vector scaling is available but not the walk-forward headline
- BTTS book optional/absent; O/U book only when columns exist
- No tipster language in docs
- No fixtures API / prediction store / Next.js in this plan
- Skip automatic git commits unless Leonard asks

## File map

| Path | Responsibility |
|---|---|
| `src/config.py` | `DIXON_COLES_XI`, `DIXON_COLES_MAX_GOALS`, `DIXON_COLES_L2`, `CALIBRATION_MIN_ROWS`, Elo defaults |
| `src/models/markets.py` | Grid → 1X2 / BTTS / O/U helpers |
| `src/models/dixon_coles.py` | MLE fit, τ, scoreline grid, predict wrapper |
| `src/models/elo.py` | Elo ratings → 1X2 |
| `src/eval/calibration.py` | Keep reliability plots; add temperature + vector scaling |
| `src/clean.py` | Map O/U odds columns when present |
| `src/features.py` | De-vig O/U implied probs |
| `src/eval/walkforward.py` | New model lines + optional O/U metrics |
| `src/evaluate.py` | Season gate includes new lines |
| `src/models/predict.py` | Offline helper: calibrated 1X2 + BTTS + O/U for a match table |
| `docs/EVAL_PROTOCOL.md` | Document new lines |
| `results/WALKFORWARD.md` | Regenerated model list |
| `tests/test_markets.py` | Market aggregation |
| `tests/test_dixon_coles.py` | τ, ρ=0, fit/predict, shrinkage |
| `tests/test_elo.py` | Rating direction / probs |
| `tests/test_calibration.py` | Extended temperature / vector |
| `tests/test_features.py` / `test_clean.py` | O/U columns |
| `tests/test_walkforward.py` | Smoke with new lines on synthetic data |

---

### Task 1: Market aggregation from a scoreline grid

**Files:**
- Create: `src/models/markets.py`
- Create: `tests/test_markets.py`

**Interfaces:**
- Consumes: `numpy.ndarray` grid with shape `(max_goals+1, max_goals+1)`, index `(home_goals, away_goals)`, already non-negative
- Produces:
  - `normalize_grid(grid: np.ndarray) -> np.ndarray`
  - `markets_from_grid(grid: np.ndarray) -> dict` with keys `home`, `draw`, `away`, `btts_yes`, `btts_no`, `over_25`, `under_25` (floats summing appropriately)

- [ ] **Step 1: Write the failing test**

Create `tests/test_markets.py`:

```python
import numpy as np

from src.models.markets import markets_from_grid, normalize_grid


def test_normalize_grid_sums_to_one():
    grid = np.ones((3, 3), dtype=float)
    out = normalize_grid(grid)
    assert abs(out.sum() - 1.0) < 1e-12


def test_markets_from_hand_grid():
    # 2x2 toy: mass on 1-0 and 0-1 only after normalize
    grid = np.zeros((2, 2), dtype=float)
    grid[1, 0] = 0.6  # home win 1-0
    grid[0, 1] = 0.4  # away win 0-1
    m = markets_from_grid(grid)
    assert abs(m["home"] - 0.6) < 1e-12
    assert abs(m["away"] - 0.4) < 1e-12
    assert abs(m["draw"] - 0.0) < 1e-12
    assert abs(m["btts_yes"] - 0.0) < 1e-12
    assert abs(m["btts_no"] - 1.0) < 1e-12
    assert abs(m["under_25"] - 1.0) < 1e-12
    assert abs(m["over_25"] - 0.0) < 1e-12


def test_btts_and_over_on_larger_grid():
    grid = np.zeros((4, 4), dtype=float)
    grid[2, 2] = 1.0  # 2-2: BTTS yes, over 2.5
    m = markets_from_grid(grid)
    assert abs(m["btts_yes"] - 1.0) < 1e-12
    assert abs(m["over_25"] - 1.0) < 1e-12
    assert abs(m["draw"] - 1.0) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_markets.py -v`  
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `src/models/markets.py`**

```python
"""Derive match markets by summing cells of a scoreline probability grid."""

from __future__ import annotations

import numpy as np


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=float)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("grid must be a square 2D array")
    total = g.sum()
    if total <= 0:
        raise ValueError("grid must have positive total mass")
    return g / total


def markets_from_grid(grid: np.ndarray) -> dict[str, float]:
    """Return 1X2, BTTS, and O/U 2.5 probabilities from a scoreline grid."""
    g = normalize_grid(grid)
    n = g.shape[0]
    home = draw = away = 0.0
    btts_yes = btts_no = 0.0
    over_25 = under_25 = 0.0
    for i in range(n):
        for j in range(n):
            p = float(g[i, j])
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
            if i >= 1 and j >= 1:
                btts_yes += p
            else:
                btts_no += p
            if i + j >= 3:
                over_25 += p
            else:
                under_25 += p
    return {
        "home": home,
        "draw": draw,
        "away": away,
        "btts_yes": btts_yes,
        "btts_no": btts_no,
        "over_25": over_25,
        "under_25": under_25,
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_markets.py -v`  
Expected: PASS

- [ ] **Step 5: Commit** (only if Leonard asks)

```bash
git add src/models/markets.py tests/test_markets.py
git commit -m "$(cat <<'EOF'
Add scoreline-grid market aggregation helpers.

EOF
)"
```

---

### Task 2: Dixon–Coles τ, grid, and ρ = 0 equivalence

**Files:**
- Create: `src/models/dixon_coles.py` (stubs + grid helpers first)
- Create: `tests/test_dixon_coles.py`
- Modify: `src/config.py` (add constants)

**Interfaces:**
- Produces:
  - `tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float`
  - `scoreline_grid_from_rates(lam, mu, rho, max_goals) -> np.ndarray`
  - `DixonColesModel` dataclass (fields filled in Task 3): `attack`, `defence`, `home_advantage`, `rho`, `xi`, `as_of`

- [ ] **Step 1: Add config defaults**

In `src/config.py` add:

```python
# Dixon–Coles: ξ in per-day units (~1 year half-life ≈ ln(2)/365)
DIXON_COLES_XI = 0.0019
DIXON_COLES_MAX_GOALS = 8
DIXON_COLES_L2 = 0.01
CALIBRATION_MIN_ROWS = 200
ELO_K = 20.0
ELO_HOME_BONUS = 60.0
ELO_INITIAL = 1500.0
```

- [ ] **Step 2: Write failing tests for τ / grid**

Create `tests/test_dixon_coles.py`:

```python
import math

import numpy as np

from src.models.dixon_coles import scoreline_grid_from_rates, tau
from src.models.markets import markets_from_grid


def test_tau_rho_zero_is_one():
    assert tau(0, 0, 1.2, 1.1, 0.0) == 1.0
    assert tau(1, 1, 1.2, 1.1, 0.0) == 1.0


def test_tau_formulas():
    lam, mu, rho = 1.5, 1.2, -0.1
    assert abs(tau(0, 0, lam, mu, rho) - (1 - lam * mu * rho)) < 1e-12
    assert abs(tau(1, 0, lam, mu, rho) - (1 + mu * rho)) < 1e-12
    assert abs(tau(0, 1, lam, mu, rho) - (1 + lam * rho)) < 1e-12
    assert abs(tau(1, 1, lam, mu, rho) - (1 - rho)) < 1e-12
    assert tau(2, 1, lam, mu, rho) == 1.0


def test_rho_zero_grid_matches_independent_poisson_1x2():
    lam, mu = 1.4, 1.1
    max_goals = 8
    grid = scoreline_grid_from_rates(lam, mu, rho=0.0, max_goals=max_goals)
    m = markets_from_grid(grid)

    # Independent Poisson reference
    goals = np.arange(0, max_goals + 1)
    ph = np.exp(-lam) * np.power(lam, goals) / np.array([math.factorial(int(k)) for k in goals])
    pa = np.exp(-mu) * np.power(mu, goals) / np.array([math.factorial(int(k)) for k in goals])
    ref = np.outer(ph, pa)
    ref = ref / ref.sum()
    home = float(np.tril(ref, k=-1).sum())
    draw = float(np.trace(ref))
    away = float(np.triu(ref, k=1).sum())
    assert abs(m["home"] - home) < 1e-9
    assert abs(m["draw"] - draw) < 1e-9
    assert abs(m["away"] - away) < 1e-9
```

- [ ] **Step 3: Run tests — expect FAIL**

Run: `pytest tests/test_dixon_coles.py -v`

- [ ] **Step 4: Implement τ + `scoreline_grid_from_rates` in `src/models/dixon_coles.py`**

```python
"""Dixon–Coles goals model with time decay."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.models.markets import markets_from_grid


@dataclass
class DixonColesModel:
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float
    xi: float
    as_of: pd.Timestamp | None = None


def tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam * mu * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + mu * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def scoreline_grid_from_rates(
    lam: float,
    mu: float,
    rho: float,
    max_goals: int = 8,
) -> np.ndarray:
    goals = np.arange(0, max_goals + 1)
    home_p = np.exp(-lam) * np.power(lam, goals) / np.array(
        [math.factorial(int(k)) for k in goals]
    )
    away_p = np.exp(-mu) * np.power(mu, goals) / np.array(
        [math.factorial(int(k)) for k in goals]
    )
    grid = np.outer(home_p, away_p)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            grid[i, j] *= tau(i, j, lam, mu, rho)
    total = grid.sum()
    if total <= 0:
        raise ValueError("non-positive scoreline grid; check rho/lambda/mu")
    return grid / total
```

(Leave `fit_dixon_coles` / `predict_match` for Task 3.)

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/test_dixon_coles.py -v`

- [ ] **Step 6: Commit** (only if asked)

---

### Task 3: Dixon–Coles MLE fit (decay, shrinkage, ρ bounds)

**Files:**
- Modify: `src/models/dixon_coles.py`
- Modify: `tests/test_dixon_coles.py`

**Interfaces:**
- Produces:
  - `fit_dixon_coles(matches: pd.DataFrame, as_of=None, xi: float | None = None, l2: float | None = None, max_goals: int | None = None) -> DixonColesModel`
  - `expected_goals(model, home_team, away_team) -> tuple[float, float]`
  - `scoreline_grid(model, home_team, away_team, max_goals=None) -> np.ndarray`
  - `predict_match(model, home_team, away_team, max_goals=None) -> dict[str, float]` with at least `home`/`draw`/`away` (and preferably full markets keys)
- Consumes: matches with `home_team`, `away_team`, `home_goals`, `away_goals`, `date`; config defaults

**Fit requirements (implement all):**
1. Exponential weights `exp(-xi * days_ago)` vs `as_of` (default max date in `matches`)
2. Parameters: per-team attack/defence, home advantage, ρ
3. Constraint: mean(attack) = 1 after each unpack (or via reparameterisation)
4. L2 penalty on (attack−1) and (defence−1), scaled roughly by `1/(match_count+1)` contribution — simplest acceptable form: `l2 * sum((a-1)^2 + (d-1)^2)`
5. ρ box bounds: start with conservative `(-0.2, 0.2)` clipped further by admissible interval from current λ, μ if easy; always reject τ ≤ 0 inside NLL with a large penalty
6. Unseen teams at predict time → attack/defence = 1.0
7. On optimizer failure: set `rho=0` and fall back to strengths from a simple Poisson init (or re-raise only in tests that expect success) — prefer logging + ρ=0 with attack/defence from init

- [ ] **Step 1: Add failing fit/predict tests**

Append to `tests/test_dixon_coles.py`:

```python
import pandas as pd

from src.models.dixon_coles import fit_dixon_coles, predict_match, scoreline_grid


def _toy_matches() -> pd.DataFrame:
    rows = []
    teams = ["Alpha", "Beta", "Gamma"]
    start = pd.Timestamp("2024-01-01")
    for i in range(30):
        h = teams[i % 3]
        a = teams[(i + 1) % 3]
        rows.append(
            {
                "date": start + pd.Timedelta(days=i * 3),
                "home_team": h,
                "away_team": a,
                "home_goals": 1 + (i % 2),
                "away_goals": i % 2,
            }
        )
    return pd.DataFrame(rows)


def test_fit_predict_probs_sum_to_one():
    model = fit_dixon_coles(_toy_matches())
    probs = predict_match(model, "Alpha", "Beta")
    assert abs(probs["home"] + probs["draw"] + probs["away"] - 1.0) < 1e-8


def test_scoreline_grid_renormalized():
    model = fit_dixon_coles(_toy_matches())
    grid = scoreline_grid(model, "Alpha", "Beta")
    assert abs(grid.sum() - 1.0) < 1e-8


def test_unseen_team_does_not_crash():
    model = fit_dixon_coles(_toy_matches())
    probs = predict_match(model, "NewTeam", "Alpha")
    assert abs(sum(probs[k] for k in ("home", "draw", "away")) - 1.0) < 1e-8
```

- [ ] **Step 2: Run — expect FAIL** on missing `fit_dixon_coles`

- [ ] **Step 3: Implement fit + predict**

Implementation sketch (engineer fills complete function bodies in `dixon_coles.py`):

```python
from scipy.optimize import minimize

from src.config import DIXON_COLES_L2, DIXON_COLES_MAX_GOALS, DIXON_COLES_XI


def expected_goals(model: DixonColesModel, home_team: str, away_team: str) -> tuple[float, float]:
    a_h = model.attack.get(home_team, 1.0)
    a_a = model.attack.get(away_team, 1.0)
    d_h = model.defence.get(home_team, 1.0)
    d_a = model.defence.get(away_team, 1.0)
    lam = a_h * d_a * model.home_advantage
    mu = a_a * d_h
    return float(lam), float(mu)


def scoreline_grid(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    max_goals: int | None = None,
) -> np.ndarray:
    max_goals = DIXON_COLES_MAX_GOALS if max_goals is None else max_goals
    lam, mu = expected_goals(model, home_team, away_team)
    return scoreline_grid_from_rates(lam, mu, model.rho, max_goals=max_goals)


def predict_match(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    max_goals: int | None = None,
) -> dict[str, float]:
    return markets_from_grid(scoreline_grid(model, home_team, away_team, max_goals=max_goals))


def fit_dixon_coles(
    matches: pd.DataFrame,
    as_of=None,
    xi: float | None = None,
    l2: float | None = None,
    max_goals: int | None = None,
) -> DixonColesModel:
    # 1. Clean rows; parse dates
    # 2. Build team index; init attack/defence/home≈1.3/rho≈-0.03 from averages
    # 3. weights = exp(-xi * days_ago)
    # 4. minimize weighted NLL + L2 with scipy.optimize.minimize
    # 5. unpack with mean(attack)=1
    # 6. return DixonColesModel(...)
    ...
```

Use `scipy.special.gammaln` for log factorial in the NLL for speed. Vectorise τ with boolean masks on observed scores.

- [ ] **Step 4: Run `pytest tests/test_dixon_coles.py -v` — expect PASS**

- [ ] **Step 5: Optional shrinkage sanity test**

```python
def test_mean_attack_near_one():
    model = fit_dixon_coles(_toy_matches())
    mean_att = sum(model.attack.values()) / len(model.attack)
    assert abs(mean_att - 1.0) < 1e-5
```

- [ ] **Step 6: Commit** (only if asked)

---

### Task 4: Elo comparator

**Files:**
- Create: `src/models/elo.py`
- Create: `tests/test_elo.py`

**Interfaces:**
- `fit_elo(matches: pd.DataFrame, k=None, home_bonus=None, initial=None) -> EloModel`
- `predict_match(model: EloModel, home_team: str, away_team: str) -> dict[str, float]` with `home`/`draw`/`away`
- `EloModel`: `ratings: dict[str, float]`, `home_bonus: float`, `k: float`

**1X2 mapping (lock this):**
1. Chronological Elo updates on train matches (H=1, D=0.5, A=0 home score)
2. Expected home score: `1 / (1 + 10 ** (-(r_home + home_bonus - r_away) / 400))`
3. Convert two-way expectation `e` to 1X2 by inserting a draw mass that shrinks with rating gap:

```python
gap = abs((r_home + home_bonus) - r_away)
p_draw = 0.28 * math.exp(-gap / 400.0)  # in (0, 0.28]
p_home = e * (1 - p_draw)
p_away = (1 - e) * (1 - p_draw)
```

Unknown teams start at `ELO_INITIAL`.

- [ ] **Step 1: Failing tests**

```python
import pandas as pd

from src.models.elo import fit_elo, predict_match


def test_elo_probs_sum_to_one():
    matches = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "home_team": ["A", "B"] * 5,
            "away_team": ["B", "A"] * 5,
            "result": ["H", "A", "H", "D", "H", "A", "H", "H", "D", "H"],
        }
    )
    model = fit_elo(matches)
    p = predict_match(model, "A", "B")
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9


def test_elo_winner_gains_rating():
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "result": ["H", "H"],
        }
    )
    model = fit_elo(matches)
    assert model.ratings["A"] > model.ratings["B"]
```

- [ ] **Step 2: Implement `elo.py` and pass tests**

- [ ] **Step 3: Commit** (only if asked)

---

### Task 5: Temperature + vector scaling

**Files:**
- Modify: `src/eval/calibration.py`
- Modify: `tests/test_calibration.py`

**Interfaces:**
- Keep `reliability_table`, `plot_reliability`
- Add:
  - `apply_temperature(P: np.ndarray, temperature: float) -> np.ndarray`  # shape (n, 3)
  - `fit_temperature(y_true: list[str] | np.ndarray, P: np.ndarray) -> float`
  - `apply_vector_scaling(P: np.ndarray, weights: np.ndarray) -> np.ndarray`  # weights shape (3,)
  - `fit_vector_scaling(y_true, P) -> np.ndarray`

Outcome order: columns `[home, draw, away]` aligned with `H,D,A`.

Temperature: `logits = log(clip(P)); softmax(logits / T)`. Fit `T > 0` by minimising log loss via `scipy.optimize.minimize_scalar` or 1D bounded minimize.

Vector scaling (lab): multiply columns by positive weights, renormalise rows; fit weights by log loss (fix one weight to 1 for identifiability).

- [ ] **Step 1: Extend `tests/test_calibration.py`**

```python
import numpy as np

from src.eval.calibration import apply_temperature, fit_temperature


def test_temperature_one_leaves_probs_almost_unchanged():
    P = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]], dtype=float)
    out = apply_temperature(P, 1.0)
    assert np.allclose(out, P, atol=1e-10)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_fit_temperature_returns_positive():
    y = ["H", "D", "A", "H", "A", "D", "H", "A"]
    P = np.array(
        [
            [0.6, 0.2, 0.2],
            [0.2, 0.5, 0.3],
            [0.2, 0.2, 0.6],
            [0.55, 0.25, 0.2],
            [0.25, 0.25, 0.5],
            [0.3, 0.4, 0.3],
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
        ],
        dtype=float,
    )
    t = fit_temperature(y, P)
    assert t > 0
```

- [ ] **Step 2: Implement and pass tests** (include a small vector-scaling smoke test)

- [ ] **Step 3: Commit** (only if asked)

---

### Task 6: O/U columns in clean + features

**Files:**
- Modify: `src/clean.py`
- Modify: `src/features.py`
- Modify: `tests/test_clean.py`, `tests/test_features.py`

**Column mapping (prefer first present):**
- Over 2.5: `B365>2.5` → `over_25_odds_close` (also accept `Avg>2.5` as fallback if Bet365 missing)
- Under 2.5: `B365<2.5` → `under_25_odds_close` (fallback `Avg<2.5`)

Note: `>` / `<` in CSV headers are valid pandas column names when read from football-data.co.uk.

**Features:**
- `implied_over_25`, `implied_under_25` = de-vig `(1/odds) / sum` when both present

- [ ] **Step 1: Failing clean/features tests with a tiny synthetic raw frame** including `B365>2.5` / `B365<2.5`

- [ ] **Step 2: Extend `clean_frame` odds_map and column list; extend `add_implied_odds`**

- [ ] **Step 3: Rebuild processed data locally**

Run: `python -m src.pipeline`  
Confirm new columns appear when present in raw CSVs (some older seasons may lack them — that is fine).

- [ ] **Step 4: Commit** (only if asked)

---

### Task 7: Wire models into walk-forward + season gate

**Files:**
- Modify: `src/eval/walkforward.py`
- Modify: `src/evaluate.py`
- Modify: `tests/test_walkforward.py`
- Update artefacts text in `write_walkforward_artefacts`

**Behaviour:**
1. Each fold fit: poisson (existing), `fit_dixon_coles(train)`, `fit_elo(train)`, baseline/book (existing)
2. Predict Dixon–Coles raw 1X2 via `predict_match`
3. Maintain running store of prior DC predictions + results; when `len(prior) >= CALIBRATION_MIN_ROWS`, `fit_temperature` on prior only and apply to current fold → model name `dixon_coles_calibrated`; else skip calibrated line or copy raw (document: skip until warm)
4. Elo line name: `elo`
5. Optionally record `p_over_25` / `p_under_25` on DC pred frames for later; add O/U log-loss vs binary over if easy — minimum for this task: 1X2 metrics for new lines; O/U book row when `implied_over_25` present (binary log loss helper is OK)
6. Season gate (`run_backtest`): add dixon_coles + elo (+ calibrated if enough holdout — for season gate, fit temperature on a trailing slice of train predictions or skip calibrated and only report raw DC; **prefer:** fit DC on train, predict test raw; fit temperature on a chronological last 20% of train via internal CV-style predict — if too heavy, season gate reports raw DC + elo only and walk-forward owns calibrated)

**Pinned season-gate choice:** report `dixon_coles` and `elo` on the season gate; calibrated line is walk-forward-only for v1 (simpler leakage story).

- [ ] **Step 1: Extend synthetic walk-forward test** so `run_walkforward` on a larger synthetic frame includes `dixon_coles` in summary models (may need `min_train_rows=0` path or call internal pieces). Prefer testing that imports succeed and `fit_dixon_coles` + elo are invoked on a tiny hand loop rather than full CSV.

Minimal test addition:

```python
def test_new_models_import_and_fit_on_synthetic():
    from src.models.dixon_coles import fit_dixon_coles, predict_match as dc_predict
    from src.models.elo import fit_elo, predict_match as elo_predict

    matches = _synthetic_matches(40)
    matches["home_goals"] = 1
    matches["away_goals"] = 0
    dc = fit_dixon_coles(matches)
    elo = fit_elo(matches.assign(result=["H"] * 40))
    assert "home" in dc_predict(dc, "Arsenal", "Chelsea")
    assert "home" in elo_predict(elo, "Arsenal", "Chelsea")
```

- [ ] **Step 2: Implement walk-forward wiring + season-gate rows**

- [ ] **Step 3: Run**

```bash
pytest tests/test_walkforward.py tests/test_evaluate.py tests/test_dixon_coles.py tests/test_elo.py tests/test_markets.py tests/test_calibration.py -v
python -m src.evaluate --mode season
python -m src.evaluate --mode walkforward
```

Expect: new rows in CSVs; calibration PNGs for new models; runtime of walk-forward may be several minutes because of MLE per fold — acceptable offline.

- [ ] **Step 4: Commit** (only if asked)

---

### Task 8: Offline predict helper + protocol docs

**Files:**
- Create: `src/models/predict.py`
- Modify: `docs/EVAL_PROTOCOL.md`
- Modify: `results/WALKFORWARD.md` (via regenerate)
- Optionally: short note in `README.md` under evaluation / models

**Interfaces:**
- `predict_matches(model: DixonColesModel, matches: pd.DataFrame, temperature: float | None = None) -> pd.DataFrame`  
  Columns: `match_id` (if present), teams, `p_home`, `p_draw`, `p_away`, `p_btts_yes`, `p_over_25`, …  
  If `temperature` is set, apply to 1X2 only; leave BTTS/O/U from the raw grid (document that calibration is 1X2-only in v1).

- [ ] **Step 1: Implement helper + a small unit test**

- [ ] **Step 2: Update `EVAL_PROTOCOL.md` Comparators table** to include Dixon–Coles (raw/calibrated), Elo, derived markets, O/U book rule, ξ note

- [ ] **Step 3: Full pytest**

```bash
pytest -v
```

Expected: all green

- [ ] **Step 4: Commit** (only if asked)

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| Dixon–Coles MLE + time decay | 2–3 |
| τ / ρ bounds / mean attack / shrinkage / unseen avg | 3 |
| Grid → 1X2 / BTTS / O/U | 1, 3 |
| Temperature default + vector lab | 5 |
| Elo comparator | 4 |
| Keep independent Poisson | 7 (unchanged line) |
| O/U book columns when present | 6–7 |
| Walk-forward + season gate | 7 |
| Offline historical outputs | 8 |
| EVAL_PROTOCOL update | 8 |
| Fixtures / store / UI | Out of scope |

## Plan self-review notes

- No joint ξ MLE; config constant only
- Calibrated line is walk-forward-only in v1 to keep season-gate leakage simple
- BTTS/O/U stay on the raw grid even when 1X2 is temperature-scaled (called out in Task 8)
- scipy already listed in `requirements.txt`
