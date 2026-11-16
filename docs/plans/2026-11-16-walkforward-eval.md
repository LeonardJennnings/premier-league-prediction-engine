# Walk-forward evaluation harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.  
> **For this repo’s preferred path:** work the plan in tutor mode (explain → learner types → verify). Skip automatic commits unless Leonard asks to commit.

**Goal:** Add a gameweek walk-forward eval with calibration plots, while keeping the existing fixed-season regression gate.

**Architecture:** New `src/eval/` package owns gameweek assignment, fold iteration, and calibration. `src/evaluate.py` stays the CLI front door (`--mode season|walkforward`). Metrics (`log_loss`, `brier_score`, `accuracy`) stay in `evaluate.py` and are imported by the walk-forward runner.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn (existing models), matplotlib (new), pytest.

**Spec:** [`docs/specs/walkforward-eval.md`](../specs/walkforward-eval.md) · Protocol: [`docs/EVAL_PROTOCOL.md`](../EVAL_PROTOCOL.md)

## Global Constraints

- Premier League only; historical E0 data via existing clean/features pipeline
- No tipster language in docs or UI copy touched by this plan
- Leakage rule: train only on matches with `date` strictly before the test fold
- Gameweek = date-ordered chunks of up to 10 matches within each `season_code`
- Refit models every fold
- Minimum train rows to score a fold: `200`
- Bookmaker line drops rows missing `implied_home/draw/away`; model lines may use a larger `n` — record `n_test` per line
- Season gate (`2526`) behaviour and `results/eval_results.csv` must keep working
- Do not put process/tutor meta in portfolio docs
- Pin decisions already agreed: hybrid protocols, `src/eval/` layout, summary + per-GW CSVs + calibration PNGs

## File map

| Path | Responsibility |
|---|---|
| `src/eval/__init__.py` | Package marker |
| `src/eval/gameweeks.py` | `assign_gameweeks(matches, matches_per_gw=10) -> DataFrame` |
| `src/eval/calibration.py` | `reliability_table(...)`, `plot_reliability(...)` |
| `src/eval/walkforward.py` | `iter_folds(...)`, `run_walkforward(...)`, artefact writers |
| `src/evaluate.py` | Keep metrics + season gate; add argparse `--mode` |
| `tests/test_gameweeks.py` | GW assignment |
| `tests/test_calibration.py` | Reliability buckets |
| `tests/test_walkforward.py` | Fold boundaries / min-train skip on tiny synthetic data |
| `results/WALKFORWARD.md` | How to regenerate artefacts |
| `requirements.txt` | Ensure matplotlib pinned after install |
| `docs/EVAL_PROTOCOL.md` | Mark walk-forward as implemented when done |
| `README.md` | One short pointer to eval modes |

---

### Task 1: Gameweek assignment

**Files:**
- Create: `src/eval/__init__.py`
- Create: `src/eval/gameweeks.py`
- Create: `tests/test_gameweeks.py`

**Interfaces:**
- Consumes: `pandas.DataFrame` with at least `season_code`, `date`
- Produces: `assign_gameweeks(matches: pd.DataFrame, matches_per_gw: int = 10) -> pd.DataFrame` — same rows plus integer `gameweek` starting at 1 within each season

- [ ] **Step 1: Create the package marker**

Create empty `src/eval/__init__.py`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_gameweeks.py`:

```python
import pandas as pd

from src.eval.gameweeks import assign_gameweeks


def test_assign_gameweeks_chunks_of_ten_within_season():
    dates = pd.date_range("2024-08-01", periods=25, freq="D")
    matches = pd.DataFrame(
        {
            "season_code": ["2425"] * 25,
            "date": dates,
            "home_team": [f"H{i}" for i in range(25)],
            "away_team": [f"A{i}" for i in range(25)],
        }
    )
    out = assign_gameweeks(matches, matches_per_gw=10)
    assert list(out["gameweek"][:10]) == [1] * 10
    assert list(out["gameweek"][10:20]) == [2] * 10
    assert list(out["gameweek"][20:]) == [3] * 5


def test_assign_gameweeks_restarts_each_season():
    matches = pd.DataFrame(
        {
            "season_code": ["2324"] * 3 + ["2425"] * 3,
            "date": pd.to_datetime(
                [
                    "2023-08-01",
                    "2023-08-02",
                    "2023-08-03",
                    "2024-08-01",
                    "2024-08-02",
                    "2024-08-03",
                ]
            ),
        }
    )
    out = assign_gameweeks(matches, matches_per_gw=2)
    assert out.loc[out["season_code"] == "2324", "gameweek"].tolist() == [1, 1, 2]
    assert out.loc[out["season_code"] == "2425", "gameweek"].tolist() == [1, 1, 2]
```

- [ ] **Step 3: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_gameweeks.py -v
```

Expected: `ModuleNotFoundError` or import error for `assign_gameweeks`.

- [ ] **Step 4: Implement `assign_gameweeks`**

Create `src/eval/gameweeks.py`:

```python
"""Assign synthetic gameweek ids for walk-forward folds."""

from __future__ import annotations

import pandas as pd


def assign_gameweeks(matches: pd.DataFrame, matches_per_gw: int = 10) -> pd.DataFrame:
    """Return a copy with a 1-based gameweek column within each season_code.

    Matches are sorted by date within each season, then chunked into rounds of
    at most ``matches_per_gw`` rows. This is an evaluation convenience — the
    source CSVs do not ship an official gameweek field.
    """
    if matches_per_gw < 1:
        raise ValueError("matches_per_gw must be >= 1")

    out = matches.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["season_code", "date", "home_team", "away_team"]).reset_index(drop=True)

    weeks: list[int] = []
    for _, season_df in out.groupby("season_code", sort=False):
        n = len(season_df)
        # 0,0,..,0 (10 times), 1,1,... then +1 for 1-based gameweek
        gw = [i // matches_per_gw + 1 for i in range(n)]
        weeks.extend(gw)

    out["gameweek"] = weeks
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_gameweeks.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit (only if Leonard asks)**

```bash
git add src/eval/__init__.py src/eval/gameweeks.py tests/test_gameweeks.py
git commit -m "$(cat <<'EOF'
Add gameweek assignment helpers for walk-forward evaluation.

EOF
)"
```

---

### Task 2: Calibration reliability helpers

**Files:**
- Create: `src/eval/calibration.py`
- Create: `tests/test_calibration.py`
- Modify: install/pin `matplotlib` in the venv + `requirements.txt`

**Interfaces:**
- Consumes: 1d predicted probabilities for one outcome, binary indicators (0/1) for that outcome
- Produces:
  - `reliability_table(y_true: array-like, p_hat: array-like, n_bins: int = 10) -> pd.DataFrame` with columns `bin_left`, `bin_right`, `bin_center`, `count`, `mean_predicted`, `mean_observed`
  - `plot_reliability(table: pd.DataFrame, path: Path, title: str) -> Path`

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration.py`:

```python
import numpy as np

from src.eval.calibration import reliability_table


def test_reliability_table_hand_checked_two_bins():
    # Four predictions: two in (0, 0.5], two in (0.5, 1.0]
    p_hat = np.array([0.25, 0.25, 0.75, 0.75])
    y_true = np.array([0, 1, 1, 1])  # observed rates: 0.5 and 1.0
    table = reliability_table(y_true, p_hat, n_bins=2)
    assert len(table) == 2
    assert table.loc[0, "count"] == 2
    assert abs(table.loc[0, "mean_predicted"] - 0.25) < 1e-9
    assert abs(table.loc[0, "mean_observed"] - 0.5) < 1e-9
    assert abs(table.loc[1, "mean_predicted"] - 0.75) < 1e-9
    assert abs(table.loc[1, "mean_observed"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_calibration.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement calibration helpers**

Create `src/eval/calibration.py`:

```python
"""Reliability / calibration tables and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def reliability_table(y_true, p_hat, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predicted probabilities and compare to observed frequencies."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_hat, dtype=float)
    if len(y) != len(p):
        raise ValueError("y_true and p_hat must have the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-inclusive on the last edge so p=1.0 lands in the final bin
    bin_idx = np.digitize(p, edges[1:-1], right=False)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            mean_pred = np.nan
            mean_obs = np.nan
        else:
            mean_pred = float(p[mask].mean())
            mean_obs = float(y[mask].mean())
        rows.append(
            {
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]),
                "bin_center": float((edges[b] + edges[b + 1]) / 2),
                "count": count,
                "mean_predicted": mean_pred,
                "mean_observed": mean_obs,
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(table: pd.DataFrame, path: Path, title: str) -> Path:
    """Save a reliability plot; empty bins are skipped."""
    import matplotlib.pyplot as plt

    usable = table.dropna(subset=["mean_predicted", "mean_observed"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    if not usable.empty:
        ax.plot(
            usable["mean_predicted"],
            usable["mean_observed"],
            marker="o",
            label="model",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
```

- [ ] **Step 4: Install matplotlib and re-freeze requirements**

```bash
python -m pip install matplotlib
python -c "import subprocess; from pathlib import Path; Path('requirements.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'], text=True))"
```

- [ ] **Step 5: Run calibration tests**

```bash
pytest tests/test_calibration.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit (only if Leonard asks)**

```bash
git add src/eval/calibration.py tests/test_calibration.py requirements.txt
git commit -m "$(cat <<'EOF'
Add calibration reliability helpers and matplotlib dependency.

EOF
)"
```

---

### Task 3: Walk-forward runner (core)

**Files:**
- Create: `src/eval/walkforward.py`
- Create: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `assign_gameweeks`, metrics from `src.evaluate`, `fit_poisson` / `predict_match`, `fit_baseline` / `predict_match_features`
- Produces:
  - `MIN_TRAIN_ROWS = 200`
  - `iter_folds(matches: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str, int]]` yielding `(train, test, season_code, gameweek)`
  - `run_walkforward(matches: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]`  
    returns `(summary_df, by_gw_df, prediction_frames_by_model)` where prediction frames include columns needed for calibration (`y_home`, `p_home`, etc.)

- [ ] **Step 1: Write failing fold-boundary tests**

Create `tests/test_walkforward.py`:

```python
import pandas as pd

from src.eval.gameweeks import assign_gameweeks
from src.eval.walkforward import MIN_TRAIN_ROWS, iter_folds


def _synthetic_matches(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-08-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "season_code": ["2425"] * n,
            "date": dates,
            "home_team": ["Arsenal"] * n,
            "away_team": ["Chelsea"] * n,
            "home_goals": [1] * n,
            "away_goals": [0] * n,
            "result": ["H"] * n,
            "implied_home": [0.5] * n,
            "implied_draw": [0.25] * n,
            "implied_away": [0.25] * n,
        }
    )


def test_iter_folds_train_strictly_before_test():
    matches = assign_gameweeks(_synthetic_matches(25), matches_per_gw=5)
    folds = list(iter_folds(matches, min_train_rows=0))
    assert len(folds) == 5
    for train, test, season, gw in folds:
        assert season == "2425"
        assert (train["date"] < test["date"].min()).all()
        assert set(test["gameweek"]) == {gw}


def test_iter_folds_skips_until_min_train():
    matches = assign_gameweeks(_synthetic_matches(25), matches_per_gw=5)
    folds = list(iter_folds(matches, min_train_rows=12))
    # First folds with train size 0,5,10 skipped; 15+ kept
    assert all(len(train) >= 12 for train, *_ in folds)
    assert len(folds) == 2
```

- [ ] **Step 2: Run tests — expect fail**

```bash
pytest tests/test_walkforward.py -v
```

- [ ] **Step 3: Implement `iter_folds` + `run_walkforward`**

Create `src/eval/walkforward.py` with at least:

```python
"""Gameweek walk-forward evaluation runner."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS
from src.eval.calibration import plot_reliability, reliability_table
from src.eval.gameweeks import assign_gameweeks
from src.evaluate import accuracy, brier_score, log_loss
from src.models.baseline import fit_baseline, predict_match_features
from src.models.poisson import fit_poisson, predict_match

logger = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 200
OUTCOME_ORDER = ("H", "D", "A")


def iter_folds(
    matches: pd.DataFrame,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str, int]]:
    """Yield (train, test, season_code, gameweek) with strict date cutoff."""
    if "gameweek" not in matches.columns:
        raise ValueError("matches must include gameweek; call assign_gameweeks first")

    for (season, gw), test in matches.groupby(["season_code", "gameweek"], sort=True):
        cutoff = test["date"].min()
        train = matches[matches["date"] < cutoff]
        if len(train) < min_train_rows:
            logger.warning(
                "Skipping %s GW%s: train_rows=%s < min_train_rows=%s",
                season,
                gw,
                len(train),
                min_train_rows,
            )
            continue
        yield train.copy(), test.copy(), str(season), int(gw)


def _probas_to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.array([[r["home"], r["draw"], r["away"]] for r in rows], dtype=float)


def _naive_probas(train: pd.DataFrame, n: int) -> np.ndarray:
    counts = train["result"].value_counts(normalize=True)
    p = np.array(
        [
            float(counts.get("H", 0.0)),
            float(counts.get("D", 0.0)),
            float(counts.get("A", 0.0)),
        ],
        dtype=float,
    )
    total = p.sum()
    if total <= 0:
        p = np.array([1 / 3, 1 / 3, 1 / 3])
    else:
        p = p / total
    return np.tile(p, (n, 1))


def _metric_row(run_at: str, model: str, season: str, gw: int, n_train: int, y_true, P) -> dict:
    return {
        "run_date": run_at,
        "model": model,
        "season_code": season,
        "gameweek": gw,
        "n_train": n_train,
        "n_test": len(y_true),
        "log_loss": round(log_loss(y_true, P), 4),
        "brier": round(brier_score(y_true, P), 4),
        "accuracy": round(accuracy(y_true, P), 4),
    }


def run_walkforward(matches: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run full walk-forward; return summary, by-gameweek, and per-model prediction frames."""
    from src.clean import build_matches_table
    from src.features import add_features

    if matches is None:
        matches = add_features(build_matches_table())
    matches = assign_gameweeks(matches)
    matches["date"] = pd.to_datetime(matches["date"])

    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    by_gw_rows: list[dict] = []
    pred_records: dict[str, list[dict]] = {
        "naive_frequencies": [],
        "poisson": [],
        "sklearn_baseline": [],
        "bookmaker_implied": [],
    }

    for train, test, season, gw in iter_folds(matches):
        y_all = test["result"].tolist()

        naive_P = _naive_probas(train, len(test))
        by_gw_rows.append(_metric_row(run_at, "naive_frequencies", season, gw, len(train), y_all, naive_P))
        for i, (_, row) in enumerate(test.iterrows()):
            pred_records["naive_frequencies"].append(
                {
                    "result": row["result"],
                    "p_home": naive_P[i, 0],
                    "p_draw": naive_P[i, 1],
                    "p_away": naive_P[i, 2],
                }
            )

        poisson_model = fit_poisson(train)
        poisson_rows = [
            predict_match(poisson_model, row.home_team, row.away_team)
            for row in test.itertuples(index=False)
        ]
        poisson_P = _probas_to_matrix(poisson_rows)
        by_gw_rows.append(_metric_row(run_at, "poisson", season, gw, len(train), y_all, poisson_P))
        for i, (_, row) in enumerate(test.iterrows()):
            pred_records["poisson"].append(
                {
                    "result": row["result"],
                    "p_home": poisson_P[i, 0],
                    "p_draw": poisson_P[i, 1],
                    "p_away": poisson_P[i, 2],
                }
            )

        base_train = train.dropna(subset=["implied_home", "implied_draw", "implied_away", "result"])
        base_test = test.dropna(subset=["implied_home", "implied_draw", "implied_away", "result"])
        if len(base_train) >= MIN_TRAIN_ROWS and not base_test.empty:
            baseline_model = fit_baseline(base_train)
            baseline_rows = [
                predict_match_features(baseline_model, row) for _, row in base_test.iterrows()
            ]
            baseline_P = _probas_to_matrix(baseline_rows)
            y_base = base_test["result"].tolist()
            by_gw_rows.append(
                _metric_row(run_at, "sklearn_baseline", season, gw, len(base_train), y_base, baseline_P)
            )
            for i, (_, row) in enumerate(base_test.iterrows()):
                pred_records["sklearn_baseline"].append(
                    {
                        "result": row["result"],
                        "p_home": baseline_P[i, 0],
                        "p_draw": baseline_P[i, 1],
                        "p_away": baseline_P[i, 2],
                    }
                )

            bookie_P = base_test[["implied_home", "implied_draw", "implied_away"]].to_numpy(dtype=float)
            by_gw_rows.append(
                _metric_row(run_at, "bookmaker_implied", season, gw, len(base_train), y_base, bookie_P)
            )
            for i, (_, row) in enumerate(base_test.iterrows()):
                pred_records["bookmaker_implied"].append(
                    {
                        "result": row["result"],
                        "p_home": bookie_P[i, 0],
                        "p_draw": bookie_P[i, 1],
                        "p_away": bookie_P[i, 2],
                    }
                )

    by_gw = pd.DataFrame(by_gw_rows)
    summary = (
        by_gw.groupby("model", as_index=False)
        .agg(
            run_date=("run_date", "first"),
            n_folds=("gameweek", "count"),
            n_test=("n_test", "sum"),
            log_loss=("log_loss", "mean"),
            brier=("brier", "mean"),
            accuracy=("accuracy", "mean"),
        )
    )
    # Re-round aggregates for readability
    for col in ("log_loss", "brier", "accuracy"):
        summary[col] = summary[col].round(4)

    pred_frames = {k: pd.DataFrame(v) for k, v in pred_records.items()}
    return summary, by_gw, pred_frames


def write_walkforward_artefacts(
    summary: pd.DataFrame,
    by_gw: pd.DataFrame,
    pred_frames: dict[str, pd.DataFrame],
    results_dir: Path | None = None,
) -> dict[str, Path]:
    """Write CSVs, home-win reliability plots, and WALKFORWARD.md."""
    results_dir = Path(results_dir or RESULTS)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    summary_path = results_dir / "walkforward_summary.csv"
    by_gw_path = results_dir / "walkforward_by_gameweek.csv"
    summary.to_csv(summary_path, index=False)
    by_gw.to_csv(by_gw_path, index=False)
    paths["summary"] = summary_path
    paths["by_gameweek"] = by_gw_path

    for model_name, frame in pred_frames.items():
        if frame.empty:
            continue
        y_home = (frame["result"] == "H").astype(float).to_numpy()
        p_home = frame["p_home"].to_numpy(dtype=float)
        table = reliability_table(y_home, p_home, n_bins=10)
        plot_path = results_dir / f"calibration_{model_name}_home.png"
        plot_reliability(table, plot_path, title=f"{model_name} — home win reliability")
        paths[f"calibration_{model_name}"] = plot_path

    readme = results_dir / "WALKFORWARD.md"
    readme.write_text(
        """# Walk-forward artefacts

## Protocol

- Folds: synthetic gameweeks (≤10 matches, date-ordered within each season)
- Train: all matches with date strictly before the fold
- Minimum train rows: 200
- Models: naive_frequencies, poisson, sklearn_baseline, bookmaker_implied
- Calibration plots: home-win reliability (predicted P(home) vs observed home frequency)

## Regenerate

```bash
source .venv/bin/activate
python -m src.evaluate --mode walkforward
```

Season-gate regression (unchanged):

```bash
python -m src.evaluate --mode season
```

See `docs/EVAL_PROTOCOL.md` for interpretation.
""",
        encoding="utf-8",
    )
    paths["readme"] = readme
    return paths
```

Note while tutoring: if `groupby(..., sort=True)` on `(season_code, gameweek)` feels unclear, draw one season on paper first (GW1…GW3) before coding the loop.

- [ ] **Step 4: Run unit tests**

```bash
pytest tests/test_walkforward.py tests/test_gameweeks.py tests/test_calibration.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit (only if Leonard asks)**

---

### Task 4: CLI wiring in `evaluate.py`

**Files:**
- Modify: `src/evaluate.py`

**Interfaces:**
- Consumes: `run_walkforward`, `write_walkforward_artefacts`
- Produces: `python -m src.evaluate --mode season|walkforward` (default `season`)

- [ ] **Step 1: Replace the `__main__` block**

Keep all existing metric + `run_backtest` / `save_results` code. Change only the bottom of `src/evaluate.py` to:

```python
if __name__ == "__main__":
    import argparse

    from src.config import LOG_LEVEL
    from src.eval.walkforward import run_walkforward, write_walkforward_artefacts
    from src.logging_setup import setup_logging

    setup_logging(LOG_LEVEL)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Premier League evaluation")
    parser.add_argument(
        "--mode",
        choices=("season", "walkforward"),
        default="season",
        help="season = fixed 2526 gate; walkforward = gameweek walk-forward",
    )
    args = parser.parse_args()

    if args.mode == "season":
        table = run_backtest()
        out = save_results(table)
        logger.info("\n%s", table.to_string(index=False))
        logger.info("Saved %s", out)
    else:
        summary, by_gw, pred_frames = run_walkforward()
        paths = write_walkforward_artefacts(summary, by_gw, pred_frames)
        logger.info("\n%s", summary.to_string(index=False))
        for label, path in paths.items():
            logger.info("Wrote %s -> %s", label, path)
```

- [ ] **Step 2: Smoke-run season mode (regression)**

```bash
python -m src.evaluate --mode season
```

Expected: familiar four-row table; `results/eval_results.csv` updated.

- [ ] **Step 3: Run walk-forward (may take several minutes)**

```bash
python -m src.evaluate --mode walkforward
```

Expected: summary table logged; files appear under `results/` (`walkforward_summary.csv`, `walkforward_by_gameweek.csv`, `calibration_*_home.png`, `WALKFORWARD.md`).

- [ ] **Step 4: Full pytest**

```bash
pytest -v
```

Expected: all green (including older clean/features/poisson/evaluate tests).

- [ ] **Step 5: Commit (only if Leonard asks)**

---

### Task 5: Docs pointers

**Files:**
- Modify: `docs/EVAL_PROTOCOL.md` (status — walk-forward command is live)
- Modify: `README.md` (short Evaluation subsection if missing)
- Verify: `results/WALKFORWARD.md` from Task 3/4

- [ ] **Step 1: Update EVAL_PROTOCOL command section**

Ensure the walk-forward regenerate command is documented as:

```bash
python -m src.evaluate --mode walkforward
```

Remove any “not yet implemented” wording if present.

- [ ] **Step 2: Add a brief README note**

Under Results / Evaluation, add roughly:

```markdown
### Evaluation modes

- Season gate: `python -m src.evaluate --mode season` → `results/eval_results.csv`
- Walk-forward: `python -m src.evaluate --mode walkforward` → `results/walkforward_*.csv` + calibration plots

Details: [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md).
```

- [ ] **Step 3: Sanity-check artefact files exist and open one calibration PNG**

- [ ] **Step 4: Commit docs + results artefacts (only if Leonard asks)**

---

## Self-review vs spec

| Spec requirement | Task |
|---|---|
| Gameweek walk-forward primary protocol | Task 3–4 |
| Keep `2526` season gate | Task 4 default `--mode season` |
| ≤10 match rounds | Task 1 |
| Refit every fold | Task 3 `run_walkforward` loop |
| `src/eval/` layout | Tasks 1–3 |
| Summary + per-GW CSV + calibration PNGs + notes | Task 3 `write_walkforward_artefacts` |
| Metrics reuse | Task 3 imports from `evaluate.py` |
| Tests for GW + calibration | Tasks 1–2 |
| Leakage / min train | Task 3 `iter_folds` + `MIN_TRAIN_ROWS = 200` |
| Home-win reliability plots in v1 | Task 3 writer (follow-up: all three outcomes later) |

No placeholder steps left. CLI choice fixed to `--mode`. Min train fixed to 200. Calibration v1 = home-win panels per model.

---

## Execution handoff

Plan saved to `docs/plans/2026-11-16-walkforward-eval.md`.

Because you chose **tutor mode**, the default next step is: we walk **Task 1** together (you type, I explain).

If you prefer otherwise:

1. **Tutor walkthrough (recommended for you)** — start Task 1 now in this chat  
2. **Inline execution** — I implement the plan in this session with checkpoints  
3. **Subagent-driven** — fresh agent per task (faster, less teaching)

Which do you want?
