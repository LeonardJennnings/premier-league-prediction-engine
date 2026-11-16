"""Reliability / calibration tables, plots, and 1X2 probability scaling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

# Columns of P are [home, draw, away], aligned with labels H, D, A.
_OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
_EPS = 1e-15
_T_LO = 0.05
_T_HI = 10.0
_WEIGHT_LO = 1e-4
_WEIGHT_HI = 1e3


def _as_proba_matrix(P) -> np.ndarray:
    arr = np.asarray(P, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("P must have shape (n, 3) for home, draw, away")
    return arr


def _log_loss(y_true, P, eps: float = _EPS) -> float:
    clipped = np.clip(_as_proba_matrix(P), eps, 1.0)
    idx = np.array([_OUTCOME_INDEX[y] for y in y_true], dtype=int)
    return float(-np.log(clipped[np.arange(len(idx)), idx]).mean())


def apply_temperature(P: np.ndarray, temperature: float) -> np.ndarray:
    """Scale 1X2 probabilities: softmax(log(clip(P)) / T). T=1 is a no-op."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = np.log(np.clip(_as_proba_matrix(P), _EPS, 1.0))
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(y_true: list[str] | np.ndarray, P: np.ndarray) -> float:
    """Return T > 0 that minimises log loss of temperature-scaled P."""
    arr = _as_proba_matrix(P)
    y = list(y_true)

    def nll(t: float) -> float:
        return _log_loss(y, apply_temperature(arr, t))

    result = minimize_scalar(nll, bounds=(_T_LO, _T_HI), method="bounded")
    return float(result.x)


def apply_vector_scaling(P: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multiply 1X2 columns by positive weights and renormalise each row."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (3,):
        raise ValueError("weights must have shape (3,)")
    if np.any(w <= 0):
        raise ValueError("weights must be positive")
    scaled = _as_proba_matrix(P) * w
    totals = scaled.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("scaled rows must have positive mass")
    return scaled / totals


def fit_vector_scaling(y_true: list[str] | np.ndarray, P: np.ndarray) -> np.ndarray:
    """Fit positive 1X2 weights by log loss; home weight is fixed at 1."""
    arr = _as_proba_matrix(P)
    y = list(y_true)

    def nll(w_da: np.ndarray) -> float:
        weights = np.array([1.0, w_da[0], w_da[1]], dtype=float)
        return _log_loss(y, apply_vector_scaling(arr, weights))

    result = minimize(
        nll,
        x0=np.array([1.0, 1.0]),
        bounds=[(_WEIGHT_LO, _WEIGHT_HI), (_WEIGHT_LO, _WEIGHT_HI)],
        method="L-BFGS-B",
    )
    return np.array([1.0, float(result.x[0]), float(result.x[1])], dtype=float)


def reliability_table(y_true, p_hat, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predicted probabilities and compare to observed frequencies."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_hat, dtype=float)
    if len(y) != len(p):
        raise ValueError("y_true and p_hat must have the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Put each probability into a bin index 0 .. n_bins-1
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
    