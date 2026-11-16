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
    