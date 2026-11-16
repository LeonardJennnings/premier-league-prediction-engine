"""OOS Benter-style log-odds blend of model + de-vigged book probabilities.

Second-stage fusion only. Fit on prior-fold out-of-sample predictions —
never on in-sample Dixon–Coles fitted probabilities. The book line is a
competitor first; the blend is labelled fusion, not tipster edge.

Benter (1994) form for discrete outcomes:

    c_i ∝ exp(α log f_i + β log π_i)

where ``f`` is the fundamental (model) probability and ``π`` is the
de-vigged public / book probability. Report ``(α, β)`` and
``w_model = α / (α + β)`` (``w_model ≈ 0`` means the market dominates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

# Columns of P are [home, draw, away], aligned with labels H, D, A.
_OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
_EPS = 1e-15
_COEF_LO = 0.0
_COEF_HI = 10.0


@dataclass(frozen=True)
class BenterWeights:
    """Fitted Benter coefficients for a multinomial (1X2) blend."""

    alpha: float
    beta: float

    @property
    def w_model(self) -> float:
        """Share of log-model weight; 0 ⇒ all book, 1 ⇒ all model."""
        total = float(self.alpha + self.beta)
        if total <= 0:
            return 0.0
        return float(self.alpha / total)


@dataclass(frozen=True)
class BinaryBlendWeights:
    """Fitted log-odds coefficients for a binary (O/U) blend."""

    alpha: float
    beta: float

    @property
    def w_model(self) -> float:
        total = float(self.alpha + self.beta)
        if total <= 0:
            return 0.0
        return float(self.alpha / total)


def _as_proba_matrix(P) -> np.ndarray:
    arr = np.asarray(P, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("P must have shape (n, 3) for home, draw, away")
    return arr


def _log_loss_1x2(y_true, P, eps: float = _EPS) -> float:
    clipped = np.clip(_as_proba_matrix(P), eps, 1.0)
    idx = np.array([_OUTCOME_INDEX[y] for y in y_true], dtype=int)
    return float(-np.log(clipped[np.arange(len(idx)), idx]).mean())


def apply_benter_blend(
    P_model: np.ndarray,
    P_book: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Blend 1X2 rows: softmax(α log f + β log π)."""
    log_f = np.log(np.clip(_as_proba_matrix(P_model), _EPS, 1.0))
    log_pi = np.log(np.clip(_as_proba_matrix(P_book), _EPS, 1.0))
    logits = float(alpha) * log_f + float(beta) * log_pi
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_benter_blend(
    y_true: list[str] | np.ndarray,
    P_model: np.ndarray,
    P_book: np.ndarray,
) -> BenterWeights:
    """Fit non-negative (α, β) by minimising 1X2 log loss on OOS pairs."""
    f = _as_proba_matrix(P_model)
    pi = _as_proba_matrix(P_book)
    if f.shape != pi.shape:
        raise ValueError("P_model and P_book must have the same shape")
    y = list(y_true)
    if len(y) != f.shape[0]:
        raise ValueError("y_true length must match probability rows")
    if len(y) == 0:
        raise ValueError("need at least one row to fit Benter blend")

    def nll(coef: np.ndarray) -> float:
        return _log_loss_1x2(y, apply_benter_blend(f, pi, coef[0], coef[1]))

    result = minimize(
        nll,
        x0=np.array([1.0, 1.0], dtype=float),
        bounds=[(_COEF_LO, _COEF_HI), (_COEF_LO, _COEF_HI)],
        method="L-BFGS-B",
    )
    alpha = float(result.x[0])
    beta = float(result.x[1])
    return BenterWeights(alpha=alpha, beta=beta)


def apply_binary_logodds_blend(
    p_model: np.ndarray,
    p_book: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Blend binary probs on the logit scale, then sigmoid."""
    f = np.clip(np.asarray(p_model, dtype=float), _EPS, 1.0 - _EPS)
    pi = np.clip(np.asarray(p_book, dtype=float), _EPS, 1.0 - _EPS)
    blended = float(alpha) * logit(f) + float(beta) * logit(pi)
    return expit(blended)


def _binary_log_loss(y_true, p, eps: float = _EPS) -> float:
    y = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)).mean())


def fit_binary_logodds_blend(
    y_true: list[float] | np.ndarray,
    p_model: np.ndarray,
    p_book: np.ndarray,
) -> BinaryBlendWeights:
    """Fit non-negative (α, β) for a binary log-odds blend."""
    f = np.asarray(p_model, dtype=float).reshape(-1)
    pi = np.asarray(p_book, dtype=float).reshape(-1)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    if not (len(y) == len(f) == len(pi)):
        raise ValueError("y_true, p_model, and p_book must have the same length")
    if len(y) == 0:
        raise ValueError("need at least one row to fit binary blend")

    def nll(coef: np.ndarray) -> float:
        return _binary_log_loss(
            y, apply_binary_logodds_blend(f, pi, coef[0], coef[1])
        )

    result = minimize(
        nll,
        x0=np.array([1.0, 1.0], dtype=float),
        bounds=[(_COEF_LO, _COEF_HI), (_COEF_LO, _COEF_HI)],
        method="L-BFGS-B",
    )
    return BinaryBlendWeights(alpha=float(result.x[0]), beta=float(result.x[1]))
