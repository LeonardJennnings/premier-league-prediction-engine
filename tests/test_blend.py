"""Unit tests for OOS Benter / log-odds blend helpers."""

import numpy as np

from src.eval.blend import (
    apply_benter_blend,
    apply_binary_logodds_blend,
    fit_benter_blend,
    fit_binary_logodds_blend,
)


def test_unit_sum_coefs_equal_inputs_leave_probs_unchanged():
    """When f = π, softmax((α+β) log f) equals f only if α+β = 1."""
    P = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]], dtype=float)
    out = apply_benter_blend(P, P, alpha=0.4, beta=0.6)
    assert np.allclose(out, P, atol=1e-10)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_alpha_zero_returns_book():
    model = np.array([[0.8, 0.1, 0.1]], dtype=float)
    book = np.array([[0.2, 0.3, 0.5]], dtype=float)
    out = apply_benter_blend(model, book, alpha=0.0, beta=1.0)
    assert np.allclose(out, book, atol=1e-10)


def test_beta_zero_returns_model():
    model = np.array([[0.8, 0.1, 0.1]], dtype=float)
    book = np.array([[0.2, 0.3, 0.5]], dtype=float)
    out = apply_benter_blend(model, book, alpha=1.0, beta=0.0)
    assert np.allclose(out, model, atol=1e-10)


def test_fit_prefers_book_when_model_is_wrong():
    """When the book matches outcomes and the model is confidently wrong, α→0."""
    y = ["A"] * 40
    book = np.tile(np.array([0.15, 0.15, 0.70]), (40, 1))
    model = np.tile(np.array([0.80, 0.10, 0.10]), (40, 1))
    weights = fit_benter_blend(y, model, book)
    assert weights.alpha < weights.beta
    assert weights.w_model < 0.5


def test_fit_prefers_model_when_book_is_wrong():
    y = ["H"] * 40
    model = np.tile(np.array([0.75, 0.15, 0.10]), (40, 1))
    book = np.tile(np.array([0.10, 0.15, 0.75]), (40, 1))
    weights = fit_benter_blend(y, model, book)
    assert weights.alpha > weights.beta
    assert weights.w_model > 0.5


def test_binary_blend_unit_sum_coefs_equal_inputs():
    p = np.array([0.3, 0.6, 0.8], dtype=float)
    out = apply_binary_logodds_blend(p, p, alpha=0.35, beta=0.65)
    assert np.allclose(out, p, atol=1e-10)


def test_fit_binary_prefers_book_when_model_is_wrong():
    y = np.ones(40, dtype=float)
    book = np.full(40, 0.8)
    model = np.full(40, 0.2)
    weights = fit_binary_logodds_blend(y, model, book)
    assert weights.alpha < weights.beta
    assert weights.w_model < 0.5
