import numpy as np

from src.eval.calibration import (
    apply_temperature,
    apply_vector_scaling,
    fit_temperature,
    fit_vector_scaling,
    reliability_table,
)


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


_Y_CAL = ["H", "D", "A", "H", "A", "D", "H", "A"]
_P_CAL = np.array(
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


def test_temperature_one_leaves_probs_almost_unchanged():
    P = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]], dtype=float)
    out = apply_temperature(P, 1.0)
    assert np.allclose(out, P, atol=1e-10)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_fit_temperature_returns_positive():
    t = fit_temperature(_Y_CAL, _P_CAL)
    assert t > 0


def test_fit_temperature_softens_overconfident_errors():
    y = ["A"] * 8
    P = np.tile(np.array([0.9, 0.05, 0.05]), (8, 1))
    t = fit_temperature(y, P)
    assert t > 1.0


def test_temperature_greater_than_one_flattens():
    P = np.array([[0.8, 0.15, 0.05]], dtype=float)
    out = apply_temperature(P, 2.0)
    assert out[0, 0] < P[0, 0]
    assert np.allclose(out.sum(axis=1), 1.0)


def test_vector_scaling_ones_leaves_probs_unchanged():
    P = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]], dtype=float)
    out = apply_vector_scaling(P, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(out, P, atol=1e-12)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_fit_vector_scaling_returns_positive_weights():
    w = fit_vector_scaling(_Y_CAL, _P_CAL)
    assert w.shape == (3,)
    assert np.all(w > 0)
    assert abs(w[0] - 1.0) < 1e-9
    out = apply_vector_scaling(_P_CAL, w)
    assert np.allclose(out.sum(axis=1), 1.0)
 