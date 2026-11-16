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
    