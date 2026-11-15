"""Tests for evaluation metrics."""

import math

from src.evaluate import accuracy, brier_score, log_loss


def test_metrics_hand_checked_home_favourite():
    y_true = ["H"]
    probas = [[0.7, 0.2, 0.1]]  # H, D, A

    assert abs(log_loss(y_true, probas) - (-math.log(0.7))) < 1e-9

    expected_brier = ((0.7 - 1) ** 2 + (0.2 - 0) ** 2 + (0.1 - 0) ** 2) / 3
    assert abs(brier_score(y_true, probas) - expected_brier) < 1e-9

    assert accuracy(y_true, probas) == 1.0


def test_accuracy_zero_when_wrong():
    y_true = ["A"]
    probas = [[0.7, 0.2, 0.1]]  # predicts H
    assert accuracy(y_true, probas) == 0.0
