"""The score-table scale and the steady low-score correction."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eval.grid_calibration import (
    choose_pair,
    pair_name,
    render_grid_calibration_note,
    stop_decision,
)
from src.eval.grid_calibration import GridCalibrationResult
from src.models.dixon_coles import (
    DixonColesModel,
    adjusted_rates,
    calibrated_score_grid,
    expected_goals,
    scoreline_grid,
    scoreline_grid_from_rates,
)


def _model() -> DixonColesModel:
    return DixonColesModel(
        attack={"Alpha": 1.1, "Beta": 0.9},
        defence={"Alpha": 1.0, "Beta": 1.2},
        home_advantage=1.25,
        rho=-0.04,
        xi=0.0035,
    )


def test_scale_moves_both_rates_by_the_same_factor():
    lam, mu, _rho = adjusted_rates(1.2, 1.0, -0.04, scale=1.05, rho=-0.10)
    assert lam == pytest.approx(1.26)
    assert mu == pytest.approx(1.05)


def test_constant_correction_changes_the_nil_nil_cell():
    raw = scoreline_grid_from_rates(1.4, 1.1, -0.02)
    grid, _lam, _mu, rho = calibrated_score_grid(
        1.4, 1.1, -0.02, scale=1.0, rho=-0.10
    )
    assert rho == pytest.approx(-0.10)
    assert grid[0, 0] != pytest.approx(raw[0, 0])


def test_published_grid_uses_a_kept_pair(monkeypatch):
    monkeypatch.setattr("src.config.GRID_CALIBRATION_SCALE", 1.05)
    monkeypatch.setattr("src.config.GRID_CALIBRATION_RHO", -0.10)
    model = _model()
    lam, mu = expected_goals(model, "Alpha", "Beta")
    grid = scoreline_grid(model, "Alpha", "Beta")
    expected, _, _, _ = calibrated_score_grid(
        lam, mu, model.rho, scale=1.05, rho=-0.10
    )
    assert grid == pytest.approx(expected)


def test_choose_pair_ignores_later_seasons_and_keeps_the_smaller_tie():
    scales = (0.95, 1.0, 1.05)
    rhos = (-0.10, -0.05, 0.0)
    rows = []
    for scale in scales:
        for rho in rhos:
            loss = 3.5
            if (scale, rho) in {(1.0, 0.0), (1.05, -0.10)}:
                loss = 3.0
            rows.append(
                {
                    "season_code": "2223",
                    "model": pair_name(scale, rho),
                    "scoreline_loss": loss,
                }
            )
            # A later season that would pick the larger change if it were used.
            rows.append(
                {
                    "season_code": "2526",
                    "model": pair_name(scale, rho),
                    "scoreline_loss": 1.0 if (scale, rho) == (1.05, -0.10) else 4.0,
                }
            )
    scale, rho = choose_pair(pd.DataFrame(rows), scales, rhos, ("2223",))
    assert (scale, rho) == (1.0, 0.0)


def test_stop_drops_a_gain_under_the_line_and_keeps_one_on_it():
    dropped, reason = stop_decision(0.0049)
    assert dropped
    assert "under 0.005" in reason
    kept, kept_reason = stop_decision(0.005)
    assert not kept
    assert "not under 0.005" in kept_reason


def _result(gain: float, *, dropped: bool, reason: str) -> GridCalibrationResult:
    def row(name: str, mean: float) -> dict:
        return {
            "comparison": name,
            "model_a": "dixon_coles",
            "model_b": "grid_calibration",
            "mean_diff": mean,
            "ci_low": mean - 0.01,
            "ci_high": mean + 0.01,
            "n_matches": 490,
            "n_rounds": 49,
            "n_bootstrap": 4000,
            "seed": 1,
            "note": "",
        }

    paired = pd.DataFrame(
        [
            row("setting_scoreline", 0.01),
            row("later_scoreline", gain),
            row("later_result", 0.001),
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "scale": 1.0,
                "rho": 0.0,
                "scoreline_log_loss": 3.0,
                "matches": 940,
                "result_log_loss": 0.96,
                "chosen": True,
            }
        ]
    )
    return GridCalibrationResult(
        chosen_scale=1.0,
        chosen_rho=0.0,
        dropped=dropped,
        reason=reason,
        candidates=candidates,
        paired=paired,
        matches=pd.DataFrame(),
        elapsed_seconds=60.0,
        run_date="25 Nov 2026",
    )


def test_result_note_quotes_a_range_and_does_not_claim_a_small_gain():
    dropped, reason = stop_decision(0.004)
    text = render_grid_calibration_note(_result(0.004, dropped=dropped, reason=reason))
    assert "95% range" in text
    assert "490 matches" in text
    assert "49 rounds" in text
    assert "published score table stays the previous one" in text
    assert "No closing home/draw/away line" in text
    assert "beat" not in text.lower()


def test_kept_note_calls_a_sub_hundredth_gain_a_point_estimate():
    kept, reason = stop_decision(0.007)
    text = render_grid_calibration_note(_result(0.007, dropped=kept, reason=reason))
    assert "not a success" in text
    assert "not a claim about the bookmaker" in text
    assert "uses scale 1.00" in text
