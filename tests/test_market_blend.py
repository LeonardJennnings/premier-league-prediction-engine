"""One weight between the model rates and the price before kickoff."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eval.book_grid import fit_book_grid
from src.eval.market_blend import (
    MarketBlendResult,
    blended_rates,
    choose_weight,
    published_blend_table,
    render_market_blend_note,
    stop_decision,
    weight_name,
)
from src.models.dixon_coles import DixonColesModel, expected_goals, scoreline_grid_from_rates


def test_weight_zero_is_the_price_rates_and_one_is_the_model():
    lam, mu = blended_rates(2.0, 1.0, 1.2, 1.4, 0.0)
    assert (lam, mu) == pytest.approx((1.2, 1.4))
    lam, mu = blended_rates(2.0, 1.0, 1.2, 1.4, 1.0)
    assert (lam, mu) == pytest.approx((2.0, 1.0))
    lam, mu = blended_rates(2.0, 1.0, 1.2, 1.4, 0.25)
    assert lam == pytest.approx(0.25 * 2.0 + 0.75 * 1.2)


def test_published_blend_table_follows_the_weight_and_leaves_rho_at_zero():
    model = DixonColesModel(
        attack={"Arsenal": 1.3, "Chelsea": 0.9},
        defence={"Arsenal": 0.8, "Chelsea": 1.1},
        home_advantage=1.25,
        rho=-0.08,
        xi=0.0035,
    )
    p_home, p_draw, p_away, p_over = 0.46, 0.27, 0.27, 0.52
    book = fit_book_grid(p_home, p_draw, p_away, p_over)
    model_lam, model_mu = expected_goals(model, "Arsenal", "Chelsea")
    table = published_blend_table(
        model, "Arsenal", "Chelsea", p_home, p_draw, p_away, p_over, 0.25
    )
    lam, mu = blended_rates(model_lam, model_mu, book.lam, book.mu, 0.25)
    assert table["lambda_home"] == pytest.approx(lam)
    assert table["lambda_away"] == pytest.approx(mu)
    assert table["rho"] == 0.0
    assert table["blend_weight"] == pytest.approx(0.25)
    price_only = published_blend_table(
        model, "Arsenal", "Chelsea", p_home, p_draw, p_away, p_over, 0.0
    )
    assert price_only["lambda_home"] == pytest.approx(book.lam)
    assert price_only["lambda_away"] == pytest.approx(book.mu)


def test_blend_grid_uses_no_low_score_correction():
    lam, mu = blended_rates(1.4, 1.1, 1.3, 1.0, 0.5)
    grid = scoreline_grid_from_rates(lam, mu, 0.0)
    shifted = scoreline_grid_from_rates(lam, mu, -0.1)
    assert grid[0, 0] != pytest.approx(shifted[0, 0])


def test_choose_weight_ignores_later_seasons_and_keeps_the_smaller_tie():
    weights = (0.0, 0.05, 0.10)
    rows = []
    for weight in weights:
        rows.append(
            {
                "season_code": "2223",
                "model": weight_name(weight),
                "scoreline_loss": 3.0 if weight in (0.0, 0.05) else 3.2,
            }
        )
        rows.append(
            {
                "season_code": "2526",
                "model": weight_name(weight),
                "scoreline_loss": 1.0 if weight == 0.10 else 4.0,
            }
        )
    assert choose_weight(pd.DataFrame(rows), weights, ("2223",)) == 0.0


def test_stop_needs_both_a_low_weight_and_a_short_gain():
    dropped, reason = stop_decision(0.05, 0.004)
    assert dropped
    assert "under 0.10" in reason
    assert "under 0.01" in reason
    kept_weight, _ = stop_decision(0.10, 0.0)
    assert not kept_weight
    kept_gain, _ = stop_decision(0.05, 0.01)
    assert not kept_gain


def _result(weight: float, gain: float, *, dropped: bool, reason: str) -> MarketBlendResult:
    def row(name: str, mean: float) -> dict:
        return {
            "comparison": name,
            "model_a": "a",
            "model_b": "b",
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
            row("setting_pure_minus_blend_scoreline", 0.02),
            row("later_book_minus_blend_scoreline", gain),
            row("later_pure_minus_blend_scoreline", 0.03),
            row("later_pure_minus_blend_result", 0.01),
            row("later_blend_minus_closing_result", 0.02),
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "weight": weight,
                "scoreline_log_loss": 2.96,
                "matches": 940,
                "result_log_loss": 0.96,
                "chosen": True,
            }
        ]
    )
    return MarketBlendResult(
        chosen_weight=weight,
        dropped=dropped,
        reason=reason,
        candidates=candidates,
        paired=paired,
        matches=pd.DataFrame(),
        elapsed_seconds=60.0,
        run_date="25 Nov 2026",
        skipped_matches=0,
    )


def test_dropped_note_keeps_the_pure_model_and_quotes_the_range():
    dropped, reason = stop_decision(0.05, 0.004)
    text = render_market_blend_note(_result(0.05, 0.004, dropped=dropped, reason=reason))
    assert "95% range" in text
    assert "490 matches" in text
    assert "49 rounds" in text
    assert "pure model stays the published probability" in text
    assert "weight on the model is 0.05" in text
    assert "beat" not in text.lower()


def test_kept_note_does_not_call_a_short_gap_a_success():
    kept, reason = stop_decision(0.25, 0.004)
    text = render_market_blend_note(_result(0.25, 0.004, dropped=kept, reason=reason))
    assert "model plus market" in text
    assert "not a success" in text
    assert "not a claim about the bookmaker" in text
    assert "price available before kickoff" in text
    assert "pure model only" in text
