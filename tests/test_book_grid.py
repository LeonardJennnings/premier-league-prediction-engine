"""The book's score table is rebuilt from its prices, and rho stays at zero."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.book_grid import (
    BOOK_SCORE_TABLE_LABEL,
    CLOSING_PRICE_COLUMNS,
    OPENING_PRICE_COLUMNS,
    PRICE_LOG_LOSS_TOLERANCE,
    BookGridUnfaithful,
    assert_prices_reproduced,
    fit_book_grid,
    market_log_losses,
    markets_from_rates,
    prices_complete,
)
from src.eval.scoreline import render_measurement_notes
from src.models.dixon_coles import scoreline_grid_from_rates
from src.models.markets import markets_from_grid


def test_label_is_the_plan_sentence():
    assert BOOK_SCORE_TABLE_LABEL == (
        "our reconstruction of the book's score table from its prices, "
        "not the bookmaker's correct-score odds"
    )


def test_rate_markets_match_the_shared_grid():
    grid = scoreline_grid_from_rates(1.7, 1.1, 0.0)
    shared = markets_from_grid(grid)
    direct = markets_from_rates(1.7, 1.1)
    for key in ("home", "draw", "away", "over_25", "under_25"):
        assert direct[key] == pytest.approx(shared[key], abs=1e-12)


def test_fit_reproduces_the_prices_it_was_given():
    target = markets_from_rates(1.65, 1.25)
    fitted = fit_book_grid(target["home"], target["draw"], target["away"], target["over_25"])
    assert fitted.squared_error < 1e-8
    assert fitted.home == pytest.approx(target["home"], abs=1e-4)
    assert fitted.draw == pytest.approx(target["draw"], abs=1e-4)
    assert fitted.away == pytest.approx(target["away"], abs=1e-4)
    assert fitted.over_25 == pytest.approx(target["over_25"], abs=1e-4)
    assert fitted.lam == pytest.approx(1.65, abs=0.05)
    assert fitted.mu == pytest.approx(1.25, abs=0.05)
    # The stored grid is the independent Poisson table. Rho is not fitted.
    again = markets_from_grid(scoreline_grid_from_rates(fitted.lam, fitted.mu, 0.0))
    assert again["home"] == pytest.approx(fitted.home, abs=1e-12)


def test_unfaithful_reconstruction_stops():
    faithful = {
        "loss_1x2_grid": 1.0,
        "loss_1x2_price": 1.001,
        "loss_ou_grid": 0.67,
        "loss_ou_price": 0.672,
    }
    assert_prices_reproduced([faithful, faithful])
    drifted = dict(faithful)
    drifted["loss_1x2_grid"] = faithful["loss_1x2_price"] + PRICE_LOG_LOSS_TOLERANCE + 0.01
    with pytest.raises(BookGridUnfaithful, match="home/draw/away"):
        assert_prices_reproduced([drifted])


def test_market_log_loss_uses_the_outcome_that_happened():
    losses = market_log_losses(
        result="H",
        home_goals=2,
        away_goals=0,
        price_home=0.5,
        price_draw=0.3,
        price_away=0.2,
        price_over=0.4,
        price_under=0.6,
        grid_home=0.45,
        grid_draw=0.3,
        grid_away=0.25,
        grid_over=0.42,
        grid_under=0.58,
    )
    assert losses["loss_1x2_price"] == pytest.approx(-math.log(0.5))
    assert losses["loss_1x2_grid"] == pytest.approx(-math.log(0.45))
    # 2-0 is under 2.5, so the under price is the one that is scored.
    assert losses["loss_ou_price"] == pytest.approx(-math.log(0.6))
    assert losses["loss_ou_grid"] == pytest.approx(-math.log(0.58))
    assert np.isfinite(losses["loss_ou_grid"])


def test_an_earlier_total_is_not_a_closing_target():
    frame = pd.DataFrame(
        {
            "implied_home_open": [0.45],
            "implied_draw_open": [0.28],
            "implied_away_open": [0.27],
            "implied_over_25_open": [0.52],
            "implied_under_25_open": [0.48],
        }
    )
    assert prices_complete(frame, OPENING_PRICE_COLUMNS)
    assert not prices_complete(frame, CLOSING_PRICE_COLUMNS)


def test_notes_carry_the_label_and_a_ranged_gap():
    summary = pd.DataFrame(
        [
            {
                "model": "dixon_coles",
                "market": "scoreline",
                "n_test": 4,
                "log_loss": 3.0,
                "modal_hit": 0.1,
                "top3_hit": 0.3,
                "mae_home": 1.0,
                "mae_away": 1.0,
                "rps": 1.0,
            },
            {
                "model": "book_score_reconstruction",
                "market": "scoreline",
                "n_test": 4,
                "log_loss": 2.96,
                "modal_hit": 0.12,
                "top3_hit": 0.3,
                "mae_home": 1.0,
                "mae_away": 1.0,
                "rps": 1.0,
            },
            {
                "model": "league_score_frequencies",
                "market": "scoreline",
                "n_test": 4,
                "log_loss": 3.09,
                "modal_hit": 0.1,
                "top3_hit": 0.3,
                "mae_home": 1.0,
                "mae_away": 1.0,
                "rps": 1.0,
            },
            {
                "model": "always_1_1",
                "market": "scoreline",
                "n_test": 4,
                "log_loss": float("nan"),
                "modal_hit": 0.11,
                "top3_hit": float("nan"),
                "mae_home": float("nan"),
                "mae_away": float("nan"),
                "rps": float("nan"),
            },
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "metric": "scoreline_log_loss",
                "model_a": "dixon_coles",
                "model_b": "book_score_reconstruction",
                "sample": "all",
                "n_matches": 4,
                "n_rounds": 1,
                "mean_diff": 0.04,
                "ci_low": 0.01,
                "ci_high": 0.07,
                "note": "A positive difference means the first line has the higher log loss",
            }
        ]
    )
    scores = pd.DataFrame(
        {
            "market": ["scoreline", "scoreline", "book_grid_check"],
            "model": ["dixon_coles", "dixon_coles", "book_score_reconstruction"],
            "season_code": ["2223", "2526", "2223"],
            "loss_1x2_grid": [float("nan"), float("nan"), 0.966],
            "loss_1x2_price": [float("nan"), float("nan"), 0.965],
            "loss_ou_grid": [float("nan"), float("nan"), 0.670],
            "loss_ou_price": [float("nan"), float("nan"), 0.670],
        }
    )
    text = render_measurement_notes(summary, paired, scores)
    assert BOOK_SCORE_TABLE_LABEL in text
    assert "95% range 0.0100 to 0.0700" in text
    assert "4 matches" in text
    lowered = text.lower()
    assert "beat the book" not in lowered
    assert "beats the book" not in lowered


def test_committed_book_grid_reproduces_the_prices_and_the_review():
    """The committed reconstruction stays inside the price tolerance and near 2.96."""
    check = pd.read_csv(Path("results/walkforward_book_grid_check.csv"))
    summary = pd.read_csv(Path("results/walkforward_summary.csv"))
    notes = Path("results/WALKFORWARD.md").read_text(encoding="utf-8")
    closing = check.loc[check["model"] == "book_score_reconstruction"].iloc[0]
    opening = check.loc[check["model"] == "book_score_reconstruction_open"].iloc[0]
    assert int(closing["n_matches"]) == int(opening["n_matches"])
    assert abs(float(closing["gap_1x2"])) <= PRICE_LOG_LOSS_TOLERANCE
    assert abs(float(closing["gap_ou"])) <= PRICE_LOG_LOSS_TOLERANCE
    assert abs(float(opening["gap_1x2"])) <= PRICE_LOG_LOSS_TOLERANCE
    assert abs(float(opening["gap_ou"])) <= PRICE_LOG_LOSS_TOLERANCE
    score = summary.loc[summary["market"] == "scoreline"]
    book = score.loc[score["model"] == "book_score_reconstruction"].iloc[0]
    assert int(book["n_test"]) == int(closing["n_matches"])
    assert float(book["log_loss"]) == pytest.approx(2.960, abs=0.015)
    assert float(book["modal_hit"]) == pytest.approx(0.125, abs=0.015)
    assert BOOK_SCORE_TABLE_LABEL in notes
    assert "beat the book" not in notes.lower()
    assert "beats the book" not in notes.lower()
