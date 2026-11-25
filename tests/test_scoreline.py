"""Scoreline metrics, the 9-0 tail, and paired comparisons on one match set."""

import math

import numpy as np
import pandas as pd
import pytest

from src.config import RESULTS

from src.eval.scoreline import (
    assert_one_match_set,
    block_bootstrap_mean_diff,
    empirical_frequency_grid,
    exact_cell_probability,
    goal_difference_rps,
    paired_comparisons,
    render_measurement_notes,
    score_scoreline,
)
from src.models.dixon_coles import scoreline_grid_from_rates


def _grid():
    # Goals 0 and 1 only.
    return np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )


def test_cell_log_loss_modal_and_goal_errors():
    scored = score_scoreline(_grid(), 1, 0)
    assert scored["p_cell"] == pytest.approx(0.3)
    assert scored["cell_log_loss"] == pytest.approx(-math.log(0.3))
    assert scored["modal_hit"] == 0.0  # mode is 1-1
    assert scored["top3_hit"] == 1.0
    assert scored["exp_home"] == pytest.approx(0.7)
    assert scored["exp_away"] == pytest.approx(0.6)
    assert scored["abs_err_home"] == pytest.approx(0.3)


def test_goal_difference_rps_hand_value():
    # Differences -1, 0, +1 have probabilities 0.2, 0.5, 0.3.
    # Actual 1-0 is difference +1.
    # Squared CDF errors, dropping the last bin: 0.2^2 + 0.7^2 = 0.53.
    assert goal_difference_rps(_grid(), 1, 0) == pytest.approx(0.53)


def test_nine_nil_uses_a_finite_exact_cell_not_a_neighbour():
    grid = scoreline_grid_from_rates(1.5, 1.2, 0.0, max_goals=8)
    scored = score_scoreline(grid, 9, 0, lam=1.5, mu=1.2, rho=0.0)
    expected = exact_cell_probability(1.5, 1.2, 0.0, 9, 0)
    assert expected > 0.0
    assert math.isfinite(expected)
    assert scored["p_cell"] == pytest.approx(expected)
    assert math.isfinite(scored["cell_log_loss"])
    assert math.isfinite(scored["rps"])
    # 9-0 is not one of the cells inside 0..8, so it is not a listed hit.
    assert scored["modal_hit"] == 0.0
    assert scored["top3_hit"] == 0.0


def test_unseen_frequency_uses_one_over_n_plus_one():
    train_home = [1, 1, 2, 0]
    train_away = [0, 1, 0, 0]
    grid = empirical_frequency_grid(train_home, train_away, max_goals=8)
    seen = score_scoreline(grid, 1, 0)
    assert seen["p_cell"] == pytest.approx(1 / 4)
    unseen = score_scoreline(grid, 5, 3, unseen_probability=1 / 5)
    assert unseen["p_cell"] == pytest.approx(1 / 5)
    assert math.isfinite(unseen["cell_log_loss"])


def test_scoreline_table_rejects_mixed_match_counts():
    rows = pd.DataFrame({"model": ["a", "b"], "n_test": [10, 8]})
    with pytest.raises(ValueError, match="different matches"):
        assert_one_match_set(rows)


def test_paired_comparison_rejects_partial_overlap_and_is_repeatable():
    scores = pd.DataFrame(
        [
            {"match_id": "m1", "round_id": "r1", "season_code": "2526", "market": "1x2", "model": "a", "loss": 1.0, "hit": 1.0},
            {"match_id": "m2", "round_id": "r1", "season_code": "2526", "market": "1x2", "model": "a", "loss": 2.0, "hit": 0.0},
            {"match_id": "m3", "round_id": "r2", "season_code": "2526", "market": "1x2", "model": "a", "loss": 1.5, "hit": 1.0},
            {"match_id": "m1", "round_id": "r1", "season_code": "2526", "market": "1x2", "model": "b", "loss": 0.5, "hit": 0.0},
            {"match_id": "m2", "round_id": "r1", "season_code": "2526", "market": "1x2", "model": "b", "loss": 0.5, "hit": 1.0},
            {"match_id": "m4", "round_id": "r2", "season_code": "2526", "market": "1x2", "model": "b", "loss": 0.5, "hit": 1.0},
        ]
    )
    with pytest.raises(ValueError, match="different matches"):
        paired_comparisons(scores, [("1x2", "loss", "a", "b", "all")], n_boot=20, seed=1)

    # Drop the exclusive rows so the shared set is a real pair.
    shared = scores.loc[scores["match_id"].isin(["m1", "m2"])]
    first = paired_comparisons(shared, [("1x2", "loss", "a", "b", "all")], n_boot=30, seed=7)
    second = paired_comparisons(shared, [("1x2", "loss", "a", "b", "all")], n_boot=30, seed=7)
    assert first["n_matches"].iloc[0] == 2
    assert first["ci_low"].iloc[0] <= first["ci_high"].iloc[0]
    assert first["mean_diff"].iloc[0] == pytest.approx(second["mean_diff"].iloc[0])
    assert first["ci_low"].iloc[0] == pytest.approx(second["ci_low"].iloc[0])


def test_block_bootstrap_resamples_rounds_not_matches():
    # Two rounds. The gap is constant inside a round, so a match-level story
    # and a round-level story can disagree about the spread. The call just
    # has to use the round ids and return a finite range.
    diff = np.array([1.0, 1.0, -1.0, -1.0])
    rounds = np.array(["r1", "r1", "r2", "r2"])
    mean, low, high, n_rounds = block_bootstrap_mean_diff(diff, rounds, n_boot=50, seed=3)
    assert mean == pytest.approx(0.0)
    assert n_rounds == 2
    assert low <= high


def test_measurement_notes_quote_range_and_match_count():
    summary = pd.DataFrame(
        [
            {
                "model": "dixon_coles",
                "market": "scoreline",
                "n_test": 4,
                "log_loss": 3.059,
                "modal_hit": 0.107,
                "top3_hit": 0.29,
                "mae_home": 0.99,
                "mae_away": 0.90,
                "rps": 0.94,
            },
            {
                "model": "league_score_frequencies",
                "market": "scoreline",
                "n_test": 4,
                "log_loss": 3.093,
                "modal_hit": 0.108,
                "top3_hit": 0.25,
                "mae_home": 1.0,
                "mae_away": 1.0,
                "rps": 1.0,
            },
            {
                "model": "always_1_1",
                "market": "scoreline",
                "n_test": 4,
                "log_loss": float("nan"),
                "modal_hit": 0.113,
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
                "model_b": "league_score_frequencies",
                "sample": "all",
                "n_matches": 4,
                "n_rounds": 1,
                "mean_diff": -0.034,
                "ci_low": -0.050,
                "ci_high": -0.010,
                "note": "positive means model_a has the higher log loss",
            }
        ]
    )
    scores = pd.DataFrame(
        {
            "market": ["scoreline"] * 4,
            "model": ["dixon_coles"] * 4,
            "season_code": ["2223", "2324", "2526", "2627"],
        }
    )
    text = render_measurement_notes(summary, paired, scores)
    assert "2223" in text
    assert "2526" in text
    assert "480" in text
    assert "95% range -0.0500 to -0.0100" in text
    assert "4 matches" in text
    assert "not a probability forecast" in text
    lowered = text.lower()
    assert "beat the book" not in lowered
    assert "beats the book" not in lowered


def test_committed_scoreline_matches_the_review_within_rounding():
    """The committed backtest reproduces the review's scoreline measurements."""
    summary = pd.read_csv(RESULTS / "walkforward_summary.csv")
    score = summary.loc[summary["market"] == "scoreline"]
    assert_one_match_set(score)
    dc = score.loc[score["model"] == "dixon_coles"].iloc[0]
    freq = score.loc[score["model"] == "league_score_frequencies"].iloc[0]
    always = score.loc[score["model"] == "always_1_1"].iloc[0]
    assert int(dc["n_test"]) == int(freq["n_test"]) == int(always["n_test"])
    assert dc["log_loss"] == pytest.approx(3.059, abs=0.005)
    assert freq["log_loss"] == pytest.approx(3.093, abs=0.005)
    assert dc["modal_hit"] == pytest.approx(0.107, abs=0.005)
    assert freq["modal_hit"] == pytest.approx(0.108, abs=0.005)
    assert always["modal_hit"] == pytest.approx(0.113, abs=0.005)
    assert math.isfinite(dc["log_loss"])
    assert pd.isna(always["log_loss"])

    paired = pd.read_csv(RESULTS / "walkforward_paired.csv")
    assert not paired.empty
    for row in paired.itertuples(index=False):
        assert int(row.n_matches) > 0
        assert math.isfinite(row.mean_diff)
        assert math.isfinite(row.ci_low) and math.isfinite(row.ci_high)
        assert row.ci_low <= row.ci_high
    # The headline scoreline gap is on the full shared set, not a shorter one.
    gap = paired.loc[
        (paired["metric"] == "scoreline_log_loss")
        & (paired["model_a"] == "dixon_coles")
        & (paired["model_b"] == "league_score_frequencies")
        & (paired["sample"] == "all")
    ].iloc[0]
    assert int(gap["n_matches"]) == int(dc["n_test"])
