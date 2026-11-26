"""The saved scoreline gap, split into rates and shape. No refit."""

import numpy as np
import pandas as pd
import pytest

from src.config import RESULTS
from src.eval.gap_split import (
    RATE_MISS_HIGH,
    RATE_MISS_LOW,
    build_match_table,
    locate_tenth,
    rate_miss_bin,
    render_gap_split,
    run_gap_split,
    summarise_match_table,
    write_gap_split_artefacts,
)
from src.eval.scoreline import score_scoreline
from src.models.dixon_coles import calibrated_score_grid, scoreline_grid_from_rates


def test_rate_miss_cuts_stay_at_0_15_and_0_40():
    assert RATE_MISS_LOW == 0.15
    assert RATE_MISS_HIGH == 0.40
    assert rate_miss_bin(0.0) == "under 0.15"
    assert rate_miss_bin(0.149999) == "under 0.15"
    assert rate_miss_bin(0.15) == "from 0.15 to 0.40"
    assert rate_miss_bin(0.40) == "from 0.15 to 0.40"
    assert rate_miss_bin(0.400001) == "over 0.40"


def test_locate_tenth_uses_the_small_gap_line():
    assert locate_tenth(0.0986, 0.0009) == "rates"
    assert locate_tenth(0.0004, 0.08) == "shape"
    assert locate_tenth(0.04, 0.05) == "both"
    assert locate_tenth(0.009, 0.009) == "both"


def _match_rows(
    match_id: str,
    round_id: str,
    season: str,
    model_lam: float,
    model_mu: float,
    book_lam: float,
    book_mu: float,
    rho: float,
    home_goals: int,
    away_goals: int,
):
    rho0_grid = scoreline_grid_from_rates(model_lam, model_mu, 0.0)
    model_grid, used_lam, used_mu, used_rho = calibrated_score_grid(
        model_lam, model_mu, rho, rho=rho
    )
    book_grid = scoreline_grid_from_rates(book_lam, book_mu, 0.0)
    rho0 = score_scoreline(rho0_grid, home_goals, away_goals, lam=model_lam, mu=model_mu, rho=0.0)
    model = score_scoreline(
        model_grid, home_goals, away_goals, lam=used_lam, mu=used_mu, rho=used_rho
    )
    book = score_scoreline(book_grid, home_goals, away_goals, lam=book_lam, mu=book_mu, rho=0.0)

    def line(model_name: str, loss: float) -> dict:
        return {
            "match_id": match_id,
            "round_id": round_id,
            "season_code": season,
            "market": "scoreline",
            "model": model_name,
            "loss": loss,
            "exp_home": rho0["exp_home"],
            "exp_away": rho0["exp_away"],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "lam": np.nan,
            "mu": np.nan,
        }

    return [
        line("dixon_coles", model["cell_log_loss"]),
        line("dixon_coles_rho0", rho0["cell_log_loss"]),
        line("book_score_reconstruction", book["cell_log_loss"]),
        {
            "match_id": match_id,
            "round_id": round_id,
            "season_code": season,
            "market": "book_grid_check",
            "model": "book_score_reconstruction",
            "loss": np.nan,
            "exp_home": np.nan,
            "exp_away": np.nan,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "lam": book_lam,
            "mu": book_mu,
        },
    ]


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_same_rates_leave_the_rate_piece_at_zero():
    rows = _match_rows("m1", "r1", "2526", 1.4, 1.1, 1.4, 1.1, -0.08, 1, 0)
    table = build_match_table(_frame(rows))
    assert table["rate_gap"].iloc[0] == pytest.approx(0.0, abs=1e-8)
    assert table["shape_gap"].iloc[0] == pytest.approx(table["total_gap"].iloc[0])
    assert table["rho"].iloc[0] == pytest.approx(-0.08, abs=1e-4)
    assert table["model_replay_error"].iloc[0] < 1e-8
    assert table["book_replay_error"].iloc[0] < 1e-8


def test_zero_correction_leaves_the_shape_piece_at_zero():
    rows = _match_rows("m1", "r1", "2526", 1.8, 0.9, 1.2, 1.3, 0.0, 2, 1)
    table = build_match_table(_frame(rows))
    assert table["shape_gap"].iloc[0] == pytest.approx(0.0, abs=1e-8)
    assert table["rate_gap"].iloc[0] == pytest.approx(table["total_gap"].iloc[0])
    assert table["total_rate_bin"].iloc[0] == "from 0.15 to 0.40"


def test_two_matches_in_one_round_share_one_correction():
    rows = _match_rows("m1", "r1", "2425", 1.5, 1.2, 1.1, 1.4, -0.06, 1, 0)
    rows += _match_rows("m2", "r1", "2425", 1.7, 0.8, 2.1, 0.7, -0.06, 0, 0)
    table = build_match_table(_frame(rows))
    assert table["rho"].nunique() == 1
    assert table["rho"].iloc[0] == pytest.approx(-0.06, abs=1e-4)
    assert np.max(np.abs(table["rate_gap"] + table["shape_gap"] - table["total_gap"])) < 1e-9


def test_unidentified_correction_stays_at_zero():
    rows = _match_rows("m1", "r1", "2526", 1.6, 1.1, 1.3, 1.0, -0.07, 2, 2)
    frame = _frame(rows)
    rho0_loss = frame.loc[frame["model"] == "dixon_coles_rho0", "loss"].iloc[0]
    frame.loc[frame["model"] == "dixon_coles", "loss"] = rho0_loss
    table = build_match_table(frame)
    assert table["rho"].iloc[0] == 0.0
    assert table["model_replay_error"].iloc[0] < 1e-8


def test_different_match_sets_are_rejected():
    rows = _match_rows("m1", "r1", "2526", 1.4, 1.1, 1.2, 1.0, -0.05, 1, 0)
    rows += _match_rows("m2", "r2", "2526", 1.4, 1.1, 1.2, 1.0, -0.05, 2, 0)
    frame = _frame(rows)
    frame = frame.loc[~((frame["match_id"] == "m2") & (frame["model"] == "book_score_reconstruction") & (frame["market"] == "scoreline"))]
    with pytest.raises(ValueError, match="different matches"):
        build_match_table(frame)


def test_groups_stay_in_fixed_order_and_the_note_does_not_rank_them(tmp_path):
    rows = _match_rows("early", "r1", "2223", 1.5, 1.2, 1.45, 1.15, 0.0, 1, 1)
    rows += _match_rows("late", "r2", "2627", 2.4, 0.6, 1.0, 1.6, 0.0, 3, 0)
    result = summarise_match_table(
        build_match_table(_frame(rows)),
        n_boot=40,
        seed=3,
        run_date="26 Nov 2026",
    )
    seasons = [gap.group for gap in result.gaps if gap.family == "season"]
    assert seasons == ["2223", "2627"]
    families = [gap.family for gap in result.gaps]
    assert families.index("season") < families.index("total_rate")
    text = render_gap_split(result)
    assert text.index("## By season") < text.index("## By the size of the total-rate miss")
    assert text.index("| 2223 |") < text.index("| 2627 |")
    assert "·" not in text
    lowered = text.lower()
    assert "beats the book" not in lowered
    assert "beat the book" not in lowered
    assert "95% range" in text
    assert "matches" in text
    paths = write_gap_split_artefacts(result, tmp_path)
    saved = pd.read_csv(paths["table"])
    assert list(saved["group"].head(3)) == [
        "model minus closing reconstruction",
        "rate piece",
        "shape piece",
    ]
    assert list(saved.loc[saved["family"] == "season", "group"]) == ["2223", "2627"]


def test_saved_walkforward_reprints_the_committed_gap():
    """The unsplit gap has to match the committed paired row. A miss means the join is wrong."""
    result = run_gap_split()
    gap = next(
        item
        for item in result.gaps
        if item.family == "same_matches" and item.group == "model minus closing reconstruction"
    )
    paired = pd.read_csv(RESULTS / "walkforward_paired.csv")
    published = paired.loc[
        (paired["metric"] == "scoreline_log_loss")
        & (paired["model_a"] == "dixon_coles")
        & (paired["model_b"] == "book_score_reconstruction")
        & (paired["sample"] == "all")
    ].iloc[0]
    assert gap.mean_diff == pytest.approx(published["mean_diff"])
    assert gap.ci_low == pytest.approx(published["ci_low"])
    assert gap.ci_high == pytest.approx(published["ci_high"])
    assert gap.n_matches == int(published["n_matches"]) == 1430
    assert gap.n_rounds == int(published["n_rounds"]) == 143
    assert gap.seed == int(published["seed"]) == 20261124
    assert result.location == "rates"
    rate = next(item for item in result.gaps if item.group == "rate piece")
    shape = next(item for item in result.gaps if item.group == "shape piece")
    assert rate.mean_diff + shape.mean_diff == pytest.approx(gap.mean_diff, abs=0.00015)
    assert rate.n_matches == shape.n_matches == 1430
    total_rate = [item for item in result.gaps if item.family == "total_rate"]
    split = [item for item in result.gaps if item.family == "rate_split"]
    assert [item.group for item in total_rate] == [
        "under 0.15",
        "from 0.15 to 0.40",
        "over 0.40",
    ]
    assert sum(item.n_matches for item in total_rate) == 1430
    assert sum(item.n_matches for item in split) == 1430
    note = render_gap_split(result)
    assert "The tenth is in the rates." in note
    assert "0.0995" in note
    assert "0.0608 to 0.1419" in note
    assert "1430 matches, 143 rounds" in note
    assert "·" not in note
    assert "beats the book" not in note.lower()
    committed = (RESULTS / "GAP_SPLIT.md").read_text(encoding="utf-8")
    assert "The tenth is in the rates." in committed
    assert "0.0995" in committed
    assert "0.0608 to 0.1419" in committed
    assert "·" not in committed
