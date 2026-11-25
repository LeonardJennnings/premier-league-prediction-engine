import pandas as pd

from src.eval.thin_prior import (
    annotate_matches,
    choose_kappa,
    prior_appearance_counts,
    render_thin_prior_note,
    stop_decision,
)
from src.eval.thin_prior import ThinPriorResult


def test_same_day_matches_do_not_count_as_earlier():
    dates = pd.to_datetime(["2024-08-01", "2024-08-01", "2024-08-08"])
    counts = prior_appearance_counts(
        pd.Series(["2425", "2425", "2425"]),
        pd.Series(["Alpha", "Alpha", "Alpha"]),
        pd.Series(dates),
    )
    assert list(counts) == [0, 0, 2]


def test_early_flag_is_the_first_eight_matches():
    dates = pd.date_range("2024-08-01", periods=9, freq="7D")
    frame = pd.DataFrame(
        {
            "season_code": ["2425"] * 9,
            "date": dates,
            "home_team": ["Alpha"] * 9,
            "away_team": ["Beta"] * 9,
            "home_goals": [1] * 9,
            "away_goals": [0] * 9,
            "result": ["H"] * 9,
        }
    )
    marked = annotate_matches(frame, early_limit=8)
    assert marked["early"].tolist()[:8] == [True] * 8
    assert bool(marked["early"].iloc[8]) is False


def test_choose_kappa_ignores_later_seasons_and_keeps_the_smaller_tie():
    rows = []
    # Setting era: strength 8 is worse, 2 and 4 tie, so 2 wins.
    for season, losses in {
        "2223": {"prior_2": 1.0, "prior_4": 1.0, "prior_8": 1.2},
        "2526": {"prior_2": 9.0, "prior_4": 0.1, "prior_8": 0.1},
    }.items():
        for model, loss in losses.items():
            rows.append(
                {
                    "season_code": season,
                    "early": True,
                    "model": model,
                    "scoreline_loss": loss,
                }
            )
    # A non-early row in the setting era must not decide the choice.
    rows.append(
        {
            "season_code": "2223",
            "early": False,
            "model": "prior_8",
            "scoreline_loss": 0.0,
        }
    )
    chosen = choose_kappa(pd.DataFrame(rows), (2.0, 4.0, 8.0), ("2223", "2324", "2425"))
    assert chosen == 2.0


def test_stop_decision_drops_when_early_matches_are_not_better():
    dropped, reason = stop_decision(0.0, -0.01, -0.01)
    assert dropped
    assert "early-match" in reason


def test_stop_decision_drops_when_the_overall_score_is_worse():
    dropped, reason = stop_decision(-0.02, 0.001, -0.01)
    assert dropped
    assert "overall scoreline" in reason


def test_stop_decision_keeps_a_lower_early_score_when_overall_is_not_worse():
    dropped, reason = stop_decision(-0.02, -0.001, 0.0)
    assert not dropped
    assert "lower" in reason


def test_result_note_quotes_a_range_and_a_match_count():
    paired = pd.DataFrame(
        [
            _gap("later_early_scoreline", -0.02, -0.05, 0.01, 40, 8),
            _gap("later_early_1x2", -0.01, -0.04, 0.02, 40, 8),
            _gap("later_overall_scoreline", -0.003, -0.01, 0.004, 200, 20),
            _gap("later_overall_1x2", -0.002, -0.01, 0.006, 200, 20),
            _gap("later_new_early_scoreline", -0.04, -0.1, 0.02, 12, 6),
            _gap("setting_early_scoreline", -0.01, -0.03, 0.01, 80, 16),
            _gap("later_book_prior", 0.04, 0.01, 0.07, 190, 19),
            _gap("later_book_previous", 0.042, 0.012, 0.072, 190, 19),
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "kappa": 0.0,
                "early_scoreline_log_loss": 3.10,
                "early_matches": 80,
                "overall_scoreline_log_loss": 3.06,
                "overall_matches": 400,
                "early_result_log_loss": 1.05,
                "overall_result_log_loss": 1.02,
                "chosen": False,
            },
            {
                "kappa": 8.0,
                "early_scoreline_log_loss": 3.09,
                "early_matches": 80,
                "overall_scoreline_log_loss": 3.058,
                "overall_matches": 400,
                "early_result_log_loss": 1.04,
                "overall_result_log_loss": 1.018,
                "chosen": True,
            },
        ]
    )
    result = ThinPriorResult(
        chosen_kappa=8.0,
        dropped=False,
        reason="early-match scoreline log loss is lower on the later seasons, and the overall scores are not higher",
        candidates=candidates,
        paired=paired,
        matches=pd.DataFrame(),
        elapsed_seconds=120.0,
        run_date="25 Nov 2026",
    )
    text = render_thin_prior_note(result)
    assert "95% range -0.0500 to 0.0100, 40 matches, 8 rounds" in text
    assert "200 matches" in text
    assert "not a success" in text
    assert "not a claim about the bookmaker" in text
    assert "beat" not in text.lower()
    assert "·" not in text


def test_result_note_says_the_previous_model_stays_when_the_rule_drops():
    paired = pd.DataFrame(
        [
            _gap("later_early_scoreline", 0.01, -0.02, 0.04, 40, 8),
            _gap("later_early_1x2", 0.01, -0.02, 0.04, 40, 8),
            _gap("later_overall_scoreline", 0.002, -0.01, 0.014, 200, 20),
            _gap("later_overall_1x2", 0.0, -0.01, 0.01, 200, 20),
            _gap("later_new_early_scoreline", 0.0, -0.05, 0.05, 12, 6),
            _gap("setting_early_scoreline", 0.0, 0.0, 0.0, 80, 16),
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "kappa": 0.0,
                "early_scoreline_log_loss": 3.1,
                "early_matches": 80,
                "overall_scoreline_log_loss": 3.06,
                "overall_matches": 400,
                "early_result_log_loss": 1.05,
                "overall_result_log_loss": 1.02,
                "chosen": True,
            }
        ]
    )
    result = ThinPriorResult(
        chosen_kappa=0.0,
        dropped=True,
        reason="early-match scoreline log loss is not lower on the later seasons",
        candidates=candidates,
        paired=paired,
        matches=pd.DataFrame(),
        elapsed_seconds=90.0,
        run_date="25 Nov 2026",
    )
    text = render_thin_prior_note(result)
    assert "published fit stays the previous model" in text
    assert "No closing home/draw/away line" in text


def _gap(name, mean, low, high, matches, rounds):
    return {
        "comparison": name,
        "model_a": "thin_prior",
        "model_b": "dixon_coles",
        "mean_diff": mean,
        "ci_low": low,
        "ci_high": high,
        "n_matches": matches,
        "n_rounds": rounds,
        "n_bootstrap": 10,
        "seed": 1,
        "note": "",
        "mean_diff_exact": mean,
    }
