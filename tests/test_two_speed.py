"""One weight between a slow Dixon-Coles fit and a fast Dixon-Coles fit."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import (
    DIXON_COLES_PRIOR_KAPPA,
    DIXON_COLES_XI,
    GRID_CALIBRATION_RHO,
    GRID_CALIBRATION_SCALE,
    MARKET_BLEND_WEIGHT,
    TWO_SPEED_FAST_XI,
    TWO_SPEED_SLOW_XI,
    XI_TUNE_CANDIDATES,
)
from src.eval.book_grid import markets_from_rates
from src.eval.gameweeks import assign_gameweeks
from src.eval.two_speed import (
    TwoSpeedResult,
    choose_weight,
    edge_met,
    line_function_sources,
    mixed_rates,
    range_includes_zero,
    render_two_speed_note,
    run_two_speed,
    stop_decision,
    two_speed_grid,
    weight_name,
)
from src.eval.walkforward import iter_folds
from src.models.dixon_coles import DixonColesModel, expected_goals, scoreline_grid

_TEAMS = ("Arsenal", "Chelsea", "Liverpool", "Spurs")


def test_weight_zero_is_the_slow_rates_and_one_is_the_fast_rates():
    lam, mu = mixed_rates(1.4, 1.1, 2.0, 0.8, 0.0)
    assert (lam, mu) == pytest.approx((1.4, 1.1))
    lam, mu = mixed_rates(1.4, 1.1, 2.0, 0.8, 1.0)
    assert (lam, mu) == pytest.approx((2.0, 0.8))
    lam, mu = mixed_rates(1.4, 1.1, 2.0, 0.8, 0.25)
    assert lam == pytest.approx(0.75 * 1.4 + 0.25 * 2.0)
    assert mu == pytest.approx(0.75 * 1.1 + 0.25 * 0.8)


def test_weight_zero_grid_is_the_slow_fit():
    slow = _model(0.0035, 1.2)
    fast = _model(0.0120, 1.7)
    grid, lam, mu, _rho = two_speed_grid(slow, fast, "Arsenal", "Chelsea", 0.0)
    published = scoreline_grid(slow, "Arsenal", "Chelsea")
    slow_lam, slow_mu = expected_goals(slow, "Arsenal", "Chelsea")
    assert grid == pytest.approx(published)
    assert (lam, mu) == pytest.approx((slow_lam, slow_mu))
    mixed, mixed_lam, mixed_mu, _mixed_rho = two_speed_grid(
        slow, fast, "Arsenal", "Chelsea", 1.0
    )
    fast_lam, fast_mu = expected_goals(fast, "Arsenal", "Chelsea")
    assert (mixed_lam, mixed_mu) == pytest.approx((fast_lam, fast_mu))
    assert abs(mixed - published).max() > 1e-6


def test_line_functions_do_not_name_a_price_column():
    sources = line_function_sources()
    assert set(sources) == {"mixed_rates", "two_speed_grid"}
    for name, source in sources.items():
        for token in ("implied_", "B365", "odds", "book_p_"):
            assert token not in source, f"{name} mentions {token}"


def test_fast_decay_is_already_on_the_list_and_the_published_model_is_unchanged():
    assert TWO_SPEED_SLOW_XI == DIXON_COLES_XI == 0.0035
    assert TWO_SPEED_FAST_XI in XI_TUNE_CANDIDATES
    assert TWO_SPEED_FAST_XI > TWO_SPEED_SLOW_XI
    assert DIXON_COLES_PRIOR_KAPPA == 32.0
    assert MARKET_BLEND_WEIGHT == 0.25
    assert GRID_CALIBRATION_SCALE == 1.0
    assert GRID_CALIBRATION_RHO is None


def test_choose_weight_ignores_later_seasons_and_keeps_the_smaller_tie():
    weights = (0.0, 0.25, 1.0)
    rows = []
    for weight in weights:
        rows.append(
            {
                "season_code": "2223",
                "model": weight_name(weight),
                "scoreline_loss": 3.0 if weight in (0.0, 0.25) else 3.2,
            }
        )
        rows.append(
            {
                "season_code": "2526",
                "model": weight_name(weight),
                "scoreline_loss": 1.0 if weight == 1.0 else 4.0,
            }
        )
    assert choose_weight(pd.DataFrame(rows), weights, ("2223",)) == 0.0


def test_stop_fires_on_a_low_weight_a_short_gain_or_a_range_covering_zero():
    low_weight, reason = stop_decision(0.05, 0.02, 0.01, 0.03)
    assert low_weight
    assert "under 0.10" in reason
    short, reason = stop_decision(0.25, 0.004, 0.001, 0.007)
    assert short
    assert "under 0.005" in reason
    covers, reason = stop_decision(0.25, 0.008, -0.001, 0.02)
    assert covers
    assert "includes zero" in reason
    kept, _reason = stop_decision(0.10, 0.008, 0.001, 0.02)
    assert not kept
    assert range_includes_zero(-0.00004, 0.01)
    assert not range_includes_zero(0.00006, 0.01)


def test_edge_needs_a_lower_loss_and_a_range_above_zero():
    assert edge_met(0.02, 0.01, 0.03)
    assert not edge_met(0.02, -0.001, 0.03)
    assert not edge_met(-0.02, -0.03, -0.01)


def test_a_new_decay_is_rejected():
    frame = _frame()
    with pytest.raises(ValueError, match="already listed"):
        run_two_speed(
            frame,
            weights=(0.0, 1.0),
            fast_xi=0.02,
            setting_seasons=("2223",),
            later_seasons=("2526",),
            min_train_rows=8,
        )


def test_fits_use_earlier_matches_and_ignore_closing_prices(monkeypatch):
    calls: list[tuple[pd.Timestamp, float]] = []

    def fake_fit(matches, xi=None, **_kwargs):
        calls.append((pd.to_datetime(matches["date"]).max(), float(xi)))
        # The slow fit is a poor rate for these scores, so the fast fit is chosen.
        level = 2.5 if abs(float(xi) - 0.0035) < 1e-12 else 0.9
        return _model(float(xi), level)

    monkeypatch.setattr("src.eval.two_speed.fit_dixon_coles", fake_fit)
    frame = _frame()
    folded = assign_gameweeks(frame)
    expected = [
        (train, test, season, gw)
        for train, test, season, gw in iter_folds(folded, min_train_rows=8)
        if season in {"2223", "2526"}
    ]
    first = run_two_speed(
        frame,
        weights=(0.0, 1.0),
        slow_xi=0.0035,
        fast_xi=0.0120,
        setting_seasons=("2223",),
        later_seasons=("2526",),
        min_train_rows=8,
    )
    assert first.chosen_weight == 1.0
    assert len(calls) == 2 * len(expected)
    for index, (train, test, _season, _gw) in enumerate(expected):
        train_max = pd.to_datetime(train["date"]).max()
        assert calls[2 * index][0] == train_max
        assert calls[2 * index + 1][0] == train_max
        assert train_max < pd.to_datetime(test["date"]).min()
        assert calls[2 * index][1] == pytest.approx(0.0035)
        assert calls[2 * index + 1][1] == pytest.approx(0.0120)
    assert set(first.matches["season_code"]) <= {"2223", "2526"}

    calls.clear()
    poisoned = frame.copy()
    other = markets_from_rates(1.05, 1.7)
    poisoned["implied_home"] = other["home"]
    poisoned["implied_draw"] = other["draw"]
    poisoned["implied_away"] = other["away"]
    poisoned["implied_over_25"] = other["over_25"]
    poisoned["implied_under_25"] = other["under_25"]
    second = run_two_speed(
        poisoned,
        weights=(0.0, 1.0),
        slow_xi=0.0035,
        fast_xi=0.0120,
        setting_seasons=("2223",),
        later_seasons=("2526",),
        min_train_rows=8,
    )
    left = first.matches.loc[first.matches["model"] == "two_speed", ["match_id", "scoreline_loss"]]
    right = second.matches.loc[
        second.matches["model"] == "two_speed", ["match_id", "scoreline_loss"]
    ]
    merged = left.merge(right, on="match_id", suffixes=("_a", "_b"))
    assert len(merged) == len(left) > 0
    assert (merged["scoreline_loss_a"] - merged["scoreline_loss_b"]).abs().max() == 0.0
    book_a = first.matches.loc[first.matches["model"] == "book_closing_table", "lambda_home"].iloc[0]
    book_b = second.matches.loc[
        second.matches["model"] == "book_closing_table", "lambda_home"
    ].iloc[0]
    assert book_a != pytest.approx(book_b)


def test_dropped_note_keeps_the_pure_model_and_states_the_expected_band():
    dropped, reason = stop_decision(0.05, 0.004, -0.002, 0.009)
    text = render_two_speed_note(
        _result(0.05, dropped=dropped, edge=False, replaces=False, reason=reason, pure_gain=0.004)
    )
    assert "published model stays the current pure model" in text
    assert "0.000 to 0.015" in text
    assert "not enough, on its own, to clear the reconstructed closing table" in text
    assert "edge definition is not met" in text
    assert "95% range" in text
    assert "490 matches" in text
    assert "49 rounds" in text
    assert "not a claim about the bookmaker" in text
    assert "Kalman" in text
    assert "beat the book" not in text.lower()
    assert "·" not in text


def test_a_kept_line_stays_labelled_until_the_edge_definition_is_met():
    kept, reason = stop_decision(0.25, 0.008, 0.002, 0.014)
    assert not kept
    text = render_two_speed_note(
        _result(0.25, dropped=False, edge=False, replaces=False, reason=reason, pure_gain=0.008)
    )
    assert "labelled alternative" in text
    assert "does not replace the pure model" in text
    assert "edge definition is not met" in text
    assert "beat the book" not in text.lower()

    replaced = render_two_speed_note(
        _result(
            0.25,
            dropped=False,
            edge=True,
            replaces=True,
            reason=reason,
            pure_gain=0.008,
            book_gain=0.02,
            book_low=0.01,
            book_high=0.03,
        )
    )
    assert "replaces the pure model" in replaced
    assert "edge definition is met" in replaced
    assert "beat the book" not in replaced.lower()


def _model(xi: float, attack: float) -> DixonColesModel:
    return DixonColesModel(
        attack={team: attack for team in _TEAMS},
        defence={team: 1.0 for team in _TEAMS},
        home_advantage=1.25,
        rho=-0.04,
        xi=xi,
        prior_kappa=32.0,
    )


def _frame() -> pd.DataFrame:
    prices = markets_from_rates(1.45, 1.15)
    rows = []
    for season, start in (("2223", "2022-08-01"), ("2526", "2025-08-01")):
        dates = pd.date_range(start, periods=20, freq="D")
        for index, date in enumerate(dates):
            rows.append(
                {
                    "season_code": season,
                    "date": date,
                    "home_team": _TEAMS[index % 4],
                    "away_team": _TEAMS[(index + 1) % 4],
                    "home_goals": 1,
                    "away_goals": 0 if index % 2 == 0 else 1,
                    "result": "H" if index % 2 == 0 else "D",
                    "implied_home": prices["home"],
                    "implied_draw": prices["draw"],
                    "implied_away": prices["away"],
                    "implied_over_25": prices["over_25"],
                    "implied_under_25": prices["under_25"],
                }
            )
    return pd.DataFrame(rows)


def _gap(name: str, mean: float, low: float, high: float) -> dict:
    return {
        "comparison": name,
        "model_a": "a",
        "model_b": "b",
        "mean_diff": mean,
        "ci_low": low,
        "ci_high": high,
        "n_matches": 490,
        "n_rounds": 49,
        "n_bootstrap": 4000,
        "seed": 20261124,
        "note": "",
        "mean_diff_exact": mean,
    }


def _result(
    weight: float,
    *,
    dropped: bool,
    edge: bool,
    replaces: bool,
    reason: str,
    pure_gain: float,
    book_gain: float = -0.09,
    book_low: float = -0.16,
    book_high: float = -0.02,
) -> TwoSpeedResult:
    paired = pd.DataFrame(
        [
            _gap("setting_pure_minus_two_speed_scoreline", 0.01, -0.01, 0.03),
            _gap("later_pure_minus_two_speed_scoreline", pure_gain, pure_gain - 0.006, pure_gain + 0.006),
            _gap("later_pure_minus_two_speed_result", 0.001, -0.004, 0.006),
            _gap("later_pure_minus_two_speed_ou", 0.0, -0.003, 0.003),
            _gap("later_book_minus_two_speed_scoreline", book_gain, book_low, book_high),
            _gap("later_book_minus_two_speed_result", -0.02, -0.04, -0.005),
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "weight": weight,
                "scoreline_log_loss": 2.90,
                "matches": 940,
                "result_log_loss": 0.96,
                "chosen": True,
            }
        ]
    )
    return TwoSpeedResult(
        chosen_weight=weight,
        slow_xi=0.0035,
        fast_xi=0.0120,
        dropped=dropped,
        edge=edge,
        replaces_pure=replaces,
        reason=reason,
        candidates=candidates,
        paired=paired,
        matches=pd.DataFrame(),
        elapsed_seconds=90.0,
        run_date="26 Nov 2026",
        skipped_book_matches=0,
    )
