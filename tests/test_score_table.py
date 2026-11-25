"""Score table: markets are cell sums, and a 9-0 result stays finite."""

import math

import pytest

from src.models.dixon_coles import score_table_from_rates
from src.models.markets import markets_from_score_table, scoreline_log_loss


def _hand_table():
    cells = [
        [0.10, 0.20, 0.00],
        [0.30, 0.15, 0.05],
        [0.00, 0.10, 0.05],
    ]
    return {"max_goals": 2, "cells": cells, "other": 0.05}


def test_stored_markets_equal_cell_sums():
    table = _hand_table()
    markets = markets_from_score_table(table)
    cells = table["cells"]
    other = table["other"]
    home = draw = away = 0.0
    btts_yes = 0.0
    over_25 = other
    under_25 = 0.0
    for i, row in enumerate(cells):
        for j, probability in enumerate(row):
            if i > j:
                home += probability
            elif i == j:
                draw += probability
            else:
                away += probability
            if i >= 1 and j >= 1:
                btts_yes += probability
            if i + j >= 3:
                over_25 += probability
            else:
                under_25 += probability
    assert markets["home"] == pytest.approx(home)
    assert markets["draw"] == pytest.approx(draw)
    assert markets["away"] == pytest.approx(away)
    assert markets["btts_yes"] == pytest.approx(btts_yes)
    assert markets["over_25"] == pytest.approx(over_25)
    assert markets["under_25"] == pytest.approx(under_25)
    assert home == pytest.approx(0.40)
    assert draw == pytest.approx(0.30)
    assert away == pytest.approx(0.25)
    assert over_25 == pytest.approx(0.25)
    assert under_25 == pytest.approx(0.75)
    # The other bucket is always over 4.5 on this small grid.
    assert markets["over_45"] == pytest.approx(other)
    assert markets["under_45"] == pytest.approx(0.95)


def test_nine_nil_uses_the_other_bucket():
    table = score_table_from_rates(1.35, 1.15, -0.05, max_goals=8)
    cell_mass = sum(sum(row) for row in table["cells"])
    assert table["other"] > 0
    assert cell_mass + table["other"] == pytest.approx(1.0)
    loss = scoreline_log_loss(table, 9, 0)
    assert math.isfinite(loss)
    assert loss == pytest.approx(-math.log(table["other"]))
    inside = scoreline_log_loss(table, 1, 0)
    assert math.isfinite(inside)
    assert inside != pytest.approx(loss)
