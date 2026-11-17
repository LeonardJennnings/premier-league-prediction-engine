import numpy as np
import pandas as pd

from src.eval.calibration import apply_temperature
from src.models.dixon_coles import fit_dixon_coles, predict_match
from src.models.predict import predict_matches


def _toy_matches() -> pd.DataFrame:
    rows = []
    teams = ["Alpha", "Beta", "Gamma"]
    start = pd.Timestamp("2024-01-01")
    for i in range(30):
        h = teams[i % 3]
        a = teams[(i + 1) % 3]
        rows.append(
            {
                "date": start + pd.Timedelta(days=i * 3),
                "home_team": h,
                "away_team": a,
                "home_goals": 1 + (i % 2),
                "away_goals": i % 2,
            }
        )
    return pd.DataFrame(rows)


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "home_team": ["Alpha", "Beta"],
            "away_team": ["Beta", "Gamma"],
        }
    )


def test_predict_matches_raw_matches_single_predict():
    model = fit_dixon_coles(_toy_matches())
    fixtures = _fixtures()
    out = predict_matches(model, fixtures)

    assert list(out["match_id"]) == ["m1", "m2"]
    expected = {
        "match_id",
        "home_team",
        "away_team",
        "p_home",
        "p_draw",
        "p_away",
        "p_btts_yes",
        "p_over_25",
    }
    assert expected <= set(out.columns)
    # Publish path must stay on the scoreline grid — lab totals helpers are absent.
    assert "p_poisson_total_over_25" not in out.columns
    assert "p_totals_logistic_over_25" not in out.columns

    single = predict_match(model, "Alpha", "Beta")
    row = out.iloc[0]
    assert abs(row["p_home"] - single["home"]) < 1e-12
    assert abs(row["p_draw"] - single["draw"]) < 1e-12
    assert abs(row["p_away"] - single["away"]) < 1e-12
    assert abs(row["p_btts_yes"] - single["btts_yes"]) < 1e-12
    assert abs(row["p_over_25"] - single["over_25"]) < 1e-12
    assert abs(row["p_home"] + row["p_draw"] + row["p_away"] - 1.0) < 1e-8


def test_temperature_scales_1x2_only():
    model = fit_dixon_coles(_toy_matches())
    fixtures = _fixtures()
    raw = predict_matches(model, fixtures)
    scaled = predict_matches(model, fixtures, temperature=2.0)

    np.testing.assert_allclose(
        scaled[["p_btts_yes", "p_over_25"]].to_numpy(dtype=float),
        raw[["p_btts_yes", "p_over_25"]].to_numpy(dtype=float),
    )
    expected_1x2 = apply_temperature(
        raw[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float),
        2.0,
    )
    np.testing.assert_allclose(
        scaled[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float),
        expected_1x2,
    )
    assert not np.allclose(
        scaled[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float),
        raw[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        scaled[["p_home", "p_draw", "p_away"]].sum(axis=1).to_numpy(dtype=float),
        np.ones(len(scaled)),
        atol=1e-12,
    )


def test_match_id_omitted_when_absent():
    model = fit_dixon_coles(_toy_matches())
    fixtures = _fixtures().drop(columns=["match_id"])
    out = predict_matches(model, fixtures)
    assert "match_id" not in out.columns
    assert list(out["home_team"]) == ["Alpha", "Beta"]
