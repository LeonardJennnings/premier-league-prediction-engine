"""Batch Dixon–Coles predictions for historical or upcoming matches."""

from __future__ import annotations

import pandas as pd

from src.eval.calibration import apply_temperature
from src.models.dixon_coles import DixonColesModel, predict_match

_PROB_COLS = (
    "p_home",
    "p_draw",
    "p_away",
    "p_btts_yes",
    "p_btts_no",
    "p_over_25",
    "p_under_25",
)


def predict_matches(
    model: DixonColesModel,
    matches: pd.DataFrame,
    temperature: float | None = None,
) -> pd.DataFrame:
    """Return 1X2, BTTS, and O/U 2.5 for each row.

    If ``temperature`` is set, it is applied to home/draw/away only. BTTS and
    over/under stay on the raw scoreline grid (calibration is 1X2-only in v1).
    ``match_id`` is copied through when present.
    """
    required = ["home_team", "away_team"]
    missing = [column for column in required if column not in matches.columns]
    if missing:
        raise ValueError(f"matches missing columns: {missing}")

    keep_match_id = "match_id" in matches.columns
    columns = (["match_id"] if keep_match_id else []) + [
        "home_team",
        "away_team",
        *_PROB_COLS,
    ]
    if matches.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for row in matches.itertuples(index=False):
        markets = predict_match(model, row.home_team, row.away_team)
        rec: dict[str, object] = {
            "home_team": row.home_team,
            "away_team": row.away_team,
            "p_home": markets["home"],
            "p_draw": markets["draw"],
            "p_away": markets["away"],
            "p_btts_yes": markets["btts_yes"],
            "p_btts_no": markets["btts_no"],
            "p_over_25": markets["over_25"],
            "p_under_25": markets["under_25"],
        }
        if keep_match_id:
            rec["match_id"] = row.match_id
        rows.append(rec)

    out = pd.DataFrame(rows, columns=columns)
    if temperature is not None:
        scaled = apply_temperature(
            out[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float),
            temperature,
        )
        out["p_home"] = scaled[:, 0]
        out["p_draw"] = scaled[:, 1]
        out["p_away"] = scaled[:, 2]
    return out
