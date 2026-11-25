"""Build one prediction line from the score table. Markets are sums of that table."""

from __future__ import annotations

from src.models.dixon_coles import DixonColesModel, score_table
from src.models.markets import TOTAL_LINES, markets_from_score_table, ranked_scores

FORECAST_LINE_MODEL = "model"


def league_average_teams(model: DixonColesModel, home: str, away: str) -> list[str]:
    """Mapped clubs with no fitted attack or defence. Rates fall back to 1.0."""
    flagged: list[str] = []
    for team in (home, away):
        if team not in model.attack or team not in model.defence:
            flagged.append(team)
    return flagged


def build_prediction_line(
    model: DixonColesModel,
    fixture: dict,
    home: str,
    away: str,
    *,
    model_version: str,
    artefact_version: str,
    published_at: str,
    as_of_date: str,
    forecast_line: str = FORECAST_LINE_MODEL,
) -> dict:
    """One public row. Home/draw/away are the plain sum of the stored table."""
    table = score_table(model, home, away)
    markets = markets_from_score_table(table)
    ranked = ranked_scores(table)
    if len(ranked) < 3:
        raise ValueError("score table does not have three scores")
    flagged = league_average_teams(model, home, away)
    totals = {
        f"{line:.1f}": {
            "over": markets[f"over_{str(line).replace('.', '')}"],
            "under": markets[f"under_{str(line).replace('.', '')}"],
        }
        for line in TOTAL_LINES
    }
    return {
        "match_id": int(fixture["match_id"]),
        "matchday": fixture.get("matchday"),
        "kickoff_utc": fixture["kickoff_utc"],
        "home_team": home,
        "away_team": away,
        "competition": fixture.get("competition") or "PL",
        "forecast_line": forecast_line,
        "model_version": model_version,
        "artefact_version": artefact_version,
        "published_at": published_at,
        "as_of_date": as_of_date,
        "lambda_home": table["lambda_home"],
        "lambda_away": table["lambda_away"],
        "rho": table["rho"],
        "score_table": {
            "max_goals": table["max_goals"],
            "cells": table["cells"],
            "other": table["other"],
        },
        "most_likely_score": ranked[0],
        "next_scores": ranked[1:3],
        "p_home": markets["home"],
        "p_draw": markets["draw"],
        "p_away": markets["away"],
        "p_btts_yes": markets["btts_yes"],
        "p_btts_no": markets["btts_no"],
        "p_over_25": markets["over_25"],
        "p_under_25": markets["under_25"],
        "totals": totals,
        "league_average_teams": flagged,
        "uses_league_average": bool(flagged),
        "score_omitted_reason": None,
    }
