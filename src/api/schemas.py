from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MatchOut(BaseModel):
    match_id: int
    home_team: str
    away_team: str
    kickoff_utc: str
    matchday: int | None = None
    competition: str
    status: str
    p_home: float
    p_draw: float
    p_away: float
    p_btts_yes: float
    p_btts_no: float
    p_over_25: float
    p_under_25: float
    book_p_home: float | None = None
    book_p_draw: float | None = None
    book_p_away: float | None = None
    book_p_over_25: float | None = None
    book_p_under_25: float | None = None
    book_source: str | None = None
    published_at: str
    as_of_date: str
    model_version: str
    home_goals: int | None = None
    away_goals: int | None = None
    result: str | None = None
    settled_at: str | None = None


class UpcomingOut(BaseModel):
    model_version: str
    matches: list[MatchOut]


class WeekSummary(BaseModel):
    iso_week: str
    match_count: int


class WeeksOut(BaseModel):
    model_version: str
    weeks: list[WeekSummary]


class WeekOut(BaseModel):
    model_version: str
    iso_week: str
    matches: list[MatchOut]


class ModelOut(BaseModel):
    model_version: str
    as_of_date: str
    created_at: str
    artefact_path: str
    hyperparams: dict[str, Any] | None = None
