from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from src.api import service
from src.api.deps import get_db
from src.api.schemas import UpcomingOut, WeekOut, WeeksOut

router = APIRouter(prefix="/predictions", tags=["predictions"])

IsoWeek = Annotated[str, Path(pattern=r"^\d{4}-W\d{2}$")]


@router.get("/upcoming", response_model=UpcomingOut)
def upcoming(model_version: str | None = Query(default=None), conn=Depends(get_db)):
    try:
        return service.get_upcoming(conn, model_version)
    except service.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/weeks", response_model=WeeksOut)
def weeks(model_version: str | None = Query(default=None), conn=Depends(get_db)):
    try:
        return service.list_weeks(conn, model_version)
    except service.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/weeks/{iso_week}", response_model=WeekOut)
def week(
    iso_week: IsoWeek,
    model_version: str | None = Query(default=None),
    conn=Depends(get_db),
):
    try:
        return service.get_week(conn, iso_week, model_version)
    except service.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
