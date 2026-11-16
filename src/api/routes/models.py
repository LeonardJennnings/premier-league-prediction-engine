from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api import service
from src.api.deps import get_db
from src.api.schemas import ModelOut

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/latest", response_model=ModelOut)
def latest(conn=Depends(get_db)):
    try:
        return service.get_latest_model(conn)
    except service.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{model_version}", response_model=ModelOut)
def model(model_version: str, conn=Depends(get_db)):
    try:
        return service.get_model(conn, model_version)
    except service.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
