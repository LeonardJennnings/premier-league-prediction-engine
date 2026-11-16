from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.health import router as health_router
from src.api.routes.models import router as models_router
from src.api.routes.predictions import router as predictions_router
from src.config import API_CORS_ORIGINS


def create_app(db_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Premier League predictions")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API_CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )
    if db_path is not None:
        app.state.db_path = db_path
    app.include_router(health_router)
    app.include_router(predictions_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    return app


app = create_app()
