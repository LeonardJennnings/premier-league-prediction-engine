from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from fastapi import HTTPException, Request

from src.config import PREDICTIONS_DB
from src.store.predictions import connect


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    path: Path = getattr(request.app.state, "db_path", None) or PREDICTIONS_DB
    if not path.is_file():
        raise HTTPException(status_code=503, detail="predictions database unavailable")
    try:
        conn = connect(path)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="predictions database unavailable") from exc
    try:
        yield conn
    finally:
        conn.close()
