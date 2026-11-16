from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config import PUBLISH_WINDOW_DAYS
from src.store import predictions as store


class NotFoundError(Exception):
    pass


def resolve_model_version(conn, model_version: str | None) -> str:
    if model_version is not None:
        if store.get_model_artefact(conn, model_version) is None:
            raise NotFoundError(f"model_version not found: {model_version}")
        return model_version
    latest = store.get_latest_model_version(conn)
    if latest is None:
        raise NotFoundError("no model artefacts published")
    return latest


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_upcoming(conn, model_version: str | None, *, now_utc: str | None = None) -> dict:
    version = resolve_model_version(conn, model_version)
    rows = store.list_upcoming_predictions(
        conn,
        version,
        now_utc=now_utc or _now_utc_iso(),
        window_days=PUBLISH_WINDOW_DAYS,
    )
    return {"model_version": version, "matches": rows}


def list_weeks(conn, model_version: str | None) -> dict:
    version = resolve_model_version(conn, model_version)
    return {"model_version": version, "weeks": store.list_settled_weeks(conn, version)}


def get_week(conn, iso_week: str, model_version: str | None) -> dict:
    version = resolve_model_version(conn, model_version)
    rows = store.list_settled_predictions_for_week(conn, version, iso_week)
    if not rows:
        raise NotFoundError(f"no settled matches for iso_week={iso_week}")
    return {"model_version": version, "iso_week": iso_week, "matches": rows}


def _artefact_payload(row: dict) -> dict:
    raw = row.get("hyperparams")
    if raw is None or raw == "":
        hyper = None
    elif isinstance(raw, str):
        hyper = json.loads(raw)
    else:
        hyper = raw
    return {
        "model_version": row["model_version"],
        "as_of_date": row["as_of_date"],
        "created_at": row["created_at"],
        "artefact_path": row["artefact_path"],
        "hyperparams": hyper,
    }


def get_latest_model(conn) -> dict:
    version = resolve_model_version(conn, None)
    row = store.get_model_artefact(conn, version)
    assert row is not None
    return _artefact_payload(row)


def get_model(conn, model_version: str) -> dict:
    row = store.get_model_artefact(conn, model_version)
    if row is None:
        raise NotFoundError(f"model_version not found: {model_version}")
    return _artefact_payload(row)
