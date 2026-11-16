"""Insert-once predictions and settlement that never rewrites probabilities."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

INSERT_COLUMNS = (
    "match_id",
    "model_version",
    "kickoff_utc",
    "home_team",
    "away_team",
    "matchday",
    "competition",
    "p_home",
    "p_draw",
    "p_away",
    "p_btts_yes",
    "p_btts_no",
    "p_over_25",
    "p_under_25",
    "book_p_home",
    "book_p_draw",
    "book_p_away",
    "book_p_over_25",
    "book_p_under_25",
    "book_source",
    "published_at",
    "as_of_date",
    "status",
)


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def get_prediction(
    conn: sqlite3.Connection, match_id: int, model_version: str
) -> dict | None:
    row = conn.execute(
        "SELECT * FROM predictions WHERE match_id = ? AND model_version = ?",
        (match_id, model_version),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def insert_prediction(conn: sqlite3.Connection, row: dict) -> bool:
    existing = conn.execute(
        "SELECT 1 FROM predictions WHERE match_id = ? AND model_version = ?",
        (row["match_id"], row["model_version"]),
    ).fetchone()
    if existing is not None:
        return False
    cols = ", ".join(INSERT_COLUMNS)
    placeholders = ", ".join(f":{name}" for name in INSERT_COLUMNS)
    conn.execute(
        f"INSERT INTO predictions ({cols}) VALUES ({placeholders})",
        {name: row[name] for name in INSERT_COLUMNS},
    )
    conn.commit()
    return True


def settle_prediction(
    conn: sqlite3.Connection,
    match_id: int,
    model_version: str,
    home_goals: int,
    away_goals: int,
    result: str,
) -> bool:
    settled_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.execute(
        """
        UPDATE predictions
        SET home_goals = ?, away_goals = ?, result = ?, settled_at = ?, status = 'settled'
        WHERE match_id = ? AND model_version = ?
        """,
        (home_goals, away_goals, result, settled_at, match_id, model_version),
    )
    conn.commit()
    return cur.rowcount > 0


def upsert_model_artefact(
    conn: sqlite3.Connection,
    model_version: str,
    as_of_date: str,
    artefact_path: str,
    hyperparams: dict | str | None = None,
    created_at: str | None = None,
) -> None:
    """Insert or replace one ``model_artefacts`` row for this version."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(hyperparams, str) or hyperparams is None:
        payload = hyperparams
    else:
        payload = json.dumps(hyperparams, sort_keys=True)
    conn.execute(
        """
        INSERT INTO model_artefacts (
            model_version, as_of_date, created_at, artefact_path, hyperparams
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(model_version) DO UPDATE SET
            as_of_date = excluded.as_of_date,
            created_at = excluded.created_at,
            artefact_path = excluded.artefact_path,
            hyperparams = excluded.hyperparams
        """,
        (model_version, as_of_date, created_at, artefact_path, payload),
    )
    conn.commit()
