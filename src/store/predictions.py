"""Insert-once probabilities. Later writes may change kickoff, status, and scores only."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
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


def connect_readonly(path: Path) -> sqlite3.Connection:
    """Open the prediction DB for reads only (API request path)."""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
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


# Kickoff, lifecycle, and settlement. Never probabilities or book columns.
_STATE_FIELDS = (
    "kickoff_utc",
    "status",
    "home_goals",
    "away_goals",
    "result",
    "settled_at",
)


def _insert_new(conn: sqlite3.Connection, row: dict) -> bool:
    existing = conn.execute(
        "SELECT 1 FROM predictions WHERE match_id = ? AND model_version = ?",
        (row["match_id"], row["model_version"]),
    ).fetchone()
    if existing is not None:
        return False
    columns = list(INSERT_COLUMNS)
    payload = {name: row[name] for name in INSERT_COLUMNS}
    if "detail_json" in row:
        columns.append("detail_json")
        payload["detail_json"] = row["detail_json"]
    cols = ", ".join(columns)
    placeholders = ", ".join(f":{name}" for name in columns)
    conn.execute(
        f"INSERT INTO predictions ({cols}) VALUES ({placeholders})",
        payload,
    )
    return True


def insert_prediction(conn: sqlite3.Connection, row: dict) -> bool:
    inserted = _insert_new(conn, row)
    conn.commit()
    return inserted


def insert_predictions(conn: sqlite3.Connection, rows: list[dict]) -> dict:
    """Insert many rows and commit once.

    Duplicate keys are skipped. If a row fails mid-batch, the new inserts
    from this call are rolled back.
    """
    inserted = 0
    skipped = 0
    try:
        for row in rows:
            if _insert_new(conn, row):
                inserted += 1
            else:
                skipped += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"inserted": inserted, "skipped_duplicate": skipped}


def update_prediction_state(
    conn: sqlite3.Connection, match_id: int, model_version: str, **fields
) -> bool:
    """Update kickoff, status, or settlement columns for one existing row.

    ``p_*`` and ``book_*`` are refused. Omitted fields are left unchanged.
    """
    unknown = set(fields) - set(_STATE_FIELDS)
    if unknown:
        listed = ", ".join(sorted(unknown))
        raise ValueError(
            "Refusing to update fields outside kickoff, status, and settlement: "
            f"{listed}"
        )
    present = [name for name in _STATE_FIELDS if name in fields]
    if not present:
        return False
    assignments = ", ".join(f"{name} = ?" for name in present)
    values = [fields[name] for name in present]
    cur = conn.execute(
        f"UPDATE predictions SET {assignments} WHERE match_id = ? AND model_version = ?",
        (*values, match_id, model_version),
    )
    conn.commit()
    return cur.rowcount > 0


def settle_prediction(
    conn: sqlite3.Connection,
    match_id: int,
    model_version: str,
    home_goals: int,
    away_goals: int,
    result: str,
) -> bool:
    settled_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return update_prediction_state(
        conn,
        match_id,
        model_version,
        home_goals=home_goals,
        away_goals=away_goals,
        result=result,
        settled_at=settled_at,
        status="settled",
    )


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


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso_week_key(kickoff_utc: str) -> str:
    dt = _parse_utc(kickoff_utc)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


# Local demo rows. They must not hide a real fitted version.
SAMPLE_MODEL_VERSION = "sample-local"


def get_latest_model_version(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT model_version FROM model_artefacts
        WHERE model_version != ?
        ORDER BY created_at DESC, model_version DESC
        LIMIT 1
        """,
        (SAMPLE_MODEL_VERSION,),
    ).fetchone()
    if row is None:
        row = conn.execute(
            """
            SELECT model_version FROM model_artefacts
            ORDER BY created_at DESC, model_version DESC
            LIMIT 1
            """
        ).fetchone()
    return None if row is None else row["model_version"]


def list_public_predictions(conn: sqlite3.Connection) -> list[dict]:
    """Earliest published row for each match. A newer fit does not replace it."""
    rows = conn.execute(
        """
        SELECT * FROM predictions
        ORDER BY published_at ASC, model_version ASC, match_id ASC
        """
    ).fetchall()
    chosen: dict[int, dict] = {}
    for row in rows:
        match_id = int(row["match_id"])
        if match_id not in chosen:
            chosen[match_id] = dict(row)
    return [chosen[match_id] for match_id in sorted(chosen)]


def get_model_artefact(conn: sqlite3.Connection, model_version: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM model_artefacts WHERE model_version = ?",
        (model_version,),
    ).fetchone()
    return None if row is None else dict(row)


def list_upcoming_predictions(
    conn: sqlite3.Connection,
    model_version: str,
    *,
    now_utc: str,
    window_days: int,
) -> list[dict]:
    now = _parse_utc(now_utc)
    end = now + timedelta(days=window_days)
    rows = conn.execute(
        """
        SELECT * FROM predictions
        WHERE model_version = ?
          AND status IN ('scheduled', 'postponed')
        ORDER BY kickoff_utc, match_id
        """,
        (model_version,),
    ).fetchall()
    out = []
    for row in rows:
        kick = _parse_utc(row["kickoff_utc"])
        if now <= kick < end:
            out.append(dict(row))
    return out


def list_public_upcoming(
    conn: sqlite3.Connection, *, now_utc: str, window_days: int
) -> list[dict]:
    now = _parse_utc(now_utc)
    end = now + timedelta(days=window_days)
    out = []
    for row in list_public_predictions(conn):
        if row["status"] not in ("scheduled", "postponed"):
            continue
        kick = _parse_utc(row["kickoff_utc"])
        if now <= kick < end:
            out.append(row)
    out.sort(key=lambda row: (row["kickoff_utc"], row["match_id"]))
    return out


def list_public_settled_weeks(conn: sqlite3.Connection) -> list[dict]:
    counts: dict[str, int] = {}
    for row in list_public_predictions(conn):
        if row["status"] != "settled":
            continue
        key = _iso_week_key(row["kickoff_utc"])
        counts[key] = counts.get(key, 0) + 1
    return [
        {"iso_week": week, "match_count": counts[week]}
        for week in sorted(counts.keys(), reverse=True)
    ]


def list_public_settled_for_week(conn: sqlite3.Connection, iso_week: str) -> list[dict]:
    rows = [
        row
        for row in list_public_predictions(conn)
        if row["status"] == "settled" and _iso_week_key(row["kickoff_utc"]) == iso_week
    ]
    rows.sort(key=lambda row: (row["kickoff_utc"], row["match_id"]))
    return rows


def list_settled_weeks(conn: sqlite3.Connection, model_version: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT kickoff_utc FROM predictions
        WHERE model_version = ? AND status = 'settled'
        """,
        (model_version,),
    ).fetchall()
    counts: dict[str, int] = {}
    for row in rows:
        key = _iso_week_key(row["kickoff_utc"])
        counts[key] = counts.get(key, 0) + 1
    return [
        {"iso_week": week, "match_count": counts[week]}
        for week in sorted(counts.keys(), reverse=True)
    ]


def list_settled_predictions_for_week(
    conn: sqlite3.Connection, model_version: str, iso_week: str
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT * FROM predictions
        WHERE model_version = ? AND status = 'settled'
        ORDER BY kickoff_utc, match_id
        """,
        (model_version,),
    ).fetchall()
    return [dict(r) for r in rows if _iso_week_key(r["kickoff_utc"]) == iso_week]
