"""SQLite schema for the append-only prediction store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import PREDICTIONS_DB

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS team_name_map (
  fd_org_name TEXT PRIMARY KEY,
  canonical_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_artefacts (
  model_version TEXT PRIMARY KEY,
  as_of_date TEXT NOT NULL,
  created_at TEXT NOT NULL,
  artefact_path TEXT NOT NULL,
  hyperparams TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
  match_id INTEGER NOT NULL,
  model_version TEXT NOT NULL,
  kickoff_utc TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  matchday INTEGER,
  competition TEXT NOT NULL DEFAULT 'PL',
  p_home REAL NOT NULL,
  p_draw REAL NOT NULL,
  p_away REAL NOT NULL,
  p_btts_yes REAL NOT NULL,
  p_btts_no REAL NOT NULL,
  p_over_25 REAL NOT NULL,
  p_under_25 REAL NOT NULL,
  book_p_home REAL,
  book_p_draw REAL,
  book_p_away REAL,
  book_p_over_25 REAL,
  book_p_under_25 REAL,
  book_source TEXT,
  published_at TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  -- scheduled | postponed | cancelled | settled
  -- postponed: football-data.org POSTPONED or SUSPENDED
  -- cancelled: football-data.org CANCELLED
  -- settled: full-time score stored; a later correction may replace goals
  status TEXT NOT NULL,
  home_goals INTEGER,
  away_goals INTEGER,
  result TEXT,
  settled_at TEXT,
  PRIMARY KEY (match_id, model_version)
);
"""


def init_db(path: Path | None = None) -> Path:
    db_path = PREDICTIONS_DB if path is None else path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return db_path
