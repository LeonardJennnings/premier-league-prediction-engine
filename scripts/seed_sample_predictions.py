#!/usr/bin/env python3
"""Seed a local predictions.sqlite with sample upcoming + settled rows for Epic 5.

Use when the store is empty so the Astro site can talk to a live API:

    python scripts/seed_sample_predictions.py
    uvicorn src.api.app:app --reload
    cd web && npm run dev

Writes to PREDICTIONS_DB (default data/processed/predictions.sqlite).
Safe to re-run: uses a dedicated model_version and skips duplicate inserts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.store.predictions import (  # noqa: E402
    connect,
    insert_prediction,
    settle_prediction,
    upsert_model_artefact,
)
from src.store.schema import init_db  # noqa: E402

VERSION = "sample-local"
SETTLED_JSON = ROOT / "web" / "src" / "lib" / "sample-settled.json"
# (home, away, days_ahead, hour, model probs, book probs)
UPCOMING_TEAMS = [
    ("Arsenal", "Chelsea", 3, 12, 0.48, 0.26, 0.26, 0.45, 0.28, 0.27),
    ("Liverpool", "Manchester City", 3, 17, 0.36, 0.27, 0.37, 0.34, 0.28, 0.38),
    ("Tottenham", "Brighton", 4, 14, 0.41, 0.29, 0.30, 0.39, 0.30, 0.31),
    ("Newcastle", "Aston Villa", 4, 16, 0.44, 0.28, 0.28, 0.42, 0.29, 0.29),
]


def _row(
    *,
    match_id: int,
    home: str,
    away: str,
    kickoff: str,
    p_home: float,
    p_draw: float,
    p_away: float,
    book_h: float | None,
    book_d: float | None,
    book_a: float | None,
    matchday: int,
    status: str = "scheduled",
) -> dict:
    return {
        "match_id": match_id,
        "model_version": VERSION,
        "kickoff_utc": kickoff,
        "home_team": home,
        "away_team": away,
        "matchday": matchday,
        "competition": "PL",
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts_yes": 0.55,
        "p_btts_no": 0.45,
        "p_over_25": 0.52,
        "p_under_25": 0.48,
        "book_p_home": book_h,
        "book_p_draw": book_d,
        "book_p_away": book_a,
        "book_p_over_25": 0.5,
        "book_p_under_25": 0.5,
        "book_source": "sample close" if book_h is not None else None,
        "published_at": "2026-11-16T12:00:00Z",
        "as_of_date": "2026-11-16",
        "status": status,
    }


def main() -> None:
    now = datetime.now(timezone.utc)
    as_of = now.strftime("%Y-%m-%d")
    published = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    db = init_db()
    conn = connect(db)
    real = conn.execute(
        "SELECT model_version FROM model_artefacts WHERE model_version != ? LIMIT 1",
        (VERSION,),
    ).fetchone()
    if real is not None:
        conn.close()
        print(
            "error: refusing to seed a store that already has a real model version "
            f"({real['model_version']}).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    upsert_model_artefact(
        conn,
        VERSION,
        as_of,
        "fixtures/sample-local",
        {"note": "seed_sample_predictions.py"},
        created_at=published,
    )

    for i, (h, a, days, hour, ph, pd, pa, bh, bd, ba) in enumerate(
        UPCOMING_TEAMS, start=9001
    ):
        kick_dt = (now + timedelta(days=days)).replace(
            hour=hour, minute=30, second=0, microsecond=0
        )
        kick = kick_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        row = _row(
            match_id=i,
            home=h,
            away=a,
            kickoff=kick,
            p_home=ph,
            p_draw=pd,
            p_away=pa,
            book_h=bh,
            book_d=bd,
            book_a=ba,
            matchday=12,
        )
        row["published_at"] = published
        row["as_of_date"] = as_of
        inserted = insert_prediction(conn, row)
        print(f"upcoming {i}: {'inserted' if inserted else 'exists'}")

    settled = json.loads(SETTLED_JSON.read_text())
    for m in settled:
        row = _row(
            match_id=int(m["match_id"]),
            home=m["home_team"],
            away=m["away_team"],
            kickoff=m["kickoff_utc"],
            p_home=float(m["p_home"]),
            p_draw=float(m["p_draw"]),
            p_away=float(m["p_away"]),
            book_h=m.get("book_p_home"),
            book_d=m.get("book_p_draw"),
            book_a=m.get("book_p_away"),
            matchday=int(m.get("matchday") or 10),
            status="scheduled",
        )
        inserted = insert_prediction(conn, row)
        settle_prediction(
            conn,
            int(m["match_id"]),
            VERSION,
            int(m["home_goals"]),
            int(m["away_goals"]),
            str(m["result"]),
        )
        print(f"settled {m['match_id']}: {'inserted+settled' if inserted else 'resealed'}")

    conn.close()
    print(f"Seeded {db} with model_version={VERSION}")
    print("Start API: uvicorn src.api.app:app --reload")


if __name__ == "__main__":
    main()
