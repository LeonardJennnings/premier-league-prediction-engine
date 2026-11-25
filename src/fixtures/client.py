"""football-data.org Premier League fixtures client with on-disk JSON cache."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from src.config import FD_ORG_BASE, FIXTURES_CACHE_DIR, FOOTBALL_DATA_ORG_TOKEN

REQUEST_TIMEOUT = 30


def normalize_match(raw: dict) -> dict:
    home = raw.get("homeTeam") or {}
    away = raw.get("awayTeam") or {}
    full_time = (raw.get("score") or {}).get("fullTime") or {}
    return {
        "match_id": raw["id"],
        "kickoff_utc": raw["utcDate"],
        "home_team": home["name"],
        "away_team": away["name"],
        "matchday": raw.get("matchday"),
        "status": raw.get("status"),
        "home_goals": full_time.get("home"),
        "away_goals": full_time.get("away"),
        # The provider's own update time. Absent on older cached payloads.
        "provider_updated_at": raw.get("lastUpdated"),
    }


def _cache_path(
    cache_dir: Path, date_from: str, date_to: str, status: str | None = None
) -> Path:
    suffix = f"_{status}" if status else ""
    return cache_dir / f"PL_{date_from}_{date_to}{suffix}.json"


def _matches_from_payload(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        raw_matches = payload.get("matches") or []
    elif isinstance(payload, list):
        raw_matches = payload
    else:
        raw_matches = []
    return [normalize_match(match) for match in raw_matches]


def _write_cache_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def backoff_seconds(headers, *, status_code: int = 200) -> float:
    """Seconds to wait before the next fixtures call.

    A missing budget header does not wait. Remaining-minute of 0, or HTTP 429,
    waits for the provider reset and never longer than 60 seconds.
    """
    remaining = _header(headers, "x-requests-available-minute")
    reset = _header(headers, "x-requestcounter-reset")
    try:
        left = int(remaining) if remaining is not None else None
    except ValueError:
        left = None
    try:
        wait = int(reset) if reset is not None else 60
    except ValueError:
        wait = 60
    wait = float(min(max(wait, 0), 60))
    if status_code == 429 or left == 0:
        return wait
    return 0.0


def fetch_pl_matches(
    date_from: str,
    date_to: str,
    *,
    token: str | None = None,
    cache_dir: Path | None = None,
    session=None,
    status: str | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> list[dict]:
    cache_dir = FIXTURES_CACHE_DIR if cache_dir is None else cache_dir
    cache_file = _cache_path(cache_dir, date_from, date_to, status)
    if use_cache and cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return _matches_from_payload(payload)

    auth = FOOTBALL_DATA_ORG_TOKEN if token is None else token
    if not auth:
        raise ValueError(
            "FOOTBALL_DATA_ORG_TOKEN is missing or empty; required to fetch fixtures"
        )

    url = (
        f"{FD_ORG_BASE}/competitions/PL/matches"
        f"?dateFrom={date_from}&dateTo={date_to}"
    )
    if status:
        url += f"&status={status}"
    headers = {"X-Auth-Token": auth}
    getter = session.get if session is not None else requests.get
    response = _get_with_backoff(getter, url, headers)
    payload = response.json()
    if write_cache:
        _write_cache_atomic(cache_file, payload)
    return _matches_from_payload(payload)


def _get_with_backoff(getter, url: str, headers: dict):
    response = getter(url, headers=headers, timeout=REQUEST_TIMEOUT)
    status = getattr(response, "status_code", 200)
    if status == 429:
        delay = backoff_seconds(getattr(response, "headers", None), status_code=429)
        if delay:
            time.sleep(delay)
        response = getter(url, headers=headers, timeout=REQUEST_TIMEOUT)
        status = getattr(response, "status_code", 200)
    response.raise_for_status()
    delay = backoff_seconds(getattr(response, "headers", None), status_code=status)
    if delay:
        time.sleep(delay)
    return response


def _header(headers, name: str) -> str | None:
    if not headers:
        return None
    getter = getattr(headers, "get", None)
    if getter is not None:
        value = getter(name)
        if value is None:
            value = getter(name.upper())
        if value is not None:
            return str(value)
    try:
        pairs = list(headers.items())
    except Exception:
        return None
    for key, value in pairs:
        if str(key).lower() == name and value is not None:
            return str(value)
    return None
