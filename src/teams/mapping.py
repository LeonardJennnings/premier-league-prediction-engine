"""Map football-data.org team names to football-data.co.uk canonical names."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import TEAM_NAME_MAP_PATH


def load_team_name_map(path: Path | None = None) -> dict[str, str]:
    map_path = TEAM_NAME_MAP_PATH if path is None else path
    with map_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"team name map empty or invalid: {map_path}")
    return {str(k): str(v) for k, v in data.items()}


def resolve_team_name(fd_org_name: str, mapping: dict[str, str]) -> str:
    name = fd_org_name.strip()
    if name in mapping:
        return mapping[name]
    raise KeyError(f"Unmapped team: {fd_org_name!r}")


def resolve_fixture_teams(
    home: str, away: str, mapping: dict[str, str]
) -> tuple[str, str]:
    return resolve_team_name(home, mapping), resolve_team_name(away, mapping)
