import pytest

from src.teams.mapping import load_team_name_map, resolve_team_name


def test_resolve_known_alias():
    mapping = load_team_name_map()
    assert resolve_team_name("Manchester United FC", mapping) == "Man United"
    assert resolve_team_name("Manchester City FC", mapping) == "Man City"
    assert resolve_team_name("Nottingham Forest FC", mapping) == "Nott'm Forest"
    # 2026/27 PL promoted sides (football-data.org long names)
    assert resolve_team_name("Coventry City FC", mapping) == "Coventry"
    assert resolve_team_name("Hull City AFC", mapping) == "Hull"


def test_unknown_name_fails_closed():
    mapping = {"Arsenal FC": "Arsenal"}
    with pytest.raises(KeyError, match="Unmapped team"):
        resolve_team_name("Unknown United FC", mapping)
