from pathlib import Path

import pandas as pd
import pytest

from src.models.artefacts import (
    build_model_version,
    load_artefact,
    save_artefact,
    teams_in_model,
)
from src.models.dixon_coles import DixonColesModel


def test_artefact_round_trip(tmp_path: Path):
    model = DixonColesModel(
        attack={"Arsenal": 1.1, "Chelsea": 0.9},
        defence={"Arsenal": 0.95, "Chelsea": 1.05},
        home_advantage=1.3,
        rho=-0.05,
        xi=0.0019,
        as_of=pd.Timestamp("2026-11-16"),
    )
    version, path = save_artefact(
        model,
        temperature=1.2,
        hyperparams={"xi": 0.0019, "max_goals": 8},
        models_dir=tmp_path,
    )
    assert path.exists()
    loaded, temperature, hyperparams = load_artefact(version, models_dir=tmp_path)
    assert temperature == pytest.approx(1.2)
    assert loaded.attack["Arsenal"] == pytest.approx(1.1)
    assert "Arsenal" in teams_in_model(loaded)
    assert "Spurs" not in teams_in_model(loaded)
    assert version.startswith("dc-20261116-")
    assert hyperparams["xi"] == pytest.approx(0.0019)
    assert hyperparams["max_goals"] == 8
    assert loaded.defence["Chelsea"] == pytest.approx(1.05)
    assert loaded.home_advantage == pytest.approx(1.3)
    assert loaded.rho == pytest.approx(-0.05)
    assert loaded.xi == pytest.approx(0.0019)
    assert loaded.as_of == pd.Timestamp("2026-11-16")


def _toy_model(**overrides) -> DixonColesModel:
    base = dict(
        attack={"Arsenal": 1.1, "Chelsea": 0.9},
        defence={"Arsenal": 0.95, "Chelsea": 1.05},
        home_advantage=1.3,
        rho=-0.05,
        xi=0.0019,
        as_of=pd.Timestamp("2026-11-16"),
    )
    base.update(overrides)
    return DixonColesModel(**base)


def test_load_artefact_none_uses_latest(tmp_path: Path):
    save_artefact(
        _toy_model(as_of=pd.Timestamp("2026-11-01"), attack={"Arsenal": 1.0}),
        temperature=1.0,
        hyperparams={"tag": "older"},
        models_dir=tmp_path,
    )
    newer, _ = save_artefact(
        _toy_model(as_of=pd.Timestamp("2026-11-16"), attack={"Arsenal": 1.2, "Chelsea": 0.8}),
        temperature=1.4,
        hyperparams={"tag": "newer"},
        models_dir=tmp_path,
    )
    loaded, temperature, hyperparams = load_artefact(None, models_dir=tmp_path)
    assert temperature == pytest.approx(1.4)
    assert hyperparams["tag"] == "newer"
    assert loaded.attack["Arsenal"] == pytest.approx(1.2)
    assert (tmp_path / "latest").read_text(encoding="utf-8").strip() == newer


def test_teams_in_model_uses_attack_keys():
    model = _toy_model(
        attack={"Arsenal": 1.1},
        defence={"Arsenal": 0.95, "Spurs": 1.0},
    )
    assert teams_in_model(model) == {"Arsenal"}
    assert "Spurs" not in teams_in_model(model)


def test_build_model_version_format():
    version = build_model_version(pd.Timestamp("2026-11-16"), "abcdef12deadbeef")
    assert version == "dc-20261116-abcdef12"
