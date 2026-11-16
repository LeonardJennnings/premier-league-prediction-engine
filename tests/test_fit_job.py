from pathlib import Path

import pandas as pd
import pytest

from src.jobs.fit import fit_and_save, load_training_matches, main
from src.models.artefacts import load_artefact
from src.store.predictions import connect


def _tiny_matches() -> pd.DataFrame:
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City"]
    rows = []
    start = pd.Timestamp("2024-01-01")
    for i in range(40):
        home = teams[i % 4]
        away = teams[(i + 1) % 4]
        rows.append(
            {
                "date": start + pd.Timedelta(days=i),
                "home_team": home,
                "away_team": away,
                "home_goals": i % 3,
                "away_goals": (i + 1) % 3,
                "result": "H" if i % 3 > (i + 1) % 3 else ("A" if i % 3 < (i + 1) % 3 else "D"),
            }
        )
    return pd.DataFrame(rows)


def _patch_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.jobs.fit.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("src.jobs.fit.PREDICTIONS_DB", tmp_path / "predictions.sqlite")


def test_fit_and_save_writes_artefact(tmp_path: Path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    version, path = fit_and_save(_tiny_matches(), as_of=pd.Timestamp("2024-02-20"))
    assert path.exists()
    assert version.startswith("dc-")


def test_fit_and_save_registers_model_artefact(tmp_path: Path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    version, path = fit_and_save(_tiny_matches(), as_of=pd.Timestamp("2024-02-20"))
    conn = connect(tmp_path / "predictions.sqlite")
    row = conn.execute(
        "SELECT * FROM model_artefacts WHERE model_version = ?",
        (version,),
    ).fetchone()
    conn.close()
    assert row is not None
    assert row["as_of_date"] == "2024-02-20"
    assert Path(row["artefact_path"]) == path
    _, temperature, hyperparams = load_artefact(version, models_dir=tmp_path / "models")
    assert temperature == pytest.approx(1.0)
    assert hyperparams["temperature"] == pytest.approx(1.0)


def test_skips_temperature_when_prior_rows_are_short(tmp_path: Path, monkeypatch):
    """Last N matches are a holdout only when at least N strictly earlier rows exist."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("src.jobs.fit.CALIBRATION_MIN_ROWS", 30)

    def boom(*args, **kwargs):
        raise AssertionError("fit_temperature should not run without enough prior rows")

    monkeypatch.setattr("src.jobs.fit.fit_temperature", boom)
    version, _ = fit_and_save(_tiny_matches(), as_of=pd.Timestamp("2024-02-20"))
    _, temperature, _ = load_artefact(version, models_dir=tmp_path / "models")
    assert temperature == pytest.approx(1.0)


def test_holdout_temperature_is_stored(tmp_path: Path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("src.jobs.fit.CALIBRATION_MIN_ROWS", 10)
    seen = {}

    def fake_fit(y_true, P):
        seen["n"] = len(list(y_true))
        seen["rows"] = len(P)
        return 1.37

    monkeypatch.setattr("src.jobs.fit.fit_temperature", fake_fit)
    version, _ = fit_and_save(_tiny_matches(), as_of=pd.Timestamp("2024-02-20"))
    _, temperature, hyperparams = load_artefact(version, models_dir=tmp_path / "models")
    assert seen["n"] == 10
    assert seen["rows"] == 10
    assert temperature == pytest.approx(1.37)
    assert hyperparams["temperature"] == pytest.approx(1.37)


def test_load_training_matches_prefers_processed_csv(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "matches.csv"
    _tiny_matches().to_csv(csv_path, index=False)
    monkeypatch.setattr("src.jobs.fit.DATA_PROCESSED", tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("processed CSV should win over a rebuild")

    monkeypatch.setattr("src.clean.build_matches_table", boom)
    frame = load_training_matches()
    assert len(frame) == 40
    assert set(frame["home_team"]) == {"Arsenal", "Chelsea", "Liverpool", "Man City"}


def test_main_logs_version_path(tmp_path: Path, monkeypatch, caplog):
    csv_path = tmp_path / "matches.csv"
    _tiny_matches().to_csv(csv_path, index=False)
    monkeypatch.setattr("src.jobs.fit.DATA_PROCESSED", tmp_path)
    artefact = tmp_path / "models" / "dc-test"
    monkeypatch.setattr(
        "src.jobs.fit.fit_and_save",
        lambda matches, as_of=None, models_dir=None, db_path=None: ("dc-test", artefact),
    )
    with caplog.at_level("INFO", logger="src.jobs.fit"):
        main()
    assert "dc-test" in caplog.text
    assert str(artefact) in caplog.text
