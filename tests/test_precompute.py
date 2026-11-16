"""precompute fits an artefact, then publishes that version."""

from pathlib import Path

from src.jobs.precompute import main


def test_precompute_main_fits_then_publishes_that_version(monkeypatch, caplog):
    order = []

    def fake_load():
        order.append("load")
        return "matches"

    def fake_fit(matches, as_of=None, models_dir=None, db_path=None):
        order.append(("fit", matches))
        return "dc-20261116-abcd1234", Path("models/dc-20261116-abcd1234")

    def fake_publish(model_version=None, **kwargs):
        order.append(("publish", model_version))
        return {
            "fetched": 4,
            "in_window": 2,
            "inserted": 2,
            "skipped_duplicate": 0,
        }

    monkeypatch.setattr("src.jobs.precompute.load_training_matches", fake_load)
    monkeypatch.setattr("src.jobs.precompute.fit_and_save", fake_fit)
    monkeypatch.setattr("src.jobs.precompute.publish_slate", fake_publish)

    with caplog.at_level("INFO", logger="src.jobs.precompute"):
        main()

    assert order == [
        "load",
        ("fit", "matches"),
        ("publish", "dc-20261116-abcd1234"),
    ]
    assert "inserted=2" in caplog.text
    assert "dc-20261116-abcd1234" in caplog.text
