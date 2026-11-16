"""Fit a Dixon–Coles artefact, then publish the slate for that version."""

from __future__ import annotations

import logging

from src.jobs.fit import fit_and_save, load_training_matches
from src.jobs.publish import publish_slate
from src.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def precompute() -> dict:
    """Fit on the historical match table, then publish the artefact just saved."""
    matches = load_training_matches()
    version, path = fit_and_save(matches)
    counts = publish_slate(version)
    logger.info("Precompute fitted %s -> %s", version, path)
    return {"model_version": version, "artefact_path": str(path), **counts}


def main() -> None:
    setup_logging()
    result = precompute()
    logger.info(
        "Precompute %s fetched=%s in_window=%s inserted=%s skipped_duplicate=%s",
        result["model_version"],
        result["fetched"],
        result["in_window"],
        result["inserted"],
        result["skipped_duplicate"],
    )


if __name__ == "__main__":
    main()
