"""Download Premier League CSVs from football-data.co.uk."""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from src.config import DATA_RAW, SEASON_CODES, raw_path, season_url

logger = logging.getLogger(__name__)


def download_season(season_code: str, force: bool = False) -> Path:
    """Download one season CSV into data/raw/, or skip if it already exists."""
    url = season_url(season_code)
    dest = raw_path(season_code)

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        logger.info("Skipping %s: already have %s", season_code, dest.name)
        return dest

    logger.info("Downloading %s from %s", season_code, url)
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to download season %s from %s", season_code, url)
        raise

    dest.write_bytes(response.content)
    logger.info("Saved %s (%s bytes)", dest, dest.stat().st_size)
    return dest


def download_all(season_codes=None, force: bool = False) -> list[Path]:
    """Download every configured season (or a custom list)."""
    if season_codes is None:
        season_codes = SEASON_CODES
    paths = []
    for code in season_codes:
        paths.append(download_season(code, force=force))
    return paths


if __name__ == "__main__":
    from src.logging_setup import setup_logging

    setup_logging()
    download_all()
