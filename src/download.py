"""Download Premier League CSVs from football-data.co.uk."""

from pathlib import Path

import requests

from src.config import DATA_RAW, SEASON_CODES, raw_path, season_url


def download_season(season_code: str, force: bool = False) -> Path:
    """Download one season CSV into data/raw/, or skip if it already exists."""
    url = season_url(season_code)
    dest = raw_path(season_code)

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"Skipping {season_code}: already have {dest.name}")
        return dest

    print(f"Downloading {season_code} from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Saved {dest}")
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
    download_all()