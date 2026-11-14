"""Project paths and settings used by the rest of the code."""

from pathlib import Path

# This file lives in src/. parents[1] is the project root (one level up).
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

SEASON_CODES = ["2223", "2324", "2425", "2526", "2627"]
DIVISION = "E0"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Build each season's download URL and local raw-file path from shared settings.
def season_url(season_code: str) -> str:
    """Return the football-data.co.uk CSV URL for one season."""
    return f"{BASE_URL}/{season_code}/{DIVISION}.csv"

def raw_path(season_code: str) -> Path:
    """Return where that season's CSV should live under data/raw/."""
    return DATA_RAW / f"{DIVISION}_{season_code}.csv"    