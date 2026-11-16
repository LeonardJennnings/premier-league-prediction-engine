"""Project paths and settings used by the rest of the code."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# This file lives in src/. parents[1] is the project root (one level up).
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

# Load local secrets/settings if present (.env is gitignored).
load_dotenv(ROOT / ".env")

SEASON_CODES = ["2223", "2324", "2425", "2526", "2627"]
DIVISION = "E0"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Dixon–Coles: ξ in per-day units (~1 year half-life ≈ ln(2)/365)
DIXON_COLES_XI = 0.0019
DIXON_COLES_MAX_GOALS = 8
DIXON_COLES_L2 = 0.01
CALIBRATION_MIN_ROWS = 200
ELO_K = 20.0
ELO_HOME_BONUS = 60.0
ELO_INITIAL = 1500.0

# Optional future API token for football-data.org fixtures (never commit real values).
FOOTBALL_DATA_ORG_TOKEN = os.getenv("FOOTBALL_DATA_ORG_TOKEN", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

PREDICTIONS_DB = DATA_PROCESSED / "predictions.sqlite"
# Comma-separated browser origins for the read-only API (local Next.js by default).
API_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("API_CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
MODELS_DIR = ROOT / "models"
FIXTURES_CACHE_DIR = DATA_RAW / "fixtures"
# Upcoming odds list from football-data.co.uk (not the football-data.org JSON cache).
FD_UK_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FD_UK_FIXTURES_CSV = DATA_RAW / "fixtures.csv"
TEAM_NAME_MAP_PATH = ROOT / "data" / "reference" / "team_name_map.json"
PUBLISH_WINDOW_DAYS = 10
FD_ORG_BASE = "https://api.football-data.org/v4"


# Build each season's download URL and local raw-file path from shared settings.
def season_url(season_code: str) -> str:
    """Return the football-data.co.uk CSV URL for one season."""
    return f"{BASE_URL}/{season_code}/{DIVISION}.csv"


def raw_path(season_code: str) -> Path:
    """Return where that season's CSV should live under data/raw/."""
    return DATA_RAW / f"{DIVISION}_{season_code}.csv"
