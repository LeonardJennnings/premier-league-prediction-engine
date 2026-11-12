"""Run the project steps in order."""

from src.config import ROOT, DATA_RAW, SEASON_CODES

def main() -> None:
    print("project root:", ROOT)
    print("Raw data folder:", DATA_RAW)
    print("Seasons:", ", ".join(SEASON_CODES))

if __name__ == "__main__":
    main()
