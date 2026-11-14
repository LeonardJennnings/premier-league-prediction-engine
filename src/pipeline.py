"""Run the project steps in order."""

from src.clean import build_matches_table, save_matches
from src.download import download_all
from src.features import add_features


def main() -> None:
    print("Downloading raw CSVs (skips files you already have)...")
    download_all()

    print("Building cleaned matches table...")
    matches = build_matches_table()

    print("Adding features...")
    matches = add_features(matches)

    path = save_matches(matches)
    print(f"Saved {len(matches)} matches to {path}")
    print("Columns:", ", ".join(matches.columns.astype(str)))


if __name__ == "__main__":
    main()