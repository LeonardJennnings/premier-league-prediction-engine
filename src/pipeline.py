"""Run the project steps in order."""

from src.clean import build_matches_table, save_matches
from src.download import download_all
from src.features import add_features
from src.models.baseline import fit_baseline, predict_match_features
from src.models.poisson import fit_poisson, predict_match


def _fmt(probs: dict[str, float]) -> str:
    return (
        f"home={probs['home']:.3f}  "
        f"draw={probs['draw']:.3f}  "
        f"away={probs['away']:.3f}"
    )


def main() -> None:
    print("Downloading raw CSVs (skips files you already have)...")
    download_all()

    print("Building cleaned matches table...")
    matches = build_matches_table()

    print("Adding features...")
    matches = add_features(matches)

    path = save_matches(matches)
    print(f"Saved {len(matches)} matches to {path}")

    print("Fitting Poisson and baseline models...")
    poisson = fit_poisson(matches)
    baseline = fit_baseline(matches)

    # Demo only: one recent completed-looking row with odds present.
    sample = matches.dropna(
        subset=["implied_home", "implied_draw", "implied_away"]
    ).iloc[-1]

    p_pois = predict_match(poisson, sample["home_team"], sample["away_team"])
    p_base = predict_match_features(baseline, sample)
    p_mkt = {
        "home": float(sample["implied_home"]),
        "draw": float(sample["implied_draw"]),
        "away": float(sample["implied_away"]),
    }

    print(
        f"Sample: {sample['date']}  "
        f"{sample['home_team']} vs {sample['away_team']}  "
        f"(result={sample['result']})"
    )
    print("  market  ", _fmt(p_mkt))
    print("  poisson ", _fmt(p_pois))
    print("  baseline", _fmt(p_base))


if __name__ == "__main__":
    main()