"""Run the project steps in order."""

from __future__ import annotations

import logging

from src.clean import build_matches_table, save_matches
from src.config import LOG_LEVEL
from src.download import download_all
from src.features import add_features
from src.logging_setup import setup_logging
from src.models.baseline import fit_baseline, predict_match_features
from src.models.poisson import fit_poisson, predict_match

logger = logging.getLogger(__name__)


def _fmt(probs: dict[str, float]) -> str:
    return (
        f"home={probs['home']:.3f}  "
        f"draw={probs['draw']:.3f}  "
        f"away={probs['away']:.3f}"
    )


def main() -> None:
    setup_logging(LOG_LEVEL)
    logger.info("Downloading raw CSVs (skips files you already have)...")
    download_all()

    logger.info("Building cleaned matches table...")
    matches = build_matches_table()

    logger.info("Adding features...")
    matches = add_features(matches)

    path = save_matches(matches)
    logger.info("Saved %s matches to %s", len(matches), path)

    logger.info("Fitting Poisson and baseline models...")
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

    logger.info(
        "Sample: %s  %s vs %s  (result=%s)",
        sample["date"],
        sample["home_team"],
        sample["away_team"],
        sample["result"],
    )
    logger.info("  market   %s", _fmt(p_mkt))
    logger.info("  poisson  %s", _fmt(p_pois))
    logger.info("  baseline %s", _fmt(p_base))


if __name__ == "__main__":
    main()
