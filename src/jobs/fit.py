"""Fit Dixon–Coles, estimate 1X2 temperature, and register the artefact."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CALIBRATION_MIN_ROWS,
    DATA_PROCESSED,
    DIXON_COLES_L2,
    DIXON_COLES_MAX_GOALS,
    MODELS_DIR,
    PREDICTIONS_DB,
)
from src.eval.calibration import fit_temperature
from src.logging_setup import setup_logging
from src.models.artefacts import save_artefact
from src.models.dixon_coles import fit_dixon_coles, predict_match
from src.store.predictions import connect, upsert_model_artefact
from src.store.schema import init_db

logger = logging.getLogger(__name__)


def fit_and_save(
    matches: pd.DataFrame,
    as_of=None,
    models_dir: Path | None = None,
    db_path: Path | None = None,
) -> tuple[str, Path]:
    """Fit on matches up to ``as_of``, save the artefact, and upsert its store row.

    Temperature is fit on a chronological holdout: the last
    ``CALIBRATION_MIN_ROWS`` matches, predicted by a Dixon–Coles model trained
    only on rows strictly before that window. That holdout is used only when
    at least ``CALIBRATION_MIN_ROWS`` earlier rows exist; otherwise T = 1.
    The saved model is refit on every match up to ``as_of``.
    """
    temperature = _holdout_temperature(matches, as_of)
    model = fit_dixon_coles(matches, as_of=as_of)
    hyperparams = {
        "xi": float(model.xi),
        "l2": float(DIXON_COLES_L2),
        "max_goals": int(DIXON_COLES_MAX_GOALS),
        "temperature": float(temperature),
    }
    root = MODELS_DIR if models_dir is None else Path(models_dir)
    version, path = save_artefact(model, temperature, hyperparams, models_dir=root)

    database = PREDICTIONS_DB if db_path is None else Path(db_path)
    init_db(database)
    conn = connect(database)
    try:
        upsert_model_artefact(
            conn,
            version,
            _as_of_date(as_of, model),
            str(path),
            hyperparams,
        )
    finally:
        conn.close()
    return version, path


def load_training_matches() -> pd.DataFrame:
    """Load ``data/processed/matches.csv`` when present, else rebuild it."""
    path = Path(DATA_PROCESSED) / "matches.csv"
    if path.is_file():
        logger.info("Loading matches from %s", path)
        return pd.read_csv(path)
    logger.info("Processed matches CSV missing; building from raw seasons")
    from src.clean import build_matches_table
    from src.features import add_features

    return add_features(build_matches_table())


def main() -> None:
    setup_logging()
    matches = load_training_matches()
    version, path = fit_and_save(matches)
    logger.info("Fitted %s -> %s", version, path)


def _holdout_temperature(matches: pd.DataFrame, as_of) -> float:
    """Return holdout T, or 1.0 when a proper prior set is not available."""
    frame = _eligible_matches(matches, as_of)
    n = int(CALIBRATION_MIN_ROWS)
    if n < 1 or len(frame) < n:
        logger.info(
            "Temperature 1.0; fewer than %s matches available for a holdout",
            n,
        )
        return 1.0

    holdout = frame.iloc[-n:].copy()
    cutoff = pd.Timestamp(holdout["date"].iloc[0])
    train = frame.loc[frame["date"] < cutoff]
    if len(train) < n:
        logger.info(
            "Temperature 1.0; %s prior rows before %s (need %s)",
            len(train),
            cutoff.date(),
            n,
        )
        return 1.0

    model = fit_dixon_coles(train, as_of=cutoff)
    probs = []
    for row in holdout.itertuples(index=False):
        pred = predict_match(model, row.home_team, row.away_team)
        probs.append([pred["home"], pred["draw"], pred["away"]])
    temperature = float(fit_temperature(_labels(holdout), np.asarray(probs, dtype=float)))
    logger.info(
        "Fitted temperature %.4f on %s holdout rows (%s prior)",
        temperature,
        len(holdout),
        len(train),
    )
    return temperature


def _eligible_matches(matches: pd.DataFrame, as_of) -> pd.DataFrame:
    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    missing = [column for column in required if column not in matches.columns]
    if missing:
        raise ValueError(f"matches missing columns: {missing}")
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=required)
    if as_of is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(as_of)]
    return frame.sort_values(
        ["date", "home_team", "away_team"], kind="mergesort"
    ).reset_index(drop=True)


def _labels(holdout: pd.DataFrame) -> list[str]:
    if "result" in holdout.columns and holdout["result"].notna().all():
        return [str(label) for label in holdout["result"].tolist()]
    labels: list[str] = []
    for home, away in zip(holdout["home_goals"], holdout["away_goals"], strict=True):
        if home > away:
            labels.append("H")
        elif home < away:
            labels.append("A")
        else:
            labels.append("D")
    return labels


def _as_of_date(as_of, model) -> str:
    raw = model.as_of if as_of is None else as_of
    if raw is None:
        return ""
    ts = pd.Timestamp(raw)
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
