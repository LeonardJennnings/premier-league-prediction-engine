"""Simple scikit-learn baseline model for comparison."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression


FEATURE_COLS = ["implied_home", "implied_draw", "implied_away"]


@dataclass
class BaselineModel:
    """Fitted logistic regression over pre-match features."""

    model: LogisticRegression
    feature_cols: list[str]


def fit_baseline(matches: pd.DataFrame) -> BaselineModel:
    """Train a 3-class logistic regression on implied odds features."""
    df = matches.dropna(subset=FEATURE_COLS + ["result"]).copy()
    X = df[FEATURE_COLS]
    y = df["result"]  # H / D / A

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return BaselineModel(model=clf, feature_cols=FEATURE_COLS)


def predict_match_features(model: BaselineModel, features: pd.Series | dict) -> dict[str, float]:
    """Return home/draw/away probabilities from one feature row."""
    if isinstance(features, dict):
        features = pd.Series(features)

    X = features[model.feature_cols].to_frame().T
    proba = model.model.predict_proba(X)[0]
    # Map sklearn's class order back to our keys
    raw = {cls: float(p) for cls, p in zip(model.model.classes_, proba)}
    return {
        "home": raw.get("H", 0.0),
        "draw": raw.get("D", 0.0),
        "away": raw.get("A", 0.0),
    }