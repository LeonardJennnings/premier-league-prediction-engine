"""Streamlit dashboard for the Premier League prediction engine."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when launched via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import DATA_PROCESSED, RESULTS
from src.models.baseline import fit_baseline, predict_match_features
from src.models.poisson import fit_poisson, predict_match

st.set_page_config(page_title="PL Prediction Engine", layout="wide")
st.title("Premier League Prediction Engine")
st.write(
    "A learning demo that shows **model probabilities** next to "
    "**bookmaker implied probabilities**. This is a forecast tool, not betting advice."
)


@st.cache_data
def load_matches() -> pd.DataFrame | None:
    path = DATA_PROCESSED / "matches.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_resource
def load_models(matches: pd.DataFrame):
    """Fit once per cached dataset (demo convenience; Phase 8 still owns fair eval)."""
    return fit_poisson(matches), fit_baseline(matches)


def pct(x: float) -> str:
    return f"{100 * float(x):.1f}%"


matches = load_matches()
if matches is None or matches.empty:
    st.warning(
        "No processed data found. From the project root, with the venv active, run:\n\n"
        "`python -m src.pipeline`"
    )
    st.stop()

poisson_model, baseline_model = load_models(matches)

st.header("Match probabilities")
seasons = sorted(matches["season_code"].unique())
season = st.selectbox("Season", seasons, index=len(seasons) - 1)
season_df = matches[matches["season_code"] == season].copy()
season_df["label"] = (
    season_df["date"].dt.strftime("%Y-%m-%d")
    + " | "
    + season_df["home_team"]
    + " vs "
    + season_df["away_team"]
)
choice = st.selectbox("Match", season_df["label"].tolist())
row = season_df.loc[season_df["label"] == choice].iloc[0]

p_pois = predict_match(poisson_model, row["home_team"], row["away_team"])
p_base = predict_match_features(baseline_model, row)

has_odds = pd.notna(row.get("implied_home")) and pd.notna(row.get("implied_draw")) and pd.notna(
    row.get("implied_away")
)
if has_odds:
    p_mkt = {
        "home": float(row["implied_home"]),
        "draw": float(row["implied_draw"]),
        "away": float(row["implied_away"]),
    }
else:
    p_mkt = None

st.subheader(f"{row['home_team']} vs {row['away_team']}")
st.caption(f"Date: {row['date'].date()} · Result: {row['result']}")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Poisson model**")
    st.write(f"Home: {pct(p_pois['home'])}")
    st.write(f"Draw: {pct(p_pois['draw'])}")
    st.write(f"Away: {pct(p_pois['away'])}")
with col2:
    st.markdown("**Sklearn baseline**")
    st.write(f"Home: {pct(p_base['home'])}")
    st.write(f"Draw: {pct(p_base['draw'])}")
    st.write(f"Away: {pct(p_base['away'])}")
with col3:
    st.markdown("**Bookmaker (implied)**")
    if p_mkt is None:
        st.write("Odds not available for this match.")
    else:
        st.write(f"Home: {pct(p_mkt['home'])}")
        st.write(f"Draw: {pct(p_mkt['draw'])}")
        st.write(f"Away: {pct(p_mkt['away'])}")

st.header("Evaluation results")
eval_path = RESULTS / "eval_results.csv"
if not eval_path.exists():
    st.info("No eval table yet. Run `python -m src.evaluate` to create it.")
else:
    eval_df = pd.read_csv(eval_path)
    st.dataframe(eval_df, use_container_width=True)
    if {"model", "log_loss"}.issubset(eval_df.columns):
        st.caption("Lower log loss is better.")
        st.bar_chart(eval_df.set_index("model")["log_loss"])
