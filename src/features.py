"""Build model-ready features from the cleaned matches table.

Feature contract
----------------
* **Point-in-time** columns (match id, closing implied odds) come from the
  match row itself and do not join other fixtures.
* **History-dependent** columns (future form / Elo / decay) must be computed
  only from matches with ``date < cutoff`` for the row or fold being scored.
  Use :func:`assert_history_before_cutoff` and
  :func:`add_history_dependent_features` so walk-forward cannot silently
  leak when those features land.
"""

from __future__ import annotations

import pandas as pd


def add_match_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple unique id per match."""
    out = df.copy()
    out["match_id"] = (
        out["season_code"].astype(str)
        + "_"
        + out["date"].astype(str)
        + "_"
        + out["home_team"].astype(str)
        + "_"
        + out["away_team"].astype(str)
    )
    return out


def add_implied_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Convert closing decimal odds to normalised implied probabilities."""
    out = df.copy()
    needed = ["home_odds_close", "draw_odds_close", "away_odds_close"]
    if all(c in out.columns for c in needed):
        raw_h = 1 / out["home_odds_close"]
        raw_d = 1 / out["draw_odds_close"]
        raw_a = 1 / out["away_odds_close"]
        total = raw_h + raw_d + raw_a

        out["implied_home"] = raw_h / total
        out["implied_draw"] = raw_d / total
        out["implied_away"] = raw_a / total

    ou_needed = ["over_25_odds_close", "under_25_odds_close"]
    if all(c in out.columns for c in ou_needed):
        raw_over = 1 / out["over_25_odds_close"]
        raw_under = 1 / out["under_25_odds_close"]
        total_ou = raw_over + raw_under

        out["implied_over_25"] = raw_over / total_ou
        out["implied_under_25"] = raw_under / total_ou
    return out


def assert_history_before_cutoff(
    history: pd.DataFrame,
    cutoff: pd.Timestamp | str,
) -> None:
    """Raise if ``history`` contains any row with date >= cutoff.

    Call this before any join that builds history-dependent features so a
    future-looking merge fails loudly instead of leaking into metrics.
    """
    if history.empty:
        return
    if "date" not in history.columns:
        raise ValueError("history must include a date column")
    cutoff_ts = pd.Timestamp(cutoff)
    dates = pd.to_datetime(history["date"])
    leaked = history.loc[dates >= cutoff_ts]
    if not leaked.empty:
        sample = leaked.iloc[0]
        raise ValueError(
            "history contains rows on or after cutoff "
            f"{cutoff_ts.date()}: e.g. {sample.get('home_team')} vs "
            f"{sample.get('away_team')} on {sample['date']}"
        )


def add_history_dependent_features(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Attach features derived only from past matches.

    Today this is a no-op beyond enforcing the cutoff (odds-only stack).
    Rolling form, Elo, and time-decay features should land here and must
    use ``history`` only — never rows from ``targets`` that fall on/after
    the cutoff.
    """
    out = targets.copy()
    if cutoff is None:
        if out.empty or "date" not in out.columns:
            return out
        cutoff = pd.to_datetime(out["date"]).min()
    assert_history_before_cutoff(history, cutoff)
    # Epic 2+: compute form / Elo from `history` and join onto `out`.
    return out


def add_features(
    df: pd.DataFrame,
    *,
    history: pd.DataFrame | None = None,
    cutoff: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Apply feature steps.

    Parameters
    ----------
    df:
        Rows to decorate (train, test, or full table).
    history:
        Past matches available for history-dependent features. Defaults to
        ``df`` itself. Walk-forward should pass train-only history with an
        explicit ``cutoff``.
    cutoff:
        Exclusive upper bound on history dates. When set (or when
        ``history`` is provided), :func:`add_history_dependent_features`
        enforces ``history.date < cutoff``.
    """
    out = add_match_id(df)
    out = add_implied_odds(out)

    if history is not None or cutoff is not None:
        hist = history if history is not None else out
        if cutoff is None:
            cutoff = pd.to_datetime(out["date"]).min()
        cutoff_ts = pd.Timestamp(cutoff)
        hist_dates = pd.to_datetime(hist["date"])
        hist_cut = hist.loc[hist_dates < cutoff_ts].copy()
        out = add_history_dependent_features(out, hist_cut, cutoff=cutoff_ts)
    return out


def features_for_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test features for one walk-forward fold.

    History-dependent features use only ``train`` (already ``date <`` the
    fold cutoff). Point-in-time columns are taken from each side's rows.
    """
    if test.empty:
        raise ValueError("test fold is empty")
    cutoff = pd.to_datetime(test["date"]).min()
    train_f = add_features(train, history=train, cutoff=cutoff)
    test_f = add_features(test, history=train, cutoff=cutoff)
    return train_f, test_f
