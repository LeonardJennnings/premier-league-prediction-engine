"""Build model-ready features from the cleaned matches table.

Feature contract
----------------
* **Point-in-time** columns (match id, opening/closing implied odds) come from
  the match row itself and do not join other fixtures.
* **History-dependent** columns (future form / Elo / decay) must be computed
  only from matches with ``date < cutoff`` for the row or fold being scored.
  Use :func:`assert_history_before_cutoff` and
  :func:`add_history_dependent_features` so walk-forward cannot silently
  leak when those features land.
"""

from __future__ import annotations

import numpy as np
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


def _devig_1x2(
    out: pd.DataFrame,
    home_col: str,
    draw_col: str,
    away_col: str,
    *,
    implied_home: str,
    implied_draw: str,
    implied_away: str,
) -> None:
    """Write de-vigged 1X2 implied probs when all three odds columns exist."""
    needed = [home_col, draw_col, away_col]
    if not all(c in out.columns for c in needed):
        return
    raw_h = 1 / out[home_col]
    raw_d = 1 / out[draw_col]
    raw_a = 1 / out[away_col]
    total = raw_h + raw_d + raw_a
    out[implied_home] = raw_h / total
    out[implied_draw] = raw_d / total
    out[implied_away] = raw_a / total


def _devig_ou(
    out: pd.DataFrame,
    over_col: str,
    under_col: str,
    *,
    implied_over: str,
    implied_under: str,
) -> None:
    """Write de-vigged O/U 2.5 implied probs when both odds columns exist."""
    if over_col not in out.columns or under_col not in out.columns:
        return
    raw_over = 1 / out[over_col]
    raw_under = 1 / out[under_col]
    total_ou = raw_over + raw_under
    out[implied_over] = raw_over / total_ou
    out[implied_under] = raw_under / total_ou


def add_implied_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Convert opening/closing decimal odds to normalised implied probabilities.

    Closing 1X2 keeps the historical names ``implied_home`` / ``implied_draw`` /
    ``implied_away`` (bookmaker_implied / sklearn baseline). Opening 1X2 uses
    ``implied_*_open``. Closing O/U stays ``implied_over_25`` /
    ``implied_under_25``; opening O/U uses ``implied_*_open``.
    """
    out = df.copy()
    _devig_1x2(
        out,
        "home_odds_close",
        "draw_odds_close",
        "away_odds_close",
        implied_home="implied_home",
        implied_draw="implied_draw",
        implied_away="implied_away",
    )
    _devig_1x2(
        out,
        "home_odds_open",
        "draw_odds_open",
        "away_odds_open",
        implied_home="implied_home_open",
        implied_draw="implied_draw_open",
        implied_away="implied_away_open",
    )
    _devig_ou(
        out,
        "over_25_odds_close",
        "under_25_odds_close",
        implied_over="implied_over_25",
        implied_under="implied_under_25",
    )
    _devig_ou(
        out,
        "over_25_odds_open",
        "under_25_odds_open",
        implied_over="implied_over_25_open",
        implied_under="implied_under_25_open",
    )
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


def add_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """Days since each club's previous match in this table.

    A match on the same date is not earlier. The first match in the table
    for a club has no rest day. A later row cannot change an earlier row,
    because the count uses only dates strictly before that row.
    """
    needed = ("date", "home_team", "away_team")
    missing = [name for name in needed if name not in matches.columns]
    if missing:
        raise ValueError("rest days need " + ", ".join(missing))
    out = matches.copy()
    dates = pd.to_datetime(out["date"])
    if dates.isna().any():
        raise ValueError("rest days need a date on every row")
    n = len(out)
    home_rest = np.full(n, np.nan)
    away_rest = np.full(n, np.nan)
    if n == 0:
        out["home_rest_days"] = home_rest
        out["away_rest_days"] = away_rest
        return out
    rows = np.arange(n)
    teams = np.concatenate(
        [out["home_team"].astype(str).to_numpy(), out["away_team"].astype(str).to_numpy()]
    )
    long_dates = np.concatenate([dates.to_numpy(), dates.to_numpy()])
    long_rows = np.concatenate([rows, rows])
    long_side = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    order = np.lexsort((long_rows, long_dates, teams))
    teams = teams[order]
    long_dates = long_dates[order]
    long_rows = long_rows[order]
    long_side = long_side[order]
    change = np.flatnonzero(teams[1:] != teams[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(teams)]))
    for start, end in zip(starts, ends, strict=True):
        group_dates = long_dates[start:end]
        previous = np.searchsorted(group_dates, group_dates, side="left") - 1
        has_previous = previous >= 0
        if not has_previous.any():
            continue
        day_gap = (
            group_dates[has_previous].astype("datetime64[D]")
            - group_dates[previous[has_previous]].astype("datetime64[D]")
        ).astype(int)
        for row_i, side_i, days in zip(
            long_rows[start:end][has_previous],
            long_side[start:end][has_previous],
            day_gap,
            strict=True,
        ):
            if side_i == 0:
                home_rest[int(row_i)] = int(days)
            else:
                away_rest[int(row_i)] = int(days)
    out["home_rest_days"] = home_rest
    out["away_rest_days"] = away_rest
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
