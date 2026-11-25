"""One daily pass: settle, refit, publish, capture book prices, export.

Run it twice with no new scores and it changes nothing. ``--dry-run`` prints
the counts and writes nothing. A bad team name or a failed fixtures call
writes nothing and says so.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PROCESSED,
    FD_UK_FIXTURES_CSV,
    LEDGER_DIR,
    MODELS_DIR,
    PREDICTIONS_DB,
    PUBLISH_WINDOW_DAYS,
    ROOT,
)
from src.fixtures.books import index_fixtures_csv, load_book_index, lookup_book, quotes_from_season
from src.fixtures.client import fetch_pl_matches
from src.fixtures.window import matches_in_publish_window
from src.jobs.fit import fit_and_save, load_training_matches
from src.jobs import publish as publish_module
from src.jobs.publish import _mapped_fixtures, publish_slate
from src.ledger.export_site import (
    SETTLED_NAME,
    SCORECARD_NAME,
    UPCOMING_NAME,
    build_site_files,
    resolve_match,
    write_site_files,
)
from src.ledger.log import (
    append_book_lines,
    append_settlements,
    ensure_ledger,
    read_jsonl,
)
from src.ledger.rebuild import rebuild_store
from src.logging_setup import setup_logging
from src.teams.mapping import load_team_name_map

logger = logging.getLogger(__name__)

_FINISHED = {"FINISHED", "AWARDED"}
_POSTPONED = {"POSTPONED", "SUSPENDED"}
_CANCELLED = {"CANCELLED", "CANCELED"}
_LISTED = {"TIMED", "SCHEDULED"}
_COMMIT_PATHS = [
    "data/ledger",
    "models",
    f"web/src/data/{UPCOMING_NAME}",
    f"web/src/data/{SETTLED_NAME}",
    f"web/src/data/{SCORECARD_NAME}",
]


def run_daily(
    *,
    now_utc: datetime | None = None,
    dry_run: bool = False,
    commit: bool = False,
    token: str | None = None,
    ledger_dir: Path | None = None,
    export_dir: Path | None = None,
    models_dir: Path | None = None,
    db_path: Path | None = None,
    repo_root: Path | None = None,
    season_frame: pd.DataFrame | None = None,
    book_index: dict | None = None,
    fixtures: list | None = None,
) -> dict:
    """Plan the pass, then write. Dry-run returns after the plan."""
    now = _as_utc(now_utc)
    recorded_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = Path(LEDGER_DIR if ledger_dir is None else ledger_dir)
    models = Path(MODELS_DIR if models_dir is None else models_dir)
    database = Path(PREDICTIONS_DB if db_path is None else db_path)
    exported = Path(ROOT / "web" / "src" / "data" if export_dir is None else export_dir)
    repo = Path(ROOT if repo_root is None else repo_root)

    predictions = read_jsonl(ledger / "predictions.jsonl") if (ledger / "predictions.jsonl").is_file() else []
    settlements = read_jsonl(ledger / "settlements.jsonl") if (ledger / "settlements.jsonl").is_file() else []
    book_lines = read_jsonl(ledger / "book_lines.jsonl") if (ledger / "book_lines.jsonl").is_file() else []
    prediction_bytes = (ledger / "predictions.jsonl").read_bytes() if (ledger / "predictions.jsonl").is_file() else b""

    fetched = fixtures if fixtures is not None else _fetch(predictions, settlements, now, token, dry_run)
    by_id = {int(match["match_id"]): match for match in fetched}
    settlement_lines = _plan_settlements(predictions, settlements, by_id, recorded_at)
    planned_settlements = settlements + settlement_lines
    season = _season_frame(season_frame)
    openings = _opening_index(book_index, dry_run=dry_run)
    fresh_books = _plan_books(
        predictions, planned_settlements, book_lines, season, openings, now, recorded_at
    )
    new_fixtures = _new_fixtures(predictions, fetched, now)
    if new_fixtures:
        _mapped_fixtures(new_fixtures, load_team_name_map())
    as_of = _latest_as_of(models)
    refit = _needs_refit(predictions, planned_settlements, as_of)
    if new_fixtures and as_of is None and not refit:
        raise RuntimeError("No fitted model to publish from")

    counts = {
        "added": len(new_fixtures),
        "settled": len(settlement_lines),
        "price_filled": len(fresh_books),
        "refit": refit,
    }
    _print_counts(counts, dry_run=dry_run)
    if dry_run:
        return counts

    ensure_ledger(ledger)
    settled_written = append_settlements(settlement_lines, path=ledger)["inserted"]
    books_written = append_book_lines(fresh_books, path=ledger)["inserted"]
    if refit:
        fit_and_save(load_training_matches(), models_dir=models, db_path=database)
    if new_fixtures:
        books_before = _line_count(ledger / "book_lines.jsonl")
        _publish(fetched, now, token, ledger, models, database)
        books_written += _line_count(ledger / "book_lines.jsonl") - books_before
    written = (ledger / "predictions.jsonl").read_bytes() if (ledger / "predictions.jsonl").is_file() else b""
    if prediction_bytes and not written.startswith(prediction_bytes):
        raise RuntimeError("Prediction lines were rewritten")

    predictions = read_jsonl(ledger / "predictions.jsonl")
    settlements = read_jsonl(ledger / "settlements.jsonl")
    book_lines = read_jsonl(ledger / "book_lines.jsonl")
    counts["added"] = _count_new_predictions(prediction_bytes, written)
    counts["settled"] = settled_written
    counts["price_filled"] = books_written
    documents = build_site_files(predictions, settlements, book_lines, now_utc=now)
    write_site_files(documents, exported)
    rebuild_store(
        predictions,
        settlements,
        book_lines,
        db_path=database,
        models_dir=models,
    )
    if commit:
        message = (
            f"Ledger: added {counts['added']}, settled {counts['settled']}, "
            f"price-filled {counts['price_filled']}"
        )
        counts["committed"] = _commit(repo, message)
    else:
        counts["committed"] = False
    logger.info("Daily ledger: %s", counts)
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Settle, refit, publish, capture book prices, and export.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts and write nothing")
    parser.add_argument("--commit", action="store_true", help="Commit ledger, model, and site files when they changed")
    parser.add_argument("--now", default=None, help="UTC timestamp used as the run clock")
    args = parser.parse_args(argv)
    if args.dry_run and args.commit:
        parser.error("--dry-run does not commit")
    setup_logging()
    now = _parse_utc(args.now) if args.now else datetime.now(timezone.utc)
    try:
        run_daily(now_utc=now, dry_run=args.dry_run, commit=args.commit)
    except Exception as exc:
        logger.error("ALERT: daily ledger failed: %s", exc)
        raise


def _fetch(predictions, settlements, now, token, dry_run: bool) -> list[dict]:
    start, end = _query_dates(predictions, settlements, now)
    return fetch_pl_matches(
        start,
        end,
        token=token,
        use_cache=False,
        write_cache=not dry_run,
    )


def _query_dates(predictions, settlements, now: datetime) -> tuple[str, str]:
    kicks = [_parse_utc(row["kickoff_utc"]) for row in predictions]
    start = min(kicks).date() if kicks else now.date()
    start = min(start, now.date())
    end = (now + timedelta(days=PUBLISH_WINDOW_DAYS)).date() + timedelta(days=1)
    if kicks:
        end = max(end, max(kicks).date() + timedelta(days=1))
    postponed = []
    for row in predictions:
        state = resolve_match(row, settlements)
        if state["status"] == "postponed":
            postponed.append(_parse_utc(state["kickoff_utc"]))
    if postponed:
        horizon = max(postponed).date() + timedelta(days=90)
        end = max(end, horizon + timedelta(days=1))
    return start.isoformat(), end.isoformat()


def _plan_settlements(predictions, settlements, by_id, recorded_at: str) -> list[dict]:
    planned = []
    for prediction in predictions:
        match = by_id.get(int(prediction["match_id"]))
        if match is None:
            continue
        state = resolve_match(prediction, settlements)
        line = _settlement_line(prediction, match, state, recorded_at)
        if line is not None:
            planned.append(line)
            settlements = settlements + [line]
    return planned


def _settlement_line(prediction, match, state, recorded_at: str) -> dict | None:
    status = match.get("status")
    home = match.get("home_goals")
    away = match.get("away_goals")
    base = {
        "match_id": int(prediction["match_id"]),
        "forecast_line": prediction.get("forecast_line") or "model",
        "provider_status": status,
        "provider_updated_at": match.get("provider_updated_at"),
        "recorded_at": recorded_at,
        "source": "football-data.org",
        "kickoff_utc": match.get("kickoff_utc") or prediction["kickoff_utc"],
    }
    if status in _FINISHED and home is not None and away is not None:
        home_goals = int(home)
        away_goals = int(away)
        if (
            state["status"] == "settled"
            and state["home_goals"] == home_goals
            and state["away_goals"] == away_goals
        ):
            return None
        kind = "correction" if state["status"] == "settled" else "result"
        return {
            **base,
            "kind": kind,
            "status": "settled",
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": _result_code(home_goals, away_goals),
        }
    if state["status"] == "settled":
        return None
    if status in _POSTPONED:
        if state["status"] == "postponed" and state["kickoff_utc"] == base["kickoff_utc"]:
            return None
        return {**base, "kind": "status", "status": "postponed"}
    if status in _CANCELLED:
        if state["status"] == "cancelled":
            return None
        return {**base, "kind": "status", "status": "cancelled"}
    if status in _LISTED and state["status"] == "postponed":
        return {**base, "kind": "status", "status": "scheduled"}
    return None


def _plan_books(predictions, settlements, book_lines, season, openings, now, recorded_at) -> list[dict]:
    have = {
        (int(row["match_id"]), str(row.get("forecast_line")), str(row.get("kind")))
        for row in book_lines
    }
    fresh = []
    for prediction in predictions:
        state = resolve_match(prediction, settlements)
        forecast = str(prediction.get("forecast_line") or "model")
        match_id = int(prediction["match_id"])
        kickoff = state["kickoff_utc"]
        if state["status"] == "settled":
            for kind, quote in quotes_from_season(
                season, prediction["home_team"], prediction["away_team"], kickoff
            ).items():
                _add_book(fresh, have, prediction, kind, quote, recorded_at, forecast, match_id)
        if _parse_utc(kickoff) > now and (match_id, forecast, "opening") not in have:
            quote = lookup_book(openings, prediction["home_team"], prediction["away_team"], kickoff)
            if quote:
                payload = {
                    "p_home": quote.get("book_p_home"),
                    "p_draw": quote.get("book_p_draw"),
                    "p_away": quote.get("book_p_away"),
                    "p_over_25": quote.get("book_p_over_25"),
                    "p_under_25": quote.get("book_p_under_25"),
                    "source": quote.get("book_source"),
                }
                _add_book(fresh, have, prediction, "opening", payload, recorded_at, forecast, match_id)
    return fresh


def _add_book(fresh, have, prediction, kind, quote, recorded_at, forecast, match_id) -> None:
    key = (match_id, forecast, kind)
    if key in have or quote.get("p_home") is None:
        return
    have.add(key)
    fresh.append(
        {
            "match_id": match_id,
            "forecast_line": forecast,
            "kind": kind,
            "captured_at": recorded_at,
            "kickoff_utc": prediction["kickoff_utc"],
            "source": quote.get("source"),
            "p_home": quote.get("p_home"),
            "p_draw": quote.get("p_draw"),
            "p_away": quote.get("p_away"),
            "p_over_25": quote.get("p_over_25"),
            "p_under_25": quote.get("p_under_25"),
        }
    )


def _new_fixtures(predictions, fetched, now: datetime) -> list[dict]:
    known = {int(row["match_id"]) for row in predictions}
    in_window = matches_in_publish_window(fetched, now_utc=now)
    fresh = []
    for match in in_window:
        kickoff = _parse_utc(match["kickoff_utc"])
        if kickoff <= now or int(match["match_id"]) in known:
            continue
        fresh.append(match)
    return fresh


def _needs_refit(predictions, settlements, as_of: str | None) -> bool:
    if not as_of:
        return False
    cutoff = as_of[:10]
    for prediction in predictions:
        state = resolve_match(prediction, settlements)
        if state["status"] != "settled":
            continue
        if str(state["kickoff_utc"])[:10] > cutoff:
            return True
    return False


def _latest_as_of(models_dir: Path) -> str | None:
    best = None
    if not models_dir.is_dir():
        return None
    for path in models_dir.glob("dc-*/model.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        as_of = str(payload.get("as_of") or "")[:10]
        if as_of and (best is None or as_of > best):
            best = as_of
    return best


def _season_frame(season_frame) -> pd.DataFrame | None:
    if season_frame is not None:
        return season_frame
    path = Path(DATA_PROCESSED) / "matches.csv"
    if path.is_file():
        return pd.read_csv(path)
    return None


def _opening_index(book_index, *, dry_run: bool) -> dict:
    if book_index is not None:
        return book_index
    if dry_run:
        if FD_UK_FIXTURES_CSV.is_file():
            try:
                return index_fixtures_csv(FD_UK_FIXTURES_CSV)
            except Exception as exc:
                logger.warning("Book file could not be read (%s)", exc)
        return {}
    return load_book_index()


def _publish(fetched, now, token, ledger, models, database) -> None:
    """Publish from the already fetched list, into this run's ledger."""
    publishable = [
        match for match in fetched if _parse_utc(match["kickoff_utc"]) > now
    ]
    saved = (
        publish_module.LEDGER_DIR,
        publish_module.PREDICTIONS_DB,
        publish_module.MODELS_DIR,
    )
    publish_module.LEDGER_DIR = ledger
    publish_module.PREDICTIONS_DB = database
    publish_module.MODELS_DIR = models
    try:
        publish_slate(now_utc=now, token=token, matches=publishable)
    finally:
        (
            publish_module.LEDGER_DIR,
            publish_module.PREDICTIONS_DB,
            publish_module.MODELS_DIR,
        ) = saved


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _count_new_predictions(before: bytes, after: bytes) -> int:
    if not after.startswith(before):
        return 0
    extra = after[len(before):]
    return len([line for line in extra.decode("utf-8").splitlines() if line.strip()])


def _commit(repo: Path, message: str) -> bool:
    add = subprocess.run(
        ["git", "add", "-A", "--", *_COMMIT_PATHS],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        raise RuntimeError(add.stderr.strip() or "git add failed")
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo, check=False)
    if diff.returncode == 0:
        return False
    if diff.returncode != 1:
        raise RuntimeError("git diff failed")
    commit = subprocess.run(
        [
            "git",
            "-c",
            "user.name=github-actions[bot]",
            "-c",
            "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "commit",
            "-m",
            message,
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        raise RuntimeError(commit.stderr.strip() or "git commit failed")
    return True


def _print_counts(counts: dict, *, dry_run: bool) -> None:
    if dry_run:
        print("dry-run")
    print(f"added={counts['added']}")
    print(f"settled={counts['settled']}")
    print(f"price-filled={counts['price_filled']}")
    print(f"refit={'yes' if counts['refit'] else 'no'}")


def _result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_utc(now_utc: datetime | None) -> datetime:
    if now_utc is None:
        return datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=timezone.utc)
    return now_utc.astimezone(timezone.utc)


if __name__ == "__main__":
    main()
