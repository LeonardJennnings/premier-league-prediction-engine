"""Policy tests. Each one fails if the leak it names is put back.

The checks are the priority list for the leakage slice: training dates,
the pure goals model, publish-time prices, prior-round calibration and
blend weights, closing totals, the published model hash, and the ledger.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.clean import clean_frame
from src.config import LEDGER_DIR, MODELS_DIR, ROOT
from src.eval.gameweeks import assign_gameweeks
from src.eval.walkforward import iter_folds, run_walkforward
from src.features import add_implied_odds
from src.fixtures.books import _ONE_X_TWO, _OVER_UNDER
from src.ledger.log import append_predictions, append_settlements, read_jsonl
from src.models.artefacts import _content_hash, load_artefact, save_artefact
from src.models.dixon_coles import DixonColesModel, fit_dixon_coles, score_table
from src.models.markets import markets_from_score_table
from tests.test_walkforward import _mixed_matches

_TUNING_MODULES = (
    "src/eval/xi_tune.py",
    "src/eval/thin_prior.py",
    "src/eval/grid_calibration.py",
    "src/eval/market_blend.py",
    "src/eval/calibration.py",
    "src/eval/blend.py",
    "src/eval/walkforward.py",
    "src/models/dixon_coles.py",
    "src/models/poisson.py",
    "src/jobs/fit.py",
)
_PRICE_TOKENS = ("B365", "implied_", "odds", "book_p_")
_CLOSING_MARKERS = ("B365C", "AvgC")


def _is_closing_column(name: str) -> bool:
    return name.startswith(_CLOSING_MARKERS)


def test_train_dates_stay_strictly_before_the_test_round():
    """A match on the test round's first day must not enter training.

    ``date <= cutoff`` would pull that same-day match into the train set.
    """
    early = pd.date_range("2024-08-01", periods=6, freq="D")
    same_day = pd.Timestamp("2024-08-10")
    dates = list(early) + [same_day, same_day, same_day + pd.Timedelta(days=3)]
    frame = pd.DataFrame(
        {
            "match_id": list(range(len(dates))),
            "season_code": ["2425"] * len(dates),
            "date": dates,
            "gameweek": [1, 1, 1, 1, 1, 1, 2, 3, 3],
        }
    )
    folds = list(iter_folds(frame, min_train_rows=4))
    assert [gw for _, _, _, gw in folds] == [2, 3]
    for train, test, _season, gw in folds:
        assert train["date"].max() < test["date"].min()
        assert set(train["match_id"]).isdisjoint(set(test["match_id"]))
        assert not train["date"].isin(set(test["date"])).any()
        if gw == 3:
            assert 6 not in set(train["match_id"])


def test_dixon_coles_fit_ignores_price_columns():
    frame = _mixed_matches(30)
    poisoned = frame.copy()
    poisoned["implied_home"] = 0.01
    poisoned["implied_draw"] = 0.01
    poisoned["implied_away"] = 0.98
    poisoned["B365CH"] = 50.0
    poisoned["home_odds_close"] = 50.0
    plain = fit_dixon_coles(frame, xi=0.0)
    leaked = fit_dixon_coles(poisoned, xi=0.0)
    assert plain.attack == leaked.attack
    assert plain.defence == leaked.defence
    assert plain.home_advantage == leaked.home_advantage
    assert plain.rho == leaked.rho


def test_pure_model_source_does_not_name_a_price_column():
    for relative in ("src/models/dixon_coles.py", "src/models/poisson.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in _PRICE_TOKENS:
            assert token not in text, f"{relative} mentions {token}"


def test_publish_book_lookup_does_not_read_a_closing_column():
    names = [name for group in (*_ONE_X_TWO, *_OVER_UNDER) for name in group]
    assert names
    closing = [name for name in names if _is_closing_column(name)]
    assert closing == []
    publish_source = (ROOT / "src/jobs/publish.py").read_text(encoding="utf-8")
    for token in ("B365C", "AvgC", "home_odds_close", "over_25_odds_close"):
        assert token not in publish_source


def test_earlier_total_is_not_scored_as_a_closing_price():
    raw = pd.DataFrame(
        [
            {
                "Date": "15/08/2025",
                "HomeTeam": "Liverpool",
                "AwayTeam": "Bournemouth",
                "FTHG": 4,
                "FTAG": 2,
                "FTR": "H",
                "B365>2.5": 1.80,
                "B365<2.5": 2.10,
                "Avg>2.5": 1.50,
                "Avg<2.5": 2.50,
            }
        ]
    )
    cleaned = clean_frame(raw, "2526")
    featured = add_implied_odds(cleaned)
    assert "over_25_odds_close" not in cleaned.columns
    assert "implied_over_25" not in featured.columns
    open_over = (1 / 1.80) / (1 / 1.80 + 1 / 2.10)
    assert featured.loc[0, "implied_over_25_open"] == pytest.approx(open_over)

    with_close = raw.copy()
    with_close["AvgC>2.5"] = 1.70
    with_close["AvgC<2.5"] = 2.20
    closed = add_implied_odds(clean_frame(with_close, "2526"))
    close_over = (1 / 1.70) / (1 / 1.70 + 1 / 2.20)
    assert closed.loc[0, "implied_over_25"] == pytest.approx(close_over)
    assert closed.loc[0, "implied_over_25"] != pytest.approx(
        closed.loc[0, "implied_over_25_open"]
    )


def test_calibration_and_blend_see_only_earlier_rounds(monkeypatch):
    from src.eval.blend import fit_benter_blend
    from src.eval.calibration import fit_temperature

    seen_temperature: list[list[str]] = []
    seen_blend: list[list[str]] = []

    def _temperature(y_true, P):
        seen_temperature.append([str(item) for item in y_true])
        return fit_temperature(y_true, P)

    def _blend(y_true, P_model, P_book):
        seen_blend.append([str(item) for item in y_true])
        return fit_benter_blend(y_true, P_model, P_book)

    monkeypatch.setattr("src.eval.walkforward.fit_temperature", _temperature)
    monkeypatch.setattr("src.eval.walkforward.fit_benter_blend", _blend)

    _summary, by_gw, preds, _scores = run_walkforward(
        _mixed_matches(24), min_train_rows=4, calibration_min_rows=8, blend_min_rows=8
    )
    raw = by_gw[(by_gw["model"] == "dixon_coles") & (by_gw["market"] == "1x2")]
    fold_results = [str(item) for item in preds["dixon_coles"]["result"].tolist()]
    expected: list[list[str]] = []
    prior: list[str] = []
    offset = 0
    for row in raw.itertuples(index=False):
        n_test = int(row.n_test)
        current = fold_results[offset : offset + n_test]
        if len(prior) >= 8:
            expected.append(list(prior))
        prior.extend(current)
        offset += n_test

    assert seen_temperature == expected
    assert seen_blend == expected
    assert expected
    assert all(len(fitted) < len(prior) for fitted in seen_temperature)
    assert offset == len(fold_results)


def test_settlement_does_not_replace_the_hash_stored_at_publish(tmp_path):
    model = DixonColesModel(
        attack={"Arsenal": 1.2, "Chelsea": 0.9},
        defence={"Arsenal": 1.0, "Chelsea": 1.1},
        home_advantage=1.15,
        rho=-0.04,
        xi=0.0035,
        as_of=pd.Timestamp("2026-11-01"),
    )
    version, _path = save_artefact(
        model, 1.0, {"xi": 0.0035}, models_dir=tmp_path / "models"
    )
    meta = json.loads((tmp_path / "models" / version / "meta.json").read_text())
    line = {
        "match_id": 9001,
        "forecast_line": "model",
        "kickoff_utc": "2026-12-01T15:00:00Z",
        "published_at": "2026-11-25T00:00:00Z",
        "artefact_version": version,
        "model_version": f"{version}-grid",
        "content_hash": meta["content_hash"],
        "p_home": 0.47,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
    }
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    for name in ("predictions.jsonl", "book_lines.jsonl", "settlements.jsonl"):
        (ledger / name).write_text("", encoding="utf-8")
    now = datetime(2026, 11, 25, tzinfo=timezone.utc)
    append_predictions([line], now_utc=now, path=ledger)
    append_settlements(
        [
            {
                "match_id": 9001,
                "forecast_line": "model",
                "kind": "result",
                "home_goals": 1,
                "away_goals": 0,
                "result": "H",
            }
        ],
        path=ledger,
    )
    stored = read_jsonl(ledger / "predictions.jsonl")
    assert stored == [line]
    settlement = read_jsonl(ledger / "settlements.jsonl")[0]
    for key in ("artefact_version", "model_version", "content_hash"):
        if key in settlement:
            assert settlement[key] == line[key]

    replaced = DixonColesModel(
        attack={"Arsenal": 2.0, "Chelsea": 0.4},
        defence={"Arsenal": 0.5, "Chelsea": 1.4},
        home_advantage=1.4,
        rho=0.05,
        xi=0.0035,
        as_of=pd.Timestamp("2026-11-20"),
    )
    save_artefact(replaced, 1.0, {"xi": 0.0035}, models_dir=tmp_path / "models")
    assert read_jsonl(ledger / "predictions.jsonl") == [line]


def test_committed_score_table_matches_the_publish_hash():
    predictions = read_jsonl(LEDGER_DIR / "predictions.jsonl")
    settlements = read_jsonl(LEDGER_DIR / "settlements.jsonl")
    by_key = {
        (int(row["match_id"]), str(row["forecast_line"])): row for row in predictions
    }
    assert settlements
    scored = 0
    for settlement in settlements:
        key = (int(settlement["match_id"]), str(settlement["forecast_line"]))
        published = by_key[key]
        assert published["published_at"] < published["kickoff_utc"]
        for field in ("artefact_version", "model_version", "content_hash"):
            if field in settlement:
                assert settlement[field] == published[field]
    for published in predictions:
        if not published.get("score_table"):
            continue
        version = str(published["artefact_version"])
        model_path = MODELS_DIR / version / "model.json"
        meta_path = MODELS_DIR / version / "meta.json"
        assert model_path.is_file(), version
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        digest = _content_hash(payload)
        assert digest == meta["content_hash"]
        assert version.endswith(digest[:8])
        model, _temperature, _hyper = load_artefact(version)
        table = score_table(model, published["home_team"], published["away_team"])
        markets = markets_from_score_table(table)
        assert markets["home"] == pytest.approx(published["p_home"], abs=1e-12)
        assert markets["over_25"] == pytest.approx(published["p_over_25"], abs=1e-12)
        scored += 1
    assert scored == 11


def test_tuning_code_cannot_import_the_ledger():
    for relative in _TUNING_MODULES:
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for name in modules:
                assert name != "src.ledger" and not name.startswith("src.ledger.")
        text = path.read_text(encoding="utf-8")
        assert "data/ledger" not in text


def test_site_ranges_between_the_two_looks_say_descriptive():
    """A range that is not the 380 or 1,140 look has to say it is descriptive."""
    web = ROOT / "web" / "src"
    offenders = []
    for path in web.rglob("*"):
        if path.suffix not in {".astro", ".ts", ".tsx", ".js", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        if "95%" not in text and "confidence interval" not in lowered:
            continue
        if "descriptive" in lowered:
            continue
        if "380" in text and "1140" in text:
            continue
        offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_slice_7_note_is_committed_before_that_run():
    note = ROOT / "docs" / "experiments" / "2026-11-25-thin-data-prior.md"
    template = ROOT / "docs" / "experiments" / "template.md"
    looks = ROOT / "docs" / "experiments" / "live-record-looks.md"
    text = note.read_text(encoding="utf-8")
    assert "written before the run" in text
    assert "Stop rule" in text
    assert "live ledger is not a tuning set" in text
    for heading in ("## Question", "## Candidates", "## Metric", "## Data windows", "## Stop rule"):
        assert heading in template.read_text(encoding="utf-8")
    looks_text = looks.read_text(encoding="utf-8")
    assert "380" in looks_text
    assert "1,140" in looks_text
    assert "descriptive" in looks_text
    assert "beat the book" not in looks_text.lower()
    assert "beat the book" not in text.lower()


def test_slice_8_note_is_committed_before_that_run():
    note = ROOT / "docs" / "experiments" / "2026-11-25-score-table-calibration.md"
    text = note.read_text(encoding="utf-8")
    assert "written before the run" in text
    assert "Stop rule" in text
    assert "live ledger is not a tuning set" in text
    assert "0.95" in text and "1.05" in text
    assert "-0.10" in text and "0.00" in text
    assert "0.005" in text
    assert "beat the book" not in text.lower()


def test_slice_9_note_is_committed_before_that_run():
    note = ROOT / "docs" / "experiments" / "2026-11-25-model-plus-market.md"
    text = note.read_text(encoding="utf-8")
    assert "written before the run" in text
    assert "Stop rule" in text
    assert "live ledger is not a tuning set" in text
    assert "closing price is not an input" in text.lower()
    assert "0.00" in text and "0.05" in text and "1.00" in text
    assert "0.10" in text
    assert "beat the book" not in text.lower()
