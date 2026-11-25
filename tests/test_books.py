"""Book enrichment from a local football-data.co.uk fixtures CSV. No network."""

from pathlib import Path

import pytest
import requests

from src.config import FD_UK_FIXTURES_URL
from src.fixtures.books import (
    BOOK_SOURCE,
    download_fixtures_csv,
    index_fixtures_csv,
    load_book_index,
    lookup_book,
)

CSV_PATH = Path(__file__).parent / "fixtures" / "fdco_upcoming.csv"


def _share(*odds: float) -> list[float]:
    raw = [1 / price for price in odds]
    total = sum(raw)
    return [price / total for price in raw]


def test_index_devigs_bet365_and_ignores_other_divisions():
    index = index_fixtures_csv(CSV_PATH)
    home, draw, away = _share(2.0, 3.4, 4.0)
    over, under = _share(1.80, 2.10)
    quote = lookup_book(index, "Arsenal", "Chelsea", "2026-11-20T15:00:00Z")
    assert quote is not None
    assert quote["book_p_home"] == pytest.approx(home)
    assert quote["book_p_draw"] == pytest.approx(draw)
    assert quote["book_p_away"] == pytest.approx(away)
    assert quote["book_p_over_25"] == pytest.approx(over)
    assert quote["book_p_under_25"] == pytest.approx(under)
    assert abs(quote["book_p_home"] + quote["book_p_draw"] + quote["book_p_away"] - 1) < 1e-9
    assert quote["book_source"] == BOOK_SOURCE
    # The E1 row on the same teams and date is 9/9/9. It must not win.
    assert quote["book_p_home"] != pytest.approx(1 / 3)


def test_index_leaves_over_under_null_when_one_side_is_missing():
    index = index_fixtures_csv(CSV_PATH)
    quote = lookup_book(index, "Liverpool", "Everton", "2026-11-21")
    assert quote is not None
    home, draw, away = _share(1.50, 4.20, 6.50)
    assert quote["book_p_home"] == pytest.approx(home)
    assert quote["book_p_draw"] == pytest.approx(draw)
    assert quote["book_p_away"] == pytest.approx(away)
    assert quote["book_p_over_25"] is None
    assert quote["book_p_under_25"] is None
    assert quote["book_source"] == BOOK_SOURCE


def test_lookup_uses_london_date_when_utc_date_differs():
    index = index_fixtures_csv(CSV_PATH)
    # 23:30 UTC on 15 Aug is 00:30 BST on 16 Aug.
    quote = lookup_book(index, "Arsenal", "Chelsea", "2026-08-15T23:30:00Z")
    home, _draw, _away = _share(1.80, 3.60, 4.50)
    assert quote is not None
    assert quote["book_p_home"] == pytest.approx(home)
    assert lookup_book(index, "Arsenal", "Everton", "2026-11-20T15:00:00Z") is None


def test_lookup_ignores_closing_prices_when_opening_is_blank():
    index = index_fixtures_csv(CSV_PATH)
    assert lookup_book(index, "Spurs", "Brighton", "2026-11-22T15:00:00Z") is None


def test_index_without_div_keeps_the_row(tmp_path: Path):
    path = tmp_path / "fixtures.csv"
    path.write_text(
        "Date,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
        "20/11/2026,Arsenal,Chelsea,2.0,3.4,4.0\n",
        encoding="utf-8",
    )
    quote = lookup_book(
        index_fixtures_csv(path), "Arsenal", "Chelsea", "2026-11-20T12:00:00Z"
    )
    assert quote is not None
    assert quote["book_p_over_25"] is None


def test_download_writes_cache_and_fails_soft(tmp_path: Path):
    dest = tmp_path / "fixtures.csv"
    body = CSV_PATH.read_bytes()

    class Session:
        def get(self, url, timeout=None, **kwargs):
            self.url = url
            self.timeout = timeout

            class Response:
                content = body

                def raise_for_status(self):
                    return None

            return Response()

    session = Session()
    saved = download_fixtures_csv(dest, session=session)
    assert saved == dest
    assert dest.read_bytes() == body
    assert session.url == FD_UK_FIXTURES_URL
    indexed = load_book_index(dest, session=session)
    assert ("Arsenal", "Chelsea", "2026-11-20") in indexed

    class Down:
        def get(self, url, timeout=None, **kwargs):
            raise requests.ConnectionError("offline")

    assert download_fixtures_csv(tmp_path / "missing.csv", session=Down()) is None
    assert load_book_index(tmp_path / "empty.csv", session=_EmptySession()) == {}


class _EmptySession:
    def get(self, url, timeout=None, **kwargs):
        class Response:
            content = b""

            def raise_for_status(self):
                return None

        return Response()
