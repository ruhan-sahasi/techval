"""The risk-free rate is the yield published on or before the valuation date."""

from __future__ import annotations

from datetime import date

import pytest

from techval.edgar import HttpCache
from techval.market import TREASURY_URL, CsvSource, MarketData

HEADER = "Date,1 Mo,2 Mo,3 Mo,6 Mo,1 Yr,2 Yr,3 Yr,5 Yr,7 Yr,10 Yr,20 Yr,30 Yr\n"


def curve(rows: dict[str, float]) -> bytes:
    # The feed lists the newest day first.
    lines = [f"{day},,,,,,,,,,{rate:.2f},," for day, rate in sorted(rows.items(), reverse=True)]
    return (HEADER + "\n".join(lines) + "\n").encode()


@pytest.fixture()
def cache(tmp_path):
    c = HttpCache(root=tmp_path / "cache")
    c.put(TREASURY_URL.format(year=2023), curve({"12/28/2023": 3.84, "12/29/2023": 3.88}))
    c.put(
        TREASURY_URL.format(year=2024),
        curve({"01/02/2024": 3.95, "03/27/2024": 4.20, "03/28/2024": 4.21, "03/29/2024": 4.30, "12/31/2024": 4.58}),
    )
    return c


def rate_on(cache, when):
    return MarketData(CsvSource(cache.root), cache, today=when).risk_free_rate()


def test_a_past_valuation_discounts_at_the_yield_of_its_own_day(cache):
    # Not the year's last yield, published nine months later.
    assert rate_on(cache, date(2024, 3, 28)) == (0.0421, "US Treasury 10Y constant maturity, 2024-03-28")


def test_a_day_with_no_yield_yet_takes_the_last_one_before_it(cache):
    assert rate_on(cache, date(2024, 1, 1)) == (0.0388, "US Treasury 10Y constant maturity, 2023-12-29")
    assert rate_on(cache, date(2024, 3, 30)) == (0.043, "US Treasury 10Y constant maturity, 2024-03-29")


def test_a_valuation_after_the_last_published_yield_takes_that_yield_and_dates_it(cache):
    # 2025's file holds only a yield published after the valuation date.
    cache.put(TREASURY_URL.format(year=2025), curve({"02/03/2025": 4.50}))
    assert rate_on(cache, date(2025, 2, 1))[1] == "US Treasury 10Y constant maturity, 2024-12-31"


class Counting:
    """The committed closes, counting what is asked of them."""

    name = "csv"

    def __init__(self, directory):
        self.inner = CsvSource(directory)
        self.asked = []

    def fetch(self, symbol, start, end):
        self.asked.append((symbol, start, end))
        return self.inner.fetch(symbol, start, end)


PRICES = __import__("pathlib").Path(__file__).parent / "fixtures" / "prices"


def test_a_span_fetches_each_symbol_once_and_serves_every_window_inside_it():
    from techval.market import SpanSource

    counting = Counting(PRICES)
    span = SpanSource(counting, date(2023, 6, 1), date(2026, 9, 1))
    windows = [(date(2023, 6, 1), date(2025, 3, 31)), (date(2024, 1, 2), date(2026, 9, 1))]
    for start, end in windows:
        sliced = span.fetch("DDOG", start, end)
        direct = CsvSource(PRICES).fetch("DDOG", start, end)
        assert sliced.dates == direct.dates and list(sliced.closes) == list(direct.closes)
    assert counting.asked == [("DDOG", date(2023, 6, 1), date(2026, 9, 1))]


def test_a_window_outside_the_span_goes_to_the_source_as_asked():
    from techval.market import SpanSource

    counting = Counting(PRICES)
    SpanSource(counting, date(2024, 1, 1), date(2025, 1, 1)).fetch("DDOG", date(2023, 6, 1), date(2024, 6, 1))
    assert counting.asked == [("DDOG", date(2023, 6, 1), date(2024, 6, 1))]


def test_a_symbol_the_source_cannot_price_is_refused_every_time_without_asking_again():
    from techval.errors import MissingDataError
    from techval.market import SpanSource

    counting = Counting(PRICES)
    span = SpanSource(counting, date(2024, 1, 1), date(2025, 1, 1))
    for _ in range(2):
        with pytest.raises(MissingDataError):
            span.fetch("NOPE", date(2024, 1, 1), date(2024, 6, 1))
    assert len(counting.asked) == 1


def test_a_history_reads_each_treasury_year_once(monkeypatch):
    """Twelve quarters across three years are three yield files, not twelve."""
    import json
    from pathlib import Path

    import techval.market as M
    from techval.config import Assumptions
    from techval.reverse_dcf import expectations_history, quarter_ends

    asked = []

    def fake_get(url, *, cache, headers=None, **_):
        hit = cache.get(url)
        if hit is not None:
            return hit
        asked.append(url)
        year = int(url.split("daily-treasury-rates.csv/")[1].split("/")[0])
        body = curve({f"{m:02d}/15/{year}": 4.0 for m in range(1, 13)})
        cache.put(url, body)
        return body

    monkeypatch.setattr(M, "http_get", fake_get)
    fixtures = Path(__file__).parent / "fixtures"
    payload = json.loads((fixtures / "companyfacts_DDOG.json").read_text())
    rows = expectations_history(
        payload, "DDOG", CsvSource(fixtures / "prices"), Assumptions(), quarter_ends(date(2024, 1, 1), date(2026, 9, 9))
    )
    assert sum(1 for r in rows if not r["refused"]) >= 8
    years = sorted(int(u.split("daily-treasury-rates.csv/")[1].split("/")[0]) for u in asked)
    assert years == sorted(set(years)) and len(years) <= 4
