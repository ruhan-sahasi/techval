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
