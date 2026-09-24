"""The daily value series and the return that cannot be bought with deposits.

The time-weighted case is the one that matters: money doubled by skill reads
+100% whether or not a deposit landed mid-window, and a deposit alone reads 0%.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from techval.invest.ledger import Ledger
from techval.invest.performance import (
    Series,
    benchmark_growth,
    contributions,
    twr,
    value_series,
)
from techval.invest.quotes import Quotes
from techval.market import CsvSource

PRICES = Path(__file__).resolve().parents[1] / "fixtures" / "prices"


def load(tmp_path, body: str) -> Ledger:
    path = tmp_path / "portfolio.yaml"
    path.write_text(body, encoding="utf-8")
    return Ledger.load(path)


def quotes_for(ledger: Ledger, today: date) -> Quotes:
    return Quotes(CsvSource(PRICES), start=ledger.first_date, today=today)


def test_the_value_series_walks_benchmark_trading_days(tmp_path):
    ledger = load(
        tmp_path,
        """
benchmark: SPY
transactions:
  - {date: 2024-01-02, type: deposit, amount: 10000}
  - {date: 2024-01-08, type: buy, symbol: DDOG, shares: 10, price: 120}
""",
    )
    quotes = quotes_for(ledger, date(2024, 2, 1))
    series = value_series(ledger, quotes)
    assert series.dates[0] >= date(2024, 1, 2)
    assert series.dates[-1] <= date(2024, 2, 1)
    # Before the buy, the whole value is cash.
    assert series.values[0] == pytest.approx(10000.0)
    # After the buy, cash plus shares at that day's close.
    i = series.dates.index(date(2024, 1, 8))
    close = quotes.close_on("DDOG", date(2024, 1, 8))
    assert series.values[i] == pytest.approx(10000 - 1200 + 10 * close)
    # Weekends do not appear.
    assert all(d.weekday() < 5 for d in series.dates)


def test_twr_strips_a_deposit_out_of_the_return():
    # Invest 100, it doubles, deposit 200 more, flat to the end: money went
    # from 100 to 400 but the return earned is +100%, not +33% and not +300%.
    dates = [date(2024, 1, d) for d in (1, 2, 3, 4)]
    values = np.array([100.0, 200.0, 400.0, 400.0])
    flows = {date(2024, 1, 3): 200.0}
    growth = twr(Series(dates, values), flows)
    assert growth[-1] == pytest.approx(2.0)


def test_twr_of_deposits_alone_is_flat():
    dates = [date(2024, 1, d) for d in (1, 2, 3)]
    values = np.array([100.0, 300.0, 350.0])
    flows = {date(2024, 1, 2): 200.0, date(2024, 1, 3): 50.0}
    growth = twr(Series(dates, values), flows)
    assert growth[-1] == pytest.approx(1.0)


def test_a_withdrawal_does_not_read_as_a_loss():
    dates = [date(2024, 1, d) for d in (1, 2)]
    values = np.array([100.0, 60.0])
    flows = {date(2024, 1, 2): -40.0}
    growth = twr(Series(dates, values), flows)
    assert growth[-1] == pytest.approx(1.0)


def test_benchmark_growth_aligns_to_the_series_dates(tmp_path):
    ledger = load(
        tmp_path,
        """
benchmark: SPY
transactions:
  - {date: 2024-01-02, type: deposit, amount: 1000}
""",
    )
    quotes = quotes_for(ledger, date(2024, 3, 1))
    series = value_series(ledger, quotes)
    growth = benchmark_growth(quotes, "SPY", series.dates)
    assert len(growth) == len(series.dates)
    assert growth[0] == pytest.approx(1.0)
    spy0 = quotes.close_on("SPY", series.dates[0])
    spy_last = quotes.close_on("SPY", series.dates[-1])
    assert growth[-1] == pytest.approx(spy_last / spy0)


def test_contributions_sum_to_the_gain_over_what_was_put_in(tmp_path):
    ledger = load(
        tmp_path,
        """
benchmark: SPY
transactions:
  - {date: 2024-01-02, type: deposit, amount: 50000}
  - {date: 2024-01-08, type: buy, symbol: DDOG, shares: 10, price: 120}
  - {date: 2024-01-08, type: buy, symbol: NET, shares: 20, price: 80}
  - {date: 2024-05-01, type: sell, symbol: NET, shares: 5, price: 90}
  - {date: 2024-06-03, type: dividend, symbol: DDOG, amount: 4}
""",
    )
    today = date(2024, 8, 1)
    quotes = quotes_for(ledger, today)
    rows = contributions(ledger, quotes)
    assert [r.symbol for r in rows] == sorted({"DDOG", "NET"}, key=lambda s: -next(r.total for r in rows if r.symbol == s))
    positions = ledger.positions()
    for row in rows:
        p = positions[row.symbol]
        unrealized = p.shares * quotes.close_on(row.symbol, today) - p.cost
        assert row.unrealized == pytest.approx(unrealized)
        assert row.realized == pytest.approx(p.realized)
        assert row.dividends == pytest.approx(p.dividends)
        assert row.total == pytest.approx(unrealized + p.realized + p.dividends)
