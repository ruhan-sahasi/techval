"""The daily value series and the return that cannot be bought with deposits.

The time-weighted case is the one that matters: money doubled by skill reads
+100% whether or not a deposit landed mid-window, and a deposit alone reads 0%.
"""

from __future__ import annotations

from datetime import date, timedelta
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


class AdjustedSource:
    """A vendor-style series: history restated for a 10:1 split on 2024-06-10.

    The stock never moved. Before the split it traded at 1,000; the vendor
    reports those days at 100, the post-split scale, as every source the engine
    ships does.
    """

    name = "adjusted"

    def fetch(self, symbol, start, end):
        from techval.market import PriceSeries

        days = [date(2024, 6, d) for d in (5, 6, 7, 10, 11, 12)]
        return PriceSeries(symbol, days, np.array([100.0] * len(days)), self.name)


def test_a_split_does_not_move_the_value_of_an_unchanged_holding(tmp_path):
    ledger = load(
        tmp_path,
        """
benchmark: BENCH
transactions:
  - {date: 2024-06-05, type: deposit, amount: 10000}
  - {date: 2024-06-05, type: buy, symbol: AAA, shares: 10, price: 1000}
  - {date: 2024-06-10, type: split, symbol: AAA, ratio: 10}
""",
    )
    quotes = Quotes(AdjustedSource(), start=date(2024, 6, 5), today=date(2024, 6, 12))
    series = value_series(ledger, quotes)
    # Ten shares at a restated 100 are a thousand-dollar stake at the time's
    # own price: 10,000 every day, before the split and after it.
    assert list(series.values) == pytest.approx([10000.0] * len(series.dates))
    assert twr(series, ledger.flows())[-1] == pytest.approx(1.0)


def test_max_drawdown_finds_the_worst_peak_to_trough():
    from techval.invest.performance import risk_stats

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(6)]
    growth = np.array([1.0, 1.2, 0.9, 1.1, 0.96, 1.3])
    stats = risk_stats(dates, growth)
    assert stats["max_drawdown"] == pytest.approx(0.9 / 1.2 - 1)
    assert stats["drawdown_peak"] == "2024-01-02"
    assert stats["drawdown_trough"] == "2024-01-03"


def test_a_return_is_annualized_only_over_a_year_or_more():
    from techval.invest.performance import risk_stats

    short = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    assert risk_stats(short, np.linspace(1.0, 1.1, 100))["annualized"] is None
    long = [date(2022, 1, 1), date(2024, 1, 1)]
    stats = risk_stats(long, np.array([1.0, 1.21]))
    assert stats["annualized"] == pytest.approx(1.21 ** (365.25 / 730) - 1, abs=1e-6)


def test_volatility_is_daily_log_returns_scaled_to_a_year():
    from techval.invest.performance import risk_stats

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(41)]
    up, down = 1.01, 1 / 1.01
    growth = np.cumprod([1.0] + [up if i % 2 == 0 else down for i in range(40)])
    stats = risk_stats(dates, growth)
    daily = np.diff(np.log(growth))
    assert stats["volatility"] == pytest.approx(daily.std(ddof=1) * np.sqrt(252), abs=1e-6)


def test_too_few_points_report_no_volatility():
    from techval.invest.performance import risk_stats

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    assert risk_stats(dates, np.ones(5))["volatility"] is None
