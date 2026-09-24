"""The portfolio valued daily, and the return deposits cannot buy.

The value series walks the benchmark's own trading days, so the portfolio and
the benchmark are always compared on the same calendar. The return is time
weighted: on a day with an external flow, the sub-period return divides by the
prior value plus the flow, so putting money in is not performance and taking
it out is not a loss. Contribution is the dollars each position added, three
ways: unrealized against FIFO cost, realized, and dividends.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .ledger import Ledger
from .quotes import Quotes


@dataclass
class Series:
    """Daily portfolio value, oldest first."""

    dates: list[date]
    values: np.ndarray


def value_series(ledger: Ledger, quotes: Quotes) -> Series:
    """Positions plus cash at every benchmark trading day from the first flow."""
    calendar = [
        d
        for d in quotes.series(ledger.benchmark, ledger.kind(ledger.benchmark)).dates
        if d >= ledger.first_date
    ]
    values = []
    for day in calendar:
        positions = ledger.positions(as_of=day)
        total = ledger.cash(as_of=day)
        for p in positions.values():
            if p.shares > 0:
                total += p.shares * quotes.close_on(p.symbol, day, p.kind)
        values.append(total)
    return Series(calendar, np.asarray(values, dtype=float))


def twr(series: Series, flows: dict[date, float]) -> np.ndarray:
    """Cumulative growth of one unit, with external flows stripped.

    The convention: a flow dated ``d`` is treated as arriving before ``d``'s
    close, so the sub-period return on ``d`` is ``V_d / (V_{d-1} + F_d)``.
    Flows dated before the first point are the starting money, not a return.
    """
    growth = np.ones(len(series.dates))
    for i in range(1, len(series.dates)):
        flowed = sum(
            f
            for d, f in flows.items()
            if series.dates[i - 1] < d <= series.dates[i]
        )
        base = series.values[i - 1] + flowed
        step = series.values[i] / base if base > 0 else 1.0
        growth[i] = growth[i - 1] * step
    return growth


def benchmark_growth(quotes: Quotes, benchmark: str, dates: list[date]) -> np.ndarray:
    """The benchmark's growth of one over the same dates."""
    if not dates:
        return np.ones(0)
    first = quotes.close_on(benchmark, dates[0])
    return np.asarray([quotes.close_on(benchmark, d) / first for d in dates])


@dataclass
class Contribution:
    """What one position added, in dollars, by channel."""

    symbol: str
    unrealized: float
    realized: float
    dividends: float

    @property
    def total(self) -> float:
        return self.unrealized + self.realized + self.dividends


def contributions(ledger: Ledger, quotes: Quotes) -> list[Contribution]:
    """Per-symbol dollars added, largest total first, ties by name."""
    rows = []
    for symbol, p in ledger.positions().items():
        market = p.shares * quotes.close_on(symbol, quotes.today, p.kind) if p.shares > 0 else 0.0
        rows.append(
            Contribution(
                symbol=symbol,
                unrealized=market - p.cost,
                realized=p.realized,
                dividends=p.dividends,
            )
        )
    rows.sort(key=lambda r: (-r.total, r.symbol))
    return rows
