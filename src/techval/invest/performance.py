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


def _later_splits(ledger: Ledger, symbol: str, day: date) -> float:
    """The product of the symbol's split ratios dated after ``day``."""
    factor = 1.0
    for t in ledger.transactions:
        if t.type == "split" and t.symbol == symbol and t.date > day:
            factor *= t.ratio
    return factor


def value_series(ledger: Ledger, quotes: Quotes) -> Series:
    """Positions plus cash at every benchmark trading day from the first flow.

    Every source the engine ships restates history for splits, so a close
    from before a split is quoted on the post-split scale. The ledger holds
    the share count as it stood on the day, so each holding is converted into
    today's units, multiplied by every split still to come, before it meets
    that close. Without the conversion a 10:1 split reads as a tenfold gain.
    """
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
                units = p.shares * _later_splits(ledger, p.symbol, day)
                total += units * quotes.close_on(p.symbol, day, p.kind)
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


# Under a year a return is reported as earned, never annualized: a 9% quarter
# compounded to 41% a year is a number nobody earned.
_ANNUALIZE_MIN_DAYS = 365
# Volatility from fewer daily returns than this is noise about noise.
_VOLATILITY_MIN_POINTS = 20
_TRADING_DAYS = 252


def risk_stats(dates: list[date], growth: np.ndarray) -> dict:
    """Annualized return, volatility and the worst drawdown of a growth path.

    The path is a time-weighted growth of one, so a deposit is neither return
    nor drawdown. The return annualizes only over a year or more; volatility
    is the standard deviation of daily log returns scaled by 252 sessions; the
    drawdown is the worst fall from a running peak, with the dates of both.
    """
    growth = np.asarray(growth, dtype=float)
    out: dict = {"annualized": None, "volatility": None, "max_drawdown": 0.0, "drawdown_peak": None, "drawdown_trough": None}
    if len(growth) < 2:
        return out
    days = (dates[-1] - dates[0]).days
    if days >= _ANNUALIZE_MIN_DAYS and growth[-1] > 0:
        out["annualized"] = round(float(growth[-1] ** (365.25 / days) - 1.0), 6)
    if len(growth) - 1 >= _VOLATILITY_MIN_POINTS and np.all(growth > 0):
        daily = np.diff(np.log(growth))
        out["volatility"] = round(float(daily.std(ddof=1) * np.sqrt(_TRADING_DAYS)), 6)
    peaks = np.maximum.accumulate(growth)
    drawdowns = growth / peaks - 1.0
    trough = int(np.argmin(drawdowns))
    if drawdowns[trough] < 0:
        peak = int(np.argmax(growth[: trough + 1]))
        out["max_drawdown"] = round(float(drawdowns[trough]), 6)
        out["drawdown_peak"] = dates[peak].isoformat()
        out["drawdown_trough"] = dates[trough].isoformat()
    return out
