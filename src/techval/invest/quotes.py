"""Prices for a portfolio, through the engine's own sources.

The engine already knows how to fetch and cache daily closes, so this wraps a
``PriceSource`` rather than inventing another one. Two portfolio needs are
added: a close on an arbitrary calendar date takes the last trading day's
close, and a crypto symbol maps to its USD pair. A symbol with no source
coverage refuses; nothing here invents a price.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date

import numpy as np

from ..errors import MissingDataError, TechvalError
from ..market import PriceSeries

# The USD pairs the sources quote. BTC in a ledger is the asset; BTCUSD is the
# series that prices it. An unmapped coin refuses rather than guessing a pair.
CRYPTO_USD = {
    "BTC": "BTCUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "DOGE": "DOGEUSD",
}


class Quotes:
    """Memoized daily closes per symbol, over one window."""

    def __init__(self, source, *, start: date, today: date) -> None:
        self.source = source
        self.start = start
        self.today = today
        self._series: dict[str, PriceSeries] = {}
        self._ledger = None
        # Symbols the source could not serve, priced from the ledger instead,
        # with the source's own refusal kept to show the owner.
        self.failed: dict[str, str] = {}

    def use_ledger_fallback(self, ledger) -> None:
        """Price a holding the source cannot serve at its own last trade price.

        One delisted or mistyped ticker should not take the whole page down.
        The fallback series is the symbol's trades from the ledger, each price
        divided by the splits still to come so it sits on the post-split scale
        the sources use. It never applies to a symbol the ledger never traded,
        and build_snapshot fetches the benchmark before switching it on, so an
        unquotable benchmark still refuses the run.
        """
        self._ledger = ledger

    def _from_ledger(self, symbol: str) -> PriceSeries | None:
        if self._ledger is None:
            return None
        trades = [t for t in self._ledger.transactions if t.symbol == symbol and t.type in ("buy", "sell")]
        if not trades:
            return None
        points: dict[date, float] = {}
        for t in trades:
            later = 1.0
            for s in self._ledger.transactions:
                if s.type == "split" and s.symbol == symbol and s.date > t.date:
                    later *= s.ratio
            points[t.date] = t.price / later
        days = sorted(points)
        return PriceSeries(symbol, days, np.array([points[d] for d in days]), "ledger")

    def _resolve(self, symbol: str, kind: str) -> str:
        symbol = symbol.upper()
        if kind != "crypto":
            return symbol
        pair = CRYPTO_USD.get(symbol)
        if pair is None:
            raise MissingDataError(
                "price history",
                ticker=symbol,
                hint=(
                    f"{symbol} has no USD pair in techval.invest.quotes.CRYPTO_USD; "
                    "add the pair there, or price it through a local CSV"
                ),
            )
        return pair

    def series(self, symbol: str, kind: str = "stock") -> PriceSeries:
        key = self._resolve(symbol, kind)
        if key not in self._series:
            try:
                self._series[key] = self.source.fetch(key, self.start, self.today)
            except TechvalError as err:
                fallback = self._from_ledger(symbol.upper())
                if fallback is None:
                    raise
                self.failed[symbol.upper()] = str(err)
                self._series[key] = fallback
        return self._series[key]

    def close_on(self, symbol: str, day: date, kind: str = "stock") -> float:
        """The close on ``day``, or the last trading day before it."""
        series = self.series(symbol, kind)
        i = bisect_right(series.dates, day)
        if i == 0:
            raise MissingDataError(
                "price history",
                ticker=symbol.upper(),
                hint=f"the {series.source} series starts {series.dates[0]}, after {day}",
            )
        return float(series.closes[i - 1])

    def last_close_date(self, symbol: str, kind: str = "stock") -> date:
        """The session the latest close belongs to, on or before today."""
        series = self.series(symbol, kind)
        i = bisect_right(series.dates, self.today)
        if i == 0:
            raise MissingDataError(
                "price history",
                ticker=symbol.upper(),
                hint=f"the {series.source} series has no close on or before {self.today}",
            )
        return series.dates[i - 1]

    def last_two(self, symbol: str, kind: str = "stock") -> tuple[float, float]:
        """The final close and the one before it, for a day move."""
        series = self.series(symbol, kind)
        if len(series.dates) < 2:
            last = float(series.closes[-1])
            return last, last
        return float(series.closes[-1]), float(series.closes[-2])
