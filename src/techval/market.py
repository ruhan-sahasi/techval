"""Prices and the risk-free rate.

Price history comes from Nasdaq's public quote API, which needs no key and no
account and carries roughly three years of daily closes for listed equities and
ETFs. Sources sit behind one interface so the feed can be swapped without
touching anything that consumes prices.

A note on Stooq, which this package originally targeted: it now answers plain
HTTP clients with a JavaScript proof-of-work challenge rather than CSV. Solving
that would mean defeating a bot check the operator deliberately put up, so the
Stooq source raises an explanatory error instead. Nasdaq is the default and the
CSV source exists for offline and fully reproducible runs.

The market proxy for beta is SPY rather than the S&P 500 index itself, because
the index is not available from a keyless source. The substitution is close to
free for this purpose: SPY's return differs from the index return by its dividend
distributions, and subtracting a roughly constant yield from the market series
shifts the regression intercept, not its slope. It moves alpha, not beta.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np

from .edgar import HttpCache, http_get
from .errors import DataSourceError, MissingDataError

# Nasdaq's quote API and the Treasury CSV endpoint both reject clients whose
# User-Agent they do not recognise, and neither publishes a fair-access policy
# asking automated callers to identify themselves the way the SEC's does. So a
# browser string is sent to those two hosts and nothing is claimed about it.
# This is worth stating plainly rather than leaving in the code, because the SEC
# client one module over makes a point of declaring a real contact address, and
# a reader is entitled to know the two are not held to the same standard here.
# Neither endpoint is rate limited by this package beyond ordinary use: a full
# valuation makes one request per ticker.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

NASDAQ_URL = (
    "https://api.nasdaq.com/api/quote/{symbol}/historical"
    "?assetclass={assetclass}&fromdate={start}&todate={end}&limit=9999"
)
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)

# ETFs are a separate asset class in Nasdaq's API and 404 under "stocks".
_ETFS = {"SPY", "QQQ", "IWM", "VTI", "DIA", "XLK", "IGV"}


@dataclass
class PriceSeries:
    """Daily closes for one symbol, oldest first."""

    symbol: str
    dates: list[date]
    closes: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if len(self.closes) == 0:
            raise MissingDataError(
                "price history",
                ticker=self.symbol,
                hint=(
                    f"{self.source} returned no closes for {self.symbol} inside the "
                    "requested window. A delisted or newly listed ticker, or a CSV "
                    "whose dates fall outside the history period, will do this."
                ),
            )

    @property
    def last(self) -> float:
        return float(self.closes[-1])

    @property
    def last_date(self) -> date:
        return self.dates[-1]

    def window(self, start: date) -> "PriceSeries":
        idx = [i for i, d in enumerate(self.dates) if d >= start]
        return PriceSeries(
            self.symbol,
            [self.dates[i] for i in idx],
            self.closes[idx],
            self.source,
        )

    def fifty_two_week_range(self) -> tuple[float, float]:
        cut = self.last_date - timedelta(days=365)
        w = self.window(cut)
        return float(w.closes.min()), float(w.closes.max())

    def weekly_returns(self) -> tuple[list[date], np.ndarray]:
        """Friday-to-Friday log-equivalent simple returns.

        Weekly sampling is the usual compromise for a two-year beta: daily
        returns of a mid-cap carry enough non-synchronous trading noise to bias
        the slope downward, and monthly returns leave too few observations for a
        two-year window to say anything.
        """
        by_week: dict[tuple[int, int], tuple[date, float]] = {}
        for d, c in zip(self.dates, self.closes):
            iso = d.isocalendar()
            key = (iso[0], iso[1])
            prev = by_week.get(key)
            if prev is None or d > prev[0]:
                by_week[key] = (d, float(c))
        ordered = [by_week[k] for k in sorted(by_week)]
        if len(ordered) < 3:
            return [], np.array([])
        dts = [d for d, _ in ordered][1:]
        px = np.array([p for _, p in ordered])
        return dts, px[1:] / px[:-1] - 1.0


class PriceSource(Protocol):
    name: str

    def fetch(self, symbol: str, start: date, end: date) -> PriceSeries: ...


class NasdaqSource:
    """Nasdaq's public quote API. No key, no account, daily OHLC."""

    name = "nasdaq"

    def __init__(self, cache: HttpCache) -> None:
        self.cache = cache

    def fetch(self, symbol: str, start: date, end: date) -> PriceSeries:
        sym = symbol.upper()
        url = NASDAQ_URL.format(
            symbol=sym,
            assetclass="etf" if sym in _ETFS else "stocks",
            start=start.isoformat(),
            end=end.isoformat(),
        )
        body = http_get(
            url,
            cache=self.cache,
            headers={"User-Agent": _BROWSER_UA, "Accept": "application/json"},
        )
        import json

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise DataSourceError(f"Nasdaq returned non-JSON for {sym}") from exc

        table = (payload.get("data") or {}).get("tradesTable") or {}
        rows = table.get("rows") or []
        if not rows:
            raise MissingDataError(
                "price history",
                ticker=sym,
                hint=f"Nasdaq returned no rows for {sym} between {start} and {end}",
            )

        out: list[tuple[date, float]] = []
        for r in rows:
            try:
                m, d, y = r["date"].split("/")
                close = float(r["close"].replace("$", "").replace(",", ""))
            except (KeyError, ValueError):
                continue
            out.append((date(int(y), int(m), int(d)), close))
        out.sort()
        return PriceSeries(
            sym, [d for d, _ in out], np.array([p for _, p in out]), self.name
        )


class StooqSource:
    """Stooq CSV. Currently unusable; see the module docstring."""

    name = "stooq"

    def __init__(self, cache: HttpCache) -> None:
        self.cache = cache

    def fetch(self, symbol: str, start: date, end: date) -> PriceSeries:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
        body = http_get(url, cache=self.cache, headers={"User-Agent": _BROWSER_UA})
        text = body.decode("utf-8", "replace")
        if not text.lstrip().lower().startswith("date"):
            raise DataSourceError(
                "Stooq answered with a JavaScript bot challenge rather than CSV.\n"
                "  techval will not defeat a bot check. Use the default Nasdaq "
                "source (price_source: nasdaq) or supply local CSVs "
                "(price_source: csv, price_csv_dir: <path>)."
            )
        rows = [
            (date.fromisoformat(r["Date"]), float(r["Close"]))
            for r in csv.DictReader(io.StringIO(text))
            if r.get("Close") not in (None, "", "N/A")
        ]
        rows = [(d, p) for d, p in rows if start <= d <= end]
        rows.sort()
        return PriceSeries(
            symbol.upper(),
            [d for d, _ in rows],
            np.array([p for _, p in rows]),
            self.name,
        )


class CsvSource:
    """Local ``<dir>/<SYMBOL>.csv`` with Date and Close columns.

    The offline path. Tests use it, and it makes a valuation reproducible years
    later from files committed beside the model.
    """

    name = "csv"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def fetch(self, symbol: str, start: date, end: date) -> PriceSeries:
        path = self.directory / f"{symbol.upper()}.csv"
        if not path.exists():
            raise MissingDataError(
                "price history", ticker=symbol, hint=f"no CSV at {path}"
            )
        rows = []
        for r in csv.DictReader(io.StringIO(path.read_text())):
            try:
                d = date.fromisoformat(r["Date"])
                rows.append((d, float(r["Close"])))
            except (KeyError, ValueError):
                continue
        rows = [(d, p) for d, p in rows if start <= d <= end]
        rows.sort()
        return PriceSeries(
            symbol.upper(),
            [d for d, _ in rows],
            np.array([p for _, p in rows]),
            self.name,
        )


def make_price_source(kind: str, cache: HttpCache, csv_dir: str | None = None):
    if kind == "nasdaq":
        return NasdaqSource(cache)
    if kind == "stooq":
        return StooqSource(cache)
    if kind == "csv":
        if not csv_dir:
            raise DataSourceError("price_source: csv requires price_csv_dir")
        return CsvSource(csv_dir)
    raise DataSourceError(f"unknown price source {kind!r}")


class MarketData:
    """Prices and rates for a valuation run, memoized per symbol."""

    def __init__(
        self,
        source: PriceSource,
        cache: HttpCache,
        *,
        today: date,
        history_years: float = 3.0,
    ) -> None:
        self.source = source
        self.cache = cache
        self.today = today
        self.start = today - timedelta(days=int(365.25 * history_years))
        self._series: dict[str, PriceSeries] = {}

    def prices(self, symbol: str) -> PriceSeries:
        key = symbol.upper()
        if key not in self._series:
            self._series[key] = self.source.fetch(key, self.start, self.today)
        return self._series[key]

    def spot(self, symbol: str) -> float:
        return self.prices(symbol).last

    def risk_free_rate(self, override: float | None = None) -> tuple[float, str]:
        """Ten-year constant-maturity Treasury yield, as a decimal.

        The ten-year point is the conventional anchor for a US equity discount
        rate: long enough to match the duration of the cash flows being
        discounted, liquid enough that the quote means something.
        """
        if override is not None:
            return override, "assumptions file override"

        for year in (self.today.year, self.today.year - 1):
            try:
                body = http_get(
                    TREASURY_URL.format(year=year),
                    cache=self.cache,
                    headers={"User-Agent": _BROWSER_UA},
                )
            except DataSourceError:
                continue
            rows = list(csv.DictReader(io.StringIO(body.decode("utf-8", "replace"))))
            parsed: list[tuple[date, float]] = []
            for r in rows:
                raw = r.get("10 Yr") or r.get("10 YR")
                if not raw:
                    continue
                try:
                    m, d, y = r["Date"].split("/")
                    parsed.append((date(int(y), int(m), int(d)), float(raw) / 100.0))
                except (KeyError, ValueError):
                    continue
            if parsed:
                parsed.sort()
                when, rate = parsed[-1]
                return rate, f"US Treasury 10Y constant maturity, {when}"

        raise DataSourceError(
            "could not read the 10Y Treasury yield from the daily curve feed; "
            "set market.risk_free_rate in the assumptions file to proceed"
        )
