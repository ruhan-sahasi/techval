"""Quotes: the engine's own price sources, wrapped for a portfolio's needs."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from techval.errors import MissingDataError
from techval.market import CsvSource, PriceSeries
from techval.invest.quotes import CRYPTO_USD, Quotes

PRICES = Path(__file__).resolve().parents[1] / "fixtures" / "prices"


def csv_quotes() -> Quotes:
    return Quotes(CsvSource(PRICES), start=date(2023, 9, 1), today=date(2026, 9, 9))


def test_a_series_loads_once_and_is_memoized():
    quotes = csv_quotes()
    first = quotes.series("DDOG")
    assert first is quotes.series("ddog")
    assert first.dates[0] >= date(2023, 9, 1)


def test_close_on_a_weekend_takes_the_prior_close():
    quotes = csv_quotes()
    friday = quotes.close_on("DDOG", date(2024, 3, 1))
    saturday = quotes.close_on("DDOG", date(2024, 3, 2))
    sunday = quotes.close_on("DDOG", date(2024, 3, 3))
    assert friday == saturday == sunday


def test_close_before_the_first_date_refuses():
    with pytest.raises(MissingDataError):
        csv_quotes().close_on("DDOG", date(2023, 8, 1))


def test_last_two_are_the_final_two_closes():
    quotes = csv_quotes()
    series = quotes.series("SPY")
    last, previous = quotes.last_two("SPY")
    assert last == series.closes[-1]
    assert previous == series.closes[-2]


class Recorder:
    """A source that records what symbol was asked of it."""

    name = "recorder"

    def __init__(self):
        self.asked: list[str] = []

    def fetch(self, symbol, start, end):
        self.asked.append(symbol)
        return PriceSeries(symbol, [start], np.array([1.0]), self.name)


def test_crypto_maps_to_its_usd_pair():
    source = Recorder()
    quotes = Quotes(source, start=date(2024, 1, 1), today=date(2024, 6, 1))
    quotes.series("BTC", kind="crypto")
    assert source.asked == ["BTCUSD"]
    assert CRYPTO_USD["ETH"] == "ETHUSD"


def test_an_unmapped_crypto_refuses_with_a_hint():
    quotes = Quotes(Recorder(), start=date(2024, 1, 1), today=date(2024, 6, 1))
    with pytest.raises(MissingDataError) as err:
        quotes.series("WEIRDCOIN", kind="crypto")
    assert "CRYPTO_USD" in str(err.value.hint or err.value)
