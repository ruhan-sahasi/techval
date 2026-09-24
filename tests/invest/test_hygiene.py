"""Hygiene: weights, concentration, exposure, drift and coverage.

Arithmetic on the owner's own book and targets. The only judgement here is the
bucketing: TMT names take their sub-vertical from the fade universe, ETFs and
crypto and cash are their own buckets, and a stock the engine has no taxonomy
for is 'outside TMT' rather than silently classified.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from techval.invest.hygiene import coverage, drift, exposure, top_share, weights
from techval.invest.ledger import Ledger
from techval.invest.quotes import Quotes
from techval.market import CsvSource

PRICES = Path(__file__).resolve().parents[1] / "fixtures" / "prices"

BOOK = """
benchmark: SPY
transactions:
  - {date: 2024-01-02, type: deposit, amount: 100000}
  - {date: 2024-01-08, type: buy, symbol: DDOG, shares: 100, price: 120}
  - {date: 2024-01-08, type: buy, symbol: DIS, shares: 100, price: 90}
  - {date: 2024-01-08, type: buy, symbol: SPY, shares: 50, price: 470, kind: etf}
targets:
  DDOG: 0.20
  SPY: 0.50
  cash: 0.05
"""


def book(tmp_path) -> tuple[Ledger, Quotes]:
    path = tmp_path / "portfolio.yaml"
    path.write_text(BOOK, encoding="utf-8")
    ledger = Ledger.load(path)
    return ledger, Quotes(CsvSource(PRICES), start=ledger.first_date, today=date(2024, 6, 3))


def test_weights_cover_the_whole_book_and_sum_to_one(tmp_path):
    ledger, quotes = book(tmp_path)
    rows = weights(ledger, quotes)
    assert [r.symbol for r in rows][-1] == "cash"
    assert sum(r.weight for r in rows) == pytest.approx(1.0)
    assert all(r.value > 0 for r in rows)
    by = {r.symbol: r for r in rows}
    assert by["SPY"].kind == "etf"
    assert by["DDOG"].value == pytest.approx(100 * quotes.close_on("DDOG", date(2024, 6, 3)))


def test_top_share_takes_the_largest_n(tmp_path):
    ledger, quotes = book(tmp_path)
    rows = weights(ledger, quotes)
    assert top_share(rows, n=1) == pytest.approx(max(r.weight for r in rows))
    assert top_share(rows, n=len(rows)) == pytest.approx(1.0)


def test_exposure_buckets_tmt_etf_and_cash(tmp_path):
    ledger, quotes = book(tmp_path)
    buckets = exposure(weights(ledger, quotes))
    assert set(buckets) == {"infrastructure software", "media entertainment", "index funds", "cash"}
    assert sum(buckets.values()) == pytest.approx(1.0)


def test_drift_measures_the_gap_to_each_target(tmp_path):
    ledger, quotes = book(tmp_path)
    rows = weights(ledger, quotes)
    gaps = drift(rows, ledger.targets)
    assert [g["symbol"] for g in gaps] == ["DDOG", "SPY", "cash"]
    by = {r.symbol: r.weight for r in rows}
    for g in gaps:
        assert g["gap"] == pytest.approx(by[g["symbol"]] - g["target"])


def test_drift_without_targets_is_empty(tmp_path):
    ledger, quotes = book(tmp_path)
    assert drift(weights(ledger, quotes), {}) == []


def test_coverage_is_the_share_of_value_the_engine_can_value(tmp_path):
    ledger, quotes = book(tmp_path)
    rows = weights(ledger, quotes)
    cov = coverage(rows)
    by = {r.symbol: r for r in rows}
    covered = by["DDOG"].value + by["DIS"].value
    total = sum(r.value for r in rows)
    assert cov["covered_value_share"] == pytest.approx(covered / total)
    assert cov["covered_positions"] == 2
    assert cov["positions"] == 3
