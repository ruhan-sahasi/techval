"""The portfolio ledger: what parses, and every way a file can refuse.

A ledger is the owner's own record, so nothing here is forgiving: a row that
cannot be true, a sell of shares never bought, a cash balance below zero, all
refuse with the row named rather than loading something almost right.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from techval.errors import ConfigError
from techval.invest.ledger import KINDS, TYPES, Ledger, Transaction


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "portfolio.yaml"
    path.write_text(body, encoding="utf-8")
    return path


GOOD = """
name: Example
benchmark: SPY
transactions:
  - {date: 2024-02-01, type: deposit, amount: 20000}
  - {date: 2024-02-05, type: buy, symbol: DDOG, shares: 40, price: 120.00}
  - {date: 2024-01-15, type: deposit, amount: 1000}
  - {date: 2024-03-01, type: dividend, symbol: DDOG, amount: 12.50}
  - {date: 2024-04-02, type: buy, symbol: SPY, shares: 10, price: 500.00, kind: etf}
targets:
  DDOG: 0.3
  cash: 0.1
"""


def test_a_good_file_loads_sorted_with_its_targets(tmp_path):
    ledger = Ledger.load(write(tmp_path, GOOD))
    assert ledger.name == "Example"
    assert ledger.benchmark == "SPY"
    assert [t.date for t in ledger.transactions] == sorted(t.date for t in ledger.transactions)
    assert ledger.transactions[0].type == "deposit"
    assert ledger.symbols() == ["DDOG", "SPY"]
    assert ledger.kind("DDOG") == "stock"
    assert ledger.kind("SPY") == "etf"
    assert ledger.targets == {"DDOG": 0.3, "cash": 0.1}
    assert ledger.first_date == date(2024, 1, 15)


def test_the_benchmark_defaults_to_spy(tmp_path):
    ledger = Ledger.load(write(tmp_path, "transactions:\n  - {date: 2024-01-02, type: deposit, amount: 5}\n"))
    assert ledger.benchmark == "SPY"
    assert ledger.name == "Portfolio"


def test_type_and_kind_vocabularies_are_fixed():
    assert TYPES == ("buy", "sell", "deposit", "withdraw", "dividend", "split")
    assert KINDS == ("stock", "etf", "crypto")


@pytest.mark.parametrize(
    "row, phrase",
    [
        ("- {date: 2024-01-02, type: lease, symbol: DDOG, shares: 1, price: 1}", "lease"),
        ("- {date: 2024-01-02, type: buy, symbol: DDOG, shares: 1, price: 1, kind: bond}", "bond"),
        ("- {date: 2024-01-02, type: buy, shares: 1, price: 1}", "symbol"),
        ("- {date: 2024-01-02, type: buy, symbol: DDOG, price: 1}", "shares"),
        ("- {date: 2024-01-02, type: buy, symbol: DDOG, shares: 0, price: 1}", "shares"),
        ("- {date: 2024-01-02, type: buy, symbol: DDOG, shares: 1}", "price"),
        ("- {date: 2024-01-02, type: buy, symbol: DDOG, shares: 1, price: -4}", "price"),
        ("- {date: 2024-01-02, type: deposit}", "amount"),
        ("- {date: 2024-01-02, type: withdraw, amount: -1}", "amount"),
        ("- {date: 2024-01-02, type: dividend, amount: 5}", "symbol"),
        ("- {date: 2024-01-02, type: split, symbol: DDOG}", "ratio"),
        ("- {date: 2024-01-02, type: split, symbol: DDOG, ratio: 0}", "ratio"),
        ("- {date: 02/01/2024, type: deposit, amount: 5}", "date"),
    ],
)
def test_each_impossible_row_refuses_naming_the_field(tmp_path, row, phrase):
    with pytest.raises(ConfigError) as err:
        Ledger.load(write(tmp_path, f"transactions:\n  {row}\n"))
    message = str(err.value)
    assert "transaction 1" in message
    assert phrase in message


def test_an_impossible_calendar_date_refuses_at_parse(tmp_path):
    # YAML itself constructs matching timestamps, so a 13th month dies in the
    # parser rather than in a row; the refusal still says what went wrong.
    with pytest.raises(ConfigError) as err:
        Ledger.load(write(tmp_path, "transactions:\n  - {date: 2024-13-40, type: deposit, amount: 5}\n"))
    assert "impossible" in str(err.value)


def test_a_symbol_may_not_change_kind_between_rows(tmp_path):
    body = """
transactions:
  - {date: 2024-01-02, type: deposit, amount: 100000}
  - {date: 2024-01-03, type: buy, symbol: BTC, shares: 1, price: 40000, kind: crypto}
  - {date: 2024-01-04, type: buy, symbol: BTC, shares: 1, price: 40000, kind: etf}
"""
    with pytest.raises(ConfigError) as err:
        Ledger.load(write(tmp_path, body))
    assert "BTC" in str(err.value) and "crypto" in str(err.value)


def test_a_missing_file_refuses_with_the_path(tmp_path):
    with pytest.raises(ConfigError) as err:
        Ledger.load(tmp_path / "nowhere.yaml")
    assert "nowhere.yaml" in str(err.value)


def test_an_empty_ledger_refuses(tmp_path):
    with pytest.raises(ConfigError):
        Ledger.load(write(tmp_path, "transactions: []\n"))


def test_transactions_are_frozen():
    txn = Transaction(date(2024, 1, 2), "deposit", amount=5.0)
    with pytest.raises(AttributeError):
        txn.amount = 6.0
