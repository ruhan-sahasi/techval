"""Positions by FIFO lot: hand-computed cases that pin the accounting.

The numbers here are small enough to check on paper, which is the point: a
partial sell consumes the oldest lot first, a split multiplies shares and
divides the cost, dividends land in cash and on the position, and cash can
never go below zero without the ledger refusing.
"""

from __future__ import annotations

from datetime import date

import pytest

from techval.errors import ConfigError
from techval.invest.ledger import Ledger


def load(tmp_path, body: str) -> Ledger:
    path = tmp_path / "portfolio.yaml"
    path.write_text(body, encoding="utf-8")
    return Ledger.load(path)


BOOK = """
transactions:
  - {date: 2024-01-02, type: deposit, amount: 10000}
  - {date: 2024-01-03, type: buy, symbol: AAA, shares: 10, price: 100}
  - {date: 2024-02-01, type: buy, symbol: AAA, shares: 10, price: 200}
  - {date: 2024-03-01, type: sell, symbol: AAA, shares: 15, price: 300}
"""


def test_a_partial_sell_consumes_the_oldest_lot_first(tmp_path):
    ledger = load(tmp_path, BOOK)
    positions = ledger.positions()
    aaa = positions["AAA"]
    assert aaa.shares == pytest.approx(5.0)
    # The first lot of 10 at 100 is gone, 5 of the 200 lot were sold too.
    assert len(aaa.lots) == 1
    assert aaa.lots[0].cost_per_share == pytest.approx(200.0)
    assert aaa.cost == pytest.approx(5 * 200.0)
    # Proceeds 15 x 300 = 4500 against basis 10 x 100 + 5 x 200 = 2000.
    assert aaa.realized == pytest.approx(2500.0)
    # Cash: 10000 - 1000 - 2000 + 4500.
    assert ledger.cash() == pytest.approx(11500.0)


def test_positions_as_of_an_earlier_date_ignore_the_future(tmp_path):
    ledger = load(tmp_path, BOOK)
    positions = ledger.positions(as_of=date(2024, 1, 31))
    assert positions["AAA"].shares == pytest.approx(10.0)
    assert ledger.cash(as_of=date(2024, 1, 31)) == pytest.approx(9000.0)


def test_a_split_multiplies_shares_and_divides_the_cost(tmp_path):
    ledger = load(
        tmp_path,
        """
transactions:
  - {date: 2024-01-02, type: deposit, amount: 5000}
  - {date: 2024-01-03, type: buy, symbol: BBB, shares: 4, price: 1000}
  - {date: 2024-06-10, type: split, symbol: BBB, ratio: 10}
  - {date: 2024-07-01, type: sell, symbol: BBB, shares: 30, price: 120}
""",
    )
    bbb = ledger.positions()["BBB"]
    assert bbb.shares == pytest.approx(10.0)
    assert bbb.lots[0].cost_per_share == pytest.approx(100.0)
    # Basis consumed 30 x 100 = 3000 against proceeds 3600.
    assert bbb.realized == pytest.approx(600.0)


def test_dividends_accrue_to_cash_and_to_the_position(tmp_path):
    ledger = load(
        tmp_path,
        """
transactions:
  - {date: 2024-01-02, type: deposit, amount: 1000}
  - {date: 2024-01-03, type: buy, symbol: CCC, shares: 5, price: 100}
  - {date: 2024-04-01, type: dividend, symbol: CCC, amount: 12.5}
""",
    )
    assert ledger.positions()["CCC"].dividends == pytest.approx(12.5)
    assert ledger.cash() == pytest.approx(512.5)


def test_flows_carry_deposits_and_withdrawals_only(tmp_path):
    ledger = load(
        tmp_path,
        """
transactions:
  - {date: 2024-01-02, type: deposit, amount: 1000}
  - {date: 2024-01-03, type: buy, symbol: DDD, shares: 5, price: 100}
  - {date: 2024-02-01, type: deposit, amount: 250}
  - {date: 2024-03-01, type: withdraw, amount: 100}
""",
    )
    assert ledger.flows() == {date(2024, 1, 2): 1000.0, date(2024, 2, 1): 250.0, date(2024, 3, 1): -100.0}


def test_selling_more_than_held_refuses_with_the_date(tmp_path):
    with pytest.raises(ConfigError) as err:
        load(
            tmp_path,
            """
transactions:
  - {date: 2024-01-02, type: deposit, amount: 1000}
  - {date: 2024-01-03, type: buy, symbol: EEE, shares: 5, price: 100}
  - {date: 2024-02-01, type: sell, symbol: EEE, shares: 6, price: 100}
""",
        ).positions()
    assert "EEE" in str(err.value) and "2024-02-01" in str(err.value)


def test_cash_below_zero_refuses_with_the_date(tmp_path):
    with pytest.raises(ConfigError) as err:
        load(
            tmp_path,
            """
transactions:
  - {date: 2024-01-02, type: deposit, amount: 100}
  - {date: 2024-01-03, type: buy, symbol: FFF, shares: 5, price: 100}
""",
        ).positions()
    assert "2024-01-03" in str(err.value) and "cash" in str(err.value).lower()


def test_a_dividend_on_shares_never_held_refuses(tmp_path):
    with pytest.raises(ConfigError) as err:
        load(
            tmp_path,
            """
transactions:
  - {date: 2024-01-02, type: deposit, amount: 100}
  - {date: 2024-01-03, type: dividend, symbol: GGG, amount: 5}
""",
        ).positions()
    assert "GGG" in str(err.value)
