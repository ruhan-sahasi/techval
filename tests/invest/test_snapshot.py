"""The snapshot: everything the page reads, assembled once and validated."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.invest.ledger import Ledger
from techval.invest.quotes import Quotes
from techval.invest.snapshot import build_snapshot, validate_snapshot, write_snapshot
from techval.market import CsvSource

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

BOOK = """
name: Example book
benchmark: SPY
transactions:
  - {date: 2024-01-02, type: deposit, amount: 100000}
  - {date: 2024-01-08, type: buy, symbol: DDOG, shares: 100, price: 120}
  - {date: 2024-01-08, type: buy, symbol: DIS, shares: 100, price: 90}
  - {date: 2024-01-08, type: buy, symbol: SPY, shares: 50, price: 470, kind: etf}
  - {date: 2024-05-01, type: sell, symbol: DDOG, shares: 20, price: 115}
targets:
  DDOG: 0.20
"""

TODAY = date(2024, 6, 3)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    path = tmp_path_factory.mktemp("book") / "portfolio.yaml"
    path.write_text(BOOK, encoding="utf-8")
    ledger = Ledger.load(path)
    quotes = Quotes(CsvSource(FIXTURES / "prices"), start=ledger.first_date, today=TODAY)
    return build_snapshot(
        ledger, quotes, fixtures=FIXTURES, assumptions=Assumptions(), today=TODAY
    )


def test_the_snapshot_validates_and_carries_every_pane(snapshot):
    validate_snapshot(snapshot)
    assert set(snapshot) == {
        "meta", "overview", "positions", "performance", "hygiene", "engine", "ideas", "activity",
    }
    assert snapshot["meta"]["name"] == "Example book"
    assert snapshot["meta"]["generated"] == TODAY.isoformat()


def test_the_overview_adds_up(snapshot):
    over = snapshot["overview"]
    total = sum(p["value"] for p in snapshot["positions"]) + over["cash"]
    assert over["value"] == pytest.approx(total)
    assert over["n_positions"] == 3
    assert over["cheap"] + over["rich"] + over["uncovered"] == 3
    assert len(over["movers"]) <= 3


def test_positions_carry_weights_that_sum_with_cash_to_one(snapshot):
    weights = sum(p["weight"] for p in snapshot["positions"])
    over = snapshot["overview"]
    assert weights + over["cash"] / over["value"] == pytest.approx(1.0)
    ddog = next(p for p in snapshot["positions"] if p["symbol"] == "DDOG")
    assert ddog["shares"] == pytest.approx(80)
    assert ddog["engine"] is not None and ddog["engine"]["call"] in ("rich", "cheap")
    spy = next(p for p in snapshot["positions"] if p["symbol"] == "SPY")
    assert spy["engine"] is None and spy["kind"] == "etf"


def test_performance_series_share_one_calendar(snapshot):
    perf = snapshot["performance"]
    assert len(perf["dates"]) == len(perf["growth"]) == len(perf["benchmark_growth"])
    assert perf["growth"][0] == pytest.approx(1.0)
    assert perf["benchmark_growth"][0] == pytest.approx(1.0)


def test_engine_pane_has_a_read_per_symbol(snapshot):
    assert set(snapshot["engine"]) == {"DDOG", "DIS", "SPY"}
    assert snapshot["engine"]["SPY"]["covered"] is False
    assert snapshot["engine"]["DDOG"]["fade"] is not None
    # No facts loader was passed, so the DCF pane is a refusal, not a number.
    assert snapshot["engine"]["DDOG"]["dcf"] is None
    assert any(r["what"] == "DCF" for r in snapshot["engine"]["DDOG"]["refusals"])


def test_activity_is_newest_first(snapshot):
    days = [row["date"] for row in snapshot["activity"]]
    assert days == sorted(days, reverse=True)


def test_the_validator_names_a_missing_key(snapshot):
    broken = json.loads(json.dumps(snapshot))
    del broken["overview"]["twr_pct"]
    with pytest.raises(ValueError) as err:
        validate_snapshot(broken)
    assert "twr_pct" in str(err.value)


def test_write_is_byte_stable(snapshot, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_snapshot(snapshot, a)
    write_snapshot(snapshot, b)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes().endswith(b"\n")
