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


def test_prices_are_dated_and_a_mismatched_close_is_named(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        """
benchmark: SPY
transactions:
  - {date: 2024-01-02, type: deposit, amount: 100000}
  - {date: 2024-01-08, type: buy, symbol: DDOG, shares: 10, price: 120}
  - {date: 2024-01-08, type: buy, symbol: DIS, shares: 10, price: 90}
""",
        encoding="utf-8",
    )
    ledger = Ledger.load(path)
    today = date(2026, 9, 28)
    quotes = Quotes(CsvSource(FIXTURES / "prices"), start=ledger.first_date, today=today)
    snap = build_snapshot(ledger, quotes, fixtures=FIXTURES, assumptions=Assumptions(), today=today)
    meta = snap["meta"]
    assert meta["prices_as_of"] == "2026-09-09"
    assert meta["prices_age_days"] == 19
    by = {p["symbol"]: p for p in snap["positions"]}
    assert by["DDOG"]["price_date"] == "2026-09-09"
    assert by["DIS"]["price_date"] == "2026-09-10"
    assert snap["overview"]["mismatched_closes"] == ["DIS"]


def test_a_fresh_book_has_no_mismatched_closes(snapshot):
    assert snapshot["meta"]["prices_as_of"] == "2024-06-03"
    assert snapshot["meta"]["prices_age_days"] == 0
    assert snapshot["overview"]["mismatched_closes"] == []


def test_missing_model_panels_refuse_by_name_and_the_rest_still_builds(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text(BOOK, encoding="utf-8")
    ledger = Ledger.load(path)
    quotes = Quotes(CsvSource(FIXTURES / "prices"), start=ledger.first_date, today=TODAY)
    empty = tmp_path / "no-panels"
    empty.mkdir()
    snap = build_snapshot(ledger, quotes, fixtures=empty, assumptions=Assumptions(), today=TODAY)
    validate_snapshot(snap)
    assert len(snap["positions"]) == 3
    ddog = snap["engine"]["DDOG"]
    assert ddog["fade"] is None and ddog["warranted"] is None
    whats = {r["what"]: r["why"] for r in ddog["refusals"]}
    assert "fade_companyfacts.json.gz" in whats["Fade path"]
    assert "observations.json.gz" in whats["Warranted multiple"]
    assert "--panels" in whats["Fade path"]
    ideas = snap["ideas"]
    assert ideas["cheap"] == [] and ideas["rich"] == [] and ideas["as_of"] is None
    assert "observations.json.gz" in ideas["refusal"]
