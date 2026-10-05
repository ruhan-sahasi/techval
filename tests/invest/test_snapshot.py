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


def test_risk_statistics_ride_with_performance(snapshot):
    risk = snapshot["performance"]["risk"]
    assert set(risk) == {"portfolio", "benchmark"}
    for side in risk.values():
        assert side["max_drawdown"] <= 0
        assert side["volatility"] is not None and side["volatility"] > 0
    # Five months of history is not annualized.
    assert risk["portfolio"]["annualized"] is None


def test_realized_gains_are_bucketed_by_year_and_term(snapshot):
    years = snapshot["performance"]["realized_by_year"]
    assert [y["year"] for y in years] == [2024]
    # 20 DDOG sold on 2024-05-01 at 115 against a 120 lot opened 2024-01-08.
    assert years[0]["short"] == pytest.approx(20 * (115 - 120))
    assert years[0]["long"] == 0
    assert years[0]["total"] == pytest.approx(snapshot["performance"]["realized_total"])


class PartlyBroken:
    """A source that serves the committed CSVs except for one symbol it has never heard of."""

    name = "partly-broken"

    def __init__(self, missing: str):
        self.missing = missing
        self.inner = CsvSource(FIXTURES / "prices")

    def fetch(self, symbol, start, end):
        from techval.errors import DataSourceError

        if symbol == self.missing:
            raise DataSourceError(f"request for {symbol} failed with HTTP 404 after 5 attempts")
        return self.inner.fetch(symbol, start, end)


def test_an_unquotable_holding_is_priced_from_the_ledger_and_named(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        """
benchmark: SPY
transactions:
  - {date: 2024-01-02, type: deposit, amount: 100000}
  - {date: 2024-01-08, type: buy, symbol: DDOG, shares: 10, price: 120}
  - {date: 2024-01-08, type: buy, symbol: GONE, shares: 100, price: 20}
  - {date: 2024-03-01, type: buy, symbol: GONE, shares: 50, price: 25}
""",
        encoding="utf-8",
    )
    ledger = Ledger.load(path)
    quotes = Quotes(PartlyBroken("GONE"), start=ledger.first_date, today=TODAY)
    snap = build_snapshot(ledger, quotes, fixtures=FIXTURES, assumptions=Assumptions(), today=TODAY)
    validate_snapshot(snap)
    gone = next(p for p in snap["positions"] if p["symbol"] == "GONE")
    assert gone["price"] == 25.0
    assert gone["price_date"] == "2024-03-01"
    assert gone["value"] == pytest.approx(150 * 25.0)
    assert gone["day_pct"] == 0
    [unpriced] = snap["overview"]["unpriced"]
    assert unpriced["symbol"] == "GONE"
    assert "404" in unpriced["reason"]
    assert unpriced["priced_at"] == 25.0 and unpriced["price_date"] == "2024-03-01"
    ddog = next(p for p in snap["positions"] if p["symbol"] == "DDOG")
    assert ddog["price_date"] == "2024-06-03"


def test_an_unquotable_benchmark_still_refuses(tmp_path):
    from techval.errors import TechvalError

    path = tmp_path / "portfolio.yaml"
    path.write_text(BOOK, encoding="utf-8")
    ledger = Ledger.load(path)
    quotes = Quotes(PartlyBroken("SPY"), start=ledger.first_date, today=TODAY)
    with pytest.raises(TechvalError):
        build_snapshot(ledger, quotes, fixtures=FIXTURES, assumptions=Assumptions(), today=TODAY)
