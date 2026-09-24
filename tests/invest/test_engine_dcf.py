"""The DCF read: the engine's own valuation pipeline against the market price."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.edgar import CompanyFacts, HttpCache
from techval.invest.engine_read import EngineRead, attach_dcf
from techval.market import CsvSource, MarketData

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
AS_OF = date(2026, 9, 10)


@pytest.fixture(scope="module")
def assumptions() -> Assumptions:
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    return a


@pytest.fixture(scope="module")
def market() -> MarketData:
    return MarketData(CsvSource(FIXTURES / "prices"), HttpCache(enabled=False), today=AS_OF)


def facts_for(symbol: str) -> CompanyFacts | None:
    path = FIXTURES / f"companyfacts_{symbol}.json"
    if not path.is_file():
        return None
    return CompanyFacts(json.loads(path.read_text()), symbol)


def test_ddog_values_at_the_engines_own_number(assumptions, market):
    reads = {"DDOG": EngineRead(symbol="DDOG", covered=True, sub_vertical="infrastructure_software")}
    attach_dcf(reads, facts_for=facts_for, market=market, assumptions=assumptions)
    dcf = reads["DDOG"].dcf
    assert dcf["per_share"] == pytest.approx(36.65, abs=0.02)
    assert dcf["price"] == pytest.approx(market.spot("DDOG"))
    assert dcf["gap_pct"] == pytest.approx(dcf["per_share"] / dcf["price"] - 1.0)
    assert 0.05 < dcf["wacc"] < 0.20


def test_a_name_with_no_facts_refuses_and_moves_on(assumptions, market):
    reads = {
        "ZZXX": EngineRead(symbol="ZZXX", covered=True, sub_vertical="application_software"),
    }
    attach_dcf(reads, facts_for=facts_for, market=market, assumptions=assumptions)
    assert reads["ZZXX"].dcf is None
    assert any(r["what"] == "DCF" and "ZZXX" in r["why"] for r in reads["ZZXX"].refusals)


def test_an_uncovered_name_is_not_valued(assumptions, market):
    reads = {"JPM": EngineRead(symbol="JPM", covered=False, sub_vertical=None)}
    attach_dcf(reads, facts_for=facts_for, market=market, assumptions=assumptions)
    assert reads["JPM"].dcf is None
