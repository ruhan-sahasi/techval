"""Fixtures. No test in this suite touches the network.

Company facts are frozen SEC `companyfacts` payloads retrieved on 2026-09-10,
pruned to the us-gaap concepts techval reads and to facts ending on or after
2023-01-01. Prices are frozen daily closes from the same date. Between them the
whole engine runs offline and deterministically.

The companies are chosen for what they break:

    DDOG  calendar fiscal year, deep in-the-money convertibles, thin GAAP EBITDA
    CRWD  January fiscal year end, a four-for-one split mid-2026, no combined
          D&A tag, non-controlling interest
    MDB   January fiscal year end, finance leases as well as operating leases
    ZS    January fiscal year end, marketable securities under a tag outside the
          obvious ladder
    VZ    debt under a concept that bundles finance leases with it, a current
          portion under a third concept, and a LongTermDebtNoncurrent whose
          newest value is dated 2013
    DIS   segment reporting, and a media pack the TMT commands read
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.edgar import CompanyFacts, HttpCache
from techval.ev_bridge import build_ev_bridge
from techval.financials import build_financials
from techval.market import CsvSource, MarketData

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = FIXTURES / "prices"

# The date the fixtures were retrieved. Pinning it keeps the 52-week window and
# the beta lookback fixed, so every assertion below stays true indefinitely.
AS_OF = date(2026, 9, 10)


def load_facts(ticker: str) -> CompanyFacts:
    path = FIXTURES / f"companyfacts_{ticker.upper()}.json"
    return CompanyFacts(json.loads(path.read_text()), ticker)


class FixtureClient:
    """Stands in for EdgarClient, reading the committed payloads."""

    def company_facts(self, ticker: str) -> CompanyFacts:
        return load_facts(ticker)

    def ticker_to_cik(self, ticker: str) -> int:
        return load_facts(ticker).cik


@pytest.fixture
def client() -> FixtureClient:
    return FixtureClient()


@pytest.fixture
def market() -> MarketData:
    return MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=AS_OF)


@pytest.fixture
def assumptions() -> Assumptions:
    """Defaults, with a risk-free rate pinned so no test reaches the Treasury."""
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    a.comps.peers = ["CRWD", "MDB", "ZS"]
    return a


@pytest.fixture
def ddog():
    return build_financials("DDOG", facts=load_facts("DDOG"))


@pytest.fixture
def crwd():
    return build_financials("CRWD", facts=load_facts("CRWD"))


@pytest.fixture
def mdb():
    return build_financials("MDB", facts=load_facts("MDB"))


@pytest.fixture
def zs():
    return build_financials("ZS", facts=load_facts("ZS"))


@pytest.fixture
def vz():
    """Verizon: the debt ladder's worst case, and the reason it was rewritten.

    Long-term debt under ``LongTermDebtAndCapitalLeaseObligations`` with finance
    leases inside it, a current portion under a different concept again, and a
    ``LongTermDebtNoncurrent`` whose newest value is dated 2013. The fixture
    keeps the debt and lease concepts back to the first filing rather than
    trimming them at 2023, because the thirteen-year-old value IS the test.
    """
    return build_financials("VZ", facts=load_facts("VZ"))


@pytest.fixture
def ddog_bridge(ddog, assumptions, market):
    return build_ev_bridge(ddog, market.spot("DDOG"), assumptions)
