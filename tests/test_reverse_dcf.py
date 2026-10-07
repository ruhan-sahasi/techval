"""The DCF run backwards: what the market price assumes, lever by lever.

Datadog is the case: a 36.65 base case against a 225.27 close. Every solve is
checked the only way that matters, by putting the implied value back into the
engine's own DCF and getting the price out.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.dcf import run_dcf
from techval.edgar import CompanyFacts, HttpCache
from techval.ev_bridge import build_ev_bridge
from techval.financials import build_financials
from techval.market import CsvSource, MarketData
from techval.reverse_dcf import Case, implied_discount_rate, solve_monotone
from techval.wacc import compute_wacc

FIXTURES = Path(__file__).parent / "fixtures"
PRICE = 225.27


@pytest.fixture(scope="module")
def case() -> Case:
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    facts = CompanyFacts(json.loads((FIXTURES / "companyfacts_DDOG.json").read_text()), "DDOG")
    fin = build_financials("DDOG", facts=facts)
    market = MarketData(CsvSource(FIXTURES / "prices"), HttpCache(enabled=False), today=date(2026, 9, 10))
    bridge = build_ev_bridge(fin, market.spot("DDOG"), a)
    wacc = compute_wacc(fin, bridge, market, a)
    return Case(fin=fin, bridge=bridge, wacc=wacc, assumptions=a)


def test_the_solver_finds_a_root_of_a_monotone_function():
    root = solve_monotone(lambda x: x**3, target=8.0, lo=0.0, hi=5.0)
    assert root.reached and root.x == pytest.approx(2.0, abs=1e-6)


def test_the_solver_reports_the_edge_when_the_target_is_out_of_reach():
    miss = solve_monotone(lambda x: x, target=9.0, lo=0.0, hi=5.0)
    assert not miss.reached and miss.x is None
    assert miss.edge_x == 5.0 and miss.edge_value == 5.0


def test_the_solver_handles_a_decreasing_function():
    root = solve_monotone(lambda x: 10 - x, target=4.0, lo=0.0, hi=10.0)
    assert root.reached and root.x == pytest.approx(6.0, abs=1e-6)


def test_the_base_case_value_implies_the_engines_own_discount_rate(case):
    base = case.value()
    solve = implied_discount_rate(case, base)
    assert solve.reached
    assert solve.implied == pytest.approx(case.wacc.wacc, abs=1e-6)
    assert solve.assumed == pytest.approx(case.wacc.wacc)


def test_datadogs_price_implies_a_return_under_four_percent(case):
    solve = implied_discount_rate(case, PRICE)
    assert solve.reached
    assert 0.035 < solve.implied < 0.04
    # The round trip: the engine's DCF at the implied rate is the price.
    assert case.value(wacc_override=solve.implied) == pytest.approx(PRICE, rel=1e-5)
    assert "a year" in solve.sentence


def test_a_price_below_any_plausible_value_says_so(case):
    solve = implied_discount_rate(case, 0.01)
    assert not solve.reached and solve.implied is None
    assert solve.edge == 0.40
    assert "40%" in solve.sentence
