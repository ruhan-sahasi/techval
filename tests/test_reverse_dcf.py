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


def test_the_base_case_value_implies_the_assumed_first_year_growth(case):
    from techval.reverse_dcf import implied_first_year_growth

    solve = implied_first_year_growth(case, case.value())
    assert solve.reached
    assert solve.implied == pytest.approx(case.assumptions.dcf.revenue_growth_start, abs=1e-5)


def test_datadogs_price_needs_triple_digit_growth_next_year(case):
    from techval.reverse_dcf import implied_first_year_growth

    solve = implied_first_year_growth(case, PRICE)
    assert solve.reached
    assert 1.0 < solve.implied < 2.0
    assert case.value(revenue_growth_start=solve.implied) == pytest.approx(PRICE, rel=1e-5)
    path = solve.extras["path"]
    years = case.assumptions.dcf.projection_years
    assert len(path) == years
    assert path[0] == pytest.approx(solve.implied)
    assert path[-1] == pytest.approx(case.assumptions.dcf.revenue_growth_terminal)
    assert "next year" in solve.sentence


def test_a_growth_lever_that_cannot_reach_the_price_reports_its_edge(case):
    from techval.reverse_dcf import GROWTH_BRACKET, implied_first_year_growth

    solve = implied_first_year_growth(case, 1e6)
    assert not solve.reached
    assert solve.edge == GROWTH_BRACKET[1]
    assert solve.edge_value < 1e6
    assert "300%" in solve.sentence


def test_the_base_case_value_implies_the_assumed_terminal_margin(case):
    from techval.reverse_dcf import implied_terminal_margin

    solve = implied_terminal_margin(case, case.value())
    assert solve.reached
    assert solve.implied == pytest.approx(case.assumptions.dcf.ebit_margin_terminal, abs=1e-5)


def test_no_terminal_margin_alone_reaches_datadogs_price(case):
    from techval.reverse_dcf import MARGIN_BRACKET, implied_terminal_margin

    solve = implied_terminal_margin(case, PRICE)
    assert not solve.reached
    assert solve.edge == MARGIN_BRACKET[1]
    assert solve.edge_value < PRICE
    assert case.value(ebit_margin_terminal=MARGIN_BRACKET[1]) == pytest.approx(solve.edge_value)
    assert "95%" in solve.sentence


def test_a_reachable_margin_round_trips(case):
    from techval.reverse_dcf import implied_terminal_margin

    target = case.value(ebit_margin_terminal=0.45)
    solve = implied_terminal_margin(case, target)
    assert solve.reached and solve.implied == pytest.approx(0.45, abs=1e-5)
