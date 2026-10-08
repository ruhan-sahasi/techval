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


def test_duration_is_the_fewest_whole_years_of_held_growth_that_reach_the_price(case):
    from techval.reverse_dcf import DURATION_HORIZON, duration_path, implied_duration

    solve = implied_duration(case, PRICE, held_growth=0.277)
    assert solve.reached
    k = int(solve.implied)
    assert 10 <= k <= DURATION_HORIZON
    value = lambda years: case.value(growth_path=duration_path(0.277, years, case), projection_years=DURATION_HORIZON)
    assert value(k) >= PRICE > value(k - 1)
    assert solve.extras["value_held_zero"] == pytest.approx(value(0))
    assert len(solve.extras["path"]) == DURATION_HORIZON
    assert solve.extras["path"][:k] == [pytest.approx(0.277)] * k
    assert "27.7%" in solve.sentence and str(k) in solve.sentence


def test_a_duration_the_horizon_cannot_hold_reports_fifteen_years_short(case):
    from techval.reverse_dcf import DURATION_HORIZON, implied_duration

    solve = implied_duration(case, 1e6, held_growth=0.277)
    assert not solve.reached
    assert solve.edge == DURATION_HORIZON
    assert "15 years" in solve.sentence


def test_the_duration_path_holds_then_fades_to_terminal(case):
    from techval.reverse_dcf import duration_path

    path = duration_path(0.30, 10, case)
    assert path[:10] == [0.30] * 10
    assert path[-1] == pytest.approx(case.assumptions.dcf.revenue_growth_terminal)
    assert all(a >= b for a, b in zip(path[10:], path[11:]))


class Obs:
    """A stand-in panel row: trailing growth and forward growth by horizon."""

    def __init__(self, growth, labels):
        self.growth = growth
        self.labels = labels


def test_path_cagr_compounds_the_first_h_years():
    from techval.reverse_dcf import path_cagr

    assert path_cagr([0.10, 0.10, 0.10], 3) == pytest.approx(0.10)
    assert path_cagr([1.0, 0.0], 2) == pytest.approx(2 ** 0.5 - 1)


def test_base_rates_count_realised_cagrs_at_or_above_the_implied_one():
    from techval.reverse_dcf import growth_base_rate

    panel = [
        Obs(0.30, {1: 0.40, 2: 0.40}),
        Obs(0.35, {1: 0.20, 2: 0.10}),
        Obs(0.05, {1: 0.50, 2: 0.50}),
        Obs(0.30, {1: 0.40}),  # no two-year label: not counted at h=2
    ]
    rate = growth_base_rate(panel, implied_cagr=0.30, horizon=2, trailing=0.32)
    assert (rate["all"]["n"], rate["all"]["hits"]) == (3, 2)
    assert rate["all"]["share"] == pytest.approx(2 / 3)
    # Similar starters: trailing growth within ten points of 32%.
    assert (rate["similar"]["n"], rate["similar"]["hits"]) == (2, 1)
    assert rate["horizon"] == 2 and rate["implied_cagr"] == 0.30


def test_without_a_trailing_rate_there_is_no_similar_bucket():
    from techval.reverse_dcf import growth_base_rate

    rate = growth_base_rate([Obs(0.1, {1: 0.2})], implied_cagr=0.1, horizon=1)
    assert rate["similar"] is None


@pytest.fixture(scope="module")
def panel():
    from techval.invest.engine_read import fade_panel

    return fade_panel(FIXTURES)


def test_datadogs_implied_growth_has_almost_never_happened(case, panel):
    from techval.reverse_dcf import growth_base_rate, implied_first_year_growth, path_cagr

    solve = implied_first_year_growth(case, PRICE)
    implied = path_cagr(solve.extras["path"], 5)
    rate = growth_base_rate(panel.observations, implied_cagr=implied, horizon=5, trailing=0.277)
    assert rate["all"]["n"] > 1000
    assert rate["all"]["share"] < 0.02
    assert rate["similar"]["n"] > 100


def test_the_frontier_trades_margin_for_growth(case):
    from techval.reverse_dcf import FRONTIER_MARGINS, growth_margin_frontier, implied_first_year_growth

    rows = growth_margin_frontier(case, PRICE)
    assert [r["margin"] for r in rows] == list(FRONTIER_MARGINS)
    reached = [r["implied_growth"] for r in rows if r["implied_growth"] is not None]
    assert len(reached) >= 3
    # A fatter margin needs less growth.
    assert all(a > b for a, b in zip(reached, reached[1:]))
    at_assumed = next(r for r in rows if r["margin"] == case.assumptions.dcf.ebit_margin_terminal)
    assert at_assumed["implied_growth"] == pytest.approx(implied_first_year_growth(case, PRICE).implied, abs=1e-5)


def test_with_dcf_leaves_the_original_case_alone(case):
    before = case.assumptions.dcf.ebit_margin_terminal
    other = case.with_dcf(ebit_margin_terminal=0.5)
    assert other.assumptions.dcf.ebit_margin_terminal == 0.5
    assert case.assumptions.dcf.ebit_margin_terminal == before


def test_market_expectations_gathers_every_lever_with_base_rates(case, panel):
    import json as _json

    from techval.reverse_dcf import market_expectations

    exp = market_expectations(case, PRICE, trailing_growth=0.277, observations=panel.observations)
    assert exp.price == PRICE and exp.base_value == pytest.approx(case.value())
    assert {s.lever for s in exp.solves} == {"discount_rate", "first_year_growth", "terminal_margin", "duration"}
    assert exp.base_rates["first_year_growth"]["all"]["share"] < 0.02
    assert exp.base_rates["duration"]["horizon"] == 5
    text = " ".join(exp.summary())
    assert "3.71% a year" in text
    assert "155.2%" in text
    assert "of 1,662" in text
    # The panel stops at five years, and the summary says so for a longer duration.
    assert "nothing in it tests" in text
    payload = exp.to_dict()
    _json.dumps(payload)
    assert payload["frontier"][0]["margin"] == 0.10


def test_without_a_panel_there_are_no_base_rates_and_it_says_why(case):
    from techval.reverse_dcf import market_expectations

    exp = market_expectations(case, PRICE, trailing_growth=None)
    assert exp.base_rates == {}
    assert any("no fade panel" in s for s in exp.summary())
    # Without a trailing rate, the duration holds the assumed first-year growth.
    duration = next(s for s in exp.solves if s.lever == "duration")
    assert duration.extras["held_growth"] == case.assumptions.dcf.revenue_growth_start


def test_the_persistence_curve_counts_who_held_a_rate_for_each_horizon():
    from techval.reverse_dcf import persistence_curve

    panel = [
        Obs(0.30, {1: 0.30, 2: 0.31, 3: 0.40}),  # held 3 years
        Obs(0.30, {1: 0.35, 2: 0.10, 3: 0.50}),  # broke in year 2
        Obs(0.30, {1: 0.20}),  # broke in year 1, and no later labels
        Obs(0.30, {1: 0.40, 2: 0.40}),  # held 2, no year 3 label
    ]
    # One cohort for every horizon, the company-years labelled all three years
    # out, so each year can only lose companies and the curve never rises.
    curve = persistence_curve(panel, growth=0.30, horizons=3)
    assert [(c["horizon"], c["n"], c["held"]) for c in curve] == [(1, 2, 2), (2, 2, 1), (3, 2, 1)]
    assert curve[1]["share"] == pytest.approx(0.5)


def test_datadogs_rate_rarely_lasts_five_years(panel):
    from techval.reverse_dcf import persistence_curve

    curve = persistence_curve(panel.observations, growth=0.277, horizons=5)
    shares = [c["share"] for c in curve]
    # Each extra year can only lose companies that held every year before it.
    assert all(a >= b for a, b in zip(shares, shares[1:]))
    assert shares[0] > shares[-1]
    assert shares[-1] < 0.10


def test_expectations_carry_the_persistence_curve_for_the_held_rate(case, panel):
    from techval.reverse_dcf import market_expectations

    exp = market_expectations(case, PRICE, trailing_growth=0.277, observations=panel.observations)
    assert [c["horizon"] for c in exp.persistence] == [1, 2, 3, 4, 5]
    text = " ".join(exp.summary())
    assert "year after year" in text and "for five" in text
    assert exp.to_dict()["persistence"] == exp.persistence
