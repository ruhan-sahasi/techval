"""Bear, base and bull from what similar company-years went on to do.

The scenarios are not opinions. Each is the mean year-by-year revenue growth of
the company-years around one percentile of realised five-year growth, among
those that started near the company's own growth rate, run through the
engine's own DCF with everything else held at the base case.
"""

from __future__ import annotations

import pytest

from techval.scenarios import quantile_paths, realised_paths


class Obs:
    """A stand-in panel row: trailing growth, revenue and forward growth by horizon."""

    def __init__(self, growth, labels, revenue=1e9):
        self.growth = growth
        self.labels = labels
        self.revenue = revenue


def flat(rate, horizon=5):
    return {h: rate for h in range(1, horizon + 1)}


def test_only_company_years_with_every_label_and_a_similar_start_are_kept():
    panel = [
        Obs(0.30, flat(0.20)),
        Obs(0.25, flat(0.10)),
        Obs(0.30, {1: 0.4, 2: 0.4}),  # labels stop at two years: not a five-year outcome
        Obs(0.05, flat(0.50)),  # started far from 30%
        Obs(None, flat(0.30)),  # no trailing growth to compare
    ]
    pool = realised_paths(panel, trailing=0.30, min_similar=1)
    assert pool.paths == [[0.20] * 5, [0.10] * 5]
    assert pool.similar and pool.n == 2


def test_too_few_similar_starters_widen_the_pool_to_every_company_year_and_say_so():
    panel = [Obs(0.30, flat(0.2)), Obs(0.05, flat(0.5)), Obs(0.9, flat(0.1))]
    pool = realised_paths(panel, trailing=0.30, min_similar=2)
    assert not pool.similar and pool.n == 3
    assert "fewer than 2" in pool.note


def test_each_scenario_is_the_mean_path_of_its_percentile_neighbourhood():
    # 100 company-years growing 0%, 1%, ..., 99% a year, flat.
    paths = [[g / 100] * 5 for g in range(100)]
    bear, base, bull = quantile_paths(paths)
    assert (bear.name, base.name, bull.name) == ("bear", "base", "bull")
    # Each neighbourhood is ten points wide: ranks 5 to 14 for the 10th percentile.
    assert bear.n == 10 and bear.growth == pytest.approx([0.095] * 5)
    assert base.growth == pytest.approx([0.495] * 5)
    assert bull.growth == pytest.approx([0.895] * 5)
    assert bear.cagr < base.cagr < bull.cagr


def test_a_path_keeps_its_shape_year_by_year():
    # A scenario is a path, not a constant rate: a company that decelerated
    # stays decelerating in its neighbourhood's mean.
    slowing = [0.30, 0.10, 0.00, -0.10, -0.20]
    rising = [0.10, 0.20, 0.30, 0.40, 0.50]
    bear, _, bull = quantile_paths([slowing] * 50 + [rising] * 50)
    assert bear.growth == pytest.approx(slowing)
    assert bull.growth == pytest.approx(rising)


def test_no_paths_means_no_scenarios():
    assert quantile_paths([]) == []


def test_datadogs_similar_starters_run_from_stalling_to_a_third_a_year():
    from pathlib import Path

    from techval.invest.engine_read import fade_panel

    panel = fade_panel(Path(__file__).parent / "fixtures")
    pool = realised_paths(panel.observations, trailing=0.276754)
    assert pool.similar and pool.n == 307
    bear, base, bull = quantile_paths(pool.paths)
    assert [s.n for s in (bear, base, bull)] == [31, 31, 31]
    assert 0.02 < bear.cagr < 0.06
    assert 0.15 < base.cagr < 0.21
    assert 0.30 < bull.cagr < 0.36


def test_a_path_is_cut_to_a_short_projection_and_faded_beyond_the_panel():
    from techval.scenarios import fit_path

    assert fit_path([0.3, 0.2, 0.1], 2, 0.05) == [0.3, 0.2]
    assert fit_path([0.3, 0.2, 0.1], 5, 0.05) == pytest.approx([0.3, 0.2, 0.1, 0.075, 0.05])


@pytest.fixture(scope="module")
def case():
    import json
    from datetime import date
    from pathlib import Path

    from techval.config import Assumptions
    from techval.edgar import CompanyFacts, HttpCache
    from techval.ev_bridge import build_ev_bridge
    from techval.financials import build_financials
    from techval.market import CsvSource, MarketData
    from techval.reverse_dcf import Case
    from techval.wacc import compute_wacc

    fixtures = Path(__file__).parent / "fixtures"
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    facts = CompanyFacts(json.loads((fixtures / "companyfacts_DDOG.json").read_text()), "DDOG")
    fin = build_financials("DDOG", facts=facts)
    market = MarketData(CsvSource(fixtures / "prices"), HttpCache(enabled=False), today=date(2026, 9, 10))
    bridge = build_ev_bridge(fin, market.spot("DDOG"), a)
    return Case(fin=fin, bridge=bridge, wacc=compute_wacc(fin, bridge, market, a), assumptions=a)


@pytest.fixture(scope="module")
def panel():
    from pathlib import Path

    from techval.invest.engine_read import fade_panel

    return fade_panel(Path(__file__).parent / "fixtures")


PRICE = 225.27
TRAILING = 0.276754


def test_each_scenario_is_the_engines_own_dcf_on_its_path(case, panel):
    from techval.scenarios import value_scenarios

    result = value_scenarios(case, PRICE, panel.observations, trailing=TRAILING)
    bear, base, bull = result.scenarios
    assert bear.value < base.value < bull.value
    for s in result.scenarios:
        assert s.value == pytest.approx(case.value(growth_path=s.growth))
    assert [s.weight for s in result.scenarios] == [0.30, 0.40, 0.30]
    assert result.weighted == pytest.approx(0.3 * bear.value + 0.4 * base.value + 0.3 * bull.value)
    assert result.base_value == pytest.approx(36.65, abs=0.01)
    # Datadog: 28.52, 44.20 and 71.90, and the price is three times the bull case.
    assert (round(bear.value, 2), round(base.value, 2), round(bull.value, 2)) == (28.52, 44.20, 71.90)
    assert result.to_dict()["pool"]["n"] == 307


def test_weights_that_do_not_sum_to_one_are_refused(case, panel):
    from techval.errors import ConfigError
    from techval.scenarios import value_scenarios

    with pytest.raises(ConfigError, match="sum to"):
        value_scenarios(
            case, PRICE, panel.observations, trailing=TRAILING,
            weights={"bear": 0.3, "base": 0.3, "bull": 0.3},
        )


def test_an_empty_panel_values_no_scenarios(case):
    from techval.scenarios import value_scenarios

    assert value_scenarios(case, PRICE, [], trailing=TRAILING) is None
