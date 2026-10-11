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


@pytest.mark.parametrize(
    "body, message",
    [
        ("scenarios:\n  weights: {bear: 0.3, base: 0.3, bull: 0.3}\n", "sum to 0.9000"),
        ("scenarios:\n  quantiles: {bear: 0.5, base: 0.5, bull: 0.9}\n", "rise strictly"),
        ("scenarios:\n  weights: {bear: 0.5, bull: 0.5}\n", "bear, base and bull"),
    ],
)
def test_scenario_settings_that_cannot_be_meant_are_refused(tmp_path, body, message):
    from techval.config import Assumptions
    from techval.errors import ConfigError

    path = tmp_path / "assumptions.yaml"
    path.write_text(body)
    with pytest.raises(ConfigError, match=message):
        Assumptions.load(path)


def test_the_scenarios_follow_the_assumptions(case, panel):
    from techval.scenarios import value_scenarios

    wider = case.with_dcf()
    wider.assumptions.scenarios.quantiles = {"bear": 0.25, "base": 0.5, "bull": 0.75}
    wider.assumptions.scenarios.weights = {"bear": 0.25, "base": 0.5, "bull": 0.25}
    result = value_scenarios(wider, PRICE, panel.observations, trailing=TRAILING)
    usual = value_scenarios(case, PRICE, panel.observations, trailing=TRAILING)
    assert usual.scenario("bear").value < result.scenario("bear").value
    assert result.scenario("bull").value < usual.scenario("bull").value
    assert [s.weight for s in result.scenarios] == [0.25, 0.5, 0.25]


def test_an_empty_panel_values_no_scenarios(case):
    from techval.scenarios import value_scenarios

    assert value_scenarios(case, PRICE, [], trailing=TRAILING) is None


def three(bear=10.0, base=20.0, bull=40.0):
    from techval.scenarios import Scenario

    return [
        Scenario("bear", 0.1, 0.3, [], 0.0, 1, bear),
        Scenario("base", 0.5, 0.4, [], 0.0, 1, base),
        Scenario("bull", 0.9, 0.3, [], 0.0, 1, bull),
    ]


def test_a_price_at_the_weighted_value_implies_swansons_own_weight():
    from techval.scenarios import implied_weight

    # 0.3 x 10 + 0.4 x 20 + 0.3 x 40 = 23.
    implied = implied_weight(three(), 23.0)
    assert implied["side"] == "bull" and implied["weight"] == pytest.approx(0.30)


def test_a_richer_price_leans_on_the_bull_case_and_a_cheaper_one_on_the_bear():
    from techval.scenarios import implied_weight

    rich = implied_weight(three(), 30.0)
    # Bear and base held 3:4 are worth 11/0.7; the bull weight that lifts that to 30.
    assert rich["side"] == "bull"
    assert rich["weight"] == pytest.approx((30 - 11 / 0.7) / (40 - 11 / 0.7))
    cheap = implied_weight(three(), 15.0)
    assert cheap["side"] == "bear"
    assert cheap["weight"] == pytest.approx((15 - 20 / 0.7) / (10 - 20 / 0.7))
    assert "on the bear case, against 30%" in cheap["sentence"]


def test_a_price_outside_the_scenarios_is_named_not_weighted():
    from techval.scenarios import implied_weight

    above = implied_weight(three(), 50.0)
    assert above["side"] == "above" and above["weight"] is None
    assert "1.2 times" in above["sentence"]
    below = implied_weight(three(), 5.0)
    assert below["side"] == "below" and "below even the bear case" in below["sentence"]


def test_datadogs_price_is_above_even_its_bull_case(case, panel):
    from techval.scenarios import value_scenarios

    result = value_scenarios(case, PRICE, panel.observations, trailing=TRAILING)
    assert result.implied["side"] == "above"
    assert "3.1 times" in result.implied["sentence"]
    first, weighted, implied = result.sentences
    assert "307 company-years" in first and "4.1%, 18.0% and 32.9%" in first
    assert "47.81" in weighted
    assert result.to_dict()["implied"]["side"] == "above"


class Dated(Obs):
    def __init__(self, as_of, growth, rate):
        super().__init__(growth, flat(rate))
        self.as_of = as_of


def test_a_band_is_built_only_from_outcomes_public_by_then():
    from datetime import date

    from techval.scenarios import band_coverage

    panel = [
        Dated(date(2000, 3, 1), 0.2, 0.00),
        Dated(date(2000, 3, 1), 0.2, 0.10),
        # Filed 2003: nothing it could be scored against was public yet.
        Dated(date(2003, 3, 1), 0.2, 0.05),
        # Filed 2010: all three earlier outcomes are public; 20% beats their 90th percentile.
        Dated(date(2010, 3, 1), 0.2, 0.20),
    ]
    cover = band_coverage(panel, test_from=date(2001, 1, 1), min_similar=2)
    assert (cover["n"], cover["above"], cover["inside"], cover["below"]) == (1, 1, 0, 0)


def test_on_held_out_company_years_the_band_runs_narrow_on_the_upside(panel):
    from datetime import date

    from techval.scenarios import band_coverage

    cover = band_coverage(panel.observations, test_from=date(2016, 1, 1))
    assert (cover["n"], cover["inside"], cover["below"], cover["above"]) == (674, 466, 78, 130)
    # Nominal 80% inside, 10% each side; what followed ran faster than the history.
    assert 0.65 < cover["share_inside"] < 0.75
    assert cover["share_above"] > cover["share_below"]
    assert cover["share_above_median"] > 0.55
