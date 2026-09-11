"""Metric packs, on the fixture filings and on synthetic media and telecom cases.

The four fixture companies are all software, which is what they were frozen for,
so the media and tower cases are built by hand. Every number a synthetic case
asserts is one that can be checked on paper, which is the point: a metric pack
whose arithmetic only agrees with itself is not evidence of anything.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from conftest import load_facts
from techval import tags
from techval.errors import ConfigError
from techval.ev_bridge import build_ev_bridge
from techval.financials import Financials
from techval.tmt.metrics import (
    MetricPack,
    build_metrics,
    media_metrics,
    software_metrics,
    telecom_metrics,
)


def prior_year_revenue(fin: Financials, facts) -> float:
    """The prior twelve months of revenue, anchored on a period the filer reported.

    The same anchoring the comp engine uses. A flat 365 day step lands one day
    off a 52/53 week filer's year end, which is most of this fixture set.
    """
    target = fin.as_of - timedelta(days=365)
    _, series, _ = facts.resolve_duration_series("revenue", tags.REVENUE)
    ends = [f.end for f in series if abs((f.end - target).days) <= 10]
    anchor = min(ends, key=lambda d: abs((d - target).days)) if ends else target
    raw, _ = facts.resolve_ttm("prior revenue", tags.REVENUE, anchor, required=False)
    assert raw is not None, "fixture should carry a prior year of revenue"
    return raw / 1e6


def stub(**overrides) -> Financials:
    """A synthetic filer. Every figure is round so the arithmetic is checkable."""
    base = dict(
        ticker="TEST",
        entity_name="Test Co",
        cik=1,
        as_of=date(2026, 6, 30),
        revenue=1000.0,
        gross_profit=700.0,
        ebit=200.0,
        da=100.0,
        sbc=50.0,
        net_income=100.0,
        pretax_income=120.0,
        tax_expense=20.0,
        interest_expense=30.0,
        operating_lease_cost=10.0,
        capex=100.0,
        cfo=250.0,
        diluted_shares=100.0,
        basic_shares=95.0,
        cash=200.0,
        short_term_investments=0.0,
        straight_debt=500.0,
        convertible_debt=0.0,
        operating_lease_liability=80.0,
        finance_lease_liability=20.0,
        nci=0.0,
        preferred=0.0,
        current_assets=600.0,
        current_liabilities=400.0,
        deferred_revenue=300.0,
    )
    base.update(overrides)
    return Financials(**base)


# -- the Rule of 40 argument ---------------------------------------------- #


def test_three_rule_of_40_variants_disagree_on_one_company(ddog):
    """Datadog passes the Rule of 40 on cash and fails it on both earnings bases.

    Growth of 31.5% against a 29.8% FCF margin, a 2.0% EBITDA margin and a 0.4%
    operating margin. The same twelve months of filings score 61, 33 and 32.
    """
    facts = load_facts("DDOG")
    pack = software_metrics(ddog, {"revenue_prior": prior_year_revenue(ddog, facts)})

    ebitda = pack.metrics["Rule of 40 (EBITDA variant)"]
    fcf = pack.metrics["Rule of 40 (FCF variant)"]
    operating = pack.metrics["Rule of 40 (operating variant)"]

    assert fcf == pytest.approx(61.3, abs=0.1)
    assert ebitda == pytest.approx(33.5, abs=0.1)
    assert operating == pytest.approx(31.9, abs=0.1)

    assert fcf > 40.0 > ebitda > operating
    assert pack.metrics["Rule of 40 variant spread"] == pytest.approx(fcf - operating)
    assert any("straddle 40" in n for n in pack.notes)


def test_rule_of_40_is_quoted_in_points_not_decimals(mdb):
    """40 means 40, matching how the comp table already reports the score."""
    facts = load_facts("MDB")
    pack = software_metrics(mdb, {"revenue_prior": prior_year_revenue(mdb, facts)})
    growth = pack.metrics["Revenue growth"]
    margin = pack.metrics["EBITDA margin"]
    assert growth < 1.0 and margin < 1.0
    assert pack.metrics["Rule of 40 (EBITDA variant)"] == pytest.approx(
        100.0 * (growth + margin)
    )


def test_fcf_margin_is_struck_before_sbc_and_its_sibling_after(ddog):
    """The whole FCF-against-EBITDA argument, as an identity.

    Cash from operations adds stock compensation straight back, so the distance
    between the two cash margins is exactly the SBC share of revenue.
    """
    pack = software_metrics(ddog)
    gap = pack.metrics["FCF margin"] - pack.metrics["FCF margin less SBC"]
    assert gap == pytest.approx(pack.metrics["SBC / revenue"])
    assert pack.metrics["SBC / revenue"] == pytest.approx(ddog.sbc / ddog.revenue)


def test_growth_can_arrive_as_a_rate_or_as_a_prior_period(zs):
    """Either input reaches the same score, because one derives the other."""
    facts = load_facts("ZS")
    prior = prior_year_revenue(zs, facts)
    by_prior = software_metrics(zs, {"revenue_prior": prior})
    by_rate = software_metrics(zs, {"revenue_growth": zs.revenue / prior - 1.0})
    assert by_prior.metrics["Rule of 40 (FCF variant)"] == pytest.approx(
        by_rate.metrics["Rule of 40 (FCF variant)"]
    )


# -- sales efficiency ------------------------------------------------------ #


def test_magic_number_and_cac_payback_agree_on_paper():
    """200 of new revenue on 160 of prior selling spend is 1.25, paid back in a year."""
    fin = stub(revenue=1000.0, gross_profit=800.0)
    pack = software_metrics(
        fin, {"revenue_prior": 800.0, "sales_and_marketing_prior": 160.0}
    )
    assert pack.metrics["Magic number"] == pytest.approx(1.25)
    assert pack.metrics["CAC payback (months)"] == pytest.approx(12.0)
    assert any("above the 0.75 line" in n for n in pack.notes)


def test_cac_payback_uses_gross_profit_not_revenue():
    """Halve the gross margin and the payback doubles. Revenue does not pay back CAC."""
    thin = software_metrics(
        stub(gross_profit=400.0),
        {"revenue_prior": 800.0, "sales_and_marketing_prior": 160.0},
    )
    fat = software_metrics(
        stub(gross_profit=800.0),
        {"revenue_prior": 800.0, "sales_and_marketing_prior": 160.0},
    )
    assert thin.metrics["CAC payback (months)"] == pytest.approx(
        2.0 * fat.metrics["CAC payback (months)"]
    )


# -- missing inputs flag, they never default to zero ----------------------- #


def test_missing_sales_and_marketing_flags_rather_than_zeroing():
    pack = software_metrics(stub(), {"revenue_prior": 800.0})
    assert pack.metrics["Magic number"] is None
    assert pack.metrics["CAC payback (months)"] is None
    assert "Magic number" not in pack.computed
    assert any(
        "kpis['sales_and_marketing_prior']" in f and "Magic number" in f
        for f in pack.flags
    )


def test_missing_growth_flags_every_rule_of_40_variant():
    pack = software_metrics(stub())
    for name in (
        "Rule of 40 (EBITDA variant)",
        "Rule of 40 (FCF variant)",
        "Rule of 40 (operating variant)",
        "Rule of 40 variant spread",
    ):
        assert pack.metrics[name] is None
    assert any("kpis['revenue_growth']" in f for f in pack.flags)
    assert any("all three variants" in f for f in pack.flags)


def test_a_filer_with_no_da_tag_loses_the_ebitda_variant_only(crwd):
    """CrowdStrike reports no combined D&A tag, so there is nothing to add back.

    The cash variant of the Rule of 40 is unaffected, which is the argument for
    printing all three rather than one.
    """
    facts = load_facts("CRWD")
    pack = software_metrics(crwd, {"revenue_prior": prior_year_revenue(crwd, facts)})
    assert crwd.ebitda is None
    assert pack.metrics["EBITDA margin"] is None
    assert pack.metrics["Rule of 40 (EBITDA variant)"] is None
    assert pack.metrics["Rule of 40 (FCF variant)"] == pytest.approx(54.1, abs=0.1)
    assert any("depreciation and amortisation tag" in f for f in pack.flags)


def test_a_non_numeric_kpi_is_refused_and_named():
    pack = software_metrics(stub(), {"nrr": "115%"})
    assert pack.metrics["Net revenue retention"] is None
    assert any("is a str, not a number" in f for f in pack.flags)


def test_a_negative_denominator_is_nm_with_a_reason():
    pack = telecom_metrics(stub(ebit=-400.0, da=100.0), ev=5000.0)
    assert pack.metrics["EBITDA"] == pytest.approx(-300.0)
    assert pack.metrics["EV / EBITDA"] is None
    assert any("EV / EBITDA NM" in f and "not positive" in f for f in pack.flags)


# -- media: the content gap ------------------------------------------------ #


def test_content_spend_against_amortisation_is_the_media_tell():
    """900 of cash out against a 600 charge: earnings flatter cash by 300 a year."""
    fin = stub(revenue=4000.0, ebit=300.0, da=200.0)
    pack = media_metrics(
        fin,
        {"content_spend": 900.0, "content_amortization": 600.0},
        ev=4000.0,
    )
    assert pack.metrics["Content spend"] == pytest.approx(900.0)
    assert pack.metrics["Content spend / revenue"] == pytest.approx(0.225)
    assert pack.metrics["Content spend less amortisation"] == pytest.approx(300.0)
    assert pack.metrics["Content spend / amortisation"] == pytest.approx(1.5)
    # EBITDA of 500 charged a 600 amortisation but 900 of cash went out.
    assert pack.metrics["EBITDA less content gap"] == pytest.approx(200.0)
    assert pack.metrics["EBITDA less cash content spend"] == pytest.approx(-400.0)
    assert pack.metrics["EV / EBITDA less content gap"] == pytest.approx(20.0)
    assert any("flatter" in d for d in pack.definitions.values())


def test_content_amortisation_above_total_da_settles_which_adjustment_applies():
    """A 600 content charge inside a 200 D&A add-back is arithmetically impossible.

    So the charge is in cost of revenues, EBITDA never added it back, and the gap
    adjustment is the applicable one. The pack says so rather than leaving the
    reader to pick.
    """
    pack = media_metrics(
        stub(revenue=4000.0, ebit=300.0, da=200.0),
        {"content_spend": 900.0, "content_amortization": 600.0},
    )
    assert any("is not inside that add-back" in n for n in pack.notes)
    assert not any("Check the cash flow statement" in f for f in pack.flags)


def test_an_ambiguous_content_tagging_is_flagged_not_guessed():
    """With D&A of 900 the charge could sit inside the add-back. The pack says so."""
    pack = media_metrics(
        stub(revenue=4000.0, ebit=300.0, da=900.0),
        {"content_spend": 900.0, "content_amortization": 600.0},
    )
    assert any("Check the cash flow statement" in f for f in pack.flags)
    assert pack.metrics["EBITDA less content gap"] is not None
    assert pack.metrics["EBITDA less cash content spend"] is not None


def test_missing_content_amortisation_flags_the_gap_rather_than_assuming_zero():
    pack = media_metrics(stub(), {"content_spend": 900.0})
    assert pack.metrics["Content spend less amortisation"] is None
    assert pack.metrics["Content spend / amortisation"] is None
    assert pack.metrics["EBITDA less content gap"] is None
    assert any("kpis['content_amortization']" in f for f in pack.flags)
    # The other tagging's figure does not need the charge, so it still computes.
    assert pack.metrics["EBITDA less cash content spend"] == pytest.approx(-600.0)


# -- subscribers, ARPU and churn ------------------------------------------- #


def test_reported_arpu_is_reconciled_to_revenue():
    """1,200 of revenue on 10mm subscribers is 10 a month. Say so when it is not."""
    fin = stub(revenue=1200.0)
    clean = media_metrics(fin, {"subscribers": 10.0, "arpu": 10.0})
    assert clean.metrics["Implied revenue per subscriber (monthly)"] == pytest.approx(
        10.0
    )
    assert clean.metrics["ARPU consistency gap"] == pytest.approx(0.0)
    assert not any("differ by" in f for f in clean.flags)

    partial = media_metrics(fin, {"subscribers": 10.0, "arpu": 8.0})
    assert partial.metrics["ARPU consistency gap"] == pytest.approx(0.25)
    assert any("differ by" in f for f in partial.flags)


def test_an_annual_arpu_supplied_as_monthly_is_caught():
    pack = media_metrics(stub(revenue=1200.0), {"subscribers": 10.0, "arpu": 120.0})
    assert any("looks like an annual ARPU" in f for f in pack.flags)


def test_churn_gives_a_subscriber_life_and_names_its_assumption():
    pack = media_metrics(stub(), {"churn": 0.02})
    assert pack.metrics["Implied subscriber life (months)"] == pytest.approx(50.0)
    assert "constant hazard" in pack.definitions["Implied subscriber life (months)"]
    assert pack.metrics["Subscribers"] is None
    assert any("kpis['subscribers']" in f for f in pack.flags)


# -- telecom, towers and fiber --------------------------------------------- #


def test_capex_intensity_and_the_multiple_the_sector_actually_quotes():
    """A carrier at 18% capex intensity is priced at 13x EBITDA and 25x what is left."""
    fin = stub(revenue=1000.0, ebit=280.0, da=100.0, capex=180.0)
    pack = telecom_metrics(fin, ev=5000.0, net_debt=1200.0)
    assert pack.metrics["Capex intensity"] == pytest.approx(0.18)
    assert pack.metrics["EBITDA"] == pytest.approx(380.0)
    assert pack.metrics["EBITDA less capex"] == pytest.approx(200.0)
    assert pack.metrics["EBITDA less capex margin"] == pytest.approx(0.20)
    assert pack.metrics["EV / EBITDA"] == pytest.approx(13.158, abs=0.001)
    assert pack.metrics["EV / EBITDA less capex"] == pytest.approx(25.0)
    assert pack.metrics["Net debt / EBITDA"] == pytest.approx(3.158, abs=0.001)
    assert any("15 to 20" in d for d in pack.definitions.values())


def test_missing_capex_flags_rather_than_pricing_the_network_as_free():
    pack = telecom_metrics(stub(capex=None), ev=5000.0)
    assert pack.metrics["Capex intensity"] is None
    assert pack.metrics["EBITDA less capex"] is None
    assert pack.metrics["EV / EBITDA less capex"] is None
    assert pack.metrics["EV / EBITDA"] is not None
    assert any("no capital expenditure line" in f for f in pack.flags)


def test_missing_net_debt_says_leverage_is_a_bridge_decision():
    pack = telecom_metrics(stub(), ev=5000.0)
    assert pack.metrics["Net debt / EBITDA"] is None
    assert any("bridge decision" in f for f in pack.flags)


def test_towers_get_an_affo_proxy_and_the_reit_caveat():
    """700 of EBITDA less 200 of interest less 30 of maintenance capex is 470."""
    fin = stub(
        revenue=1000.0, ebit=600.0, da=100.0, capex=250.0, interest_expense=200.0
    )
    pack = telecom_metrics(
        fin,
        {"maintenance_capex": 30.0},
        market_cap=9400.0,
        sub_vertical="towers",
    )
    assert pack.metrics["AFFO proxy"] == pytest.approx(470.0)
    assert pack.metrics["Price / AFFO proxy"] == pytest.approx(20.0)
    assert any("REIT" in n for n in pack.notes)
    assert "not the company's AFFO" in pack.definitions["AFFO proxy"]


def test_total_capex_is_never_substituted_for_maintenance_capex():
    fin = stub(capex=250.0, interest_expense=200.0)
    pack = telecom_metrics(fin, market_cap=9400.0, sub_vertical="towers")
    assert fin.capex is not None
    assert pack.metrics["AFFO proxy"] is None
    assert pack.metrics["Price / AFFO proxy"] is None
    assert any("would understate AFFO" in f for f in pack.flags)


def test_the_telecom_pack_without_towers_carries_no_affo_line():
    pack = telecom_metrics(stub(), sub_vertical="telecom")
    assert "AFFO proxy" not in pack.metrics


# -- the pack contract ----------------------------------------------------- #


def test_a_metric_without_a_definition_is_refused():
    with pytest.raises(ConfigError, match="every metric must carry its definition"):
        MetricPack(sub_vertical="software", metrics={"Mystery": 1.0}, definitions={})


@pytest.mark.parametrize("ticker", ["DDOG", "CRWD", "MDB", "ZS"])
def test_every_fixture_company_builds_a_self_documenting_pack(ticker, assumptions):
    from techval.financials import build_financials

    facts = load_facts(ticker)
    fin = build_financials(ticker, facts=facts)
    pack = build_metrics(
        fin, "software", assumptions, {"revenue_prior": prior_year_revenue(fin, facts)}
    )
    assert set(pack.metrics) <= set(pack.definitions)
    assert all(d.strip() for d in pack.definitions.values())
    frame = pack.to_frame()
    assert list(frame.columns) == ["Value", "Definition"]
    assert len(frame) == len(pack.metrics)
    assert len(pack.rows()) == len(pack.metrics)
    # Nothing that could not be sourced is sitting in the table as a zero.
    assert all(v is not None for v in pack.computed.values())


def test_the_pack_reports_kpi_keys_it_did_not_read(ddog, assumptions):
    pack = build_metrics(ddog, "software", assumptions, {"arpu_monthly": 12.0})
    assert any("reached no metric" in f and "arpu_monthly" in f for f in pack.flags)


# -- routing and the bridge ------------------------------------------------ #


@pytest.mark.parametrize(
    "name,expected",
    [("saas", "software"), ("streaming", "media"), ("Fiber", "telecom")],
)
def test_sub_vertical_aliases_route_to_a_pack(name, expected, ddog, assumptions):
    pack = build_metrics(ddog, name, assumptions)
    assert pack.sub_vertical == name.lower()
    assert any(f"on the {expected} metric pack" in n for n in pack.notes)


def test_an_unknown_sub_vertical_raises_rather_than_defaulting(ddog, assumptions):
    with pytest.raises(ConfigError, match="no metric pack is written"):
        build_metrics(ddog, "semiconductors", assumptions)


def test_no_sub_vertical_anywhere_raises(ddog, assumptions):
    with pytest.raises(ConfigError, match="no sub-vertical"):
        build_metrics(ddog, None, assumptions)


def test_the_bridge_supplies_enterprise_value_equity_value_and_net_debt(
    ddog, ddog_bridge, assumptions
):
    pack = build_metrics(
        ddog, "telecom", assumptions, {"revenue_growth": 0.3}, ddog_bridge
    )
    assert pack.metrics["Net debt"] == pytest.approx(ddog_bridge.net_debt)
    assert pack.metrics["EV / EBITDA"] == pytest.approx(
        ddog_bridge.enterprise_value / ddog.ebitda
    )
    assert any("lease convention" in n for n in pack.notes)


def test_a_lease_inclusive_bridge_is_paired_with_ebitdar(ddog, assumptions, market):
    """Counting the lease liability as debt and then dividing by post-rent EBITDA
    would charge the lease twice. The bridge hands down EBITDAR and the pack uses it.
    """
    assumptions.leases.capitalize_operating_leases = True
    bridge = build_ev_bridge(ddog, market.spot("DDOG"), assumptions)
    pack = build_metrics(ddog, "telecom", assumptions, None, bridge)
    assert ddog.ebitdar > ddog.ebitda
    assert pack.metrics["EBITDA"] == pytest.approx(ddog.ebitdar)
    assert "EBITDAR" in pack.definitions["EBITDA"]


def test_software_without_a_bridge_flags_the_multiples_it_cannot_strike(ddog):
    pack = software_metrics(ddog, {"arr": 4000.0})
    assert pack.metrics["EV / ARR"] is None
    assert pack.metrics["FCF yield on equity"] is None
    assert any("no enterprise value was supplied" in f for f in pack.flags)
    assert any("no equity value was supplied" in f for f in pack.flags)
