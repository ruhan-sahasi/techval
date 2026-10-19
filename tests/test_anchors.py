"""The record case: DCF assumptions read off the filings, each with its source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from techval.anchors import record_anchors, trailing_growth, unanchored
from techval.config import Assumptions
from techval.edgar import CompanyFacts
from techval.financials import build_financials

FIXTURES = Path(__file__).parent / "fixtures"


def company(ticker):
    facts = CompanyFacts(json.loads((FIXTURES / f"companyfacts_{ticker}.json").read_text()), ticker)
    return build_financials(ticker, facts=facts), facts


def by_field(anchors):
    return {a.field: a for a in anchors}


def test_datadogs_growth_is_its_trailing_twelve_months_against_a_year_before():
    fin, facts = company("DDOG")
    growth, how = trailing_growth(fin, facts)
    assert growth == pytest.approx(0.315, abs=0.001)
    assert "twelve months to 2026-06-30" in how


def test_a_year_the_quarters_cannot_tile_falls_back_to_the_fiscal_years():
    # Disney's twelve months to mid-2025 cannot be tiled from the pruned fixture.
    fin, facts = company("DIS")
    growth, how = trailing_growth(fin, facts)
    assert "fiscal-year revenue" in how and -0.2 < growth < 0.2


def test_each_anchor_says_what_the_assumptions_file_had_and_where_the_figure_came_from():
    fin, facts = company("DDOG")
    a = Assumptions()
    anchors = by_field(record_anchors(fin, facts, a))
    assert anchors["revenue_growth_start"].assumed == 0.20
    assert anchors["ebit_margin_terminal"].value == pytest.approx(fin.ebit_margin)
    assert anchors["capex_pct_revenue"].value == pytest.approx(fin.capex / fin.revenue)
    assert anchors["da_pct_revenue"].value == pytest.approx(fin.da / fin.revenue)
    # The fade ends at the lower of the trailing and the assumed year-N rate.
    assert anchors["revenue_growth_terminal"].value == pytest.approx(0.05)
    assert all(x.source for x in anchors.values())


def test_a_figure_the_filings_cannot_support_is_left_at_the_assumption_and_named():
    # CrowdStrike's D&A does not resolve from its fixture.
    fin, facts = company("CRWD")
    anchors = record_anchors(fin, facts, Assumptions())
    assert "da_pct_revenue" not in by_field(anchors)
    assert unanchored(a.field for a in anchors) == ["da_pct_revenue"]


def test_growth_outside_the_band_is_held_to_it_and_says_so(monkeypatch):
    import techval.anchors as A

    fin, facts = company("DDOG")
    monkeypatch.setattr(A, "trailing_growth", lambda fin, facts: (1.40, "revenue doubled and more"))
    start = by_field(record_anchors(fin, facts, Assumptions()))["revenue_growth_start"]
    assert start.value == A.GROWTH_CEILING
    assert "140.0% held to 60%" in start.source


def test_with_the_switch_off_the_assumptions_come_back_as_given():
    from techval.anchors import record_case

    fin, facts = company("DDOG")
    a = Assumptions()
    same, anchors = record_case(a, fin, facts)
    assert same is a and anchors == []


def test_with_the_switch_on_the_dcf_takes_the_record_and_the_original_is_untouched():
    from techval.anchors import record_case
    from techval.dcf import run_dcf
    from techval.ev_bridge import build_ev_bridge

    fin, facts = company("DDOG")
    a = Assumptions()
    a.dcf.from_record = True
    a.market.risk_free_rate = 0.0483
    record, anchors = record_case(a, fin, facts)
    assert record.dcf.revenue_growth_start == pytest.approx(0.315, abs=0.001)
    assert record.dcf.ebit_margin_terminal == pytest.approx(fin.ebit_margin)
    assert a.dcf.revenue_growth_start == 0.20 and a.dcf.ebit_margin_terminal == 0.20
    assert {x.field for x in anchors} >= {"revenue_growth_start", "ebit_margin_terminal"}
