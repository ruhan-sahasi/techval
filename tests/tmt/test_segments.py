"""Segment and geography economics.

Disney's fiscal 2025 10-K is committed as a pruned instance-fact fixture and
carries every real case at once: three segments that do not sum to consolidated
revenue because intersegment revenue is eliminated, the same revenue tagged again
by geography and again by product so that a careless read counts it three times,
a segment whose depreciation and capital expenditure exist only split across a
second axis, and no segment assets at all. Datadog and Zscaler stand for the
single-reportable-segment software filer.

Everything else, the mechanics and the failure paths, is built from hand-written
facts so the arithmetic under test is visible in the test. Amounts in those are
written in USD millions and scaled up to the units a filing tags them in.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from conftest import FIXTURES
from techval.config import Assumptions
from techval.edgar import CompanyFacts, DimensionedFact
from techval.errors import ConfigError, MissingDataError
from techval.tmt.segments import (
    RECONCILIATION_TOLERANCE,
    build_segments,
)

SEGMENT_AXIS = "us-gaap:StatementBusinessSegmentsAxis"
GEOGRAPHY_AXIS = "us-gaap:StatementGeographicalAxis"
PRODUCT_AXIS = "us-gaap:ProductOrServiceAxis"
QUALIFIER_AXIS = "us-gaap:ConsolidationItemsAxis"
OPERATING_SEGMENTS = "us-gaap:OperatingSegmentsMember"

REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"
YEAR = (date(2025, 1, 1), date(2025, 12, 31))
ONE = {SEGMENT_AXIS: "x:OneMember"}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def load_instance(ticker: str):
    payload = json.loads((FIXTURES / f"instance_facts_{ticker}.json").read_text())
    facts = [
        DimensionedFact(
            tag=row["tag"],
            value=row["value"],
            unit=row["unit"],
            start=date.fromisoformat(row["start"]) if row["start"] else None,
            end=date.fromisoformat(row["end"]),
            dimensions=row["dimensions"],
        )
        for row in payload["facts"]
    ]
    return facts, payload["_accession"], date.fromisoformat(payload["_filed"])


class FilingClient:
    """Serves the committed instance facts, and company facts beside them."""

    def instance_facts(self, ticker, forms=("10-K", "10-Q")):
        return load_instance(ticker)

    def company_facts(self, ticker):
        path = FIXTURES / f"companyfacts_{ticker.upper()}.json"
        return CompanyFacts(json.loads(path.read_text()), ticker)


class HandWritten:
    """Serves facts written in the test, and nothing else.

    No ``company_facts``: a filing that cannot be read on its own terms should
    fail on its own terms, and these tests are about what the instance document
    says.
    """

    def __init__(self, facts, accession="0001000000-26-000001"):
        self.facts = facts
        self.accession = accession

    def instance_facts(self, ticker, forms=("10-K", "10-Q")):
        return self.facts, self.accession, date(2026, 2, 1)


def fact(tag, millions, dims=None, period=YEAR, instant=None, unit="USD"):
    """One fact, written in USD millions and tagged in units, like a filer."""
    return DimensionedFact(
        tag=tag,
        value=millions * 1e6,
        unit=unit,
        start=None if instant else period[0],
        end=instant or period[1],
        dimensions=dims or {},
    )


@pytest.fixture
def dis(assumptions):
    return build_segments("DIS", FilingClient(), assumptions)


@pytest.fixture
def names(dis):
    return [s.name for s in dis.segments]


# --------------------------------------------------------------------------- #
# The filing
# --------------------------------------------------------------------------- #


def test_the_three_reported_businesses_come_off_the_segment_axis(dis):
    """Disney fiscal 2025, the year ended 27 September 2025.

        Entertainment   42,466 revenue    4,674 operating income
        Experiences     36,156            9,995
        Sports          17,672            2,882

    Nothing here reaches companyfacts: the endpoint publishes undimensioned
    facts only, so a filer's whole segment note is invisible to it.
    """
    assert dis.ticker == "DIS"
    assert dis.as_of == date(2025, 9, 27)
    assert dis.revenue_tag == "Revenues"
    assert dis.accession == "0001744489-25-000155"

    assert [(s.name, s.revenue, s.operating_income) for s in dis.segments] == [
        ("Entertainment", 42466.0, 4674.0),
        ("Experiences", 36156.0, 9995.0),
        ("Sports", 17672.0, 2882.0),
    ]
    assert all(s.period_start == date(2024, 9, 29) for s in dis.segments)
    assert all(s.period_end == date(2025, 9, 27) for s in dis.segments)


def test_the_margins_are_the_reason_not_to_value_the_average(dis):
    """27.6% at the parks against 11.0% in entertainment.

    The company average of 18.6% describes neither business. A single EV/EBITDA
    struck on it prices a theme park like a streaming service, which is the
    error this module exists to make visible.
    """
    margins = {s.name: s.margin for s in dis.segments}
    assert margins["Experiences"] == pytest.approx(9995.0 / 36156.0)
    assert margins["Entertainment"] == pytest.approx(4674.0 / 42466.0)
    assert margins["Sports"] == pytest.approx(2882.0 / 17672.0)

    blended = sum(s.operating_income for s in dis.segments) / dis.consolidated_revenue
    assert blended == pytest.approx(0.1859, abs=5e-5)
    assert margins["Entertainment"] < blended < margins["Experiences"]


def test_the_table_foots_and_the_residual_is_named(dis):
    """96,294 of segments against 94,425 consolidated, and 1,869 eliminated.

    The gap is intersegment revenue, and it is 1.98% of the company: inside the
    2% tolerance, so the table reconciles, but reported either way. Absorbing it
    into the segments would overstate two of the three businesses.
    """
    allocated = sum(s.revenue for s in dis.segments)
    assert allocated == 96294.0
    assert dis.consolidated_revenue == 94425.0
    assert dis.unallocated == pytest.approx(-1869.0)
    assert dis.unallocated == pytest.approx(dis.consolidated_revenue - allocated)

    assert abs(dis.unallocated) / dis.consolidated_revenue < RECONCILIATION_TOLERANCE
    assert dis.reconciles is True
    assert dis.flags == []

    assert any("residual of -1,869 is 1.98%" in n for n in dis.notes)
    assert any("the Eliminations And Other the filer tags" in n for n in dis.notes)


def test_revenue_shares_are_struck_on_consolidated_revenue(dis):
    """So they sum to more than one, by exactly the eliminated revenue.

    This is worth seeing rather than hiding. A reader asking what share of
    Disney's revenue is Experiences wants 38.3% of the company, not 37.5% of a
    segment total that includes revenue the group sells to itself.
    """
    shares = {s.name: s.revenue_share for s in dis.segments}
    assert shares["Experiences"] == pytest.approx(36156.0 / 94425.0)
    assert sum(shares.values()) == pytest.approx(96294.0 / 94425.0)
    assert sum(shares.values()) > 1.0


def test_a_fact_on_two_axes_joins_neither_total(dis):
    """The double count, and the whole reason this module is careful.

    Entertainment revenue is tagged once whole at 42,466, again split across
    three geographies (33,815 + 2,334 + 6,317 = 42,466), and again split
    third-party against intersegment on the product axis. Read the cells beside
    the totals and the segment is counted twice over.
    """
    entertainment = next(s for s in dis.segments if s.name == "Entertainment")
    assert entertainment.revenue == 42466.0
    assert entertainment.revenue != 42466.0 + 33815.0 + 2334.0 + 6317.0

    geographies = sum(g.revenue for g in dis.geographies)
    assert geographies == pytest.approx(dis.consolidated_revenue)

    # A cell colliding with the total it sits under would land here as an
    # ambiguity rather than a double count, so the empty flag list is part of
    # the assertion: the cells were excluded, not merely deduplicated.
    assert dis.flags == []


def test_a_figure_tagged_only_across_two_axes_is_left_blank(dis):
    """Experiences depreciation is 1,933 domestic plus 782 foreign, and nothing
    else. Summing the cells back into a segment total is reconstruction rather
    than disclosure, so the figure is absent and the row says why.
    """
    experiences = next(s for s in dis.segments if s.name == "Experiences")
    assert experiences.depreciation is None
    assert experiences.capex is None
    assert any("only cut by a second axis" in n for n in experiences.notes)

    entertainment = next(s for s in dis.segments if s.name == "Entertainment")
    assert (entertainment.depreciation, entertainment.capex) == (773.0, 1155.0)


def test_product_lines_and_eliminations_are_not_segments(dis, names):
    """Advertising is 11,123 of revenue and is not a business Disney reports."""
    assert names == ["Entertainment", "Experiences", "Sports"]
    assert not any("Advertising" in n or "Subscription" in n for n in names)
    assert not any("Elimination" in n for n in names)

    assert any("Eliminations And Other is a reconciling item" in n for n in dis.notes)
    assert any("Segment Eliminations is a reconciling item" in n for n in dis.notes)


def test_depreciation_and_combined_da_are_separate_lines(dis):
    """825 of Entertainment D&A against 773 of depreciation.

    The 52 between them is amortisation of acquired intangibles. Treating
    combined D&A as depreciation overstates the maintenance spend the business
    needs, which for a media company is the difference between a cash cost and
    the run-off of a purchase price.
    """
    entertainment = next(s for s in dis.segments if s.name == "Entertainment")
    assert entertainment.da == 825.0
    assert entertainment.depreciation == 773.0
    assert entertainment.da - entertainment.depreciation == 52.0


def test_segment_assets_are_absent_and_say_so_rather_than_reading_zero(dis):
    """Disney tags total assets only for the group. ASC 280 requires them by
    segment only where the decision maker reviews them, so the column is a
    disclosure fact about the filer, not a gap to be filled with a zero.
    """
    assert all(s.assets is None for s in dis.segments)
    assert any("ASC 280" in n for s in dis.segments for n in s.notes)


def test_geography_is_read_through_the_operating_segments_qualifier(dis):
    """ConsolidationItemsAxis says which part of the reconciliation a row is,
    rather than cutting the company up, so a fact carrying it plus one geography
    is still a single-axis fact. Without that, Disney has no readable geography
    table at all.

        Americas       76,430 revenue    61,888 long-lived assets
        Europe         11,090            13,227
        Asia Pacific    6,905            10,799
    """
    assert [(g.name, g.revenue, g.assets) for g in dis.geographies] == [
        ("Americas", 76430.0, 61888.0),
        ("Europe", 11090.0, 13227.0),
        ("Asia Pacific", 6905.0, 10799.0),
    ]
    assert sum(g.revenue for g in dis.geographies) == dis.consolidated_revenue
    assert any("LongLivedAssets" in n for n in dis.geographies[0].notes)
    assert any("second cut of the same revenue" in n for n in dis.notes)


def test_herfindahl_says_this_is_a_conglomerate(dis):
    """0.37 on shares of segment revenue, which sum to one by construction.

    Read it as the number of businesses the company really is. Disney at 0.37 is
    three genuinely different businesses on three different multiples, and a
    sum-of-the-parts values it better than any single multiple can.
    """
    base = sum(s.revenue for s in dis.segments)
    expected = sum((s.revenue / base) ** 2 for s in dis.segments)

    assert dis.herfindahl == pytest.approx(expected)
    assert dis.herfindahl == pytest.approx(0.3691, abs=5e-5)
    assert dis.herfindahl < 0.5
    assert dis.largest.name == "Entertainment"


def test_the_frame_carries_the_lines_that_make_it_foot(dis):
    """A revenue column that does not add up to the company is the failure this
    module is built to catch, so the residual and the total are rows in it.
    """
    frame = dis.to_frame()
    assert list(frame["Segment"])[-2:] == [
        "Unallocated / eliminations",
        "Consolidated revenue",
    ]
    body = frame.iloc[:-1]["Revenue"].sum()
    assert body == pytest.approx(dis.consolidated_revenue)

    geography = dis.geography_frame()
    assert geography.iloc[:-1]["Revenue"].sum() == pytest.approx(
        dis.consolidated_revenue
    )


def test_the_knowledge_date_reads_the_year_that_was_current_then(assumptions):
    """Set to 2024, the same filing gives fiscal 2024: 91,361 of revenue against
    94,425, and 1,595 of eliminations rather than 1,869. A comparative year in a
    later filing is still a point-in-time answer if the period is chosen by date.
    """
    assumptions.as_of = "2024-12-31"
    report = build_segments("DIS", FilingClient(), assumptions)

    assert report.as_of == date(2024, 9, 28)
    assert report.consolidated_revenue == 91361.0
    assert report.unallocated == pytest.approx(-1595.0)
    assert [(s.name, s.revenue) for s in report.segments] == [
        ("Entertainment", 41186.0),
        ("Experiences", 34151.0),
        ("Sports", 17619.0),
    ]


# --------------------------------------------------------------------------- #
# One segment is an answer
# --------------------------------------------------------------------------- #


def test_a_single_segment_filer_returns_one_segment_and_a_note(assumptions):
    """Datadog reports one reportable segment, which is how most infrastructure
    software files. Nothing is raised, the whole company is the segment, and the
    concentration index is 1.0 by definition.
    """
    report = build_segments("DDOG", FilingClient(), assumptions)

    (segment,) = report.segments
    assert segment.name == "Datadog, Inc."
    assert segment.revenue == pytest.approx(report.consolidated_revenue)
    assert segment.revenue_share == 1.0
    assert report.herfindahl == 1.0
    assert report.reconciles is True
    assert report.unallocated == 0.0
    assert any("single-reportable-segment filer" in n for n in report.notes)


def test_a_single_segment_filer_keeps_its_own_fiscal_year(assumptions):
    """Zscaler's year ends in July. The period comes from the filer, never from
    a calendar the engine prefers.
    """
    report = build_segments("ZS", FilingClient(), assumptions)

    assert report.as_of == date(2026, 7, 31)
    assert report.segments[0].period_start == date(2025, 8, 1)
    assert report.herfindahl == 1.0


# --------------------------------------------------------------------------- #
# The reconciliation is the control
# --------------------------------------------------------------------------- #


def test_a_missed_member_fails_the_reconciliation(assumptions):
    """Two segments of 400 against a company of 1,000. The 200 is a business
    somebody did not read, and at 20% of revenue it is not a rounding error: the
    table is flagged and reconciles is false.
    """
    facts = [
        fact(REVENUE, 1000.0),
        fact(REVENUE, 400.0, {SEGMENT_AXIS: "cloud:CloudMember"}),
        fact(REVENUE, 400.0, {SEGMENT_AXIS: "cloud:ServicesMember"}),
    ]
    report = build_segments("TEST", HandWritten(facts), assumptions)

    assert report.unallocated == pytest.approx(200.0)
    assert report.reconciles is False
    assert any("does not reconcile" in f and "20.00%" in f for f in report.flags)


def test_a_residual_matching_nothing_disclosed_is_flagged(assumptions):
    """Inside the tolerance but unexplained. The filer tags eliminations of 5
    and the table is short by 10, so five of revenue is somewhere nobody named,
    and saying the table reconciles without saying that would be the dishonest
    answer.
    """
    facts = [
        fact(REVENUE, 1000.0),
        fact(REVENUE, 600.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact(REVENUE, 390.0, {SEGMENT_AXIS: "x:TwoMember"}),
        fact(REVENUE, -5.0, {SEGMENT_AXIS: "x:IntersegmentEliminationMember"}),
    ]
    report = build_segments("TEST", HandWritten(facts), assumptions)

    assert report.reconciles is True
    assert report.unallocated == pytest.approx(10.0)
    assert any("matches none of the reconciling items" in f for f in report.flags)


def test_eliminations_explain_the_residual_under_either_sign_convention(assumptions):
    """Disney tags intersegment revenue as a negative adjustment. Other filers
    tag the same thing positive and tell the reader to deduct it. The control is
    whether the residual is accounted for, not which way round the filer wrote
    it, so the match is on magnitude and a presentation choice raises no flag.
    """
    facts = [
        fact(REVENUE, 1000.0),
        fact(REVENUE, 600.0, ONE),
        fact(REVENUE, 410.0, {SEGMENT_AXIS: "x:TwoMember"}),
        fact(REVENUE, 10.0, {SEGMENT_AXIS: "x:IntersegmentRevenueMember"}),
    ]
    report = build_segments("TEST", HandWritten(facts), assumptions)

    assert report.unallocated == pytest.approx(-10.0)
    assert report.flags == []
    assert any("the Intersegment Revenue the filer tags" in n for n in report.notes)


def test_matrix_cells_never_join_a_single_axis_total(assumptions):
    """The trap in miniature. One segment of 600 tagged whole, then split 400
    domestic and 200 foreign. Reading all four facts gives a segment of 1,200
    and a company that appears to have lost 200 of revenue it never had.
    """
    facts = [
        fact(REVENUE, 1000.0),
        fact(REVENUE, 600.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact(REVENUE, 400.0, {SEGMENT_AXIS: "x:TwoMember"}),
        fact(
            REVENUE,
            400.0,
            {GEOGRAPHY_AXIS: "country:US", SEGMENT_AXIS: "x:OneMember"},
        ),
        fact(
            REVENUE,
            200.0,
            {SEGMENT_AXIS: "x:OneMember", GEOGRAPHY_AXIS: "country:NonUs"},
        ),
        fact(REVENUE, 250.0, {PRODUCT_AXIS: "x:LicenceMember"}),
    ]
    report = build_segments("TEST", HandWritten(facts), assumptions)

    assert [(s.name, s.revenue) for s in report.segments] == [
        ("One", 600.0),
        ("Two", 400.0),
    ]
    assert report.geographies == []
    assert report.reconciles is True
    assert report.unallocated == 0.0
    assert report.flags == []


def test_an_aggregate_member_is_dropped_rather_than_reconciled(assumptions):
    """A total-of-segments member carries every other member inside it. Reading
    it doubles the company, and reporting it as a residual would hide that.
    """
    facts = [
        fact(REVENUE, 1000.0),
        fact(REVENUE, 600.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact(REVENUE, 400.0, {SEGMENT_AXIS: "x:TwoMember"}),
        fact(REVENUE, 1000.0, {SEGMENT_AXIS: "us-gaap:TotalSegmentsMember"}),
    ]
    report = build_segments("TEST", HandWritten(facts), assumptions)

    assert [s.name for s in report.segments] == ["One", "Two"]
    assert report.unallocated == 0.0
    assert any("aggregate of the other members" in n for n in report.notes)


def test_the_total_of_segments_row_is_not_the_consolidated_total(assumptions):
    """A fact carrying only the operating-segments qualifier is the sum of the
    segments, which for a filer with a corporate bucket is a different number
    from company revenue. Taking it as the total makes every table foot by
    construction and proves nothing.
    """
    facts = [
        fact(REVENUE, 900.0, {QUALIFIER_AXIS: OPERATING_SEGMENTS}),
        fact(
            REVENUE,
            500.0,
            {QUALIFIER_AXIS: OPERATING_SEGMENTS, SEGMENT_AXIS: "x:OneMember"},
        ),
        fact(
            REVENUE,
            400.0,
            {QUALIFIER_AXIS: OPERATING_SEGMENTS, SEGMENT_AXIS: "x:TwoMember"},
        ),
    ]
    with pytest.raises(MissingDataError) as exc:
        build_segments("TEST", HandWritten(facts), assumptions)

    assert "consolidated revenue" in str(exc.value)
    assert "cannot be reconciled" in str(exc.value)


# --------------------------------------------------------------------------- #
# Periods
# --------------------------------------------------------------------------- #


def test_a_quarter_is_never_paired_with_a_year(assumptions):
    """The quarterly facts are the most recent in the filing and are not the
    answer. A segment measured over three months against a company measured over
    twelve reads as a quarter of the business it is.
    """
    quarter = (date(2026, 1, 1), date(2026, 3, 31))
    facts = [
        fact(REVENUE, 1000.0),
        fact(REVENUE, 600.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact(REVENUE, 400.0, {SEGMENT_AXIS: "x:TwoMember"}),
        fact(REVENUE, 300.0, period=quarter),
        fact(REVENUE, 180.0, {SEGMENT_AXIS: "x:OneMember"}, period=quarter),
        fact(REVENUE, 120.0, {SEGMENT_AXIS: "x:TwoMember"}, period=quarter),
    ]
    report = build_segments("TEST", HandWritten(facts), assumptions)

    assert report.as_of == date(2025, 12, 31)
    assert report.consolidated_revenue == 1000.0
    assert [s.revenue for s in report.segments] == [600.0, 400.0]


def test_assets_are_taken_at_the_period_end_and_nowhere_else(assumptions):
    """A later balance sheet in the same filing belongs to a later period. Pairing
    it with this year's revenue produces an asset turn nobody reported.
    """
    facts = [
        fact(REVENUE, 1000.0),
        fact(REVENUE, 1000.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact("Assets", 800.0, ONE, instant=date(2025, 12, 31)),
        fact("Assets", 950.0, ONE, instant=date(2026, 6, 30)),
    ]
    report = build_segments("TEST", HandWritten(facts), assumptions)

    (segment,) = report.segments
    assert segment.assets == 800.0
    assert segment.period_end == date(2025, 12, 31)


def test_composite_da_fires_only_where_every_component_is_present(assumptions):
    """Some filers never tag a combined D&A line. The components are summed where
    both are tagged against the same member and the same period, and where only
    one is, nothing is reported: a partial sum passed off as a total is worse
    than a blank, because it looks like an answer.
    """
    facts = [
        fact(REVENUE, 1000.0),
        fact(REVENUE, 600.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact(REVENUE, 400.0, {SEGMENT_AXIS: "x:TwoMember"}),
        fact("Depreciation", 40.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact("AmortizationOfIntangibleAssets", 15.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact("AmortizationOfIntangibleAssets", 9.0, {SEGMENT_AXIS: "x:TwoMember"}),
    ]
    report = build_segments("TEST", HandWritten(facts), assumptions)

    one, two = report.segments
    assert one.da == 55.0
    assert one.depreciation == 40.0
    assert two.da is None
    assert two.depreciation is None


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_two_values_under_one_key_are_flagged_not_chosen_between(assumptions):
    """Inline XBRL renders a fact once per table it appears in, so duplicates are
    normal and collapse. Two different values for one tag, period and member is
    not a duplicate: it is an ambiguity, and picking the convenient one silently
    is how a wrong number gets into a memo.
    """
    facts = [
        fact(REVENUE, 1000.0),
        fact(REVENUE, 600.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact(REVENUE, 620.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact(REVENUE, 400.0, {SEGMENT_AXIS: "x:TwoMember"}),
    ]
    report = build_segments("TEST", HandWritten(facts), assumptions)

    assert any("tagged twice" in f and "600" in f and "620" in f for f in report.flags)
    assert report.segments[0].revenue == 600.0


def test_a_segment_with_an_operating_result_and_no_revenue_is_flagged(assumptions):
    """It carries a margin that cannot be struck and it is missing from the
    reconciliation, so both are said rather than one being inferred.
    """
    facts = [
        fact(REVENUE, 1000.0),
        fact(REVENUE, 1000.0, {SEGMENT_AXIS: "x:OneMember"}),
        fact("OperatingIncomeLoss", 40.0, {SEGMENT_AXIS: "x:VenturesMember"}),
    ]
    report = build_segments("TEST", HandWritten(facts), assumptions)

    ventures = next(s for s in report.segments if s.name == "Ventures")
    assert ventures.revenue is None
    assert ventures.margin is None
    assert ventures.operating_income == 40.0
    assert any("no revenue" in f and "Ventures" in f for f in report.flags)
    assert any("outside the revenue reconciliation" in n for n in ventures.notes)


def test_a_filing_with_no_annual_revenue_raises_rather_than_guessing(assumptions):
    """Nothing to reconcile against and nothing to reconcile. The error names the
    tags that were tried and the filing that was read, so the gap can be audited
    by hand.
    """
    facts = [fact("Assets", 500.0, instant=date(2025, 12, 31))]
    with pytest.raises(MissingDataError) as exc:
        build_segments("TEST", HandWritten(facts), assumptions)

    assert "segment revenue" in str(exc.value)
    assert REVENUE in str(exc.value)
    assert "0001000000-26-000001" in str(exc.value)


def test_switching_segments_off_refuses_rather_than_returning_nothing(assumptions):
    """An empty table and a switched-off feature look identical in a rendered
    report, which is exactly why one of them raises.
    """
    assumptions.tmt.segments = False
    with pytest.raises(ConfigError, match="tmt.segments"):
        build_segments("DIS", FilingClient(), assumptions)
