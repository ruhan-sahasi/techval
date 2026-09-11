"""Operating metrics: what the filings really tag, and what the prose really says.

Nothing here touches the network. The XBRL side runs against four instance
documents committed under ``tests/fixtures/instance_kpis_*.json``, pruned from
the real filings on 2026-09-11 and chosen for what they prove:

    DDOG  a software filer that tags remaining performance obligation and
          nothing else the sector quotes, which is the ordinary case
    NET   the same, plus the percentage of the obligation due inside a year, so
          the current portion can be derived rather than guessed
    NFLX  content spend and content amortisation as filer extension elements,
          which is the case a us-gaap-only engine cannot see at all
    TMUS  three different undimensioned values for one concept at one date, so
          the conflict path runs against a real filing rather than a mock

The text side runs against synthetic MD&A, because the point is to cover the
phrasings rather than one company's house style, and because a hedged or
mis-scaled sentence has to be written deliberately to be tested at all.

The billings arithmetic is checked against the worked example in
``derive_billings``, on the committed Datadog facts: if the docstring and the
code ever disagree, this fails.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from techval import tags
from techval.config import Assumptions
from techval.edgar import CompanyFacts, DimensionedFact
from techval.errors import ConfigError, MissingDataError
from techval.tmt.kpis import (
    ALL_METRICS,
    CONFIDENCE_REFUSED,
    KPI,
    KPISet,
    KPISettings,
    MEDIA_METRICS,
    SOFTWARE_METRICS,
    apply_sanity_bounds,
    build_kpis,
    derive_billings,
    extract_from_text,
    extract_from_xbrl,
    settings_from_assumptions,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# The date the instance fixtures were retrieved, and the date every assertion
# below is made as of. Pinning it keeps "latest" fixed for good.
AS_OF = date(2026, 9, 10)

# A filer with no company facts at all. Used where the point of the test is the
# instance document, so that anything the assertion sees demonstrably came from
# the tagged filing and not from a us-gaap fallback underneath it.
NO_COMPANY_FACTS = {"cik": 0, "entityName": "Instance Only Inc", "facts": {"us-gaap": {}}}


def load_instance(ticker: str) -> tuple[list[DimensionedFact], str, date]:
    """One committed instance document, in the shape ``EdgarClient`` returns."""
    payload = json.loads((FIXTURES / f"instance_kpis_{ticker}.json").read_text())
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


def load_company_facts(ticker: str) -> CompanyFacts:
    path = FIXTURES / f"companyfacts_{ticker}.json"
    return CompanyFacts(json.loads(path.read_text()), ticker)


class KpiClient:
    """An offline client carrying both halves of a filing.

    The conftest fixture client answers ``company_facts`` only, because nothing
    before this module needed dimensioned facts. This one adds
    ``instance_facts`` so the extension-tag path can run, and lets the two halves
    be mixed: ``instance`` names which committed instance document to serve, so a
    media instance can be paired with an empty fact set.
    """

    def __init__(
        self,
        instance: str | None = None,
        facts: str | None = None,
        knowledge_date: date | None = None,
    ) -> None:
        self.instance = instance
        self.facts = facts
        if knowledge_date is not None:
            self.knowledge_date = knowledge_date

    def company_facts(self, ticker: str) -> CompanyFacts:
        if self.facts is None:
            return CompanyFacts(NO_COMPANY_FACTS, ticker)
        return load_company_facts(self.facts)

    def instance_facts(
        self, ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q")
    ) -> tuple[list[DimensionedFact], str, date]:
        if self.instance is None:
            raise MissingDataError("XBRL instance document", ticker=ticker)
        return load_instance(self.instance)


class NoInstanceClient:
    """A client from before dimensioned facts existed, as conftest's still is."""

    def company_facts(self, ticker: str) -> CompanyFacts:
        return load_company_facts(ticker)


def software(**over) -> Assumptions:
    a = Assumptions()
    a.tmt.sub_vertical = "infrastructure_software"
    for key, value in over.items():
        setattr(a, key, value)
    return a


def media() -> Assumptions:
    a = Assumptions()
    a.tmt.sub_vertical = "media_entertainment"
    return a


def tagged(ticker: str, as_of: date = AS_OF) -> dict[str, KPI]:
    """The XBRL half on its own, with no company facts underneath it."""
    return extract_from_xbrl(
        CompanyFacts(NO_COMPANY_FACTS, ticker), KpiClient(instance=ticker), ticker, as_of
    )


# --------------------------------------------------------------------------- #
# Tagged facts
# --------------------------------------------------------------------------- #


def test_rpo_comes_off_the_standard_concept_exactly():
    """Datadog tags 3,471.4mm at 30 June 2026. A tagged number is not rounded."""
    rpo = tagged("DDOG")["rpo"]
    assert rpo.value == pytest.approx(3_471.4)
    assert rpo.unit == "usd_mm"
    assert rpo.period_end == date(2026, 6, 30)
    assert rpo.source == "xbrl_extension"
    assert rpo.tag_or_phrase == "RevenueRemainingPerformanceObligation"
    assert rpo.confidence == 1.0
    assert rpo.usable


def test_the_note_names_the_element_the_accession_and_which_taxonomy():
    """A reader has to be able to find the fact in the filing without asking."""
    note = tagged("DDOG")["rpo"].notes
    assert "RevenueRemainingPerformanceObligation" in note
    assert "standard us-gaap concept" in note
    assert "0001628280-26-054458" in note


def test_a_filer_extension_is_read_and_is_labelled_as_one():
    """Netflix's content tags are its own, and the note must not imply otherwise.

    This is the case a generic engine cannot reach: neither element exists in
    us-gaap, so neither appears in companyfacts, and there are no company facts
    behind this test at all.
    """
    out = tagged("NFLX")
    assert out["content_spend"].value == pytest.approx(9_774.44)
    assert out["content_spend"].tag_or_phrase == "AdditionstoStreamingContentAssets"
    assert "filer extension element" in out["content_spend"].notes

    amortisation = out["content_amortisation"]
    assert amortisation.value == pytest.approx(8_529.209)
    assert amortisation.tag_or_phrase == (
        "CostofServicesAmortizationofStreamingContentAssets"
    )
    assert "filer extension element" in amortisation.notes


def test_a_flow_that_cannot_be_tiled_to_a_year_says_so_rather_than_annualising():
    """Netflix's 10-Q reports half years, and four times a half year is fiction."""
    note = tagged("NFLX")["content_spend"].notes
    assert "181 days from 2026-01-01 to 2026-06-30" in note
    assert "not annualised" in note


def test_dimensioned_facts_are_parts_and_never_summed_into_a_total():
    """The licensed and produced components must not become a third reading.

    Netflix tags 8,529.209 undimensioned and, separately, 4,546.776 licensed and
    3,982.433 produced along the major-class axis. Summing those to the same
    total by accident would be lucky; summing a set that is not exhaustive would
    be wrong, so parts are dropped and only the tagged total is read.
    """
    assert tagged("NFLX")["content_amortisation"].value == pytest.approx(8_529.209)


def test_a_concept_tagged_only_along_an_axis_is_refused_not_absent():
    """Silence and unreadability are different findings and must look different."""
    fact = DimensionedFact(
        tag="AnnualRecurringRevenue",
        value=4.05e9,
        unit="USD",
        start=None,
        end=date(2026, 6, 30),
        dimensions={"us-gaap:StatementBusinessSegmentsAxis": "x:CloudMember"},
    )
    kpi = _one_concept("arr", [fact])
    assert kpi is not None
    assert kpi.confidence == CONFIDENCE_REFUSED
    assert not kpi.usable
    assert "reported along an axis" in kpi.notes


def test_an_undimensioned_extension_resolves_at_full_confidence():
    """The same element without the axis is the total, and is believed."""
    fact = DimensionedFact(
        tag="AnnualRecurringRevenue",
        value=4.05e9,
        unit="USD",
        start=None,
        end=date(2026, 6, 30),
        dimensions={},
    )
    kpi = _one_concept("arr", [fact])
    assert kpi is not None
    assert kpi.value == pytest.approx(4_050.0)
    assert kpi.confidence == 1.0


def test_a_count_tagged_in_dollars_is_refused():
    """A USD balance dropped into a customer count would never be questioned."""
    fact = DimensionedFact(
        tag="NumberOfCustomers",
        value=3_390.0,
        unit="USD",
        start=None,
        end=date(2026, 6, 30),
        dimensions={},
    )
    kpi = _one_concept("customers", [fact])
    assert kpi is not None
    assert kpi.confidence == CONFIDENCE_REFUSED
    assert "a count tagged in USD is not a count" in kpi.notes


def _one_concept(name: str, facts: list[DimensionedFact]) -> KPI | None:
    """Run the tagged path over a handful of synthetic facts."""

    class OneFiling:
        def company_facts(self, ticker: str) -> CompanyFacts:
            return CompanyFacts(NO_COMPANY_FACTS, ticker)

        def instance_facts(self, ticker, forms=("10-K", "10-Q")):
            return facts, "0000000000-26-000000", date(2026, 8, 6)

    out = extract_from_xbrl(
        CompanyFacts(NO_COMPANY_FACTS, "TEST"), OneFiling(), "TEST", AS_OF
    )
    return out.get(name)


# --------------------------------------------------------------------------- #
# The conflict path
# --------------------------------------------------------------------------- #


def test_three_undimensioned_values_at_one_date_are_all_reported_and_none_believed():
    """T-Mobile's 10-Q is the real case, and the honest answer is to refuse.

    694mm, 1,000mm and 2,300mm are tagged to the same element at the same date
    with no axis on any of them. Picking the largest, the first or the sum would
    each be a guess dressed as a number.
    """
    rpo = tagged("TMUS")["rpo"]
    assert rpo.confidence == CONFIDENCE_REFUSED
    assert not rpo.usable
    assert "3 different undimensioned values" in rpo.notes
    for shown in ("694", "1,000", "2,300"):
        assert shown in rpo.notes


def test_a_flow_tagged_twice_over_one_window_with_two_values_is_refused():
    """The instant path is not the only way a filing can contradict itself.

    Content spend is a flow, so it goes through the period algebra rather than
    the balance-at-a-date path, and a filer that tags two different amounts over
    the same six months has given no total to tile. The conflict has to surface
    at the period it sits at, not at the as-of date.
    """
    facts = [
        DimensionedFact(
            tag="AdditionstoStreamingContentAssets",
            value=value,
            unit="USD",
            start=date(2026, 1, 1),
            end=date(2026, 6, 30),
            dimensions={},
        )
        for value in (9_774.44e6, 8_100.0e6)
    ]
    kpi = _one_concept("content_spend", facts)
    assert kpi is not None
    assert kpi.confidence == CONFIDENCE_REFUSED
    assert kpi.period_end == date(2026, 6, 30)
    assert "2 different undimensioned values" in kpi.notes
    assert "9,774" in kpi.notes and "8,100" in kpi.notes


def test_a_refused_conflict_never_reaches_the_model_facing_view():
    a = Assumptions()
    a.tmt.sub_vertical = "application_software"
    result = build_kpis("TMUS", KpiClient(instance="TMUS"), a, as_of=AS_OF)
    assert result.kpis["rpo"].value == pytest.approx(694.0)
    assert result.as_dict()["rpo"] is None
    assert any("rpo" in flag for flag in result.flags)


def test_a_tagged_value_and_a_text_value_that_disagree_are_both_kept():
    """Silently preferring one is how a wrong KPI reaches a page."""
    result = build_kpis(
        "DDOG",
        KpiClient(instance="DDOG", facts="DDOG"),
        software(),
        as_of=AS_OF,
        mdna_text="Remaining performance obligations (RPO) of $5.10 billion at period end.",
    )
    assert result.kpis["rpo"].value == pytest.approx(3_471.4)
    assert result.kpis["rpo"].source == "xbrl_extension"
    assert result.kpis["rpo__text"].value == pytest.approx(5_100.0)
    assert result.kpis["rpo__text"].source == "text"
    flag = next(f for f in result.flags if f.startswith("rpo:"))
    assert "disagree" in flag
    assert "3,471" in flag and "5,100" in flag


def test_a_text_value_that_agrees_is_recorded_as_corroboration_not_a_conflict():
    """Within the tolerance the two readings are one reading, and say so."""
    result = build_kpis(
        "DDOG",
        KpiClient(instance="DDOG", facts="DDOG"),
        software(),
        as_of=AS_OF,
        mdna_text="Remaining performance obligations (RPO) of $3.47 billion at period end.",
    )
    assert "rpo__text" not in result.kpis
    assert "the text agrees" in result.kpis["rpo"].notes
    assert not any(f.startswith("rpo:") for f in result.flags)


def test_two_different_values_in_one_text_are_reported_together_and_refused():
    out = extract_from_text(
        "Net revenue retention rate of 115%. On a trailing basis, net revenue "
        "retention rate was 108%.",
        AS_OF,
    )
    nrr = out["net_revenue_retention"]
    assert nrr.confidence == CONFIDENCE_REFUSED
    assert "2 different values" in nrr.notes
    assert "115%" in nrr.tag_or_phrase and "108%" in nrr.tag_or_phrase


# --------------------------------------------------------------------------- #
# Near misses. A keyword search would take all of these.
# --------------------------------------------------------------------------- #


# The elements a substring search would take, and the filing each sits in. These
# are the reason the concept ladders are anchored full-match patterns.
NEAR_MISSES = {
    "TMUS": [
        "NumberOfCustomerAccountsImpacted",
        "NumberofCustomerClasses",
        "NumberOfCustomerCategories",
        "NumberOfCompaniesIncludedInJointVenture",
    ],
    "NET": [
        "NumberOfSegmentManagers",
        "NumberOfReportableSegments",
        "ProvisionForDoubtfulAccountsExcludingPayAsYouGoCustomers",
        "RevenueRemainingPerformanceObligationPercentage",
    ],
    "NFLX": ["ContentAssetsNetNoncurrent", "NumberOfOperatingSegments"],
    "DDOG": ["NumberOfCommonStockClasses", "ContractWithCustomerLiabilityCurrent"],
}


@pytest.mark.parametrize("ticker", sorted(NEAR_MISSES))
def test_the_near_miss_elements_really_are_in_these_filings(ticker: str):
    """The trap has to exist for the test that avoids it to mean anything."""
    present = {fact.tag for fact in load_instance(ticker)[0]}
    assert set(NEAR_MISSES[ticker]) <= present


@pytest.mark.parametrize("ticker", sorted(NEAR_MISSES))
@pytest.mark.parametrize(
    "metric",
    [
        "arr",
        "customers",
        "customers_over_100k",
        "subscribers",
        "paid_subscribers",
        "net_revenue_retention",
        "gross_revenue_retention",
        "arpu",
        "churn",
    ],
)
def test_no_filer_tags_the_metrics_that_live_only_in_prose(ticker: str, metric: str):
    """Nine metrics, four filings across software, streaming and telecom, nothing.

    This is the measurement the whole text path rests on. If a later taxonomy
    release gives one of these a standard concept, or a filer starts extending
    for it, this test turns red and the module's central claim needs rewriting
    rather than quietly surviving.

    It also proves the near misses above stay out: T-Mobile's
    NumberOfCustomerCategories is not a customer count, and Cloudflare's
    ProvisionForDoubtfulAccountsExcludingPayAsYouGoCustomers is not one either.
    """
    assert metric not in tagged(ticker)


def test_the_rpo_percentage_element_is_not_read_as_an_obligation():
    """0.64 is a share, not 0.64mm of remaining performance obligation."""
    rpo = tagged("NET")["rpo"]
    assert rpo.value == pytest.approx(2_732.0)
    assert rpo.tag_or_phrase == "RevenueRemainingPerformanceObligation"


# --------------------------------------------------------------------------- #
# The derived current portion of RPO
# --------------------------------------------------------------------------- #


def test_rpo_current_is_derived_from_the_obligation_and_the_tagged_share():
    """2,732.0 x 0.64 = 1,748.48, and the filing states neither the product nor
    the current portion, so it is derived and labelled as such."""
    current = tagged("NET")["rpo_current"]
    assert current.value == pytest.approx(2_732.0 * 0.64)
    assert current.value == pytest.approx(1_748.48)
    assert current.source == "derived"
    assert current.confidence == 0.8
    assert "RevenueRemainingPerformanceObligationPercentage" in current.tag_or_phrase


def test_the_window_read_into_an_unnamed_percentage_is_stated_not_buried():
    note = tagged("NET")["rpo_current"].notes
    assert "does not name its window" in note
    assert "is an assumption" in note


def test_two_percentages_at_one_date_derive_nothing():
    """Different horizons, and the document does not say which is the year."""
    facts = [
        DimensionedFact(
            tag="RevenueRemainingPerformanceObligation",
            value=1_000e6,
            unit="USD",
            start=None,
            end=date(2026, 6, 30),
            dimensions={},
        ),
        DimensionedFact(
            tag="RevenueRemainingPerformanceObligationPercentage",
            value=0.45,
            unit="pure",
            start=None,
            end=date(2026, 6, 30),
            dimensions={},
        ),
        DimensionedFact(
            tag="RevenueRemainingPerformanceObligationPercentage",
            value=0.90,
            unit="pure",
            start=None,
            end=date(2026, 6, 30),
            dimensions={},
        ),
    ]
    current = _one_concept("rpo_current", facts)
    assert current is not None
    assert current.confidence == CONFIDENCE_REFUSED
    assert "0.45" in current.notes and "0.9" in current.notes


# --------------------------------------------------------------------------- #
# Billings
# --------------------------------------------------------------------------- #


def test_billings_matches_the_hand_arithmetic_in_its_own_docstring():
    """The worked example in ``derive_billings``, on the committed DDOG facts:

        revenue                            3,966.725
        deferred revenue at 2026-06-30     1,286.690 + 50.860 = 1,337.550
        deferred revenue at 2025-06-30       966.442 + 29.866 =   996.308
        billings = 3,966.725 + 1,337.550 - 996.308 = 4,307.967

    If the code and the docstring ever part company, this is where it shows.
    """
    billings = derive_billings(load_company_facts("DDOG"), AS_OF)
    assert billings is not None
    assert billings.value == pytest.approx(3_966.725 + 1_337.550 - 996.308)
    assert billings.value == pytest.approx(4_307.967)
    assert billings.period_end == date(2026, 6, 30)
    assert billings.unit == "usd_mm"


def test_billings_is_marked_derived_and_carries_the_caveat():
    """It approximates invoiced value. A page must not read it as disclosure."""
    billings = derive_billings(load_company_facts("DDOG"), AS_OF)
    assert billings is not None
    assert billings.source == "derived"
    assert billings.confidence == 0.8
    assert "approximates invoiced value" in billings.notes
    assert "contract terms shift" in billings.notes
    assert "acquisition" in billings.notes


@pytest.mark.parametrize("ticker", ["DDOG", "CRWD", "MDB", "ZS"])
def test_billings_exceeds_revenue_for_every_growing_software_filer(ticker: str):
    """Not an identity, but a check the derivation is the right way round.

    A subscription business whose deferred revenue balance is growing invoices
    more than it recognises. A sign error in the change would invert this for all
    four at once, which is exactly the bug a single-company test would miss.
    """
    facts = load_company_facts(ticker)
    billings = derive_billings(facts, AS_OF)
    assert billings is not None
    revenue, _prov = facts.resolve_ttm("revenue", tags.REVENUE, billings.period_end)
    assert revenue is not None
    assert billings.value > revenue / 1_000_000.0


def test_billings_refuses_rather_than_guessing_an_opening_balance():
    """No facts at all means no billings, not a billings figure built on zero."""
    assert derive_billings(CompanyFacts(NO_COMPANY_FACTS, "TEST"), AS_OF) is None


def test_billings_uses_total_deferred_revenue_not_only_the_current_part():
    """A multi-year prepayment is invoiced value whichever side of a year it lands.

    Datadog's noncurrent balances are 50.860 and 29.866, so the current-only
    convention would come in 21.0mm lower. Asserting the gap keeps the choice
    visible rather than implicit.
    """
    billings = derive_billings(load_company_facts("DDOG"), AS_OF)
    assert billings is not None
    current_only = 3_966.725 + 1_286.690 - 966.442
    assert billings.value - current_only == pytest.approx(50.860 - 29.866, abs=1e-6)


# --------------------------------------------------------------------------- #
# Text: retention
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("Our net revenue retention rate of 115% reflects expansion.", 1.15),
        ("Dollar-based net retention rate was 111% as of June 30, 2026.", 1.11),
        ("Net dollar retention rate 109% in the period.", 1.09),
        ("Our dollar based net expansion rate was 120 percent.", 1.20),
        ("Net revenue retention reached 118%.", 1.18),
    ],
)
def test_every_retention_phrasing_reads_to_the_same_number(sentence, expected):
    kpi = extract_from_text(sentence, AS_OF)["net_revenue_retention"]
    assert kpi.value == pytest.approx(expected)
    assert kpi.unit == "ratio"
    assert kpi.source == "text"
    assert kpi.confidence == 0.75
    assert kpi.period_end == AS_OF


def test_the_matched_phrase_travels_with_the_figure():
    """The point of the phrase is that a reader can grep the filing for it."""
    kpi = extract_from_text(
        "As disclosed, our dollar-based net retention rate was 111% for the period.",
        AS_OF,
    )["net_revenue_retention"]
    assert kpi.tag_or_phrase == "dollar-based net retention rate was 111%"


@pytest.mark.parametrize(
    "sentence, qualifier",
    [
        ("Net dollar retention of over 120% for the period.", "over"),
        ("Net revenue retention rate of more than 130%.", "more than"),
        ("Net revenue retention rate in excess of 125%.", "in excess of"),
        ("Net revenue retention rate of at least 110%.", "at least"),
    ],
)
def test_a_bounded_hedge_is_flagged_as_a_bound_and_dropped_in_confidence(
    sentence, qualifier
):
    """"Over 120%" is not 120%. It is wrong in a known direction, every time."""
    kpi = extract_from_text(sentence, AS_OF)["net_revenue_retention"]
    assert kpi.confidence == 0.4
    assert "hedged" in kpi.notes
    assert qualifier in kpi.notes
    assert "a bound and not the figure" in kpi.notes


def test_an_estimate_hedge_is_flagged_differently_from_a_bound():
    """"Approximately 108%" is a centre, not a floor, and the note must say so."""
    kpi = extract_from_text("Net revenue retention was approximately 108%.", AS_OF)[
        "net_revenue_retention"
    ]
    assert kpi.value == pytest.approx(1.08)
    assert kpi.confidence == 0.4
    assert "own estimate rather than a measurement" in kpi.notes
    assert "bound" not in kpi.notes


def test_a_rate_without_a_percent_sign_is_refused():
    """115 is 115% or 1.15, and the sentence does not settle it."""
    kpi = extract_from_text("Net revenue retention of 115 for the period.", AS_OF)[
        "net_revenue_retention"
    ]
    assert kpi.confidence == CONFIDENCE_REFUSED
    assert not kpi.usable
    assert "no percent sign" in kpi.notes


def test_gross_retention_is_a_separate_metric_from_net_retention():
    out = extract_from_text(
        "Gross revenue retention rate was 97% while our net revenue retention rate "
        "of 115% reflects expansion.",
        AS_OF,
    )
    assert out["gross_revenue_retention"].value == pytest.approx(0.97)
    assert out["net_revenue_retention"].value == pytest.approx(1.15)


# --------------------------------------------------------------------------- #
# Text: money and scale
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("Annual recurring revenue (ARR) of $1.25 billion at period end.", 1_250.0),
        ("ARR was $4,050 million as of June 30, 2026.", 4_050.0),
        ("Annualized recurring revenue of $850 million.", 850.0),
        ("ARR of $1,250,000,000 at period end.", 1_250.0),
    ],
)
def test_a_dollar_figure_with_a_resolvable_scale_is_read(sentence, expected):
    kpi = extract_from_text(sentence, AS_OF)["arr"]
    assert kpi.value == pytest.approx(expected)
    assert kpi.unit == "usd_mm"
    assert kpi.confidence == 0.75


def test_a_dollar_figure_with_no_scale_word_below_the_floor_is_refused():
    """"$1,250" is 1,250 dollars or 1,250 million, six orders of magnitude apart."""
    kpi = extract_from_text("ARR of $1,250 at the end of the quarter.", AS_OF)["arr"]
    assert kpi.confidence == CONFIDENCE_REFUSED
    assert "no scale word" in kpi.notes
    assert kpi.tag_or_phrase == "ARR of $1,250"


def test_the_scale_floor_is_configurable_and_the_refusal_moves_with_it():
    """A filer that writes its tables in thousands can be told so, once."""
    loose = KPISettings(text_scale_floor=1_000.0)
    kpi = extract_from_text("ARR of $1,250 at quarter end.", AS_OF, settings=loose)["arr"]
    assert kpi.usable
    assert kpi.value == pytest.approx(1_250.0 / 1_000_000.0)


# --------------------------------------------------------------------------- #
# Text: counts and scale
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sentence, metric, expected",
    [
        ("We ended the quarter with 2.4 million subscribers.", "subscribers", 2_400_000.0),
        ("We had 3,390 customers as of June 30, 2026.", "customers", 3_390.0),
        ("Postpaid phone net additions of 830,000 in the quarter.", "postpaid_net_adds", 830_000.0),
        ("301.6 million paid memberships worldwide", "paid_subscribers", 301_600_000.0),
        ("We reported 1.7 million total subscribers.", "subscribers", 1_700_000.0),
    ],
)
def test_a_count_with_a_resolvable_scale_is_read(sentence, metric, expected):
    kpi = extract_from_text(sentence, AS_OF)[metric]
    assert kpi.value == pytest.approx(expected)
    assert kpi.unit == "count"


def test_a_fractional_count_with_no_scale_word_is_refused():
    """"2.4 subscribers" means millions and is still not something to infer."""
    kpi = extract_from_text("We ended the quarter with 2.4 subscribers.", AS_OF)[
        "subscribers"
    ]
    assert kpi.confidence == CONFIDENCE_REFUSED
    assert "fractional count" in kpi.notes


def test_counts_stay_counts_and_are_not_scaled_into_millions():
    """3,390 customers is 3,390, not 0.003390. Money is millions; people are not."""
    kpi = extract_from_text("We had 3,390 customers at period end.", AS_OF)["customers"]
    assert kpi.value == 3_390.0


# --------------------------------------------------------------------------- #
# Text: ARPU
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sentence, unit",
    [
        ("revenue of $54.32 per subscriber per month", "usd_per_month"),
        ("Average revenue per user was $54.32 per month.", "usd_per_month"),
        ("ARPU of $162.96 per quarter.", "usd_per_quarter"),
        ("Average revenue per membership was $12.15 per month.", "usd_per_month"),
    ],
)
def test_arpu_carries_the_period_it_is_per(sentence, unit):
    """A price with no denominator is not a price, so the denominator is in the unit."""
    kpi = extract_from_text(sentence, AS_OF)["arpu"]
    assert kpi.unit == unit
    assert kpi.confidence == 0.75


def test_arpu_without_a_period_is_reported_at_low_confidence_with_the_reason():
    kpi = extract_from_text("Average revenue per user was $54.32.", AS_OF)["arpu"]
    assert kpi.value == pytest.approx(54.32)
    assert kpi.unit == "usd_per_period"
    assert kpi.confidence == 0.4
    assert "does not say per what period" in kpi.notes


def test_an_arpu_in_millions_is_a_misread_unit_and_is_refused():
    kpi = extract_from_text("ARPU of $54.32 million in the quarter.", AS_OF)["arpu"]
    assert kpi.confidence == CONFIDENCE_REFUSED
    assert "misread unit" in kpi.notes


# --------------------------------------------------------------------------- #
# Text: ordering
# --------------------------------------------------------------------------- #


def test_the_large_customer_phrase_is_consumed_before_arr_can_misread_it():
    """"3,390 customers with ARR of $100,000 or more" contains no ARR figure.

    Read by the ARR rule it would say the company's annual recurring revenue is
    one hundred thousand dollars, which is the kind of error that survives review
    because the number is printable and the label is right.
    """
    out = extract_from_text(
        "We had 3,390 customers with ARR of $100,000 or more, up from 3,010.", AS_OF
    )
    assert out["customers_over_100k"].value == 3_390.0
    assert "arr" not in out
    assert "customers" not in out


def test_a_paragraph_gives_each_metric_once():
    """The whole point, on prose shaped like a real Item 7."""
    out = extract_from_text(
        "Revenue grew 25% year over year. Our dollar-based net retention rate was "
        "111% as of June 30, 2026, and we ended the period with 3,390 customers "
        "with ARR of $100,000 or more. Annual recurring revenue (ARR) of $4.05 "
        "billion. Remaining performance obligations (RPO) of $3.47 billion. Gross "
        "revenue retention rate was 97%.",
        AS_OF,
    )
    assert out["net_revenue_retention"].value == pytest.approx(1.11)
    assert out["customers_over_100k"].value == 3_390.0
    assert out["arr"].value == pytest.approx(4_050.0)
    assert out["rpo"].value == pytest.approx(3_470.0)
    assert out["gross_revenue_retention"].value == pytest.approx(0.97)


def test_prose_that_names_the_metrics_without_stating_them_yields_nothing():
    """Risk factors are where a loose pattern goes to manufacture numbers.

    Every metric this module looks for is named in the paragraph below, and the
    paragraph is full of dollar figures, and not one of them is an operating
    metric. A rule that skipped words rather than only whitespace between the
    label and the number would report revenue as ARR and a Fortune 100 reference
    as a customer count.
    """
    assert (
        extract_from_text(
            "Risk Factors. If we are unable to retain customers, our net revenue "
            "retention rate could decline. Our churn could increase. Our revenue "
            "was $1,234.5 million for the year. Cost of revenue was $310.2 "
            "million. We had cash and cash equivalents of $2.1 billion. Our "
            "customers include 12 of the Fortune 100. We expect net revenue "
            "retention to remain above our historical range. Subscribers to our "
            "newsletter receive updates.",
            AS_OF,
        )
        == {}
    )


def test_empty_or_missing_text_finds_nothing_and_does_not_raise():
    assert extract_from_text(None, AS_OF) == {}
    assert extract_from_text("", AS_OF) == {}
    assert extract_from_text("   \n  ", AS_OF) == {}
    assert extract_from_text("This filing discusses risk factors at length.", AS_OF) == {}


# --------------------------------------------------------------------------- #
# Sanity bounds
# --------------------------------------------------------------------------- #


def test_a_misread_percent_sign_is_flagged_and_never_corrected():
    """11500% is not 115%. Dividing by a hundred makes three errors look like one."""
    kpi = extract_from_text("Net revenue retention rate of 11500%.", AS_OF)[
        "net_revenue_retention"
    ]
    assert kpi.value == pytest.approx(115.0)
    assert kpi.confidence == CONFIDENCE_REFUSED
    assert "outside [0.5, 2.0]" in kpi.notes
    assert "percent sign" in kpi.notes


def test_churn_above_one_is_flagged():
    kpi = extract_from_text("Annual churn rate of 140%.", AS_OF)["churn"]
    assert kpi.value == pytest.approx(1.4)
    assert kpi.confidence == CONFIDENCE_REFUSED
    assert "cannot exceed it" in kpi.notes


def _kpi(name: str, value: float, unit: str = "ratio") -> KPI:
    return KPI(
        name=name,
        value=value,
        unit=unit,
        period_end=AS_OF,
        source="text",
        tag_or_phrase="synthetic",
        confidence=0.75,
    )


def test_every_bound_flags_without_changing_the_value():
    out, flags = apply_sanity_bounds(
        {
            "net_revenue_retention": _kpi("net_revenue_retention", 115.0),
            "churn": _kpi("churn", 1.4),
            "arpu": _kpi("arpu", -3.0, "usd_per_month"),
            "customers": _kpi("customers", 0.0, "count"),
            "billings": _kpi("billings", -500.0, "usd_mm"),
        },
        KPISettings(),
    )
    assert len(flags) == 5
    assert out["net_revenue_retention"].value == pytest.approx(115.0)
    assert out["churn"].value == pytest.approx(1.4)
    assert out["arpu"].value == pytest.approx(-3.0)
    assert out["billings"].value == pytest.approx(-500.0)
    assert all(not kpi.usable for kpi in out.values())


def test_arr_an_order_of_magnitude_from_revenue_is_flagged():
    """A run rate and a trailing year differ by growth, never by a factor of ten."""
    out, flags = apply_sanity_bounds(
        {"arr": _kpi("arr", 40_500.0, "usd_mm")}, KPISettings(), revenue=3_966.725
    )
    assert not out["arr"].usable
    assert "times trailing revenue" in flags[0]


def test_arr_within_an_order_of_magnitude_of_revenue_passes():
    out, flags = apply_sanity_bounds(
        {"arr": _kpi("arr", 4_050.0, "usd_mm")}, KPISettings(), revenue=3_966.725
    )
    assert out["arr"].usable
    assert flags == []


def test_bounds_do_not_re_flag_a_figure_that_was_already_refused():
    """The gate runs in both the text reader and the assembly, so it must be idempotent."""
    refused = KPI(
        name="churn",
        value=1.4,
        unit="ratio",
        period_end=AS_OF,
        source="text",
        tag_or_phrase="synthetic",
        confidence=CONFIDENCE_REFUSED,
        notes="already refused",
    )
    out, flags = apply_sanity_bounds({"churn": refused}, KPISettings())
    assert flags == []
    assert out["churn"].notes == "already refused"


def test_inverted_bounds_are_a_config_error_not_a_silent_reject_everything():
    with pytest.raises(ConfigError, match="0 < min < max"):
        KPISettings(nrr_min=2.0, nrr_max=0.5)
    with pytest.raises(ConfigError, match="conflict_tolerance"):
        KPISettings(conflict_tolerance=0.0)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


MDNA = (
    "Item 7. Management's Discussion and Analysis of Financial Condition. "
    "Our dollar-based net retention rate was 111% as of June 30, 2026. We ended "
    "the period with 3,390 customers with ARR of $100,000 or more. Annual "
    "recurring revenue (ARR) of $4.05 billion. Gross revenue retention rate was 97%."
)


def test_a_software_filer_assembles_tagged_derived_and_parsed_figures_together():
    result = build_kpis(
        "DDOG",
        KpiClient(instance="DDOG", facts="DDOG"),
        software(),
        as_of=AS_OF,
        mdna_text=MDNA,
    )
    assert isinstance(result, KPISet)
    assert result.ticker == "DDOG"
    assert result.as_of == AS_OF

    assert result.kpis["rpo"].source == "xbrl_extension"
    assert result.kpis["billings"].source == "derived"
    assert result.kpis["arr"].source == "text"
    assert result.kpis["net_revenue_retention"].value == pytest.approx(1.11)
    assert result.kpis["customers_over_100k"].value == 3_390.0
    assert result.kpis["billings"].value == pytest.approx(4_307.967)


def test_absence_is_reported_as_a_reason_rather_than_an_omission():
    """A filer that does not disclose a metric and a broken run must not look alike."""
    result = build_kpis(
        "DDOG", KpiClient(instance="DDOG", facts="DDOG"), software(), as_of=AS_OF
    )
    assert set(result.missing) | set(result.kpis) >= set(SOFTWARE_METRICS)
    assert "no MD&A text was supplied" in result.missing["net_revenue_retention"]
    for reason in result.missing.values():
        assert reason


def test_a_company_with_none_of_these_metrics_returns_an_empty_set_with_reasons():
    """A software filer measured against the media pack discloses nothing in it."""
    result = build_kpis(
        "DDOG", KpiClient(instance="DDOG", facts="DDOG"), media(), as_of=AS_OF
    )
    assert result.kpis == {}
    assert set(result.missing) == set(MEDIA_METRICS)
    assert all(result.as_dict()[name] is None for name in MEDIA_METRICS)
    assert result.to_frame().empty


def test_a_sub_vertical_with_no_pack_says_so_instead_of_searching_for_nothing():
    """A chip maker has no net revenue retention, and hunting for one is not coverage."""
    a = Assumptions()
    a.tmt.sub_vertical = "semiconductors"
    result = build_kpis("DDOG", KpiClient(instance="DDOG", facts="DDOG"), a, as_of=AS_OF)
    assert result.kpis == {}
    assert result.missing == {}
    assert any("no operating metric pack applies" in note for note in result.notes)


def test_an_unknown_sub_vertical_is_a_config_error():
    a = Assumptions()
    a.tmt.sub_vertical = "biotech"
    with pytest.raises(ConfigError, match="no operating metric pack"):
        build_kpis("DDOG", KpiClient(instance="DDOG", facts="DDOG"), a, as_of=AS_OF)


def test_no_sub_vertical_searches_every_pack():
    a = Assumptions()
    result = build_kpis("DDOG", KpiClient(instance="DDOG", facts="DDOG"), a, as_of=AS_OF)
    assert set(result.missing) | set(result.kpis) == set(ALL_METRICS)


def test_extraction_can_be_switched_off_and_says_which_knob_did_it():
    a = software()
    a.tmt.kpi_extraction = False
    result = build_kpis("DDOG", KpiClient(instance="DDOG", facts="DDOG"), a, as_of=AS_OF)
    assert result.kpis == {}
    assert set(result.missing) == set(SOFTWARE_METRICS)
    assert all("switched off" in reason for reason in result.missing.values())


def test_a_client_without_instance_facts_degrades_to_company_facts_and_says_so():
    """The conftest client is one of these, and a missing method is not a crash."""
    result = build_kpis("DDOG", NoInstanceClient(), software(), as_of=AS_OF)
    assert "billings" in result.kpis
    assert any("no instance_facts" in note for note in result.notes)


def test_a_media_filer_reads_its_extensions_and_its_prose_together():
    """Netflix's content tags plus the subscriber language no filer ever tags.

    The company facts behind this client are deliberately empty, so every figure
    asserted here demonstrably came from the committed instance document or from
    the prose, and none of it from a us-gaap fallback.
    """
    result = build_kpis(
        "NFLX",
        KpiClient(instance="NFLX"),
        media(),
        as_of=AS_OF,
        mdna_text=(
            "We ended the quarter with approximately 301.6 million paid memberships. "
            "Average revenue per membership was $12.15 per month. Monthly churn "
            "rate of 1.9%."
        ),
    )
    assert result.kpis["content_spend"].source == "xbrl_extension"
    assert result.kpis["content_amortisation"].value == pytest.approx(8_529.209)
    assert result.kpis["paid_subscribers"].value == pytest.approx(301_600_000.0)
    assert result.kpis["paid_subscribers"].confidence == 0.4  # hedged: "approximately"
    assert result.kpis["arpu"].unit == "usd_per_month"
    assert result.kpis["churn"].value == pytest.approx(0.019)
    assert "subscribers" in result.missing


# --------------------------------------------------------------------------- #
# Point in time
# --------------------------------------------------------------------------- #


def test_nothing_filed_after_the_as_of_date_is_read():
    """Datadog tags 3,461.2mm at 2025-12-31 and 3,471.4mm at 2026-06-30.

    Asked as of March 2026 the answer has to be the December figure. Reading the
    June one is hindsight, and a backtest built on hindsight measures nothing.
    """
    client = KpiClient(instance="DDOG", facts="DDOG")
    now = build_kpis("DDOG", client, software(), as_of=AS_OF)
    then = build_kpis("DDOG", client, software(), as_of=date(2026, 3, 31))
    assert now.kpis["rpo"].value == pytest.approx(3_471.4)
    assert now.kpis["rpo"].period_end == date(2026, 6, 30)
    assert then.kpis["rpo"].value == pytest.approx(3_461.2)
    assert then.kpis["rpo"].period_end == date(2025, 12, 31)


def test_the_client_knowledge_date_outranks_the_assumptions_file():
    """A client refusing later filings must not be handed a later as-of date."""
    a = software()
    a.as_of = "2026-09-10"
    client = KpiClient(instance="DDOG", facts="DDOG", knowledge_date=date(2026, 3, 31))
    result = build_kpis("DDOG", client, a)
    assert result.as_of == date(2026, 3, 31)
    assert result.kpis["rpo"].value == pytest.approx(3_461.2)


def test_the_assumptions_as_of_is_used_when_the_client_has_no_knowledge_date():
    a = software()
    a.as_of = "2026-03-31"
    result = build_kpis("DDOG", KpiClient(instance="DDOG", facts="DDOG"), a)
    assert result.as_of == date(2026, 3, 31)


def test_an_unparseable_as_of_is_a_config_error():
    a = software()
    a.as_of = "March 2026"
    with pytest.raises(ConfigError, match="ISO date"):
        build_kpis("DDOG", KpiClient(instance="DDOG", facts="DDOG"), a)


# --------------------------------------------------------------------------- #
# Settings, output shape and determinism
# --------------------------------------------------------------------------- #


def test_thresholds_come_from_assumptions_where_they_are_configured():
    """The knob is a bool today and a settings object later. Both must work."""
    assert settings_from_assumptions(Assumptions()).source == "module defaults"

    class Knob:
        nrr_max = 1.6
        text_scale_floor = 1_000.0

    class Tmt:
        kpi_extraction = Knob()
        sub_vertical = None

    class Fake:
        tmt = Tmt()

    configured = settings_from_assumptions(Fake())
    assert configured.nrr_max == 1.6
    assert configured.text_scale_floor == 1_000.0
    assert configured.nrr_min == KPISettings().nrr_min
    assert "nrr_max" in configured.source and "text_scale_floor" in configured.source


def test_the_frame_carries_every_column_a_reader_needs_to_audit_a_figure():
    result = build_kpis(
        "DDOG",
        KpiClient(instance="DDOG", facts="DDOG"),
        software(),
        as_of=AS_OF,
        mdna_text=MDNA,
    )
    frame = result.to_frame()
    assert list(frame.columns) == [
        "name",
        "value",
        "unit",
        "period_end",
        "source",
        "tag_or_phrase",
        "confidence",
        "notes",
    ]
    assert len(frame) == len(result.kpis)
    assert frame["name"].is_monotonic_increasing


def test_an_empty_set_still_declares_its_columns():
    empty = KPISet(ticker="TEST", as_of=AS_OF)
    assert empty.to_frame().empty
    assert "tag_or_phrase" in empty.to_frame().columns
    assert empty.get("arr") is None


def test_as_dict_withholds_every_figure_nobody_will_stand_behind():
    result = build_kpis(
        "DDOG",
        KpiClient(instance="DDOG", facts="DDOG"),
        software(),
        as_of=AS_OF,
        mdna_text="Net revenue retention of 115 for the period. " + MDNA,
    )
    assert result.kpis["net_revenue_retention"].confidence == CONFIDENCE_REFUSED
    assert result.as_dict()["net_revenue_retention"] is None
    assert result.as_dict()["rpo"] == pytest.approx(3_471.4)
    assert result.as_dict()["customers"] is None


@pytest.mark.parametrize("ticker", ["DDOG", "NET", "NFLX", "TMUS"])
def test_each_fixture_says_where_it_came_from_and_the_claim_holds(ticker: str):
    """A pruned fixture is evidence only while its own provenance note is true.

    The note states the substring filter used to prune the instance document. If
    a later hand edit adds an element the filter would not have kept, this fails,
    and the file stops being something a reader can reproduce from EDGAR.
    """
    payload = json.loads((FIXTURES / f"instance_kpis_{ticker}.json").read_text())
    assert payload["_source"] == "SEC EDGAR XBRL instance document"
    assert payload["_ticker"] == ticker
    assert date.fromisoformat(payload["_retrieved"]) == date(2026, 9, 11)
    filed = date.fromisoformat(payload["_filed"])

    words = re.search(r"contains any of: ([^.]+)\.", payload["_note"]).group(1)
    keywords = [word.strip() for word in words.split(",")]
    for row in payload["facts"]:
        assert any(word in row["tag"].lower() for word in keywords), row["tag"]
        assert date.fromisoformat(row["end"]) <= filed


def test_two_runs_on_the_same_inputs_agree_exactly():
    """Nothing here samples or fits, so there is nothing to seed and no drift."""
    def run() -> KPISet:
        return build_kpis(
            "DDOG",
            KpiClient(instance="DDOG", facts="DDOG"),
            software(),
            as_of=AS_OF,
            mdna_text=MDNA,
        )

    first, second = run(), run()
    assert first.rows() == second.rows()
    assert first.flags == second.flags
    assert first.missing == second.missing
    assert first.as_dict() == second.as_dict()
