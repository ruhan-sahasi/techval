"""Normalization: tag resolution, period algebra, splits, controls."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from techval.edgar import (
    CompanyFacts,
    Fact,
    cover_window,
    derive_periods,
    trailing_twelve_months,
)
from techval.errors import MissingDataError, StaleDataError
from techval.financials import build_financials

from conftest import FIXTURES, load_facts


def _fact(start: str, end: str, val: float, tag="X", form="10-Q", filed="2026-01-01"):
    return Fact(
        tag=tag,
        val=val,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        form=form,
        filed=date.fromisoformat(filed),
        unit="USD",
    )


# --------------------------------------------------------------------------- #
# period algebra
# --------------------------------------------------------------------------- #


def test_derive_tail_period_from_shared_start():
    """Nine months less six months is the third quarter."""
    facts = [
        _fact("2026-01-01", "2026-06-30", 100.0),
        _fact("2026-01-01", "2026-09-30", 160.0),
    ]
    derived = {(f.start, f.end): f for f in derive_periods(facts)}
    q3 = derived[(date(2026, 7, 1), date(2026, 9, 30))]
    assert q3.val == pytest.approx(60.0)
    assert q3.derived_from is not None


def test_derive_head_period_from_shared_end():
    """A half year less a discrete second quarter is the first quarter.

    CrowdStrike files exactly this shape: a year-to-date half and a discrete Q2,
    with Q1 never reported on its own.
    """
    facts = [
        _fact("2026-02-01", "2026-07-31", 50.1),
        _fact("2026-05-01", "2026-07-31", 25.7),
    ]
    derived = {(f.start, f.end): f for f in derive_periods(facts)}
    q1 = derived[(date(2026, 2, 1), date(2026, 4, 30))]
    assert q1.val == pytest.approx(24.4)


def test_fourth_quarter_is_recovered_from_the_annual_figure():
    """Q4 is never a filed period. It is the year less the first nine months."""
    facts = [
        _fact("2025-01-01", "2025-09-30", 2473.964, form="10-Q"),
        _fact("2025-01-01", "2025-12-31", 3427.158, form="10-K"),
    ]
    derived = {(f.start, f.end): f for f in derive_periods(facts)}
    q4 = derived[(date(2025, 10, 1), date(2025, 12, 31))]
    assert q4.val == pytest.approx(953.194)


def test_cover_window_tiles_without_gap_or_overlap():
    facts = [
        _fact("2025-07-01", "2025-12-31", 1.0),
        _fact("2026-01-01", "2026-06-30", 2.0),
    ]
    tiles = cover_window(facts, date(2025, 7, 1), date(2026, 6, 30))
    assert tiles is not None
    assert len(tiles) == 2
    for a, b in zip(tiles, tiles[1:]):
        assert b.start == a.end + timedelta(days=1)
    assert sum(t.days for t in tiles) == 365


def test_ttm_returns_none_when_the_window_cannot_be_tiled():
    """Annual-only data cannot produce a twelve months ending mid-year.

    Snowflake and CrowdStrike both report operating lease cost this way. The
    right answer is to say so, not to prorate.
    """
    facts = [
        _fact("2024-02-01", "2025-01-31", 59.9, form="10-K"),
        _fact("2025-02-01", "2026-01-31", 66.5, form="10-K"),
    ]
    assert trailing_twelve_months(facts, date(2026, 7, 31)) is None


def test_averages_are_day_weighted_not_summed():
    """Share counts are averages over their window, so they add as integrals.

    Two halves at 100mm and 200mm shares, running 184 and 181 days, average to
    (100*184 + 200*181) / 365 = 149.59mm. Summing them would report 300mm, and
    subtracting one from the other to recover a missing quarter would report
    100mm. Both are nonsense, and both are what a flow-shaped aggregation does to
    a weighted average.

    The weighting is not decoration: the halves are unequal, so a plain mean
    would give 150.0 and be wrong by the drift in share count across the longer
    period.
    """
    facts = [
        _fact("2025-07-01", "2025-12-31", 100.0),
        _fact("2026-01-01", "2026-06-30", 200.0),
    ]
    value, tiles, method = trailing_twelve_months(
        facts, date(2026, 6, 30), kind="average"
    )
    assert value == pytest.approx((100.0 * 184 + 200.0 * 181) / 365)
    assert value == pytest.approx(149.589, abs=1e-3)
    assert value != pytest.approx(150.0, abs=1e-3)
    assert len(tiles) == 2


# --------------------------------------------------------------------------- #
# tag resolution
# --------------------------------------------------------------------------- #


def test_revenue_ladder_falls_through_to_the_including_assessed_tax_variant(crwd):
    """CrowdStrike tags revenue under the second entry, not the first."""
    tag = crwd.provenance["revenue"].tag
    assert tag == "RevenueFromContractWithCustomerIncludingAssessedTax"


def test_stale_tag_is_skipped_for_a_current_one(zs):
    """Zscaler's marketable securities are not under ShortTermInvestments.

    ``AvailableForSaleSecuritiesDebtSecuritiesCurrent`` exists in its fact set
    but stops in 2022. Resolving by first-tag-present would return that stale
    zero and overstate enterprise value by the whole balance.
    """
    assert zs.short_term_investments == pytest.approx(2545.8, rel=1e-3)
    assert zs.provenance["short-term investments"].tag == (
        "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent"
    )


def test_absent_concept_defaults_to_zero_and_says_so(ddog):
    """Datadog has no non-controlling interest, and zero is the right reading."""
    assert ddog.nci == 0.0
    assert ddog.provenance["non-controlling interest"].method == "absent, defaulted"


def test_non_controlling_interest_is_picked_up_where_it_exists(crwd):
    assert crwd.nci == pytest.approx(37.8, rel=1e-2)


def test_missing_required_concept_raises_naming_the_tags():
    facts = load_facts("DDOG")
    with pytest.raises(MissingDataError) as exc:
        facts.resolve_ttm("invented", ["NotARealTag", "AlsoNotReal"], date(2026, 6, 30))
    assert "NotARealTag" in str(exc.value)
    assert "invented" in str(exc.value)


# --------------------------------------------------------------------------- #
# fiscal year boundaries and splits
# --------------------------------------------------------------------------- #


def test_ttm_crosses_a_january_fiscal_year_end(mdb):
    """MongoDB's year ends 31 January, so its TTM spans two fiscal years."""
    assert mdb.as_of == date(2026, 7, 31)
    assert mdb.revenue == pytest.approx(2782.8, rel=1e-3)
    periods = mdb.provenance["revenue"].periods
    assert any(p.startswith("2025-") for p in periods)
    assert any(p.startswith("2026-") for p in periods)


def test_stock_split_is_detected_and_restated(crwd):
    """CrowdStrike split four-for-one in mid-2026.

    Periods filed before the split keep pre-split share counts in companyfacts
    forever. Mixing them across the TTM window gives roughly 271mm shares
    against a true post-split 1,023mm.
    """
    facts = load_facts("CRWD")
    splits = facts._split_factors()
    assert len(splits) == 1
    when, factor = splits[0]
    assert factor == pytest.approx(4.0)
    assert when == date(2026, 8, 27)
    assert crwd.diluted_shares == pytest.approx(1022.8, rel=1e-3)


def test_no_split_is_invented_where_none_happened(ddog, mdb, zs):
    for fin in (ddog, mdb, zs):
        assert not any("split" in w for w in fin.warnings)


def test_split_warning_reaches_the_user(crwd):
    assert any("4-for-1 share split" in w for w in crwd.warnings)


# --------------------------------------------------------------------------- #
# derived measures and controls
# --------------------------------------------------------------------------- #


def test_ebitda_is_ebit_plus_da(ddog):
    assert ddog.ebitda == pytest.approx(ddog.ebit + ddog.da)
    assert ddog.ebitda == pytest.approx(78.3, rel=1e-2)


def test_ebitda_is_none_when_da_cannot_be_built(crwd):
    """CrowdStrike publishes no combined D&A covering the TTM window.

    EBITDA is then unavailable, and saying so is the only honest option. The
    comps table flags the peer rather than dropping it.
    """
    assert crwd.da is None
    assert crwd.ebitda is None


def test_ebitdar_adds_back_operating_lease_cost(ddog):
    assert ddog.ebitdar == pytest.approx(ddog.ebitda + ddog.operating_lease_cost)


def test_effective_tax_rate_is_withheld_when_not_meaningful(zs):
    """A pre-tax loss produces a rate that describes valuation allowances."""
    assert zs.pretax_income < 0
    assert zs.effective_tax_rate is None


def test_working_capital_excludes_cash_and_investments(ddog):
    nwc = ddog.net_working_capital
    assert nwc == pytest.approx(
        ddog.current_assets
        - ddog.cash
        - ddog.short_term_investments
        - ddog.current_liabilities
    )


def test_eps_tie_out_control_is_silent_on_clean_data(ddog, mdb, zs):
    for fin in (ddog, mdb, zs):
        assert not any("EPS tie-out" in w for w in fin.warnings)


def test_provenance_records_every_reported_line(ddog):
    rows = {r["concept"]: r for r in ddog.provenance_rows()}
    assert rows["revenue"]["tag"] == (
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    )
    assert rows["revenue"]["form"]
    assert rows["revenue"]["filed"]
    assert "diluted shares" in rows


def test_build_is_deterministic():
    a = build_financials("DDOG", facts=load_facts("DDOG"))
    b = build_financials("DDOG", facts=load_facts("DDOG"))
    assert a.revenue == b.revenue
    assert a.diluted_shares == b.diluted_shares
    assert a.ebit == b.ebit


def test_audited_report_beats_a_later_proxy():
    """A proxy summarises the financials; it does not restate them.

    CrowdStrike's fiscal 2024 net income is 72.2mm in the 10-K that restated it
    and 73.4mm in the proxy filed six weeks later. Taking the latest filing
    regardless of form would swap the audited figure for a summary of it.
    """
    facts = load_facts("CRWD")
    annual = {
        (f.start, f.end): f
        for f in facts.facts("NetIncomeLoss")
        if f.start and 350 <= f.days <= 380
    }
    fy2024 = annual[(date(2023, 2, 1), date(2024, 1, 31))]
    assert fy2024.form == "10-K"
    assert fy2024.val / 1e6 == pytest.approx(72.2, rel=1e-2)

    # Every annual net income must come from a periodic report, never a proxy.
    assert all(f.form.startswith(("10-K", "10-Q")) for f in annual.values())


def test_latest_periodic_report_still_wins_within_its_class():
    """Genuine restatements between two 10-Ks must be picked up.

    The same CrowdStrike year reads 89.3mm in the 10-K filed in 2024 and 72.2mm
    in the one filed in 2026. Form class decides first, filing date second.
    """
    facts = load_facts("CRWD")
    fy2024 = next(
        f
        for f in facts.facts("NetIncomeLoss")
        if f.start == date(2023, 2, 1) and f.end == date(2024, 1, 31)
    )
    assert fy2024.filed == date(2026, 3, 5)
    assert fy2024.val / 1e6 == pytest.approx(72.2, rel=1e-2)


# --------------------------------------------------------------------------- #
# point-in-time knowledge
# --------------------------------------------------------------------------- #


def test_knowledge_date_hides_facts_filed_later():
    """A valuation dated in the past may only see what was filed by then.

    Without this every historical run quietly reads restatements, later
    comparatives and split adjustments that nobody could have seen at the time,
    and a backtest built on it measures hindsight rather than the model.

    Datadog's June 2026 quarter was filed on 2026-08-06. As of 1 March 2026 the
    newest revenue period knowable is the December 2025 year end, from the 10-K
    filed on 2026-02-18.
    """
    import json

    from techval.edgar import CompanyFacts

    from pathlib import Path

    raw = json.loads(
        (Path(__file__).parent / "fixtures" / "companyfacts_DDOG.json").read_text()
    )
    tag = "RevenueFromContractWithCustomerExcludingAssessedTax"

    full = CompanyFacts(raw, "DDOG")
    past = CompanyFacts(raw, "DDOG", knowledge_date=date(2026, 3, 1))

    latest = lambda c: max(f.end for f in c.facts(tag) if not f.is_instant)
    assert latest(full) == date(2026, 6, 30)
    assert latest(past) == date(2025, 12, 31)
    assert all(f.filed <= date(2026, 3, 1) for f in past.facts(tag))


def test_knowledge_date_falls_back_to_the_filing_of_record(crwd):
    """Hiding a later restatement must expose the earlier figure, not a gap.

    CrowdStrike's fiscal 2024 net income reads 89.3mm in the 10-K filed in 2024
    and 72.2mm in the one filed in 2026. An analyst working in 2025 saw the
    first. Filtering only after deduplication would have let the 2026 filing win
    selection and then be dropped, losing the period altogether.
    """
    import json

    from techval.edgar import CompanyFacts

    raw = json.loads(
        (__import__("pathlib").Path(__file__).parent / "fixtures"
         / "companyfacts_CRWD.json").read_text()
    )
    period = (date(2023, 2, 1), date(2024, 1, 31))

    then = CompanyFacts(raw, "CRWD", knowledge_date=date(2025, 6, 30))
    fact = next(
        f for f in then.facts("NetIncomeLoss")
        if (f.start, f.end) == period
    )
    assert fact.val / 1e6 == pytest.approx(89.3, rel=1e-2)
    assert fact.filed <= date(2025, 6, 30)


def test_a_point_in_time_statement_is_internally_consistent():
    """The whole statement moves back together, not just one line."""
    import json

    from techval.edgar import CompanyFacts

    raw = json.loads(
        (__import__("pathlib").Path(__file__).parent / "fixtures"
         / "companyfacts_DDOG.json").read_text()
    )
    fin = build_financials(
        "DDOG", facts=CompanyFacts(raw, "DDOG", knowledge_date=date(2026, 3, 1))
    )
    assert fin.as_of == date(2025, 12, 31)
    assert fin.revenue == pytest.approx(3427.2, rel=1e-3)   # FY2025 as filed
    for p in fin.provenance.values():
        if p.filed and p.filed != "-":
            assert p.filed <= "2026-03-01"


# --------------------------------------------------------------------------- #
# dimensioned facts from the filing instance
# --------------------------------------------------------------------------- #


_INSTANCE = b"""<?xml version="1.0" encoding="UTF-8"?>
<xbrl xmlns="http://www.xbrl.org/2003/instance"
      xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
      xmlns:us-gaap="http://fasb.org/us-gaap/2024"
      xmlns:dei="http://xbrl.sec.gov/dei/2024">
  <context id="cA">
    <entity><identifier scheme="x">1</identifier>
      <segment><xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">us-gaap:CommonClassAMember</xbrldi:explicitMember></segment>
    </entity>
    <period><instant>2026-06-30</instant></period>
  </context>
  <context id="cB">
    <entity><identifier scheme="x">1</identifier>
      <segment><xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">us-gaap:CommonClassBMember</xbrldi:explicitMember></segment>
    </entity>
    <period><instant>2026-06-30</instant></period>
  </context>
  <context id="cPlain">
    <entity><identifier scheme="x">1</identifier></entity>
    <period><startDate>2026-01-01</startDate><endDate>2026-06-30</endDate></period>
  </context>
  <unit id="shares"><measure>xbrli:shares</measure></unit>
  <unit id="usdPerShare"><measure>iso4217:USD</measure><measure>xbrli:shares</measure></unit>
  <dei:EntityCommonStockSharesOutstanding contextRef="cA" unitRef="shares">334904614</dei:EntityCommonStockSharesOutstanding>
  <dei:EntityCommonStockSharesOutstanding contextRef="cB" unitRef="shares">24170410</dei:EntityCommonStockSharesOutstanding>
  <us-gaap:ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingNumber contextRef="cPlain" unitRef="shares">3474619</us-gaap:ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingNumber>
  <us-gaap:NotANumber contextRef="cPlain">n/a</us-gaap:NotANumber>
</xbrl>
"""


def test_instance_parsing_recovers_dimensioned_facts():
    """companyfacts publishes only undimensioned facts, so this is the way in.

    A dual-class issuer reports shares outstanding once per class, and both
    vanish from companyfacts. Summing them is the only route to a point-in-time
    share count for Datadog, Alphabet or any other multi-class filer.
    """
    from techval.edgar import parse_instance

    facts = parse_instance(_INSTANCE)
    shares = [f for f in facts if f.tag == "EntityCommonStockSharesOutstanding"]

    assert len(shares) == 2
    assert sum(f.value for f in shares) == pytest.approx(359_075_024)
    assert {f.axis("StatementClassOfStockAxis").split(":")[-1] for f in shares} == {
        "CommonClassAMember",
        "CommonClassBMember",
    }
    assert all(f.is_instant and f.end == date(2026, 6, 30) for f in shares)


def test_instance_parsing_keeps_duration_and_skips_non_numeric():
    """Durations survive with both endpoints; unparseable values are dropped.

    Coercing a non-numeric fact to zero would put a fabricated figure into a
    share count, which is the one thing this engine does not do.
    """
    from techval.edgar import parse_instance

    facts = parse_instance(_INSTANCE)
    options = next(f for f in facts if "OptionsOutstandingNumber" in f.tag)

    assert options.value == pytest.approx(3_474_619)
    assert options.start == date(2026, 1, 1) and options.end == date(2026, 6, 30)
    assert not options.is_instant
    assert options.unit == "shares"
    assert not any(f.tag == "NotANumber" for f in facts)


# A minimal fact set for testing the balance-sheet ladders in isolation. Four
# discrete quarters tile the trailing twelve months exactly, so the income
# statement resolves without argument and the assertion is only ever about the
# balance-sheet concept the test passed in.
_QUARTERS = (
    ("2025-07-01", "2025-09-30"),
    ("2025-10-01", "2025-12-31"),
    ("2026-01-01", "2026-03-31"),
    ("2026-04-01", "2026-06-30"),
)


def _facts_with_debt(balances: dict[str, list[tuple[str, float]]]) -> CompanyFacts:
    def flow(value: float, unit: str = "USD") -> dict:
        return {
            "units": {
                unit: [
                    {
                        "start": start,
                        "end": end,
                        "val": value,
                        "form": "10-Q",
                        "filed": "2026-07-31",
                        "fy": 2026,
                        "fp": "Q2",
                    }
                    for start, end in _QUARTERS
                ]
            }
        }

    units: dict[str, dict] = {
        "Revenues": flow(2_500e6),
        "OperatingIncomeLoss": flow(250e6),
        "NetIncomeLoss": flow(200e6),
        "WeightedAverageNumberOfDilutedSharesOutstanding": flow(1_000e6, "shares"),
        "CashAndCashEquivalentsAtCarryingValue": {
            "units": {
                "USD": [
                    {
                        "end": "2026-06-30",
                        "val": 1_000e6,
                        "form": "10-Q",
                        "filed": "2026-07-31",
                    }
                ]
            }
        },
    }
    for concept, rows in balances.items():
        units[concept] = {
            "units": {
                "USD": [
                    {"end": end, "val": val, "form": "10-Q", "filed": "2026-07-31"}
                    for end, val in rows
                ]
            }
        }
    return CompanyFacts(
        {"cik": 1, "entityName": "Test Filer", "facts": {"us-gaap": units}}, "TEST"
    )


# --------------------------------------------------------------------------- #
# the debt ladder
#
# The worst failure this engine is capable of, because it is the only one that
# prints a wrong number instead of refusing. Verizon's Q2 2026 balance sheet
# (accession 0000732712-26-000046, R5.htm) reads:
#
#     Debt maturing within one year        21,783     LongTermDebtCurrent
#     Long-term debt                      143,448     LongTermDebtAndCapitalLeaseObligations
#     Cash and cash equivalents             1,752
#
# The old ladder held LongTermDebtNoncurrent, LongTermNotesPayable,
# SecuredDebtNoncurrent and NotesPayableNoncurrent for the long-term line.
# Verizon stopped using the first in 2013 and has never used the other three, so
# the ladder resolved NOTHING, the bridge printed 21,783mm of debt, and
# enterprise value came out at 231,813mm against 375,261mm.
# --------------------------------------------------------------------------- #


def test_long_term_debt_is_read_from_the_debt_and_leases_concept(vz):
    """The fix, against the face of the filing.

    143,448 plus 21,783 is 165,231, which is the sum of the two debt lines on
    Verizon's balance sheet to the dollar.
    """
    assert vz.as_of == date(2026, 6, 30)
    assert vz.straight_debt == pytest.approx(165_231.0)
    assert "LongTermDebtAndCapitalLeaseObligations" in vz.debt_basis
    assert "LongTermDebtCurrent" in vz.debt_basis


def test_the_thirteen_year_old_tag_is_not_what_resolved(vz):
    """The stale value is present in the fixture and is not the answer.

    LongTermDebtNoncurrent sits first in the ladder and carries 89,658mm dated
    2013-12-31. Resolving it would have been worse than resolving nothing, and
    the point of keeping it in the fixture is that the ladder has to walk past it
    every time.
    """
    stale = [f for f in load_facts("VZ").facts("LongTermDebtNoncurrent") if f.is_instant]
    assert stale[-1].end == date(2013, 12, 31)
    assert stale[-1].val == pytest.approx(89_658e6)
    assert vz.provenance["long-term debt"].tag == "LongTermDebtAndCapitalLeaseObligations"


def test_the_bridge_says_what_is_inside_the_debt_figure(vz):
    """A number nobody can trace is the thing this engine refuses to print.

    Verizon's debt concept bundles finance leases with borrowings, so the figure
    is not comparable with a filer who reports the two separately unless the
    difference is stated.
    """
    assert "finance leases inside it" in vz.debt_basis


def test_a_finance_lease_inside_the_debt_concept_is_not_added_twice():
    """The double count the old ladder could not have hit, because it read zero.

    Verizon at 2025-12-31 reports LongTermDebtAndCapitalLeaseObligations
    139,532mm, DebtCurrent 18,618mm, FinanceLeaseLiabilityNoncurrent 1,568mm and
    FinanceLeaseLiabilityCurrent 943mm. Its own fair-value note gives debt
    EXCLUDING capital leases as 155,639mm, and 139,532 + 18,618 - 155,639 is
    2,511, exactly the two lease figures. They are the same obligation twice, so
    the bridge takes the debt line and suppresses the lease line, and the total
    lands on the 158,150mm Verizon tags as its own combined debt.
    """
    facts = CompanyFacts(
        json.loads((FIXTURES / "companyfacts_VZ.json").read_text()),
        "VZ",
        knowledge_date=date(2026, 3, 1),
    )
    fin = build_financials("VZ", facts=facts)

    assert fin.as_of == date(2025, 12, 31)
    assert fin.straight_debt == pytest.approx(158_150.0)
    assert fin.finance_lease_inside_debt == pytest.approx(2_511.0)
    assert fin.finance_lease_liability == 0.0
    assert any("inside the debt figure" in w for w in fin.warnings)


def test_a_separately_presented_finance_lease_is_still_added(mdb):
    """The other camp, and the reason the suppression is not unconditional.

    MongoDB reports finance leases under their own concepts and its debt under
    none, so there is nothing to suppress and the lease stands on its own line.
    Oracle is the same shape at scale: DebtCurrent for borrowings and the finance
    lease in other liabilities, with the lease note's extensible enumeration
    saying so.
    """
    assert mdb.finance_lease_liability > 0
    assert mdb.finance_lease_inside_debt == 0.0


def test_a_combined_tag_is_never_added_to_its_own_components():
    """Three concepts describing the same balance, summed, is twice the debt.

    Verizon at 2025-12-31 publishes the non-current line, the current line and
    LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities at
    157,709mm, which is the first two together. The resolution takes the split
    and never reaches the combined ladder.
    """
    facts = CompanyFacts(
        json.loads((FIXTURES / "companyfacts_VZ.json").read_text()),
        "VZ",
        knowledge_date=date(2026, 3, 1),
    )
    combined = [
        f
        for f in facts.facts("LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities")
        if f.is_instant and f.end == date(2025, 12, 31)
    ]
    assert combined and combined[-1].val == pytest.approx(157_709e6)

    fin = build_financials("VZ", facts=facts)
    assert "total debt" not in fin.provenance, "the combined ladder was consulted"
    assert fin.straight_debt == pytest.approx(158_150.0)


def test_the_current_leg_alone_no_longer_passes_for_a_debt_figure():
    """The exact mechanism of the Verizon bug, on a synthetic filer.

    A long-term line the ladder cannot reach and a current line it can used to
    return the current line, silently. Now the combined figure is preferred
    where it is larger, and it replaces the split rather than joining it.
    """
    facts = _facts_with_debt(
        {
            "LongTermDebtCurrent": [("2026-06-30", 2_000e6)],
            "DebtLongtermAndShorttermCombinedAmount": [("2026-06-30", 30_000e6)],
        }
    )
    fin = build_financials("TEST", facts=facts)
    assert fin.straight_debt == pytest.approx(30_000.0)
    assert fin.debt_basis == "DebtLongtermAndShorttermCombinedAmount"


def test_a_total_by_definition_concept_is_not_added_to_a_current_leg():
    """Western Digital's shape, where the obvious reading doubles the debt.

    LongTermDebtNoncurrent is an explicit zero, LongTermDebtCurrent is 1,052mm
    and LongTermDebt is 1,052mm. Reading LongTermDebt as a non-current line and
    adding the current one reports 2,104mm against 1,052mm on the balance sheet.
    The explicit zero is the filer saying it has no non-current debt, and that is
    what settles it.
    """
    facts = _facts_with_debt(
        {
            "LongTermDebtNoncurrent": [("2026-06-30", 0.0)],
            "LongTermDebtCurrent": [("2026-06-30", 1_052e6)],
            "LongTermDebt": [("2026-06-30", 1_052e6)],
        }
    )
    fin = build_financials("TEST", facts=facts)
    assert fin.straight_debt == pytest.approx(1_052.0)


def test_a_non_current_line_tagged_long_term_debt_still_takes_its_current_leg():
    """Qualcomm's shape, which is the opposite reading of the same concept.

    Its balance sheet tags the non-current line LongTermDebt at 12,781mm with
    2,489mm of short-term debt above it, and there is no explicit non-current
    zero, so the two are different lines and add.
    """
    facts = _facts_with_debt(
        {
            "LongTermDebt": [("2026-06-30", 12_781e6)],
            "DebtCurrent": [("2026-06-30", 2_489e6)],
        }
    )
    fin = build_financials("TEST", facts=facts)
    assert fin.straight_debt == pytest.approx(15_270.0)


def test_a_finance_lease_reported_only_as_a_total_is_not_read_as_zero():
    """Microsoft's 66,594mm, which the split-only lease ladder could not see.

    Bigger than Microsoft's own 40,294mm of straight debt, and the largest single
    balance the old ladders missed anywhere in the seed universe.
    """
    facts = _facts_with_debt(
        {
            "LongTermDebtNoncurrent": [("2026-06-30", 31_067e6)],
            "LongTermDebtCurrent": [("2026-06-30", 9_227e6)],
            "FinanceLeaseLiability": [("2026-06-30", 66_594e6)],
        }
    )
    fin = build_financials("TEST", facts=facts)
    assert fin.straight_debt == pytest.approx(40_294.0)
    assert fin.finance_lease_liability == pytest.approx(66_594.0)


# --------------------------------------------------------------------------- #
# the debt control
# --------------------------------------------------------------------------- #


def test_debt_outside_every_ladder_refuses_rather_than_reading_zero():
    """Digital Realty's shape: a real debt stack no single tag adds up.

    16,014mm of SeniorNotes and 842mm of SecuredDebt, with no total concept of
    any kind. Summing instrument classes assumes they do not overlap, which
    nothing in the fact set establishes, so the engine refuses and names what it
    can see. An enterprise value of equity less cash would have been the
    alternative.
    """
    facts = _facts_with_debt({"SeniorNotes": [("2026-06-30", 16_014e6)]})
    with pytest.raises(MissingDataError) as exc:
        build_financials("TEST", facts=facts)
    assert "SeniorNotes" in str(exc.value)
    assert "16,014" in str(exc.value)


def test_a_debt_balance_that_vanishes_between_filings_refuses():
    """T-Mobile's shape, and the reason the refusal is StaleDataError.

    Its Q2 2026 balance sheet carries 78,504mm of long-term debt tagged
    LongTermDebtNoncurrent under a related-party axis, which the companyfacts
    API does not return. The newest undimensioned value is the 81,147mm of the
    December 10-K, so the ladder sees only 6,117mm of short-term borrowings at
    the balance-sheet date. Read as zero it looks like a carrier that repaid
    eighty billion dollars in a quarter, which is possible, which is exactly why
    the engine will not decide it on its own.
    """
    facts = _facts_with_debt(
        {
            "ShortTermBorrowings": [("2026-06-30", 6_117e6)],
            "LongTermDebtNoncurrent": [("2025-12-31", 81_147e6)],
        }
    )
    with pytest.raises(StaleDataError) as exc:
        build_financials("TEST", facts=facts)
    assert "81,147" in str(exc.value)
    assert "2025-12-31" in str(exc.value)


def test_a_debt_balance_repaid_years_ago_does_not_refuse():
    """The other half of the same test, and the one that keeps it usable.

    MongoDB's LongTermDebt stops at 216.9mm in January 2019, the New York
    Times's DebtAndCapitalLeaseObligations at 431.2mm in December 2015 and
    Fortinet's at 987.5mm in June 2021. All three repaid. A control that refused
    on those would refuse on a third of the universe, so it looks back one
    reporting cycle and no further.
    """
    facts = _facts_with_debt({"LongTermDebt": [("2019-01-31", 216.9e6)]})
    fin = build_financials("TEST", facts=facts)
    assert fin.straight_debt == 0.0


def test_a_filer_that_repaid_is_believed_when_it_publishes_a_current_total():
    """Micron's shape: a stale figure above, and the filer's own total below it.

    LongTermDebt 8,844mm at 2025-11-27 against 5,722mm of
    DebtAndCapitalLeaseObligations at 2026-05-28, which the resolution matches to
    the dollar. That is three billion dollars repaid, and the filer's own current
    total is what says so.
    """
    facts = _facts_with_debt(
        {
            "LongTermDebtAndCapitalLeaseObligations": [("2026-06-30", 5_140e6)],
            "DebtCurrent": [("2026-06-30", 582e6)],
            "DebtAndCapitalLeaseObligations": [("2026-06-30", 5_722e6)],
            "LongTermDebt": [("2025-11-27", 8_844e6)],
        }
    )
    fin = build_financials("TEST", facts=facts)
    assert fin.straight_debt == pytest.approx(5_722.0)


def test_an_ambiguous_current_concept_follows_the_filers_own_tagging():
    """Micron again, on the question the taxonomy does not settle.

    DebtCurrent is defined as "debt AND LEASE obligation, classified as current",
    and Micron's 582mm of it is finance lease to the last dollar while Oracle's
    7,199mm of it excludes its lease entirely. The tie-break is the OTHER leg: a
    filer whose non-current line is tagged "debt and capital lease obligations"
    has shown that it bundles the two.
    """
    bundled = _facts_with_debt(
        {
            "LongTermDebtAndCapitalLeaseObligations": [("2026-06-30", 5_140e6)],
            "DebtCurrent": [("2026-06-30", 582e6)],
            "DebtAndCapitalLeaseObligations": [("2026-06-30", 5_722e6)],
            "FinanceLeaseLiabilityCurrent": [("2026-06-30", 582e6)],
            "FinanceLeaseLiabilityNoncurrent": [("2026-06-30", 2_088e6)],
        }
    )
    fin = build_financials("TEST", facts=bundled)
    assert fin.straight_debt == pytest.approx(5_722.0)
    assert fin.finance_lease_liability == 0.0
    assert fin.finance_lease_inside_debt == pytest.approx(2_670.0)

    separate = _facts_with_debt(
        {
            "LongTermDebtNoncurrent": [("2026-06-30", 122_342e6)],
            "DebtCurrent": [("2026-06-30", 7_199e6)],
            "FinanceLeaseLiabilityCurrent": [("2026-06-30", 620e6)],
            "FinanceLeaseLiabilityNoncurrent": [("2026-06-30", 7_081e6)],
        }
    )
    fin = build_financials("TEST", facts=separate)
    assert fin.straight_debt == pytest.approx(129_541.0)
    assert fin.finance_lease_liability == pytest.approx(7_701.0)
    assert fin.finance_lease_inside_debt == 0.0


def test_a_small_gap_against_a_reported_total_is_not_worth_a_flag():
    """Broadcom's 61,079mm of principal against 59,419mm of carrying value.

    The concepts being compared are struck on different bases, so a control that
    printed a flag for a 2.8% difference would teach a reader to skip the block
    on the company where it means something. That is the failure mode the whole
    control is written against.
    """
    facts = _facts_with_debt(
        {
            "LongTermDebtNoncurrent": [("2026-06-30", 57_167e6)],
            "DebtCurrent": [("2026-06-30", 2_252e6)],
            "DebtLongtermAndShorttermCombinedAmount": [("2026-06-30", 61_079e6)],
        }
    )
    fin = build_financials("TEST", facts=facts)
    assert fin.straight_debt == pytest.approx(59_419.0)
    assert not [w for w in fin.warnings if "debt resolved" in w]


def test_a_material_gap_against_a_reported_total_is_flagged_and_not_refused():
    """PayPal at 2026-03-31, before the combined ladder reached it.

    LongTermDebtNoncurrent 9,409mm against LongTermDebt 10,876mm, with the
    1,467mm current portion tagged under a dimension companyfacts does not
    return. A gap of that size against a figure the engine cannot fully explain
    is worth a line of print and is not worth refusing a valuation over.
    """
    facts = _facts_with_debt(
        {
            "LongTermDebtNoncurrent": [("2026-06-30", 9_409e6)],
            "SeniorNotes": [("2026-06-30", 10_876e6)],
        }
    )
    fin = build_financials("TEST", facts=facts)
    assert fin.straight_debt == pytest.approx(9_409.0)
    assert [w for w in fin.warnings if "SeniorNotes" in w and "10,876" in w]


def test_a_genuinely_debt_free_filer_still_resolves_zero(ddog):
    """The control has to stay quiet on the companies it does not apply to.

    Datadog has no straight debt and no finance leases, and reads zero for both
    because zero is the truth. A control that refused here would have taken the
    engine's most-run valuation offline to fix a telecom.
    """
    assert ddog.straight_debt == 0.0
    assert ddog.convertible_debt > 0
    assert not [w for w in ddog.warnings if "refused" in w]
