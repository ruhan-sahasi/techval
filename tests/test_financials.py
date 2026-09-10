"""Normalization: tag resolution, period algebra, splits, controls."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from techval.edgar import Fact, cover_window, derive_periods, trailing_twelve_months
from techval.errors import MissingDataError
from techval.financials import build_financials

from conftest import load_facts


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
