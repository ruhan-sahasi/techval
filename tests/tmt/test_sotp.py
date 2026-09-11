"""Sum of the parts: the arithmetic, the corporate line, the discount, the gap.

The segments here are synthetic and the numbers are round, so every figure the
module produces can be checked on paper rather than against itself. That is
deliberate twice over: the segment builder lives in another module which this
one does not import, so the tests exercise the documented input protocol rather
than one particular producer of it, and a valuation test that can only be
verified by rerunning the valuation proves nothing.

One test runs the real DDOG fixture through a real priced bridge, to prove the
walk from enterprise value to equity is the engine's own and not a private one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from techval.errors import ConfigError, MissingDataError, NotMeaningfulError
from techval.ev_bridge import EVBridge, equity_value_from_ev
from techval.financials import Financials
from techval.tmt.sotp import run_sotp, value_segments

# --------------------------------------------------------------------------- #
# a synthetic conglomerate. USD millions, shares millions, dollars per share.
# --------------------------------------------------------------------------- #

CABLE = {"name": "Cable", "revenue": 6_000.0, "operating_income": 1_500.0, "da": 900.0}
STUDIOS = {
    "name": "Studios",
    "revenue": 4_000.0,
    "operating_income": 400.0,
    "da": 100.0,
}

MULTIPLES = {
    "Cable": ("ebitda", 7.0, "peer median EV/EBITDA for US cable, Charter and Altice"),
    "Studios": ("ebitda", 12.0, "peer median EV/EBITDA for the listed studios"),
}

CORP = {
    "corporate_cost": 300.0,
    "corporate_multiple": 8.0,
    "corporate_multiple_source": "blended segment earnings multiple, weighted by value",
}


def _fin(revenue: float = 10_000.0) -> Financials:
    """Consolidated revenue is all the sum of the parts reads off the filer."""
    return Financials(
        ticker="TMTCO",
        entity_name="TMT Holdings Inc",
        cik=1,
        as_of=date(2026, 9, 10),
        revenue=revenue,
        gross_profit=None,
        ebit=1_600.0,
        da=1_000.0,
        sbc=None,
        net_income=900.0,
        pretax_income=None,
        tax_expense=None,
        interest_expense=None,
        operating_lease_cost=None,
        capex=None,
        cfo=None,
        diluted_shares=1_000.0,
        basic_shares=None,
        cash=1_000.0,
        short_term_investments=0.0,
        straight_debt=5_000.0,
        convertible_debt=0.0,
        operating_lease_liability=0.0,
        finance_lease_liability=0.0,
        nci=0.0,
        preferred=0.0,
        current_assets=None,
        current_liabilities=None,
        deferred_revenue=0.0,
    )


def _bridge() -> EVBridge:
    """Priced at $12.00 on 1,000mm shares: equity 12,000, enterprise value 16,000."""
    return EVBridge(
        ticker="TMTCO",
        price=12.0,
        diluted_shares=1_000.0,
        equity_value=12_000.0,
        straight_debt=5_000.0,
        convertible_debt=0.0,
        convertible_in_debt=0.0,
        finance_lease=0.0,
        operating_lease=0.0,
        operating_lease_in_debt=0.0,
        preferred=0.0,
        nci=0.0,
        cash=1_000.0,
        short_term_investments=0.0,
        enterprise_value=16_000.0,
        ev_excluding_leases=16_000.0,
        ev_including_leases=16_000.0,
        net_debt=4_000.0,
        lease_convention="operating leases excluded from debt",
        convertible_treatment="none outstanding",
    )


def _run(assumptions, segments=None, multiples=None, **kwargs):
    return run_sotp(
        _fin(),
        _bridge(),
        segments if segments is not None else [CABLE, STUDIOS],
        multiples if multiples is not None else MULTIPLES,
        assumptions,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #


def test_two_segment_case_worked_by_hand(assumptions):
    """The whole bridge, on paper.

        Cable     EBITDA 1,500 + 900 = 2,400 at  7.0x     16,800
        Studios   EBITDA   400 + 100 =   500 at 12.0x      6,000
        Gross value of the parts                          22,800
        Corporate cost of 300 a year at 8.0x              -2,400
        Subtotal                                          20,400
        Conglomerate discount, 10% of 20,400              -2,040
        Enterprise value, sum of the parts                18,360
        less straight debt 5,000, plus cash 1,000
        Equity value                                      14,360
        over 1,000mm shares                               $14.36

    The market pays $12.00 for those shares, so the consolidated enterprise
    value is 12,000 + 5,000 - 1,000 = 16,000. The break-up gap is
    18,360 - 16,000 = 2,360, which is 14.75% of the market enterprise value and
    $2.36 a share. That gap is the entire point of the exercise.
    """
    r = _run(assumptions, conglomerate_discount=0.10, **CORP)

    cable, studios = r.segments
    assert cable.metric_value == pytest.approx(2_400.0)
    assert cable.enterprise_value == pytest.approx(16_800.0)
    assert studios.metric_value == pytest.approx(500.0)
    assert studios.enterprise_value == pytest.approx(6_000.0)

    assert r.gross_enterprise_value == pytest.approx(22_800.0)
    assert r.unallocated_corporate == pytest.approx(-2_400.0)
    assert r.conglomerate_discount == pytest.approx(-2_040.0)
    assert r.net_enterprise_value == pytest.approx(18_360.0)
    assert r.equity_value == pytest.approx(14_360.0)
    assert r.per_share == pytest.approx(14.36)

    gap = r.implied_vs_consolidated
    assert gap.consolidated_enterprise_value == pytest.approx(16_000.0)
    assert gap.gap == pytest.approx(2_360.0)
    assert gap.gap_pct == pytest.approx(0.1475)
    assert gap.gap_per_share == pytest.approx(2.36)
    assert gap.market_price == pytest.approx(12.0)


def test_the_bridge_adds_down(assumptions):
    """Both deductions are stored negative, so the printed column sums."""
    r = _run(assumptions, conglomerate_discount=0.10, **CORP)
    assert r.unallocated_corporate < 0
    assert r.conglomerate_discount < 0
    assert (
        r.gross_enterprise_value + r.unallocated_corporate + r.conglomerate_discount
    ) == pytest.approx(r.net_enterprise_value)


def test_segment_shares_are_of_the_gross_and_add_to_one(assumptions):
    r = _run(assumptions, **CORP)
    assert sum(s.share_of_total for s in r.segments) == pytest.approx(1.0)
    assert r.segments[0].share_of_total == pytest.approx(16_800.0 / 22_800.0)


def test_the_multiple_source_survives_to_the_output(assumptions):
    """A bare 8.5x is not a valuation view, so the argument travels with it."""
    r = _run(assumptions, **CORP)
    assert "Charter" in r.segments[0].multiple_source
    assert r.corporate_multiple_source == CORP["corporate_multiple_source"]
    frame = r.to_frame()
    assert len(frame) == 2
    assert "Multiple source" in frame.columns


def test_value_segments_stands_alone(assumptions):
    """The segment table is worth printing before anything is netted off it."""
    vals = value_segments([CABLE, STUDIOS], MULTIPLES, assumptions, ticker="TMTCO")
    assert [v.name for v in vals] == ["Cable", "Studios"]
    assert vals[0].multiple_label == "EV/EBITDA"
    assert sum(v.enterprise_value for v in vals) == pytest.approx(22_800.0)


def test_a_segment_without_a_multiple_is_refused(assumptions):
    """Dropping a segment from the sum values that business at zero."""
    with pytest.raises(ConfigError, match="no multiple supplied"):
        _run(assumptions, multiples={"Cable": MULTIPLES["Cable"]}, **CORP)


def test_a_multiple_without_a_source_is_refused(assumptions):
    multiples = dict(MULTIPLES)
    multiples["Studios"] = ("ebitda", 12.0, "  ")
    with pytest.raises(ConfigError, match="no source"):
        _run(assumptions, multiples=multiples, **CORP)


def test_both_input_shapes_are_read_the_same_way(assumptions):
    """The module takes mappings or objects, and does not import the builder."""

    @dataclass
    class Seg:
        name: str
        revenue: float
        operating_income: float
        da: float

    objects = [Seg(**CABLE), Seg(**STUDIOS)]
    assert _run(assumptions, segments=objects, **CORP).net_enterprise_value == (
        pytest.approx(_run(assumptions, **CORP).net_enterprise_value)
    )


def test_ebitda_needs_either_an_ebitda_or_a_da_line(assumptions):
    """ASC 280 gives segment profit; D&A only where the CODM reviews it."""
    bare = {"name": "Cable", "revenue": 6_000.0, "operating_income": 1_500.0}
    with pytest.raises(MissingDataError, match="segment EBITDA"):
        _run(assumptions, segments=[bare, STUDIOS], **CORP)


# --------------------------------------------------------------------------- #
# the holding company
# --------------------------------------------------------------------------- #


def test_corporate_cost_reduces_the_total_by_the_capitalised_amount(assumptions):
    """Segment profit is struck before unallocated expense, so it must come off."""
    without = _run(assumptions)
    with_corp = _run(assumptions, **CORP)

    assert without.gross_enterprise_value == with_corp.gross_enterprise_value
    assert with_corp.unallocated_corporate == pytest.approx(-300.0 * 8.0)
    assert with_corp.net_enterprise_value == pytest.approx(
        without.net_enterprise_value - 2_400.0
    )
    assert with_corp.per_share < without.per_share


def test_omitting_corporate_cost_flags_the_total_as_overstated(assumptions):
    """Silence here is the single most common way a break-up number comes out high."""
    r = _run(assumptions)
    assert r.unallocated_corporate == 0.0
    assert r.corporate_multiple is None
    assert any("overstated" in f for f in r.flags)


def test_reported_corporate_line_must_be_flipped_to_a_positive_cost(assumptions):
    """The footnote reports overhead as negative operating income."""
    with pytest.raises(ConfigError, match="negative"):
        _run(assumptions, corporate_cost=-300.0, corporate_multiple=8.0,
             corporate_multiple_source="the segment footnote")


def test_default_corporate_multiple_is_the_weighted_earnings_average(assumptions):
    """Overhead is an earnings charge, so it is capitalised the way earnings are.

    (7.0 x 16,800 + 12.0 x 6,000) / 22,800 = 8.3158x.
    """
    r = _run(assumptions, corporate_cost=300.0)
    assert r.corporate_multiple == pytest.approx(189_600.0 / 22_800.0)
    assert "weighted average" in r.corporate_multiple_source
    assert r.unallocated_corporate == pytest.approx(-300.0 * 189_600.0 / 22_800.0)


def test_no_earnings_multiple_means_the_caller_is_asked_rather_than_guessed_at(
    assumptions,
):
    """Averaging an EV/Revenue with an EV/EBITDA averages different denominators."""
    multiples = {
        "Cable": ("revenue", 2.8, "peer median EV/Revenue for US cable"),
        "Studios": ("revenue", 1.5, "peer median EV/Revenue for the studios"),
    }
    with pytest.raises(ConfigError, match="no segment was valued on an earnings"):
        _run(assumptions, multiples=multiples, corporate_cost=300.0)


def test_corporate_cost_swallowing_the_parts_is_not_meaningful(assumptions):
    with pytest.raises(NotMeaningfulError, match="nothing to value"):
        _run(assumptions, corporate_cost=4_000.0, corporate_multiple=8.0,
             corporate_multiple_source="the segment footnote")


# --------------------------------------------------------------------------- #
# the conglomerate discount
# --------------------------------------------------------------------------- #


def test_the_discount_is_zero_unless_it_is_chosen(assumptions):
    """A discount assumed rather than observed can rescue any answer."""
    r = _run(assumptions, **CORP)
    assert r.conglomerate_discount_rate == 0.0
    assert r.conglomerate_discount == 0.0
    assert any("No conglomerate discount" in n for n in r.notes)


def test_the_discount_applies_after_the_corporate_deduction(assumptions):
    """Applying it to the gross would charge a discount on value already removed."""
    r = _run(assumptions, conglomerate_discount=0.12, **CORP)
    subtotal = r.gross_enterprise_value + r.unallocated_corporate

    assert r.conglomerate_discount == pytest.approx(-0.12 * subtotal)
    assert r.conglomerate_discount != pytest.approx(-0.12 * r.gross_enterprise_value)
    assert r.net_enterprise_value == pytest.approx(subtotal * 0.88)


def test_a_non_zero_discount_states_the_argument_against_itself(assumptions):
    r = _run(assumptions, conglomerate_discount=0.12, **CORP)
    note = " ".join(r.notes)
    assert "5 to 15 percent" in note
    assert "tuned" in note


def test_a_conglomerate_premium_needs_a_different_argument(assumptions):
    with pytest.raises(ConfigError, match="outside"):
        _run(assumptions, conglomerate_discount=-0.10, **CORP)


# --------------------------------------------------------------------------- #
# segments that do not earn anything
# --------------------------------------------------------------------------- #

LOSING = {
    "name": "Streaming",
    "revenue": 4_000.0,
    "operating_income": -600.0,
    "da": 200.0,
}


def test_an_earnings_multiple_on_a_loss_making_segment_is_refused(assumptions):
    """Negative EBITDA times a positive multiple claims the business is a liability."""
    with pytest.raises(NotMeaningfulError, match="worth nothing rather than"):
        _run(
            assumptions,
            segments=[CABLE, LOSING],
            multiples={
                "Cable": MULTIPLES["Cable"],
                "Streaming": ("ebitda", 10.0, "peer median EV/EBITDA for streaming"),
            },
            **CORP,
        )


def test_a_loss_making_segment_can_be_carried_at_nil_deliberately(assumptions):
    """Nil is a floor and an assumption, and the row says so."""
    r = _run(
        assumptions,
        segments=[CABLE, LOSING],
        multiples={
            "Cable": MULTIPLES["Cable"],
            "Streaming": ("ebitda", 10.0, "peer median EV/EBITDA for streaming"),
        },
        on_negative="nil",
        **CORP,
    )
    streaming = r.segments[1]
    assert streaming.enterprise_value == 0.0
    assert streaming.share_of_total == 0.0
    assert any("Carried at nil" in n for n in streaming.notes)
    assert r.gross_enterprise_value == pytest.approx(16_800.0)
    assert any("carried at nil" in c for c in r.checks)


def test_a_loss_making_segment_valued_on_revenue_says_why(assumptions):
    r = _run(
        assumptions,
        segments=[CABLE, LOSING],
        multiples={
            "Cable": MULTIPLES["Cable"],
            "Streaming": ("revenue", 1.5, "peer median EV/Revenue for streaming"),
        },
        **CORP,
    )
    streaming = r.segments[1]
    assert streaming.enterprise_value == pytest.approx(6_000.0)
    assert any("Valued on revenue because" in n for n in streaming.notes)
    assert any("no operating profit" in c for c in r.checks)
    assert any("valued on EV/Revenue" in c for c in r.checks)


# --------------------------------------------------------------------------- #
# the parts have to be the company
# --------------------------------------------------------------------------- #


def test_segments_that_do_not_cover_the_company_are_refused(assumptions):
    """A missing segment is a business valued at zero without anyone deciding to."""
    short = {**STUDIOS, "revenue": 2_000.0}
    with pytest.raises(MissingDataError, match="80.0% of the company"):
        _run(assumptions, segments=[CABLE, short], **CORP)


def test_segments_adding_to_more_than_the_company_are_refused(assumptions):
    """Usually a dropped eliminations line: intersegment revenue counted twice."""
    over = {**STUDIOS, "revenue": 6_000.0}
    with pytest.raises(MissingDataError, match="reconciling to consolidated"):
        _run(assumptions, segments=[CABLE, over], **CORP)


def test_rounding_in_the_segment_footnote_is_tolerated(assumptions):
    """Two percent absorbs rounding and nothing that changes an answer."""
    rounded = {**STUDIOS, "revenue": 4_100.0}
    r = _run(assumptions, segments=[CABLE, rounded], **CORP)
    assert any("101.0% of the company" in c for c in r.checks)


def test_repeated_segment_names_are_refused(assumptions):
    """Multiples are matched by name, so a duplicate is valued without being asked."""
    twin = {**STUDIOS, "name": "Cable"}
    with pytest.raises(ConfigError, match="repeat"):
        _run(assumptions, segments=[CABLE, twin], **CORP)


def test_a_multiple_for_a_segment_that_was_not_passed_in_is_refused(assumptions):
    multiples = {**MULTIPLES, "Theme Parks": ("ebitda", 11.0, "peer median for parks")}
    with pytest.raises(ConfigError, match="not passed in"):
        _run(assumptions, multiples=multiples, **CORP)


def test_no_segments_at_all_is_refused(assumptions):
    with pytest.raises(MissingDataError, match="segment detail"):
        _run(assumptions, segments=[], **CORP)


# --------------------------------------------------------------------------- #
# the checks printed beside the total
# --------------------------------------------------------------------------- #


def test_the_mix_check_compares_share_of_value_against_share_of_revenue(assumptions):
    """Cable is 73.7% of the value of the parts on 60% of the revenue."""
    r = _run(assumptions, **CORP)
    line = next(c for c in r.checks if "Largest segment" in c)
    assert "73.7%" in line
    assert "60.0%" in line
    assert line.startswith("FLAG:")


def test_multiples_spanning_an_implausible_range_are_flagged(assumptions):
    multiples = {
        "Cable": ("ebitda", 3.0, "distressed cable peers"),
        "Studios": ("ebitda", 20.0, "the streaming comparables"),
    }
    r = _run(assumptions, multiples=multiples, corporate_cost=300.0)
    assert any(c.startswith("FLAG:") and "EV/EBITDA multiples supplied" in c
               for c in r.checks)


def test_a_plausible_range_is_stated_without_a_flag(assumptions):
    r = _run(assumptions, **CORP)
    line = next(c for c in r.checks if "EV/EBITDA multiples supplied" in c)
    assert not line.startswith("FLAG:")


def test_a_wide_gap_is_flagged_as_an_error_before_it_is_a_mispricing(assumptions):
    r = _run(assumptions, **CORP)
    assert r.implied_vs_consolidated.gap_pct == pytest.approx(20_400.0 / 16_000.0 - 1)
    assert any(c.startswith("FLAG:") and "break-up argument" in c for c in r.checks)


def test_lease_conventions_that_do_not_match_are_flagged(assumptions):
    """Segment footnotes do not allocate lease liabilities, so the gap is mixed."""
    bridge = _bridge()
    bridge.operating_lease_in_debt = 800.0
    bridge.enterprise_value = 16_800.0
    assumptions.leases.capitalize_operating_leases = True

    r = run_sotp(_fin(), bridge, [CABLE, STUDIOS], MULTIPLES, assumptions, **CORP)
    assert any("two conventions" in c for c in r.checks)
    assert any("post-rent convention" in n for n in r.segments[0].notes)


# --------------------------------------------------------------------------- #
# the walk to equity is the engine's own
# --------------------------------------------------------------------------- #


def test_equity_uses_the_engines_debt_definition(ddog, ddog_bridge, assumptions):
    """A sum-of-the-parts price and a DCF price have to be comparable.

    DDOG is one business, so the two segments below are synthetic. They are
    split off real consolidated revenue precisely so the coverage check passes
    and the bridge under test is a real priced one.
    """
    platform = {
        "name": "Platform",
        "revenue": ddog.revenue * 0.7,
        "operating_income": 900.0,
        "da": 150.0,
    }
    security = {
        "name": "Security",
        "revenue": ddog.revenue * 0.3,
        "operating_income": 100.0,
        "da": 50.0,
    }
    multiples = {
        "Platform": ("ebitda", 30.0, "peer median EV/EBITDA for observability"),
        "Security": ("revenue", 9.0, "peer median EV/Revenue for security software"),
    }
    r = run_sotp(
        ddog,
        ddog_bridge,
        [platform, security],
        multiples,
        assumptions,
        corporate_cost=200.0,
        corporate_multiple=25.0,
        corporate_multiple_source="the platform multiple, since overhead runs it",
    )

    assert r.equity_value == pytest.approx(
        equity_value_from_ev(r.net_enterprise_value, ddog, ddog_bridge)
    )
    assert r.per_share == pytest.approx(r.equity_value / ddog_bridge.diluted_shares)
    assert r.implied_vs_consolidated.gap == pytest.approx(
        r.net_enterprise_value - ddog_bridge.enterprise_value
    )
    assert r.implied_vs_consolidated.gap_per_share == pytest.approx(
        r.per_share - ddog_bridge.price
    )


def test_rows_carry_the_parts_the_deductions_and_the_gap(assumptions):
    r = _run(assumptions, conglomerate_discount=0.10, **CORP)
    rows = dict(r.rows())
    assert rows["Gross value of the parts"] == pytest.approx(22_800.0)
    assert rows["Capitalised corporate cost"] == pytest.approx(-2_400.0)
    assert rows["Conglomerate discount"] == pytest.approx(-2_040.0)
    assert rows["Enterprise value, sum of the parts"] == pytest.approx(18_360.0)
    assert rows["Value per share"] == pytest.approx(14.36)
    assert rows["Gap, dollars"] == pytest.approx(2_360.0)
    assert rows["Gap per share"] == pytest.approx(2.36)
    assert any(k.startswith("Cable at 7.0x") for k in rows)


def test_a_net_cash_filer_reports_the_gap_without_a_percentage(assumptions):
    """Dividing by an enterprise value at or below zero gives a sign, not a fact."""
    bridge = _bridge()
    bridge.cash = 20_000.0
    bridge.enterprise_value = -3_000.0
    bridge.net_debt = -15_000.0

    r = run_sotp(_fin(), bridge, [CABLE, STUDIOS], MULTIPLES, assumptions, **CORP)
    assert r.implied_vs_consolidated.gap_pct is None
    assert r.implied_vs_consolidated.gap == pytest.approx(20_400.0 + 3_000.0)
    assert any("not positive" in n for n in r.implied_vs_consolidated.notes)
    assert "Gap, % of market EV" not in dict(r.rows())
