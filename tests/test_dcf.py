"""Projection mechanics, discounting conventions and terminal value."""

from __future__ import annotations

import numpy as np
import pytest

from techval.config import Assumptions
from techval.dcf import (
    discount_factors,
    exit_multiple_terminal_value,
    gordon_terminal_value,
    project,
    run_dcf,
    sensitivity_wacc_growth,
    value_driver_terminal_value,
)
from techval.edgar import CompanyFacts
from techval.errors import ConfigError
from techval.wacc import compute_wacc

# Pinned so the per-share figures quoted in the docstrings below stay true
# whatever the CAPM build does next. Every test that reads a number rather than
# a relationship sets this.
PINNED_WACC = 0.105


@pytest.fixture
def wacc_result(ddog, ddog_bridge, market, assumptions):
    return compute_wacc(ddog, ddog_bridge, market, assumptions)


def _steady_year(rows, assumptions, tax_rate, g):
    """Year N+1 rebuilt independently of the module, to check it against.

    Deliberately arithmetic rather than an import of the module's own helper: a
    test that reuses the implementation only proves the implementation equals
    itself.
    """
    cfg = assumptions.dcf
    revenue = rows[-1].revenue * (1 + g)
    ebit = cfg.ebit_margin_terminal * revenue
    nopat = ebit - tax_rate * max(ebit, 0.0)
    if cfg.sbc_treatment == "addback":
        nopat += (rows[-1].sbc / rows[-1].revenue) * revenue
    da = cfg.da_pct_revenue * revenue
    capex = cfg.capex_pct_revenue * revenue
    delta_nwc = cfg.nwc_pct_revenue * rows[-1].revenue * g
    return nopat, capex + delta_nwc - da


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #


def test_growth_fades_linearly_from_start_to_terminal(ddog, assumptions):
    assumptions.dcf.projection_years = 5
    assumptions.dcf.revenue_growth_start = 0.20
    assumptions.dcf.revenue_growth_terminal = 0.08
    rows = project(ddog, assumptions)

    assert rows[0].growth == pytest.approx(0.20)
    assert rows[-1].growth == pytest.approx(0.08)
    steps = np.diff([r.growth for r in rows])
    assert np.allclose(steps, steps[0])


def test_margin_path_anchors_on_the_realised_margin_when_unset(ddog, assumptions):
    assumptions.dcf.ebit_margin_start = None
    assumptions.dcf.ebit_margin_terminal = 0.22
    rows = project(ddog, assumptions)
    assert rows[0].ebit_margin == pytest.approx(ddog.ebit_margin)
    assert rows[-1].ebit_margin == pytest.approx(0.22)


def test_revenue_compounds_off_the_ttm_base(ddog, assumptions):
    rows = project(ddog, assumptions)
    assert rows[0].revenue == pytest.approx(ddog.revenue * (1 + rows[0].growth))
    for prev, cur in zip(rows, rows[1:]):
        assert cur.revenue == pytest.approx(prev.revenue * (1 + cur.growth))


def test_negative_working_capital_releases_cash_as_revenue_grows(ddog, assumptions):
    """Subscription software bills ahead of delivery, so growth funds itself."""
    assumptions.dcf.nwc_pct_revenue = -0.06
    rows = project(ddog, assumptions)
    assert all(r.delta_nwc < 0 for r in rows)
    assert all(r.fcff > r.nopat - r.capex + r.da - 1e-9 for r in rows)


def test_no_tax_benefit_is_booked_on_a_loss_year(ddog, assumptions):
    """A loss creates a carryforward, not a refund, and no NOL balance is tracked."""
    assumptions.dcf.ebit_margin_start = -0.10
    assumptions.dcf.ebit_margin_terminal = -0.05
    rows = project(ddog, assumptions)
    assert all(r.ebit < 0 for r in rows)
    assert all(r.taxes == 0 for r in rows)


def test_sbc_expensed_is_the_default_and_adds_nothing_back(ddog, assumptions):
    """GAAP EBIT is already net of SBC, so the default must not touch it.

    The build specification's formula subtracts SBC from an EBIT that already
    contains it, which charges the same compensation twice.
    """
    assumptions.dcf.sbc_treatment = "expense"
    rows = project(ddog, assumptions)
    r = rows[0]
    assert r.fcff == pytest.approx(r.nopat + r.da - r.capex - r.delta_nwc)


def test_sbc_addback_raises_free_cash_flow_by_the_charge(ddog, assumptions):
    expensed = project(ddog, assumptions)
    assumptions.dcf.sbc_treatment = "addback"
    added = project(ddog, assumptions)
    for a, b in zip(expensed, added):
        assert b.fcff > a.fcff
        assert b.fcff - a.fcff == pytest.approx(b.sbc, rel=1e-6)


# --------------------------------------------------------------------------- #
# discounting
# --------------------------------------------------------------------------- #


def test_mid_year_factors_use_t_minus_a_half():
    f = discount_factors(3, 0.10, True)
    assert f[0] == pytest.approx(1.10**-0.5)
    assert f[1] == pytest.approx(1.10**-1.5)
    assert f[2] == pytest.approx(1.10**-2.5)


def test_year_end_factors_use_whole_periods():
    f = discount_factors(3, 0.10, False)
    assert f[0] == pytest.approx(1.10**-1)
    assert f[2] == pytest.approx(1.10**-3)


def test_mid_year_discounting_is_worth_more_than_year_end():
    mid = discount_factors(5, 0.10, True)
    end = discount_factors(5, 0.10, False)
    assert (mid > end).all()


# --------------------------------------------------------------------------- #
# terminal value
# --------------------------------------------------------------------------- #


def test_gordon_is_the_perpetuity_formula():
    assert gordon_terminal_value(100.0, 0.10, 0.025) == pytest.approx(
        100.0 * 1.025 / (0.10 - 0.025)
    )


def test_gordon_requires_growth_below_the_discount_rate():
    with pytest.raises(ConfigError) as exc:
        gordon_terminal_value(100.0, 0.05, 0.06)
    assert "0.05" in str(exc.value) or "5" in str(exc.value)


def test_exit_multiple_is_the_multiple_times_terminal_ebitda():
    assert exit_multiple_terminal_value(500.0, 14.0) == pytest.approx(7000.0)


def test_terminal_methods_discount_on_different_clocks(ddog, ddog_bridge, assumptions,
                                                       wacc_result):
    """The distinction bankers routinely get wrong.

    A Gordon terminal value capitalises a stream whose flows arrive mid-year
    under the mid-year convention, so it carries a half-year uplift. An exit
    multiple is a price struck at a date, the end of year N, so it is discounted
    over whole periods with no uplift. Applying the mid-year factor to both
    overstates the exit-multiple value by roughly half a year of WACC.
    """
    assumptions.dcf.mid_year_convention = True
    assumptions.dcf.exit_multiple = 20.0
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    n = assumptions.dcf.projection_years
    w = d.wacc
    gordon_ratio = d.terminal_gordon.pv / d.terminal_gordon.value
    exit_ratio = d.terminal_exit.pv / d.terminal_exit.value

    assert gordon_ratio == pytest.approx((1 + w) ** -(n - 0.5), rel=1e-9)
    assert exit_ratio == pytest.approx((1 + w) ** -n, rel=1e-9)
    assert gordon_ratio > exit_ratio


def test_gordon_and_exit_reconcile_through_the_implied_multiple(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """Setting the exit multiple to the one Gordon implies makes them agree.

    Agreement means equal PRESENT values. The implied multiple is restated onto
    the exit method's whole-period clock, so under mid-year discounting the two
    end-of-year-N amounts deliberately differ by the half-year factor while the
    discounted answers coincide.
    """
    assumptions.dcf.exit_multiple = 20.0
    first = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    assumptions.dcf.exit_multiple = first.terminal_gordon.implied_exit_multiple
    second = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    assert second.terminal_exit.pv == pytest.approx(
        second.terminal_gordon.pv, rel=1e-9
    )
    assert second.terminal_exit.value == pytest.approx(
        second.terminal_gordon.value * (1 + second.wacc) ** 0.5, rel=1e-9
    )


def test_terminal_value_share_is_flagged_when_it_dominates(
    ddog, ddog_bridge, assumptions, wacc_result
):
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    assert 0.0 < d.terminal_gordon.pct_of_ev < 1.0
    if d.terminal_gordon.pct_of_ev > 0.75:
        assert any("terminal" in c.lower() and "FLAG" in c for c in d.checks)


def test_reinvestment_check_ties_growth_to_return_on_capital(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """In perpetuity g = ROIC x reinvestment rate, and the model should say so.

    The single best test of whether a terminal assumption is coherent: growth
    has to be paid for out of reinvestment, and the return that reinvestment
    earns is implied by the two together.
    """
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    assert any("ROIC" in c or "reinvestment" in c.lower() for c in d.checks)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def test_enterprise_value_is_pv_of_flows_plus_terminal(
    ddog, ddog_bridge, assumptions, wacc_result
):
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    explicit = sum(p.pv for p in d.projections)
    assert d.enterprise_value_gordon == pytest.approx(
        explicit + d.terminal_gordon.pv
    )


def test_per_share_walks_the_bridge_backwards(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """A DCF equity value must use the same debt definition as the comps."""
    from techval.ev_bridge import equity_value_from_ev

    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    expected = equity_value_from_ev(d.enterprise_value_gordon, ddog, ddog_bridge)
    assert d.equity_value_gordon == pytest.approx(expected)
    assert d.per_share_gordon == pytest.approx(expected / ddog.diluted_shares)


def test_exit_valuation_is_omitted_rather_than_guessed(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """With no exit multiple supplied and no peer median, there is no answer."""
    assumptions.dcf.exit_multiple = None
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions, peer_median_ev_ebitda=None)
    assert d.enterprise_value_exit is None
    assert d.per_share_exit is None
    assert any("exit" in n.lower() for n in d.notes + d.checks)


def test_terminal_growth_above_wacc_is_rejected(
    ddog, ddog_bridge, assumptions, wacc_result
):
    assumptions.dcf.terminal_growth = 0.055
    object.__setattr__(wacc_result, "wacc", 0.05)
    with pytest.raises(ConfigError):
        run_dcf(ddog, ddog_bridge, wacc_result, assumptions)


def test_sensitivity_grid_is_monotonic_in_both_directions(ddog, ddog_bridge, assumptions):
    """Higher discount rate lowers value; higher terminal growth raises it."""
    grid = sensitivity_wacc_growth(ddog, ddog_bridge, assumptions, 0.12)
    values = grid.to_numpy(dtype=float)
    finite = np.isfinite(values)
    for col in range(values.shape[1]):
        col_vals = values[finite[:, col], col]
        assert np.all(np.diff(col_vals) <= 1e-6), "value should fall as WACC rises"
    for row in range(values.shape[0]):
        row_vals = values[row, finite[row, :]]
        assert np.all(np.diff(row_vals) >= -1e-6), "value should rise with growth"


def test_dcf_equity_walk_never_subtracts_operating_leases(
    ddog, assumptions, market, wacc_result
):
    """The DCF flows pay rent every year, so the lease is already serviced.

    Under the capitalise-leases convention the bridge counts the liability as
    debt for multiples, but a DCF enterprise value built on post-rent FCFF is a
    lease-excluded EV by construction. Subtracting the liability on the walk to
    equity would charge the same lease twice, so the equity value must be
    identical under either convention.
    """
    from techval.ev_bridge import build_ev_bridge

    price = market.spot("DDOG")
    plain = run_dcf(ddog, build_ev_bridge(ddog, price, assumptions), wacc_result,
                    assumptions)

    assumptions.leases.capitalize_operating_leases = True
    leased_bridge = build_ev_bridge(ddog, price, assumptions)
    leased = run_dcf(ddog, leased_bridge, wacc_result, assumptions)

    assert ddog.operating_lease_liability > 0
    assert leased.equity_value_gordon == pytest.approx(plain.equity_value_gordon)
    assert leased.per_share_gordon == pytest.approx(plain.per_share_gordon)


def test_implied_exit_multiple_sits_on_the_exit_clock(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """Feeding the implied multiple back must reproduce Gordon's PRESENT value.

    The two terminal methods discount on different clocks under the mid-year
    convention, so the multiple that means "the same answer as Gordon" is the
    one that equates present values, not end-of-year-N amounts. Equal PVs make
    equal enterprise values and equal per-share figures, which is the property
    a reader actually wants from the cross-check.
    """
    assumptions.dcf.mid_year_convention = True
    assumptions.dcf.exit_multiple = 20.0
    first = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    assumptions.dcf.exit_multiple = first.terminal_gordon.implied_exit_multiple
    second = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    assert second.terminal_exit.pv == pytest.approx(
        second.terminal_gordon.pv, rel=1e-9
    )
    assert second.per_share_exit == pytest.approx(second.per_share_gordon, rel=1e-9)

    # And the multiple itself carries the half-year gross-up against the raw
    # end-of-year ratio.
    raw = second.terminal_gordon.value / (
        second.terminal_exit.value / first.terminal_gordon.implied_exit_multiple
    )
    assert first.terminal_gordon.implied_exit_multiple == pytest.approx(
        raw * (1 + first.wacc) ** 0.5, rel=1e-9
    )


def test_reinvestment_check_reads_a_g_consistent_steady_state(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """The identity must be read at the perpetuity's growth, not the fade path's.

    The terminal explicit year grows at revenue_growth_terminal, so its
    working-capital release is sized for that faster growth and flatters the
    reinvestment rate. The check builds the first perpetuity year at g and
    quotes that rate instead, and it discloses when the shortcut terminal flow
    disagrees with the steady state it claims to capitalise.
    """
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    line = next(c for c in d.checks if "reinvestment" in c.lower())
    assert "Steady-state" in line
    assert f"{assumptions.dcf.terminal_growth:.2%}" in line


# --------------------------------------------------------------------------- #
# SBC dilution
# --------------------------------------------------------------------------- #


def test_sbc_dilution_issues_the_shares_the_addback_is_paid_with(ddog, assumptions):
    """Each year issues SBC dollars divided by the share price, and they compound."""
    assumptions.dcf.sbc_treatment = "addback"
    assumptions.dcf.sbc_dilution = True
    price = 200.0
    rows = project(ddog, assumptions, price=price)

    count = ddog.diluted_shares
    for r in rows:
        assert r.sbc_shares_issued == pytest.approx(r.sbc / price)
        count += r.sbc_shares_issued
        assert r.shares_outstanding == pytest.approx(count)
    assert rows[-1].shares_outstanding > ddog.diluted_shares


def test_sbc_dilution_does_nothing_while_the_charge_is_expensed(ddog, assumptions):
    """Under expensing there is no benefit taken, so there is no cost to charge.

    The dilution is the other half of the addback, not a separate view of the
    share count. Leaving SBC inside EBIT already charges the compensation in
    full, and issuing shares on top of that would count it twice.
    """
    assumptions.dcf.sbc_treatment = "expense"
    assumptions.dcf.sbc_dilution = True
    rows = project(ddog, assumptions, price=200.0)
    assert all(r.sbc_shares_issued == 0.0 for r in rows)
    assert all(r.shares_outstanding == ddog.diluted_shares for r in rows)


def test_a_projection_without_a_price_cannot_issue_shares(ddog, assumptions):
    """No price, no share count, and the column says so rather than guessing."""
    assumptions.dcf.sbc_treatment = "addback"
    rows = project(ddog, assumptions)
    assert all(r.sbc_shares_issued == 0.0 for r in rows)
    assert all(r.shares_outstanding == ddog.diluted_shares for r in rows)


def test_a_zero_price_is_refused_rather_than_dividing_by_it(ddog, assumptions):
    assumptions.dcf.sbc_treatment = "addback"
    with pytest.raises(ConfigError):
        project(ddog, assumptions, price=0.0)


def test_per_share_divides_by_the_terminal_share_count(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """The register the terminal value belongs to is the one at year N.

    Every share the addback pays for has been issued by the end of the explicit
    period, so the ending count is the denominator. Dividing by the opening
    count would take the cash the company saved by paying in equity and hand it
    to holders who were diluted to provide it.
    """
    assumptions.dcf.sbc_treatment = "addback"
    assumptions.dcf.sbc_dilution = True
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    assert d.shares_for_value == pytest.approx(d.projections[-1].shares_outstanding)
    assert d.shares_for_value > ddog.diluted_shares
    assert d.per_share_gordon == pytest.approx(
        d.equity_value_gordon / d.shares_for_value
    )


def test_cumulative_dilution_is_reported_as_a_percentage(
    ddog, ddog_bridge, assumptions, wacc_result
):
    assumptions.dcf.sbc_treatment = "addback"
    assumptions.dcf.sbc_dilution = True
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    line = next(c for c in d.checks if "cumulative dilution" in c)
    opening = ddog.diluted_shares
    expected = d.shares_for_value / opening - 1.0
    assert f"{expected:.1%}" in line


def test_dilution_closes_part_of_the_gap_between_the_two_sbc_camps(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """The payoff, and it is a partial one. Datadog at a pinned 10.5% WACC:

        expense                      $39.55
        addback, no dilution         $85.67
        addback, with dilution       $79.60

    Modelling the shares closes 13% of the gap, and the two camps are still a
    factor of two apart. That is the honest answer, not a convergence, and the
    reason is structural rather than a matter of calibration. Roughly 71% of the
    addback's uplift in enterprise value sits in the terminal value, which
    capitalises the SBC addback in perpetuity, while the modelled share count
    stops growing at year five. Five years of issuance is 7.6% of the register
    and it cannot pay for a perpetuity.

    Two further pressures push the same way. Datadog's SBC runs at a fifth of
    revenue, so the addback is enormous relative to a 0.4% GAAP EBIT margin. And
    the shares are issued at the 225.27 market price while the model says the
    equity is worth 39.55, so each dollar of compensation buys very few shares.
    Issuing at the model's own value per share would dilute far harder.

    What the test therefore asserts is the direction and the size of the move,
    and that the checks tell the reader the rest.
    """
    assumptions.dcf.wacc_override = PINNED_WACC

    assumptions.dcf.sbc_treatment = "expense"
    expensed = run_dcf(ddog, ddog_bridge, wacc_result, assumptions).per_share_gordon

    assumptions.dcf.sbc_treatment = "addback"
    assumptions.dcf.sbc_dilution = False
    undiluted = run_dcf(ddog, ddog_bridge, wacc_result, assumptions).per_share_gordon

    assumptions.dcf.sbc_dilution = True
    diluted = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    assert expensed == pytest.approx(39.55, rel=1e-3)
    assert undiluted == pytest.approx(85.67, rel=1e-3)
    assert diluted.per_share_gordon == pytest.approx(79.60, rel=1e-3)

    # Closer, and by enough to matter, but nowhere near equal.
    assert abs(diluted.per_share_gordon - expensed) < abs(undiluted - expensed)
    assert diluted.per_share_gordon > 1.5 * expensed

    # The perpetuity is where the rest of the gap lives, and a reader is told so.
    assert any(
        "capitalises the addback forever" in c and c.startswith("FLAG:")
        for c in diluted.checks
    )


def test_the_terminal_dilution_rate_is_set_against_terminal_growth(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """A company issuing 1.7% of itself a year is not growing at 2.5% per share.

    This is the check that makes the unclosed half of the loop legible. The
    terminal year's issuance rate is a perpetual drag on per-share cash flow,
    and set beside the terminal growth rate it says what the addback camp is
    really claiming.
    """
    assumptions.dcf.wacc_override = PINNED_WACC
    assumptions.dcf.sbc_treatment = "addback"
    assumptions.dcf.sbc_dilution = True
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    terminal = d.projections[-1]
    rate = terminal.sbc / (ddog_bridge.price * terminal.shares_outstanding)
    per_share_growth = (1 + assumptions.dcf.terminal_growth) / (1 + rate) - 1

    line = next(c for c in d.checks if "stock compensation" in c and "FLAG" in c)
    assert f"{rate:.2%}" in line
    assert f"{per_share_growth:.2%}" in line
    assert per_share_growth < assumptions.dcf.terminal_growth


# --------------------------------------------------------------------------- #
# net operating losses
# --------------------------------------------------------------------------- #


def test_the_limitation_binds_however_large_the_carryforward(ddog, assumptions):
    """The post-2017 rule, and the one people forget.

    Federal losses carry forward indefinitely but shelter only 80% of taxable
    income in any year, so a company sitting on a decade of accumulated losses
    still writes a cheque the moment it turns profitable. A model that shelters
    the whole of taxable income overstates free cash flow by a fifth of the tax
    bill for as long as the balance lasts.
    """
    assumptions.dcf.nol.track = True
    assumptions.dcf.nol.opening_balance = 50_000.0  # far larger than any usage
    rows = project(ddog, assumptions)
    rate = assumptions.tax.marginal_tax_rate
    limit = assumptions.dcf.nol.annual_limitation_pct

    for r in rows:
        assert r.ebit > 0
        assert r.nol_used == pytest.approx(limit * r.ebit)
        assert r.cash_taxes == pytest.approx(rate * (1 - limit) * r.ebit)
        assert r.cash_taxes > 0
        assert r.nopat == pytest.approx(r.ebit - r.cash_taxes)


def test_a_loss_year_adds_to_the_balance_and_pays_nothing(ddog, assumptions):
    """A loss creates a carryforward, which is the whole point of tracking one."""
    assumptions.dcf.ebit_margin_start = -0.10
    assumptions.dcf.ebit_margin_terminal = -0.05
    assumptions.dcf.nol.track = True
    assumptions.dcf.nol.opening_balance = 1_000.0
    rows = project(ddog, assumptions)

    for r in rows:
        assert r.ebit < 0
        assert r.cash_taxes == 0.0
        assert r.nol_used == 0.0
        assert r.nol_closing == pytest.approx(r.nol_opening - r.ebit)
    assert rows[-1].nol_closing > 1_000.0


def test_the_balance_runs_down_as_it_is_used(ddog, assumptions):
    assumptions.dcf.nol.track = True
    assumptions.dcf.nol.opening_balance = 2_000.0
    rows = project(ddog, assumptions)

    assert rows[0].nol_opening == 2_000.0
    for prev, cur in zip(rows, rows[1:]):
        assert cur.nol_opening == pytest.approx(prev.nol_closing)
        assert cur.nol_closing == pytest.approx(cur.nol_opening - cur.nol_used)
    assert rows[-1].nol_closing < rows[0].nol_opening


def test_the_shield_is_reported_in_present_value(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """A gross NOL balance is a headline. The discounted tax it stops is the value."""
    assumptions.dcf.wacc_override = PINNED_WACC
    assumptions.dcf.nol.track = True
    assumptions.dcf.nol.opening_balance = 3_000.0
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    rate = assumptions.tax.marginal_tax_rate
    shielded = sum(rate * p.nol_used for p in d.projections)
    pv = sum(rate * p.nol_used * p.discount_factor for p in d.projections)
    assert 0 < pv < shielded

    line = next(c for c in d.checks if "shelters" in c)
    assert f"{shielded:,.0f}mm" in line
    assert f"{pv:,.0f}mm" in line
    assert any("section 382" in c for c in d.checks)


def test_tracking_the_carryforward_raises_the_valuation(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """The shield is cash, so it has to reach the answer, not just the footnotes."""
    assumptions.dcf.wacc_override = PINNED_WACC
    base = run_dcf(ddog, ddog_bridge, wacc_result, assumptions).per_share_gordon

    assumptions.dcf.nol.track = True
    assumptions.dcf.nol.opening_balance = 3_000.0
    sheltered = run_dcf(ddog, ddog_bridge, wacc_result, assumptions).per_share_gordon
    assert sheltered > base


def test_the_opening_balance_is_read_from_the_filings_when_the_assumption_is_null(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """OperatingLossCarryforwards is an instant, and an annual one.

    The tax footnote is written once a year, so the newest fact can be almost
    four quarters behind the balance sheet the rest of the model runs on.
    Rejecting it for staleness would reject every filer, so it is taken and the
    note says how old it is.
    """
    facts = CompanyFacts(
        {
            "entityName": "Datadog, Inc.",
            "cik": 1561550,
            "facts": {
                "us-gaap": {
                    "OperatingLossCarryforwards": {
                        "units": {
                            "USD": [
                                {
                                    "val": 4_200_000_000,
                                    "end": "2025-12-31",
                                    "filed": "2026-02-20",
                                    "form": "10-K",
                                }
                            ]
                        }
                    }
                }
            },
        },
        "DDOG",
    )
    ddog.facts = facts
    assumptions.dcf.nol.track = True
    assumptions.dcf.nol.opening_balance = None

    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    assert d.projections[0].nol_opening == pytest.approx(4_200.0)
    note = next(n for n in d.notes if "NOL carryforwards open" in n)
    assert "OperatingLossCarryforwards" in note
    assert "2025-12-31" in note


def test_the_assumption_beats_the_tag(ddog, ddog_bridge, assumptions, wacc_result):
    """An analyst reading the tax footnote knows more than the tag does.

    The gross federal carryforward is usually prose in the footnote, split by
    jurisdiction and by expiry. Where the assumptions file carries a number it is
    the number a human looked up, so it wins.
    """
    ddog.facts = CompanyFacts(
        {
            "facts": {
                "us-gaap": {
                    "OperatingLossCarryforwards": {
                        "units": {
                            "USD": [
                                {
                                    "val": 4_200_000_000,
                                    "end": "2025-12-31",
                                    "filed": "2026-02-20",
                                    "form": "10-K",
                                }
                            ]
                        }
                    }
                }
            }
        },
        "DDOG",
    )
    assumptions.dcf.nol.track = True
    assumptions.dcf.nol.opening_balance = 900.0
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    assert d.projections[0].nol_opening == pytest.approx(900.0)


def test_tracking_without_a_balance_anywhere_is_an_error(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """Dimensioned facts do not reach companyfacts, so absence is common.

    A filer that tags the carryforward only by jurisdiction publishes nothing
    consolidated, and the balance is sitting in the 10-K where a person can read
    it. Guessing at zero would quietly hand the company a tax bill it will not
    pay; guessing at something else would be worse.
    """
    assumptions.dcf.nol.track = True
    assumptions.dcf.nol.opening_balance = None
    with pytest.raises(ConfigError) as exc:
        run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    assert "nol.opening_balance" in str(exc.value)


def test_the_carryforward_columns_stay_off_the_page_when_unused(
    ddog, ddog_bridge, assumptions, wacc_result
):
    base = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    assert "NOL opening" not in base.to_frame().columns

    assumptions.dcf.nol.track = True
    assumptions.dcf.nol.opening_balance = 3_000.0
    tracked = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    for column in ("NOL opening", "NOL used", "NOL closing", "Cash taxes"):
        assert column in tracked.to_frame().columns


# --------------------------------------------------------------------------- #
# value-driver terminal value
# --------------------------------------------------------------------------- #


def test_value_driver_reinvests_what_the_return_requires():
    """TV = NOPAT * (1 - g/ROIC) / (WACC - g), and the bracket is the point."""
    tv = value_driver_terminal_value(1_000.0, 0.10, 0.03, 0.15)
    assert tv == pytest.approx(1_000.0 * (1 - 0.03 / 0.15) / (0.10 - 0.03))


def test_at_roic_equal_to_wacc_growth_is_worth_exactly_nothing():
    """The cleanest statement of competitive equilibrium available.

    When a business earns precisely its cost of capital, the reinvestment growth
    demands costs exactly what the growth is worth, so the terminal value is
    NOPAT/WACC whatever g happens to be. Any model where g still moves the answer
    under that assumption has a return assumption hidden in it somewhere.
    """
    for g in (0.0, 0.01, 0.025, 0.04):
        assert value_driver_terminal_value(1_000.0, 0.10, g, 0.10) == pytest.approx(
            1_000.0 / 0.10
        )


def test_a_non_positive_terminal_roic_is_refused():
    """With ROIC below zero the bracket exceeds one and growth adds value."""
    with pytest.raises(ConfigError):
        value_driver_terminal_value(1_000.0, 0.10, 0.03, -0.05)
    with pytest.raises(ConfigError):
        value_driver_terminal_value(1_000.0, 0.10, 0.03, 0.0)


def test_the_null_roic_falls_back_to_the_cost_of_capital(
    ddog, ddog_bridge, assumptions, wacc_result
):
    assumptions.dcf.wacc_override = PINNED_WACC
    assumptions.dcf.terminal.terminal_roic = None
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    tax_rate = assumptions.tax.marginal_tax_rate
    nopat, _ = _steady_year(
        d.projections, assumptions, tax_rate, assumptions.dcf.terminal_growth
    )
    assert d.terminal_value_driver.value == pytest.approx(nopat / PINNED_WACC)
    assert any("competitive" in n.lower() for n in d.notes)


def test_the_value_driver_is_discounted_on_the_gordon_clock(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """A perpetuity of flows, not a price at a date.

    The value driver capitalises a stream in exactly the way Gordon does, so it
    carries the same half-year uplift under the mid-year convention. Only the
    exit multiple, which is a price observed at the end of year N, is discounted
    over whole periods.
    """
    assumptions.dcf.mid_year_convention = True
    assumptions.dcf.exit_multiple = 20.0
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    n = assumptions.dcf.projection_years

    vd = d.terminal_value_driver
    assert vd.pv / vd.value == pytest.approx((1 + d.wacc) ** -(n - 0.5), rel=1e-9)
    assert vd.pv / vd.value == pytest.approx(
        d.terminal_gordon.pv / d.terminal_gordon.value, rel=1e-9
    )
    assert vd.pv / vd.value > d.terminal_exit.pv / d.terminal_exit.value


def test_value_driver_and_gordon_agree_when_the_reinvestment_agrees(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """The two perpetuities differ in one input, and the check has to say so.

    Both are (NOPAT less reinvestment) / (WACC - g) on the first perpetuity year.
    Gordon takes the reinvestment from the capex and working-capital assumptions
    and leaves the return implied; the value driver takes the return and derives
    the reinvestment. Set the ROIC to the one the Gordon path implies and the two
    terminal values are the same number, which is the cleanest demonstration that
    they are one formula with two different inputs held fixed.
    """
    assumptions.dcf.wacc_override = PINNED_WACC
    g = assumptions.dcf.terminal_growth
    tax_rate = assumptions.tax.marginal_tax_rate

    first = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    nopat, reinvestment = _steady_year(first.projections, assumptions, tax_rate, g)
    implied_roic = nopat * g / reinvestment

    assumptions.dcf.terminal.terminal_roic = implied_roic
    second = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    steady_gordon = (nopat - reinvestment) / (PINNED_WACC - g)
    assert second.terminal_value_driver.value == pytest.approx(steady_gordon, rel=1e-9)

    line = next(c for c in second.checks if "Value-driver terminal value" in c)
    assert f"{reinvestment:,.0f}mm" in line
    assert "reinvests 0mm" in line


def test_a_return_below_the_cost_of_capital_is_flagged(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """Growth at a return below the WACC destroys value, and raising g hurts."""
    assumptions.dcf.wacc_override = PINNED_WACC
    assumptions.dcf.terminal.terminal_roic = 0.05
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    assert any(
        c.startswith("FLAG:") and "5.0%" in c and "destroying value" in c
        for c in d.checks
    )

    assumptions.dcf.terminal_growth = 0.035
    faster = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    assert faster.terminal_value_driver.value < d.terminal_value_driver.value


def test_a_franchise_return_is_flagged(ddog, ddog_bridge, assumptions, wacc_result):
    assumptions.dcf.wacc_override = PINNED_WACC
    assumptions.dcf.terminal.terminal_roic = 0.80
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    assert any(c.startswith("FLAG:") and "80.0%" in c for c in d.checks)


def test_all_three_terminal_methods_are_reported_together(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """Never blended, and never quietly dropped either."""
    assumptions.dcf.wacc_override = PINNED_WACC
    assumptions.dcf.exit_multiple = 20.0
    d = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    assert d.terminal_gordon.method == "gordon"
    assert d.terminal_exit.method == "exit_multiple"
    assert d.terminal_value_driver.method == "value_driver"
    assert d.enterprise_value_value_driver == pytest.approx(
        sum(p.pv for p in d.projections) + d.terminal_value_driver.pv
    )
    assert d.per_share_value_driver == pytest.approx(
        d.equity_value_value_driver / d.shares_for_value
    )


def test_the_method_setting_picks_the_headline_and_leaves_the_rest_alone(
    ddog, ddog_bridge, assumptions, wacc_result
):
    """Downstream reads the Gordon fields, so they must not move under it."""
    assumptions.dcf.wacc_override = PINNED_WACC
    assumptions.dcf.exit_multiple = 20.0

    gordon_led = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)
    assert gordon_led.headline_method == "gordon"
    assert gordon_led.per_share == pytest.approx(gordon_led.per_share_gordon)

    assumptions.dcf.terminal.method = "value_driver"
    driver_led = run_dcf(ddog, ddog_bridge, wacc_result, assumptions)

    assert driver_led.headline_method == "value_driver"
    assert driver_led.per_share == pytest.approx(driver_led.per_share_value_driver)
    assert driver_led.enterprise_value == pytest.approx(
        driver_led.enterprise_value_value_driver
    )
    # The per-method figures are identical under either setting.
    assert driver_led.per_share_gordon == pytest.approx(gordon_led.per_share_gordon)
    assert driver_led.per_share_exit == pytest.approx(gordon_led.per_share_exit)
