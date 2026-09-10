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
)
from techval.errors import ConfigError
from techval.wacc import compute_wacc


@pytest.fixture
def wacc_result(ddog, ddog_bridge, market, assumptions):
    return compute_wacc(ddog, ddog_bridge, market, assumptions)


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
