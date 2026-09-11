"""Adjusted present value, and the one construction that proves it.

Most of these tests are properties: Ku below the levered cost of equity when
there is debt and equal to it when there is none, the Modigliani-Miller shield
above the Harris-Pringle shield whenever debt is cheaper than the assets, a net
cash filer reconciling to nothing.

The test that actually proves the module is ``test_constant_leverage_firm``. It
builds a firm for which APV and the WACC DCF must give the same number to the
last decimal, so any error in the discounting, the terminal shield or the
recovery of the terminal flow shows up as a failure rather than as a plausible
looking difference. The algebra is in that test's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pytest

from techval.apv import APVResult, run_apv, unlevered_cost_of_equity
from techval.config import Assumptions
from techval.dcf import project, run_dcf
from techval.errors import ConfigError, NotMeaningfulError
from techval.ev_bridge import EVBridge, build_ev_bridge
from techval.financials import Financials
from techval.wacc import WACCResult, compute_wacc

# Datadog's convertible conversion price from the notes footnote. With the stock
# far above it the notes are equity under ASU 2020-06, which leaves the company
# with no debt at all: the clean no-shield case.
DDOG_CONVERSION_PRICE = 148.15


# -- helpers over the fixture companies ------------------------------------- #


def _stack(fin, assumptions, market):
    """Bridge, WACC, DCF and APV for one filer, in the order they depend."""
    bridge = build_ev_bridge(fin, market.spot(fin.ticker), assumptions)
    wacc_result = compute_wacc(fin, bridge, market, assumptions)
    dcf_result = run_dcf(fin, bridge, wacc_result, assumptions)
    result = run_apv(fin, bridge, wacc_result, dcf_result, assumptions)
    return bridge, wacc_result, dcf_result, result


def _named(fin_by_ticker, ticker):
    return fin_by_ticker[ticker]


@pytest.fixture
def filers(ddog, crwd, mdb, zs):
    return {"DDOG": ddog, "CRWD": crwd, "MDB": mdb, "ZS": zs}


# -- Ku against the levered cost of equity ---------------------------------- #


@pytest.mark.parametrize("ticker", ["ZS", "CRWD", "MDB"])
def test_ku_sits_below_the_levered_cost_of_equity_with_debt(
    filers, ticker, assumptions, market
):
    """Unlevering removes financial risk, so the asset rate has to be lower.

    The gap is not approximate. Both rates are CAPM on the same risk-free rate,
    the same premium and the same size adjustment, so they differ by exactly the
    beta spread times the equity risk premium and nothing else.
    """
    fin = _named(filers, ticker)
    bridge, wacc_result, _, result = _stack(fin, assumptions, market)

    assert bridge.total_debt > 0
    assert result.unlevered_cost_of_equity < wacc_result.cost_of_equity

    beta_spread = wacc_result.levered_beta - wacc_result.unlevered_beta
    assert result.unlevered_cost_of_equity == pytest.approx(
        wacc_result.cost_of_equity - beta_spread * wacc_result.erp, rel=1e-12
    )


def test_ku_equals_the_cost_of_equity_when_there_is_no_debt(
    ddog, assumptions, market
):
    """No debt, no unlevering: the asset beta is the equity beta.

    Datadog's convertibles are deep in the money, so under ASU 2020-06 their
    shares already sit in diluted WASO and the notes are equity rather than debt.
    That leaves the company with a zero debt balance, and Hamada with D/E of zero
    is the identity.
    """
    assumptions.convertibles.conversion_price = DDOG_CONVERSION_PRICE
    bridge, wacc_result, _, result = _stack(ddog, assumptions, market)

    assert bridge.total_debt == 0
    assert result.unlevered_cost_of_equity == pytest.approx(
        wacc_result.cost_of_equity, rel=1e-12
    )


# -- the shield, and which rate discounts it -------------------------------- #


@pytest.mark.parametrize("ticker", ["ZS", "CRWD", "MDB"])
def test_modigliani_miller_values_the_shield_above_harris_pringle(
    filers, ticker, assumptions, market
):
    """The predicted sign, and the reason for it.

    Discounting at the cost of debt treats the shield as exactly as safe as the
    debt. Discounting at the unlevered cost of equity treats it as as risky as
    the firm. Whenever debt is cheaper than the assets, which is the ordinary
    case, the first rate is lower and therefore the first value is higher. The
    gap is the value of a fixed dollar debt schedule over a rebalanced one.
    """
    fin = _named(filers, ticker)

    assumptions.apv.shield_discount_rate = "cost_of_debt"
    _, wacc_result, _, mm = _stack(fin, assumptions, market)

    assumptions.apv.shield_discount_rate = "unlevered_cost_of_equity"
    _, _, _, hp = _stack(fin, assumptions, market)

    assert wacc_result.pretax_cost_of_debt < hp.unlevered_cost_of_equity
    assert mm.pv_tax_shield > hp.pv_tax_shield
    assert mm.total_value > hp.total_value

    # And the statement the module docstring makes: at the cost of debt, APV
    # comes out above the constant-WACC answer, because a constant WACC has
    # already assumed the rebalancing that makes the shield risky.
    assert mm.total_value > mm.wacc_enterprise_value


def test_a_cost_of_debt_above_ku_reverses_the_sign_and_is_flagged(
    ddog, assumptions, market
):
    """The synthetic rating can price debt above the assets, which cannot be true.

    Datadog's interest coverage is thin, so the synthetic credit model assigns a
    spread that puts the cost of debt fractionally above the unlevered cost of
    equity. That says lenders bear more systematic risk than the assets they lend
    against. The ordering of the two shield values reverses with it, and the
    module says so rather than presenting the reversal as a finding.
    """
    assumptions.apv.shield_discount_rate = "cost_of_debt"
    _, wacc_result, _, mm = _stack(ddog, assumptions, market)

    assumptions.apv.shield_discount_rate = "unlevered_cost_of_equity"
    _, _, _, hp = _stack(ddog, assumptions, market)

    assert wacc_result.pretax_cost_of_debt > hp.unlevered_cost_of_equity
    assert mm.pv_tax_shield < hp.pv_tax_shield
    assert any(
        "more systematic risk than the assets" in line for line in mm.checks
    )


def test_debt_is_held_flat_at_the_bridge_balance(zs, assumptions, market):
    """Every year carries the current balance, and the notes say so.

    Nothing in the assumptions describes an amortisation schedule, so none is
    invented. The consequence is that this shield is the one a company that never
    repays would claim, and that has to be stated rather than discovered.
    """
    bridge, wacc_result, _, result = _stack(zs, assumptions, market)

    assert len(result.shields) == assumptions.dcf.projection_years
    assert all(s.debt_balance == bridge.total_debt for s in result.shields)
    assert all(
        s.interest == pytest.approx(
            wacc_result.pretax_cost_of_debt * bridge.total_debt
        )
        for s in result.shields
    )
    assert all(
        s.shield == pytest.approx(wacc_result.tax_rate * s.interest)
        for s in result.shields
    )
    assert any("held flat" in note for note in result.notes)


def test_the_unlevered_value_re_discounts_the_dcf_flows_and_nothing_else(
    zs, assumptions, market
):
    """Same flows, same terminal dollar, one different rate.

    Rebuilt here by hand from ``dcf_result.projections`` and from the terminal
    value backed out as ``TV * (w - g)``. If the module re-forecast anything, the
    two would not land on the same number.
    """
    assumptions.apv.shield_discount_rate = "unlevered_cost_of_equity"
    _, _, dcf_result, result = _stack(zs, assumptions, market)

    ku = result.unlevered_cost_of_equity
    g = assumptions.dcf.terminal_growth
    n = len(dcf_result.projections)
    flows = np.array([p.fcff for p in dcf_result.projections])

    exponents = np.arange(1, n + 1, dtype=float) - 0.5
    pv_explicit = float(np.sum(flows * (1.0 + ku) ** -exponents))
    terminal_flow = dcf_result.terminal_gordon.value * (dcf_result.wacc - g)
    pv_terminal = (terminal_flow / (ku - g)) * (1.0 + ku) ** -(n - 0.5)

    assert result.unlevered_value == pytest.approx(pv_explicit + pv_terminal, rel=1e-12)


# -- the net cash reconciliation -------------------------------------------- #


def test_net_cash_datadog_reconciles_to_the_wacc_answer(ddog, assumptions, market):
    """A company with no debt has no financing side effect to value.

    Datadog is net cash by roughly four billion, and once its in-the-money
    convertibles are treated as the equity they are, it carries no debt at all.
    The shield is zero, Ku is the cost of equity, the WACC is the cost of equity,
    and APV is the WACC DCF run a second time. Anything other than agreement here
    is a bug in the discounting.
    """
    assumptions.convertibles.conversion_price = DDOG_CONVERSION_PRICE
    bridge, _, dcf_result, result = _stack(ddog, assumptions, market)

    assert bridge.net_debt < 0
    assert bridge.total_debt == 0
    assert result.pv_tax_shield == 0.0
    assert result.total_value == pytest.approx(
        dcf_result.enterprise_value_gordon, rel=1e-12
    )
    assert result.difference_pct == pytest.approx(0.0, abs=1e-12)
    assert any("the same calculation" in note for note in result.notes)


def test_no_debt_returns_a_zero_shield_rather_than_dividing_by_leverage(
    ddog, assumptions, market
):
    """The degenerate case returns a result, not an exception and not a NaN."""
    assumptions.convertibles.conversion_price = DDOG_CONVERSION_PRICE
    _, _, _, result = _stack(ddog, assumptions, market)

    assert isinstance(result, APVResult)
    assert result.debt_balance == 0
    assert result.shields and all(s.shield == 0.0 for s in result.shields)
    assert result.terminal_shield_value == 0.0
    assert result.pv_shield_explicit == 0.0
    assert result.pv_shield_terminal == 0.0
    assert result.mid_year_wedge == pytest.approx(0.0, abs=1e-9)


def test_a_shield_worth_almost_nothing_leaves_the_two_models_almost_agreed(
    mdb, assumptions, market
):
    """MongoDB's only debt is a finance lease, so the shield is rounding.

    Twenty six million of lease liability against a twenty nine billion market
    capitalisation weights debt at under a tenth of a percent. There is nothing
    for the two models to disagree about, and the reconciliation should show it.
    """
    assumptions.apv.shield_discount_rate = "unlevered_cost_of_equity"
    bridge, wacc_result, _, result = _stack(mdb, assumptions, market)

    assert 0 < wacc_result.weight_debt < 0.01
    assert abs(result.difference_pct) < 0.005
    assert result.shield_share < 0.001


# -- the synthetic firm: the real test -------------------------------------- #

_RF = 0.04
_ERP = 0.05
_TAX = 0.24
_KU = 0.10
_KD = 0.06
_DEBT = 400.0
_SHARES = 100.0


def _flat_financials() -> Financials:
    """A business that earns the same dollar forever, so its value is a ratio.

    Revenue does not grow, margins do not move, and working capital does not
    swing, which makes free cash flow identical in every year. Nothing about the
    company matters to this test except that the flow is flat.
    """
    revenue = 1_000.0
    ebit = 200.0
    return Financials(
        ticker="FLAT",
        entity_name="Flat Perpetuity Corp",
        cik=1,
        as_of=date(2026, 9, 10),
        revenue=revenue,
        gross_profit=None,
        ebit=ebit,
        da=25.0,
        sbc=0.0,
        net_income=ebit * (1.0 - _TAX),
        pretax_income=ebit,
        tax_expense=ebit * _TAX,
        interest_expense=_KD * _DEBT,
        operating_lease_cost=None,
        capex=30.0,
        cfo=None,
        diluted_shares=_SHARES,
        basic_shares=_SHARES,
        cash=0.0,
        short_term_investments=0.0,
        straight_debt=_DEBT,
        convertible_debt=0.0,
        operating_lease_liability=0.0,
        finance_lease_liability=0.0,
        nci=0.0,
        preferred=0.0,
        current_assets=None,
        current_liabilities=None,
        deferred_revenue=0.0,
    )


def _flat_assumptions(mid_year: bool) -> Assumptions:
    a = Assumptions()
    a.market.risk_free_rate = _RF
    a.market.equity_risk_premium = _ERP
    a.market.size_premium = 0.0
    a.tax.marginal_tax_rate = _TAX
    a.dcf.projection_years = 5
    a.dcf.revenue_growth_start = 0.0
    a.dcf.revenue_growth_terminal = 0.0
    a.dcf.ebit_margin_start = 0.20
    a.dcf.ebit_margin_terminal = 0.20
    a.dcf.terminal_growth = 0.0
    a.dcf.mid_year_convention = mid_year
    a.dcf.sbc_treatment = "expense"
    a.apv.shield_discount_rate = "unlevered_cost_of_equity"
    return a


@dataclass
class _ConstantLeverageCase:
    fin: Financials
    assumptions: Assumptions
    bridge: EVBridge
    wacc_result: WACCResult
    flow: float
    shield: float
    wacc: float
    firm_value: float


def _constant_leverage_case(mid_year: bool = False) -> _ConstantLeverageCase:
    """A firm whose WACC and APV must agree to the last decimal.

    Take a flat free cash flow ``F`` in perpetuity, a tax rate ``t``, debt ``D``
    at a pre-tax cost ``Kd``, and an unlevered cost of equity ``Ku``. Zero growth
    means the firm is worth the same on every future date, so holding debt flat
    in dollars IS holding it at a constant percentage of value, and the two
    financing policies the module distinguishes coincide. That is what makes the
    identity testable without arguing about rebalancing.

    Write ``S = t * Kd * D`` for the annual shield. Then:

        V_unlevered  = F / Ku
        PV(shield)   = S / Ku          discounted at Ku, Harris-Pringle
        APV          = (F + S) / Ku

    The WACC side. Under rebalancing the consistent discount rate is

        w = Ku - (D / V) * t * Kd

    and the levered value of a flat perpetuity is ``V = F / w``. Substituting:

        w = Ku - t * Kd * D * w / F
        w * (1 + t * Kd * D / F) = Ku
        w = Ku * F / (F + S)

    so

        V = F / w = (F + S) / Ku = APV.

    Exactly, not approximately. The construction below sets ``w`` from that last
    expression, sets the market weights to the leverage that same ``w`` implies,
    and relevers the cost of equity as ``Ke = Ku + (D/E) * (Ku - Kd)``, which is
    the Harris-Pringle relevering with no tax term and the one that makes
    ``E/V * Ke + D/V * Kd * (1 - t)`` return ``w``. Every condition the module
    lists for agreement is satisfied by construction, so the only thing left that
    can move the answer is the arithmetic under test.
    """
    fin = _flat_financials()
    assumptions = _flat_assumptions(mid_year)

    flows = project(fin, assumptions)
    flow = flows[0].fcff
    assert all(p.fcff == pytest.approx(flow, rel=1e-12) for p in flows)

    shield = _TAX * _KD * _DEBT
    wacc = _KU * flow / (flow + shield)
    firm_value = flow / wacc
    equity = firm_value - _DEBT
    cost_of_equity = _KU + (_DEBT / equity) * (_KU - _KD)

    # The weights have to reproduce that WACC, or the case is not the one the
    # docstring describes.
    assert (equity / firm_value) * cost_of_equity + (_DEBT / firm_value) * _KD * (
        1.0 - _TAX
    ) == pytest.approx(wacc, rel=1e-12)

    bridge = EVBridge(
        ticker="FLAT",
        price=equity / _SHARES,
        diluted_shares=_SHARES,
        equity_value=equity,
        straight_debt=_DEBT,
        convertible_debt=0.0,
        convertible_in_debt=0.0,
        finance_lease=0.0,
        operating_lease=0.0,
        operating_lease_in_debt=0.0,
        preferred=0.0,
        nci=0.0,
        cash=0.0,
        short_term_investments=0.0,
        enterprise_value=firm_value,
        ev_excluding_leases=firm_value,
        ev_including_leases=firm_value,
        net_debt=_DEBT,
        lease_convention="operating leases excluded, EBITDA basis (ASC 842)",
        convertible_treatment="none",
    )
    wacc_result = WACCResult(
        cost_of_equity=cost_of_equity,
        after_tax_cost_of_debt=_KD * (1.0 - _TAX),
        pretax_cost_of_debt=_KD,
        weight_equity=equity / firm_value,
        weight_debt=_DEBT / firm_value,
        wacc=wacc,
        risk_free_rate=_RF,
        erp=_ERP,
        levered_beta=(cost_of_equity - _RF) / _ERP,
        unlevered_beta=(_KU - _RF) / _ERP,
        tax_rate=_TAX,
        credit_rating=None,
    )
    return _ConstantLeverageCase(
        fin=fin,
        assumptions=assumptions,
        bridge=bridge,
        wacc_result=wacc_result,
        flow=flow,
        shield=shield,
        wacc=wacc,
        firm_value=firm_value,
    )


def test_constant_leverage_firm_reconciles_exactly():
    """APV at Ku equals the WACC enterprise value when every condition holds.

    This is the test the module exists to pass. See ``_constant_leverage_case``
    for the algebra. A tolerance of one part in a trillion leaves no room for a
    misplaced half year, a dropped terminal shield or a terminal flow rebuilt
    from the assumptions rather than recovered from the DCF.
    """
    case = _constant_leverage_case(mid_year=False)
    dcf_result = run_dcf(case.fin, case.bridge, case.wacc_result, case.assumptions)

    # The DCF collapses to F/w, which is where the algebra starts.
    assert dcf_result.enterprise_value_gordon == pytest.approx(
        case.firm_value, rel=1e-12
    )

    result = run_apv(
        case.fin, case.bridge, case.wacc_result, dcf_result, case.assumptions
    )

    assert result.unlevered_cost_of_equity == pytest.approx(_KU, rel=1e-12)
    assert result.unlevered_value == pytest.approx(case.flow / _KU, rel=1e-12)
    assert result.pv_tax_shield == pytest.approx(case.shield / _KU, rel=1e-12)
    assert result.total_value == pytest.approx(
        result.wacc_enterprise_value, rel=1e-12
    )
    assert result.difference_pct == pytest.approx(0.0, abs=1e-12)


def test_constant_leverage_firm_agrees_at_the_cost_of_debt_only_by_accident():
    """Move the shield to the cost of debt and the identity breaks, upward.

    Same firm, same flows, one changed assumption about financing policy. The
    cost of debt is below Ku, so the shield is worth more and APV exceeds the
    WACC answer. Nothing has gone wrong: the two numbers now describe different
    debt policies, which is the whole reason the setting exists.
    """
    case = _constant_leverage_case(mid_year=False)
    case.assumptions.apv.shield_discount_rate = "cost_of_debt"
    dcf_result = run_dcf(case.fin, case.bridge, case.wacc_result, case.assumptions)
    result = run_apv(
        case.fin, case.bridge, case.wacc_result, dcf_result, case.assumptions
    )

    assert result.shield_rate == pytest.approx(_KD)
    assert result.pv_tax_shield == pytest.approx(case.shield / _KD, rel=1e-12)
    assert result.total_value > result.wacc_enterprise_value
    # And the excess is exactly the shield revalued, nothing else having moved.
    assert result.difference == pytest.approx(
        case.shield / _KD - case.shield / _KU, rel=1e-12
    )


def test_the_mid_year_convention_is_the_entire_residual():
    """With mid-year discounting the identity picks up one known factor.

    Every flow moving half a year earlier multiplies a valuation by
    ``(1 + r)^0.5`` at its own rate. APV compounds at Ku and the WACC answer at
    w, and those differ, so the two stop agreeing by exactly
    ``((1 + Ku) / (1 + w))^0.5``. ``mid_year_wedge`` reports that amount, and
    taking it off restores the identity to the same tolerance as the year-end
    case. Which is the point: the residual is a timing convention, not a
    financing disagreement.
    """
    case = _constant_leverage_case(mid_year=True)
    dcf_result = run_dcf(case.fin, case.bridge, case.wacc_result, case.assumptions)
    result = run_apv(
        case.fin, case.bridge, case.wacc_result, dcf_result, case.assumptions
    )

    assert result.difference > 0
    assert result.mid_year_wedge == pytest.approx(result.difference, rel=1e-10)
    assert result.total_value - result.mid_year_wedge == pytest.approx(
        result.wacc_enterprise_value, rel=1e-12
    )
    # The wedge is the gross-up factor applied to the whole value, not a fudge.
    assert result.mid_year_wedge == pytest.approx(
        result.total_value * (1.0 - ((1.0 + case.wacc) / (1.0 + _KU)) ** 0.5),
        rel=1e-12,
    )


def test_the_terminal_shield_is_required_for_agreement():
    """Shield only the explicit years and APV falls short by a known amount.

    A constant WACC capitalises the tax benefit in perpetuity, because the rate
    that discounts the terminal value is itself net of it. An APV that stops
    shielding at year five is therefore guaranteed to come out lower whatever the
    financing policy, and an analyst reading it would conclude that leverage is
    worth less than it is. The shortfall here is exactly the terminal shield.
    """
    case = _constant_leverage_case(mid_year=False)
    dcf_result = run_dcf(case.fin, case.bridge, case.wacc_result, case.assumptions)
    result = run_apv(
        case.fin, case.bridge, case.wacc_result, dcf_result, case.assumptions
    )

    explicit_only = result.unlevered_value + result.pv_shield_explicit
    assert explicit_only < result.wacc_enterprise_value
    assert result.wacc_enterprise_value - explicit_only == pytest.approx(
        result.pv_shield_terminal, rel=1e-12
    )
    assert result.pv_shield_terminal > 0
    assert any("rigged" in note for note in result.notes)


def test_the_reconciliation_names_the_conditions():
    """Every condition for agreement is reported, satisfied or not."""
    case = _constant_leverage_case(mid_year=False)
    dcf_result = run_dcf(case.fin, case.bridge, case.wacc_result, case.assumptions)
    result = run_apv(
        case.fin, case.bridge, case.wacc_result, dcf_result, case.assumptions
    )

    text = "\n".join(result.checks)
    for condition in ("Condition 1", "Condition 2", "Condition 3", "Condition 4"):
        assert condition in text
    assert "agree to within rounding" in text
    assert not any(line.startswith("FLAG:") for line in result.checks)


def test_a_wacc_that_its_own_asset_beta_does_not_imply_is_flagged():
    """Condition 2 is the term that usually breaks on a real filer.

    Shift the WACC off the rate its asset beta implies under rebalancing and the
    module says so in basis points and in dollars, instead of reporting a
    difference and leaving the reader to assume model error.
    """
    case = _constant_leverage_case(mid_year=False)
    shifted = replace(case.wacc_result, wacc=case.wacc + 0.004)
    dcf_result = run_dcf(case.fin, case.bridge, shifted, case.assumptions)
    result = run_apv(case.fin, case.bridge, shifted, dcf_result, case.assumptions)

    flagged = [line for line in result.checks if line.startswith("FLAG:")]
    assert any("Condition 2" in line for line in flagged)
    assert any("40bp" in line for line in flagged)
    # A higher discount rate is a lower WACC value, so APV now sits above it.
    assert result.difference > 0


def test_market_leverage_far_from_model_leverage_is_flagged(zs, assumptions, market):
    """Condition 3: the WACC weights are the market's, the shield is the model's.

    Zscaler's DCF enterprise value is a fraction of its market enterprise value,
    so the same debt balance is six percent of the market's firm and nineteen
    percent of the model's. The WACC takes only six percent of a shield out of
    the discount rate while APV adds the whole of it onto the smaller base, and
    that is most of the gap between them.
    """
    assumptions.apv.shield_discount_rate = "unlevered_cost_of_equity"
    _, wacc_result, dcf_result, result = _stack(zs, assumptions, market)

    model_leverage = result.debt_balance / result.wacc_enterprise_value
    assert model_leverage > wacc_result.weight_debt * 2
    assert any(
        line.startswith("FLAG:") and "Condition 3" in line for line in result.checks
    )


# -- refusals ---------------------------------------------------------------- #


def test_a_size_premium_from_a_different_run_raises():
    """The cost of equity has to rebuild from the parts it is said to have.

    Ku is assembled from the WACC's risk-free rate and premium plus the
    assumptions' size premium. If those two objects are not from the same run,
    the rate is half of one model and half of another, and the reconciliation it
    produces would mean nothing. That is a refusal, not a warning.
    """
    case = _constant_leverage_case(mid_year=False)
    case.assumptions.market.size_premium = 0.02

    with pytest.raises(ConfigError, match="not from the same run"):
        unlevered_cost_of_equity(case.wacc_result, case.bridge, case.assumptions)


def test_an_asset_beta_above_the_equity_beta_raises():
    """Unlevering cannot raise a beta, so this pair is not one capital structure."""
    case = _constant_leverage_case(mid_year=False)
    broken = replace(
        case.wacc_result, unlevered_beta=case.wacc_result.levered_beta + 0.5
    )

    with pytest.raises(ConfigError, match="above the levered beta"):
        unlevered_cost_of_equity(broken, case.bridge, case.assumptions)


def test_a_shield_rate_at_or_below_terminal_growth_raises():
    """A perpetuity growing at least as fast as its discount rate is not a number."""
    case = _constant_leverage_case(mid_year=False)
    case.assumptions.apv.shield_discount_rate = "cost_of_debt"
    case.assumptions.dcf.terminal_growth = 0.0
    dcf_result = run_dcf(case.fin, case.bridge, case.wacc_result, case.assumptions)

    # Push terminal growth above the cost of debt after the DCF has been built,
    # which is the only way to reach the shield guard: the DCF's own guard fires
    # first on anything at or above the WACC.
    case.assumptions.dcf.terminal_growth = _KD + 0.005

    with pytest.raises(ConfigError, match="infinite or negative perpetuity"):
        run_apv(case.fin, case.bridge, case.wacc_result, dcf_result, case.assumptions)


def test_terminal_growth_above_ku_raises_even_when_the_wacc_clears_it():
    """The shield inside the WACC can hide a growth rate the business cannot carry.

    A WACC net of the tax benefit sits below Ku, so there is a band in which
    terminal growth clears the WACC and fails against the unlevered rate. A DCF
    run in that band is capitalising a perpetuity the business only supports
    because of how it is financed, which is exactly the crutch APV removes.
    """
    case = _constant_leverage_case(mid_year=False)
    dcf_result = run_dcf(case.fin, case.bridge, case.wacc_result, case.assumptions)
    case.assumptions.dcf.terminal_growth = _KU + 0.01

    with pytest.raises(ConfigError, match="not above terminal growth"):
        run_apv(case.fin, case.bridge, case.wacc_result, dcf_result, case.assumptions)


def test_a_zero_enterprise_value_is_not_meaningful():
    """No percentage can be taken against nothing, and none is invented."""
    case = _constant_leverage_case(mid_year=False)
    dcf_result = run_dcf(case.fin, case.bridge, case.wacc_result, case.assumptions)
    dcf_result.enterprise_value_gordon = 0.0

    with pytest.raises(NotMeaningfulError, match="is zero"):
        run_apv(case.fin, case.bridge, case.wacc_result, dcf_result, case.assumptions)


# -- output shape ------------------------------------------------------------ #


def test_rows_and_frame_are_raw_numbers(zs, assumptions, market):
    """Rendering lives in the renderer. These carry numbers and labels only."""
    _, _, _, result = _stack(zs, assumptions, market)

    rows = result.rows()
    assert all(isinstance(label, str) for label, _ in rows)
    assert all(isinstance(value, float) for _, value in rows)
    assert ("Adjusted present value", result.total_value) in rows

    frame = result.to_frame()
    assert len(frame) == assumptions.dcf.projection_years
    assert list(frame.columns) == [
        "Debt balance",
        "Interest",
        "Tax shield",
        "Discount factor",
        "PV",
    ]


def test_the_synthetic_yield_gap_against_filed_interest_is_flagged(
    zs, assumptions, market
):
    """A deduction is taken on interest paid, not on a rating model's yield.

    Zscaler's convertibles carry a coupon far below the synthetic spread its
    coverage implies, so the modelled shield is several times the one the tax
    return will show. The number is kept, because changing it would break the
    reconciliation the module exists to perform, and the gap is reported instead.
    """
    _, _, _, result = _stack(zs, assumptions, market)

    assert any(
        line.startswith("FLAG:") and "interest paid" in line for line in result.checks
    )
