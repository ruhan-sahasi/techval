"""Adjusted present value: value the business first, price the financing second.

    APV = unlevered enterprise value + present value of the financing side effects

A WACC discounted cash flow buries the tax benefit of debt inside the discount
rate. APV takes it back out and prices it as a stream in its own right. The two
are the same valuation seen from different angles, and the reason to run both is
the size of the gap between them. Where they disagree the disagreement is a
statement about financing policy, not an arithmetic error, and this module exists
to say which statement. Money is USD millions.

**The unlevered value.** The flows are not re-forecast. They are lifted straight
off ``dcf_result.projections``, the same unlevered free cash flows the DCF built,
and re-discounted at the unlevered cost of equity instead of the WACC. The
terminal value is recovered the same way: whatever dollar of flow the DCF
capitalised is backed out as ``TV * (w - g)`` and re-capitalised at ``Ku - g``,
so the Gordon and value-driver paths both carry over without being rebuilt. Two
valuations that re-derive their own cash flows are two models, and the difference
between them is then a mixture of financing policy and forecasting drift with no
way to tell which is which. Here the only difference is the discount rate.

**The unlevered cost of equity** is CAPM on the asset beta the WACC build already
produced:

    Ku = rf + beta_unlevered * ERP + size premium

That is the rate the business would be discounted at if it carried no debt at
all, so it sits below the levered cost of equity by exactly
``(beta_levered - beta_unlevered) * ERP``, and equals it when there is no debt.

**Which rate discounts the shield. This is the whole argument.**

Discounting the shield at the COST OF DEBT is Modigliani-Miller. It says the
shield is exactly as safe as the interest payment that creates it: the company
either pays the coupon and takes the deduction or it does neither, so the two
streams carry one risk. That is right when the debt schedule is FIXED IN
DOLLARS, a term loan amortising on a stated schedule, because then next year's
interest is known today whatever happens to the enterprise.

Discounting the shield at the UNLEVERED COST OF EQUITY is Harris-Pringle. It
says the shield is as risky as the firm. That is right when debt is REBALANCED
to a constant percentage of firm value continuously, because then the debt
balance, and with it the interest and the shield, moves with the enterprise and
inherits its risk.

Miles-Ezzell is the same policy with the rebalancing done once a year, and it is
neither of the two above. The balance is reset at each year end, so every year's
interest, and the shield on it, is known one year before it is paid. Every
year's shield is therefore discounted at the cost of debt for its final year and
at the unlevered cost of equity for each year before that:

    PV(TS_t) = TS_t / ((1 + Ku)^(t-1) * (1 + Kd))

Set beside Harris-Pringle's ``TS_t / (1 + Ku)^t``, that is the Harris-Pringle
value of each year's shield, the first year's included, multiplied by
``(1 + Ku) / (1 + Kd)``, and so, on year-end timing, the whole Harris-Pringle
shield, terminal included, scaled by the same factor. This module computes
Modigliani-Miller and Harris-Pringle. It does not compute Miles-Ezzell, and
``apv.shield_discount_rate`` does not offer it.

A constant-WACC model already assumes constant leverage. That is what a single
discount rate applied to every year means: the weights never move, so the debt is
rebalanced to value in every period. Therefore the rate consistent with a WACC
DCF is the SECOND one, Ku. Since the cost of debt is normally below Ku, APV run
at the cost of debt will systematically come out ABOVE the WACC answer. That gap
is not an error to be tuned away. It is the price of a fixed debt schedule over a
rebalanced one, and a company that really does hold its debt flat in dollars
while its equity compounds is genuinely worth more than a constant-WACC model
says.

**The debt schedule.** Debt is held flat at the bridge's current balance across
the explicit period, and the interest on it is the same pre-tax cost of debt the
WACC used, not the filed interest expense. Two reasons, and one cost. The rate
has to be the WACC's own Kd or the reconciliation is comparing two different
credit views rather than two different financing policies. The balance is held
flat because nothing in the assumptions describes a paydown, and inventing an
amortisation schedule would put a financing forecast into a module whose job is
to expose one. The cost is that a filer whose coupon is far below its synthetic
yield, which is the zero-coupon convertible signature, gets a modelled shield far
larger than the deduction it will actually claim. The checks report that gap
against filed interest expense rather than leaving it to be discovered.

**The terminal shield, without which the comparison is rigged.** A WACC DCF
capitalises the tax benefit forever, because the rate that discounts the terminal
value is itself net of it. An APV that shields only the explicit years is
therefore guaranteed to come out lower, and the analyst reading it would conclude
that leverage is worth less than it is. The terminal shield here is the final
explicit year's shield capitalised as a perpetuity on the same growth rate the
terminal value uses. That does assume debt grows at ``g`` in perpetuity while the
explicit period held it flat, which is an inconsistency inside the schedule
rather than a hidden one: it is stated in the notes on every run, and it vanishes
when ``g`` is zero.

**The reconciliation, which is the point of the module.** APV and the WACC answer
agree when four conditions hold together, and each one that fails shows up in
``checks`` with the basis points or the dollars it costs:

1. The shield is discounted at Ku, not at Kd.
2. The WACC is the one its own asset beta implies under rebalancing,
   ``w = Ku - (D/V) * t * Kd``. The WACC build unlevers with Hamada, which is the
   fixed-debt relation, and relevers with a debt beta of zero, so the WACC handed
   in is normally a few basis points away from that. This is usually the largest
   single term in the difference, and it is a property of the beta convention,
   not of the APV.
3. The leverage in the WACC weights, which is market debt over market
   capitalisation, equals the leverage in the model, which is debt over the
   enterprise value the DCF produced. A model that disagrees with the market by
   thirty percent is discounting at a capital structure it does not believe.
4. The mid-year convention is off, or its effect is taken out. Moving every flow
   half a year earlier multiplies each valuation by ``(1 + r)^0.5`` at its own
   rate, and the rates differ, so the identity picks up a factor of
   ``((1 + Ku) / (1 + w))^0.5``. ``mid_year_wedge`` reports that amount exactly,
   and subtracting it restores the identity.

With no debt there is nothing to reconcile. The shield is zero, Ku equals the
levered cost of equity because the asset beta equals the equity beta, and the two
valuations are the same arithmetic run twice. That case returns a zero shield and
says so rather than dividing by a leverage ratio that does not exist.

``apv.enabled`` is read by the runner, not here. Calling ``run_apv`` computes the
reconciliation whatever that flag says.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import APVAssumptions, Assumptions
from .dcf import DCFResult, discount_factors
from .errors import ConfigError, NotMeaningfulError
from .ev_bridge import EVBridge
from .financials import Financials

if TYPE_CHECKING:  # the WACC build is a peer module, imported for typing only
    from .wacc import WACCResult

# Diagnostic thresholds, not valuation inputs. They move no number, only whether
# a line in ``checks`` carries a FLAG, so they stay out of the assumptions file
# where every entry changes a price.
_WACC_GAP_FLAG = 0.0005
_RECONCILE_FLAG = 0.01
_LEVERAGE_GAP_FLAG = 0.05
_INTEREST_GAP_FLAG = 0.25
_TERMINAL_SHARE_FLAG = 0.75

# The cost of equity is rebuilt from its own parts to confirm the WACC result and
# the assumptions describe the same run. Anything looser than this is a genuine
# mismatch rather than floating point.
_RATE_TOLERANCE = 1e-8

# The DCF enterprise value is rebuilt from the flows before it is reconciled
# against. A residual above this means the rebuild is not the same calculation.
_REBUILD_TOLERANCE = 1e-6


@dataclass
class TaxShield:
    """One year of interest deduction, USD millions except the factor.

    ``debt_balance`` is the balance the interest is charged on, held flat at the
    bridge's current debt. ``shield`` is the tax actually not paid, interest times
    the marginal rate, and ``pv`` discounts it on the same clock the free cash
    flows use, because interest accrues through the year exactly as they do.
    """

    year: int
    debt_balance: float
    interest: float
    shield: float
    discount_factor: float
    pv: float


@dataclass
class APVResult:
    """The unlevered business, the financing, and the gap against the WACC answer.

    ``unlevered_value`` and ``wacc_enterprise_value`` are both enterprise values,
    which is where the comparison belongs: the walk down to equity is the same
    line-by-line subtraction in both models, so putting it in front of the
    reconciliation would only add a constant to both sides.

    ``shield_discount_rate`` is the policy chosen, ``'cost_of_debt'`` or
    ``'unlevered_cost_of_equity'``; ``shield_rate`` is the number it resolved to.

    ``mid_year_wedge`` is the part of ``difference`` created by the mid-year
    convention alone: the two valuations are grossed up half a year at different
    rates, and this is the dollar consequence. Subtracting it from
    ``difference`` leaves the part that is about financing policy.

    Lines in ``checks`` that begin with ``FLAG:`` failed a threshold; the rest are
    stated for the record.
    """

    unlevered_cost_of_equity: float
    unlevered_value: float
    pv_tax_shield: float
    total_value: float
    shields: list[TaxShield]
    wacc_enterprise_value: float
    difference: float
    difference_pct: float
    shield_discount_rate: str
    shield_rate: float
    wacc: float
    pv_shield_explicit: float
    pv_shield_terminal: float
    terminal_shield_value: float
    debt_balance: float
    mid_year_wedge: float
    terminal_method: str
    checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def shield_share(self) -> float:
        """What share of the enterprise is financing rather than operations."""
        if not self.total_value:
            raise NotMeaningfulError(
                "the adjusted present value is zero, so the shield cannot be "
                "expressed as a share of it."
            )
        return self.pv_tax_shield / self.total_value

    def rows(self) -> list[tuple[str, float]]:
        """The build-up, read top to bottom as a bridge from Ku to the difference."""
        return [
            ("Unlevered cost of equity (Ku)", self.unlevered_cost_of_equity),
            ("WACC", self.wacc),
            ("Unlevered enterprise value", self.unlevered_value),
            ("+ PV of tax shield, explicit period", self.pv_shield_explicit),
            ("+ PV of tax shield, terminal", self.pv_shield_terminal),
            ("Adjusted present value", self.total_value),
            ("WACC enterprise value", self.wacc_enterprise_value),
            ("Difference", self.difference),
            ("Difference, %", self.difference_pct),
            ("of which mid-year convention", self.mid_year_wedge),
        ]

    def to_frame(self) -> pd.DataFrame:
        """The shield schedule as raw numbers. Formatting belongs to the renderer."""
        return pd.DataFrame(
            [
                {
                    "Year": s.year,
                    "Debt balance": s.debt_balance,
                    "Interest": s.interest,
                    "Tax shield": s.shield,
                    "Discount factor": s.discount_factor,
                    "PV": s.pv,
                }
                for s in self.shields
            ]
        ).set_index("Year")


# -- the unlevered rate ---------------------------------------------------- #


def unlevered_cost_of_equity(
    wacc_result: "WACCResult", bridge: EVBridge, assumptions: Assumptions
) -> float:
    """CAPM on the asset beta: the rate this business earns with no debt at all.

        Ku = rf + beta_unlevered * ERP + size premium

    Every input is the one the WACC build actually used, so Ku and the levered
    cost of equity sit on the same stack and differ only in the beta. That makes
    the spread between them exactly ``(beta_levered - beta_unlevered) * ERP``,
    which is the whole of what leverage does to a CAPM discount rate, and it goes
    to zero when there is no debt to unlever.

    The size premium is the one input not carried on ``WACCResult``, so it comes
    from the assumptions and the cost of equity is then rebuilt from all four
    parts and checked against the figure the WACC reported. If they disagree the
    two objects describe different runs, and quietly discounting at a rate built
    from half of one and half of the other would produce a reconciliation that
    means nothing. That raises instead.
    """
    risk_free = wacc_result.risk_free_rate
    erp = wacc_result.erp
    size_premium = assumptions.market.size_premium

    rebuilt = risk_free + wacc_result.levered_beta * erp + size_premium
    if abs(rebuilt - wacc_result.cost_of_equity) > _RATE_TOLERANCE:
        raise ConfigError(
            f"the cost of equity rebuilds to {rebuilt:.6%} from a {risk_free:.2%} "
            f"risk-free rate, a levered beta of {wacc_result.levered_beta:,.4f} "
            f"against a {erp:.2%} equity risk premium and a {size_premium:.2%} size "
            f"premium, but the WACC result reports "
            f"{wacc_result.cost_of_equity:.6%}. The WACC and the assumptions passed "
            "to APV are not from the same run, most likely a different "
            "market.size_premium. Ku built from these parts would not be comparable "
            "to that WACC and no reconciliation is attempted."
        )

    beta_u = wacc_result.unlevered_beta
    beta_l = wacc_result.levered_beta
    if bridge.total_debt > 0 and beta_u > beta_l:
        raise ConfigError(
            f"the unlevered beta of {beta_u:,.4f} is above the levered beta of "
            f"{beta_l:,.4f} while {bridge.ticker} carries {bridge.total_debt:,.0f}mm "
            "of debt. Unlevering removes financial risk, so an asset beta cannot "
            "exceed the equity beta it came from. These two figures are not from "
            "the same capital structure."
        )
    if bridge.total_debt <= 0 and abs(beta_u - beta_l) > _RATE_TOLERANCE:
        raise ConfigError(
            f"{bridge.ticker} carries no debt in this bridge, so the asset beta and "
            f"the equity beta are the same number, yet they read {beta_u:,.4f} and "
            f"{beta_l:,.4f}. The WACC result was built on a different capital "
            "structure from the bridge handed to APV."
        )

    return risk_free + beta_u * erp + size_premium


# -- discounting ------------------------------------------------------------ #


def _value_at(
    rate: float,
    flows: np.ndarray,
    terminal_flow: float,
    g: float,
    mid_year: bool,
) -> tuple[float, float, np.ndarray]:
    """Explicit PV, terminal PV and the factors, on the DCF's own clock.

    The terminal flow is capitalised and then carried back ``n - 0.5`` years
    under the mid-year convention, which is what the DCF does with its Gordon
    value: the perpetuity's own flows arrive mid-year too, half a year ahead of
    where the formula puts them. Using the whole-period exponent here instead
    would make the unlevered terminal value sit on a different clock from the
    levered one and put a timing difference inside a financing comparison.
    """
    n = len(flows)
    factors = discount_factors(n, rate, mid_year)
    pv_explicit = float(np.sum(flows * factors))
    exponent = (n - 0.5) if mid_year else float(n)
    pv_terminal = (terminal_flow / (rate - g)) * (1.0 + rate) ** (-exponent)
    return pv_explicit, pv_terminal, factors


def _headline_terminal(dcf_result: DCFResult) -> tuple[float, str]:
    """The terminal value the DCF's headline figures were struck on.

    The caller recovers the flow behind it as ``TV * (w - g)`` rather than
    rebuilding it from the assumptions. For the Gordon path that is ``FCFF_N * (1 + g)`` and for the
    value driver it is steady-state NOPAT net of the reinvestment that funds
    growth, and in both cases it is the dollar the DCF actually capitalised. Any
    attempt to reconstruct it would risk a second difference creeping into a
    comparison that is supposed to isolate one.
    """
    method = dcf_result.headline_method
    if method == "value_driver" and dcf_result.terminal_value_driver is not None:
        terminal = dcf_result.terminal_value_driver
    else:
        method = "gordon"
        terminal = dcf_result.terminal_gordon
    return terminal.value, method


# -- the run ---------------------------------------------------------------- #


def run_apv(
    fin: Financials,
    bridge: EVBridge,
    wacc_result: "WACCResult",
    dcf_result: DCFResult,
    assumptions: Assumptions,
) -> APVResult:
    """Value the firm unlevered, price the shield, reconcile against the WACC."""
    cfg = assumptions.apv
    notes: list[str] = []
    checks: list[str] = []

    g = assumptions.dcf.terminal_growth
    mid_year = assumptions.dcf.mid_year_convention
    wacc = dcf_result.wacc
    tax_rate = wacc_result.tax_rate
    kd = wacc_result.pretax_cost_of_debt
    debt = bridge.total_debt

    ku = unlevered_cost_of_equity(wacc_result, bridge, assumptions)
    if ku <= g:
        raise ConfigError(
            f"the unlevered cost of equity of {ku:.2%} is not above terminal growth "
            f"of {g:.2%}, so the unlevered terminal value cannot be formed. The WACC "
            f"of {wacc:.2%} clears that bar only because the tax shield is inside it, "
            "which is precisely the crutch APV removes. Either the growth rate or the "
            "asset beta has to move."
        )

    # -- the same flows, a different rate ---------------------------------- #
    flows = np.array([p.fcff for p in dcf_result.projections])
    n = len(flows)
    terminal_value_w, method = _headline_terminal(dcf_result)
    terminal_flow = terminal_value_w * (wacc - g)

    pv_explicit_u, pv_terminal_u, _ = _value_at(ku, flows, terminal_flow, g, mid_year)
    unlevered_value = pv_explicit_u + pv_terminal_u

    wacc_enterprise_value = (
        dcf_result.enterprise_value_value_driver
        if method == "value_driver"
        else dcf_result.enterprise_value_gordon
    )
    if wacc_enterprise_value is None:
        wacc_enterprise_value = dcf_result.enterprise_value_gordon
    if wacc_enterprise_value == 0:
        raise NotMeaningfulError(
            f"the {method} enterprise value for {fin.ticker} is zero, so the APV "
            "difference cannot be expressed as a percentage of it and the "
            "reconciliation conveys nothing. The forecast, not the financing, is "
            "the thing to look at."
        )

    # -- the shield --------------------------------------------------------- #
    shield_rate = ku if cfg.shield_discount_rate == "unlevered_cost_of_equity" else kd
    if debt > 0 and shield_rate <= g:
        raise ConfigError(
            f"the tax shield is being discounted at {shield_rate:.2%}, which is not "
            f"above terminal growth of {g:.2%}, so the terminal shield is an infinite "
            "or negative perpetuity. A shield growing forever at least as fast as the "
            "rate that discounts it is worth more than the firm that pays for it."
        )

    interest = kd * debt
    annual_shield = tax_rate * interest
    shield_factors = (
        discount_factors(n, shield_rate, mid_year) if debt > 0 else np.zeros(n)
    )
    shields = [
        TaxShield(
            year=i + 1,
            debt_balance=debt,
            interest=interest,
            shield=annual_shield,
            discount_factor=float(shield_factors[i]),
            pv=annual_shield * float(shield_factors[i]),
        )
        for i in range(n)
    ]
    pv_shield_explicit = float(sum(s.pv for s in shields))

    if debt > 0:
        terminal_shield_value = annual_shield * (1.0 + g) / (shield_rate - g)
        exponent = (n - 0.5) if mid_year else float(n)
        pv_shield_terminal = terminal_shield_value * (1.0 + shield_rate) ** (-exponent)
    else:
        terminal_shield_value = 0.0
        pv_shield_terminal = 0.0
    pv_tax_shield = pv_shield_explicit + pv_shield_terminal

    total_value = unlevered_value + pv_tax_shield
    difference = total_value - wacc_enterprise_value
    difference_pct = difference / wacc_enterprise_value

    # The mid-year convention multiplies a valuation by (1+r)^0.5 at its own
    # rate, and the two models do not share a rate, so restating APV onto the
    # WACC model's clock isolates the timing part of the difference from the
    # financing part.
    if mid_year:
        on_wacc_clock = unlevered_value * ((1.0 + wacc) / (1.0 + ku)) ** 0.5
        if debt > 0:
            on_wacc_clock += pv_tax_shield * ((1.0 + wacc) / (1.0 + shield_rate)) ** 0.5
        mid_year_wedge = total_value - on_wacc_clock
    else:
        mid_year_wedge = 0.0

    notes.extend(
        _conventions(
            fin=fin,
            cfg=cfg,
            ku=ku,
            wacc=wacc,
            kd=kd,
            shield_rate=shield_rate,
            tax_rate=tax_rate,
            debt=debt,
            interest=interest,
            annual_shield=annual_shield,
            terminal_shield_value=terminal_shield_value,
            g=g,
            n=n,
            method=method,
            mid_year=mid_year,
            wacc_result=wacc_result,
        )
    )
    checks.extend(
        _reconciliation(
            fin=fin,
            bridge=bridge,
            wacc_result=wacc_result,
            dcf_result=dcf_result,
            cfg=cfg,
            ku=ku,
            wacc=wacc,
            kd=kd,
            tax_rate=tax_rate,
            debt=debt,
            interest=interest,
            g=g,
            flows=flows,
            terminal_flow=terminal_flow,
            mid_year=mid_year,
            pv_tax_shield=pv_tax_shield,
            pv_shield_terminal=pv_shield_terminal,
            total_value=total_value,
            wacc_enterprise_value=wacc_enterprise_value,
            difference=difference,
            difference_pct=difference_pct,
            mid_year_wedge=mid_year_wedge,
        )
    )

    return APVResult(
        unlevered_cost_of_equity=ku,
        unlevered_value=unlevered_value,
        pv_tax_shield=pv_tax_shield,
        total_value=total_value,
        shields=shields,
        wacc_enterprise_value=wacc_enterprise_value,
        difference=difference,
        difference_pct=difference_pct,
        shield_discount_rate=cfg.shield_discount_rate,
        shield_rate=shield_rate if debt > 0 else 0.0,
        wacc=wacc,
        pv_shield_explicit=pv_shield_explicit,
        pv_shield_terminal=pv_shield_terminal,
        terminal_shield_value=terminal_shield_value,
        debt_balance=debt,
        mid_year_wedge=mid_year_wedge,
        terminal_method=method,
        checks=checks,
        notes=notes,
    )


# -- what was assumed ------------------------------------------------------- #


def _conventions(
    *,
    fin: Financials,
    cfg: APVAssumptions,
    ku: float,
    wacc: float,
    kd: float,
    shield_rate: float,
    tax_rate: float,
    debt: float,
    interest: float,
    annual_shield: float,
    terminal_shield_value: float,
    g: float,
    n: int,
    method: str,
    mid_year: bool,
    wacc_result: "WACCResult",
) -> list[str]:
    """Every judgment the number rests on, stated before it is argued about."""
    out = [
        f"The unlevered cost of equity is {ku:.2%}: a {wacc_result.risk_free_rate:.2%} "
        f"risk-free rate plus an asset beta of {wacc_result.unlevered_beta:,.2f} "
        f"against a {wacc_result.erp:.2%} equity risk premium. It sits "
        f"{(wacc_result.cost_of_equity - ku) * 1e4:,.0f}bp below the levered cost of "
        f"equity of {wacc_result.cost_of_equity:.2%}, which is the whole of what "
        "leverage does to a CAPM rate.",
        f"The unlevered value re-discounts the same free cash flows the DCF "
        f"projected, {n} explicit years and the {method} terminal value, at {ku:.2%} "
        f"rather than {wacc:.2%}. Nothing is re-forecast, so the two valuations "
        "differ in the discount rate and in nothing else.",
    ]

    if debt <= 0:
        out.append(
            f"{fin.ticker} carries no debt in this bridge, so there is no interest, "
            "no shield, and nothing for APV to add. With no debt the asset beta is "
            "the equity beta, Ku is the cost of equity, and the WACC is the cost of "
            "equity as well: APV and the WACC DCF are the same calculation performed "
            "twice, and any difference between them is arithmetic, not finance."
        )
        return out

    out.append(
        f"Debt is held flat at {debt:,.0f}mm, the bridge's current balance, for all "
        f"{n} explicit years. Nothing in the assumptions describes a paydown or a "
        "draw, and inventing an amortisation schedule would put a financing forecast "
        "inside the module whose job is to expose one. Interest is that balance at "
        f"the {kd:.2%} pre-tax cost of debt the WACC used, {interest:,.0f}mm a year, "
        f"and the shield is {tax_rate:.1%} of it, {annual_shield:,.0f}mm."
    )

    if cfg.shield_discount_rate == "cost_of_debt":
        out.append(
            f"The shield is discounted at the {kd:.2%} cost of debt, the "
            "Modigliani-Miller treatment: the shield is exactly as safe as the "
            "interest payment that creates it, which is right when the debt schedule "
            "is fixed in dollars. Note that this model holds the balance flat, so "
            "that is the policy being priced. A constant-WACC DCF assumes the "
            "opposite, debt rebalanced to a constant share of value, so this APV will "
            f"come out above the WACC answer by construction while {kd:.2%} sits "
            f"below the {ku:.2%} unlevered cost of equity. The gap is the value of a "
            "fixed schedule over a rebalanced one, not an error."
        )
    else:
        out.append(
            f"The shield is discounted at the {ku:.2%} unlevered cost of equity, the "
            "Harris-Pringle treatment. It says the shield is as risky as the firm, "
            "which is right when debt is rebalanced to a constant percentage of firm "
            "value. That is the policy a "
            "constant-WACC model already assumes, so this is the setting under which "
            "the two valuations are supposed to agree."
        )

    out.append(
        f"The terminal shield capitalises the year {n} shield at {g:.2%} growth, "
        f"{terminal_shield_value:,.0f}mm at the end of year "
        f"{n}. Without it the comparison would be rigged: the WACC discount rate "
        "capitalises the tax benefit forever, so an APV shielding only the explicit "
        "years is guaranteed to come out lower whatever the financing policy. It does "
        "mean debt grows at g in perpetuity while the explicit period holds it flat, "
        "which is an inconsistency inside the schedule rather than a hidden one."
    )
    if mid_year:
        out.append(
            "Shields are discounted on the same mid-year clock as the free cash "
            "flows, since interest accrues through the year exactly as they do. "
            "Putting the shield at year end while the flows arrive mid-year would "
            "add a timing difference to a comparison meant to isolate financing."
        )
    return out


# -- what the difference is made of ----------------------------------------- #


def _reconciliation(
    *,
    fin: Financials,
    bridge: EVBridge,
    wacc_result: "WACCResult",
    dcf_result: DCFResult,
    cfg: APVAssumptions,
    ku: float,
    wacc: float,
    kd: float,
    tax_rate: float,
    debt: float,
    interest: float,
    g: float,
    flows: np.ndarray,
    terminal_flow: float,
    mid_year: bool,
    pv_tax_shield: float,
    pv_shield_terminal: float,
    total_value: float,
    wacc_enterprise_value: float,
    difference: float,
    difference_pct: float,
    mid_year_wedge: float,
) -> list[str]:
    """Take the gap apart term by term and name the condition each one violates.

    Reporting a difference without decomposing it invites the reader to treat it
    as model error and average the two answers, which is the one response that is
    certainly wrong. Each line below is a condition for agreement, quantified in
    basis points or dollars, so the reader can see which assumption is doing the
    work.
    """
    out: list[str] = []

    # The rebuild has to reproduce the DCF before anything is compared against
    # it. If it does not, the two are not on the same clock and every line below
    # is measuring the wrong thing.
    pv_explicit_w, pv_terminal_rebuilt, _ = _value_at(
        wacc, flows, terminal_flow, g, mid_year
    )
    rebuilt = pv_explicit_w + pv_terminal_rebuilt
    if abs(rebuilt - wacc_enterprise_value) > _REBUILD_TOLERANCE * max(
        abs(wacc_enterprise_value), 1.0
    ):
        out.append(
            f"FLAG: re-discounting the DCF's own flows at {wacc:.2%} gives "
            f"{rebuilt:,.0f}mm against the {wacc_enterprise_value:,.0f}mm the DCF "
            "reports, so APV is not rebuilding the same valuation and the "
            "reconciliation below is not measuring financing policy."
        )

    head = (
        f"APV of {total_value:,.0f}mm against a WACC enterprise value of "
        f"{wacc_enterprise_value:,.0f}mm: a difference of {difference:,.0f}mm, "
        f"{difference_pct:+.1%}."
    )
    if abs(difference_pct) > _RECONCILE_FLAG:
        out.append("FLAG: " + head + " The conditions for agreement are below.")
    else:
        out.append(head + " The two models agree to within rounding.")

    if debt <= 0:
        out.append(
            f"{fin.ticker} carries no debt, so there is no shield to discount and no "
            "financing policy to disagree about. Ku equals the levered cost of equity "
            f"at {ku:.2%} and the WACC is {wacc:.2%}; the two valuations are the same "
            "calculation and should differ only in the last decimal."
        )
        if abs(ku - wacc) > _WACC_GAP_FLAG:
            out.append(
                f"FLAG: with no debt the WACC should equal the cost of equity, yet it "
                f"reads {wacc:.2%} against Ku of {ku:.2%}, a gap of "
                f"{(wacc - ku) * 1e4:,.0f}bp. Either the DCF is running a WACC "
                "override or the WACC was built on a different capital structure from "
                "this bridge."
            )
        return out

    # -- condition 1: the rate on the shield -------------------------------- #
    other_rate = kd if cfg.shield_discount_rate == "unlevered_cost_of_equity" else ku
    rate_used = ku if cfg.shield_discount_rate == "unlevered_cost_of_equity" else kd
    if other_rate > g:
        n = len(flows)
        exponent = (n - 0.5) if mid_year else float(n)
        annual = tax_rate * interest
        alternative = float(
            np.sum(annual * discount_factors(n, other_rate, mid_year))
        ) + (annual * (1.0 + g) / (other_rate - g)) * (1.0 + other_rate) ** (-exponent)
        swing = pv_tax_shield - alternative
        label = (
            "Modigliani-Miller at the cost of debt"
            if cfg.shield_discount_rate == "cost_of_debt"
            else "Harris-Pringle at the unlevered cost of equity"
        )
        other_label = (
            "Harris-Pringle at the unlevered cost of equity"
            if cfg.shield_discount_rate == "cost_of_debt"
            else "Modigliani-Miller at the cost of debt"
        )
        out.append(
            f"Condition 1, the rate on the shield. {label} discounts it at "
            f"{rate_used:.2%} and values it at {pv_tax_shield:,.0f}mm; "
            f"{other_label} at {other_rate:.2%} would value it at "
            f"{alternative:,.0f}mm, a swing of {abs(swing):,.0f}mm. The constant-WACC "
            "model assumes leverage is held at a constant share of value, therefore "
            "that debt is rebalanced, therefore that the shield carries the firm's "
            "risk, so the unlevered cost of equity is the rate consistent with the "
            "answer being reconciled against."
        )
    if kd >= ku:
        out.append(
            f"FLAG: the pre-tax cost of debt of {kd:.2%} is at or above the unlevered "
            f"cost of equity of {ku:.2%}. That says the lenders bear more systematic "
            "risk than the assets they lend against, which cannot be true of a "
            "solvent borrower. It is the synthetic-rating signature on a thin-coverage "
            "filer, and it reverses the usual sign: here Modigliani-Miller values the "
            "shield BELOW Harris-Pringle rather than above it."
        )

    # -- condition 2: the WACC its own asset beta implies -------------------- #
    weight_debt = wacc_result.weight_debt
    implied_wacc = ku - weight_debt * tax_rate * kd
    gap = wacc - implied_wacc
    line = (
        f"Condition 2, the WACC implied by the asset beta. Under rebalancing the "
        f"consistent rate is Ku - (D/V) x t x Kd = {ku:.2%} - {weight_debt:.2%} x "
        f"{tax_rate:.1%} x {kd:.2%} = {implied_wacc:.2%}, against the {wacc:.2%} the "
        f"DCF used, a gap of {gap * 1e4:,.0f}bp."
    )
    if implied_wacc > g:
        value_at_implied = sum(
            _value_at(implied_wacc, flows, terminal_flow, g, mid_year)[:2]
        )
        line += (
            f" Discounting the same flows at it is worth "
            f"{value_at_implied - wacc_enterprise_value:,.0f}mm of enterprise value."
        )
    else:
        line += (
            f" That rate is at or below terminal growth of {g:.2%}, so the valuation "
            "it implies cannot even be formed, which is itself the answer: this "
            "capital structure does not support the growth assumption."
        )
    line += (
        " The WACC build unlevers with Hamada, which is the fixed-debt relation, and "
        "relevers with a debt beta of zero while charging a spread over the "
        "risk-free rate, so the two conventions are not the same one."
    )
    out.append(("FLAG: " + line) if abs(gap) > _WACC_GAP_FLAG else line)

    # -- condition 3: whose leverage ----------------------------------------- #
    model_leverage = debt / wacc_enterprise_value
    leverage_gap = model_leverage - weight_debt
    line = (
        f"Condition 3, whose capital structure. The WACC weights debt at "
        f"{weight_debt:.2%} of market capitalisation, while the DCF's own enterprise "
        f"value of {wacc_enterprise_value:,.0f}mm puts the same {debt:,.0f}mm of debt "
        f"at {model_leverage:.2%}. A model discounts at the leverage it believes or "
        "at the leverage the market quotes, and those are the same number only when "
        "the model agrees with the market."
    )
    out.append(
        ("FLAG: " + line) if abs(leverage_gap) > _LEVERAGE_GAP_FLAG else line
    )

    # -- condition 4: the mid-year clock ------------------------------------- #
    if mid_year:
        out.append(
            f"Condition 4, the mid-year convention. Moving every flow half a year "
            f"earlier multiplies each valuation by (1+r)^0.5 at its own rate, and the "
            f"rates differ, so the identity picks up a factor of "
            f"((1+{ku:.2%})/(1+{wacc:.2%}))^0.5. That is worth "
            f"{mid_year_wedge:,.0f}mm of the {difference:,.0f}mm difference, leaving "
            f"{difference - mid_year_wedge:,.0f}mm that is about financing policy. "
            "Turning the convention off removes this term exactly."
        )
    else:
        out.append(
            "Condition 4, the mid-year convention, is off, so both valuations sit on "
            "the same year-end clock and contribute nothing to the difference."
        )

    # -- the terminal shield, and the interest that is really paid ----------- #
    share = pv_shield_terminal / pv_tax_shield if pv_tax_shield else float("nan")
    line = (
        f"The terminal shield is {pv_shield_terminal:,.0f}mm of the "
        f"{pv_tax_shield:,.0f}mm total, {share:.0%}. Dropping it would understate "
        f"APV by that amount and the comparison would be rigged, because the WACC "
        "capitalises the tax benefit in perpetuity inside its own discount rate."
    )
    out.append(("FLAG: " + line) if share > _TERMINAL_SHARE_FLAG else line)

    filed = fin.interest_expense
    if filed is None:
        out.append(
            f"{fin.ticker} reports no interest expense, so the {interest:,.0f}mm of "
            f"modelled interest cannot be checked against a filed figure. The shield "
            "rests entirely on the synthetic yield."
        )
    elif filed <= 0 or abs(interest / filed - 1.0) > _INTEREST_GAP_FLAG:
        out.append(
            f"FLAG: modelled interest of {interest:,.0f}mm is "
            f"{interest / filed:,.1f}x the {filed:,.0f}mm this filer actually reports "
            f"on {debt:,.0f}mm of debt. A deduction is taken on interest paid, not on "
            "a synthetic yield, so the shield here is larger than the one the tax "
            "return will show. This is the low-coupon convertible signature: the "
            "lender was paid in an equity option, which is not deductible. Read the "
            "shield as an upper bound."
        )
    else:
        out.append(
            f"Modelled interest of {interest:,.0f}mm is within "
            f"{_INTEREST_GAP_FLAG:.0%} of the {filed:,.0f}mm filed, so the shield is "
            "close to the deduction actually being claimed."
        )

    if bridge.net_debt < 0:
        out.append(
            f"{fin.ticker} is net cash by {-bridge.net_debt:,.0f}mm. The shield still "
            f"attaches to the {debt:,.0f}mm of gross debt outstanding, since a "
            "deduction is generated by interest paid and not by a net position, but "
            f"at {pv_tax_shield / total_value:.1%} of value the financing side of "
            "this valuation is close to immaterial and the two models have little "
            "room to disagree."
        )

    if abs(dcf_result.wacc - wacc_result.wacc) > _RATE_TOLERANCE:
        out.append(
            f"FLAG: the DCF discounted at {dcf_result.wacc:.2%} while the WACC build "
            f"produced {wacc_result.wacc:.2%}, so dcf.wacc_override is in force. Ku "
            "comes from the beta stack behind the computed WACC, and the value it is "
            "being reconciled against does not, so the difference below mixes the "
            "override in with the financing policy."
        )
    return out
