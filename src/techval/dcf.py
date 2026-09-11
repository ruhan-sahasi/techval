"""Unlevered discounted cash flow.

Free cash flow to the firm, discounted at the weighted average cost of capital,
then walked back to equity with the same debt definition the trading comps use,
so a DCF price and a comps price are reconcilable rather than merely adjacent.
Money is USD millions, share counts are millions, per-share figures are dollars.

    FCFF = EBIT * (1 - t) + D&A - capex - change in net working capital

**Stock-based compensation, and the double count.** GAAP EBIT is already net of
SBC. It is an operating expense on the face of the income statement, not a line
below it. So under the default treatment this model adds nothing back, because
nothing was removed. A formula written as ``EBIT*(1-t) + D&A - capex - dNWC -
SBC`` charges the expense twice and understates free cash flow by the whole SBC
amount, which for a subscription software company is routinely a tenth to a
fifth of revenue. That formula is wrong and is not implemented here.

Both camps deserve a hearing. Expensing, the default, treats SBC as what it is:
compensation the company elected to settle in shares instead of cash. An
engineer paid in equity is paid. Adding it back, available as
``sbc_treatment='addback'``, treats it as a non-cash charge in the manner of
depreciation. That is only honest when the dilution it creates is modelled
alongside it, because the cash the company did not spend was funded by existing
holders giving up claim. When the addback is on, it sits after tax next to D&A,
since the award is deductible when it vests.

**Closing that loop: SBC dilution.** With ``sbc_dilution`` on, which is the
default, every projected year issues ``SBC dollars / share price`` new shares and
the count carried forward grows by them. Three judgments sit inside that line.

The price is held at today's price for the whole forecast. A rising share price
would issue fewer shares for the same dollars, so a constant price is not a
neutral assumption, it is a choice. Growing the price at the cost of equity would
make the share count depend on the answer the model is trying to produce, and the
whole point of the exercise is to hold an intrinsic value up against the price
quoted today. Where the model's own per-share value is far below that price, as
it is for most of this cohort, shares are being issued at a valuation the model
does not believe, so the dilution charged here is the smallest defensible one.

The per-share value divides by the share count at the END of the explicit
period, not by the starting count and not by the average. Every share the
addback pays for has been issued by then, and the terminal value belongs to the
register as it stands at that date. Against the year one flows that is too many
shares. Against the terminal value it is far too few, for the reason below, and
the second error is much the larger of the two.

The loop does not close all the way, and pretending otherwise would be worse
than leaving it open. On Datadog, whose SBC runs at a fifth of revenue, at a
10.5% discount rate and the default assumptions, expensing gives 39.55 a share,
the addback without dilution gives 85.67, and the addback with dilution gives
79.60. Five years of issuance is 7.6% of the count, and it
cannot pay for a terminal value that capitalises the addback in perpetuity:
roughly 71% of the uplift sits in the terminal value, which no modelled share
issuance funds. The checks report the cumulative dilution and the terminal year's
issuance rate against the terminal growth rate, because a company issuing 1.7% of
itself every year while growing at 2.5% is growing at 0.8% per share, and that is
the number the addback camp has to defend.

**Net operating losses.** Off by default, and on it replaces the tax on positive
EBIT with a cash tax. A projected loss adds to the carryforward balance and pays
nothing. A profitable year shelters the lesser of the balance and
``annual_limitation_pct`` of taxable income, and pays tax on the rest. The
limitation is the post-2017 federal rule, and it is the part people forget: a
company with a decade of accumulated losses still pays cash tax the moment it
turns profitable, because only 80% of taxable income can be sheltered however
large the carryforward. NOPAT is struck on the cash tax, so the shield reaches
free cash flow in the year it is used rather than being asserted in a footnote.

What the NOL model does not do: section 382, which caps the annual use of a
carryforward after a change of control at roughly the equity value times the
long-term tax-exempt rate, so any deal case here overstates the shield; state
carryforwards, which have their own expiry rules and apportionment; and the
valuation allowance, so the balance used is the gross federal carryforward and
not the deferred tax asset the filer believes it will realise. Nor is the
residual balance carried into the terminal value, which is struck at the full
rate, so a large unused balance at year N is worth something this model does not
count.

**Terminal value the value-driver way.** The Gordon path assumes capex and
working capital in the terminal year and leaves the implied return on capital to
be checked afterwards. ``terminal.method='value_driver'`` inverts that:

    TV_N = NOPAT_{N+1} * (1 - g / ROIC) / (WACC - g)

``g / ROIC`` is the reinvestment rate that growth of ``g`` requires at a return
of ``ROIC``, so ``(1 - g / ROIC)`` is the share of NOPAT that reaches investors.
Reinvestment is derived from the return the analyst will defend rather than
assumed independently, and the terminal value cannot embed a return nobody signed
up for. With ``terminal_roic`` null the return falls back to the WACC and the
term becomes ``(1 - g / WACC)``, the competitive-equilibrium view that growth
creates exactly zero value. At ROIC equal to WACC the formula collapses to
``NOPAT / WACC`` for any g at all, which is the cleanest statement of that idea
available: if a business earns its cost of capital and nothing more, how fast it
grows does not change what it is worth.

All three terminal values are computed on every run and reported side by side.
``terminal.method`` picks which one drives the headline figures; the Gordon and
exit-multiple fields keep their own meanings whatever it is set to.

**Working capital.** Non-cash working capital is carried at a fixed share of
revenue, and for most subscription software that share is negative: customers
are billed ahead of delivery and the deferred revenue balance funds the
business. Growth therefore releases cash rather than consuming it, and the
change in NWC is a source in every growing year. The year one change is measured
against ``nwc_pct_revenue`` applied to TTM revenue, not against the reported
balance-sheet NWC. Anchoring on the reported balance would put a one-time
step, the gap between the filer's actual working capital and the assumed ratio,
into year one and leave the rest of the path clean, which reads as an operating
event that was never forecast. Measuring against the ratio keeps the whole path
internally consistent at the cost of ignoring where the balance sheet actually
starts. That trade is stated rather than hidden.

**Mid-year convention.** Explicit-period flows arrive through the year, not on
its last day, so they are discounted at ``(1+w)^-(t-0.5)``.

**Terminal value timing, which is where models quietly break.** The two terminal
methods do not take the same treatment, and assuming they do is a common error.

The Gordon formula ``TV_N = FCFF_N * (1+g) / (w - g)`` values a perpetuity of
end-of-year flows as of the end of year N. When the mid-year convention is on,
the flows in that perpetuity also arrive mid-year, half a year earlier than the
formula assumes, so the terminal value must be scaled by ``(1+w)^0.5`` before
being discounted back N years. Equivalently, discount it at ``(1+w)^-(N-0.5)``.
That is what this module does.

An exit multiple is not a flow. It is an observed market price at a point in
time, the end of year N, in the same way a share price is a price on a date. It
is discounted at ``(1+w)^-N`` with no mid-year adjustment, because there is no
stream arriving early to adjust for. Applying the half-year uplift to an exit
multiple inflates the terminal value by roughly half the cost of capital, and
bankers do it often enough that the two methods here are deliberately given
different code paths.

**Terminal EBITDA** for the exit multiple is terminal EBIT plus terminal D&A,
with D&A at ``da_pct_revenue`` of terminal revenue. It is a post-rent figure
under ASC 842, exactly the EBITDA definition the comps module quotes multiples
on, so a peer median multiple can be applied to it without silently switching
conventions. If operating leases are capitalised into debt the comp denominator
becomes EBITDAR and that equivalence breaks, which the checks report.

**Reinvestment consistency** is the check worth reading first. In perpetuity,
``g = ROIC * reinvestment rate``, so any terminal growth assumption implies a
return on capital whether the analyst states one or not. Backing it out of the
terminal year is the fastest way to see whether a model is coherent or is simply
capitalising an optimistic number. A negative implied ROIC means growth is being
funded by disinvestment, an implied ROIC above sixty percent means a perpetual
franchise no competitor ever attacks, and an implied ROIC below the WACC means
the model is capitalising growth that destroys value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import Assumptions
from .errors import ConfigError, MissingDataError, NotMeaningfulError
from .ev_bridge import EVBridge, equity_value_from_dcf_ev
from .financials import Financials

if TYPE_CHECKING:  # the WACC build is a peer module, imported for typing only
    from .wacc import WACCResult

# Diagnostic thresholds, not valuation inputs. They change no number in the
# model, only whether a line in ``checks`` is flagged, so they stay out of the
# assumptions file where every entry moves a price.
_TV_SHARE_FLAG = 0.75
_ROIC_CEILING = 0.60
_TERMINAL_DILUTION_FLAG = 0.005
_SENSITIVITY_POINTS = 5
_EXIT_MULTIPLE_SPREAD = 0.25

# Unit convention, not an assumption: companyfacts reports dollars and every
# figure in this engine is millions.
_MM = 1e6

# The gross federal carryforward. Deliberately one entry: the deferred-tax-asset
# tags that look like near neighbours are the balance already tax-effected.
_NOL_TAGS = ("OperatingLossCarryforwards",)

# A tax-footnote instant is published once a year, so the newest one can sit
# almost four quarters behind the balance sheet the rest of the model uses.
_NOL_TOLERANCE_DAYS = 400

_TV_LABELS = {
    "gordon": "Gordon",
    "exit_multiple": "Exit multiple",
    "value_driver": "Value driver",
}


@dataclass
class ProjectionYear:
    """One explicit forecast year, USD millions except the rates and the counts.

    ``taxes`` is the charge at the tax rate on positive EBIT, before any
    carryforward. ``cash_taxes`` is what is actually paid after the carryforward
    shelters what it is allowed to shelter, and NOPAT is struck on that, because
    a shield that never reaches the cash flow is not worth anything. With NOL
    tracking off the two are the same number.

    ``shares_outstanding`` is the count at the END of the year, after that year's
    stock compensation has been issued.
    """

    year: int
    revenue: float
    growth: float
    ebit: float
    ebit_margin: float
    taxes: float
    nopat: float
    da: float
    capex: float
    delta_nwc: float
    sbc: float
    fcff: float
    discount_factor: float
    pv: float
    cash_taxes: float = 0.0
    nol_opening: float = 0.0
    nol_used: float = 0.0
    nol_closing: float = 0.0
    sbc_shares_issued: float = 0.0
    shares_outstanding: float = 0.0

    @property
    def ebitda(self) -> float:
        """Post-rent, per ASC 842, matching the comps denominator."""
        return self.ebit + self.da


@dataclass
class TerminalValue:
    """A terminal value and the cross-check that falls out of it."""

    method: str
    value: float
    pv: float
    implied_exit_multiple: float | None
    implied_growth: float | None
    pct_of_ev: float
    note: str


@dataclass
class DCFResult:
    """All three terminal methods carried side by side, never blended.

    Averaging a Gordon value and an exit-multiple value produces a number no one
    can defend in a meeting, so each is reported separately and the checks say
    what each implies about the others. Lines in ``checks`` that begin with
    ``FLAG:`` failed a threshold; the rest are stated for the record.

    ``enterprise_value``, ``equity_value`` and ``per_share`` are the headline
    figures, and they repeat whichever method ``dcf.terminal.method`` selected.
    The per-method fields keep their own meanings whatever that is set to.

    ``shares_for_value`` is the count the per-share figures divide by: the
    diluted share count at the end of the explicit period, which equals the
    starting count unless SBC dilution is being modelled.
    """

    projections: list[ProjectionYear]
    terminal_gordon: TerminalValue
    terminal_exit: TerminalValue | None
    wacc: float
    enterprise_value_gordon: float
    enterprise_value_exit: float | None
    equity_value_gordon: float
    equity_value_exit: float | None
    per_share_gordon: float
    per_share_exit: float | None
    terminal_value_driver: TerminalValue | None = None
    enterprise_value_value_driver: float | None = None
    equity_value_value_driver: float | None = None
    per_share_value_driver: float | None = None
    headline_method: str = "gordon"
    enterprise_value: float = float("nan")
    equity_value: float = float("nan")
    per_share: float = float("nan")
    shares_for_value: float = float("nan")
    checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """The projection as raw numbers. Formatting belongs to the renderer.

        The carryforward and share-count blocks appear only when they carry
        information. A column of zeros beside twelve live ones is not disclosure,
        it is clutter, and on a transposed terminal table it costs five rows.
        """
        show_nol = any(p.nol_opening or p.nol_used for p in self.projections)
        show_dilution = any(p.sbc_shares_issued for p in self.projections)

        records = []
        for p in self.projections:
            row = {
                "Year": p.year,
                "Revenue": p.revenue,
                "Growth": p.growth,
                "EBIT": p.ebit,
                "EBIT margin": p.ebit_margin,
                "Taxes": p.taxes,
            }
            if show_nol:
                row |= {
                    "NOL opening": p.nol_opening,
                    "NOL used": p.nol_used,
                    "NOL closing": p.nol_closing,
                    "Cash taxes": p.cash_taxes,
                }
            row |= {
                "NOPAT": p.nopat,
                "D&A": p.da,
                "Capex": p.capex,
                "Change in NWC": p.delta_nwc,
                "SBC": p.sbc,
            }
            if show_dilution:
                row |= {
                    "SBC shares issued": p.sbc_shares_issued,
                    "Shares outstanding": p.shares_outstanding,
                }
            row |= {
                "FCFF": p.fcff,
                "Discount factor": p.discount_factor,
                "PV": p.pv,
            }
            records.append(row)
        return pd.DataFrame(records).set_index("Year")


# -- the projection ------------------------------------------------------- #


def _fade(start: float, end: float, n: int) -> np.ndarray:
    """Year 1 at ``start``, year n at ``end``, straight line between."""
    if n == 1:
        # With a single explicit year that year is also the terminal year, so
        # the terminal assumption is the one that has to hold.
        return np.array([end])
    return np.linspace(start, end, n)


def _tax_rate(fin: Financials, assumptions: Assumptions) -> tuple[float, str | None]:
    marginal = assumptions.tax.marginal_tax_rate
    if not assumptions.tax.use_effective_rate:
        return marginal, None
    effective = fin.effective_tax_rate
    if effective is None:
        return marginal, (
            "The assumptions ask for the filed effective tax rate, but this filer's "
            "rate is not economically meaningful: it reflects valuation allowances "
            f"and discrete items rather than the burden on an incremental dollar. The "
            f"marginal rate of {marginal:.1%} is used instead."
        )
    return effective, (
        f"NOPAT is taxed at the filed effective rate of {effective:.1%} rather than "
        f"the marginal {marginal:.1%}, at the request of the assumptions file."
    )


def _sbc_ratio(fin: Financials, assumptions: Assumptions) -> tuple[float, str | None]:
    if fin.sbc is not None:
        return fin.sbc / fin.revenue, None
    if assumptions.dcf.sbc_treatment == "addback":
        raise MissingDataError(
            "stock-based compensation",
            ticker=fin.ticker,
            hint=(
                "sbc_treatment is 'addback', which cannot proceed without an SBC "
                "figure to add back. Set sbc_treatment to 'expense' to leave the "
                "charge inside EBIT where the filer already put it."
            ),
        )
    return 0.0, (
        "No SBC figure was resolved, so the SBC column is zero. The charge is still "
        "inside EBIT and still borne by free cash flow; only the disclosure is absent."
    )


def _opening_nol(fin: Financials, assumptions: Assumptions) -> tuple[float, str]:
    """The carryforward balance the projection starts from, and where it came from.

    The assumptions file wins when it carries a number, because the gross federal
    carryforward usually lives in the tax footnote as prose and an analyst reading
    that footnote knows more than any tag does. Failing that the filer's own
    ``OperatingLossCarryforwards`` instant is used if it is there.

    Two traps. The first is the tag next door: ``DeferredTaxAssets``-prefixed
    carryforward tags are the balance already multiplied by a tax rate, so taking
    one as a gross NOL understates the shelter by roughly three quarters. Only the
    gross tag is read here. The second is dimensions. Filers commonly tag this
    concept only by jurisdiction, federal against state against foreign, and the
    companyfacts endpoint publishes consolidated facts, so the concept can be
    absent from the fact set while sitting in plain sight in the 10-K. That is a
    reason to raise and ask for the number, not to guess at one.

    ``Financials`` carries no handle back to the fact set, so a caller that has
    one attaches it as ``fin.facts`` and the filings path opens. With nothing
    attached the assumption is the only source, which is the common case.
    """
    cfg = assumptions.dcf.nol
    if cfg.opening_balance is not None:
        return cfg.opening_balance, (
            f"NOL carryforwards open at {cfg.opening_balance:,.0f}mm, from the "
            "assumptions file rather than from a tag."
        )

    facts = getattr(fin, "facts", None)
    if facts is not None:
        # The carryforward is a tax-footnote disclosure made once a year, so the
        # newest instant can be almost a full year behind the balance-sheet date
        # the rest of the model runs on. Refusing it for staleness would reject
        # every filer; carrying it and saying how old it is, is the honest trade.
        value, prov = facts.resolve_instant(
            "operating loss carryforwards",
            _NOL_TAGS,
            fin.as_of,
            tolerance_days=_NOL_TOLERANCE_DAYS,
            required=False,
        )
        if value is not None:
            balance = value / _MM
            period = prov.periods[0] if prov.periods else "an undated instant"
            return balance, (
                f"NOL carryforwards open at {balance:,.0f}mm, read from "
                f"{prov.tag} as of {period}. That is a fiscal year end "
                "disclosure, so it is carried forward to the valuation date "
                "without the losses or usage of the intervening quarters."
            )

    raise ConfigError(
        f"dcf.nol.track is on, but no opening carryforward balance could be "
        f"sourced for {fin.ticker}. Filers commonly tag OperatingLossCarryforwards "
        "only by jurisdiction, and dimensioned facts do not reach the companyfacts "
        "endpoint, so the balance has to come from the tax footnote by hand. Set "
        "dcf.nol.opening_balance in USD millions, or turn dcf.nol.track off."
    )


def project(
    fin: Financials,
    assumptions: Assumptions,
    price: float | None = None,
) -> list[ProjectionYear]:
    """Build the explicit forecast.

    Discount factors are stamped only when the assumptions already fix a WACC.
    Otherwise they come back as NaN and ``run_dcf`` fills them once the cost of
    capital is known. An undiscounted flow carrying a factor of 1.0 looks exactly
    like a discounted one on a printed page, which is why it is not done.

    ``price`` is the share price new stock compensation is issued at. It is
    optional because a projection is a projection whether or not anyone has
    priced the equity, and without it the share count is held flat and the issued
    column reads zero, which is visible rather than silent. ``run_dcf`` always
    passes the bridge's price, so the valuation path never runs blind.
    """
    cfg = assumptions.dcf
    if fin.revenue <= 0:
        raise NotMeaningfulError(
            f"{fin.ticker} reports {fin.revenue:,.1f}mm of trailing revenue. Every "
            "assumption in the projection is a percentage of revenue, so a "
            "revenue-driven forecast cannot be started from a zero or negative base."
        )

    n = cfg.projection_years
    tax_rate, _ = _tax_rate(fin, assumptions)
    sbc_ratio, _ = _sbc_ratio(fin, assumptions)
    addback_sbc = cfg.sbc_treatment == "addback"
    dilute = addback_sbc and cfg.sbc_dilution and price is not None
    if dilute and price <= 0:
        raise ConfigError(
            f"SBC dilution divides the charge by a share price, and {price:,.2f} "
            "cannot issue shares. Supply a positive price or set "
            "dcf.sbc_dilution to false and read the per-share figure as overstated."
        )
    nol_balance = _opening_nol(fin, assumptions)[0] if cfg.nol.track else 0.0

    growth = _fade(cfg.revenue_growth_start, cfg.revenue_growth_terminal, n)
    margin_start = (
        cfg.ebit_margin_start if cfg.ebit_margin_start is not None else fin.ebit_margin
    )
    margin = _fade(margin_start, cfg.ebit_margin_terminal, n)

    factors = (
        discount_factors(n, cfg.wacc_override, cfg.mid_year_convention)
        if cfg.wacc_override is not None
        else np.full(n, np.nan)
    )

    rows: list[ProjectionYear] = []
    revenue = fin.revenue
    nwc_prev = cfg.nwc_pct_revenue * fin.revenue
    shares = fin.diluted_shares
    for i in range(n):
        revenue = revenue * (1.0 + float(growth[i]))
        ebit = revenue * float(margin[i])
        # No tax benefit is taken on a loss year. A loss creates a carryforward,
        # not a refund, so crediting one here would book cash the company will
        # not receive whether or not the balance is being tracked.
        taxes = tax_rate * max(ebit, 0.0)
        nol_opening = nol_balance
        nol_used = 0.0
        if cfg.nol.track:
            if ebit < 0:
                # The loss itself joins the balance. It shelters nothing this
                # year because there is nothing to shelter.
                nol_balance = nol_opening - ebit
            else:
                nol_used = min(nol_opening, cfg.nol.annual_limitation_pct * ebit)
                nol_balance = nol_opening - nol_used
            cash_taxes = tax_rate * max(ebit - nol_used, 0.0)
        else:
            cash_taxes = taxes
        nopat = ebit - cash_taxes
        da = cfg.da_pct_revenue * revenue
        capex = cfg.capex_pct_revenue * revenue
        nwc = cfg.nwc_pct_revenue * revenue
        delta_nwc = nwc - nwc_prev
        nwc_prev = nwc
        sbc = sbc_ratio * revenue
        fcff = nopat + da - capex - delta_nwc + (sbc if addback_sbc else 0.0)
        issued = sbc / price if dilute else 0.0
        shares += issued
        rows.append(
            ProjectionYear(
                year=i + 1,
                revenue=revenue,
                growth=float(growth[i]),
                ebit=ebit,
                ebit_margin=float(margin[i]),
                taxes=taxes,
                nopat=nopat,
                da=da,
                capex=capex,
                delta_nwc=delta_nwc,
                sbc=sbc,
                fcff=fcff,
                discount_factor=float(factors[i]),
                pv=fcff * float(factors[i]),
                cash_taxes=cash_taxes,
                nol_opening=nol_opening,
                nol_used=nol_used,
                nol_closing=nol_balance,
                sbc_shares_issued=issued,
                shares_outstanding=shares,
            )
        )
    return rows


# -- discounting ---------------------------------------------------------- #


def discount_factors(n: int, wacc: float, mid_year: bool = True) -> np.ndarray:
    """Factors for years 1..n, at t-0.5 under the mid-year convention."""
    if wacc <= -1.0:
        raise ConfigError(f"a WACC of {wacc:.2%} cannot discount anything.")
    t = np.arange(1, n + 1, dtype=float)
    if mid_year:
        t = t - 0.5
    return (1.0 + wacc) ** (-t)


def gordon_terminal_value(fcff_n: float, wacc: float, g: float) -> float:
    """Perpetuity of end-of-year flows, valued as of the end of year N."""
    if g >= wacc:
        raise ConfigError(
            f"terminal growth of {g:.2%} is not below the WACC of {wacc:.2%}. The "
            "Gordon formula divides by (WACC - g), so growth at or above the "
            "discount rate prices the perpetuity as infinite or negative. Lower "
            "dcf.terminal_growth or revisit the cost of capital, but do not run this."
        )
    return fcff_n * (1.0 + g) / (wacc - g)


def exit_multiple_terminal_value(ebitda_n: float, multiple: float) -> float:
    """Terminal EBITDA at an observed multiple. A price, not a flow."""
    return ebitda_n * multiple


def value_driver_terminal_value(
    nopat_next: float, wacc: float, g: float, roic: float
) -> float:
    """Koller's value-driver formula, valued as of the end of year N.

        TV_N = NOPAT_{N+1} * (1 - g / ROIC) / (WACC - g)

    Growth of ``g`` at a return of ``ROIC`` costs ``g / ROIC`` of every dollar of
    NOPAT, so the bracket is the share left for investors. Reinvestment is derived
    from the return rather than assumed beside it, which is the whole point: the
    Gordon path lets an analyst assume capex, working capital and growth
    independently and only discover the implied return afterwards, if at all.

    At ``ROIC == WACC`` the expression reduces to ``NOPAT / WACC`` for every g,
    because the bracket and the denominator cancel. That is the competitive
    equilibrium result stated as plainly as it can be: a business earning exactly
    its cost of capital is worth the same whatever its growth rate, since the
    reinvestment growth demands is worth precisely what it costs.
    """
    if roic <= 0:
        raise ConfigError(
            f"terminal ROIC of {roic:.2%} is not positive, so the reinvestment the "
            "value-driver formula derives is meaningless: a negative return makes "
            "(1 - g/ROIC) exceed one and growth would raise the terminal value "
            "rather than cost anything. Set dcf.terminal.terminal_roic above zero, "
            "or leave it null to fall back to the WACC."
        )
    if g >= wacc:
        raise ConfigError(
            f"terminal growth of {g:.2%} is not below the WACC of {wacc:.2%}, so "
            "the value-driver perpetuity cannot be formed either."
        )
    return nopat_next * (1.0 - g / roic) / (wacc - g)


def _pv_terminal_gordon(tv: float, wacc: float, n: int, mid_year: bool) -> float:
    # The perpetuity's own flows arrive mid-year under this convention, so the
    # value dated end of year N is carried back only N-0.5 years.
    exponent = (n - 0.5) if mid_year else float(n)
    return tv * (1.0 + wacc) ** (-exponent)


def _pv_terminal_exit(tv: float, wacc: float, n: int) -> float:
    # No mid-year adjustment: a multiple is a price struck at the end of year N.
    return tv * (1.0 + wacc) ** (-float(n))


def _implied_growth(tv_end_year: float, fcff_n: float, wacc: float) -> float | None:
    """Solve TV = FCFF_N*(1+g)/(w-g) for g."""
    denominator = tv_end_year + fcff_n
    if denominator == 0:
        return None
    return (tv_end_year * wacc - fcff_n) / denominator


# -- the first perpetuity year -------------------------------------------- #


@dataclass
class _SteadyState:
    """Year N+1 built to be consistent with g, which year N is not.

    The last explicit year grows at ``revenue_growth_terminal`` and its capital
    spending and working-capital swing are sized for that growth. The perpetuity
    grows at ``g``. Reading reinvestment, return on capital or a terminal NOPAT
    off year N therefore mixes two growth rates. Every terminal calculation that
    needs a steady state takes it from here instead.
    """

    revenue: float
    ebit: float
    nopat: float
    da: float
    capex: float
    delta_nwc: float
    sbc: float

    @property
    def reinvestment(self) -> float:
        """Capital absorbed net of what depreciation replaces."""
        return self.capex + self.delta_nwc - self.da

    @property
    def fcff(self) -> float:
        return self.nopat + self.da - self.capex - self.delta_nwc + self.sbc

    @property
    def nopat_for_value(self) -> float:
        """NOPAT on the same SBC convention the explicit years were built on.

        Under the addback the projected flows carry stock compensation as a
        non-cash charge restored, so a terminal NOPAT struck without it would
        switch conventions between year N and year N+1 and the two terminal
        methods would stop being comparable.
        """
        return self.nopat + self.sbc


def _steady_state(
    terminal: ProjectionYear,
    g: float,
    assumptions: Assumptions,
    tax_rate: float,
) -> _SteadyState:
    """One notch of g past the terminal year, at terminal margins and intensity.

    Taxes are struck at the full rate even when a carryforward is being tracked.
    A carryforward is a finite asset: whatever survives year N shelters a few
    more years and is gone, so capitalising a sheltered tax rate in perpetuity
    would value a shield that does not exist.
    """
    cfg = assumptions.dcf
    revenue = terminal.revenue * (1.0 + g)
    ebit = cfg.ebit_margin_terminal * revenue
    sbc = (
        (terminal.sbc / terminal.revenue) * revenue
        if cfg.sbc_treatment == "addback" and terminal.revenue
        else 0.0
    )
    return _SteadyState(
        revenue=revenue,
        ebit=ebit,
        nopat=ebit - tax_rate * max(ebit, 0.0),
        da=cfg.da_pct_revenue * revenue,
        capex=cfg.capex_pct_revenue * revenue,
        # Working capital scales with revenue, so in steady state its change is
        # the NWC ratio applied to one year of growth, not to the terminal
        # year's faster revenue step.
        delta_nwc=cfg.nwc_pct_revenue * terminal.revenue * g,
        sbc=sbc,
    )


# -- the run -------------------------------------------------------------- #


def run_dcf(
    fin: Financials,
    bridge: EVBridge,
    wacc_result: "WACCResult",
    assumptions: Assumptions,
    peer_median_ev_ebitda: float | None = None,
) -> DCFResult:
    """Value the enterprise on both terminal methods and cross-check them."""
    cfg = assumptions.dcf
    notes: list[str] = []
    checks: list[str] = []

    if fin.diluted_shares <= 0:
        raise MissingDataError(
            "diluted shares outstanding",
            ticker=fin.ticker,
            hint="a per-share value cannot be formed without a share count.",
        )

    wacc = cfg.wacc_override if cfg.wacc_override is not None else wacc_result.wacc
    if cfg.wacc_override is not None:
        notes.append(
            f"WACC is overridden at {wacc:.2%}. The computed cost of capital of "
            f"{wacc_result.wacc:.2%} is reported but not used."
        )
    g = cfg.terminal_growth
    if g >= wacc:
        raise ConfigError(
            f"terminal growth of {g:.2%} is not below the WACC of {wacc:.2%}. Set "
            "dcf.terminal_growth below the cost of capital, or explain why this "
            "business compounds faster forever than the rate its investors demand."
        )

    tax_rate, tax_note = _tax_rate(fin, assumptions)
    if tax_note:
        notes.append(tax_note)
    _, sbc_note = _sbc_ratio(fin, assumptions)
    if sbc_note:
        notes.append(sbc_note)
    if cfg.sbc_treatment == "addback" and cfg.sbc_dilution:
        notes.append(
            "SBC is added back as a non-cash charge, and the shares that pay for "
            f"it are issued: each year's charge divided by the {bridge.price:,.2f} "
            "share price, at that price throughout, because a forecast share price "
            "would make the count depend on the answer. Per-share value divides by "
            "the count at the end of the explicit period, since every share the "
            "addback pays for has been issued by then."
        )
    elif cfg.sbc_treatment == "addback":
        notes.append(
            "SBC is added back as a non-cash charge. GAAP EBIT was already net of "
            "it, so this raises free cash flow by the full amount. The share count "
            "is held flat, so the dilution that pays for the charge is not in the "
            "per-share result and value per share is overstated. Set "
            "dcf.sbc_dilution to true to close that loop."
        )
    else:
        notes.append(
            "SBC is left inside EBIT, where the filer charged it. Nothing is added "
            "back, because nothing was taken out."
        )
    if cfg.nwc_pct_revenue < 0:
        reported = (
            f"the reported balance of {fin.net_working_capital:,.0f}mm"
            if fin.net_working_capital is not None
            else "the reported balance sheet"
        )
        notes.append(
            f"Working capital is carried at {cfg.nwc_pct_revenue:.1%} of revenue, so "
            "growth releases cash: customers are billed ahead of delivery. The year "
            "one change is measured against that same ratio applied to TTM revenue, "
            f"not against {reported}, which keeps the path free of a one-time step "
            "that was never an operating event."
        )

    if cfg.nol.track:
        notes.append(_opening_nol(fin, assumptions)[1])

    projections = project(fin, assumptions, price=bridge.price)
    n = len(projections)
    fcff = np.array([p.fcff for p in projections])
    factors = discount_factors(n, wacc, cfg.mid_year_convention)
    for p, f in zip(projections, factors):
        p.discount_factor = float(f)
        p.pv = p.fcff * float(f)
    pv_explicit = float(np.sum(fcff * factors))

    terminal = projections[-1]
    terminal_ebitda = terminal.ebitda
    # The register the terminal value belongs to, which is the starting count
    # unless dilution is being modelled.
    shares = terminal.shares_outstanding
    steady = _steady_state(terminal, g, assumptions, tax_rate)

    # -- Gordon ------------------------------------------------------------ #
    tv_gordon = gordon_terminal_value(terminal.fcff, wacc, g)
    pv_gordon = _pv_terminal_gordon(tv_gordon, wacc, n, cfg.mid_year_convention)
    ev_gordon = pv_explicit + pv_gordon
    # Restated onto the exit method's clock so the two cross-checks convert
    # between conventions the same way. Under mid-year discounting the Gordon
    # value carries a half-year uplift the exit method does not, so the multiple
    # that reproduces Gordon's PRESENT VALUE under the exit convention is the
    # end-of-year-N Gordon value grossed up by (1+w)^0.5. Quoting the raw ratio
    # instead would understate the implied multiple by half a year of WACC and
    # feed an inconsistency into the comparison against the peer median.
    _clock = (1 + wacc) ** 0.5 if cfg.mid_year_convention else 1.0
    implied_multiple = (
        tv_gordon * _clock / terminal_ebitda if terminal_ebitda > 0 else None
    )
    terminal_gordon = TerminalValue(
        method="gordon",
        value=tv_gordon,
        pv=pv_gordon,
        implied_exit_multiple=implied_multiple,
        implied_growth=g,
        pct_of_ev=pv_gordon / ev_gordon if ev_gordon else float("nan"),
        note=(
            f"Perpetuity of end-of-year flows growing at {g:.2%}, valued at the end "
            f"of year {n} and discounted at (1+w)^-{n - 0.5:g} because the mid-year "
            "convention places those flows half a year earlier."
            if cfg.mid_year_convention
            else (
                f"Perpetuity of end-of-year flows growing at {g:.2%}, valued at the "
                f"end of year {n} and discounted at (1+w)^-{n}."
            )
        ),
    )

    # -- exit multiple ----------------------------------------------------- #
    multiple = cfg.exit_multiple if cfg.exit_multiple is not None else peer_median_ev_ebitda
    terminal_exit: TerminalValue | None = None
    ev_exit: float | None = None
    if multiple is None:
        notes.append(
            "No exit multiple is available, so only the Gordon value is reported. "
            "Set dcf.exit_multiple or run the comps to supply a peer median. A "
            "multiple is not invented here."
        )
    elif terminal_ebitda <= 0:
        notes.append(
            f"Terminal EBITDA of {terminal_ebitda:,.0f}mm is not positive, so an "
            "EV/EBITDA exit multiple is arithmetic without meaning and the exit "
            "method is omitted."
        )
    else:
        tv_exit = exit_multiple_terminal_value(terminal_ebitda, multiple)
        pv_exit = _pv_terminal_exit(tv_exit, wacc, n)
        ev_exit = pv_explicit + pv_exit
        # Compare like with like: restate the exit terminal value as the
        # end-of-year-N Gordon value carrying the same present value, then solve
        # for g. Without this step the implied growth silently absorbs the
        # half-year timing difference between the two methods.
        equivalent = pv_exit * (1.0 + wacc) ** (
            (n - 0.5) if cfg.mid_year_convention else float(n)
        )
        terminal_exit = TerminalValue(
            method="exit_multiple",
            value=tv_exit,
            pv=pv_exit,
            implied_exit_multiple=multiple,
            implied_growth=_implied_growth(equivalent, terminal.fcff, wacc),
            pct_of_ev=pv_exit / ev_exit if ev_exit else float("nan"),
            note=(
                f"{multiple:.1f}x terminal EBITDA of {terminal_ebitda:,.0f}mm, "
                f"discounted at (1+w)^-{n} with no mid-year adjustment: the multiple "
                f"is a price observed at the end of year {n}, not a flow received "
                "through it."
            ),
        )
        if cfg.exit_multiple is None:
            notes.append(
                f"The exit multiple of {multiple:.1f}x is the peer median from the "
                "comp set, applied to a terminal EBITDA built on the same post-rent "
                "definition the comps are quoted on."
            )

    # -- value driver ------------------------------------------------------ #
    roic = wacc if cfg.terminal.terminal_roic is None else cfg.terminal.terminal_roic
    terminal_vd: TerminalValue | None = None
    ev_vd: float | None = None
    if steady.nopat_for_value <= 0:
        notes.append(
            f"Steady-state NOPAT of {steady.nopat_for_value:,.0f}mm is not "
            "positive, so the value-driver terminal value would capitalise a loss "
            "and is omitted. A terminal margin that never turns a profit is the "
            "assumption to revisit."
        )
    else:
        if cfg.terminal.terminal_roic is None:
            notes.append(
                f"The value-driver terminal value uses a terminal ROIC of "
                f"{wacc:.2%}, the WACC, because dcf.terminal.terminal_roic is "
                "null. That is the competitive-equilibrium assumption: growth in "
                "perpetuity creates no value, and the terminal value is "
                "NOPAT/WACC whatever g is set to."
            )
        tv_vd = value_driver_terminal_value(steady.nopat_for_value, wacc, g, roic)
        # Same clock as Gordon: this is a perpetuity of flows, not a price struck
        # at a date, so under the mid-year convention it carries the half-year
        # uplift an exit multiple must not.
        pv_vd = _pv_terminal_gordon(tv_vd, wacc, n, cfg.mid_year_convention)
        ev_vd = pv_explicit + pv_vd
        terminal_vd = TerminalValue(
            method="value_driver",
            value=tv_vd,
            pv=pv_vd,
            implied_exit_multiple=(
                tv_vd * _clock / terminal_ebitda if terminal_ebitda > 0 else None
            ),
            implied_growth=g,
            pct_of_ev=pv_vd / ev_vd if ev_vd else float("nan"),
            note=(
                f"Steady-state NOPAT of {steady.nopat_for_value:,.0f}mm growing at "
                f"{g:.2%} at a {roic:.1%} return on capital, so "
                f"{g / roic:.1%} of NOPAT is reinvested to fund the growth and the "
                f"rest is capitalised. Discounted on the Gordon clock."
            ),
        )

    # The walk back to equity never subtracts operating leases: the projected
    # flows pay rent every year, so the obligation is serviced inside the DCF
    # and taking the liability out as debt would charge it twice.
    equity_gordon = equity_value_from_dcf_ev(ev_gordon, fin, bridge)
    equity_exit = (
        None if ev_exit is None else equity_value_from_dcf_ev(ev_exit, fin, bridge)
    )
    equity_vd = None if ev_vd is None else equity_value_from_dcf_ev(ev_vd, fin, bridge)

    headline = cfg.terminal.method
    if headline == "value_driver" and ev_vd is None:
        headline = "gordon"
        notes.append(
            "dcf.terminal.method asks for the value-driver terminal value, which "
            "could not be formed, so the headline figures are the Gordon ones. "
            "They are not the same construction and should not be read as such."
        )
    ev_headline = ev_vd if headline == "value_driver" else ev_gordon
    equity_headline = equity_vd if headline == "value_driver" else equity_gordon

    checks.extend(
        _cross_checks(
            fin=fin,
            bridge=bridge,
            assumptions=assumptions,
            wacc=wacc,
            wacc_result=wacc_result,
            g=g,
            tax_rate=tax_rate,
            projections=projections,
            factors=factors,
            steady=steady,
            price=bridge.price,
            shares=shares,
            roic=roic,
            terminal_ebitda=terminal_ebitda,
            terminal_gordon=terminal_gordon,
            terminal_exit=terminal_exit,
            terminal_vd=terminal_vd,
            peer_median_ev_ebitda=peer_median_ev_ebitda,
        )
    )

    return DCFResult(
        projections=projections,
        terminal_gordon=terminal_gordon,
        terminal_exit=terminal_exit,
        wacc=wacc,
        enterprise_value_gordon=ev_gordon,
        enterprise_value_exit=ev_exit,
        equity_value_gordon=equity_gordon,
        equity_value_exit=equity_exit,
        per_share_gordon=equity_gordon / shares,
        per_share_exit=None if equity_exit is None else equity_exit / shares,
        terminal_value_driver=terminal_vd,
        enterprise_value_value_driver=ev_vd,
        equity_value_value_driver=equity_vd,
        per_share_value_driver=None if equity_vd is None else equity_vd / shares,
        headline_method=headline,
        enterprise_value=ev_headline,
        equity_value=equity_headline,
        per_share=equity_headline / shares,
        shares_for_value=shares,
        checks=checks,
        notes=notes,
    )


def _cross_checks(
    *,
    fin: Financials,
    bridge: EVBridge,
    assumptions: Assumptions,
    wacc: float,
    wacc_result: "WACCResult",
    g: float,
    tax_rate: float,
    projections: list[ProjectionYear],
    factors: np.ndarray,
    steady: _SteadyState,
    price: float,
    shares: float,
    roic: float,
    terminal_ebitda: float,
    terminal_gordon: TerminalValue,
    terminal_exit: TerminalValue | None,
    terminal_vd: TerminalValue | None,
    peer_median_ev_ebitda: float | None,
) -> list[str]:
    """What each terminal assumption implies about the others, and about returns."""
    out: list[str] = []
    terminal = projections[-1]

    # The rate the CAPM build actually used, which is what g has to be judged
    # against. The assumptions file leaves it null when it is pulled live.
    risk_free = getattr(wacc_result, "risk_free_rate", None)
    if risk_free is None:
        risk_free = assumptions.market.risk_free_rate
    if risk_free is not None and g > risk_free:
        out.append(
            f"FLAG: terminal growth of {g:.2%} exceeds the risk-free rate of "
            f"{risk_free:.2%}. The long bond yield embeds expected nominal growth "
            "plus a term premium, so a firm compounding above it in perpetuity "
            "eventually becomes the economy. Defensible only with an explicit case."
        )
    elif risk_free is not None:
        out.append(
            f"Terminal growth of {g:.2%} sits below the {risk_free:.2%} risk-free "
            "rate, so the perpetuity does not outgrow the long bond."
        )

    if terminal_gordon.implied_exit_multiple is not None:
        line = (
            f"The Gordon terminal value implies an exit multiple of "
            f"{terminal_gordon.implied_exit_multiple:.1f}x terminal EBITDA of "
            f"{terminal_ebitda:,.0f}mm, restated onto the exit method's "
            "whole-period discount clock so the comparison is like for like."
        )
        if peer_median_ev_ebitda:
            gap = terminal_gordon.implied_exit_multiple / peer_median_ev_ebitda - 1.0
            line += (
                f" The peer median today is {peer_median_ev_ebitda:.1f}x, a gap of "
                f"{gap:+.0%}."
            )
        out.append(line)

    if terminal_exit is not None and terminal_exit.implied_growth is not None:
        implied_g = terminal_exit.implied_growth
        line = (
            f"The {terminal_exit.implied_exit_multiple:.1f}x exit multiple implies "
            f"perpetuity growth of {implied_g:.2%} against a WACC of {wacc:.2%}."
        )
        if implied_g >= wacc:
            line = "FLAG: " + line + (
                " That is at or above the discount rate, so the multiple prices a "
                "perpetuity the Gordon formula cannot even express."
            )
        elif implied_g < 0:
            line += (
                " A negative implied growth rate says the multiple is cheaper than a "
                "no-growth perpetuity on the terminal year's cash flow."
            )
        out.append(line)

    for tv in (terminal_gordon, terminal_exit, terminal_vd):
        if tv is None:
            continue
        label = _TV_LABELS[tv.method]
        if tv.pct_of_ev > _TV_SHARE_FLAG:
            out.append(
                f"FLAG: {label} terminal value is {tv.pct_of_ev:.0%} of enterprise "
                f"value, above the {_TV_SHARE_FLAG:.0%} mark. The valuation is a bet "
                "on the terminal assumption, not on the forecast. Extending the "
                "explicit period until the business is mature moves the judgment "
                "into years that can be argued about."
            )
        else:
            out.append(
                f"{label} terminal value is {tv.pct_of_ev:.0%} of enterprise value."
            )

    out.extend(_reinvestment_check(terminal, steady, g, wacc, assumptions))
    out.extend(
        _value_driver_check(
            steady=steady,
            terminal_gordon=terminal_gordon,
            terminal_vd=terminal_vd,
            g=g,
            wacc=wacc,
            roic=roic,
            explicit_roic=assumptions.dcf.terminal.terminal_roic is not None,
        )
    )
    out.extend(_dilution_check(projections, assumptions, price, shares, g))
    out.extend(_nol_check(projections, factors, assumptions, tax_rate))

    if bridge.operating_lease_in_debt > 0 and terminal_exit is not None:
        out.append(
            "FLAG: operating leases are capitalised into debt, so the comp set is "
            "quoted on EBITDAR, before rent. The terminal EBITDA here is after rent "
            "and the projection does not model lease cost separately, so the exit "
            "multiple is being applied across two different denominators. Either "
            "turn off leases.capitalize_operating_leases or read the exit value as "
            "understated."
        )

    if fin.capex is not None and fin.revenue:
        realised_capex = fin.capex / fin.revenue
        assumed = assumptions.dcf.capex_pct_revenue
        if abs(realised_capex - assumed) > 0.02:
            out.append(
                f"Capex is assumed at {assumed:.1%} of revenue against a trailing "
                f"{realised_capex:.1%}. The gap is deliberate or it is an oversight; "
                "it is not neutral."
            )

    if assumptions.dcf.nol.track:
        out.append(
            f"NOPAT is struck on cash taxes at a {tax_rate:.1%} rate after the "
            "carryforward shelters what the limitation allows, so a loss year pays "
            "nothing and adds to the balance."
        )
    else:
        out.append(
            f"NOPAT is struck at a {tax_rate:.1%} tax rate, with no benefit taken on "
            "a loss year, since a loss creates a carryforward rather than a refund. "
            "Set dcf.nol.track to carry that carryforward forward."
        )
    return out


def _reinvestment_check(
    terminal: ProjectionYear,
    steady: _SteadyState,
    g: float,
    wacc: float,
    assumptions: Assumptions,
) -> list[str]:
    """g = ROIC * reinvestment rate, read out of a g-consistent steady state.

    The terminal assumption is a claim about returns whether or not the analyst
    states one, and this identity is the single best signal of whether it is
    coherent. The subtlety is which year to read it from. The final explicit
    year grows at ``revenue_growth_terminal``, so its capital spending and its
    working-capital swing are sized for that growth, not for the perpetuity's
    ``g``. Reading the reinvestment rate straight off that year mixes the two
    growth rates and misstates the implied return, usually upward, because the
    working-capital release that fast growth produces flatters the rate.

    So the check builds the first perpetuity year properly: revenue one notch of
    ``g`` beyond the terminal year, margins and capital intensity held at their
    terminal settings, and the working-capital change sized by ``g`` alone. That
    is the steady state the Gordon formula claims to capitalise, and it is the
    construction Damodaran recommends for exactly this reason.

    The same construction exposes a second, quieter issue. The terminal value is
    computed off ``FCFF_N * (1 + g)``, the standard shortcut, and that flow
    inherits the terminal year's reinvestment, which was sized for the faster
    growth. Where the shortcut flow and the steady-state flow disagree by more
    than a couple of percent, the check says so and by how much, because the gap
    is a bias in the terminal value itself, not a stylistic quibble.
    """
    nopat_next = steady.nopat
    if nopat_next <= 0:
        return [
            f"FLAG: steady-state NOPAT of {nopat_next:,.0f}mm at the terminal "
            "margin is not positive, so no reinvestment rate can be formed and "
            "the terminal growth rate is being capitalised without any implied "
            "return standing behind it."
        ]

    out: list[str] = []

    fcff_steady = steady.fcff
    fcff_shortcut = terminal.fcff * (1.0 + g)
    if fcff_shortcut:
        gap = fcff_shortcut / fcff_steady - 1.0 if fcff_steady else float("inf")
        if abs(gap) > 0.02:
            out.append(
                f"FLAG: the terminal value capitalises FCFF of "
                f"{fcff_shortcut:,.0f}mm, the final explicit year grown at "
                f"{g:.2%}, but a steady state at that growth supports "
                f"{fcff_steady:,.0f}mm. The shortcut flow inherits reinvestment "
                f"sized for {terminal.growth:.1%} growth, so the Gordon value is "
                f"{'overstated' if gap > 0 else 'understated'} by roughly "
                f"{abs(gap):.1%} before discounting."
            )

    reinvestment = steady.reinvestment
    rate = reinvestment / nopat_next
    head = (
        f"Steady-state reinvestment at {g:.2%} growth is {reinvestment:,.0f}mm "
        f"on {nopat_next:,.0f}mm of NOPAT, a reinvestment rate of {rate:.1%}."
    )
    if abs(rate) < 1e-6:
        out.append(
            "FLAG: " + head + f" Growing at {g:.2%} on no reinvestment implies "
            "an infinite return on capital. Either capex and working capital "
            "have to rise in the terminal state or the growth rate has to come "
            "down."
        )
        return out

    roic = g / rate
    if rate < 0:
        out.append(
            "FLAG: " + head + " It is negative, so the steady state releases "
            f"more capital than it absorbs while still growing at {g:.2%}. The "
            f"implied return on capital is {roic:.1%}, which is not a return at "
            "all: growth is being funded by disinvestment. Raise capex, or hold "
            "working capital flat as a share of revenue."
        )
    elif roic > _ROIC_CEILING:
        out.append(
            "FLAG: " + head + f" It implies a terminal ROIC of {roic:.1%}, "
            f"above the {_ROIC_CEILING:.0%} mark. A perpetual return that far "
            "above the cost of capital assumes no competitor ever arrives."
        )
    elif roic < wacc:
        out.append(
            "FLAG: " + head + f" It implies a terminal ROIC of {roic:.1%} "
            f"against a WACC of {wacc:.2%}. Growth at that return destroys "
            "value, yet the model capitalises it as though it were worth paying "
            "for. Either the growth is worth less than zero or the reinvestment "
            "assumption is too heavy."
        )
    else:
        out.append(
            head + f" It implies a terminal ROIC of {roic:.1%} against a WACC "
            f"of {wacc:.2%}, so terminal growth creates value and the "
            "assumption hangs together."
        )
    return out


def _value_driver_check(
    *,
    steady: _SteadyState,
    terminal_gordon: TerminalValue,
    terminal_vd: TerminalValue | None,
    g: float,
    wacc: float,
    roic: float,
    explicit_roic: bool,
) -> list[str]:
    """Reconcile the two perpetuities, which differ only in their reinvestment.

    Gordon capitalises a flow and lets the reinvestment inside it imply a return.
    The value driver states a return and derives the reinvestment. Written out,
    both are ``(NOPAT - reinvestment) / (WACC - g)`` on the first perpetuity
    year, so they are the same number whenever the two reinvestment figures
    agree, and the size of the disagreement is the size of the argument. Quoting
    both is the only way to see which of the two assumptions is doing the work.

    The reported Gordon value capitalises ``FCFF_N * (1 + g)`` rather than the
    steady-state flow, which is a second and separate gap already reported by
    the reinvestment check, so both Gordon figures appear here.
    """
    if terminal_vd is None:
        return []

    out: list[str] = []
    nopat = steady.nopat_for_value
    reinvestment_vd = nopat * g / roic
    tv_steady_gordon = steady.fcff / (wacc - g)

    if not explicit_roic:
        out.append(
            f"The value driver runs at ROIC equal to the {wacc:.2%} WACC, so its "
            f"terminal value is NOPAT/WACC, {nopat / wacc:,.0f}mm, and the "
            f"{g:.2%} growth rate changes it by nothing at all. Growth priced at "
            "the cost of capital is worth exactly what it costs."
        )
    elif roic > _ROIC_CEILING:
        out.append(
            f"FLAG: the value driver assumes a terminal ROIC of {roic:.1%}, above "
            f"the {_ROIC_CEILING:.0%} mark. Only {g / roic:.1%} of NOPAT has to be "
            "reinvested to fund the growth at that return, which is why the "
            "terminal value is as large as it is. A perpetual franchise that "
            "productive assumes no competitor ever arrives."
        )
    elif roic < wacc:
        out.append(
            f"FLAG: the value driver assumes a terminal ROIC of {roic:.1%} against "
            f"a WACC of {wacc:.2%}, so every dollar reinvested to grow returns "
            "less than it cost. Growth is destroying value, and raising g from "
            "here lowers the terminal value rather than raising it."
        )
    else:
        out.append(
            f"The value driver assumes a terminal ROIC of {roic:.1%} against a "
            f"WACC of {wacc:.2%}, funding {g:.2%} growth out of {g / roic:.1%} of "
            "NOPAT and capitalising the rest."
        )
    if g >= roic:
        out.append(
            f"FLAG: terminal growth of {g:.2%} is at or above the {roic:.1%} "
            "return that funds it, so the value-driver terminal value is zero or "
            "negative: the business would have to reinvest more than it earns to "
            "keep growing."
        )

    gap = reinvestment_vd - steady.reinvestment
    line = (
        f"Value-driver terminal value of {terminal_vd.value:,.0f}mm against "
        f"{tv_steady_gordon:,.0f}mm for Gordon on the same steady state. The two "
        f"are one statement whenever the reinvestment agrees: Gordon spends "
        f"{steady.reinvestment:,.0f}mm in the first perpetuity year, capex plus "
        f"the working-capital change less D&A, and the value driver derives "
        f"{reinvestment_vd:,.0f}mm, being {g:.2%} growth at a {roic:.1%} return on "
        f"{nopat:,.0f}mm of NOPAT. The value driver reinvests {abs(gap):,.0f}mm "
        f"{'more' if gap >= 0 else 'less'}, which is {abs(gap) / (wacc - g):,.0f}mm "
        f"{'off' if gap >= 0 else 'onto'} the terminal value."
    )
    if steady.sbc:
        line += (
            f" That NOPAT carries the SBC addback the projected flows carry, so it "
            f"sits above the {steady.nopat:,.0f}mm the reinvestment rate is "
            "measured against."
        )
    out.append(line)
    out.append(
        f"Gordon as reported is {terminal_gordon.value:,.0f}mm, struck on the "
        f"final explicit year grown at {g:.2%} rather than on that steady state. "
        "The difference between the two Gordon figures is the shortcut flow, not "
        "a difference of method."
    )
    return out


def _dilution_check(
    projections: list[ProjectionYear],
    assumptions: Assumptions,
    price: float,
    shares: float,
    g: float,
) -> list[str]:
    """What the addback costs the existing holder, stated as a share count.

    The explicit period's issuance is arithmetic. The line worth reading is the
    second one: the terminal year issues a fixed percentage of the company every
    year, and if that rate persists alongside the terminal growth rate then cash
    flow PER SHARE compounds at the difference, not at g. The terminal value
    capitalises the addback in perpetuity while the modelled share count stops
    growing at year N, so the per-share answer is still generous however the
    explicit years are handled.
    """
    cfg = assumptions.dcf
    if cfg.sbc_treatment != "addback" or not cfg.sbc_dilution:
        return []
    issued = sum(p.sbc_shares_issued for p in projections)
    if issued <= 0:
        return []

    opening = shares - issued
    out = [
        f"Stock compensation issues {issued:,.1f}mm shares over the "
        f"{len(projections)} explicit years at {price:,.2f}, taking the count from "
        f"{opening:,.1f}mm to {shares:,.1f}mm: cumulative dilution of "
        f"{issued / opening:.1%}, which the per-share figures divide by."
    ]

    rate = projections[-1].sbc / (price * shares)
    if rate >= _TERMINAL_DILUTION_FLAG:
        per_share_growth = (1.0 + g) / (1.0 + rate) - 1.0
        out.append(
            f"FLAG: the terminal year issues {rate:.2%} of the share count in "
            f"stock compensation. Held at that rate beside {g:.2%} terminal "
            f"growth, cash flow per share compounds at {per_share_growth:.2%}, "
            f"not {g:.2%}. The terminal value capitalises the addback forever and "
            "nothing beyond year N is modelled to pay for it, so the per-share "
            "value here is still the generous reading of this treatment."
        )
    return out


def _nol_check(
    projections: list[ProjectionYear],
    factors: np.ndarray,
    assumptions: Assumptions,
    tax_rate: float,
) -> list[str]:
    """What the carryforward is worth, and the three things it does not include.

    The shield is reported in present value because that is the only form in
    which it is comparable to anything else in the model. An NOL balance quoted
    gross is a headline; the tax it actually stops the company paying, discounted,
    is the part that belongs in a valuation.
    """
    cfg = assumptions.dcf.nol
    if not cfg.track:
        return []

    sheltered = sum(p.nol_used for p in projections)
    shielded = tax_rate * sheltered
    pv_shield = float(
        sum(tax_rate * p.nol_used * f for p, f in zip(projections, factors))
    )
    closing = projections[-1].nol_closing
    out = [
        f"The carryforward shelters {sheltered:,.0f}mm of taxable income across "
        f"the explicit period, worth {shielded:,.0f}mm of cash tax not paid and "
        f"{pv_shield:,.0f}mm discounted. That present value is what the NOL is "
        "worth to this valuation."
    ]

    paying = next(
        (p for p in projections if p.nol_used > 0 and p.cash_taxes > 0), None
    )
    if paying is not None:
        out.append(
            f"The {cfg.annual_limitation_pct:.0%} limitation binds in year "
            f"{paying.year}: cash tax of {paying.cash_taxes:,.0f}mm is still paid "
            f"against {paying.taxes:,.0f}mm at the full rate, on an opening "
            f"balance of {paying.nol_opening:,.0f}mm. Post-2017 federal losses "
            "carry forward indefinitely but shelter only part of taxable income, "
            "so a profitable year pays cash tax however large the carryforward."
        )
    if closing > 0:
        out.append(
            f"{closing:,.0f}mm of carryforward is unused at the end of year "
            f"{projections[-1].year}. The terminal value is struck at the full "
            f"{tax_rate:.1%} rate, so that balance shelters nothing in this "
            "valuation and the answer is conservative by whatever it is worth."
        )
    out.append(
        "Not modelled in the carryforward: section 382, which caps annual use "
        "after a change of control and would cut the shield in any deal case; "
        "state carryforwards, with their own expiry and apportionment; and the "
        "valuation allowance, so this is the gross federal balance rather than "
        "the deferred tax asset the filer expects to realise."
    )
    return out


# -- sensitivities -------------------------------------------------------- #


def _grid_axis(centre: float, step: float) -> np.ndarray:
    return centre + np.linspace(-step, step, _SENSITIVITY_POINTS)


def _per_share(ev: float, fin: Financials, bridge: EVBridge, shares: float) -> float:
    # The same denominator run_dcf uses, so a grid cell and the headline are the
    # same number when the axes cross at the base case.
    return equity_value_from_dcf_ev(ev, fin, bridge) / shares


def sensitivity_wacc_growth(
    fin: Financials,
    bridge: EVBridge,
    assumptions: Assumptions,
    wacc: float,
) -> pd.DataFrame:
    """Implied price per share, WACC down the side against terminal g across.

    Cells where g has reached the discount rate are left empty rather than
    filled with the large negative number the formula would return, because a
    printed number invites someone to read it.
    """
    cfg = assumptions.dcf
    projections = project(fin, assumptions, price=bridge.price)
    n = len(projections)
    fcff = np.array([p.fcff for p in projections])
    shares = projections[-1].shares_outstanding

    waccs = _grid_axis(wacc, cfg.sensitivity_wacc_step)
    growths = _grid_axis(cfg.terminal_growth, cfg.sensitivity_growth_step)
    grid = np.full((_SENSITIVITY_POINTS, _SENSITIVITY_POINTS), np.nan)

    for i, w in enumerate(waccs):
        pv_explicit = float(
            np.sum(fcff * discount_factors(n, float(w), cfg.mid_year_convention))
        )
        for j, gg in enumerate(growths):
            if gg >= w:
                continue
            tv = gordon_terminal_value(float(fcff[-1]), float(w), float(gg))
            ev = pv_explicit + _pv_terminal_gordon(
                tv, float(w), n, cfg.mid_year_convention
            )
            grid[i, j] = _per_share(ev, fin, bridge, shares)

    frame = pd.DataFrame(
        grid,
        index=[f"{w:.1%}" for w in waccs],
        columns=[f"{x:.1%}" for x in growths],
    )
    frame.index.name = "WACC"
    frame.columns.name = "Terminal growth"
    return frame


def sensitivity_wacc_exit(
    fin: Financials,
    bridge: EVBridge,
    assumptions: Assumptions,
    wacc: float,
    peer_median_ev_ebitda: float | None = None,
) -> pd.DataFrame:
    """Implied price per share, WACC down the side against the exit multiple.

    The multiple axis spans the centre multiple plus or minus a quarter, which is
    about the range a comp set's interquartile spread covers for software.
    """
    cfg = assumptions.dcf
    centre = cfg.exit_multiple if cfg.exit_multiple is not None else peer_median_ev_ebitda
    if centre is None:
        raise ConfigError(
            "an exit-multiple sensitivity needs a multiple to centre on. Set "
            "dcf.exit_multiple, or run the comps so a peer median EV/EBITDA can be "
            "passed in."
        )

    projections = project(fin, assumptions, price=bridge.price)
    n = len(projections)
    fcff = np.array([p.fcff for p in projections])
    shares = projections[-1].shares_outstanding
    terminal_ebitda = projections[-1].ebitda
    if terminal_ebitda <= 0:
        raise NotMeaningfulError(
            f"terminal EBITDA of {terminal_ebitda:,.0f}mm is not positive, so an "
            "EV/EBITDA exit multiple grid would price a negative denominator."
        )

    waccs = _grid_axis(wacc, cfg.sensitivity_wacc_step)
    multiples = centre * np.linspace(
        1.0 - _EXIT_MULTIPLE_SPREAD, 1.0 + _EXIT_MULTIPLE_SPREAD, _SENSITIVITY_POINTS
    )
    grid = np.empty((_SENSITIVITY_POINTS, _SENSITIVITY_POINTS))

    for i, w in enumerate(waccs):
        pv_explicit = float(
            np.sum(fcff * discount_factors(n, float(w), cfg.mid_year_convention))
        )
        for j, m in enumerate(multiples):
            tv = exit_multiple_terminal_value(terminal_ebitda, float(m))
            ev = pv_explicit + _pv_terminal_exit(tv, float(w), n)
            grid[i, j] = _per_share(ev, fin, bridge, shares)

    frame = pd.DataFrame(
        grid,
        index=[f"{w:.1%}" for w in waccs],
        columns=[f"{m:.1f}x" for m in multiples],
    )
    frame.index.name = "WACC"
    frame.columns.name = "Exit EV/EBITDA"
    return frame
