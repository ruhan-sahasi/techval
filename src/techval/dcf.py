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
holders giving up claim. This model projects cash flow, not the share count, so
the addback path overstates value per share and says so in the notes. When it is
on, the addback sits after tax next to D&A, since the award is deductible when
it vests.

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
from .ev_bridge import EVBridge, equity_value_from_ev
from .financials import Financials

if TYPE_CHECKING:  # the WACC build is a peer module, imported for typing only
    from .wacc import WACCResult

# Diagnostic thresholds, not valuation inputs. They change no number in the
# model, only whether a line in ``checks`` is flagged, so they stay out of the
# assumptions file where every entry moves a price.
_TV_SHARE_FLAG = 0.75
_ROIC_CEILING = 0.60
_SENSITIVITY_POINTS = 5
_EXIT_MULTIPLE_SPREAD = 0.25


@dataclass
class ProjectionYear:
    """One explicit forecast year, USD millions except the rates."""

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
    """Both terminal methods carried side by side, never blended.

    Averaging a Gordon value and an exit-multiple value produces a number no one
    can defend in a meeting, so the two are reported separately and the checks
    say what each implies about the other. Lines in ``checks`` that begin with
    ``FLAG:`` failed a threshold; the rest are stated for the record.
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
    checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """The projection as raw numbers. Formatting belongs to the renderer."""
        frame = pd.DataFrame(
            [
                {
                    "Year": p.year,
                    "Revenue": p.revenue,
                    "Growth": p.growth,
                    "EBIT": p.ebit,
                    "EBIT margin": p.ebit_margin,
                    "Taxes": p.taxes,
                    "NOPAT": p.nopat,
                    "D&A": p.da,
                    "Capex": p.capex,
                    "Change in NWC": p.delta_nwc,
                    "SBC": p.sbc,
                    "FCFF": p.fcff,
                    "Discount factor": p.discount_factor,
                    "PV": p.pv,
                }
                for p in self.projections
            ]
        )
        return frame.set_index("Year")


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


def project(fin: Financials, assumptions: Assumptions) -> list[ProjectionYear]:
    """Build the explicit forecast.

    Discount factors are stamped only when the assumptions already fix a WACC.
    Otherwise they come back as NaN and ``run_dcf`` fills them once the cost of
    capital is known. An undiscounted flow carrying a factor of 1.0 looks exactly
    like a discounted one on a printed page, which is why it is not done.
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
    for i in range(n):
        revenue = revenue * (1.0 + float(growth[i]))
        ebit = revenue * float(margin[i])
        # No tax benefit is taken on a loss year. A loss creates a carryforward,
        # not a refund, and this model does not track the NOL balance, so
        # crediting it here would book cash the company will not receive.
        taxes = tax_rate * max(ebit, 0.0)
        nopat = ebit - taxes
        da = cfg.da_pct_revenue * revenue
        capex = cfg.capex_pct_revenue * revenue
        nwc = cfg.nwc_pct_revenue * revenue
        delta_nwc = nwc - nwc_prev
        nwc_prev = nwc
        sbc = sbc_ratio * revenue
        fcff = nopat + da - capex - delta_nwc + (sbc if addback_sbc else 0.0)
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
    if cfg.sbc_treatment == "addback":
        notes.append(
            "SBC is added back as a non-cash charge. GAAP EBIT was already net of "
            "it, so this raises free cash flow by the full amount. The share count "
            "is held flat, so the dilution that pays for the charge is not in the "
            "per-share result and value per share is overstated."
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

    projections = project(fin, assumptions)
    n = len(projections)
    fcff = np.array([p.fcff for p in projections])
    factors = discount_factors(n, wacc, cfg.mid_year_convention)
    for p, f in zip(projections, factors):
        p.discount_factor = float(f)
        p.pv = p.fcff * float(f)
    pv_explicit = float(np.sum(fcff * factors))

    terminal = projections[-1]
    terminal_ebitda = terminal.ebitda

    # -- Gordon ------------------------------------------------------------ #
    tv_gordon = gordon_terminal_value(terminal.fcff, wacc, g)
    pv_gordon = _pv_terminal_gordon(tv_gordon, wacc, n, cfg.mid_year_convention)
    ev_gordon = pv_explicit + pv_gordon
    implied_multiple = tv_gordon / terminal_ebitda if terminal_ebitda > 0 else None
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

    equity_gordon = equity_value_from_ev(ev_gordon, fin, bridge)
    equity_exit = None if ev_exit is None else equity_value_from_ev(ev_exit, fin, bridge)

    checks.extend(
        _cross_checks(
            fin=fin,
            bridge=bridge,
            assumptions=assumptions,
            wacc=wacc,
            wacc_result=wacc_result,
            g=g,
            tax_rate=tax_rate,
            terminal=terminal,
            terminal_ebitda=terminal_ebitda,
            terminal_gordon=terminal_gordon,
            terminal_exit=terminal_exit,
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
        per_share_gordon=equity_gordon / fin.diluted_shares,
        per_share_exit=None if equity_exit is None else equity_exit / fin.diluted_shares,
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
    terminal: ProjectionYear,
    terminal_ebitda: float,
    terminal_gordon: TerminalValue,
    terminal_exit: TerminalValue | None,
    peer_median_ev_ebitda: float | None,
) -> list[str]:
    """What each terminal assumption implies about the other, and about returns."""
    out: list[str] = []

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
            f"{terminal_ebitda:,.0f}mm."
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

    for tv in (terminal_gordon, terminal_exit):
        if tv is None:
            continue
        label = "Gordon" if tv.method == "gordon" else "Exit multiple"
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

    out.extend(_reinvestment_check(terminal, g, wacc))

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

    out.append(
        f"NOPAT is struck at a {tax_rate:.1%} tax rate, with no benefit taken on a "
        "loss year, since a loss creates a carryforward rather than a refund."
    )
    return out


def _reinvestment_check(terminal: ProjectionYear, g: float, wacc: float) -> list[str]:
    """g = ROIC * reinvestment rate, read backwards out of the terminal year.

    The terminal assumption is a claim about returns whether or not the analyst
    states one. This is the single best signal of whether it is coherent.
    """
    reinvestment = terminal.capex + terminal.delta_nwc - terminal.da
    if terminal.nopat <= 0:
        return [
            f"FLAG: terminal NOPAT of {terminal.nopat:,.0f}mm is not positive, so no "
            "reinvestment rate can be formed and the terminal growth rate is being "
            "capitalised without any implied return standing behind it."
        ]

    rate = reinvestment / terminal.nopat
    head = (
        f"Terminal reinvestment is {reinvestment:,.0f}mm on {terminal.nopat:,.0f}mm of "
        f"NOPAT, a reinvestment rate of {rate:.1%}."
    )
    if abs(rate) < 1e-6:
        return [
            "FLAG: " + head + f" Growing at {g:.2%} on no reinvestment implies an "
            "infinite return on capital. Either capex and working capital have to "
            "rise in the terminal year or the growth rate has to come down."
        ]

    roic = g / rate
    if rate < 0:
        return [
            "FLAG: " + head + f" It is negative, so the terminal year releases more "
            f"capital than it absorbs while still growing at {g:.2%}. The implied "
            f"return on capital is {roic:.1%}, which is not a return at all: growth "
            "is being funded by disinvestment. Raise capex, or hold working capital "
            "flat as a share of revenue in the terminal year."
        ]
    if roic > _ROIC_CEILING:
        return [
            "FLAG: " + head + f" It implies a terminal ROIC of {roic:.1%}, above the "
            f"{_ROIC_CEILING:.0%} mark. A perpetual return that far above the cost of "
            "capital assumes no competitor ever arrives."
        ]
    if roic < wacc:
        return [
            "FLAG: " + head + f" It implies a terminal ROIC of {roic:.1%} against a "
            f"WACC of {wacc:.2%}. Growth at that return destroys value, yet the model "
            "capitalises it as though it were worth paying for. Either the growth is "
            "worth less than zero or the reinvestment assumption is too heavy."
        ]
    return [
        head + f" It implies a terminal ROIC of {roic:.1%} against a WACC of "
        f"{wacc:.2%}, so terminal growth creates value and the assumption hangs "
        "together."
    ]


# -- sensitivities -------------------------------------------------------- #


def _grid_axis(centre: float, step: float) -> np.ndarray:
    return centre + np.linspace(-step, step, _SENSITIVITY_POINTS)


def _per_share(ev: float, fin: Financials, bridge: EVBridge) -> float:
    return equity_value_from_ev(ev, fin, bridge) / fin.diluted_shares


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
    projections = project(fin, assumptions)
    n = len(projections)
    fcff = np.array([p.fcff for p in projections])

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
            grid[i, j] = _per_share(ev, fin, bridge)

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

    projections = project(fin, assumptions)
    n = len(projections)
    fcff = np.array([p.fcff for p in projections])
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
            grid[i, j] = _per_share(ev, fin, bridge)

    frame = pd.DataFrame(
        grid,
        index=[f"{w:.1%}" for w in waccs],
        columns=[f"{m:.1f}x" for m in multiples],
    )
    frame.index.name = "WACC"
    frame.columns.name = "Exit EV/EBITDA"
    return frame
