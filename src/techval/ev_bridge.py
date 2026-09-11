"""Equity value to enterprise value, and the two traps in it.

    EV = equity value + debt + preferred + non-controlling interest
         - cash - short-term investments

Simple to write down. The judgment is entirely in what counts as debt, and two
items decide it for a technology company.

**Operating leases.** ASC 842 put the liability on the balance sheet but left the
expense in operating income, as a single straight-line rent charge. It is not
split into depreciation and interest the way IFRS 16 requires. So a US filer's
EBITDA is already *after* rent, and adding the lease liability to debt while
pairing it with that EBITDA counts the same obligation twice: once in the
numerator as a claim, and again in the denominator as a charge that suppressed
earnings. Either treatment is defensible so long as it is consistent:

    EV excluding lease liabilities   pairs with   EBITDA   (after rent)
    EV including lease liabilities   pairs with   EBITDAR  (before rent)

This module computes both and labels them. ``multiple_denominator`` returns the
earnings figure that matches whichever convention is configured, and it is the
only sanctioned way to pick one, so a mismatch cannot happen by accident. The
default excludes operating leases, matching how a US software comp set is
normally quoted. Finance leases go into debt under either convention: their
interest and amortisation already sit outside EBITDA.

**Convertible notes.** ASU 2020-06 removed the treasury-stock option and made
if-converted mandatory for convertible instruments. Whenever a convertible is
dilutive, its full conversion shares are already inside diluted weighted-average
shares outstanding. Equity value computed on that share count therefore already
contains the converted value, and adding the principal to debt as well
double-counts the instrument. The correct pairing:

    convertible in the money    shares in diluted WASO   -> treat as equity, not debt
    convertible out of the money   shares excluded       -> treat as debt

Getting this wrong is worth real money. Datadog carries roughly a billion
dollars of convertible notes against a conversion price far below the current
share price; counting them as debt on top of a diluted share count that already
reflects conversion overstates enterprise value by about that billion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import tags
from .config import Assumptions
from .financials import Financials


@dataclass
class EVBridge:
    """A priced bridge from share price to enterprise value, in USD millions."""

    ticker: str
    price: float
    diluted_shares: float
    equity_value: float

    straight_debt: float
    convertible_debt: float
    convertible_in_debt: float
    finance_lease: float
    operating_lease: float
    operating_lease_in_debt: float
    preferred: float
    nci: float
    cash: float
    short_term_investments: float

    enterprise_value: float
    ev_excluding_leases: float
    ev_including_leases: float
    net_debt: float

    lease_convention: str
    convertible_treatment: str
    notes: list[str] = field(default_factory=list)

    # Finance leases the filer tags inside its own debt caption, so they are part
    # of ``straight_debt`` and are NOT in ``finance_lease``. Reported so the debt
    # row can say what is inside it.
    finance_lease_in_straight_debt: float = 0.0
    debt_basis: str = ""

    @property
    def total_debt(self) -> float:
        return (
            self.straight_debt
            + self.convertible_in_debt
            + self.finance_lease
            + self.operating_lease_in_debt
        )

    def multiple_denominator(self, fin: Financials) -> tuple[float | None, str]:
        """The earnings figure that legitimately pairs with this EV.

        The single guard against quoting a lease-inclusive enterprise value over
        a post-rent EBITDA.
        """
        if self.operating_lease_in_debt > 0:
            return fin.ebitdar, "EBITDAR (before operating lease cost)"
        return fin.ebitda, "EBITDA (after operating lease cost, per ASC 842)"

    def ebit_denominator(self, fin: Financials) -> tuple[float | None, str]:
        """The operating profit figure that legitimately pairs with this EV.

        EV/EBIT carries the same lease trap as EV/EBITDA and for the same reason:
        under ASC 842 the rent charge sits inside operating income, so a
        lease-inclusive enterprise value divided by GAAP EBIT counts the lease
        twice. Under that convention the denominator is EBIT before rent.
        """
        if self.operating_lease_in_debt > 0:
            return fin.ebitr, "EBIT before operating lease cost"
        return fin.ebit, "EBIT (after operating lease cost, per ASC 842)"

    def rows(self) -> list[tuple[str, float]]:
        """The bridge as line items, each named for what is inside it.

        The debt row changes its own label rather than its value when the filer
        bundles finance leases into its debt caption. A reader who sees
        "Straight debt 143,448" beside "Finance leases 0.0" would reasonably
        conclude Verizon has no finance leases; what is true is that they are in
        the row above, because the concept the filer tags is called "long-term
        debt and capital lease obligations" and there is no split to recover.
        """
        debt_label = "+ Straight debt"
        if self.finance_lease_in_straight_debt:
            debt_label = "+ Debt, finance leases inside"
        elif "finance leases inside it" in self.debt_basis:
            debt_label = "+ Debt and finance leases"
        return [
            ("Share price", self.price),
            ("Diluted shares (mm)", self.diluted_shares),
            ("Equity value", self.equity_value),
            (debt_label, self.straight_debt),
            ("+ Convertible notes", self.convertible_in_debt),
            ("+ Finance leases", self.finance_lease),
            ("+ Operating leases", self.operating_lease_in_debt),
            ("+ Preferred stock", self.preferred),
            ("+ Non-controlling interest", self.nci),
            ("- Cash and equivalents", -self.cash),
            ("- Short-term investments", -self.short_term_investments),
            ("Enterprise value", self.enterprise_value),
        ]


def _check_conversion_shares_are_present(fin: Financials, conv_cfg) -> list[str]:
    """Confirm the conversion shares really are inside diluted WASO.

    Treating an in-the-money convertible as equity rests entirely on the claim
    that its shares are already in the diluted count. That claim is testable: the
    gap between diluted and basic weighted-average shares has to be at least as
    large as the shares conversion would create. If it is not, the filer excluded
    them as antidilutive despite the price, the assumption behind the treatment is
    false, and enterprise value is understated by the note principal.

    Datadog: 366.9mm diluted against 352.3mm basic leaves a 14.6mm gap, and
    conversion accounts for 6.7mm of it, with the balance option and RSU
    overhang. The treatment holds.
    """
    if fin.basic_shares is None or not conv_cfg.conversion_price:
        return []
    implied = fin.convertible_debt / conv_cfg.conversion_price
    gap = fin.diluted_shares - fin.basic_shares
    if gap >= implied:
        return [
            f"Checked: conversion would create {implied:,.1f}mm shares and diluted "
            f"WASO runs {gap:,.1f}mm above basic, so the shares are inside the "
            "count the equity value is built on."
        ]
    return [
        f"Warning: conversion would create {implied:,.1f}mm shares but diluted "
        f"WASO runs only {gap:,.1f}mm above basic, so they do not appear to be in "
        "the diluted count. The filer may have excluded them as antidilutive. "
        "Treating the notes as debt is then the correct call, and enterprise "
        f"value here is understated by up to {fin.convertible_debt:,.0f}mm."
    ]


def build_ev_bridge(
    fin: Financials, price: float, assumptions: Assumptions
) -> EVBridge:
    """Bridge from a share price to enterprise value under stated conventions."""
    notes: list[str] = []
    conv_cfg = assumptions.convertibles

    # -- convertibles ------------------------------------------------------ #
    treatment = conv_cfg.treatment
    if fin.convertible_debt <= 0:
        treatment = "none"
    elif treatment == "auto":
        if conv_cfg.conversion_price:
            if price > conv_cfg.conversion_price:
                treatment = "if_converted"
                notes.append(
                    f"Convertible notes of {fin.convertible_debt:,.0f}mm are in the "
                    f"money at {price:,.2f} against a {conv_cfg.conversion_price:,.2f} "
                    "conversion price. Under ASU 2020-06 their shares are already in "
                    "diluted WASO, so they are carried as equity, not debt."
                )
                notes.extend(_check_conversion_shares_are_present(fin, conv_cfg))
            else:
                treatment = "debt"
                notes.append(
                    f"Convertible notes of {fin.convertible_debt:,.0f}mm are out of "
                    f"the money at {price:,.2f} against a "
                    f"{conv_cfg.conversion_price:,.2f} conversion price, so their "
                    "shares are excluded from diluted WASO and they are debt."
                )
        else:
            treatment = "debt"
            notes.append(
                f"Convertible notes of {fin.convertible_debt:,.0f}mm are carried as "
                "debt because no conversion price was supplied. If they are in the "
                "money their shares are already inside diluted WASO and enterprise "
                "value is overstated by up to that amount. Set "
                "convertibles.conversion_price from the notes footnote to resolve this."
            )

    convertible_in_debt = fin.convertible_debt if treatment == "debt" else 0.0

    # -- leases ------------------------------------------------------------ #
    capitalize = assumptions.leases.capitalize_operating_leases
    operating_lease_in_debt = fin.operating_lease_liability if capitalize else 0.0
    finance_lease = (
        fin.finance_lease_liability
        if assumptions.leases.include_finance_leases_in_debt
        else 0.0
    )
    if capitalize:
        notes.append(
            "Operating lease liabilities are counted as debt, so every multiple "
            "built on this bridge uses EBITDAR, before operating lease cost."
        )
        if fin.operating_lease_cost is None and fin.operating_lease_liability > 0:
            notes.append(
                "Warning: leases are capitalised but the filer does not report an "
                "operating lease cost, so EBITDAR cannot be formed and EV/EBITDA "
                "multiples will be suppressed rather than shown inconsistently."
            )

    # The concepts behind the debt figure, but only where they change what the
    # figure means. A non-current line plus a current line is what the row label
    # already says, and a note on every company in a comp set trains a reader to
    # skip the block on the company where it matters, which is the failure the
    # read-as-zero flags were criticised for. Two cases do change the meaning: a
    # concept that bundles finance leases into the debt, and a total read off a
    # single tag because one leg of the balance sheet could not be resolved.
    from_combined = len(fin.debt_tags) == 1 and fin.debt_tags[0] in tags.DEBT_COMBINED
    if set(fin.debt_tags) & tags.INCLUDES_FINANCE_LEASES or from_combined:
        notes.append(f"Debt is read from {fin.debt_basis}.")
    if fin.finance_lease_inside_debt:
        notes.append(
            f"{fin.finance_lease_inside_debt:,.0f}mm of finance lease liabilities "
            "sits inside that debt figure rather than on the finance lease line, "
            "because the filer tags the two together. The finance lease row below "
            "shows only what is tagged separately, so nothing is counted twice."
        )

    shares = fin.shares_for_valuation

    # A treasury-stock count is built from outstanding shares and award tables,
    # so it does NOT contain the shares an in-the-money convertible would
    # create, whereas diluted WASO does through the mandatory if-converted
    # method. Pairing that count with a bridge that also carries the note as
    # equity would drop the instrument out of both sides of the valuation. Add
    # the conversion shares back here, where the treatment is decided.
    if (
        fin.valuation_shares
        and treatment == "if_converted"
        and conv_cfg.conversion_price
    ):
        conversion_shares = fin.convertible_debt / conv_cfg.conversion_price
        shares += conversion_shares
        notes.append(
            f"Added {conversion_shares:,.2f}mm conversion shares to the treasury "
            "stock count. That count is built from outstanding shares and award "
            "tables and so does not contain them, while the notes are carried as "
            "equity rather than debt, and the instrument has to appear on one "
            "side of the bridge or the other."
        )

    equity_value = price * shares
    liquid = fin.cash + fin.short_term_investments

    core = (
        equity_value
        + fin.straight_debt
        + convertible_in_debt
        + finance_lease
        + fin.preferred
        + fin.nci
        - liquid
    )
    # ``core`` never contains operating leases, so both variants derive from it
    # directly. Both are always reported; the convention only picks the headline.
    ev_excluding = core
    ev_including = core + fin.operating_lease_liability
    enterprise_value = ev_including if capitalize else ev_excluding

    # An enterprise value below the market capitalisation is arithmetic, not an
    # error: it says the claims ahead of the common are smaller than the liquid
    # assets behind it. It is also what a debt figure read as zero looks like, so
    # the bridge states which of the two it is instead of leaving the reader to
    # guess. Comcast printed an enterprise value below its own market
    # capitalisation for a quarter of a trillion dollars of reasons, and the
    # table said nothing at all.
    if enterprise_value < equity_value:
        total = (
            fin.straight_debt
            + convertible_in_debt
            + finance_lease
            + operating_lease_in_debt
            + fin.preferred
            + fin.nci
        )
        notes.append(
            f"Enterprise value sits {equity_value - enterprise_value:,.0f}mm below "
            f"the market capitalisation because cash and short-term investments of "
            f"{liquid:,.0f}mm exceed the {total:,.0f}mm of debt, preferred and "
            "minority interest ahead of the common. Read it against the balance "
            "sheet: a debt figure that failed to resolve looks exactly like this."
        )

    return EVBridge(
        ticker=fin.ticker,
        price=price,
        diluted_shares=shares,
        equity_value=equity_value,
        straight_debt=fin.straight_debt,
        convertible_debt=fin.convertible_debt,
        convertible_in_debt=convertible_in_debt,
        finance_lease=finance_lease,
        operating_lease=fin.operating_lease_liability,
        operating_lease_in_debt=operating_lease_in_debt,
        preferred=fin.preferred,
        nci=fin.nci,
        cash=fin.cash,
        short_term_investments=fin.short_term_investments,
        enterprise_value=enterprise_value,
        ev_excluding_leases=ev_excluding,
        ev_including_leases=ev_including,
        net_debt=enterprise_value - equity_value,
        lease_convention=(
            "operating leases in debt, EBITDAR basis"
            if capitalize
            else "operating leases excluded, EBITDA basis (ASC 842)"
        ),
        convertible_treatment=treatment,
        notes=notes,
        finance_lease_in_straight_debt=fin.finance_lease_inside_debt,
        debt_basis=fin.debt_basis,
    )


def equity_value_from_dcf_ev(
    enterprise_value: float, fin: Financials, bridge: EVBridge
) -> float:
    """Walk a DCF enterprise value to equity.

    Differs from ``equity_value_from_ev`` in exactly one line item: operating
    lease liabilities are never subtracted, whatever convention the bridge was
    built under. A DCF enterprise value is the present value of free cash flow
    that pays rent in every projected year and in the terminal perpetuity, so
    the lease obligation is already serviced inside the flows. Subtracting the
    liability as well would charge the same lease twice, which is the DCF-side
    twin of the EV/EBITDA pairing trap the bridge exists to prevent.

    Finance leases stay in the walk: their interest and principal are financing
    flows that unlevered FCFF deliberately excludes, so the claim is outstanding
    against the enterprise value exactly as straight debt is.
    """
    return (
        enterprise_value
        - bridge.straight_debt
        - bridge.convertible_in_debt
        - bridge.finance_lease
        - bridge.preferred
        - bridge.nci
        + bridge.cash
        + bridge.short_term_investments
    )


def equity_value_from_ev(
    enterprise_value: float, fin: Financials, bridge: EVBridge
) -> float:
    """Walk back the other way, for a DCF that values the enterprise.

    Uses the same debt definition the bridge used, so a DCF equity value and a
    comps equity value are reconcilable rather than merely similar.
    """
    return (
        enterprise_value
        - bridge.straight_debt
        - bridge.convertible_in_debt
        - bridge.finance_lease
        - bridge.operating_lease_in_debt
        - bridge.preferred
        - bridge.nci
        + bridge.cash
        + bridge.short_term_investments
    )
