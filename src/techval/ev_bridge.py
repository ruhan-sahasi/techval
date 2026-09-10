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

    def rows(self) -> list[tuple[str, float]]:
        return [
            ("Share price", self.price),
            ("Diluted shares (mm)", self.diluted_shares),
            ("Equity value", self.equity_value),
            ("+ Straight debt", self.straight_debt),
            ("+ Convertible notes", self.convertible_in_debt),
            ("+ Finance leases", self.finance_lease),
            ("+ Operating leases", self.operating_lease_in_debt),
            ("+ Preferred stock", self.preferred),
            ("+ Non-controlling interest", self.nci),
            ("- Cash and equivalents", -self.cash),
            ("- Short-term investments", -self.short_term_investments),
            ("Enterprise value", self.enterprise_value),
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

    equity_value = price * fin.diluted_shares
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

    return EVBridge(
        ticker=fin.ticker,
        price=price,
        diluted_shares=fin.diluted_shares,
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
