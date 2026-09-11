"""Accretion/(dilution): does the deal add to next year's earnings per share.

Pro forma EPS is the acquirer's earnings plus the target's, less what funding the
purchase costs after tax, over the acquirer's share count plus whatever new
shares were printed to pay for it. Everything else in this module is bookkeeping
around that one sentence.

**Convention.** Year one, both sides on trailing twelve month figures, at the
acquirer's marginal tax rate throughout. The acquirer's rate is the right one:
the incremental dollar of synergy and the incremental dollar of interest are
earned and deducted inside the acquirer's tax posture, not the target's.

One-time deal fees are funded in the sources and uses table but are deliberately
kept out of the earnings walk. A fee paid once at close is not part of the
run-rate earnings the market capitalises, and leaving it in would make every
large deal look dilutive in year one for a reason that has nothing to do with
whether the businesses fit. Financing fees go the other way: they are capitalised
and amortised over the life of the facility, so their annual charge does belong
in the walk.

**The trap this module is built around.** Accretion is quoted as a percentage of
standalone EPS. When standalone EPS is negative that percentage is worse than
useless, it inverts: a pro forma EPS of -0.40 against a standalone -0.50 is an
improvement of ten cents, and the ratio -0.10 / -0.50 prints as +20% accretive,
while a further loss prints as dilution with the sign flipped. A base near zero
fails the other way: four cents of dilution against four cents of standalone EPS
prints as minus one hundred percent, which measures the denominator and not the
deal. Loss-making and barely profitable software acquirers hit both constantly.
So the percentage is computed only over a base large enough to divide by, and
otherwise the module reports the cents figure alone and says why. Nothing is
silently substituted, and the sensitivity grid switches units on the same test so
the two can never disagree.

**Breakeven synergies, solved rather than searched.** With S the run-rate pre-tax
synergy, phi the year-one phase-in, t the tax rate, N_pf pro forma shares and D
the sum of every after-tax funding charge:

    PF NI = NI_acq + NI_tgt + S * phi * (1 - t) - D

Set pro forma EPS equal to standalone EPS and solve directly:

    EPS_standalone * N_pf = NI_acq + NI_tgt + S * phi * (1 - t) - D
    S = [EPS_standalone * N_pf - (NI_acq + NI_tgt - D)] / (1 - t) / phi

N_pf does not depend on S, so this is exact and needs no iteration. The bracket
is the standalone earnings the deal has to reproduce less the earnings it brings
before synergies, which is the number a banker actually wants: how much has to
come out of the cost base to stand still.

**The rule of thumb, which is the whole of the intuition.** A deal is accretive
when what you buy yields more than what you pay with. What you buy is always the
target's earnings yield at the offer price, target net income over the equity
purchase price. What you pay with depends on the currency:

    Stock funded: the currency costs the acquirer's own earnings yield, 1 / its
    P/E, because every share printed hands a claim on those earnings to someone
    new. So the deal is accretive when the target's yield at the offer exceeds
    the acquirer's earnings yield, which is the same statement as the acquirer's
    P/E being above the P/E it is paying. Note the direction: a high-multiple
    acquirer has a low earnings yield, which is precisely why its paper is cheap
    currency. Reading the comparison the other way round is the commonest error
    in this calculation, and it inverts the answer on every deal.

    Cash funded: the currency costs the after-tax coupon on new debt, or the
    after-tax yield surrendered on balance sheet cash spent. Accretive when the
    target's yield at the offer exceeds that.

Both are computed in ``checks``, blended at the actual mix, with the offer price
at which each crosses over, and the module states whether the full calculation
agrees. Where it disagrees the difference is the synergies and the financing
charges, which is exactly what the rule of thumb leaves out. The rule is
undefined when either side loses money, and in that case it says so rather than
printing a negative earnings yield and pretending it means something.

**What the screen leaves out.** Left at its default the module stops here: no
opening balance sheet, no goodwill, no deferred tax liability on the step-up, no
deferred revenue haircut, no debt paydown or share buyback over time, and no
purchase-accounting step-up amortisation unless the config asks for it. Year-one
EPS built on two sets of trailing twelve month figures is a screening tool that
tells you the shape of a deal and roughly what it costs. It is not a merger
model, and no one should take a bid to a board on it.

**What ``merger.purchase_accounting.enabled`` adds.** Switched on, the module
builds the opening balance sheet and rolls the combined company forward, and the
result is attached as ``MergerResult.purchase_accounting``. Four pieces of it are
the ones a reviewer should look at first, because they are the ones most often
got wrong.

*Goodwill is a plug, and the deferred tax liability makes it bigger.* The excess
of the equity purchase price over the book equity acquired is allocated first to
identifiable intangibles, at the configured share, and whatever is left is
goodwill. A stock deal is tax free to the seller and carries over the seller's
tax basis, so the write-up exists for book and not for tax. That temporary
difference is a liability assumed at close:

    DTL      = intangible write-up * tax rate
    goodwill = excess - intangibles + DTL

The plus sign is the part people get backwards. Goodwill is consideration less
the fair value of net assets acquired, and the DTL is a liability inside those
net assets, so booking it takes net assets down and pushes goodwill up. Skip it
and goodwill is understated by roughly a quarter of the write-up, and the opening
balance sheet does not balance.

*Goodwill is not amortised, intangibles are.* ASC 350 stopped goodwill
amortisation in 2001 and replaced it with impairment testing, so goodwill never
touches EPS until the day it is written down, all at once. Identifiable
intangibles, the acquired technology and the customer relationships, are
amortised over their useful lives and hit EPS every quarter.

*That amortisation has no cash tax deduction behind it.* The DTL unwinds as the
intangible amortises, which produces a deferred tax benefit, so GAAP tax expense
looks normal and the book charge is the usual ``amortisation * (1 - t)``. Cash
tax is not reduced at all, because there was never any tax basis to amortise. So
free cash flow gets back only ``(1 - t)`` of the amortisation, not the whole of
it. Adding back the full charge overstates cash generation by the tax on it, and
therefore overstates how fast the acquisition debt is repaid.

*The deferred revenue haircut destroys revenue outright.* Acquired deferred
revenue is remeasured at fair value, which is the cost of delivering the service
plus a normal margin rather than the amount billed. The difference is revenue the
target would have recognised and the combined company never will. For a
subscription business with a year of billings sitting in deferred revenue that is
a visible hole in year-one revenue, and it is why SaaS deals look worse in the
first year than the run rate says they should.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Assumptions
from .errors import ConfigError, MissingDataError, NotMeaningfulError
from .ev_bridge import EVBridge, build_ev_bridge
from .financials import Financials

# Sources and uses is an identity, not an estimate. Anything above rounding noise
# on figures of this size is a construction error.
_BALANCE_TOLERANCE = 1e-6


@dataclass
class Consideration:
    """What is offered, what it buys, and how it is funded. USD millions."""

    offer_price: float
    premium: float
    equity_purchase_price: float
    enterprise_purchase_price: float
    cash_consideration: float
    stock_consideration: float
    new_shares_issued: float
    exchange_ratio: float
    new_debt: float
    cash_used: float
    pct_cash: float
    pct_stock: float

    @property
    def target_net_debt(self) -> float:
        return self.enterprise_purchase_price - self.equity_purchase_price

    def rows(self) -> list[tuple[str, float]]:
        return [
            ("Offer price per share", self.offer_price),
            ("Premium to unaffected price", self.premium),
            ("Equity purchase price", self.equity_purchase_price),
            ("+ Target net debt assumed", self.target_net_debt),
            ("Enterprise purchase price", self.enterprise_purchase_price),
            ("Cash consideration", self.cash_consideration),
            ("Stock consideration", self.stock_consideration),
            ("Exchange ratio (acquirer shares per target share)", self.exchange_ratio),
            ("New acquirer shares issued (mm)", self.new_shares_issued),
            ("New debt raised", self.new_debt),
            ("Balance sheet cash used", self.cash_used),
            ("Mix: cash", self.pct_cash),
            ("Mix: stock", self.pct_stock),
        ]


@dataclass
class AccretionResult:
    """Year-one EPS effect and the walk that produces it.

    ``components`` sum exactly to ``pro_forma_net_income``. ``accretion_pct`` is
    None when standalone EPS is not positive, in which case only the cents figure
    carries meaning and ``notes`` says so.
    """

    acquirer_eps_standalone: float
    pro_forma_eps: float
    accretion_dollars: float
    accretion_pct: float | None
    pro_forma_net_income: float
    pro_forma_shares: float
    breakeven_synergies: float | None
    components: list[tuple[str, float]]
    notes: list[str] = field(default_factory=list)

    @property
    def is_accretive(self) -> bool:
        return self.accretion_dollars > 0

    def rows(self) -> list[tuple[str, float]]:
        out = list(self.components)
        out.append(("Pro forma net income", self.pro_forma_net_income))
        out.append(("Pro forma diluted shares (mm)", self.pro_forma_shares))
        out.append(("Acquirer standalone EPS", self.acquirer_eps_standalone))
        out.append(("Pro forma EPS", self.pro_forma_eps))
        out.append(("Accretion / (dilution), $", self.accretion_dollars))
        if self.accretion_pct is not None:
            out.append(("Accretion / (dilution), %", self.accretion_pct))
        return out


@dataclass
class ContributionRow:
    """One earnings metric split between the two sides of the combination."""

    metric: str
    acquirer: float
    target: float
    acquirer_pct: float | None
    target_pct: float | None

    def row(self) -> tuple[str, float, float, float | None, float | None]:
        return (
            self.metric,
            self.acquirer,
            self.target,
            self.acquirer_pct,
            self.target_pct,
        )


@dataclass
class OpeningBalanceSheet:
    """What the acquirer books at close. USD millions.

    The identity that has to hold, and that ``_opening_balance_sheet`` asserts
    rather than trusts:

        book equity acquired + intangibles + goodwill - DTL = equity purchase price

    Goodwill is the residual in that line, never an input. ``deferred_revenue_haircut``
    and ``target_debt_refinanced`` are memo figures: they carry earnings and
    funding consequences that are handled elsewhere in this module, and they are
    deliberately outside the allocation above so the reconciliation stays the one
    a reader can check in their head.
    """

    equity_purchase_price: float
    target_book_equity: float
    excess_purchase_price: float
    intangibles_created: float
    goodwill_created: float
    deferred_tax_liability: float
    deferred_revenue_haircut: float
    target_debt_refinanced: float

    def rows(self) -> list[tuple[str, float]]:
        return [
            ("Equity purchase price", self.equity_purchase_price),
            ("- Book equity acquired", _charge(self.target_book_equity)),
            ("Excess over book equity", self.excess_purchase_price),
            (
                "- Allocated to identifiable intangibles",
                _charge(self.intangibles_created),
            ),
            ("+ Deferred tax liability on the write-up", self.deferred_tax_liability),
            ("Goodwill created", self.goodwill_created),
            ("Memo: deferred revenue written down", self.deferred_revenue_haircut),
            ("Memo: target debt assumed or refinanced", self.target_debt_refinanced),
        ]


@dataclass
class ProFormaYear:
    """One projected year of the combined company. USD millions, EPS in dollars.

    ``interest`` is the charge on acquisition debt alone, so it falls as the
    sweep repays the facility. Interest on debt both companies already carried,
    and the interest income their cash earns, sit in the non-operating line
    between ``ebit`` and ``pretax_income`` and are held flat.

    ``debt_balance`` is the acquisition facility at the END of the year, after
    ``debt_repaid``, while ``interest`` is charged on the balance at the start of
    it. Sweeping out of cash the year has not generated yet would credit the deal
    with a repayment before the money arrives.

    The gap between ``ebitda`` and ``ebit`` is the existing depreciation and
    amortisation of the two businesses plus ``intangible_amortisation``.
    """

    year: int
    revenue: float
    ebitda: float
    intangible_amortisation: float
    ebit: float
    interest: float
    pretax_income: float
    taxes: float
    net_income: float
    shares: float
    eps: float
    debt_balance: float
    cash_flow: float
    debt_repaid: float


@dataclass
class PurchaseAccounting:
    """The opening balance sheet and the years that follow it.

    ``accretion_by_year`` holds (year, dollars per share, percent or None). The
    percent entry follows exactly the test the headline figure uses, so the two
    can never disagree about the unit: it is None wherever standalone EPS is too
    small or too negative to divide by, and the cents figure is then the only one
    that carries meaning.
    """

    opening: OpeningBalanceSheet
    years: list[ProFormaYear]
    accretion_by_year: list[tuple[int, float, float | None]]
    notes: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Year": y.year,
                    "Revenue": y.revenue,
                    "EBITDA": y.ebitda,
                    "Intangible amortisation": y.intangible_amortisation,
                    "EBIT": y.ebit,
                    "Interest on acquisition debt": y.interest,
                    "Pre-tax income": y.pretax_income,
                    "Taxes": y.taxes,
                    "Net income": y.net_income,
                    "Diluted shares (mm)": y.shares,
                    "Pro forma EPS": y.eps,
                    "Acquisition debt, closing": y.debt_balance,
                    "Free cash flow": y.cash_flow,
                    "Debt repaid": y.debt_repaid,
                }
                for y in self.years
            ]
        )


@dataclass
class MergerResult:
    """A full accretion/(dilution) screen.

    ``acquirer`` and ``target`` are the Financials the run was built on, so a
    renderer has the tickers and the as-of dates without carrying them separately.
    ``sources_uses`` is a list of (side, label, amount) with side in {"Use",
    "Source", "Total"}. ``pro_forma_ownership`` maps "acquirer" and "target" to
    their share of the pro forma count.

    ``purchase_accounting`` is None unless ``merger.purchase_accounting.enabled``
    is set. Everything above it is the year-one screen and is unaffected by that
    switch: turning purchase accounting on adds a result, it does not restate one.
    """

    acquirer: Financials
    target: Financials
    consideration: Consideration
    accretion: AccretionResult
    contribution: list[ContributionRow]
    pro_forma_ownership: dict[str, float]
    sources_uses: list[tuple[str, str, float]]
    sensitivity: pd.DataFrame
    checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    purchase_accounting: PurchaseAccounting | None = None

    def contribution_frame(self) -> pd.DataFrame:
        """Earnings contribution against pro forma ownership, side by side.

        The comparison is the point: if the target's owners walk away with more
        of the combined company than the earnings they bring to it, the deal
        transfers value to them, and that shows up as dilution.
        """
        data = [
            {
                "Metric": r.metric,
                self.acquirer.ticker: r.acquirer,
                self.target.ticker: r.target,
                f"{self.acquirer.ticker} %": r.acquirer_pct,
                f"{self.target.ticker} %": r.target_pct,
            }
            for r in self.contribution
        ]
        data.append(
            {
                "Metric": "Pro forma ownership",
                self.acquirer.ticker: None,
                self.target.ticker: None,
                f"{self.acquirer.ticker} %": self.pro_forma_ownership["acquirer"],
                f"{self.target.ticker} %": self.pro_forma_ownership["target"],
            }
        )
        return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def _tax_rate(acq_fin: Financials, assumptions: Assumptions, notes: list[str]) -> float:
    """The rate applied to synergies and to every financing charge.

    The acquirer's rate, on both sides of the walk. The target's earnings arrive
    already taxed at its own rate and are not regrossed: this is a screen, not a
    structuring exercise.
    """
    rate = assumptions.tax.marginal_tax_rate
    if assumptions.tax.use_effective_rate:
        filed = acq_fin.effective_tax_rate
        if filed is None:
            notes.append(
                "tax.use_effective_rate is set but the acquirer's filed effective "
                "rate is not meaningful (a loss, or a benefit against a profit), so "
                f"the marginal rate of {rate:.1%} was used instead"
            )
        else:
            rate = filed
            notes.append(
                f"synergies and financing charges taxed at the acquirer's filed "
                f"effective rate of {rate:.1%} rather than the marginal rate"
            )
    if not 0.0 <= rate < 1.0:
        raise ConfigError(
            f"tax rate of {rate:.1%} is outside [0, 100). Every after-tax figure "
            "here divides or multiplies by (1 - t), so a rate at or above 100% is "
            "not recoverable."
        )
    return rate


def _offer_price(unaffected_price: float, assumptions: Assumptions) -> float:
    m = assumptions.merger
    if (m.offer_price_per_share is None) == (m.offer_premium is None):
        raise ConfigError(
            "set exactly one of merger.offer_price_per_share or merger.offer_premium. "
            "Setting both lets the file state two different offers and gives no rule "
            "for which wins; setting neither leaves the deal unpriced."
        )
    if m.offer_price_per_share is not None:
        if m.offer_price_per_share <= 0:
            raise ConfigError(
                f"merger.offer_price_per_share of {m.offer_price_per_share} is not a "
                "price"
            )
        return m.offer_price_per_share
    assert m.offer_premium is not None
    price = unaffected_price * (1.0 + m.offer_premium)
    if price <= 0:
        raise ConfigError(
            f"merger.offer_premium of {m.offer_premium:.1%} against an unaffected "
            f"price of {unaffected_price:,.2f} implies a non-positive offer"
        )
    return price


def _net_debt(bridge: EVBridge) -> float:
    """Net debt on exactly the definition the EV bridge used.

    Written out rather than taken from ``bridge.net_debt`` so the reader can see
    which claims are being assumed. It reproduces that property by construction,
    including the bridge's convertible and operating lease decisions, so a
    purchase price here and an enterprise value elsewhere are reconcilable.
    """
    return (
        bridge.straight_debt
        + bridge.convertible_in_debt
        + bridge.finance_lease
        + bridge.operating_lease_in_debt
        + bridge.preferred
        + bridge.nci
        - bridge.cash
        - bridge.short_term_investments
    )


def build_consideration(
    acq_fin: Financials,
    tgt_fin: Financials,
    acq_price: float,
    tgt_price: float,
    assumptions: Assumptions,
    tgt_bridge: EVBridge | None = None,
    notes: list[str] | None = None,
) -> Consideration:
    """Price the offer and fund it.

    ``tgt_price`` is the unaffected price, the reference the premium is quoted
    against. Where no target bridge is supplied one is built at the offer price
    rather than the unaffected price, because a convertible that is out of the
    money today is usually in the money at a control premium, and the change of
    control provision settles it either way.
    """
    m = assumptions.merger
    notes = notes if notes is not None else []

    if acq_price <= 0 or tgt_price <= 0:
        raise ConfigError(
            f"prices must be positive to price an offer: acquirer {acq_price}, "
            f"target {tgt_price}"
        )
    if tgt_fin.diluted_shares <= 0:
        raise MissingDataError(
            "target diluted shares",
            ticker=tgt_fin.ticker,
            hint="an equity purchase price cannot be formed without a share count",
        )

    offer_price = _offer_price(tgt_price, assumptions)
    premium = offer_price / tgt_price - 1.0

    if tgt_bridge is None:
        tgt_bridge = build_ev_bridge(tgt_fin, offer_price, assumptions)

    equity_purchase_price = offer_price * tgt_fin.diluted_shares
    enterprise_purchase_price = equity_purchase_price + _net_debt(tgt_bridge)

    pct_cash = m.pct_cash
    pct_stock = 1.0 - pct_cash
    cash_consideration = pct_cash * equity_purchase_price
    stock_consideration = pct_stock * equity_purchase_price

    # New shares follow from the dollars of stock, not from the exchange ratio
    # alone: at a partial stock mix only that fraction of the ratio is printed.
    exchange_ratio = offer_price / acq_price
    new_shares_issued = stock_consideration / acq_price

    # Fees are funded like the cash portion. Balance sheet cash goes first, new
    # debt takes the remainder, which is what "cash and new debt" means in a
    # press release.
    cash_need = cash_consideration + m.deal_fees + m.financing_fees
    available = acq_fin.cash + acq_fin.short_term_investments
    if m.balance_sheet_cash_used > available:
        raise ConfigError(
            f"merger.balance_sheet_cash_used of {m.balance_sheet_cash_used:,.0f}mm "
            f"exceeds the acquirer's {available:,.0f}mm of cash and short-term "
            "investments. Fund the gap with new debt or lower the figure."
        )
    cash_used = min(m.balance_sheet_cash_used, cash_need)
    if cash_used < m.balance_sheet_cash_used:
        notes.append(
            f"only {cash_used:,.0f}mm of the {m.balance_sheet_cash_used:,.0f}mm of "
            "balance sheet cash configured is needed; the cash portion plus fees is "
            "smaller than the amount offered, so the excess stays on the balance "
            "sheet and keeps earning"
        )
    new_debt = cash_need - cash_used

    return Consideration(
        offer_price=offer_price,
        premium=premium,
        equity_purchase_price=equity_purchase_price,
        enterprise_purchase_price=enterprise_purchase_price,
        cash_consideration=cash_consideration,
        stock_consideration=stock_consideration,
        new_shares_issued=new_shares_issued,
        exchange_ratio=exchange_ratio,
        new_debt=new_debt,
        cash_used=cash_used,
        pct_cash=pct_cash,
        pct_stock=pct_stock,
    )


def _sources_and_uses(
    cons: Consideration, assumptions: Assumptions
) -> list[tuple[str, str, float]]:
    """The funding identity, asserted rather than assumed."""
    m = assumptions.merger
    uses = [
        ("Use", "Purchase of target equity", cons.equity_purchase_price),
        ("Use", "Advisory and other deal fees", m.deal_fees),
        ("Use", "Financing fees", m.financing_fees),
    ]
    sources = [
        ("Source", "New debt", cons.new_debt),
        ("Source", "Balance sheet cash", cons.cash_used),
        ("Source", "Acquirer stock issued", cons.stock_consideration),
    ]
    total_uses = sum(v for _, _, v in uses)
    total_sources = sum(v for _, _, v in sources)
    assert abs(total_uses - total_sources) <= _BALANCE_TOLERANCE * max(1.0, total_uses), (
        f"sources and uses do not balance: uses {total_uses:,.6f}mm against sources "
        f"{total_sources:,.6f}mm"
    )
    return [
        *uses,
        ("Total", "Total uses", total_uses),
        *sources,
        ("Total", "Total sources", total_sources),
    ]


# --------------------------------------------------------------------------- #
# the earnings walk
# --------------------------------------------------------------------------- #


def breakeven_synergies(
    *,
    standalone_eps: float,
    pro_forma_shares: float,
    net_income_before_synergies: float,
    tax_rate: float,
    phase_in: float,
) -> float | None:
    """Run-rate pre-tax synergy that makes the deal exactly EPS neutral.

    Solved from the identity in the module docstring, not searched. Returns None
    when the year-one phase-in is zero, because then no level of run-rate synergy
    changes year-one EPS and the breakeven does not exist rather than being large.
    """
    if phase_in <= 0:
        return None
    required_after_tax = standalone_eps * pro_forma_shares - net_income_before_synergies
    return required_after_tax / (1.0 - tax_rate) / phase_in


# Below this, a percentage against standalone EPS says more about how close the
# base is to zero than about the deal. Roughly a penny a quarter.
_MIN_EPS_FOR_PERCENT = 0.10


def _charge(value: float) -> float:
    """A deduction, signed for the walk, without a printed negative zero."""
    return -value if value else 0.0


def _accretion_percent(accretion_dollars: float, standalone_eps: float) -> float:
    """Accretion as a share of standalone EPS, or a refusal.

    Raised and caught rather than returned as None from the arithmetic, so the
    reason travels with the refusal and reaches the note the user reads.

    Two ways the percentage stops meaning anything, and loss-making or barely
    profitable software acquirers hit both. A negative base inverts the sign, so
    a deal that improves a loss prints as dilution. A base near zero explodes the
    ratio: CrowdStrike's TTM diluted EPS of about four cents turns four cents of
    dilution into minus eight hundred percent, which describes the denominator
    rather than the transaction.
    """
    if standalone_eps <= 0:
        raise NotMeaningfulError(
            f"standalone EPS of {standalone_eps:,.2f} is not positive, so percentage "
            "accretion inverts: an improvement in a loss divided by a negative base "
            "prints with the wrong sign. The cents per share figure is the only one "
            "that means anything here."
        )
    if standalone_eps < _MIN_EPS_FOR_PERCENT:
        raise NotMeaningfulError(
            f"standalone EPS of {standalone_eps:,.2f} is below the "
            f"{_MIN_EPS_FOR_PERCENT:,.2f} floor this engine will express a percentage "
            "against. Dividing by a base that close to zero produces a figure that "
            "moves with the denominator rather than with the deal. Read the cents "
            "per share instead."
        )
    return accretion_dollars / standalone_eps


def _run_accretion(
    acq_fin: Financials,
    tgt_fin: Financials,
    cons: Consideration,
    assumptions: Assumptions,
    tax_rate: float,
) -> AccretionResult:
    m = assumptions.merger
    syn = m.synergies
    notes: list[str] = []
    after_tax = 1.0 - tax_rate

    if acq_fin.diluted_shares <= 0:
        raise MissingDataError(
            "acquirer diluted shares",
            ticker=acq_fin.ticker,
            hint="EPS cannot be formed without a share count",
        )
    if m.financing_fees and m.financing_fee_amort_years <= 0:
        raise ConfigError(
            "merger.financing_fee_amort_years must be positive when financing fees "
            "are charged; the fees are capitalised and written off over the life of "
            "the facility"
        )
    if m.include_intangible_amortization and m.intangible_life_years <= 0:
        raise ConfigError(
            "merger.intangible_life_years must be positive when intangible "
            "amortisation is switched on"
        )

    synergy_pretax = (syn.pretax_cost_synergies + syn.pretax_revenue_synergies)
    synergy_after_tax = synergy_pretax * syn.phase_in_year_one * after_tax
    interest_new_debt = cons.new_debt * m.cost_of_new_debt * after_tax
    interest_foregone = cons.cash_used * m.foregone_cash_yield * after_tax
    financing_amort = (
        (m.financing_fees / m.financing_fee_amort_years) * after_tax
        if m.financing_fees
        else 0.0
    )
    intangible_amort = (
        (m.intangible_step_up / m.intangible_life_years) * after_tax
        if m.include_intangible_amortization
        else 0.0
    )

    components: list[tuple[str, float]] = [
        (f"{acq_fin.ticker} net income (standalone)", acq_fin.net_income),
        (f"+ {tgt_fin.ticker} net income", tgt_fin.net_income),
        ("+ After-tax synergies", synergy_after_tax),
        ("- After-tax interest on new debt", _charge(interest_new_debt)),
        ("- After-tax interest foregone on cash used", _charge(interest_foregone)),
        ("- After-tax financing fee amortisation", _charge(financing_amort)),
        ("- After-tax intangible amortisation", _charge(intangible_amort)),
    ]
    pro_forma_net_income = sum(v for _, v in components)
    pro_forma_shares = acq_fin.diluted_shares + cons.new_shares_issued

    standalone_eps = acq_fin.net_income / acq_fin.diluted_shares
    pro_forma_eps = pro_forma_net_income / pro_forma_shares
    accretion_dollars = pro_forma_eps - standalone_eps
    try:
        accretion_pct: float | None = _accretion_percent(
            accretion_dollars, standalone_eps
        )
    except NotMeaningfulError as exc:
        accretion_pct = None
        notes.append(str(exc))

    notes.append(
        f"deal fees of {m.deal_fees:,.0f}mm are funded in sources and uses but are "
        "excluded from this walk. They are incurred once at close, so charging them "
        "against run-rate earnings would understate the earnings the deal actually "
        "leaves behind. Financing fees are in the walk, amortised over "
        f"{m.financing_fee_amort_years} years, because they are a cost of carrying "
        "the debt rather than of doing the deal."
    )
    if synergy_pretax and syn.phase_in_year_one < 1.0:
        notes.append(
            f"{synergy_pretax:,.0f}mm of run-rate pre-tax synergies are phased in at "
            f"{syn.phase_in_year_one:.0%} in year one, so "
            f"{synergy_pretax * syn.phase_in_year_one:,.0f}mm is credited here"
        )
    if m.include_intangible_amortization:
        notes.append(
            f"intangible step-up of {m.intangible_step_up:,.0f}mm is amortised over "
            f"{m.intangible_life_years} years. No deferred tax liability is set up "
            "against it and no goodwill is computed, so this is the EPS charge only, "
            "not purchase accounting."
        )
    else:
        notes.append(
            "purchase-accounting step-up amortisation is off, so this is a cash EPS "
            "basis. That is how the street quotes it, and it is the convention that "
            "makes a stock deal and a cash deal comparable."
        )

    return AccretionResult(
        acquirer_eps_standalone=standalone_eps,
        pro_forma_eps=pro_forma_eps,
        accretion_dollars=accretion_dollars,
        accretion_pct=accretion_pct,
        pro_forma_net_income=pro_forma_net_income,
        pro_forma_shares=pro_forma_shares,
        breakeven_synergies=breakeven_synergies(
            standalone_eps=standalone_eps,
            pro_forma_shares=pro_forma_shares,
            net_income_before_synergies=pro_forma_net_income - synergy_after_tax,
            tax_rate=tax_rate,
            phase_in=syn.phase_in_year_one,
        ),
        components=components,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# contribution, sensitivity, checks
# --------------------------------------------------------------------------- #


def _contribution(
    acq_fin: Financials, tgt_fin: Financials, notes: list[str]
) -> list[ContributionRow]:
    pairs: list[tuple[str, float | None, float | None]] = [
        ("Revenue", acq_fin.revenue, tgt_fin.revenue),
        ("EBITDA", acq_fin.ebitda, tgt_fin.ebitda),
        ("EBIT", acq_fin.ebit, tgt_fin.ebit),
        ("Net income", acq_fin.net_income, tgt_fin.net_income),
    ]
    rows: list[ContributionRow] = []
    negative: list[str] = []
    for metric, a, t in pairs:
        if a is None or t is None:
            notes.append(
                f"{metric} contribution is not shown: one side does not report the "
                "components for it, and a share of a combined figure that is missing "
                "half its input would be a made-up number"
            )
            continue
        combined = a + t
        if combined == 0:
            rows.append(ContributionRow(metric, a, t, None, None))
            continue
        # A share of a combined negative figure is not a contribution. Two
        # loss-makers produce shares that sum to one and rank the larger loss as
        # the larger contributor, which reverses the meaning of the exercise. The
        # amounts are still shown; only the percentages are withheld.
        if a < 0 or t < 0:
            negative.append(metric)
            rows.append(ContributionRow(metric, a, t, None, None))
            continue
        rows.append(ContributionRow(metric, a, t, a / combined, t / combined))
    if negative:
        notes.append(
            "contribution shares for "
            + ", ".join(negative)
            + " are withheld because one side is negative. Dividing by a combined "
            "figure that a loss has pulled toward zero produces shares above 100% on "
            "one side and below zero on the other, and ranks the larger loss as the "
            "larger contributor. The amounts are shown; read those"
        )
    return rows


def accretion_sensitivity(
    acq_fin: Financials,
    tgt_fin: Financials,
    tgt_bridge: EVBridge,
    acq_price: float,
    tgt_price: float,
    assumptions: Assumptions,
    tax_rate: float,
    *,
    premium_step: float = 0.05,
    stock_steps: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0),
) -> pd.DataFrame:
    """Accretion across offer premium by stock mix, five by five.

    The two axes are the only terms a buyer really negotiates: how much it pays
    and what it pays with. Everything else, synergies included, is held at the
    configured level so the grid isolates deal terms.

    Grid spacing is a display choice rather than a valuation assumption, so it is
    a function argument and not an entry in the assumptions file. Premium steps
    are centred on the deal as configured and are not floored at zero: a cell
    below the unaffected price is a discount, and it is labelled as one.

    The target's net debt is held at the supplied bridge across every cell. Only
    the equity being purchased moves with the premium; the debt assumed does not.

    Values are percent accretion where standalone EPS is a base worth dividing
    by, and cents per share otherwise, on the same test ``_accretion_percent``
    applies to the headline figure so the grid and the headline never disagree
    about the unit. ``.attrs["unit"]`` and ``.attrs["label"]`` say which.
    """
    base_premium = _offer_price(tgt_price, assumptions) / tgt_price - 1.0
    premiums = [base_premium + k * premium_step for k in (-2, -1, 0, 1, 2)]
    standalone_eps = acq_fin.net_income / acq_fin.diluted_shares
    in_percent = standalone_eps >= _MIN_EPS_FOR_PERCENT

    data: list[list[float]] = []
    for prem in premiums:
        row: list[float] = []
        for stock in stock_steps:
            cell_merger = assumptions.merger.model_copy(
                update={
                    "offer_price_per_share": None,
                    "offer_premium": prem,
                    "pct_cash": 1.0 - stock,
                }
            )
            cell = assumptions.model_copy(update={"merger": cell_merger})
            # No bridge is passed, so the target is re-bridged at each cell's
            # offer price and a convertible that crosses into the money at a
            # higher premium is treated accordingly in that cell.
            cons = build_consideration(
                acq_fin, tgt_fin, acq_price, tgt_price, cell, None
            )
            res = _run_accretion(acq_fin, tgt_fin, cons, cell, tax_rate)
            row.append(
                res.accretion_dollars / standalone_eps
                if in_percent
                else res.accretion_dollars * 100.0
            )
        data.append(row)

    frame = pd.DataFrame(
        data,
        index=[f"{p:.1%}" for p in premiums],
        columns=[f"{s:.0%} stock" for s in stock_steps],
    )
    frame.index.name = "Offer premium"
    frame.columns.name = "Stock consideration"
    frame.attrs["unit"] = "percent" if in_percent else "cents"
    frame.attrs["label"] = (
        "Accretion / (dilution), % of standalone EPS"
        if in_percent
        else "Accretion / (dilution), cents per share"
    )
    return frame


def _rule_of_thumb_checks(
    acq_fin: Financials,
    tgt_fin: Financials,
    acq_bridge: EVBridge,
    cons: Consideration,
    assumptions: Assumptions,
    tax_rate: float,
    accretion_dollars: float,
) -> list[str]:
    """What you buy against what you pay with.

    Both yields are quoted after tax and so are directly comparable: the target's
    net income is already post-tax, and the cost of cash is taken down by the
    shield. The stock leg needs no tax adjustment at all, because paying in
    equity costs earnings, not interest.
    """
    m = assumptions.merger
    checks: list[str] = []

    acq_equity = acq_bridge.equity_value
    acq_ni = acq_fin.net_income
    tgt_ni = tgt_fin.net_income

    acq_yield = acq_ni / acq_equity if acq_ni > 0 and acq_equity > 0 else None
    tgt_yield = tgt_ni / cons.equity_purchase_price if tgt_ni > 0 else None

    funded = cons.new_debt + cons.cash_used
    pretax_cash_cost = (
        (cons.new_debt * m.cost_of_new_debt + cons.cash_used * m.foregone_cash_yield)
        / funded
        if funded > 0
        else m.cost_of_new_debt
    )
    cash_cost = pretax_cash_cost * (1.0 - tax_rate)

    if acq_yield is None:
        checks.append(
            f"Stock-funded rule: undefined. {acq_fin.ticker} earns "
            f"{acq_ni:,.0f}mm, so it has no earnings yield and no P/E to compare "
            "against the multiple it is paying. A loss-making acquirer paying in "
            "stock dilutes its loss per share whenever the target loses money too, "
            "and the rule of thumb has nothing to say about it."
        )
    elif tgt_yield is None:
        checks.append(
            f"Stock-funded rule: undefined. {tgt_fin.ticker} earns {tgt_ni:,.0f}mm, "
            f"so the yield bought at the offer is negative. Against "
            f"{acq_fin.ticker}'s {acq_yield:.2%} the deal is dilutive on any stock "
            "mix until synergies turn the target's earnings positive."
        )
    else:
        # Accretive when the yield bought beats the yield given up by printing
        # shares, which is the acquirer's own. Equivalently when the acquirer's
        # multiple is above the multiple it pays, and the crossover is the offer
        # at which the two multiples meet.
        crossover_price = tgt_ni / (acq_yield * tgt_fin.diluted_shares)
        checks.append(
            f"Stock-funded rule: the offer buys {tgt_yield:.2%} "
            f"({cons.equity_purchase_price / tgt_ni:,.1f}x earnings) and is paid for "
            f"with {acq_fin.ticker} paper yielding {acq_yield:.2%} "
            f"({acq_equity / acq_ni:,.1f}x). A pure stock deal is "
            + ("accretive" if tgt_yield > acq_yield else "dilutive")
            + " before synergies. The two multiples meet at an offer of "
            f"{crossover_price:,.2f} per share, a "
            f"{crossover_price / cons.offer_price * (1.0 + cons.premium) - 1.0:.1%} "
            "premium to the unaffected price."
        )

    if tgt_yield is None:
        checks.append(
            "Cash-funded rule: undefined. A negative earnings yield cannot be "
            f"compared with the {cash_cost:.2%} after-tax cost of the cash; the "
            "target has to earn something before borrowing to buy it can pay."
        )
    elif cash_cost <= 0:
        checks.append(
            "Cash-funded rule: the cash leg is assumed to cost nothing, so any "
            "positive earnings yield clears it and there is no crossover offer to "
            "quote. Set merger.cost_of_new_debt or merger.foregone_cash_yield to a "
            "real rate to make the comparison say anything."
        )
    else:
        crossover_price = tgt_ni / (cash_cost * tgt_fin.diluted_shares)
        checks.append(
            f"Cash-funded rule: the target yields {tgt_yield:.2%} at the offer "
            f"against an after-tax cost of cash of {cash_cost:.2%} "
            f"({pretax_cash_cost:.2%} pre-tax at a {tax_rate:.0%} rate). A pure cash "
            "deal is "
            + ("accretive" if tgt_yield > cash_cost else "dilutive")
            + f" before synergies. The crossover offer is {crossover_price:,.2f} per "
            f"share, a "
            f"{crossover_price / cons.offer_price * (1.0 + cons.premium) - 1.0:.1%} "
            "premium to the unaffected price."
        )

    if acq_yield is not None and tgt_yield is not None:
        blended = cons.pct_stock * acq_yield + cons.pct_cash * cash_cost
        predicted = tgt_yield > blended
        checks.append(
            f"At {cons.pct_cash:.0%} cash and {cons.pct_stock:.0%} stock the blended "
            f"cost of the consideration is {blended:.2%} against {tgt_yield:.2%} "
            "bought, so the rule predicts "
            + ("accretion" if predicted else "dilution")
            + f". The full calculation gives {accretion_dollars:+,.3f} per share, "
            "which "
            + (
                "agrees."
                if predicted == (accretion_dollars > 0)
                else "disagrees. The difference is what the rule of thumb leaves "
                "out: after-tax synergies, financing fee amortisation and the yield "
                "given up on cash spent."
            )
        )

    return checks


# --------------------------------------------------------------------------- #
# purchase accounting
# --------------------------------------------------------------------------- #


def _target_book_equity(tgt_fin: Financials, notes: list[str]) -> float:
    """Book equity acquired, built from the balance sheet items this engine carries.

    Goodwill is the price paid less the book equity bought, so this figure sets
    the whole allocation. It is assembled rather than read, because ``Financials``
    carries the claims a valuation needs and not a full balance sheet, and the
    two places that bite are worth naming.

    Operating and finance lease liabilities are NOT deducted. ASC 842 books a
    right-of-use asset against the liability at inception and the two stay close
    for the life of the lease. Subtracting the liability while the matching asset
    is absent from this object would understate book equity by the whole lease
    balance, which for a company with a large office footprint is hundreds of
    millions of goodwill conjured out of an accounting entry.

    Non-current operating assets are absent for the same reason: property, existing
    goodwill and intangibles, and capitalised commissions are not carried here. So
    is any current portion of borrowings, which sits inside current liabilities and
    is deducted a second time with total debt below. Both errors point the same
    way, understating book equity, which overstates the excess, the write-up and
    the amortisation charge. The deal therefore looks more dilutive here than a
    full opening balance sheet would make it, not less.
    """
    missing = [
        label
        for label, value in (
            ("current assets", tgt_fin.current_assets),
            ("current liabilities", tgt_fin.current_liabilities),
        )
        if value is None
    ]
    if missing:
        raise MissingDataError(
            "target book equity",
            ticker=tgt_fin.ticker,
            hint=(
                "purchase accounting allocates the price paid less the book equity "
                "acquired, and " + " and ".join(missing) + " did not resolve from the "
                "filings, so that book equity cannot be formed. Allocating the entire "
                "purchase price to goodwill instead would be a made-up number, so the "
                "run stops here"
            ),
        )
    assert tgt_fin.current_assets is not None
    assert tgt_fin.current_liabilities is not None

    book = (
        (tgt_fin.current_assets - tgt_fin.current_liabilities)
        - tgt_fin.total_debt_ex_leases
        - tgt_fin.nci
        - tgt_fin.preferred
    )
    notes.append(
        f"Book equity acquired of {book:,.0f}mm is working capital of "
        f"{tgt_fin.current_assets - tgt_fin.current_liabilities:,.0f}mm less "
        f"{tgt_fin.total_debt_ex_leases:,.0f}mm of borrowings and any preferred or "
        "minority interest. Lease liabilities are left out because their "
        "right-of-use assets are not carried here and deducting one side alone "
        "would invent goodwill; non-current operating assets are missing for the "
        "same reason. Book equity is therefore understated and goodwill overstated, "
        "which makes the deal look more dilutive here, not less."
    )
    return book


def _opening_balance_sheet(
    tgt_fin: Financials,
    tgt_bridge: EVBridge,
    cons: Consideration,
    assumptions: Assumptions,
    tax_rate: float,
    notes: list[str],
) -> OpeningBalanceSheet:
    """Allocate the price paid: intangibles first, deferred tax on them, goodwill last.

    The deferred tax liability is the step most models miss. A stock deal is a
    carryover-basis transaction, so the intangibles are written up for book and
    not for tax. The difference reverses over the life of the asset at the
    acquirer's rate, and the liability that represents it is assumed at close, so
    it lifts goodwill rather than reducing it.
    """
    pa = assumptions.merger.purchase_accounting

    price = cons.equity_purchase_price
    book = _target_book_equity(tgt_fin, notes)
    excess = price - book

    if excess <= 0:
        # A price below book is a bargain purchase. ASC 805 requires the buyer to
        # reassess the fair values and then take the remaining difference to
        # earnings immediately rather than carry negative goodwill. Nothing is
        # written up and nothing is amortised; the gain is not run through the
        # pro forma below, so it is flagged instead of being quietly booked.
        intangibles = 0.0
        dtl = 0.0
        goodwill = excess
        notes.append(
            f"The offer of {price:,.0f}mm is at or below the {book:,.0f}mm of book "
            "equity acquired, so there is nothing to write up. Under ASC 805 the "
            f"difference of {-excess:,.0f}mm is a bargain purchase gain taken to "
            "earnings at close once fair values have been reassessed. It is shown "
            "here as negative goodwill and is deliberately kept out of the pro forma "
            "income statement, because a one-time gain is not run-rate earnings."
        )
    else:
        intangibles = pa.intangible_pct_of_excess * excess
        dtl = intangibles * tax_rate
        goodwill = excess - intangibles + dtl

    haircut = pa.deferred_revenue_haircut * tgt_fin.deferred_revenue
    refinanced = tgt_bridge.total_debt

    opening = OpeningBalanceSheet(
        equity_purchase_price=price,
        target_book_equity=book,
        excess_purchase_price=excess,
        intangibles_created=intangibles,
        goodwill_created=goodwill,
        deferred_tax_liability=dtl,
        deferred_revenue_haircut=haircut,
        target_debt_refinanced=refinanced,
    )

    # The allocation is an identity. A residual here means the deferred tax was
    # signed the wrong way, which is the failure this construction exists to avoid.
    reconciled = book + intangibles + goodwill - dtl
    assert abs(reconciled - price) <= _BALANCE_TOLERANCE * max(1.0, abs(price)), (
        f"opening balance sheet does not reconcile: assets acquired of "
        f"{reconciled:,.6f}mm against consideration of {price:,.6f}mm"
    )

    notes.append(
        f"Intangibles of {intangibles:,.0f}mm are written up against a deferred tax "
        f"liability of {dtl:,.0f}mm at {tax_rate:.0%}, because a stock deal carries "
        f"over the seller's tax basis. Goodwill of {goodwill:,.0f}mm is the residual "
        "and includes that liability. Goodwill is not amortised under ASC 350, so it "
        "never touches EPS until an impairment takes it all at once; the intangibles "
        f"are amortised over {pa.intangible_life_years} years and touch EPS every "
        "quarter."
    )
    notes.append(
        f"Deferred revenue of {tgt_fin.deferred_revenue:,.0f}mm is written down by "
        f"{pa.deferred_revenue_haircut:.0%} to {haircut:,.0f}mm of fair value "
        "adjustment. The write-down is shown as a memo rather than inside the "
        "allocation above: a full fair value exercise would add it to identifiable "
        "net assets and take the same amount back out of goodwill, which nets to "
        "nothing for EPS. Its earnings effect, which does not net to nothing, is "
        "charged against year-one revenue below."
    )
    notes.append(
        f"{refinanced:,.0f}mm of target borrowings are assumed at close and carried "
        "on unchanged terms, so their interest stays inside the non-operating line "
        "and no funding for them appears in sources and uses. Where a change of "
        "control clause forces repayment instead, that amount has to be added to the "
        "acquisition facility and the whole walk rerun on the larger balance."
    )
    if assumptions.tax.use_effective_rate:
        notes.append(
            "The deferred tax liability is measured at the same rate the rest of this "
            "module uses, which tax.use_effective_rate has set to the acquirer's filed "
            "effective rate. ASC 740 measures deferred tax at the enacted statutory "
            "rate expected when the difference reverses, not at an effective rate that "
            "carries discrete items. The figures here stay internally consistent, so "
            "the liability still unwinds to exactly zero over the life of the asset, "
            "but the opening balance is not the one the auditors would book."
        )
    return opening


def _pro_forma_basis(
    acq_fin: Financials, tgt_fin: Financials
) -> tuple[float, float, float, float, float]:
    """Combined revenue, margins and the non-operating line, from TTM figures.

    The non-operating line is combined pre-tax income less combined EBIT, which
    is net interest expense, interest income on the cash both companies hold, and
    anything else below the operating line. It is carried as one flat dollar
    figure rather than grown with revenue, because none of it scales with sales:
    a coupon is fixed by the indenture and interest income is fixed by the cash
    balance. For a software company with several billion earning five percent it
    is a large positive number, and dropping it would understate pro forma EPS
    badly enough to change the answer.
    """
    missing = [
        label
        for label, value in (
            (f"{acq_fin.ticker} D&A", acq_fin.da),
            (f"{tgt_fin.ticker} D&A", tgt_fin.da),
            (f"{acq_fin.ticker} capital expenditure", acq_fin.capex),
            (f"{tgt_fin.ticker} capital expenditure", tgt_fin.capex),
            (f"{acq_fin.ticker} pre-tax income", acq_fin.pretax_income),
            (f"{tgt_fin.ticker} pre-tax income", tgt_fin.pretax_income),
        )
        if value is None
    ]
    if missing:
        raise MissingDataError(
            "pro forma income statement",
            hint=(
                "a multi-year pro forma needs D&A to split EBITDA from EBIT, capital "
                "expenditure to form the cash flow that repays the debt, and pre-tax "
                "income to locate the non-operating line. Missing: "
                + ", ".join(missing)
            ),
        )
    assert acq_fin.da is not None and tgt_fin.da is not None
    assert acq_fin.capex is not None and tgt_fin.capex is not None
    assert acq_fin.pretax_income is not None and tgt_fin.pretax_income is not None

    revenue = acq_fin.revenue + tgt_fin.revenue
    if revenue <= 0:
        raise NotMeaningfulError(
            "combined revenue is not positive, so margins held flat against it would "
            "project nothing"
        )
    ebitda = (acq_fin.ebit + acq_fin.da) + (tgt_fin.ebit + tgt_fin.da)
    ebit = acq_fin.ebit + tgt_fin.ebit
    capex = acq_fin.capex + tgt_fin.capex
    nonoperating = (acq_fin.pretax_income - acq_fin.ebit) + (
        tgt_fin.pretax_income - tgt_fin.ebit
    )
    return revenue, ebitda / revenue, ebit / revenue, capex / revenue, nonoperating


def build_purchase_accounting(
    acq_fin: Financials,
    tgt_fin: Financials,
    acq_bridge: EVBridge,
    tgt_bridge: EVBridge,
    cons: Consideration,
    assumptions: Assumptions,
    tax_rate: float,
) -> PurchaseAccounting:
    """Open the balance sheet, then run the combined company forward.

    **Year one is the trailing twelve months.** Growth starts in year two, so
    year one of this projection covers the same period as the year-one screen
    above and the two are directly comparable. The gap between them is exactly
    the purchase accounting: the intangible amortisation, the deferred revenue
    haircut, and a combined tax charge struck at one rate.

    **Revenue grows, margins do not.** The top line compounds at
    ``dcf.revenue_growth_start`` and the EBITDA and EBIT margins, the capital
    intensity and the non-operating line are all held where the trailing twelve
    months put them. That is a deliberate simplification and it is a generous one
    for an acquirer buying a faster-growing, lower-margin target, because the
    blended margin is held at a mix that the growth itself keeps shifting. The
    DCF fades its growth rate; nothing here does.

    **Taxes.** The combined pre-tax result is taxed at one rate, the acquirer's,
    which is what a pro forma income statement does and what the rest of this
    module assumes. Where either filer's reported effective rate differs from
    that rate, year one here will not reproduce the headline walk exactly, since
    the walk starts from reported net income and this starts from revenue. No
    loss carryforwards are tracked, so a pro forma loss books a full tax benefit
    at the marginal rate.

    **The sweep.** Free cash flow is net income, plus the after-tax intangible
    amortisation, plus existing D&A, less capital expenditure. Only ``(1 - t)`` of
    the amortisation comes back because the deferred tax benefit inside net income
    is not cash. A configured share of that cash flow repays the acquisition
    facility at the end of each year, and the next year's interest is charged on
    what is left.
    """
    m = assumptions.merger
    pa = m.purchase_accounting
    syn = m.synergies
    notes: list[str] = []
    checks: list[str] = []
    after_tax = 1.0 - tax_rate

    opening = _opening_balance_sheet(
        tgt_fin, tgt_bridge, cons, assumptions, tax_rate, notes
    )
    basis = _pro_forma_basis(acq_fin, tgt_fin)
    base_revenue, ebitda_margin, ebit_margin, capex_pct, nonoperating = basis
    da_pct = ebitda_margin - ebit_margin

    growth = assumptions.dcf.revenue_growth_start
    shares = acq_fin.diluted_shares + cons.new_shares_issued
    synergy_pretax = syn.pretax_cost_synergies + syn.pretax_revenue_synergies
    annual_amortisation = opening.intangibles_created / pa.intangible_life_years
    # The yield given up on cash spent at close is permanent: the cash is gone,
    # so it is charged in every projected year, not only in year one.
    foregone = cons.cash_used * m.foregone_cash_yield
    fee_amortisation = (
        m.financing_fees / m.financing_fee_amort_years if m.financing_fees else 0.0
    )

    years: list[ProFormaYear] = []
    debt = cons.new_debt
    for y in range(1, pa.projection_years + 1):
        compounding = (1.0 + growth) ** (y - 1)
        haircut = opening.deferred_revenue_haircut if y == 1 else 0.0
        revenue = base_revenue * compounding - haircut
        synergy = synergy_pretax * (syn.phase_in_year_one if y == 1 else 1.0)
        amortisation = annual_amortisation if y <= pa.intangible_life_years else 0.0

        # Applying the margin to revenue already net of the haircut takes the
        # margin on the destroyed revenue out of EBITDA with it. The blended
        # margin is the mild version of that charge: deferred revenue is written
        # down to cost plus a normal margin, so what the haircut actually removes
        # is close to a full-margin dollar of the target's revenue.
        ebitda = revenue * ebitda_margin + synergy
        da = revenue * da_pct
        ebit = ebitda - da - amortisation
        interest = debt * m.cost_of_new_debt
        fee = fee_amortisation if y <= m.financing_fee_amort_years else 0.0

        pretax = ebit + nonoperating - foregone - fee - interest
        taxes = pretax * tax_rate
        net_income = pretax - taxes
        eps = net_income / shares

        cash_flow = net_income + amortisation * after_tax + da - revenue * capex_pct
        repaid = min(max(cash_flow, 0.0) * pa.debt_repayment_pct_of_fcf, debt)
        debt -= repaid

        years.append(
            ProFormaYear(
                year=y,
                revenue=revenue,
                ebitda=ebitda,
                intangible_amortisation=amortisation,
                ebit=ebit,
                interest=interest,
                pretax_income=pretax,
                taxes=taxes,
                net_income=net_income,
                shares=shares,
                eps=eps,
                debt_balance=debt,
                cash_flow=cash_flow,
                debt_repaid=repaid,
            )
        )

    # The acquirer standalone, on identical conventions: same growth, same flat
    # margins, same single tax rate. Only that way is the difference between the
    # two paths the deal rather than the bookkeeping. At year one it reproduces
    # reported EPS whenever the filed effective rate equals the rate in use.
    acq_nonoperating = acq_fin.pretax_income - acq_fin.ebit
    standalone_ttm_eps = acq_fin.net_income / acq_fin.diluted_shares
    in_percent = standalone_ttm_eps >= _MIN_EPS_FOR_PERCENT

    accretion_by_year: list[tuple[int, float, float | None]] = []
    standalone_eps_path: list[float] = []
    for year in years:
        compounding = (1.0 + growth) ** (year.year - 1)
        standalone_pretax = (
            acq_fin.revenue * compounding * acq_fin.ebit_margin + acq_nonoperating
        )
        standalone_eps = standalone_pretax * after_tax / acq_fin.diluted_shares
        standalone_eps_path.append(standalone_eps)
        dollars = year.eps - standalone_eps
        pct: float | None = None
        if in_percent:
            try:
                pct = _accretion_percent(dollars, standalone_eps)
            except NotMeaningfulError:
                pct = None
        accretion_by_year.append((year.year, dollars, pct))

    notes.append(
        f"Revenue compounds at {growth:.1%}, the DCF's year-one growth rate, with "
        "every margin, the capital intensity and the non-operating line held where "
        "the trailing twelve months put them. Year one carries no growth, so it "
        "covers the same period as the year-one screen and the two are comparable. "
        "No fade, no working capital movement and no buyback: the only capital "
        "action modelled after close is the debt sweep."
    )
    notes.append(
        f"Cash not swept to the facility, {1.0 - pa.debt_repayment_pct_of_fcf:.0%} of "
        "free cash flow, piles up earning nothing in this model. That understates "
        "later years by the interest it would earn, and it leaves the sweep "
        "percentage moving interest expense rather than net debt."
    )
    notes.append(
        "The combined pre-tax result is taxed at one rate throughout, so year one "
        "here will not tie exactly to the headline walk unless both filers' reported "
        "effective rates equal that rate. The walk starts from reported net income; "
        "this starts from revenue."
    )
    notes.append(
        "Accretion by year is quoted in "
        + (
            "both dollars per share and percent of standalone EPS"
            if in_percent
            else "dollars per share only. Standalone EPS of "
            f"{standalone_ttm_eps:,.2f} is not a base worth dividing by, on the same "
            "test the headline figure applies, so no percentage is shown for any year"
        )
        + "."
    )
    if m.include_intangible_amortization:
        notes.append(
            f"merger.intangible_step_up of {m.intangible_step_up:,.0f}mm is the "
            "hand-entered figure the year-one screen charges. It is ignored here: the "
            f"write-up in this projection is the {opening.intangibles_created:,.0f}mm "
            "derived from the price actually paid. Where the two differ, the derived "
            "figure is the one the opening balance sheet supports."
        )

    checks.extend(
        _purchase_accounting_checks(
            acq_fin,
            tgt_fin,
            acq_bridge,
            tgt_bridge,
            cons,
            opening,
            years,
            accretion_by_year,
            standalone_eps_path,
            assumptions,
            tax_rate,
        )
    )
    return PurchaseAccounting(
        opening=opening,
        years=years,
        accretion_by_year=accretion_by_year,
        notes=notes,
        checks=checks,
    )


def _purchase_accounting_checks(
    acq_fin: Financials,
    tgt_fin: Financials,
    acq_bridge: EVBridge,
    tgt_bridge: EVBridge,
    cons: Consideration,
    opening: OpeningBalanceSheet,
    years: list[ProFormaYear],
    accretion_by_year: list[tuple[int, float, float | None]],
    standalone_eps_path: list[float],
    assumptions: Assumptions,
    tax_rate: float,
) -> list[str]:
    """The four questions a reader asks of an opening balance sheet."""
    checks: list[str] = []
    pa = assumptions.merger.purchase_accounting
    horizon = years[-1]

    price = opening.equity_purchase_price
    checks.append(
        f"Goodwill of {opening.goodwill_created:,.0f}mm is "
        f"{opening.goodwill_created / price:.0%} of the equity purchase price, with "
        f"{opening.intangibles_created:,.0f}mm "
        f"({opening.intangibles_created / price:.0%}) in identifiable intangibles and "
        f"{opening.target_book_equity / price:.0%} in book equity acquired. Goodwill "
        "that large is the whole of the deal thesis sitting on one line, tested for "
        "impairment once a year and written off in a single quarter when it fails."
    )

    turns = next((y for y, dollars, _ in accretion_by_year if dollars > 0), None)
    if turns == 1:
        checks.append(
            f"Accretive from year one, at {accretion_by_year[0][1]:+,.3f} per share "
            "after the intangible amortisation and the deferred revenue haircut."
        )
    elif turns is not None:
        checks.append(
            f"Dilutive in year one at {accretion_by_year[0][1]:+,.3f} per share and "
            f"crossing into accretion in year {turns} at "
            f"{accretion_by_year[turns - 1][1]:+,.3f}. The crossover is driven by the "
            "debt sweep taking interest down and by synergies reaching their run rate, "
            "not by the amortisation, which is flat until the asset is written off."
        )
    else:
        checks.append(
            f"The deal does not turn accretive inside the {pa.projection_years} year "
            f"horizon: year {horizon.year} is still "
            f"{accretion_by_year[-1][1]:+,.3f} per share dilutive. Extending the "
            "horizon only helps once the amortisation runs off in year "
            f"{pa.intangible_life_years + 1}, and a crossover that far out is a "
            "synergy assumption, not a result."
        )

    # Leverage is quoted on whichever earnings figure the lease convention allows,
    # the same rule the headline check follows. Under the lease-inclusive
    # convention net debt carries the lease liabilities, so the denominator has to
    # be taken before rent or the same obligation is counted twice.
    rent = 0.0
    label = "EBITDA"
    quotable = True
    if acq_bridge.operating_lease_in_debt or tgt_bridge.operating_lease_in_debt:
        label = "EBITDAR"
        if acq_fin.operating_lease_cost is None or tgt_fin.operating_lease_cost is None:
            quotable = False
        else:
            rent = acq_fin.operating_lease_cost + tgt_fin.operating_lease_cost

    net_debt_close = (
        _net_debt(acq_bridge) + _net_debt(tgt_bridge) + cons.new_debt + cons.cash_used
    )
    # Every dollar of free cash flow reduces net debt, whether it repays the
    # facility or sits in the bank. The sweep decides how much interest is saved,
    # not how levered the company is.
    net_debt_horizon = net_debt_close - sum(y.cash_flow for y in years)
    open_den = years[0].ebitda + rent
    horizon_den = horizon.ebitda + rent
    if not quotable:
        checks.append(
            "Pro forma leverage is not shown: operating leases are counted as debt, so "
            "the denominator has to be taken before rent, and one of the two filers "
            "does not report an operating lease cost for EBITDAR to be formed."
        )
    elif open_den > 0 and horizon_den > 0:
        # The direction is read off the figures rather than assumed. A combined
        # company that burns cash levers up over the horizon, and a check that
        # says "falling" on the way to a worse number is worse than no check. A
        # negative balance is quoted as net cash, on the headline check's wording.
        direction = "falling" if net_debt_horizon < net_debt_close else "rising"
        at_close = (
            f"net debt of {net_debt_close:,.0f}mm"
            if net_debt_close >= 0
            else f"net cash of {-net_debt_close:,.0f}mm"
        )
        at_horizon = (
            f"net debt of {net_debt_horizon:,.0f}mm"
            if net_debt_horizon >= 0
            else f"net cash of {-net_debt_horizon:,.0f}mm"
        )
        checks.append(
            f"Pro forma {at_close} at close is {net_debt_close / open_den:,.1f}x "
            f"{label}, {direction} to {at_horizon} and "
            f"{net_debt_horizon / horizon_den:,.1f}x by year {horizon.year}. The "
            f"acquisition facility itself runs from {cons.new_debt:,.0f}mm to "
            f"{horizon.debt_balance:,.0f}mm; the rest of the move is cash retained on "
            "the balance sheet, or drawn down to fund losses."
        )
    else:
        checks.append(
            f"Pro forma leverage is not shown: combined {label} is not positive in "
            "every year of the horizon, and a debt multiple of a negative number is "
            "not a leverage ratio."
        )

    first = years[0]
    if accretion_by_year[0][1] < 0 and first.intangible_amortisation > 0:
        per_share_charge = (
            first.intangible_amortisation * (1.0 - tax_rate) / first.shares
        )
        cash_accretion = first.eps + per_share_charge - standalone_eps_path[0]
        if cash_accretion > 0:
            checks.append(
                "The whole of the year-one dilution is the intangible amortisation. "
                f"Excluding it, {first.intangible_amortisation:,.0f}mm pre-tax and "
                f"{per_share_charge:,.3f} per share after tax, the deal is "
                f"{cash_accretion:+,.3f} accretive. That is the case for quoting cash "
                "EPS: the charge is non-cash, it has no tax deduction behind it in a "
                "stock deal, and it runs off on a schedule set at close rather than by "
                "how the business performs."
            )
        else:
            checks.append(
                f"Year-one dilution of {accretion_by_year[0][1]:+,.3f} per share is "
                "not explained by the intangible amortisation. Excluding that charge "
                f"the deal is still {cash_accretion:+,.3f} per share dilutive, so cash "
                "EPS does not rescue it and the price, the funding mix or the "
                "synergies have to move."
            )
    return checks


def run_merger(
    acq_fin: Financials,
    tgt_fin: Financials,
    acq_bridge: EVBridge,
    tgt_bridge: EVBridge,
    acq_price: float,
    tgt_price: float,
    assumptions: Assumptions,
) -> MergerResult:
    """Screen one deal: price it, fund it, and run year-one EPS.

    ``acq_price`` and ``tgt_price`` are unaffected prices. Both bridges must have
    been built under the same assumptions, which is what keeps the net debt
    assumed here and the enterprise values quoted elsewhere on one definition.
    """
    notes: list[str] = []
    tax_rate = _tax_rate(acq_fin, assumptions, notes)

    # The consideration prices the target's balance sheet at the OFFER, not at
    # the unaffected market price, because moneyness at the offer is what
    # decides whether a convertible converts in the deal. Passing no bridge lets
    # build_consideration apply its documented rule; the market-priced bridge
    # the caller supplied still serves the standalone comparisons below.
    cons = build_consideration(
        acq_fin, tgt_fin, acq_price, tgt_price, assumptions, None, notes
    )
    sources_uses = _sources_and_uses(cons, assumptions)
    accretion = _run_accretion(acq_fin, tgt_fin, cons, assumptions, tax_rate)
    contribution = _contribution(acq_fin, tgt_fin, notes)
    ownership = {
        "acquirer": acq_fin.diluted_shares / accretion.pro_forma_shares,
        "target": cons.new_shares_issued / accretion.pro_forma_shares,
    }
    sensitivity = accretion_sensitivity(
        acq_fin, tgt_fin, tgt_bridge, acq_price, tgt_price, assumptions, tax_rate
    )
    checks = _rule_of_thumb_checks(
        acq_fin,
        tgt_fin,
        acq_bridge,
        cons,
        assumptions,
        tax_rate,
        accretion.accretion_dollars,
    )

    total_uses = next(v for side, label, v in sources_uses if label == "Total uses")
    checks.append(
        f"Sources and uses balance at {total_uses:,.0f}mm: "
        f"{cons.new_debt:,.0f}mm of new debt, {cons.cash_used:,.0f}mm of balance "
        f"sheet cash and {cons.stock_consideration:,.0f}mm of stock against "
        f"{cons.equity_purchase_price:,.0f}mm of equity purchased plus "
        f"{assumptions.merger.deal_fees + assumptions.merger.financing_fees:,.0f}mm "
        "of fees."
    )

    # Leverage is quoted on whichever earnings figure the lease convention
    # permits. The bridge owns that decision, so it is asked rather than assumed.
    acq_den, den_label = acq_bridge.multiple_denominator(acq_fin)
    tgt_den, _ = tgt_bridge.multiple_denominator(tgt_fin)
    if acq_den is not None and tgt_den is not None:
        syn = assumptions.merger.synergies
        pf_earnings = (
            acq_den
            + tgt_den
            + (syn.pretax_cost_synergies + syn.pretax_revenue_synergies)
            * syn.phase_in_year_one
        )
        # Cash spent on the deal leaves the balance sheet, so it RAISES net debt.
        # Post-deal debt is D_acq + D_tgt + new debt; post-deal cash is
        # C_acq + C_tgt - cash used. Net debt is the difference, which puts cash
        # used on the same side as new borrowing, not against it.
        pf_net_debt = (
            _net_debt(acq_bridge)
            + _net_debt(tgt_bridge)
            + cons.new_debt
            + cons.cash_used
        )
        if pf_earnings > 0:
            position = (
                f"net debt of {pf_net_debt:,.0f}mm"
                if pf_net_debt >= 0
                else f"net cash of {-pf_net_debt:,.0f}mm"
            )
            line = (
                f"Pro forma {position} is {pf_net_debt / pf_earnings:,.1f}x pro forma "
                f"{den_label}"
            )
            if acq_den > 0:
                line += f", against {_net_debt(acq_bridge) / acq_den:,.1f}x standalone"
            checks.append(line + ".")
        else:
            checks.append(
                f"Pro forma leverage is not shown: combined {den_label} of "
                f"{pf_earnings:,.0f}mm is not positive, so a debt multiple of it "
                "would be a negative number pretending to be a leverage ratio."
            )

    ni_row = next((r for r in contribution if r.metric == "Net income"), None)
    if ni_row is not None and ni_row.target_pct is not None:
        checks.append(
            f"{tgt_fin.ticker} takes {ownership['target']:.1%} of the pro forma "
            f"share count for {ni_row.target_pct:.1%} of combined net income. Its "
            "owners receive "
            + ("more" if ownership["target"] > ni_row.target_pct else "less")
            + " of the combined company than the earnings they bring. On an "
            "all-stock deal that is the same statement as the EPS effect above, "
            "read off the share register. Where cash funds part of the price the "
            "two diverge, because cash buys earnings without issuing the shares "
            "that would show up in ownership."
        )

    notes.append(
        f"Exchange ratio: an offer of {cons.offer_price:,.2f} against "
        f"{acq_fin.ticker} at {acq_price:,.2f} is {cons.exchange_ratio:,.4f} acquirer "
        f"shares per target share. On {tgt_fin.diluted_shares:,.1f}mm target shares a "
        f"full stock deal would print {cons.exchange_ratio * tgt_fin.diluted_shares:,.1f}mm "
        f"new shares; at {cons.pct_stock:.0%} stock it prints "
        f"{cons.new_shares_issued:,.1f}mm."
    )
    notes.append(
        "Target shares are the trailing twelve month diluted weighted average. What "
        "an acquirer actually buys is the fully diluted count at close, after change "
        "of control vesting of options and restricted stock and after any "
        "convertible settles. For a company still issuing equity every quarter the "
        "average understates that count, so the equity purchase price here is a "
        "floor, not a bid."
    )
    # Purchase accounting is additive. Everything above this line is computed the
    # same way whether the switch is on or off, so a run with it on and a run with
    # it off can be compared line by line.
    purchase_accounting: PurchaseAccounting | None = None
    if assumptions.merger.purchase_accounting.enabled:
        purchase_accounting = build_purchase_accounting(
            acq_fin, tgt_fin, acq_bridge, tgt_bridge, cons, assumptions, tax_rate
        )
        notes.append(
            "Purchase accounting is on. The opening balance sheet, the intangible "
            "write-up and the deferred tax on it, the deferred revenue haircut and a "
            f"{assumptions.merger.purchase_accounting.projection_years} year pro forma "
            "with a debt sweep are in purchase_accounting. The year-one figures above "
            "are unchanged by it and stay on a cash EPS basis, so the difference "
            "between the two is the purchase accounting itself."
        )
    else:
        notes.append(
            "Not modelled: the opening balance sheet, goodwill, deferred tax on any "
            "step-up, the deferred revenue haircut, debt paydown and buybacks after "
            "year one, and the transaction structure. This is year-one EPS on two sets "
            "of trailing figures. It screens deals; it does not price one."
        )

    return MergerResult(
        acquirer=acq_fin,
        target=tgt_fin,
        consideration=cons,
        accretion=accretion,
        contribution=contribution,
        pro_forma_ownership=ownership,
        sources_uses=sources_uses,
        sensitivity=sensitivity,
        checks=checks,
        notes=notes,
        purchase_accounting=purchase_accounting,
    )
