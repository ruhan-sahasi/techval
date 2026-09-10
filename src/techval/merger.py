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

**What is not modelled.** No opening balance sheet, no goodwill, no deferred tax
liability on the step-up, no deferred revenue haircut, no debt paydown or share
buyback over time, and no purchase-accounting step-up amortisation unless the
config asks for it. Year-one EPS built on two sets of trailing twelve month
figures is a screening tool that tells you the shape of a deal and roughly what
it costs. It is not a merger model, and no one should take a bid to a board on it.
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
class MergerResult:
    """A full accretion/(dilution) screen.

    ``acquirer`` and ``target`` are the Financials the run was built on, so a
    renderer has the tickers and the as-of dates without carrying them separately.
    ``sources_uses`` is a list of (side, label, amount) with side in {"Use",
    "Source", "Total"}. ``pro_forma_ownership`` maps "acquirer" and "target" to
    their share of the pro forma count.
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
            cons = build_consideration(
                acq_fin, tgt_fin, acq_price, tgt_price, cell, tgt_bridge
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

    cons = build_consideration(
        acq_fin, tgt_fin, acq_price, tgt_price, assumptions, tgt_bridge, notes
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
        pf_net_debt = (
            _net_debt(acq_bridge) + _net_debt(tgt_bridge) + cons.new_debt - cons.cash_used
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
            + " of the combined company than the earnings they bring, which is the "
            "same statement as the EPS effect above, read off the share register."
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
    )
