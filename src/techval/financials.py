"""Normalized financial statements on a trailing-twelve-month basis.

Everything downstream reads this object and nothing else. Money is USD millions,
share counts are millions of shares, per-share figures are dollars.

Two accounting positions are taken here and are worth stating plainly, because
they drive the enterprise value bridge and the multiples built on it.

**EBITDA is after operating lease cost.** Under ASC 842 a US filer keeps its
operating lease expense inside operating income as a single straight-line rent
charge. It is not split into depreciation and interest the way IFRS 16 requires,
so it is not in the D&A that gets added back. US-GAAP EBITDA is therefore a
post-rent number. ``ebitdar`` is published alongside it for the case where lease
liabilities are treated as debt, and the EV bridge refuses to mix the two.

**Convertible notes are resolved separately from straight debt.** Whether they
are debt or equity depends on where their shares already sit, which is a bridge
decision, not a statement one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from . import tags
from .edgar import CompanyFacts, EdgarClient, Provenance
from .errors import MissingDataError, StaleDataError


@dataclass
class Financials:
    """Normalized TTM statements plus the latest balance sheet, in USD millions."""

    ticker: str
    entity_name: str
    cik: int
    as_of: date

    # Income statement, trailing twelve months
    revenue: float
    gross_profit: float | None
    ebit: float
    da: float | None
    sbc: float | None
    net_income: float
    pretax_income: float | None
    tax_expense: float | None
    interest_expense: float | None
    operating_lease_cost: float | None

    # Cash flow, trailing twelve months
    capex: float | None
    cfo: float | None

    # Shares, day-weighted over the same twelve months
    diluted_shares: float
    basic_shares: float | None

    # Balance sheet, latest reported instant
    cash: float
    short_term_investments: float
    straight_debt: float
    convertible_debt: float
    operating_lease_liability: float
    finance_lease_liability: float
    nci: float
    preferred: float
    current_assets: float | None
    current_liabilities: float | None
    deferred_revenue: float

    # A point-in-time share count, when one has been built. ``diluted_shares``
    # stays as filed so the two can always be compared and so the EPS tie-out
    # control keeps comparing like with like.
    valuation_shares: float | None = None

    # The finance lease liability that is already inside ``straight_debt``,
    # because the filer tags its debt line with a concept whose name is "debt and
    # capital lease obligations". It is reported here and excluded from
    # ``finance_lease_liability`` so that every consumer which adds the two
    # together stays right without knowing about the case.
    finance_lease_inside_debt: float = 0.0

    # The us-gaap concepts ``straight_debt`` was built from: the tuple for code to
    # test, the clause for a report to print. A debt figure that cannot be traced
    # to a tag is the one number in this object most likely to be silently wrong,
    # so both travel with it.
    debt_tags: tuple[str, ...] = ()
    debt_basis: str = ""

    provenance: dict[str, Provenance] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # -- derived ----------------------------------------------------------- #

    @property
    def ebitda(self) -> float | None:
        """EBIT plus D&A. After operating lease cost, per ASC 842."""
        return None if self.da is None else self.ebit + self.da

    @property
    def ebitdar(self) -> float | None:
        """EBITDA before operating lease cost.

        The only earnings figure that may be paired with an enterprise value
        which counts lease liabilities as debt.
        """
        if self.ebitda is None or self.operating_lease_cost is None:
            return None
        return self.ebitda + self.operating_lease_cost

    @property
    def ebitr(self) -> float | None:
        """EBIT before operating lease cost.

        The EBIT counterpart of ``ebitdar``, and the only operating profit figure
        that may be paired with an enterprise value counting lease liabilities as
        debt.
        """
        if self.operating_lease_cost is None:
            return None
        return self.ebit + self.operating_lease_cost

    @property
    def ebitda_margin(self) -> float | None:
        e = self.ebitda
        return None if e is None or not self.revenue else e / self.revenue

    @property
    def ebit_margin(self) -> float:
        return self.ebit / self.revenue if self.revenue else 0.0

    @property
    def gross_margin(self) -> float | None:
        if self.gross_profit is None or not self.revenue:
            return None
        return self.gross_profit / self.revenue

    @property
    def effective_tax_rate(self) -> float | None:
        """Filed effective rate, or None when it is not economically meaningful.

        A negative pre-tax result, or a tax benefit against a profit, produces a
        rate that describes valuation allowances and discrete items rather than
        the burden on an incremental dollar. Those cases return None so callers
        fall back to the configured marginal rate rather than quietly using a
        nonsense number.
        """
        if self.pretax_income is None or self.tax_expense is None:
            return None
        if self.pretax_income <= 0:
            return None
        rate = self.tax_expense / self.pretax_income
        return rate if 0.0 <= rate <= 0.60 else None

    @property
    def total_debt_ex_leases(self) -> float:
        """Borrowings excluding lease liabilities, as far as the tags allow.

        ``finance_lease_inside_debt`` is the exception and it is not stripped
        out: where a filer tags its debt line as "debt and capital lease
        obligations" there is no split to remove, and inventing one by
        subtracting a separately tagged lease would assume the two footnotes
        reconcile. The amount is reported on the object so a reader can see how
        much of this figure is lease rather than borrowing.
        """
        return self.straight_debt + self.convertible_debt

    @property
    def net_working_capital(self) -> float | None:
        """Non-cash working capital.

        Cash and short-term investments come out of current assets because they
        are financing, not operations, and are already counted in the EV bridge.
        Leaving them in would let a cash pile masquerade as operating capital.

        Current liabilities are taken whole, which means any current debt and
        current lease liability sit inside this figure even though they are
        financing too. Stripping them would need those balances at the same
        instant and they are resolved for the bridge, not here. The DCF does not
        read this property, it projects working capital as a share of revenue, so
        the inconsistency does not reach a valuation; it matters only if you use
        this figure directly.
        """
        if self.current_assets is None or self.current_liabilities is None:
            return None
        return (self.current_assets - self.cash - self.short_term_investments) - (
            self.current_liabilities
        )

    @property
    def shares_for_valuation(self) -> float:
        """The count equity value and per-share figures divide by.

        The treasury-stock count when one has been built, otherwise trailing
        diluted weighted-average shares. Keeping both on the object rather than
        overwriting one with the other means a reader can always see which was
        used and what the other would have given.
        """
        return self.valuation_shares or self.diluted_shares

    @property
    def eps_diluted(self) -> float:
        return self.net_income / self.diluted_shares if self.diluted_shares else 0.0

    def provenance_rows(self) -> list[dict[str, str]]:
        return [p.as_row() for p in self.provenance.values()]


_MM = 1e6

# How far a balance-sheet fact may sit from the balance-sheet date and still be
# read as describing it. Mirrors ``CompanyFacts.resolve_instant``'s own default,
# and is repeated here rather than imported because the controls need to ask the
# same question of concepts the ladders never resolve from.
_INSTANT_TOLERANCE_DAYS = 20

# A revenue tag reporting more than ten times another tag's figure for the same
# window is the total and the other is a component. Ten rather than two: the
# genuine scope disagreements in the TMT universe, revenue before or after
# billable expenses at Interpublic, services revenue against total at Starz, run
# two to five times and picking the larger of those by rule would be a thumb on
# the scale. See CompanyFacts.resolve_ttm.
_REVENUE_COMPONENT_GUARD = 10.0


@dataclass
class _DebtLadder:
    """What the three debt ladders resolved, and out of which tags.

    Kept as a record rather than a bare float because two downstream decisions
    need the winning tag and not just its value: whether the figure already
    contains finance leases, and whether the resolution is complete enough to
    stand beside a balance sheet at all.
    """

    total: float = 0.0
    noncurrent: float = 0.0
    current: float = 0.0
    noncurrent_tag: str | None = None
    current_tag: str | None = None
    combined_tag: str | None = None

    @property
    def source_tags(self) -> list[str]:
        if self.combined_tag:
            return [self.combined_tag]
        return [t for t in (self.noncurrent_tag, self.current_tag) if t]

    @property
    def _bundles_leases(self) -> bool:
        """Whether this filer demonstrably puts finance leases inside debt.

        True when any leg resolved from a concept whose name says so. It is a
        property of the filer rather than of the leg, which is what lets an
        ambiguous concept on the other leg be read the same way: Micron tags its
        non-current line ``LongTermDebtAndCapitalLeaseObligations`` and its
        current line ``DebtCurrent``, and its current line is finance lease to
        the last dollar.
        """
        return any(tag in tags.INCLUDES_FINANCE_LEASES for tag in self.source_tags)

    def _swallows(self, tag: str | None) -> bool:
        if tag in tags.INCLUDES_FINANCE_LEASES:
            return True
        return self._bundles_leases and tag in tags.MAY_INCLUDE_FINANCE_LEASES

    @property
    def swallows_noncurrent_lease(self) -> bool:
        return self._swallows(self.combined_tag or self.noncurrent_tag)

    @property
    def swallows_current_lease(self) -> bool:
        return self._swallows(self.combined_tag or self.current_tag)

    def describe(self) -> str:
        """One clause naming the tags the figure came from, for the bridge."""
        if not self.source_tags:
            return "no us-gaap debt concept resolved at the balance-sheet date"
        leases = (
            ", finance leases inside it"
            if self.swallows_noncurrent_lease or self.swallows_current_lease
            else ""
        )
        return " plus ".join(self.source_tags) + leases


def _explicitly_zero(facts: CompanyFacts, as_of: date, tag: str) -> bool:
    """Whether the filer states a zero balance for this concept AT this date.

    Distinct from the concept being absent. An absent tag is silence and a
    reported zero is an assertion, and the difference decides whether a
    total-by-definition concept is being used as a total or as a non-current
    line.
    """
    at_date = [
        f
        for f in facts.facts(tag)
        if f.is_instant
        and f.end <= as_of
        and (as_of - f.end).days <= _INSTANT_TOLERANCE_DAYS
    ]
    return bool(at_date) and at_date[-1].val == 0


def _straight_debt(
    instant, prov: dict, facts: CompanyFacts, as_of: date
) -> _DebtLadder:
    """Borrowings, from the balance-sheet split, or a combined tag if that is all.

    The precedence is a judgment and not a lookup, because a combined tag
    overlaps its own components. Verizon at 2025-12-31 reports
    ``LongTermDebtAndCapitalLeaseObligations`` 139,532mm, ``DebtCurrent``
    18,618mm and ``LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities``
    157,709mm for the same balance sheet; adding all three triple counts the
    current maturities and returns roughly twice the debt the company has.

    So the split wins whenever both of its legs resolve, and the combined ladder
    is consulted in two cases only.

    *Neither leg resolved.* Some filers publish only a total and never the pair.
    Resolving the pair alone returns zero for them, which reads as a debt-free
    company and understates enterprise value by the whole balance.

    *One leg resolved and the other did not.* This is the case that produced the
    Verizon error and it is the reason the old rule was not enough. The old code
    consulted the combined ladder only when the split was completely empty, so a
    filer whose current portion resolved and whose long-term portion did not
    returned the current portion alone and looked entirely healthy doing it:
    Verizon printed 21,783mm of debt against 165,231mm on the face of its Q2 2026
    balance sheet. Where one leg is missing the combined figure is the better
    answer whenever it is larger, and it is taken INSTEAD of the split rather
    than added to it, so the overlap can never be counted twice.

    Nothing here fabricates the missing leg. A filer that reports no debt concept
    at all still resolves zero, which for Datadog is the truth; ``_debt_control``
    is what separates a true zero from a lost one.
    """

    def pick(concept: str, ladder) -> tuple[float, str | None]:
        value = instant(concept, ladder, default=0.0) or 0.0
        return value, prov[concept].tag

    noncurrent, nc_tag = pick("long-term debt", tags.DEBT_NONCURRENT)
    current, cur_tag = pick("current debt", tags.DEBT_CURRENT)
    split = noncurrent + current

    # ``LongTermDebt`` is a non-current line for Adobe and Qualcomm and a total
    # for Microsoft and Dell, and the taxonomy does not settle which. Where the
    # filer states a non-current balance of ZERO at the same date the figure can
    # only be a total, and adding a current leg to it would report the debt
    # twice. See ``tags.DEBT_TOTAL_BY_DEFINITION``.
    if nc_tag in tags.DEBT_TOTAL_BY_DEFINITION and any(
        _explicitly_zero(facts, as_of, tag) for tag in tags.DEBT_STRICTLY_NONCURRENT
    ):
        return _DebtLadder(max(noncurrent, current), combined_tag=nc_tag)

    if nc_tag and cur_tag:
        return _DebtLadder(split, noncurrent, current, nc_tag, cur_tag)

    combined, comb_tag = pick("total debt", tags.DEBT_COMBINED)
    if comb_tag and combined > split:
        return _DebtLadder(combined, combined_tag=comb_tag)
    return _DebtLadder(split, noncurrent, current, nc_tag, cur_tag)


def _lease(
    instant, prov: dict, label: str, noncurrent, current, combined
) -> tuple[float, float]:
    """A lease liability as (non-current, current), from its split or its total.

    Same precedence as the debt ladder and for the same reason: a combined figure
    overlaps its own components and must replace them rather than join them.
    Microsoft's 2026 10-K tags ``FinanceLeaseLiability`` and neither of the split
    concepts, so a split-only resolution reads its 66,594mm of finance leases as
    zero, against 40,294mm of straight debt.

    A combined figure cannot be split back apart, so it is returned whole against
    the non-current leg. That matters only where the debt ladder swallows one leg
    and not the other, and attributing an unsplit lease to the long end is the
    conservative reading of a balance whose weighted average term runs to years.
    """
    nc = instant(f"{label}, non-current", noncurrent, default=0.0) or 0.0
    cur = instant(f"{label}, current", current, default=0.0) or 0.0
    if prov[f"{label}, non-current"].tag and prov[f"{label}, current"].tag:
        return nc, cur
    total = instant(f"{label}, total", combined, default=0.0) or 0.0
    return (total, 0.0) if total > nc + cur else (nc, cur)


def build_financials(
    ticker: str,
    *,
    client: EdgarClient | None = None,
    facts: CompanyFacts | None = None,
) -> Financials:
    """Assemble normalized TTM statements for one filer."""
    if facts is None:
        if client is None:
            client = EdgarClient()
        facts = client.company_facts(ticker)

    prov: dict[str, Provenance] = {}
    warnings: list[str] = []

    def flow(
        concept: str,
        ladder,
        *,
        required=True,
        default=None,
        annual_ok=False,
        kind="flow",
        scale=_MM,
        component_guard=None,
    ):
        val, p = facts.resolve_ttm(
            concept,
            ladder,
            as_of,
            kind=kind,
            required=required,
            default_when_absent=default,
            allow_annual_fallback=annual_ok,
            component_guard=component_guard,
        )
        prov[concept] = p
        if p.method.startswith("latest filed fiscal year"):
            warnings.append(
                f"{concept}: only reported annually, so the figure shown is "
                f"{p.periods[0] if p.periods else 'the latest fiscal year'}, not "
                "the trailing twelve months"
            )
        return None if val is None else val / scale

    def instant(concept: str, ladder, *, required=True, default=None, scale=_MM):
        val, p = facts.resolve_instant(
            concept, ladder, as_of, required=required, default_when_absent=default
        )
        prov[concept] = p
        return None if val is None else val / scale

    # The balance-sheet date is the end of the most recent period for which the
    # filer reported revenue. Anchoring on revenue rather than on any balance
    # item keeps the income statement and the balance sheet on the same filing.
    # The latest period end across EVERY revenue tag in the ladder, not just the
    # first one that has any facts at all. A filer that migrated from one revenue
    # tag to another leaves the old tag in place forever, and anchoring on it
    # would date the whole valuation to whenever that tag was retired, then pull
    # a balance sheet to match. The result looks entirely normal and is years stale.
    ends = [
        f.end
        for tag in tags.REVENUE
        for f in facts.facts(tag)
        if not f.is_instant
    ]
    if not ends:
        raise MissingDataError(
            "revenue",
            ticker=ticker,
            tags_tried=list(tags.REVENUE),
            hint="no revenue tag in the ladder is reported by this filer",
        )
    as_of = max(ends)

    # Revenue takes the component guard. It is the only concept in this
    # statement that needs one and the only one where the test means anything:
    # revenue is strictly positive for every filer this engine values, so an
    # order-of-magnitude gap between two ladder entries covering the same window
    # is a scope error rather than a disagreement. Charter tags
    # RevenueFromContractWithCustomerIncludingAssessedTax at 889mm for fiscal
    # 2025 against 54,774mm of Revenues, and the ladder ranks the smaller first.
    # Nothing below the top line gets the guard: EBIT and net income pass
    # through zero, and a ratio test on a concept that passes through zero fires
    # at random.
    revenue = flow("revenue", tags.REVENUE, component_guard=_REVENUE_COMPONENT_GUARD)
    ebit = flow("EBIT", tags.EBIT)
    assert revenue is not None and ebit is not None

    # Debt and leases are resolved together because the answer to one changes the
    # other: a debt tag named "debt and capital lease obligations" already carries
    # the lease the lease ladder is about to resolve.
    debt = _straight_debt(instant, prov, facts, as_of)
    finance_lease_nc, finance_lease_cur = _lease(
        instant,
        prov,
        "finance lease liability",
        tags.FINANCE_LEASE_NONCURRENT,
        tags.FINANCE_LEASE_CURRENT,
        tags.FINANCE_LEASE_COMBINED,
    )
    operating_lease_nc, operating_lease_cur = _lease(
        instant,
        prov,
        "operating lease liability",
        tags.OPERATING_LEASE_NONCURRENT,
        tags.OPERATING_LEASE_CURRENT,
        tags.OPERATING_LEASE_COMBINED,
    )

    inside_debt = 0.0
    if debt.swallows_noncurrent_lease and finance_lease_nc:
        inside_debt += finance_lease_nc
        finance_lease_nc = 0.0
    if debt.swallows_current_lease and finance_lease_cur:
        inside_debt += finance_lease_cur
        finance_lease_cur = 0.0
    if inside_debt:
        warnings.append(
            f"{inside_debt:,.1f}mm of finance lease liabilities is inside the "
            f"debt figure, because the filer tags its debt line as "
            f"{' and '.join(debt.source_tags)}. It is not added a second time "
            "through the lease line"
        )

    fin = Financials(
        ticker=ticker.upper(),
        entity_name=facts.entity_name,
        cik=facts.cik,
        as_of=as_of,
        revenue=revenue,
        gross_profit=flow("gross profit", tags.GROSS_PROFIT, required=False),
        ebit=ebit,
        da=flow("D&A", tags.DA, required=False, annual_ok=True),
        sbc=flow("stock-based compensation", tags.SBC, required=False, annual_ok=True),
        net_income=flow("net income", tags.NET_INCOME) or 0.0,
        pretax_income=flow("pre-tax income", tags.PRETAX_INCOME, required=False),
        tax_expense=flow("income tax expense", tags.TAX_EXPENSE, required=False),
        interest_expense=flow("interest expense", tags.INTEREST_EXPENSE, required=False),
        operating_lease_cost=flow(
            "operating lease cost",
            tags.OPERATING_LEASE_COST,
            required=False,
            annual_ok=True,
        ),
        capex=flow("capital expenditure", tags.CAPEX, required=False, annual_ok=True),
        cfo=flow("cash from operations", tags.CFO, required=False, annual_ok=True),
        diluted_shares=flow("diluted shares", tags.DILUTED_SHARES, kind="average"),
        basic_shares=flow(
            "basic shares", tags.BASIC_SHARES, kind="average", required=False
        ),
        cash=instant("cash and equivalents", tags.CASH),
        short_term_investments=instant(
            "short-term investments", tags.SHORT_TERM_INVESTMENTS, default=0.0
        ),
        straight_debt=debt.total,
        convertible_debt=(
            (
                instant(
                    "convertible notes, non-current",
                    tags.CONVERTIBLE_NONCURRENT,
                    default=0.0,
                )
                or 0.0
            )
            + (
                instant(
                    "convertible notes, current", tags.CONVERTIBLE_CURRENT, default=0.0
                )
                or 0.0
            )
        ),
        operating_lease_liability=operating_lease_nc + operating_lease_cur,
        finance_lease_liability=finance_lease_nc + finance_lease_cur,
        finance_lease_inside_debt=inside_debt,
        debt_tags=tuple(debt.source_tags),
        debt_basis=debt.describe(),
        nci=instant("non-controlling interest", tags.NCI, default=0.0) or 0.0,
        preferred=instant("preferred stock", tags.PREFERRED, default=0.0) or 0.0,
        current_assets=instant("current assets", tags.CURRENT_ASSETS, required=False),
        current_liabilities=instant(
            "current liabilities", tags.CURRENT_LIABILITIES, required=False
        ),
        deferred_revenue=(
            (
                instant(
                    "deferred revenue, current",
                    tags.DEFERRED_REVENUE_CURRENT,
                    default=0.0,
                )
                or 0.0
            )
            + (
                instant(
                    "deferred revenue, non-current",
                    tags.DEFERRED_REVENUE_NONCURRENT,
                    default=0.0,
                )
                or 0.0
            )
        ),
        provenance=prov,
        warnings=warnings,
    )

    _run_controls(fin, facts, as_of, warnings)
    return fin


# Below this a debt gap is a rounding artefact rather than a missing line. USD
# millions, absolute rather than relative: a hundred million dollars of debt that
# the ladder cannot see is worth naming on any company large enough to be in a
# TMT comp set, and a relative threshold would switch the control off on exactly
# the balance sheets where it matters most.
_MATERIAL_DEBT = 100.0

# And a relative floor beside it, because the concepts being compared are struck
# on different bases. Broadcom tags DebtLongtermAndShorttermCombinedAmount at
# 61,079mm for 2026-08-02 against 59,419mm of carrying value on the balance
# sheet: the difference is unamortised discount, not a missing line. A control
# that printed a flag for that would teach a reader to skip the flag on the
# company where it means something, which is the failure mode the whole block is
# written against.
_DEBT_GAP_SHARE = 0.05

# How far short the ladder may fall before the figure is refused rather than
# flagged. Half again is deliberately loose. A concept in ``DEBT_CROSSCHECK`` is
# bounded above by total borrowings, so any excess at all is evidence of a gap,
# but the concepts are struck on different bases: a footnote total may be before
# unamortised discount, a fair-value table may be dated a quarter earlier. Under
# this multiple the reader gets a warning and the number; over it the resolution
# is not a debt figure, it is a fragment of one, and printing it is worse than
# printing nothing.
_DEBT_REFUSAL_MULTIPLE = 1.5

# How far back the control will look for a debt balance the current balance sheet
# no longer shows. One annual reporting cycle plus slack, and the bound is what
# makes the stale arm mean anything: a balance reported at the last period end
# and absent at this one is a missing tag until proved otherwise, while a balance
# last reported five years ago is a company that repaid its debt. MongoDB's
# ``LongTermDebt`` stops at 216.9mm in January 2019, the New York Times's
# ``DebtAndCapitalLeaseObligations`` at 431.2mm in December 2015, and Fortinet's
# at 987.5mm in June 2021; all three are genuinely repaid and all three refused
# before this window was added. T-Mobile is what the window is for: 81,147mm at
# 2025-12-31 against a resolution of 6,117mm at 2026-06-30, because the Q2
# balance sheet tags its long-term debt under a related-party axis the
# companyfacts API does not return.
_DEBT_LOOKBACK_DAYS = 400


def _debt_control(
    fin: Financials, facts: CompanyFacts, as_of: date, warnings: list[str]
) -> None:
    """Whether the resolved debt figure can stand beside this balance sheet.

    The ladders cannot enumerate every concept a filer might use, and the failure
    when they miss one is silent: enterprise value comes back as equity less
    cash, every multiple built on it is too low, and nothing on the page suggests
    a number is missing. Verizon printed an enterprise value of 231,813mm against
    375,261mm for exactly this reason, and its peer table put Comcast's
    enterprise value below Comcast's own market capitalisation.

    So the resolved figure is tested against the filer's own fact set rather than
    against the ladder. Every concept in ``tags.DEBT_CROSSCHECK`` is bounded above
    by total borrowings, whatever the ladder did, so one of them reporting more
    than the ladder resolved is proof that something was missed. What it is proof
    of varies and the message says which:

    *A live concept above the resolved figure* is a tagging gap. Digital Realty
    reports 16,014mm of ``SeniorNotes`` and 842mm of ``SecuredDebt`` at
    2026-03-31 and no total concept at all, so a single-tag ladder resolves zero
    against a real debt stack and there is no honest way to add the pieces up
    without knowing they do not overlap.

    *A stale concept above the resolved figure* is a retired tag or a
    dimensioned one, and the two cannot be told apart from company facts.
    T-Mobile's Q2 2026 balance sheet carries 78,504mm of long-term debt tagged
    ``LongTermDebtNoncurrent`` under a related-party axis, which the companyfacts
    API does not return, so the newest undimensioned value is the 81,147mm of the
    December 10-K and the ladder sees nothing at the balance-sheet date. Read as
    zero it looks like a carrier that repaid eighty billion dollars in a quarter.
    A company can genuinely repay, so the refusal names the figure and its date
    and asks the reader to decide rather than deciding for them.

    The reverse direction is a warning and never a refusal, because the only way
    to overstate here is a small overlap between a debt caption and a separately
    tagged lease, and suppressing a real lease to make an identity close would be
    the worse error.
    """
    resolved = (
        fin.straight_debt + fin.convertible_debt + fin.finance_lease_liability
    )

    # The filer's own total, where it publishes one at this date. It settles the
    # question on its own: a resolution that reaches the filer's stated total
    # cannot be missing a line, whatever an older figure says. Micron is the case
    # the check exists for. It resolves 5,722mm at 2026-05-28, exactly the
    # ``DebtAndCapitalLeaseObligations`` it tags for that date, while its
    # ``LongTermDebt`` stops at 8,844mm six months earlier. That is three billion
    # dollars repaid, not three billion dollars missed, and the filer's own
    # current total is what says so.
    live_totals = [
        f.val / _MM
        for tag in tags.DEBT_TOTALS
        for f in facts.facts(tag)
        if f.is_instant and abs((as_of - f.end).days) <= _INSTANT_TOLERANCE_DAYS
    ]
    reconciled = bool(live_totals) and resolved + _MATERIAL_DEBT >= max(live_totals)

    floor = as_of - timedelta(days=_DEBT_LOOKBACK_DAYS)
    worst: tuple[str, float, date] | None = None
    for tag in tags.DEBT_CROSSCHECK:
        candidates = [
            f for f in facts.facts(tag) if f.is_instant and floor <= f.end <= as_of
        ]
        if not candidates:
            continue
        newest = candidates[-1]
        stale = (as_of - newest.end).days > _INSTANT_TOLERANCE_DAYS
        if stale and reconciled:
            continue
        value = newest.val / _MM
        if worst is None or value > worst[1]:
            worst = (tag, value, newest.end)

    material = max(_MATERIAL_DEBT, _DEBT_GAP_SHARE * resolved)
    if worst is not None and worst[1] - resolved >= material:
        tag, value, end = worst
        live = end >= as_of - timedelta(days=_INSTANT_TOLERANCE_DAYS)
        where = (
            f"at the balance-sheet date of {as_of}"
            if live
            else f"at {end}, the newest instant it carries"
        )
        detail = (
            f"debt resolved to {resolved:,.1f}mm from "
            f"{fin.debt_basis or 'no concept at all'}, but {tag} reports "
            f"{value:,.1f}mm {where}"
        )
        if resolved <= 0.0 or value > resolved * _DEBT_REFUSAL_MULTIPLE:
            error = StaleDataError if not live else MissingDataError
            raise error(
                "straight debt",
                ticker=fin.ticker,
                tags_tried=list(tags.DEBT_CROSSCHECK),
                period=f"as of {as_of}",
                hint=(
                    detail
                    + ". Enterprise value built on the resolved figure would be "
                    f"understated by up to {value - resolved:,.0f}mm, so it is "
                    "refused rather than printed. Read the debt off the balance "
                    "sheet and supply it, or value the equity directly"
                ),
            )
        warnings.append(detail + ". The gap is not explained by the ladder")

    # The overstatement arm compares straight debt alone, not the whole of the
    # claims. A filer's total-debt concept need not contain its convertibles or
    # its separately presented finance leases: Oracle's 129,541mm is "notes
    # payable and other borrowings" and excludes the 7,701mm of finance lease it
    # reports in other liabilities, and Super Micro's 4,056mm excludes its
    # 4,664mm of convertible notes. Comparing the full stack against either would
    # flag a balance sheet that is already right.
    if live_totals and fin.straight_debt - max(live_totals) >= material:
        warnings.append(
            f"straight debt resolved to {fin.straight_debt:,.1f}mm against a "
            f"reported total of {max(live_totals):,.1f}mm at {as_of}. The usual "
            "cause is a finance lease counted both inside a debt caption and "
            "again on its own tag"
        )


def _run_controls(
    fin: Financials, facts: CompanyFacts, as_of: date, warnings: list[str]
) -> None:
    """Tie-out checks against the risk of a silent zero.

    Several balance-sheet concepts default to zero when no tag reports them,
    because for most filers zero is the truth: they have no preferred stock and
    no minority interest. The danger is the filer who does hold the asset but
    tags it under a concept outside the ladder. Zscaler reports its marketable
    securities as ``DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent``;
    a ladder without that entry returns zero and overstates enterprise value by
    the whole balance, with nothing on screen to suggest anything went wrong.

    So current assets are tied back to their components. An unexplained residual
    does not stop the run, but it is reported, because a two-billion-dollar hole
    in a bridge should never be invisible.
    """
    _debt_control(fin, facts, as_of, warnings)

    if (
        fin.total_debt_ex_leases == 0
        and fin.finance_lease_liability == 0
        and fin.interest_expense
        and fin.interest_expense > 1.0
    ):
        warnings.append(
            f"no debt tag resolved, yet {fin.interest_expense:,.1f}mm of interest "
            "expense was reported; the filer may tag borrowings under a concept "
            "outside the ladder in tags.py"
        )

    note = facts.split_note()
    if note:
        warnings.append(
            f"{note}. Share counts and per-share figures from earlier filings have "
            "been restated into current units before any period was aggregated"
        )

    # Tie share count back to the filer's own arithmetic. Net income divided by
    # diluted shares has to reproduce reported diluted EPS; where it does not,
    # the share count is in the wrong units or the wrong period, which is exactly
    # the failure a stock split causes and the one most likely to go unnoticed.
    eps_reported, _ = facts.resolve_ttm(
        "reported diluted EPS", tags.EPS_DILUTED, as_of, required=False
    )
    if eps_reported is not None and fin.diluted_shares:
        implied = fin.net_income / fin.diluted_shares
        # Compared in cents per share rather than as a ratio. A relative test
        # switches itself off exactly where it is needed most: a filer near
        # breakeven has an EPS close to zero, so any tolerance expressed as a
        # percentage of it is met by everything. CrowdStrike, the one company in
        # the fixture set that actually split, earns about four cents.
        tolerance = max(0.02, 0.10 * abs(eps_reported))
        if abs(implied - eps_reported) > tolerance:
            warnings.append(
                f"EPS tie-out: net income over diluted shares gives "
                f"{implied:,.2f}, against {eps_reported:,.2f} summed from the "
                "filings. A gap this size usually means share counts from either "
                "side of a corporate action have been mixed"
            )

    if fin.current_assets:
        receivables, _ = facts.resolve_instant(
            "receivables", tags.RECEIVABLES, as_of, required=False
        )
        accounted = fin.cash + fin.short_term_investments + (receivables or 0.0) / _MM
        residual = fin.current_assets - accounted
        if residual > 0.25 * fin.current_assets:
            warnings.append(
                f"balance-sheet tie-out: {residual:,.0f}mm of current assets "
                f"({residual / fin.current_assets:.0%}) is not explained by cash, "
                "short-term investments or receivables. If any of it is marketable "
                "securities tagged outside the ladder, enterprise value is "
                "overstated by that amount"
            )
