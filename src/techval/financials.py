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
from datetime import date

from . import tags
from .edgar import CompanyFacts, EdgarClient, Provenance
from .errors import MissingDataError


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
    def eps_diluted(self) -> float:
        return self.net_income / self.diluted_shares if self.diluted_shares else 0.0

    def provenance_rows(self) -> list[dict[str, str]]:
        return [p.as_row() for p in self.provenance.values()]


_MM = 1e6


def _straight_debt(instant) -> float:
    """Borrowings, split current and non-current, or a combined tag if that is all.

    Some filers publish only ``LongTermDebt`` and never the current/non-current
    pair. Resolving the pair alone returns zero for them, which reads as a
    debt-free company and understates enterprise value by the whole balance. The
    combined tag is consulted only when the split resolves to nothing, so a filer
    that reports both is not double counted.
    """
    noncurrent = instant("long-term debt", tags.DEBT_NONCURRENT, default=0.0) or 0.0
    current = instant("current debt", tags.DEBT_CURRENT, default=0.0) or 0.0
    if noncurrent or current:
        return noncurrent + current
    return instant("total debt", tags.DEBT_COMBINED, default=0.0) or 0.0


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
    ):
        val, p = facts.resolve_ttm(
            concept,
            ladder,
            as_of,
            kind=kind,
            required=required,
            default_when_absent=default,
            allow_annual_fallback=annual_ok,
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

    revenue = flow("revenue", tags.REVENUE)
    ebit = flow("EBIT", tags.EBIT)
    assert revenue is not None and ebit is not None

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
        straight_debt=_straight_debt(instant),
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
        operating_lease_liability=(
            (
                instant(
                    "operating lease liability, non-current",
                    tags.OPERATING_LEASE_NONCURRENT,
                    default=0.0,
                )
                or 0.0
            )
            + (
                instant(
                    "operating lease liability, current",
                    tags.OPERATING_LEASE_CURRENT,
                    default=0.0,
                )
                or 0.0
            )
        ),
        finance_lease_liability=(
            (
                instant(
                    "finance lease liability, non-current",
                    tags.FINANCE_LEASE_NONCURRENT,
                    default=0.0,
                )
                or 0.0
            )
            + (
                instant(
                    "finance lease liability, current",
                    tags.FINANCE_LEASE_CURRENT,
                    default=0.0,
                )
                or 0.0
            )
        ),
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
