"""Operating metric packs for the TMT sub-verticals.

A discounted cash flow and a comp table price every company the same way. A
coverage banker does not. The first page of a software deck leads with the Rule
of 40, net revenue retention and the magic number. A media page leads with cash
content spend against the amortisation running through reported earnings. A
telecom or tower page leads with capex intensity and EBITDA less capex, because
that is the figure the sector's multiples are actually struck on. None of those
numbers exist anywhere else in this engine.

Three conventions carry through the module.

**Every definition here is contested, so every definition travels with its
number.** ``MetricPack.definitions`` holds one prose definition per metric, keyed
identically to ``metrics``, and a pack refuses to be constructed if a metric
arrives without one. A Rule of 40 quoted without saying which margin went into it
is not a fact, it is an assertion. The pack computes all three margins and lets
the reader see which one the company is passing on.

**Units.** Money is USD millions, subscribers are millions, ARPU is dollars per
subscriber per month, growth rates and margins are decimals, and the Rule of 40
variants are in percentage points because that is how the market quotes them: 25%
growth at a 15% margin scores 40, not 0.40. Every definition states its own unit
first, so the printed table needs no separate legend.

**What the statements cannot support is None with a flag naming the input that
was missing.** The normalized statement set is an income statement, a cash flow
statement and a balance sheet. It does not carry sales and marketing expense,
cash content spend, subscribers or churn, and it is a single trailing twelve
month window so it does not carry a prior period either. Those arrive through the
optional ``kpis`` dict, whose accepted keys are documented in ``KPI_INPUTS``. A
metric that needed one and did not get it is never quietly zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import Assumptions
from ..errors import ConfigError
from ..ev_bridge import EVBridge
from ..financials import Financials

# Market conventions, not valuation assumptions. The Rule of 40 is named for the
# 40, and the sell side stops calling sales efficiency good somewhere around
# 0.75. They colour notes and flags only. No number this module computes depends
# on them, which is why they are not in the assumptions file.
RULE_OF_40_PASS = 40.0
MAGIC_NUMBER_GOOD = 0.75

# How far implied revenue per subscriber may sit from a reported ARPU before the
# pack says so. A reporting-consistency tolerance, not an assumption: a gap of a
# few percent is the arithmetic of a growing subscriber base, a gap of a quarter
# means the two figures are measuring different things.
ARPU_TOLERANCE = 0.10

# Accepted ``kpis`` keys, with the unit each is read in. A key outside this map
# is reported back as unread rather than ignored, because the usual way a metric
# pack goes quietly wrong is a caller supplying "arpu_monthly" to a reader that
# only looks for "arpu".
KPI_INPUTS: dict[str, str] = {
    "revenue_growth": "year on year growth of trailing twelve month revenue, decimal",
    "revenue_prior": "revenue of the prior comparable twelve months, USD millions",
    "arr": "annual recurring revenue at the period end, USD millions",
    "nrr": (
        "net revenue retention over the last twelve months, decimal, so 1.15 for 115%"
    ),
    "rpo": "remaining performance obligations at the period end, USD millions",
    "billings": "calculated or reported billings for the period, USD millions",
    "sales_and_marketing": "sales and marketing expense for the period, USD millions",
    "sales_and_marketing_prior": (
        "sales and marketing expense of the prior comparable period, USD millions"
    ),
    "content_spend": (
        "cash spent on content additions in the period, USD millions, from the "
        "cash flow statement"
    ),
    "content_amortization": (
        "content amortisation charged through the income statement in the period, "
        "USD millions"
    ),
    "subscribers": "subscribers at the period end, millions",
    "arpu": "average revenue per subscriber, dollars per subscriber per month",
    "churn": "subscriber churn, decimal per month",
    "maintenance_capex": (
        "capex required to hold the existing asset base, USD millions, as "
        "disclosed; total capex is not a substitute"
    ),
}

# Which pack a sub-vertical is priced on. A name outside this map raises rather
# than falling through to the software pack, because a semiconductor company
# scored on the Rule of 40 is a worse answer than no answer.
SUB_VERTICALS: dict[str, str] = {
    "software": "software",
    "saas": "software",
    "internet": "software",
    "media": "media",
    "streaming": "media",
    "entertainment": "media",
    "telecom": "telecom",
    "wireless": "telecom",
    "cable": "telecom",
    "fiber": "telecom",
    "towers": "towers",
}


@dataclass
class MetricPack:
    """The operating metrics one sub-vertical is priced on, each with its meaning.

    ``metrics`` and ``definitions`` are keyed identically and that is enforced,
    so a printed table can never carry a number whose definition was left behind.
    ``flags`` say why a metric is None. ``notes`` are the reading of the numbers
    that did compute.
    """

    sub_vertical: str
    metrics: dict[str, float | None]
    definitions: dict[str, str]
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        undefined = [k for k in self.metrics if k not in self.definitions]
        if undefined:
            raise ConfigError(
                "every metric must carry its definition, and these do not: "
                + ", ".join(undefined)
            )

    @property
    def computed(self) -> dict[str, float]:
        """The metrics that resolved to a number, for callers that want only those."""
        return {k: v for k, v in self.metrics.items() if v is not None}

    def rows(self) -> list[dict[str, object]]:
        return [
            {"Metric": name, "Value": value, "Definition": self.definitions[name]}
            for name, value in self.metrics.items()
        ]

    def to_frame(self) -> pd.DataFrame:
        """Raw numbers beside their definitions. Formatting belongs to the renderer."""
        return pd.DataFrame(self.rows()).set_index("Metric")


def _kpi(
    kpis: dict | None,
    key: str,
    flags: list[str],
    needed_for: str | None = None,
) -> float | None:
    """One supplied KPI, or None with a flag naming what the caller has to supply.

    ``needed_for`` is the metric that wanted it. Left None the absence is silent,
    which is right where two keys are alternative routes to the same figure and
    only one of them has to arrive.
    """
    value = (kpis or {}).get(key)
    if value is None:
        if needed_for:
            flags.append(
                f"{needed_for}: not computed, needs kpis[{key!r}] ({KPI_INPUTS[key]})"
            )
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        flags.append(
            f"kpis[{key!r}] is a {type(value).__name__}, not a number, so it was "
            "not used"
        )
        return None
    return float(value)


def _divide(
    numerator: float | None,
    denominator: float | None,
    *,
    metric: str,
    denominator_label: str,
    flags: list[str],
) -> float | None:
    """Divide, or refuse and say why.

    A missing input passes through silently, because whatever could not source it
    has already flagged itself and repeating that helps nobody. A non-positive
    denominator is different: the inputs are there and the ratio is the thing
    that is meaningless, so it is flagged here in the sector's own language
    rather than raised, because one dead metric should not take the page down
    with it.
    """
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        flags.append(
            f"{metric} NM: {denominator_label} of {denominator:,.1f} is not positive"
        )
        return None
    return numerator / denominator


def _ebitda(
    fin: Financials,
    basis: tuple[float | None, str] | None,
    flags: list[str],
) -> tuple[float | None, str]:
    """The earnings figure EV multiples are struck on, and its name.

    Defaults to GAAP EBITDA, which under ASC 842 is after the straight-line rent
    charge. When an EV bridge has counted operating leases as debt it hands down
    EBITDAR instead, and taking the default would count the lease once in the
    numerator and again in the denominator. Towers and cable sit on ground leases
    large enough to move the multiple by a turn on that alone.
    """
    if basis is not None:
        value, label = basis
    else:
        value = fin.ebitda
        label = "EBITDA (after operating lease cost, per ASC 842)"
    if value is None:
        flags.append(
            "EBITDA is not available: this filer reports no combined depreciation "
            "and amortisation tag, so there is nothing to add back to EBIT"
        )
    return value, label


def _free_cash_flow(fin: Financials, flags: list[str]) -> float | None:
    """Cash from operations less capital expenditure.

    Cash interest is already inside CFO, so this is a levered figure and belongs
    against equity value rather than enterprise value.
    """
    if fin.cfo is None or fin.capex is None:
        missing = (
            "cash flow from operations" if fin.cfo is None else "capital expenditure"
        )
        flags.append(
            f"Free cash flow: not computed, {missing} is not reported in this "
            "filer's cash flow statement"
        )
        return None
    return fin.cfo - fin.capex


def _growth(
    fin: Financials, kpis: dict | None, flags: list[str], metric: str
) -> tuple[float | None, float | None]:
    """Revenue growth and the prior period revenue behind it.

    Either input gets you both, so neither is flagged on its own. A supplied
    growth rate wins over one derived here, because the comp engine anchors the
    prior window on a period the filer actually reported rather than on a 365 day
    step, and a 52/53 week filer closes its year one day off that step.
    """
    growth = _kpi(kpis, "revenue_growth", flags)
    prior = _kpi(kpis, "revenue_prior", flags)
    if growth is None and prior is not None:
        if prior > 0:
            growth = fin.revenue / prior - 1.0
        else:
            flags.append(
                f"{metric} NM: the supplied prior period revenue of {prior:,.1f}mm "
                "is not positive"
            )
    if prior is None and growth is not None and growth > -1.0:
        prior = fin.revenue / (1.0 + growth)
    if growth is None:
        flags.append(
            f"{metric}: not computed, needs kpis['revenue_growth'] "
            f"({KPI_INPUTS['revenue_growth']}) or kpis['revenue_prior'] "
            f"({KPI_INPUTS['revenue_prior']}); the normalized statements are a "
            "single trailing twelve month window and carry no prior period"
        )
    return growth, prior


def _subscriber_block(
    fin: Financials,
    kpis: dict | None,
    metrics: dict[str, float | None],
    definitions: dict[str, str],
    flags: list[str],
    notes: list[str],
) -> None:
    """Subscribers, ARPU and churn, with the consistency check that matters.

    Reported ARPU and revenue divided by subscribers disagree at almost every
    company, and the disagreement is informative rather than an error: the
    company is quoting ARPU on one product, or one geography, or on average
    rather than period-end subscribers. The check is here so the gap gets stated
    instead of discovered in a meeting.
    """
    subs = _kpi(kpis, "subscribers", flags, "Subscribers")
    arpu = _kpi(kpis, "arpu", flags, "ARPU (monthly)")
    churn = _kpi(kpis, "churn", flags, "Monthly churn")

    implied = _divide(
        fin.revenue,
        None if subs is None else subs * 12.0,
        metric="Implied revenue per subscriber (monthly)",
        denominator_label="subscriber months",
        flags=flags,
    )

    metrics["Subscribers"] = subs
    definitions["Subscribers"] = (
        "Millions. Subscribers at the period end as supplied. Period-end rather "
        "than average, which is what nearly every filer discloses."
    )
    metrics["ARPU (monthly)"] = arpu
    definitions["ARPU (monthly)"] = (
        "Dollars per subscriber per month, as reported by the company. Read the "
        "filer's own definition before comparing it across names: some report it "
        "on paid subscribers, some on total accounts, some net of a partner "
        "revenue share and some gross of it."
    )
    metrics["Implied revenue per subscriber (monthly)"] = implied
    definitions["Implied revenue per subscriber (monthly)"] = (
        "Dollars per subscriber per month. Trailing twelve month revenue divided "
        "by period-end subscribers and by twelve. It understates ARPU at a "
        "company still adding subscribers, because the revenue was earned on a "
        "smaller average base than the count it is divided by."
    )

    gap = None
    if implied is not None and arpu:
        gap = implied / arpu - 1.0
        multiple = arpu / implied
        if 10.0 <= multiple <= 14.0:
            flags.append(
                f"kpis['arpu'] of {arpu:,.2f} is about {multiple:,.1f} times the "
                f"{implied:,.2f} implied by revenue over subscribers, so it looks "
                "like an annual ARPU. This pack reads ARPU as dollars per "
                "subscriber per month."
            )
        elif abs(gap) > ARPU_TOLERANCE:
            flags.append(
                f"Reported ARPU of {arpu:,.2f} and the {implied:,.2f} implied by "
                f"revenue over subscribers differ by {gap:+.1%}, so the two are "
                "not measuring the same revenue line or the same subscriber base"
            )
    metrics["ARPU consistency gap"] = gap
    definitions["ARPU consistency gap"] = (
        "Decimal. Implied revenue per subscriber over reported ARPU, less one. "
        "Zero means the reported figure reconciles to total revenue. A large gap "
        "means ARPU is quoted on a subset of revenue or on a different subscriber "
        "definition, and anything built on it is not what it looks like."
    )

    life = None
    if churn is not None:
        if churn <= 0:
            flags.append(
                f"Implied subscriber life NM: monthly churn of {churn:,.4f} is not "
                "positive"
            )
        else:
            life = 1.0 / churn
    metrics["Monthly churn"] = churn
    definitions["Monthly churn"] = (
        "Decimal per month. Subscribers lost in a month over the opening base. "
        "Annualise it by compounding, not by multiplying by twelve."
    )
    metrics["Implied subscriber life (months)"] = life
    definitions["Implied subscriber life (months)"] = (
        "Months. One over monthly churn, the mean life of a geometric survival "
        "curve. It assumes a constant hazard rate, which overstates life wherever "
        "churn is front-loaded into the months after sign-up, and that is the "
        "normal shape for a consumer subscription."
    )
    if life is not None:
        notes.append(
            f"At {churn:.2%} monthly churn the average subscriber stays "
            f"{life:,.0f} months. Any lifetime value built on that base inherits "
            "the constant-hazard assumption behind it."
        )


def software_metrics(
    fin: Financials,
    kpis: dict | None = None,
    market_cap: float | None = None,
    ev: float | None = None,
) -> MetricPack:
    """The software page: growth, the three Rule of 40 variants, sales efficiency.

    The Rule of 40 is one rule and three different numbers, and which margin goes
    into it is the whole argument. The free cash flow variant is what growth
    investors quote, because it is the cash the business actually threw off. The
    EBITDA variant is what this engine's comp table reports, because it comes off
    filed lines every company has and so stays comparable across the set. The
    operating variant is the strictest, since it is after depreciation and after
    stock compensation. One company can pass on one variant and fail on another
    by twenty points, which is why all three are printed rather than one being
    chosen here.
    """
    metrics: dict[str, float | None] = {}
    definitions: dict[str, str] = {}
    flags: list[str] = []
    notes: list[str] = []

    growth, prior_revenue = _growth(fin, kpis, flags, "Revenue growth")
    ebitda, _ = _ebitda(fin, None, flags)
    fcf = _free_cash_flow(fin, flags)

    ebitda_margin = _divide(
        ebitda,
        fin.revenue,
        metric="EBITDA margin",
        denominator_label="revenue",
        flags=flags,
    )
    fcf_margin = _divide(
        fcf, fin.revenue, metric="FCF margin", denominator_label="revenue", flags=flags
    )
    operating_margin = _divide(
        fin.ebit,
        fin.revenue,
        metric="Operating margin",
        denominator_label="revenue",
        flags=flags,
    )
    fcf_less_sbc = None if fcf is None or fin.sbc is None else fcf - fin.sbc
    if fcf is not None and fin.sbc is None:
        flags.append(
            "FCF margin less SBC: not computed, this filer reports no share-based "
            "compensation tag in its cash flow statement"
        )

    metrics["Revenue growth"] = growth
    definitions["Revenue growth"] = (
        "Decimal. Trailing twelve month revenue against the prior comparable "
        "twelve months."
    )
    metrics["Gross margin"] = fin.gross_margin
    definitions["Gross margin"] = (
        "Decimal. Gross profit over revenue, as filed. Hosting and customer "
        "support sit in cost of revenue for a software company, so this is not "
        "the licence-model gross margin of the 1990s."
    )
    if fin.gross_margin is None:
        flags.append(
            "Gross margin: not computed, this filer does not report a gross profit "
            "line"
        )
    metrics["EBITDA margin"] = ebitda_margin
    definitions["EBITDA margin"] = (
        "Decimal. EBITDA over revenue, after stock-based compensation, because SBC "
        "is an operating expense inside EBIT and nothing here adds it back."
    )
    metrics["Operating margin"] = operating_margin
    definitions["Operating margin"] = "Decimal. GAAP EBIT over revenue."
    metrics["Free cash flow"] = fcf
    definitions["Free cash flow"] = (
        "USD millions. Cash from operations less capital expenditure. Cash "
        "interest is already inside CFO, so this is a levered figure and belongs "
        "against equity value rather than enterprise value."
    )
    metrics["FCF margin"] = fcf_margin
    definitions["FCF margin"] = (
        "Decimal. (CFO less capex) over revenue. Stock-based compensation is added "
        "back inside CFO, so this margin is struck before the cost of paying "
        "people in stock, which is exactly why it flatters a heavy issuer against "
        "the EBITDA and operating margins. Collections land here too: a company "
        "billing annually in advance carries a working capital tailwind in this "
        "line that revenue growth alone does not show."
    )
    metrics["FCF margin less SBC"] = _divide(
        fcf_less_sbc,
        fin.revenue,
        metric="FCF margin less SBC",
        denominator_label="revenue",
        flags=flags,
    )
    definitions["FCF margin less SBC"] = (
        "Decimal. (CFO less capex less stock-based compensation) over revenue. The "
        "same cash figure with the SBC add-back reversed out, which is the honest "
        "comparison against an EBITDA margin struck after that cost."
    )
    metrics["SBC / revenue"] = _divide(
        fin.sbc,
        fin.revenue,
        metric="SBC / revenue",
        denominator_label="revenue",
        flags=flags,
    )
    definitions["SBC / revenue"] = (
        "Decimal. Stock-based compensation over revenue. This is the size of the "
        "wedge between the FCF and EBITDA variants of the Rule of 40, and it is "
        "also the dilution the share count absorbs each year."
    )

    variant_definitions = {
        "Rule of 40 (EBITDA variant)": (
            "Percentage points. Revenue growth plus EBITDA margin. The variant "
            "this engine's comp table reports, because EBITDA comes off filed "
            "lines every company has and so the score stays comparable across the "
            "set. Under ASC 842 it is after the operating lease charge."
        ),
        "Rule of 40 (FCF variant)": (
            "Percentage points. Revenue growth plus free cash flow margin. What "
            "growth investors mean when they say Rule of 40, and the most generous "
            "of the three at any company paying a large part of its compensation "
            "in stock or collecting cash a year ahead of revenue."
        ),
        "Rule of 40 (operating variant)": (
            "Percentage points. Revenue growth plus GAAP operating margin. The "
            "strictest of the three: after depreciation, after amortisation of "
            "acquired intangibles and after stock compensation."
        ),
    }
    variants = (
        ("Rule of 40 (EBITDA variant)", ebitda_margin),
        ("Rule of 40 (FCF variant)", fcf_margin),
        ("Rule of 40 (operating variant)", operating_margin),
    )
    for name, margin in variants:
        metrics[name] = (
            None if growth is None or margin is None else 100.0 * (growth + margin)
        )
        definitions[name] = variant_definitions[name]
    if growth is None:
        flags.append(
            "Rule of 40 (all three variants): not computed, revenue growth is "
            "missing"
        )
    elif any(m is None for _, m in variants):
        flags.append(
            "Rule of 40: any variant left blank above is blank because its margin "
            "could not be struck, not because the score is low"
        )

    scored = {name: metrics[name] for name, _ in variants if metrics[name] is not None}
    spread = None if len(scored) < 2 else max(scored.values()) - min(scored.values())
    metrics["Rule of 40 variant spread"] = spread
    definitions["Rule of 40 variant spread"] = (
        "Percentage points. Highest Rule of 40 variant less the lowest. The width "
        "of the disagreement between three definitions of one rule, on one set of "
        "filings."
    )
    if scored:
        best = max(scored, key=lambda k: scored[k])
        worst = min(scored, key=lambda k: scored[k])
        if scored[best] >= RULE_OF_40_PASS > scored[worst]:
            notes.append(
                f"The Rule of 40 variants straddle {RULE_OF_40_PASS:.0f}: this "
                f"company passes at {scored[best]:,.1f} on the "
                f"{best.split('(')[1].rstrip(')')} and fails at "
                f"{scored[worst]:,.1f} on the {worst.split('(')[1].rstrip(')')}, "
                "off one set of filings. Which number gets quoted is a choice, so "
                "state it."
            )
        if fin.sbc is not None and fin.revenue:
            notes.append(
                f"Stock-based compensation runs {fin.sbc / fin.revenue:.1%} of "
                "revenue. EBITDA is charged with it and cash flow from operations "
                "adds it straight back, so that figure is most of the distance "
                "between the FCF and EBITDA variants."
            )

    sm = _kpi(kpis, "sales_and_marketing", flags, "Sales and marketing / revenue")
    sm_prior = _kpi(kpis, "sales_and_marketing_prior", flags, "Magic number")
    increment = None if prior_revenue is None else fin.revenue - prior_revenue

    metrics["Sales and marketing / revenue"] = _divide(
        sm,
        fin.revenue,
        metric="Sales and marketing / revenue",
        denominator_label="revenue",
        flags=flags,
    )
    definitions["Sales and marketing / revenue"] = (
        "Decimal. Sales and marketing expense over revenue. Not separable from the "
        "normalized statement set, so it has to be supplied off the income "
        "statement itself."
    )

    magic = _divide(
        increment,
        sm_prior,
        metric="Magic number",
        denominator_label="prior period sales and marketing spend",
        flags=flags,
    )
    metrics["Magic number"] = magic
    definitions["Magic number"] = (
        "Multiple. The increase in revenue over the period, annualised, over the "
        "prior period's sales and marketing spend. Computed here on twelve months "
        "against twelve months, which needs no annualising. The quarterly variant "
        "the sell side quotes takes one quarter's increase times four; it measures "
        "the same efficiency through a noisier lens and turns a seasonally strong "
        f"quarter into a sales story. Above {MAGIC_NUMBER_GOOD} the market reads "
        "sales efficiency as good and funds more hiring."
    )
    if magic is not None:
        verdict = "above" if magic >= MAGIC_NUMBER_GOOD else "below"
        notes.append(
            f"A magic number of {magic:,.2f} is {verdict} the "
            f"{MAGIC_NUMBER_GOOD} line. It assumes last period's selling spend "
            "bought this period's revenue, which imposes a one-period lag on a "
            "sales cycle that may not have one."
        )

    payback = None
    if magic is not None and fin.gross_margin is None:
        flags.append(
            "CAC payback (months): not computed, this filer reports no gross profit "
            "line, and paying sales spend back out of revenue rather than gross "
            "profit would ignore the cost of serving the customer"
        )
    elif magic is not None:
        payback = _divide(
            12.0,
            magic * fin.gross_margin,
            metric="CAC payback (months)",
            denominator_label="magic number times gross margin",
            flags=flags,
        )
    metrics["CAC payback (months)"] = payback
    definitions["CAC payback (months)"] = (
        "Months. Prior period sales and marketing spend over the annualised "
        "increase in gross profit it bought, times twelve. Identical to twelve "
        "over the magic number times gross margin. Gross profit rather than "
        "revenue in the denominator, because a dollar of revenue that costs forty "
        "cents to serve does not pay back a dollar of selling cost."
    )

    nrr = _kpi(kpis, "nrr", flags, "Net revenue retention")
    metrics["Net revenue retention"] = nrr
    definitions["Net revenue retention"] = (
        "Decimal. Revenue from the customers of a year ago, measured today, "
        "including expansion, downgrades and churn but excluding new logos. Above "
        "1.0 the installed base grows on its own with zero new customers, which is "
        "the part of the multiple that is not paying for the sales force. It is a "
        "disclosed figure rather than a derived one, and the definition varies by "
        "filer."
    )
    if nrr is not None and nrr > 1.0:
        notes.append(
            f"Net revenue retention of {nrr:.0%} means the installed base grows "
            f"{nrr - 1.0:.0%} a year before a single new logo. A figure durably "
            "above one is most of the argument for a growth multiple."
        )

    arr = _kpi(kpis, "arr", flags, "EV / ARR")
    metrics["ARR"] = arr
    definitions["ARR"] = (
        "USD millions. Annual recurring revenue at the period end as disclosed. A "
        "point-in-time run rate, so it leads trailing revenue at a growing company "
        "and is not comparable to it."
    )
    metrics["EV / ARR"] = _divide(
        ev, arr, metric="EV / ARR", denominator_label="ARR", flags=flags
    )
    definitions["EV / ARR"] = (
        "Multiple. Enterprise value over period-end ARR. Lower than EV/Revenue at "
        "a growing company, because ARR is a forward run rate and revenue is "
        "trailing, so the two are never quoted side by side as though they were "
        "the same multiple."
    )
    if ev is None:
        flags.append(
            "EV / ARR: not computed, no enterprise value was supplied; pass an EV "
            "bridge to build_metrics or ev= to this function"
        )

    rpo = _kpi(kpis, "rpo", flags, "RPO coverage")
    metrics["RPO"] = rpo
    definitions["RPO"] = (
        "USD millions. Remaining performance obligations, the contracted revenue "
        "not yet recognised. It includes non-cancellable multi-year commitments, "
        "which makes it the hardest forward number a software filer publishes."
    )
    metrics["RPO coverage"] = _divide(
        rpo,
        fin.revenue,
        metric="RPO coverage",
        denominator_label="revenue",
        flags=flags,
    )
    definitions["RPO coverage"] = (
        "Multiple, read as years. RPO over trailing twelve month revenue. Rising "
        "coverage means contracts are lengthening, which supports the multiple and "
        "also means less of next year is still to be sold."
    )

    billings = _kpi(kpis, "billings", flags, "Billings / revenue")
    metrics["Billings"] = billings
    definitions["Billings"] = (
        "USD millions. Billings for the period as supplied, usually revenue plus "
        "the change in deferred revenue."
    )
    metrics["Billings / revenue"] = _divide(
        billings,
        fin.revenue,
        metric="Billings / revenue",
        denominator_label="revenue",
        flags=flags,
    )
    definitions["Billings / revenue"] = (
        "Multiple. Billings over revenue for the same period. Above one the "
        "deferred revenue balance is building and cash is arriving ahead of "
        "recognition, which is the working capital tailwind sitting inside the FCF "
        "variant of the Rule of 40."
    )

    metrics["FCF yield on equity"] = _divide(
        fcf,
        market_cap,
        metric="FCF yield on equity",
        denominator_label="equity value",
        flags=flags,
    )
    definitions["FCF yield on equity"] = (
        "Decimal. Free cash flow over equity value. Equity value rather than "
        "enterprise value in the denominator, because CFO is already struck after "
        "cash interest, so the numerator belongs to the shareholders."
    )
    if market_cap is None:
        flags.append(
            "FCF yield on equity: not computed, no equity value was supplied; pass "
            "an EV bridge to build_metrics or market_cap= to this function"
        )

    return MetricPack(
        sub_vertical="software",
        metrics=metrics,
        definitions=definitions,
        flags=flags,
        notes=notes,
    )


def media_metrics(
    fin: Financials,
    kpis: dict | None = None,
    ev: float | None = None,
    *,
    ebitda_basis: tuple[float | None, str] | None = None,
) -> MetricPack:
    """The media page: what content costs in cash against what earnings were charged.

    A studio or a streamer capitalises the content it makes and amortises it over
    the years it expects the title to be watched. Cash goes out now, the charge
    arrives later, and while a library is growing the charge is smaller than the
    spend. Reported EBITDA is therefore struck on a cost the company did not
    incur this year, and the gap between cash content spend and content
    amortisation is the single most important adjustment in the sector. A
    widening gap is a company investing through the income statement's blind
    spot. A narrowing one is a library maturing, and earnings quality improving
    whether or not earnings do.

    The normalized statements cannot say whether content amortisation was inside
    the depreciation and amortisation added back to reach EBITDA, because that
    depends on how the filer tagged it. Both cash-adjusted figures are shown for
    that reason, each naming the tagging it assumes.
    """
    metrics: dict[str, float | None] = {}
    definitions: dict[str, str] = {}
    flags: list[str] = []
    notes: list[str] = []

    growth, _ = _growth(fin, kpis, flags, "Revenue growth")
    ebitda, ebitda_label = _ebitda(fin, ebitda_basis, flags)
    spend = _kpi(kpis, "content_spend", flags, "Content spend")
    amort = _kpi(kpis, "content_amortization", flags, "Content amortisation")

    metrics["Revenue growth"] = growth
    definitions["Revenue growth"] = (
        "Decimal. Trailing twelve month revenue against the prior comparable "
        "twelve months."
    )
    metrics["EBITDA margin"] = _divide(
        ebitda,
        fin.revenue,
        metric="EBITDA margin",
        denominator_label="revenue",
        flags=flags,
    )
    definitions["EBITDA margin"] = f"Decimal. {ebitda_label} over revenue."

    metrics["Content spend"] = spend
    definitions["Content spend"] = (
        "USD millions. Cash paid for content additions in the period, off the cash "
        "flow statement. This is the cheque, not the charge."
    )
    metrics["Content spend / revenue"] = _divide(
        spend,
        fin.revenue,
        metric="Content spend / revenue",
        denominator_label="revenue",
        flags=flags,
    )
    definitions["Content spend / revenue"] = (
        "Decimal. Cash content spend over revenue. The sector's capital intensity "
        "measure, and the first line of any streaming model."
    )
    metrics["Content amortisation"] = amort
    definitions["Content amortisation"] = (
        "USD millions. The content charge that ran through the income statement "
        "this period. At most filers it sits inside cost of revenues rather than "
        "in a depreciation and amortisation line."
    )

    gap = None if spend is None or amort is None else spend - amort
    metrics["Content spend less amortisation"] = gap
    definitions["Content spend less amortisation"] = (
        "USD millions. Cash out less the charge taken. Positive means the library "
        "is being built faster than it is being written off and reported earnings "
        "flatter the cash reality by that amount. Negative means the company is "
        "harvesting a library it has already paid for."
    )
    metrics["Content spend / amortisation"] = _divide(
        spend,
        amort,
        metric="Content spend / amortisation",
        denominator_label="content amortisation",
        flags=flags,
    )
    definitions["Content spend / amortisation"] = (
        "Multiple. Cash content spend over content amortisation. The scale-free "
        "version of the gap: 1.0 is a steady-state library, above 1.0 is "
        "investment the income statement has not been charged for yet, and a "
        "company holding above 1.0 for years is one whose earnings and cash flows "
        "are telling different stories."
    )

    metrics["EBITDA less content gap"] = (
        None if ebitda is None or gap is None else ebitda - gap
    )
    definitions["EBITDA less content gap"] = (
        f"USD millions. {ebitda_label} less cash content spend plus content "
        "amortisation. The right figure where content amortisation sits in cost of "
        "revenues and so was never added back to reach EBITDA, which is the usual "
        "tagging. It puts the cash cheque into an earnings number that had only "
        "been charged the accounting estimate."
    )
    metrics["EBITDA less cash content spend"] = (
        None if ebitda is None or spend is None else ebitda - spend
    )
    definitions["EBITDA less cash content spend"] = (
        f"USD millions. {ebitda_label} less cash content spend, with no add-back. "
        "The right figure under the other tagging, where content amortisation was "
        "inside the depreciation and amortisation added back to reach EBITDA and "
        "has therefore already been removed. Using it under the first tagging "
        "charges the content cost twice."
    )
    if gap is not None:
        if fin.da is not None and amort > fin.da:
            notes.append(
                f"Content amortisation of {amort:,.0f}mm exceeds the "
                f"{fin.da:,.0f}mm of total depreciation and amortisation added "
                "back to reach EBITDA, so the content charge is not inside that "
                "add-back and 'EBITDA less content gap' is the applicable figure."
            )
        else:
            flags.append(
                "Which cash-adjusted EBITDA applies depends on whether this "
                "filer's content amortisation sits inside the depreciation and "
                "amortisation added back to reach EBITDA. The statements do not "
                "say. Check the cash flow statement before quoting either figure."
            )
        notes.append(
            f"Cash content spend runs {gap:+,.0f}mm against the amortisation "
            "charged. That is the annual distance between reported earnings and "
            "the cash the content library absorbs."
        )

    metrics["EV / EBITDA less content gap"] = _divide(
        ev,
        metrics["EBITDA less content gap"],
        metric="EV / EBITDA less content gap",
        denominator_label="EBITDA less the content gap",
        flags=flags,
    )
    definitions["EV / EBITDA less content gap"] = (
        "Multiple. Enterprise value over the cash-adjusted earnings figure above. "
        "A streamer in a build phase looks materially more expensive on this than "
        "on headline EV/EBITDA, and that difference is the reason to compute it."
    )
    if ev is None:
        flags.append(
            "EV / EBITDA less content gap: not computed, no enterprise value was "
            "supplied; pass an EV bridge to build_metrics or ev= to this function"
        )

    _subscriber_block(fin, kpis, metrics, definitions, flags, notes)

    return MetricPack(
        sub_vertical="media",
        metrics=metrics,
        definitions=definitions,
        flags=flags,
        notes=notes,
    )


def telecom_metrics(
    fin: Financials,
    kpis: dict | None = None,
    market_cap: float | None = None,
    ev: float | None = None,
    *,
    net_debt: float | None = None,
    ebitda_basis: tuple[float | None, str] | None = None,
    sub_vertical: str = "telecom",
) -> MetricPack:
    """The telecom, tower and fiber page: capex intensity and what survives it.

    A carrier spends 15 to 20 percent of revenue every year to stand still,
    because the network is the product and it depreciates. EV/EBITDA prices the
    business as though that spending were optional, so the sector is quoted on
    EBITDA less capex instead, and the distance between the two multiples is the
    entire reason the convention exists. Towers sit at the other end of the same
    spectrum: the structure is already built, augmentation capex is small, and
    the tenant leases are contracted with fixed escalators, which is why towers
    are REITs quoted on AFFO multiples rather than on EV/EBITDA at all.

    Leverage is not a footnote here. The sector is levered by design, the debt is
    priced off contracted cash flows, and net debt to EBITDA is the constraint
    the equity story is actually running against.
    """
    metrics: dict[str, float | None] = {}
    definitions: dict[str, str] = {}
    flags: list[str] = []
    notes: list[str] = []

    growth, _ = _growth(fin, kpis, flags, "Revenue growth")
    ebitda, ebitda_label = _ebitda(fin, ebitda_basis, flags)
    if fin.capex is None:
        flags.append(
            "Capex intensity and EBITDA less capex: not computed, this filer "
            "reports no capital expenditure line in its cash flow statement"
        )

    metrics["Revenue growth"] = growth
    definitions["Revenue growth"] = (
        "Decimal. Trailing twelve month revenue against the prior comparable "
        "twelve months."
    )
    metrics["EBITDA"] = ebitda
    definitions["EBITDA"] = (
        f"USD millions. {ebitda_label}. Ground leases are large in this sector, so "
        "which of EBITDA and EBITDAR is quoted has to match how the enterprise "
        "value treated the lease liability."
    )
    metrics["EBITDA margin"] = _divide(
        ebitda,
        fin.revenue,
        metric="EBITDA margin",
        denominator_label="revenue",
        flags=flags,
    )
    definitions["EBITDA margin"] = f"Decimal. {ebitda_label} over revenue."
    metrics["Capex"] = fin.capex
    definitions["Capex"] = (
        "USD millions. Cash capital expenditure over the trailing twelve months. "
        "Success-based, growth and maintenance spend are not separable from this "
        "line."
    )

    intensity = _divide(
        fin.capex,
        fin.revenue,
        metric="Capex intensity",
        denominator_label="revenue",
        flags=flags,
    )
    metrics["Capex intensity"] = intensity
    definitions["Capex intensity"] = (
        "Decimal. Capex over revenue. A wireline or wireless carrier runs 15 to 20 "
        "percent through a normal cycle and higher through a spectrum build or a "
        "fiber push. A tower REIT runs far below that, because the expensive part "
        "of a tower is built once. It is the number that decides whether EV/EBITDA "
        "is telling you anything."
    )
    if intensity is not None:
        notes.append(
            f"Capex absorbs {intensity:.1%} of revenue, so EV/EBITDA prices that "
            "spending as though it were free. EBITDA less capex is the sector's "
            "answer to exactly that."
        )

    less_capex = None if ebitda is None or fin.capex is None else ebitda - fin.capex
    metrics["EBITDA less capex"] = less_capex
    definitions["EBITDA less capex"] = (
        f"USD millions. {ebitda_label} less cash capex. A pre-tax, pre-interest "
        "proxy for what the network throws off after the spending that keeps it "
        "competitive. It charges growth capex as though it were maintenance, which "
        "penalises a carrier mid-build, and that is the known cost of the "
        "convention."
    )
    metrics["EBITDA less capex margin"] = _divide(
        less_capex,
        fin.revenue,
        metric="EBITDA less capex margin",
        denominator_label="revenue",
        flags=flags,
    )
    definitions["EBITDA less capex margin"] = (
        "Decimal. EBITDA less capex over revenue."
    )
    metrics["EV / EBITDA"] = _divide(
        ev, ebitda, metric="EV / EBITDA", denominator_label="EBITDA", flags=flags
    )
    definitions["EV / EBITDA"] = (
        f"Multiple. Enterprise value over {ebitda_label}. Carried so it can be read "
        "against the line below rather than instead of it."
    )
    metrics["EV / EBITDA less capex"] = _divide(
        ev,
        less_capex,
        metric="EV / EBITDA less capex",
        denominator_label="EBITDA less capex",
        flags=flags,
    )
    definitions["EV / EBITDA less capex"] = (
        "Multiple. Enterprise value over EBITDA less capex. The sector's "
        "convention, and the only one of the two that prices a network operator "
        "and an asset-light business on comparable terms."
    )
    if ev is None:
        flags.append(
            "EV / EBITDA and EV / EBITDA less capex: not computed, no enterprise "
            "value was supplied; pass an EV bridge to build_metrics or ev= to this "
            "function"
        )

    if net_debt is None:
        flags.append(
            "Net debt / EBITDA: not computed, no net debt was supplied. It is a "
            "bridge output rather than a statement line, because whether lease "
            "liabilities and convertibles count as debt is a bridge decision; pass "
            "an EV bridge to build_metrics or net_debt= to this function"
        )
    metrics["Net debt"] = net_debt
    definitions["Net debt"] = (
        "USD millions. Gross debt less cash and short-term investments, on the "
        "lease and convertible conventions the enterprise value bridge used. "
        "Negative means net cash."
    )
    metrics["Net debt / EBITDA"] = _divide(
        net_debt,
        ebitda,
        metric="Net debt / EBITDA",
        denominator_label="EBITDA",
        flags=flags,
    )
    definitions["Net debt / EBITDA"] = (
        "Multiple, quoted as turns. Net debt over EBITDA. The sector is levered by "
        "design and the leverage is the equity story: at four turns, a point of "
        "EBITDA margin moves the equity several times as far as it moves the "
        "enterprise. Compare it against the filer's own covenant definition, which "
        "usually runs on a different EBITDA."
    )
    if net_debt is not None and ebitda is not None and ebitda > 0:
        notes.append(
            f"Leverage of {net_debt / ebitda:,.1f} turns sits inside an enterprise "
            "value that already counts that debt. The equity is the residual, so "
            "it carries the whole of any multiple move."
        )

    if sub_vertical == "towers":
        maintenance = _kpi(kpis, "maintenance_capex", flags, "AFFO proxy")
        if maintenance is None and fin.capex is not None:
            flags.append(
                "AFFO proxy: total capex was not used in place of maintenance "
                "capex. A tower REIT's capex is mostly discretionary augmentation "
                "and land purchases, so substituting it would understate AFFO by "
                "the whole growth programme"
            )
        affo = None
        if ebitda is not None and maintenance is not None:
            if fin.interest_expense is None:
                flags.append(
                    "AFFO proxy: not computed, interest expense is not reported, "
                    "and AFFO is struck after the cost of the debt that funded the "
                    "portfolio"
                )
            else:
                affo = ebitda - fin.interest_expense - maintenance
        metrics["AFFO proxy"] = affo
        definitions["AFFO proxy"] = (
            f"USD millions. {ebitda_label} less interest expense less maintenance "
            "capex. A proxy, not the company's AFFO: a REIT's own definition also "
            "adds back stock compensation, straight-line rent and amortisation of "
            "deferred financing costs, and uses cash interest rather than the "
            "accrued charge. Rank names on it, do not quote it."
        )
        metrics["Price / AFFO proxy"] = _divide(
            market_cap,
            affo,
            metric="Price / AFFO proxy",
            denominator_label="the AFFO proxy",
            flags=flags,
        )
        definitions["Price / AFFO proxy"] = (
            "Multiple. Equity value over the AFFO proxy. Towers are REITs and the "
            "market quotes them on AFFO per share multiples rather than EV/EBITDA, "
            "because the distribution is the product and AFFO is what funds it. An "
            "EV/EBITDA comparison against a non-REIT peer set misses both the "
            "payout requirement and the tax status."
        )
        if market_cap is None:
            flags.append(
                "Price / AFFO proxy: not computed, no equity value was supplied; "
                "pass an EV bridge to build_metrics or market_cap= to this function"
            )
        notes.append(
            "Towers are REITs. Rank them on AFFO multiples and on tenants per "
            "tower, and read the EV/EBITDA lines above as a bridge to the rest of "
            "the sector rather than as the valuation."
        )

    _subscriber_block(fin, kpis, metrics, definitions, flags, notes)

    return MetricPack(
        sub_vertical=sub_vertical,
        metrics=metrics,
        definitions=definitions,
        flags=flags,
        notes=notes,
    )


def build_metrics(
    fin: Financials,
    sub_vertical: str | None,
    assumptions: Assumptions,
    kpis: dict | None = None,
    bridge: EVBridge | None = None,
) -> MetricPack:
    """The metric pack for one company, routed by sub-vertical and priced off a bridge.

    The bridge supplies enterprise value, equity value and net debt, and it also
    supplies the earnings figure those multiples are allowed to pair with. Where
    it counted operating leases as debt the denominator is EBITDAR, because
    quoting a lease-inclusive enterprise value over a post-rent EBITDA counts the
    lease twice. In towers and cable that trap is worth more than a turn.

    An unknown sub-vertical raises rather than falling back to the software pack.
    A semiconductor company scored on the Rule of 40 is a worse answer than no
    answer, and a silent default is the kind of thing that survives into a
    printed page.
    """
    # Read the sub-vertical through getattr so an assumptions file without a TMT
    # block still runs; the caller can always name it directly instead.
    name = sub_vertical or getattr(
        getattr(assumptions, "tmt", None), "sub_vertical", None
    )
    if not name:
        raise ConfigError(
            "no sub-vertical to build a metric pack for. Pass one to build_metrics "
            "or set tmt.sub_vertical in the assumptions file. Known: "
            + ", ".join(sorted(SUB_VERTICALS))
        )
    key = str(name).strip().lower()
    pack = SUB_VERTICALS.get(key)
    if pack is None:
        raise ConfigError(
            f"no metric pack is written for sub-vertical {name!r}. Known: "
            + ", ".join(sorted(SUB_VERTICALS))
            + ". The operating metrics a sector is priced on do not generalise, so "
            "there is no default pack."
        )

    ev = bridge.enterprise_value if bridge else None
    market_cap = bridge.equity_value if bridge else None
    basis = bridge.multiple_denominator(fin) if bridge else None

    if pack == "software":
        result = software_metrics(fin, kpis, market_cap, ev)
    elif pack == "media":
        result = media_metrics(fin, kpis, ev, ebitda_basis=basis)
    else:
        result = telecom_metrics(
            fin,
            kpis,
            market_cap,
            ev,
            net_debt=bridge.net_debt if bridge else None,
            ebitda_basis=basis,
            sub_vertical=pack,
        )

    result.sub_vertical = key
    result.notes.insert(
        0,
        f"{fin.entity_name}: trailing twelve months to {fin.as_of}, USD millions, "
        f"on the {pack} metric pack.",
    )
    if basis is not None:
        result.notes.insert(
            1,
            f"Multiples pair {basis[1]} with an enterprise value built on the "
            f"{bridge.lease_convention} lease convention.",
        )
    unread = sorted(set(kpis or {}) - set(KPI_INPUTS))
    if unread:
        result.flags.append(
            "kpis carried keys this pack does not read, so they reached no metric: "
            + ", ".join(unread)
        )
    return result
