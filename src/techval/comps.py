"""Trading comparables, and the four ways a comp table lies.

A comp set is a claim that a group of companies is priced on the same economics,
so the target should be priced on them too. The arithmetic is one division. The
work is making sure the numerator and the denominator describe the same firm,
and refusing to print a figure that survives the division but says nothing.

**Numerator and denominator must share a lease convention.** ASC 842 leaves
operating lease expense inside US-GAAP operating income as a single rent charge,
so a US filer's EBITDA is already after rent. An enterprise value that counts
the lease liability as debt therefore has to be divided by EBITDAR, before rent,
or the same obligation is charged twice: once as a claim in the numerator and
again as a cost that suppressed the denominator. Every earnings figure here
comes from ``bridge.multiple_denominator``, the only sanctioned way to pick the
matching line, so the mismatch cannot be introduced by hand.

**A meaningless multiple is suppressed, not dropped.** A company running a two
percent GAAP EBITDA margin trades at some enormous EV/EBITDA. That number is
arithmetic, not valuation, and printed into a column it drags the median with
it. Multiples on a non-positive denominator, and any EV/EBITDA above
``comps.ev_ebitda_nm_threshold``, are returned as None with a flag saying why.
The peer keeps its row and its other multiples, because a business that is
unprofitable this year is still evidence of what the market pays for its revenue.

**A peer never disappears quietly.** A name that cannot be built at all goes to
``exclusions`` with the reason text, for the CLI to print. A comp set that
silently shrank from twelve names to five is the most common way a median gets
quoted with more confidence than it has earned. For the same reason every column
of ``stats`` carries its own ``n``: a median over two peers is not a median.

**EV/Gross Profit sits beside EV/Revenue.** Gross margin across software runs
from the low forties for infrastructure that resells compute to the high eighties
for application software. EV/Revenue treats those dollars as interchangeable and
quietly rewards a company for passing through low-margin hosting. EV/Gross Profit
compares what the revenue is actually worth once the cost of delivering it is out.

Revenue growth is year-over-year TTM: the trailing twelve months against the
twelve months ending a year earlier, both tiled from the filer's own reported
periods. Where the earlier window cannot be tiled, growth is None and flagged.
It is never annualised from a quarter, and never interpolated.

Rule of 40 is TTM revenue growth in points plus TTM EBITDA margin in points. The
growth-equity convention often substitutes free cash flow margin, which is nearer
to what an owner keeps. EBITDA margin is used here because it can be built for
every name in the set from filings alone, and a Rule of 40 available for six of
ten peers ranks nothing.

**A band says what the neighbours trade at, not what the target deserves.** The
median knows nothing about the company it is applied to, so it quietly penalises
a name growing faster than its set and rewards one growing slower. Across
software, EV/Revenue is largely explained by growth and margin, which is a
testable claim rather than a saying, so ``fit_warranted_multiple`` regresses the
peers' EV/Revenue on those two by ordinary least squares and evaluates the fitted
line at the target's own fundamentals. The residual, actual less warranted, is
the number worth arguing about: what the market pays over or under what the
fundamentals explain. It is a cross-check on the percentile band, not a
replacement for it, and it refuses to run at all on a sample too small to carry
its parameters.

Percentiles are numpy's default: linear interpolation between order statistics.
With eight peers the 25th percentile is an interpolated value, not an observed
trade, which is worth remembering before it is quoted to two decimal places. The
statistics table always reports quartiles because that is the standard summary of
a sample; the pair actually applied to the target is ``comps.apply_percentiles``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from . import tags
from .config import Assumptions
from .edgar import CompanyFacts, EdgarClient
from .errors import ConfigError, DataSourceError, MissingDataError, NotMeaningfulError
from .ev_bridge import EVBridge, build_ev_bridge, equity_value_from_ev
from .financials import Financials, build_financials
from .market import MarketData

_MM = 1e6

# Attribute, column label, the target metric the multiple is applied to, and how
# that metric reads in a sentence. One table so the peer columns, the statistics
# and the implied-value walk can never drift apart.
MULTIPLES: tuple[tuple[str, str, str, str], ...] = (
    ("ev_revenue", "EV/Revenue", "revenue", "revenue"),
    ("ev_gross_profit", "EV/Gross Profit", "gross_profit", "gross profit"),
    ("ev_ebitda", "EV/EBITDA", "ebitda", "EBITDA"),
    ("ev_ebit", "EV/EBIT", "ebit", "EBIT"),
    ("pe", "P/E", "net_income", "net income"),
)

# P/E is a claim on equity, not on the enterprise. Applying it gives an equity
# value directly, so the bridge is walked backwards for that row rather than
# forwards, and the two are not interchangeable.
_EQUITY_BASIS = frozenset({"pe"})

_BY_ATTR = {attr: (label, metric, text) for attr, label, metric, text in MULTIPLES}

# The sign economics predicts for each driver the regression knows. Not an
# assumption to be tuned: a multiple that falls as growth rises is not a view of
# the world, it is a symptom that one peer is carrying the fit.
_EXPECTED_SIGN: dict[str, float] = {
    "revenue_growth": 1.0,
    "ebitda_margin": 1.0,
    "gross_margin": 1.0,
    "rule_of_40": 1.0,
}


@dataclass
class PeerMetrics:
    """One company priced and reduced to the lines a comp table quotes.

    Money is USD millions and prices are dollars. ``revenue_growth``,
    ``ebitda_margin`` and ``gross_margin`` are decimal fractions. ``rule_of_40``
    is in percentage points, because that is how the market quotes it: a company
    growing 25% at a 15% margin scores 40, not 0.40.
    """

    ticker: str
    name: str
    price: float
    market_cap: float
    enterprise_value: float
    # Gross book debt, carried so a caller can unlever this peer's beta at the
    # same leverage definition the target is relevered at. Net debt would floor
    # at zero for the cash-rich and quietly unlever them at 0.0x.
    gross_debt: float
    revenue: float
    ebitda: float | None
    ebit: float
    net_income: float
    gross_profit: float | None
    revenue_growth: float | None
    ebitda_margin: float | None
    gross_margin: float | None
    rule_of_40: float | None
    ev_revenue: float | None
    ev_gross_profit: float | None
    ev_ebitda: float | None
    ev_ebit: float | None
    pe: float | None
    flags: list[str] = field(default_factory=list)

    def row(self) -> dict[str, object]:
        return {
            "Ticker": self.ticker,
            "Company": self.name,
            "Price": self.price,
            "Market cap": self.market_cap,
            "EV": self.enterprise_value,
            "Revenue": self.revenue,
            "Rev growth": self.revenue_growth,
            "Gross margin": self.gross_margin,
            "EBITDA margin": self.ebitda_margin,
            "Rule of 40": self.rule_of_40,
            "EV/Revenue": self.ev_revenue,
            "EV/Gross Profit": self.ev_gross_profit,
            "EV/EBITDA": self.ev_ebitda,
            "EV/EBIT": self.ev_ebit,
            "P/E": self.pe,
            "Flags": "; ".join(self.flags),
        }


@dataclass
class PeerExclusion:
    """A name that was asked for and could not be built. Always reported."""

    ticker: str
    reason: str


@dataclass
class RegressionFit:
    """A cross-sectional fit of one multiple on fundamentals, with its diagnostics.

    ``warranted_multiple`` is the fitted line read at the target's own growth and
    margin: what the peer cross-section says a company with those characteristics
    should trade at. ``actual_multiple`` is where it does trade, and ``residual``
    is actual less warranted, so a positive residual means the market is paying
    above what the fundamentals in the regression explain. Actual and residual are
    None when the target's own multiple was suppressed as not meaningful, because
    a residual against nothing is nothing.

    ``standard_error`` is the residual standard error, in multiple turns: the
    typical distance between a peer's actual multiple and the line. Quote it next
    to the residual, since a 1.5x residual against a 2.0x standard error is noise.

    ``adj_r_squared`` is the honest one. Raw R-squared cannot fall when a
    regressor is added, so on ten observations and two drivers it flatters the
    fit; the adjusted figure charges for the degrees of freedom spent.
    """

    dependent: str
    drivers: list[str]
    coefficients: dict[str, float]
    intercept: float
    r_squared: float
    adj_r_squared: float
    n_observations: int
    standard_error: float
    t_stats: dict[str, float]
    warranted_multiple: float
    actual_multiple: float | None
    residual: float | None
    implied_ev: float
    implied_price: float
    notes: list[str] = field(default_factory=list)

    def rows(self) -> list[tuple[str, float]]:
        label = _BY_ATTR[self.dependent][0]
        out: list[tuple[str, float]] = [("Intercept", self.intercept)]
        for driver in self.drivers:
            human = driver.replace("_", " ")
            out.append((f"Coefficient on {human}", self.coefficients[driver]))
            out.append((f"t-statistic, {human}", self.t_stats[driver]))
        out.extend(
            [
                ("R-squared", self.r_squared),
                ("Adjusted R-squared", self.adj_r_squared),
                ("Residual standard error (turns)", self.standard_error),
                ("Observations", float(self.n_observations)),
                (f"Warranted {label}", self.warranted_multiple),
            ]
        )
        # Dropped rather than shown as a blank when the target's own multiple was
        # suppressed: the renderer formats numbers, and a residual against a
        # suppressed actual does not exist to be formatted.
        if self.actual_multiple is not None:
            out.append((f"Actual {label}", self.actual_multiple))
        if self.residual is not None:
            out.append(("Residual, actual less warranted", self.residual))
        out.append(("Implied enterprise value", self.implied_ev))
        out.append(("Implied price per share", self.implied_price))
        return out


@dataclass
class CompsResult:
    target: PeerMetrics
    peers: list[PeerMetrics]
    exclusions: list[PeerExclusion]
    stats: pd.DataFrame
    implied: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    # None whenever comps.regression.enabled is off, and also whenever it is on
    # but the sample will not carry a fit. The reason is in ``notes`` either way.
    regression: RegressionFit | None = None

    def table(self) -> pd.DataFrame:
        """The peer set, one row each. The target is held separately.

        The target is deliberately not in this frame: a company included in its
        own comp set pulls the median toward the answer it is being tested
        against, which is circular.
        """
        # Columns come from the target so the frame keeps its shape even when
        # every requested peer ended up in ``exclusions``.
        return pd.DataFrame(
            [p.row() for p in self.peers], columns=list(self.target.row())
        )


def _ratio(numerator: float | None, denominator: float | None) -> float:
    """Divide, or refuse.

    Raises rather than returning a figure that reads like a valuation and is
    not. Callers catch this and record a flag, so the reason survives to the
    printed table instead of being lost as a blank cell.
    """
    if numerator is None:
        raise NotMeaningfulError("the numerator is not meaningful")
    if denominator is None:
        raise NotMeaningfulError("the denominator is not reported")
    if denominator <= 0:
        raise NotMeaningfulError(f"denominator of {denominator:,.0f}mm is not positive")
    return numerator / denominator


def _observed(peers: list[PeerMetrics], attr: str) -> np.ndarray:
    """The peers that actually contributed a value to one multiple."""
    vals = [getattr(p, attr) for p in peers]
    return np.array([v for v in vals if v is not None], dtype=float)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q * 100.0))


def _pct_label(q: float) -> str:
    return f"p{q * 100:g}"


def _revenue_growth(
    fin: Financials, facts: CompanyFacts, flags: list[str]
) -> float | None:
    """Year-over-year growth of trailing twelve month revenue.

    The prior window is anchored on the filer's **own** reported period end
    nearest a year before the current balance-sheet date, then handed to the same
    tiler, so the comparison is twelve reported months against twelve reported
    months.

    A fixed 365-day step looks equivalent and is not. A 52/53-week filer closes
    its year 364 days back, one day before where the step lands, and the tiler
    cannot cover a window whose end falls between two reported periods. Growth
    and Rule of 40 would then read NM for every such filer, permanently, and that
    calendar is common among US technology names. Anchoring on a date the company
    actually reported sidesteps the arithmetic entirely, and also avoids the 29
    February problem that ``date.replace`` would raise on.
    """
    target = fin.as_of - timedelta(days=365)
    _, series, _ = facts.resolve_duration_series("revenue", tags.REVENUE)
    candidates = [f.end for f in series if abs((f.end - target).days) <= 10]
    prior_as_of = (
        min(candidates, key=lambda d: abs((d - target).days)) if candidates else target
    )
    raw, prov = facts.resolve_ttm(
        "prior-year revenue", tags.REVENUE, prior_as_of, required=False
    )
    if raw is None:
        flags.append(
            "Revenue growth NM: the twelve months ended "
            f"{prior_as_of} cannot be tiled from the periods this filer reports"
        )
        return None

    prior = raw / _MM
    if prior <= 0:
        flags.append(
            f"Revenue growth NM: prior-year revenue of {prior:,.0f}mm is not positive"
        )
        return None

    # A growth rate measured across two different revenue tags compares two
    # different definitions of revenue. It is still shown, and it is flagged.
    current = fin.provenance["revenue"].tag if "revenue" in fin.provenance else None
    if current and prov.tag and prov.tag != current:
        flags.append(
            f"Revenue growth spans two tags: {current} this year against "
            f"{prov.tag} a year ago"
        )
    return fin.revenue / prior - 1.0


def _for_company(
    fin: Financials, assumptions: Assumptions, is_target: bool
) -> Assumptions:
    """Assumptions as they apply to one company in the set.

    ``convertibles.conversion_price`` is read from the target's own notes
    footnote and means nothing for anybody else. Left global it would be applied
    to every peer, silently reclassifying each peer's convertibles as equity or
    debt on the strength of a number taken from a different company's indenture.
    MongoDB carries no converts and Cloudflare's convert at a different strike;
    neither is served by Datadog's 148.15.

    So a peer is priced with the treatment set to ``auto`` and no conversion
    price, which is the honest position: its notes go into debt and the bridge
    says out loud that enterprise value may be overstated by that amount. Supply
    each peer's own strike by running it as the target if the multiple matters.
    """
    if is_target:
        return assumptions
    return assumptions.model_copy(
        update={
            "convertibles": assumptions.convertibles.model_copy(
                update={"treatment": "auto", "conversion_price": None}
            )
        }
    )


def compute_peer_metrics(
    fin: Financials,
    facts: CompanyFacts,
    price: float,
    assumptions: Assumptions,
    *,
    is_target: bool = False,
) -> PeerMetrics:
    """Price one company at ``price`` and cut it into multiples.

    Nothing here reaches the network. The caller supplies the normalized
    statements, the fact set behind them and the quote, so the same peer can be
    repriced at a different price without another fetch.
    """
    bridge = build_ev_bridge(fin, price, _for_company(fin, assumptions, is_target))
    ev = bridge.enterprise_value
    flags: list[str] = list(bridge.notes) if not is_target else []

    # A negative enterprise value means the market prices the equity below the
    # cash on hand. It happens, and every EV multiple built on it comes out
    # negative and would drag the whole percentile distribution below zero, so
    # the EV multiples are withheld with the reason on show while P/E survives.
    ev_numerator: float | None = ev
    if ev <= 0:
        flags.append(
            f"EV multiples NM: enterprise value of {ev:,.0f}mm is not positive; "
            "cash and investments exceed market capitalisation plus debt"
        )
        ev_numerator = None

    # The single point where the lease convention enters the multiples. Note
    # what follows from it: the columns headed "EBITDA margin" and "Rule of 40"
    # are computed on this same convention-matched figure, so under the
    # capitalise-leases convention they are EBITDAR-based. That is deliberate,
    # because a margin on one basis sitting beside a multiple on another would
    # describe two different firms, and the table note says which basis is live.
    ebitda, basis = bridge.multiple_denominator(fin)
    ebitda_margin = None if ebitda is None or not fin.revenue else ebitda / fin.revenue
    growth = _revenue_growth(fin, facts, flags)
    rule_of_40 = (
        None
        if growth is None or ebitda_margin is None
        else (growth + ebitda_margin) * 100.0
    )

    def take(
        metric: str, numerator: float, denominator: float | None, note: str
    ) -> float | None:
        try:
            return _ratio(numerator, denominator)
        except NotMeaningfulError as exc:
            flags.append(f"{metric} NM: {note or exc}")
            return None

    if ebitda is None:
        ebitda_note = f"{basis} cannot be formed from this filer's tags"
    elif ebitda_margin is None:
        ebitda_note = "no revenue reported"
    else:
        ebitda_note = f"EBITDA margin {ebitda_margin:.1%}"

    def cap(metric: str, value: float | None, ceiling: float, basis_note: str):
        """Withhold a multiple whose denominator has collapsed toward zero.

        Past the cut-off the ratio is reporting the size of the denominator, not
        the price of the business, and letting it into a percentile drags the
        whole distribution with it.
        """
        if value is None or value <= ceiling:
            return value
        flags.append(
            f"{metric} NM: {value:,.0f}x exceeds the {ceiling:,.0f}x cut-off {basis_note}"
        )
        return None

    # EV/EBIT carries the same lease trap as EV/EBITDA, so it takes its
    # denominator from the bridge too rather than reaching for GAAP EBIT.
    ebit_den, ebit_basis = bridge.ebit_denominator(fin)
    ebit_note = (
        f"{ebit_basis} cannot be formed from this filer's tags"
        if ebit_den is None
        else f"EBIT margin {fin.ebit_margin:.1%}"
    )

    ev_ebitda = take("EV/EBITDA", ev_numerator, ebitda, ebitda_note)
    if ev_ebitda is not None and ev_ebitda > assumptions.comps.ev_ebitda_nm_threshold:
        # Above the cut-off the multiple is measuring how close the margin is to
        # zero, not how the market prices the business.
        flags.append(
            f"EV/EBITDA NM: {ev_ebitda:,.0f}x exceeds the "
            f"{assumptions.comps.ev_ebitda_nm_threshold:,.0f}x cut-off on an "
            f"EBITDA margin of {ebitda_margin:.1%}"
        )
        ev_ebitda = None

    conv = assumptions.convertibles.conversion_price
    if bridge.convertible_in_debt > 0 and conv is None:
        # Worth flagging on every peer that carries converts: if they are in the
        # money their shares already sit inside diluted WASO, and this EV counts
        # the instrument twice.
        flags.append(
            f"EV carries {bridge.convertible_in_debt:,.0f}mm of convertible notes as "
            "debt with no conversion price supplied; if they are in the money EV is "
            "overstated by up to that amount"
        )

    return PeerMetrics(
        ticker=fin.ticker,
        name=fin.entity_name,
        price=price,
        market_cap=bridge.equity_value,
        enterprise_value=ev,
        gross_debt=bridge.total_debt,
        revenue=fin.revenue,
        ebitda=ebitda,
        ebit=fin.ebit,
        net_income=fin.net_income,
        gross_profit=fin.gross_profit,
        revenue_growth=growth,
        ebitda_margin=ebitda_margin,
        gross_margin=fin.gross_margin,
        rule_of_40=rule_of_40,
        ev_revenue=take("EV/Revenue", ev_numerator, fin.revenue, ""),
        ev_gross_profit=take(
            "EV/Gross Profit",
            ev,
            fin.gross_profit,
            "gross profit is not tagged in this filer's statements"
            if fin.gross_profit is None
            else f"gross margin {fin.gross_margin:.1%}",
        ),
        ev_ebitda=ev_ebitda,
        ev_ebit=cap(
            "EV/EBIT",
            take("EV/EBIT", ev_numerator, ebit_den, ebit_note),
            assumptions.comps.ev_ebitda_nm_threshold,
            f"on an EBIT margin of {fin.ebit_margin:.1%}",
        ),
        pe=cap(
            "P/E",
            take(
                "P/E",
                bridge.equity_value,
                fin.net_income,
                "GAAP net loss" if fin.net_income < 0 else "",
            ),
            assumptions.comps.pe_nm_threshold,
            f"on a net margin of {(fin.net_income / fin.revenue if fin.revenue else 0):.1%}",
        ),
        flags=flags,
    )


def _stats_frame(peers: list[PeerMetrics]) -> pd.DataFrame:
    """Sample statistics per multiple, over non-null values only.

    The ``n`` row is the point of the table as much as the median is. A column
    where two of nine peers contributed is a quotation from two companies, and
    the reader has to be able to see that without counting blanks.
    """
    data: dict[str, list[float]] = {}
    for attr, label, _, _ in MULTIPLES:
        vals = _observed(peers, attr)
        if vals.size == 0:
            data[label] = [0.0] + [float("nan")] * 5
            continue
        data[label] = [
            float(vals.size),
            float(vals.min()),
            _percentile(vals, 0.25),
            _percentile(vals, 0.50),
            _percentile(vals, 0.75),
            float(vals.max()),
        ]
    return pd.DataFrame(data, index=["n", "Min", "p25", "Median", "p75", "Max"])


_IMPLIED_COLUMNS = [
    "Methodology",
    "n",
    "Target metric",
    "Low multiple",
    "High multiple",
    "Implied EV low",
    "Implied EV high",
    "Implied price low",
    "Implied price high",
]


def _implied_frame(
    target: PeerMetrics,
    target_fin: Financials,
    bridge: EVBridge,
    peers: list[PeerMetrics],
    assumptions: Assumptions,
    notes: list[str],
) -> pd.DataFrame:
    """Apply the peer percentile range to the target and walk to a share price.

    Each row is one methodology: peer multiple times the target's own metric
    gives an implied enterprise value, and ``equity_value_from_ev`` subtracts the
    same debt the bridge added to get back to equity. Using the bridge in both
    directions is what makes a comps price and a DCF price reconcilable rather
    than merely similar.
    """
    low_q, high_q = assumptions.comps.apply_percentiles
    rows: list[dict[str, object]] = []

    for attr, label, metric_attr, metric_text in MULTIPLES:
        metric = getattr(target, metric_attr)
        vals = _observed(peers, attr)

        if metric is None or metric <= 0:
            shown = (
                "is not reported"
                if metric is None
                else f"of {metric:,.0f}mm is not positive"
            )
            notes.append(
                f"{label} is not applied: {target.ticker}'s own {metric_text} {shown}. "
                "A positive multiple on a negative denominator returns a negative "
                "value, which is not a valuation."
            )
            continue
        if vals.size == 0:
            notes.append(
                f"{label} is not applied: no peer in the set produced a meaningful "
                f"{label}, so there is no percentile to apply."
            )
            continue

        low_m = _percentile(vals, low_q)
        high_m = _percentile(vals, high_q)
        if attr in _EQUITY_BASIS:
            equity_low, equity_high = low_m * metric, high_m * metric
            # Net debt is added back only so this row can be read on the same
            # scale as the enterprise rows above it.
            ev_low, ev_high = equity_low + bridge.net_debt, equity_high + bridge.net_debt
            notes.append(
                f"{label} is an equity multiple, so implied equity value is the "
                "multiple times net income directly. The enterprise value shown for "
                "that row is that equity value plus net debt."
            )
        else:
            ev_low, ev_high = low_m * metric, high_m * metric
            equity_low = equity_value_from_ev(ev_low, target_fin, bridge)
            equity_high = equity_value_from_ev(ev_high, target_fin, bridge)

        rows.append(
            {
                "Methodology": label,
                "n": int(vals.size),
                "Target metric": metric,
                "Low multiple": low_m,
                "High multiple": high_m,
                "Implied EV low": ev_low,
                "Implied EV high": ev_high,
                "Implied price low": equity_low / target_fin.diluted_shares,
                "Implied price high": equity_high / target_fin.diluted_shares,
            }
        )

    frame = pd.DataFrame(rows, columns=_IMPLIED_COLUMNS)
    frame = frame.rename(
        columns={
            "Low multiple": f"Peer {_pct_label(low_q)}",
            "High multiple": f"Peer {_pct_label(high_q)}",
        }
    )
    return frame.set_index("Methodology")


def _walk_to_equity(
    enterprise_value: float,
    target: PeerMetrics,
    target_fin: Financials | None,
    bridge: EVBridge | None,
) -> float:
    """Enterprise value to equity value, on the target's own bridge.

    The percentile walk uses ``equity_value_from_ev``, so this one does too. A
    warranted price and a p25 price are only comparable if they subtract the same
    debt, and two walks written separately drift apart the first time a lease or
    a convertible is reclassified.

    Where the caller did not hand over the bridge, enterprise value less market
    capitalisation is that identical walk in reduced form: the gap between the two
    is by construction the sum of every item the bridge added, under either lease
    convention and either convertible treatment.
    """
    if target_fin is not None and bridge is not None:
        return equity_value_from_ev(enterprise_value, target_fin, bridge)
    return enterprise_value - (target.enterprise_value - target.market_cap)


def fit_warranted_multiple(
    peers: list[PeerMetrics],
    target: PeerMetrics,
    assumptions: Assumptions,
    dependent: str = "ev_revenue",
    *,
    target_fin: Financials | None = None,
    bridge: EVBridge | None = None,
    notes: list[str] | None = None,
) -> RegressionFit | None:
    """Fit one multiple on peer fundamentals and read it at the target.

        EV/Revenue_i = a + b1 * revenue_growth_i + b2 * ebitda_margin_i + e_i

    estimated by ordinary least squares with an intercept across the peer set,
    then evaluated at the target's own growth and margin. The output is the
    multiple the target's characteristics warrant, and the residual against what
    it actually trades at.

    **This is a cross-check on the percentile band, not a replacement for it.** A
    comp set of ten names is a cross-section of ten points. It can tell you that
    the market pays for growth and roughly how much; it cannot support a claim
    that a coefficient is different from zero at any particular confidence. The
    t-statistics are reported because a coefficient of the wrong sign or a
    t-statistic of 0.3 tells the analyst the fit is being carried by one name, and
    below about ten observations they are indicative only, never inferential. No
    p-value is printed, because none of these samples earns one.

    **It refuses rather than overfits.** Fitting two parameters to six points
    memorises the peer set: the line passes near every name by construction, the
    R-squared is high, and the warranted multiple is a restatement of the inputs.
    So the fit is withheld when usable observations fall below
    ``comps.regression.min_observations``, when they fall below three per
    regressor, or when the design matrix is singular because two drivers move
    together across the set. The reason goes to ``notes``, and the honest fix is
    usually a wider ``comps.peers``.

    **Nothing is imputed.** A peer missing the dependent variable or any driver is
    dropped and named. Filling a missing margin with the peer mean would pull that
    name straight onto the fitted line and inflate the R-squared with a number
    nobody reported.
    """
    cfg = assumptions.comps.regression
    if dependent not in _BY_ATTR:
        raise ConfigError(
            f"{dependent!r} is not a multiple this module computes; expected one of "
            + ", ".join(_BY_ATTR)
        )

    drivers = list(cfg.drivers)
    known = set(PeerMetrics.__dataclass_fields__)
    unknown = [d for d in drivers if d not in known]
    if unknown:
        raise ConfigError(
            f"comps.regression.drivers names {', '.join(unknown)}, which are not "
            "fields of PeerMetrics. The drivers this module knows the economics of "
            "are " + ", ".join(_EXPECTED_SIGN) + "."
        )
    if not drivers:
        raise ConfigError(
            "comps.regression.drivers is empty, so there is nothing to explain the "
            "multiple with. An intercept alone is the mean of the peer set, which "
            "the statistics table already reports."
        )

    label, metric_attr, metric_text = _BY_ATTR[dependent]
    detail: list[str] = []

    def refuse(reason: str) -> None:
        detail.append(f"Warranted {label} regression is not available: {reason}")
        if notes is not None:
            notes.extend(detail)

    # -- usable observations ------------------------------------------------ #
    rows: list[list[float]] = []
    observed: list[float] = []
    for peer in peers:
        y = getattr(peer, dependent)
        missing = [] if y is not None else [label]
        missing += [d.replace("_", " ") for d in drivers if getattr(peer, d) is None]
        if missing:
            detail.append(
                f"{peer.ticker} is out of the regression: no "
                f"{', '.join(missing)}. Dropped rather than filled in, because a "
                "peer given the sample mean sits on the fitted line by construction."
            )
            continue
        rows.append([float(getattr(peer, d)) for d in drivers])
        observed.append(float(y))

    n = len(rows)
    k = len(drivers)
    if n < cfg.min_observations:
        refuse(
            f"{n} of {len(peers)} peers carry {label} and every driver, against "
            f"the {cfg.min_observations} that comps.regression.min_observations "
            f"requires. Fitting {k + 1} parameters to {n} points memorises the peer "
            "set instead of measuring the cross-section. Widen comps.peers, or lower "
            "that minimum and read what comes back as description, not as evidence."
        )
        return None
    if n < 3 * k:
        refuse(
            f"{n} observations against {k} regressors is under three points per "
            "regressor, which is not a cross-section. Widen comps.peers or drop a "
            "driver."
        )
        return None

    design = np.array(rows, dtype=float)
    y_vec = np.array(observed, dtype=float)
    x_matrix = np.column_stack([np.ones(n), design])

    sst = float(((y_vec - y_vec.mean()) ** 2).sum())
    if sst <= 0.0:
        refuse(
            f"every peer trades on the same {label}, so there is nothing in the "
            "cross-section for the drivers to explain."
        )
        return None

    beta, _, rank, _ = np.linalg.lstsq(x_matrix, y_vec, rcond=None)
    if rank < k + 1:
        refuse(
            "the design matrix is singular: two drivers move together across this "
            "set, or one does not move at all, so their separate effects cannot be "
            "identified. Drop one of them."
        )
        return None
    try:
        xtx_inv = np.linalg.inv(x_matrix.T @ x_matrix)
    except np.linalg.LinAlgError:
        refuse("the design matrix could not be inverted, so no fit is reported.")
        return None

    # -- fit and diagnostics ------------------------------------------------ #
    resid = y_vec - x_matrix @ beta
    sse = float(resid @ resid)
    dof = n - k - 1
    sigma_sq = sse / dof
    standard_error = float(np.sqrt(sigma_sq))
    r_squared = 1.0 - sse / sst
    # The adjusted figure is the one to quote. Raw R-squared cannot fall when a
    # regressor is added, so on a sample this size it rewards spending degrees of
    # freedom whether or not the driver carries information.
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / dof

    se_beta = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma_sq, 0.0))
    # A standard error of exactly zero means the plane passes through every peer,
    # so the coefficient carries no measurement error and the t-statistic diverges.
    t_values = [
        float(b / s) if s > 0 else float(np.sign(b) * np.inf)
        for b, s in zip(beta, se_beta)
    ]
    intercept = float(beta[0])
    coefficients = {d: float(b) for d, b in zip(drivers, beta[1:])}
    t_stats = {d: t for d, t in zip(drivers, t_values[1:])}
    t_stats["intercept"] = t_values[0]

    for driver, coef in coefficients.items():
        expected = _EXPECTED_SIGN.get(driver)
        if expected is not None and coef * expected < 0:
            detail.append(
                f"Sign check: the coefficient on {driver.replace('_', ' ')} is "
                f"{coef:,.2f}, the wrong way round. Every driver here raises what a "
                "dollar of revenue is worth, so a negative loading points at one "
                "peer with an extreme multiple driving the fit rather than at "
                "economics. Look at the set before using the warranted figure."
            )
    if n < 10:
        detail.append(
            f"With {n} observations the t-statistics are indicative, not "
            "inferential. They are shown so an unsupported coefficient is visible, "
            "not so a confidence level can be quoted."
        )
    if adj_r_squared < 0.0:
        detail.append(
            f"Adjusted R-squared is {adj_r_squared:,.2f}: after charging for the "
            "degrees of freedom spent, these drivers explain less than the peer "
            "mean does. The percentile band is the better read here."
        )

    # -- read the line at the target ---------------------------------------- #
    target_x = [getattr(target, d) for d in drivers]
    absent = [d.replace("_", " ") for d, v in zip(drivers, target_x) if v is None]
    if absent:
        refuse(
            f"{target.ticker}'s own {', '.join(absent)} is not available, so the "
            "fitted line cannot be read at its fundamentals. The coefficients above "
            "stand; the warranted multiple does not exist."
        )
        return None

    for column, driver, value in zip(design.T, drivers, target_x):
        lo, hi = float(column.min()), float(column.max())
        if value < lo or value > hi:
            detail.append(
                f"{target.ticker}'s {driver.replace('_', ' ')} of {value:,.3f} sits "
                f"outside the peer range {lo:,.3f} to {hi:,.3f}, so the warranted "
                "multiple extrapolates the line rather than interpolating inside "
                "the set. That is where a linear fit is least trustworthy."
            )

    warranted = intercept + sum(
        coefficients[d] * float(v) for d, v in zip(drivers, target_x)
    )
    if warranted <= 0.0:
        refuse(
            f"the fitted line warrants {warranted:,.2f}x at {target.ticker}'s "
            f"fundamentals. A negative multiple on positive {metric_text} is not a "
            "valuation, and it means the target sits well outside the range this "
            "peer set spans."
        )
        return None

    metric = getattr(target, metric_attr)
    if metric is None or metric <= 0:
        shown = (
            "is not reported"
            if metric is None
            else f"of {metric:,.0f}mm is not positive"
        )
        refuse(
            f"{target.ticker}'s own {metric_text} {shown}, so the warranted "
            f"{label} of {warranted:,.2f}x has nothing meaningful to multiply."
        )
        return None

    if dependent in _EQUITY_BASIS:
        # An equity multiple lands on equity value directly. Net debt is added
        # back only so the enterprise figure can be read beside the EV rows.
        equity = warranted * metric
        net_debt = (
            bridge.net_debt
            if bridge is not None
            else target.enterprise_value - target.market_cap
        )
        implied_ev = equity + net_debt
    else:
        implied_ev = warranted * metric
        equity = _walk_to_equity(implied_ev, target, target_fin, bridge)

    shares = (
        bridge.diluted_shares if bridge is not None else target.market_cap / target.price
    )
    actual = getattr(target, dependent)
    if actual is None:
        detail.append(
            f"{target.ticker}'s own {label} was suppressed as not meaningful, so "
            "there is a warranted multiple here but no residual to read against it."
        )

    return RegressionFit(
        dependent=dependent,
        drivers=drivers,
        coefficients=coefficients,
        intercept=intercept,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        n_observations=n,
        standard_error=standard_error,
        t_stats=t_stats,
        warranted_multiple=warranted,
        actual_multiple=None if actual is None else float(actual),
        residual=None if actual is None else float(actual) - warranted,
        implied_ev=implied_ev,
        implied_price=equity / shares,
        notes=detail,
    )


def run_comps(
    target_ticker: str,
    assumptions: Assumptions,
    client: EdgarClient,
    market_data: MarketData,
    *,
    peers: list[str] | None = None,
) -> CompsResult:
    """Build the comp set, the statistics over it, and what it implies for the target.

    The target is priced the same way as every peer and on the same conventions,
    so the row printed above the set is directly comparable with it. If the
    target itself cannot be built the error propagates: there is nothing to value.
    """
    requested = list(peers if peers is not None else assumptions.comps.peers)
    notes: list[str] = []

    wanted: list[str] = []
    seen = {target_ticker.upper()}
    for name in requested:
        symbol = name.upper()
        if symbol in seen:
            continue
        seen.add(symbol)
        wanted.append(symbol)

    if len(wanted) < len(requested):
        notes.append(
            f"{target_ticker.upper()} and any duplicate names were removed from the "
            "peer list. A company in its own comp set biases the median toward the "
            "answer being tested."
        )
    if not wanted:
        raise ConfigError(
            "no peers to compare against: set comps.peers in the assumptions file "
            "to a list of tickers, or pass them on the command line"
        )

    target_facts = client.company_facts(target_ticker)
    target_fin = build_financials(target_ticker, facts=target_facts)
    target_price = market_data.spot(target_ticker)
    target_bridge = build_ev_bridge(target_fin, target_price, assumptions)
    target = compute_peer_metrics(
        target_fin, target_facts, target_price, assumptions, is_target=True
    )

    built: list[PeerMetrics] = []
    exclusions: list[PeerExclusion] = []
    for symbol in wanted:
        try:
            facts = client.company_facts(symbol)
            fin = build_financials(symbol, facts=facts)
            built.append(
                compute_peer_metrics(fin, facts, market_data.spot(symbol), assumptions)
            )
        except (MissingDataError, DataSourceError) as exc:
            exclusions.append(PeerExclusion(ticker=symbol, reason=str(exc)))

    stats = _stats_frame(built)

    notes.append(f"Enterprise value on the basis: {target_bridge.lease_convention}.")
    notes.append(
        "Earnings multiples use the figure returned by the EV bridge, so a "
        "lease-inclusive enterprise value is never divided by a post-rent EBITDA."
    )
    notes.append(
        "Percentiles use numpy's linear interpolation between order statistics. "
        "With a small set they are interpolated values, not observed trades."
    )
    notes.append(
        "Rule of 40 is TTM revenue growth plus TTM EBITDA margin, in points. The "
        "free cash flow margin variant is common in growth equity; EBITDA margin "
        "is used here because it is available for the whole set from filings alone."
    )
    for _, label, _, _ in MULTIPLES:
        n = int(stats.loc["n", label])
        if n == 0:
            notes.append(f"{label}: no peer produced a meaningful value.")
        elif n < 3:
            notes.append(
                f"{label}: {n} of {len(built)} peers contributed a value. A "
                "percentile over that many names is a quotation, not a distribution."
            )
    if exclusions:
        notes.append(
            f"{len(exclusions)} of {len(wanted)} requested peers could not be built "
            "and are listed separately. The statistics above are over the rest."
        )

    implied = _implied_frame(
        target, target_fin, target_bridge, built, assumptions, notes
    )

    regression = None
    if assumptions.comps.regression.enabled:
        regression = fit_warranted_multiple(
            built,
            target,
            assumptions,
            target_fin=target_fin,
            bridge=target_bridge,
            notes=notes,
        )

    return CompsResult(
        target=target,
        peers=built,
        exclusions=exclusions,
        stats=stats,
        implied=implied,
        notes=notes,
        regression=regression,
    )
