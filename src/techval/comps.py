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
class CompsResult:
    target: PeerMetrics
    peers: list[PeerMetrics]
    exclusions: list[PeerExclusion]
    stats: pd.DataFrame
    implied: pd.DataFrame
    notes: list[str] = field(default_factory=list)

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


def _ratio(numerator: float, denominator: float | None) -> float:
    """Divide, or refuse.

    Raises rather than returning a figure that reads like a valuation and is
    not. Callers catch this and record a flag, so the reason survives to the
    printed table instead of being lost as a blank cell.
    """
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

    The prior window is anchored 365 days before the current balance-sheet date
    and handed to the same tiler, so the comparison is twelve reported months
    against twelve reported months. Stepping back with ``date.replace`` instead
    would raise on a 29 February year end, and a 52/53-week filer does not report
    on the calendar anniversary anyway.
    """
    prior_as_of = fin.as_of - timedelta(days=365)
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


def compute_peer_metrics(
    fin: Financials,
    facts: CompanyFacts,
    price: float,
    assumptions: Assumptions,
) -> PeerMetrics:
    """Price one company at ``price`` and cut it into multiples.

    Nothing here reaches the network. The caller supplies the normalized
    statements, the fact set behind them and the quote, so the same peer can be
    repriced at a different price without another fetch.
    """
    bridge = build_ev_bridge(fin, price, assumptions)
    ev = bridge.enterprise_value
    flags: list[str] = []

    # The single point where the lease convention enters the multiples.
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

    ev_ebitda = take("EV/EBITDA", ev, ebitda, ebitda_note)
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
        revenue=fin.revenue,
        ebitda=ebitda,
        ebit=fin.ebit,
        net_income=fin.net_income,
        gross_profit=fin.gross_profit,
        revenue_growth=growth,
        ebitda_margin=ebitda_margin,
        gross_margin=fin.gross_margin,
        rule_of_40=rule_of_40,
        ev_revenue=take("EV/Revenue", ev, fin.revenue, ""),
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
            take("EV/EBIT", ev, fin.ebit, f"EBIT margin {fin.ebit_margin:.1%}"),
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
        target_fin, target_facts, target_price, assumptions
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

    return CompsResult(
        target=target,
        peers=built,
        exclusions=exclusions,
        stats=stats,
        implied=implied,
        notes=notes,
    )
