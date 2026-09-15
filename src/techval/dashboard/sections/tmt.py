"""TMT layer: precedent transactions, segments, the sum of the parts, operating KPIs.

Not a model section. Nothing here is fitted, so there is no baseline to beat and
no headline; what the page shows is what the TMT modules read out of committed
filings, and what they refused to read. Three reads, each run offline against
its own fixture and each pinned to the date that fixture was recorded, because a
recording is a statement about the filing record on that day: run the merger
set as of today and every 2026 deal still pending in it would drift towards
"unresolved" for no reason in the evidence.

**Precedents** come from ``build_precedents`` over every target under
``merger/``, through a client that serves the SEC endpoints from those files and
raises rather than reaching the network. The page draws the two premia side by
side because the module computes both and they disagree exactly where it
matters, and it keeps the rule that a precedent is not a trading comp in the
subtitle of that chart, where it cannot be skipped. A cell the module left
empty is a refusal, and its reason is the module's own text.

**Segments** come from ``build_segments`` on Disney's instance fixture, drawn as
two waterfalls on one axis each rather than one chart on two. Revenue bridges to
consolidated revenue, and the kit itself checks that the parts foot to the total
it is handed. Operating income bridges to the undimensioned
``OperatingIncomeLoss`` for the same year, with the gap shown even when it is
zero, and the title says what a zero means here: that total equals the segment
sum to the dollar, so it is the segments' own total and not operating income
after corporate cost. The margins are a tile row.

**The sum of the parts is refused.** ``run_sotp`` needs a multiple and its source
for every segment and ships no default by design, and no committed fixture
carries one: the only plans in the repository are typed into tests. A waterfall
drawn on multiples written into this collector would be exactly the typed-in
number the dashboard exists not to show. So the collector asks
``value_segments`` to value the segments with no multiple and puts the module's
own refusal on the page, followed by what the fixtures do hold: the segment
EBITDA a plan would price, what is tagged against the corporate member, and
Disney's last committed close.

**KPIs** come from ``build_kpis`` over the four ``instance_kpis_*`` fixtures, with
Datadog's committed 10-K text as its Item 7 and its company facts for the
billings arithmetic. Refused figures are rows, not omissions. Disney's misread
subscriber footnote cannot be one of them: no Disney 10-K text is committed, so
that refusal is recorded as not reproducible offline.

The collector is split so the part that turns module results into figures
(``shape_precedents``, ``shape_segments``, ``shape_kpis``, ``build_takeaway``) is
pure and testable on small fakes, and everything that touches a fixture is in
``collect``.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ...errors import DataSourceError, TechvalError

ID = "tmt"
TITLE = "TMT layer"

MERGER_DIR = "merger"
SEGMENT_TICKER = "DIS"
# The closes the sum of the parts would be set against, named in its refusal.
SEGMENT_PRICES = f"prices/{SEGMENT_TICKER}.csv"
KPI_TICKERS = ("DDOG", "NET", "NFLX", "TMUS")
# The one 10-K text among the KPI filers. Datadog's Item 7 is read for the
# metrics that exist only in prose.
KPI_TEXT = {"DDOG": "filing_text_DDOG_2025.json.gz"}
KPI_FACTS = {"DDOG": "companyfacts_DDOG.json"}

INPUTS: list[str] = [
    MERGER_DIR,
    f"instance_facts_{SEGMENT_TICKER}.json",
    f"companyfacts_{SEGMENT_TICKER}.json",
    SEGMENT_PRICES,
    *(f"instance_kpis_{t}.json" for t in KPI_TICKERS),
    *KPI_FACTS.values(),
    *KPI_TEXT.values(),
]

EP_PRECEDENTS = "techval.tmt.precedents.build_precedents"
EP_SEGMENTS = "techval.tmt.segments.build_segments"
EP_KPIS = "techval.tmt.kpis.build_kpis"

# The precedent figures, in page order. tmt.js lays every figure out by id.
PRECEDENT_FIGURES = ("precedent_tiles", "premia", "ev_revenue", "deals")


class NothingToShow(TechvalError):
    """A module ran and returned nothing a figure could be drawn from."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _figure(kind: str, title: str, subtitle: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "title": title, "subtitle": subtitle, "data": data}


def _refusal(what: str, why: str) -> dict[str, str]:
    return {"what": what, "why": _sentence(why)}


def _sentence(text: str) -> str:
    """Module text as a page sentence: one line, a capital, a full stop."""
    flat = re.sub(r"\s*\n\s*", "; ", str(text)).strip()
    flat = re.sub(r"\s+", " ", flat)
    if not flat:
        return flat
    flat = flat[0].upper() + flat[1:]
    return flat if flat.endswith((".", ")")) else flat + "."


def _clause(text: str) -> str:
    """Module text quoted mid-sentence: one line and a full stop, its case kept."""
    flat = re.sub(r"\s+", " ", re.sub(r"\s*\n\s*", "; ", str(text))).strip()
    return flat if flat.endswith((".", ")")) else flat + "."


def _strip_ticker(flag: str, ticker: str) -> str:
    prefix = f"{ticker}: "
    return flag[len(prefix):] if flag.startswith(prefix) else flag


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _join(names: Sequence[str]) -> str:
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _group_reasons(items: Iterable[tuple[str, str]]) -> list[tuple[list[str], str]]:
    """(ticker, reason) pairs grouped by identical reason, first appearance first."""
    order: list[str] = []
    groups: dict[str, list[str]] = {}
    for ticker, reason in items:
        if reason not in groups:
            order.append(reason)
            groups[reason] = []
        if ticker not in groups[reason]:
            groups[reason].append(ticker)
    return [(groups[r], r) for r in order]


# --------------------------------------------------------------------------- #
# shaping: precedents
# --------------------------------------------------------------------------- #

# Notes on an unpriced deal that say why no single price exists. Everything else
# in Transaction.notes is about dating and conventions, not the refusal.
_UNPRICED_MARKERS = (
    "exchange ratio",
    "stock leg cannot be marked",
    "no per-share cash amount",
    "no single cash figure",
)


def shape_precedents(
    transactions: Sequence[Any],
    stats: Mapping[str, Mapping[str, float]],
    *,
    min_deals: int,
    leak_points: float,
    window_days: int,
    columns: Sequence[tuple[str, str]],
    sub_vertical_stats: Iterable[str] = (),
) -> dict[str, Any]:
    """Figures and refusals from a precedent set.

    ``transactions`` carry ``Transaction``'s attributes; ``stats`` maps each
    column label to its statistic rows (``n``, ``Median``); ``columns`` is the
    module's ``MULTIPLES``. Raises ``NothingToShow`` for an empty set, since a
    set with no deal has no figure to draw and the caller refuses it.
    """
    deals = list(transactions)
    if not deals:
        raise NothingToShow(
            "build_precedents found no merger agreement for any target in the "
            "merger fixtures, so there is no precedent to show"
        )
    figures: dict[str, Any] = {}
    refusals: list[dict[str, str]] = []
    attach: dict[str, list[str]] = {fid: [] for fid in PRECEDENT_FIGURES}

    def refuse(figure: str, what: str, why: str) -> None:
        refusals.append(_refusal(what, why))
        attach[figure].append(what)

    n_deals = len(deals)
    priced = [t for t in deals if t.offer_price is not None]
    unpriced = [t for t in deals if t.offer_price is None]
    both = [t for t in deals if _finite(t.premium_1d) and _finite(t.premium_30d)]
    flagged = [t for t in both if abs(t.premium_gap_points) > leak_points]
    status_counts = Counter(t.status for t in deals)

    # Refusals, in the module's own words.
    for t in unpriced:
        reasons = [n for n in t.notes if any(m in n for m in _UNPRICED_MARKERS)]
        why = " ".join(_sentence(r) for r in reasons) or " ".join(
            _sentence(_strip_ticker(f, t.target_ticker)) for f in t.flags
        )
        if t.cash_per_share is not None:
            why += (
                f" The cash leg of {t.cash_per_share:,.2f} a share is not the offer "
                "and is not put in its place."
            )
        refuse("deals", f"{t.target_ticker} offer price", why)

    no_premium = [
        (t.target_ticker, _strip_ticker(f, t.target_ticker))
        for t in priced
        if not (_finite(t.premium_1d) and _finite(t.premium_30d))
        for f in t.flags
        if "premium" in f and "unaffected" in f
    ]
    for tickers, reason in _group_reasons(no_premium):
        refuse("premia", f"Premia for {_join(tickers)}", reason)

    no_multiple = [
        (t.target_ticker, _strip_ticker(f, t.target_ticker))
        for t in priced
        if t.ev_revenue is None
        for f in t.flags
        if "EV/Revenue" in f or "multiple" in f
    ]
    for tickers, reason in _group_reasons(no_multiple):
        refuse("ev_revenue", f"EV/Revenue for {_join(tickers)}", reason)

    for _attr, label in columns:
        n = int(stats.get(label, {}).get("n", 0))
        if n < min_deals:
            target = "premia" if "Premium" in label else "ev_revenue"
            refuse(
                target,
                f"{label} statistic",
                f"{n} of {n_deals} deals carry a value, below the {min_deals}-deal "
                "floor, so no median is reported for the column",
            )

    buckets = Counter(t.sub_vertical or "unclassified" for t in deals)
    if not list(sub_vertical_stats):
        name, size = sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        refuse(
            "ev_revenue",
            "Statistics by sub-vertical",
            f"no sub-vertical reaches the {min_deals}-deal floor; the largest, "
            f"{name}, holds {size}",
        )

    rev = stats.get("EV/Revenue", {})
    rev_n = int(rev.get("n", 0))
    rev_median = rev.get("Median") if rev_n >= min_deals and _finite(rev.get("Median")) else None

    figures["precedent_tiles"] = _figure(
        "tiles",
        (
            f"{len(priced)} of {n_deals} deals read from merger filings carry an offer "
            f"price, and {len(both)} carry both premia"
        ),
        "Deals, offer prices and premia as the extractor read them from the targets' own filings",
        {
            "tiles": [
                {
                    "label": "Deals read from merger filings",
                    "value": n_deals,
                    "format": "int",
                    "sub": ", ".join(
                        f"{count} {status}" for status, count in sorted(status_counts.items())
                    ),
                },
                {
                    "label": "Offer price fixed at announcement",
                    "value": len(priced),
                    "format": "int",
                    "sub": (
                        f"refused for {_join([t.target_ticker for t in unpriced])}"
                        if unpriced
                        else f"all {n_deals} deals"
                    ),
                },
                {
                    "label": "Both premia measured",
                    "value": len(both),
                    "format": "int",
                    "sub": f"{len(flagged)} move more than {leak_points:.0f} points between the two",
                },
                {
                    "label": "Median EV/Revenue",
                    "value": rev_median,
                    "format": "mult:1",
                    "sub": f"{rev_n} deals carry the multiple; the floor is {min_deals}",
                    **({} if rev_median is not None else {"status": "refused"}),
                },
            ]
        },
    )
    if both:
        rows = sorted(both, key=lambda t: (t.premium_gap_points, t.target_ticker))
        figures["premia"] = _figure(
            "dot",
            f"{len(flagged)} of {len(both)} premia move more than {leak_points:.0f} "
            "points with the choice of unaffected price",
            f"Offer over the last close before announcement, and over the mean close "
            f"of the {window_days} days before; {len(both)} deals with price history. "
            "A precedent is not a trading comp: every precedent price holds a "
            "control premium and the buyer's synergies.",
            {
                "rows": [
                    {
                        "label": t.target_ticker,
                        "values": {"one_day": t.premium_1d, "thirty_day": t.premium_30d},
                        "gap_points": t.premium_gap_points,
                        "flagged": abs(t.premium_gap_points) > leak_points,
                        "acquirer": t.acquirer_name,
                        "announced": t.announced,
                    }
                    for t in rows
                ],
                "series": [
                    {"key": "one_day", "name": "Premium to last close", "role": "model"},
                    {"key": "thirty_day", "name": f"Premium to {window_days}-day mean", "role": "alt"},
                ],
                "format": "pct:1",
                "gapFormat": "pct:1",
                "gapLabel": "Last close minus mean",
                "labelHeader": "Target",
                "zero": True,
                "leakPoints": leak_points,
            },
        )
    else:
        refuse(
            "deals",
            "Premia",
            f"none of the {n_deals} deals carries both an offer price and the closes "
            "before its announcement, so no premium can be drawn",
        )

    with_multiple = [t for t in deals if _finite(t.ev_revenue)]
    if with_multiple:
        ordered = sorted(with_multiple, key=lambda t: (-t.ev_revenue, t.target_ticker))
        title = (
            f"Acquirers paid a median {rev_median:.1f}x trailing revenue across {rev_n} deals"
            if rev_median is not None
            else f"EV/Revenue on {len(with_multiple)} deals, too few for a median"
        )
        figures["ev_revenue"] = _figure(
            "hbar",
            title,
            "Offer enterprise value over the target's trailing twelve months of "
            f"revenue as filed at announcement, {len(with_multiple)} deals. A "
            "precedent multiple sits above a trading multiple by construction.",
            {
                "rows": [
                    {
                        "label": t.target_ticker,
                        "value": t.ev_revenue,
                        "acquirer": t.acquirer_name,
                        "status": t.status,
                        "announced": t.announced,
                    }
                    for t in ordered
                ],
                "median": rev_median,
                "format": "mult:1",
                "valueLabel": "EV/Revenue",
                "labelHeader": "Target",
            },
        )

    figures["deals"] = _figure(
        "table",
        f"{len(both)} of {n_deals} deals carry both a fixed price and the closes to "
        "measure a premium",
        "Every merger agreement read from the targets' own filings, newest first. "
        "A refused cell is one the module declined to fill; each reason sits under "
        "the figure it concerns.",
        {
            "rows": [
                {
                    "ticker": t.target_ticker,
                    "target": t.target_name,
                    "acquirer": t.acquirer_name,
                    "announced": t.announced,
                    "status": t.status,
                    "closed": t.closed,
                    "consideration": t.consideration,
                    "offer_price": t.offer_price,
                    "premium_1d": t.premium_1d,
                    "premium_30d": t.premium_30d,
                    "ev_revenue": t.ev_revenue,
                }
                for t in deals
            ],
        },
    )

    for fid, whats in attach.items():
        if fid in figures and whats:
            figures[fid]["data"]["refusals"] = whats

    widest = max(both, key=lambda t: (abs(t.premium_gap_points), t.target_ticker)) if both else None
    summary = {
        "n_deals": n_deals,
        "n_both": len(both),
        "n_flagged": len(flagged),
        "leak_points": leak_points,
        "widest": (
            {
                "ticker": widest.target_ticker,
                "one_day": widest.premium_1d,
                "thirty_day": widest.premium_30d,
            }
            if widest is not None
            else None
        ),
    }
    return {"figures": figures, "refusals": refusals, "summary": summary}


# --------------------------------------------------------------------------- #
# shaping: segments and the sum of the parts
# --------------------------------------------------------------------------- #


# The reconciling line ``build_segments`` matched the revenue residual to. The
# module states the match in a note rather than a field, so it is read from there
# and the figure falls back to a neutral label when the note is absent.
_EXPLAINED = re.compile(r"The residual is the (.+?) the filer tags, to the dollar")


def _explained_residual(notes: Iterable[str]) -> str | None:
    for note in notes:
        match = _EXPLAINED.search(str(note))
        if match:
            return match.group(1)
    return None


def shape_segments(
    report: Any,
    *,
    operating_total: float | None,
    corporate_tags: Sequence[str],
    sotp_reason: str,
    price: tuple[float, date] | None,
    tolerance: float,
) -> dict[str, Any]:
    """Figures from a ``SegmentReport``, and the refusal of the sum of the parts.

    ``operating_total`` is the undimensioned ``OperatingIncomeLoss`` the instance
    document tags for the report's own period, or None. ``corporate_tags`` are
    the elements tagged against a corporate member for that period, which is
    what decides whether the corporate line could have been read off the filing.
    ``sotp_reason`` is what ``techval.tmt.sotp`` said when it was asked to value
    the segments with no multiple, ``price`` the last committed close on or
    before the recording date with its date, and ``tolerance`` the share of
    revenue ``build_segments`` lets a residual reach and still foot.
    """
    segments = [s for s in report.segments if _finite(s.revenue)]
    if not segments:
        raise NothingToShow(
            f"build_segments returned no segment with revenue for {report.ticker}"
        )
    figures: dict[str, Any] = {}
    refusals: list[dict[str, str]] = []
    attach: dict[str, list[str]] = {}
    ticker = report.ticker
    period = f"fiscal year ended {report.as_of}"

    def refuse(figure: str, what: str, why: str) -> None:
        refusals.append(_refusal(what, why))
        attach.setdefault(figure, []).append(what)

    # -- revenue, bridged to the consolidated total ----------------------------
    allocated = sum(s.revenue for s in segments)
    consolidated = report.consolidated_revenue
    residual = report.unallocated
    explained = _explained_residual(report.notes)
    if residual < 0:
        title = f"Segment revenue overshoots consolidated revenue by {abs(residual):,.0f}"
    elif residual > 0:
        title = f"Segment revenue falls {residual:,.0f} short of consolidated revenue"
    else:
        title = f"Segment revenue foots exactly to consolidated revenue of {consolidated:,.0f}"
    if residual and explained:
        title += f", exactly the {explained} line the filer tags"
    share = abs(residual) / consolidated if consolidated else None
    steps = [{"label": s.name, "value": s.revenue} for s in segments]
    if residual:
        steps.append({"label": explained or "Unallocated / eliminations", "value": residual})
    figures["segment_revenue"] = _figure(
        "waterfall",
        title,
        f"{ticker} revenue by reportable segment, {period}, USD millions, bridged to "
        f"consolidated revenue"
        + (
            f"; the residual is {share:.2%} of revenue, "
            + ("inside" if report.reconciles else "outside")
            + f" the {tolerance:.0%} tolerance"
            if residual and share is not None
            else ""
        )
        + f". 10-K {report.accession}.",
        {
            "steps": steps,
            "total": {"label": "Consolidated revenue", "value": consolidated},
            "format": "num:0",
            "valueLabel": "USD millions",
            "labelHeader": "Line",
            "upLabel": "Segment revenue",
            "downLabel": "Removed on consolidation",
            "totalLegend": "Consolidated revenue",
        },
    )

    # -- operating income, reconciled to what the filing tags for the company --
    earning = [s for s in segments if _finite(s.operating_income)]
    segment_profit = sum(s.operating_income for s in earning)
    if earning:
        steps = [{"label": s.name, "value": s.operating_income} for s in earning]
        if _finite(operating_total):
            gap = operating_total - segment_profit
            steps.append({"label": "Outside the segments", "value": gap})
            total = {"label": "Tagged for the company", "value": operating_total}
            if abs(gap) < 0.5:
                title = (
                    f"Operating income foots to the dollar, because the "
                    f"{operating_total:,.0f} tagged for the company is the segments' own sum"
                )
                tail = (
                    "The undimensioned OperatingIncomeLoss equals the segment sum, so it "
                    "is total segment operating income, struck before corporate and "
                    "unallocated cost. That cost is not in the fixture, and a gap of zero "
                    "says nothing about its size."
                )
            else:
                title = (
                    f"Items outside the segments {'take' if gap < 0 else 'add'} "
                    f"{abs(gap):,.0f} {'off' if gap < 0 else 'to'} segment operating "
                    f"income of {segment_profit:,.0f}"
                )
                tail = (
                    "The gap is the corporate line and every other reconciling item "
                    "between segment profit and the total tagged for the company."
                )
        else:
            total = {"label": "Sum of the segments", "value": segment_profit}
            title = (
                f"Segment operating income sums to {segment_profit:,.0f}, with no company "
                "total to reconcile it to"
            )
            tail = ""
            refuse(
                "segment_operating_income",
                "Operating income reconciliation",
                f"the instance document tags no undimensioned OperatingIncomeLoss for the "
                f"{period}, so segment operating income cannot be reconciled to a company "
                "total, and none is made up for it",
            )
        figures["segment_operating_income"] = _figure(
            "waterfall",
            title,
            f"{ticker} operating income by reportable segment, {period}, USD millions. "
            + tail,
            {
                "steps": steps,
                "total": total,
                "format": "num:0",
                "valueLabel": "USD millions",
                "labelHeader": "Line",
                "upLabel": "Segment operating income",
                "downLabel": "Taken off outside the segments",
                "totalLegend": total["label"],
            },
        )

        with_margin = [s for s in earning if _finite(s.margin)]
        if with_margin:
            top = max(with_margin, key=lambda s: (s.margin, s.name))
            low = min(with_margin, key=lambda s: (s.margin, s.name))
            title = f"{top.name} earns a {top.margin:.1%} operating margin"
            if top is not low and low.margin > 0:
                title += (
                    f", {top.margin / low.margin:.1f} times {_possessive(low.name)} {low.margin:.1%}"
                )
            earning_revenue = sum(s.revenue for s in earning)
            tiles = [
                {
                    "label": s.name,
                    "value": s.margin,
                    "format": "pct:1",
                    "sub": f"{s.operating_income:,.0f} on {s.revenue:,.0f} of revenue",
                }
                for s in with_margin
            ]
            if len(earning) > 1 and earning_revenue > 0:
                tiles.append(
                    {
                        "label": "All segments",
                        "value": segment_profit / earning_revenue,
                        "format": "pct:1",
                        "sub": (
                            f"{segment_profit:,.0f} on {earning_revenue:,.0f}, "
                            "before corporate cost"
                        ),
                    }
                )
            figures["segment_margins"] = _figure(
                "tiles", title, f"{ticker} segment operating margin, {period}.", {"tiles": tiles}
            )

    # -- the sum of the parts: refused, with what is and is not there ----------
    names = _join([s.name for s in segments])
    bases = [
        f"{s.operating_income + s.da:,.0f} in {s.name}"
        for s in segments
        if _finite(s.operating_income) and _finite(s.da)
    ]
    parts = [
        f"No committed fixture carries a multiple and its source for {names}; the only "
        "plans in the repository are typed into tests, and a waterfall drawn on one "
        "would be a typed-in number.",
        f"Asked to value the segments without one, the module refuses: {_clause(sotp_reason)}",
    ]
    if bases:
        parts.append(
            f"What a plan would price is in the filing: segment EBITDA of {_join(bases)}."
        )
    if any("OperatingIncome" in t or "Expense" in t or "Cost" in t for t in corporate_tags):
        parts.append("The corporate line is tagged, but there is nothing to capitalise it at.")
    elif corporate_tags:
        parts.append(
            "The corporate cost it would capitalise is not: the corporate member carries "
            f"only {_join(list(corporate_tags))}."
        )
    else:
        parts.append(
            "The corporate cost it would capitalise is not: nothing is tagged against a "
            "corporate member."
        )
    if price is not None:
        close, day = price
        parts.append(
            f"The price it would be set against is {close:,.2f}, {_possessive(ticker)} close on {day}."
        )
    else:
        parts.append(f"No {ticker} close is committed to set a value per share against.")
    where = "segment_operating_income" if "segment_operating_income" in figures else "segment_revenue"
    refuse(where, "Sum of the parts", " ".join(parts))

    for fid, whats in attach.items():
        figures[fid]["data"]["refusals"] = whats

    summary = {
        "ticker": ticker,
        "n_segments": len(segments),
        "allocated": allocated,
        "consolidated": consolidated,
        "residual": residual,
        "explained": explained,
        "top_margin": (top.name, top.margin) if earning and with_margin else None,
        "low_margin": (low.name, low.margin) if earning and with_margin and low is not top else None,
    }
    return {"figures": figures, "refusals": refusals, "summary": summary}


# --------------------------------------------------------------------------- #
# shaping: operating KPIs
# --------------------------------------------------------------------------- #


def _read_from(filer: Mapping[str, Any]) -> str:
    accession = filer.get("text_accession")
    period = filer.get("text_period")
    where = f"10-K {accession}" if accession else "the filing text"
    return f"{where}, year to {period}" if period else where


def shape_kpis(filers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The KPI table across filers.

    Each filer is ``{ticker, rows, missing, instance_accession, instance_filed,
    text_accession, text_period}``, where ``rows`` are ``KPI.row()`` dictionaries
    with an ``evidence`` label added. A refused figure stays a row with its
    reason.

    A figure read out of prose carries no period of its own: ``build_kpis``
    stamps it with the date it was read. The table says so and names the
    filing the phrase came from instead of passing the read date off as a
    period end.
    """
    table: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for filer in filers:
        for row in filer["rows"]:
            usable = float(row["confidence"]) > 0.0
            prose = row["source"] == "text"
            table.append(
                {
                    "ticker": filer["ticker"],
                    "metric": row["name"],
                    "value": row["value"],
                    "unit": row["unit"],
                    "period_end": None if prose else row["period_end"],
                    "period_stated": not prose,
                    "read_from": _read_from(filer) if prose else None,
                    "evidence": row["evidence"],
                    "tag_or_phrase": row["tag_or_phrase"],
                    "confidence": row["confidence"],
                    "refused": not usable,
                    "reason": _sentence(row["notes"]) if not usable else None,
                }
            )
        sources.append(
            {
                "ticker": filer["ticker"],
                "instance_accession": filer.get("instance_accession"),
                "instance_filed": filer.get("instance_filed"),
                "text_accession": filer.get("text_accession"),
                "text_period": filer.get("text_period"),
                "missing": filer.get("missing", 0),
            }
        )
    if not table:
        raise NothingToShow(
            "build_kpis found no operating figure in any of the "
            f"{len(filers)} KPI fixtures"
        )
    n_total = len(table)
    n_refused = sum(1 for r in table if r["refused"])
    n_tagged = sum(
        1 for r in table if not r["refused"] and str(r["evidence"]).startswith("XBRL")
    )
    from_prose = [r for r in table if not r["period_stated"]]
    n_prose_refused = sum(1 for r in from_prose if r["refused"])
    tickers = [f["ticker"] for f in filers]
    prose = [f["ticker"] for f in filers if f.get("text_accession")]
    if n_refused and from_prose:
        title = (
            f"{n_refused} of {n_total} operating figures are refused, "
            f"{n_prose_refused} of them among the {len(from_prose)} read from prose"
        )
    elif n_refused:
        title = f"{n_refused} of {n_total} operating figures are refused with a reason"
    else:
        title = f"All {n_total} operating figures stand, {n_tagged} of them tagged facts"
    figures = {
        "kpis": _figure(
            "table",
            title,
            f"Operating KPIs read from {len(tickers)} filers' XBRL instance documents"
            + (f", with Item 7 prose and filed arithmetic for {_join(prose)}" if prose else "")
            + ". Each row gives the value, its period and where it came from.",
            {"rows": table, "sources": sources},
        )
    }
    refusals = [
        _refusal(
            "Disney subscriber count",
            "No Disney 10-K text is committed under the fixtures, so the Key Metrics "
            "footnote that was misread as a subscriber base cannot be read again "
            "offline, and its refusal is not reproduced here. The test that pins it "
            "builds the figure by hand.",
        )
    ]
    figures["kpis"]["data"]["refusals"] = ["Disney subscriber count"]
    summary = {
        "n_total": n_total,
        "n_refused": n_refused,
        "n_tagged": n_tagged,
        "n_prose": len(from_prose),
        "n_prose_refused": n_prose_refused,
    }
    return {"figures": figures, "refusals": refusals, "summary": summary}


# --------------------------------------------------------------------------- #
# the takeaway
# --------------------------------------------------------------------------- #


def build_takeaway(
    precedents: Mapping[str, Any] | None,
    segments: Mapping[str, Any] | None,
    kpis: Mapping[str, Any] | None,
) -> str:
    """One short paragraph, every number in it taken from the summaries."""
    out: list[str] = []
    if precedents and precedents.get("widest"):
        w = precedents["widest"]
        out.append(
            f"On {precedents['n_flagged']} of the {precedents['n_both']} deals with price "
            "history, the premium moves by more than "
            f"{precedents['leak_points']:.0f} points depending on which close counts as "
            f"unaffected, most at {w['ticker']}: {w['one_day']:.1%} over the last close "
            f"against {w['thirty_day']:.1%} over the month before."
        )
    elif precedents:
        out.append(
            f"None of the {precedents['n_deals']} precedent deals carries both premia."
        )
    if segments:
        gap = segments["residual"]
        if gap < 0:
            foot = f"overshoots consolidated revenue by {abs(gap):,.0f}"
        elif gap > 0:
            foot = f"falls {gap:,.0f} short of consolidated revenue"
        else:
            foot = "foots exactly to consolidated revenue"
        if gap and segments.get("explained"):
            foot += f", exactly the {segments['explained']} line it tags"
        sentence = f"{_possessive(segments['ticker'])} segment revenue {foot}"
        top, low = segments.get("top_margin"), segments.get("low_margin")
        if top and low:
            sentence += (
                f"; {top[0]} earns a {top[1]:.1%} margin against {_possessive(low[0])} {low[1]:.1%}"
            )
        out.append(
            sentence + ". Its sum of the parts is refused, because no committed fixture "
            "carries a multiple for any segment."
        )
    if kpis:
        text = f"Of {kpis['n_total']} operating figures read, {kpis['n_refused']} are refused"
        if kpis.get("n_prose"):
            text += f", {kpis['n_prose_refused']} of them among the {kpis['n_prose']} read from prose"
        out.append(text + ", and each keeps its row and its reason.")
    return " ".join(out)


# --------------------------------------------------------------------------- #
# fixture clients
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _instance(path: Path):
    """An instance fixture in the shape ``EdgarClient.instance_facts`` returns."""
    from ...edgar import DimensionedFact

    payload = _read_json(path)
    facts = [
        DimensionedFact(
            tag=row["tag"],
            value=row["value"],
            unit=row["unit"],
            start=date.fromisoformat(row["start"]) if row["start"] else None,
            end=date.fromisoformat(row["end"]),
            dimensions=row["dimensions"],
        )
        for row in payload["facts"]
    ]
    return facts, payload["_accession"], date.fromisoformat(payload["_filed"])


def _merger_client(directory: Path, knowledge_date: date):
    """The real ``EdgarClient`` with its JSON endpoints served from the merger fixtures.

    The same construction as ``tests/tmt/test_precedents.py``: subclassing keeps
    ``filings`` and the knowledge-date filtering on the production code path. A
    URL no fixture answers is a ``DataSourceError``, never a network call.
    """
    from ...edgar import (
        SEC_FACTS_URL,
        SEC_SUBMISSIONS_URL,
        SEC_TICKERS_URL,
        EdgarClient,
        HttpCache,
    )

    class MergerFixtureClient(EdgarClient):
        def __init__(self) -> None:
            super().__init__(cache=HttpCache(enabled=False), knowledge_date=knowledge_date)
            self._json: dict[Path, Any] = {}
            self._by_cik: dict[int, str] = {}
            for path in sorted(directory.glob("submissions_*.json")):
                ticker = path.stem.split("_", 1)[1].upper()
                self._by_cik[int(self._load(path)["cik"])] = ticker

        def _load(self, path: Path) -> Any:
            if path not in self._json:
                self._json[path] = _read_json(path)
            return self._json[path]

        @property
        def tickers(self) -> list[str]:
            return sorted(self._by_cik.values())

        def _get_json(self, url: str) -> dict:
            if url == SEC_TICKERS_URL:
                raw = self._load(directory / "company_tickers.json")
                return {k: v for k, v in raw.items() if k != "_fixture"}
            for cik, ticker in self._by_cik.items():
                if url == SEC_SUBMISSIONS_URL.format(cik=cik):
                    return self._load(directory / f"submissions_{ticker}.json")
                if url == SEC_FACTS_URL.format(cik=cik):
                    return self._load(directory / f"companyfacts_{ticker}.json")
            raise DataSourceError(
                f"{url} is not among the committed merger fixtures, and the dashboard "
                "collects offline"
            )

        def ticker_to_cik(self, ticker: str) -> int:
            for cik, known in self._by_cik.items():
                if known == ticker.upper():
                    return cik
            return super().ticker_to_cik(ticker)

        def filing_text(self, ticker: str, filing: dict) -> str:
            # A filing with no committed text stands in for a document carrying no
            # merger agreement, as it does in the precedent tests.
            path = directory / "text" / f"{filing['accession']}.txt"
            packed = path.with_suffix(".txt.gz")
            if path.exists():
                return path.read_text(encoding="utf-8")
            if packed.exists():
                with gzip.open(packed, "rt", encoding="utf-8") as handle:
                    return handle.read()
            return ""

    return MergerFixtureClient()


class _SegmentClient:
    def __init__(self, instance: Path, facts: Path) -> None:
        self._instance = instance
        self._facts = facts

    def instance_facts(self, ticker, forms=("10-K", "10-Q")):
        return _instance(self._instance)

    def company_facts(self, ticker):
        from ...edgar import CompanyFacts

        return CompanyFacts(_read_json(self._facts), ticker)


class _KpiClient:
    """An instance document, company facts where committed, and a 10-K text where committed."""

    def __init__(self, instance: Path, facts: Path | None, text: Path | None) -> None:
        self._instance = instance
        self._facts = facts
        self._text = text
        self._text_payload: dict | None = None

    def _payload(self) -> dict:
        if self._text_payload is None:
            with gzip.open(self._text, "rt", encoding="utf-8") as handle:
                self._text_payload = json.load(handle)
        return self._text_payload

    def company_facts(self, ticker):
        from ...edgar import CompanyFacts

        if self._facts is None:
            return CompanyFacts({"cik": 0, "entityName": ticker, "facts": {"us-gaap": {}}}, ticker)
        return CompanyFacts(_read_json(self._facts), ticker)

    def instance_facts(self, ticker, forms=("10-K", "10-Q")):
        return _instance(self._instance)

    def filings(self, ticker, forms=("10-K",), since=None, limit=20):
        if self._text is None:
            return []
        head = self._payload()
        if head["form"] not in forms:
            return []
        return [
            {
                "accession": head["accession"],
                "filed": date.fromisoformat(head["filed"]),
                "form": head["form"],
                "document": None,
                "period": head["period"],
            }
        ][:limit]

    def filing_text(self, ticker, filing) -> str:
        return self._payload()["text"]


def _corporate_tags(instance: Path, period_end: date) -> list[str]:
    """Elements tagged against a corporate member for the period ending ``period_end``."""
    tags = set()
    for row in _read_json(instance)["facts"]:
        if row["end"] != period_end.isoformat():
            continue
        members = [str(m).split(":")[-1].lower() for m in row["dimensions"].values()]
        if any("corporate" in m for m in members):
            tags.add(row["tag"])
    return sorted(tags)


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #


def _pinned(assumptions, as_of: str):
    """A copy of the assumptions read as of a fixture's recording date."""
    pinned = assumptions.model_copy(deep=True)
    pinned.as_of = as_of
    # A sub-vertical override in the assumptions is a statement about one
    # company. Applied here it would put every deal and every filer in one bucket.
    pinned.tmt.sub_vertical = None
    return pinned


def _collect_precedents(ctx) -> dict[str, Any]:
    from ...market import CsvSource
    from ...tmt import precedents as P

    directory = ctx.input(MERGER_DIR)
    manifest = _read_json(directory / "MANIFEST.json")
    recorded = date.fromisoformat(manifest["retrieved"])
    client = _merger_client(directory, recorded)
    result = P.build_precedents(
        client.tickers,
        client,
        _pinned(ctx.assumptions, recorded.isoformat()),
        prices=CsvSource(directory / "prices"),
    )
    stats = {
        str(label): {str(row): float(result.stats.loc[row, label]) for row in result.stats.index}
        for label in result.stats.columns
    }
    return shape_precedents(
        result.transactions,
        stats,
        min_deals=result.min_deals_for_stats,
        leak_points=P.LEAK_FLAG_POINTS,
        window_days=P.UNAFFECTED_WINDOW_DAYS,
        columns=P.MULTIPLES,
        sub_vertical_stats=list(result.sub_vertical_stats),
    )


def _operating_total(instance: Path, start: date | None, end: date) -> float | None:
    """The undimensioned OperatingIncomeLoss for one period, in USD millions, or None."""
    for row in _read_json(instance)["facts"]:
        if (
            row["tag"] == "OperatingIncomeLoss"
            and not row["dimensions"]
            and row["end"] == end.isoformat()
            and (start is None or row["start"] == start.isoformat())
        ):
            return float(row["value"]) / 1e6
    return None


def _sotp_refusal(report, assumptions) -> str:
    """What ``value_segments`` says when it is handed the segments and no multiple.

    The sum of the parts is not attempted on multiples written here. Asking the
    module with none puts its own refusal on the page, so the page cannot
    claim a stricter or a looser rule than the code enforces.
    """
    from ...tmt.sotp import value_segments

    try:
        value_segments(report.segments, {}, assumptions, ticker=report.ticker)
    except TechvalError as exc:
        return str(exc)
    raise AssertionError(
        "value_segments valued segments with no multiple supplied, which its "
        "contract says it never does"
    )


def _last_close(path: Path, ticker: str, on_or_before: date) -> tuple[float, date] | None:
    from ...market import CsvSource

    try:
        series = CsvSource(path.parent).fetch(
            ticker, date(on_or_before.year - 1, 1, 1), on_or_before
        )
    except TechvalError:
        # A missing close costs the refusal one clause, not the segment figures.
        return None
    pairs = [(d, c) for d, c in zip(series.dates, series.closes) if d <= on_or_before and _finite(c)]
    if not pairs:
        return None
    day, close = pairs[-1]
    return float(close), day


def _collect_segments(ctx) -> dict[str, Any]:
    from ...tmt.segments import RECONCILIATION_TOLERANCE, build_segments

    instance = ctx.input(f"instance_facts_{SEGMENT_TICKER}.json")
    facts = ctx.input(f"companyfacts_{SEGMENT_TICKER}.json")
    recorded = _read_json(instance)["_retrieved"]
    assumptions = _pinned(ctx.assumptions, recorded)
    report = build_segments(SEGMENT_TICKER, _SegmentClient(instance, facts), assumptions)
    start = report.segments[0].period_start if report.segments else None
    return shape_segments(
        report,
        operating_total=_operating_total(instance, start, report.as_of),
        corporate_tags=_corporate_tags(instance, report.as_of),
        sotp_reason=_sotp_refusal(report, assumptions),
        price=_last_close(
            ctx.input(SEGMENT_PRICES), SEGMENT_TICKER, date.fromisoformat(recorded)
        ),
        tolerance=RECONCILIATION_TOLERANCE,
    )


def _collect_kpis(ctx) -> dict[str, Any]:
    from ...commands_tmt import _evidence_label
    from ...nlp.sections import load_sections
    from ...tmt.kpis import build_kpis

    filers = []
    for ticker in KPI_TICKERS:
        instance = ctx.input(f"instance_kpis_{ticker}.json")
        header = _read_json(instance)
        facts = ctx.input(KPI_FACTS[ticker]) if ticker in KPI_FACTS else None
        text = ctx.input(KPI_TEXT[ticker]) if ticker in KPI_TEXT else None
        client = _KpiClient(instance, facts, text)
        mdna = None
        text_accession = None
        text_period = None
        if text is not None:
            sections = load_sections(ticker, client)
            if sections.mdna is not None:
                mdna = sections.mdna.text
                text_accession = sections.accession
                text_period = sections.period
        recorded = date.fromisoformat(header["_retrieved"])
        kpi_set = build_kpis(
            ticker,
            client,
            _pinned(ctx.assumptions, recorded.isoformat()),
            as_of=recorded,
            mdna_text=mdna,
        )
        rows = [dict(row, evidence=_evidence_label(row)) for row in kpi_set.rows()]
        filers.append(
            {
                "ticker": ticker,
                "rows": rows,
                "missing": len(kpi_set.missing),
                "instance_accession": header["_accession"],
                "instance_filed": header["_filed"],
                "text_accession": text_accession,
                "text_period": text_period,
            }
        )
    return shape_kpis(filers)


def _run_part(
    ctx,
    name: str,
    entry_point: str,
    inputs: list[str],
    primary: str,
    build: Callable[[Any], dict[str, Any]],
    refusals: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Run one read, record provenance for each figure it made, or refuse the read.

    The timed block is recorded against ``primary``, the figure the shaping
    always produces when it returns at all. The other figures come out of the
    same run, so their rows carry the same entry point and inputs with no time
    of their own. A read that raises ``TechvalError`` records nothing and
    becomes a refusal of that read alone, so a failure in one module does not
    take the other two off the page.
    """
    try:
        with ctx.record(primary, entry_point, inputs):
            part = build(ctx)
            if primary not in part["figures"]:
                raise NothingToShow(f"{name}: the read produced no {primary} figure")
    except TechvalError as exc:
        refusals.append(_refusal(name, str(exc)))
        return None
    for fid in part["figures"]:
        if fid != primary:
            with ctx.record(fid, entry_point, inputs):
                pass
    refusals.extend(part["refusals"])
    return part


def collect(ctx) -> dict:
    refusals: list[dict[str, str]] = []
    figures: dict[str, Any] = {}

    precedents = _run_part(
        ctx,
        "Precedent transactions",
        EP_PRECEDENTS,
        [MERGER_DIR],
        "deals",
        _collect_precedents,
        refusals,
    )
    segments = _run_part(
        ctx,
        "Segments",
        EP_SEGMENTS,
        [
            f"instance_facts_{SEGMENT_TICKER}.json",
            f"companyfacts_{SEGMENT_TICKER}.json",
            SEGMENT_PRICES,
        ],
        "segment_revenue",
        _collect_segments,
        refusals,
    )
    kpis = _run_part(
        ctx,
        "Operating KPIs",
        EP_KPIS,
        [f"instance_kpis_{t}.json" for t in KPI_TICKERS]
        + list(KPI_FACTS.values())
        + list(KPI_TEXT.values()),
        "kpis",
        _collect_kpis,
        refusals,
    )
    for part in (precedents, segments, kpis):
        if part is not None:
            figures.update(part["figures"])

    if not figures:
        return {"status": "refused", "takeaway": "", "refusals": refusals, "headline": None, "figures": {}}
    return {
        "status": "ok",
        "takeaway": build_takeaway(
            precedents["summary"] if precedents else None,
            segments["summary"] if segments else None,
            kpis["summary"] if kpis else None,
        ),
        "refusals": refusals,
        "headline": None,
        "figures": figures,
    }
