"""Sum of the parts, and the three places it goes wrong.

A TMT conglomerate is not one business. Comcast is a cable network, a film
studio, a streaming service and a theme park operator, and the market pays
something in the mid single digits of EBITDA for the first and low double digits
for the last. Valuing the whole on one consolidated multiple prices it as the
average of things nobody would average, and the error runs in whichever
direction the mix happens to point. Every break-up pitch and every spin-off
argument an advisory firm makes is this arithmetic: value each business on the
multiple its own peers trade at, add them up, take off what the holding company
costs to run, and compare the total against what the market pays for the whole.

**The multiple is an argument, not a number.** Each segment is valued on a
multiple the caller supplies together with the sentence that justifies it, and
that sentence travels to the printed table. "8.5x" is not a valuation view.
"Peer median EV/EBITDA for US cable, Charter and Altice, spot at the valuation
date" is a view, and it can be disagreed with, which is the point. Nothing here
invents a multiple and there is no default: a peer set for a cable network has
no business living in a config file shared with a software DCF.

**Corporate cost is where most sum-of-the-parts go wrong.** Segment operating
income in a 10-K is struck before unallocated corporate expense. The segment
footnote reconciles the sum of segment profit down to consolidated operating
income through a corporate and other line, and that line is a cost. Add the
segments up and stop, and the holding company's own cost of existing has been
valued at zero, which is the single most common way a break-up number comes out
too high. Corporate overhead is a perpetual charge on the enterprise in exactly
the way segment earnings are a perpetual credit, so it is capitalised at a
multiple and subtracted. The multiple used here, absent one from the caller, is
the enterprise-value-weighted average of the *earnings* multiples the segments
were valued on, because the overhead is an earnings charge and because averaging
an EV/Revenue against an EV/EBITDA would be averaging different denominators.
Where no segment was valued on earnings there is no such average and the engine
asks for a multiple rather than inventing one. Where no corporate cost is
supplied at all the total is still produced, and flagged as overstated by an
amount the user can compute, because a silent omission here is worth billions on
a large cap.

**The conglomerate discount is an assumption, and it defaults to zero.** The
empirical literature going back to Lang and Stulz and to Berger and Ofek finds
diversified firms trading at something like 5 to 15 percent below the sum of
their imputed segment values. The counter-argument is that a discount assumed
rather than observed is a fudge factor which can rescue any answer: pick 12
percent and any sum of the parts can be made to agree with any market price, and
the exercise stops being evidence. So it is zero unless the user sets it
deliberately, and when it is set it is applied to the enterprise value *after*
the corporate deduction, not to the gross. That ordering matters and it is worth
saying why it is still not clean: the measured discount in that literature is
the conglomerate's market value against a sum of segment values which typically
does not deduct capitalised corporate cost, so part of what those studies
measured *is* the overhead. Apply the full literature discount on top of a
corporate deduction and some of the same cost is charged twice.

**The gap against the consolidated valuation is the entire exercise.** A sum of
the parts that is not reconciled against what the market pays for the whole is a
number without a claim attached to it. ``implied_vs_consolidated`` puts the two
enterprise values side by side and reports the difference in dollars, in percent
and per share. That difference is the break-up argument, and it is also the
first place to look for an error: a 60 percent gap is usually a wrong multiple
or a missing corporate line long before it is a mispricing.

The walk from enterprise value down to equity uses ``equity_value_from_ev``, the
same debt definition the rest of the engine uses, so a sum-of-the-parts price
per share and a DCF price per share are comparable rather than merely similar.

**Segment input shape.** This module deliberately does not import the segment
builder. It accepts any sequence of mappings or objects carrying:

    name              str, the segment label as the filer reports it
    revenue           float, USD millions, TTM or fiscal year, the caller's choice
    operating_income  float, USD millions, segment profit as reported, which is
                      before unallocated corporate expense and may be negative
    da                float, optional, USD millions, segment depreciation and
                      amortisation where the filer discloses it
    ebitda            float, optional, USD millions, used directly when given and
                      otherwise built as operating_income + da

Mappings are read by key and objects by attribute, so a dataclass from the
segment builder and a hand-typed dict both work, and the periods have to be
consistent because nothing here can check that they are.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from ..config import Assumptions
from ..errors import ConfigError, MissingDataError, NotMeaningfulError
from ..ev_bridge import EVBridge, equity_value_from_ev
from ..financials import Financials

# Metric key -> (how it prints, whether it is an earnings figure). Only revenue
# survives a loss-making segment, so the distinction is carried in one table
# rather than re-derived at each call site.
_METRICS: dict[str, tuple[str, bool]] = {
    "revenue": ("Revenue", False),
    "ebitda": ("EBITDA", True),
    "ebit": ("EBIT", True),
}

# Diagnostic thresholds, not valuation inputs. They move no number, only whether
# a line in ``checks`` carries a FLAG, so they stay out of the assumptions file
# where every entry changes a price.
_CONCENTRATION_FLAG = 0.10
_MULTIPLE_SPREAD_FLAG = 4.0
_CORPORATE_SHARE_FLAG = 0.15
_GAP_FLAG = 0.25

# Segment revenue has to reconcile to consolidated revenue or the parts are not
# the company. Two percent absorbs rounding in a segment footnote and nothing
# else: a real gap means an undisclosed segment, an eliminations line the caller
# dropped, or two different periods being added together.
_COVERAGE_TOLERANCE = 0.02


@dataclass
class SegmentValuation:
    """One business valued on its own peers' multiple. USD millions.

    ``metric_value`` is the segment figure the multiple was applied to, so the
    row can be checked by multiplication without reference to anything else.
    ``share_of_total`` is a decimal fraction of the gross sum of segment values,
    before corporate cost and before any discount, because that is the only base
    against which the segment shares add to one.
    """

    name: str
    metric_name: str
    metric_value: float
    multiple: float
    multiple_source: str
    enterprise_value: float
    share_of_total: float
    notes: list[str] = field(default_factory=list)

    @property
    def multiple_label(self) -> str:
        return f"EV/{self.metric_name}"

    def row(self) -> dict[str, object]:
        return {
            "Segment": self.name,
            "Metric": self.metric_name,
            "Metric value": self.metric_value,
            "Multiple": self.multiple,
            "Basis": self.multiple_label,
            "Enterprise value": self.enterprise_value,
            "% of parts": self.share_of_total,
            "Multiple source": self.multiple_source,
            "Notes": "; ".join(self.notes),
        }


@dataclass
class BreakupGap:
    """The sum of the parts against what the market pays for the whole.

    ``gap`` is the sum-of-the-parts enterprise value less the consolidated
    enterprise value from the priced bridge, so a positive number is the
    break-up case: the parts are worth more apart than the market pays together.

    ``gap_pct`` is None for a company whose consolidated enterprise value is not
    positive, which happens to a net cash filer. Dividing by a denominator at or
    below zero would produce a percentage with the wrong sign or an arbitrary
    magnitude, so the dollars and the per-share figure stand alone and the
    reason is in ``notes``.
    """

    sotp_enterprise_value: float
    consolidated_enterprise_value: float
    gap: float
    gap_pct: float | None
    sotp_per_share: float
    market_price: float
    gap_per_share: float
    notes: list[str] = field(default_factory=list)

    def rows(self) -> list[tuple[str, float]]:
        out = [
            ("Enterprise value, sum of the parts", self.sotp_enterprise_value),
            ("Enterprise value, market", self.consolidated_enterprise_value),
            ("Gap, dollars", self.gap),
        ]
        if self.gap_pct is not None:
            out.append(("Gap, % of market EV", self.gap_pct))
        out.extend(
            [
                ("Value per share, sum of the parts", self.sotp_per_share),
                ("Share price", self.market_price),
                ("Gap per share", self.gap_per_share),
            ]
        )
        return out


@dataclass
class SOTPResult:
    """The parts, the holding company, the discount and the gap.

    ``unallocated_corporate`` and ``conglomerate_discount`` are both stored as
    negative numbers because both are deductions, so the bridge adds down the
    page and a reader never has to guess which way a line points:

        gross_enterprise_value + unallocated_corporate + conglomerate_discount
            = net_enterprise_value

    ``conglomerate_discount_rate`` is the decimal rate that produced the dollar
    line. ``corporate_multiple`` is None only when no corporate cost was
    supplied, in which case ``flags`` says so and the total is overstated.

    Lines in ``checks`` that begin with ``FLAG:`` failed a threshold; the rest
    are stated for the record. ``flags`` carries the conditions that make the
    total itself unreliable rather than merely worth a look.
    """

    segments: list[SegmentValuation]
    gross_enterprise_value: float
    unallocated_corporate: float
    conglomerate_discount: float
    net_enterprise_value: float
    equity_value: float
    per_share: float
    implied_vs_consolidated: BreakupGap
    conglomerate_discount_rate: float
    corporate_cost: float | None = None
    corporate_multiple: float | None = None
    corporate_multiple_source: str | None = None
    checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def rows(self) -> list[tuple[str, float]]:
        """The bridge, read top to bottom from the parts to the gap."""
        gap = self.implied_vs_consolidated
        out: list[tuple[str, float]] = [
            (
                f"{s.name} at {s.multiple:,.1f}x {s.multiple_label}",
                s.enterprise_value,
            )
            for s in self.segments
        ]
        out.extend(
            [
                ("Gross value of the parts", self.gross_enterprise_value),
                ("Capitalised corporate cost", self.unallocated_corporate),
                ("Conglomerate discount", self.conglomerate_discount),
                ("Enterprise value, sum of the parts", self.net_enterprise_value),
                ("Equity value", self.equity_value),
                ("Value per share", self.per_share),
                ("Enterprise value, market", gap.consolidated_enterprise_value),
                ("Gap, dollars", gap.gap),
            ]
        )
        # Dropped rather than shown as a blank when the market enterprise value is
        # not positive: the renderer formats numbers, and a percentage that does
        # not exist has nothing to format.
        if gap.gap_pct is not None:
            out.append(("Gap, % of market EV", gap.gap_pct))
        out.append(("Share price", gap.market_price))
        out.append(("Gap per share", gap.gap_per_share))
        return out

    def to_frame(self) -> pd.DataFrame:
        """The segment table as raw numbers. Formatting belongs to the renderer."""
        return pd.DataFrame([s.row() for s in self.segments])


# --------------------------------------------------------------------------- #
# reading the caller's segments
# --------------------------------------------------------------------------- #


def _read(segment: Any, key: str) -> Any:
    """One accessor for both supported shapes. See the module docstring."""
    if isinstance(segment, Mapping):
        return segment.get(key)
    return getattr(segment, key, None)


def _name(segment: Any, position: int) -> str:
    name = _read(segment, "name")
    if not name:
        raise ConfigError(
            f"the segment in position {position} has no name, so its multiple "
            "cannot be matched to it. Every segment must carry the label the "
            "filer reports it under."
        )
    return str(name)


def _required(segment: Any, key: str, name: str, ticker: str | None) -> float:
    value = _read(segment, key)
    if value is None:
        raise MissingDataError(
            f"{key} for segment {name!r}",
            ticker=ticker,
            hint=(
                "a sum of the parts needs revenue and operating income for every "
                "segment, revenue to reconcile the parts against the consolidated "
                "company and operating income to know whether the segment is "
                "profitable enough to carry an earnings multiple."
            ),
        )
    return float(value)


def _metric_value(segment: Any, metric: str, name: str, ticker: str | None) -> float:
    """The segment figure the multiple is applied to, or a raise.

    Segment EBITDA is the one figure a filer may not give directly. ASC 280
    requires reported segment profit and, where the chief operating decision
    maker reviews it, segment depreciation and amortisation, so EBITDA is built
    as the sum of the two rather than taken on faith. Where the filer discloses
    neither an EBITDA nor a D&A line for the segment it cannot be built, and the
    segment has to be valued on revenue or on operating income instead.
    """
    if metric == "revenue":
        return _required(segment, "revenue", name, ticker)
    if metric == "ebit":
        return _required(segment, "operating_income", name, ticker)

    given = _read(segment, "ebitda")
    if given is not None:
        return float(given)
    da = _read(segment, "da")
    if da is None:
        raise MissingDataError(
            f"segment EBITDA for {name!r}",
            ticker=ticker,
            hint=(
                "neither an ebitda nor a da figure was supplied for this segment, "
                "so EBITDA cannot be built from operating income. Value it on EBIT "
                "or on revenue, or supply segment D&A from the segment footnote."
            ),
        )
    return _required(segment, "operating_income", name, ticker) + float(da)


def _multiple_for(
    name: str, multiples: Mapping[str, tuple[str, float, str]]
) -> tuple[str, float, str]:
    """Resolve and validate one segment's multiple, or refuse.

    The source string is enforced, not merely carried. A sum of the parts is a
    stack of judgments about which peers a business belongs with, and a bare
    number in that stack is untraceable the moment the model leaves the desk it
    was built on.
    """
    if name not in multiples:
        supplied = ", ".join(sorted(multiples)) or "none"
        raise ConfigError(
            f"no multiple supplied for segment {name!r}. Multiples were given for: "
            f"{supplied}. Every segment needs one, because dropping a segment from "
            "the sum silently values that business at zero."
        )
    entry = multiples[name]
    if len(entry) != 3:
        raise ConfigError(
            f"the multiple for segment {name!r} must be (metric, multiple, source); "
            f"got {len(entry)} items."
        )
    metric, multiple, source = entry
    if metric not in _METRICS:
        raise ConfigError(
            f"segment {name!r} was given metric {metric!r}, which is not one of "
            f"{', '.join(sorted(_METRICS))}."
        )
    multiple = float(multiple)
    if multiple <= 0:
        raise ConfigError(
            f"segment {name!r} was given a multiple of {multiple:,.2f}x. A "
            "non-positive multiple is not a view about a business."
        )
    if not str(source).strip():
        raise ConfigError(
            f"segment {name!r} was given a multiple of {multiple:,.1f}x with no "
            "source. State which peer set and which statistic it came from: a bare "
            "multiple is a number, not an argument."
        )
    return metric, multiple, str(source)


# --------------------------------------------------------------------------- #
# valuing the parts
# --------------------------------------------------------------------------- #


def value_segments(
    segments: Sequence[Any],
    multiples: Mapping[str, tuple[str, float, str]],
    assumptions: Assumptions,
    *,
    ticker: str | None = None,
    on_negative: Literal["raise", "nil"] = "raise",
) -> list[SegmentValuation]:
    """Value each segment on the multiple supplied for it.

    ``multiples`` maps the segment name to ``(metric, multiple, source)``, where
    metric is one of revenue, ebitda or ebit, and source is the sentence that
    justifies the number. All three are required for every segment.

    **Loss-making segments.** Multiplying a negative EBITDA by a positive
    multiple produces a large negative enterprise value, which is the claim that
    the business is a liability rather than merely unprofitable. That is almost
    never true: a segment losing money can be closed or sold, so its value has a
    floor at roughly zero and an option value above it. ``on_negative`` decides
    what happens. The default, ``raise``, refuses, because the caller asked for
    an earnings multiple on a business with no earnings and the honest answer is
    to go back and choose a revenue multiple. ``nil`` carries the segment at
    zero, which is a deliberate assumption and is recorded as one on the row.
    Valuing such a segment on revenue needs no special case: pass it a revenue
    multiple and the note explains why.

    ``assumptions`` supplies the lease convention. The multiples themselves are
    deliberately not in it, for the reason given in the module docstring.
    """
    if not segments:
        raise MissingDataError(
            "segment detail",
            ticker=ticker,
            hint=(
                "a sum of the parts needs the segments. A single-segment filer has "
                "nothing to break up and should be valued on the consolidated "
                "multiple instead."
            ),
        )

    names = [_name(s, i) for i, s in enumerate(segments)]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(
            f"segment names repeat: {', '.join(duplicates)}. Multiples are matched "
            "by name, so two segments sharing one would be valued on the same "
            "multiple without either being asked about."
        )
    unmatched = sorted(set(multiples) - set(names))
    if unmatched:
        raise ConfigError(
            "multiples were supplied for segments that were not passed in: "
            f"{', '.join(unmatched)}. Either the name is misspelled or a segment "
            "is missing from the sum, and both change the answer."
        )

    lease_inclusive = assumptions.leases.capitalize_operating_leases

    valuations: list[SegmentValuation] = []
    for segment, name in zip(segments, names):
        metric, multiple, source = _multiple_for(name, multiples)
        label, is_earnings = _METRICS[metric]
        value = _metric_value(segment, metric, name, ticker)
        notes: list[str] = []

        if value <= 0:
            if on_negative == "raise":
                raise NotMeaningfulError(
                    f"segment {name!r} has {label} of {value:,.0f}mm, so the "
                    f"{multiple:,.1f}x supplied would value it at "
                    f"{value * multiple:,.0f}mm. A loss-making business is worth "
                    "nothing rather than worth less than nothing, since it can be "
                    "closed or sold. Value it on revenue, or pass "
                    "on_negative='nil' to carry it at zero deliberately."
                )
            enterprise_value = 0.0
            notes.append(
                f"Carried at nil: {label} of {value:,.0f}mm is not positive, so the "
                f"{multiple:,.1f}x supplied cannot be applied. Nil is a floor and an "
                "assumption, not a measurement, and it ignores whatever the business "
                "would fetch in a sale."
            )
        else:
            enterprise_value = value * multiple

        operating_income = _read(segment, "operating_income")
        if (
            metric == "revenue"
            and operating_income is not None
            and float(operating_income) <= 0
        ):
            notes.append(
                f"Valued on revenue because segment operating income of "
                f"{float(operating_income):,.0f}mm will not carry an earnings "
                "multiple."
            )

        if is_earnings and lease_inclusive:
            notes.append(
                "Segment earnings are struck after rent under ASC 842 while the "
                "engine is configured to count operating lease liabilities as debt. "
                "No segment carries a lease liability, so this value is on the "
                "post-rent convention and does not match the consolidated bridge."
            )

        valuations.append(
            SegmentValuation(
                name=name,
                metric_name=label,
                metric_value=value,
                multiple=multiple,
                multiple_source=source,
                enterprise_value=enterprise_value,
                share_of_total=0.0,
                notes=notes,
            )
        )

    gross = sum(v.enterprise_value for v in valuations)
    if gross <= 0:
        raise NotMeaningfulError(
            "every segment came out at or below zero, so the parts have no value "
            "to take shares of. Check the multiples and the metrics supplied."
        )
    for v in valuations:
        v.share_of_total = v.enterprise_value / gross
    return valuations


# --------------------------------------------------------------------------- #
# the holding company
# --------------------------------------------------------------------------- #


def _corporate_multiple(
    valuations: list[SegmentValuation],
    supplied: float | None,
    source: str | None,
) -> tuple[float, str]:
    """Pick the multiple that capitalises unallocated corporate cost.

    Corporate overhead is an annual earnings charge, so the defensible default
    is the enterprise-value-weighted average of the earnings multiples the
    segments were valued on: the overhead exists to run those businesses, and it
    is capitalised the way their earnings are. Segments valued on revenue are
    excluded from the average, because an EV/Revenue and an EV/EBITDA have
    different denominators and their average is not a multiple of anything.
    Where no segment was valued on earnings there is nothing to average and the
    caller is asked for a multiple instead of being handed an invented one.
    """
    if supplied is not None:
        if not (source or "").strip():
            raise ConfigError(
                f"a corporate multiple of {float(supplied):,.1f}x was supplied with "
                "no source. Say why that multiple: it decides how much of the "
                "holding company's cost is capitalised against the parts."
            )
        if float(supplied) <= 0:
            raise ConfigError(
                f"the corporate multiple of {float(supplied):,.1f}x is not positive. "
                "Capitalising a cost at a negative multiple turns overhead into value."
            )
        return float(supplied), str(source)

    earnings = [v for v in valuations if v.metric_name in ("EBITDA", "EBIT")]
    weight = sum(v.enterprise_value for v in earnings)
    if not earnings or weight <= 0:
        raise ConfigError(
            "no segment was valued on an earnings multiple, so there is no earnings "
            "multiple to capitalise corporate cost at. Supply corporate_multiple and "
            "corporate_multiple_source, or value at least one segment on EBITDA or "
            "EBIT."
        )
    blended = sum(v.multiple * v.enterprise_value for v in earnings) / weight
    bases = sorted({v.multiple_label for v in earnings})
    return blended, (
        f"enterprise-value-weighted average of the segment earnings multiples "
        f"({', '.join(bases)}), because corporate overhead is an earnings charge "
        "on the businesses it supports"
    )


def _coverage(
    fin: Financials, segments: Sequence[Any], names: Sequence[str]
) -> tuple[dict[str, float], float, float]:
    """Segment revenue against consolidated revenue, or a refusal.

    The parts have to be the company. If the segments add to less than
    consolidated revenue there is a business in the filing that this valuation
    is carrying at zero, and if they add to more there is an eliminations line
    that has been dropped, most often intersegment revenue which is real revenue
    for the segment and not revenue for the group. Either way the total is not a
    valuation of this company, so it raises rather than printing.
    """
    by_name = {
        name: _required(segment, "revenue", name, fin.ticker)
        for segment, name in zip(segments, names)
    }
    total = sum(by_name.values())
    if fin.revenue <= 0:
        raise NotMeaningfulError(
            f"consolidated revenue of {fin.revenue:,.0f}mm is not positive, so the "
            "segments cannot be reconciled against it."
        )
    coverage = total / fin.revenue
    if abs(coverage - 1.0) > _COVERAGE_TOLERANCE:
        raise MissingDataError(
            "segment revenue reconciling to consolidated revenue",
            ticker=fin.ticker,
            hint=(
                f"the {len(names)} segments supplied add to {total:,.0f}mm against "
                f"consolidated revenue of {fin.revenue:,.0f}mm, which is "
                f"{coverage:.1%} of the company. A sum of the parts over segments "
                "that do not cover the company values the missing part at zero. "
                "Add the missing segment, or the eliminations line if the segments "
                "overlap."
            ),
        )
    return by_name, total, coverage


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def run_sotp(
    fin: Financials,
    bridge: EVBridge,
    segments: Sequence[Any],
    multiples: Mapping[str, tuple[str, float, str]],
    assumptions: Assumptions,
    *,
    corporate_cost: float | None = None,
    corporate_multiple: float | None = None,
    corporate_multiple_source: str | None = None,
    conglomerate_discount: float = 0.0,
    on_negative: Literal["raise", "nil"] = "raise",
) -> SOTPResult:
    """Value the segments separately, net off the holding company, reconcile.

    ``corporate_cost`` is the annual unallocated corporate expense in USD
    millions, passed as a **positive** number. Segment footnotes report it as
    negative operating income in a corporate and other line, so the sign has to
    be flipped deliberately rather than absorbed: passing the reported negative
    would capitalise the overhead as value and move the answer by twice the
    right amount in the wrong direction. A negative argument therefore raises.

    ``conglomerate_discount`` is a decimal rate, zero by default, applied to the
    enterprise value after the corporate deduction. Both sides of that argument
    are in the module docstring and both are repeated in ``notes`` on every run
    where it is non-zero, because a discount is the one input here that can be
    chosen to produce a desired answer.
    """
    names = [_name(s, i) for i, s in enumerate(segments)] if segments else []
    valuations = value_segments(
        segments, multiples, assumptions, ticker=fin.ticker, on_negative=on_negative
    )
    revenue_by_name, segment_revenue, coverage = _coverage(fin, segments, names)
    income_by_name = {
        name: _required(s, "operating_income", name, fin.ticker)
        for s, name in zip(segments, names)
    }

    checks: list[str] = []
    notes: list[str] = []
    flags: list[str] = []

    gross = sum(v.enterprise_value for v in valuations)

    # -- the holding company ----------------------------------------------- #
    if corporate_cost is None:
        unallocated = 0.0
        used_multiple: float | None = None
        used_source: str | None = None
        flags.append(
            "No corporate cost supplied. Segment operating income is struck before "
            "unallocated corporate expense, so this total values the holding "
            "company's own overhead at zero and is overstated by the capitalised "
            "cost of running it. Take the corporate and other line from the segment "
            "footnote and pass it as corporate_cost."
        )
    else:
        if corporate_cost < 0:
            raise ConfigError(
                f"corporate_cost of {corporate_cost:,.0f}mm is negative. Pass the "
                "annual unallocated corporate expense as a positive cost. The "
                "segment footnote reports it as negative operating income, and "
                "carrying that sign through would add the overhead to value "
                "instead of subtracting it."
            )
        used_multiple, used_source = _corporate_multiple(
            valuations, corporate_multiple, corporate_multiple_source
        )
        unallocated = -corporate_cost * used_multiple
        notes.append(
            f"Corporate cost of {corporate_cost:,.0f}mm a year capitalised at "
            f"{used_multiple:,.1f}x for {-unallocated:,.0f}mm. Multiple source: "
            f"{used_source}."
        )

    subtotal = gross + unallocated
    if subtotal <= 0:
        raise NotMeaningfulError(
            f"capitalised corporate cost of {-unallocated:,.0f}mm is at or above the "
            f"{gross:,.0f}mm the segments are worth, leaving nothing to value. "
            "Either the corporate cost or the segment multiples are wrong."
        )

    # -- the discount -------------------------------------------------------- #
    rate = float(conglomerate_discount)
    if not 0.0 <= rate < 1.0:
        raise ConfigError(
            f"conglomerate_discount of {rate:.1%} is outside [0, 1). A negative "
            "figure is a conglomerate premium, which needs a different argument "
            "than the diversification literature supplies."
        )
    discount = -subtotal * rate
    net_ev = subtotal + discount
    if rate > 0:
        notes.append(
            f"A {rate:.1%} conglomerate discount takes off {-discount:,.0f}mm. The "
            "diversification literature supports something in the 5 to 15 percent "
            "range; against that, a discount assumed rather than observed can be "
            "tuned until the sum of the parts agrees with whatever answer was "
            "wanted, and it is applied here after the corporate deduction, which "
            "already charges part of what those studies were measuring."
        )
    else:
        notes.append(
            "No conglomerate discount applied. The parts are valued as if each "
            "would trade at its peers' multiple on its own, which is the break-up "
            "case rather than the status quo."
        )

    # -- down to equity ------------------------------------------------------ #
    equity = equity_value_from_ev(net_ev, fin, bridge)
    shares = bridge.diluted_shares
    if shares <= 0:
        raise NotMeaningfulError(
            "the bridge carries no shares, so a per-share value cannot be formed."
        )
    per_share = equity / shares

    consolidated_ev = bridge.enterprise_value
    gap = net_ev - consolidated_ev
    gap_notes: list[str] = []
    if consolidated_ev > 0:
        gap_pct: float | None = gap / consolidated_ev
    else:
        gap_pct = None
        gap_notes.append(
            f"Consolidated enterprise value of {consolidated_ev:,.0f}mm is not "
            "positive, which happens to a filer holding more cash than debt, so the "
            "gap is reported in dollars and per share only."
        )
    gap_per_share = per_share - bridge.price
    reconciliation = BreakupGap(
        sotp_enterprise_value=net_ev,
        consolidated_enterprise_value=consolidated_ev,
        gap=gap,
        gap_pct=gap_pct,
        sotp_per_share=per_share,
        market_price=bridge.price,
        gap_per_share=gap_per_share,
        notes=gap_notes,
    )

    checks.extend(
        _checks(
            fin=fin,
            bridge=bridge,
            valuations=valuations,
            revenue_by_name=revenue_by_name,
            income_by_name=income_by_name,
            segment_revenue=segment_revenue,
            coverage=coverage,
            gross=gross,
            unallocated=unallocated,
            net_ev=net_ev,
            gap=gap,
            gap_pct=gap_pct,
        )
    )

    return SOTPResult(
        segments=valuations,
        gross_enterprise_value=gross,
        unallocated_corporate=unallocated,
        conglomerate_discount=discount,
        net_enterprise_value=net_ev,
        equity_value=equity,
        per_share=per_share,
        implied_vs_consolidated=reconciliation,
        conglomerate_discount_rate=rate,
        corporate_cost=corporate_cost,
        corporate_multiple=used_multiple,
        corporate_multiple_source=used_source,
        checks=checks,
        notes=notes,
        flags=flags,
    )


def _checks(
    *,
    fin: Financials,
    bridge: EVBridge,
    valuations: list[SegmentValuation],
    revenue_by_name: Mapping[str, float],
    income_by_name: Mapping[str, float],
    segment_revenue: float,
    coverage: float,
    gross: float,
    unallocated: float,
    net_ev: float,
    gap: float,
    gap_pct: float | None,
) -> list[str]:
    """The lines worth printing beside the total.

    Every one of these is a question an analyst would ask of somebody else's
    sum of the parts, which is why they are computed rather than left to the
    reader to notice.
    """
    out: list[str] = []

    out.append(
        f"Segments add to {segment_revenue:,.0f}mm of revenue against consolidated "
        f"{fin.revenue:,.0f}mm, {coverage:.1%} of the company."
    )

    # Concentration. A segment that is a third of revenue and two thirds of value
    # is the story of the company, and it is also where the whole answer sits: if
    # that one multiple is wrong, nothing else in the table matters.
    largest = max(valuations, key=lambda v: v.enterprise_value)
    revenue_share = revenue_by_name[largest.name] / segment_revenue
    wedge = largest.share_of_total - revenue_share
    line = (
        f"Largest segment {largest.name!r} is {largest.share_of_total:.1%} of the "
        f"value of the parts at {largest.multiple:,.1f}x {largest.multiple_label} "
        f"and {revenue_share:.1%} of revenue, a wedge of {wedge:+.1%}."
    )
    if abs(wedge) > _CONCENTRATION_FLAG:
        line = "FLAG: " + line + (
            " The mix is the valuation: the consolidated multiple this company "
            "trades on is the wrong multiple for most of its revenue."
        )
    out.append(line)

    by_name = {v.name: v for v in valuations}
    loss_making: list[str] = []
    for name in sorted(n for n, oi in income_by_name.items() if oi <= 0):
        v = by_name[name]
        how = (
            "carried at nil"
            if v.enterprise_value == 0.0
            else f"valued on {v.multiple_label}"
        )
        loss_making.append(f"{name} at {income_by_name[name]:,.0f}mm, {how}")
    if loss_making:
        out.append(
            "FLAG: segments reporting no operating profit: "
            + "; ".join(loss_making)
            + ". A revenue multiple prices the top line without asking what it "
            "costs to deliver, an EBITDA multiple on a segment with negative EBIT "
            "rests entirely on the depreciation addback, and nil is an assumption "
            "about a business that could be sold rather than a measurement of one."
        )

    # Multiple range. Within one basis a set spanning more than a few turns is
    # either a genuinely diversified group or a peer set applied too loosely, and
    # the total cannot tell the two apart.
    for basis in sorted({v.multiple_label for v in valuations}):
        band = [v.multiple for v in valuations if v.multiple_label == basis]
        if len(band) < 2:
            continue
        low, high = min(band), max(band)
        spread = high / low
        line = (
            f"{basis} multiples supplied span {low:,.1f}x to {high:,.1f}x, "
            f"{spread:,.1f} times."
        )
        if spread > _MULTIPLE_SPREAD_FLAG:
            line = "FLAG: " + line + (
                " A range that wide inside one basis usually means one of the peer "
                "sets does not fit the business it was applied to."
            )
        out.append(line)

    if unallocated < 0:
        share = -unallocated / gross
        line = (
            f"Capitalised corporate cost is {-unallocated:,.0f}mm, {share:.1%} of "
            "the gross value of the parts."
        )
        if share > _CORPORATE_SHARE_FLAG:
            line = "FLAG: " + line + (
                " Overhead at that scale is a real part of the answer, so the "
                "multiple it is capitalised at deserves as much argument as the "
                "segment multiples do."
            )
        out.append(line)

    if bridge.operating_lease_in_debt > 0:
        out.append(
            "FLAG: the bridge counts "
            f"{bridge.operating_lease_in_debt:,.0f}mm of operating lease "
            "liabilities as debt, so the consolidated enterprise value is lease "
            "inclusive while the segment values are not. Segment footnotes do not "
            "allocate lease liabilities, so the gap below is measured across two "
            "conventions and is overstated in favour of the parts."
        )

    if gap_pct is None:
        out.append(
            f"Sum of the parts is {net_ev:,.0f}mm against a market enterprise value "
            f"of {net_ev - gap:,.0f}mm, a gap of {gap:+,.0f}mm."
        )
    else:
        line = (
            f"Sum of the parts is {gap_pct:+.1%} against the market enterprise "
            f"value, {gap:+,.0f}mm."
        )
        if abs(gap_pct) > _GAP_FLAG:
            line = "FLAG: " + line + (
                " A gap that size is the break-up argument if the multiples are "
                "right, and a wrong multiple or a missing corporate line if they "
                "are not. Check the second before pitching the first."
            )
        out.append(line)

    return out
