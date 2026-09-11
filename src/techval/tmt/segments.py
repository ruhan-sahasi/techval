"""Segment and geography economics, read from dimensioned XBRL.

A media conglomerate valued on its consolidated margin is being valued as an
average of businesses that do not trade at the same multiple. Disney's FY2025
Experiences segment earns a 27.6% operating margin and Entertainment earns 11.0%;
the company average of 18.6% describes neither, and a single EV/EBITDA struck on
it prices parks like a streaming service. The detail that fixes this is in the
filing, tagged under ``StatementBusinessSegmentsAxis``, and it is invisible to
the ``companyfacts`` endpoint, which publishes undimensioned facts only. It has
to be read out of the instance document itself.

**The reconciliation is the control.** Segment revenue must sum to consolidated
revenue. Where it does not, the difference is intersegment eliminations, a
corporate or unallocated bucket, or a member this code failed to read, and it is
reported here as a named residual rather than spread across the segments or
quietly dropped. A segment table that does not foot is the single most common
way segment analysis goes wrong, because every share, every margin and every
sum-of-the-parts weight built on it is then wrong by an amount nobody has
measured. Disney FY2025 is the honest case: the three segments sum to 96,294
against consolidated revenue of 94,425, and the 1,869 difference is the
intersegment revenue the filer eliminates. It is reported, not absorbed.

**The double-counting trap.** Filers tag the same dollar of revenue several
times over, on different cuts. Disney tags Entertainment revenue once whole
(42,466), again split across three geographies, and again split third-party
against intersegment on the product-or-service axis. Any fact carrying more than
one slicing axis is therefore excluded from both the segment table and the
geography table: a fact on segment *and* geography belongs to a cell of the
matrix, not to either margin, and summing the cells alongside the totals counts
the revenue twice. Total-like members (``TotalSegmentsMember``) are dropped for
the same reason, and eliminations, corporate and unallocated members are held out
of the segment list because they are reconciling items rather than businesses.

**One qualifier, not a slice.** ``ConsolidationItemsAxis`` says which part of the
reconciliation a row belongs to rather than cutting the company up. A fact on
that axis with ``OperatingSegmentsMember`` is still a single-axis fact and is
read; on any other member it is a reconciling row and is not. This is what makes
Disney's geography table readable at all: its revenue by geography is tagged with
the operating-segments qualifier beside the geography, and the three regions sum
to consolidated revenue exactly.

**Periods do not mix.** Flows are taken over one annual period, the latest ending
on or before the knowledge date, and every flow in the table comes from that same
period. Assets are taken at the instant equal to that period end and nowhere
else. Pairing a quarterly segment revenue with an annual consolidated figure
produces a segment that looks like a quarter of the business it is.

**Two traps worth naming before a reader uses the output.** The undimensioned
``OperatingIncomeLoss`` in a segment footnote is often *total segment operating
income*, not GAAP operating income: Disney's income statement carries no
operating-income subtotal at all, so the 17,551 sitting undimensioned in FY2025
is the segment total and comparing a segment margin against a consolidated GAAP
EBIT margin from elsewhere in the engine compares two different things. And
segment assets are disclosed only where the chief operating decision maker
reviews them (ASC 280-10-50-22), so a missing asset column is a disclosure fact
about the filer, reported as ``None`` with a note, never as a zero.

Many software companies report one segment. That is a valid answer rather than a
failure: the report comes back with a single segment covering the whole company,
a Herfindahl of 1.0 and a note saying so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .. import tags
from ..config import Assumptions
from ..edgar import CompanyFacts, DimensionedFact
from ..errors import ConfigError, MissingDataError

_MM = 1e6

# Materiality on the reconciliation. This is a disclosure convention rather than
# a valuation input, which is why it is a named constant here and not an
# assumption: it decides what gets flagged to the reader, never what a number is.
# Two percent is the level at which a residual stops being rounding and starts
# being a business somebody forgot to read.
RECONCILIATION_TOLERANCE = 0.02

# Axis local names, lowercased. Filers bind taxonomies to different prefixes, so
# only the local name is worth comparing.
_SEGMENT_AXIS_SUFFIX = "segmentsaxis"
_GEOGRAPHY_AXIS = "statementgeographicalaxis"
_QUALIFIER_AXIS = "consolidationitemsaxis"
_OPERATING_SEGMENTS = "operatingsegments"

# Members that are an aggregate of other members. Reading one is a double count
# of everything under it, so they are dropped outright rather than reconciled.
_AGGREGATE_MEMBERS = (
    "totalsegment",
    "segmenttotal",
    "allsegments",
    "consolidatedentity",
    "entitytotal",
)

# Members that are reconciling items rather than businesses. They are held out of
# the segment list and named in the notes, and the residual is what accounts for
# them. Note that a filer can tag the same elimination twice under two different
# member names (Disney uses both EliminationsAndOtherMember and
# SegmentEliminationsMember for the same 1,869), so these values are reported as
# disclosed and never summed.
_RECONCILING_MEMBERS = (
    "elimination",
    "intersegment",
    "corporate",
    "unallocated",
    "reconcilingitem",
    "nonsegment",
)

# An annual period, in days. A 52/53-week retailer's year runs 364 or 371 days
# and a calendar year runs 365, so the window has to be wider than a point.
_ANNUAL_MIN, _ANNUAL_MAX = 350, 380

# Long-lived assets by geography, which is what ASC 280-10-50-41 requires: total
# assets by country is not disclosed and must not be substituted for it. The
# segment table uses total assets instead; the two columns therefore mean
# different things and each row says which tag produced it.
GEOGRAPHY_ASSETS = ("LongLivedAssets", "PropertyPlantAndEquipmentNet")

# Depreciation alone, kept separate from the combined D&A ladder. For a media
# company the gap between them is the real number: amortisation of acquired
# intangibles and of content is most of Disney's Entertainment D&A, and a
# capital-intensity read that treats combined D&A as depreciation overstates the
# maintenance capex the parks business needs.
DEPRECIATION = ("Depreciation", "DepreciationNonproduction")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Segment:
    """One reported segment, or one geography, over one period.

    Money is USD millions. ``margin`` is operating income over revenue, ``None``
    where revenue is absent or non-positive rather than a ratio nobody can read.
    ``revenue_share`` is a fraction of consolidated revenue, so the shares of a
    group with intersegment eliminations sum to more than one, by exactly the
    eliminated amount.

    A geography row is the same shape with fewer fields filled: ``assets`` there
    carries long-lived assets rather than total assets, because that is the split
    the standard requires by geography. The notes on each row say which tag every
    figure came from and name anything the filing did not disclose.
    """

    name: str
    revenue: float | None
    operating_income: float | None
    assets: float | None
    capex: float | None
    da: float | None
    depreciation: float | None
    margin: float | None
    revenue_share: float | None
    period_start: date | None
    period_end: date
    notes: list[str] = field(default_factory=list)


@dataclass
class SegmentReport:
    """The segment table, its geography cut, and whether it foots.

    ``unallocated`` is the residual: consolidated revenue less the sum of the
    reported segments, in USD millions. It is negative when intersegment revenue
    is eliminated on consolidation, which is the usual case for a group whose
    segments sell to each other, and positive when part of the company sits in a
    corporate or unallocated bucket the filer did not put in a segment. Either
    way it is stated. ``reconciles`` is false when it exceeds
    ``RECONCILIATION_TOLERANCE`` of consolidated revenue.
    """

    ticker: str
    as_of: date
    segments: list[Segment]
    geographies: list[Segment]
    consolidated_revenue: float
    reconciles: bool
    unallocated: float
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    accession: str | None = None
    revenue_tag: str | None = None

    @property
    def largest(self) -> Segment | None:
        """The biggest segment by revenue, which is what the multiple follows."""
        priced = [s for s in self.segments if s.revenue is not None]
        return max(priced, key=lambda s: s.revenue) if priced else None

    @property
    def herfindahl(self) -> float | None:
        """Revenue concentration: the sum of squared segment revenue shares.

        Computed on shares of total segment revenue rather than of consolidated
        revenue, so the shares sum to one and the index keeps its meaning;
        eliminations are not a business and cannot hold a share of one.

        Read it as the number of businesses the company really is. At 0.9 it is
        one business with a rounding error attached and a consolidated multiple
        is the right lens. At 0.25 it is a genuine conglomerate: four comparable
        businesses on four different multiples, and a sum-of-the-parts values it
        better than any single multiple can, because the blended multiple the
        market applies is a weighted average nobody chose.
        """
        base = sum(s.revenue for s in self.segments if s.revenue and s.revenue > 0)
        if base <= 0:
            return None
        return sum(
            (s.revenue / base) ** 2
            for s in self.segments
            if s.revenue and s.revenue > 0
        )

    def to_frame(self) -> pd.DataFrame:
        """The segment table as raw numbers, with the lines that make it foot.

        The residual and the consolidated total are rows, not footnotes. A reader
        adding the revenue column of a segment table and landing somewhere other
        than the company's revenue has found a real problem, and the table should
        show them where it went rather than leave them to discover it.
        """
        return _frame(self.segments, self.unallocated, self.consolidated_revenue)

    def geography_frame(self) -> pd.DataFrame:
        """The geography cut, with its own residual against consolidated revenue.

        A different slice of the same revenue, so it foots to the same total and
        must never be added to the segment table.
        """
        total = sum(g.revenue for g in self.geographies if g.revenue is not None)
        return _frame(
            self.geographies,
            self.consolidated_revenue - total,
            self.consolidated_revenue,
            residual_label="Unallocated / other geographies",
        )


def _frame(
    rows: list[Segment],
    residual: float,
    consolidated: float,
    residual_label: str = "Unallocated / eliminations",
) -> pd.DataFrame:
    records = [
        {
            "Segment": s.name,
            "Revenue": s.revenue,
            "% of revenue": s.revenue_share,
            "Operating income": s.operating_income,
            "Margin": s.margin,
            "D&A": s.da,
            "Depreciation": s.depreciation,
            "Capex": s.capex,
            "Assets": s.assets,
        }
        for s in rows
    ]
    records.append({"Segment": residual_label, "Revenue": residual})
    records.append({"Segment": "Consolidated revenue", "Revenue": consolidated})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Fact handling
# --------------------------------------------------------------------------- #

# (tag, start, end, kind, member). kind is "c" consolidated, "s" segment,
# "g" geography; member is empty for consolidated.
_Key = tuple[str, date | None, date, str, str]


def _classify(fact: DimensionedFact) -> tuple[str, str] | None:
    """Which single-axis total a fact belongs to, or ``None`` to exclude it.

    This is where the double count is prevented. A fact carrying a segment
    member *and* a geography member is one cell of a matrix: it belongs to
    neither the segment total nor the geography total, because both of those are
    already tagged whole elsewhere in the same filing, and adding the cells to
    the totals counts the same revenue twice. The same goes for any fact cut by a
    third axis such as product or service.
    """
    dims = {
        axis.split(":")[-1].lower(): member.split(":")[-1]
        for axis, member in fact.dimensions.items()
    }
    qualifier = dims.pop(_QUALIFIER_AXIS, None)
    if qualifier is not None and _OPERATING_SEGMENTS not in qualifier.lower():
        # Intersegment eliminations, corporate and other reconciling rows travel
        # on this axis. They are not operating slices and cannot join a total.
        return None
    if not dims:
        # A bare fact is the consolidated figure. A fact carrying only the
        # operating-segments qualifier is the total *of the segments*, which for
        # a filer with a corporate bucket is not the same number, so it is not
        # admitted as the consolidated total.
        return ("c", "") if qualifier is None else None
    if len(dims) > 1:
        return None
    axis, member = next(iter(dims.items()))
    if axis.endswith(_SEGMENT_AXIS_SUFFIX):
        return "s", member
    if axis == _GEOGRAPHY_AXIS:
        return "g", member
    return None


def _index(
    facts: list[DimensionedFact], as_of: date | None
) -> tuple[dict[_Key, float], set[tuple[str, date | None, date, str]], list[str]]:
    """Facts reduced to one value per tag, period, axis and member, in millions.

    Also returns the multi-axis facts that were excluded, keyed by tag, period
    and each member they carry, so a blank in the table can say why it is blank.
    A figure that exists in the filing only as the cells of a two-axis split is
    reported as absent, and the row says so, rather than being reconstructed by
    addition: Disney tags Experiences depreciation only as domestic plus foreign.

    Inline XBRL renders the same fact once per table it appears in, so duplicates
    are the norm and are collapsed. Two *different* values under one key is a
    different matter: it means the tag is ambiguous in this filing, and the first
    is kept with both reported in a flag rather than one being chosen silently.
    """
    out: dict[_Key, float] = {}
    crossed: set[tuple[str, date | None, date, str]] = set()
    flags: list[str] = []
    for fact in facts:
        if fact.unit != "USD" or (as_of is not None and fact.end > as_of):
            continue
        kind = _classify(fact)
        if kind is None:
            for member in fact.dimensions.values():
                crossed.add((fact.tag, fact.start, fact.end, member.split(":")[-1]))
            continue
        key = (fact.tag, fact.start, fact.end, kind[0], kind[1])
        value = fact.value / _MM
        prior = out.get(key)
        if prior is None:
            out[key] = value
        elif abs(prior - value) > 0.5:
            flags.append(
                f"{fact.tag} is tagged twice for {kind[1] or 'the consolidated total'} "
                f"over the period ending {fact.end} with different values "
                f"({prior:,.0f} and {value:,.0f}); the first is used"
            )
    return out, crossed, flags


def _annual(key: _Key) -> bool:
    start, end = key[1], key[2]
    return start is not None and _ANNUAL_MIN <= (end - start).days + 1 <= _ANNUAL_MAX


def _pick_revenue(
    index: dict[_Key, float]
) -> tuple[str | None, tuple[date | None, date] | None]:
    """The revenue tag and the annual period the whole table is built on.

    The ladder is walked in order and the first tag that carries segment-level
    revenue wins, because the reconciliation only means anything when the
    segments and the total come from the same concept. Where no tag carries
    segment detail, the first tag with an annual consolidated figure is taken and
    the filer is treated as a single segment.
    """
    ladder = [t for t in tags.REVENUE if isinstance(t, str)]
    for wanted_kind in ("s", "c"):
        for tag in ladder:
            periods = [
                (k[1], k[2])
                for k in index
                if k[0] == tag and k[3] == wanted_kind and _annual(k)
            ]
            if periods:
                return tag, max(periods, key=lambda p: p[1])
    return None, None


def _value(
    index: dict[_Key, float],
    ladder,
    period: tuple[date | None, date],
    kind: str,
    member: str,
) -> tuple[float | None, str | None]:
    """First ladder entry carrying this member over this period, and its tag.

    Composite entries are summed only where every component is present for this
    member and period, the same all-or-nothing rule the company-facts ladders
    use. A partial sum passed off as a total is worse than a blank: Disney tags
    Experiences depreciation only split by geography, so a composite that fired
    on amortisation alone would report a parks segment with no depreciation in it.
    """
    start, end = period
    for entry in ladder:
        if isinstance(entry, tuple):
            parts = [index.get((t, start, end, kind, member)) for t in entry]
            if all(p is not None for p in parts):
                return sum(parts), " + ".join(entry)
            continue
        hit = index.get((entry, start, end, kind, member))
        if hit is not None:
            return hit, entry
    return None, None


def _members(
    index: dict[_Key, float], revenue_tag: str, period, kind: str
) -> list[str]:
    """Members reported for this axis and period, by revenue or operating result.

    Those two define a reportable segment. Members that appear only under a cost
    line are cost allocations rather than businesses: Disney tags a corporate
    depreciation and a corporate capital expenditure and no corporate revenue,
    and a corporate row in a revenue table is an invitation to add it in.
    """
    start, end = period
    wanted = {revenue_tag, *tags.EBIT}
    return sorted(
        {
            k[4]
            for k in index
            if k[0] in wanted and k[1] == start and k[2] == end and k[3] == kind
        }
    )


def _is(member: str, needles: tuple[str, ...]) -> bool:
    low = member.lower()
    return any(n in low for n in needles)


def _label(member: str) -> str:
    """A member name a banker would recognise, out of an XBRL member id."""
    name = member.split(":")[-1]
    for suffix in ("Member", "Segment"):
        while name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
    spaced = "".join(
        f" {ch}" if i and ch.isupper() and not name[i - 1].isupper() else ch
        for i, ch in enumerate(name)
    )
    return spaced.strip() or member


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def _split_only(
    crossed: set[tuple[str, date | None, date, str]],
    ladder,
    period: tuple[date | None, date],
    member: str,
) -> str | None:
    """The tag that carries this figure for this member, but only cross-cut."""
    start, end = period
    for entry in ladder:
        for tag in (entry,) if isinstance(entry, str) else entry:
            if (tag, start, end, member) in crossed:
                return tag
    return None


def _row(
    index: dict[_Key, float],
    crossed: set[tuple[str, date | None, date, str]],
    member: str,
    kind: str,
    period: tuple[date | None, date],
    revenue_tag: str,
    consolidated: float,
) -> Segment:
    """One segment or geography row, with every figure's tag recorded."""
    start, end = period
    axis = "segment" if kind == "s" else "geography"
    notes = [f"Tagged as {member} on the {axis} axis."]

    revenue = index.get((revenue_tag, start, end, kind, member))
    operating_income, oi_tag = _value(index, tags.EBIT, period, kind, member)
    capex, capex_tag = _value(index, tags.CAPEX, period, kind, member)
    da, da_tag = _value(index, tags.DA, period, kind, member)
    depreciation, dep_tag = _value(index, DEPRECIATION, period, kind, member)

    asset_ladder = GEOGRAPHY_ASSETS if kind == "g" else tags.TOTAL_ASSETS
    assets, assets_tag = _value(index, asset_ladder, (None, end), kind, member)

    for label, tag in (
        ("Operating income", oi_tag),
        ("Capital expenditure", capex_tag),
        ("D&A", da_tag),
        ("Depreciation", dep_tag),
        ("Assets", assets_tag),
    ):
        if tag is not None:
            notes.append(f"{label} from {tag}.")

    for label, value, ladder, when in (
        ("Operating income", operating_income, tags.EBIT, period),
        ("Capital expenditure", capex, tags.CAPEX, period),
        ("D&A", da, tags.DA, period),
        ("Depreciation", depreciation, DEPRECIATION, period),
        ("Assets", assets, asset_ladder, (None, end)),
    ):
        if value is not None:
            continue
        split = _split_only(crossed, ladder, when, member)
        if split is not None:
            notes.append(
                f"{label} is tagged for this member only cut by a second axis "
                f"({split}), so it is left blank rather than added back up. A cell "
                "of a two-axis table is not a total, and the same cells are what "
                "double count when they are read alongside the single-axis rows."
            )

    split_assets = _split_only(crossed, asset_ladder, (None, end), member)
    if assets is None and split_assets is None:
        notes.append(
            "Long-lived assets are not tagged for this geography at the period end."
            if kind == "g"
            else (
                "Assets are not tagged for this segment at the period end. ASC 280 "
                "requires them only where the chief operating decision maker reviews "
                "them, so this is a disclosure fact about the filer rather than a "
                "gap in the read."
            )
        )
    if revenue is None:
        notes.append(
            f"No revenue is tagged for this member under {revenue_tag}, so the row "
            "sits outside the revenue reconciliation below."
        )

    margin = (
        operating_income / revenue
        if operating_income is not None and revenue and revenue > 0
        else None
    )
    if margin is None and operating_income is not None:
        notes.append("Margin is not meaningful without positive segment revenue.")

    return Segment(
        name=_label(member),
        revenue=revenue,
        operating_income=operating_income,
        assets=assets,
        capex=capex,
        da=da,
        depreciation=depreciation,
        margin=margin,
        revenue_share=(
            revenue / consolidated if revenue is not None and consolidated else None
        ),
        period_start=start,
        period_end=end,
        notes=notes,
    )


def _rows(
    index: dict[_Key, float],
    crossed: set[tuple[str, date | None, date, str]],
    kind: str,
    period: tuple[date | None, date],
    revenue_tag: str,
    consolidated: float,
) -> tuple[list[Segment], list[str], list[str], list[tuple[str, float | None]]]:
    """Operating rows for one axis, the notes the excluded members earn, and
    the reconciling items the filer tags, which the residual is checked against."""
    rows: list[Segment] = []
    notes: list[str] = []
    flags: list[str] = []
    reconcilers: list[tuple[str, float | None]] = []
    start, end = period
    for member in _members(index, revenue_tag, period, kind):
        value = index.get((revenue_tag, start, end, kind, member))
        if _is(member, _AGGREGATE_MEMBERS):
            notes.append(
                f"{_label(member)} is an aggregate of the other members and is "
                "excluded; reading it would count everything under it twice."
            )
            continue
        if _is(member, _RECONCILING_MEMBERS):
            reconcilers.append((_label(member), value))
            amount = f"{value:,.0f}" if value is not None else "no revenue"
            notes.append(
                f"{_label(member)} is a reconciling item rather than a business, "
                f"tagged at {amount}, and is left out of the table. The residual "
                "below is what accounts for it."
            )
            continue
        row = _row(index, crossed, member, kind, period, revenue_tag, consolidated)
        if row.revenue is None:
            flags.append(
                f"{row.name} is reported with an operating result and no revenue "
                f"under {revenue_tag}, so it carries a margin that cannot be struck "
                "and it is missing from the revenue reconciliation"
            )
        rows.append(row)
    rows.sort(key=lambda s: (s.revenue is None, -(s.revenue or 0.0)))
    return rows, notes, flags, reconcilers


def _consolidated_from_facts(
    facts: CompanyFacts, period: tuple[date | None, date]
) -> float | None:
    """Consolidated revenue for exactly this period, out of company facts.

    Exactly: a period that merely overlaps is a different number, and the point
    of the reconciliation is ruined by a total struck over a window the segments
    were not.
    """
    _tag, series, _ladder = facts.resolve_duration_series("revenue", tags.REVENUE)
    for fact in series:
        if (fact.start, fact.end) == period:
            return fact.val / _MM
    return None


def build_segments(
    ticker: str,
    client,
    assumptions: Assumptions,
    facts: CompanyFacts | None = None,
) -> SegmentReport:
    """Segment and geography economics for one filer, from its latest 10-K.

    The annual report is used rather than the latest filing of any kind. A 10-Q
    segment note covers three and nine months, and a multiple is struck on an
    annual figure; pairing a quarterly segment against an annual consolidated
    total is the mistake this module exists to make impossible.

    ``facts`` is only consulted where the instance document carries no
    undimensioned revenue for the chosen period, which happens for filers whose
    segment note is the only place revenue is tagged. The fallback is flagged,
    because a total and its parts then come from two different sources.
    """
    if not assumptions.tmt.segments:
        raise ConfigError(
            "segment detail is switched off; set tmt.segments true to read the "
            "business-segments axis out of the filing"
        )

    as_of = date.fromisoformat(assumptions.as_of) if assumptions.as_of else None
    dimensioned, accession, _filed = client.instance_facts(ticker, forms=("10-K",))
    index, crossed, flags = _index(dimensioned, as_of)
    revenue_tag, period = _pick_revenue(index)

    notes: list[str] = []
    if revenue_tag is None or period is None:
        if facts is None:
            facts = _company_facts(client, ticker)
        if facts is None:
            raise MissingDataError(
                "segment revenue",
                ticker=ticker,
                tags_tried=[t for t in tags.REVENUE if isinstance(t, str)],
                period="latest annual",
                hint=f"{accession} tags no annual revenue, dimensioned or not",
            )
        revenue_tag, period, consolidated = _annual_from_facts(facts, as_of, ticker)
        flags.append(
            f"{accession} tags no annual revenue at all, so consolidated revenue "
            f"for the year ended {period[1]} is taken from company facts instead"
        )
    else:
        consolidated = index.get((revenue_tag, period[0], period[1], "c", ""))
        if consolidated is None:
            if facts is None:
                facts = _company_facts(client, ticker)
            consolidated = (
                _consolidated_from_facts(facts, period) if facts is not None else None
            )
            if consolidated is None:
                raise MissingDataError(
                    "consolidated revenue",
                    ticker=ticker,
                    tags_tried=[revenue_tag],
                    period=f"{period[0]} to {period[1]}",
                    hint=(
                        f"{accession} tags segment revenue but no undimensioned total "
                        "for the same period, so the segments cannot be reconciled"
                    ),
                )
            flags.append(
                f"{accession} tags no undimensioned revenue for the year ended "
                f"{period[1]}; the total is from company facts and the segments are "
                "from the filing, so the reconciliation crosses two sources"
            )

    segments, segment_notes, segment_flags, reconcilers = _rows(
        index, crossed, "s", period, revenue_tag, consolidated
    )
    geographies, geo_notes, geo_flags, _ = _rows(
        index, crossed, "g", period, revenue_tag, consolidated
    )
    notes += segment_notes + geo_notes
    flags += segment_flags + geo_flags

    if not segments:
        operating_income = _value(index, tags.EBIT, period, "c", "")[0]
        segments = [
            Segment(
                name=facts.entity_name if facts is not None else ticker.upper(),
                revenue=consolidated,
                operating_income=operating_income,
                assets=_value(index, tags.TOTAL_ASSETS, (None, period[1]), "c", "")[0],
                capex=_value(index, tags.CAPEX, period, "c", "")[0],
                da=_value(index, tags.DA, period, "c", "")[0],
                depreciation=_value(index, DEPRECIATION, period, "c", "")[0],
                margin=(
                    operating_income / consolidated
                    if operating_income is not None and consolidated > 0
                    else None
                ),
                revenue_share=1.0,
                period_start=period[0],
                period_end=period[1],
                notes=["The whole company, undimensioned."],
            )
        ]
        notes.append(
            f"{accession} tags no revenue on a business-segments axis. That is how a "
            "single-reportable-segment filer reports under ASC 280, and it is an "
            "answer rather than a gap: there is nothing to sum the parts of, and a "
            "consolidated multiple is the right lens."
        )

    allocated = sum(s.revenue for s in segments if s.revenue is not None)
    unallocated = consolidated - allocated
    reconciles = (
        abs(unallocated) <= RECONCILIATION_TOLERANCE * consolidated
        if consolidated
        else False
    )
    if not reconciles:
        flags.append(
            f"segment revenue of {allocated:,.0f} does not reconcile to consolidated "
            f"revenue of {consolidated:,.0f}: the residual of {unallocated:,.0f} is "
            f"{abs(unallocated) / consolidated:.2%} of the company, past the "
            f"{RECONCILIATION_TOLERANCE:.0%} tolerance. Treat every share and margin "
            "below as approximate until the missing member is found."
        )
    elif unallocated:
        notes.append(
            f"The segments sum to {allocated:,.0f} against consolidated revenue of "
            f"{consolidated:,.0f}. The residual of {unallocated:,.0f} is "
            f"{abs(unallocated) / consolidated:.2%} of revenue and is reported as "
            "unallocated rather than pushed into a segment."
        )

    # Whether the residual is explained is a stronger control than whether it is
    # small. A gap that equals the eliminations the filer tags is the table
    # working; a gap of the same size that matches nothing disclosed means a
    # member was missed, and the two deserve different words.
    #
    # Matched on magnitude, not on sign. Filers tag intersegment revenue both
    # ways: Disney as a negative adjustment of -1,869, others as a positive
    # amount the reader is told to deduct. Failing the control on a presentation
    # convention would cry wolf on a table that is in fact complete.
    explained = [
        name
        for name, value in reconcilers
        if value is not None and abs(abs(value) - abs(unallocated)) <= 0.5
    ]
    if unallocated and explained:
        notes.append(
            f"The residual is the {explained[0]} the filer tags, to the dollar, so "
            "no reportable segment is missing from the table."
        )
    elif unallocated and reconcilers:
        flags.append(
            f"the residual of {unallocated:,.0f} matches none of the reconciling "
            f"items the filer tags ({', '.join(name for name, _ in reconcilers)}), "
            "so something in the segment note has not been read"
        )

    if geographies:
        geo_total = sum(g.revenue for g in geographies if g.revenue is not None)
        gap = consolidated - geo_total
        if abs(gap) > RECONCILIATION_TOLERANCE * consolidated:
            flags.append(
                f"geography revenue of {geo_total:,.0f} leaves {gap:,.0f} against "
                "consolidated revenue. The geography table is a different cut of the "
                "same revenue, so it should foot to the same total; treat it as a "
                "partial split"
            )
        notes.append(
            "The geography rows are a second cut of the same revenue, not an "
            "addition to the segment table."
        )

    return SegmentReport(
        ticker=ticker.upper(),
        as_of=period[1],
        segments=segments,
        geographies=geographies,
        consolidated_revenue=consolidated,
        reconciles=reconciles,
        unallocated=unallocated,
        notes=notes,
        flags=flags,
        accession=accession,
        revenue_tag=revenue_tag,
    )


def _company_facts(client, ticker: str) -> CompanyFacts | None:
    """Company facts, where the client can serve them and the caller did not.

    A client that cannot serve them at all is a legitimate caller and gets a
    ``None`` to fall through on. A client that can and then fails is not caught:
    the reason the source failed is more useful to whoever has to audit the gap
    than a tidier error raised over the top of it.
    """
    getter = getattr(client, "company_facts", None)
    return getter(ticker) if getter is not None else None


def _annual_from_facts(
    facts: CompanyFacts, as_of: date | None, ticker: str
) -> tuple[str, tuple[date | None, date], float]:
    """Latest annual revenue from company facts, for a filing with none tagged."""
    tag, series, ladder = facts.resolve_duration_series("revenue", tags.REVENUE)
    annuals = [
        f
        for f in series
        if f.start is not None
        and _ANNUAL_MIN <= f.days <= _ANNUAL_MAX
        and (as_of is None or f.end <= as_of)
    ]
    if not annuals or tag is None:
        raise MissingDataError(
            "annual revenue",
            ticker=ticker,
            tags_tried=[t for t in ladder if isinstance(t, str)],
            period="latest annual",
            hint="neither the instance document nor company facts carry a full year",
        )
    latest = max(annuals, key=lambda f: f.end)
    return tag, (latest.start, latest.end), latest.val / _MM
