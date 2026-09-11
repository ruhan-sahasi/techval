"""The operating metrics TMT is priced on, and the reasons most of them are hard.

A software company is quoted on annual recurring revenue, net revenue retention,
remaining performance obligation and billings. A streamer is quoted on paid
subscribers, average revenue per user and content spend. A carrier is quoted on
postpaid net additions, ARPU and churn. Not one of those is a us-gaap concept.
The taxonomy has ``Revenues`` and ``ContractWithCustomerLiabilityCurrent``; it
has no ARR, no NRR, no ARPU and no churn, so a generic engine reading standard
tags is structurally blind to the numbers the sector actually trades on. That
gap is the whole reason this module exists.

There are exactly three places the numbers can come from, and they are not of
equal quality, so each figure records which one it came from.

**Tagged facts, and the namespace problem.** ``companyfacts`` publishes only the
standard taxonomies, so a filer's own extension elements are invisible there and
have to be read out of the filing's XBRL instance document instead. Four such
documents are committed with this package, one per business model in the sector,
and what they tag decides the shape of everything below. Every claim here is
checked by a test against those files rather than asserted:

    Remaining performance obligation is tagged, as the us-gaap concept
    ``RevenueRemainingPerformanceObligation``. Datadog, Cloudflare and T-Mobile
    all carry it, across software and telecom alike.

    Netflix tags content spend and content amortisation as its own extension
    elements, ``AdditionstoStreamingContentAssets`` and
    ``CostofServicesAmortizationofStreamingContentAssets``, which is why the
    concept ladders here have to reach outside us-gaap at all.

    Not one of the four tags ARR, a customer count, net revenue retention, a
    subscriber count, ARPU or churn. Netflix, whose entire equity story is paid
    memberships, does not tag a membership count. Those metrics exist only in
    prose, which is why the text path here is not a convenience: without it,
    half the sector's metrics do not exist at all.

    What the four do carry, in quantity, is near misses. T-Mobile tags
    ``NumberOfCustomerAccountsImpacted``, ``NumberofCustomerClasses`` and
    ``NumberOfCustomerCategories``; Cloudflare tags ``NumberOfSegmentManagers``
    and ``RevenueRemainingPerformanceObligationPercentage``. A substring search
    for "customer" would return all of those as customer counts. That is why the
    ladders below are anchored full-match patterns and not keyword filters.

The parser reduces every element to its local name, so ``ddog:ARR`` and
``us-gaap:Revenues`` arrive alike. That is deliberate, because prefixes differ
between filers and between taxonomy versions and matching on local names
survives both. It also means the namespace cannot be read back off the fact, so
whether a matched element is a standard concept or a filer extension is decided
against ``STANDARD_CONCEPTS`` below and recorded in the note, rather than
claimed from evidence the document no longer carries.

**Derived figures, and billings above all.** Billings is the clearest case of a
number every software analyst quotes and no filer reports:

    billings = revenue + the change in deferred revenue over the same window

Revenue is what was delivered; deferred revenue is what was invoiced and not yet
delivered. Adding the period's increase in the second to the first recovers what
was invoiced. It is an approximation and the places it breaks are known: a shift
from annual to multi-year prepayment inflates it without any change in demand, a
shift to monthly billing deflates it the same way, and an acquisition brings
acquired deferred revenue onto the balance sheet as a step that never passed
through invoicing. So it is marked ``derived``, and the note says what it
approximates rather than presenting it as disclosure.

**Prose, parsed conservatively.** "Net revenue retention rate of 115%" is a
number. "Net dollar retention of over 120%" is not 120%, it is a floor, and a
model fed the second as though it were the first is being fed a fiction that is
always wrong in the same direction. Every text match therefore carries the
qualifier it was hedged with, the exact phrase it came from, and a confidence
that reflects both. Scale is resolved explicitly or the match is refused: "2.4
million subscribers" is 2,400,000, "2.4 subscribers" is a sentence this module
will not guess at.

**Refusal, conflict and bounds.** Three rules, and they exist because the
failure they prevent is silent:

    A figure the extractor cannot resolve is reported with confidence 0.0 and
    the reason attached. It is visible to a reader and invisible to a model:
    ``as_dict`` returns None for it. Reporting nothing at all would hide that
    the filing said something the engine could not read.

    Where two extractions disagree, both are kept and the disagreement is
    flagged. Silently preferring one is how a wrong KPI reaches a page. Where
    the disagreement is inside one filing, between several undimensioned facts
    tagged to the same element and date, nothing is chosen at all: T-Mobile's
    latest 10-Q carries three different undimensioned values for RPO, and any
    one of them picked out of that set would be a coin toss presented as data.

    Sanity bounds flag, they never correct. An NRR of 11500% means a percent
    sign was misread, and the honest response is to say so and withhold the
    number, not to divide by 100 and hope.

**Point in time.** Nothing here reaches past ``as_of``, and the client's own
knowledge date already filters the facts underneath it, so a KPI set built for a
historical date carries only what was readable then.

**Determinism.** Nothing in this module is stochastic. There is no sampling, no
fitting and no seed to set: the same filing and the same text produce the same
KPI set, byte for byte, on every run.

Units follow the house convention with one deliberate exception. Money is USD
millions and rates are decimals, but counts are actual counts. A filing that
reports 3,390 customers should print 3,390, not 0.003390, and a subscriber base
is a population rather than a balance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd

from .. import tags
from ..edgar import CompanyFacts, Fact, trailing_twelve_months
from ..errors import ConfigError, DataSourceError, MissingDataError

_MM = 1_000_000.0

# --------------------------------------------------------------------------- #
# Extraction settings
# --------------------------------------------------------------------------- #

# These are extractor calibration rather than valuation assumptions: no number
# below moves a valuation, each one decides whether a figure is believed at all.
# They are read off ``assumptions.tmt.kpi_extraction`` by name where that knob
# grows into a settings object, and otherwise come from here, with the choice
# recorded in the KPI set's notes so a printed table says which was used. The
# knob is a bool today, and False switches the whole step off.

DEFAULT_NRR_MIN = 0.5
DEFAULT_NRR_MAX = 2.0
# A retention rate outside a half and double is not a retention rate. The low
# end is charitable: a company losing half its revenue base in a year is in
# liquidation, not in a comp set. The high end catches the percent-sign error,
# which is the one that actually happens.

DEFAULT_CHURN_MIN = 0.0
DEFAULT_CHURN_MAX = 1.0
# Churn is a share of a base. Above one is arithmetically impossible and below
# zero is a sign error.

DEFAULT_ARR_TO_REVENUE_MIN = 0.1
DEFAULT_ARR_TO_REVENUE_MAX = 10.0
# ARR is a run rate and revenue is a trailing twelve months, so they differ by
# roughly the growth rate and never by an order of magnitude. A factor of ten in
# either direction means a scale word was misread, not that the business grew.

DEFAULT_ARPU_MIN = 0.0
# ARPU is a price. Zero or negative is not a price.

DEFAULT_CONFLICT_TOLERANCE = 0.01
# Relative gap below which two readings of the same metric are the same reading.
# One percent absorbs rounding between a filer's own tagged figure and the same
# figure written out in prose to three significant digits.

DEFAULT_TEXT_SCALE_FLOOR = 1_000_000.0
# A dollar figure written without a scale word is believed only above this. A
# filing that says "ARR of $1,250" means 1,250 million about as often as it
# means 1,250 dollars, and the two differ by six orders of magnitude.

CONFIDENCE_XBRL = 1.0
CONFIDENCE_DERIVED = 0.8
CONFIDENCE_TEXT = 0.75
CONFIDENCE_HEDGED = 0.4
CONFIDENCE_REFUSED = 0.0
# The rungs, most trustworthy first. A tagged fact is exact and machine readable,
# so it is 1.0. A derivation off two tagged facts inherits their exactness minus
# the modelling step between them. An unambiguous phrase with a unit is high but
# is still a regular expression reading English. A hedged or period-ambiguous
# phrase is low and always flagged. Zero means the extractor refused, and a
# zero-confidence figure never reaches ``as_dict``.


@dataclass(frozen=True)
class KPISettings:
    """Thresholds that decide whether an extracted figure is believed.

    Built by :func:`settings_from_assumptions`, which reads each field by name
    off ``assumptions.tmt.kpi_extraction`` and falls back to the documented
    default above. Reading by name rather than by type means this keeps working
    when that knob grows from a bool into a settings model, without this module
    having to be changed in step with the config schema.
    """

    nrr_min: float = DEFAULT_NRR_MIN
    nrr_max: float = DEFAULT_NRR_MAX
    churn_min: float = DEFAULT_CHURN_MIN
    churn_max: float = DEFAULT_CHURN_MAX
    arr_to_revenue_min: float = DEFAULT_ARR_TO_REVENUE_MIN
    arr_to_revenue_max: float = DEFAULT_ARR_TO_REVENUE_MAX
    arpu_min: float = DEFAULT_ARPU_MIN
    conflict_tolerance: float = DEFAULT_CONFLICT_TOLERANCE
    text_scale_floor: float = DEFAULT_TEXT_SCALE_FLOOR
    confidence_xbrl: float = CONFIDENCE_XBRL
    confidence_derived: float = CONFIDENCE_DERIVED
    confidence_text: float = CONFIDENCE_TEXT
    confidence_hedged: float = CONFIDENCE_HEDGED
    source: str = "module defaults"

    def __post_init__(self) -> None:
        if not 0.0 < self.nrr_min < self.nrr_max:
            raise ConfigError(
                f"kpi_extraction retention bounds must satisfy 0 < min < max; got "
                f"{self.nrr_min} and {self.nrr_max}. An inverted band would reject "
                "every retention rate while appearing to check them."
            )
        if not 0.0 <= self.churn_min < self.churn_max:
            raise ConfigError(
                f"kpi_extraction churn bounds must satisfy 0 <= min < max; got "
                f"{self.churn_min} and {self.churn_max}."
            )
        if not 0.0 < self.arr_to_revenue_min < self.arr_to_revenue_max:
            raise ConfigError(
                "kpi_extraction ARR-to-revenue bounds must satisfy 0 < min < max; got "
                f"{self.arr_to_revenue_min} and {self.arr_to_revenue_max}."
            )
        if not 0.0 < self.conflict_tolerance < 1.0:
            raise ConfigError(
                f"kpi_extraction conflict_tolerance must sit in (0, 1); got "
                f"{self.conflict_tolerance}. At zero every rounded restatement of the "
                "same figure reads as a conflict; at one nothing ever does."
            )


def _extraction_knob(assumptions: Any) -> Any:
    return getattr(getattr(assumptions, "tmt", None), "kpi_extraction", True)


def extraction_enabled(assumptions: Any) -> bool:
    """Whether ``assumptions.tmt.kpi_extraction`` switches this step on.

    False is off. Anything else, including the settings object the knob may
    become, is on, so growing the schema never silently disables extraction.
    """
    return _extraction_knob(assumptions) is not False


def settings_from_assumptions(assumptions: Any) -> KPISettings:
    """Extraction thresholds, from assumptions where they are configured."""
    knob = _extraction_knob(assumptions)
    tunable = [f for f in dc_fields(KPISettings) if f.name != "source"]
    values = {f.name: getattr(knob, f.name, f.default) for f in tunable}
    configured = sorted(f.name for f in tunable if hasattr(knob, f.name))
    source = (
        "assumptions.tmt.kpi_extraction: " + ", ".join(configured)
        if configured
        else "module defaults"
    )
    return KPISettings(**values, source=source)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KPI:
    """One operating metric, with everything needed to argue about it.

    ``value`` is USD millions for money, an actual count for populations, a
    decimal for rates, and dollars per subscriber per period for ARPU. ``unit``
    names which, because a figure whose unit is inferred from its name is a
    figure waiting to be misread.

    ``source`` is one of ``xbrl_extension`` for anything read off a tagged fact,
    ``text`` for anything parsed out of prose, and ``derived`` for anything this
    module computed from other figures. The first covers both a filer's own
    extension element and the handful of standard concepts that exist only
    because the sector needed them, and ``notes`` says which of the two it was.

    ``tag_or_phrase`` is the exact element name or the exact matched sentence
    fragment, so the figure can be found in the filing by searching for it.

    ``confidence`` is a rung, not a probability. 0.0 means the extractor refused:
    the figure is reported so a reader can see what the filing said, and
    withheld from :meth:`KPISet.as_dict` so no model is trained on it.
    """

    name: str
    value: float
    unit: str
    period_end: date
    source: str
    tag_or_phrase: str
    confidence: float
    notes: str = ""

    @property
    def usable(self) -> bool:
        """Whether this figure may be fed to anything downstream."""
        return self.confidence > 0.0

    def row(self) -> dict[str, Any]:
        """One flat record. Values stay raw; rendering and NM live in cli.py."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "period_end": str(self.period_end),
            "source": self.source,
            "tag_or_phrase": self.tag_or_phrase,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class KPISet:
    """Every operating metric found for one company, and every one that was not.

    ``missing`` maps a metric name to the reason it is absent, because absence is
    itself a finding. A software company with no disclosed customer count and a
    carrier with no disclosed churn are different from a run that failed, and a
    set that simply omitted them would be indistinguishable from a broken
    extractor.

    ``kpis`` is keyed by metric name. Where an XBRL reading and a text reading
    disagree the text one is kept under ``<name>__text`` beside the tagged one,
    so both appear in the table and neither is discarded.
    """

    ticker: str
    as_of: date
    kpis: dict[str, KPI] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def get(self, name: str) -> KPI | None:
        return self.kpis.get(name)

    def rows(self) -> list[dict[str, Any]]:
        return [self.kpis[k].row() for k in sorted(self.kpis)]

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.rows())
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "name",
                    "value",
                    "unit",
                    "period_end",
                    "source",
                    "tag_or_phrase",
                    "confidence",
                    "notes",
                ]
            )
        return frame

    def as_dict(self) -> dict[str, float | None]:
        """The model-facing view: a number, or None where there is not one.

        A metric that was not found is None. So is one the extractor refused,
        because a figure nobody is willing to stand behind must not reach a
        model just because it happens to be printable. The reason in either case
        is in ``missing`` or in the KPI's own notes.
        """
        out: dict[str, float | None] = {name: None for name in self.missing}
        for name, kpi in self.kpis.items():
            out[name] = kpi.value if kpi.usable else None
        return out


# --------------------------------------------------------------------------- #
# Metric packs
# --------------------------------------------------------------------------- #

SOFTWARE_METRICS = (
    "arr",
    "net_revenue_retention",
    "gross_revenue_retention",
    "rpo",
    "rpo_current",
    "billings",
    "customers",
    "customers_over_100k",
)

MEDIA_METRICS = (
    "subscribers",
    "paid_subscribers",
    "arpu",
    "churn",
    "content_spend",
    "content_amortisation",
)

TELECOM_METRICS = (
    "subscribers",
    "postpaid_net_adds",
    "arpu",
    "churn",
)

# Which pack a sub-vertical is priced on. Semiconductors, hardware and IT
# services are deliberately absent: none of these metrics is how those companies
# are quoted, and searching a chip maker's filing for net revenue retention
# would produce noise and call it coverage. Internet and gaming take the media
# pack because they are subscriber and engagement businesses; payments takes the
# software pack because the processors report on volume, customers and retention
# the way a software company does.
METRIC_PACKS: dict[str, tuple[str, ...]] = {
    "infrastructure_software": SOFTWARE_METRICS,
    "application_software": SOFTWARE_METRICS,
    "payments": SOFTWARE_METRICS,
    "internet": MEDIA_METRICS,
    "gaming": MEDIA_METRICS,
    "media_entertainment": MEDIA_METRICS,
    "telecom": TELECOM_METRICS,
    "towers_fiber": TELECOM_METRICS,
    "semiconductors": (),
    "hardware": (),
    "it_services": (),
}

ALL_METRICS = tuple(
    dict.fromkeys(SOFTWARE_METRICS + MEDIA_METRICS + TELECOM_METRICS)
)


def target_metrics(assumptions: Any) -> tuple[str, ...]:
    """Which metrics to look for, from ``assumptions.tmt.sub_vertical``.

    Unset, every pack is searched. That is noisier, because a software company
    then reports no subscribers and no churn as missing, but the alternative is
    to classify the company here and a classification made inside a metric
    extractor is a classification nobody will ever see or argue with.
    """
    raw = getattr(getattr(assumptions, "tmt", None), "sub_vertical", None)
    if raw is None:
        return ALL_METRICS
    key = getattr(raw, "value", raw)
    key = str(key).strip().lower()
    if key not in METRIC_PACKS:
        raise ConfigError(
            f"tmt.sub_vertical is {key!r}, which has no operating metric pack. "
            f"Known: {', '.join(sorted(METRIC_PACKS))}."
        )
    return METRIC_PACKS[key]


# --------------------------------------------------------------------------- #
# XBRL concepts
# --------------------------------------------------------------------------- #

# Element names that belong to the standard taxonomy rather than to a filer's
# own extension. The instance parser reduces every element to its local name, so
# the namespace is no longer on the fact and this is how the note tells a reader
# whether the number came from a concept the SEC defines or one the company
# invented. Everything matched that is not in here is treated as an extension.
STANDARD_CONCEPTS = frozenset(
    {
        "RevenueRemainingPerformanceObligation",
        "RevenueRemainingPerformanceObligationPercentage",
        "RevenueRemainingPerformanceObligationPercentageNextTwelveMonths",
    }
)


@dataclass(frozen=True)
class _Concept:
    """What to look for, in what units, and which element names count.

    ``patterns`` is an ordered ladder of full-match regular expressions against
    the normalised element name, most specific first, exactly like the us-gaap
    ladders in ``tags.py``. The first pattern that matches anything wins and the
    rest are not consulted. That ordering is what keeps a filer who tags both a
    content amortisation total and its licensed and produced components from
    reading as three conflicting totals.
    """

    name: str
    patterns: tuple[str, ...]
    kind: str  # usd_mm | count | ratio | usd_per_period
    instant: bool | None  # None where a filer may tag either


_CONCEPTS: tuple[_Concept, ...] = (
    _Concept(
        name="arr",
        patterns=(
            r"(total)?annual(ized|ised)?recurringrevenue",
            r"arr",
        ),
        kind="usd_mm",
        instant=True,
    ),
    _Concept(
        name="rpo",
        patterns=(
            r"revenueremainingperformanceobligations?",
            r"remainingperformanceobligations?",
        ),
        kind="usd_mm",
        instant=True,
    ),
    _Concept(
        name="rpo_current",
        patterns=(
            r"(revenue)?remainingperformanceobligations?"
            r"(expectedtobe)?(recognized|recognised|satisfied)?"
            r"(with)?in(the)?nexttwelvemonths",
            r"(revenue)?remainingperformanceobligations?current",
        ),
        kind="usd_mm",
        instant=True,
    ),
    _Concept(
        name="net_revenue_retention",
        patterns=(
            r"(dollarbased)?net(revenue|dollar)(retention|expansion)rate",
            r"(dollarbased)?net(retention|expansion)rate",
            r"net(revenue|dollar)retention",
        ),
        kind="ratio",
        instant=None,
    ),
    _Concept(
        name="gross_revenue_retention",
        patterns=(
            r"gross(revenue|dollar)retentionrate",
            r"grossretentionrate",
        ),
        kind="ratio",
        instant=None,
    ),
    _Concept(
        name="customers",
        patterns=(
            r"(total)?numberofcustomers",
            r"customercount",
        ),
        kind="count",
        instant=True,
    ),
    _Concept(
        name="customers_over_100k",
        patterns=(
            r"numberofcustomers(with)?.*100(000|k).*",
            r"numberoflargecustomers",
        ),
        kind="count",
        instant=True,
    ),
    _Concept(
        name="paid_subscribers",
        patterns=(
            r"(total)?numberofpaid(subscribers|members|memberships)",
            r"paid(subscribers|members|memberships)",
        ),
        kind="count",
        instant=True,
    ),
    _Concept(
        name="subscribers",
        patterns=(
            r"(total)?numberofsubscribers",
            r"subscribers",
        ),
        kind="count",
        instant=True,
    ),
    _Concept(
        name="arpu",
        patterns=(
            r"averagerevenueper(user|subscriber|account|membership|member|customer)",
            r"arpu",
        ),
        kind="usd_per_period",
        instant=None,
    ),
    _Concept(
        name="churn",
        patterns=(
            r"(monthly|annual|quarterly)?churnrate",
            r"churn",
        ),
        kind="ratio",
        instant=None,
    ),
    _Concept(
        name="postpaid_net_adds",
        patterns=(
            r"(branded)?postpaid(phone)?net(customer)?additions",
            r"postpaidnetadds",
        ),
        kind="count",
        instant=False,
    ),
    _Concept(
        name="content_spend",
        patterns=(
            r"additionsto(streaming)?contentassets",
            r"paymentstoacquire(streaming)?contentassets",
        ),
        kind="usd_mm",
        instant=False,
    ),
    _Concept(
        name="content_amortisation",
        patterns=(
            r"(costofservices)?amortization(expense)?of(streaming)?contentassets",
            r"amortizationofcontentassets",
            r"amortization(expense)?of(licensed|produced)contentassets",
        ),
        kind="usd_mm",
        instant=False,
    ),
)

# The percentage of RPO a filer expects to recognise inside a year. Not a metric
# on its own, and only ever used to split RPO into its current part.
_RPO_PERCENTAGE_PATTERNS = (
    r"(revenue)?remainingperformanceobligationpercentagenexttwelvemonths",
    r"(revenue)?remainingperformanceobligationpercentage",
)


def _norm(tag: str) -> str:
    """Element name reduced to lower-case alphanumerics.

    Filers write the same concept as ``AnnualRecurringRevenue``,
    ``Annual_Recurring_Revenue`` and ``ARRAnnualRecurringRevenue``, and casing
    is inconsistent inside a single document: Netflix tags
    ``CostofServicesAmortizationofStreamingContentAssets`` with two lower-case
    prepositions in the middle of a camel-case name.
    """
    return re.sub(r"[^a-z0-9]", "", tag.lower())


# --------------------------------------------------------------------------- #
# XBRL extraction
# --------------------------------------------------------------------------- #


def _instance_facts(client: Any, ticker: str) -> tuple[list[Any], str | None, date | None, str | None]:
    """Dimensioned facts from the latest periodic filing, or the reason there are none.

    The client is duck typed on purpose. Anything exposing ``instance_facts``
    works, which covers the real ``EdgarClient`` and the offline fixture clients
    the tests use, and a client without it is a stated limitation rather than a
    crash.
    """
    getter = getattr(client, "instance_facts", None)
    if getter is None:
        return [], None, None, (
            "this client exposes no instance_facts, so only the standard "
            "taxonomy in company facts could be searched"
        )
    try:
        facts, accession, filed = getter(ticker, forms=("10-K", "10-Q"))
    except (MissingDataError, DataSourceError) as exc:
        return [], None, None, f"the filing's instance document could not be read: {exc}"
    return list(facts), accession, filed, None


def _relative_gap(a: float, b: float) -> float:
    """Relative difference, scaled by the larger magnitude."""
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def _distinct(values: Iterable[float], tolerance: float) -> list[float]:
    """Values that genuinely differ, in the order first seen.

    A filing repeats the same fact in several contexts. For the six months to
    30 June 2026 Netflix tags content amortisation five times: 8,529.209 bare
    twice over, 8,529.209 again under the reportable-segment axis, and the
    licensed and produced components that sum to it. Only the two bare repeats
    reach here, since the dimensioned facts are parts rather than totals and are
    dropped upstream, and two identical readings are one reading. A conflict is
    a real difference in value, not a filer tagging carefully.
    """
    out: list[float] = []
    for v in values:
        if not any(_relative_gap(v, seen) <= tolerance for seen in out):
            out.append(v)
    return out


def _convert(value: float, unit: str | None, kind: str) -> tuple[float, str, str | None]:
    """Scale a tagged value into house units, or refuse and say why.

    The unit is checked rather than assumed. A count tagged in USD is not a
    count, and taking it as one would put a dollar balance into a subscriber
    field where nothing downstream would ever question it.
    """
    u = (unit or "").strip()
    if kind == "usd_mm":
        if u != "USD":
            return value, "usd_mm", f"tagged in {u or 'no unit'} rather than USD"
        return value / _MM, "usd_mm", None
    if kind == "usd_per_period":
        if u != "USD":
            return value, "usd_per_period", f"tagged in {u or 'no unit'} rather than USD"
        return value, "usd_per_period", None
    if kind == "ratio":
        if u != "pure":
            return value, "ratio", f"tagged in {u or 'no unit'} rather than pure"
        return value, "ratio", None
    if u in ("USD", "USD/shares"):
        return value, "count", f"a count tagged in {u} is not a count"
    return value, "count", None


def _select(
    candidates: list[Any], concept: _Concept, as_of: date, filed: date | None
) -> tuple[list[Fact], list[str]]:
    """Reduce a concept's matching facts to the ones that answer the question.

    Only undimensioned facts are considered. A disaggregation along a product,
    segment or timing axis is a part, and summing parts that may or may not be
    exhaustive is how a total gets invented. Where a filer publishes only parts,
    the concept reads as unreadable rather than as absent, because the filer did
    tag it and the engine should say that it could not use what was tagged.
    """
    reasons: list[str] = []
    kept = [f for f in candidates if not f.dimensions and f.end <= as_of]
    if not kept:
        if candidates:
            reasons.append(
                "every matching fact is reported along an axis, so only parts of "
                "the total are tagged and no total can be read from them"
            )
        return [], reasons

    if concept.instant is True:
        kept = [f for f in kept if f.start is None]
    elif concept.instant is False:
        kept = [f for f in kept if f.start is not None]
    if not kept:
        reasons.append(
            "the matching facts are tagged over the wrong kind of period for this "
            "concept, as an instant where a duration was needed or the reverse"
        )
        return [], reasons

    stamp = filed or as_of
    facts = [
        Fact(
            tag=f.tag,
            val=f.value,
            end=f.end,
            start=f.start,
            form="instance document",
            filed=stamp,
            unit=f.unit or "",
        )
        for f in kept
    ]
    return facts, reasons


def _resolve_instant(facts: list[Fact], tolerance: float) -> tuple[Fact, list[float]]:
    """The latest instant, and every distinct value tagged at it."""
    latest = max(f.end for f in facts)
    at_date = [f for f in facts if f.end == latest]
    values = _distinct([f.val for f in at_date], tolerance)
    return at_date[0], values


def _resolve_duration(
    facts: list[Fact], tolerance: float
) -> tuple[float | None, Fact | None, str, tuple[date, list[float]] | None]:
    """Twelve months where the filing supports it, the longest filed period otherwise.

    A quarterly content spend and an annual one are not comparable, and
    annualising a quarter by multiplying by four would manufacture a number the
    filing does not support. So the window is either a genuine twelve months
    tiled out of filed periods, using the same period algebra the statements use,
    or it is the longest period the filer actually reported and the note says so.

    The fourth element of the return is the disagreement, where there is one:
    the period end it sits at and every distinct value tagged over that period.
    A concept tagged twice over the same window with two different values cannot
    be tiled into anything, because the tiling would have to pick one first.
    """
    by_period: dict[tuple[date | None, date], list[float]] = {}
    for f in facts:
        by_period.setdefault((f.start, f.end), []).append(f.val)
    for period, values in by_period.items():
        distinct = _distinct(values, tolerance)
        if len(distinct) > 1:
            return None, None, "", (period[1], distinct)

    unique = [
        next(f for f in facts if (f.start, f.end) == period) for period in by_period
    ]
    latest = max(f.end for f in unique)
    ttm = trailing_twelve_months(unique, latest)
    if ttm is not None:
        value, tiles, method = ttm
        return value, tiles[-1], f"twelve months ended {latest}, {method}", None

    at_latest = [f for f in unique if f.end == latest]
    longest = max(at_latest, key=lambda f: f.days)
    return (
        longest.val,
        longest,
        f"the {longest.days} days from {longest.start} to {longest.end}, which is the "
        "longest period this filing reports for the concept; it is not annualised",
        None,
    )


def extract_from_xbrl(
    facts: CompanyFacts,
    client: Any,
    ticker: str,
    as_of: date,
    *,
    settings: KPISettings | None = None,
    instance: tuple[list[Any], str | None, date | None, str | None] | None = None,
) -> dict[str, KPI]:
    """Operating metrics from tagged facts, extension namespaces included.

    ``companyfacts`` carries only the standard taxonomies, so the filer's own
    elements are read out of the latest 10-K or 10-Q instance document instead
    and matched on local names. ``facts`` is still needed: it is what the
    standard concepts fall back to when the instance document cannot be fetched,
    and it is what the billings derivation runs on.

    Returns every concept that resolved, including ``rpo_current`` where the
    filer tagged both the obligation and the share of it expected inside a year.
    Concepts that did not resolve are simply absent; :func:`build_kpis` turns
    that absence into a stated reason.

    ``instance`` is the already fetched output of the instance-document read,
    passed by :func:`build_kpis` so one filing is not downloaded twice in a run.
    """
    settings = settings or KPISettings()
    rows, accession, filed, _unreadable = instance or _instance_facts(client, ticker)
    out: dict[str, KPI] = {}

    by_norm: dict[str, list[Any]] = {}
    for fact in rows:
        by_norm.setdefault(_norm(fact.tag), []).append(fact)

    for concept in _CONCEPTS:
        kpi = _concept_from_instance(
            concept, by_norm, as_of, filed, accession, settings
        )
        if kpi is not None:
            out[concept.name] = kpi

    if "rpo" not in out:
        fallback = _rpo_from_company_facts(facts, as_of, settings)
        if fallback is not None:
            out["rpo"] = fallback

    if "rpo_current" not in out and "rpo" in out and out["rpo"].usable:
        current = _rpo_current_from_percentage(
            out["rpo"], by_norm, as_of, accession, settings
        )
        if current is not None:
            out["rpo_current"] = current

    return out


def _concept_from_instance(
    concept: _Concept,
    by_norm: dict[str, list[Any]],
    as_of: date,
    filed: date | None,
    accession: str | None,
    settings: KPISettings,
) -> KPI | None:
    """Walk one concept's pattern ladder and build the KPI the winner supports."""
    for pattern in concept.patterns:
        rx = re.compile(pattern + r"\Z")
        matched = [
            fact for norm, facts in by_norm.items() if rx.match(norm) for fact in facts
        ]
        if not matched:
            continue

        candidates, reasons = _select(matched, concept, as_of, filed)
        if not candidates:
            if reasons:
                # The elements are there and the facts are not usable. Report the
                # concept as refused rather than absent: the two are different
                # and only one of them is the filer's silence.
                return KPI(
                    name=concept.name,
                    value=0.0,
                    unit=concept.kind,
                    period_end=as_of,
                    source="xbrl_extension",
                    tag_or_phrase=sorted({f.tag for f in matched})[0],
                    confidence=CONFIDENCE_REFUSED,
                    notes="; ".join(reasons),
                )
            continue

        tag_name = sorted({f.tag for f in candidates})[0]
        standard = tag_name in STANDARD_CONCEPTS
        origin = (
            f"{tag_name} is a standard us-gaap concept"
            if standard
            else f"{tag_name} is a filer extension element, not a us-gaap concept"
        )
        if accession:
            origin += f", tagged in {accession}"

        # An instant is preferred wherever the filer tagged one, because a
        # balance at a date needs no window to be interpreted. Only a concept
        # that is a flow, or one the filer tagged solely as a duration, goes
        # through the period algebra.
        instants = [f for f in candidates if f.start is None]
        if instants and concept.instant is not False:
            anchor, values = _resolve_instant(instants, settings.conflict_tolerance)
            if len(values) > 1:
                return _conflicted(
                    concept, tag_name, anchor.end, values, anchor.unit, origin
                )
            scaled, unit, refusal = _convert(values[0], anchor.unit, concept.kind)
            return KPI(
                name=concept.name,
                value=scaled,
                unit=unit,
                period_end=anchor.end,
                source="xbrl_extension",
                tag_or_phrase=tag_name,
                confidence=CONFIDENCE_REFUSED if refusal else settings.confidence_xbrl,
                notes=f"{origin}; {refusal}" if refusal else origin,
            )

        value, anchor, window, conflicting = _resolve_duration(
            candidates, settings.conflict_tolerance
        )
        if conflicting is not None:
            clash_end, clash_values = conflicting
            return _conflicted(
                concept, tag_name, clash_end, clash_values, candidates[0].unit, origin
            )
        assert value is not None and anchor is not None
        scaled, unit, refusal = _convert(value, anchor.unit, concept.kind)
        notes = f"{origin}; {window}"
        return KPI(
            name=concept.name,
            value=scaled,
            unit=unit,
            period_end=anchor.end,
            source="xbrl_extension",
            tag_or_phrase=tag_name,
            confidence=CONFIDENCE_REFUSED if refusal else settings.confidence_xbrl,
            notes=f"{notes}; {refusal}" if refusal else notes,
        )
    return None


def _conflicted(
    concept: _Concept,
    tag_name: str,
    period_end: date,
    values: list[float],
    unit: str,
    origin: str,
) -> KPI:
    """One element, several undimensioned values, no way to choose between them.

    T-Mobile's 10-Q tags ``RevenueRemainingPerformanceObligation`` three times at
    the same date with no axis on any of them, at 694mm, 1,000mm and 2,300mm.
    Those are pieces of a disclosure whose structure lives in the narrative and
    not in the XBRL, and picking the largest, the first or the sum would each be
    a guess dressed as a number. Every value is reported and none is believed.
    """
    scaled = [_convert(v, unit, concept.kind)[0] for v in values]
    house_unit = _convert(values[0], unit, concept.kind)[1]
    listed = ", ".join(f"{v:,.4g}" for v in scaled)
    return KPI(
        name=concept.name,
        value=scaled[0],
        unit=house_unit,
        period_end=period_end,
        source="xbrl_extension",
        tag_or_phrase=tag_name,
        confidence=CONFIDENCE_REFUSED,
        notes=(
            f"{origin}; the filing carries {len(values)} different undimensioned "
            f"values at {period_end} ({listed}), so which one is the total cannot "
            "be read from the document"
        ),
    )


def _rpo_from_company_facts(
    facts: CompanyFacts, as_of: date, settings: KPISettings
) -> KPI | None:
    """Remaining performance obligation off the standard taxonomy.

    RPO is the one metric here that is a us-gaap concept, so it survives in
    ``companyfacts`` when the instance document cannot be reached. Resolved
    through ``resolve_instant`` so the retired-tag defence applies: a filer who
    stopped tagging RPO two years ago reads as absent rather than as flat.
    """
    try:
        value, prov = facts.resolve_instant(
            "remaining performance obligation",
            ["RevenueRemainingPerformanceObligation"],
            as_of,
            tolerance_days=120,
            required=False,
        )
    except MissingDataError:
        return None
    if value is None or prov.tag is None:
        return None
    period = date.fromisoformat(prov.periods[0]) if prov.periods else as_of
    return KPI(
        name="rpo",
        value=value / _MM,
        unit="usd_mm",
        period_end=period,
        source="xbrl_extension",
        tag_or_phrase=prov.tag,
        confidence=settings.confidence_xbrl,
        notes=(
            f"{prov.tag} is a standard us-gaap concept, read from company facts "
            f"rather than from the instance document; {prov.method}"
        ),
    )


def _rpo_current_from_percentage(
    rpo: KPI,
    by_norm: dict[str, list[Any]],
    as_of: date,
    accession: str | None,
    settings: KPISettings,
) -> KPI | None:
    """Split RPO into the part due inside a year, where the filer tagged the share.

    Filers tag the obligation and, separately, the percentage of it they expect
    to recognise as revenue over the following twelve months. The Cloudflare 10-Q
    committed with this package carries 2,732mm and 0.64. The product is the
    current portion, and it is ``derived`` rather than disclosed because the
    filing states the two inputs and not the answer.

    The assumption inside it is stated rather than buried: where the element name
    does not itself carry the window, the note says that twelve months is what is
    being read into it. Filers also disclose the share expected over two or three
    horizons, and where more than one undimensioned percentage is tagged at the
    same date nothing is derived at all, because the document does not say which
    of them is the next twelve months and the largest is not the answer by virtue
    of being the largest.
    """
    for pattern in _RPO_PERCENTAGE_PATTERNS:
        rx = re.compile(pattern + r"\Z")
        matched = [
            fact
            for norm, facts in by_norm.items()
            if rx.match(norm)
            for fact in facts
            if not fact.dimensions and fact.start is None and fact.end <= as_of
        ]
        if not matched:
            continue
        latest = max(f.end for f in matched)
        at_date = [f for f in matched if f.end == latest]
        values = _distinct([f.value for f in at_date], settings.conflict_tolerance)
        tag_name = sorted({f.tag for f in at_date})[0]
        if len(values) > 1:
            listed = ", ".join(f"{v:.4g}" for v in values)
            return KPI(
                name="rpo_current",
                value=0.0,
                unit="usd_mm",
                period_end=latest,
                source="derived",
                tag_or_phrase=f"{rpo.tag_or_phrase} x {tag_name}",
                confidence=CONFIDENCE_REFUSED,
                notes=(
                    f"{tag_name} carries {len(values)} undimensioned values at "
                    f"{latest} ({listed}), which are different horizons, and "
                    "nothing in the document says which one is the next twelve "
                    "months"
                ),
            )
        share = values[0]
        if not 0.0 < share <= 1.0:
            return KPI(
                name="rpo_current",
                value=share,
                unit="ratio",
                period_end=latest,
                source="derived",
                tag_or_phrase=tag_name,
                confidence=CONFIDENCE_REFUSED,
                notes=(
                    f"{tag_name} is tagged at {share:,.4g}, which is not a share of "
                    "one; multiplying an obligation by it would produce a current "
                    "portion larger than the obligation"
                ),
            )
        explicit = "nexttwelvemonths" in _norm(tag_name)
        note = (
            f"{rpo.value:,.1f} of remaining performance obligation multiplied by the "
            f"{share:.0%} the filer tagged under {tag_name}"
        )
        note += (
            "; the element names the twelve month window"
            if explicit
            else "; the element does not name its window and twelve months is read "
            "into it, which is the standard ASC 606 disclosure but is an assumption"
        )
        if accession:
            note += f"; both facts from {accession}"
        return KPI(
            name="rpo_current",
            value=rpo.value * share,
            unit="usd_mm",
            period_end=latest,
            source="derived",
            tag_or_phrase=f"{rpo.tag_or_phrase} x {tag_name}",
            confidence=settings.confidence_derived,
            notes=note,
        )
    return None


# --------------------------------------------------------------------------- #
# Billings
# --------------------------------------------------------------------------- #


def derive_billings(
    facts: CompanyFacts,
    as_of: date,
    *,
    settings: KPISettings | None = None,
) -> KPI | None:
    """Invoiced value over the trailing twelve months, which nobody reports.

    The identity every software analyst is expected to know:

        billings = revenue + (deferred revenue now less deferred revenue a year ago)

    Revenue is what was delivered in the window. Deferred revenue is what was
    invoiced and not yet delivered, so its increase over the same window is the
    part of this period's invoicing that has not reached the income statement.
    Add them and the result approximates what was billed.

    Worked, on the Datadog facts committed with this package, twelve months ended
    30 June 2026, USD millions:

        revenue                            3,966.725
        deferred revenue at 2026-06-30     1,286.690 current + 50.860 noncurrent
                                         = 1,337.550
        deferred revenue at 2025-06-30       966.442 current + 29.866 noncurrent
                                         =   996.308
        billings = 3,966.725 + 1,337.550 - 996.308 = 4,307.967

    Total deferred revenue is used, current plus noncurrent, because a multi-year
    prepayment is invoiced value whichever side of twelve months it is delivered
    on. The current-only convention is the other common one and runs lower by the
    change in the noncurrent balance; for Datadog over this window that is 21.0mm,
    or half a percent.

    What it is not: invoicing. The approximation fails where contract terms shift,
    and the direction is predictable. A move from annual to multi-year prepayment
    raises deferred revenue with no change in demand and billings rises with it. A
    move to monthly billing does the reverse. An acquisition brings acquired
    deferred revenue onto the balance sheet as a step that was never invoiced by
    this company at all, and purchase accounting then writes it down, so both the
    step and the write-down land in billings as though they were trading. None of
    that is detectable from the two balances, which is why this is marked derived
    and carries the caveat into its own notes.

    Returns None where the pieces are not all present, since a billings figure
    built on a guessed opening balance would be worse than no billings figure.
    """
    settings = settings or KPISettings()

    ends = [
        f.end
        for tag in tags.REVENUE
        for f in facts.facts(tag)
        if not f.is_instant and f.end <= as_of
    ]
    if not ends:
        return None
    statement_end = max(ends)

    try:
        revenue, prov = facts.resolve_ttm("revenue", tags.REVENUE, statement_end)
    except MissingDataError:
        return None
    if revenue is None:
        return None

    now = _deferred_revenue(facts, statement_end)
    if now is None:
        return None
    prior_end = _prior_year_end(facts, statement_end)
    if prior_end is None:
        return None
    then = _deferred_revenue(facts, prior_end)
    if then is None:
        return None

    value = (revenue + now - then) / _MM
    return KPI(
        name="billings",
        value=value,
        unit="usd_mm",
        period_end=statement_end,
        source="derived",
        tag_or_phrase=(
            "revenue + change in ContractWithCustomerLiabilityCurrent "
            "+ ContractWithCustomerLiabilityNoncurrent"
        ),
        confidence=settings.confidence_derived,
        notes=(
            f"revenue of {revenue / _MM:,.1f} over the twelve months ended "
            f"{statement_end} ({prov.method}) plus deferred revenue of "
            f"{now / _MM:,.1f} at {statement_end} less {then / _MM:,.1f} at "
            f"{prior_end}; approximates invoiced value and diverges where contract "
            "terms shift between annual, multi-year and monthly billing or where "
            "an acquisition brings deferred revenue on"
        ),
    )


def _deferred_revenue(facts: CompanyFacts, when: date) -> float | None:
    """Total deferred revenue at a date, current plus noncurrent.

    The noncurrent piece is optional because a filer with no multi-year
    prepayments legitimately has none, and reading its absence as a missing input
    would drop billings for exactly the companies whose billings is cleanest.
    """
    try:
        current, _prov = facts.resolve_instant(
            "deferred revenue, current",
            tags.DEFERRED_REVENUE_CURRENT,
            when,
            required=False,
        )
        noncurrent, _p2 = facts.resolve_instant(
            "deferred revenue, noncurrent",
            tags.DEFERRED_REVENUE_NONCURRENT,
            when,
            required=False,
            default_when_absent=0.0,
        )
    except MissingDataError:
        return None
    if current is None:
        return None
    return current + (noncurrent or 0.0)


def _prior_year_end(facts: CompanyFacts, statement_end: date) -> date | None:
    """The balance-sheet date twelve months before, as the filer reported it.

    A fiscal quarter is thirteen weeks, so the anniversary of a period end is
    rarely the same calendar day and a 53 week year moves it by a whole week. The
    date is therefore looked up among the dates the filer actually reported,
    inside a three week window around the anniversary, rather than computed.
    """
    target = statement_end - timedelta(days=365)
    candidates = [
        f.end
        for tag in tags.DEFERRED_REVENUE_CURRENT
        for f in facts.facts(tag)
        if f.is_instant and abs((f.end - target).days) <= 21
    ]
    return min(candidates, key=lambda d: abs((d - target).days)) if candidates else None


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #

_NUM = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_SCALE = r"(?P<scale>thousand|million|billion|trillion)"
_QUAL = (
    r"(?P<qual>approximately|approx\.?|about|roughly|nearly|almost|around|"
    r"more than|greater than|at least|in excess of|no less than|up to|"
    r"less than|fewer than|under|over|above|below|just over|just under|"
    r"north of|exceeding|exceeded)"
)

_SCALES = {
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
}

# Hedges that are bounds rather than estimates. "Over 120%" is a floor and
# "approximately 120%" is a centre, and while both are hedged only the first is
# guaranteed to be wrong in a known direction, so the note says which.
_BOUNDING_QUALIFIERS = frozenset(
    {
        "more than",
        "greater than",
        "at least",
        "in excess of",
        "no less than",
        "up to",
        "less than",
        "fewer than",
        "under",
        "over",
        "above",
        "below",
        "just over",
        "just under",
        "north of",
        "exceeding",
        "exceeded",
    }
)

_LINK = r"(?:\s*(?:of|was|were|is|are|at|to|reached|totall?ed|stood\s+at|came\s+in\s+at|grew\s+to|ended\s+at|,)\s*)"

# The linking word is optional and so is the whitespace after it, because a
# filer writes "net revenue retention rate of 115%", "net revenue retention rate
# was 115%" and "net revenue retention rate 115%" for the same disclosure, and a
# tail that demanded the word would read the first two and silently miss the
# third. Only whitespace may be skipped, never punctuation or another word, so a
# sentence that names the metric and then gives an unrelated number further on
# still does not match.
_RATE_TAIL = rf"(?:\s*rate)?{_LINK}?\s*(?:\b{_QUAL}\s+)?{_NUM}\s*(?P<pct>%|percent)?"
_MONEY_TAIL = rf"{_LINK}?\s*(?:\b{_QUAL}\s+)?\$\s*{_NUM}\s*(?:{_SCALE})?"
_COUNT_HEAD = rf"(?:\b{_QUAL}\s+)?{_NUM}\s*(?:{_SCALE}\s+)?"


@dataclass(frozen=True)
class _TextRule:
    """One metric's phrasings, in the order they are tried.

    Order across rules matters as much as order inside one. Matched spans are
    blanked out as each rule runs, so "3,390 customers with ARR of $100,000 or
    more" is consumed by the large-customer rule and cannot then be read by the
    ARR rule as an ARR of one hundred thousand dollars. Rules that carry their
    own distinguishing context therefore run before the general ones.
    """

    name: str
    kind: str  # ratio | money | count | arpu
    patterns: tuple[str, ...]


_TEXT_RULES: tuple[_TextRule, ...] = (
    _TextRule(
        name="customers_over_100k",
        kind="count",
        patterns=(
            _COUNT_HEAD
            + r"customers?\s+(?:with|generating|contributing)\s+(?:an?\s+)?"
            r"(?:ARR|annual\s+recurring\s+revenue)\s+(?:of\s+)?\$\s*100,?000\s+"
            r"or\s+(?:more|greater|above)",
            _COUNT_HEAD
            + r"customers?\s+(?:with|generating|contributing)\s+\$\s*100,?000\s+"
            r"or\s+(?:more|greater|above)\s+in\s+(?:ARR|annual\s+recurring\s+revenue)",
        ),
    ),
    _TextRule(
        name="net_revenue_retention",
        kind="ratio",
        patterns=(
            r"(?:dollar[-\s]based\s+)?net\s+(?:revenue|dollar)\s+"
            r"(?:retention|expansion)" + _RATE_TAIL,
            r"(?:dollar[-\s]based\s+)?net\s+(?:retention|expansion)" + _RATE_TAIL,
        ),
    ),
    _TextRule(
        name="gross_revenue_retention",
        kind="ratio",
        patterns=(
            r"gross\s+(?:revenue|dollar)\s+retention" + _RATE_TAIL,
            r"gross\s+retention" + _RATE_TAIL,
        ),
    ),
    _TextRule(
        name="churn",
        kind="ratio",
        patterns=(
            r"(?:monthly|quarterly|annual)\s+churn" + _RATE_TAIL,
            r"churn" + _RATE_TAIL,
        ),
    ),
    _TextRule(
        name="arr",
        kind="money",
        patterns=(
            r"(?:annual|annualized|annualised)\s+recurring\s+revenue"
            r"(?:\s*\(\s*ARR\s*\))?" + _MONEY_TAIL,
            r"\bARR\b" + _MONEY_TAIL,
        ),
    ),
    _TextRule(
        name="rpo",
        kind="money",
        patterns=(
            r"remaining\s+performance\s+obligations?(?:\s*\(\s*RPO\s*\))?" + _MONEY_TAIL,
            r"\bRPO\b" + _MONEY_TAIL,
        ),
    ),
    _TextRule(
        name="arpu",
        kind="arpu",
        patterns=(
            rf"\$\s*{_NUM}\s*(?:{_SCALE}\s+)?per\s+"
            r"(?:subscriber|user|account|membership|member|customer)\s+per\s+"
            r"(?P<per>month|quarter|year)",
            # "Membership" is here because the streamers do not say user. Netflix
            # quotes average revenue per membership and calls it ARM, and a
            # pattern list written only from the software vocabulary would report
            # no ARPU for the company the metric matters most to.
            r"(?:average\s+revenue\s+per\s+"
            r"(?:user|subscriber|account|membership|member|customer)"
            r"(?:\s*\(\s*AR[PM]U?\s*\))?|\bARPU\b)"
            + _MONEY_TAIL
            # The money tail has already eaten the space before "per", so the
            # period suffix must tolerate none. Written as \s+ it never fires and
            # every ARPU silently loses its denominator.
            + r"(?:\s*per\s+(?P<per>month|quarter|year))?",
        ),
    ),
    _TextRule(
        name="postpaid_net_adds",
        kind="count",
        patterns=(
            r"postpaid\s+(?:phone\s+)?net\s+(?:customer\s+)?(?:additions?|adds)"
            + _LINK
            + rf"?(?:\b{_QUAL}\s+)?{_NUM}\s*(?:{_SCALE})?",
            r"added\s+" + _COUNT_HEAD + r"postpaid\s+net\s+(?:customers|additions|adds)",
        ),
    ),
    _TextRule(
        name="paid_subscribers",
        kind="count",
        patterns=(
            _COUNT_HEAD + r"paid\s+(?:subscribers|memberships|members)",
            _COUNT_HEAD + r"paid\s+(?:streaming\s+)?subscriptions",
        ),
    ),
    # Adjectives are a ladder rather than an alternation, because a filing that
    # gives both a postpaid and a prepaid base has given two populations and not
    # one figure with two spellings. Taking the most specific phrasing present
    # and stopping there names which base was read; matching them all at once
    # would read two different bases as a conflict over a single number.
    _TextRule(
        name="subscribers",
        kind="count",
        patterns=(
            _COUNT_HEAD + r"total\s+subscribers",
            _COUNT_HEAD + r"postpaid\s+(?:phone\s+)?(?:subscribers|customers)",
            _COUNT_HEAD + r"(?:wireless|retail|ending)\s+subscribers",
            _COUNT_HEAD + r"subscribers",
            _COUNT_HEAD + r"(?:total\s+)?subscriptions",
        ),
    ),
    _TextRule(
        name="customers",
        kind="count",
        patterns=(_COUNT_HEAD + r"(?:total\s+)?customers",),
    ),
    _TextRule(
        name="content_spend",
        kind="money",
        patterns=(
            r"(?:cash\s+)?(?:spent|spend(?:ing)?|invested)\s+on\s+content" + _MONEY_TAIL,
            r"content\s+(?:cash\s+)?(?:spend(?:ing)?|obligations?)" + _MONEY_TAIL,
        ),
    ),
    _TextRule(
        name="content_amortisation",
        kind="money",
        patterns=(
            r"amorti[sz]ation\s+of\s+(?:streaming\s+)?content\s+assets" + _MONEY_TAIL,
            r"content\s+amorti[sz]ation" + _MONEY_TAIL,
        ),
    ),
)


_TEXT_UNITS = {
    "ratio": "ratio",
    "money": "usd_mm",
    "count": "count",
    "arpu": "usd_per_period",
}


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _resolve_money(
    match: re.Match[str], floor: float
) -> tuple[float | None, str | None]:
    """Dollars from a matched phrase, in millions, or the reason it is not readable.

    Scale is resolved explicitly. A scale word settles it. Without one, a figure
    written with thousands separators and standing above the floor is read as
    dollars as written, because "$1,234,567,000" is unambiguous. Anything else is
    refused: "$1,250" in a filing is 1,250 dollars or 1,250 million depending on
    a table header this parser cannot see, and the two differ by a factor of a
    million.
    """
    raw = _to_float(match.group("num"))
    scale = (match.groupdict().get("scale") or "").lower()
    if scale:
        return raw * _SCALES[scale] / _MM, None
    if raw >= floor:
        return raw / _MM, None
    return None, (
        f"the phrase gives ${match.group('num')} with no scale word, and below "
        f"${floor:,.0f} that is as likely to mean dollars as millions"
    )


def _resolve_count(match: re.Match[str]) -> tuple[float | None, str | None]:
    """A population from a matched phrase, or the reason it is not readable.

    A scale word settles it. Without one, an integer is a literal count, which is
    how customer counts are written: "3,390 customers" is 3,390 customers. A
    fractional number without a scale word is refused, because "2.4 subscribers"
    is 2.4 million subscribers in every filing that has ever been written and is
    still not something to infer.
    """
    text = match.group("num")
    raw = _to_float(text)
    scale = (match.groupdict().get("scale") or "").lower()
    if scale:
        return raw * _SCALES[scale], None
    if "." in text:
        return None, (
            f"the phrase gives {text} with no scale word, and a fractional count "
            "has to mean thousands or millions without saying which"
        )
    return raw, None


def _qualifier_note(qual: str | None) -> tuple[str, bool]:
    """How a hedge should be described, and whether it is a hedge at all."""
    if not qual:
        return "", False
    clean = re.sub(r"\s+", " ", qual.strip().lower()).rstrip(".")
    if clean in _BOUNDING_QUALIFIERS:
        return (
            f"hedged: the filing says {clean!r}, which is a bound and not the "
            "figure, so this is wrong in a known direction",
            True,
        )
    return (
        f"hedged: the filing says {clean!r}, so the figure is the company's own "
        "estimate rather than a measurement",
        True,
    )


def extract_from_text(
    mdna_text: str | None,
    as_of: date,
    *,
    settings: KPISettings | None = None,
) -> dict[str, KPI]:
    """Operating metrics out of prose, read conservatively.

    The phrasings vary more than the metrics do. Net revenue retention appears as
    "net revenue retention rate of 115%", as "dollar-based net retention rate was
    111%" and as "net dollar retention of over 120%", and those are the same
    disclosure written three ways by three companies. Every rule therefore
    captures the number, the qualifier that hedged it and the exact fragment it
    matched, and the fragment travels on the KPI so the sentence can be found in
    the filing.

    Three refusals, each of which would otherwise be a silent error:

        A rate with no percent sign. "Retention of 115" is 115% or 1.15 and the
        difference is a factor of a hundred.

        A dollar figure with no scale word below the floor. See
        :func:`_resolve_money`.

        A fractional count with no scale word. See :func:`_resolve_count`.

    A refusal is reported with confidence 0.0 and the reason attached, not
    dropped, because a filing that stated a metric this parser could not read is
    a different situation from a filing that did not state it.

    Where one text gives a metric two different values, both are reported and
    neither is believed. That happens for real: a 10-K quotes retention in the
    overview and again in the quarterly comparison, and where those differ it is
    because they are different periods, which the prose knows and this does not.

    Before returning, the sanity bounds that need no second number are applied,
    so a retention rate of 11500% leaves here already refused rather than at full
    confidence waiting for a caller to check it.

    ``period_end`` is ``as_of`` for every text match. Prose does not date its own
    figures, and attributing them to the as-of date is the honest reading rather
    than a claim about which quarter the sentence meant.
    """
    settings = settings or KPISettings()
    if not mdna_text or not mdna_text.strip():
        return {}

    working = mdna_text
    out: dict[str, KPI] = {}

    for rule in _TEXT_RULES:
        found: list[tuple[float | None, str | None, str | None, str]] = []
        spans: list[tuple[int, int]] = []
        for pattern in rule.patterns:
            for match in re.finditer(pattern, working, flags=re.IGNORECASE):
                value, refusal, qual, phrase = _read_match(rule, match, settings)
                found.append((value, refusal, qual, phrase))
                spans.append(match.span())
            if found:
                break
        if not found:
            continue

        working = _blank(working, spans)
        kpi = _text_kpi(rule, found, as_of, settings)
        if kpi is not None:
            out[rule.name] = kpi

    # The bounds that need no second number are applied here as well as in
    # build_kpis, because this function is public and a caller reaching for the
    # text reader on its own is exactly the caller who will not remember to gate
    # its output. The ARR-to-revenue bound is not among them: it needs trailing
    # revenue, which prose does not carry. Running the gate twice is harmless,
    # since a figure already refused is passed through untouched.
    gated, _flags = apply_sanity_bounds(out, settings)
    return gated


def _read_match(
    rule: _TextRule, match: re.Match[str], settings: KPISettings
) -> tuple[float | None, str | None, str | None, str]:
    """One match reduced to a value or a refusal, plus its hedge and its phrase."""
    phrase = re.sub(r"\s+", " ", match.group(0).strip())
    qual = match.groupdict().get("qual")

    if rule.kind == "ratio":
        if not match.groupdict().get("pct"):
            return (
                None,
                (
                    "the phrase carries no percent sign, so whether "
                    f"{match.group('num')} means a percentage or a decimal is not "
                    "something the sentence settles"
                ),
                qual,
                phrase,
            )
        return _to_float(match.group("num")) / 100.0, None, qual, phrase

    if rule.kind == "money":
        value, refusal = _resolve_money(match, settings.text_scale_floor)
        return value, refusal, qual, phrase

    if rule.kind == "count":
        value, refusal = _resolve_count(match)
        return value, refusal, qual, phrase

    # ARPU is a price per subscriber per period, so it is neither scaled into
    # millions nor a population. A scale word on it is a misread and refused.
    raw = _to_float(match.group("num"))
    if match.groupdict().get("scale"):
        return (
            None,
            (
                f"the phrase scales the figure by {match.group('scale')}, and an "
                "average revenue per user in millions is a misread unit rather than "
                "a very expensive subscriber"
            ),
            qual,
            phrase,
        )
    return raw, None, qual, phrase


def _text_kpi(
    rule: _TextRule,
    found: list[tuple[float | None, str | None, str | None, str]],
    as_of: date,
    settings: KPISettings,
) -> KPI | None:
    """Fold one rule's matches into a single KPI, refusing where they disagree."""
    readable = [(v, q, p) for v, r, q, p in found if v is not None]
    refusals = [(r, p) for v, r, q, p in found if v is None and r]

    if not readable:
        if not refusals:
            return None
        reason, phrase = refusals[0]
        return KPI(
            name=rule.name,
            value=0.0,
            unit=_TEXT_UNITS[rule.kind],
            period_end=as_of,
            source="text",
            tag_or_phrase=phrase,
            confidence=CONFIDENCE_REFUSED,
            notes=reason,
        )

    values = _distinct([v for v, _q, _p in readable], settings.conflict_tolerance)
    if len(values) > 1:
        listed = ", ".join(f"{v:,.4g}" for v in values)
        phrases = "; ".join(p for _v, _q, p in readable)
        return KPI(
            name=rule.name,
            value=values[0],
            unit=_TEXT_UNITS[rule.kind],
            period_end=as_of,
            source="text",
            tag_or_phrase=phrases,
            confidence=CONFIDENCE_REFUSED,
            notes=(
                f"the text states {len(values)} different values for this metric "
                f"({listed}), which are usually different periods the prose "
                "distinguishes and this parser does not"
            ),
        )

    value, qual, phrase = readable[0]
    assert value is not None
    hedge_note, hedged = _qualifier_note(qual)
    notes = hedge_note

    unit = _TEXT_UNITS[rule.kind]
    confidence = settings.confidence_hedged if hedged else settings.confidence_text

    if rule.kind == "arpu":
        per = _period_word(phrase)
        if per:
            unit = f"usd_per_{per}"
        else:
            confidence = min(confidence, settings.confidence_hedged)
            notes = (
                notes + "; " if notes else ""
            ) + (
                "the phrase does not say per what period, so the figure is a price "
                "without a denominator until a reader supplies one"
            )

    if refusals:
        notes = (notes + "; " if notes else "") + (
            f"{len(refusals)} further match in this text could not be read: "
            + refusals[0][0]
        )

    return KPI(
        name=rule.name,
        value=value,
        unit=unit,
        period_end=as_of,
        source="text",
        tag_or_phrase=phrase,
        confidence=confidence,
        notes=notes,
    )


def _period_word(phrase: str) -> str | None:
    match = re.search(r"per\s+(month|quarter|year)\b", phrase, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace matched spans with spaces so a later rule cannot re-read them.

    Spaces rather than deletion, so every remaining offset stays where it was and
    the phrases reported by later rules still line up with the original text.
    """
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Sanity bounds
# --------------------------------------------------------------------------- #


def apply_sanity_bounds(
    kpis: dict[str, KPI],
    settings: KPISettings,
    *,
    revenue: float | None = None,
) -> tuple[dict[str, KPI], list[str]]:
    """Flag figures that cannot be what they claim to be, and withhold them.

    Nothing is corrected. An NRR of 11500% is a percent sign read twice, and the
    arithmetic fix is obvious, which is exactly why it must not be applied: the
    same value could equally be a transcription error, a footnote about a single
    customer cohort, or a genuinely different metric, and dividing by a hundred
    would make three different errors look like one clean number.

    So the figure keeps its value, keeps its phrase, is reported with the bound it
    broke, and is dropped to confidence 0.0 so it never reaches a model through
    ``as_dict``. The flag names the metric and the bound.

    ``revenue`` is trailing twelve month revenue in USD millions, used only for
    the ARR check, which is the one bound that needs a second number to mean
    anything.
    """
    flagged: dict[str, KPI] = {}
    flags: list[str] = []

    for key, kpi in kpis.items():
        breach = _breach(kpi, settings, revenue)
        if breach is None or not kpi.usable:
            flagged[key] = kpi
            continue
        flags.append(f"{key}: {breach}")
        flagged[key] = KPI(
            name=kpi.name,
            value=kpi.value,
            unit=kpi.unit,
            period_end=kpi.period_end,
            source=kpi.source,
            tag_or_phrase=kpi.tag_or_phrase,
            confidence=CONFIDENCE_REFUSED,
            notes=(kpi.notes + "; " if kpi.notes else "") + breach,
        )
    return flagged, flags


def _breach(kpi: KPI, settings: KPISettings, revenue: float | None) -> str | None:
    """The bound this figure broke, or None."""
    base = kpi.name
    if base in ("net_revenue_retention", "gross_revenue_retention"):
        if not settings.nrr_min <= kpi.value <= settings.nrr_max:
            return (
                f"a retention rate of {kpi.value:,.4g} sits outside "
                f"[{settings.nrr_min}, {settings.nrr_max}]; the usual cause is a "
                "percent sign read as part of the number"
            )
    if base == "churn":
        if not settings.churn_min <= kpi.value <= settings.churn_max:
            return (
                f"a churn rate of {kpi.value:,.4g} sits outside "
                f"[{settings.churn_min}, {settings.churn_max}]; churn is a share of "
                "a base and cannot exceed it"
            )
    if base == "arr":
        if kpi.value <= 0:
            return f"ARR of {kpi.value:,.4g} is not positive"
        if revenue and revenue > 0:
            ratio = kpi.value / revenue
            if not settings.arr_to_revenue_min <= ratio <= settings.arr_to_revenue_max:
                return (
                    f"ARR of {kpi.value:,.1f} is {ratio:,.2f} times trailing revenue "
                    f"of {revenue:,.1f}, outside [{settings.arr_to_revenue_min}, "
                    f"{settings.arr_to_revenue_max}]; a run rate and a trailing "
                    "twelve months differ by the growth rate, not by an order of "
                    "magnitude"
                )
    if base == "arpu" and kpi.value <= settings.arpu_min:
        return f"ARPU of {kpi.value:,.4g} is not a positive price"
    if base in ("customers", "customers_over_100k", "subscribers", "paid_subscribers"):
        if kpi.value <= 0:
            return f"a population of {kpi.value:,.4g} is not a population"
    if base in ("rpo", "rpo_current", "billings") and kpi.value < 0:
        return f"{base} of {kpi.value:,.1f} is negative, which it cannot be"
    return None


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_kpis(
    ticker: str,
    client: Any,
    assumptions: Any,
    as_of: date | None = None,
    mdna_text: str | None = None,
) -> KPISet:
    """Every operating metric one filer discloses, and the reasons for the rest.

    Tagged facts first, because a tagged number is exact and a parsed one is a
    reading. Then the derivations, then the text, and where the text disagrees
    with a tagged fact both are kept: the tagged one under the metric's own name
    and the text one under ``<name>__text``, with the disagreement flagged. The
    tagged one stays usable because it is the better evidence, but nothing about
    that choice is silent.

    ``mdna_text`` is Item 7 of the latest 10-K or the equivalent discussion in a
    10-Q, as plain text with markup already stripped, which is what
    ``EdgarClient.filing_text`` returns. It is not fetched here even when the
    client could fetch it, and that is deliberate: locating Item 7's boundaries
    is a separate job, and running these patterns over an entire 10-K reads risk
    factors and forward-looking statements as though they were results. Without
    it, the text-only metrics are reported as missing with that reason, not as
    absent from the filing.

    Returns an empty ``KPISet`` with reasons rather than raising when a company
    discloses none of this. A semiconductor company has no net revenue retention,
    and saying so is the answer.
    """
    as_of = as_of or _valuation_date(client, assumptions)
    settings = settings_from_assumptions(assumptions)
    wanted = target_metrics(assumptions)
    result = KPISet(ticker=ticker.upper(), as_of=as_of)
    result.notes.append(f"extraction thresholds from {settings.source}")

    if not extraction_enabled(assumptions):
        result.notes.append(
            "assumptions.tmt.kpi_extraction is off, so no operating metric was read"
        )
        result.missing = {
            name: "KPI extraction is switched off in assumptions" for name in wanted
        }
        return result

    if not wanted:
        result.notes.append(
            "no operating metric pack applies to this sub-vertical; companies here "
            "are priced on the statements rather than on a recurring revenue or "
            "subscriber disclosure"
        )
        return result

    facts = client.company_facts(ticker)
    instance = _instance_facts(client, ticker)
    tagged = extract_from_xbrl(
        facts, client, ticker, as_of, settings=settings, instance=instance
    )
    _rows, accession, filed, unreadable = instance
    if unreadable:
        result.notes.append(unreadable)
    elif accession:
        result.notes.append(f"tagged facts read from {accession}, filed {filed}")

    merged: dict[str, KPI] = {
        name: kpi for name, kpi in tagged.items() if name in wanted
    }

    if "billings" in wanted and "billings" not in merged:
        billings = derive_billings(facts, as_of, settings=settings)
        if billings is not None:
            merged["billings"] = billings

    if mdna_text:
        from_text = extract_from_text(mdna_text, as_of, settings=settings)
        for name, kpi in from_text.items():
            if name not in wanted:
                continue
            existing = merged.get(name)
            if existing is None:
                merged[name] = kpi
                continue
            if not (existing.usable and kpi.usable):
                merged[f"{name}__text"] = kpi
                continue
            gap = _relative_gap(existing.value, kpi.value)
            if gap <= settings.conflict_tolerance:
                merged[name] = KPI(
                    name=existing.name,
                    value=existing.value,
                    unit=existing.unit,
                    period_end=existing.period_end,
                    source=existing.source,
                    tag_or_phrase=existing.tag_or_phrase,
                    confidence=existing.confidence,
                    notes=(
                        existing.notes + "; " if existing.notes else ""
                    )
                    + f"the text agrees, at {kpi.value:,.4g} from {kpi.tag_or_phrase!r}",
                )
                continue
            merged[f"{name}__text"] = kpi
            result.flags.append(
                f"{name}: the tagged value of {existing.value:,.4g} and the text "
                f"value of {kpi.value:,.4g} disagree by {gap:.1%}; both are reported "
                f"and the tagged one is the usable figure. Text phrase: "
                f"{kpi.tag_or_phrase!r}"
            )
    else:
        result.notes.append(
            "no MD&A text was supplied, so metrics that exist only in prose could "
            "not be looked for; pass mdna_text to reach retention, subscribers, "
            "ARPU and churn"
        )

    revenue = _trailing_revenue(facts, as_of)
    merged, bound_flags = apply_sanity_bounds(merged, settings, revenue=revenue)
    result.flags.extend(bound_flags)

    for key, kpi in merged.items():
        if not kpi.usable:
            result.flags.append(f"{key}: not usable, {kpi.notes}")
        elif kpi.confidence <= settings.confidence_hedged:
            result.flags.append(f"{key}: low confidence, {kpi.notes}")

    result.kpis = dict(sorted(merged.items()))
    result.missing = _missing_reasons(wanted, merged, mdna_text)
    return result


def _missing_reasons(
    wanted: Iterable[str], found: dict[str, KPI], mdna_text: str | None
) -> dict[str, str]:
    """Why each metric that was looked for is not here.

    Absence is a fact about the filer, not a failure of the run, and the reason
    separates the two cases that matter: the filer tags and writes nothing about
    this metric, or the only place it could have been was prose that nobody
    supplied.
    """
    text_only = {
        "net_revenue_retention",
        "gross_revenue_retention",
        "customers",
        "customers_over_100k",
        "subscribers",
        "paid_subscribers",
        "arpu",
        "churn",
        "postpaid_net_adds",
    }
    out: dict[str, str] = {}
    for name in wanted:
        if name in found:
            continue
        if name in text_only and not mdna_text:
            out[name] = (
                "no filer in this sector tags this concept, and no MD&A text was "
                "supplied to read it out of prose"
            )
        elif name == "billings":
            out[name] = (
                "revenue and the two deferred revenue balances twelve months apart "
                "are not all present in this filer's company facts, and billings "
                "built on a guessed opening balance would be worse than none"
            )
        else:
            out[name] = (
                "no element in the filing's instance document matches this concept "
                "and no phrasing matched in the text supplied, so this filer does "
                "not disclose it"
            )
    return out


def _trailing_revenue(facts: CompanyFacts, as_of: date) -> float | None:
    """Trailing twelve month revenue in USD millions, for the ARR bound only."""
    ends = [
        f.end
        for tag in tags.REVENUE
        for f in facts.facts(tag)
        if not f.is_instant and f.end <= as_of
    ]
    if not ends:
        return None
    try:
        value, _prov = facts.resolve_ttm("revenue", tags.REVENUE, max(ends))
    except MissingDataError:
        return None
    return None if value is None else value / _MM


def _valuation_date(client: Any, assumptions: Any) -> date:
    """The client's knowledge date, then the assumptions file, then today.

    The client outranks the assumptions file because a client built with a
    knowledge date is already refusing to hand back anything filed after it, and
    reading metrics past that date would put facts in the KPI set that the rest
    of the run cannot see.
    """
    from_client = getattr(client, "knowledge_date", None)
    if from_client is not None:
        return from_client
    raw = getattr(assumptions, "as_of", None)
    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError as exc:
            raise ConfigError(f"as_of must be an ISO date, got {raw!r}: {exc}") from exc
    return date.today()
