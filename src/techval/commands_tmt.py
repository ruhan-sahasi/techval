"""The TMT fundamentals surface: operating metrics, segments, sum of the parts.

Three commands over ``techval.tmt``. They are the least model-ish part of the
package and the closest to what a coverage banker actually puts on a page, so
the discipline they carry is about evidence rather than about method.

    techval kpis     TICKER   the operating metrics the sector is priced on
    techval segments TICKER   revenue, operating income and margin by segment
    techval sotp     TICKER   each segment on its own multiple, summed

**Everything printed says where it came from.** That is not decoration. Net
revenue retention read off a tagged fact, net revenue retention parsed out of a
sentence in Item 7, and net revenue retention a reader typed on the command line
are three different kinds of claim, and a table that prints them in the same
column without saying which is which invites a reader to trust the weakest one
as much as the strongest. Every row in the KPI table carries the exact element
name or the exact matched sentence fragment, so the figure can be found in the
filing by searching for it.

**The two vocabularies, and why this module holds the map.**
``taxonomy.SubVertical`` has eleven buckets. ``metrics.SUB_VERTICALS``, which
decides which computed metric pack applies, is keyed on a different eleven
strings, and the two overlap on exactly two values: ``internet`` and
``telecom``. Hand ``metrics.build_metrics`` any of the other nine and it raises
``ConfigError``, which is the right behaviour for that module: a semiconductor
company scored on the Rule of 40 is a worse answer than no answer. A tripwire
test in ``tests/ml/test_warranted.py`` measures the gap and exists to fail if
somebody reconciles the two vocabularies.

So the map lives here, in the command layer, where a judgment about which pack
a bucket belongs to can be read and argued with, rather than inside a module
whose contract is to refuse what it was not written for.
``SUB_VERTICAL_TO_METRIC_PACK`` below states each choice and its reason.
Three buckets map to nothing and the command refuses them by name rather than
falling through to a default pack.

**The KPI bridge, and the second vocabulary gap nobody had hit yet.**
``kpis.KPISet`` and ``metrics.KPI_INPUTS`` are also keyed differently, and the
mismatch is quieter than the first because nothing raises. Passing
``KPISet.as_dict()`` straight into ``build_metrics`` looks like it works and
silently delivers almost nothing: the retention figure is called
``net_revenue_retention`` on one side and ``nrr`` on the other, content
amortisation is spelled with an s on one side and a z on the other, and
subscribers are an absolute count on one side and millions on the other, so a
carrier with 130 million subscribers would arrive as 130,000,000 and produce an
ARPU consistency gap of nine orders of magnitude. ``_bridge_kpis`` does the
translation explicitly, converts the units, and prints one line per figure that
did not cross and why. Nothing crosses silently and nothing is guessed:

    ARPU is a price per subscriber per *some* period. The extractor records
    which period in the unit when the sentence said so and leaves the unit as
    ``usd_per_period`` when it did not. A quarterly ARPU passed to a pack that
    documents monthly ARPU is wrong by three, so the quarterly and annual cases
    are divided down and the unknown case is refused.

    Churn carries no period at all, only ``ratio``. Monthly churn of 1.5% and
    annual churn of 1.5% are different businesses, and one over the wrong one is
    an implied subscriber life wrong by a factor of twelve. Churn therefore
    crosses only where the tag or the matched sentence says month.

**Mounting.** This module exposes ``app`` with no name set, so ``cli.py`` can
decide whether to mount it as a group or merge the commands in at the top level.
Nothing here imports ``cli``.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import tags
from .config import Assumptions
from .edgar import EdgarClient, HttpCache
from .errors import ConfigError, MissingDataError, TechvalError
from .ev_bridge import build_ev_bridge
from .financials import build_financials
from .market import MarketData, make_price_source
from .nlp.sections import load_sections
from .tmt import kpis as kpi_module
from .tmt import metrics as metric_module
from .tmt.metrics import build_metrics
from .tmt.segments import RECONCILIATION_TOLERANCE, build_segments
from .tmt.sotp import run_sotp
from .tmt.taxonomy import SubVertical, classify

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="TMT fundamentals: operating metrics, segment economics, sum of the parts.",
)

# Same reasoning as cli.py: a segment table is nine columns wide and rich falls
# back to eighty when stdout is not a terminal, which crushes every column to
# three characters in a piped or redirected run.
console = Console(width=None if sys.stdout.isatty() else 120)

_CFG = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file.")
_NOCACHE = typer.Option(False, "--no-cache", help="Bypass the HTTP cache.")
_ASOF = typer.Option(
    None,
    "--as-of",
    help=(
        "Read the company as it was knowable on this date (YYYY-MM-DD). Facts "
        "filed later are discarded and prices stop there."
    ),
)


# --------------------------------------------------------------------------- #
# the sub-vertical map
# --------------------------------------------------------------------------- #

# Which computed metric pack each taxonomy bucket is priced on, or None where no
# pack has been written for it.
#
# The rule, so the map can be checked rather than taken on trust: it carries
# across the judgment ``kpis.METRIC_PACKS`` already made about which *disclosure*
# pack a bucket belongs to, because that module is keyed on the taxonomy
# vocabulary and had to answer the same question. The three buckets that get
# nothing here are exactly the three ``kpis.METRIC_PACKS`` gives an empty tuple.
#
# There is one deliberate divergence. ``kpis`` puts internet on the media
# disclosure pack, because an internet company discloses subscribers and ARPU.
# ``metrics.SUB_VERTICALS`` already holds an explicit opinion that internet is
# computed on the software pack, and internet is one of the two values where the
# two vocabularies do meet, so the module that owns the vocabulary wins and this
# map does not overturn it. An internet name therefore hunts for subscriber
# disclosure and is scored on the Rule of 40, which is how the sell side does in
# fact cover Meta and Netflix's advertising tier.
#
# Two entries are weaker than the rest and are named as weak in the printed
# output rather than buried here:
#
#   gaming -> media. A game studio capitalises development cost and amortises it
#   through earnings on exactly the mechanism a streamer capitalises content, so
#   cash spend against the amortisation charge is the right lens. What the media
#   pack does not carry is bookings, daily actives and ARPDAU, which is most of
#   how gaming is actually quoted. Treat the media pack as the accounting half of
#   a gaming page, not the whole of it.
#
#   towers_fiber -> towers. The taxonomy merges towers and fibre into one bucket
#   for good reason: both are contracted-cash-flow infrastructure. The metric
#   packs do not merge them. A tower REIT is quoted on AFFO and this pack
#   computes an AFFO proxy; a fibre operator is quoted on EBITDA less capex and a
#   tower pack understates how much of its spend is not optional. Pass
#   --metric-pack telecom for a fibre name.
SUB_VERTICAL_TO_METRIC_PACK: dict[SubVertical, str | None] = {
    SubVertical.INFRASTRUCTURE_SOFTWARE: "software",
    SubVertical.APPLICATION_SOFTWARE: "software",
    SubVertical.INTERNET: "internet",
    SubVertical.PAYMENTS: "software",
    SubVertical.MEDIA_ENTERTAINMENT: "media",
    SubVertical.GAMING: "media",
    SubVertical.TELECOM: "telecom",
    SubVertical.TOWERS_FIBER: "towers",
    SubVertical.SEMICONDUCTORS: None,
    SubVertical.HARDWARE: None,
    SubVertical.IT_SERVICES: None,
}

# Buckets whose mapping is a judgment worth a second opinion, with the sentence
# the command prints when one of them is used.
WEAK_MAPPINGS: dict[SubVertical, str] = {
    SubVertical.GAMING: (
        "gaming is scored on the media pack because a studio capitalises "
        "development cost and amortises it through earnings the way a streamer "
        "capitalises content. Bookings, daily actives and ARPDAU are how gaming "
        "is actually quoted and none of them is in this pack, so read it as the "
        "accounting half of a gaming page rather than the whole of it."
    ),
    SubVertical.TOWERS_FIBER: (
        "the taxonomy merges towers and fibre into one bucket and the metric "
        "packs do not. This is the tower pack, which computes an AFFO proxy and "
        "treats capex as largely discretionary. That is right for a tower REIT "
        "and wrong for a fibre operator mid-build. Pass --metric-pack telecom "
        "for a fibre name."
    ),
}

# Why a bucket has no pack. Printed instead of a table, because naming the gap is
# the answer and a default pack would not be.
NO_PACK_REASON: dict[SubVertical, str] = {
    SubVertical.SEMICONDUCTORS: (
        "a chip maker is priced on the cycle: book to bill, utilisation, "
        "inventory weeks and design wins. Not one of those is in any pack here, "
        "and the Rule of 40 applied to a fab is arithmetic without a claim."
    ),
    SubVertical.HARDWARE: (
        "hardware is priced on unit volume, attach rate and the mix between "
        "product and the services sold against the installed base. No pack here "
        "computes any of them."
    ),
    SubVertical.IT_SERVICES: (
        "an IT services firm is priced on bookings, book to bill, utilisation "
        "and headcount pyramid. No pack here computes any of them."
    ),
}


def resolve_metric_pack(
    sub_vertical: SubVertical | None, override: str | None
) -> tuple[str, str]:
    """Which metric pack to build, and the sentence explaining the choice.

    Raises rather than choosing a default. The whole reason this map lives in the
    command layer is that a silent fallback to the software pack is the failure
    it exists to prevent, and a command that guessed would reintroduce it one
    level up from the module that refused to.
    """
    if override:
        key = override.strip().lower()
        if key not in metric_module.SUB_VERTICALS:
            raise ConfigError(
                f"--metric-pack {override!r} is not a metric pack. Known: "
                + ", ".join(sorted(metric_module.SUB_VERTICALS))
            )
        return key, "forced with --metric-pack, overriding the classified bucket"

    if sub_vertical is None:
        raise ConfigError(
            "the filer could not be classified into a TMT sub-vertical, so there "
            "is no metric pack to apply. Set tmt.sub_vertical in the assumptions "
            "file, or pass --sub-vertical, or pass --metric-pack to name the pack "
            "directly. Known sub-verticals: "
            + ", ".join(v.value for v in SubVertical)
        )

    pack = SUB_VERTICAL_TO_METRIC_PACK.get(sub_vertical)
    if pack is None:
        reason = NO_PACK_REASON.get(
            sub_vertical, "no pack here covers this bucket"
        ).rstrip(".")
        raise ConfigError(
            f"no computed metric pack is written for {sub_vertical.value}. "
            f"{reason[0].upper() + reason[1:]}. The disclosed metrics above stand "
            "on their own. Pass --metric-pack to apply another sector's pack "
            "deliberately, knowing that it is another sector's."
        )
    return pack, (
        f"mapped from {sub_vertical.value} by commands_tmt."
        "SUB_VERTICAL_TO_METRIC_PACK"
    )


# --------------------------------------------------------------------------- #
# the KPI bridge
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Crossing:
    """One route from a ``KPISet`` name to a ``metrics.KPI_INPUTS`` key."""

    kpi_name: str
    input_key: str
    units: tuple[str, ...]
    factor: float = 1.0
    note: str = ""


# The straightforward crossings: same quantity, same unit, different spelling.
_CROSSINGS: tuple[_Crossing, ...] = (
    _Crossing("arr", "arr", ("usd_mm",)),
    _Crossing(
        "net_revenue_retention",
        "nrr",
        ("ratio",),
        note="named net_revenue_retention by the extractor and nrr by the pack",
    ),
    _Crossing("rpo", "rpo", ("usd_mm",)),
    _Crossing("billings", "billings", ("usd_mm",)),
    _Crossing("content_spend", "content_spend", ("usd_mm",)),
    _Crossing(
        "content_amortisation",
        "content_amortization",
        ("usd_mm",),
        note="spelled with an s by the extractor and with a z by the pack",
    ),
)

# Subscribers, in preference order. A streamer's paid membership count is the one
# the equity story runs on; a total that includes free tiers is the fallback.
_SUBSCRIBER_SOURCES = ("paid_subscribers", "subscribers")

# A refusal gate on the subscriber count, in dollars of revenue per subscriber
# per month. It moves no number: it decides only whether a count crosses into the
# pack or is reported as read and refused, with the arithmetic printed so a
# reader can overrule it with --kpi subscribers=.
#
# T-Mobile is the case that earned it, and it is worth stating because the
# failure is invisible without the check. The text extractor's subscriber ladder
# matches ``<count> postpaid phone customers``. T-Mobile's Item 7 says it "added
# 3,287,000 postpaid phone customers", which is a year of net additions against a
# base of something over a hundred million, and the sentence does not contain the
# word "net" so the postpaid_net_adds rule above it does not claim the figure
# first. Passed through, the pack reports a carrier with 3.3 million subscribers
# and an implied revenue per subscriber of 2,337 dollars a month, and both
# numbers look like numbers.
#
# The band is set where no consumer subscription business can live. US wireless
# ARPU is under 50 dollars, cable is under 150, and the most expensive bundle any
# listed filer sells does not reach 500. The floor exists for the mirror failure,
# a count read in units rather than millions.
SUBSCRIBER_IMPLIED_FLOOR = 1.0
SUBSCRIBER_IMPLIED_CEILING = 500.0

# ...and why the band alone was not enough, which is the second half of the same
# lesson and cost a printed number to learn.
#
# The band caught T-Mobile and passed Disney. `kpis DIS` read
# ``subscribers 27,100,000`` at confidence 0.75, crossed it, and printed
# "Implied revenue per subscriber (monthly) $304.00", which is inside the band
# and is not a subscriber figure at all. The sentence behind it, in footnote (2)
# to the Key Metrics table of the FY2025 10-K (dis-20250927.htm, accession
# 0001744489-25-000155), reads:
#
#     Includes 43.7 million and 27.1 million subscribers to bundles that have
#     both Disney+ and Hulu as of September 27, 2025 and September 28, 2024,
#     respectively.
#
# So 27.1 million is the PRIOR year's half of an "X and Y ... respectively" pair
# and it counts bundle overlap rather than any base. Disney's Disney+ base on the
# same page is 131.6 million. The extractor matched it because it is the number
# adjacent to the word "subscribers", the two-value guard did not fire because
# only one of the two numbers sits next to that word, and the band could not
# fire because 27.1 million happens to divide into group revenue at a number a
# cable bundle really could cost.
#
# Neither failure is fixable by another arithmetic bound, and both have the same
# shape: a count in prose is a number attached to a population, the sentence
# says how many, and the SUBJECT of the sentence says how many of what. The
# parser sees the number and the noun beside it and never the subject. So a
# subscriber count whose only evidence is a sentence does not reach the computed
# pack, and the refusal prints the fragment and the override.
#
# The cost is real and belongs beside the reason. Almost no filer tags a
# subscriber base in XBRL: Netflix, T-Mobile, Disney and Cloudflare tag none of
# their operating counts, so this rule costs the media and telecom packs their
# subscriber line, their implied revenue per subscriber, their ARPU consistency
# gap and their implied subscriber life at nearly every company. The argument for
# paying that is the alternative on the page above: $304.00 a month for Disney,
# printed at two decimals beside figures that were read off tagged facts, with
# nothing in the computed table to say which was which. One override, --kpi
# subscribers=131.6, buys all four metrics back and puts the reader's eyes on the
# filing while they type it.
#
# Sources whose count may still cross, and why each is different from a sentence.
# A tagged fact names its population in the element name, which is the thing the
# prose case is missing. A statement read and a supplied figure were both fixed
# by somebody who could see what they were looking at. A derived count would be
# arithmetic this engine did on two figures it can name; nothing produces one
# today and the entry is here so that a future one is admitted deliberately
# rather than caught by a rule written about prose.
_SUBSCRIBER_EVIDENCE = ("xbrl_extension", "statements", "supplied", "derived")

# ARPU by the period the extractor managed to read off the sentence.
_ARPU_FACTORS = {"usd_per_month": 1.0, "usd_per_quarter": 1.0 / 3.0, "usd_per_year": 1.0 / 12.0}

# KPISet names with no counterpart in metrics.KPI_INPUTS at all. Listed rather
# than left to fall off the end, because a reader looking at a customer count in
# the evidence table and not finding it in the computed table deserves to know it
# was never going to be there.
_NO_COUNTERPART = {
    "gross_revenue_retention": (
        "no pack computes a gross retention figure; it is reported above as "
        "disclosed and is the floor under net retention"
    ),
    "rpo_current": (
        "the pack reads total RPO only, and computes RPO coverage against "
        "trailing revenue from it"
    ),
    "customers": "no pack computes a customer count",
    "customers_over_100k": "no pack computes a large-customer count",
    "postpaid_net_adds": (
        "the telecom pack computes on the subscriber level and not on the "
        "period's additions"
    ),
}


def _bridge_kpis(kpi_set, fin) -> tuple[dict[str, float], list[str]]:
    """Translate a ``KPISet`` into the dict ``metrics.build_metrics`` reads.

    Returns the dict and one prose line per decision, including every figure
    that did not cross and the reason. The lines are the point: the failure this
    guards against is not an exception, it is a table of metrics that quietly
    came back empty because two modules spell retention differently.

    Rates and prices cross from prose; populations do not. Net revenue retention,
    ARPU and churn are quantities whose meaning is carried by the words next to
    the number, and the extractor checks those words. A count is not: "27.1
    million subscribers" is the same fragment whether the sentence is about a
    base, a period's additions, one product, one geography or a footnote about
    bundle overlap, and only the subject of the sentence separates them. See the
    note above ``_SUBSCRIBER_EVIDENCE`` for the two filings that settled it and
    for what the rule costs.
    """
    out: dict[str, float] = {}
    lines: list[str] = []

    for crossing in _CROSSINGS:
        kpi = kpi_set.get(crossing.kpi_name)
        if kpi is None:
            continue
        if not kpi.usable:
            lines.append(
                f"{crossing.kpi_name}: not passed to the pack, the extractor "
                f"refused it ({kpi.notes})"
            )
            continue
        if kpi.unit not in crossing.units:
            lines.append(
                f"{crossing.kpi_name}: not passed to the pack, it came back in "
                f"{kpi.unit} and the pack documents "
                f"{' or '.join(crossing.units)}"
            )
            continue
        out[crossing.input_key] = kpi.value * crossing.factor
        detail = f", {crossing.note}" if crossing.note else ""
        lines.append(
            f"{crossing.kpi_name} -> kpis[{crossing.input_key!r}]{detail}"
        )

    # -- subscribers: an absolute count on one side, millions on the other --- #
    for name in _SUBSCRIBER_SOURCES:
        kpi = kpi_set.get(name)
        if kpi is None:
            continue
        if not kpi.usable:
            lines.append(
                f"{name}: not passed to the pack, the extractor refused it "
                f"({kpi.notes})"
            )
            continue
        if kpi.unit != "count":
            lines.append(
                f"{name}: not passed to the pack, it came back in {kpi.unit} "
                "rather than as a count"
            )
            continue
        millions = kpi.value / 1e6
        implied = (
            fin.revenue / (millions * 12.0) if millions > 0 else None
        )
        if kpi.source not in _SUBSCRIBER_EVIDENCE:
            lines.append(
                f"FLAG: {name} of {kpi.value:,.0f} was read out of the fragment "
                f"{kpi.tag_or_phrase!r} and is NOT passed to the pack, because a "
                "count parsed out of prose is a number attached to a population "
                "the parser cannot see. The sentence says how many; its subject "
                "says how many of what. Measured on two filings: T-Mobile's "
                "'3,287,000 postpaid phone customers' is a year of net additions "
                "against a base near 130 million, and Disney's '27.1 million "
                "subscribers' is the prior year's half of a footnote counting "
                "subscribers to bundles carrying both Disney+ and Hulu, against a "
                "Disney+ base of 131.6 million on the same page. Both read as a "
                "base and neither is one."
            )
            lines.append(
                f"{name}: for reference only, {kpi.value:,.0f} against "
                f"{fin.revenue:,.0f}mm of trailing revenue would imply "
                + (
                    f"{implied:,.2f} dollars"
                    if implied is not None
                    else "an undefined amount"
                )
                + " of revenue per subscriber per month. That arithmetic is "
                "printed rather than acted on: the numerator is the whole "
                "company's revenue, so at a filer whose subscription business is "
                "one segment among several the ratio is not a revenue per "
                "subscriber at all, and the band it would be judged against was "
                "set on subscription price points. Pass --kpi subscribers="
                f"{kpi.value / 1e6:,.1f} if that count is the base you want, or "
                "the right figure once you have read the sentence in the filing."
            )
            break
        if implied is None or not (
            SUBSCRIBER_IMPLIED_FLOOR <= implied <= SUBSCRIBER_IMPLIED_CEILING
        ):
            lines.append(
                f"FLAG: {name} of {kpi.value:,.0f} was read out of "
                f"{kpi.tag_or_phrase!r} and is NOT passed to the pack. Against "
                f"{fin.revenue:,.0f}mm of trailing revenue it implies "
                + (
                    f"{implied:,.0f} dollars"
                    if implied is not None
                    else "an undefined amount"
                )
                + " of revenue per subscriber per month, outside the "
                f"{SUBSCRIBER_IMPLIED_FLOOR:,.0f} to "
                f"{SUBSCRIBER_IMPLIED_CEILING:,.0f} band any consumer "
                "subscription business occupies. The usual cause is a sentence "
                "about a period's net additions read as the subscriber base. "
                "Override with --kpi subscribers=<millions> if the count is right."
            )
            break
        out["subscribers"] = millions
        lines.append(
            f"{name} -> kpis['subscribers'], divided by a million: the extractor "
            f"reports an absolute count ({kpi.value:,.0f}) and the pack documents "
            f"millions. It implies {implied:,.2f} dollars of revenue per "
            "subscriber per month, which is inside the plausible band."
        )
        break

    # -- ARPU: a price per subscriber per some period ------------------------ #
    arpu = kpi_set.get("arpu")
    if arpu is not None:
        if not arpu.usable:
            lines.append(
                f"arpu: not passed to the pack, the extractor refused it "
                f"({arpu.notes})"
            )
        elif arpu.unit in _ARPU_FACTORS:
            out["arpu"] = arpu.value * _ARPU_FACTORS[arpu.unit]
            period = arpu.unit.removeprefix("usd_per_")
            detail = (
                "already monthly"
                if period == "month"
                else f"divided from {period}ly to monthly"
            )
            lines.append(f"arpu -> kpis['arpu'], {detail}")
        else:
            lines.append(
                "arpu: not passed to the pack. The filing gives a price per "
                "subscriber without saying per what period, and the pack "
                "documents dollars per subscriber per month. A quarterly figure "
                "read as monthly is wrong by three."
            )

    # -- churn: a rate with no period on it ---------------------------------- #
    churn = kpi_set.get("churn")
    if churn is not None:
        if not churn.usable:
            lines.append(
                f"churn: not passed to the pack, the extractor refused it "
                f"({churn.notes})"
            )
        elif re.search(r"month", churn.tag_or_phrase, flags=re.IGNORECASE):
            out["churn"] = churn.value
            lines.append(
                "churn -> kpis['churn']; the evidence says monthly, which is what "
                "the pack documents"
            )
        else:
            lines.append(
                "churn: not passed to the pack. The extractor carries a rate with "
                "no period attached and the evidence "
                f"({churn.tag_or_phrase!r}) does not say month. Implied "
                "subscriber life is one over monthly churn, so an annual rate "
                "read as monthly is wrong by twelve."
            )

    for name, reason in _NO_COUNTERPART.items():
        if kpi_set.get(name) is not None:
            lines.append(f"{name}: reaches no computed metric, {reason}")

    return out, lines


def _pack_input_evidence(
    kpi_set, merged, statement_rows, override_rows
) -> list[tuple[str, str, str]]:
    """One row per figure the pack was given: the key, the rung, the evidence.

    This exists because the two tables this command prints are not the same kind
    of claim and only the first of them says so. The disclosed table carries an
    Evidence column and a confidence rung on every row. The computed table
    carries a name, a value and a unit, so a Rule of 40 built on a tagged fact
    and an implied revenue per subscriber built on a sentence in Item 7 print
    identically, and the provenance a reader was shown twenty lines earlier does
    not survive the crossing. Printing the inputs under the computed table is
    what closes that, and it is cheap: every figure here already carries its own
    label, and nothing new is asserted about any of them.
    """
    # One pack input to the KPISet names it can have come from, in the order the
    # bridge tries them. Subscribers is the one with two, and the preference
    # order has to match ``_SUBSCRIBER_SOURCES`` or the table would name the
    # wrong sentence beside the right number.
    by_input: dict[str, tuple[str, ...]] = {
        c.input_key: (c.kpi_name,) for c in _CROSSINGS
    }
    by_input["arpu"] = ("arpu",)
    by_input["churn"] = ("churn",)
    by_input["subscribers"] = _SUBSCRIBER_SOURCES
    rows: list[tuple[str, str, str]] = []
    supplied = {str(r["name"]): r for r in override_rows}
    filed = {str(r["name"]): r for r in statement_rows}
    for key in sorted(merged):
        if key in supplied:
            rows.append((key, _evidence_label(supplied[key]), str(supplied[key]["tag_or_phrase"])))
            continue
        if key in filed:
            rows.append((key, _evidence_label(filed[key]), str(filed[key]["tag_or_phrase"])))
            continue
        kpi = next(
            (k for k in (kpi_set.get(n) for n in by_input.get(key, (key,))) if k is not None),
            None,
        )
        if kpi is None:
            rows.append((key, "unknown", "this bridge did not record where it came from"))
            continue
        rows.append((key, _evidence_label(kpi.row()), str(kpi.tag_or_phrase)))
    return rows


def _render_pack_inputs(rows: list[tuple[str, str, str]]) -> None:
    if not rows:
        return
    console.print("\n[bold]What the pack was given, and the evidence behind each[/bold]")
    t = Table(box=None, pad_edge=False)
    t.add_column("Pack input", no_wrap=True)
    t.add_column("Evidence", no_wrap=True)
    t.add_column("Tag or phrase", overflow="fold", style="dim")
    weak = {"prose", "prose, hedged", "command line", "unknown"}
    for key, label, evidence in rows:
        style = "yellow" if label in weak else ""
        t.add_row(Text(key, style=style), Text(label, style=style), evidence)
    console.print(t)
    console.print(
        "[dim]Every metric above was computed from these and from the filed "
        "statements. A metric is no better than the weakest row it stands on, and "
        "the computed table has no column that says so.[/dim]"
    )


# --------------------------------------------------------------------------- #
# the figures the packs need that live on the statements
# --------------------------------------------------------------------------- #

# Sales and marketing expense, which the normalized statement set does not carry
# and which the magic number and CAC payback both need. One entry deliberately.
# SG&A is not sales and marketing: folding general and administrative cost into
# the denominator makes the magic number smaller and the payback longer, and it
# does so invisibly because both numbers stay plausible. Where a filer tags only
# SG&A the figure is reported as unavailable, which is the true answer.
SALES_AND_MARKETING = ["SellingAndMarketingExpense"]

_MM = 1e6


def _prior_anchor(fin, facts) -> date:
    """The period end the filer actually reported nearest a year back.

    A fixed 365 day step looks equivalent and is not. A 52/53 week filer closes
    its year 364 days back, one day before where the step lands, the tiler cannot
    cover a window whose end falls between two reported periods, and every such
    filer would report no growth and no magic number forever.

    The candidate period ends are pooled across **every** tag on the revenue
    ladder rather than taken from the first one that carries any facts at all.
    That distinction is not academic and Nvidia is the case that earned it.
    Nvidia stopped tagging ``RevenueFromContractWithCustomerExcludingAssessedTax``
    after fiscal 2022 and reports under ``Revenues`` now, but the retired tag
    still has facts, so a reader that stops at the first tag with a series gets a
    list of period ends that stop in 2022, finds nothing within ten days of the
    target, falls back to the fixed 365 day step and lands on 2025-07-26 against
    a quarter the company closed on 2025-07-27. The window cannot be tiled, and
    prior-year revenue, revenue growth, the Rule of 40 and the magic number all
    come back unavailable for the largest company in the sector. See the note in
    the pull request: ``comps._revenue_growth`` anchors the same way and has the
    same exposure.
    """
    target = fin.as_of - timedelta(days=365)
    ends: set[date] = set()
    for tag in tags.REVENUE:
        if not isinstance(tag, str):
            continue
        ends.update(f.end for f in facts.facts(tag) if not f.is_instant)
    near = [d for d in ends if abs((d - target).days) <= 10]
    return min(near, key=lambda d: abs((d - target).days)) if near else target


def _statement_kpis(fin, facts) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Prior period revenue and sales and marketing spend, off the filed lines.

    These are the inputs ``metrics`` documents and cannot get for itself: the
    normalized statement set is one trailing twelve month window, so it carries
    no prior period, and it has no sales and marketing line at all. Both are
    read here from company facts over the same tiling the rest of the engine
    uses, so they are filed figures rather than supplied ones, and each comes
    back with the tag and the window it was resolved over.
    """
    values: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    prior_end = _prior_anchor(fin, facts)

    for key, concept, ladder, when in (
        ("revenue_prior", "prior-year revenue", tags.REVENUE, prior_end),
        ("sales_and_marketing", "sales and marketing", SALES_AND_MARKETING, fin.as_of),
        (
            "sales_and_marketing_prior",
            "prior-year sales and marketing",
            SALES_AND_MARKETING,
            prior_end,
        ),
    ):
        raw, prov = facts.resolve_ttm(concept, ladder, when, required=False)
        if raw is None:
            continue
        values[key] = raw / _MM
        rows.append(
            {
                "name": key,
                "value": raw / _MM,
                "unit": "usd_mm",
                "period_end": str(when),
                "source": "statements",
                "tag_or_phrase": prov.tag or ", ".join(ladder),
                "confidence": 1.0,
                "notes": (
                    f"trailing twelve months to {when}, resolved over the same "
                    "period tiling every other figure in this engine uses"
                ),
            }
        )
    return values, rows


def bridge_caveats(fin) -> list[str]:
    """What would make every enterprise-value multiple on the page wrong.

    ``build_financials`` defaults a balance-sheet concept it cannot source to
    zero rather than raising, which is the right call for a filer that genuinely
    has no debt and the wrong one for a filer whose tag went stale. The
    difference is in the provenance and nowhere else, so a command that prints
    EV/EBITDA beside a defaulted debt line has to say so.

    T-Mobile is the live case and it is not small. Its Q2 2026 10-Q tags
    ``ShortTermBorrowings`` at 6,117mm and tags ``LongTermDebtNoncurrent`` only
    as of the last year end, so the staleness guard correctly refuses the stale
    figure and the default then reads long-term debt as zero. Enterprise value
    comes out at 201,807mm against a real figure nearer 275,000mm, EV/EBITDA
    prints 6.2x against something nearer 8.5x, and nothing on the page says so.
    The fix belongs in ``financials``; naming it here is what this command can do.

    **Two defaults that look identical and are not.** The first version of this
    block put the same warning on both and taught a reader to skip it. Measured:
    ``kpis DDOG`` printed five of these, each ending "Every enterprise-value
    multiple below rests on that zero", and Datadog's enterprise value is right.
    It has no straight debt, no current debt and no finance leases, its
    convertible notes resolve under ``ConvertibleLongTermNotesPayable`` at
    985.5mm, and all five statements were true with a false conclusion attached.
    Verizon's block looks exactly the same and its conclusion is true by 143bn.

    The difference is already on the object, in ``Provenance.note``, and this
    reads it rather than adding a heuristic:

        *no tag reports this* means the whole ladder came back empty across every
        period the filer has ever reported. The concept is absent from this
        company's accounts, which is what a debt-free balance sheet looks like.
        Stated once, quietly, in a list.

        *only stale tags found* means the filer DOES tag the concept and the
        newest fact predates the balance sheet this page is built on, so the zero
        is standing in for a number that exists somewhere in the filing history.
        That is the dangerous one and it gets the loud clause, the tag name and
        the date it went stale.

    **And one corroborating check, because a stale tag is not the only way debt
    goes missing.** Verizon's long-term debt reads as zero because it reports
    under ``LongTermDebtAndCapitalLeaseObligations``, which is on no ladder here,
    and the provenance for that is a clean stale-tag note about a tag it stopped
    using in 2013. The independent evidence is on the income statement: 7,348mm
    of interest expense against 21,783mm of resolved debt is an implied cost of
    debt of 34%, which no investment-grade carrier pays. The check is one
    division and it is reported as a question rather than a verdict, because a
    filer that repaid most of its debt during the year shows the same signature
    honestly: a full year of interest against a period-end balance that is nearly
    gone. Both readings are printed.
    """
    out = [f"FLAG: {w}" if "overstated" in w or "not explained" in w else w
           for w in fin.warnings]
    absent: list[str] = []
    for concept, prov in sorted(fin.provenance.items()):
        low = concept.lower()
        if not any(k in low for k in ("debt", "cash", "investment", "lease")):
            continue
        if "defaulted" not in (prov.method or ""):
            continue
        note = prov.note or "no note"
        if "stale" in note:
            out.append(
                f"FLAG: {concept} was read as zero even though this filer tags "
                f"it: {note}. A zero standing in for a figure the company does "
                "report is not an absence, it is a gap the size of whatever that "
                "figure is now. Every enterprise-value multiple below rests on "
                "it, so read the balance sheet before quoting one."
            )
        else:
            absent.append(concept)
    if absent:
        out.append(
            f"{len(absent)} balance-sheet concept(s) read as zero because no tag "
            "in the ladder reports them in any period this filer has ever filed: "
            + "; ".join(absent)
            + ". That is what a company which genuinely does not have the line "
            "looks like, so it is reported rather than flagged. The same default "
            "becomes a FLAG where the filer does tag the concept and the newest "
            "fact is stale."
        )
    out.extend(_leverage_crosscheck(fin))
    return out


# Interest expense divided by debt, above which the debt side of the bridge is
# not believable. Set at a level no rated issuer pays on its whole stack: US
# high-yield coupons top out in the low teens, and an issuer paying twice that
# would not be reporting zero debt. It is deliberately far above the false
# positive it has to tolerate, which is a company that repaid its debt during the
# year and carries a full year of interest against almost none of it.
IMPLIED_COST_OF_DEBT_CEILING = 0.25


def _leverage_crosscheck(fin) -> list[str]:
    """Does the interest on the income statement fit the debt on the page.

    A second opinion on the balance sheet, from the one part of the filing that
    cannot be missed by a tag ladder: a company pays interest on debt whether or
    not this engine found the debt. It catches the case the provenance cannot,
    where every tag resolved cleanly and the ladder simply has no entry for the
    element this filer uses.
    """
    interest = fin.interest_expense
    if interest is None or interest <= 0:
        # A filer that reports interest net of interest income can carry a
        # negative figure here, and a ratio built on it means nothing.
        return []
    debt = fin.straight_debt + fin.convertible_debt + fin.finance_lease_liability
    if debt <= 0:
        return [
            f"FLAG: the income statement carries {interest:,.1f}mm of interest "
            "expense and the debt side of this bridge is zero. A company with no "
            "debt does not pay interest on it, so either the interest is on "
            "something this bridge does not count as debt, or the debt is tagged "
            "under an element the ladder does not read. Enterprise value is "
            "understated in the second case and nothing here can tell you which."
        ]
    implied = interest / debt
    if implied <= IMPLIED_COST_OF_DEBT_CEILING:
        return []
    return [
        f"FLAG: {interest:,.1f}mm of interest expense against {debt:,.1f}mm of "
        f"debt is an implied cost of debt of {implied:.1%}, above the "
        f"{IMPLIED_COST_OF_DEBT_CEILING:.0%} this engine will believe. The "
        "reading that costs money is that the debt is understated because part "
        "of it is tagged under an element the ladder does not read, which makes "
        "every enterprise-value multiple below too low. The innocent reading is "
        "that the company repaid its debt during the year, so a full year of "
        "interest sits against a period-end balance that is nearly gone. Both "
        "are visible on the balance sheet in about a minute."
    ]


def _parse_kpi_overrides(raw: list[str] | None) -> tuple[dict[str, float], list[dict]]:
    """``--kpi name=value``, validated against the keys the packs actually read.

    A key outside ``metrics.KPI_INPUTS`` is refused rather than carried, because
    the usual way a supplied figure goes missing is a caller writing
    ``arpu_monthly`` to a reader that only looks for ``arpu``, and the value then
    sits in the dict reaching nothing.
    """
    values: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for item in raw or []:
        if "=" not in item:
            raise ConfigError(
                f"--kpi {item!r} is not name=value. Example: "
                "--kpi sales_and_marketing=1234.5"
            )
        name, _, text = item.partition("=")
        name = name.strip()
        if name not in metric_module.KPI_INPUTS:
            raise ConfigError(
                f"--kpi {name!r} is not a figure any pack reads. Known: "
                + ", ".join(sorted(metric_module.KPI_INPUTS))
            )
        try:
            value = float(text.strip())
        except ValueError as exc:
            raise ConfigError(
                f"--kpi {name}={text.strip()!r} is not a number."
            ) from exc
        values[name] = value
        rows.append(
            {
                "name": name,
                "value": value,
                "unit": metric_module.KPI_INPUTS[name].rsplit(",", 1)[-1].strip(),
                "period_end": "",
                "source": "supplied",
                "tag_or_phrase": f"--kpi {name}={text.strip()}",
                "confidence": 1.0,
                "notes": (
                    "typed on the command line, not read from any filing. "
                    + metric_module.KPI_INPUTS[name]
                ),
            }
        )
    return values, rows


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _rule(title: str) -> None:
    console.print()
    console.rule(Text(title, style="bold"), style="dim")


def _money(v: float | None, dp: int = 0) -> str:
    if v is None:
        return "n/a"
    # Rounded to nothing prints as nothing. Without this, a discount of exactly
    # zero arrives as -0.0 from the module's sign convention and renders "-0",
    # which reads as a rounding artefact rather than as a line that was not used.
    if abs(v) < 0.5 / (10**dp):
        return f"{0.0:,.{dp}f}"
    return f"({abs(v):,.{dp}f})" if v < 0 else f"{v:,.{dp}f}"


def _dollars(v: float | None, dp: int = 2) -> str:
    """A per-share figure. The sign goes outside the dollar sign, not inside it."""
    if v is None:
        return "n/a"
    return f"$({abs(v):,.{dp}f})" if v < 0 else f"${v:,.{dp}f}"


def _pct(v: float | None, dp: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{dp}%}"


def _notes(items, *, heading: str = "Notes") -> None:
    items = [i for i in items if i]
    if not items:
        return
    console.print(f"\n[bold]{heading}[/bold]")
    for n in items:
        style = "yellow" if str(n).upper().startswith(("FLAG", "WARN")) else "dim"
        console.print(Text(f"  - {n}", style=style))


# How the printed value is formatted, read off the first words of the metric's
# own definition. Every definition in ``metrics`` states its unit first and the
# pack refuses to be constructed if a metric arrives without one, so the renderer
# never has to keep a second list of metric names in step with that module.
def _format_metric(value: float | None, definition: str) -> str:
    if value is None:
        return "n/a"
    head = definition.strip().lower()
    if head.startswith("decimal"):
        return _pct(value)
    if head.startswith("percentage points"):
        return f"{value:,.1f}"
    if head.startswith("usd millions"):
        return _money(value, 1)
    if head.startswith("millions"):
        return f"{value:,.2f}mm"
    if head.startswith("multiple"):
        return f"{value:,.2f}x"
    if head.startswith("months"):
        return f"{value:,.1f}"
    if head.startswith("dollars per"):
        return f"${value:,.2f}"
    return f"{value:,.2f}"


# Evidence labels, ordered worst to best so the sort key below reads as quality.
_EVIDENCE_RANK = {
    "supplied": 0,
    "text": 1,
    "derived": 2,
    "statements": 3,
    "xbrl_extension": 4,
}


def _evidence_label(row: dict) -> str:
    """Plain English for where one figure came from.

    ``xbrl_extension`` covers both a filer's own extension element and the
    handful of standard concepts the sector needed, and the two are not the same
    evidence: a us-gaap concept means the same thing at every filer, and an
    extension element means whatever this filer decided it means. The extractor
    records which in the notes, because the instance parser reduces every element
    to its local name and the namespace is no longer on the fact.
    """
    source = str(row.get("source", ""))
    notes = str(row.get("notes", ""))
    if source == "xbrl_extension":
        if "standard us-gaap concept" in notes:
            return "XBRL, us-gaap"
        return "XBRL, extension"
    if source == "statements":
        return "XBRL, us-gaap"
    if source == "derived":
        return "derived"
    if source == "supplied":
        return "command line"
    if source == "text":
        return "prose, hedged" if "hedged" in notes else "prose"
    return source or "unknown"


def _format_kpi_value(value: float, unit: str) -> str:
    if unit == "usd_mm":
        return _money(value, 1)
    if unit == "ratio":
        return _pct(value, 2)
    if unit == "count":
        return f"{value:,.0f}"
    if unit.startswith("usd_per"):
        return f"${value:,.2f}"
    return f"{value:,.4g}"


# --------------------------------------------------------------------------- #
# setup, the same shape cli.py uses
# --------------------------------------------------------------------------- #


def _setup(config: Path | None, no_cache: bool, as_of: str | None):
    assumptions = Assumptions.load(config)
    if as_of:
        assumptions.as_of = as_of
    knowledge = date.fromisoformat(assumptions.as_of) if assumptions.as_of else None
    cache = HttpCache(enabled=not no_cache)
    client = EdgarClient(cache, knowledge_date=knowledge)
    source = make_price_source(assumptions.price_source, cache, assumptions.price_csv_dir)
    market = MarketData(source, cache, today=knowledge or date.today())
    return assumptions, client, market


def _priced_bridge(ticker: str, fin, market, assumptions):
    """The EV bridge, or None with the reason.

    A metric pack is better without its enterprise-value multiples than not
    printed at all: net revenue retention and the Rule of 40 do not need a price
    and a price source that is down should not take them with it.
    """
    try:
        price = market.spot(ticker)
    except TechvalError as exc:
        return None, None, f"no price, so no enterprise value multiples: {exc}"
    return build_ev_bridge(fin, price, assumptions), price, None


def _classify_filer(ticker: str, client, assumptions, override: str | None):
    """The sub-vertical, its confidence and the evidence, or the override.

    Item 1 is fetched because a SIC-only classification is the weakest rung on
    the ladder and Item 1 is the company describing its own business. The fetch
    failing is not fatal: the code alone still classifies, and the note says the
    call was made on thinner evidence than it could have been.
    """
    if override:
        try:
            picked = SubVertical(override.strip().lower())
        except ValueError as exc:
            raise ConfigError(
                f"--sub-vertical {override!r} is not one of: "
                + ", ".join(v.value for v in SubVertical)
            ) from exc
        return picked, 1.0, "named on the command line, overriding the classifier"

    configured = getattr(assumptions.tmt, "sub_vertical", None)
    if configured:
        try:
            picked = SubVertical(str(configured).strip().lower())
        except ValueError as exc:
            raise ConfigError(
                f"tmt.sub_vertical is {configured!r}, which is not one of: "
                + ", ".join(v.value for v in SubVertical)
            ) from exc
        return picked, 1.0, "set in the assumptions file, overriding the classifier"

    payload = client.submissions(ticker)
    business = None
    caveat = ""
    try:
        sections = load_sections(ticker, client)
        business = sections.business.text if sections.business else None
    except TechvalError as exc:
        caveat = f"; Item 1 could not be read, so the code decided alone: {exc}"
    picked, confidence, source = classify(
        payload.get("sic"), (payload.get("name") or ticker).strip(), business, assumptions
    )
    return picked, confidence, source + caveat


def _mdna(ticker: str, client) -> tuple[str | None, str]:
    """Item 7, and a sentence about how it went.

    Half the metrics in this sector exist only in prose, so this is not a
    convenience. It is also the one fetch that can fail in a way worth stating
    plainly: without Item 7 there is no retention, no subscriber count, no ARPU
    and no churn, and the table would otherwise look like a company that
    discloses none of them.
    """
    try:
        sections = load_sections(ticker, client)
    except TechvalError as exc:
        return None, f"Item 7 is unavailable, so prose metrics were not looked for: {exc}"
    section = sections.mdna
    if section is None:
        return None, (
            f"the split of {sections.accession} found no Item 7, so prose metrics "
            "were not looked for"
        )
    return section.text, (
        f"Item 7 of {sections.accession} read, {section.word_count:,} words"
    )


# --------------------------------------------------------------------------- #
# kpis
# --------------------------------------------------------------------------- #


def _render_classification(ticker, fin, sub_vertical, confidence, source, pack, why):
    _rule(f"{fin.entity_name} ({ticker.upper()})  operating metrics")
    console.print(
        f"[dim]CIK {fin.cik}   twelve months ended {fin.as_of}   USD millions[/dim]\n"
    )
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", overflow="fold")
    t.add_row("Sub-vertical", Text(sub_vertical.value if sub_vertical else "unclassified", style="bold"))
    t.add_row("Confidence", f"{confidence:,.2f}")
    t.add_row("Evidence", source)
    t.add_row("Metric pack", Text(f"{pack}  ({why})", style="bold"))
    console.print(t)


def _render_evidence(rows: list[dict]) -> None:
    """Every figure that reached this page, worst evidence last.

    The Evidence column is the column. A number tagged in the filing's instance
    document under a us-gaap concept, a number this engine computed out of two
    other filed lines, and a number a sentence in Item 7 hedged with "about" are
    three different kinds of claim, and the ordering here puts the strongest
    first so a reader scanning down the page is reading down the quality of the
    evidence as well.
    """
    _rule("Disclosed operating metrics, and where each came from")
    if not rows:
        console.print("[dim]Nothing was found. See the table below for why.[/dim]")
        return
    t = Table(box=None, pad_edge=False)
    t.add_column("Metric", no_wrap=True)
    t.add_column("Value", justify="right")
    t.add_column("Unit", style="dim")
    t.add_column("Period end", style="dim", no_wrap=True)
    t.add_column("Evidence", no_wrap=True)
    t.add_column("Conf", justify="right")
    t.add_column("Tag or phrase", overflow="fold", style="dim")
    ordered = sorted(
        rows,
        key=lambda r: (-_EVIDENCE_RANK.get(str(r["source"]), 0), str(r["name"])),
    )
    for r in ordered:
        usable = float(r["confidence"]) > 0.0
        style = "" if usable else "yellow"
        shown = (
            _format_kpi_value(float(r["value"]), str(r["unit"]))
            if usable
            else "refused"
        )
        t.add_row(
            Text(str(r["name"]), style=style),
            Text(shown, style=style or "bold"),
            str(r["unit"]),
            str(r["period_end"]),
            Text(_evidence_label(r), style=style),
            f"{float(r['confidence']):,.2f}",
            str(r["tag_or_phrase"]),
        )
    console.print(t)
    console.print(
        "\n[dim]A refused row is printed so a reader can see what the filing said "
        "and is withheld from every calculation below. Confidence is a rung, not "
        "a probability: 1.00 is a tagged fact, 0.80 arithmetic on two filed "
        "lines, 0.75 a clean sentence, 0.40 a hedged one.[/dim]"
    )
    console.print(
        "[dim]Tagged facts come from the latest periodic filing and prose comes "
        "from the latest 10-K, so for a filer that has reported a quarter since "
        "its annual report the two halves of this table are read out of two "
        "different documents. Both accessions are under Sources. A prose row "
        "carries the valuation date as its period end because the sentence "
        "rarely says which period it measured.[/dim]"
    )


def _render_missing(missing: dict[str, str]) -> None:
    if not missing:
        return
    _rule("Looked for and not disclosed")
    t = Table(box=None, pad_edge=False)
    t.add_column("Metric", no_wrap=True)
    t.add_column("Why", overflow="fold", style="dim")
    for name in sorted(missing):
        t.add_row(name, missing[name])
    console.print(t)
    console.print(
        "\n[dim]Absence is a finding. A software company that discloses no "
        "customer count and a carrier that discloses no churn are different from "
        "a run that failed.[/dim]"
    )


def _render_pack(pack, *, definitions: bool) -> None:
    _rule(f"Computed operating metrics: the {pack.sub_vertical} pack")
    t = Table(box=None, pad_edge=False)
    t.add_column("Metric", no_wrap=True)
    t.add_column("Value", justify="right")
    t.add_column("Unit", style="dim", no_wrap=True)
    for name, value in pack.metrics.items():
        definition = pack.definitions[name]
        unit = definition.split(".")[0].strip()
        bold = value is not None and (
            name.startswith("Rule of 40") or name in ("Net revenue retention", "Magic number")
        )
        style = "bold" if bold else ("" if value is not None else "dim")
        t.add_row(
            Text(name, style=style),
            Text(_format_metric(value, definition), style=style),
            unit if len(unit) < 40 else unit[:37] + "...",
        )
    console.print(t)

    if definitions:
        console.print("\n[bold]Definitions[/bold]")
        for name, value in pack.metrics.items():
            if value is None:
                continue
            console.print(Text(f"  {name}", style="bold"))
            console.print(Text(f"    {pack.definitions[name]}", style="dim"))


@app.command()
def kpis(
    ticker: str,
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    as_of: str = _ASOF,
    sub_vertical: str = typer.Option(
        None,
        "--sub-vertical",
        help="Override the classifier, e.g. infrastructure_software, media_entertainment.",
    ),
    metric_pack: str = typer.Option(
        None,
        "--metric-pack",
        help=(
            "Force the computed pack: software, media, telecom or towers. Use it "
            "for a bucket this map refuses, knowing it is another sector's pack."
        ),
    ),
    kpi: list[str] = typer.Option(
        None,
        "--kpi",
        help=(
            "Supply a figure the filing does not tag, as name=value, repeatable. "
            "Example: --kpi maintenance_capex=420."
        ),
    ),
    no_text: bool = typer.Option(
        False,
        "--no-text",
        help="Skip Item 7. Faster, and gives up retention, subscribers, ARPU and churn.",
    ),
    definitions: bool = typer.Option(
        True,
        "--definitions/--no-definitions",
        help="Print each computed metric's full definition under the table.",
    ),
) -> None:
    """The operating metrics a TMT valuation actually turns on.

    Two tables and they are not the same kind of evidence. The first is what the
    company disclosed, each row carrying the exact element name or the exact
    sentence it was read out of. The second is what this engine computed from
    those disclosures and from the filed statements, each row carrying the
    definition it was computed under, because a Rule of 40 quoted without saying
    which margin went into it is an assertion rather than a fact.
    """
    try:
        assumptions, client, market = _setup(config, no_cache, as_of)
        if assumptions.as_of:
            console.print(
                f"[yellow]Point in time: reading {ticker.upper()} as it was "
                f"knowable on {assumptions.as_of}.[/yellow]"
            )

        facts = client.company_facts(ticker)
        fin = build_financials(ticker, facts=facts)
        picked, confidence, source = _classify_filer(
            ticker, client, assumptions, sub_vertical
        )

        overrides, override_rows = _parse_kpi_overrides(kpi)

        # The disclosure pack is routed off the taxonomy value, which is the
        # vocabulary kpis.METRIC_PACKS is already keyed on, so this needs no map.
        assumptions.tmt.sub_vertical = picked.value if picked else None

        pack_key = pack_why = None
        pack_refusal = None
        try:
            pack_key, pack_why = resolve_metric_pack(picked, metric_pack)
        except ConfigError as exc:
            pack_refusal = str(exc)

        _render_classification(
            ticker,
            fin,
            picked,
            confidence,
            source,
            pack_key or "none",
            pack_why or "refused, see below",
        )
        if picked is not None and metric_pack is None and picked in WEAK_MAPPINGS:
            console.print(
                f"\n[yellow]FLAG: {WEAK_MAPPINGS[picked]}[/yellow]"
            )

        mdna_text, mdna_note = (None, "Item 7 skipped with --no-text")
        if not no_text:
            mdna_text, mdna_note = _mdna(ticker, client)

        kpi_set = kpi_module.build_kpis(
            ticker, client, assumptions, mdna_text=mdna_text
        )
        statement_values, statement_rows = _statement_kpis(fin, facts)

        _render_evidence(kpi_set.rows() + statement_rows + override_rows)
        _render_missing(kpi_set.missing)

        bridged, bridge_lines = _bridge_kpis(kpi_set, fin)
        # Filed figures beat parsed ones, and a figure typed on the command line
        # beats both, because the reader who typed it is looking at the filing.
        merged = {**bridged, **statement_values, **overrides}

        if pack_refusal is not None:
            _rule("No computed metric pack")
            console.print(f"[yellow]{pack_refusal}[/yellow]")
        else:
            bridge, _price, price_note = _priced_bridge(ticker, fin, market, assumptions)
            pack = build_metrics(fin, pack_key, assumptions, kpis=merged, bridge=bridge)
            _render_pack(pack, definitions=definitions)
            _render_pack_inputs(
                _pack_input_evidence(kpi_set, merged, statement_rows, override_rows)
            )
            _notes(pack.flags, heading="Not computed")
            _notes(([price_note] if price_note else []) + pack.notes)

        _notes(bridge_lines, heading="How the disclosed figures reached the pack")
        _notes(bridge_caveats(fin), heading="Data quality")
        _notes([mdna_note] + kpi_set.notes, heading="Sources")
        _notes(kpi_set.flags, heading="Extraction flags")
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #


def _segment_table(rows, *, title: str, residual_label: str, consolidated: float,
                   residual: float, geography: bool) -> Table:
    t = Table(title=title, title_justify="left", box=None, pad_edge=False)
    t.add_column("Segment" if not geography else "Geography", overflow="fold")
    t.add_column("Revenue", justify="right")
    t.add_column("% of total", justify="right")
    if not geography:
        t.add_column("Operating income", justify="right")
        t.add_column("Margin", justify="right")
        t.add_column("D&A", justify="right")
        t.add_column("Capex", justify="right")
    t.add_column("Assets" if not geography else "Long-lived assets", justify="right")
    for s in rows:
        cells = [s.name, _money(s.revenue), _pct(s.revenue_share)]
        if not geography:
            cells += [
                _money(s.operating_income),
                _pct(s.margin),
                _money(s.da),
                _money(s.capex),
            ]
        cells.append(_money(s.assets))
        t.add_row(*cells)
    blanks = [""] * (4 if not geography else 0)
    t.add_row(
        Text(residual_label, style="yellow"),
        Text(_money(residual), style="yellow"),
        "",
        *blanks,
        "",
    )
    t.add_row(
        Text("Consolidated revenue", style="bold"),
        Text(_money(consolidated), style="bold"),
        "",
        *blanks,
        "",
    )
    return t


def _render_reconciliation(report) -> None:
    """The control that decides whether anything above is worth reading.

    Segment revenue must sum to consolidated revenue. Where it does not, the
    difference is intersegment eliminations, a corporate or unallocated bucket,
    or a member the reader failed to spot, and every share, every margin and
    every sum-of-the-parts weight built on the table is then wrong by an amount
    nobody has measured.
    """
    total = sum(s.revenue for s in report.segments if s.revenue is not None)
    _rule("Reconciliation")
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", justify="right")
    t.add_column("", style="dim", overflow="fold")
    t.add_row("Sum of reported segments", _money(total), "")
    t.add_row("Consolidated revenue", _money(report.consolidated_revenue), "")
    share = (
        report.unallocated / report.consolidated_revenue
        if report.consolidated_revenue
        else None
    )
    t.add_row(
        Text("Residual", style="bold"),
        Text(_money(report.unallocated), style="bold"),
        (
            f"{_pct(share, 2)} of consolidated revenue"
            if share is not None
            else "consolidated revenue is not positive"
        ),
    )
    verdict = "foots" if report.reconciles else "DOES NOT FOOT"
    t.add_row(
        Text(f"Within {RECONCILIATION_TOLERANCE:.0%} tolerance", style="bold"),
        Text(verdict, style="bold" if report.reconciles else "bold yellow"),
        (
            "a negative residual is intersegment revenue eliminated on "
            "consolidation, a positive one is a corporate or unallocated bucket"
        ),
    )
    console.print(t)


@app.command()
def segments(
    ticker: str,
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    as_of: str = _ASOF,
    geography: bool = typer.Option(
        True, "--geography/--no-geography", help="Print the geography cut as well."
    ),
) -> None:
    """Revenue, operating income and margin by reportable segment and geography.

    Read out of the dimensioned XBRL in the latest 10-K, which is where segment
    detail lives: the ``companyfacts`` endpoint publishes undimensioned facts
    only and is structurally blind to it.

    The segment total is printed against the consolidated total and the gap is
    shown, because reconciling items and unallocated corporate cost are real. A
    conglomerate valued on its consolidated margin is being valued as the average
    of businesses that do not trade at the same multiple, and the average
    describes none of them.
    """
    try:
        assumptions, client, _market = _setup(config, no_cache, as_of)
        report = build_segments(ticker, client, assumptions)

        _rule(f"{ticker.upper()}  segment economics, USD millions")
        period = report.segments[0].period_start if report.segments else None
        console.print(
            f"[dim]Year ended {report.as_of}"
            + (f", from {period}" if period else "")
            + f"   {report.accession or 'no accession'}"
            + (f"   revenue tagged as {report.revenue_tag}" if report.revenue_tag else "")
            + "[/dim]\n"
        )
        console.print(
            _segment_table(
                report.segments,
                title="",
                residual_label="Unallocated / eliminations",
                consolidated=report.consolidated_revenue,
                residual=report.unallocated,
                geography=False,
            )
        )
        _render_reconciliation(report)

        h = report.herfindahl
        if h is not None:
            reading = (
                "one business with a rounding error attached, and a consolidated "
                "multiple is the right lens"
                if h >= 0.7
                else (
                    "a genuine conglomerate: a sum of the parts values it better "
                    "than any single multiple, because the blended multiple the "
                    "market applies is a weighted average nobody chose"
                    if h <= 0.4
                    else "concentrated but not a single business"
                )
            )
            console.print(
                f"\n  [bold]Revenue concentration (Herfindahl)   {h:,.3f}[/bold]"
            )
            console.print(f"  [dim]{reading}.[/dim]")

        if geography:
            geo_total = sum(g.revenue for g in report.geographies if g.revenue is not None)
            _rule("By geography")
            if not report.geographies:
                console.print(
                    "[yellow]The filing tags no single-axis geography total. ASC 280 "
                    "requires revenue and long-lived assets by country or region, but "
                    "a filer that tags them only crossed with another axis leaves no "
                    "row that can be read without double counting.[/yellow]"
                )
            else:
                console.print(
                    _segment_table(
                        report.geographies,
                        title="",
                        residual_label="Unallocated / other geographies",
                        consolidated=report.consolidated_revenue,
                        residual=report.consolidated_revenue - geo_total,
                        geography=True,
                    )
                )
                console.print(
                    "\n[dim]A second cut of the same revenue, not an addition to the "
                    "segment table. Operating income by geography is absent because "
                    "ASC 280-10-50-41 does not require it: what the standard asks for "
                    "by geography is revenue and long-lived assets, and a margin "
                    "column here would be a figure no filer reports.[/dim]"
                )

        _notes(report.notes, heading="How the table was read")
        _notes(report.flags, heading="Flags")
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# sotp
# --------------------------------------------------------------------------- #


def load_plan(path: Path) -> dict:
    """Read and validate the sum-of-the-parts plan.

    The multiples are deliberately not in the assumptions file. A peer set for a
    cable network has no business living in a config shared with a software DCF,
    and every multiple here has to travel with the sentence that justifies it, so
    the plan is its own document that can be circulated and argued with.
    """
    if not path.exists():
        raise ConfigError(f"plan file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} should be a mapping with a 'segments' key. See --emit-plan."
        )
    block = raw.get("segments")
    if not isinstance(block, dict) or not block:
        raise ConfigError(
            f"{path} carries no 'segments' mapping. Every segment needs a metric, "
            "a multiple and a source. Run with --emit-plan to write a skeleton "
            "carrying this filer's own segment names."
        )
    multiples: dict[str, tuple[str, float, str]] = {}
    for name, entry in block.items():
        if not isinstance(entry, dict):
            raise ConfigError(
                f"segment {name!r} in {path} should be a mapping with metric, "
                "multiple and source."
            )
        missing = [k for k in ("metric", "multiple", "source") if k not in entry]
        if missing:
            raise ConfigError(
                f"segment {name!r} in {path} is missing: {', '.join(missing)}. A "
                "bare multiple is a number, not an argument: state which peer set "
                "and which statistic it came from."
            )
        multiples[str(name)] = (
            str(entry["metric"]),
            float(entry["multiple"]),
            str(entry["source"]),
        )
    corporate = raw.get("corporate") or {}
    if not isinstance(corporate, dict):
        raise ConfigError(f"'corporate' in {path} should be a mapping.")

    def number(key: str, block: dict, default=None):
        value = block.get(key, default)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{key!r} in {path} is {value!r}, which is not a number."
            ) from exc

    return {
        "multiples": multiples,
        "corporate_cost": number("cost", corporate),
        "corporate_multiple": number("multiple", corporate),
        "corporate_multiple_source": corporate.get("source"),
        "conglomerate_discount": number("conglomerate_discount", raw, 0.0) or 0.0,
    }


def plan_skeleton(report) -> str:
    """A starter plan carrying this filer's own segment names.

    Nobody can guess the names: they are whatever the filer tagged on the segment
    axis, and a plan whose keys do not match is refused segment by segment. The
    multiples are left at a placeholder and the sources are left empty on purpose,
    because a skeleton that shipped a default multiple would be a default nobody
    chose showing up in a valuation.
    """
    lines = [
        "# Sum-of-the-parts plan. Every segment needs a metric, a multiple and the",
        "# sentence that justifies the multiple. 'metric' is revenue, ebitda or ebit.",
        "# The source is not optional: a multiple without a peer set behind it is a",
        "# number, not a view, and it is untraceable the moment this file leaves the",
        "# desk it was built on.",
        "segments:",
    ]
    for s in report.segments:
        lines.append(f"  {s.name}:")
        lines.append("    metric: ebitda")
        lines.append("    multiple: 0.0   # replace")
        lines.append('    source: ""      # which peers, which statistic, which date')
    lines += [
        "",
        "# Unallocated corporate expense as a POSITIVE annual cost, in USD millions,",
        "# from the corporate and other line of the segment footnote. Leaving it out",
        "# values the holding company's cost of existing at zero and overstates the",
        "# total. Leave multiple and source null to capitalise it at the",
        "# enterprise-value-weighted average of the segment earnings multiples.",
        "corporate:",
        "  cost: null",
        "  multiple: null",
        "  source: null",
        "",
        "# A decimal rate, zero by default, applied after the corporate deduction.",
        "# The literature finds 5 to 15 percent; against that, a discount assumed",
        "# rather than observed can be tuned until the answer agrees with whatever",
        "# was wanted.",
        "conglomerate_discount: 0.0",
        "",
    ]
    return "\n".join(lines)


def _annual_filed(client, ticker: str, accession: str | None) -> date | None:
    """The date the annual report behind the segment table reached EDGAR.

    Returned so the FLAG below can name a knowledge date that works instead of
    describing one. ``None`` when the submissions feed cannot be read or does not
    carry the accession, in which case the FLAG falls back to describing the
    window, which is still true and still better than the date it used to print.
    """
    if not accession:
        return None
    try:
        for filing in client.filings(ticker, forms=("10-K",), limit=20):
            if filing.get("accession") == accession:
                filed = filing.get("filed")
                return filed if isinstance(filed, date) else None
    except TechvalError:
        return None
    return None


def _period_gap_flag(client, ticker: str, report, fin) -> str:
    """The two windows, and the pin that actually closes them.

    This sentence used to end "pin the whole run to the annual report with
    --as-of {report.as_of}", and that instruction makes the problem worse rather
    than better. The last day of a fiscal year is weeks or months before the
    annual report covering it reaches EDGAR, so pinning the knowledge date there
    puts the run BEHIND that filing: the engine falls back to the prior year's
    segment footnote and the two windows are a period apart again, one year
    earlier. Measured on Disney, whose FY2025 10-K was filed 2025-11-13:
    ``sotp DIS --as-of 2025-09-27`` returns segments for the year ended
    2024-09-28, and that footnote carries no depreciation and amortisation for
    Experiences, so the run then fails with a different ``MissingDataError`` and
    a reader who followed the instruction is further from an answer than when
    they started. The knowledge date that works is the day after the annual
    report was filed, where the 10-K is the newest periodic filing on record and
    the parts and the whole are read out of the same document.

    The integration agent corrected the copy of this sentence in ``cli.py``
    during the same wave. This one is the copy that was missed.
    """
    filed = _annual_filed(client, ticker, report.accession)
    head = (
        f"FLAG: the segments cover the year ended {report.as_of} and the "
        f"consolidated figures cover the twelve months to {fin.as_of}, because "
        f"{ticker.upper()} has filed at least one quarter since its annual "
        "report. Segment detail is annual, so the two cannot be brought together "
        "by reading more filings. Either accept that the parts are measured a "
        "period behind the whole, or pin the knowledge date so that the annual "
        "report is the newest filing on record."
    )
    if filed is not None:
        pin = filed + timedelta(days=1)
        return (
            head
            + f" That date is {pin.isoformat()}, the day after the {report.as_of} "
            f"annual report reached EDGAR on {filed.isoformat()}: run with --as-of "
            f"{pin.isoformat()}, which also prices the company at that date's "
            f"close. Do NOT pin to {report.as_of.isoformat()}. The annual report "
            "did not exist on the last day of the year it covers, so that date "
            "reads the PRIOR year's segment note and the two windows are a period "
            "apart again."
        )
    return (
        head
        + " That is a date shortly after the annual report was filed, which is a "
        f"month or two after the year end rather than the year end itself. Do NOT "
        f"pin to {report.as_of.isoformat()}: the annual report did not exist on "
        "the last day of the year it covers, so that date reads the PRIOR year's "
        "segment note and the two windows are a period apart again. The "
        "submissions feed did not give this run the filing date, so the exact day "
        "has to come from the filing index."
    )


def _render_sotp(result, fin, report) -> None:
    _rule("The parts")
    t = Table(box=None, pad_edge=False)
    t.add_column("Segment", overflow="fold")
    t.add_column("Metric", no_wrap=True)
    t.add_column("Value", justify="right")
    t.add_column("Multiple", justify="right")
    t.add_column("Basis", no_wrap=True)
    t.add_column("Enterprise value", justify="right")
    t.add_column("% of parts", justify="right")
    for s in result.segments:
        t.add_row(
            s.name,
            s.metric_name,
            _money(s.metric_value),
            f"{s.multiple:,.1f}x",
            s.multiple_label,
            Text(_money(s.enterprise_value), style="bold"),
            _pct(s.share_of_total),
        )
    console.print(t)

    _rule("Where each multiple came from")
    t = Table(box=None, pad_edge=False)
    t.add_column("Segment", no_wrap=True)
    t.add_column("Applied", justify="right", no_wrap=True)
    t.add_column("Source", overflow="fold", style="dim")
    for s in result.segments:
        t.add_row(s.name, f"{s.multiple:,.1f}x {s.multiple_label}", s.multiple_source)
    if result.corporate_multiple is not None:
        t.add_row(
            Text("Corporate cost", style="yellow"),
            Text(f"{result.corporate_multiple:,.1f}x", style="yellow"),
            Text(str(result.corporate_multiple_source), style="yellow"),
        )
    console.print(t)

    _rule("The holding company")
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", justify="right")
    t.add_column("", style="dim", overflow="fold")
    if result.corporate_cost is None:
        t.add_row(
            Text("Unallocated corporate cost", style="yellow"),
            Text("not supplied", style="yellow"),
            "segment operating income is struck before it, so the total below "
            "values the cost of running the holding company at zero",
        )
    else:
        t.add_row(
            "Annual unallocated corporate cost",
            _money(result.corporate_cost),
            "as supplied in the plan, as a positive cost. Segment operating "
            "income is struck before it, which is why it is a deduction here "
            "rather than already inside the parts",
        )
        t.add_row(
            "Capitalised at",
            f"{result.corporate_multiple:,.1f}x",
            str(result.corporate_multiple_source),
        )
        t.add_row(
            Text("Deducted from the parts", style="bold"),
            Text(_money(result.unallocated_corporate), style="bold"),
            "overhead is a perpetual charge in exactly the way segment earnings "
            "are a perpetual credit",
        )
        share = (
            abs(result.unallocated_corporate) / result.gross_enterprise_value
            if result.gross_enterprise_value
            else None
        )
        if share is not None:
            t.add_row("As a share of the parts", _pct(share), "")
    console.print(t)

    _rule("Bridge to equity value per share")
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", justify="right")
    for label, value in result.rows():
        low = label.lower()
        if low.startswith(("value per share", "share price", "gap per share")):
            shown = _dollars(value)
        elif low.startswith("gap, %"):
            shown = _pct(value)
        else:
            shown = _money(value)
        bold = low.startswith(
            (
                "gross value",
                "enterprise value, sum",
                "equity value",
                "value per share",
                "gap",
            )
        )
        t.add_row(Text(label, style="bold" if bold else ""),
                  Text(shown, style="bold" if bold else ""))
    console.print(t)

    gap = result.implied_vs_consolidated
    direction = "above" if gap.gap > 0 else "below"
    console.print(
        f"\n[dim]The parts come out {_money(abs(gap.gap))}mm {direction} what the "
        "market pays for the whole. That gap is the break-up argument and it is "
        "also the first place to look for an error: a wide gap is usually a wrong "
        "multiple or a missing corporate line long before it is a mispricing. The "
        "sum of the parts routinely exceeds market value across the whole "
        "conglomerate literature, and the excess is not automatically an "
        "opportunity: the segments are valued as if each would trade at its own "
        "peers' multiple standing alone, which is the break-up case rather than "
        "the status quo, and a break-up has tax leakage, stranded corporate cost "
        "and execution risk that none of these multiples carry.[/dim]"
    )
    if result.conglomerate_discount_rate == 0.0:
        console.print(
            f"[dim]No conglomerate discount was applied. Lang and Stulz and Berger "
            "and Ofek find diversified firms trading 5 to 15 percent below the sum "
            "of imputed segment values; against that, a discount assumed rather "
            "than observed can be set to make any sum of the parts agree with any "
            "price. Set conglomerate_discount in the plan to apply one "
            "deliberately.[/dim]"
        )

    console.print(
        f"\n[dim]Segments read from {report.accession or 'the filing'} for the year "
        f"ended {report.as_of}; the bridge to equity uses the trailing twelve "
        f"months to {fin.as_of}. Where those two periods differ, the parts and the "
        "whole are measured over different windows.[/dim]"
    )
    _notes(result.checks, heading="Cross-checks")
    _notes(result.flags, heading="Flags")
    _notes(result.notes)


@app.command()
def sotp(
    ticker: str,
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    as_of: str = _ASOF,
    plan: Path = typer.Option(
        None,
        "--plan",
        help="YAML naming the multiple, the metric and the source for every segment.",
    ),
    emit_plan: Path = typer.Option(
        None,
        "--emit-plan",
        help="Write a skeleton plan carrying this filer's own segment names, and stop.",
    ),
    discount: float = typer.Option(
        None,
        "--discount",
        help="Conglomerate discount as a decimal, overriding the plan. Default zero.",
    ),
    on_negative: str = typer.Option(
        "raise",
        "--on-negative",
        help=(
            "What to do with a loss-making segment given an earnings multiple: "
            "raise (default) or nil."
        ),
    ),
) -> None:
    """Value each reported segment on its own multiple and reconcile to the market.

    This is the command with the most room to mislead, so it prints the most.
    Every multiple travels with the sentence that justifies it, the corporate
    cost is shown with the multiple it was capitalised at, and the walk from the
    sum of the parts down to a price per share is printed line by line. A sum of
    the parts that does not show its corporate cost allocation is an argument for
    whatever answer the author wanted.
    """
    try:
        assumptions, client, market = _setup(config, no_cache, as_of)
        report = build_segments(ticker, client, assumptions)

        if emit_plan is not None:
            emit_plan.write_text(plan_skeleton(report))
            console.print(
                f"Wrote a skeleton for {len(report.segments)} segment(s) to "
                f"[bold]{emit_plan}[/bold]. Fill in every multiple and every "
                "source, then rerun with --plan."
            )
            return

        if plan is None:
            console.print(
                "[red]No plan supplied.[/red] A sum of the parts needs a multiple "
                "and a stated source for every segment, and there is no default: a "
                "peer set for a cable network has no business living in a config "
                "shared with a software DCF."
            )
            console.print(
                f"\n{ticker.upper()} reports {len(report.segments)} segment(s): "
                + ", ".join(s.name for s in report.segments)
            )
            console.print(
                "\nRun with [bold]--emit-plan plan.yaml[/bold] to write a skeleton "
                "carrying those names, fill it in, then rerun with --plan."
            )
            raise typer.Exit(1)

        loaded = load_plan(plan)
        rate = loaded["conglomerate_discount"] if discount is None else discount

        fin = build_financials(ticker, client=client)
        bridge, price, price_note = _priced_bridge(ticker, fin, market, assumptions)
        if bridge is None:
            raise MissingDataError(
                "share price",
                ticker=ticker,
                hint=(
                    "a sum of the parts is only an argument against what the market "
                    f"pays for the whole, and that needs a price. {price_note}"
                ),
            )

        _rule(f"{fin.entity_name} ({fin.ticker})  sum of the parts, USD millions")
        console.print(
            f"[dim]Segments from {report.accession or 'the filing'}, year ended "
            f"{report.as_of}. Consolidated figures are the trailing twelve months "
            f"to {fin.as_of} at {price:,.2f} a share.[/dim]\n"
        )
        console.print(
            _segment_table(
                report.segments,
                title="As reported",
                residual_label="Unallocated / eliminations",
                consolidated=report.consolidated_revenue,
                residual=report.unallocated,
                geography=False,
            )
        )
        _render_reconciliation(report)

        # The parts and the whole are read over two different windows whenever a
        # quarter has been filed since the annual report, and the coverage check
        # inside run_sotp will then refuse on a gap that is growth rather than a
        # missing segment. Say so before it does, and name a fix that works.
        if report.as_of != fin.as_of:
            console.print(f"\n[yellow]{_period_gap_flag(client, ticker, report, fin)}[/yellow]")

        result = run_sotp(
            fin,
            bridge,
            report.segments,
            loaded["multiples"],
            assumptions,
            corporate_cost=loaded["corporate_cost"],
            corporate_multiple=loaded["corporate_multiple"],
            corporate_multiple_source=loaded["corporate_multiple_source"],
            conglomerate_discount=float(rate),
            on_negative="nil" if str(on_negative).lower() == "nil" else "raise",
        )
        _render_sotp(result, fin, report)
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
