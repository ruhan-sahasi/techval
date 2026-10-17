"""The record case: a DCF set from what the filings show rather than from the assumptions file.

The assumptions file carries one set of DCF defaults, and they describe a
subscription software company: growth fading from 20% to 5%, a 20% terminal
margin, capex at 3% of revenue. Applied to a cable operator they value it at
twenty times its price, because Charter spends close to a fifth of its revenue
on capex and has not grown in years. That is not a bug in the engine; it is the
assumptions file being asked about a business it was not written for.

``record_anchors`` reads the alternative off the filings: revenue growth from
the company's own trailing growth, fading toward a rate no higher than that and
no lower than the perpetuity rate; today's EBIT margin, held; capex and D&A at
today's share of revenue. Each anchor says what the assumptions file had and
where the new figure came from, and an anchor the filings cannot support is
left at the assumption and said to be. The result is the company as it stands,
with nothing improving and nothing getting worse: a baseline to set the
assumption-driven case against, not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from . import tags
from .errors import TechvalError

# Growth read off the record is held inside this band. A trailing rate outside it
# is an acquisition, a disposal or a recovery from a collapse, none of which a
# company repeats at the same rate.
GROWTH_FLOOR = -0.30
GROWTH_CEILING = 0.60

# A fiscal year, by the length of its reporting window.
_ANNUAL_DAYS = (330, 400)


@dataclass(frozen=True)
class Anchor:
    """One DCF assumption set from the record, beside what the assumptions file had."""

    field: str
    value: float
    assumed: float | None
    source: str


def _a_year_before(day: date) -> date:
    try:
        return day.replace(year=day.year - 1)
    except ValueError:  # 29 February
        return day.replace(year=day.year - 1, day=28)


def trailing_growth(fin, facts) -> tuple[float, str] | None:
    """Revenue growth over the last year, and how it was measured, or None.

    The trailing twelve months against the twelve months a year earlier where
    both can be tiled; otherwise the latest fiscal year against the one before.
    """
    try:
        prior, _ = facts.resolve_ttm("revenue", tags.REVENUE, _a_year_before(fin.as_of))
    except TechvalError:
        prior = None
    if prior:
        prior_mm = prior / 1e6
        return fin.revenue / prior_mm - 1.0, (
            f"revenue of {fin.revenue:,.0f}mm over the twelve months to {fin.as_of} "
            f"against {prior_mm:,.0f}mm a year earlier"
        )
    years: dict[date, float] = {}
    for tag in tags.REVENUE:
        for fact in facts.facts(tag):
            if fact.is_instant or not (_ANNUAL_DAYS[0] <= fact.days <= _ANNUAL_DAYS[1]):
                continue
            if fact.end <= fin.as_of:
                years.setdefault(fact.end, fact.val)
    ends = sorted(years)
    if len(ends) < 2 or (ends[-1] - ends[-2]).days > _ANNUAL_DAYS[1]:
        return None
    latest, before = years[ends[-1]], years[ends[-2]]
    if before <= 0:
        return None
    return latest / before - 1.0, (
        f"fiscal-year revenue of {latest / 1e6:,.0f}mm to {ends[-1]} against "
        f"{before / 1e6:,.0f}mm the year before"
    )


def record_anchors(fin, facts, assumptions) -> list[Anchor]:
    """The DCF assumptions the filings support, each with its source."""
    cfg = assumptions.dcf
    out: list[Anchor] = []
    measured = trailing_growth(fin, facts)
    if measured is not None:
        raw, how = measured
        start = min(max(raw, GROWTH_FLOOR), GROWTH_CEILING)
        if start != raw:
            how += f", {raw:.1%} held to {start:.0%} since no company repeats that"
        out.append(Anchor("revenue_growth_start", start, cfg.revenue_growth_start, how))
        terminal = max(cfg.terminal_growth, min(start, cfg.revenue_growth_terminal))
        out.append(
            Anchor(
                "revenue_growth_terminal",
                terminal,
                cfg.revenue_growth_terminal,
                "the lower of the trailing rate and the assumed year-N rate, and no "
                "lower than the perpetuity rate",
            )
        )
    out.append(
        Anchor(
            "ebit_margin_terminal",
            fin.ebit_margin,
            cfg.ebit_margin_terminal,
            f"today's EBIT margin, {fin.ebit:,.0f}mm on {fin.revenue:,.0f}mm of revenue, held",
        )
    )
    if fin.capex is not None and fin.revenue:
        out.append(
            Anchor(
                "capex_pct_revenue",
                fin.capex / fin.revenue,
                cfg.capex_pct_revenue,
                f"trailing capex of {fin.capex:,.0f}mm on {fin.revenue:,.0f}mm of revenue",
            )
        )
    if fin.da is not None and fin.revenue:
        out.append(
            Anchor(
                "da_pct_revenue",
                fin.da / fin.revenue,
                cfg.da_pct_revenue,
                f"trailing D&A of {fin.da:,.0f}mm on {fin.revenue:,.0f}mm of revenue",
            )
        )
    return out


def unanchored(anchors: list[Anchor]) -> list[str]:
    """The record-case fields the filings could not set, left at the assumption."""
    wanted = ("revenue_growth_start", "ebit_margin_terminal", "capex_pct_revenue", "da_pct_revenue")
    have = {a.field for a in anchors}
    return [f for f in wanted if f not in have]


def record_case(assumptions, fin, facts):
    """The assumptions with the record anchors applied, and the anchors; unchanged when the switch is off.

    ``dcf.from_record`` is the switch. With it off this returns the assumptions
    as given and no anchors, so every caller can route through it.
    """
    if not assumptions.dcf.from_record:
        return assumptions, []
    anchors = record_anchors(fin, facts, assumptions)
    dcf = assumptions.dcf.model_copy(update={a.field: a.value for a in anchors})
    return assumptions.model_copy(update={"dcf": dcf}), anchors
