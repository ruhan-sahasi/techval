"""What the sample can support: the count beside each result against the count that limits it.

Every model section on the page quotes its score against a count in the
thousands, and none of those counts is the sample size. This section reads the
counts the model sections already published and sets them side by side, so a
reader can see what methodology section 16 says in words: roughly a hundred
companies over ten to fifteen years is a few dozen independent observations,
however many rows they fill.

**It reads, it does not compute.** Everything here comes from ``ctx.results``,
the sections collected before this one. No model is refitted and no fixture is
opened, so ``INPUTS`` is empty and the provenance of each count is the section
that published it, named beside the count in the table view. The only
arithmetic is the kind a reader would do with the numbers in front of them: a
ratio of two published counts, and the years between two published dates.

**A count that is not published is refused, not recomputed.** Each reader names
the field it reads. If the field is absent, is not a number, or sits in a
section that was not collected, the count becomes a refusal whose reason names
that field, and the figure draws its row empty. Three numbers are published
only inside a sentence, and are read with patterns that must match the whole
clause: the value signal's company-dates, in the sub-line of its
effective-observations tile, where the number of dates the sentence names must
also equal the signal's headline ``n``; and the encoder's excerpt length and
empty-feature count, in a subtitle and a note. A reworded sentence refuses the
number rather than yielding whatever number happens to be in it.

The work is split so the page can be tested without a collection.
``read(results)`` returns plain values and the reasons for anything it could
not read, and ``shape(facts)`` turns those into figures, a takeaway and
refusals.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable, Mapping

ID = "sample"
TITLE = "What the sample can support"
INPUTS: list[str] = []

FIGURE_IDS = ("counts", "record", "limits")

# The function each figure's provenance names. Each reads published results and
# nothing else.
ENTRY_POINTS = {
    "counts": "techval.dashboard.sections.sample.count_ladders",
    "record": "techval.dashboard.sections.sample.record_depth",
    "limits": "techval.dashboard.sections.sample.limitations",
}

# How each model is named in a sentence, and its title on the chart when its
# section is absent and cannot supply its own.
PROSE_NAMES = {
    "signal": "the value signal",
    "encoder": "the peer encoder",
    "warranted": "the warranted multiple",
    "fade": "the revenue fade",
    "propensity": "the propensity screen",
}
FALLBACK_TITLES = {
    "signal": "The value signal",
    "encoder": "Peer encoder",
    "warranted": "Warranted multiple",
    "fade": "Revenue fade",
    "propensity": "M&A propensity",
}

MINUS = "\u2212"


# --------------------------------------------------------------------------- #
# reading published results
# --------------------------------------------------------------------------- #


class Unpublished(Exception):
    """A number this section needs is not in the results it was handed.

    ``field`` is the path of what was looked for, written the way the table view
    names a source, and ``reason`` says what was found there instead.
    """

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


@dataclass(frozen=True)
class Quote:
    """One number read from another section's results, with where it was read."""

    value: float
    source: str


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _section(results: Mapping[str, Any], sid: str, source: str) -> dict[str, Any]:
    section = results.get(sid)
    if not isinstance(section, dict):
        raise Unpublished(source, f"the snapshot holds no {sid} section")
    status = section.get("status")
    if status != "ok":
        state = {"refused": "refused", "not_built": "not collected"}.get(status, f"in state {status!r}")
        raise Unpublished(source, f"the {sid} section is {state}")
    return section


def _figure_data(results: Mapping[str, Any], sid: str, fid: str, source: str) -> dict[str, Any]:
    figure = (_section(results, sid, source).get("figures") or {}).get(fid)
    if not isinstance(figure, dict) or not isinstance(figure.get("data"), dict):
        raise Unpublished(source, f"the {sid} section has no {fid} figure")
    return figure["data"]


def _number(value: Any, source: str) -> Quote:
    if value is None:
        raise Unpublished(source, "the field is absent")
    if not _is_number(value):
        raise Unpublished(source, f"the field holds {value!r}, not a number")
    return Quote(value, source)


def headline_value(results: Mapping[str, Any], sid: str, key: str) -> Quote:
    source = f"{sid}.headline.{key}"
    headline = _section(results, sid, source).get("headline")
    if not isinstance(headline, dict):
        raise Unpublished(source, "the section carries no headline")
    return _number(headline.get(key), source)


def data_field(results: Mapping[str, Any], sid: str, fid: str, *keys: str) -> Quote:
    """A number at ``<sid>.figures.<fid>.data.<keys...>``."""
    source = f"{sid}.figures.{fid}.data." + ".".join(keys)
    node: Any = _figure_data(results, sid, fid, source)
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return _number(node, source)


def _tile(
    results: Mapping[str, Any], sid: str, fid: str, attr: str, *, label: str | None = None, key: str | None = None
) -> tuple[dict, str]:
    by, name = ("key", key) if key is not None else ("label", label)
    source = f'{sid}.figures.{fid}.data.tiles[{by}="{name}"].{attr}'
    tiles = _figure_data(results, sid, fid, source).get("tiles")
    matches = [t for t in (tiles if isinstance(tiles, list) else []) if isinstance(t, dict) and t.get(by) == name]
    if len(matches) != 1:
        raise Unpublished(source, f"{'no tile' if not matches else f'{len(matches)} tiles'} with that {by}")
    return matches[0], source


def tile_value(
    results: Mapping[str, Any], sid: str, fid: str, *, label: str | None = None, key: str | None = None
) -> Quote:
    tile, source = _tile(results, sid, fid, "value", label=label, key=key)
    return _number(tile.get("value"), source)


_INTEGER = r"(\d{1,3}(?:,\d{3})+|\d+)"


def _integer_text(text: str) -> int:
    return int(text.replace(",", ""))


# The value signal states its company-dates only in this sentence.
SIGNAL_EVIDENCE_SUB = re.compile(rf"From {_INTEGER} company-dates on {_INTEGER} dates")


def signal_company_dates(results: Mapping[str, Any]) -> Quote:
    tile, source = _tile(results, "signal", "evidence", "sub", label="Effective observations")
    sub = tile.get("sub")
    match = SIGNAL_EVIDENCE_SUB.fullmatch(sub) if isinstance(sub, str) else None
    if match is None:
        raise Unpublished(
            source,
            f'the sub-line reads {sub!r}, not "From <n> company-dates on <n> dates", '
            "and the signal publishes its company-dates nowhere else",
        )
    dates = _integer_text(match.group(2))
    headline = headline_value(results, "signal", "n")
    if dates != headline.value:
        raise Unpublished(
            source,
            f"the sub-line names {dates} dates where signal.headline.n is "
            f"{headline.value:g}, so it may not describe the result on the page",
        )
    return Quote(_integer_text(match.group(1)), source)


# --------------------------------------------------------------------------- #
# the facts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rung:
    """One count on a model's ladder: a quote, or the reason it could not be read."""

    key: str
    label: str
    unit: str
    step: str  # "nominal", "headline" or "limiting"
    quote: Quote | None
    missing: Unpublished | None = None


@dataclass(frozen=True)
class Ladder:
    """One model's counts, its largest first and the one that limits it last."""

    model: str
    title: str
    why: str
    rungs: tuple[Rung, ...]

    def rung(self, step: str) -> Rung:
        return next(r for r in self.rungs if r.step == step)

    @property
    def ratio(self) -> float | None:
        nominal, limiting = self.rung("nominal").quote, self.rung("limiting").quote
        return nominal.value / limiting.value if nominal and limiting else None


@dataclass(frozen=True)
class Span:
    """One row of the record's depth, in years, or the reason it could not be read."""

    key: str
    label: str
    years: float | None
    detail: str
    sources: tuple[str, ...]
    missing: Unpublished | None = None


@dataclass(frozen=True)
class Limit:
    """One limit in words, with every number the words quote."""

    what: str
    why: str
    quotes: tuple[tuple[str, Quote, str], ...] = ()  # (what the number is, the quote, its format)


@dataclass(frozen=True)
class Facts:
    ladders: tuple[Ladder, ...]
    spans: tuple[Span, ...]
    limits: tuple[Limit, ...]


def _rung(key: str, label: str, unit: str, step: str, reader: Callable[[], Quote]) -> Rung:
    try:
        quote = reader()
    except Unpublished as exc:
        return Rung(key, label, unit, step, None, exc)
    if quote.value <= 0:
        reason = f"the count is {quote.value:g}, which a log scale cannot place"
        return Rung(key, label, unit, step, None, Unpublished(quote.source, reason))
    return Rung(key, label, unit, step, quote)


def _title(results: Mapping[str, Any], sid: str) -> str:
    section = results.get(sid)
    title = section.get("title") if isinstance(section, dict) else None
    return title if isinstance(title, str) and title.strip() else FALLBACK_TITLES[sid]


def count_ladders(results: Mapping[str, Any]) -> tuple[Ladder, ...]:
    """Each model's largest count, the count its headline is quoted on, and the one that limits it.

    Which count limits a model is a judgment, made here and stated in ``why``.
    The limiting count is not chosen for being the smallest on its ladder but
    for being the unit that varies independently: rows that share a company, a
    committee's list or an overlapping return window do not. The models are in
    page order.
    """
    r = results
    return (
        Ladder(
            "signal",
            _title(r, "signal"),
            "The return windows of neighbouring rebalance dates overlap, so the dates are "
            "not independent either; the Newey-West count is what is left once the overlap "
            "is allowed for",
            (
                _rung("company_dates", "Company-dates scored", "company-dates", "nominal",
                      lambda: signal_company_dates(r)),
                _rung("dates", "Rebalance dates, the headline's n", "rebalance dates", "headline",
                      lambda: headline_value(r, "signal", "n")),
                _rung("effective", "Effective observations after the overlap correction",
                      "effective observations", "limiting",
                      lambda: tile_value(r, "signal", "evidence", label="Effective observations")),
            ),
        ),
        Ladder(
            "encoder",
            _title(r, "encoder"),
            "Every pair a filer contributes comes from one committee's list, so the pairs and "
            "queries of one filer are not independent of each other",
            (
                _rung("pairs", "Training pairs", "training pairs", "nominal",
                      lambda: data_field(r, "datalayer", "peer_label_counts", "counts", "pairs")),
                _rung("queries", "Test queries, the headline's n", "test queries", "headline",
                      lambda: headline_value(r, "encoder", "n")),
                _rung("filers", "Filers whose peer groups became pairs", "filers", "limiting",
                      lambda: data_field(r, "datalayer", "peer_label_counts", "counts", "filers_kept")),
            ),
        ),
        Ladder(
            "warranted",
            _title(r, "warranted"),
            "A company's features and its relative multiple barely move in a quarter, so its "
            "consecutive quarters are close to one observation repeated",
            (
                _rung("company_quarters", "Company-quarters in the panel", "company-quarters", "nominal",
                      lambda: tile_value(r, "warranted", "panel", key="observations")),
                _rung("scored", "Company-quarters scored, the headline's n", "company-quarters scored",
                      "headline", lambda: headline_value(r, "warranted", "n")),
                _rung("companies", "Companies in the panel", "companies", "limiting",
                      lambda: tile_value(r, "warranted", "panel", key="companies")),
            ),
        ),
        Ladder(
            "fade",
            _title(r, "fade"),
            "Consecutive years of one filer share its growth path, so the filer is the unit "
            "that varies",
            (
                _rung("company_years", "Company-years in the panel", "company-years", "nominal",
                      lambda: data_field(r, "fade", "depth", "stats", "observations")),
                _rung("scored", "Company-years scored at one year, the headline's n",
                      "company-years scored", "headline", lambda: headline_value(r, "fade", "n")),
                _rung("filers", "Filers with revenue rows", "filers", "limiting",
                      lambda: data_field(r, "fade", "depth", "stats", "filers_with_rows")),
            ),
        ),
        Ladder(
            "propensity",
            _title(r, "propensity"),
            "An AUC ranks the acquired rows against the rest, and each deal labels several "
            "quarters of its target, so the targets are the unit that varies",
            (
                _rung("labelled", "Labelled company-quarters", "labelled company-quarters", "nominal",
                      lambda: data_field(r, "propensity", "sample", "counts", "n_labelled")),
                _rung("scored", "Rows scored out of sample, the headline's n", "rows scored", "headline",
                      lambda: data_field(r, "propensity", "sample", "counts", "n_scored")),
                # scored_deals counts distinct targets, not announcements.
                _rung("targets", "Targets behind the AUC", "targets", "limiting",
                      lambda: data_field(r, "propensity", "sample", "counts", "scored_deals")),
            ),
        ),
    )


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _month_index(value: Any) -> int | None:
    match = re.fullmatch(r"(\d{4})-(\d{2})", value) if isinstance(value, str) else None
    if match is None or not 1 <= int(match.group(2)) <= 12:
        return None
    return int(match.group(1)) * 12 + int(match.group(2)) - 1


CADENCE = {1: "monthly", 3: "quarterly", 6: "half-yearly", 12: "annual"}


def _closes_span(results: Mapping[str, Any]) -> Span:
    label = "Nvidia's closes, first week to last"
    source = "datalayer.figures.nvda_closes.data.series[0].values[*].x"
    try:
        series = _figure_data(results, "datalayer", "nvda_closes", source).get("series")
        first_series = series[0] if isinstance(series, list) and series and isinstance(series[0], dict) else {}
        values = first_series.get("values")
        ends = [v.get("x") if isinstance(v, dict) else None for v in (values[0], values[-1])] if values else [None, None]
        first, last = _day(ends[0]), _day(ends[1])
        if first is None or last is None or last <= first:
            raise Unpublished(source, "the series has no first and last close with ISO dates in order")
        days = data_field(results, "datalayer", "nvda_closes", "counts", "days")
    except Unpublished as exc:
        return Span("closes", label, None, "", (exc.field,), exc)
    return Span(
        "closes",
        label,
        (last - first).days / 365.25,
        f"Weekly, {first.isoformat()} to {last.isoformat()}, drawn from {_count(days.value)} "
        "trading days in the signal's closes",
        (source, days.source),
    )


def _rebalance_span(results: Mapping[str, Any]) -> Span:
    label = "The value signal's rebalance dates, first to last"
    source = "signal.figures.ic_by_date.data.rows[*].label"
    try:
        rows = _figure_data(results, "signal", "ic_by_date", source).get("rows")
        labels = [row.get("label") if isinstance(row, dict) else None for row in (rows if isinstance(rows, list) else [])]
        months = [_month_index(x) for x in labels]
        if len(months) < 2 or None in months or months != sorted(set(months)):
            raise Unpublished(source, "the IC series has no run of YYYY-MM dates in order")
    except Unpublished as exc:
        return Span("rebalance", label, None, "", (exc.field,), exc)
    gaps = {b - a for a, b in zip(months, months[1:])}
    cadence = CADENCE.get(next(iter(gaps)), "") if len(gaps) == 1 else ""
    return Span(
        "rebalance",
        label,
        (months[-1] - months[0]) / 12,
        f"{len(months)} {cadence + ' ' if cadence else ''}dates, {labels[0]} to {labels[-1]}",
        (source,),
    )


def _median_span(results: Mapping[str, Any]) -> Span:
    label = "Fiscal years of revenue, median filer"
    try:
        median = data_field(results, "fade", "depth", "stats", "median_years")
        filers = data_field(results, "fade", "depth", "stats", "filers_with_rows")
        first = data_field(results, "fade", "depth", "stats", "first_filed")
        last = data_field(results, "fade", "depth", "stats", "last_filed")
    except Unpublished as exc:
        return Span("median_filer", label, None, "", (exc.field,), exc)
    return Span(
        "median_filer",
        label,
        median.value,
        f"Across {_count(filers.value)} filers with revenue rows, filed {int(first.value)} to {int(last.value)}",
        (median.source, filers.source, first.source, last.source),
    )


def _deepest_span(results: Mapping[str, Any]) -> Span:
    label = "Fiscal years of revenue, deepest filers"
    try:
        deepest = data_field(results, "fade", "depth", "stats", "max_years")
        at_max = data_field(results, "fade", "depth", "stats", "at_max")
        filers = data_field(results, "fade", "depth", "stats", "filers_with_rows")
    except Unpublished as exc:
        return Span("deepest_filers", label, None, "", (exc.field,), exc)
    return Span(
        "deepest_filers",
        label,
        deepest.value,
        f"{_count(at_max.value)} of the {_count(filers.value)} filers reach it",
        (deepest.source, at_max.source, filers.source),
    )


def record_depth(results: Mapping[str, Any]) -> tuple[Span, ...]:
    """How far back the record reaches, in years: closes, rebalance dates and filings."""
    return (_closes_span(results), _rebalance_span(results), _median_span(results), _deepest_span(results))


def _optional(reader: Callable[[], Quote]) -> Quote | None:
    try:
        return reader()
    except Unpublished:
        return None


def _has_refusal(results: Mapping[str, Any], sid: str, what: str) -> bool:
    section = results.get(sid)
    refusals = section.get("refusals") if isinstance(section, dict) else None
    return any(isinstance(x, dict) and x.get("what") == what for x in (refusals or []))


EXCERPT_SUBTITLE = re.compile(rf"\bwhich stop at {_INTEGER} characters\b")
EMPTY_FEATURES_NOTE = re.compile(rf"^{_INTEGER} of the panel's {_INTEGER} features are empty on every one of its ")


def _stated_integer(
    results: Mapping[str, Any], sid: str, fid: str, where: str, pattern: re.Pattern, group: int
) -> Quote:
    """An integer stated in a figure's subtitle or its data note."""
    source = f"{sid}.figures.{fid}.{'subtitle' if where == 'subtitle' else 'data.note'}"
    figure = (_section(results, sid, source).get("figures") or {}).get(fid)
    text: Any = None
    if isinstance(figure, dict):
        text = figure.get("subtitle") if where == "subtitle" else (figure.get("data") or {}).get("note")
    match = pattern.search(text) if isinstance(text, str) else None
    if match is None:
        raise Unpublished(source, "the sentence that states this number is not there")
    return Quote(_integer_text(match.group(group)), source)


def _ablation(results: Mapping[str, Any], tower: str, column: str) -> Quote:
    source = f'encoder.figures.tower_ablation.data.table.rows[cut_key="headline",tower="{tower}"].{column}'
    table = _figure_data(results, "encoder", "tower_ablation", source).get("table")
    rows = table.get("rows") if isinstance(table, dict) else None
    matches = [
        row
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and row.get("cut_key") == "headline" and row.get("tower") == tower
    ]
    if len(matches) != 1:
        raise Unpublished(source, "no single row for the headline cut without that tower")
    return _number(matches[0].get(column), source)


def limitations(results: Mapping[str, Any]) -> tuple[Limit, ...]:
    """Methodology section 16's limits, each quoting only numbers the results publish.

    The words follow the methodology. A limit whose number is not published is
    stated without the number rather than with one typed in.
    """
    r = results
    out: list[Limit] = []

    dates = _optional(lambda: headline_value(r, "signal", "n"))
    why = (
        "A hundred names inside one quarter share a sector, a rate cycle and a market, so they "
        "are not a hundred draws. The overlap correction deals with dependence through time"
        + (f", between the value signal's {_count(dates.value)} rebalance dates," if dates else ",")
        + " and does nothing about dependence across the names on one date, which nothing in "
        "the package corrects."
    )
    out.append(
        Limit(
            "Names on the same date are not independent draws",
            why,
            (("Signal rebalance dates", dates, "int"),) if dates else (),
        )
    )

    company_dates = _optional(lambda: signal_company_dates(r))
    why = (
        "The SEC's ticker file is today's list and the price sources drop a company the day it "
        "stops trading, so every model here is biased by the outcomes it cannot see."
    )
    quotes: tuple = ()
    if company_dates and _has_refusal(r, "signal", "Survivorship correction"):
        why += (
            " It is why the value signal refuses a survivorship correction across its "
            f"{_count(company_dates.value)} company-dates."
        )
        quotes = (("Signal company-dates", company_dates, "int"),)
    out.append(Limit("There is no point-in-time universe", why, quotes))

    chars = _optional(lambda: _stated_integer(r, "encoder", "truncation", "subtitle", EXCERPT_SUBTITLE, 1))
    score = _optional(lambda: headline_value(r, "encoder", "score"))
    quotes = ()
    if chars:
        why = f"The committed Item 1 text stops at {_count(chars.value)} characters"
        quotes += (("Characters kept of each Item 1", chars, "int"),)
    else:
        why = "The committed Item 1 text keeps only the opening of each description"
    why += ", roughly the first two pages of each business description"
    if score:
        why += f", and the encoder's {_fixed(score.value, 4)} NDCG@10 is measured on it"
        quotes += (("Encoder NDCG@10", score, "num:4"),)
    why += ". What it would score on the full documents cannot be reproduced from this repository."
    out.append(Limit("The text corpus is an excerpt", why, quotes))

    empty = _optional(lambda: _stated_integer(r, "encoder", "tower_ablation", "note", EMPTY_FEATURES_NOTE, 1))
    total = _optional(lambda: _stated_integer(r, "encoder", "tower_ablation", "note", EMPTY_FEATURES_NOTE, 2))
    damage = _optional(lambda: _ablation(r, "fundamentals", "damage"))
    spread = _optional(lambda: _ablation(r, "fundamentals", "fold_sd"))
    quotes = ()
    if empty and total:
        why = (
            f"{_count(empty.value)} of the peer panel's {_count(total.value)} features are empty "
            "on every row, the market features among them, so the fundamentals tower is judged "
            "on a degraded input."
        )
        quotes += (("Features empty on every row", empty, "int"), ("Features in the peer panel", total, "int"))
    else:
        why = "The peer panel carries no market features, so the fundamentals tower is judged on a degraded input."
    if damage and spread:
        why += (
            f" Taking that tower out moves the encoder's pair ranking by {_fixed(damage.value, 3)}, "
            f"inside a fold standard deviation of {_fixed(spread.value, 3)}, and that is not a "
            "verdict on fundamentals in general."
        )
        quotes += (
            ("Pair ranking lost without fundamentals", damage, "num:4"),
            ("Its fold standard deviation", spread, "num:4"),
        )
    else:
        why += " Its failure to clear the noise is not a verdict on fundamentals in general."
    out.append(Limit("There is no market feed in the peer panel", why, quotes))

    kept = _optional(lambda: data_field(r, "datalayer", "peer_label_counts", "counts", "groups_kept"))
    disclosed = _optional(lambda: data_field(r, "datalayer", "peer_label_counts", "counts", "groups_recorded"))
    why = (
        "A board committee picks a peer group partly for competition over executive talent, and "
        "the encoder inherits that choice"
    )
    quotes = ()
    if kept and disclosed:
        why += f" from the {_count(kept.value)} of {_count(disclosed.value)} disclosed groups it can use"
        quotes = (("Peer groups used", kept, "int"), ("Peer groups disclosed", disclosed, "int"))
    out.append(Limit("The labels are compensation peers, not trading comparables", why + ".", quotes))

    out.append(
        Limit(
            "A warranted multiple cannot say the market is wrong",
            "It is fitted on the market's own pricing, so a mispricing the whole sector shares is "
            "invisible to it by construction. Its residual says where a company sits, not that the "
            "gap will close.",
        )
    )
    out.append(
        Limit(
            "There is no total-return series",
            "The price layer carries closes, so a dividend payer's realised return is understated "
            "by its yield. The convention is stated here rather than assumed.",
        )
    )
    return tuple(out)


def read(results: Mapping[str, Any]) -> Facts:
    """Everything this section shows, read from the sections collected before it."""
    return Facts(ladders=count_ladders(results), spans=record_depth(results), limits=limitations(results))


# --------------------------------------------------------------------------- #
# words for numbers
# --------------------------------------------------------------------------- #


def _half_up(value: float, places: int = 0) -> Decimal:
    return Decimal(repr(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def _count(value: float) -> str:
    """A count as the page states it: to the unit, with commas, so 13.03 reads 13."""
    return f"{int(_half_up(value)):,}"


def _fixed(value: float, dp: int) -> str:
    body = f"{abs(value):,.{dp}f}"
    return (MINUS if value < 0 and any(c in body for c in "123456789") else "") + body


def _times(ratio: float) -> str:
    return _count(ratio) if ratio >= 10 else str(_half_up(ratio, 1))


_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _word(n: int) -> str:
    return _WORDS[n] if 0 <= n < len(_WORDS) else f"{n:,}"


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _span_of(values: list[float]) -> str:
    lo, hi = _count(min(values)), _count(max(values))
    return lo if lo == hi else f"{lo} to {hi}"


# --------------------------------------------------------------------------- #
# the section
# --------------------------------------------------------------------------- #


def _section_missing(ladder: Ladder) -> Unpublished | None:
    """The one reason for every missing count on a ladder, when it is that the section is absent."""
    missing = [r.missing for r in ladder.rungs]
    if not all(missing) or len({m.reason for m in missing}) != 1:
        return None
    reason = missing[0].reason
    return missing[0] if reason.startswith(("the snapshot holds no", f"the {ladder.model} section is")) else None


def _refusal_name(ladder: Ladder, rung: Rung | None) -> str:
    return f"{ladder.title}: {'every count' if rung is None else rung.unit}"


def _span_refusal_name(span: Span) -> str:
    return f"Depth of the record: {span.label}"


def _complete(ladders: tuple[Ladder, ...]) -> list[Ladder]:
    return [ladder for ladder in ladders if ladder.ratio is not None]


def _counts_figure(facts: Facts) -> dict[str, Any] | None:
    if not any(r.quote for ladder in facts.ladders for r in ladder.rungs):
        return None
    groups, rows, table = [], [], []
    for ladder in facts.ladders:
        whole = _section_missing(ladder)
        nominal = ladder.rung("nominal").quote
        groups.append(
            {
                "key": ladder.model,
                "label": ladder.title,
                "ratio": ladder.ratio,
                "ratioText": f"{_times(ladder.ratio)} times fewer" if ladder.ratio is not None else "",
                "why": ladder.why,
            }
        )
        for rung in ladder.rungs:
            value = rung.quote.value if rung.quote else None
            source = rung.quote.source if rung.quote else rung.missing.field
            rows.append(
                {
                    "key": f"{ladder.model}.{rung.key}",
                    "group": ladder.model,
                    "label": rung.label,
                    "unit": rung.unit,
                    "step": rung.step,
                    "role": "alt" if rung.step == "limiting" else "baseline",
                    "values": {"count": value},
                    "source": source,
                    "refusal": None if rung.quote else _refusal_name(ladder, None if whole else rung),
                }
            )
            table.append(
                {
                    "model": ladder.title,
                    "label": rung.label,
                    "count": value if rung.quote else "refused",
                    # Blank on the largest count itself, n/a where either count is missing.
                    "ratio": "" if rung.step == "nominal" else (nominal.value / value if nominal and value else None),
                    "source": source,
                }
            )

    complete = _complete(facts.ladders)
    if len(complete) > 1:
        title = (
            f"The counts that limit the {_word(len(complete))} models run from "
            f"{_span_of([x.rung('limiting').quote.value for x in complete])}, against largest "
            f"counts of {_span_of([x.rung('nominal').quote.value for x in complete])}"
        )
    elif complete:
        one = complete[0]
        title = (
            f"{one.title} counts {_count(one.rung('nominal').quote.value)} {one.rung('nominal').unit} "
            f"and is limited by {_count(one.rung('limiting').quote.value)} {one.rung('limiting').unit}"
        )
    else:
        title = "No model publishes both its largest count and the count that limits it"
    return {
        "kind": "dot",
        "title": title,
        "subtitle": (
            "For each model, its largest count, the count its headline is quoted on and the count "
            "that limits what the result can support, on a log scale. The unit changes from row "
            "to row, so every row names it, and every count is read from the section that "
            "published it."
        ),
        "data": {
            "scale": "log",
            "format": "num:0",
            "labelHeader": "Count of",
            "valueLabel": "Count",
            "series": [{"key": "count", "name": "Count", "role": "baseline"}],
            "legend": [
                {"label": "Largest count and the headline's n", "role": "baseline"},
                {"label": "The count that limits the result", "role": "alt"},
            ],
            "groups": groups,
            "rows": rows,
            "wide": True,
            "table": {
                "columns": [
                    {"key": "model", "label": "Model"},
                    {"key": "label", "label": "Count of"},
                    {"key": "count", "label": "Count", "align": "right", "format": "auto"},
                    {"key": "ratio", "label": "Largest count over this", "align": "right", "format": "num:1"},
                    {"key": "source", "label": "Read from", "mono": True},
                ],
                "rows": table,
            },
        },
    }


def _record_figure(facts: Facts) -> dict[str, Any] | None:
    drawn = [s for s in facts.spans if s.years is not None]
    if not drawn:
        return None
    spans = {s.key: s for s in drawn}
    parts = []
    if "closes" in spans:
        parts.append(f"Nvidia's closes reach back {_count(spans['closes'].years)} years")
    if "median_filer" in spans:
        parts.append(
            ("the median filer's" if parts else "The median filer's")
            + f" revenue {_count(spans['median_filer'].years)} fiscal years"
        )
    title = ", ".join(parts) if parts else f"{drawn[0].label} span {_half_up(drawn[0].years, 1)} years"
    return {
        "kind": "hbar",
        "title": title,
        "subtitle": (
            "Years between the first and last close and between the first and last rebalance "
            "date, and fiscal years of revenue per filer in the fade panel. The price record is "
            "measured on the one name the data layer draws from the signal's closes."
        ),
        "data": {
            "format": "num:1",
            "valueLabel": "Years",
            "labelHeader": "Record",
            "labels": "all",
            "wide": True,
            "height": 136,
            "rows": [{"key": s.key, "label": s.label, "value": s.years, "role": "total"} for s in drawn],
            # The refusals for spans that could not be read, by name, so the page
            # hangs them under this figure rather than at the foot of the section.
            "refusals": [_span_refusal_name(s) for s in facts.spans if s.missing is not None],
            "table": {
                "columns": [
                    {"key": "label", "label": "Record"},
                    {"key": "years", "label": "Years", "align": "right", "format": "auto"},
                    {"key": "detail", "label": "Detail"},
                    {"key": "source", "label": "Read from", "mono": True},
                ],
                "rows": [
                    {"label": s.label, "years": s.years, "detail": s.detail, "source": ", ".join(s.sources)}
                    for s in drawn
                ],
            },
        },
    }


def _limits_figure(facts: Facts) -> dict[str, Any]:
    items = facts.limits
    return {
        "kind": "list",
        "title": f"{_word(len(items)).capitalize()} limits that no amount of care downstream fixes",
        "subtitle": (
            "From methodology section 16. Every number in them is read from the section that "
            "published it, and a limit whose number is not published is stated without it."
        ),
        "data": {
            "items": [
                {
                    "what": item.what,
                    "why": item.why,
                    "quotes": [
                        {"label": label, "value": q.value, "format": fmt, "source": q.source}
                        for label, q, fmt in item.quotes
                    ],
                }
                for item in items
            ],
            "wide": True,
            "table": {
                "columns": [
                    {"key": "what", "label": "Limit"},
                    {"key": "label", "label": "Number quoted"},
                    {"key": "value", "label": "Value", "align": "right", "format": "auto"},
                    {"key": "source", "label": "Read from", "mono": True},
                ],
                "rows": [
                    {"what": item.what, "label": label, "value": q.value, "source": q.source}
                    for item in items
                    for label, q, _ in item.quotes
                ],
            },
        },
    }


def _takeaway(facts: Facts) -> str:
    complete = _complete(facts.ladders)
    if not complete:
        return ""
    nominal = [ladder.rung("nominal").quote.value for ladder in complete]
    ratios = [ladder.ratio for ladder in complete]
    clauses = [
        f"{_count(ladder.rung('limiting').quote.value)} {ladder.rung('limiting').unit} behind "
        f"{PROSE_NAMES[ladder.model]}'s {_count(ladder.rung('nominal').quote.value)} "
        f"{ladder.rung('nominal').unit}"
        for ladder in complete
    ]
    scale = "in thousands of rows" if min(nominal) >= 1000 else "in rows"
    if len(complete) > 1:
        lo, hi = _times(min(ratios)), _times(max(ratios))
        scarcer = f"{lo} times scarcer" if lo == hi else f"{lo} to {hi} times scarcer"
        lead = (
            f"The {_word(len(complete))} models count their evidence {scale}, and the unit that "
            f"limits each is {scarcer}: "
        )
    else:
        lead = (
            f"Only one model publishes both counts. It counts its evidence {scale}, and the unit "
            f"that limits it is {_times(ratios[0])} times scarcer: "
        )
    text = lead + _join(clauses) + "."
    spans = {s.key: s for s in facts.spans if s.years is not None}
    if "closes" in spans and "median_filer" in spans:
        text += (
            f" Under all of it, Nvidia's closes reach back {_count(spans['closes'].years)} years and "
            f"the median filer's revenue {_count(spans['median_filer'].years)} fiscal years."
        )
    return text


def _refusals(facts: Facts) -> list[dict[str, str]]:
    out = []
    for ladder in facts.ladders:
        whole = _section_missing(ladder)
        if whole is not None:
            out.append(
                {
                    "what": _refusal_name(ladder, None),
                    "why": f"None of its counts can be read, because {whole.reason}. Nothing is recomputed in their place.",
                }
            )
            continue
        for rung in ladder.rungs:
            if rung.missing is not None:
                out.append(
                    {
                        "what": _refusal_name(ladder, rung),
                        "why": (
                            f"Not read from {rung.missing.field}: {rung.missing.reason}. The count is "
                            "not recomputed here, so its row is left empty."
                        ),
                    }
                )
    for span in facts.spans:
        if span.missing is not None:
            out.append(
                {
                    "what": _span_refusal_name(span),
                    "why": (
                        f"Not read from {span.missing.field}: {span.missing.reason}. The span is not "
                        "recomputed from the fixtures, so it is not drawn."
                    ),
                }
            )
    return out


BUILDERS: dict[str, Callable[[Facts], dict[str, Any] | None]] = {
    "counts": _counts_figure,
    "record": _record_figure,
    "limits": _limits_figure,
}


def shape(facts: Facts) -> dict[str, Any]:
    """The section as ``collect`` returns it, from the facts alone."""
    refusals = _refusals(facts)
    if not any(r.quote for ladder in facts.ladders for r in ladder.rungs):
        return {
            "status": "refused",
            "takeaway": "",
            "refusals": [
                {
                    "what": TITLE,
                    "why": (
                        "No model section in this snapshot publishes a count this section can "
                        "read, so there is nothing to set side by side."
                    ),
                }
            ]
            + refusals,
            "headline": None,
            "figures": {},
        }
    figures = {}
    for fid in FIGURE_IDS:
        figure = BUILDERS[fid](facts)
        if figure is not None:
            figures[fid] = figure
    return {
        "status": "ok",
        "takeaway": _takeaway(facts),
        "refusals": refusals,
        "headline": None,
        "figures": figures,
    }


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #


def collect(ctx) -> dict:
    section = shape(read(ctx.results))
    # Each figure is recorded against the reader that produced its numbers. The
    # reads were done above and take milliseconds, so the blocks attach the rows
    # and time nothing. The inputs are empty: the fixtures behind every number
    # are on the provenance of the section that published it.
    for fid in section["figures"]:
        with ctx.record(fid, ENTRY_POINTS[fid], []):
            pass
    return section
