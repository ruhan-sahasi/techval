"""Warranted multiple: a better ranking than the incumbent, and how little of it is new.

The section makes one argument in two halves, and the page keeps them together.
The network ranks EV/Revenue multiples better than the strongest baseline built
from peers, which is the comps OLS refit on each date. And almost all of that
ranking is company identity inherited from history: differenced against each
company's own previous observation, the same rank correlation on the same rows
falls to a few hundredths.

Everything here comes from one fit of ``fit_warranted`` on the recorded panel,
loaded by the same loader ``techval screen`` uses, so the skips come back with
the observations and the refusal count on the page is the panel's own.

**One figure needs more than the model returns.** ``EvalResult`` keeps the
model's per-fold scores and not the baseline's, so the fold-by-fold dumbbell and
the stricter noise test cannot be read off the result. ``_rebuild`` recovers
them with the model's own private helpers (the leave-one-out date means, the
memoised comps OLS, the persistence carry) and hands the arrays back to
``evaluate_regression``. Nothing is reimplemented, and the rebuild is checked
before it is used: it must reproduce every baseline's pooled score, baseline
score and count, the card's fold scores, and the differenced rank correlation,
to 1e-9. A rebuild that drifts from the model is a ``ValueError``, because a
figure drawn from it would be a second model wearing the first one's name.

**The panel predates the debt-ladder fix, and the section says so first.** The
panel was recorded before the fix that stopped Verizon's long-term debt reading
as zero, so its enterprise values were bridged with the old ladders.
``ev_audit.json.gz`` rebuilds each of them with today's code, and the section
reads it rather than trusting the panel: the audit's counts open the section,
a caution under them says what they mean for the headline, and the screen is
refused when the audit shows its rich and cheap calls rest on enterprise values
now known to be wrong. Two things make that last call. A name on the screen
whose own observation is flagged refuses it directly. And because every
residual is measured against a model fitted on the whole panel, the section
also refits once with only the enterprise values rebuilt, and refuses a screen
whose names or sides that refit changes. That refit is partial, since the
features came from the same old ladder, and every sentence quoting it says so;
re-recording the panel is the fix, and it is not made here.

The part that turns those results into figures is ``shape``, a pure function of
an ``Inputs`` record, so the page can be tested on small fakes without a fit.
Each figure builder either returns a figure or raises ``FigureRefused`` with a
reason, and ``shape`` records provenance only for the figures that exist.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Callable, ContextManager

ID = "warranted"
TITLE = "Warranted multiple"
PANEL = "warranted/observations.json.gz"
AUDIT = "warranted/ev_audit.json.gz"
INPUTS: list[str] = [PANEL, AUDIT]

# Both the recorded and the rebuilt enterprise value are rounded to a thousandth
# of a million, so two figures within twice that step are the same figure.
IDENTICAL_WITHIN_MM = 0.002

# The audit's two thresholds: a gap worth counting, and a gap that makes an
# observation stale for the screen.
GAP_NOTED = 0.01
GAP_STALE = 0.05

# Names shown at each end of the screen.
SCREEN_EACH_END = 8

# How close the rebuild must come to the model's own numbers.
REBUILD_TOLERANCE = 1e-9

# The takeaway says "almost all of the ranking is company identity" when the
# differenced rank correlation is below this, and softens the sentence above it.
# It is a wording threshold for one sentence, not a statistical test.
IDENTITY_WORDING_THRESHOLD = 0.1

ENTRY_FIT = "techval.ml.warranted.fit_warranted"
ENTRY_EVALUATE = "techval.ml.evaluation.evaluate_regression"
ENTRY_EXTREMES = "techval.ml.warranted.WarrantedModel.extremes"
ENTRY_LOAD = "techval.commands_peers._load_observations"
ENTRY_AUDIT = "techval.dashboard.sections.warranted.audit_counts"
ENTRY_FILERS = "techval.dashboard.sections.warranted.affected_filers"

# Short row labels for the model's baseline names. A name not listed here is
# shown as the model wrote it, up to its first parenthesis.
SHORT_BASELINE_NAMES = {
    "comps.py OLS refit on the date, sub-vertical peers": "Comps OLS on sub-vertical peers",
    "comps.py OLS refit on the date, whole TMT cross-section": "Comps OLS on the whole TMT cross-section",
    "sub-vertical median multiple on the date (the comps convention)": "Sub-vertical median multiple",
    (
        "the company's own multiple last quarter (persistence: NOT a warranted "
        "multiple, it reads the answer)"
    ): "Last quarter's own multiple",
}

METRIC_NAMES = {"spearman": "Spearman rank correlation"}

MODEL_NAMES = {"mlp": "network", "ridge": "ridge"}

# Why a baseline stronger than the card's is left off the card, for the baselines
# the model is known to exclude. The exclusion itself is derived from the scores.
EXCLUSION_REASONS = {
    (
        "the company's own multiple last quarter (persistence: NOT a warranted "
        "multiple, it reads the answer)"
    ): (
        "it is the company's own multiple carried forward, so it reads the price a "
        "warranted multiple exists to be compared against."
    ),
}

# The deflation figure draws the pooled score in a tint of the model's colour and
# the differenced score in the full colour: both are the model, one is the point.
MUTED_MODEL_ROLE = "model-muted"


class FigureRefused(Exception):
    """A figure that cannot be drawn from what the fit returned, and why.

    Deliberately not a ``TechvalError``: raised inside one figure's
    ``ctx.record`` block it removes that figure and nothing else, and one that
    escaped ``shape`` would be a bug to see, not a section to refuse.
    """

    def __init__(self, what: str, why: str) -> None:
        super().__init__(why)
        self.what = what
        self.why = why


# --------------------------------------------------------------------------- #
# What the shaping reads
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Comparison:
    """The model against one baseline, on the observations both could score."""

    name: str
    model_score: float
    baseline_score: float
    n: int
    on_card: bool


@dataclass(frozen=True)
class FoldResult:
    """One walk-forward test block, the model against the card's baseline."""

    index: int
    test_start: date
    test_end: date
    n: int
    model_score: float
    baseline_score: float


@dataclass(frozen=True)
class ScreenRow:
    ticker: str
    sub_vertical: str
    actual: float
    warranted: float
    residual_log: float
    z: float


@dataclass(frozen=True)
class AuditRow:
    """One recorded observation beside what today's code rebuilds it to.

    ``ev_rebuilt`` is None when today's code refuses to build the row, and
    ``refused`` then names the category; a row a check refuses after it built
    carries both.
    """

    ticker: str
    as_of: date
    sub_vertical: str
    ev_recorded: float
    ev_rebuilt: float | None
    refused: str | None

    @property
    def gap(self) -> float | None:
        """The rebuilt enterprise value over the recorded one, less one."""
        if self.ev_rebuilt is None:
            return None
        return self.ev_rebuilt / self.ev_recorded - 1.0

    @property
    def identical(self) -> bool:
        return (
            self.ev_rebuilt is not None
            and abs(self.ev_rebuilt - self.ev_recorded) <= IDENTICAL_WITHIN_MM
        )


@dataclass(frozen=True)
class SkippedRow:
    """A row the panel skipped as debt outside the ladder, and what today's code makes of it."""

    ticker: str
    as_of: date
    today: str

    @property
    def admitted(self) -> bool:
        return self.today == "admitted"


@dataclass(frozen=True)
class Audit:
    """The committed enterprise-value audit of the panel, as plain values."""

    recorded: str
    code_commit: str
    observations: tuple[AuditRow, ...]
    skipped: tuple[SkippedRow, ...]


@dataclass(frozen=True)
class AuditCounts:
    observations: int
    compared: int
    identical: int
    off_noted: int
    off_stale: int
    filers_stale: int
    refused: int
    skipped: int
    admitted: int


@dataclass(frozen=True)
class FilerGap:
    """One filer with an observation more than ``GAP_STALE`` off."""

    ticker: str
    sub_vertical: str
    compared: int
    stale: int
    median_gap: float
    largest_gap: float

    @property
    def share(self) -> float:
        return self.stale / self.compared


@dataclass(frozen=True)
class PartialRefit:
    """The model refitted with only the enterprise values rebuilt.

    Partial by construction: the features, and the rows today's code would
    refuse or admit, stay as recorded.
    """

    score: float
    n: int
    swapped: int
    screen: tuple[tuple[str, bool], ...]  # (ticker, rich) on the screen's date


@dataclass(frozen=True)
class Inputs:
    """Plain values from one fit, which is everything ``shape`` needs."""

    target_label: str
    model_kind: str
    metric: str
    higher_is_better: bool
    comparisons: tuple[Comparison, ...]
    verdict_text: str
    fold_score_sd: float | None
    folds: tuple[FoldResult, ...]
    change: float | None
    n_changes: int
    pooled_on_change_rows: float | None
    screen_date: date | None
    screen_names: int
    screen: tuple[ScreenRow, ...]
    date_means: tuple[tuple[date, float], ...]
    between_date_share: float
    within_date_sd: float
    n_observations: int
    n_companies: int
    n_dates: int
    n_refused: int
    # None when no audit describes this panel; ``audit_problem`` then says why.
    audit: Audit | None = None
    audit_problem: str | None = None
    refit: PartialRefit | None = None

    @property
    def card(self) -> Comparison:
        cards = [c for c in self.comparisons if c.on_card]
        if len(cards) != 1:
            raise ValueError(
                f"expected exactly one comparison on the model card, found {len(cards)}"
            )
        return cards[0]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _lift(model: float, baseline: float, higher_is_better: bool) -> float:
    return model - baseline if higher_is_better else baseline - model


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sd(values: list[float]) -> float | None:
    """Sample standard deviation, ddof 1, as ``EvalResult.fold_sd`` takes it."""
    if len(values) < 2:
        return None
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _short(name: str) -> str:
    if name in SHORT_BASELINE_NAMES:
        return SHORT_BASELINE_NAMES[name]
    head = name.split(" (")[0].strip()
    return head[:1].upper() + head[1:]


def _lower_first(text: str) -> str:
    """Lower-case a label's first letter for use mid-sentence, unless it starts an acronym."""
    if len(text) > 1 and text[1].isupper():
        return text
    return text[:1].lower() + text[1:]


def _model_label(inp: Inputs) -> str:
    return f"Warranted multiple, {MODEL_NAMES.get(inp.model_kind, inp.model_kind)}"


def _metric_name(metric: str) -> str:
    return METRIC_NAMES.get(metric, metric)


def _signed(value: float, dp: int) -> str:
    """A signed number with a real minus sign, as the kit formats one on the page."""
    return f"{value:+.{dp}f}".replace("-", "\u2212")


def _row_height(rows: int, per_row: int) -> int:
    """The height to ask a dot chart for, so two-line row labels do not touch."""
    return rows * per_row + 34


def _month(d: date) -> str:
    return d.strftime("%b %Y")


def _figure(kind: str, title: str, subtitle: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "title": title, "subtitle": subtitle, "data": data}


def fold_lifts(inp: Inputs) -> list[float]:
    return [
        _lift(f.model_score, f.baseline_score, inp.higher_is_better) for f in inp.folds
    ]


# --------------------------------------------------------------------------- #
# The enterprise-value audit
# --------------------------------------------------------------------------- #


def _pct(value: float, dp: int = 0) -> str:
    """A signed percentage with a real minus sign."""
    return _signed(value * 100, dp) + "%"


def _names(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _count_word(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def audit_counts(audit: Audit) -> AuditCounts:
    """What the audit found, counted the way the section quotes it."""
    rebuilt = [r for r in audit.observations if r.ev_rebuilt is not None]
    stale = [r for r in rebuilt if abs(r.gap) > GAP_STALE]
    return AuditCounts(
        observations=len(audit.observations),
        compared=len(rebuilt),
        identical=sum(1 for r in rebuilt if r.identical),
        off_noted=sum(1 for r in rebuilt if abs(r.gap) > GAP_NOTED),
        off_stale=len(stale),
        filers_stale=len({r.ticker for r in stale}),
        refused=sum(1 for r in audit.observations if r.refused is not None),
        skipped=len(audit.skipped),
        admitted=sum(1 for s in audit.skipped if s.admitted),
    )


def affected_filers(audit: Audit) -> list[FilerGap]:
    """Each filer with an observation more than ``GAP_STALE`` off, most affected first.

    The share is of the filer's observations today's code could rebuild, and the
    median is of the gaps that crossed the threshold, since those are the ones
    the screen treats as stale.
    """
    by_filer: dict[str, list[AuditRow]] = {}
    for r in audit.observations:
        if r.ev_rebuilt is not None:
            by_filer.setdefault(r.ticker, []).append(r)
    out = []
    for ticker, rows in by_filer.items():
        gaps = [r.gap for r in rows if abs(r.gap) > GAP_STALE]
        if not gaps:
            continue
        out.append(
            FilerGap(
                ticker=ticker,
                sub_vertical=rows[0].sub_vertical,
                compared=len(rows),
                stale=len(gaps),
                median_gap=statistics.median(gaps),
                largest_gap=max(gaps, key=abs),
            )
        )
    return sorted(out, key=lambda f: (-f.share, -abs(f.median_gap), f.ticker))


REFUSED_TODAY = "refused by today's code"
ADMITTED_TODAY = "skipped when recorded and admitted by today's code"


def flagged_on(audit: Audit, when: date) -> dict[str, str]:
    """Every name the audit flags on one date, with how, in words.

    An observation more than ``GAP_STALE`` off, one today's code refuses, and a
    skipped row today's code would admit are all flagged: the first is priced
    wrong, the second should not be in the panel, and the third should be.
    """
    out: dict[str, str] = {}
    for r in audit.observations:
        if r.as_of != when:
            continue
        if r.refused is not None:
            out[r.ticker] = REFUSED_TODAY
        elif abs(r.gap) > GAP_STALE:
            way = "higher" if r.gap > 0 else "lower"
            out[r.ticker] = f"its enterprise value rebuilds {abs(r.gap):.0%} {way}"
    for s in audit.skipped:
        if s.as_of == when and s.admitted:
            out[s.ticker] = ADMITTED_TODAY
    return out


def screen_refusal(inp: Inputs) -> str | None:
    """Why the screen may not be drawn, or None when the audit leaves it standing.

    Refused when a name on it is flagged on the screen's date, and when the
    partial refit changes which names are on it or which side they are on. The
    second test exists because a residual is a distance from a model fitted on
    the whole panel, so a screen whose own names are clean can still be ranked
    by a fit the stale rows moved.
    """
    if not inp.screen or inp.screen_date is None:
        return None
    if inp.audit is None:
        return (
            "No enterprise-value audit describes this panel"
            + (f" ({inp.audit_problem})" if inp.audit_problem else "")
            + ", so whether its rich and cheap calls rest on enterprise values now "
            "known to be wrong cannot be checked."
        )
    when = inp.screen_date
    names = [r.ticker for r in sorted(inp.screen, key=lambda r: r.residual_log, reverse=True)]
    flagged = flagged_on(inp.audit, when)
    hit = [t for t in names if t in flagged]

    left: list[str] = []
    joined: list[str] = []
    crossed: list[str] = []
    if inp.refit is not None:
        recorded = {r.ticker: r.residual_log > 0 for r in inp.screen}
        refit = dict(inp.refit.screen)
        left = [t for t in names if t not in refit]
        joined = [t for t, _ in inp.refit.screen if t not in recorded]
        crossed = [t for t in names if t in refit and refit[t] != recorded[t]]
    if not hit and not (left or joined or crossed):
        return None

    parts: list[str] = []
    if hit:
        verb = "is" if len(hit) == 1 else "are"
        parts.append(
            f"{len(hit)} of the screen's {len(names)} names on {when.isoformat()} {verb} "
            "flagged by the enterprise-value audit: "
            + "; ".join(f"{t}, {flagged[t]}" for t in hit)
            + "."
        )
    else:
        ranked = [t for t in sorted(flagged) if flagged[t] != ADMITTED_TODAY]
        missing = [t for t in sorted(flagged) if flagged[t] == ADMITTED_TODAY]
        context = []
        if ranked:
            context.append(
                f"{len(ranked):,} of the {inp.screen_names:,} names ranked that day "
                f"({_names(ranked)})"
            )
        if missing:
            context.append(
                f"{_count_word(len(missing), 'row', 'rows')} today's code would admit "
                f"and the ranking leaves out ({_names(missing)})"
            )
        parts.append(
            f"None of the screen's {len(names)} names on {when.isoformat()} is itself "
            "flagged by the enterprise-value audit, but every residual is measured "
            "against a model fitted on the whole panel"
            + (f", and on that date the audit flags {' and '.join(context)}" if context else "")
            + "."
        )
    if left or joined or crossed:
        changed = len(left) + len(crossed)
        clauses = []
        if left:
            clauses.append(f"{_names(left)} {'leaves' if len(left) == 1 else 'leave'} it")
        if joined:
            clauses.append(f"{_names(joined)} {'joins' if len(joined) == 1 else 'join'}")
        if crossed:
            clauses.append(
                f"{_names(crossed)} {'changes' if len(crossed) == 1 else 'change'} "
                "between rich and cheap"
            )
        what = f"{changed} of its {len(names)} names" if changed else "the screen"
        parts.append(
            "Refitting with only the enterprise values rebuilt, a partial refit that "
            f"leaves the features as recorded, changes {what}: {_names(clauses)}."
        )
    parts.append(
        "A screen that calls a company rich or cheap on enterprise values now known "
        "to be wrong is not drawn, and re-recording the panel with today's debt "
        "ladders is the fix."
    )
    return " ".join(parts)


def _excluded(inp: Inputs) -> list[Comparison]:
    """Baselines stronger than the card's that the card still does not use.

    The card is scored against the strongest baseline answering the same
    question, so a baseline scoring above the card's own and left off it can only
    have been left off because it answers a different one.
    """
    card = inp.card
    return [
        c
        for c in inp.comparisons
        if not c.on_card
        and _lift(c.baseline_score, card.baseline_score, inp.higher_is_better) > 0
    ]


# --------------------------------------------------------------------------- #
# Headline and takeaway
# --------------------------------------------------------------------------- #


def verdict_status(inp: Inputs) -> str:
    """The chip, by the stricter of the two noise tests.

    The rule: "loses" when the pooled lift over the card's baseline is not
    positive. Otherwise "inside_noise" when the mean of the per-fold lifts is
    smaller than the standard deviation of those lifts, and "beats" when it is
    not. With fewer than two folds there is no dispersion to judge the lift
    against, and the chip says "not_significant".

    ``EvalResult.verdict`` compares the pooled lift with the spread of the
    model's own score across folds, which is a different yardstick; the noise
    figure shows both, so a reader can see why the chip and the sentence differ.
    """
    card = inp.card
    if _lift(card.model_score, card.baseline_score, inp.higher_is_better) <= 0:
        return "loses"
    lifts = fold_lifts(inp)
    sd = _sd(lifts)
    if sd is None:
        return "not_significant"
    return "inside_noise" if _mean(lifts) < sd else "beats"


def headline(inp: Inputs) -> dict[str, Any]:
    card = inp.card
    return {
        "metric": _metric_name(inp.metric),
        "score": card.model_score,
        "baseline_name": card.name,
        "baseline_score": card.baseline_score,
        "lift": _lift(card.model_score, card.baseline_score, inp.higher_is_better),
        "n": card.n,
        "higher_is_better": inp.higher_is_better,
        "verdict_status": verdict_status(inp),
        "verdict_text": inp.verdict_text,
    }


def takeaway(inp: Inputs) -> str:
    card = inp.card
    lift = _lift(card.model_score, card.baseline_score, inp.higher_is_better)
    ranks = "ranks" if lift > 0 else "does not rank"
    first = (
        f"The {MODEL_NAMES.get(inp.model_kind, inp.model_kind)} {ranks} "
        f"{inp.target_label} multiples better than the strongest peer baseline, "
        f"{_lower_first(_short(card.name))}: "
        f"{card.model_score:.4f} against {card.baseline_score:.4f} on {card.n:,} "
        "observations."
    )
    if inp.change is None:
        second = (
            " The differenced score could not be computed, so this page "
            "cannot say how much of that ranking is company identity."
        )
    else:
        if inp.change < IDENTITY_WORDING_THRESHOLD:
            reading = "so almost all of the ranking is company identity inherited from history."
        else:
            reading = "so a real part of the ranking survives differencing."
        second = (
            " Differenced against each company's own previous observation it "
            f"predicts the change at {_signed(inp.change, 4)} on {inp.n_changes:,}, {reading}"
        )
    return first + second + " " + audit_sentence(inp)


def audit_sentence(inp: Inputs) -> str:
    """The takeaway's last sentence: what the enterprise-value audit found, and what it costs."""
    screen = screen_refusal(inp)
    if inp.audit is None:
        return (
            "No enterprise-value audit describes this panel, so the screen is not drawn."
            if screen
            else "No enterprise-value audit describes this panel."
        )
    c = audit_counts(inp.audit)
    if c.off_stale == 0 and c.refused == 0 and c.admitted == 0:
        found = (
            "Rebuilt with today's debt ladders, every enterprise value in the panel is "
            f"within {GAP_STALE:.0%} of the one recorded"
        )
    else:
        found = (
            "The panel predates the debt-ladder fix: rebuilt with today's code, "
            f"{c.off_stale:,} of its {c.compared:,} enterprise values move by more than "
            f"{GAP_STALE:.0%}, at {_count_word(c.filers_stale, 'filer', 'filers')}"
        )
    if screen:
        return found + ", so the screen is not drawn and every score here is the panel as recorded."
    return found + "."


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def build_deflation(inp: Inputs) -> dict[str, Any]:
    if inp.change is None or inp.pooled_on_change_rows is None:
        raise FigureRefused(
            "deflation",
            "The fit returned no differenced rank correlation, so there is no score "
            f"for the change to set beside the level ({inp.n_changes:,} scored "
            "observations carry a previous observation).",
        )
    pooled, change, n = inp.pooled_on_change_rows, inp.change, inp.n_changes
    verb = "falls" if change < pooled else "rises"
    return _figure(
        "hbar",
        f"Differenced against last quarter, the score {verb} from {pooled:.2f} to "
        f"{change:.2f}".replace("-", "\u2212"),
        f"{_metric_name(inp.metric)}, out of sample, on the same {n:,} observations "
        "for both bars. "
        "Differenced means the actual and the fitted relative multiple, each less "
        "the company's own previous observation.",
        {
            "rows": [
                {
                    "key": "pooled",
                    "label": "Ranking the level",
                    "n": n,
                    "value": pooled,
                    "role": MUTED_MODEL_ROLE,
                },
                {
                    "key": "differenced",
                    "label": "Predicting the change",
                    "n": n,
                    "value": change,
                    "role": "model",
                },
            ],
            "format": "num:4",
            "valueLabel": _metric_name(inp.metric),
            "labelHeader": "Scored on",
            "legend": False,
        },
    )


def build_baselines(inp: Inputs) -> dict[str, Any]:
    card = inp.card
    excluded = _excluded(inp)
    ordered = (
        [card]
        + sorted(
            (c for c in inp.comparisons if not c.on_card and c not in excluded),
            key=lambda c: -c.n,
        )
        + excluded
    )
    rows = []
    for c in ordered:
        note = "the card's baseline, " if c.on_card else "left off the card, " if c in excluded else ""
        rows.append(
            {
                "key": c.name,
                "label": f"{_short(c.name)}, {note}n = {c.n:,}",
                "n": c.n,
                "on_card": c.on_card,
                "values": {"model": c.model_score, "baseline": c.baseline_score},
            }
        )
    peers = [c for c in inp.comparisons if c not in excluded]
    ahead = [c for c in peers if _lift(c.model_score, c.baseline_score, inp.higher_is_better) > 0]
    behind_excluded = [
        c for c in excluded if _lift(c.model_score, c.baseline_score, inp.higher_is_better) < 0
    ]
    if len(ahead) == len(peers) and excluded and len(behind_excluded) == len(excluded):
        title = "Ahead of every baseline built from peers, behind only " + " and ".join(
            _lower_first(_short(c.name)) for c in excluded
        )
    elif len(ahead) == len(peers):
        title = "Ahead of every baseline built from peers"
    else:
        title = f"Ahead of {len(ahead)} of {len(peers)} baselines built from peers"
    subtitle = (
        f"{_metric_name(inp.metric)} on the observations each baseline can score, so "
        "the model's own score moves with n."
    )
    for c in excluded:
        subtitle += (
            f" {_short(c.name)} scores above the card's baseline and stays off the card: "
            + EXCLUSION_REASONS.get(
                c.name, "the model does not count it as answering the same question."
            )
        )
    return _figure(
        "dot",
        title,
        subtitle,
        {
            "rows": rows,
            "series": [
                {"key": "model", "name": _model_label(inp), "role": "model"},
                {"key": "baseline", "name": "Baseline", "role": "baseline"},
            ],
            "format": "num:4",
            "gapLabel": "Lift",
            "gapFormat": "signed:4",
            "labelHeader": "Compared against",
            "height": _row_height(len(rows), 46),
        },
    )


def build_folds(inp: Inputs) -> dict[str, Any]:
    if len(inp.folds) < 2:
        raise FigureRefused(
            "folds",
            f"The fit produced {len(inp.folds)} walk-forward fold(s), and one fold is "
            "a single split with nothing to compare it to.",
        )
    card = inp.card
    lifts = fold_lifts(inp)
    wins = sum(1 for v in lifts if v > 0)
    rows = [
        {
            "key": f"fold_{f.index}",
            "label": f"{_month(f.test_start)} to {_month(f.test_end)}",
            "n": f.n,
            "values": {"model": f.model_score, "baseline": f.baseline_score},
        }
        for f in inp.folds
    ]
    return _figure(
        "dot",
        f"Ahead of {_lower_first(_short(card.name))} in "
        f"{wins} of {len(lifts)} folds, the lift running from "
        f"{_signed(min(lifts), 2)} to {_signed(max(lifts), 2)}",
        f"{_metric_name(inp.metric)} in each walk-forward test block, on the "
        f"observations the baseline could score: {min(f.n for f in inp.folds):,} to "
        f"{max(f.n for f in inp.folds):,} a block, {card.n:,} in all.",
        {
            "rows": rows,
            "series": [
                {"key": "model", "name": _model_label(inp), "role": "model"},
                {"key": "baseline", "name": _short(card.name), "role": "baseline"},
            ],
            "format": "num:4",
            "gapLabel": "Lift",
            "gapFormat": "signed:4",
            "labelHeader": "Test block",
            "height": _row_height(len(rows), 38),
            "table": {
                "columns": [
                    {"key": "block", "label": "Test block"},
                    {"key": "n", "label": "n", "align": "right", "format": "int"},
                    {"key": "model", "label": _model_label(inp), "align": "right", "format": "num:4"},
                    {"key": "baseline", "label": _short(card.name), "align": "right", "format": "num:4"},
                    {"key": "lift", "label": "Lift", "align": "right", "format": "signed:4"},
                ],
                "rows": [
                    {
                        "block": row["label"],
                        "n": f.n,
                        "model": f.model_score,
                        "baseline": f.baseline_score,
                        "lift": lift,
                    }
                    for row, f, lift in zip(rows, inp.folds, lifts)
                ],
            },
        },
    )


def _call(mid: float, half: float) -> str:
    """Where a lift sits against its yardstick, the rule ``EvalResult.verdict`` uses."""
    if abs(mid) < half:
        return "inside the noise"
    return "outside the noise, above zero" if mid > 0 else "outside the noise, below zero"


def build_noise(inp: Inputs) -> dict[str, Any]:
    lifts = fold_lifts(inp)
    strict_sd = _sd(lifts)
    if inp.fold_score_sd is None or strict_sd is None:
        raise FigureRefused(
            "noise",
            f"Both noise tests need at least two folds, and the fit produced {len(inp.folds)}.",
        )
    card = inp.card
    pooled_lift = _lift(card.model_score, card.baseline_score, inp.higher_is_better)
    mean_lift = _mean(lifts)
    tests = [
        {
            "key": "shipped",
            "label": "Verdict sentence: pooled lift, sd of the model's fold scores",
            "short": "Verdict sentence",
            "mid": pooled_lift,
            "half": inp.fold_score_sd,
            "noise": "sd of the model's fold scores",
        },
        {
            "key": "strict",
            "label": "Chip: mean fold lift, sd of the fold lifts",
            "short": "Chip",
            "mid": mean_lift,
            "half": strict_sd,
            "noise": "sd of the fold lifts",
        },
    ]
    shipped_call = _call(tests[0]["mid"], tests[0]["half"])
    strict_call = _call(tests[1]["mid"], tests[1]["half"])
    if shipped_call.startswith("outside") and strict_call.startswith("inside"):
        title = "Outside the noise by the verdict sentence's test, inside it by the stricter one"
    elif shipped_call == strict_call:
        title = f"Both noise tests put the lift {shipped_call}"
    else:
        title = f"The verdict sentence says {shipped_call}, the stricter test says {strict_call}"
    return _figure(
        "range",
        title,
        "Each bar is a lift plus and minus one standard deviation, so a bar that "
        "crosses zero is inside the noise. The sentence judges the pooled lift by "
        "how much the model's score moves between folds; the chip judges the lift "
        "by how much the lift itself moves.",
        {
            "rows": [
                {
                    "key": t["key"],
                    "label": t["label"],
                    "lo": t["mid"] - t["half"],
                    "mid": t["mid"],
                    "hi": t["mid"] + t["half"],
                    "role": "model",
                }
                for t in tests
            ],
            "format": "signed:4",
            "reference": [{"value": 0.0, "label": "No lift"}],
            "labelHeader": "Test",
            "table": {
                "columns": [
                    {"key": "test", "label": "Test"},
                    {"key": "lift", "label": "Lift", "align": "right", "format": "signed:4"},
                    {"key": "yardstick", "label": "Yardstick"},
                    {"key": "sd", "label": "Standard deviation", "align": "right", "format": "num:4"},
                    {"key": "call", "label": "Lift is"},
                ],
                "rows": [
                    {
                        "test": t["short"],
                        "lift": t["mid"],
                        "yardstick": t["noise"],
                        "sd": t["half"],
                        "call": _call(t["mid"], t["half"]),
                    }
                    for t in tests
                ],
            },
        },
    )


def _human(sub_vertical: str) -> str:
    return sub_vertical.replace("_", " ")


def build_screen(inp: Inputs) -> dict[str, Any]:
    if not inp.screen or inp.screen_date is None:
        raise FigureRefused(
            "screen",
            "The fit holds no out-of-sample reads on its latest date, so there is no "
            "screen to draw.",
        )
    stale = screen_refusal(inp)
    if stale is not None:
        raise FigureRefused("screen", stale)
    rows = sorted(inp.screen, key=lambda r: r.residual_log, reverse=True)
    rich = [r for r in rows if r.residual_log > 0]
    cheap = [r for r in rows if r.residual_log <= 0]

    def crowd(group: list[ScreenRow]) -> tuple[str, int] | None:
        if not group:
            return None
        name, count = Counter(r.sub_vertical for r in group).most_common(1)[0]
        return (name, count) if count * 2 >= len(group) else None

    top, bottom = crowd(rich), crowd(cheap)
    if top and bottom and top[0] != bottom[0]:
        title = (
            f"{_human(top[0]).capitalize()} crowd the rich end of the screen, "
            f"{_human(bottom[0])} the cheap end"
        )
    else:
        title = (
            f"{rows[0].ticker} trades furthest above its warranted multiple, "
            f"{rows[-1].ticker} furthest below"
        )
    return _figure(
        "hbar",
        title,
        f"Out-of-sample residual, log of the traded over the warranted {inp.target_label}, "
        f"for the {len(rich)} richest and {len(cheap)} cheapest of {inp.screen_names:,} "
        f"names on {inp.screen_date.isoformat()}. A residual says a company is priced "
        "unlike its characteristics, never that the market is wrong.",
        {
            "rows": [
                {
                    "key": r.ticker,
                    "label": r.ticker,
                    "value": r.residual_log,
                    "role": "pos" if r.residual_log > 0 else "neg",
                    "note": (
                        f"{_human(r.sub_vertical)}, {r.actual:.1f}x against "
                        f"{r.warranted:.1f}x warranted"
                    ),
                }
                for r in rows
            ],
            "format": "signed:2",
            "valueLabel": "Residual, log points",
            "labelHeader": "Ticker",
            "roleLabels": {
                "pos": "Trades above its warranted multiple",
                "neg": "Trades below it",
            },
            "wide": True,
            "table": {
                "columns": [
                    {"key": "ticker", "label": "Ticker"},
                    {"key": "sub_vertical", "label": "Sub-vertical"},
                    {"key": "actual", "label": "Trades at", "align": "right", "format": "mult:1"},
                    {"key": "warranted", "label": "Warranted", "align": "right", "format": "mult:1"},
                    {"key": "residual_log", "label": "Residual, log", "align": "right", "format": "signed:2"},
                    {"key": "z", "label": "Within-date sd", "align": "right", "format": "signed:2"},
                ],
                "rows": [
                    {
                        "ticker": r.ticker,
                        "sub_vertical": _human(r.sub_vertical),
                        "actual": r.actual,
                        "warranted": r.warranted,
                        "residual_log": r.residual_log,
                        "z": r.z,
                    }
                    for r in rows
                ],
            },
        },
    )


def build_rerating(inp: Inputs) -> dict[str, Any]:
    if len(inp.date_means) < 2:
        raise FigureRefused(
            "rerating",
            f"The panel holds {len(inp.date_means)} quarter end(s), and a re-rating "
            "needs at least two.",
        )
    points = sorted(inp.date_means)
    sd = inp.within_date_sd
    values = [
        {
            "x": d.isoformat(),
            "y": math.exp(m),
            "lo": math.exp(m - sd),
            "hi": math.exp(m + sd),
        }
        for d, m in points
    ]
    peak = max(points, key=lambda p: p[1])
    trough = min(points, key=lambda p: p[1])
    first, second = (peak, trough) if peak[0] < trough[0] else (trough, peak)
    verb = "fell" if first is peak else "rose"
    # "yet" only where the calendar is the smaller part of the variance.
    joint = "yet the calendar is only" if inp.between_date_share < 0.5 else "and the calendar is"
    return _figure(
        "line",
        f"The panel {verb} from {math.exp(first[1]):.1f}x to {math.exp(second[1]):.1f}x "
        f"{inp.target_label}, {joint} {inp.between_date_share:.1%} of the variance",
        f"Geometric mean {inp.target_label} of the panel at each of {inp.n_dates} quarter "
        f"ends, {inp.n_observations:,} observations. The band is one within-date "
        "standard deviation of the log multiple either side, the spread between "
        f"companies on a typical date; {inp.between_date_share:.1%} of the pooled "
        "variance of the log multiple lies between dates.",
        {
            "series": [{"name": "Panel mean", "role": "total", "values": values}],
            "x": {"label": "Quarter end"},
            "format": "mult:1",
            "wide": True,
            "between_date_share": inp.between_date_share,
            "table": {
                "columns": [
                    {"key": "x", "label": "Quarter end"},
                    {"key": "y", "label": "Geometric mean", "align": "right", "format": "mult:2"},
                    {"key": "lo", "label": "One sd below", "align": "right", "format": "mult:2"},
                    {"key": "hi", "label": "One sd above", "align": "right", "format": "mult:2"},
                ],
                "rows": values,
            },
        },
    )


def build_panel(inp: Inputs) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tiles": [
            {"key": "observations", "label": "Company-quarters", "value": inp.n_observations, "format": "int"},
            {"key": "companies", "label": "Companies", "value": inp.n_companies, "format": "int"},
            {"key": "dates", "label": "Quarter ends", "value": inp.n_dates, "format": "int"},
            {
                "key": "refused",
                "label": "Company-dates refused",
                "value": inp.n_refused,
                "format": "int",
                "sub": "Kept with their reasons in the panel's skips",
            },
        ]
    }
    if inp.audit is not None and _audit_found_anything(inp.audit):
        c = audit_counts(inp.audit)
        data["notes"] = [
            {
                "what": "Recorded before the debt-ladder fix",
                "why": (
                    f"These are the panel's counts as recorded. Rebuilt with today's code, "
                    f"{c.off_stale:,} of its {c.compared:,} enterprise values move by more "
                    f"than {GAP_STALE:.0%}, {c.refused:,} of its observations are refused "
                    f"and {c.admitted:,} of its skips are admitted, so re-recorded it would "
                    "hold different rows as well as different multiples."
                ),
            }
        ]
    return _figure("tiles", "The panel behind every figure here", "", data)


def _audit_found_anything(audit: Audit) -> bool:
    c = audit_counts(audit)
    return bool(c.off_stale or c.refused or c.admitted)


def _audit_caution(inp: Inputs) -> dict[str, str] | None:
    """The caution under the audit: what the counts mean for the headline above them."""
    audit = inp.audit
    if audit is None or not _audit_found_anything(audit):
        return None
    c = audit_counts(audit)
    card = inp.card
    metric = _metric_name(inp.metric)
    why = (
        "The headline above and every figure below are fitted on the panel as recorded, "
        f"in which {c.off_stale:,} enterprise values are more than {GAP_STALE:.0%} off at "
        f"{_count_word(c.filers_stale, 'filer', 'filers')}, {c.refused:,} observations are "
        f"ones today's code refuses, and {c.admitted:,} rows today's code admits are missing."
    )
    if inp.refit is not None:
        why += (
            " Refitting with only the enterprise values rebuilt moves the headline "
            f"{metric} from {card.model_score:.4f} on {card.n:,} observations to "
            f"{inp.refit.score:.4f} on {inp.refit.n:,}. That is a partial refit and it "
            "understates the change: the features, the debt-to-capital ratio among "
            "them, came from the same old ladder, and the refused and admitted rows "
            "stay as recorded."
        )
    else:
        why += " No refit on the rebuilt enterprise values was run, so how far the headline moves is not shown."
    why += " Re-recording the panel with today's debt ladders is the fix, and it has not been made."
    return {"what": "The panel predates the debt-ladder fix", "why": why}


def build_audit(inp: Inputs) -> dict[str, Any]:
    if inp.audit is None:
        raise FigureRefused(
            "audit",
            "No enterprise-value audit describes this panel"
            + (f": {inp.audit_problem}" if inp.audit_problem else "")
            + ". Run tests/fixtures/warranted/record_ev_audit.py against it.",
        )
    audit = inp.audit
    c = audit_counts(audit)
    if _audit_found_anything(audit):
        title = (
            f"Rebuilt with today's debt ladders, {c.off_stale:,} of the panel's enterprise "
            f"values move by more than {GAP_STALE:.0%}, at "
            f"{_count_word(c.filers_stale, 'filer', 'filers')}"
        )
    else:
        title = (
            "Rebuilt with today's debt ladders, no enterprise value in the panel moves by "
            f"more than {GAP_STALE:.0%}"
        )
    refused = Counter(r.refused for r in audit.observations if r.refused is not None)
    words = {k: REFUSAL_WORDS.get(k, k.replace("_", " ")) for k in refused}
    if not refused:
        refused_sub = "Today's code admits every observation in the panel"
    elif len(refused) == 1:
        refused_sub = f"In the panel as recorded, all {next(iter(words.values()))}"
    else:
        refused_sub = "In the panel as recorded: " + _names(
            [f"{n:,} {words[k]}" for k, n in sorted(refused.items())]
        )
    data: dict[str, Any] = {
        "tiles": [
            {
                "key": "compared",
                "label": "Enterprise values rebuilt",
                "value": c.compared,
                "format": "int",
                "sub": f"Of {c.observations:,} observations in the panel",
            },
            {
                "key": "identical",
                "label": "Identical",
                "value": c.identical,
                "format": "int",
                "sub": "Within the rounding both figures carry",
            },
            {
                "key": "off_noted",
                "label": f"More than {GAP_NOTED:.0%} off",
                "value": c.off_noted,
                "format": "int",
            },
            {
                "key": "off_stale",
                "label": f"More than {GAP_STALE:.0%} off",
                "value": c.off_stale,
                "format": "int",
                "sub": f"At {_count_word(c.filers_stale, 'filer', 'filers')}",
            },
            {
                "key": "refused",
                "label": "Refused by today's code",
                "value": c.refused,
                "format": "int",
                "sub": refused_sub,
            },
            {
                "key": "admitted",
                "label": "Admitted by today's code",
                "value": c.admitted,
                "format": "int",
                "sub": (
                    f"Of {c.skipped:,} rows skipped as debt outside the ladder; the float "
                    "check needs a price and is not run on them"
                ),
            },
        ],
        "wide": True,
    }
    caution = _audit_caution(inp)
    if caution is not None:
        data["notes"] = [caution]
    return _figure(
        "tiles",
        title,
        (
            f"Each of the panel's {c.observations:,} observations rebuilt by "
            f"record_ev_audit.py at {audit.code_commit[:7]}, pinned to its own date: the "
            "recorded equity value plus today's net debt, then today's row checks. The "
            f"audit was recorded on {audit.recorded}; the features are not rebuilt."
        ),
        data,
    )


# How a refusal category reads in a tile's sub-line.
REFUSAL_WORDS = {
    "not_built": "not built by today's code",
    "revenue_is_a_component": "failing today's revenue check",
    "debt_outside_the_ladder": "failing today's debt check",
    "price_disagrees_with_float": "failing today's float check",
}


def build_audit_filers(inp: Inputs) -> dict[str, Any]:
    if inp.audit is None:
        raise FigureRefused("audit_filers", "No enterprise-value audit describes this panel.")
    filers = affected_filers(inp.audit)
    if not filers:
        raise FigureRefused(
            "audit_filers",
            f"No filer has an enterprise value more than {GAP_STALE:.0%} off, so there is "
            "no filer to draw.",
        )
    everywhere = [f for f in filers if f.stale == f.compared]
    if everywhere:
        worst = max(everywhere, key=lambda f: abs(f.median_gap))
        title = (
            f"{_names([f.ticker for f in everywhere])} "
            f"{'is' if len(everywhere) == 1 else 'are'} more than {GAP_STALE:.0%} off at "
            f"every observation, {worst.ticker} by a median of {_pct(worst.median_gap)}"
        )
    else:
        top = filers[0]
        title = (
            f"{top.ticker} is the most affected, {top.stale} of its {top.compared} "
            f"observations more than {GAP_STALE:.0%} off"
        )
    return _figure(
        "dot",
        title,
        (
            f"Share of each filer's rebuilt observations more than {GAP_STALE:.0%} off, and "
            "the median of those gaps, rebuilt over recorded less one. A positive gap is "
            "net debt today's code finds and the recorded panel did not."
        ),
        {
            "rows": [
                {
                    "key": f.ticker,
                    "label": f.ticker,
                    "values": {"share": f.share},
                    "aside": _pct(f.median_gap),
                    "tip": [
                        {"label": f"More than {GAP_STALE:.0%} off", "value": f"{f.stale} of {f.compared}"},
                        {"label": "Largest gap", "value": _pct(f.largest_gap)},
                    ],
                }
                for f in filers
            ],
            "series": [
                {"key": "share", "name": f"Share more than {GAP_STALE:.0%} off", "role": "total"}
            ],
            "format": "pct:0",
            "domain": [0, 1],
            "labels": "all",
            "asideHeader": "Median gap",
            "labelHeader": "Filer",
            "legend": False,
            "height": _row_height(len(filers), 26),
            "table": {
                "columns": [
                    {"key": "ticker", "label": "Filer"},
                    {"key": "sub_vertical", "label": "Sub-vertical"},
                    {"key": "compared", "label": "Rebuilt", "align": "right", "format": "int"},
                    {"key": "stale", "label": f"More than {GAP_STALE:.0%} off", "align": "right", "format": "int"},
                    {"key": "share", "label": "Share", "align": "right", "format": "pct:0"},
                    {"key": "median_gap", "label": "Median gap", "align": "right"},
                    {"key": "largest_gap", "label": "Largest gap", "align": "right"},
                ],
                "rows": [
                    {
                        "ticker": f.ticker,
                        "sub_vertical": _human(f.sub_vertical),
                        "compared": f.compared,
                        "stale": f.stale,
                        "share": f.share,
                        "median_gap": _pct(f.median_gap),
                        "largest_gap": _pct(f.largest_gap),
                    }
                    for f in filers
                ],
            },
        },
    )


# (figure id, entry point that computed it, builder), in page order.
FIGURES: tuple[tuple[str, str, Callable[[Inputs], dict[str, Any]]], ...] = (
    ("audit", ENTRY_AUDIT, build_audit),
    ("audit_filers", ENTRY_FILERS, build_audit_filers),
    ("deflation", ENTRY_FIT, build_deflation),
    ("baselines", ENTRY_FIT, build_baselines),
    ("folds", ENTRY_EVALUATE, build_folds),
    ("noise", ENTRY_EVALUATE, build_noise),
    ("screen", ENTRY_EXTREMES, build_screen),
    ("rerating", ENTRY_FIT, build_rerating),
    ("panel", ENTRY_LOAD, build_panel),
)

# The declared inputs behind each figure. The audit card's caution quotes the
# partial refit, which reads both files, and the screen is checked against the
# audit before it is drawn; every other figure is the panel's alone.
FIGURE_INPUTS: dict[str, list[str]] = {
    "audit": [AUDIT, PANEL],
    "audit_filers": [AUDIT],
    "screen": [PANEL, AUDIT],
}


def _no_record(figure_id: str, entry_point: str) -> ContextManager[None]:
    return nullcontext()


def shape(
    inp: Inputs,
    record: Callable[[str, str], ContextManager[None]] = _no_record,
) -> dict[str, Any]:
    """Turn one fit's values into the section: figures, refusals, headline, takeaway.

    ``record(figure_id, entry_point)`` wraps each figure's construction. A
    refused figure raises inside its block, so the block records nothing and
    the refusal takes the figure's place.
    """
    figures: dict[str, dict[str, Any]] = {}
    refusals: list[dict[str, str]] = []
    for figure_id, entry_point, build in FIGURES:
        try:
            with record(figure_id, entry_point):
                figures[figure_id] = build(inp)
        except FigureRefused as refused:
            refusals.append({"what": refused.what, "why": refused.why})
    return {
        "status": "ok",
        "takeaway": takeaway(inp),
        "refusals": refusals,
        "headline": headline(inp),
        "figures": figures,
    }


# --------------------------------------------------------------------------- #
# From a fit to Inputs
# --------------------------------------------------------------------------- #


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=0.0, abs_tol=REBUILD_TOLERANCE)


def _rebuild(panel, model, assumptions) -> dict[str, Any]:
    """The arrays ``fit_warranted`` scored, recovered and checked against the fit.

    Returns the value array behind each of the model's baselines, keyed by the
    model's own baseline name, with the demeaned target, the model's
    predictions, the dates, and the mask of observations the differenced score
    was taken on.
    """
    import numpy as np

    from ...ml import warranted as W
    from ...ml.evaluation import evaluate_regression

    observations = sorted(panel.observations, key=lambda o: (o.as_of, o.ticker))
    y_raw = np.asarray([o.log_multiple for o in observations], dtype=float)
    dates = np.asarray([o.as_of for o in observations], dtype=object)
    loo, _ = W._leave_one_out_means(y_raw, dates)
    add_back = loo if model.demeaned else np.zeros_like(y_raw)
    y = y_raw - add_back

    prediction = np.full(y.shape, np.nan)
    for i, obs in enumerate(observations):
        read = model.reads.get((obs.ticker, obs.as_of))
        if read is not None:
            prediction[i] = math.log(read.warranted_multiple) - add_back[i]
    scored = np.isfinite(prediction)

    incumbent = W._comps_baselines(panel, observations, assumptions)
    relative_to = loo if model.demeaned else None
    persistence = W._persistence(observations, y)
    candidates = {
        "sub-vertical median": W._sub_vertical_median(observations, y, dates),
        "comps sub-vertical": W._to_relative(incumbent["sub_vertical"][0], relative_to),
        "comps universe": W._to_relative(incumbent["universe"][0], relative_to),
        "persistence": persistence,
    }

    matched: dict[str, Any] = {}
    for name, result in model.baselines.items():
        for values in candidates.values():
            mask = scored & np.isfinite(values)
            if int(mask.sum()) != result.n_observations:
                continue
            rebuilt = evaluate_regression(
                y[mask], prediction[mask], values[mask], dates[mask],
                folds=model.folds, metric=result.metric,
            )
            if (
                _close(rebuilt.score, result.score)
                and _close(rebuilt.baseline_score, result.baseline_score)
                and len(rebuilt.folds) == len(result.folds)
                and all(_close(a, b) for a, b in zip(rebuilt.folds, result.folds))
            ):
                matched[name] = values
                break
        else:
            raise ValueError(
                f"the dashboard's rebuild of the warranted fit does not reproduce the "
                f"baseline {name!r} ({result.score:.6f} against {result.baseline_score:.6f} "
                f"on {result.n_observations}); techval.ml.warranted has changed how it "
                "scores, and sections/warranted.py must follow it"
            )

    usable = scored & np.isfinite(persistence)
    if model.change_rank_correlation is not None:
        from ...ml.protocol import spearman

        change = spearman(y[usable] - persistence[usable], prediction[usable] - persistence[usable])
        if int(usable.sum()) != model.n_changes or not _close(change, model.change_rank_correlation):
            raise ValueError(
                "the dashboard's rebuild does not reproduce the differenced rank "
                f"correlation ({change} on {int(usable.sum())} against "
                f"{model.change_rank_correlation} on {model.n_changes})"
            )
    return {
        "values": matched,
        "y": y,
        "prediction": prediction,
        "dates": dates,
        "usable_for_change": usable,
    }


def extract(panel, model, assumptions) -> Inputs:
    """Read one fit into ``Inputs``, rebuilding the fold scores the result does not keep."""
    import numpy as np

    from ...ml.evaluation import evaluate_regression
    from ...ml.protocol import spearman
    from ...ml.warranted import TARGETS

    card = model.card.evaluation
    if card is None:
        raise ValueError("the warranted model card carries no evaluation")
    rebuilt = _rebuild(panel, model, assumptions)
    y, prediction, dates = rebuilt["y"], rebuilt["prediction"], rebuilt["dates"]

    comparisons = tuple(
        Comparison(
            name=name,
            model_score=result.score,
            baseline_score=result.baseline_score,
            n=result.n_observations,
            on_card=result is card,
        )
        for name, result in model.baselines.items()
    )

    # The card's baseline scored as though it were the model, against the model:
    # its fold scores are then the baseline's fold scores on the card's own split.
    card_values = rebuilt["values"][card.baseline_name]
    mask = np.isfinite(prediction) & np.isfinite(card_values)
    swapped = evaluate_regression(
        y[mask], card_values[mask], prediction[mask], dates[mask],
        folds=model.folds, metric=card.metric,
    )
    if not (_close(swapped.score, card.baseline_score) and _close(swapped.baseline_score, card.score)):
        raise ValueError("scoring the card's baseline against the model did not reproduce the card")
    masked_dates = dates[mask]
    folds = tuple(
        FoldResult(
            index=fold.index,
            test_start=fold.test_start,
            test_end=fold.test_end,
            n=int(((masked_dates >= fold.test_start) & (masked_dates <= fold.test_end)).sum()),
            model_score=model_fold,
            baseline_score=baseline_fold,
        )
        for fold, model_fold, baseline_fold in zip(model.folds, card.folds, swapped.folds)
    )

    usable = rebuilt["usable_for_change"]
    pooled = (
        spearman(y[usable], prediction[usable])
        if model.change_rank_correlation is not None
        else None
    )

    latest = model.latest
    frame = model.extremes(SCREEN_EACH_END, when=latest)
    screen = tuple(
        ScreenRow(
            ticker=str(row["ticker"]),
            sub_vertical=str(row["sub_vertical"]),
            actual=float(row["actual"]),
            warranted=float(row["warranted"]),
            residual_log=float(row["residual_log"]),
            z=float(row["z"]),
        )
        for row in frame.to_dict("records")
    )

    return Inputs(
        target_label=TARGETS[model.target][1],
        model_kind=str(model.card.hyperparameters.get("model", "")),
        metric=card.metric,
        higher_is_better=card.higher_is_better,
        comparisons=comparisons,
        verdict_text=model.verdict(),
        fold_score_sd=card.fold_sd,
        folds=folds,
        change=model.change_rank_correlation,
        n_changes=model.n_changes,
        pooled_on_change_rows=pooled,
        screen_date=latest,
        screen_names=sum(1 for (_, when) in model.reads if when == latest),
        screen=screen,
        date_means=tuple(sorted(model.rerating.date_means.items())),
        between_date_share=model.rerating.between_date_share,
        within_date_sd=model.rerating.within_date_sd,
        n_observations=len(panel.observations),
        n_companies=len(panel.tickers),
        n_dates=len(panel.dates),
        n_refused=len(panel.skips),
    )


def read_audit(path, panel) -> tuple[Audit | None, str | None]:
    """The committed audit, checked against the panel it claims to describe.

    An audit of another panel is not evidence about this one, so a mismatch in
    the company-dates or in any recorded enterprise value returns no audit and
    the reason, and the figures that need it refuse with that reason.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    header = raw.get("_techval_fixture") or {}
    rows = tuple(
        AuditRow(
            ticker=str(r["ticker"]),
            as_of=date.fromisoformat(r["as_of"]),
            sub_vertical=str(r["sub_vertical"]),
            ev_recorded=float(r["ev_recorded"]),
            ev_rebuilt=None if r["ev_rebuilt"] is None else float(r["ev_rebuilt"]),
            refused=r["refused"],
        )
        for r in raw["observations"]
    )
    recorded = {(o.ticker, o.as_of): o.enterprise_value for o in panel.observations}
    audited = {(r.ticker, r.as_of): r.ev_recorded for r in rows}
    if set(recorded) != set(audited) or len(audited) != len(rows):
        only_panel = len(set(recorded) - set(audited))
        only_audit = len(set(audited) - set(recorded))
        return None, (
            f"the audit covers {len(audited):,} company-dates and the panel holds "
            f"{len(recorded):,}; {only_panel:,} are in the panel only and {only_audit:,} "
            "in the audit only, so it audits another recording of the panel"
        )
    moved = [k for k, ev in recorded.items() if abs(ev - audited[k]) > IDENTICAL_WITHIN_MM]
    if moved:
        return None, (
            f"{len(moved):,} of the audit's recorded enterprise values do not match the "
            "panel's, so it audits another recording of the panel"
        )
    skipped = tuple(
        SkippedRow(ticker=str(s["ticker"]), as_of=date.fromisoformat(s["as_of"]), today=str(s["today"]))
        for s in raw["skipped"]
    )
    return (
        Audit(
            recorded=str(header.get("recorded", "")),
            code_commit=str(header.get("code_commit", "")),
            observations=rows,
            skipped=skipped,
        ),
        None,
    )


def partial_refit(panel, audit: Audit, assumptions, model_kind: str, when: date | None) -> PartialRefit:
    """Refit with each rebuilt enterprise value in place of the recorded one, and nothing else.

    A row today's code refuses keeps its recorded value, a row it would admit
    stays out, and the features stay as recorded, which is why the result is
    called partial wherever it is quoted. The comps memo is emptied, because the
    incumbent baseline is a function of the multiples and would otherwise be
    read from the recorded panel's fits.
    """
    from ...ml.warranted import fit_warranted

    rebuilt = {(r.ticker, r.as_of): r for r in audit.observations}
    observations = []
    swapped = 0
    for o in panel.observations:
        row = rebuilt[(o.ticker, o.as_of)]
        if row.ev_rebuilt is None or row.identical:
            observations.append(o)
            continue
        if row.ev_rebuilt <= 0:
            raise ValueError(
                f"{o.ticker} on {o.as_of} rebuilds to an enterprise value of "
                f"{row.ev_rebuilt:,.0f}mm, which has no log multiple"
            )
        multiple = row.ev_rebuilt / o.denominator
        observations.append(
            replace(
                o,
                enterprise_value=row.ev_rebuilt,
                multiple=multiple,
                log_multiple=math.log(multiple),
            )
        )
        swapped += 1
    refit_panel = replace(panel, observations=observations, _comps_memo={})
    model = fit_warranted(refit_panel, assumptions, model=model_kind)
    card = model.card.evaluation
    if card is None:
        raise ValueError("the partial refit's model card carries no evaluation")
    screen: tuple[tuple[str, bool], ...] = ()
    if when is not None and any(d == when for (_, d) in model.reads):
        frame = model.extremes(SCREEN_EACH_END, when=when)
        screen = tuple(
            (str(r["ticker"]), float(r["residual_log"]) > 0) for r in frame.to_dict("records")
        )
    return PartialRefit(score=card.score, n=card.n_observations, swapped=swapped, screen=screen)


def collect(ctx) -> dict:
    from ...commands_peers import _load_observations
    from ...ml.warranted import fit_warranted

    panel = _load_observations(ctx.input(PANEL))
    model = fit_warranted(panel, ctx.assumptions, model="mlp")
    inputs = extract(panel, model, ctx.assumptions)
    audit, problem = read_audit(ctx.input(AUDIT), panel)
    refit = (
        None
        if audit is None
        else partial_refit(panel, audit, ctx.assumptions, "mlp", inputs.screen_date)
    )
    inputs = replace(inputs, audit=audit, audit_problem=problem, refit=refit)
    return shape(
        inputs,
        record=lambda fid, entry: ctx.record(fid, entry, FIGURE_INPUTS.get(fid, [PANEL])),
    )
