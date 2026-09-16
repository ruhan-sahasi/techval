"""M&A propensity: the weakest model in the package, shown with the numbers that say so.

The collector fits the acquisition screen once from ``tests/fixtures/mna`` the
way ``techval targets`` does (``load_dataset`` then ``fit_propensity``), refits
it twice on reduced label sets, ranks the latest date, and hands every number
it read to ``shape``, a pure function that turns them into the page. ``shape``
is where the titles and the takeaway are written, as f-strings over those
numbers, so a sentence on the page cannot say something the fit did not.

Four choices here are judgments and are recorded where they are made.

**The headline n is the scored count.** The AUC is computed on the rows that
fell inside a walk-forward test window, which is also the count the model's own
verdict sentence quotes. The labelled sample and the design matrix are larger,
and the number that governs the evidence is smaller still: the distinct
companies among the scored positives. The sample tiles carry all of them, so no
count stands in for another.

**A target is not a deal, and the page never calls one by the other's name.**
``events.json`` holds announced deals, 107 of them as its manifest records.
``LabelReport.n_distinct_deals`` counts something else despite its name: the
distinct companies with a positive label. On this fixture every deal has its
own target, and still only 97 are labelled, because a deal announced before
any panel date's window could reach it, or after the last window that has
closed, labels no company. The targets among the scored rows are fewer again.
So the tiles say targets labelled and targets scored, the label sets in the
recovered-deals figure are counted in announced deals, and one sub-line puts
the three side by side.

**The recovered deals are identified from the fixture's own audit.**
``MANIFEST.json`` names the seven transactions the dual-class extraction fix
recovered and splits them into four sales and three it doubts. The names are
held below and checked against that audit and against ``events.json`` at
collection time; if either no longer agrees, the figure is refused rather than
drawn from a guessed subset.

**Colour follows the entity.** The fitted model is blue and the size-only sort
grey everywhere. What actually happened (the share announced as targets) is
orange, in the base rate by year and in the calibration dumbbell, where it is
the second entity of a literal comparison. The base rate as a benchmark for a
list is a reference rule, not a series.

**No cross-unit comparison is drawn.** The base rate moving by a factor of two
and the AUC moving by two hundredths are not on one scale, so the page states
each and compares neither.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from statistics import fmean, stdev
from typing import Any, Sequence

import numpy as np

from ...errors import ConfigError, MissingDataError, NotMeaningfulError, TechvalError

ID = "propensity"
TITLE = "M&A propensity"

MNA = "mna"
UNIVERSE = f"{MNA}/universe.json"
EVENTS = f"{MNA}/events.json"
PANEL = f"{MNA}/panel.csv.gz"
MANIFEST = f"{MNA}/MANIFEST.json"
PRICES = f"{MNA}/price_availability.json"

# The three files load_dataset reads (it opens panel.csv.gz and never looks for
# the plain panel.csv when the gzip exists), the manifest the as-of date and the
# recovered-deal audit come from, and the price probe behind the blind-spot note.
INPUTS: list[str] = [UNIVERSE, EVENTS, PANEL, MANIFEST, PRICES]
FIT_INPUTS = [UNIVERSE, EVENTS, PANEL, MANIFEST]

# Figure ids in page order. The snapshot is written with sorted keys, so the
# renderer draws from this order and not from the file's.
FIGURES = (
    "sample",
    "auc_by_fold",
    "recovered_deals",
    "coefficient_signs",
    "base_rate_by_year",
    "calibration",
    "precision_at_k",
    "recall_at_k",
    "target_list",
)

# The seven transactions MANIFEST.json says the dual-class fix in
# _consideration_clause recovered, as its audit names and groups them. They are
# identities, not figures: every count and score drawn from them is computed.
RECOVERED_SALES = ("PowerSchool", "Endeavor", "DouYu", "DISH")
RECOVERED_DOUBTED = ("DraftKings Holdings", "SIRIUS XM", "Gen Digital")
_AUDIT_COUNT = re.compile(r"Of the (\d+) transactions recovered")
_AUDIT_SPLIT = "target-side acquisitions"

# Presentation thresholds, not results. A calibration bucket whose stated
# probability starts at LOUD or above gets its gap labelled on the chart, if it
# is not thin. SIGN_BREAKS are the magnitudes at which a coefficient cell moves
# to a stronger colour class; they are fixed rather than scaled to the largest
# coefficient, because fold one's coefficients are several times the size of
# fold five's and a linear scale would paint the later folds' signs neutral.
LOUD = 0.3
SIGN_BREAKS = (0.1, 0.5)


# --------------------------------------------------------------------------- #
# What shape reads: plain values, so a test can hand it fakes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Refused:
    """A figure that could not be computed, and why."""

    what: str
    why: str


@dataclass(frozen=True)
class Evaluation:
    metric: str
    score: float
    baseline_name: str
    baseline_score: float
    lift: float
    fold_sd: float | None
    n_scored: int
    n_folds: int
    higher_is_better: bool
    beat_baseline: bool
    verdict_text: str
    chance: float


@dataclass(frozen=True)
class Sample:
    n_labelled: int
    positives: int
    # Distinct TARGETS with a positive label, read from
    # ``LabelReport.n_distinct_deals``; the field keeps the library's name so the
    # published count keeps its key.
    distinct_deals: int
    base_rate: float
    scored_positives: int
    # Distinct targets among the scored positives, under the same naming rule.
    scored_deals: int
    # Announced deals in ``events.json``: the one count here that is of deals.
    announced_deals: int


@dataclass(frozen=True)
class FoldScore:
    index: int
    test_start: date
    test_end: date
    n: int
    positives: int
    model: float
    size: float


@dataclass(frozen=True)
class YearRate:
    year: int
    n: int
    positives: int
    rate: float
    dates: int


@dataclass(frozen=True)
class Coefficient:
    feature: str
    folds: tuple[float, ...]
    mean: float
    sd: float
    flips: bool


@dataclass(frozen=True)
class Coefficients:
    fold_labels: tuple[str, ...]
    fold_train: tuple[int, ...]
    rows: tuple[Coefficient, ...]


@dataclass(frozen=True)
class Bucket:
    low: float
    high: float
    n: int
    stated: float
    realised: float
    gap: float
    thin: bool


@dataclass(frozen=True)
class Calibration:
    buckets: tuple[Bucket, ...]
    thin_below: int
    n: int


@dataclass(frozen=True)
class TopK:
    k: int
    dates: int
    precision_model: float
    precision_size: float
    base_rate: float
    recall_model: float
    recall_size: float
    random_recall: float


@dataclass(frozen=True)
class Refit:
    events: int
    score: float
    baseline_score: float
    lift: float
    fold_sd: float | None
    beat_baseline: bool


@dataclass(frozen=True)
class Recovered:
    committed: Refit
    without_doubted: Refit
    without_all: Refit
    n_recovered: int
    n_doubted: int


@dataclass(frozen=True)
class ScreenRow:
    rank: int
    ticker: str
    name: str
    sub_vertical: str | None
    probability: float
    drivers: tuple[str, ...]


@dataclass(frozen=True)
class Screen:
    as_of: date
    n_ranked: int
    rows: tuple[ScreenRow, ...]


@dataclass(frozen=True)
class PriceLeak:
    auc: float
    positives_departed: float
    negatives_departed: float
    served: tuple[tuple[str, str, int], ...]  # (symbol, source, rows) that returned prices
    unserved: tuple[str, ...]  # symbols no source returned a row for
    fitted_columns: int


@dataclass(frozen=True)
class Facts:
    evaluation: Evaluation
    sample: Sample
    folds: tuple[FoldScore, ...] | Refused
    recovered: Recovered | Refused
    coefficients: Coefficients | Refused
    base_rates: tuple[YearRate, ...] | Refused
    calibration: Calibration | Refused
    top_k: TopK | Refused
    screen: Screen | Refused
    price: PriceLeak | Refused | None


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #


def verdict_status(
    lift: float, fold_sd: float | None, fold_lifts: Sequence[float] = ()
) -> str:
    """The chip for a model score, from its signed lift and the noise in that lift.

    ``lift`` is signed so that positive means better, as ``EvalResult.lift`` is.
    The rule, applied in this order:

    1. a lift of exactly zero ties;
    2. a negative lift loses, whatever the noise, because the model's own
       verdict then says to use the baseline and the chip must not soften it;
    3. a positive lift is inside the noise when the mean of the per-fold lifts
       over the size sort is smaller than their standard deviation, the stricter
       test the warranted multiple's chip applies, so the scoreboard compares
       like with like; with fewer than two fold lifts it falls back to the lift
       against the fold standard deviation of the model's own AUC;
    4. anything else beats the baseline, including a positive lift with no
       fold dispersion to judge it against.
    """
    if lift == 0:
        return "ties"
    if lift < 0:
        return "loses"
    if len(fold_lifts) >= 2:
        return "inside_noise" if fmean(fold_lifts) < stdev(fold_lifts) else "beats"
    if fold_sd is not None and abs(lift) < fold_sd:
        return "inside_noise"
    return "beats"


# --------------------------------------------------------------------------- #
# shape: numbers in, page out
# --------------------------------------------------------------------------- #


def _figure(kind: str, title: str, subtitle: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "title": title, "subtitle": subtitle, "data": data}


def _pct(v: float, dp: int = 1) -> str:
    return f"{v * 100:.{dp}f}%"


def _month(d: date) -> str:
    return f"{d.strftime('%b')} {d.year}"


def _day(d: date) -> str:
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _moves(delta: float) -> str:
    if delta > 0:
        return f"rises {delta:.3f}"
    if delta < 0:
        return f"falls {abs(delta):.3f}"
    return "does not move"


def _one_in(p: float) -> str:
    return f"one in {1 / p:.0f}" if p > 0 else "none"


def _fold_lifts(folds: Any) -> tuple[float, ...]:
    """Model AUC less the size sort's on each test fold, or none if the folds were refused."""
    if isinstance(folds, Refused):
        return ()
    return tuple(f.model - f.size for f in folds)


def _headline(ev: Evaluation, fold_lifts: Sequence[float] = ()) -> dict[str, Any]:
    return {
        "metric": ev.metric,
        "score": ev.score,
        "baseline_name": ev.baseline_name,
        "baseline_score": ev.baseline_score,
        "lift": ev.lift,
        "n": ev.n_scored,
        "higher_is_better": ev.higher_is_better,
        "verdict_status": verdict_status(ev.lift, ev.fold_sd, fold_lifts),
        "verdict_text": ev.verdict_text,
    }


def _takeaway(facts: Facts) -> str:
    ev, sample = facts.evaluation, facts.sample
    if ev.lift > 0:
        noise = ""
        lifts = _fold_lifts(facts.folds)
        if len(lifts) >= 2:
            mean, sd = fmean(lifts), stdev(lifts)
            where = "inside" if mean < sd else "outside"
            noise = (
                f", {where} the fold noise (a mean fold lift of {mean:+.4f} against a "
                f"fold-to-fold standard deviation of {sd:.4f})"
            )
        elif ev.fold_sd is not None:
            where = "inside" if abs(ev.lift) < ev.fold_sd else "outside"
            noise = f", {where} a fold standard deviation of {ev.fold_sd:.4f}"
        first = (
            f"The screen's walk-forward AUC of {ev.score:.4f} beats sorting companies "
            f"smallest first ({ev.baseline_score:.4f}) by {ev.lift:+.4f}{noise}, "
            f"and it rests on {sample.scored_deals} companies announced as targets."
        )
    else:
        first = (
            f"The screen's walk-forward AUC of {ev.score:.4f} does not beat sorting "
            f"companies smallest first ({ev.baseline_score:.4f}), on "
            f"{sample.scored_deals} companies announced as targets."
        )
    rest = []
    coefs = facts.coefficients
    if isinstance(coefs, Coefficients):
        flipped = sum(c.flips for c in coefs.rows)
        rest.append(
            f"{flipped} of its {len(coefs.rows)} coefficients change sign between folds"
        )
    rec = facts.recovered
    if isinstance(rec, Recovered) and rec.committed.beat_baseline != rec.without_all.beat_baseline:
        rest.append(
            f"removing the {rec.n_recovered} deals a later extraction fix recovered "
            "flips the verdict"
        )
    if not rest:
        return first
    second = ", and ".join(rest)
    return f"{first} {second[0].upper()}{second[1:]}."


def _sample_figure(facts: Facts) -> dict[str, Any]:
    ev, s = facts.evaluation, facts.sample
    tiles = [
        {
            "label": "Targets scored",
            "value": s.scored_deals,
            "format": "int",
            "sub": (
                f"{s.scored_positives} positive rows among {ev.n_scored:,} scored; "
                f"{s.distinct_deals} targets labelled of the "
                f"{s.announced_deals} announced deals in the fixture"
            ),
        },
        {
            "label": "Base rate",
            "value": s.base_rate,
            "format": "pct:2",
            "sub": f"{s.positives} positives in {s.n_labelled:,} labelled company-quarters",
        },
    ]
    if isinstance(facts.coefficients, Coefficients):
        rows = facts.coefficients.rows
        tiles.append(
            {
                "label": "Coefficients that change sign",
                "value": f"{sum(c.flips for c in rows)} of {len(rows)}",
                "sub": f"across {len(facts.coefficients.fold_labels)} walk-forward fits",
            }
        )
    if ev.fold_sd is not None:
        tiles.append(
            {
                "label": "Fold standard deviation",
                "value": ev.fold_sd,
                "format": "num:4",
                "sub": f"{ev.n_folds} folds; the {ev.lift:+.4f} lift is "
                f"{abs(ev.lift) / ev.fold_sd:.0%} of it",
            }
        )
    title = (
        f"{s.scored_deals} targets stand behind an AUC computed on "
        f"{ev.n_scored:,} rows"
    )
    return {
        "kind": "tiles",
        "title": title,
        "data": {
            "tiles": tiles,
            "counts": {
                "n_labelled": s.n_labelled,
                "positives": s.positives,
                "distinct_deals": s.distinct_deals,
                "base_rate": s.base_rate,
                "n_scored": ev.n_scored,
                "scored_positives": s.scored_positives,
                "scored_deals": s.scored_deals,
                "announced_deals": s.announced_deals,
            },
        },
    }


_SERIES = [
    {"key": "model", "name": "Fitted model", "role": "model"},
    {"key": "size", "name": "Size-only sort", "role": "baseline"},
]


def _folds_figure(ev: Evaluation, folds: tuple[FoldScore, ...]) -> dict[str, Any]:
    wins = sum(f.model > f.size for f in folds)
    title = f"The model beats the size sort in {wins} of {len(folds)} test folds"
    sd = f"fold standard deviation {ev.fold_sd:.4f}" if ev.fold_sd is not None else "no fold dispersion"
    gaps = [f.model - f.size for f in folds]
    # The widest fold in each direction carries its gap on the chart.
    marked = {gaps.index(max(gaps)), gaps.index(min(gaps))}
    return _figure(
        "dot",
        title,
        f"Walk-forward AUC per test window; {sd} against a pooled lift of "
        f"{ev.lift:+.4f}; {ev.n_scored:,} scored rows",
        {
            "rows": [
                {
                    "label": f"Fold {f.index + 1}, {_month(f.test_start)} to {_month(f.test_end)}",
                    "values": {"model": f.model, "size": f.size},
                    "gap": gap,
                    "labelled": i in marked,
                    "n": f.n,
                    "positives": f.positives,
                }
                for i, (f, gap) in enumerate(zip(folds, gaps))
            ],
            "series": _SERIES,
            "format": "num:3",
            "gapLabel": "Model minus size sort",
            "gapFormat": "signed:3",
            "labelHeader": "Test fold",
            "reference": [{"value": ev.chance, "label": f"Coin toss {ev.chance:.1f}"}],
            "fold_sd": ev.fold_sd,
            "lift": ev.lift,
        },
    )


def _recovered_figure(rec: Recovered) -> dict[str, Any]:
    ds = rec.without_all.baseline_score - rec.committed.baseline_score
    dm = rec.without_all.score - rec.committed.score
    flipped = rec.committed.beat_baseline != rec.without_all.beat_baseline
    lead = f"Without the {rec.n_recovered} recovered deals the size sort {_moves(ds)} and the model {_moves(dm)}"
    title = lead + (", and the verdict flips" if flipped else "")
    doubted = rec.without_doubted
    if doubted.fold_sd is not None:
        noise = (
            "inside" if abs(doubted.lift) < doubted.fold_sd else "outside"
        ) + f" the {doubted.fold_sd:.4f} fold standard deviation"
    else:
        noise = "with no fold dispersion to judge it against"
    if doubted.lift > 0:
        after = f"leaves the model ahead by {doubted.lift:+.4f}, {noise}"
    else:
        after = f"leaves the model behind by {abs(doubted.lift):.4f}"
    note = (
        f"{rec.n_doubted} of the {rec.n_recovered} recovered deals are, on the fixture "
        "manifest's own audit, an acquirer-side document or an internal reorganisation "
        f"the extractor cannot tell from a sale. Dropping only those {rec.n_doubted} "
        f"{after}."
    )
    rows = [
        ("As committed", rec.committed),
        (f"Less the {rec.n_doubted} doubted", rec.without_doubted),
        (f"Less all {rec.n_recovered} recovered", rec.without_all),
    ]
    return _figure(
        "dot",
        title,
        "Walk-forward AUC refitted on three label sets, each counted in announced "
        "deals, the recovered deals as named in the fixture manifest; the label is "
        "the model's lift",
        {
            "rows": [
                {
                    "label": f"{label}, {refit.events} announced deals",
                    "values": {"model": refit.score, "size": refit.baseline_score},
                    # The sign of the lift is the verdict, so every row carries it.
                    "gap": refit.score - refit.baseline_score,
                    "labelled": True,
                    "lift": refit.lift,
                    "fold_sd": refit.fold_sd,
                    "beats": refit.beat_baseline,
                }
                for label, refit in rows
            ],
            "series": _SERIES,
            "format": "num:3",
            "tableFormat": "num:4",
            "gapLabel": "Model minus size sort",
            "gapFormat": "signed:4",
            "labelHeader": "Label set",
            "notes": [note],
        },
    )


def _coefficients_figure(coefs: Coefficients) -> dict[str, Any]:
    # Rows that change sign first, each group from least to most stable.
    ordered = sorted(coefs.rows, key=lambda c: (not c.flips, -c.sd, c.feature))
    flipped = [c for c in ordered if c.flips]
    annotate = flipped[0].feature if flipped else None
    first, last = coefs.fold_train[0], coefs.fold_train[-1]
    return _figure(
        "heat",
        f"{len(flipped)} of {len(coefs.rows)} coefficients change sign between folds",
        f"Standardised logistic coefficient per walk-forward fit, {coefs.fold_labels[0]} "
        f"trained on {first:,} rows and {coefs.fold_labels[-1]} on {last:,}",
        {
            "rows": [c.feature for c in ordered],
            "cols": list(coefs.fold_labels),
            "values": [list(c.folds) for c in ordered],
            "scale": "diverging",
            "format": "signed:2",
            "scaleLabel": "Standardised coefficient",
            "valueLabel": "Coefficient",
            "rowHeader": "Feature",
            "breaks": list(SIGN_BREAKS),
            "flips": [c.flips for c in ordered],
            "mean": [c.mean for c in ordered],
            "sd": [c.sd for c in ordered],
            "annotate": annotate,
            "n_flipped": len(flipped),
            "groups": [
                {"label": f"Changes sign, {len(flipped)}", "count": len(flipped)},
                {
                    "label": f"Same sign in every fit, {len(ordered) - len(flipped)}",
                    "count": len(ordered) - len(flipped),
                },
            ],
        },
    )


def _base_rates_figure(years: tuple[YearRate, ...], sample: Sample) -> dict[str, Any]:
    full = max(y.dates for y in years)
    whole = [y for y in years if y.dates == full]
    lo = min(whole, key=lambda y: y.rate)
    hi = max(whole, key=lambda y: y.rate)
    ratio = hi.rate / lo.rate if lo.rate > 0 else float("inf")
    if ratio >= 2:
        verb = "more than doubles"
    else:
        verb = f"moves by a factor of {ratio:.1f}"
    if hi.year > lo.year:
        title = f"The base rate {verb} from {lo.year} ({_pct(lo.rate, 2)}) to {hi.year} ({_pct(hi.rate, 2)})"
    else:
        title = f"The base rate falls from {hi.year} ({_pct(hi.rate, 2)}) to {lo.year} ({_pct(lo.rate, 2)})"
    partial = [y for y in years if y.dates < full]
    subtitle = (
        "Share of labelled company-quarters announced as a target within 12 months, "
        "by feature year"
    )
    if partial:
        subtitle += "; " + ", ".join(
            f"{y.year} holds {y.dates} screen date{'s' if y.dates != 1 else ''} of {full}"
            for y in partial
        ) + ", its later windows still open"
    return _figure(
        "column",
        title,
        subtitle,
        {
            "rows": [
                {
                    "label": str(y.year),
                    "value": y.rate,
                    "role": "alt",
                    # The title quotes these two years, so the chart labels them.
                    "labelled": y is lo or y is hi,
                    "n": y.n,
                    "positives": y.positives,
                    "dates": y.dates,
                }
                for y in years
            ],
            "format": "pct:2",
            "valueLabel": "Base rate",
            "labelHeader": "Year",
            "reference": [
                {"value": sample.base_rate, "label": f"Pooled {_pct(sample.base_rate, 2)}"}
            ],
        },
    )


def _calibration_figure(cal: Calibration) -> dict[str, Any]:
    solid = [b for b in cal.buckets if not b.thin]
    over_from = None
    for b in sorted(solid, key=lambda b: b.low, reverse=True):
        if b.gap < 0:
            over_from = b.low
        else:
            break
    worst = min(solid, key=lambda b: b.gap) if solid else None
    thin_over = [b for b in cal.buckets if b.thin and b.gap < 0]
    deeper = (
        f"; thin buckets run to {abs(min(b.gap for b in thin_over)) * 100:.0f}"
        if thin_over and worst is not None and min(b.gap for b in thin_over) < worst.gap
        else ""
    )
    if over_from is not None and worst is not None and worst.gap < 0:
        where = f"above {over_from:.0%}" if over_from > 0 else "at every level"
        title = (
            f"Every well-populated bucket {where} over-promises, by up to "
            f"{abs(worst.gap) * 100:.0f} points{deeper}"
        )
    elif worst is not None and worst.gap < 0:
        title = (
            f"The well-populated buckets over-promise by up to "
            f"{abs(worst.gap) * 100:.0f} points{deeper}"
        )
    else:
        title = "No well-populated bucket over-promises"
    return _figure(
        "dot",
        title,
        "Out-of-sample stated probability against the share announced as a target "
        "within 12 months, "
        f"by bucket; hollow marks hold fewer than {cal.thin_below} rows; n = {cal.n:,}",
        {
            "rows": [
                {
                    "label": f"{b.low:.0%} to {b.high:.0%}",
                    "values": {"stated": b.stated, "realised": b.realised},
                    "n": b.n,
                    "thin": b.thin,
                    # First series less second, as every dumbbell here: positive
                    # is a promise the world did not keep. calibration_table's
                    # own gap is the same number with the opposite sign.
                    "gap": b.stated - b.realised,
                    "model_gap": b.gap,
                    "labelled": (not b.thin) and b.low >= LOUD,
                }
                for b in cal.buckets
            ],
            "series": [
                {"key": "stated", "name": "Stated probability", "role": "model"},
                {"key": "realised", "name": "Share acquired", "role": "alt"},
            ],
            "format": "pct:0",
            "tableFormat": "pct:1",
            "gapLabel": "Stated minus realised",
            "gapFormat": "points",
            "labelHeader": "Stated probability",
            "domain": [0, 1],
            "showCounts": True,
            "thin_below": cal.thin_below,
        },
    )


def _precision_figures(top: TopK) -> tuple[dict[str, Any], dict[str, Any]]:
    precision = _figure(
        "hbar",
        f"{_one_in(top.precision_model).capitalize()} on the model's top {top.k} was "
        f"bought within a year, {_one_in(top.precision_size)} on the size sort's",
        f"Precision at {top.k}, out of sample, mean over {top.dates} screen dates; the "
        f"rule is the mean base rate, what {top.k} random names would score",
        {
            "rows": [
                {"label": "Fitted model", "value": top.precision_model, "role": "model"},
                {"label": "Size-only sort", "value": top.precision_size, "role": "baseline"},
            ],
            "format": "pct:1",
            "valueLabel": f"Precision at {top.k}",
            "labelHeader": "Screen",
            "legend": [],
            "reference": [{"value": top.base_rate, "label": f"Base rate {_pct(top.base_rate)}"}],
            "k": top.k,
            "dates": top.dates,
            "aggregation": "per-date mean",
        },
    )
    if top.recall_size < top.random_recall:
        title = (
            f"The size sort's top {top.k} catch fewer of each date's targets than "
            f"{top.k} random names would"
        )
    else:
        title = (
            f"The model's top {top.k} catch {_pct(top.recall_model)} of each date's "
            f"targets, the size sort's {_pct(top.recall_size)}"
        )
    recall = _figure(
        "hbar",
        title,
        f"Recall at {top.k}, out of sample, mean over {top.dates} screen dates; the rule "
        f"is {top.k} over each date's scored cross-section, what a random list recovers",
        {
            "rows": [
                {"label": "Fitted model", "value": top.recall_model, "role": "model"},
                {"label": "Size-only sort", "value": top.recall_size, "role": "baseline"},
            ],
            "format": "pct:1",
            "valueLabel": f"Recall at {top.k}",
            "labelHeader": "Screen",
            "legend": [],
            "reference": [
                {"value": top.random_recall, "label": f"{top.k} random names {_pct(top.random_recall)}"}
            ],
            "k": top.k,
            "dates": top.dates,
            "aggregation": "per-date mean",
        },
    )
    return precision, recall


def _vertical(name: str | None) -> str:
    return name.replace("_", " ") if name else "unclassified"


def _price_note(price: PriceLeak, ev: Evaluation) -> str:
    served = "; ".join(
        f"{source.capitalize()} served {rows:,} days of prices for {sym}"
        for sym, source, rows in price.served
    )
    unserved = ", ".join(price.unserved)
    probe = (
        f" In the recorded probe {served}, and no source served a single day for {unserved}."
        if price.served and price.unserved
        else ""
    )
    return (
        f"None of the {price.fitted_columns} fitted columns uses a price, so this list "
        "cannot say whether cheap companies get bought, and a price column could not "
        "have been added honestly. On the scored rows, whether a company has since left "
        f"the filing record separates targets with an AUC of {price.auc:.4f} on its own, "
        f"against {ev.score:.4f} for the model: {price.positives_departed:.0%} of "
        f"positive rows are on companies that left, against "
        f"{price.negatives_departed:.0%} of negative rows.{probe} A price feature's gaps "
        "would carry that separation into the fit."
    )


def _screen_figure(screen: Screen, calibration: Calibration | Refused, price: PriceLeak | Refused | None, ev: Evaluation) -> dict[str, Any]:
    counts = Counter(_vertical(r.sub_vertical) for r in screen.rows)
    vertical, count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    title = f"{count} of the top {len(screen.rows)} on {_day(screen.as_of)} are {vertical}"
    subtitle = (
        f"Final model fitted on every labelled row, ranked across {screen.n_ranked:,} "
        "registrants with no deal announced by that date; the last column names the "
        "three features contributing most to each score"
    )
    if isinstance(calibration, Calibration):
        loud = [b for b in calibration.buckets if not b.thin and b.low >= LOUD]
        if loud:
            subtitle += (
                f". Out of sample, in buckets of at least {calibration.thin_below} rows, "
                f"stated probabilities of {min(b.low for b in loud):.0%} or more were "
                f"realised at {_pct(min(b.realised for b in loud))} to "
                f"{_pct(max(b.realised for b in loud))}"
            )
    notes = [_price_note(price, ev)] if isinstance(price, PriceLeak) else []
    return _figure(
        "hbar",
        title,
        subtitle,
        {
            "rows": [
                {"label": r.ticker, "value": r.probability, "role": "model"} for r in screen.rows
            ],
            "format": "pct:1",
            "valueLabel": "Stated probability",
            "labelHeader": "Ticker",
            "table": [
                {
                    "rank": r.rank,
                    "ticker": r.ticker,
                    "name": r.name,
                    "sub_vertical": _vertical(r.sub_vertical),
                    "probability": r.probability,
                    "drivers": ", ".join(r.drivers),
                }
                for r in screen.rows
            ],
            "as_of": screen.as_of,
            "notes": notes,
        },
    )


def shape(facts: Facts) -> dict[str, Any]:
    """The section as the collector returns it, from the numbers alone.

    Every component that is ``Refused`` becomes a refusal naming it and no
    figure. Nothing here reads a file, fits a model or knows a number it was
    not handed.
    """
    refusals: list[dict[str, str]] = []
    figures: dict[str, dict[str, Any]] = {"sample": _sample_figure(facts)}

    def refuse(item: Refused) -> None:
        refusals.append({"what": item.what, "why": item.why})

    ev = facts.evaluation
    if isinstance(facts.folds, Refused):
        refuse(facts.folds)
    else:
        figures["auc_by_fold"] = _folds_figure(ev, facts.folds)
    if isinstance(facts.recovered, Refused):
        refuse(facts.recovered)
    else:
        figures["recovered_deals"] = _recovered_figure(facts.recovered)
    if isinstance(facts.coefficients, Refused):
        refuse(facts.coefficients)
    else:
        figures["coefficient_signs"] = _coefficients_figure(facts.coefficients)
    if isinstance(facts.base_rates, Refused):
        refuse(facts.base_rates)
    else:
        figures["base_rate_by_year"] = _base_rates_figure(facts.base_rates, facts.sample)
    if isinstance(facts.calibration, Refused):
        refuse(facts.calibration)
    else:
        figures["calibration"] = _calibration_figure(facts.calibration)
    if isinstance(facts.top_k, Refused):
        refuse(facts.top_k)
    else:
        figures["precision_at_k"], figures["recall_at_k"] = _precision_figures(facts.top_k)
    if isinstance(facts.screen, Refused):
        refuse(facts.screen)
    else:
        figures["target_list"] = _screen_figure(facts.screen, facts.calibration, facts.price, ev)
    if isinstance(facts.price, Refused):
        refuse(facts.price)

    return {
        "status": "ok",
        "takeaway": _takeaway(facts),
        "refusals": refusals,
        "headline": _headline(ev, _fold_lifts(facts.folds)),
        "figures": figures,
    }


# --------------------------------------------------------------------------- #
# Reading the fit
# --------------------------------------------------------------------------- #


def _as_of(manifest: dict[str, Any]) -> date:
    raw = manifest.get("as_of")
    if not raw:
        raise ConfigError(
            f"{MANIFEST} carries no as_of date, and the label windows cannot be "
            "resolved against a date the fixtures do not state"
        )
    return date.fromisoformat(str(raw))


def _evaluation(result) -> Evaluation:
    ev = result.evaluation
    name, _, _ = ev.baseline_name.partition(", walk-forward")
    return Evaluation(
        metric=f"walk-forward {ev.metric.upper()}",
        score=float(ev.score),
        baseline_name=name,
        baseline_score=float(ev.baseline_score),
        lift=float(ev.lift),
        fold_sd=ev.fold_sd,
        n_scored=int(ev.n_observations),
        n_folds=len(ev.folds),
        higher_is_better=bool(ev.higher_is_better),
        beat_baseline=bool(ev.beat_baseline),
        # The baseline's full name carries its own score for a reader of the model
        # card. The verdict states that score already, so the page trims the name
        # the same way the tile beside it is trimmed.
        verdict_text=result.verdict().replace(ev.baseline_name, name),
        chance=float(result.base_rate_result.baseline_score),
    )


def _scored(result) -> np.ndarray:
    return np.isfinite(result.out_of_sample)


def _sample(result, events: list) -> Sample:
    labels = result.labels
    scored = _scored(result)
    y = result.matrix.y
    tickers = result.matrix.tickers
    return Sample(
        n_labelled=len(labels.observations),
        positives=int(labels.n_positive),
        distinct_deals=int(labels.n_distinct_deals),
        base_rate=float(labels.base_rate),
        scored_positives=int(y[scored].sum()),
        scored_deals=len({t for t, keep, lab in zip(tickers, scored, y) if keep and lab}),
        announced_deals=len(events),
    )


def _fold_scores(result) -> tuple[FoldScore, ...]:
    """The model's and the size sort's AUC on each test fold, from the evaluation harness.

    The model's fold AUCs are the ones ``fit_propensity`` already reported. The
    size sort's are not kept on the result, so the same harness is run on the
    size scores over the same folds. Both lists skip a fold with no event in
    its test window, so the folds are matched to them by the same test here.
    """
    from ...ml.evaluation import evaluate_classification

    scored = _scored(result)
    dates = [d for d, keep in zip(result.matrix.dates, scored) if keep]
    y = result.matrix.y[scored]
    ordinals = np.asarray([d.toordinal() for d in dates])
    kept = [
        f for f in result.folds if any(f.test_start <= d <= f.test_end for d in dates)
    ]
    size = evaluate_classification(
        y,
        result.size_scores[scored],
        dates=dates,
        folds=kept,
        embargo_days=kept[0].embargo_days if kept else 0,
    )
    eventful = []
    for fold in kept:
        idx = (ordinals >= fold.test_start.toordinal()) & (ordinals <= fold.test_end.toordinal())
        events = int(y[idx].sum())
        if 0 < events < int(idx.sum()):
            eventful.append((fold, int(idx.sum()), events))
    model_folds = list(result.evaluation.folds)
    if not (len(eventful) == len(model_folds) == len(size.folds)):
        raise ValueError(
            f"{len(eventful)} eventful folds against {len(model_folds)} model and "
            f"{len(size.folds)} size-sort fold scores"
        )
    if len(eventful) < 2:
        raise NotMeaningfulError(
            f"{len(eventful)} test fold holds an event, so there is no fold-to-fold "
            "comparison to draw"
        )
    return tuple(
        FoldScore(
            index=fold.index,
            test_start=fold.test_start,
            test_end=fold.test_end,
            n=n,
            positives=events,
            model=float(m),
            size=float(s),
        )
        for (fold, n, events), m, s in zip(eventful, model_folds, size.folds)
    )


def _coefficients(result) -> Coefficients:
    stability = result.coefficient_stability()
    if stability.empty:
        raise NotMeaningfulError("no walk-forward fold was fitted, so no coefficient can be compared")
    weights = np.array([m.weights for _, m in result.fold_models], dtype=float)
    columns = list(result.model.columns)
    rows = []
    for rec in stability.to_dict("records"):
        j = columns.index(rec["feature"])
        rows.append(
            Coefficient(
                feature=str(rec["feature"]),
                folds=tuple(float(w) for w in weights[:, j]),
                mean=float(rec["mean"]),
                sd=float(rec["sd"]),
                flips=int(rec["sign_flips"]) > 0,
            )
        )
    return Coefficients(
        fold_labels=tuple(f"Fold {fold.index + 1}" for fold, _ in result.fold_models),
        fold_train=tuple(int(fold.n_train) for fold, _ in result.fold_models),
        rows=tuple(rows),
    )


def _base_rates(result) -> tuple[YearRate, ...]:
    frame = result.base_rates
    if frame is None or frame.empty:
        raise NotMeaningfulError("the label report holds no observations to take a base rate over")
    dates: dict[int, set[date]] = {}
    for obs in result.labels.observations:
        dates.setdefault(obs.as_of.year, set()).add(obs.as_of)
    return tuple(
        YearRate(
            year=int(rec["year"]),
            n=int(rec["n"]),
            positives=int(rec["positives"]),
            rate=float(rec["base_rate"]),
            dates=len(dates[int(rec["year"])]),
        )
        for rec in frame.to_dict("records")
    )


def _calibration(result) -> Calibration:
    from ...ml.evaluation import THIN_BIN_OBSERVATIONS

    frame = result.calibration
    if frame is None:
        raise NotMeaningfulError(
            "fit_propensity returned no calibration table: the scored sample is below "
            "the harness floor, so the probabilities may be ranked but not read"
        )
    buckets = tuple(
        Bucket(
            low=float(rec["low"]),
            high=float(rec["high"]),
            n=int(rec["n"]),
            stated=float(rec["predicted"]),
            realised=float(rec["realised"]),
            gap=float(rec["gap"]),
            thin=bool(rec["thin"]),
        )
        for rec in frame.to_dict("records")
        if int(rec["n"]) > 0
    )
    return Calibration(
        buckets=buckets,
        thin_below=int(THIN_BIN_OBSERVATIONS),
        n=int(sum(b.n for b in buckets)),
    )


def _top_k(result) -> TopK:
    frame = result.precision_at_k
    if frame is None or frame.empty:
        raise NotMeaningfulError("no screen date held a positive, so precision at k is undefined")
    ks = sorted({int(k) for k in frame["k"]})
    if len(ks) != 1:
        raise NotMeaningfulError(
            f"the screens were cut at different depths ({', '.join(map(str, ks))}), so "
            "a mean precision would average lists of different lengths"
        )
    return TopK(
        k=ks[0],
        dates=len(frame),
        precision_model=float(frame["precision_at_k_model"].mean()),
        precision_size=float(frame["precision_at_k_size"].mean()),
        base_rate=float(frame["base_rate"].mean()),
        recall_model=float(frame["recall_at_k_model"].mean()),
        recall_size=float(frame["recall_at_k_size"].mean()),
        random_recall=float((frame["k"] / frame["n"]).mean()),
    )


def _refit(result, events: int) -> Refit:
    ev = result.evaluation
    return Refit(
        events=events,
        score=float(ev.score),
        baseline_score=float(ev.baseline_score),
        lift=float(ev.lift),
        fold_sd=ev.fold_sd,
        beat_baseline=bool(ev.beat_baseline),
    )


def _recovered_events(manifest: dict[str, Any], events: list) -> tuple[list, list]:
    """The events the manifest's audit names as recovered: (all seven, the doubted three).

    Raises ``MissingDataError`` when the audit or the events no longer agree
    with the names held in this module, so the figure is refused rather than
    drawn from a subset that is not the one the audit describes.
    """
    audit = (((manifest.get("files") or {}).get("events.json") or {}).get("audit")) or ""
    hint = f"{MANIFEST} files.events.json.audit"
    match = _AUDIT_COUNT.search(audit)
    named = len(RECOVERED_SALES) + len(RECOVERED_DOUBTED)
    if not match or int(match.group(1)) != named:
        raise MissingDataError(
            f"an audit naming {named} recovered transactions",
            hint=f"{hint} does not say how many transactions the dual-class fix recovered, or says a different number",
        )
    head, split, tail = audit.partition(_AUDIT_SPLIT)
    if not split:
        raise MissingDataError("the audit's split between sales and doubted deals", hint=hint)
    for names, part, group in ((RECOVERED_SALES, head, "sales"), (RECOVERED_DOUBTED, tail, "doubted deals")):
        missing = [n for n in names if n not in part]
        if missing:
            raise MissingDataError(
                f"the recovered {group} {', '.join(missing)}",
                hint=f"{hint} no longer names them where this section expects",
            )

    def find(name: str):
        hits = [e for e in events if e.name.upper().startswith(name.upper())]
        if len(hits) != 1:
            raise MissingDataError(
                f"one deal event named {name!r}",
                hint=f"{EVENTS} holds {len(hits)} events whose name starts with it",
            )
        return hits[0]

    sales = [find(n) for n in RECOVERED_SALES]
    doubted = [find(n) for n in RECOVERED_DOUBTED]
    return sales + doubted, doubted


def _price_leak(result, universe, probe: dict[str, Any]) -> PriceLeak:
    from ...ml.evaluation import evaluate_classification

    scored = _scored(result)
    members = universe.by_ticker
    y = result.matrix.y[scored]
    departed = np.array(
        [
            1.0 if members[t].departed is not None else 0.0
            for t, keep in zip(result.matrix.tickers, scored)
            if keep
        ]
    )
    pooled = evaluate_classification(y, departed)
    served: list[tuple[str, str, int]] = []
    unserved: list[str] = []
    for symbol, sources in sorted((probe.get("results") or {}).items()):
        rows = {name: int((got or {}).get("rows") or 0) for name, got in sources.items()}
        hits = [(symbol, name, n) for name, n in sorted(rows.items()) if n > 0]
        if hits:
            served.extend(hits)
        else:
            unserved.append(symbol)
    return PriceLeak(
        auc=float(pooled.score),
        positives_departed=float(departed[y == 1].mean()),
        negatives_departed=float(departed[y == 0].mean()),
        served=tuple(served),
        unserved=tuple(unserved),
        fitted_columns=len(result.model.columns),
    )


def _screen(result, panel, universe, events, extras) -> Screen:
    from ...ml.mna import DEFAULT_TOP_K

    latest = max(panel.dates)
    # Ranked in full, as ``techval targets`` does, so the count of names the
    # list was drawn from is the model's own filter and not a copy of it.
    full = result.model.rank(
        panel,
        universe,
        latest,
        top_k=len(universe.members),
        extras=extras,
        events=events,
    )
    frame = full.head(DEFAULT_TOP_K)
    rows = tuple(
        ScreenRow(
            rank=int(rec["rank"]),
            ticker=str(rec["ticker"]),
            name=str(rec["name"]),
            sub_vertical=rec["sub_vertical"],
            probability=float(rec["probability"]),
            drivers=tuple(str(rec[c]) for c in ("driver_1", "driver_2", "driver_3") if rec[c]),
        )
        for rec in frame.to_dict("records")
    )
    return Screen(as_of=latest, n_ranked=len(full), rows=rows)


def _attempt(ctx, figure: str, entry_point: str, inputs: list[str], what: str, compute):
    """Run one computation under its provenance record, or turn a refusal into ``Refused``.

    A block that raises leaves no provenance row, which is what keeps the
    record and the figures in step: a refused figure is neither drawn nor
    cited.
    """
    try:
        with ctx.record(figure, entry_point, inputs):
            return compute()
    except TechvalError as exc:
        return Refused(what, str(exc))


def collect(ctx) -> dict[str, Any]:
    from ...commands_mna import load_dataset
    from ...ml.mna import fit_propensity

    manifest = json.loads(ctx.input(MANIFEST).read_text(encoding="utf-8"))
    as_of = _as_of(manifest)
    directory = ctx.input(UNIVERSE).parent
    for name in (EVENTS, PANEL):
        if ctx.input(name).parent != directory:
            raise ValueError(f"{name} is not beside {UNIVERSE}, which load_dataset requires")

    # The fit every figure reads. A refusal here refuses the section.
    with ctx.record("sample", "techval.ml.mna.fit_propensity", FIT_INPUTS):
        panel, universe, events, extras = load_dataset(directory)
        result = fit_propensity(
            panel, universe, events, ctx.assumptions, as_of=as_of, extras=extras
        )
        evaluation = _evaluation(result)
        sample = _sample(result, events)

    folds = _attempt(
        ctx, "auc_by_fold", "techval.ml.evaluation.evaluate_classification", FIT_INPUTS,
        "AUC by fold", lambda: _fold_scores(result),
    )

    def recovered() -> Recovered:
        seven, doubted = _recovered_events(manifest, events)

        def refit(drop: list) -> Refit:
            kept = [e for e in events if e not in drop]
            fitted = fit_propensity(
                panel, universe, kept, ctx.assumptions, as_of=as_of, extras=extras
            )
            return _refit(fitted, len(kept))

        return Recovered(
            committed=_refit(result, len(events)),
            without_doubted=refit(doubted),
            without_all=refit(seven),
            n_recovered=len(seven),
            n_doubted=len(doubted),
        )

    recovered_deals = _attempt(
        ctx, "recovered_deals", "techval.ml.mna.fit_propensity", FIT_INPUTS,
        "The deals the dual-class fix recovered", recovered,
    )
    coefficients = _attempt(
        ctx, "coefficient_signs", "techval.ml.mna.PropensityResult.coefficient_stability",
        FIT_INPUTS, "Coefficient signs across folds", lambda: _coefficients(result),
    )
    base_rates = _attempt(
        ctx, "base_rate_by_year", "techval.ml.mna.base_rate_by_year", FIT_INPUTS,
        "Base rate by year", lambda: _base_rates(result),
    )
    calibration = _attempt(
        ctx, "calibration", "techval.ml.evaluation.calibration_table", FIT_INPUTS,
        "Calibration", lambda: _calibration(result),
    )
    top_k = _top_k_attempt(ctx, result)
    screen = _attempt(
        ctx, "target_list", "techval.ml.mna.PropensityModel.rank", FIT_INPUTS,
        "The target list", lambda: _screen(result, panel, universe, events, extras),
    )
    price: PriceLeak | Refused | None = None
    if isinstance(screen, Screen):
        probe = json.loads(ctx.input(PRICES).read_text(encoding="utf-8"))
        price = _attempt(
            ctx, "target_list", "techval.ml.evaluation.evaluate_classification",
            [UNIVERSE, EVENTS, PANEL, MANIFEST, PRICES],
            "What the target list cannot see", lambda: _price_leak(result, universe, probe),
        )

    section = shape(
        Facts(
            evaluation=evaluation,
            sample=sample,
            folds=folds,
            recovered=recovered_deals,
            coefficients=coefficients,
            base_rates=base_rates,
            calibration=calibration,
            top_k=top_k,
            screen=screen,
            price=price,
        )
    )
    cited = {row["figure"] for row in ctx.provenance}
    if cited != set(section["figures"]):
        raise ValueError(
            f"provenance cites {sorted(cited)} and the section drew {sorted(section['figures'])}"
        )
    return section


def _top_k_attempt(ctx, result) -> TopK | Refused:
    """Precision and recall come from one table and are cited as the two figures they draw."""
    entry = "techval.ml.evaluation.precision_recall_at_k"
    try:
        with ctx.record("precision_at_k", entry, FIT_INPUTS), ctx.record(
            "recall_at_k", entry, FIT_INPUTS
        ):
            return _top_k(result)
    except TechvalError as exc:
        return Refused("Precision and recall at k", str(exc))
