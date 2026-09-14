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

The part that turns those results into figures is ``shape``, a pure function of
an ``Inputs`` record, so the page can be tested on small fakes without a fit.
Each figure builder either returns a figure or raises ``FigureRefused`` with a
reason, and ``shape`` records provenance only for the figures that exist.
"""

from __future__ import annotations

import math
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, ContextManager

ID = "warranted"
TITLE = "Warranted multiple"
PANEL = "warranted/observations.json.gz"
INPUTS: list[str] = [PANEL]

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
        return (
            first + " The differenced score could not be computed, so this page "
            "cannot say how much of that ranking is company identity."
        )
    if inp.change < IDENTITY_WORDING_THRESHOLD:
        reading = "so almost all of the ranking is company identity inherited from history."
    else:
        reading = "so a real part of the ranking survives differencing."
    return (
        f"{first} Differenced against each company's own previous observation it "
        f"predicts the change at {_signed(inp.change, 4)} on {inp.n_changes:,}, {reading}"
    )


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
    return _figure(
        "tiles",
        "The panel behind every figure here",
        "",
        {
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
        },
    )


# (figure id, entry point that computed it, builder), in page order.
FIGURES: tuple[tuple[str, str, Callable[[Inputs], dict[str, Any]]], ...] = (
    ("deflation", ENTRY_FIT, build_deflation),
    ("baselines", ENTRY_FIT, build_baselines),
    ("folds", ENTRY_EVALUATE, build_folds),
    ("noise", ENTRY_EVALUATE, build_noise),
    ("screen", ENTRY_EXTREMES, build_screen),
    ("rerating", ENTRY_FIT, build_rerating),
    ("panel", ENTRY_LOAD, build_panel),
)


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


def collect(ctx) -> dict:
    from ...commands_peers import _load_observations
    from ...ml.warranted import fit_warranted

    panel = _load_observations(ctx.input(PANEL))
    model = fit_warranted(panel, ctx.assumptions, model="mlp")
    inputs = extract(panel, model, ctx.assumptions)
    return shape(inputs, record=lambda fid, entry: ctx.record(fid, entry, [PANEL]))
