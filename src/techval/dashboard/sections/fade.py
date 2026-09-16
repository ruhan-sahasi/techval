"""Revenue fade: the growth schedule a DCF types in, against one fitted on filings.

The DCF's most consequential typed-in input is the growth schedule, and
``techval.ml.forecast`` replaces it with one fitted on 10-K filings. What a reader
needs from this section is the comparison, so every figure puts the fitted fade
beside the thing it would replace: persistence and two sector means at every
horizon, the ``assumptions.dcf`` straight line on the curve, and both paths on one
real company.

Collection is split in two. ``collect`` builds the panel from the committed blob,
fits the model and values Datadog, and reduces what it finds to the plain records
defined below. ``shape`` turns those records into the section. It fits nothing and
reads nothing, so it can be tested on small fakes, and the page cannot hold a
number ``collect`` did not hand it.

**The verdict rule.** ``verdict_status`` calls a lift a tie when its size is under
the fold standard deviation, whichever way it points, and a win or a loss by its
sign otherwise. That is the comparison ``EvalResult.verdict`` prints, and the rule
``techval fade`` applies to persistence in its horizon table.

**What is refused, and why.** The DCF under each path needs a risk-free rate. The
engine reads the ten-year Treasury yield from a live feed, the committed fixtures
record none, and the valuation tests pin one by hand in ``tests/conftest.py``. So
the per-share values are drawn only when the assumptions handed to the collection
set ``market.risk_free_rate``, and are refused with that reason otherwise. The
growth paths, and the revenue they compound to, need no market input and are drawn
either way.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any, Sequence

from ...commands_forecast import _recorded_panel, _truncate_panel
from ...dcf import _fade, project
from ...edgar import CompanyFacts, HttpCache
from ...errors import TechvalError
from ...ev_bridge import build_ev_bridge
from ...financials import build_financials
from ...market import CsvSource, MarketData
from ...ml.forecast import compare_fade, fit_fade, is_delisted
from ...wacc import compute_wacc

ID = "fade"
TITLE = "Revenue fade"

PANEL = "fade_companyfacts.json.gz"
COMPANY = "DDOG"
COMPANY_NAME = "Datadog"
COMPANY_FACTS = f"companyfacts_{COMPANY}.json"
PRICES = "prices"
INPUTS: list[str] = [PANEL, COMPANY_FACTS, PRICES]

# The refusal of the per-share values names this, and the renderer hangs that
# refusal under the revenue tiles by matching it, so the note sits beside the
# figures it would have joined.
DCF_REFUSAL = f"{COMPANY_NAME}'s DCF value on each path"

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


# --------------------------------------------------------------------------- #
# what collect hands to shape
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Refusal:
    what: str
    why: str


@dataclass(frozen=True)
class HorizonScore:
    """One horizon's out-of-fold mean absolute errors of growth, lower is better."""

    horizon: int
    model: float
    persistence: float
    training_mean: float
    sub_vertical_mean: float
    fold_sd: float | None
    n: int
    verdict_text: str
    fold_lifts: tuple[float, ...] = ()


@dataclass(frozen=True)
class Curve:
    """The one-year persistence regression, and the typed schedule it is set against."""

    start: float
    slope: float
    intercept: float
    reversion_level: float
    half_life_years: float
    typed: tuple[float, ...]
    n: int


@dataclass(frozen=True)
class Survivorship:
    """How much higher forward growth looks without the filers that left, by horizon."""

    leavers: int
    filers: int
    gaps: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class CompanyPaths:
    ticker: str
    name: str
    trailing: float | None
    fiscal_year_end: str
    filed: str
    typed: tuple[float, ...]
    fitted: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    basis: tuple[str, ...]
    terminal: float


@dataclass(frozen=True)
class CompanyRevenue:
    """Final projection year revenue, USD millions, compounded on each path."""

    trailing: float
    trailing_as_of: str
    typed: float
    fitted: float
    low: float
    high: float


@dataclass(frozen=True)
class Valuation:
    """One DCF run on each growth path, USD per share and USD millions.

    Every path is discounted at the same cost of capital, so the gap between
    them is the growth path and nothing else. That cost of capital is built from
    Datadog's own regression beta, the construction ``tests/ml/test_forecast.py``
    pins, which is not the peer-median beta the engine section values on. The
    rate, the beta and the risk-free rate travel with the figure so a reader
    comparing the two sections sees why their per-share values differ.
    """

    per_share_typed: float
    per_share_fitted: float
    per_share_low: float
    per_share_high: float
    ev_typed: float
    ev_fitted: float
    price_date: str
    wacc: float
    beta: float
    risk_free_rate: float


@dataclass(frozen=True)
class Depth:
    """Fiscal years of revenue each filer supports, and the panel they build."""

    years: dict[int, int]
    observations: int
    filers_with_rows: int
    unbuilt: tuple[str, ...]
    first_filed: int
    last_filed: int


@dataclass(frozen=True)
class Results:
    """Everything ``shape`` draws. ``None`` is a part an earlier refusal already covers."""

    horizons: tuple[HorizonScore, ...]
    curve: Curve | Refusal
    survivorship: Survivorship
    depth: Depth
    paths: CompanyPaths | Refusal
    revenue: CompanyRevenue | Refusal | None
    valuation: Valuation | Refusal | None


# --------------------------------------------------------------------------- #
# the verdict rule
# --------------------------------------------------------------------------- #


def verdict_status(
    lift: float, fold_sd: float | None, fold_lifts: Sequence[float] = ()
) -> str:
    """``ties`` inside the fold noise, otherwise ``beats`` or ``loses`` by the sign.

    ``lift`` is signed so that positive means the model is better. The noise is
    judged on the lift itself: the mean of the per-fold lifts against their
    standard deviation, the same stricter test the warranted multiple's chip
    applies, so the scoreboard compares like with like. The spread of the model's
    own fold errors is the wrong yardstick for a chip, because much of it is
    variation the baseline shares. A lift inside that noise is a tie whichever
    way it points: calling a small negative lift a loss, or a small positive one
    a win, is reading a sign off noise.

    With fewer than two fold lifts the older comparison against ``fold_sd`` is
    the only one available, and with no fold spread at all the sign decides.
    """
    if len(fold_lifts) >= 2:
        return "ties" if abs(fmean(fold_lifts)) < stdev(fold_lifts) else ("beats" if lift > 0 else "loses")
    if fold_sd is not None and abs(lift) < fold_sd:
        return "ties"
    return "beats" if lift > 0 else "loses"


# --------------------------------------------------------------------------- #
# shaping: records in, section out
# --------------------------------------------------------------------------- #


def _word(n: int) -> str:
    return _WORDS.get(n, str(n))


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _years(horizons: Sequence[int]) -> str:
    """``[2, 3]`` reads "two and three years"; ``[1]`` reads "one year"."""
    return _join([_word(h) for h in horizons]) + (" year" if list(horizons) == [1] else " years")


def _check_finite(where: str, *values: float | None) -> None:
    for v in values:
        if v is None or not math.isfinite(v):
            raise ValueError(f"{where}: {v!r} is not a finite number")


def _status(h: HorizonScore) -> str:
    return verdict_status(h.persistence - h.model, h.fold_sd, h.fold_lifts)


def _strongest(h: HorizonScore) -> tuple[str, float]:
    options = {
        "persistence": h.persistence,
        "the training mean": h.training_mean,
        "the sub-vertical mean": h.sub_vertical_mean,
    }
    return min(options.items(), key=lambda kv: kv[1])


_VERB = {"ties": "ties", "beats": "beats", "loses": "loses to"}


def _against_persistence(horizons: Sequence[HorizonScore], name: str = "this year's growth") -> str:
    """"ties this year's growth at one year and beats it at two and three years"."""
    groups: list[tuple[str, list[int]]] = []
    for h in horizons:
        status = _status(h)
        if groups and groups[-1][0] == status:
            groups[-1][1].append(h.horizon)
        else:
            groups.append((status, [h.horizon]))
    phrases = []
    for i, (status, hs) in enumerate(groups):
        phrases.append(f"{_VERB[status]} {name if i == 0 else 'it'} at {_years(hs)}")
    return _join(phrases)


def _horizon_label(h: int) -> str:
    return _word(h).capitalize() + (" year" if h == 1 else " years")


def _error_figure(horizons: Sequence[HorizonScore]) -> dict[str, Any]:
    def series(name: str, role: str, key: str, band: bool = False) -> dict[str, Any]:
        values = []
        for h in horizons:
            y = getattr(h, key)
            point: dict[str, Any] = {"x": _horizon_label(h.horizon), "y": y}
            if band and h.fold_sd is not None:
                point["lo"] = y - h.fold_sd
                point["hi"] = y + h.fold_sd
            values.append(point)
        return {"name": name, "role": role, "values": values}

    return {
        "kind": "line",
        "title": f"The fitted fade {_against_persistence(horizons)}",
        "subtitle": (
            "Mean absolute error of revenue growth, out of fold; lower is better. The "
            "band is the model's score plus or minus one fold standard deviation, so a "
            "baseline inside it is inside the noise. "
            + _join([f"{h.n:,}" for h in horizons])
            + " scored company-years."
        ),
        "data": {
            "series": [
                series("Fitted fade", "model", "model", band=True),
                series("Persistence", "alt", "persistence"),
                series("Sub-vertical mean", "baseline", "sub_vertical_mean"),
                series("Training mean", "baseline", "training_mean"),
            ],
            "x": {"label": "Horizon"},
            "format": "num:4",
        },
    }


def _curve_path(curve: Curve) -> list[float]:
    """The one-year regression applied year after year from the schedule's own start."""
    path = [curve.start]
    while len(path) < len(curve.typed):
        path.append(curve.intercept + curve.slope * path[-1])
    return path


def _survivorship_note(s: Survivorship) -> dict[str, str]:
    why = (
        "A company whose growth collapses is bought or delisted and stops filing, so "
        "a curve fitted on the survivors fades too slowly and a DCF built on it is too "
        f"optimistic. The panel keeps the {s.leavers} of its {s.filers} filers that "
        "later left until their last filing."
    )
    if s.gaps:
        why += (
            " Dropping them would lift mean forward growth by "
            + _join([f"{g * 100:.1f}" for _, g in s.gaps])
            + " points at "
            + _years([h for h, _ in s.gaps])
            + "."
        )
    why += (
        " The years after they left are in no filing, so what is left of the bias "
        "points the same way."
    )
    return {"what": "Survivorship flatters a fade curve", "why": why}


def _curve_figure(curve: Curve, survivorship: Survivorship) -> dict[str, Any]:
    labels = [f"Year {i + 1}" for i in range(len(curve.typed))]
    fitted = _curve_path(curve)
    return {
        "kind": "line",
        "title": (
            f"The filings level off near {curve.reversion_level:.0%}; the typed "
            f"schedule falls to {curve.typed[-1]:.0%}"
        ),
        "subtitle": (
            f"Revenue growth by projection year, both paths starting at "
            f"{curve.start:.0%}. Fitted on {curve.n:,} company-years: next year's "
            f"growth is {curve.intercept * 100:.1f} points plus {curve.slope:.2f} "
            f"times this year's, so half the gap to {curve.reversion_level:.1%} "
            f"closes every {curve.half_life_years:.2f} years."
        ),
        "note": _survivorship_note(survivorship),
        "data": {
            "series": [
                {"name": "Fitted fade", "role": "model", "values": [{"x": x, "y": y} for x, y in zip(labels, fitted)]},
                {"name": "Typed schedule", "role": "alt", "values": [{"x": x, "y": y} for x, y in zip(labels, curve.typed)]},
            ],
            "x": {"label": "Projection year"},
            "format": "pct:1",
            "zero": True,
            "reference": [
                {"value": curve.reversion_level, "label": f"Filings revert to {curve.reversion_level:.1%}"}
            ],
        },
    }


def _paths_figure(p: CompanyPaths) -> dict[str, Any]:
    labels = [f"Year {i + 1}" for i in range(len(p.fitted))]
    fitted_years = [i for i, b in enumerate(p.basis) if b == "fitted"]
    assumed_years = [i + 1 for i, b in enumerate(p.basis) if b != "fitted"]
    k = len(fitted_years)
    held = sum(p.fitted[i] for i in fitted_years) / k
    tail = ""
    if assumed_years:
        many = len(assumed_years) > 1
        tail = (
            f" Year{'s' if many else ''} {_join([str(y) for y in assumed_years])} "
            f"{'are' if many else 'is'} not fitted: path and band run straight to the "
            f"{p.terminal:.0%} terminal assumption."
        )
    reference = []
    if p.trailing is not None:
        reference.append({"value": p.trailing, "label": f"Trailing growth {p.trailing:.1%}"})
    return {
        "kind": "line",
        "title": (
            f"Fitted on filings, {p.name} holds near {held:.0%} growth for "
            f"{_word(k)} year{'s' if k != 1 else ''}; the typed schedule is down to "
            f"{p.typed[k - 1]:.1%} by then"
        ),
        "subtitle": (
            f"{p.ticker} revenue growth by projection year. The fitted path carries its "
            "10th to 90th percentile out-of-fold band and starts from the fiscal year "
            f"ended {p.fiscal_year_end}, filed {p.filed}; the typed schedule is "
            "assumptions.dcf." + tail
        ),
        "data": {
            "series": [
                {
                    "name": "Fitted fade",
                    "role": "model",
                    "values": [
                        {"x": x, "y": y, "lo": lo, "hi": hi}
                        for x, y, lo, hi in zip(labels, p.fitted, p.lower, p.upper)
                    ],
                },
                {"name": "Typed schedule", "role": "alt", "values": [{"x": x, "y": y} for x, y in zip(labels, p.typed)]},
            ],
            "x": {"label": "Projection year"},
            "format": "pct:1",
            "zero": True,
            "reference": reference,
        },
    }


def _revenue_figure(r: CompanyRevenue, name: str, years: int) -> dict[str, Any]:
    gap = r.fitted / r.typed - 1.0
    return {
        "kind": "tiles",
        "title": (
            f"The fitted path ends year {_word(years)} with {abs(gap):.0%} "
            f"{'more' if gap >= 0 else 'less'} revenue than the typed schedule"
        ),
        "subtitle": (
            f"{name} revenue in year {years}, USD millions, compounded on each path "
            f"from trailing revenue of {r.trailing:,.0f}mm to {r.trailing_as_of}."
        ),
        "data": {
            "tiles": [
                {"label": "Typed schedule", "value": r.typed, "format": "mm", "sub": f"Year {years} revenue"},
                {"label": "Fitted fade", "value": r.fitted, "format": "mm", "sub": f"{gap:+.1%} against the typed schedule"},
                {"label": "Fitted, low band", "value": r.low, "format": "mm", "sub": "10th percentile path"},
                {"label": "Fitted, high band", "value": r.high, "format": "mm", "sub": "90th percentile path"},
            ],
            "gap": gap,
            "refusalSlot": DCF_REFUSAL,
        },
    }


def _valuation_figure(v: Valuation, name: str) -> dict[str, Any]:
    gap = v.per_share_fitted - v.per_share_typed
    return {
        "kind": "tiles",
        "title": (
            f"The fitted path moves {name}'s DCF by {gap:+,.2f} a share, inside a band "
            f"of {v.per_share_low:,.2f} to {v.per_share_high:,.2f}"
        ),
        "subtitle": (
            f"USD per share, prices to {v.price_date}. Every path is discounted at "
            f"{v.wacc:.2%}, on {name}'s own regression beta of {v.beta:.2f} and a "
            f"{v.risk_free_rate:.2%} risk-free rate that is an assumption pinned for "
            "offline runs, not a Treasury quote. The engine section values on a "
            "peer-median beta, so its per-share figure differs by the discount rate, "
            "not by the growth path."
        ),
        "data": {
            "tiles": [
                {"label": "Typed schedule", "value": v.per_share_typed, "format": "num:2", "sub": f"EV {v.ev_typed:,.0f}mm"},
                {"label": "Fitted fade", "value": v.per_share_fitted, "format": "num:2", "sub": f"EV {v.ev_fitted:,.0f}mm, {gap:+,.2f} a share"},
                {"label": "Fitted, low band", "value": v.per_share_low, "format": "num:2", "sub": "10th percentile path"},
                {"label": "Fitted, high band", "value": v.per_share_high, "format": "num:2", "sub": "90th percentile path"},
            ],
            "gap": gap,
        },
    }


def _depth_figure(d: Depth) -> dict[str, Any]:
    deepest = max(d.years)
    expanded = [y for y, c in sorted(d.years.items()) for _ in range(c)]
    mid = median(expanded)
    unbuilt = ""
    if d.unbuilt:
        one = len(d.unbuilt) == 1
        unbuilt = (
            f" {_join(list(d.unbuilt))} {'has' if one else 'have'} no fiscal year the "
            "revenue ladder could build."
        )
    return {
        "kind": "column",
        "title": (
            f"The median filer supports {mid:g} fiscal years of revenue; "
            f"{d.years[deepest]} reach the full {deepest}"
        ),
        "subtitle": (
            f"Filers by fiscal years of revenue in the committed panel: "
            f"{len(expanded)} filers, {d.observations:,} company-years filed "
            f"{d.first_filed} to {d.last_filed}.{unbuilt}"
        ),
        "data": {
            "rows": [
                {"label": str(y), "value": d.years.get(y, 0), "role": "total"}
                for y in range(0, deepest + 1)
            ],
            "format": "int",
            "valueLabel": "Filers",
            "labelHeader": "Fiscal years",
            "stats": {
                "companies": len(expanded),
                "filers_with_rows": d.filers_with_rows,
                "observations": d.observations,
                "median_years": mid,
                "max_years": deepest,
                "at_max": d.years[deepest],
                "first_filed": d.first_filed,
                "last_filed": d.last_filed,
            },
        },
    }


def _takeaway(r: Results) -> str:
    one = r.horizons[0]
    lift = one.persistence - one.model
    status = _status(one)
    if len(one.fold_lifts) >= 2:
        mean, sd = fmean(one.fold_lifts), stdev(one.fold_lifts)
        spread = (
            f"{'inside' if abs(mean) < sd else 'outside'} the fold noise (a mean fold "
            f"lift of {mean:+.4f} against a fold-to-fold standard deviation of {sd:.4f})"
        )
    elif one.fold_sd is not None:
        spread = (
            f"{'inside' if abs(lift) < one.fold_sd else 'outside'} a fold standard "
            f"deviation of {one.fold_sd:.4f}"
        )
    else:
        spread = "with no fold spread to judge it against"
    text = (
        f"At {_years([one.horizon])} out the fitted fade {_VERB[status]} this year's "
        f"growth: a mean absolute error of {one.model:.4f} against {one.persistence:.4f}, "
        f"a lift of {lift:+.4f} {spread}."
    )
    later = list(r.horizons[1:])
    if later:
        lifts = _join([f"{h.persistence - h.model:+.4f}" for h in later])
        statuses = {_status(h) for h in later}
        if statuses == {"beats"}:
            text += (
                f" At {_years([h.horizon for h in later])} it beats persistence by "
                f"{lifts}, outside the fold noise."
            )
        else:
            text += (
                f" Further out it {_against_persistence(later, 'persistence')}, with "
                f"lifts of {lifts}."
            )
    if all(h.fold_sd is not None for h in r.horizons):
        if all(abs(_strongest(h)[1] - h.model) < h.fold_sd for h in r.horizons):
            text += (
                " Against the strongest baseline at each horizon the gap is smaller than "
                "one fold standard deviation of the model's own error"
            )
            names = {_strongest(h)[0] for h in later}
            if len(names) == 1 and names != {"persistence"}:
                text += f", and from {_word(later[0].horizon)} years out that baseline is {names.pop()}"
            text += "."
    if isinstance(r.paths, CompanyPaths) and isinstance(r.valuation, Valuation):
        v = r.valuation
        text += (
            f" Discounted at {v.wacc:.2%} on its own beta, {r.paths.name} is worth "
            f"{v.per_share_fitted:,.2f} a share on the fitted path and "
            f"{v.per_share_typed:,.2f} on the typed one."
        )
    return text


def shape(r: Results) -> dict[str, Any]:
    """The section, from the records ``collect`` produced. Nothing here fits or reads."""
    if not r.horizons:
        raise ValueError("the fade section needs at least one horizon")
    for h in r.horizons:
        _check_finite(f"horizon {h.horizon}", h.model, h.persistence, h.training_mean, h.sub_vertical_mean)
    if isinstance(r.curve, Curve):
        # collect refuses a curve that does not revert, so a NaN here is a bug.
        c = r.curve
        _check_finite("fade curve", c.start, c.slope, c.intercept, c.reversion_level, c.half_life_years, *c.typed)

    refusals: list[dict[str, str]] = []
    figures: dict[str, Any] = {"error_by_horizon": _error_figure(r.horizons)}
    parts: list[tuple[str, Any, Any]] = [
        ("fade_curve", r.curve, lambda c: _curve_figure(c, r.survivorship)),
        ("ddog_paths", r.paths, _paths_figure),
        ("ddog_revenue", r.revenue, lambda x: _revenue_figure(x, COMPANY_NAME, len(r.paths.fitted))),
        ("ddog_dcf", r.valuation, lambda x: _valuation_figure(x, COMPANY_NAME)),
    ]
    for key, part, build in parts:
        if part is None:
            continue
        if isinstance(part, Refusal):
            refusals.append({"what": part.what, "why": part.why})
        else:
            figures[key] = build(part)
    figures["depth"] = _depth_figure(r.depth)

    one = r.horizons[0]
    lift = one.persistence - one.model
    headline = {
        "metric": f"MAE of {_word(one.horizon)}-year revenue growth",
        "score": one.model,
        "baseline_name": "Persistence, this year's growth carried forward",
        "baseline_score": one.persistence,
        "lift": lift,
        "n": one.n,
        "higher_is_better": False,
        "verdict_status": verdict_status(lift, one.fold_sd, one.fold_lifts),
        "verdict_text": one.verdict_text,
    }
    return {
        "status": "ok",
        "takeaway": _takeaway(r),
        "refusals": refusals,
        "headline": headline,
        "figures": figures,
    }


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #


def _last_close(path: Path) -> date:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(date.fromisoformat(row["Date"]) for row in csv.DictReader(handle) if row.get("Date"))


def collect(ctx) -> dict:
    assumptions = copy.deepcopy(ctx.assumptions)
    # A fitted path reaches a DCF only behind this switch. Showing that path is
    # what this section is for, so the switch is on for this collection alone.
    assumptions.ml.forecast.enabled = True
    knowledge = date.fromisoformat(assumptions.as_of) if assumptions.as_of else None
    dcf = assumptions.dcf

    with ctx.record("depth", "techval.ml.forecast.build_fade_panel", [PANEL]):
        panel = _recorded_panel(ctx.input(PANEL), assumptions, None)
        if knowledge is not None:
            panel = _truncate_panel(panel, knowledge)
        filed = [o.as_of for o in panel.observations]
        counts: dict[int, int] = {}
        for n in panel.depth.values():
            counts[n] = counts.get(n, 0) + 1
        depth = Depth(
            years=dict(sorted(counts.items())),
            observations=len(panel.observations),
            filers_with_rows=len(panel.tickers),
            unbuilt=tuple(sorted(t for t, _ in panel.failures)),
            first_filed=min(filed).year,
            last_filed=max(filed).year,
        )

    with ctx.record("error_by_horizon", "techval.ml.forecast.fit_fade", [PANEL]):
        model = fit_fade(panel, assumptions)
        horizons = tuple(
            HorizonScore(
                horizon=h,
                model=fit.evaluation.score,
                persistence=fit.evaluation.baseline_score,
                training_mean=fit.baselines["training_mean"],
                sub_vertical_mean=fit.baselines["sub_vertical_mean"],
                fold_sd=fit.evaluation.fold_sd,
                n=fit.evaluation.n_observations,
                verdict_text=fit.evaluation.verdict(),
                fold_lifts=tuple(fit.evaluation.fold_lifts),
            )
            for h, fit in sorted(model.fits.items())
        )

    survivorship = Survivorship(
        leavers=sum(1 for t in panel.tickers if is_delisted(t)),
        filers=len(panel.tickers),
        gaps=tuple(
            (h, model.survivorship[f"gap_{h}y"])
            for h in sorted(model.fits)
            if f"gap_{h}y" in model.survivorship
        ),
    )
    typed = tuple(
        float(g) for g in _fade(dcf.revenue_growth_start, dcf.revenue_growth_terminal, dcf.projection_years)
    )
    curve: Curve | Refusal
    if not (math.isfinite(model.reversion_level) and math.isfinite(model.half_life_years)):
        curve = Refusal(
            "The fade curve",
            f"The one-year persistence slope is {model.persistence_slope:.4f}, so growth "
            "does not revert on this panel and there is no level or half-life to draw.",
        )
    else:
        with ctx.record("fade_curve", "techval.ml.forecast.fit_fade", [PANEL]):
            curve = Curve(
                start=dcf.revenue_growth_start,
                slope=model.persistence_slope,
                intercept=model.persistence_intercept,
                reversion_level=model.reversion_level,
                half_life_years=model.half_life_years,
                typed=typed,
                n=len(panel.labelled(1)),
            )

    paths: CompanyPaths | Refusal
    revenue: CompanyRevenue | Refusal | None = None
    valuation: Valuation | Refusal | None = None
    try:
        with ctx.record("ddog_paths", "techval.ml.forecast.FadeModel.path", [PANEL]):
            path = model.path(COMPANY, dcf.projection_years)
            latest = model.latest[COMPANY]
            paths = CompanyPaths(
                ticker=COMPANY,
                name=COMPANY_NAME,
                trailing=latest.growth,
                fiscal_year_end=latest.fiscal_year_end.isoformat(),
                filed=latest.as_of.isoformat(),
                typed=typed,
                fitted=tuple(path.growth),
                lower=tuple(path.lower),
                upper=tuple(path.upper),
                basis=tuple(path.basis),
                terminal=model.terminal_growth,
            )
    except TechvalError as exc:
        # The revenue and the DCF are built on this path, so this one refusal
        # covers all three and the page does not repeat it under each.
        paths = Refusal(f"{COMPANY_NAME}, assumed against fitted", str(exc))

    fin = None
    if isinstance(paths, CompanyPaths):
        try:
            with ctx.record("ddog_revenue", "techval.dcf.project", [PANEL, COMPANY_FACTS]):
                payload = json.loads(ctx.input(COMPANY_FACTS).read_text(encoding="utf-8"))
                fin = build_financials(
                    COMPANY, facts=CompanyFacts(payload, COMPANY, knowledge_date=knowledge)
                )

                def final_year(growth: Sequence[float] | None) -> float:
                    return project(
                        fin, assumptions, growth_path=None if growth is None else list(growth)
                    )[-1].revenue

                revenue = CompanyRevenue(
                    trailing=fin.revenue,
                    trailing_as_of=fin.as_of.isoformat(),
                    typed=final_year(None),
                    fitted=final_year(path.growth),
                    low=final_year(path.lower),
                    high=final_year(path.upper),
                )
        except TechvalError as exc:
            fin = None
            revenue = Refusal(f"{COMPANY_NAME} revenue on each path", str(exc))

    if fin is not None and assumptions.market.risk_free_rate is None:
        valuation = Refusal(
            DCF_REFUSAL,
            "The DCF needs a risk-free rate and the committed fixtures record none. The "
            "engine reads the ten-year Treasury yield from a live feed, and the "
            "valuation tests pin a rate by hand in tests/conftest.py. Set "
            "market.risk_free_rate in the assumptions passed to techval dashboard to "
            "value both paths.",
        )
    elif fin is not None:
        try:
            with ctx.record(
                "ddog_dcf", "techval.ml.forecast.compare_fade", [PANEL, COMPANY_FACTS, PRICES]
            ):
                prices = ctx.input(PRICES)
                today = knowledge or _last_close(prices / f"{COMPANY}.csv")
                market = MarketData(CsvSource(prices), HttpCache(enabled=False), today=today)
                bridge = build_ev_bridge(fin, market.spot(COMPANY), assumptions)
                wacc = compute_wacc(fin, bridge, market, assumptions)
                comparison = compare_fade(fin, bridge, wacc, assumptions, model, ticker=COMPANY)
                valuation = Valuation(
                    per_share_typed=comparison.assumed.per_share,
                    per_share_fitted=comparison.fitted.per_share,
                    per_share_low=comparison.low.per_share,
                    per_share_high=comparison.high.per_share,
                    ev_typed=comparison.assumed.enterprise_value,
                    ev_fitted=comparison.fitted.enterprise_value,
                    price_date=today.isoformat(),
                    wacc=wacc.wacc,
                    beta=wacc.levered_beta,
                    risk_free_rate=wacc.risk_free_rate,
                )
        except TechvalError as exc:
            valuation = Refusal(DCF_REFUSAL, str(exc))

    return shape(
        Results(
            horizons=horizons,
            curve=curve,
            survivorship=survivorship,
            depth=depth,
            paths=paths,
            revenue=revenue,
            valuation=valuation,
        )
    )
