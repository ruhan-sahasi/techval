"""The value signal: did cheapness predict forward returns, and does the answer survive the overlap?

This section runs the recorded run of ``techval signal`` and nothing else. The
score panel and the closes are read with the command's own loaders and the
command's own sign convention for ``cheapness``, and handed to
``techval.ml.signals.test_signal`` with the assumptions in force, so the page
shows exactly what this prints::

    techval signal --score cheapness \\
        --scores tests/fixtures/signals/ev_revenue.csv.gz \\
        --prices tests/fixtures/signals/closes.csv.gz

The collector is split in two so the page can be tested without a run.
``facts_from_result`` reads the numbers off a ``SignalResult`` into a plain
``SignalFacts``, and ``shape`` turns those numbers into the section: figures,
takeaway, headline and refusals. ``shape`` does no arithmetic a reader would
call a result; it formats, orders and picks the words that match the signs.

Two judgments are made here and stated where they are made. The headline's
``n`` is the number of rebalance dates, not company-dates, because the mean IC
is a mean of one coefficient per date and the whole argument of the section is
that the company-date count overstates the evidence. And the verdict status is
read off the Newey-West p-value, which is the module's statistic of record.

The one limit this section cannot fix is survivorship. The panel holds no name
that was taken over or delisted. The seed names with no price rows and no score
are found by comparing the seed universe with the fixtures, so the refusal note
lists exactly the names the data lacks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ID = "signal"
TITLE = "The value signal"

SCORES = "signals/ev_revenue.csv.gz"
CLOSES = "signals/closes.csv.gz"
INPUTS: list[str] = [SCORES, CLOSES]

ENTRY_POINT = "techval.ml.signals.test_signal"

# The score this section tests, as ``techval signal`` names it.
SCORE_NAME = "cheapness"

# Two-sided 5% on the normal, as ``techval.ml.signals.SIGNIFICANT_T`` has it.
# Kept here so ``shape`` imports nothing from the model module; the test file
# checks the two are the same number.
SIGNIFICANT_T = 1.959963984540054
SIGNIFICANT_P = 0.05

# The figure recorded with the run's own timing. Every other figure comes from
# the same run and is recorded against the same entry point.
LEAD_FIGURE = "t_statistics"

FIGURE_IDS = ("evidence", "t_statistics", "ic_by_date", "ic_autocorrelation", "bucket_returns")

MINUS = "\u2212"


# --------------------------------------------------------------------------- #
# the numbers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SignalFacts:
    """Everything the page states, read off one ``SignalResult``.

    ``bucket_returns`` is in the module's order, bucket one holding the lowest
    score. The score is cheapness, so bucket one is the most expensive fifth
    and the last bucket the cheapest.
    """

    label: str
    horizon_months: int
    companies: int
    n_scored: int
    dates: tuple[date, ...]
    ic: tuple[float, ...]
    counts: tuple[int, ...]
    mean_ic: float
    share_positive: float
    t_naive: float
    t_newey_west: float
    p_newey_west: float
    lag: int
    spacing_months: int
    inflation: float
    effective_n: float
    autocorrelations: tuple[float, ...]
    permutation_p: float | None
    baseline_draws: int
    baseline_name: str
    baseline_score: float
    lift: float
    bucket_returns: tuple[float, ...]
    bucket_dates: int
    spread: float
    spread_t: float
    monotonicity: float | None
    acquired: int
    delisted: int
    absent: tuple[str, ...]
    verdict: str


def facts_from_result(result: Any, absent: Sequence[str]) -> SignalFacts:
    """Read a ``SignalResult`` into the numbers the page shows."""
    from techval.ml.signals import DAYS_PER_MONTH

    ic = result.ic
    counts = result.exit_counts()
    # The spacing in months, rounded the way ``overlap_lag`` rounds it, so the
    # window overlap drawn beside the autocorrelations uses the same clock as
    # the lag the correction used.
    spacing_months = max(1, int(round(ic.spacing_days / DAYS_PER_MONTH)))
    frame = result.buckets.frame
    return SignalFacts(
        label=result.label,
        horizon_months=int(result.horizon_months),
        companies=len({o.ticker for o in result.outcomes if o.scored}),
        n_scored=int(result.n_scored),
        dates=tuple(ic.dates),
        ic=tuple(float(v) for v in ic.values),
        counts=tuple(int(c) for c in ic.counts),
        mean_ic=float(ic.mean),
        share_positive=float(ic.share_positive),
        t_naive=float(ic.t_naive),
        t_newey_west=float(ic.t_newey_west),
        p_newey_west=float(result.p_newey_west),
        lag=int(ic.lag),
        spacing_months=spacing_months,
        inflation=float(ic.inflation),
        effective_n=float(ic.effective_n),
        autocorrelations=tuple(float(r) for r in ic.autocorrelations),
        permutation_p=None if result.permutation_p is None else float(result.permutation_p),
        baseline_draws=int(result.baseline_draws),
        baseline_name=str(result.evaluation.baseline_name),
        baseline_score=float(result.evaluation.baseline_score),
        lift=float(result.evaluation.lift),
        bucket_returns=tuple(float(v) for v in frame["Mean return"].to_list()),
        bucket_dates=int(result.buckets.n_dates),
        spread=float(result.buckets.spread),
        spread_t=float(result.buckets.spread_t),
        monotonicity=None if result.buckets.monotonicity is None else float(result.buckets.monotonicity),
        acquired=int(counts.get("acquired", 0)),
        delisted=int(counts.get("delisted", 0)),
        absent=tuple(sorted(absent)),
        verdict=result.verdict(),
    )


# --------------------------------------------------------------------------- #
# words for numbers
# --------------------------------------------------------------------------- #


def _finite(v: float | None) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _num(v: float | None, dp: int = 2) -> str:
    if not _finite(v):
        return "n/a"
    body = f"{abs(v):,.{dp}f}"
    return (MINUS if v < 0 and any(c in body for c in "123456789") else "") + body


def _signed(v: float | None, dp: int = 2) -> str:
    if not _finite(v):
        return "n/a"
    body = f"{abs(v):,.{dp}f}"
    if not any(c in body for c in "123456789"):
        return body
    return ("+" if v > 0 else MINUS) + body


def _signed_pct(v: float | None, dp: int = 1) -> str:
    return "n/a" if not _finite(v) else _signed(v * 100, dp) + "%"


def _p(p: float | None) -> str:
    if not _finite(p):
        return "n/a"
    if p < 0.001:
        return "below 0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


_WORDS = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")


def _cadence(months: int) -> str:
    return {1: "monthly", 3: "quarterly", 6: "half-yearly", 12: "annual"}.get(
        months, f"{months}-monthly"
    )


def _longest_run(values: Sequence[float], negative: bool) -> int:
    best = run = 0
    for v in values:
        hit = v < 0 if negative else v > 0
        run = run + 1 if hit else 0
        best = max(best, run)
    return best


def verdict_status(p_newey_west: float, lift: float) -> str:
    """The headline chip, read off the statistic of record.

    The rule: a Newey-West p-value that is not below 5% is ``not_significant``
    whatever the sign of the coefficient, because a mean that the overlap
    correction cannot tell from zero has not beaten or lost to anything. Below
    5% the sign of the lift over the permutation baseline decides: ``beats``
    when the mean IC is above it, ``loses`` when it is below. The threshold is
    strict, as ``SignalResult.significant`` has it, so a p of exactly 0.05 is
    not significant.
    """
    if not _finite(p_newey_west) or p_newey_west >= SIGNIFICANT_P:
        return "not_significant"
    return "beats" if lift > 0 else "loses"


# --------------------------------------------------------------------------- #
# the section
# --------------------------------------------------------------------------- #


def _figure(kind: str, title: str, subtitle: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "title": title, "subtitle": subtitle, "data": data}


def _takeaway(f: SignalFacts) -> str:
    sign = "the wrong sign" if f.mean_ic < 0 else "the sign value investing expects"
    naive_clears = abs(f.t_naive) >= SIGNIFICANT_T
    corrected_clears = abs(f.t_newey_west) >= SIGNIFICANT_T
    if naive_clears and not corrected_clears:
        verdict = (
            f"and it is not significant once the overlapping windows are corrected: "
            f"the naive t of {_signed(f.t_naive)} clears the 5% line, the Newey-West "
            f"t of {_signed(f.t_newey_west)} (p = {_p(f.p_newey_west)}) does not."
        )
    elif corrected_clears:
        verdict = (
            f"and it survives the overlap correction, with a Newey-West t of "
            f"{_signed(f.t_newey_west)} (p = {_p(f.p_newey_west)})."
        )
    else:
        verdict = (
            f"and neither t-statistic reaches the 5% line: naive {_signed(f.t_naive)}, "
            f"Newey-West {_signed(f.t_newey_west)} (p = {_p(f.p_newey_west)})."
        )
    return (
        f"Cheapness on trailing EV/Revenue across {f.companies} TMT names predicted "
        f"{f.horizon_months}-month returns with {sign}, a mean IC of "
        f"{_signed(f.mean_ic, 3)} over {len(f.ic)} {_cadence(f.spacing_months)} "
        f"dates, {verdict}"
    )


def _evidence(f: SignalFacts) -> dict[str, Any]:
    n = len(f.ic)
    positive = sum(1 for v in f.ic if v > 0)
    if f.permutation_p is None:
        perm = "No permutation baseline was drawn"
    else:
        # Where no permutation reached the observed mean the p-value sits at the
        # add-one floor, 1 / (draws + 1), and saying so is part of reading it.
        floor = 1.0 / (f.baseline_draws + 1) if f.baseline_draws else None
        at_floor = floor is not None and abs(f.permutation_p - floor) < 1e-12
        perm = (
            f"Permutation p {f.permutation_p:.4f}"
            + (f", the floor of {f.baseline_draws} draws," if at_floor else "")
            + f" treats the {n} dates as independent and is not the test"
        )
    return _figure(
        "tiles",
        f"About {_num(f.effective_n, 0)} effective observations, not {f.n_scored:,}",
        "",
        {
            "tiles": [
                {
                    "label": "Newey-West t",
                    "value": f.t_newey_west,
                    "format": "signed:2",
                    "sub": f"Naive t {_signed(f.t_naive)}, corrected at lag {f.lag}",
                },
                {"label": "Newey-West p", "value": f.p_newey_west, "format": "num:3", "sub": perm},
                {
                    "label": "Dates with a positive IC",
                    "value": f.share_positive,
                    "format": "pct:0",
                    "sub": f"{positive} of {n} rebalance dates",
                },
                {
                    "label": "Effective observations",
                    "value": f.effective_n,
                    "format": "num:0",
                    "sub": f"From {f.n_scored:,} company-dates on {n} dates",
                },
            ]
        },
    )


def _t_statistics(f: SignalFacts) -> dict[str, Any]:
    naive_clears = abs(f.t_naive) >= SIGNIFICANT_T
    corrected_clears = abs(f.t_newey_west) >= SIGNIFICANT_T
    if naive_clears and not corrected_clears:
        title = "The naive t clears the 5% line and the corrected t does not"
    elif naive_clears and corrected_clears:
        title = "Both t-statistics clear the 5% line"
    elif corrected_clears:
        title = "Only the corrected t clears the 5% line"
    else:
        title = "Neither t-statistic reaches the 5% line"
    subtitle = (
        f"t-statistic of the mean IC over {len(f.ic)} {_cadence(f.spacing_months)} "
        f"dates. The naive one treats the dates as independent; Newey-West at lag "
        f"{f.lag} widens the standard error {_num(f.inflation, 2)}x for the "
        f"{f.horizon_months}-month windows they share."
    )
    return _figure(
        "dot",
        title,
        subtitle,
        {
            "rows": [
                {
                    "label": "Mean IC",
                    "values": {"newey_west": f.t_newey_west, "naive": f.t_naive},
                }
            ],
            "series": [
                {"key": "newey_west", "name": f"Newey-West, lag {f.lag}", "role": "model"},
                {"key": "naive", "name": "Naive", "role": "baseline"},
            ],
            "format": "signed:2",
            "gapLabel": "Newey-West minus naive",
            "gapFormat": "signed:2",
            "labelHeader": "t-statistic of",
            "reference": [
                {"value": -SIGNIFICANT_T, "label": f"5% line {_signed(-SIGNIFICANT_T)}"},
                {"value": SIGNIFICANT_T, "label": f"5% line {_signed(SIGNIFICANT_T)}"},
            ],
            "zero": True,
            "height": 130,
            "wide": True,
        },
    )


def _ic_by_date(f: SignalFacts) -> dict[str, Any]:
    n = len(f.ic)
    negative = f.mean_ic < 0
    side = sum(1 for v in f.ic if (v < 0 if negative else v > 0))
    run = _longest_run(f.ic, negative)
    title = f"The IC was {'below' if negative else 'above'} zero on {side} of {n} dates"
    if run >= 2:
        title += f", {run} of them in a row"
    subtitle = (
        f"Rank IC of cheapness against the {f.horizon_months}-month forward return, "
        f"one per {_cadence(f.spacing_months)} rebalance date, "
        f"{f.dates[0]:%Y-%m} to {f.dates[-1]:%Y-%m}, {min(f.counts)} to "
        f"{max(f.counts)} names a date"
    )
    # Symmetric about zero, so neither sign is drawn larger than the other, and
    # at least a tenth beyond the largest coefficient, so the label under the
    # most negative column stays clear of the date labels.
    bound = (math.floor(max(abs(v) for v in f.ic) * 10) + 2) / 10
    return _figure(
        "column",
        title,
        subtitle,
        {
            "rows": [{"label": f"{d:%Y-%m}", "value": v} for d, v in zip(f.dates, f.ic)],
            "diverging": True,
            "format": "signed:2",
            "valueLabel": "IC",
            "labelHeader": "Rebalance date",
            "reference": [{"value": f.mean_ic, "label": f"Mean {_signed(f.mean_ic, 3)}"}],
            "domain": [-bound, bound],
            "wide": True,
        },
    )


def _ic_autocorrelation(f: SignalFacts) -> dict[str, Any]:
    rows = []
    for k, rho in enumerate(f.autocorrelations, start=1):
        apart = k * f.spacing_months
        overlap = max(0.0, 1.0 - apart / f.horizon_months)
        rows.append(
            {"label": f"{apart} months apart", "values": {"measured": rho, "overlap": overlap}}
        )
    first = rows[0]["values"]
    title = (
        f"Adjacent ICs correlate at {_signed(first['measured'])} and their windows "
        f"overlap by {_num(first['overlap'], 2)}"
    )
    subtitle = (
        f"Autocorrelation of the per-date IC at each Newey-West lag, against the "
        f"share of the {f.horizon_months}-month return window two dates that far apart "
        "have in common. Overlap alone does not autocorrelate a cross-sectional IC. "
        "Overlap with a persistent score does, a trailing multiple persists, and "
        "that is what makes the naive standard error too small."
    )
    return _figure(
        "dot",
        title,
        subtitle,
        {
            "rows": rows,
            "series": [
                {"key": "measured", "name": "Cheapness IC", "role": "model"},
                {"key": "overlap", "name": "Window overlap", "role": "baseline"},
            ],
            "format": "num:2",
            "gapLabel": "IC minus overlap",
            "gapFormat": "signed:2",
            "labelHeader": "Dates",
            "zero": True,
        },
    )


def _bucket_returns(f: SignalFacts) -> dict[str, Any]:
    k = len(f.bucket_returns)
    # Cheapest first. The module numbers buckets from the lowest score, and the
    # score is cheapness, so its last bucket is the cheapest. The ends are named
    # cheap and dear, the valuation pair, because "most expensive" is too wide
    # to label its column in a half-width card.
    ordered = list(reversed(f.bucket_returns))
    labels = ["Cheapest"] + [str(i) for i in range(2, k)] + ["Dearest"]
    group = "fifth" if k == 5 else "bucket"
    points = abs(f.spread) * 100
    if f.spread < 0:
        title = f"The cheapest {group} trailed the dearest by {points:.1f} percentage points"
    elif f.spread > 0:
        title = f"The cheapest {group} beat the dearest by {points:.1f} percentage points"
    else:
        title = f"The cheapest and dearest {group} earned the same"
    if f.monotonicity is None:
        shape_words = "monotonicity undefined"
    else:
        direction = "fall" if f.monotonicity < 0 else "rise"
        shape_words = (
            f"monotonicity {_signed(f.monotonicity)}, so returns {direction} as "
            "cheapness rises"
        )
    subtitle = (
        f"Mean {f.horizon_months}-month forward return by cheapness {group}, equal "
        f"weight, averaged over {f.bucket_dates} dates. Cheapest minus dearest "
        f"{_signed_pct(f.spread)}, Newey-West t {_signed(f.spread_t)}; {shape_words}."
    )
    return _figure(
        "column",
        title,
        subtitle,
        {
            "rows": [{"label": lab, "value": v} for lab, v in zip(labels, ordered)],
            "format": "pct:1",
            "valueLabel": f"Mean {f.horizon_months}-month return",
            "labelHeader": "Cheapness " + group,
        },
    )


def _survivorship(f: SignalFacts) -> dict[str, str] | None:
    if f.acquired or f.delisted:
        return None
    why = (
        f"Not one of the {f.n_scored:,} holding periods ended at a takeover or a "
        "delisting, so the harness had no exit to terminate at a deal or score at a "
        "delisting return."
    )
    if f.absent:
        count = _WORDS[len(f.absent)] if len(f.absent) < len(_WORDS) else f"{len(f.absent):,}"
        why += (
            f" {count} names in the seed universe ({', '.join(f.absent)}) have "
            "no price rows and no score in the fixtures and are missing from every "
            "cross-section."
        )
    why += (
        " Survivorship is measured here, not corrected. A takeover carries its "
        "premium into whichever bucket held the target, so if the missing names "
        "skewed cheap their absence biases the coefficient against cheapness, and "
        "the true figure is if anything higher than the one shown."
    )
    return {"what": "Survivorship correction", "why": why}


def shape(f: SignalFacts) -> dict[str, Any]:
    """The section as ``collect`` returns it, from the numbers alone."""
    figures: dict[str, Any] = {
        "evidence": _evidence(f),
        "t_statistics": _t_statistics(f),
        "ic_by_date": _ic_by_date(f),
    }
    refusals: list[dict[str, str]] = []

    if f.lag > 0 and f.autocorrelations:
        figures["ic_autocorrelation"] = _ic_autocorrelation(f)
    else:
        refusals.append(
            {
                "what": "IC autocorrelation by lag",
                "why": (
                    f"The rebalance spacing of {f.spacing_months} months is at least "
                    f"the {f.horizon_months}-month horizon, so no two windows overlap, "
                    "the Newey-West lag is zero and there is no autocorrelation for "
                    "the correction to act on."
                ),
            }
        )

    if f.bucket_dates > 0 and len(f.bucket_returns) >= 2 and all(_finite(v) for v in f.bucket_returns):
        figures["bucket_returns"] = _bucket_returns(f)
    else:
        refusals.append(
            {
                "what": "Returns by cheapness bucket",
                "why": (
                    f"No rebalance date carried enough scored names to fill "
                    f"{len(f.bucket_returns)} buckets with two names each, so there "
                    "is no bucket return to draw."
                ),
            }
        )

    survivorship = _survivorship(f)
    if survivorship is not None:
        refusals.append(survivorship)

    return {
        "status": "ok",
        "takeaway": _takeaway(f),
        "refusals": refusals,
        # n is the number of dates: the mean IC is a mean of one coefficient per
        # date, and the company-date count is exactly the overstatement this
        # section exists to take apart.
        "headline": {
            "metric": "mean IC",
            "score": f.mean_ic,
            "baseline_name": f.baseline_name,
            "baseline_score": f.baseline_score,
            "lift": f.lift,
            "n": len(f.ic),
            "higher_is_better": True,
            "verdict_status": verdict_status(f.p_newey_west, f.lift),
            "verdict_text": f.verdict,
        },
        "figures": figures,
    }


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #


def collect(ctx) -> dict:
    from techval.commands_forecast import _KNOWN_SCORES, _prices_from_csv, _scores_from_csv
    from techval.ml.signals import test_signal
    from techval.tmt.taxonomy import SEED

    inputs = [SCORES, CLOSES]
    with ctx.record(LEAD_FIGURE, ENTRY_POINT, inputs):
        spec = _KNOWN_SCORES[SCORE_NAME]
        scores = _scores_from_csv(ctx.input(SCORES), spec["column"], spec["negate"], None)
        prices = _prices_from_csv(ctx.input(CLOSES), None)
        result = test_signal(scores, prices, assumptions=ctx.assumptions, label=spec["label"])
        scored = {s.ticker for s in scores}
        absent = sorted(t for t in SEED if t not in prices and t not in scored)
        section = shape(facts_from_result(result, absent))

    # One run computed every figure. The lead figure carries its time; the rest
    # are recorded against the same entry point and inputs, which is where
    # their numbers came from.
    for figure_id in section["figures"]:
        if figure_id != LEAD_FIGURE:
            with ctx.record(figure_id, ENTRY_POINT, inputs):
                pass
    return section
