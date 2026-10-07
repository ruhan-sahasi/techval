"""Market-implied expectations: the engine's DCF solved for the market price.

The DCF says what a company is worth on stated assumptions. When that number
is far from the price, the useful question is the reverse one: what does the
price assume? Each solve here changes one assumption, holds every other at its
configured value, and finds where the engine's own ``run_dcf`` per-share value
equals the price. Nothing is re-implemented, so the implied figures inherit
every adjustment the forward DCF makes: the bridge, dilution, NOLs, the
terminal method the assumptions select.

A lever that cannot reach the price inside its bracket is reported as such,
with the value the DCF reaches at the bracket's edge. That is an answer, not
a failure: "no terminal margin below 95% gets there" says more about a price
than an extrapolated 140% margin would.

Money is USD millions, per-share figures dollars, rates decimals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .config import Assumptions
from .dcf import run_dcf
from .errors import TechvalError

# The discount rate's bracket. Below terminal growth the Gordon value is not
# defined, and half a point above it is already a perpetuity near infinity;
# 40% is above any equity a DCF should be applied to.
RATE_FLOOR_ABOVE_G = 0.005
RATE_CEILING = 0.40


@dataclass
class Case:
    """Everything a forward DCF needs, so a solve can revalue under one change."""

    fin: object
    bridge: object
    wacc: object
    assumptions: Assumptions

    def with_dcf(self, **dcf_changes) -> "Case":
        """The same company under changed ``assumptions.dcf`` fields; the original is untouched."""
        a = self.assumptions.model_copy(deep=True)
        for name, v in dcf_changes.items():
            setattr(a.dcf, name, v)
        return Case(fin=self.fin, bridge=self.bridge, wacc=self.wacc, assumptions=a)

    def value(self, growth_path=None, **dcf_changes) -> float:
        """Per-share value with some ``assumptions.dcf`` fields changed."""
        a = self.assumptions.model_copy(deep=True)
        for name, v in dcf_changes.items():
            setattr(a.dcf, name, v)
        return float(run_dcf(self.fin, self.bridge, self.wacc, a, growth_path=growth_path).per_share)


@dataclass(frozen=True)
class Root:
    """A bracketed solve: the root, or the bracket edge nearest the target."""

    x: float | None
    edge_x: float | None = None
    edge_value: float | None = None

    @property
    def reached(self) -> bool:
        return self.x is not None


def solve_monotone(
    f: Callable[[float], float],
    target: float,
    lo: float,
    hi: float,
    tol: float = 1e-7,
    iterations: int = 200,
) -> Root:
    """Bisect a monotone f for f(x) = target on [lo, hi], either direction.

    Out of reach, the edge whose value lies nearer the target is returned with
    its value, so the caller can say how far the bracket gets.
    """
    f_lo, f_hi = f(lo), f(hi)
    if not (min(f_lo, f_hi) <= target <= max(f_lo, f_hi)):
        if abs(f_lo - target) < abs(f_hi - target):
            return Root(None, lo, f_lo)
        return Root(None, hi, f_hi)
    rising = f_hi >= f_lo
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        f_mid = f(mid)
        if (f_mid < target) == rising:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return Root((lo + hi) / 2.0)


@dataclass(frozen=True)
class Solve:
    """One lever's answer: the assumed value, the implied one, and a sentence."""

    lever: str
    price: float
    assumed: float | None
    implied: float | None
    bracket: tuple[float, float]
    edge: float | None = None
    edge_value: float | None = None
    sentence: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def reached(self) -> bool:
        return self.implied is not None


def _guarded(f: Callable[[float], float]) -> Callable[[float], float]:
    """A DCF that refuses at some input is treated as worthless there, not as a crash."""

    def inner(x: float) -> float:
        try:
            return f(x)
        except TechvalError:
            return float("-inf")

    return inner


def implied_discount_rate(case: Case, price: float) -> Solve:
    """The rate at which the base case is worth exactly the price.

    Value falls as the rate rises and explodes as it nears terminal growth, so
    for any price above the value at 40% a root exists: it is the annual
    return a buyer at today's price earns if every other assumption holds.
    """
    g = case.assumptions.dcf.terminal_growth
    lo, hi = g + RATE_FLOOR_ABOVE_G, RATE_CEILING
    root = solve_monotone(_guarded(lambda r: case.value(wacc_override=r)), price, lo, hi)
    assumed = case.assumptions.dcf.wacc_override or case.wacc.wacc
    if root.reached:
        sentence = (
            f"At {price:,.2f} a buyer earns {root.x:.2%} a year if the base case holds, "
            f"against a cost of capital of {assumed:.2%}."
        )
        return Solve("discount_rate", price, assumed, root.x, (lo, hi), sentence=sentence)
    sentence = (
        f"Even discounted at {RATE_CEILING:.0%} the base case is worth {root.edge_value:,.2f}, "
        f"above the {price:,.2f} price: the price assumes less than the base case."
        if root.edge_x == hi
        else f"No rate above terminal growth plus half a point reaches {price:,.2f}."
    )
    return Solve(
        "discount_rate", price, assumed, None, (lo, hi), root.edge_x, root.edge_value, sentence
    )


# First-year growth from a halving to a quadrupling of revenue. Above the top
# the answer is "no growth rate a company has printed", which the edge says.
GROWTH_BRACKET = (-0.50, 3.00)


def implied_first_year_growth(case: Case, price: float) -> Solve:
    """The first-year growth, faded on the engine's straight line, that the price needs.

    Every other year follows from it exactly as the forward DCF builds them, a
    straight line to ``dcf.revenue_growth_terminal`` in the final projection
    year, so the implied path is in ``extras['path']`` for the base rates.
    """
    from .dcf import _fade

    cfg = case.assumptions.dcf
    lo, hi = GROWTH_BRACKET
    root = solve_monotone(_guarded(lambda g: case.value(revenue_growth_start=g)), price, lo, hi)
    assumed = cfg.revenue_growth_start
    years = cfg.projection_years
    if root.reached:
        path = [float(x) for x in _fade(root.x, cfg.revenue_growth_terminal, years)]
        sentence = (
            f"At {price:,.2f} the price needs {root.x:.1%} revenue growth next year, fading in a "
            f"straight line to {cfg.revenue_growth_terminal:.1%} by year {years}, against "
            f"{assumed:.1%} assumed."
        )
        return Solve(
            "first_year_growth", price, assumed, root.x, (lo, hi), sentence=sentence, extras={"path": path}
        )
    if root.edge_x == hi:
        sentence = (
            f"No first-year growth up to {hi:.0%} reaches {price:,.2f}: at {hi:.0%}, fading to "
            f"{cfg.revenue_growth_terminal:.1%}, the DCF is worth {root.edge_value:,.2f}."
        )
    else:
        sentence = (
            f"Even shrinking {abs(lo):.0%} next year the DCF is worth {root.edge_value:,.2f}, above "
            f"the {price:,.2f} price: the price assumes a contraction this lever cannot express."
        )
    return Solve(
        "first_year_growth", price, assumed, None, (lo, hi), root.edge_x, root.edge_value, sentence
    )


# Terminal EBIT margin from a loss-maker's to one no listed software company
# has sustained. The top is deliberately beyond plausibility, so a miss at it
# is unambiguous.
MARGIN_BRACKET = (-0.20, 0.95)


def implied_terminal_margin(case: Case, price: float) -> Solve:
    """The year-N EBIT margin, reached on the engine's own margin path, the price needs."""
    cfg = case.assumptions.dcf
    lo, hi = MARGIN_BRACKET
    root = solve_monotone(_guarded(lambda m: case.value(ebit_margin_terminal=m)), price, lo, hi)
    assumed = cfg.ebit_margin_terminal
    if root.reached:
        sentence = (
            f"At {price:,.2f} the price needs a {root.x:.1%} EBIT margin by year "
            f"{cfg.projection_years}, against {assumed:.1%} assumed, growth unchanged."
        )
        return Solve("terminal_margin", price, assumed, root.x, (lo, hi), sentence=sentence)
    if root.edge_x == hi:
        sentence = (
            f"No terminal margin up to {hi:.0%} reaches {price:,.2f}: at {hi:.0%} the DCF is worth "
            f"{root.edge_value:,.2f}, so the price cannot be a bet on margins alone."
        )
    else:
        sentence = (
            f"Even at a {lo:.0%} terminal margin the DCF is worth {root.edge_value:,.2f}, above the "
            f"{price:,.2f} price."
        )
    return Solve(
        "terminal_margin", price, assumed, None, (lo, hi), root.edge_x, root.edge_value, sentence
    )


# The longest projection the DCF accepts; the duration solve works inside it.
DURATION_HORIZON = 15


def duration_path(held_growth: float, years_held: int, case: Case) -> list[float]:
    """Growth held for ``years_held`` years, then a straight fade to terminal by year 15."""
    terminal = case.assumptions.dcf.revenue_growth_terminal
    rest = DURATION_HORIZON - years_held
    fade = [held_growth + (terminal - held_growth) * (i + 1) / rest for i in range(rest)]
    return [held_growth] * years_held + fade


def implied_duration(case: Case, price: float, *, held_growth: float) -> Solve:
    """How many years of growth at ``held_growth`` the price needs.

    Mauboussin's competitive advantage period, read off the engine's DCF: on a
    15-year projection, growth is held for k years and then fades in a straight
    line to terminal, and the answer is the fewest whole k whose value reaches
    the price. Margins ramp to the same terminal level over the same 15 years,
    which is slower than the 5-year base case, so the value with nothing held
    in this frame is reported beside the answer rather than left implicit.
    """
    value = lambda k: case.value(growth_path=duration_path(held_growth, k, case), projection_years=DURATION_HORIZON)
    values = [value(k) for k in range(DURATION_HORIZON + 1)]
    extras = {"value_held_zero": values[0], "held_growth": held_growth}
    reached = next((k for k, v in enumerate(values) if v >= price), None)
    if reached is not None:
        extras["path"] = duration_path(held_growth, reached, case)
        if reached == DURATION_HORIZON:
            held = f"held for all {DURATION_HORIZON} years of the projection"
        else:
            held = (
                f"held for {reached} year{'s' if reached != 1 else ''} before it fades to "
                f"{case.assumptions.dcf.revenue_growth_terminal:.1%}"
            )
        sentence = (
            f"At {price:,.2f} the price needs {held_growth:.1%} growth {held}, on a "
            f"{DURATION_HORIZON}-year projection worth {values[0]:,.2f} with none held."
        )
        return Solve(
            "duration", price, None, float(reached), (0.0, float(DURATION_HORIZON)), sentence=sentence, extras=extras
        )
    extras["path"] = duration_path(held_growth, DURATION_HORIZON, case)
    sentence = (
        f"Even {held_growth:.1%} growth held for all {DURATION_HORIZON} years is worth {values[-1]:,.2f}, "
        f"short of {price:,.2f}."
    )
    return Solve(
        "duration", price, None, None, (0.0, float(DURATION_HORIZON)), float(DURATION_HORIZON), values[-1], sentence, extras
    )


# Base rates read an h-year CAGR; five years is the base case's horizon and
# the deepest horizon the fade panel labels.
BASE_RATE_HORIZON = 5
# "Similar starters": trailing growth within this many points of the company's.
SIMILAR_BAND = 0.10


def path_cagr(path: list[float], horizon: int) -> float:
    """The compound annual growth of the first ``horizon`` years of a path."""
    years = path[:horizon]
    growth = 1.0
    for g in years:
        growth *= 1.0 + g
    return growth ** (1.0 / len(years)) - 1.0


def growth_base_rate(
    observations,
    *,
    implied_cagr: float,
    horizon: int = BASE_RATE_HORIZON,
    trailing: float | None = None,
    band: float = SIMILAR_BAND,
) -> dict:
    """How often a TMT company-year went on to compound revenue this fast.

    Each labelled observation's realised CAGR over ``horizon`` years is built
    from the growth each later 10-K printed, so it is the record as filed. The
    share is reported across every observation with a full ``horizon`` of
    labels, and among those whose own trailing growth was within ``band`` of
    the company's, because a company growing 28% is not drawn from the same
    population as one growing 3%. The panel keeps delisted filers, so the
    companies that stopped growing and were bought are in the denominator.
    """
    def realised(obs) -> float | None:
        if not all(h in obs.labels for h in range(1, horizon + 1)):
            return None
        return path_cagr([obs.labels[h] for h in range(1, horizon + 1)], horizon)

    rows = [(obs, realised(obs)) for obs in observations]
    rows = [(obs, r) for obs, r in rows if r is not None]

    def tally(subset) -> dict:
        n = len(subset)
        hits = sum(1 for _, r in subset if r >= implied_cagr)
        return {"n": n, "hits": hits, "share": hits / n if n else None}

    similar = None
    if trailing is not None:
        near = [(o, r) for o, r in rows if o.growth is not None and abs(o.growth - trailing) <= band]
        similar = {**tally(near), "trailing": trailing, "band": band}
    return {"horizon": horizon, "implied_cagr": implied_cagr, "all": tally(rows), "similar": similar}


# Terminal margins the frontier is read at: a services business to the best
# software franchise anyone has priced.
FRONTIER_MARGINS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


def growth_margin_frontier(case: Case, price: float, margins=FRONTIER_MARGINS) -> list[dict]:
    """The first-year growth the price needs at each terminal margin.

    Two levers that each fail alone can succeed together, and this is the
    curve along which they trade: a row per margin, with the implied growth or
    the value the growth bracket's edge reaches when even 300% will not do.
    """
    rows = []
    for m in margins:
        solve = implied_first_year_growth(case.with_dcf(ebit_margin_terminal=m), price)
        rows.append(
            {
                "margin": m,
                "implied_growth": solve.implied,
                "edge_value": None if solve.reached else solve.edge_value,
            }
        )
    return rows
