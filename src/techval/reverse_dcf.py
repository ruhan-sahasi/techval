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
