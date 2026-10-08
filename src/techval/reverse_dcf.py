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
# "Similar size": revenue within this factor of the company's, either way.
SIZE_FACTOR = 2.0


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
    revenue_mm: float | None = None,
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
    # Size: large companies grow more slowly, so a company is also judged
    # against company-years of its own scale. The panel stores raw dollars;
    # Financials states millions.
    size = None
    if revenue_mm is not None and revenue_mm > 0:
        low, high = revenue_mm / SIZE_FACTOR, revenue_mm * SIZE_FACTOR
        sized = [(o, r) for o, r in rows if low <= o.revenue / 1e6 <= high]
        size = {**tally(sized), "low_mm": low, "high_mm": high}
    return {"horizon": horizon, "implied_cagr": implied_cagr, "all": tally(rows), "similar": similar, "size": size}


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


@dataclass
class MarketExpectations:
    """What one price assumes, lever by lever, with the base rates that judge it."""

    ticker: str
    price: float
    base_value: float
    solves: list[Solve]
    frontier: list[dict]
    base_rates: dict
    notes: list[str] = field(default_factory=list)
    persistence: list[dict] = field(default_factory=list)
    simulation: dict | None = None

    def solve(self, lever: str) -> Solve:
        return next(s for s in self.solves if s.lever == lever)

    def summary(self) -> list[str]:
        gap = self.price / self.base_value - 1
        out = [
            f"{self.ticker} closed at {self.price:,.2f}; the engine's base case is worth "
            f"{self.base_value:,.2f}, so the price is {abs(gap):.0%} {'above' if gap >= 0 else 'below'} it."
        ]
        out += [s.sentence for s in self.solves]
        if self.simulation:
            out.append(self.simulation["sentence"])
        for lever in ("first_year_growth", "duration"):
            rate = self.base_rates.get(lever)
            if not rate:
                continue
            every = rate["all"]
            line = (
                f"{'That path' if lever == 'first_year_growth' else 'Its first five years'} "
                f"compound{'s' if lever == 'first_year_growth' else ''} "
                f"{rate['implied_cagr']:.1%} a year over {rate['horizon']} years: {every['hits']:,} of "
                f"{every['n']:,} TMT company-years did that ({every['share']:.1%})"
            )
            similar = rate["similar"]
            if similar and similar["n"]:
                line += (
                    f", and {similar['hits']:,} of the {similar['n']:,} that started within "
                    f"{similar['band'] * 100:.0f} points of {similar['trailing']:.1%}"
                )
            size = rate.get("size")
            if size and size["n"]:
                who = "none" if size["hits"] == 0 else f"{size['hits']:,}"
                line += (
                    f"; {who} of the {size['n']:,} with revenue between {size['low_mm']:,.0f}mm and "
                    f"{size['high_mm']:,.0f}mm did"
                )
            out.append(line + ".")
            if lever == "duration":
                held = self.solve("duration")
                years = held.implied if held.reached else held.edge
                if years and years > rate["horizon"]:
                    out.append(
                        f"The panel's labels end at {rate['horizon']} years, so nothing in it tests the "
                        f"remaining {int(years) - rate['horizon']}."
                    )
        if self.persistence and self.persistence[0]["n"]:
            held = self.solve("duration").extras.get("held_growth")
            by = {c["horizon"]: c["share"] for c in self.persistence}
            out.append(
                f"Held year after year, {held:.1%} growth is rarer than its average: "
                f"{by[1]:.1%} of company-years managed it for one year, {by.get(3, 0):.1%} for three "
                f"and {by.get(5, 0):.1%} for five, of {self.persistence[0]['n']:,} labelled five years out."
            )
        out += self.notes
        return out

    def to_dict(self) -> dict:
        def solve_dict(s: Solve) -> dict:
            return {
                "lever": s.lever,
                "assumed": s.assumed,
                "implied": s.implied,
                "reached": s.reached,
                "bracket": list(s.bracket),
                "edge": s.edge,
                "edge_value": s.edge_value,
                "sentence": s.sentence,
                "path": s.extras.get("path"),
            }

        return {
            "ticker": self.ticker,
            "price": self.price,
            "base_value": self.base_value,
            "solves": [solve_dict(s) for s in self.solves],
            "frontier": self.frontier,
            "base_rates": self.base_rates,
            "persistence": self.persistence,
            "simulation": self.simulation,
            "summary": self.summary(),
        }


def market_expectations(
    case: Case,
    price: float,
    *,
    trailing_growth: float | None,
    observations=None,
) -> MarketExpectations:
    """Every lever, the frontier, and base rates when a panel is supplied.

    ``trailing_growth`` is the growth the duration lever holds and the centre
    of the similar-starters base rate; without it the duration holds the
    assumed first-year growth and says so.
    """
    notes: list[str] = []
    held = trailing_growth
    if held is None:
        held = case.assumptions.dcf.revenue_growth_start
        notes.append(
            f"No trailing growth was supplied, so the duration holds the assumed {held:.1%} first-year growth."
        )
    growth = implied_first_year_growth(case, price)
    duration = implied_duration(case, price, held_growth=held)
    solves = [implied_discount_rate(case, price), growth, implied_terminal_margin(case, price), duration]
    base_rates: dict = {}
    if observations is None:
        notes.append("There is no fade panel on this run, so no base rate judges the implied growth.")
    else:
        for solve in (growth, duration):
            path = solve.extras.get("path")
            if not path:
                continue
            horizon = min(BASE_RATE_HORIZON, len(path))
            base_rates[solve.lever] = growth_base_rate(
                observations,
                implied_cagr=path_cagr(path, horizon),
                horizon=horizon,
                trailing=trailing_growth,
                revenue_mm=getattr(case.fin, "revenue", None),
            )
    persistence = [] if observations is None else persistence_curve(observations, growth=held)
    return MarketExpectations(
        ticker=case.fin.ticker,
        price=price,
        base_value=case.value(),
        solves=solves,
        frontier=growth_margin_frontier(case, price),
        base_rates=base_rates,
        notes=notes,
        persistence=persistence,
        simulation=price_in_simulation(case, price),
    )


def persistence_curve(observations, *, growth: float, horizons: int = BASE_RATE_HORIZON) -> list[dict]:
    """The share of company-years that grew at least ``growth`` in every one of h years.

    A survival curve for a growth rate, which is the question the duration
    lever asks: not whether a five-year average reached the rate, but whether
    the company grew that fast year after year. One cohort serves every
    horizon, the company-years labelled all ``horizons`` years out, so each
    added year can only lose companies and the curve never rises.
    """
    cohort = [o for o in observations if all(h in o.labels for h in range(1, horizons + 1))]
    out = []
    survivors = cohort
    for h in range(1, horizons + 1):
        survivors = [o for o in survivors if o.labels[h] >= growth]
        n = len(cohort)
        out.append({"horizon": h, "n": n, "held": len(survivors), "share": len(survivors) / n if n else None})
    return out


def price_in_simulation(case: Case, price: float) -> dict | None:
    """Where a price sits in the engine's own Monte Carlo of value.

    The simulation draws first-year growth, terminal margin, the discount rate
    and terminal growth jointly, with the correlations its module states, and
    revalues on every draw. The share of draws worth more than the price is the
    assumed distribution's own verdict on it: a probability under the model's
    assumptions, not about the world. None when the simulation refuses.
    """
    from .simulation import run_simulation

    try:
        result = run_simulation(case.fin, case.bridge, case.wacc, case.assumptions)
    except TechvalError:
        return None
    draws = result.per_share_draws
    above = float((draws > price).mean()) if len(draws) else None
    stats = result.per_share
    if above == 0.0:
        sentence = (
            f"None of the {len(draws):,} joint draws of growth, margin, discount rate and terminal "
            f"growth reaches {price:,.2f}; the 95th percentile is {stats.p95:,.2f}."
        )
    else:
        sentence = (
            f"{above:.1%} of the {len(draws):,} joint draws of growth, margin, discount rate and "
            f"terminal growth are worth more than {price:,.2f}; the median is {stats.p50:,.2f}."
        )
    return {
        "kept": len(draws),
        "share_above": above,
        "p5": stats.p5,
        "p50": stats.p50,
        "p95": stats.p95,
        "sentence": sentence,
    }


def quarter_ends(first, last) -> list:
    """Calendar quarter ends falling between ``first`` and ``last``, inclusive."""
    from datetime import date

    return [
        date(year, month, day)
        for year in range(first.year, last.year + 1)
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31))
        if first <= date(year, month, day) <= last
    ]


def expectations_history(payload: dict, ticker: str, source, assumptions: Assumptions, dates) -> list[dict]:
    """What the price assumed at each date, from only what was public by then.

    Each date rebuilds the whole forward case point in time: the filings known
    by that date through ``CompanyFacts(knowledge_date=...)``, the close on or
    before it, a cost of capital from the price history up to it. Then the two
    levers a holder reads first are solved. A date the record cannot support,
    too little filed history or too little price history for a beta, is kept
    as a row with the refusal, never dropped.
    """
    from .edgar import CompanyFacts, HttpCache
    from .ev_bridge import build_ev_bridge
    from .financials import build_financials
    from .market import MarketData
    from .wacc import compute_wacc

    rows = []
    for when in dates:
        row = {"date": when.isoformat(), "refused": None}
        try:
            fin = build_financials(ticker, facts=CompanyFacts(payload, ticker, knowledge_date=when))
            market = MarketData(source, HttpCache(enabled=False), today=when)
            price = market.spot(ticker)
            bridge = build_ev_bridge(fin, price, assumptions)
            wacc = compute_wacc(fin, bridge, market, assumptions)
            case = Case(fin=fin, bridge=bridge, wacc=wacc, assumptions=assumptions)
            rate = implied_discount_rate(case, price)
            growth = implied_first_year_growth(case, price)
        except TechvalError as err:
            row["refused"] = str(err).splitlines()[0]
            rows.append(row)
            continue
        row.update(
            price=round(price, 4),
            base_value=round(case.value(), 4),
            wacc=round(wacc.wacc, 6),
            implied_return=None if rate.implied is None else round(rate.implied, 6),
            implied_growth=None if growth.implied is None else round(growth.implied, 6),
            revenue_mm=round(fin.revenue, 2),
            filings_through=fin.as_of.isoformat(),
        )
        rows.append(row)
    return rows
