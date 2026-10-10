"""Bear, base and bull, from what similar companies went on to do.

A scenario table is usually three sets of numbers an analyst chose. Here each
scenario is read off the fade panel's point-in-time company-years instead: of
the company-years that started within ten points of the company's own trailing
growth, the ones around the 10th, 50th and 90th percentile of realised
five-year revenue growth, and the mean year-by-year path they took. A path
keeps its shape, so a neighbourhood of companies that decelerated hard is a
bear case that decelerates hard, not a constant rate with the same CAGR.

Only the growth path changes between scenarios. Margins, reinvestment, the
discount rate and the perpetuity are the base case's in all three, because the
panel tests revenue growth for five years and nothing else; a scenario that
also moved the margin would be mixing evidence with an opinion and labelling
both as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .reverse_dcf import BASE_RATE_HORIZON, SIMILAR_BAND, path_cagr

# The percentiles each scenario is read at, and the half-width of the
# neighbourhood averaged around each: 10th percentile means ranks 5 to 15.
SCENARIO_QUANTILES = {"bear": 0.10, "base": 0.50, "bull": 0.90}
NEIGHBOURHOOD = 0.05
# Fewer similar starters than this and a percentile is a handful of companies,
# so the pool widens to every company-year and says so.
MIN_SIMILAR = 60


@dataclass(frozen=True)
class Pool:
    """The realised paths the scenarios are drawn from, and which company-years they are."""

    paths: list[list[float]]
    similar: bool
    trailing: float | None
    band: float
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.paths)


@dataclass(frozen=True)
class ScenarioPath:
    """One scenario's growth: the mean path of a percentile neighbourhood."""

    name: str
    quantile: float
    growth: list[float]
    cagr: float
    n: int
    low_cagr: float
    high_cagr: float


def realised_paths(
    observations,
    *,
    trailing: float | None,
    horizon: int = BASE_RATE_HORIZON,
    band: float = SIMILAR_BAND,
    min_similar: int = MIN_SIMILAR,
) -> Pool:
    """Every labelled company-year's realised growth path, among similar starters when there are enough."""
    def path(obs) -> list[float] | None:
        if not all(h in obs.labels for h in range(1, horizon + 1)):
            return None
        return [float(obs.labels[h]) for h in range(1, horizon + 1)]

    rows = [(obs, path(obs)) for obs in observations]
    rows = [(obs, p) for obs, p in rows if p is not None]
    if trailing is not None:
        near = [p for obs, p in rows if obs.growth is not None and abs(obs.growth - trailing) <= band]
        if len(near) >= min_similar:
            return Pool(paths=near, similar=True, trailing=trailing, band=band)
        note = (
            f"{len(near)} company-years started within {band * 100:.0f} points of {trailing:.1%}, fewer than "
            f"{min_similar}, so the scenarios are read from every labelled company-year."
        )
    else:
        note = "No trailing growth to compare, so the scenarios are read from every labelled company-year."
    return Pool(paths=[p for _, p in rows], similar=False, trailing=trailing, band=band, note=note)


def quantile_paths(
    paths: list[list[float]],
    quantiles: dict[str, float] = SCENARIO_QUANTILES,
    neighbourhood: float = NEIGHBOURHOOD,
) -> list[ScenarioPath]:
    """The mean year-by-year path of the company-years around each percentile of CAGR."""
    if not paths:
        return []
    horizon = len(paths[0])
    ranked = sorted(paths, key=lambda p: path_cagr(p, horizon))
    n = len(ranked)
    out = []
    for name, q in quantiles.items():
        lo = max(0, min(n - 1, round((q - neighbourhood) * n)))
        hi = min(n, max(lo + 1, round((q + neighbourhood) * n)))
        members = ranked[lo:hi]
        growth = [sum(p[y] for p in members) / len(members) for y in range(horizon)]
        out.append(
            ScenarioPath(
                name=name,
                quantile=q,
                growth=growth,
                cagr=path_cagr(growth, horizon),
                n=len(members),
                low_cagr=path_cagr(members[0], horizon),
                high_cagr=path_cagr(members[-1], horizon),
            )
        )
    return out


# Swanson's rule: P10, P50 and P90 weighted 30/40/30 approximate the mean of a
# skewed distribution far better than an equal three-way split, which overweights
# both tails, and it is the convention the scenario table states.
SWANSON_WEIGHTS = {"bear": 0.30, "base": 0.40, "bull": 0.30}


def fit_path(growth: list[float], years: int, terminal: float) -> list[float]:
    """A scenario path sized to the projection: cut short, or faded to terminal after the panel's years."""
    if years <= len(growth):
        return list(growth[:years])
    rest = years - len(growth)
    last = growth[-1]
    return list(growth) + [last + (terminal - last) * (i + 1) / rest for i in range(rest)]


@dataclass(frozen=True)
class Scenario:
    """One scenario valued: its path through the engine's DCF, and its weight."""

    name: str
    quantile: float
    weight: float
    growth: list[float]
    cagr: float
    n: int
    value: float


@dataclass
class Scenarios:
    """Three scenarios from the panel, valued, weighted, and set against the price."""

    ticker: str
    price: float
    base_value: float
    scenarios: list[Scenario]
    pool: Pool
    weighted: float
    implied: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def scenario(self, name: str) -> Scenario:
        return next(s for s in self.scenarios if s.name == name)

    @property
    def sentences(self) -> list[str]:
        bear, base, bull = (self.scenario(n) for n in ("bear", "base", "bull"))
        if self.pool.similar:
            who = (
                f"Of the {self.pool.n:,} company-years that started within {self.pool.band * 100:.0f} "
                f"points of {self.pool.trailing:.1%} growth"
            )
        else:
            who = f"Of all {self.pool.n:,} labelled company-years"
        return [
            f"{who}, the neighbourhoods around the 10th, 50th and 90th percentile compounded "
            f"{bear.cagr:.1%}, {base.cagr:.1%} and {bull.cagr:.1%} a year over five years; on those "
            f"paths, with everything else at the base case, the engine values {self.ticker} at "
            f"{bear.value:,.2f}, {base.value:,.2f} and {bull.value:,.2f}.",
            f"Weighted {bear.weight * 100:.0f}/{base.weight * 100:.0f}/{bull.weight * 100:.0f}, the scenarios are worth "
            f"{self.weighted:,.2f} against a close of {self.price:,.2f} and a base case of {self.base_value:,.2f}.",
            self.implied["sentence"],
        ]

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "price": self.price,
            "base_value": self.base_value,
            "weighted": self.weighted,
            "pool": {
                "n": self.pool.n,
                "similar": self.pool.similar,
                "trailing": self.pool.trailing,
                "band": self.pool.band,
                "note": self.pool.note,
            },
            "scenarios": [
                {
                    "name": s.name,
                    "quantile": s.quantile,
                    "weight": s.weight,
                    "growth": s.growth,
                    "cagr": s.cagr,
                    "n": s.n,
                    "value": s.value,
                }
                for s in self.scenarios
            ],
            "implied": self.implied,
            "sentences": self.sentences,
            "notes": self.notes,
        }


def value_scenarios(
    case,
    price: float,
    observations,
    *,
    trailing: float | None,
    weights: dict[str, float] = SWANSON_WEIGHTS,
    quantiles: dict[str, float] = SCENARIO_QUANTILES,
) -> Scenarios | None:
    """Value bear, base and bull through the engine's own DCF; None with no labelled panel."""
    from .errors import ConfigError

    if set(weights) != set(quantiles):
        raise ConfigError(f"scenario weights name {sorted(weights)} and quantiles name {sorted(quantiles)}")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ConfigError(f"scenario weights sum to {sum(weights.values()):.4f}, not 1")
    pool = realised_paths(observations, trailing=trailing)
    paths = quantile_paths(pool.paths, quantiles)
    if not paths:
        return None
    dcf = case.assumptions.dcf
    scenarios = []
    for p in paths:
        growth = fit_path(p.growth, dcf.projection_years, dcf.revenue_growth_terminal)
        scenarios.append(
            Scenario(
                name=p.name,
                quantile=p.quantile,
                weight=weights[p.name],
                growth=growth,
                cagr=p.cagr,
                n=p.n,
                value=case.value(growth_path=growth),
            )
        )
    weighted = sum(s.weight * s.value for s in scenarios)
    notes = [pool.note] if pool.note else []
    return Scenarios(
        ticker=case.fin.ticker,
        price=price,
        base_value=case.value(),
        scenarios=scenarios,
        pool=pool,
        weighted=weighted,
        implied=implied_weight(scenarios, price),
        notes=notes,
    )


def implied_weight(scenarios: list[Scenario], price: float) -> dict:
    """The weight the price puts on one tail, the other two held in their stated ratio.

    A price above the weighted value is read as a heavier bull case, with bear
    and base kept in proportion; below it, as a heavier bear case, with base and
    bull kept in proportion. Outside the three values no weighting reaches the
    price, and the reading says how far outside rather than inventing a fourth
    scenario.
    """
    by = {s.name: s for s in scenarios}
    bear, base, bull = by["bear"], by["base"], by["bull"]
    if price > bull.value:
        return {
            "side": "above",
            "weight": None,
            "sentence": (
                f"At {price:,.2f} the price is above even the bull case, {bull.value:,.2f}, "
                f"{price / bull.value:.1f} times the value of the path the 90th percentile took; "
                "no weighting of the three reaches it."
            ),
        }
    if price < bear.value:
        return {
            "side": "below",
            "weight": None,
            "sentence": (
                f"At {price:,.2f} the price is below even the bear case, {bear.value:,.2f}; "
                "no weighting of the three reaches it."
            ),
        }

    def solve(tail, a, b):
        rest = (a.weight * a.value + b.weight * b.value) / (a.weight + b.weight)
        return (price - rest) / (tail.value - rest) if tail.value != rest else tail.weight

    weighted = sum(s.weight * s.value for s in scenarios)
    tail, (a, b) = (bull, (bear, base)) if price >= weighted else (bear, (base, bull))
    w = solve(tail, a, b)
    return {
        "side": tail.name,
        "weight": w,
        "sentence": (
            f"At {price:,.2f} the price puts {w:.0%} on the {tail.name} case, against "
            f"{tail.weight:.0%} under Swanson's rule, with the other two held in proportion."
        ),
    }
