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
            f"{len(near)} company-years started within {band:.0%} of {trailing:.1%}, fewer than "
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
