"""Monte Carlo over the four assumptions that actually move a DCF.

A discounted cash flow prints one number. That number is a function of four
judgments no one can observe, so the honest output is a distribution and a
probability statement about it. This module draws year-one revenue growth, the
terminal EBIT margin, the discount rate and terminal growth together, revalues
the company on every draw, and reports the percentiles of value per share,
the percentiles of enterprise value, and the probability that value clears the
quoted price. Money is USD millions, share counts are millions, per-share
figures are dollars.

**The draw is joint, and that is the whole point.** The commonest error in DCF
Monte Carlo is four independent normals. Independence asserts that a company
which grows twice as fast as expected is no more likely than average to reach
a better terminal margin, and that the market would discount the two cases at
the same rate. Neither is true of any business anyone has ever covered. The
draws here come from a multivariate normal with this correlation matrix, which
is a module-level constant so a reader can see it and change it:

                       growth   margin     WACC   term g
        growth           1.00     0.35    -0.25     0.10
        margin           0.35     1.00    -0.20     0.05
        WACC            -0.25    -0.20     1.00     0.00
        term g           0.10     0.05     0.00     1.00

    growth / margin  +0.35  operating leverage. Fixed cost is spread over a
                            larger base, and the company that wins the market
                            is also the one that stops discounting to win it.
    growth / WACC    -0.25  the market discounts a better business at a lower
                            rate: scale buys index inclusion, lower funding
                            cost and a lower equity risk premium on the name.
    margin / WACC    -0.20  same mechanism, weaker. Cash generation reduces
                            financing risk, though a high-margin business is
                            not automatically a low-beta one.
    term g / others   0.10  and 0.05, which is near enough to independent. The
                            perpetuity growth rate is a claim about the economy
                            in fifteen years, not about this year's execution,
                            and treating it as an echo of near-term momentum is
                            how a cyclical peak gets capitalised forever.

The matrix is verified positive semi-definite before anything is drawn. A
correlation matrix that is not PSD describes no joint distribution at all: some
weighted combination of the drivers would have negative variance. The error
names the pair the offending direction loads on, because in practice the
failure is a transitivity one, two strong correlations of opposite sign forcing
a third that was entered by hand.

**What correlation does to the spread here, which is the opposite of the usual
claim.** Analysts often say correlating the drivers narrows the answer, on the
reasoning that independence generates combinations that do not occur. It
generates them, but look at which ones. Value rises with growth, rises with
margin and falls with WACC, so the three correlations above all point the same
way: the good draw is fast growth AND a fat margin AND a low discount rate, and
the bad draw is all three against you. Independence puts mass on the offsetting
middle, fast growth at a thin margin, and the middle is where the point
estimate already sits. Writing the variance of a linear approximation out,

    Var = a^2 sG^2 + b^2 sM^2 + c^2 sW^2
          + 2ab corr(G,M) sG sM - 2ac corr(G,W) sG sW - 2bc corr(M,W) sM sW

with a, b, c all positive, every one of the three cross terms is positive at
the signs above. The correlated distribution is WIDER than the independent one,
by 21% of a standard deviation on each of Datadog, CrowdStrike, MongoDB and
Zscaler at the default spreads, and it is wider for the right reason: the
scenarios that hurt arrive together. Narrowing would require believing that
growth carries its own discount rate with it, corr(growth, WACC) positive,
which is the riskier-company-higher-beta view. That is a defensible matrix and
it is one line to enter, but it is not the one argued above. The module reports
the spread the stated signs produce rather than the spread the folklore
expects.

**Truncation, not clipping.** A draw where terminal growth lands at or above the
discount rate is rejected outright, not pushed back to the boundary. The Gordon
formula divides by (WACC - g), so those draws are undefined rather than extreme,
and clipping them to g = WACC - epsilon would pile probability mass on the point
where the perpetuity is largest and manufacture a right tail that is pure
artefact. Rejections are counted and reported, and flagged above 5% of the
sample, because a rejection rate that high means the assumed spreads on WACC and
on g are inconsistent with the central case rather than merely wide.

**The probabilities are under the assumed distribution, not about the world.**
``prob_above_price`` is the share of surviving draws whose value exceeds the
quoted price. It is a statement about a model, conditional on four standard
deviations an analyst typed in and a correlation matrix an analyst chose. The
model does not know whether those spreads are right, and a tight distribution
around a wrong central case is confidently wrong. Read the number as: given
this view of the business and this much uncertainty about it, the stock screens
cheap in this fraction of cases. Nothing stronger is available from it.

**The tornado is deliberately univariate.** Each driver is moved to the 5th and
95th percentile of its own marginal while the other three are held at their
central values, which ignores every correlation the simulation is built on.
That is not an oversight. The two outputs answer different questions. The
tornado says which assumption to argue about in the meeting, since it isolates
one at a time and ranks them by the swing in price. The simulation says how
wide the answer is once the assumptions move the way they actually move
together. Reporting only the tornado understates the spread; reporting only the
distribution hides which knob produced it.

**Vectorised, and reconciled against the scalar path.** Ten thousand calls to
``run_dcf`` would rebuild the projection ten thousand times in Python. The
projection arithmetic here runs as numpy over the draw dimension, looping only
over the handful of forecast years, which is a few tens of milliseconds for ten
thousand draws. A vectorised valuation is worth nothing unless it is the same
valuation, so every run revalues the central draw through this path and checks
it against ``run_dcf`` on the same inputs. A disagreement raises rather than
prints: if the two paths have diverged, the distribution is around a number the
rest of the engine does not report.

**Limitations left in.** Only year-one revenue growth is shocked, matching the
assumption the config exposes, so the fade converges on an unshocked terminal
revenue growth rate and the spread on the final explicit year is narrower than
a parallel shift of the whole path would give. The exit-multiple terminal value
is not simulated: the exit multiple is not one of the drawn drivers, and
sampling a peer median alongside terminal growth would put two competing
terminal assumptions into one histogram. The distribution is over assumptions,
not over outcomes: nothing here models the chance that the business is a
different business than the one the projection describes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import Assumptions
from .dcf import _opening_nol, _sbc_ratio, _tax_rate, run_dcf
from .errors import ConfigError
from .ev_bridge import EVBridge, equity_value_from_dcf_ev
from .financials import Financials

if TYPE_CHECKING:  # the WACC build is a peer module, imported for typing only
    from .wacc import WACCResult


# The drawn drivers, in the order the correlation matrix and the shock array use.
DRIVERS = ("revenue_growth", "terminal_margin", "wacc", "terminal_growth")

# The assumption each driver moves, named as it appears in the assumptions file,
# because the point of the tornado is to send someone to the right line of it.
DRIVER_PATHS = {
    "revenue_growth": "dcf.revenue_growth_start",
    "terminal_margin": "dcf.ebit_margin_terminal",
    "wacc": "the computed WACC",
    "terminal_growth": "dcf.terminal_growth",
}

DRIVER_LABELS = {
    "revenue_growth": "Year-one revenue growth",
    "terminal_margin": "Terminal EBIT margin",
    "wacc": "WACC",
    "terminal_growth": "Terminal growth",
}

# The same four drivers as they read inside a sentence.
DRIVER_PROSE = {
    "revenue_growth": "year-one revenue growth",
    "terminal_margin": "the terminal margin",
    "wacc": "the WACC",
    "terminal_growth": "terminal growth",
}

# The joint view of the four drivers. Signs are argued in the module docstring;
# every omitted pair is zero. Edit this to change the shape of the distribution.
DRIVER_CORRELATIONS: dict[tuple[str, str], float] = {
    ("revenue_growth", "terminal_margin"): 0.35,
    ("revenue_growth", "wacc"): -0.25,
    ("terminal_margin", "wacc"): -0.20,
    ("revenue_growth", "terminal_growth"): 0.10,
    ("terminal_margin", "terminal_growth"): 0.05,
    ("wacc", "terminal_growth"): 0.00,
}

# Diagnostic thresholds and numerical tolerances. None of them moves a price.
_FAILED_DRAW_FLAG = 0.05
_PSD_TOLERANCE = 1e-8
_RECONCILIATION_TOLERANCE = 1e-7

# The standard normal deviate at the 95th percentile, which sets the tornado's
# endpoints so that a driver's low and high case are the same 5/95 band the
# simulation itself draws from.
_Z95 = 1.6448536269514722


# -- the joint distribution ------------------------------------------------ #


def correlation_matrix(
    correlations: Mapping[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """Assemble the driver correlation matrix and prove it is a real one.

    Any pair not named is zero. The matrix is checked for unit diagonal by
    construction, for correlations inside [-1, 1], and for positive
    semi-definiteness, which is the condition that matters: a symmetric matrix
    with a negative eigenvalue has an eigenvector whose weighted combination of
    the drivers carries negative variance, and no set of random variables
    behaves that way. Such a matrix is not a tight assumption or an aggressive
    one, it is not an assumption at all, and sampling from it would either fail
    or quietly return draws whose pairwise correlations are not the ones asked
    for.

    The usual cause is transitivity. Correlations are cosines of angles: if
    growth and margin are strongly positive and growth and WACC strongly
    negative, then margin and WACC cannot be freely chosen, and a hand-entered
    third number outside the implied window breaks the matrix. The error names
    the pair the offending direction loads on hardest.
    """
    pairs = DRIVER_CORRELATIONS if correlations is None else correlations
    index = {name: i for i, name in enumerate(DRIVERS)}
    matrix = np.eye(len(DRIVERS))

    for (left, right), rho in pairs.items():
        if left not in index or right not in index:
            raise ConfigError(
                f"correlation ({left!r}, {right!r}) names a driver this simulation "
                f"does not draw. The drivers are {', '.join(DRIVERS)}."
            )
        if not -1.0 <= rho <= 1.0:
            raise ConfigError(
                f"correlation between {left} and {right} is {rho:.2f}, outside "
                "[-1, 1]. A correlation is a cosine; there is no such angle."
            )
        i, j = index[left], index[right]
        if i == j:
            raise ConfigError(
                f"{left} cannot be given a correlation with itself. The diagonal "
                "of a correlation matrix is one by definition."
            )
        matrix[i, j] = matrix[j, i] = float(rho)

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    smallest = float(eigenvalues[0])
    if smallest < -_PSD_TOLERANCE:
        direction = eigenvectors[:, 0]
        left, right = _offending_pair(direction)
        raise ConfigError(
            f"the driver correlation matrix is not positive semi-definite: its "
            f"smallest eigenvalue is {smallest:.4f}. The combination "
            f"{_combination(direction)} would have a variance of {smallest:.4f}, "
            f"which no set of random variables can have, so this matrix describes "
            f"no joint distribution and nothing can be drawn from it. The "
            f"offending direction loads hardest on {left} and {right}, at a "
            f"correlation of {matrix[DRIVERS.index(left), DRIVERS.index(right)]:+.2f}. "
            "Correlations are not independently choosable: two strong pairings fix "
            "a window the third has to sit inside. Move that pair towards zero."
        )
    return matrix


def _offending_pair(direction: np.ndarray) -> tuple[str, str]:
    """The two drivers the negative-variance direction weights most heavily."""
    order = np.argsort(np.abs(direction))[::-1]
    i, j = sorted((int(order[0]), int(order[1])))
    return DRIVERS[i], DRIVERS[j]


def _combination(direction: np.ndarray) -> str:
    terms = [f"{w:+.2f} {name}" for w, name in zip(direction, DRIVERS)]
    return " ".join(terms).lstrip("+").strip()


def _factor(matrix: np.ndarray) -> np.ndarray:
    """A matrix L with L @ L.T equal to the correlation matrix.

    Cholesky where the matrix is strictly positive definite, which is the
    ordinary case and the cheaper one. A matrix that is positive semi-definite
    but singular, which happens the moment two drivers are given a correlation
    of exactly one, has no Cholesky factor, and the symmetric eigenvalue square
    root is used instead. Both give draws with the requested covariance; only
    the orientation of the underlying normals differs, which is why the seed and
    the factorisation are reported together.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))


def _standard_deviations(assumptions: Assumptions) -> np.ndarray:
    cfg = assumptions.simulation
    sds = np.array(
        [
            cfg.revenue_growth_sd,
            cfg.terminal_margin_sd,
            cfg.wacc_sd,
            cfg.terminal_growth_sd,
        ],
        dtype=float,
    )
    for name, sd in zip(DRIVERS, sds):
        if sd < 0:
            raise ConfigError(
                f"simulation.{name}_sd is {sd:.4f}. A standard deviation is a "
                "square root and cannot be negative; to pin a driver at its "
                "central value set it to zero."
            )
    return sds


def draw_shocks(
    assumptions: Assumptions,
    correlations: Mapping[tuple[str, str], float] | None = None,
    independent: bool = False,
) -> np.ndarray:
    """Draws of the four drivers, shaped (draws, 4), as shocks around zero.

    Seeded from ``simulation.seed`` through ``numpy.random.default_rng``, never
    from the global random state, so the same assumptions file produces the same
    distribution on any machine on any day. A valuation that moves when nothing
    moved is not evidence of anything.

    ``independent`` zeroes the off-diagonal and draws from the same underlying
    standard normals, which is what makes the comparison between the correlated
    and the independent distribution a like-for-like one rather than two
    different samples.
    """
    cfg = assumptions.simulation
    sds = _standard_deviations(assumptions)
    matrix = np.eye(len(DRIVERS)) if independent else correlation_matrix(correlations)
    rng = np.random.default_rng(cfg.seed)
    normals = rng.standard_normal((cfg.draws, len(DRIVERS)))
    return (normals @ _factor(matrix).T) * sds


# -- results --------------------------------------------------------------- #


@dataclass
class Percentiles:
    """A distribution reported at the five points that get quoted in a note.

    ``sd`` is the sample standard deviation of the surviving draws. The mean
    sits beside the median deliberately: a DCF distribution is right-skewed,
    because the Gordon denominator is convex in the discount rate, so a mean
    above the median is the normal case and a large gap between them is the
    signal that the terminal assumption is carrying the valuation.
    """

    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    mean: float
    sd: float

    @classmethod
    def of(cls, values: np.ndarray) -> "Percentiles":
        p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
        return cls(
            p5=float(p5),
            p25=float(p25),
            p50=float(p50),
            p75=float(p75),
            p95=float(p95),
            mean=float(np.mean(values)),
            sd=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        )

    def rows(self) -> list[tuple[str, float]]:
        return [
            ("5th percentile", self.p5),
            ("25th percentile", self.p25),
            ("Median", self.p50),
            ("75th percentile", self.p75),
            ("95th percentile", self.p95),
            ("Mean", self.mean),
            ("Standard deviation", self.sd),
        ]


@dataclass
class DriverSensitivity:
    """One bar of the tornado: a driver at its 5th and 95th percentile.

    ``low_value`` and ``high_value`` are the driver's own values, not shocks, so
    the bar can be read without reference to the standard deviation that
    produced it. ``low_price`` is the value per share at ``low_value``, which for
    the WACC driver is the HIGHER of the two prices, since value falls as the
    discount rate rises. ``swing`` is the absolute gap and is what the ordering
    uses, because the question a tornado answers is which assumption moves the
    answer most, not in which direction.
    """

    driver: str
    label: str
    low_value: float
    high_value: float
    low_price: float
    high_price: float
    swing: float


@dataclass
class SimulationResult:
    """The distribution, the probabilities, and what was thrown away.

    ``draws`` is the number requested. ``failed_draws`` is how many of them were
    rejected because terminal growth reached the discount rate or the terminal
    value could not be formed, so the percentiles are struck on
    ``draws - failed_draws`` survivors. Lines in ``checks`` that begin with
    ``FLAG:`` failed a threshold; the rest are stated for the record.

    ``prob_above_price`` and ``prob_upside_20pct`` are probabilities under the
    assumed distribution. They are not probabilities about the world, and the
    module docstring says why at length.
    """

    draws: int
    seed: int
    per_share: Percentiles
    enterprise_value: Percentiles
    current_price: float
    prob_above_price: float
    prob_upside_20pct: float
    tornado: list[DriverSensitivity]
    failed_draws: int
    notes: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    central_per_share: float = float("nan")

    @property
    def kept_draws(self) -> int:
        return self.draws - self.failed_draws

    def rows(self) -> list[dict]:
        return [
            {
                "Statistic": label,
                "Per share": per_share,
                "Enterprise value": enterprise,
            }
            for (label, per_share), (_, enterprise) in zip(
                self.per_share.rows(), self.enterprise_value.rows()
            )
        ]

    def to_frame(self) -> pd.DataFrame:
        """The percentile table as raw numbers. Formatting belongs to the renderer."""
        return pd.DataFrame(self.rows()).set_index("Statistic")

    def tornado_rows(self) -> list[dict]:
        return [
            {
                "Driver": bar.label,
                "Assumption": DRIVER_PATHS[bar.driver],
                "Low": bar.low_value,
                "High": bar.high_value,
                "Price at low": bar.low_price,
                "Price at high": bar.high_price,
                "Swing": bar.swing,
            }
            for bar in self.tornado
        ]

    def tornado_frame(self) -> pd.DataFrame:
        """One row per driver, already sorted widest swing first."""
        return pd.DataFrame(self.tornado_rows()).set_index("Driver")


# -- the vectorised valuation ---------------------------------------------- #


@dataclass
class _Engine:
    """The DCF rebuilt as array arithmetic over the draw dimension.

    Every constant is resolved once, from the same helpers ``run_dcf`` uses, so
    the tax rate, the SBC ratio and the opening carryforward cannot drift
    between the scalar path and this one. The reconciliation check would catch a
    drift, but resolving them twice from two pieces of code is how the drift
    gets in.

    The loop runs over forecast years, which are five or ten of them, and the
    draws are the array dimension. That is the inversion that makes ten thousand
    valuations cost about as much as one.
    """

    n: int
    mid_year: bool
    revenue0: float
    shares0: float
    margin_start: float
    growth_terminal: float
    margin_terminal: float
    growth_start: float
    base_wacc: float
    base_g: float
    tax_rate: float
    sbc_ratio: float
    addback: bool
    dilute: bool
    price: float
    da_pct: float
    capex_pct: float
    nwc_pct: float
    track_nol: bool
    opening_nol: float
    nol_limit: float
    method: str
    terminal_roic: float | None
    equity_offset: float

    def values(self, shocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-share value, enterprise value and the survivor mask for each draw.

        Rejected draws come back as NaN rather than as a boundary value. The
        arithmetic is run on the surviving subset only, so a draw where the
        Gordon denominator is zero never reaches a division at all: a warning
        suppressed is a warning someone stops reading.
        """
        shocks = np.atleast_2d(np.asarray(shocks, dtype=float))
        total = shocks.shape[0]
        growth_start = self.growth_start + shocks[:, 0]
        margin_terminal = self.margin_terminal + shocks[:, 1]
        wacc = self.base_wacc + shocks[:, 2]
        g = self.base_g + shocks[:, 3]

        # Truncation, not clipping. Terminal growth at or above the discount rate
        # leaves the perpetuity undefined, and a non-positive cost of capital is
        # not a discount rate whatever the arithmetic would return.
        alive = (g < wacc) & (wacc > 0.0)
        keep = np.flatnonzero(alive)
        per_share = np.full(total, np.nan)
        enterprise = np.full(total, np.nan)
        if keep.size == 0:
            return per_share, enterprise, alive

        growth_start = growth_start[keep]
        margin_terminal = margin_terminal[keep]
        wacc = wacc[keep]
        g = g[keep]
        live = keep.size

        if self.n == 1:
            # A single explicit year is also the terminal year, so the projection
            # takes the terminal assumptions and the year-one growth driver moves
            # nothing at all. That matches the scalar path rather than improving
            # on it: a shock that silently changed meaning with the horizon would
            # be worse than one that is inert and said to be.
            growth_path = np.repeat(self.growth_terminal, live)[:, None]
            margin_path = margin_terminal[:, None]
        else:
            growth_path = np.linspace(
                growth_start, self.growth_terminal, self.n, axis=-1
            )
            margin_path = np.linspace(self.margin_start, margin_terminal, self.n, axis=-1)

        revenue = np.full(live, self.revenue0)
        nwc_prev = np.full(live, self.nwc_pct * self.revenue0)
        shares = np.full(live, self.shares0)
        nol = np.full(live, self.opening_nol)
        pv_explicit = np.zeros(live)
        fcff = np.zeros(live)

        for i in range(self.n):
            revenue = revenue * (1.0 + growth_path[:, i])
            ebit = revenue * margin_path[:, i]
            if self.track_nol:
                loss = ebit < 0.0
                used = np.where(loss, 0.0, np.minimum(nol, self.nol_limit * ebit))
                nol = np.where(loss, nol - ebit, nol - used)
                cash_taxes = self.tax_rate * np.maximum(ebit - used, 0.0)
            else:
                cash_taxes = self.tax_rate * np.maximum(ebit, 0.0)
            nopat = ebit - cash_taxes
            da = self.da_pct * revenue
            capex = self.capex_pct * revenue
            nwc = self.nwc_pct * revenue
            delta_nwc = nwc - nwc_prev
            nwc_prev = nwc
            sbc = self.sbc_ratio * revenue
            fcff = nopat + da - capex - delta_nwc + (sbc if self.addback else 0.0)
            if self.dilute:
                shares = shares + sbc / self.price
            t = (i + 1) - 0.5 if self.mid_year else float(i + 1)
            pv_explicit = pv_explicit + fcff * (1.0 + wacc) ** (-t)

        # Both perpetuities are flows rather than prices, so both carry the
        # half-year uplift under the mid-year convention. An exit multiple would
        # not, which is why it is not simulated here.
        exponent = (self.n - 0.5) if self.mid_year else float(self.n)
        formed = np.ones(live, dtype=bool)

        if self.method == "value_driver":
            steady_revenue = revenue * (1.0 + g)
            steady_ebit = margin_terminal * steady_revenue
            steady_nopat = steady_ebit - self.tax_rate * np.maximum(steady_ebit, 0.0)
            if self.addback:
                steady_nopat = steady_nopat + self.sbc_ratio * steady_revenue
            roic = (
                wacc
                if self.terminal_roic is None
                else np.full(live, self.terminal_roic)
            )
            formed = (steady_nopat > 0.0) & (roic > 0.0)
            safe_nopat = np.where(formed, steady_nopat, 1.0)
            safe_roic = np.where(formed, roic, 1.0)
            terminal = safe_nopat * (1.0 - g / safe_roic) / (wacc - g)
        else:
            terminal = fcff * (1.0 + g) / (wacc - g)

        ev = pv_explicit + terminal * (1.0 + wacc) ** (-exponent)
        value = (ev + self.equity_offset) / shares

        survived = keep[formed]
        per_share[survived] = value[formed]
        enterprise[survived] = ev[formed]
        alive[keep[~formed]] = False
        return per_share, enterprise, alive


def _prepare(
    fin: Financials,
    bridge: EVBridge,
    wacc_result: "WACCResult",
    assumptions: Assumptions,
) -> _Engine:
    cfg = assumptions.dcf
    wacc = cfg.wacc_override if cfg.wacc_override is not None else wacc_result.wacc
    tax_rate, _ = _tax_rate(fin, assumptions)
    sbc_ratio, _ = _sbc_ratio(fin, assumptions)
    addback = cfg.sbc_treatment == "addback"
    dilute = addback and cfg.sbc_dilution and bridge.price > 0
    margin_start = (
        cfg.ebit_margin_start if cfg.ebit_margin_start is not None else fin.ebit_margin
    )
    return _Engine(
        n=cfg.projection_years,
        mid_year=cfg.mid_year_convention,
        revenue0=fin.revenue,
        shares0=fin.diluted_shares,
        margin_start=margin_start,
        growth_terminal=cfg.revenue_growth_terminal,
        margin_terminal=cfg.ebit_margin_terminal,
        growth_start=cfg.revenue_growth_start,
        base_wacc=wacc,
        base_g=cfg.terminal_growth,
        tax_rate=tax_rate,
        sbc_ratio=sbc_ratio,
        addback=addback,
        dilute=dilute,
        price=bridge.price,
        da_pct=cfg.da_pct_revenue,
        capex_pct=cfg.capex_pct_revenue,
        nwc_pct=cfg.nwc_pct_revenue,
        track_nol=cfg.nol.track,
        opening_nol=_opening_nol(fin, assumptions)[0] if cfg.nol.track else 0.0,
        nol_limit=cfg.nol.annual_limitation_pct,
        method=cfg.terminal.method,
        terminal_roic=cfg.terminal.terminal_roic,
        # The walk from enterprise value to equity is linear in the enterprise
        # value, so the whole bridge collapses to one constant and every draw
        # crosses it by the same route the scalar path does.
        equity_offset=equity_value_from_dcf_ev(0.0, fin, bridge),
    )


# -- the tornado ----------------------------------------------------------- #


def tornado(
    fin: Financials,
    bridge: EVBridge,
    wacc_result: "WACCResult",
    assumptions: Assumptions,
    peer_median_ev_ebitda: float | None = None,
) -> list[DriverSensitivity]:
    """One-at-a-time sensitivity, each driver at its 5th and 95th percentile.

    The band is the driver's own marginal, plus and minus 1.645 standard
    deviations, so a bar spans exactly the range the simulation draws that
    driver across. The other three sit at their central values throughout,
    which is what makes this univariate and therefore not a scenario: the
    combination at the end of the WACC bar is a company with a different
    discount rate and identical operations, and no such company exists. It is
    still the right ranking tool, because the question is which assumption is
    load-bearing, and the answer has to hold one thing at a time to mean
    anything.

    Bars come back sorted by swing, widest first. An endpoint that cannot be
    valued raises rather than returning a gap in the chart, because the reason
    it cannot be valued, terminal growth reaching the discount rate inside a
    one-standard-deviation band, is itself a statement that the assumed spreads
    do not hang together.

    ``peer_median_ev_ebitda`` is accepted so this can be called wherever
    ``run_dcf`` is, and is unused: the exit multiple is not one of the drawn
    drivers, so it has no bar.
    """
    engine = _prepare(fin, bridge, wacc_result, assumptions)
    sds = _standard_deviations(assumptions)
    centres = np.array(
        [
            engine.growth_start,
            engine.margin_terminal,
            engine.base_wacc,
            engine.base_g,
        ]
    )

    shocks = np.zeros((2 * len(DRIVERS), len(DRIVERS)))
    for k in range(len(DRIVERS)):
        shocks[2 * k, k] = -_Z95 * sds[k]
        shocks[2 * k + 1, k] = +_Z95 * sds[k]
    prices, _, alive = engine.values(shocks)

    bars: list[DriverSensitivity] = []
    for k, driver in enumerate(DRIVERS):
        low, high = prices[2 * k], prices[2 * k + 1]
        if not (alive[2 * k] and alive[2 * k + 1]):
            end = "5th" if not alive[2 * k] else "95th"
            raise ConfigError(
                f"the {end} percentile of {DRIVER_PATHS[driver]} cannot be valued: "
                "terminal growth reaches the discount rate, or the terminal value "
                "cannot be formed, inside the driver's own 90% band. The tornado "
                "would have a hole in it, and the hole is the finding. Reduce "
                f"simulation.{driver}_sd, or move the central case away from the "
                "boundary."
            )
        bars.append(
            DriverSensitivity(
                driver=driver,
                label=DRIVER_LABELS[driver],
                low_value=float(centres[k] - _Z95 * sds[k]),
                high_value=float(centres[k] + _Z95 * sds[k]),
                low_price=float(low),
                high_price=float(high),
                swing=float(abs(high - low)),
            )
        )
    bars.sort(key=lambda bar: bar.swing, reverse=True)
    return bars


# -- the run --------------------------------------------------------------- #


def run_simulation(
    fin: Financials,
    bridge: EVBridge,
    wacc_result: "WACCResult",
    assumptions: Assumptions,
    peer_median_ev_ebitda: float | None = None,
    correlations: Mapping[tuple[str, str], float] | None = None,
    independent: bool = False,
) -> SimulationResult:
    """Revalue the company across the joint distribution and report it.

    ``peer_median_ev_ebitda`` is accepted so the call site matches ``run_dcf``,
    and it reaches the reconciliation run only. The simulation values on the
    terminal method the assumptions select, Gordon or value driver, and never on
    an exit multiple, for the reason in the module docstring.

    ``simulation.enabled`` is the switch a report assembler reads to decide
    whether to run this at all. Calling this function is that decision, so the
    flag is not re-checked here.
    """
    cfg = assumptions.simulation
    notes: list[str] = []
    checks: list[str] = []

    engine = _prepare(fin, bridge, wacc_result, assumptions)
    central, reconciliation = _reconcile(
        engine, fin, bridge, wacc_result, assumptions, peer_median_ev_ebitda
    )
    checks.append(reconciliation)

    shocks = draw_shocks(assumptions, correlations, independent=independent)
    per_share, enterprise, alive = engine.values(shocks)
    kept = int(np.count_nonzero(alive))
    failed = cfg.draws - kept
    # The two rejection reasons are counted apart because they truncate opposite
    # sides of the distribution. A draw where g reaches the WACC is one the Gordon
    # formula cannot express, and it would have been an extreme HIGH value; a draw
    # whose steady state never turns a profit would have been an extreme LOW one.
    # A single headline count hides which tail was cut.
    undefined = int(
        np.count_nonzero(
            (engine.base_g + shocks[:, 3] >= engine.base_wacc + shocks[:, 2])
            | (engine.base_wacc + shocks[:, 2] <= 0.0)
        )
    )
    unformed = failed - undefined
    if kept == 0:
        raise ConfigError(
            f"every one of the {cfg.draws} draws was rejected: terminal growth "
            "reached the discount rate, or the terminal value could not be formed, "
            "in all of them. The central case sits on the boundary of the Gordon "
            "formula, so there is no distribution to report. Lower "
            "dcf.terminal_growth, raise the cost of capital, or narrow "
            "simulation.terminal_growth_sd and simulation.wacc_sd."
        )

    values = per_share[alive]
    evs = enterprise[alive]
    per_share_stats = Percentiles.of(values)
    ev_stats = Percentiles.of(evs)
    price = bridge.price
    prob_above = float(np.mean(values > price))
    prob_upside = float(np.mean(values > 1.2 * price))

    bars = tornado(fin, bridge, wacc_result, assumptions, peer_median_ev_ebitda)

    notes.extend(_notes(assumptions, engine, independent, correlations))
    checks.extend(
        _checks(
            assumptions=assumptions,
            per_share=per_share_stats,
            price=price,
            prob_above=prob_above,
            prob_upside=prob_upside,
            failed=failed,
            undefined=undefined,
            unformed=unformed,
            bars=bars,
            central=central,
        )
    )

    return SimulationResult(
        draws=cfg.draws,
        seed=cfg.seed,
        per_share=per_share_stats,
        enterprise_value=ev_stats,
        current_price=price,
        prob_above_price=prob_above,
        prob_upside_20pct=prob_upside,
        tornado=bars,
        failed_draws=failed,
        notes=notes,
        checks=checks,
        central_per_share=central,
    )


def _reconcile(
    engine: _Engine,
    fin: Financials,
    bridge: EVBridge,
    wacc_result: "WACCResult",
    assumptions: Assumptions,
    peer_median_ev_ebitda: float | None,
) -> tuple[float, str]:
    """Value the zero-shock draw both ways and refuse to proceed if they differ.

    The vectorised path exists for speed and is worth nothing if it is a
    different model. The central draw applies no shock at all, so it has to
    reproduce ``run_dcf`` exactly up to floating-point summation order. Where it
    does not, the two paths have diverged, most likely because a terminal
    construction was added to the scalar model and not to this one, and a
    distribution around a number the rest of the engine does not report is
    worse than no distribution.
    """
    scalar = run_dcf(fin, bridge, wacc_result, assumptions, peer_median_ev_ebitda)
    vector, _, alive = engine.values(np.zeros((1, len(DRIVERS))))
    if not alive[0]:
        raise ConfigError(
            "the central case itself cannot be valued by the simulation: terminal "
            "growth is at or above the discount rate, or the value-driver terminal "
            f"value cannot be formed at a steady state built on a "
            f"{engine.margin_terminal:.1%} terminal margin. run_dcf falls back to "
            "the Gordon value when the second one happens, and a distribution "
            "cannot take that fallback: half a sample valued one way and half the "
            "other is two histograms printed on top of each other. Set "
            "dcf.terminal.method to gordon, or raise dcf.ebit_margin_terminal until "
            "the steady state turns a profit."
        )
    vectorised = float(vector[0])
    reference = scalar.per_share
    gap = abs(vectorised - reference) / max(abs(reference), 1e-9)
    if gap > _RECONCILIATION_TOLERANCE:
        raise ConfigError(
            f"the vectorised valuation returns {vectorised:,.4f} a share on the "
            f"central draw against {reference:,.4f} from run_dcf, a gap of "
            f"{gap:.2%}. The two have diverged, so the distribution would be "
            "centred on a figure the rest of the engine does not report. The "
            f"simulation covers the '{engine.method}' terminal method; check that "
            "the scalar model still builds the same one."
        )
    return vectorised, (
        f"The vectorised path reproduces run_dcf on the central draw at "
        f"{vectorised:,.2f} a share against {reference:,.2f}, a relative gap of "
        f"{gap:.1e}. Every draw in the distribution travels that same code."
    )


def _notes(
    assumptions: Assumptions,
    engine: _Engine,
    independent: bool,
    correlations: Mapping[tuple[str, str], float] | None,
) -> list[str]:
    cfg = assumptions.simulation
    matrix = np.eye(len(DRIVERS)) if independent else correlation_matrix(correlations)
    out = [
        f"{cfg.draws:,} draws from a multivariate normal seeded at {cfg.seed}, with "
        f"standard deviations of {cfg.revenue_growth_sd:.1%} on year-one revenue "
        f"growth, {cfg.terminal_margin_sd:.1%} on the terminal EBIT margin, "
        f"{cfg.wacc_sd:.2%} on the discount rate and "
        f"{cfg.terminal_growth_sd:.2%} on terminal growth."
    ]
    if independent:
        out.append(
            "The drivers are drawn INDEPENDENTLY, which is the comparison case, "
            "not the base case. Independence assumes a company that grows faster "
            "is no more likely to reach a better margin and is discounted at the "
            "same rate, which is false of every business worth modelling."
        )
    else:
        pairs = ", ".join(
            f"{DRIVER_PROSE[a]} and {DRIVER_PROSE[b]} at "
            f"{matrix[DRIVERS.index(a), DRIVERS.index(b)]:+.2f}"
            for a, b in (
                ("revenue_growth", "terminal_margin"),
                ("revenue_growth", "wacc"),
                ("terminal_margin", "wacc"),
            )
        )
        out.append(
            f"The drivers are drawn jointly: {pairs}. Operating leverage ties "
            "growth to margin, and the market discounts a better business at a "
            "lower rate, so the favourable draws arrive together and so do the "
            "unfavourable ones. Independent draws would put mass on combinations "
            "that do not occur."
        )
    out.append(
        f"The probabilities are under this distribution, not about the world. "
        f"They are conditional on four standard deviations someone typed into the "
        f"assumptions file and on the correlation matrix in this module. A tight "
        f"distribution around a wrong central case is confidently wrong."
    )
    out.append(
        "The tornado holds three drivers still while moving the fourth, so it "
        "ignores the correlations the distribution is built on. That is why both "
        "are reported: the tornado says which assumption to argue about, the "
        "distribution says how wide the answer is once they move together."
    )
    out.append(
        "Only year-one revenue growth is shocked, so the growth path still fades "
        f"to an unshocked {engine.growth_terminal:.1%} in year {engine.n}. A "
        "parallel shift of the whole path would widen the answer materially; this "
        "is the narrower of the two and the one the assumptions file describes."
    )
    return out


def _checks(
    *,
    assumptions: Assumptions,
    per_share: Percentiles,
    price: float,
    prob_above: float,
    prob_upside: float,
    failed: int,
    undefined: int,
    unformed: int,
    bars: list[DriverSensitivity],
    central: float,
) -> list[str]:
    cfg = assumptions.simulation
    out: list[str] = []

    share = failed / cfg.draws
    if failed == 0:
        out.append(
            f"No draw was rejected: terminal growth stayed below the discount rate "
            f"in all {cfg.draws:,} of them."
        )
    elif share > _FAILED_DRAW_FLAG:
        out.append(
            f"FLAG: {failed:,} of {cfg.draws:,} draws were rejected, "
            f"{share:.1%} of the sample, above the {_FAILED_DRAW_FLAG:.0%} mark. "
            f"Terminal growth reached the discount rate in {undefined:,} of them, "
            "which means the assumed spreads on WACC and on g are not consistent "
            "with a central case sitting comfortably inside the Gordon formula. "
            "Those draws would have carried the largest values, so the surviving "
            "sample is truncated on the right and the reported percentiles are "
            "biased low. Narrow simulation.wacc_sd and simulation.terminal_growth_sd, "
            "or move the central case."
        )
    else:
        out.append(
            f"{failed:,} of {cfg.draws:,} draws were rejected, {share:.1%} of the "
            f"sample, {undefined:,} of them because terminal growth reached the "
            "discount rate. They are discarded rather than clipped to the boundary, "
            "which would pile mass on the point the perpetuity is largest and "
            "invent a right tail out of nothing."
        )
    if unformed:
        out.append(
            f"FLAG: a further {unformed:,} draws were rejected because the "
            "value-driver terminal value could not be formed: the steady state at "
            "the drawn terminal margin never turns a profit. Those are the draws "
            "that would have produced the LOWEST values, so the left tail is cut "
            "and the percentiles are biased high. A terminal margin whose 90% band "
            "reaches zero is the assumption to revisit."
        )

    out.append(
        f"Median value of {per_share.p50:,.2f} a share against a central case of "
        f"{central:,.2f}. The gap between them is the skew the Gordon denominator "
        "puts into the distribution, not a second valuation."
    )

    if per_share.p95 < price:
        out.append(
            f"FLAG: the 95th percentile of value, {per_share.p95:,.2f} a share, is "
            f"below the quoted price of {price:,.2f}. No combination inside the "
            "assumed spreads reaches the market price, so the disagreement with the "
            "market is in the central case and not in the uncertainty around it. "
            "Widening the standard deviations until the price falls inside the "
            "distribution would be fitting the spread to the answer."
        )
    elif per_share.p5 > price:
        out.append(
            f"FLAG: the 5th percentile of value, {per_share.p5:,.2f} a share, is "
            f"above the quoted price of {price:,.2f}. Every draw says the stock is "
            "cheap, which is a claim about the assumptions rather than about the "
            "company: check the terminal margin and the growth path before the "
            "conviction."
        )
    else:
        out.append(
            f"The quoted price of {price:,.2f} sits inside the distribution, "
            f"between the 5th percentile at {per_share.p5:,.2f} and the 95th at "
            f"{per_share.p95:,.2f}."
        )

    out.append(
        f"P(value > price) is {prob_above:.1%} and P(value > 1.2x price) is "
        f"{prob_upside:.1%}, under the assumed distribution."
    )

    widest = bars[0]
    narrowest = bars[-1]
    out.append(
        f"The tornado is led by {DRIVER_PROSE[widest.driver]}, which swings the "
        f"price by {widest.swing:,.2f} a share across its 90% band, against "
        f"{narrowest.swing:,.2f} for {DRIVER_PROSE[narrowest.driver]} at the foot "
        f"of it. That is the assumption to argue about, and "
        f"{DRIVER_PATHS[widest.driver]} is the line it comes from."
    )

    if per_share.sd > 0:
        out.append(
            f"The distribution has a standard deviation of {per_share.sd:,.2f} a "
            f"share on a median of {per_share.p50:,.2f}, a coefficient of variation "
            f"of {per_share.sd / abs(per_share.p50):.0%} where the median is not "
            "zero. A DCF is an estimate with that much dispersion in it before any "
            "argument about whether the central case is right."
        )
    return out
