"""Weighted average cost of capital.

    WACC = E/(D+E) * Ke + D/(D+E) * Kd * (1 - t)

Ke is CAPM: the ten-year Treasury yield, plus beta times the equity risk
premium, plus a size premium where one is configured. Weights are market on the
equity side and book on the debt side. That asymmetry is deliberate and standard:
a share price is observable every second, and the market value of a private term
loan or a revolver is not, so book value is the only honest stand-in.

Four things decide whether a technology WACC is credible, and each of them is a
place where the arithmetic keeps working long after the economics have stopped.

**The book yield on a zero-coupon convertible.** Dividing reported interest
expense by total debt gives a backward-looking book yield, not a market cost of
new borrowing. For a software issuer whose only debt is a 0% or 0.25% coupon
convertible, that ratio comes back at something like 0.5%, below the risk-free
rate, which says the market lends to this company more cheaply than it lends to
the Treasury. It does not. The coupon is low because the holder was paid in an
equity option, and the option does not show up in interest expense. So the
filings method carries a hard floor: a book yield below the risk-free rate is
rejected outright and the synthetic rating is used instead, with the substitution
stated in the output rather than buried.

**Beta and capital structure.** A raw regression beta reflects the operating risk
of the business and the leverage of the balance sheet, mixed together. Averaging
raw betas across a peer set therefore averages capital structures that have
nothing to do with each other. Hamada strips the leverage out of each peer at its
own debt-to-equity ratio and its own tax rate, the median of those unlevered
betas is taken, and the result is relevered at the target's ratio. The median
rather than the mean, because one peer with a broken regression, a recent IPO or
a takeover rumour in the window should not be able to move the answer.

**Blume.** Betas mean-revert toward one. A firm's beta this decade is a poor
predictor of its beta next decade, and the estimate itself carries sampling error
that pulls extreme readings further from the truth than moderate ones. The
Bloomberg convention shrinks the estimate two-thirds of the way from 1.0 toward
the regression result. It is applied after the OLS and before unlevering, so the
leverage adjustment operates on the beta actually used in the cost of equity.

**Vasicek.** The median and Blume share a blind spot: neither looks at how well
each beta was measured. The median treats a slope estimated on an R-squared of
0.05 exactly like one estimated on 0.40, and Blume shrinks every slope by the
same 33% whether it came from a tight regression or a scatter. Vasicek (1973) is
the Bayesian answer, and it lets the data decide the shrinkage: each peer's asset
beta is pulled toward the cross-sectional mean of the peer set by a weight that
is the ratio of the dispersion across peers to that dispersion plus the peer's
own sampling variance. A precisely measured beta keeps nearly all of its own
value, a noisy one is pulled hard toward the group, and the pooled figure is the
precision-weighted mean of what comes out. That buys information the median
throws away, at the cost of the median's robustness: one broken regression, a
peer in a takeover or three weeks of a short squeeze, can move a mean and cannot
move a median. Which is why the median remains the default and Vasicek is opt-in
through market.peer_beta_method.

**Net cash.** A company holding more cash than debt still has a debt weight of
D/(D+E) using gross debt, not net. The interest tax shield attaches to the debt
outstanding, not to a net position, and netting cash against debt in the weights
would credit the firm with a shield it does not receive. The practical
consequence is that a net-cash software company has a debt weight near zero and a
WACC that is its cost of equity to within a few basis points. The output says so,
because a WACC page that shows a 3% after-tax cost of debt invites the reader to
think it mattered.

Rates are decimals, money is USD millions, and every assumption is read from the
Assumptions object.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from .config import Assumptions
from .errors import ConfigError, MissingDataError, NotMeaningfulError
from .ev_bridge import EVBridge
from .financials import Financials
from .market import MarketData, PriceSeries

# Two years of weekly data is 104 observations. Thirty is the point below which
# the standard error on the slope is wide enough that the point estimate is not
# worth quoting, and it is roughly seven months of trading.
MIN_OBSERVATIONS = 30

# The Bloomberg weight on the regression slope, with 1 - this on the market beta
# of 1.0. Named because the Vasicek path has to rescale the standard error by the
# same number when the two adjustments are combined.
_BLUME_SLOPE = 0.67

# Vasicek needs a cross-sectional variance, and a variance of two points is a
# statement about two points. With fewer names than this the prior is the peer
# set's own noise, so the engine falls back to the median rather than pretending
# to have measured a dispersion.
MIN_PEERS_FOR_SHRINKAGE = 3

# Damodaran's synthetic rating table for large-cap issuers, keyed on the floor of
# each interest coverage band, with the spread over the risk-free rate that goes
# with it. This is a published mapping rather than a judgment call, which is why
# it lives here as a constant and not in the assumptions file. Anyone who wants a
# different credit view sets cost_of_debt.method to 'override'.
_SYNTHETIC_RATING_TABLE: tuple[tuple[float, str, float], ...] = (
    (8.50, "Aaa/AAA", 0.0069),
    (6.50, "Aa2/AA", 0.0085),
    (5.50, "A1/A+", 0.0107),
    (4.25, "A2/A", 0.0118),
    (3.00, "A3/A-", 0.0133),
    (2.50, "Baa2/BBB", 0.0183),
    (2.25, "Ba1/BB+", 0.0250),
    (2.00, "Ba2/BB", 0.0315),
    (1.75, "B1/B+", 0.0403),
    (1.50, "B2/B", 0.0489),
    (1.25, "B3/B-", 0.0644),
    (0.80, "Caa/CCC", 0.0946),
    (0.65, "Ca2/CC", 0.1083),
    (0.20, "C2/C", 0.1390),
    (float("-inf"), "D2/D", 0.1860),
)

# Where coverage cannot be formed at all, the table is entered at its lowest
# investment grade rather than at its bottom. The alternative, treating an
# undefined ratio as if it were a coverage of zero, prices a cash-rich
# unprofitable software company as a defaulted credit at an 18% spread. That is a
# worse error than the one it avoids, and it is silent.
_UNDEFINED_COVERAGE_RATING = "Baa2/BBB"

# What each buildup row measures, so the CLI can format without guessing whether
# 0.045 is a rate, a weight or a beta. Formatting itself stays in cli.py.
_INPUT_KIND: dict[str, str] = {
    "Risk-free rate": "rate",
    "Equity risk premium": "rate",
    "Size premium": "rate",
    "Levered beta": "beta",
    "Unlevered beta": "beta",
    "Regression R-squared": "ratio",
    "Regression observations": "count",
    "Pre-tax cost of debt": "rate",
    "Tax rate": "rate",
    "After-tax cost of debt": "rate",
    "Market value of equity": "usd_mm",
    "Total debt (book)": "usd_mm",
    "Debt / equity": "ratio",
    "Weight of equity": "ratio",
    "Weight of debt": "ratio",
    "Cost of equity": "rate",
    "WACC": "rate",
}


@dataclass
class BetaEstimate:
    """One regression, with enough diagnostics to judge whether to believe it.

    ``raw_beta`` is the OLS slope. ``adjusted_beta`` is what the cost of equity
    actually uses, which differs from raw only under the Blume convention.
    ``unlevered_beta`` is the adjusted beta stripped of this issuer's own
    leverage, and is the figure that may legitimately be averaged across a peer
    set.

    ``shrunk_beta`` and ``shrink_weight`` are filled in only where this estimate
    was pooled under Vasicek, and record what the shrinkage did to it: the asset
    beta after it was pulled toward the peer mean, and the weight it kept on its
    own regression. Both stay None under the median, which shrinks nothing, so a
    populated pair is proof that shrinkage was applied rather than requested.
    """

    ticker: str
    raw_beta: float
    adjusted_beta: float
    r_squared: float
    n_observations: int
    stderr: float
    unlevered_beta: float
    debt_to_equity: float
    tax_rate: float
    note: str
    adjustment: str = "raw"
    shrunk_beta: float | None = None
    shrink_weight: float | None = None

    def as_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "raw_beta": self.raw_beta,
            "adjusted_beta": self.adjusted_beta,
            "unlevered_beta": self.unlevered_beta,
            "shrunk_beta": self.shrunk_beta,
            "shrink_weight": self.shrink_weight,
            "debt_to_equity": self.debt_to_equity,
            "tax_rate": self.tax_rate,
            "r_squared": self.r_squared,
            "stderr": self.stderr,
            "n": self.n_observations,
            "note": self.note,
        }


@dataclass
class WACCResult:
    """A discount rate with its whole derivation attached."""

    cost_of_equity: float
    after_tax_cost_of_debt: float
    pretax_cost_of_debt: float
    weight_equity: float
    weight_debt: float
    wacc: float
    risk_free_rate: float
    erp: float
    levered_beta: float
    unlevered_beta: float
    tax_rate: float
    credit_rating: str | None
    inputs: list[tuple[str, float, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def buildup_rows(self) -> list[dict]:
        """Every input with the source it came from, then the two results.

        A discount rate is an assumption stack, not a measurement. Printing the
        stack is the only way a reader can tell which of the two hundred basis
        points between one analyst's WACC and another's came from the beta and
        which came from the credit view.
        """
        rows = [
            {
                "label": label,
                "value": value,
                "source": source,
                "kind": _INPUT_KIND.get(label, "number"),
            }
            for label, value, source in self.inputs
        ]
        rows.append(
            {
                "label": "Cost of equity",
                "value": self.cost_of_equity,
                "source": "CAPM: risk-free + beta x ERP + size premium",
                "kind": "rate",
            }
        )
        rows.append(
            {
                "label": "WACC",
                "value": self.wacc,
                "source": (
                    f"{self.weight_equity:.1%} x {self.cost_of_equity:.2%} + "
                    f"{self.weight_debt:.1%} x {self.after_tax_cost_of_debt:.2%}"
                ),
                "kind": "rate",
            }
        )
        return rows


# -- Hamada ---------------------------------------------------------------- #


def _leverage_factor(debt_to_equity: float, tax_rate: float) -> float:
    """The 1 + (1-t)D/E term shared by both directions of the Hamada relation."""
    factor = 1.0 + (1.0 - tax_rate) * debt_to_equity
    if factor <= 0.0:
        raise ConfigError(
            f"Hamada leverage factor of {factor:,.4f} is not positive at a D/E of "
            f"{debt_to_equity:,.4f} and a tax rate of {tax_rate:.2%}. Unlevering "
            "against it would flip the sign of beta."
        )
    return factor


def unlever(beta_l: float, debt_to_equity: float, tax_rate: float) -> float:
    """Strip financial leverage out of an observed beta.

        beta_u = beta_l / (1 + (1-t) * D/E)

    D/E is market equity against book debt, matching the WACC weights. The
    tax term is there because the debt tax shield is itself a claim whose risk
    tracks the debt, so leverage raises equity risk by less than gross D/E.
    """
    return beta_l / _leverage_factor(debt_to_equity, tax_rate)


def relever(beta_u: float, debt_to_equity: float, tax_rate: float) -> float:
    """Reapply leverage to an asset beta.

        beta_l = beta_u * (1 + (1-t) * D/E)

    The exact algebraic inverse of ``unlever``, computed from the identical
    leverage factor, so unlevering at one D/E and relevering at the same D/E
    returns the input.
    """
    return beta_u * _leverage_factor(debt_to_equity, tax_rate)


# -- beta ------------------------------------------------------------------ #


def _aligned_weekly_returns(
    stock: PriceSeries, market: PriceSeries, lookback_years: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Paired weekly returns on the dates both series actually priced.

    Regressing two return vectors by position rather than by date is the quiet
    killer here. A peer that listed eighteen months ago has fewer weeks than the
    index, and zipping them together pairs its first week against the index's
    first week two years earlier. The slope that comes back is arithmetic
    performed on unrelated numbers, and it looks entirely plausible on screen.
    """
    if lookback_years is not None:
        anchor = min(stock.last_date, market.last_date)
        cut = anchor - timedelta(days=round(365.25 * lookback_years))
        stock = stock.window(cut)
        market = market.window(cut)

    s_dates, s_ret = stock.weekly_returns()
    m_dates, m_ret = market.weekly_returns()
    by_date = dict(zip(m_dates, m_ret))

    xs: list[float] = []
    ys: list[float] = []
    for d, r in zip(s_dates, s_ret):
        if d in by_date:
            xs.append(float(by_date[d]))
            ys.append(float(r))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def estimate_beta(
    stock: PriceSeries,
    market: PriceSeries,
    *,
    adjustment: str = "raw",
    debt_to_equity: float = 0.0,
    tax_rate: float = 0.0,
    lookback_years: float | None = None,
    min_observations: int = MIN_OBSERVATIONS,
) -> BetaEstimate:
    """OLS of weekly stock returns on weekly market returns.

    Returns the slope, the standard error of the slope and R-squared. The last
    two are not decoration: a software beta of 1.4 on an R-squared of 0.06 is a
    number the market did not actually tell you, and the analyst reading the
    output needs to see that before deciding whether to use a peer beta instead.

    ``debt_to_equity`` and ``tax_rate`` belong to this issuer and are used only
    to unlever. Left at their defaults the unlevered beta equals the levered one,
    which is correct for a debt-free filer and honest for a caller that has not
    supplied a capital structure.
    """
    if adjustment not in ("raw", "blume"):
        raise ConfigError(
            f"unknown beta adjustment {adjustment!r}; expected 'raw' or 'blume'"
        )

    x, y = _aligned_weekly_returns(stock, market, lookback_years)
    n = int(x.size)
    if n < min_observations:
        raise MissingDataError(
            "beta regression",
            ticker=stock.symbol,
            hint=(
                f"only {n} weekly returns are common to {stock.symbol} and "
                f"{market.symbol}; {min_observations} is the minimum this engine "
                "will regress on. A recent listing or a long trading halt is the "
                "usual cause. Use a peer beta instead."
            ),
        )

    x_bar = float(x.mean())
    y_bar = float(y.mean())
    dx = x - x_bar
    dy = y - y_bar
    sxx = float(dx @ dx)
    if sxx <= 0.0:
        raise MissingDataError(
            "beta regression",
            ticker=stock.symbol,
            hint=f"{market.symbol} shows no return variation over the window",
        )

    raw_beta = float(dx @ dy) / sxx
    alpha = y_bar - raw_beta * x_bar
    resid = y - (alpha + raw_beta * x)
    sse = float(resid @ resid)
    syy = float(dy @ dy)

    # n-2 degrees of freedom: the slope and the intercept were both estimated.
    stderr = math.sqrt(sse / (n - 2) / sxx)
    r_squared = 0.0 if syy <= 0.0 else 1.0 - sse / syy

    if adjustment == "blume":
        adjusted = _BLUME_SLOPE * raw_beta + (1.0 - _BLUME_SLOPE) * 1.0
        note = f"Blume-adjusted from a raw slope of {raw_beta:,.2f}"
    else:
        adjusted = raw_beta
        note = "raw OLS slope, unadjusted"

    unlevered = unlever(adjusted, debt_to_equity, tax_rate)
    note += (
        f"; {n} weekly returns vs {market.symbol}, R2 {r_squared:,.2f}, "
        f"s.e. {stderr:,.2f}; unlevered at {debt_to_equity:,.2f}x D/E and a "
        f"{tax_rate:.1%} tax rate"
    )

    return BetaEstimate(
        ticker=stock.symbol,
        raw_beta=raw_beta,
        adjusted_beta=adjusted,
        r_squared=r_squared,
        n_observations=n,
        stderr=stderr,
        unlevered_beta=unlevered,
        debt_to_equity=debt_to_equity,
        tax_rate=tax_rate,
        note=note,
        adjustment=adjustment,
    )


def _median_unlevered(estimates: Sequence[BetaEstimate]) -> float:
    if not estimates:
        raise MissingDataError(
            "peer unlevered beta",
            hint="no peer beta estimates were supplied to take a median of",
        )
    return float(np.median([e.unlevered_beta for e in estimates]))


# -- Vasicek shrinkage ------------------------------------------------------ #


def vasicek_adjust(
    betas: Sequence[float], stderrs: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Shrink each beta toward the cross-sectional mean by its own precision.

        w_i     = sigma_cross^2 / (sigma_cross^2 + se_i^2)
        beta_i* = w_i * beta_i + (1 - w_i) * beta_cross_mean

    Vasicek (1973). The peer set is treated as a prior: betas are drawn from a
    distribution centred on ``beta_cross_mean`` with variance ``sigma_cross^2``,
    and each regression is a noisy reading of one draw with variance ``se_i^2``.
    The posterior mean is the precision-weighted blend above. Read the weight
    directly: a peer whose standard error is small against the spread of the
    group keeps its own number, and a peer whose standard error is as wide as the
    spread of the group is told that half of what it thinks it knows is noise.

    This is what Blume does not do. Blume moves every estimate 33% toward 1.0
    whether it was measured on an R-squared of 0.05 or 0.40, so the beta the
    market told you least about is corrected by exactly as much as the one it
    told you most about.

    ``sigma_cross^2`` is the sample variance of the betas passed in, on n-1
    degrees of freedom, because the mean was estimated from the same points. The
    betas and the standard errors must be in the same units: if the betas are
    unlevered then the standard errors must be unlevered too, or the weights are
    a ratio of two different things.

    Raises NotMeaningfulError in the three degenerate cases, so the caller can
    fall back to the median and say it did: fewer than three peers, a peer set
    with no dispersion at all, and a standard error that is zero, negative or not
    finite. None of the three is a division that should be attempted.
    """
    b = np.asarray(list(betas), dtype=float)
    s = np.asarray(list(stderrs), dtype=float)
    if b.size != s.size:
        raise ConfigError(
            f"vasicek_adjust got {b.size} betas and {s.size} standard errors; each "
            "beta must carry the standard error of its own regression"
        )
    if b.size < MIN_PEERS_FOR_SHRINKAGE:
        raise NotMeaningfulError(
            f"Vasicek shrinkage needs at least {MIN_PEERS_FOR_SHRINKAGE} peers to "
            f"measure a cross-sectional variance and {b.size} were supplied"
        )
    if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise NotMeaningfulError(
            "Vasicek shrinkage needs a positive standard error on every peer "
            "regression, and at least one is zero, negative or missing"
        )

    mean = float(b.mean())
    sigma2 = float(b.var(ddof=1))
    if sigma2 <= 0.0:
        raise NotMeaningfulError(
            "the peer asset betas have no cross-sectional variance, so the prior "
            "is a point and shrinkage toward it has no weight to compute"
        )

    weights = [sigma2 / (sigma2 + float(se) ** 2) for se in s]
    adjusted = [w * float(beta) + (1.0 - w) * mean for w, beta in zip(weights, b)]
    return adjusted, weights


def _unlevered_stderr(est: BetaEstimate) -> float:
    """Standard error of the peer's asset beta, not of its levered slope.

    This is the subtle part of applying Vasicek at the unlevered level. Both
    steps between the OLS slope and the number being pooled are multiplications
    by constants, not new estimates: Blume multiplies the slope by 0.67 and adds
    a constant, and unlevering divides by 1 + (1-t)D/E, which comes off the
    balance sheet. So the standard error rides through both on exactly the same
    scale: se divided by the leverage factor, and by a further 0.67 where the
    slope was Blume-adjusted on the way.

    Skip the division and the weights are computed on mismatched units: a levered
    peer's asset beta would be paired with the standard error of its equity beta,
    which is larger by the leverage factor. Its weight would come out too low and
    it would be dragged toward the peer mean for no reason except that it carries
    debt. The shrinkage would then be reading capital structure, which is the one
    thing unlevering exists to remove.
    """
    scale = _BLUME_SLOPE if est.adjustment == "blume" else 1.0
    return est.stderr * scale / _leverage_factor(est.debt_to_equity, est.tax_rate)


def pool_unlevered_betas(
    estimates: Sequence[BetaEstimate], method: str = "median"
) -> tuple[float, str, list[str]]:
    """Pool peer asset betas into the one figure the target is relevered at.

    Returns the pooled asset beta, the method that was actually used, and the
    notes a reader needs. The method returned is not always the method asked
    for: a Vasicek request over a degenerate peer set falls back to the median
    and says so in the notes rather than dividing by zero.

    Under 'median' the answer is the middle asset beta, which is the sell-side
    default because one peer with a broken regression cannot move it. Under
    'vasicek' each asset beta is first shrunk toward the cross-sectional mean by
    its own precision, then pooled as a precision-weighted mean with weights
    1/se^2 on the unlevered standard errors.

    The mean is the deliberate part. Having spent the effort to work out which
    estimates are well measured, taking a median of them throws that ordering
    away again: the median reads only the middle of the sorted list and cannot
    tell whether the estimate sitting there was the tightest regression in the
    set or the loosest. The mean uses all of it, and gives up the robustness in
    exchange. With a peer set that contains one genuinely broken regression, a
    peer in a takeover or one that listed nine weeks ago, the median is still the
    safer pool, because shrinkage narrows a bad estimate's influence without ever
    removing it. That trade is why 'median' is the default.

    Precision here is sampling precision, 1/se^2, rather than the posterior
    precision that would add 1/sigma_cross^2 to every peer. The posterior version
    compresses the weights toward equal and is harder to defend at a desk: the
    number quoted is the one that comes off each regression's own output.

    The estimates are updated in place with the shrunk beta and the weight, so
    a caller holding them can show what the prior did to each peer. Pooling under
    the median clears both fields, because nothing was shrunk.
    """
    if method not in ("median", "vasicek"):
        raise ConfigError(
            f"unknown peer beta method {method!r}; expected 'median' or 'vasicek'"
        )
    median = _median_unlevered(estimates)
    for e in estimates:
        e.shrunk_beta = None
        e.shrink_weight = None
    if method == "median":
        return median, "median", []

    betas = [e.unlevered_beta for e in estimates]
    stderrs = [_unlevered_stderr(e) for e in estimates]
    try:
        shrunk, weights = vasicek_adjust(betas, stderrs)
    except NotMeaningfulError as exc:
        return (
            median,
            "median",
            [
                "market.peer_beta_method is 'vasicek', but it could not be applied: "
                f"{exc}. The median asset beta of {median:,.3f} was used instead, "
                "and the peer betas reaching the cost of equity are unshrunk."
            ],
        )

    precision = [1.0 / se**2 for se in stderrs]
    pooled = sum(p * b for p, b in zip(precision, shrunk)) / sum(precision)

    for e, b, w in zip(estimates, shrunk, weights):
        e.shrunk_beta = b
        e.shrink_weight = w

    mean = float(np.mean(betas))
    sd = float(np.std(betas, ddof=1))
    notes = [
        f"Vasicek shrinkage pooled {len(estimates)} peer asset betas toward a "
        f"cross-sectional mean of {mean:,.3f} with a standard deviation of "
        f"{sd:,.3f}. The weight each peer kept on its own regression runs "
        f"{min(weights):.2f} to {max(weights):.2f}, the remainder going to that "
        f"mean. The precision-weighted pool is {pooled:,.3f}, against a median of "
        f"{median:,.3f} on the same unshrunk betas."
    ]
    if any(e.adjustment == "blume" for e in estimates):
        notes.append(
            "Peer regressions arrived here Blume-adjusted, so those estimates are "
            "shrunk twice: two thirds of the way toward 1.0 by a fixed rule, then "
            "toward the peer mean by its own precision. The two corrections are "
            "answers to the same question and stacking them pulls dispersion out of "
            "the peer set beyond what either intends. Set market.beta_adjustment to "
            "'raw' to let the data do the shrinking on its own."
        )
    return pooled, "vasicek", notes


def peer_unlevered_beta(
    peers: Sequence[tuple[PriceSeries, float, float]],
    market: PriceSeries,
    *,
    adjustment: str = "raw",
    method: str = "median",
    lookback_years: float | None = None,
    min_observations: int = MIN_OBSERVATIONS,
) -> tuple[float, list[BetaEstimate], list[str]]:
    """Pooled unlevered beta of a peer set, the estimates, and the pooling notes.

    Each peer is ``(prices, debt_to_equity, tax_rate)``, with the ratio and the
    rate belonging to that peer, not to the target. Unlevering happens before
    pooling, which is the entire point: it removes the capital structure
    differences that make raw betas incomparable, and leaves an asset beta that
    describes the business the peers have in common. The caller relevers the
    pooled figure at the target's own D/E.

    ``method`` is 'median' or 'vasicek'; see ``pool_unlevered_betas``. The notes
    come back rather than being discarded because a Vasicek request can end in a
    median, and a pooled beta that does not say how it was pooled is not
    auditable.

    A peer that cannot be regressed, typically a recent listing with too few
    weeks, is dropped rather than allowed to fail the set. The returned list is
    the peers that survived, so a caller can compare it against what it passed.
    """
    estimates: list[BetaEstimate] = []
    skipped: list[str] = []
    for prices, de, tax in peers:
        try:
            estimates.append(
                estimate_beta(
                    prices,
                    market,
                    adjustment=adjustment,
                    debt_to_equity=de,
                    tax_rate=tax,
                    lookback_years=lookback_years,
                    min_observations=min_observations,
                )
            )
        except MissingDataError as exc:
            skipped.append(f"{prices.symbol} ({exc.hint or 'no usable regression'})")

    if not estimates:
        raise MissingDataError(
            "peer unlevered beta",
            hint="no peer produced a usable regression: " + "; ".join(skipped),
        )
    pooled, _, notes = pool_unlevered_betas(estimates, method)
    return pooled, estimates, notes


# -- cost of debt ---------------------------------------------------------- #


def synthetic_credit_spread(interest_coverage: float) -> tuple[str, float]:
    """Map interest coverage to a rating and a spread over the risk-free rate.

    Damodaran's large-cap table. Pass ``float('inf')`` for a profitable filer
    with no debt and no interest expense, and ``float('nan')`` where coverage
    cannot be formed at all, which returns the lowest investment grade rather
    than the bottom of the table.
    """
    if math.isnan(interest_coverage):
        for _, rating, spread in _SYNTHETIC_RATING_TABLE:
            if rating == _UNDEFINED_COVERAGE_RATING:
                return rating, spread

    for floor_, rating, spread in _SYNTHETIC_RATING_TABLE:
        if interest_coverage >= floor_:
            return rating, spread

    _, rating, spread = _SYNTHETIC_RATING_TABLE[-1]
    return rating, spread


def _synthetic_cost_of_debt(
    fin: Financials, risk_free_rate: float
) -> tuple[float, str, str, str | None]:
    """Pre-tax cost of debt from a synthetic rating.

    Returns ``(pretax_rate, rating, source, note)``. The note is populated only
    where coverage could not be formed from the filings, because that is the case
    a reader has to be told about.
    """
    interest = fin.interest_expense
    note: str | None = None

    if fin.ebit <= 0:
        coverage = float("nan")
        note = (
            f"TTM EBIT of {fin.ebit:,.0f}mm is negative, so interest coverage carries "
            "no signal about default risk for a company spending its operating margin "
            f"on growth. The rating is pinned at {_UNDEFINED_COVERAGE_RATING}, the "
            "lowest investment grade. Set cost_of_debt.override_rate if the issuer "
            "carries an agency rating or has priced debt recently."
        )
    elif not interest or interest <= 0:
        if fin.total_debt_ex_leases <= 0 and fin.finance_lease_liability <= 0:
            coverage = float("inf")
            note = (
                "The filer reports no debt and no interest expense, so coverage is "
                "unbounded and the rating sits at the top of the table. It carries no "
                "weight in the WACC: the debt weight is zero."
            )
        else:
            coverage = float("nan")
            note = (
                f"The filer reports {fin.total_debt_ex_leases:,.0f}mm of debt but no "
                f"interest expense, so coverage cannot be formed and the rating is "
                f"pinned at {_UNDEFINED_COVERAGE_RATING}. Interest may be tagged under "
                "a concept outside the ladder in tags.py, or capitalised into an asset."
            )
    else:
        coverage = fin.ebit / interest

    rating, spread = synthetic_credit_spread(coverage)
    if math.isfinite(coverage):
        source = (
            f"synthetic: EBIT/interest of {coverage:,.1f}x maps to {rating}, "
            f"{spread:.2%} over the risk-free rate"
        )
    else:
        source = (
            f"synthetic: interest coverage not meaningful, pinned at {rating}, "
            f"{spread:.2%} over the risk-free rate"
        )
    return risk_free_rate + spread, rating, source, note


# -- WACC ------------------------------------------------------------------ #


def compute_wacc(
    fin: Financials,
    bridge: EVBridge,
    market_data: MarketData,
    assumptions: Assumptions,
    peer_betas: Sequence[BetaEstimate] | None = None,
) -> WACCResult:
    """Assemble a discount rate from filings, market data and the assumptions.

    ``peer_betas`` are estimates already produced by ``peer_unlevered_beta``.
    When supplied, their pooled unlevered beta is relevered at the target's own
    capital structure and the target's own regression is not run at all, which is
    the right call whenever the target's R-squared is poor or its listing history
    is short. The pooling is the median by default, or the Vasicek shrunk mean
    where market.peer_beta_method asks for it, and the notes say which was used.
    """
    notes: list[str] = []
    inputs: list[tuple[str, float, str]] = []
    mkt_cfg = assumptions.market

    if mkt_cfg.beta_frequency != "weekly":
        raise ConfigError(
            f"market.beta_frequency is {mkt_cfg.beta_frequency!r}, but the price "
            "layer publishes weekly returns only. Regressing weekly data and "
            "labelling it monthly would misstate the estimate's precision."
        )

    # -- tax rate ---------------------------------------------------------- #
    if assumptions.tax.use_effective_rate and fin.effective_tax_rate is not None:
        tax_rate = fin.effective_tax_rate
        tax_source = f"filed effective rate, TTM to {fin.as_of}"
    else:
        tax_rate = assumptions.tax.marginal_tax_rate
        tax_source = "assumption: tax.marginal_tax_rate"
        if assumptions.tax.use_effective_rate:
            notes.append(
                "The effective tax rate was requested but the filed rate is not "
                "economically meaningful, so the marginal rate is used for the debt "
                "shield. See Financials.effective_tax_rate for the test applied."
            )

    # -- risk free and premia ---------------------------------------------- #
    risk_free_rate, rf_source = market_data.risk_free_rate(mkt_cfg.risk_free_rate)
    erp = mkt_cfg.equity_risk_premium
    size_premium = mkt_cfg.size_premium

    # -- capital structure -------------------------------------------------- #
    equity = bridge.equity_value
    debt = bridge.total_debt
    if equity <= 0:
        raise NotMeaningfulError(
            f"{fin.ticker} has a market equity value of {equity:,.0f}mm, so market "
            "weights cannot be formed and neither can a D/E for unlevering."
        )
    capital = equity + debt
    weight_equity = equity / capital
    weight_debt = debt / capital
    debt_to_equity = debt / equity

    # -- beta --------------------------------------------------------------- #
    regression: BetaEstimate | None = None
    if peer_betas:
        unlevered_beta, pool_method, pool_notes = pool_unlevered_betas(
            peer_betas, mkt_cfg.peer_beta_method
        )
        levered_beta = relever(unlevered_beta, debt_to_equity, tax_rate)
        tickers = ", ".join(e.ticker for e in peer_betas)
        pool_label = (
            "Vasicek-shrunk precision-weighted mean"
            if pool_method == "vasicek"
            else "median"
        )
        beta_source = (
            f"{pool_label} of {len(peer_betas)} peer unlevered betas ({tickers}), "
            f"relevered at {debt_to_equity:,.2f}x D/E and a {tax_rate:.1%} tax rate"
        )
        # The regression quality has to reach the reader on this path too, or
        # the pooling laundered the uncertainty out of sight. A median of six
        # slopes whose R-squareds run 0.05 to 0.30 is a judgment, not a datum,
        # and a shrunk mean of the same six is a different judgment.
        r2s = sorted(e.r_squared for e in peer_betas)
        ses = sorted(e.stderr for e in peer_betas)
        pooling_sentence = (
            "Vasicek shrinkage pools these estimates by weighting each against how "
            "well it was measured; it does not sharpen any one of them."
            if pool_method == "vasicek"
            else "The median asset beta pools these estimates; it does not sharpen "
            "any one of them."
        )
        notes.append(
            f"Peer regressions: R-squared runs {r2s[0]:.2f} to {r2s[-1]:.2f} and "
            f"the slope standard error {ses[0]:.2f} to {ses[-1]:.2f} across "
            f"{len(peer_betas)} names, each on at least "
            f"{min(e.n_observations for e in peer_betas)} weekly observations. "
            + pooling_sentence
        )
        notes.extend(pool_notes)
        unlevered_source = (
            f"Hamada on each peer at its own D/E and tax rate, then the "
            f"{pool_label}. {mkt_cfg.beta_adjustment} adjustment applied to each "
            "peer regression"
        )
    else:
        regression = estimate_beta(
            market_data.prices(fin.ticker),
            market_data.prices(mkt_cfg.market_index),
            adjustment=mkt_cfg.beta_adjustment,
            debt_to_equity=debt_to_equity,
            tax_rate=tax_rate,
            lookback_years=mkt_cfg.beta_lookback_years,
        )
        levered_beta = regression.adjusted_beta
        unlevered_beta = regression.unlevered_beta
        beta_source = (
            f"OLS of {regression.n_observations} weekly returns on "
            f"{mkt_cfg.market_index} over {mkt_cfg.beta_lookback_years:g}y, "
            f"s.e. {regression.stderr:,.2f}, {mkt_cfg.beta_adjustment} adjustment"
        )
        unlevered_source = (
            f"Hamada at {debt_to_equity:,.2f}x D/E and a {tax_rate:.1%} tax rate"
        )
        if regression.r_squared < 0.10:
            notes.append(
                f"The beta regression explains {regression.r_squared:.0%} of "
                f"{fin.ticker}'s weekly variance, with a standard error of "
                f"{regression.stderr:,.2f} on a slope of {levered_beta:,.2f}. The "
                "market is barely driving this stock over the window, so the point "
                "estimate is weak and a peer beta is the better anchor."
            )

    cost_of_equity = risk_free_rate + levered_beta * erp + size_premium

    # -- cost of debt -------------------------------------------------------- #
    method = assumptions.cost_of_debt.method
    credit_rating: str | None = None

    if method == "override":
        override = assumptions.cost_of_debt.override_rate
        if override is None:
            raise ConfigError(
                "cost_of_debt.method is 'override' but cost_of_debt.override_rate "
                "is unset. The engine will not invent a borrowing cost."
            )
        pretax_cost_of_debt = override
        cod_source = "assumption: cost_of_debt.override_rate"
    elif method == "filings":
        # Interest expense is generated by every borrowing on the balance sheet,
        # whichever side of the EV bridge each one lands on, so the denominator
        # here is all interest-bearing debt rather than the bridge's debt.
        interest_bearing = (
            fin.straight_debt + fin.convertible_debt + fin.finance_lease_liability
        )
        interest = fin.interest_expense
        book_yield = (
            interest / interest_bearing
            if interest and interest_bearing > 0
            else None
        )
        if book_yield is None:
            pretax_cost_of_debt, credit_rating, cod_source, syn_note = (
                _synthetic_cost_of_debt(fin, risk_free_rate)
            )
            notes.append(
                "cost_of_debt.method is 'filings', but the filer reports no interest "
                "expense or no interest-bearing debt, so a book yield cannot be "
                "formed. The synthetic rating was used instead."
            )
            if syn_note:
                notes.append(syn_note)
        elif book_yield < risk_free_rate:
            pretax_cost_of_debt, credit_rating, cod_source, syn_note = (
                _synthetic_cost_of_debt(fin, risk_free_rate)
            )
            notes.append(
                f"The book yield from the filings is {book_yield:.2%}, "
                f"{interest:,.0f}mm of interest expense over {interest_bearing:,.0f}mm "
                f"of debt, which is below the {risk_free_rate:.2%} risk-free rate. No "
                "corporate borrows below the Treasury. This is the low-coupon "
                "convertible signature: the holder was paid in an equity option that "
                "never enters interest expense. The synthetic rating was used instead."
            )
            if syn_note:
                notes.append(syn_note)
        else:
            pretax_cost_of_debt = book_yield
            cod_source = (
                f"filings: {interest:,.0f}mm interest expense over "
                f"{interest_bearing:,.0f}mm of period-end interest-bearing debt"
            )
            notes.append(
                "The cost of debt is a book yield on period-end debt. Only one "
                "balance-sheet instant is normalised, so no average balance is "
                "available; for a filer that borrowed during the year the yield is "
                "understated, and in every case it is the coupon on debt already "
                "issued rather than the cost of the next dollar borrowed."
            )
    else:
        pretax_cost_of_debt, credit_rating, cod_source, syn_note = (
            _synthetic_cost_of_debt(fin, risk_free_rate)
        )
        if syn_note:
            notes.append(syn_note)

    after_tax_cost_of_debt = pretax_cost_of_debt * (1.0 - tax_rate)
    wacc = weight_equity * cost_of_equity + weight_debt * after_tax_cost_of_debt

    if wacc <= 0:
        raise NotMeaningfulError(
            f"WACC computed to {wacc:.2%} for {fin.ticker}. A non-positive discount "
            "rate cannot be used to value anything; check the risk-free rate, the "
            "equity risk premium and the beta before going further."
        )

    if bridge.net_debt < 0:
        notes.append(
            f"{fin.ticker} is net cash by {-bridge.net_debt:,.0f}mm. Weights still use "
            f"gross debt of {debt:,.0f}mm, because the tax shield attaches to debt "
            f"outstanding and not to a net position. At a {weight_debt:.1%} debt "
            "weight the WACC is its cost of equity to within a few basis points."
        )
        if bridge.convertible_debt > 0 and bridge.convertible_in_debt == 0:
            notes.append(
                f"Gross debt reads {debt:,.0f}mm despite {bridge.convertible_debt:,.0f}mm "
                "of convertible notes outstanding, because the bridge carries those as "
                "equity under the if-converted treatment: their shares already sit in "
                "the diluted count the equity weight is built on. Counting them as debt "
                "here while counting their shares in equity would weight the same "
                "instrument twice."
            )

    inputs.extend(
        [
            ("Risk-free rate", risk_free_rate, rf_source),
            ("Equity risk premium", erp, "assumption: market.equity_risk_premium"),
            ("Levered beta", levered_beta, beta_source),
            ("Unlevered beta", unlevered_beta, unlevered_source),
        ]
    )
    if regression is not None:
        inputs.extend(
            [
                (
                    "Regression R-squared",
                    regression.r_squared,
                    f"share of {fin.ticker} weekly variance explained by "
                    f"{mkt_cfg.market_index}",
                ),
                (
                    "Regression observations",
                    float(regression.n_observations),
                    f"weeks common to {fin.ticker} and {mkt_cfg.market_index}",
                ),
            ]
        )
    inputs.extend(
        [
            ("Size premium", size_premium, "assumption: market.size_premium"),
            ("Pre-tax cost of debt", pretax_cost_of_debt, cod_source),
            ("Tax rate", tax_rate, tax_source),
            (
                "After-tax cost of debt",
                after_tax_cost_of_debt,
                f"{pretax_cost_of_debt:.2%} x (1 - {tax_rate:.1%})",
            ),
            (
                "Market value of equity",
                equity,
                f"{bridge.price:,.2f} x {bridge.diluted_shares:,.1f}mm diluted shares",
            ),
            (
                "Total debt (book)",
                debt,
                f"EV bridge at book value, {bridge.lease_convention}",
            ),
            ("Debt / equity", debt_to_equity, "book debt over market equity"),
            ("Weight of equity", weight_equity, f"E / (D + E) on {capital:,.0f}mm"),
            ("Weight of debt", weight_debt, "D / (D + E) on gross debt, not net"),
        ]
    )

    return WACCResult(
        cost_of_equity=cost_of_equity,
        after_tax_cost_of_debt=after_tax_cost_of_debt,
        pretax_cost_of_debt=pretax_cost_of_debt,
        weight_equity=weight_equity,
        weight_debt=weight_debt,
        wacc=wacc,
        risk_free_rate=risk_free_rate,
        erp=erp,
        levered_beta=levered_beta,
        unlevered_beta=unlevered_beta,
        tax_rate=tax_rate,
        credit_rating=credit_rating,
        inputs=inputs,
        notes=notes,
    )
