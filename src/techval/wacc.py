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

    def as_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "raw_beta": self.raw_beta,
            "adjusted_beta": self.adjusted_beta,
            "unlevered_beta": self.unlevered_beta,
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
        adjusted = 0.67 * raw_beta + 0.33 * 1.0
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
    )


def _median_unlevered(estimates: Sequence[BetaEstimate]) -> float:
    if not estimates:
        raise MissingDataError(
            "peer unlevered beta",
            hint="no peer beta estimates were supplied to take a median of",
        )
    return float(np.median([e.unlevered_beta for e in estimates]))


def peer_unlevered_beta(
    peers: Sequence[tuple[PriceSeries, float, float]],
    market: PriceSeries,
    *,
    adjustment: str = "raw",
    lookback_years: float | None = None,
    min_observations: int = MIN_OBSERVATIONS,
) -> tuple[float, list[BetaEstimate]]:
    """Median unlevered beta of a peer set, and the estimates behind it.

    Each peer is ``(prices, debt_to_equity, tax_rate)``, with the ratio and the
    rate belonging to that peer, not to the target. Unlevering happens before
    averaging, which is the entire point: it removes the capital structure
    differences that make raw betas incomparable, and leaves an asset beta that
    describes the business the peers have in common. The caller relevers the
    median at the target's own D/E.

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
    return _median_unlevered(estimates), estimates


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
    When supplied, their median unlevered beta is relevered at the target's own
    capital structure and the target's own regression is not run at all, which is
    the right call whenever the target's R-squared is poor or its listing history
    is short.
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
        unlevered_beta = _median_unlevered(peer_betas)
        levered_beta = relever(unlevered_beta, debt_to_equity, tax_rate)
        tickers = ", ".join(e.ticker for e in peer_betas)
        beta_source = (
            f"median of {len(peer_betas)} peer unlevered betas ({tickers}), "
            f"relevered at {debt_to_equity:,.2f}x D/E and a {tax_rate:.1%} tax rate"
        )
        unlevered_source = (
            f"Hamada on each peer at its own D/E and tax rate, then the median. "
            f"{mkt_cfg.beta_adjustment} adjustment applied to each peer regression"
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
