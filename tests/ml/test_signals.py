"""The harness has to be provably unable to confirm what it is handed.

Four things are worth testing in a signal harness and they are all failure
modes rather than features.

**It must not inflate a t-statistic.** The central test in this file is a Monte
Carlo with no signal in it at all, sampled the way this package samples: twelve
month returns taken quarterly, so each coefficient shares three quarters of its
path with the next. A naive t-statistic rejects the true null about a third of
the time on that design, six times the nominal rate, and the Newey-West figure
brings it back most of the way. The gap between the two is the single number
this module exists to produce, and the test measures it rather than asserting
it.

**It must not certify hindsight.** A score built with knowledge that postdates
its own date is refused, not warned about, and both routes to that refusal are
tested: a knowledge date later than the score date, and a provenance carrying a
filing later than the score date.

**It must not let a takeout vanish.** A company acquired mid-window has the
largest return in the sample and no twelve month price. The tests check that it
is terminated at the deal rather than dropped, that dropping it moves the answer
the way the bias says it should, and that a name stopping because the data
stopped is not mistaken for one that stopped because the company did.

**It must not flatter noise.** A pure noise score must fail against its own
permutation baseline and a constructed one must beat it, both through the same
code path.

Everything except the two recorded-panel tests is generated from a seeded numpy
Generator, so no number here depends on a network or a machine.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from techval.backtest import LookaheadError, forward_return
from techval.config import Assumptions
from techval.edgar import Provenance
from techval.errors import ConfigError, NotMeaningfulError
from techval.market import PriceSeries
from techval.ml import signals
from techval.ml.protocol import spearman
from techval.ml.signals import (
    DELISTING_CONVENTIONS,
    DealTerm,
    Score,
    block_bootstrap_se,
    deal_terms_from_events,
    newey_west_se,
    overlap_lag,
    two_sided_p,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "signals"

HORIZON_DAYS = 365


# --------------------------------------------------------------------------- #
# Synthetic panels
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _business_days(start: date, end: date) -> tuple[date, ...]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return tuple(out)


def business_days(start: date, end: date) -> list[date]:
    """Trading-day calendar for the synthetic panels, built once per span.

    Every test in this file lays its prices on one of three spans and building
    the calendar is a Python loop over four thousand days, so it is cached. The
    list is copied out of the cache because callers index into it.
    """
    return list(_business_days(start, end))


def quarterly(start: date, n: int) -> list[date]:
    """Rebalance dates three months apart, pinned to the fifteenth.

    The fifteenth rather than a month end, so that no date lands on a quarter
    close and picks up the volume of an index rebalance in its entry price.
    """
    out, y, m = [], start.year, start.month
    for _ in range(n):
        out.append(date(y, m, start.day))
        m += 3
        if m > 12:
            y, m = y + 1, m - 12
    return out


def make_prices(
    rng: np.random.Generator,
    tickers: list[str],
    start: date,
    end: date,
    drift: float = 0.0003,
    vol: float = 0.02,
) -> dict[str, PriceSeries]:
    days = business_days(start, end)
    out = {}
    for t in tickers:
        steps = rng.normal(drift, vol, size=len(days))
        closes = 100.0 * np.exp(np.cumsum(steps))
        out[t] = PriceSeries(t, days, closes, "synthetic")
    return out


def make_scores(
    prices: dict[str, PriceSeries],
    dates: list[date],
    rng: np.random.Generator,
    strength: float,
    noise: float = 1.0,
    only_scored: bool = True,
) -> list[Score]:
    """Scores that carry a known amount of the answer, and no more.

    ``strength`` is how much of the realised forward return is folded into the
    score, so the harness is pointed at a sample whose true information
    coefficient is known by construction. A clairvoyant score is exactly what a
    harness should be tested against: if it cannot find skill that was put there
    on purpose, nothing it says about a real model means anything.

    These scores are fresh noise at every date and carry nothing from the last
    one. That matters more than it looks, and
    ``test_a_persistent_score_is_what_makes_the_coefficients_overlap`` is where
    it is set out: a score with no persistence produces a coefficient series
    with no autocorrelation, whatever the returns underneath it are doing.

    ``only_scored`` keeps the helper to the dates that have a forward return.
    Turn it off to exercise the censored rows at the end of the sample.
    """
    out: list[Score] = []
    for d in dates:
        for t, series in prices.items():
            fr = forward_return(t, d, series, HORIZON_DAYS)
            if fr.total_return is None:
                if only_scored:
                    continue
                out.append(Score(t, d, noise * rng.normal(0.0, 0.3)))
                continue
            out.append(
                Score(t, d, strength * fr.total_return + noise * rng.normal(0.0, 0.3))
            )
    return out


def make_tilted_prices(
    rng: np.random.Generator,
    tickers: list[str],
    start: date,
    end: date,
    tilt: float,
) -> tuple[dict[str, PriceSeries], dict[str, float]]:
    """Prices whose drift is set by a fixed per-company characteristic.

    The characteristic never moves, so a score equal to it is perfectly
    persistent and every pair of adjacent coefficients differs only through the
    overlapping returns underneath. That is the design in which the overlap
    correction has its full effect, and it is what a real valuation multiple
    looks like: a company that is cheap this quarter is cheap next quarter.

    ``tilt`` is the daily drift per unit of characteristic. Zero is a sample with
    no signal in it at all and a coefficient series that is still autocorrelated,
    which is the combination that manufactures a significant t-statistic out of
    nothing.
    """
    days = business_days(start, end)
    prices: dict[str, PriceSeries] = {}
    quality: dict[str, float] = {}
    n = len(tickers)
    for i, t in enumerate(tickers):
        q = (i - (n - 1) / 2) / ((n - 1) / 2)
        quality[t] = q
        steps = rng.normal(0.0003 + tilt * q, 0.02, size=len(days))
        prices[t] = PriceSeries(t, days, 100.0 * np.exp(np.cumsum(steps)), "synthetic")
    return prices, quality


def run(scores, prices, **kwargs):
    """``test_signal`` with the draw counts turned down, for speed.

    The defaults are four hundred permutations and a thousand bootstrap
    resamples, which is right for a result somebody will quote and wasteful
    inside a test that only needs the machinery to run.
    """
    kwargs.setdefault("baseline_draws", 40)
    kwargs.setdefault("bootstrap_draws", 80)
    kwargs.setdefault("min_names_per_date", 20)
    return signals.test_signal(scores, prices, **kwargs)


# --------------------------------------------------------------------------- #
# Trap one: the overlapping window
# --------------------------------------------------------------------------- #


def overlapping_ic_series(rng: np.random.Generator, n_dates: int) -> np.ndarray:
    """A coefficient series with no signal in it and the overlap of the real design.

    Each coefficient is the mean of twelve monthly shocks and the series is
    sampled every third month, which is exactly what a twelve month horizon
    rebalanced quarterly does to a coefficient series. Its autocorrelations are
    0.75, 0.50 and 0.25 by construction and zero from the fourth lag on, so the
    true variance of the mean is 1 + 2 * (0.75 + 0.50 + 0.25) = 4 times the
    naive one, and the true standard error is exactly double.
    """
    months = n_dates * 3 + 12
    shocks = rng.normal(size=months)
    return np.asarray([shocks[i : i + 12].mean() for i in range(0, n_dates * 3, 3)])


def test_the_naive_t_statistic_rejects_a_true_null_a_third_of_the_time():
    """The number the brief calls roughly double, measured rather than asserted.

    Three hundred replications of a sample with no signal in it. The naive
    statistic clears two on about a third of them against a nominal five per
    cent, because it is dividing by a standard error that is half the right one.
    Newey-West at the matched lag brings the rejection rate most of the way back
    without quite reaching nominal, and the shortfall is the small-sample bias of
    the estimator rather than anything wrong with the design.
    """
    rng = np.random.default_rng(20260911)
    reps, n_dates, lag = 300, 40, 3
    naive_hits = nw_hits = 0
    inflations = []
    for _ in range(reps):
        series = overlapping_ic_series(rng, n_dates)
        naive = float(np.std(series, ddof=1)) / math.sqrt(series.size)
        nw, _ = newey_west_se(series, lag)
        inflations.append(nw / naive)
        mean = float(series.mean())
        naive_hits += abs(mean / naive) >= 1.96
        nw_hits += abs(mean / nw) >= 1.96

    naive_rate = naive_hits / reps
    nw_rate = nw_hits / reps
    assert naive_rate > 0.25, f"expected a badly oversized naive test, got {naive_rate:.2%}"
    assert nw_rate < naive_rate / 1.8, (
        f"the correction has to do most of the work: naive {naive_rate:.2%}, "
        f"corrected {nw_rate:.2%}"
    )
    assert nw_rate < 0.20


def test_the_bartlett_correction_recovers_most_but_not_all_of_the_inflation():
    """The honest limit of the fix, which the module docstring has to own.

    The true standard error on this design is exactly twice the naive one. A
    Bartlett kernel truncated at the matched lag of three weights the three
    autocovariances by 0.75, 0.50 and 0.25, so even with infinite data it
    recovers sqrt(1 + 2 * (0.75 * 0.75 + 0.50 * 0.50 + 0.25 * 0.25)) = 1.66 of
    that factor of two and not the whole of it. The corrected t-statistic this
    module reports is therefore still about a fifth too large, and saying so is
    the difference between a correction and a claim.

    A longer lag closes that gap asymptotically and does NOT close it at the
    forty-odd dates this package can offer, which is the finding that settles
    the lag choice. At two thousand observations the recovered factor rises
    monotonically with the lag; at forty it peaks around the matched lag and
    falls away again, because the sample autocovariances at high lags are
    estimated from a handful of products and shrink toward zero. Reaching for a
    longer lag on a short series buys noise, not conservatism.
    """
    def recovered(n_dates: int, reps: int, lags: tuple[int, ...]) -> dict[int, float]:
        rng = np.random.default_rng(4)
        got: dict[int, list[float]] = {lag: [] for lag in lags}
        for _ in range(reps):
            series = overlapping_ic_series(rng, n_dates)
            naive = float(np.std(series, ddof=1)) / math.sqrt(series.size)
            for lag in lags:
                nw, _ = newey_west_se(series, lag)
                got[lag].append(nw / naive)
        return {lag: float(np.mean(v)) for lag, v in got.items()}

    asymptotic = recovered(2000, 40, (3, 5, 7, 11))
    assert asymptotic[3] == pytest.approx(1.658, abs=0.04)
    assert asymptotic[3] < asymptotic[5] < asymptotic[7] < asymptotic[11]
    assert asymptotic[11] < 2.0, "even a long lag understates on a Bartlett kernel"

    short = recovered(40, 400, (3, 5, 7, 11))
    assert all(1.4 < v < 1.7 for v in short.values()), short
    assert short[11] < short[5], (
        "at forty dates a longer lag must not be assumed to help: " f"{short}"
    )
    assert short[3] < asymptotic[3], "the estimator is biased downward in small samples"


def test_the_measured_autocorrelations_are_the_ones_the_design_implies():
    rng = np.random.default_rng(7)
    series = overlapping_ic_series(rng, 4000)
    _, rhos = newey_west_se(series, 5)
    assert rhos[0] == pytest.approx(0.75, abs=0.03)
    assert rhos[1] == pytest.approx(0.50, abs=0.03)
    assert rhos[2] == pytest.approx(0.25, abs=0.03)
    assert rhos[3] == pytest.approx(0.0, abs=0.03)
    assert rhos[4] == pytest.approx(0.0, abs=0.03)


def test_a_lag_of_zero_is_the_naive_standard_error_exactly():
    """One code path for both numbers, so they cannot drift apart."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=50)
    nw, rhos = newey_west_se(x, 0)
    # The naive form here divides the population variance by n, which is the
    # ddof=0 convention the Newey-West estimator uses throughout.
    assert nw == pytest.approx(float(np.std(x, ddof=0)) / math.sqrt(x.size))
    assert rhos == []


def test_newey_west_matches_the_arithmetic_in_its_own_docstring():
    x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    nw, rhos = newey_west_se(x, 2)
    dev = x - x.mean()
    n = x.size
    g0 = float(dev @ dev) / n
    g1 = float(dev[1:] @ dev[:-1]) / n
    g2 = float(dev[2:] @ dev[:-2]) / n
    expected = math.sqrt((g0 + 2 * ((2 / 3) * g1 + (1 / 3) * g2)) / n)
    assert nw == pytest.approx(expected)
    assert rhos == pytest.approx([g1 / g0, g2 / g0])


def test_negative_autocorrelation_shrinks_the_standard_error():
    """The correction is not always an inflation, and the module must not assume it.

    A strictly alternating series has a negative first autocovariance, so the
    corrected standard error is the smaller of the two and the naive figure was
    the conservative one.
    """
    x = np.asarray([1.0, -1.0] * 20)
    nw, rhos = newey_west_se(x, 3)
    naive = float(np.std(x, ddof=0)) / math.sqrt(x.size)
    assert rhos[0] < 0
    assert nw < naive


def test_a_bartlett_kernel_never_returns_a_negative_variance():
    rng = np.random.default_rng(19)
    for _ in range(200):
        x = rng.normal(size=rng.integers(5, 40))
        for lag in range(0, min(4, x.size - 1)):
            nw, _ = newey_west_se(x, lag)
            assert nw >= 0.0 and math.isfinite(nw)


def test_the_lag_matches_the_overlap_and_does_not_move_with_february():
    dates = quarterly(date(2016, 3, 15), 40)
    lag, spacing = overlap_lag(dates, 12)
    assert lag == 3
    assert 89.0 <= spacing <= 93.0
    assert overlap_lag(dates, 3)[0] == 0
    assert overlap_lag(dates, 24)[0] == 7
    monthly = [date(2020, 1 + i % 12, 15) if i < 12 else date(2021, 1 + (i - 12) % 12, 15) for i in range(24)]
    assert overlap_lag(monthly, 12)[0] == 11


def test_the_lag_refuses_to_exceed_the_series():
    with pytest.raises(ConfigError, match="more autocovariances"):
        newey_west_se(np.arange(5.0), 5)


def test_the_block_bootstrap_is_a_second_opinion_not_a_repeat_of_the_naive():
    rng = np.random.default_rng(5)
    series = overlapping_ic_series(rng, 60)
    naive = float(np.std(series, ddof=1)) / math.sqrt(series.size)
    block = block_bootstrap_se(series, 4, 500, np.random.default_rng(5))
    single = block_bootstrap_se(series, 1, 500, np.random.default_rng(5))
    assert block is not None and single is not None
    assert block > single, "blocks longer than the overlap must widen the interval"
    assert single == pytest.approx(naive, rel=0.25)


def test_two_sided_p_agrees_with_the_familiar_landmarks():
    assert two_sided_p(1.959963984540054) == pytest.approx(0.05, abs=1e-6)
    assert two_sided_p(0.0) == pytest.approx(1.0)
    assert two_sided_p(-1.959963984540054) == pytest.approx(0.05, abs=1e-6)
    assert math.isnan(two_sided_p(float("nan")))


# --------------------------------------------------------------------------- #
# Trap three: point in time, refused rather than warned about
# --------------------------------------------------------------------------- #


def test_a_score_pinned_after_its_own_date_is_refused():
    rng = np.random.default_rng(1)
    dates = quarterly(date(2018, 3, 15), 24)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(30)], date(2017, 1, 2), date(2026, 1, 2))
    scores = make_scores(prices, dates, rng, strength=0.0)
    leaked = [
        Score(s.ticker, s.as_of, s.value, knowledge_date=s.as_of + timedelta(days=90))
        for s in scores
    ]
    with pytest.raises(LookaheadError, match="pinned to"):
        run(leaked, prices)


def test_a_score_pinned_before_its_own_date_is_allowed():
    """Using less than was knowable is conservative, not dishonest."""
    rng = np.random.default_rng(1)
    dates = quarterly(date(2018, 3, 15), 24)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(30)], date(2017, 1, 2), date(2026, 1, 2))
    scores = [
        Score(s.ticker, s.as_of, s.value, knowledge_date=s.as_of - timedelta(days=30))
        for s in make_scores(prices, dates, rng, strength=0.0)
    ]
    assert run(scores, prices).ic.n > 0


def test_a_provenance_filed_after_the_score_date_is_refused():
    """The check that catches a client built without a knowledge date at all."""
    rng = np.random.default_rng(2)
    dates = quarterly(date(2018, 3, 15), 24)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(30)], date(2017, 1, 2), date(2026, 1, 2))
    scores = make_scores(prices, dates, rng, strength=0.0)
    late = Provenance(
        concept="Revenues",
        tag="Revenues",
        method="ttm",
        filed=(scores[0].as_of + timedelta(days=45)).isoformat(),
    )
    tainted = [
        Score(scores[0].ticker, scores[0].as_of, scores[0].value, provenance={"revenue": late})
    ] + scores[1:]
    with pytest.raises(LookaheadError, match="not on file until later"):
        run(tainted, prices)


def test_a_duplicate_company_date_is_refused():
    rng = np.random.default_rng(3)
    dates = quarterly(date(2018, 3, 15), 24)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(30)], date(2017, 1, 2), date(2026, 1, 2))
    scores = make_scores(prices, dates, rng, strength=0.0)
    with pytest.raises(ConfigError, match="two scores dated"):
        run(scores + [scores[0]], prices)


def test_too_few_dates_is_refused_rather_than_reported_thinly():
    rng = np.random.default_rng(4)
    dates = quarterly(date(2018, 3, 15), 4)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(30)], date(2017, 1, 2), date(2026, 1, 2))
    with pytest.raises(NotMeaningfulError, match="rebalance date"):
        run(make_scores(prices, dates, rng, strength=0.0), prices)


def test_thin_cross_sections_are_dropped_and_counted():
    rng = np.random.default_rng(6)
    dates = quarterly(date(2018, 3, 15), 24)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(30)], date(2017, 1, 2), date(2026, 1, 2))
    scores = make_scores(prices, dates, rng, strength=0.0)
    # Strip one date down to five names, which is below the floor.
    thin_date = dates[5]
    kept, seen = [], 0
    for s in scores:
        if s.as_of == thin_date:
            seen += 1
            if seen > 5:
                continue
        kept.append(s)
    result = run(kept, prices)
    assert thin_date not in result.ic.dates
    assert any("fewer than 20 scored names" in n for n in result.notes)


# --------------------------------------------------------------------------- #
# Trap two: survivorship, delisting, and the edge of the data
# --------------------------------------------------------------------------- #


def clip(series: PriceSeries, last: date) -> PriceSeries:
    idx = [i for i, d in enumerate(series.dates) if d <= last]
    return PriceSeries(series.symbol, [series.dates[i] for i in idx], series.closes[idx], series.source)


def test_an_acquisition_is_terminated_at_the_offer_price():
    rng = np.random.default_rng(8)
    dates = quarterly(date(2018, 3, 15), 24)
    names = [f"T{i:02d}" for i in range(30)]
    prices = make_prices(rng, names, date(2017, 1, 2), date(2026, 1, 2))
    scores = make_scores(prices, dates, rng, strength=0.0)

    target, announced = "T00", date(2020, 9, 1)
    prices[target] = clip(prices[target], announced)
    offer = float(prices[target].closes[-1]) * 1.4
    deals = [DealTerm(target, announced, offer_price=offer, completed=True)]

    result = run(scores, prices, deals=deals, benchmark=prices["T01"])
    taken = [o for o in result.outcomes if o.ticker == target and o.exit_kind == "acquired"]
    assert taken, "the acquisition must terminate the holding periods that straddle it"
    for o in taken:
        assert o.exit_price == pytest.approx(offer)
        assert o.forward_return is not None
        assert "40" not in o.note or "offer price" in o.note
        # The whole horizon is accounted for, not the part before the deal.
        assert o.held_days == pytest.approx(365, abs=2)


def test_a_deal_announced_on_the_score_date_does_not_count():
    """The market has already repriced it, so the premium is in the entry price."""
    rng = np.random.default_rng(9)
    dates = quarterly(date(2018, 3, 15), 24)
    names = [f"T{i:02d}" for i in range(30)]
    prices = make_prices(rng, names, date(2017, 1, 2), date(2026, 1, 2))
    scores = make_scores(prices, dates, rng, strength=0.0)
    on_the_day = dates[8]
    prices["T00"] = clip(prices["T00"], on_the_day)
    deals = [DealTerm("T00", on_the_day, offer_price=200.0)]
    result = run(scores, prices, deals=deals)
    same_day = [o for o in result.outcomes if o.ticker == "T00" and o.as_of == on_the_day]
    assert same_day and same_day[0].exit_kind != "acquired"


def test_dropping_the_takeout_biases_the_bucket_spread():
    """The bias has a direction, and this measures it rather than asserting it.

    A takeout at a premium is the largest return in its cross-section. Scoring
    the same sample with and without the deal on record changes the return of
    whichever bucket held the target, and leaving it out is the version that
    cannot see the best outcome in the sample.
    """
    rng = np.random.default_rng(10)
    dates = quarterly(date(2018, 3, 15), 28)
    names = [f"T{i:02d}" for i in range(40)]
    prices = make_prices(rng, names, date(2017, 1, 2), date(2026, 6, 1))
    scores = make_scores(prices, dates, rng, strength=0.0)

    announced = date(2021, 6, 1)
    targets = names[:6]
    deals = []
    for t in targets:
        prices[t] = clip(prices[t], announced)
        deals.append(DealTerm(t, announced, offer_price=float(prices[t].closes[-1]) * 1.5))

    with_deals = run(scores, prices, deals=deals, benchmark=prices["T39"])
    without = run(scores, prices, deals=[])

    assert with_deals.exit_counts().get("acquired", 0) >= len(targets)
    assert without.exit_counts().get("acquired", 0) == 0
    assert without.exit_counts().get("delisted", 0) >= len(targets)
    assert with_deals.n_scored > without.n_scored
    assert any("delisted company-date" in c for c in without.checks)


def test_a_delisting_is_not_the_edge_of_the_sample():
    """The distinction that stops a data cut-off looking like a wave of failures."""
    rng = np.random.default_rng(11)
    # The dates run past the end of the price data on purpose, so that the last
    # few carry no forward return for anybody and the censored case is exercised
    # beside the delisted one.
    dates = quarterly(date(2018, 3, 15), 32)
    names = [f"T{i:02d}" for i in range(40)]
    prices = make_prices(rng, names, date(2017, 1, 2), date(2026, 1, 2))
    scores = make_scores(prices, dates, rng, strength=0.0, only_scored=False)
    prices["T00"] = clip(prices["T00"], date(2022, 4, 1))

    result = run(scores, prices)
    kinds: dict[str, set[str]] = {}
    for o in result.outcomes:
        kinds.setdefault(o.ticker, set()).add(o.exit_kind)
    assert "delisted" in kinds["T00"]
    assert "delisted" not in kinds["T01"]
    # The last dates in the sample have no twelve month forward and are censored
    # for every name, which says nothing about any of them.
    assert result.exit_counts()["censored"] > 0
    censored = {o.ticker for o in result.outcomes if o.exit_kind == "censored"}
    assert len(censored) > 30, "censoring at the data edge hits the whole universe"


def test_a_stated_delisting_return_scores_the_failure_rather_than_dropping_it():
    rng = np.random.default_rng(12)
    dates = quarterly(date(2018, 3, 15), 28)
    names = [f"T{i:02d}" for i in range(40)]
    prices = make_prices(rng, names, date(2017, 1, 2), date(2026, 1, 2))
    scores = make_scores(prices, dates, rng, strength=0.0)
    for t in names[:5]:
        prices[t] = clip(prices[t], date(2022, 4, 1))

    dropped = run(scores, prices, delisting_return=None)
    scored = run(scores, prices, delisting_return="shumway_nasdaq")
    assert scored.n_scored > dropped.n_scored
    failures = [o for o in scored.outcomes if o.exit_kind == "delisted" and o.scored]
    assert failures
    assert all(o.forward_return == pytest.approx(-0.55) for o in failures)
    assert DELISTING_CONVENTIONS["shumway_nasdaq"] == -0.55


def test_an_unknown_delisting_convention_is_refused():
    rng = np.random.default_rng(13)
    dates = quarterly(date(2018, 3, 15), 24)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(30)], date(2017, 1, 2), date(2026, 1, 2))
    scores = make_scores(prices, dates, rng, strength=0.0)
    with pytest.raises(ConfigError, match="unknown delisting convention"):
        run(scores, prices, delisting_return="optimistic")


def test_a_universe_with_no_exits_at_all_is_flagged_as_survivors():
    rng = np.random.default_rng(14)
    dates = quarterly(date(2018, 3, 15), 24)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(30)], date(2017, 1, 2), date(2026, 1, 2))
    result = run(make_scores(prices, dates, rng, strength=0.0), prices)
    assert any("list of today's survivors" in c for c in result.checks)


def test_deal_terms_from_events_carry_no_price_and_say_so():
    terms = deal_terms_from_events([("ZEN", date(2022, 6, 24), True)])
    assert terms == [DealTerm("ZEN", date(2022, 6, 24), None, True, "deal_events")]


# --------------------------------------------------------------------------- #
# The coefficient, the buckets, the turnover and the baseline
# --------------------------------------------------------------------------- #


def test_a_pure_noise_score_does_not_beat_its_own_baseline():
    rng = np.random.default_rng(15)
    dates = quarterly(date(2017, 3, 15), 32)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    result = run(make_scores(prices, dates, rng, strength=0.0), prices, label="noise")
    assert abs(result.ic.mean) < 0.05
    assert abs(result.ic.t_newey_west) < 1.96
    assert not result.significant
    assert result.permutation_p is not None and result.permutation_p > 0.05
    assert "NOT" in result.verdict() or "Not significant" in result.verdict()


def test_the_baseline_comparison_opens_its_own_sentence_in_the_verdict():
    """The evaluation's line is written to stand alone, and here it follows a full stop.

    ``EvalResult.verdict`` opens on the metric's name in lower case, which reads
    correctly as a line of its own and wrongly in the middle of a paragraph:
    "... erases the spread. mean information coefficient of ...". The join
    capitalises it and changes nothing else.
    """
    rng = np.random.default_rng(15)
    dates = quarterly(date(2017, 3, 15), 32)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    result = run(make_scores(prices, dates, rng, strength=0.0), prices, label="noise")

    line = result.evaluation.verdict()
    opening = "mean information coefficient of "
    assert line.startswith(opening)
    text = result.verdict()
    assert text.endswith(" Mean information coefficient of " + line.removeprefix(opening))
    assert ". mean information coefficient" not in text


def test_a_score_built_to_work_is_found_to_work():
    """If the harness cannot find skill that was put there deliberately, nothing it says means anything."""
    rng = np.random.default_rng(16)
    dates = quarterly(date(2017, 3, 15), 32)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    result = run(make_scores(prices, dates, rng, strength=3.0), prices, label="oracle")
    assert result.ic.mean > 0.3
    assert result.ic.share_positive > 0.9
    assert result.evaluation.beat_baseline
    assert result.significant
    assert result.buckets.spread > 0
    assert result.buckets.monotone_steps == 1.0
    assert result.buckets.monotonicity == pytest.approx(1.0)


def test_the_buckets_run_from_the_lowest_score_to_the_highest():
    rng = np.random.default_rng(17)
    dates = quarterly(date(2017, 3, 15), 32)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    result = run(make_scores(prices, dates, rng, strength=3.0), prices, buckets=5)
    frame = result.buckets.frame
    assert list(frame.index) == [1, 2, 3, 4, 5]
    assert frame["Mean score"].is_monotonic_increasing
    assert frame["Mean return"].iloc[-1] > frame["Mean return"].iloc[0]
    assert result.buckets.spread == pytest.approx(
        frame["Mean return"].iloc[-1] - frame["Mean return"].iloc[0], abs=1e-9
    )


def test_the_spread_carries_the_same_overlap_correction_as_the_coefficient():
    rng = np.random.default_rng(18)
    dates = quarterly(date(2017, 3, 15), 32)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    result = run(make_scores(prices, dates, rng, strength=3.0), prices)
    naive = float(np.std(result.buckets.spread_by_date, ddof=1)) / math.sqrt(
        len(result.buckets.spread_by_date)
    )
    assert result.buckets.lag == result.ic.lag == 3
    assert result.buckets.spread_se != pytest.approx(naive)


def test_a_ranking_that_never_changes_has_no_turnover():
    rng = np.random.default_rng(21)
    dates = quarterly(date(2017, 3, 15), 32)
    names = [f"T{i:02d}" for i in range(50)]
    prices = make_prices(rng, names, date(2016, 6, 1), date(2026, 6, 1))
    fixed = {t: float(i) for i, t in enumerate(names)}
    scores = [
        Score(s.ticker, s.as_of, fixed[s.ticker]) for s in make_scores(prices, dates, rng, 0.0)
    ]
    result = run(scores, prices)
    assert result.turnover.top == pytest.approx(0.0)
    assert result.turnover.bottom == pytest.approx(0.0)
    assert result.turnover.breakeven_cost_bps is None


def test_a_ranking_redrawn_every_date_turns_over_almost_completely():
    rng = np.random.default_rng(22)
    dates = quarterly(date(2017, 3, 15), 32)
    names = [f"T{i:02d}" for i in range(50)]
    prices = make_prices(rng, names, date(2016, 6, 1), date(2026, 6, 1))
    scores = [
        Score(s.ticker, s.as_of, float(rng.normal())) for s in make_scores(prices, dates, rng, 0.0)
    ]
    result = run(scores, prices, buckets=5)
    # Ten names in a bucket of fifty drawn afresh: about four fifths are new.
    assert 0.6 < result.turnover.mean < 0.95
    assert result.turnover.rebalances_per_year == pytest.approx(4.0, abs=0.1)


def test_the_break_even_cost_falls_as_turnover_rises():
    rng = np.random.default_rng(23)
    dates = quarterly(date(2017, 3, 15), 32)
    names = [f"T{i:02d}" for i in range(50)]
    prices = make_prices(rng, names, date(2016, 6, 1), date(2026, 6, 1))
    base = make_scores(prices, dates, rng, strength=3.0)
    steady = run(base, prices)
    jittered = run(
        [Score(s.ticker, s.as_of, s.value + rng.normal(0, 3.0)) for s in base], prices
    )
    assert jittered.turnover.mean > steady.turnover.mean
    assert steady.buckets.spread > 0 and steady.turnover.breakeven_cost_bps is not None


def test_the_baseline_is_noise_on_this_sample_and_not_a_zero():
    """Permuting within date preserves the shape and destroys only the pairing."""
    rng = np.random.default_rng(24)
    dates = quarterly(date(2017, 3, 15), 32)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    scores = make_scores(prices, dates, rng, strength=0.0)
    result = signals.test_signal(
        scores, prices, baseline_draws=200, bootstrap_draws=200, min_names_per_date=20
    )
    assert "permutations within date" in result.evaluation.baseline_name
    assert abs(result.evaluation.baseline_score) < 0.02
    # A single permutation draw is not tiny even though its expectation is zero,
    # which is the whole reason the null is the distribution and not the point.
    assert result.permutation_p is not None and 0.0 <= result.permutation_p <= 1.0


def test_the_permutation_null_is_exactly_a_spearman_on_shuffled_ranks():
    """The optimisation must be algebra, not an approximation.

    ``_permutation_means`` correlates standardised ranks with a dot product
    rather than calling ``spearman`` four hundred times. That is the same number
    by construction, including under ties, and this is where it is checked
    rather than asserted in a comment.
    """
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(5, 60))
        # Rounded, so that ties are part of the test rather than a special case.
        a = rng.normal(size=n).round(int(rng.integers(0, 3)))
        b = rng.normal(size=n)
        x = signals._standardised_ranks(a)
        y = signals._standardised_ranks(b)
        assert float(x @ y) == pytest.approx(spearman(a, b), abs=1e-12)
    assert signals._standardised_ranks(np.zeros(5)) is None


def test_the_same_seed_gives_the_same_answer_twice():
    rng = np.random.default_rng(25)
    dates = quarterly(date(2017, 3, 15), 32)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    scores = make_scores(prices, dates, rng, strength=1.0)
    a = run(scores, prices, seed=7)
    b = run(scores, prices, seed=7)
    assert a.ic.values == b.ic.values
    assert a.permutation_p == b.permutation_p
    assert a.ic.block_bootstrap_se == b.ic.block_bootstrap_se
    assert a.verdict() == b.verdict()


# --------------------------------------------------------------------------- #
# Trap four: how many tests were run
# --------------------------------------------------------------------------- #


def test_the_p_value_is_adjusted_for_the_number_of_tests_the_caller_admits_to():
    rng = np.random.default_rng(26)
    dates = quarterly(date(2017, 3, 15), 32)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    # A weak signal, so the p-value is small enough to be interesting and large
    # enough to still be a number. A clairvoyant score underflows to zero and
    # the adjustment then has nothing to act on.
    scores = make_scores(prices, dates, rng, strength=0.05)
    alone = run(scores, prices, n_tests_run=1)
    searched = run(scores, prices, n_tests_run=24)
    assert alone.p_newey_west == pytest.approx(searched.p_newey_west)
    assert searched.p_adjusted > alone.p_adjusted
    assert searched.p_adjusted == pytest.approx(1 - (1 - alone.p_newey_west) ** 24)
    assert "24 tests were run" in searched.verdict()
    assert "pre-register" in searched.verdict()
    assert any("One test is claimed" in c for c in alone.checks)


def test_a_marginal_result_can_survive_alone_and_die_under_the_search():
    """The point of the adjustment, on a sample tuned to sit on the line.

    A signal strength chosen so the uncorrected-for-search p-value lands just
    inside five per cent. Reported as the only test anybody ran, it is a result.
    Reported as the best of thirty, it is what thirty tries produce on noise.
    Nothing in the arithmetic of the first reading can tell you which it was.
    """
    rng = np.random.default_rng(27)
    dates = quarterly(date(2017, 3, 15), 40)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    scores = make_scores(prices, dates, np.random.default_rng(99), strength=0.09)

    one = run(scores, prices, n_tests_run=1)
    many = run(scores, prices, n_tests_run=30)
    assert 0.005 < one.p_newey_west < 0.045, one.p_newey_west
    assert one.significant
    assert not many.significant
    assert many.p_adjusted > 0.4


# --------------------------------------------------------------------------- #
# Housekeeping the harness itself depends on
# --------------------------------------------------------------------------- #


def test_the_entry_point_is_not_collected_as_a_test():
    """``test_signal`` is the module's entry point and pytest must leave it alone.

    Any module-level callable whose name begins with ``test_`` is collected and
    called with no arguments, which would fail every time this module is
    imported by name. The flag is the documented way to opt out and it is
    asserted here so that nobody removes it.
    """
    assert signals.test_signal.__test__ is False


def test_bare_triples_are_accepted_alongside_score_objects():
    rng = np.random.default_rng(28)
    dates = quarterly(date(2017, 3, 15), 32)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    scores = make_scores(prices, dates, rng, strength=1.0)
    triples = [(s.ticker, s.as_of, s.value) for s in scores]
    assert run(triples, prices).ic.values == run(scores, prices).ic.values


def test_a_malformed_score_is_refused_with_the_shape_it_wanted():
    rng = np.random.default_rng(29)
    prices = make_prices(rng, ["T00"], date(2016, 6, 1), date(2026, 6, 1))
    with pytest.raises(ConfigError, match=r"\(ticker, date, value\)"):
        signals.test_signal([("T00", date(2020, 1, 1))], prices)


def test_the_frame_keeps_the_rows_that_could_not_be_scored():
    rng = np.random.default_rng(30)
    dates = quarterly(date(2017, 3, 15), 38)
    prices = make_prices(rng, [f"T{i:02d}" for i in range(50)], date(2016, 6, 1), date(2026, 6, 1))
    scores = make_scores(prices, dates, rng, strength=0.0, only_scored=False)
    result = run(scores, prices)
    frame = result.frame()
    assert len(frame) == len(result.outcomes)
    assert frame["Forward return"].isna().sum() > 0
    assert set(frame["Exit kind"]) >= {"held", "censored"}


def test_the_effective_sample_lands_near_the_independent_period_count():
    """The check that the lag was set sensibly, stated as a property.

    Forty quarterly coefficients on a twelve month horizon span ten years and
    contain about ten non-overlapping annual periods. Where the score is
    persistent, the Newey-West variance should say the series is worth something
    of that order rather than forty, and seeing the two land in the same
    neighbourhood is the sanity check on the whole correction.
    """
    rng = np.random.default_rng(55)
    names = [f"T{i:02d}" for i in range(50)]
    prices, quality = make_tilted_prices(rng, names, date(2016, 6, 1), date(2027, 1, 1), tilt=0.0002)
    dates = quarterly(date(2016, 9, 15), 40)
    scores = [Score(t, d, quality[t]) for d in dates for t in names]
    result = run(scores, prices)

    assert result.ic.n >= 35
    assert 8 <= result.n_independent_periods <= 11
    assert result.ic.effective_n < result.ic.n / 2
    assert result.ic.effective_n < 2 * result.n_independent_periods


def test_a_persistent_score_is_what_makes_the_coefficients_overlap():
    """The refinement the brief does not make, and it changes the advice.

    Overlapping returns on their own do NOT autocorrelate a cross-sectional
    coefficient series. The coefficient is a rank correlation inside one date, so
    a market move common to every name cancels out of it entirely, and what
    survives from one date to the next is the part of the ordering that
    persisted. Hand the harness a score redrawn from noise at every rebalance and
    the coefficient series comes back with no autocorrelation worth correcting,
    whatever the returns underneath it are doing.

    Persistence is the other half. A valuation multiple is about as persistent as
    a company characteristic gets: cheap this quarter is cheap next quarter. Pair
    that with returns that share three quarters of their path and the first-order
    autocorrelation of the coefficients lands on 0.75, which is exactly the
    overlap fraction, and the standard error needs the full correction.

    So the rule is not "overlapping windows inflate the t-statistic". It is
    "overlapping windows inflate the t-statistic of a persistent score", and the
    signals anybody actually tests are persistent.
    """
    names = [f"T{i:02d}" for i in range(50)]
    dates = quarterly(date(2016, 9, 15), 40)

    # Three seeds rather than one, because a single run of an uncorrelated
    # series lands anywhere between about 0.8 and 1.2 and the claim is about the
    # absence of a correction rather than about any one draw of it.
    fresh_inflations = []
    for seed in (56, 66):
        fresh_prices = make_prices(
            np.random.default_rng(seed), names, date(2016, 6, 1), date(2027, 1, 1)
        )
        fresh = run(
            make_scores(fresh_prices, dates, np.random.default_rng(seed + 1), strength=0.3),
            fresh_prices,
        )
        assert abs(fresh.ic.autocorrelations[0]) < 0.35
        fresh_inflations.append(fresh.ic.inflation)
    assert max(fresh_inflations) < 1.25, fresh_inflations

    prices, quality = make_tilted_prices(
        np.random.default_rng(55), names, date(2016, 6, 1), date(2027, 1, 1), tilt=0.0002
    )
    persistent = run([Score(t, d, quality[t]) for d in dates for t in names], prices)
    assert persistent.ic.autocorrelations[0] == pytest.approx(0.75, abs=0.12)
    assert persistent.ic.inflation > 1.5


def test_the_naive_statistic_clears_two_where_the_corrected_one_does_not():
    """The case the whole module exists for, end to end on a constructed sample.

    A persistent characteristic with a real but small tilt behind it. The naive
    t-statistic clears two and would be written up; the corrected one does not,
    and the verdict has to say NOT SIGNIFICANT in those words rather than
    reporting the number that flattered it.
    """
    names = [f"T{i:02d}" for i in range(50)]
    dates = quarterly(date(2016, 9, 15), 40)
    prices, quality = make_tilted_prices(
        np.random.default_rng(55), names, date(2016, 6, 1), date(2027, 1, 1), tilt=0.00008
    )
    result = run([Score(t, d, quality[t]) for d in dates for t in names], prices)

    assert abs(result.ic.t_naive) >= 1.96
    assert abs(result.ic.t_newey_west) < 1.96
    assert not result.significant
    assert "NOT SIGNIFICANT" in result.verdict()
    assert any("Newey-West at lag 3" in c for c in result.checks)


# --------------------------------------------------------------------------- #
# The recorded run on a real signal
# --------------------------------------------------------------------------- #


def load_recorded_panel() -> tuple[list[Score], dict[str, PriceSeries]]:
    """The committed EV/Revenue panel and the closes behind it.

    Both were recorded from live sources once, on the date the manifest names,
    and are read from the fixture afterwards so that two runs agree byte for
    byte. The scores carry the latest filing date behind each one, so the
    point-in-time claim is checked against the fixture rather than trusted.
    """
    scores: list[Score] = []
    with gzip.open(FIXTURES / "ev_revenue.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            scores.append(
                Score(
                    ticker=row["ticker"],
                    as_of=date.fromisoformat(row["as_of"]),
                    value=-float(row["ev_revenue"]),
                    knowledge_date=date.fromisoformat(row["as_of"]),
                    note=row["max_filed"],
                )
            )
    by_ticker: dict[str, list[tuple[date, float]]] = {}
    with gzip.open(FIXTURES / "closes.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            by_ticker.setdefault(row["ticker"], []).append(
                (date.fromisoformat(row["date"]), float(row["close"]))
            )
    prices = {
        t: PriceSeries(t, [d for d, _ in rows], np.asarray([c for _, c in rows]), "fixture")
        for t, rows in by_ticker.items()
    }
    return scores, prices


@pytest.fixture(scope="module")
def recorded():
    return load_recorded_panel()


def test_every_recorded_score_was_built_from_filings_that_predate_it(recorded):
    """The fixture carries its own point-in-time proof rather than a claim."""
    scores, _ = recorded
    late = [
        s for s in scores if s.note and date.fromisoformat(s.note) > s.as_of
    ]
    assert not late, f"{len(late)} recorded scores rest on a filing dated after them"


def test_the_recorded_panel_is_the_shape_the_manifest_says(recorded):
    scores, prices = recorded
    manifest = json.loads((FIXTURES / "MANIFEST.json").read_text())
    assert len({s.ticker for s in scores}) == manifest["companies_scored"]
    assert len({s.as_of for s in scores}) == manifest["rebalance_dates"]
    assert len(scores) == manifest["company_dates"]
    assert len(prices) == manifest["price_series"]


def test_value_in_technology_is_a_negative_signal_and_not_a_significant_one(recorded):
    """The headline result, pinned so that it cannot drift without being seen.

    Cheapness on trailing EV/Revenue, a hundred technology, media and
    telecommunications companies, thirty-five quarterly cross-sections from
    December 2016 to June 2025, twelve month forward returns.

    The mean coefficient is negative: the cheap half of the technology universe
    underperformed the expensive half over this decade, which is what everyone
    who lived through it remembers and is the opposite of what the textbook says
    value does. The naive t-statistic on that is -2.65 and would be written up.
    The Newey-West figure is -1.62 and would not. The result is a clean negative
    that is also not significant, and reporting it as anything else in either
    direction would be a choice rather than a finding.
    """
    scores, prices = recorded
    result = signals.test_signal(
        scores, prices, label="cheapness (negative trailing EV/Revenue)"
    )

    assert result.ic.n == 35
    assert result.n_scored == 2604
    assert result.ic.mean == pytest.approx(-0.0984, abs=5e-4)
    assert result.ic.share_positive == pytest.approx(0.4286, abs=5e-4)
    assert result.ic.t_naive == pytest.approx(-2.651, abs=5e-3)
    assert result.ic.t_newey_west == pytest.approx(-1.617, abs=5e-3)
    assert result.buckets.spread == pytest.approx(-0.0538, abs=5e-4)

    assert abs(result.ic.t_naive) >= 1.96
    assert abs(result.ic.t_newey_west) < 1.96
    assert not result.significant
    assert "NOT SIGNIFICANT" in result.verdict()
    assert not result.evaluation.beat_baseline
    assert "Use the baseline." in result.evaluation.verdict()


def test_the_real_coefficient_series_carries_the_overlap_the_design_implies(recorded):
    """Measured on real data: 0.76 against a theoretical 0.75.

    A twelve month return sampled quarterly shares three quarters of its path
    with the next one, so a perfectly persistent score should produce a
    coefficient series with a first-order autocorrelation of 0.75. Trailing
    EV/Revenue is about as persistent as a company characteristic gets, and the
    measured figure lands on 0.76. That is the closest thing to a proof that the
    lag is set right, and it is the reason the standard error has to move.
    """
    scores, prices = recorded
    result = signals.test_signal(scores, prices)
    assert result.ic.lag == 3
    assert result.ic.autocorrelations[0] == pytest.approx(0.76, abs=0.02)
    assert result.ic.inflation == pytest.approx(1.64, abs=0.02)
    assert result.ic.effective_n == pytest.approx(13.0, abs=0.2)
    assert result.n_independent_periods == pytest.approx(8.5, abs=0.1)
    # The block bootstrap is the second opinion and has to agree with the kernel.
    assert result.ic.block_bootstrap_se == pytest.approx(result.ic.newey_west_se, rel=0.2)


def test_the_longer_the_horizon_the_further_the_two_t_statistics_diverge(recorded):
    """Three horizons on one panel, which is also three tests and is counted as three.

    The naive statistic gets LARGER as the horizon lengthens, because a longer
    window is a smoother series and a smoother series has a smaller dispersion
    to divide by. The corrected statistic gets SMALLER over the same range,
    because a longer window overlaps more and the correction is tracking the
    overlap rather than the smoothness. They move in opposite directions, which
    is the clearest possible statement that one of them is measuring something
    other than evidence.
    """
    scores, prices = recorded
    got = {}
    for months in (6, 12, 24):
        r = signals.test_signal(
            scores, prices, horizon_months=months, n_tests_run=3, baseline_draws=40
        )
        got[months] = r

    assert [got[m].ic.lag for m in (6, 12, 24)] == [1, 3, 7]
    assert got[6].ic.inflation < got[12].ic.inflation < got[24].ic.inflation
    assert abs(got[6].ic.t_naive) < abs(got[12].ic.t_naive) < abs(got[24].ic.t_naive)
    assert abs(got[24].ic.t_newey_west) < abs(got[12].ic.t_newey_west)
    assert not any(r.significant for r in got.values())
    # Three tests were run and the adjustment is applied to all three.
    assert all(r.p_adjusted > r.p_newey_west for r in got.values())


def test_the_permutation_null_calls_it_significant_and_is_wrong_to(recorded):
    """The caveat in the module docstring, demonstrated on the real panel.

    Not one of four hundred within-date permutations reaches the observed
    coefficient, so the permutation p-value is zero and the signal looks
    overwhelming. It is the right null for whether any cross-sectional relation
    exists and the wrong one for how precisely the mean of thirty-five
    overlapping coefficients is measured, because permuting inside a date leaves
    the dates independent of each other by construction. The Newey-West interval
    is the statistic of record and it says no.
    """
    scores, prices = recorded
    result = signals.test_signal(scores, prices, baseline_draws=400)
    # One over one more than the draw count, which is the floor four hundred
    # permutations can support. Not zero: a finite resample cannot prove zero.
    assert result.permutation_p == pytest.approx(1 / 401)
    assert result.p_newey_west > 0.05
    assert abs(result.evaluation.baseline_score) < 0.01


def test_the_recorded_universe_contains_no_failures_at_all_and_is_flagged_for_it(recorded):
    """The survivorship hole, measured rather than warned about.

    Not one of the hundred companies in this panel was acquired or delisted over
    nine years, which is not what happens to a technology universe over nine
    years. Five names that were in the seed universe are missing from the panel
    entirely, absent from the SEC ticker file and with no price rows. Not all
    five stopped trading: Fiserv is still listed, and the seed names it FI where
    the ticker file carries FISV. The other four are the kind of company a
    universe built from today's listings loses, and the check says the sample is
    survivors.
    """
    scores, prices = recorded
    result = signals.test_signal(scores, prices, baseline_draws=40)
    counts = result.exit_counts()
    assert counts.get("acquired", 0) == 0
    assert counts.get("delisted", 0) == 0
    assert counts["censored"] > 0
    assert any("today's survivors" in c for c in result.checks)

    manifest = json.loads((FIXTURES / "MANIFEST.json").read_text())
    assert "EA, FI, FYBR, IPG and JNPR" in manifest["universe"]


def test_a_delisting_return_changes_nothing_when_there_are_no_delistings(recorded):
    """The sensitivity that cannot be run, which is the point.

    A harness with no failures in its universe is insensitive to what a failure
    is assumed to earn, and a reader who sees the sensitivity come back flat
    should conclude that the sample has no failures rather than that the
    assumption does not matter.
    """
    scores, prices = recorded
    plain = signals.test_signal(scores, prices, baseline_draws=20)
    harsh = signals.test_signal(
        scores, prices, delisting_return="total_loss", baseline_draws=20
    )
    assert plain.n_scored == harsh.n_scored
    assert plain.ic.mean == pytest.approx(harsh.ic.mean)
