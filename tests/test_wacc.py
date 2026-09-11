"""Beta, the Hamada relation, and the cost of debt."""

from __future__ import annotations

import numpy as np
import pytest

from techval.config import Assumptions
from techval.errors import ConfigError, MissingDataError, NotMeaningfulError
from techval.market import PriceSeries
from techval.wacc import (
    BetaEstimate,
    _unlevered_stderr,
    compute_wacc,
    estimate_beta,
    pool_unlevered_betas,
    relever,
    synthetic_credit_spread,
    unlever,
    vasicek_adjust,
)

from conftest import AS_OF


# --------------------------------------------------------------------------- #
# Hamada
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("beta_l", [0.6, 1.0, 1.35, 2.4])
@pytest.mark.parametrize("de", [0.0, 0.15, 0.8, 2.0])
@pytest.mark.parametrize("tax", [0.0, 0.21, 0.24, 0.35])
def test_unlever_relever_round_trip(beta_l, de, tax):
    """Relevering at the same capital structure returns the input.

    Exact to floating point rounding, not bit-for-bit: dividing then multiplying
    by the same factor loses the last bit for a good fraction of doubles. One ulp
    is the correct tolerance, and asserting equality would be a test that fails
    for reasons unrelated to the finance.
    """
    beta_u = unlever(beta_l, de, tax)
    assert relever(beta_u, de, tax) == pytest.approx(beta_l, rel=1e-12)


def test_leverage_raises_equity_beta():
    """More debt, more equity risk, and the tax shield damps how much."""
    beta_u = 1.0
    assert relever(beta_u, 1.0, 0.0) == pytest.approx(2.0)
    assert relever(beta_u, 1.0, 0.24) == pytest.approx(1.76)
    assert relever(beta_u, 0.0, 0.24) == pytest.approx(1.0)


def test_unlevering_a_debt_free_company_changes_nothing():
    assert unlever(1.3, 0.0, 0.24) == pytest.approx(1.3)


# --------------------------------------------------------------------------- #
# beta regression
# --------------------------------------------------------------------------- #


def _series(symbol: str, closes: list[float], start="2024-01-05") -> PriceSeries:
    """Weekly closes, seven days apart, so each lands in its own ISO week."""
    from datetime import date, timedelta

    d0 = date.fromisoformat(start)
    dates = [d0 + timedelta(days=7 * i) for i in range(len(closes))]
    return PriceSeries(symbol, dates, np.array(closes, dtype=float), "test")


def test_beta_of_a_series_that_moves_twice_the_market_is_two():
    rng = np.random.default_rng(0)
    m_ret = rng.normal(0.0, 0.02, 120)
    s_ret = 2.0 * m_ret

    m_px = 100 * np.cumprod(np.concatenate([[1.0], 1 + m_ret]))
    s_px = 100 * np.cumprod(np.concatenate([[1.0], 1 + s_ret]))

    est = estimate_beta(_series("S", list(s_px)), _series("M", list(m_px)))
    assert est.raw_beta == pytest.approx(2.0, rel=1e-6)
    assert est.r_squared == pytest.approx(1.0, rel=1e-6)


def test_blume_shrinks_beta_toward_one():
    rng = np.random.default_rng(1)
    m_ret = rng.normal(0.0, 0.02, 120)
    s_ret = 2.0 * m_ret
    m_px = 100 * np.cumprod(np.concatenate([[1.0], 1 + m_ret]))
    s_px = 100 * np.cumprod(np.concatenate([[1.0], 1 + s_ret]))

    raw = estimate_beta(_series("S", list(s_px)), _series("M", list(m_px)))
    adj = estimate_beta(
        _series("S", list(s_px)), _series("M", list(m_px)), adjustment="blume"
    )
    assert adj.adjusted_beta == pytest.approx(0.67 * raw.raw_beta + 0.33, rel=1e-9)
    assert abs(adj.adjusted_beta - 1.0) < abs(raw.raw_beta - 1.0)


def test_returns_are_paired_by_date_not_by_position():
    """A short peer series must regress on the weeks it shares with the index.

    Zipping by position would pair a recently listed company's first week
    against the index's first week two years earlier. The slope that comes back
    is arithmetic on unrelated numbers and looks entirely plausible.
    """
    rng = np.random.default_rng(2)
    m_ret = rng.normal(0.0, 0.02, 150)
    m_px = list(100 * np.cumprod(np.concatenate([[1.0], 1 + m_ret])))
    market = _series("M", m_px, start="2024-01-05")

    # Same underlying path, but the stock only starts trading 60 weeks in.
    offset = 60
    s_px = [p * 1.5 for p in m_px[offset:]]
    from datetime import date, timedelta

    d0 = date.fromisoformat("2024-01-05") + timedelta(days=7 * offset)
    stock = PriceSeries(
        "S",
        [d0 + timedelta(days=7 * i) for i in range(len(s_px))],
        np.array(s_px),
        "test",
    )

    est = estimate_beta(stock, market, lookback_years=None)
    assert est.n_observations == len(s_px) - 1
    assert est.raw_beta == pytest.approx(1.0, rel=1e-6)


def test_too_few_observations_raises_naming_the_count():
    rng = np.random.default_rng(3)
    px = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 12)))
    with pytest.raises(MissingDataError) as exc:
        estimate_beta(_series("S", px), _series("M", px))
    assert "10" in str(exc.value) or "11" in str(exc.value)


def test_unknown_adjustment_is_rejected():
    rng = np.random.default_rng(4)
    px = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 120)))
    with pytest.raises(ConfigError):
        estimate_beta(_series("S", px), _series("M", px), adjustment="nonsense")


# --------------------------------------------------------------------------- #
# cost of debt
# --------------------------------------------------------------------------- #


def test_credit_spread_widens_as_coverage_falls():
    strong, _ = synthetic_credit_spread(20.0)
    mid, _ = synthetic_credit_spread(5.0)
    weak, _ = synthetic_credit_spread(0.6)
    spreads = [synthetic_credit_spread(c)[1] for c in (20.0, 5.0, 0.6)]
    assert spreads == sorted(spreads)
    assert strong != weak


def test_book_yield_below_the_risk_free_rate_is_rejected(ddog, ddog_bridge, market):
    """A zero-coupon convertible issuer produces an absurd book cost of debt.

    Interest expense over average debt is a backward-looking accounting yield.
    For a company whose only borrowings are 0% coupon converts it lands near
    zero, which is not a rate anyone would lend at, so the engine falls through
    to the synthetic rating and says why.
    """
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    a.cost_of_debt.method = "filings"
    w = compute_wacc(ddog, ddog_bridge, market, a)
    assert w.pretax_cost_of_debt >= w.risk_free_rate
    assert any("risk-free" in n or "synthetic" in n for n in w.notes)


def test_override_is_used_verbatim(ddog, ddog_bridge, market):
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    a.cost_of_debt.method = "override"
    a.cost_of_debt.override_rate = 0.062
    w = compute_wacc(ddog, ddog_bridge, market, a)
    assert w.pretax_cost_of_debt == pytest.approx(0.062)
    assert w.after_tax_cost_of_debt == pytest.approx(0.062 * (1 - 0.24))


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def test_capm_and_weights_reconcile(ddog, ddog_bridge, market, assumptions):
    w = compute_wacc(ddog, ddog_bridge, market, assumptions)
    assert w.cost_of_equity == pytest.approx(
        w.risk_free_rate
        + w.levered_beta * w.erp
        + assumptions.market.size_premium
    )
    assert w.weight_equity + w.weight_debt == pytest.approx(1.0)
    assert w.wacc == pytest.approx(
        w.weight_equity * w.cost_of_equity
        + w.weight_debt * w.after_tax_cost_of_debt
    )
    assert 0.0 < w.wacc < 0.40


def test_net_cash_company_is_flagged(ddog, ddog_bridge, market, assumptions):
    """Its WACC is essentially its cost of equity, and the output should say so."""
    assert ddog_bridge.net_debt < 0
    w = compute_wacc(ddog, ddog_bridge, market, assumptions)
    assert any("net cash" in n.lower() for n in w.notes)


def test_buildup_names_a_source_for_every_input(ddog, ddog_bridge, market, assumptions):
    """The table is the audit trail; a row without a source is not auditable."""
    w = compute_wacc(ddog, ddog_bridge, market, assumptions)
    rows = w.buildup_rows()
    assert len(rows) >= 10
    for r in rows:
        assert r["source"], f"no source for {r['label']}"
    labels = {r["label"] for r in rows}
    assert {"Risk-free rate", "Equity risk premium", "Cost of equity", "WACC"} <= labels


def test_peer_betas_are_unlevered_then_relevered(ddog, ddog_bridge, market, assumptions):
    """Peer betas should reach the result through the median of asset betas."""
    peers = []
    for t in ("CRWD", "MDB", "ZS"):
        peers.append(
            estimate_beta(
                market.prices(t),
                market.prices("SPY"),
                debt_to_equity=0.05,
                tax_rate=0.24,
                lookback_years=2.0,
            )
        )
    w = compute_wacc(ddog, ddog_bridge, market, assumptions, peer_betas=peers)
    median_u = float(np.median([p.unlevered_beta for p in peers]))
    assert w.unlevered_beta == pytest.approx(median_u)

    # The buildup has to say the beta came from peers and name them, or the
    # number is unauditable.
    sources = {r["label"]: r["source"] for r in w.buildup_rows()}
    assert "peer" in sources["Levered beta"].lower()
    assert all(t in sources["Levered beta"] for t in ("CRWD", "MDB", "ZS"))


# --------------------------------------------------------------------------- #
# Vasicek shrinkage
# --------------------------------------------------------------------------- #


def _estimate(ticker: str, beta: float, se: float, de: float = 0.0, tax: float = 0.24):
    """A BetaEstimate with the regression fields set by hand.

    Pooling reads the levered beta only through ``unlevered_beta``, so the two
    are kept consistent here rather than left to drift.
    """
    return BetaEstimate(
        ticker=ticker,
        raw_beta=beta,
        adjusted_beta=beta,
        r_squared=0.2,
        n_observations=104,
        stderr=se,
        unlevered_beta=unlever(beta, de, tax),
        debt_to_equity=de,
        tax_rate=tax,
        note="constructed",
    )


def _three_peers():
    """The hand-calculated set: betas 1.0, 1.4, 1.8 on s.e. 0.1, 0.4, 0.2.

    Rebuilt on every call because pooling writes the shrinkage back onto the
    estimates, and a shared list would carry one test's result into the next.
    """
    return [
        _estimate("A", 1.0, 0.1),
        _estimate("B", 1.4, 0.4),
        _estimate("C", 1.8, 0.2),
    ]


def test_vasicek_formula_matches_a_hand_calculation():
    """Three betas, a cross-sectional variance of 0.16, weights on paper.

    betas 1.0, 1.4, 1.8 have a mean of 1.4 and a sample variance of 0.16 on two
    degrees of freedom. With standard errors 0.1, 0.4 and 0.2:

        w = 0.16/0.17 = 0.9412, 0.16/0.32 = 0.5000, 0.16/0.20 = 0.8000
        beta* = 1.0235, 1.4000, 1.7200

    The middle peer sits at the mean, so a weight of one half moves it nowhere;
    the tight peer keeps 94% of its own estimate, the loose one 80%.
    """
    adjusted, weights = vasicek_adjust([1.0, 1.4, 1.8], [0.1, 0.4, 0.2])
    assert weights == pytest.approx([0.16 / 0.17, 0.5, 0.8])
    assert adjusted == pytest.approx([1.0235294118, 1.4, 1.72])


def test_a_precise_estimate_barely_moves_and_a_noisy_one_moves_a_lot():
    """The whole point of Vasicek over Blume: the data sets the shrinkage.

    Two peers sit the same distance from the mean, 0.5 of beta, and the peer set
    has a cross-sectional standard deviation of 0.5. The peer measured with a
    standard error of 0.02 keeps 99.8% of its reading and moves by 0.001. The
    peer measured with a standard error of 1.0, twice the spread of the group,
    keeps a fifth of its reading and gives up 0.4 of beta.
    """
    betas = [1.0, 2.0, 1.5]
    adjusted, weights = vasicek_adjust(betas, [0.02, 1.0, 0.3])

    assert weights[0] == pytest.approx(0.25 / 0.2504)
    assert abs(adjusted[0] - betas[0]) < 0.001

    assert weights[1] == pytest.approx(0.2)
    assert adjusted[1] == pytest.approx(1.6)
    # It is pulled toward the mean of 1.5, never past it.
    assert betas[1] > adjusted[1] > 1.5


@pytest.mark.parametrize(
    "ses",
    [[0.05, 0.4, 0.9], [1e-6, 1e-6, 1e-6], [12.0, 9.0, 30.0], [0.2, 0.21, 0.19]],
)
def test_weights_are_bounded_and_shrinkage_never_leaves_the_range(ses):
    """A weight is a share of belief, so it lives in [0,1], and so does the blend."""
    betas = [0.7, 1.3, 2.1]
    adjusted, weights = vasicek_adjust(betas, ses)
    assert all(0.0 <= w <= 1.0 for w in weights)
    # A convex combination of a beta and the mean cannot escape the peer range.
    assert all(min(betas) <= b <= max(betas) for b in adjusted)


def test_shrinkage_pulls_the_dispersion_in():
    """Shrunk betas are a tighter set than the ones they came from, same centre."""
    betas = [0.8, 1.1, 1.6, 2.2]
    adjusted, _ = vasicek_adjust(betas, [0.3, 0.4, 0.25, 0.5])
    assert np.std(adjusted) < np.std(betas)


def test_fewer_than_three_peers_is_not_meaningful():
    with pytest.raises(NotMeaningfulError):
        vasicek_adjust([1.1, 1.4], [0.2, 0.3])


def test_a_peer_set_with_no_dispersion_is_not_meaningful():
    """Every peer on the same beta makes the prior a point, and the weight 0/0."""
    with pytest.raises(NotMeaningfulError):
        vasicek_adjust([1.2, 1.2, 1.2], [0.2, 0.3, 0.4])


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_a_missing_standard_error_is_not_meaningful(bad):
    with pytest.raises(NotMeaningfulError):
        vasicek_adjust([1.0, 1.4, 1.8], [0.2, bad, 0.3])


def test_a_standard_error_per_beta_is_required():
    with pytest.raises(ConfigError):
        vasicek_adjust([1.0, 1.4, 1.8], [0.2, 0.3])


# --------------------------------------------------------------------------- #
# unlevering the standard error
# --------------------------------------------------------------------------- #


def test_the_standard_error_is_unlevered_alongside_the_beta():
    """Unlevering divides the slope by 1 + (1-t)D/E, so it divides its s.e. too.

    The factor is read off the balance sheet, not estimated, so it carries no
    sampling error of its own. Leaving the standard error levered would pair a
    peer's asset beta with the uncertainty of its equity beta.
    """
    est = _estimate("LEV", beta=1.8, se=0.30, de=0.8, tax=0.25)
    factor = 1.0 + 0.75 * 0.8  # 1.6
    assert _unlevered_stderr(est) == pytest.approx(0.30 / factor)
    assert est.unlevered_beta == pytest.approx(1.8 / factor)


def test_blume_rescales_the_standard_error_by_the_same_slope():
    """0.67*raw + 0.33 is an affine map, so the s.e. of the result is 0.67 s.e."""
    rng = np.random.default_rng(11)
    m_ret = rng.normal(0.0, 0.02, 120)
    s_ret = 1.4 * m_ret + rng.normal(0.0, 0.02, 120)
    m_px = 100 * np.cumprod(np.concatenate([[1.0], 1 + m_ret]))
    s_px = 100 * np.cumprod(np.concatenate([[1.0], 1 + s_ret]))

    raw = estimate_beta(_series("S", list(s_px)), _series("M", list(m_px)))
    blume = estimate_beta(
        _series("S", list(s_px)), _series("M", list(m_px)), adjustment="blume"
    )
    assert _unlevered_stderr(blume) == pytest.approx(0.67 * _unlevered_stderr(raw))


def test_leverage_does_not_change_a_peers_shrinkage_weight():
    """Two peers with the same business risk and the same regression precision.

    One carries no debt, the other 0.5x D/E at a 24% tax rate. Their levered
    betas and levered standard errors differ by exactly the leverage factor, so
    after unlevering both the asset betas and the asset standard errors match,
    and the shrinkage must treat them identically. If the standard error were
    left levered, the indebted peer would look noisier than its twin and be
    dragged toward the mean for carrying debt, which is the one thing unlevering
    exists to remove.
    """
    factor = 1.0 + 0.76 * 0.5
    unlevered_twin = _estimate("A", beta=1.5, se=0.30, de=0.0)
    levered_twin = _estimate("B", beta=1.5 * factor, se=0.30 * factor, de=0.5)
    third = _estimate("C", beta=2.1, se=0.45, de=0.0)

    pooled, method, _ = pool_unlevered_betas(
        [unlevered_twin, levered_twin, third], "vasicek"
    )
    assert method == "vasicek"
    assert unlevered_twin.shrink_weight == pytest.approx(levered_twin.shrink_weight)
    assert unlevered_twin.shrunk_beta == pytest.approx(levered_twin.shrunk_beta)
    assert 1.5 < pooled < 2.1


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #


def test_the_vasicek_pool_is_the_precision_weighted_mean_of_the_shrunk_betas():
    """Not the median of them, and not the plain average either.

    Weights are 1/se^2 on the unlevered standard errors: 100, 6.25 and 25 for
    the 0.1, 0.4 and 0.2 peers, so the tight peer carries three quarters of the
    answer.
    """
    ests = _three_peers()
    pooled, method, notes = pool_unlevered_betas(ests, "vasicek")
    assert method == "vasicek"

    shrunk = [1.0235294118, 1.4, 1.72]
    precision = [1 / 0.1**2, 1 / 0.4**2, 1 / 0.2**2]
    expected = sum(p * b for p, b in zip(precision, shrunk)) / sum(precision)
    assert pooled == pytest.approx(expected)
    assert pooled == pytest.approx(1.1741176471)
    assert pooled != pytest.approx(float(np.median(shrunk)))


def test_the_median_path_shrinks_nothing_and_says_nothing():
    ests = _three_peers()
    pooled, method, notes = pool_unlevered_betas(ests, "median")
    assert (pooled, method, notes) == (pytest.approx(1.4), "median", [])
    assert all(e.shrunk_beta is None and e.shrink_weight is None for e in ests)


def test_a_repooled_estimate_does_not_keep_a_stale_shrinkage():
    """Pooling the same estimates under the median must clear what Vasicek wrote."""
    ests = _three_peers()
    pool_unlevered_betas(ests, "vasicek")
    assert all(e.shrink_weight is not None for e in ests)
    pool_unlevered_betas(ests, "median")
    assert all(e.shrunk_beta is None and e.shrink_weight is None for e in ests)


@pytest.mark.parametrize(
    "ests",
    [
        [_estimate("A", 1.2, 0.2), _estimate("B", 1.6, 0.3)],
        [_estimate("A", 1.3, 0.2), _estimate("B", 1.3, 0.3), _estimate("C", 1.3, 0.4)],
        [_estimate("A", 1.2, 0.2), _estimate("B", 1.6, 0.0), _estimate("C", 1.9, 0.3)],
    ],
    ids=["two peers", "no dispersion", "zero standard error"],
)
def test_a_degenerate_peer_set_falls_back_to_the_median_and_says_so(ests):
    """Too few peers, no dispersion, or a standard error of zero.

    Each of the three is a division that should not be attempted. The fallback
    is the median, which needs none of them, and the note has to say the
    configured method was not the method used.
    """
    pooled, method, notes = pool_unlevered_betas(ests, "vasicek")
    assert method == "median"
    assert pooled == pytest.approx(float(np.median([e.unlevered_beta for e in ests])))
    assert len(notes) == 1
    assert "vasicek" in notes[0].lower() and "median" in notes[0].lower()
    assert all(e.shrunk_beta is None and e.shrink_weight is None for e in ests)


def test_an_unknown_pooling_method_is_rejected():
    ests = _three_peers()
    with pytest.raises(ConfigError):
        pool_unlevered_betas(ests, "average")


def test_pooling_nothing_raises():
    with pytest.raises(MissingDataError):
        pool_unlevered_betas([], "vasicek")


# --------------------------------------------------------------------------- #
# the fixture peer set, end to end
# --------------------------------------------------------------------------- #


def _fixture_peers(market, tickers=("CRWD", "MDB", "ZS", "NET", "SNOW", "HUBS")):
    return [
        estimate_beta(
            market.prices(t),
            market.prices("SPY"),
            debt_to_equity=0.05,
            tax_rate=0.24,
            lookback_years=2.0,
        )
        for t in tickers
    ]


def test_vasicek_versus_median_on_the_fixture_peers(
    ddog, ddog_bridge, market, assumptions
):
    """The comparison on six software peers vs SPY, 104 weekly returns each.

    Each peer is unlevered at 0.05x D/E and a 24% tax rate, so the leverage
    factor is 1.038 and the asset standard errors are the levered ones divided
    by it. DDOG is relevered at its own 0.01x D/E.

        peer   raw beta   s.e.    R2     asset beta   weight   shrunk
        CRWD     1.628    0.285   0.24      1.569      0.518    1.500
        MDB      1.987    0.393   0.20      1.914      0.360    1.602
        ZS       1.155    0.294   0.13      1.113      0.502    1.269
        NET      1.407    0.332   0.15      1.356      0.441    1.395
        SNOW     1.435    0.373   0.13      1.382      0.386    1.409
        HUBS     1.270    0.352   0.11      1.224      0.413    1.343

        cross-sectional mean 1.426, standard deviation 0.284

        pooled asset beta      median 1.3690     Vasicek 1.4108
        relevered at 0.01x     levered 1.3814    levered 1.4236
        cost of equity         11.74%            11.95%
        WACC                   11.70%            11.91%

    Twenty-one basis points on the cost of equity, and the reason is MDB: the
    widest standard error in the set at 0.379 unlevered, so it keeps the least
    of its own estimate, 0.36, and comes back from 1.914 to 1.602. The median
    could not see that, because the median only reads the middle of the sorted
    list. Note also which direction the move goes. Shrinkage pulled the highest
    beta down hard, and the pooled figure still came out above the median,
    because the mean of this set sits above its middle and the median was
    ignoring three quarters of the evidence for where the centre is.
    """
    a = assumptions
    a.market.peer_beta_method = "median"
    w_med = compute_wacc(
        ddog, ddog_bridge, market, a, peer_betas=_fixture_peers(market)
    )

    a.market.peer_beta_method = "vasicek"
    peers = _fixture_peers(market)
    w_vas = compute_wacc(ddog, ddog_bridge, market, a, peer_betas=peers)

    assert w_med.unlevered_beta == pytest.approx(1.368973, abs=5e-6)
    assert w_vas.unlevered_beta == pytest.approx(1.410776, abs=5e-6)
    assert w_med.levered_beta == pytest.approx(1.381378, abs=5e-6)
    assert w_vas.levered_beta == pytest.approx(1.423560, abs=5e-6)

    # The cost of equity is CAPM on those betas, at the pinned 4.83% risk-free
    # rate and the configured ERP, so it is checked against the identity rather
    # than against a literal that moves when an assumption default moves.
    for w in (w_med, w_vas):
        assert w.cost_of_equity == pytest.approx(
            w.risk_free_rate + w.levered_beta * w.erp + a.market.size_premium
        )
    gap = w_vas.cost_of_equity - w_med.cost_of_equity
    assert gap == pytest.approx(0.0021, abs=1e-4)

    weights = {e.ticker: e.shrink_weight for e in peers}
    assert weights["MDB"] == pytest.approx(0.3603, abs=5e-4)
    assert weights["CRWD"] == pytest.approx(0.5184, abs=5e-4)
    assert min(weights.values()) == weights["MDB"]
    shrunk = {e.ticker: e.shrunk_beta for e in peers}
    assert shrunk["MDB"] == pytest.approx(1.6021, abs=5e-4)


def test_median_pooling_is_unchanged_by_the_shrinkage_work(
    ddog, ddog_bridge, market, assumptions
):
    """The default path has to produce exactly the answer it produced before.

    CRWD, MDB and ZS unlevered at 0.05x D/E: asset betas 1.5688, 1.9143 and
    1.1131, median 1.5688, which is CRWD's. Nothing about that is allowed to
    move because a second pooling method exists.
    """
    assert assumptions.market.peer_beta_method == "median"
    peers = _fixture_peers(market, ("CRWD", "MDB", "ZS"))
    w = compute_wacc(ddog, ddog_bridge, market, assumptions, peer_betas=peers)

    median_u = float(np.median([p.unlevered_beta for p in peers]))
    assert w.unlevered_beta == median_u
    assert w.unlevered_beta == pytest.approx(1.568819, abs=5e-6)
    de = {r["label"]: r["value"] for r in w.buildup_rows()}["Debt / equity"]
    assert w.levered_beta == pytest.approx(relever(median_u, de, w.tax_rate))
    assert all(p.shrunk_beta is None for p in peers)

    sources = {r["label"]: r["source"] for r in w.buildup_rows()}
    assert sources["Levered beta"].startswith("median of 3 peer unlevered betas")
    assert not any("vasicek" in n.lower() for n in w.notes)


def test_the_buildup_names_the_pooling_method(ddog, ddog_bridge, market, assumptions):
    """A pooled beta that does not say how it was pooled is not auditable."""
    assumptions.market.peer_beta_method = "vasicek"
    w = compute_wacc(
        ddog, ddog_bridge, market, assumptions, peer_betas=_fixture_peers(market)
    )
    sources = {r["label"]: r["source"] for r in w.buildup_rows()}
    assert "Vasicek" in sources["Levered beta"]
    assert "Vasicek" in sources["Unlevered beta"]


def test_the_notes_report_how_far_the_prior_moved_each_peer(
    ddog, ddog_bridge, market, assumptions
):
    """Mean, dispersion and the range of weights, or the reader cannot judge it."""
    assumptions.market.peer_beta_method = "vasicek"
    w = compute_wacc(
        ddog, ddog_bridge, market, assumptions, peer_betas=_fixture_peers(market)
    )
    note = next(n for n in w.notes if "Vasicek shrinkage pooled" in n)
    assert "1.426" in note  # cross-sectional mean
    assert "0.284" in note  # cross-sectional standard deviation
    assert "0.36 to 0.52" in note  # the weight range across the six peers


def test_blume_and_vasicek_together_are_flagged_as_double_shrinkage(
    ddog, ddog_bridge, market, assumptions
):
    """Two corrections answering the same question, applied one on top of the other.

    Neither is wrong on its own. Stacking them pulls dispersion out of the peer
    set twice, once by a fixed rule and once by the data, and the output has to
    say so rather than leave the reader to notice.
    """
    assumptions.market.peer_beta_method = "vasicek"
    assumptions.market.beta_adjustment = "blume"
    peers = [
        estimate_beta(
            market.prices(t),
            market.prices("SPY"),
            adjustment="blume",
            debt_to_equity=0.05,
            tax_rate=0.24,
            lookback_years=2.0,
        )
        for t in ("CRWD", "MDB", "ZS", "NET")
    ]
    w = compute_wacc(ddog, ddog_bridge, market, assumptions, peer_betas=peers)
    assert any("shrunk twice" in n for n in w.notes)
