"""The joint draw, the truncation, the tornado and the vectorised valuation.

Every number here comes from the offline fixtures, and the simulation is seeded
from the assumptions, so the assertions below are reproducible rather than
merely usually true. The reconciliation tests are the load-bearing ones: a
vectorised DCF that does not equal the scalar DCF is a second model nobody
asked for.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from techval.dcf import run_dcf
from techval.errors import ConfigError
from techval.ev_bridge import build_ev_bridge
from techval.simulation import (
    DRIVERS,
    DRIVER_CORRELATIONS,
    Percentiles,
    correlation_matrix,
    draw_shocks,
    run_simulation,
    tornado,
)
from techval.wacc import compute_wacc

# Pinned so the reported figures do not move when the CAPM build changes. Every
# test that reads a level rather than a relationship sets it.
PINNED_WACC = 0.105

# Enough draws to make a percentile stable without making the suite slow. The
# performance test is the one that uses the full ten thousand.
FAST_DRAWS = 2_000


@pytest.fixture
def wacc_result(ddog, ddog_bridge, market, assumptions):
    return compute_wacc(ddog, ddog_bridge, market, assumptions)


@pytest.fixture
def sim(assumptions):
    """Default assumptions with a pinned WACC and a quick sample."""
    assumptions.dcf.wacc_override = PINNED_WACC
    assumptions.simulation.draws = FAST_DRAWS
    return assumptions


def _slopes(bars):
    """Price sensitivity per unit of each driver, read off the tornado.

    A first-order approximation of the valuation around the central case, which
    is all that is needed to predict which way a correlation moves the spread.
    """
    return {
        bar.driver: (bar.high_price - bar.low_price)
        / (bar.high_value - bar.low_value)
        for bar in bars
    }


def _predicted_sd(bars, assumptions, matrix):
    """Delta-method standard deviation of price under a correlation matrix.

        sd^2 = a' C a,  a_i = (dPrice/dDriver_i) * sd_i

    Computed independently of the simulation so the two can be compared: if the
    sampled spread and the analytic one disagree on direction, one of them is
    wrong about what the correlation matrix does.
    """
    cfg = assumptions.simulation
    sds = {
        "revenue_growth": cfg.revenue_growth_sd,
        "terminal_margin": cfg.terminal_margin_sd,
        "wacc": cfg.wacc_sd,
        "terminal_growth": cfg.terminal_growth_sd,
    }
    slopes = _slopes(bars)
    vector = np.array([slopes[d] * sds[d] for d in DRIVERS])
    return float(np.sqrt(vector @ matrix @ vector))


# --------------------------------------------------------------------------- #
# the correlation matrix
# --------------------------------------------------------------------------- #


def test_the_shipped_matrix_is_a_real_correlation_matrix():
    matrix = correlation_matrix()

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.linalg.eigvalsh(matrix).min() > 0.0


def test_the_shipped_matrix_carries_the_signs_the_docstring_argues():
    matrix = correlation_matrix()
    i = {name: k for k, name in enumerate(DRIVERS)}

    # Operating leverage ties growth to margin; the market discounts a better
    # business at a lower rate, so both of those pair negatively with the WACC.
    assert matrix[i["revenue_growth"], i["terminal_margin"]] > 0
    assert matrix[i["revenue_growth"], i["wacc"]] < 0
    assert matrix[i["terminal_margin"], i["wacc"]] < 0
    assert abs(matrix[i["terminal_growth"], i["wacc"]]) <= 0.1
    assert abs(matrix[i["terminal_growth"], i["revenue_growth"]]) <= 0.1


def test_a_matrix_that_is_not_psd_is_rejected_and_the_pair_is_named():
    """A violation is always a triple, so the message names where it loads.

    Two correlations of opposite sign and high magnitude fix a window the third
    has to sit inside. Here growth is almost perfectly with margin and almost
    perfectly against the WACC, which forces margin and WACC strongly negative;
    entering them positive describes no joint distribution. The message cannot
    name a single guilty pair, because none of the three is guilty on its own,
    so it names the two the negative-variance direction loads on hardest.
    """
    broken = {
        ("revenue_growth", "terminal_margin"): 0.90,
        ("revenue_growth", "wacc"): -0.90,
        ("terminal_margin", "wacc"): 0.80,
    }

    with pytest.raises(ConfigError) as excinfo:
        correlation_matrix(broken)

    message = str(excinfo.value)
    assert "positive semi-definite" in message
    named = [d for d in DRIVERS if d in message]
    assert len(named) >= 2
    assert "terminal_growth" not in named[:2] or len(named) > 2


def test_the_non_psd_message_quotes_the_negative_variance():
    broken = {
        ("revenue_growth", "terminal_margin"): 0.95,
        ("revenue_growth", "wacc"): -0.95,
        ("terminal_margin", "wacc"): 0.95,
    }

    with pytest.raises(ConfigError) as excinfo:
        correlation_matrix(broken)

    assert "variance of -" in str(excinfo.value)


def test_a_correlation_outside_the_unit_interval_is_rejected():
    with pytest.raises(ConfigError, match="outside"):
        correlation_matrix({("revenue_growth", "wacc"): -1.4})


def test_a_correlation_naming_a_driver_that_is_not_drawn_is_rejected():
    with pytest.raises(ConfigError, match="does not draw"):
        correlation_matrix({("revenue_growth", "exit_multiple"): 0.2})


def test_a_driver_cannot_be_correlated_with_itself():
    with pytest.raises(ConfigError, match="itself"):
        correlation_matrix({("wacc", "wacc"): 0.5})


def test_a_singular_but_valid_matrix_still_produces_draws(sim):
    """Perfect correlation is PSD, has no Cholesky factor, and must still draw.

    The eigenvalue square root is the fallback, and the test that it worked is
    that the two drivers come back moving in lockstep rather than that the
    factorisation returned something.
    """
    shocks = draw_shocks(sim, {("revenue_growth", "terminal_margin"): 1.0})

    assert np.isfinite(shocks).all()
    assert np.corrcoef(shocks[:, 0], shocks[:, 1])[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_the_draws_reproduce_the_requested_correlations(assumptions):
    assumptions.simulation.draws = 20_000
    shocks = draw_shocks(assumptions)
    empirical = np.corrcoef(shocks, rowvar=False)
    requested = correlation_matrix()

    assert np.abs(empirical - requested).max() < 0.03


def test_a_negative_standard_deviation_is_rejected(sim):
    sim.simulation.wacc_sd = -0.01

    with pytest.raises(ConfigError, match="square root"):
        draw_shocks(sim)


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #


def test_the_same_seed_reproduces_the_distribution(ddog, ddog_bridge, wacc_result, sim):
    first = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    second = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    assert first.per_share.p50 == second.per_share.p50
    assert first.per_share.p5 == second.per_share.p5
    assert first.per_share.p95 == second.per_share.p95
    assert first.per_share.sd == second.per_share.sd
    assert first.prob_above_price == second.prob_above_price
    assert first.failed_draws == second.failed_draws


def test_a_different_seed_moves_the_distribution(ddog, ddog_bridge, wacc_result, sim):
    first = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    sim.simulation.seed = sim.simulation.seed + 1
    second = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    assert first.per_share.p50 != second.per_share.p50
    # Same distribution, different sample: the medians should still be close.
    assert first.per_share.p50 == pytest.approx(second.per_share.p50, rel=0.05)


def test_the_global_random_state_never_reaches_the_draws(sim):
    """Seeded from the assumptions through default_rng, never from numpy.random.

    A model that moves when someone else's code draws a random number is not
    reproducible, and the failure is silent: the answer simply differs between
    two runs of the same file.
    """
    before = draw_shocks(sim)

    np.random.seed(12345)
    np.random.random(100)
    after = draw_shocks(sim)

    assert np.array_equal(before, after)


# --------------------------------------------------------------------------- #
# the vectorised path against run_dcf
# --------------------------------------------------------------------------- #


def _configure(assumptions, case):
    cfg = assumptions.dcf
    if case == "gordon":
        pass
    elif case == "value_driver":
        cfg.terminal.method = "value_driver"
    elif case == "value_driver_roic":
        cfg.terminal.method = "value_driver"
        cfg.terminal.terminal_roic = 0.18
    elif case == "value_driver_with_sbc_addback":
        # The steady state carries the SBC addback the projected flows carry, so
        # the two paths have to switch convention at year N+1 together.
        cfg.terminal.method = "value_driver"
        cfg.sbc_treatment = "addback"
    elif case == "sbc_addback_diluted":
        cfg.sbc_treatment = "addback"
    elif case == "sbc_addback_undiluted":
        cfg.sbc_treatment = "addback"
        cfg.sbc_dilution = False
    elif case == "nol":
        cfg.nol.track = True
        cfg.nol.opening_balance = 2_000.0
    elif case == "end_year_discounting":
        cfg.mid_year_convention = False
    elif case == "one_year":
        cfg.projection_years = 1
    elif case == "twelve_years":
        cfg.projection_years = 12
    elif case == "explicit_margin_start":
        cfg.ebit_margin_start = 0.05
    elif case == "effective_tax":
        assumptions.tax.use_effective_rate = True
    else:  # pragma: no cover - a typo in the parametrisation, not a code path
        raise AssertionError(f"unknown case {case}")
    return assumptions


@pytest.mark.parametrize(
    "case",
    [
        "gordon",
        "value_driver",
        "value_driver_roic",
        "value_driver_with_sbc_addback",
        "sbc_addback_diluted",
        "sbc_addback_undiluted",
        "nol",
        "end_year_discounting",
        "one_year",
        "twelve_years",
        "explicit_margin_start",
        "effective_tax",
    ],
)
def test_the_vectorised_central_draw_reproduces_run_dcf(
    ddog, ddog_bridge, wacc_result, sim, case
):
    """The whole case for vectorising rests on this.

    Ten thousand valuations computed as array arithmetic are worth nothing
    unless the zero-shock draw is the same valuation the rest of the engine
    reports. Every terminal method, every SBC treatment, both discounting
    conventions and both ends of the horizon are checked, because a projection
    rewritten as numpy diverges from the scalar one at exactly the branches the
    author was not thinking about.
    """
    _configure(sim, case)

    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    scalar = run_dcf(ddog, ddog_bridge, wacc_result, sim)

    assert result.central_per_share == pytest.approx(scalar.per_share, rel=1e-12)


def test_the_reconciliation_is_reported_as_a_check(ddog, ddog_bridge, wacc_result, sim):
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    assert any("reproduces run_dcf" in line for line in result.checks)


def test_the_distribution_is_centred_on_the_scalar_valuation(
    ddog, ddog_bridge, wacc_result, sim
):
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    scalar = run_dcf(ddog, ddog_bridge, wacc_result, sim)

    # Not equal: the Gordon denominator is convex in the discount rate, so the
    # distribution is right-skewed and its median sits a little below the point
    # estimate. Wildly apart would mean the shocks are not centred on zero.
    assert result.per_share.p50 == pytest.approx(scalar.per_share, rel=0.05)
    assert result.per_share.mean > result.per_share.p50


def test_enterprise_value_and_per_share_percentiles_are_the_same_draws(
    ddog, ddog_bridge, wacc_result, sim
):
    """The bridge from EV to equity is linear, so the two tables must line up.

    Without SBC dilution the share count is fixed, so per-share value is an
    affine function of enterprise value and the median of one has to be the
    median of the other put through the bridge. If they disagree, the two
    percentile tables were struck on different draws.
    """
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    scalar = run_dcf(ddog, ddog_bridge, wacc_result, sim)
    offset = scalar.equity_value - scalar.enterprise_value

    implied = (result.enterprise_value.p50 + offset) / scalar.shares_for_value
    assert implied == pytest.approx(result.per_share.p50, rel=1e-12)


# --------------------------------------------------------------------------- #
# truncation
# --------------------------------------------------------------------------- #


def _tight_boundary(assumptions):
    """A central case close enough to the Gordon boundary that draws fall off it.

    The gap between the WACC and terminal growth is 1.8 points against a
    combined standard deviation of 1.4, so roughly a tenth of the sample lands
    at or above the discount rate. The band remains wide enough that no single
    driver crosses the boundary on its own, which keeps the tornado computable.
    """
    assumptions.dcf.wacc_override = 0.046
    assumptions.dcf.terminal_growth = 0.028
    assumptions.simulation.wacc_sd = 0.010
    assumptions.simulation.terminal_growth_sd = 0.010
    return assumptions


def test_draws_where_growth_reaches_the_discount_rate_are_counted(
    ddog, ddog_bridge, wacc_result, sim
):
    """Counted against the raw shocks, not against the engine's own bookkeeping."""
    _tight_boundary(sim)
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    shocks = draw_shocks(sim)
    breached = np.count_nonzero(
        sim.dcf.terminal_growth + shocks[:, 3] >= sim.dcf.wacc_override + shocks[:, 2]
    )

    assert result.failed_draws == breached
    assert result.kept_draws == sim.simulation.draws - breached
    assert 0.05 < result.failed_draws / result.draws < 0.20


def test_a_heavy_rejection_rate_is_flagged(ddog, ddog_bridge, wacc_result, sim):
    _tight_boundary(sim)
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    assert any(line.startswith("FLAG:") and "rejected" in line for line in result.checks)


def test_nothing_is_rejected_when_the_boundary_is_far_away(
    ddog, ddog_bridge, wacc_result, sim
):
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    assert result.failed_draws == 0
    assert any("No draw was rejected" in line for line in result.checks)


def test_rejected_draws_are_discarded_rather_than_clipped(
    ddog, ddog_bridge, wacc_result, sim
):
    """The signature of a clipped sample is a mean above the 95th percentile.

    Pushing a breaching draw back to terminal growth a hair below the WACC does
    not throw it away, it parks it at the point where the perpetuity is largest.
    A tenth of this sample breaches, so a tenth of it would sit at that single
    enormous value, and the mean would be dragged clear of the 95th percentile
    by a mass of draws that all say the same impossible thing. Truncation leaves
    a right-skewed but finite distribution whose mean stays inside the body of
    it, which is what is asserted here. The tail is still long: removing the
    undefined region does not cap the surviving draws, and the near-boundary
    ones are legitimately worth many times the central case.
    """
    _tight_boundary(sim)
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    assert result.failed_draws > 0
    assert np.isfinite(
        [result.per_share.p5, result.per_share.p95, result.per_share.mean]
    ).all()
    assert result.per_share.p50 < result.per_share.mean < result.per_share.p95


# --------------------------------------------------------------------------- #
# the distribution and the probabilities
# --------------------------------------------------------------------------- #


def test_percentiles_are_monotone(ddog, ddog_bridge, wacc_result, sim):
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    for table in (result.per_share, result.enterprise_value):
        assert table.p5 <= table.p25 <= table.p50 <= table.p75 <= table.p95
        assert table.sd > 0


def test_percentiles_are_monotone_under_truncation(
    ddog, ddog_bridge, wacc_result, sim
):
    _tight_boundary(sim)
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    table = result.per_share
    assert table.p5 <= table.p25 <= table.p50 <= table.p75 <= table.p95


def test_the_probabilities_are_nested_and_bounded(ddog, ddog_bridge, wacc_result, sim):
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    assert 0.0 <= result.prob_upside_20pct <= result.prob_above_price <= 1.0


def test_a_model_far_below_the_price_never_clears_it(
    ddog, ddog_bridge, wacc_result, sim
):
    """Datadog at these assumptions is worth a fraction of what it trades at.

    The point of the assertion is not the level. It is that the probability
    statement does not soften a disagreement of that size: no combination of
    four drivers inside their assumed spreads reaches the quoted price, so the
    argument is about the central case and the check says so rather than
    reporting a small non-zero chance.
    """
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    assert result.per_share.p95 < ddog_bridge.price
    assert result.prob_above_price == 0.0
    assert any("95th percentile" in line and line.startswith("FLAG:") for line in result.checks)


def test_the_probability_is_read_off_the_quoted_price(
    ddog, wacc_result, sim, assumptions
):
    """Price the bridge at the model's own answer and the coin is fair.

    Under these assumptions the enterprise-to-equity walk does not depend on
    the share price, so repricing the bridge moves the comparison point without
    moving the valuation, which isolates the probability statement.
    """
    result = run_simulation(
        ddog, build_ev_bridge(ddog, 36.65, sim), wacc_result, sim
    )

    assert 0.30 < result.prob_above_price < 0.70
    assert result.prob_upside_20pct < result.prob_above_price


# --------------------------------------------------------------------------- #
# what the correlation does to the spread
# --------------------------------------------------------------------------- #


def test_correlated_draws_are_wider_than_independent_ones(
    ddog, ddog_bridge, wacc_result, sim
):
    """The opposite of the usual claim, and the signs are why.

    Value rises with growth, rises with the terminal margin and falls with the
    WACC, so a positive growth-margin correlation and negative correlations of
    both against the WACC all push the same way: the good draws arrive
    together. Independence spreads mass over the offsetting middle, which is
    where the point estimate already is, so the independent sample is the
    NARROWER one. The delta-method prediction is computed from the tornado
    slopes, entirely outside the sampler, so the two would have to be wrong in
    the same direction to agree by accident.
    """
    correlated = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    independent = run_simulation(ddog, ddog_bridge, wacc_result, sim, independent=True)

    predicted_correlated = _predicted_sd(correlated.tornado, sim, correlation_matrix())
    predicted_independent = _predicted_sd(
        correlated.tornado, sim, np.eye(len(DRIVERS))
    )

    assert predicted_correlated > predicted_independent
    assert correlated.per_share.sd > independent.per_share.sd
    assert correlated.per_share.sd / independent.per_share.sd == pytest.approx(
        predicted_correlated / predicted_independent, rel=0.05
    )


def test_flipping_the_wacc_pairings_narrows_the_spread(
    ddog, ddog_bridge, wacc_result, sim
):
    """The riskier-company-higher-beta view, which is the matrix that narrows.

    Entered as a test rather than as the default because it is a different
    economic claim: that a faster-growing company carries its own higher
    discount rate, so growth and the discount rate offset inside the valuation.
    The module reports the spread the stated signs produce, and this shows the
    stated signs are the reason it is wide.
    """
    offsetting = dict(DRIVER_CORRELATIONS)
    offsetting[("revenue_growth", "wacc")] = +0.25
    offsetting[("terminal_margin", "wacc")] = +0.20

    base = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    flipped = run_simulation(
        ddog, ddog_bridge, wacc_result, sim, correlations=offsetting
    )

    assert flipped.per_share.sd < base.per_share.sd


def test_independent_draws_say_so_in_the_notes(ddog, ddog_bridge, wacc_result, sim):
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim, independent=True)

    assert any("INDEPENDENTLY" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# the tornado
# --------------------------------------------------------------------------- #


def test_the_tornado_is_sorted_by_swing_descending(
    ddog, ddog_bridge, wacc_result, sim
):
    bars = tornado(ddog, ddog_bridge, wacc_result, sim)

    swings = [bar.swing for bar in bars]
    assert swings == sorted(swings, reverse=True)
    assert {bar.driver for bar in bars} == set(DRIVERS)


def test_the_tornado_band_is_the_five_and_ninety_five_percent_points(
    ddog, ddog_bridge, wacc_result, sim
):
    sim.simulation.wacc_sd = 0.010
    bars = {bar.driver: bar for bar in tornado(ddog, ddog_bridge, wacc_result, sim)}

    wacc_bar = bars["wacc"]
    assert wacc_bar.low_value == pytest.approx(PINNED_WACC - 1.6448536 * 0.010)
    assert wacc_bar.high_value == pytest.approx(PINNED_WACC + 1.6448536 * 0.010)
    assert wacc_bar.high_value - wacc_bar.low_value == pytest.approx(2 * 1.6448536 * 0.010)


def test_value_falls_as_the_discount_rate_rises_and_rises_with_the_others(
    ddog, ddog_bridge, wacc_result, sim
):
    bars = {bar.driver: bar for bar in tornado(ddog, ddog_bridge, wacc_result, sim)}

    assert bars["wacc"].low_price > bars["wacc"].high_price
    assert bars["revenue_growth"].high_price > bars["revenue_growth"].low_price
    assert bars["terminal_margin"].high_price > bars["terminal_margin"].low_price
    assert bars["terminal_growth"].high_price > bars["terminal_growth"].low_price
    for bar in bars.values():
        assert bar.swing == pytest.approx(abs(bar.high_price - bar.low_price))
        assert bar.swing > 0


def test_the_tornado_centres_on_the_scalar_valuation(
    ddog, ddog_bridge, wacc_result, sim
):
    """Each bar straddles the point estimate, which is what makes it a sensitivity."""
    scalar = run_dcf(ddog, ddog_bridge, wacc_result, sim)

    for bar in tornado(ddog, ddog_bridge, wacc_result, sim):
        low, high = sorted((bar.low_price, bar.high_price))
        assert low < scalar.per_share < high


def test_a_driver_pinned_at_zero_spread_has_no_swing(
    ddog, ddog_bridge, wacc_result, sim
):
    sim.simulation.terminal_growth_sd = 0.0
    bars = tornado(ddog, ddog_bridge, wacc_result, sim)

    assert bars[-1].driver == "terminal_growth"
    assert bars[-1].swing == pytest.approx(0.0)


def test_the_terminal_margin_leads_the_tornado_for_this_filer(
    ddog, ddog_bridge, wacc_result, sim
):
    """Datadog's GAAP EBIT margin is near zero, so the whole valuation is the fade.

    The model is an argument that a company running at breakeven today reaches a
    twenty percent operating margin. Four points either side of that assumption
    moves the answer more than a hundred and sixty basis points on the discount
    rate does, and that is the assumption to bring to the meeting.
    """
    bars = tornado(ddog, ddog_bridge, wacc_result, sim)

    assert bars[0].driver == "terminal_margin"
    assert bars[0].swing > bars[1].swing


def test_a_band_that_crosses_the_gordon_boundary_raises(
    ddog, ddog_bridge, wacc_result, sim
):
    """A hole in the tornado is a finding, not a blank cell.

    Terminal growth whose own 90% band reaches the discount rate says the spread
    and the central case do not hang together, and the right response is to say
    so rather than to print three bars and leave the fourth out.
    """
    sim.dcf.wacc_override = 0.046
    sim.dcf.terminal_growth = 0.040
    sim.simulation.terminal_growth_sd = 0.010
    # Held tight so the discount rate's own bar stays clear of the boundary and
    # terminal growth is unambiguously the driver that breaches it.
    sim.simulation.wacc_sd = 0.001

    with pytest.raises(ConfigError, match="95th percentile of dcf.terminal_growth"):
        tornado(ddog, ddog_bridge, wacc_result, sim)


# --------------------------------------------------------------------------- #
# the value-driver terminal method
# --------------------------------------------------------------------------- #


def test_the_value_driver_method_is_simulated_on_its_own_construction(
    ddog, ddog_bridge, wacc_result, sim
):
    sim.dcf.terminal.method = "value_driver"
    sim.dcf.terminal.terminal_roic = 0.18

    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    scalar = run_dcf(ddog, ddog_bridge, wacc_result, sim)

    assert result.central_per_share == pytest.approx(
        scalar.per_share_value_driver, rel=1e-12
    )
    assert result.central_per_share != pytest.approx(scalar.per_share_gordon)


def test_a_value_driver_that_cannot_be_formed_refuses_to_fall_back(
    ddog, ddog_bridge, wacc_result, sim
):
    """run_dcf falls back to Gordon here. A distribution must not.

    Half a sample capitalised as a value driver and half as a Gordon perpetuity
    is two histograms printed on top of each other, and the percentiles of that
    mixture mean nothing. The scalar model reports the fallback and says so in a
    note, which is fine for one number; the simulation stops.
    """
    sim.dcf.terminal.method = "value_driver"
    sim.dcf.ebit_margin_terminal = -0.05

    scalar = run_dcf(ddog, ddog_bridge, wacc_result, sim)
    assert scalar.headline_method == "gordon"

    with pytest.raises(ConfigError, match="cannot be formed"):
        run_simulation(ddog, ddog_bridge, wacc_result, sim)


# --------------------------------------------------------------------------- #
# the result object
# --------------------------------------------------------------------------- #


def test_the_percentile_frame_carries_both_columns(
    ddog, ddog_bridge, wacc_result, sim
):
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    frame = result.to_frame()

    assert list(frame.columns) == ["Per share", "Enterprise value"]
    assert frame.loc["Median", "Per share"] == result.per_share.p50
    assert frame.loc["95th percentile", "Enterprise value"] == result.enterprise_value.p95


def test_the_tornado_frame_is_one_row_per_driver_widest_first(
    ddog, ddog_bridge, wacc_result, sim
):
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    frame = result.tornado_frame()

    assert len(frame) == len(DRIVERS)
    assert frame["Swing"].is_monotonic_decreasing
    assert frame.iloc[0]["Assumption"] == "dcf.ebit_margin_terminal"


def test_the_result_reports_the_seed_and_the_draw_count(
    ddog, ddog_bridge, wacc_result, sim
):
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    assert result.draws == FAST_DRAWS
    assert result.seed == sim.simulation.seed
    assert result.kept_draws + result.failed_draws == result.draws
    assert result.current_price == ddog_bridge.price


def test_the_notes_say_the_probabilities_are_not_about_the_world(
    ddog, ddog_bridge, wacc_result, sim
):
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)

    assert any("not about the world" in note for note in result.notes)
    assert any("only year-one revenue growth is shocked" in note.lower() for note in result.notes)


def test_percentiles_of_a_known_sample():
    values = np.arange(1.0, 101.0)
    table = Percentiles.of(values)

    assert table.p50 == pytest.approx(50.5)
    assert table.p5 == pytest.approx(5.95)
    assert table.p95 == pytest.approx(95.05)
    assert table.mean == pytest.approx(50.5)


# --------------------------------------------------------------------------- #
# performance
# --------------------------------------------------------------------------- #


def test_ten_thousand_draws_complete_quickly(ddog, ddog_bridge, wacc_result, sim):
    """The reason the projection is array arithmetic rather than a loop of run_dcf."""
    sim.simulation.draws = 10_000

    started = time.perf_counter()
    result = run_simulation(ddog, ddog_bridge, wacc_result, sim)
    elapsed = time.perf_counter() - started

    assert result.kept_draws == 10_000
    assert elapsed < 5.0


def test_simulation_uses_the_same_share_count_as_the_dcf(
    ddog, ddog_bridge, market, assumptions
):
    """A treasury stock count must reach the vectorised path too.

    The simulation reconciles its vectorised central draw against run_dcf on
    every call. Starting from diluted WASO here while run_dcf started from a
    point-in-time count put a 2.9% wedge between them, and the reconciliation
    correctly refused to run rather than publishing a distribution around a
    per-share figure the DCF itself would not produce. Both now divide by
    Financials.shares_for_valuation.
    """
    from techval.wacc import compute_wacc
    from techval.simulation import run_simulation

    w = compute_wacc(ddog, ddog_bridge, market, assumptions)
    assumptions.simulation.draws = 500

    ddog.valuation_shares = ddog.diluted_shares * 1.03
    sim = run_simulation(ddog, ddog_bridge, w, assumptions)

    # The reconciliation inside run_simulation is the real assertion; reaching
    # here at all means it held. This pins the resulting level.
    assert sim.central_per_share > 0
    assert sim.per_share.p5 < sim.per_share.p50 < sim.per_share.p95
