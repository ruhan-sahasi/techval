"""Beta, the Hamada relation, and the cost of debt."""

from __future__ import annotations

import numpy as np
import pytest

from techval.config import Assumptions
from techval.errors import ConfigError, MissingDataError
from techval.market import PriceSeries
from techval.wacc import (
    compute_wacc,
    estimate_beta,
    relever,
    synthetic_credit_spread,
    unlever,
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
