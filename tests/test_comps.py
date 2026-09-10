"""Peer multiples, suppression of the meaningless, and the implied range."""

from __future__ import annotations

import pandas as pd
import pytest

from techval.comps import run_comps

from conftest import FixtureClient


@pytest.fixture
def result(assumptions, market):
    assumptions.comps.peers = ["CRWD", "MDB", "ZS"]
    return run_comps("DDOG", assumptions, FixtureClient(), market)


def test_every_requested_peer_is_accounted_for(result, assumptions):
    """Never dropped silently: a peer is either in the table or in exclusions."""
    seen = {p.ticker for p in result.peers} | {e.ticker for e in result.exclusions}
    assert seen == set(assumptions.comps.peers)


def test_exclusions_carry_a_reason(result):
    for e in result.exclusions:
        assert e.reason and len(e.reason) > 10


# --------------------------------------------------------------------------- #
# not meaningful
# --------------------------------------------------------------------------- #


def test_peer_without_ebitda_keeps_its_other_multiples(result):
    """CrowdStrike publishes no D&A covering the window, so it has no EBITDA.

    That is not a reason to discard what the market pays for its revenue.
    """
    crwd = next(p for p in result.peers if p.ticker == "CRWD")
    assert crwd.ev_ebitda is None
    assert crwd.ev_revenue is not None and crwd.ev_revenue > 0
    assert any("EV/EBITDA" in f for f in crwd.flags)


def test_negative_earnings_suppress_the_multiple_and_flag_it(result):
    for p in result.peers:
        if p.ebit is not None and p.ebit <= 0:
            assert p.ev_ebit is None
            assert any("EV/EBIT" in f for f in p.flags)
        if p.net_income <= 0:
            assert p.pe is None


def test_multiple_past_the_cut_off_is_withheld(assumptions, market):
    """A 3,000x EV/EBITDA measures a margin near zero, not a valuation."""
    assumptions.comps.peers = ["MDB"]
    assumptions.comps.ev_ebitda_nm_threshold = 100.0
    r = run_comps("DDOG", assumptions, FixtureClient(), market)
    mdb = next(p for p in r.peers if p.ticker == "MDB")

    assert mdb.ebitda is not None and mdb.ebitda > 0
    assert mdb.ev_ebitda is None
    assert any("cut-off" in f for f in mdb.flags)


def test_raising_the_cut_off_lets_the_multiple_through(assumptions, market):
    assumptions.comps.peers = ["MDB"]
    assumptions.comps.ev_ebitda_nm_threshold = 1e9
    r = run_comps("DDOG", assumptions, FixtureClient(), market)
    mdb = next(p for p in r.peers if p.ticker == "MDB")
    assert mdb.ev_ebitda is not None and mdb.ev_ebitda > 100


def test_price_earnings_has_its_own_lower_cut_off(assumptions, market):
    """Net income sits below every other line, so it reaches zero sooner."""
    assumptions.comps.peers = ["CRWD", "MDB", "ZS"]
    assumptions.comps.pe_nm_threshold = 75.0
    r = run_comps("DDOG", assumptions, FixtureClient(), market)
    for p in r.peers:
        assert p.pe is None or p.pe <= 75.0


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def test_stats_report_how_many_peers_contributed(result):
    """A median over two names is a quotation, not a distribution."""
    assert "n" in result.stats.index
    for col in result.stats.columns:
        n = result.stats.loc["n", col]
        contributing = sum(
            1
            for p in result.peers
            if getattr(p, _attr_for(col)) is not None
        )
        assert n == contributing


def _attr_for(column: str) -> str:
    return {
        "EV/Revenue": "ev_revenue",
        "EV/Gross Profit": "ev_gross_profit",
        "EV/EBITDA": "ev_ebitda",
        "EV/EBIT": "ev_ebit",
        "P/E": "pe",
    }[column]


def test_percentiles_are_ordered(result):
    for col in result.stats.columns:
        s = result.stats[col]
        vals = [s.get(k) for k in ("Min", "p25", "Median", "p75", "Max")]
        vals = [v for v in vals if v is not None and pd.notna(v)]
        assert vals == sorted(vals)


def test_stats_ignore_suppressed_values(result):
    """A withheld multiple must not re-enter as a zero or a NaN in the median."""
    col = "EV/Revenue"
    contributing = sorted(
        p.ev_revenue for p in result.peers if p.ev_revenue is not None
    )
    assert result.stats.loc["Min", col] == pytest.approx(contributing[0])
    assert result.stats.loc["Max", col] == pytest.approx(contributing[-1])


# --------------------------------------------------------------------------- #
# implied value
# --------------------------------------------------------------------------- #


def test_implied_range_applies_peer_percentiles_to_the_target(result, ddog):
    row = result.implied.loc["EV/Revenue"]
    assert row["Target metric"] == pytest.approx(ddog.revenue)
    assert row["Implied EV low"] == pytest.approx(
        ddog.revenue * row["Peer p25"], rel=1e-6
    )
    assert row["Implied price low"] < row["Implied price high"]


def test_row_is_suppressed_where_the_target_metric_is_not_positive(
    assumptions, market
):
    """You cannot apply an EV/EBITDA to a negative EBITDA."""
    assumptions.comps.peers = ["DDOG", "MDB"]
    r = run_comps("ZS", assumptions, FixtureClient(), market)
    if "EV/EBIT" in r.implied.index:
        assert r.implied.loc["EV/EBIT"]["Target metric"] > 0
    assert any("EV/EBIT" in n for n in r.notes)


def test_no_meaningful_peer_means_no_implied_row(result):
    if result.stats.loc["n", "EV/EBITDA"] == 0:
        assert "EV/EBITDA" not in result.implied.index
        assert any("EV/EBITDA" in n for n in result.notes)


# --------------------------------------------------------------------------- #
# conventions
# --------------------------------------------------------------------------- #


def test_lease_convention_is_carried_into_the_denominator(assumptions, market):
    """EV including leases must meet EBITDAR, never a post-rent EBITDA."""
    assumptions.comps.peers = ["MDB"]
    assumptions.comps.ev_ebitda_nm_threshold = 1e9
    plain = run_comps("DDOG", assumptions, FixtureClient(), market)

    assumptions.leases.capitalize_operating_leases = True
    leased = run_comps("DDOG", assumptions, FixtureClient(), market)

    a = next(p for p in plain.peers if p.ticker == "MDB")
    b = next(p for p in leased.peers if p.ticker == "MDB")
    assert b.enterprise_value > a.enterprise_value
    assert b.ebitda > a.ebitda
    assert any("EBITDAR" in n or "lease" in n.lower() for n in leased.notes)


def test_rule_of_40_is_growth_plus_margin_in_points(result):
    for p in result.peers:
        if p.revenue_growth is None or p.ebitda_margin is None:
            assert p.rule_of_40 is None
        else:
            assert p.rule_of_40 == pytest.approx(
                (p.revenue_growth + p.ebitda_margin) * 100
            )


def test_growth_is_measured_not_guessed(result):
    for p in result.peers:
        if p.revenue_growth is not None:
            assert -1.0 < p.revenue_growth < 3.0
