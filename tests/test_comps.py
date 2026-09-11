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


def test_growth_works_for_a_52_53_week_filer(assumptions, market, monkeypatch):
    """A 13-week fiscal quarter does not land on the calendar anniversary.

    A 52/53-week year closes 364 days back, one day before a fixed 365-day step
    lands, and the tiler cannot cover a window whose end falls between two
    reported periods. Anchoring on a period end the filer actually reported is
    what makes growth available for this calendar at all. Cisco, Broadcom,
    Marvell, NetApp and Dell all use it, and all are plausible comp-set members.
    """
    import json
    from datetime import date, timedelta

    from techval.edgar import CompanyFacts

    # Thirteen-week quarters marching back from 2026-08-01.
    end = date(2026, 8, 1)
    quarters = []
    for i in range(9):
        q_end = end - timedelta(days=91 * i)
        quarters.append(
            {
                "start": (q_end - timedelta(days=90)).isoformat(),
                "end": q_end.isoformat(),
                "val": 250_000_000 - 10_000_000 * i,
                "form": "10-Q",
                "filed": (q_end + timedelta(days=30)).isoformat(),
            }
        )
    payload = {
        "cik": 1,
        "entityName": "Thirteen Weeks Inc",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": quarters}
                }
            }
        },
    }
    facts = CompanyFacts(payload, "WEEK")

    from techval.financials import Financials
    from techval.comps import _revenue_growth

    current, _ = facts.resolve_ttm(
        "revenue",
        ["RevenueFromContractWithCustomerExcludingAssessedTax"],
        end,
    )
    stub = object.__new__(Financials)
    object.__setattr__(stub, "as_of", end)
    object.__setattr__(stub, "revenue", current / 1e6)
    object.__setattr__(stub, "provenance", {})
    flags: list[str] = []
    growth = _revenue_growth(stub, facts, flags)

    assert growth is not None, f"growth unavailable for a 13-week filer: {flags}"
    assert growth > 0


def test_negative_enterprise_value_withholds_ev_multiples(assumptions, market):
    """Cash above market cap plus debt makes every EV multiple negative.

    Negative multiples would drag the whole percentile distribution below zero,
    so they are withheld with the reason on show while P/E, an equity multiple,
    survives on its own merits.
    """
    from datetime import date as _date

    from techval.comps import compute_peer_metrics
    from techval.financials import Financials

    fin = Financials(
        ticker="CASHBOX", entity_name="Cashbox", cik=1, as_of=_date(2026, 6, 30),
        revenue=500.0, gross_profit=350.0, ebit=40.0, da=10.0, sbc=0.0,
        net_income=50.0, pretax_income=60.0, tax_expense=10.0,
        interest_expense=0.0, operating_lease_cost=0.0, capex=5.0, cfo=None,
        diluted_shares=100.0, basic_shares=100.0,
        cash=2_000.0, short_term_investments=0.0, straight_debt=0.0,
        convertible_debt=0.0, operating_lease_liability=0.0,
        finance_lease_liability=0.0, nci=0.0, preferred=0.0,
        current_assets=None, current_liabilities=None, deferred_revenue=0.0,
    )
    from techval.edgar import CompanyFacts

    # Priced at 10.00 the equity is 1,000mm against 2,000mm of cash: EV -1,000mm.
    empty = CompanyFacts({"cik": 1, "entityName": "Cashbox", "facts": {}}, "CASHBOX")
    m = compute_peer_metrics(fin, empty, 10.0, assumptions)
    assert m.enterprise_value <= 0
    assert m.ev_revenue is None and m.ev_ebitda is None and m.ev_ebit is None
    assert m.pe is not None and m.pe > 0
    assert any("not positive" in f for f in m.flags)


# --------------------------------------------------------------------------- #
# warranted multiples from a regression on fundamentals
# --------------------------------------------------------------------------- #

# The line the synthetic peer sets are drawn from: a 2.0x base, twenty turns per
# point of growth and ten per point of margin. Recovering these exactly from
# noiseless data is the only proof the normal equations are being solved and not
# something that merely correlates with them.
_A, _B1, _B2 = 2.0, 20.0, 10.0


def _peer(ticker, *, ev_revenue, growth, margin, revenue=1_000.0, price=50.0):
    """One synthetic comp, carrying only the lines the regression reads."""
    from techval.comps import PeerMetrics

    equity = price * 100.0
    return PeerMetrics(
        ticker=ticker,
        name=f"{ticker} Inc",
        price=price,
        market_cap=equity,
        enterprise_value=(ev_revenue * revenue if ev_revenue is not None else equity),
        gross_debt=0.0,
        revenue=revenue,
        ebitda=None if margin is None else margin * revenue,
        ebit=0.0,
        net_income=0.0,
        gross_profit=None,
        revenue_growth=growth,
        ebitda_margin=margin,
        gross_margin=None,
        rule_of_40=None,
        ev_revenue=ev_revenue,
        ev_gross_profit=None,
        ev_ebitda=None,
        ev_ebit=None,
        pe=None,
        flags=[],
    )


def _on_the_line(n=12, *, noise=False, a=_A, b1=_B1):
    """``n`` peers drawn from a known plane, exactly or with a wobble.

    Growth marches linearly, margin cycles on a four-step pattern. That keeps the
    two drivers genuinely independent: a margin defined as a linear function of
    growth would make the design matrix singular and test the refusal path by
    accident rather than the fit.
    """
    peers = []
    for i in range(n):
        growth = 0.10 + 0.03 * i
        margin = 0.05 + 0.02 * (i % 4)
        ev_rev = a + b1 * growth + _B2 * margin
        if noise:
            ev_rev += 0.25 * (-1) ** i * (1 + i % 3)
        peers.append(_peer(f"P{i:02d}", ev_revenue=ev_rev, growth=growth, margin=margin))
    return peers


def _synthetic_target(growth=0.25, margin=0.10, ev_revenue=9.0):
    return _peer("TGT", ev_revenue=ev_revenue, growth=growth, margin=margin)


def _real_target(assumptions, *, price=100.0, growth=0.25, margin=0.10):
    """A target built from real ``Financials`` so the value walk is the real walk.

    Returns the statements, the bridge and the priced metrics together. Growth and
    margin are set on the metrics afterwards because the empty fact set carries no
    prior-year revenue to measure growth from, and the regression needs drivers,
    not a growth calculation.
    """
    from datetime import date as _date

    from techval.comps import compute_peer_metrics
    from techval.edgar import CompanyFacts
    from techval.ev_bridge import build_ev_bridge
    from techval.financials import Financials

    fin = Financials(
        ticker="TGT", entity_name="Warranted Inc", cik=1, as_of=_date(2026, 6, 30),
        revenue=1_000.0, gross_profit=800.0, ebit=60.0, da=40.0, sbc=0.0,
        net_income=50.0, pretax_income=60.0, tax_expense=10.0,
        interest_expense=12.0, operating_lease_cost=15.0, capex=30.0, cfo=None,
        diluted_shares=100.0, basic_shares=98.0,
        cash=400.0, short_term_investments=100.0, straight_debt=600.0,
        convertible_debt=0.0, operating_lease_liability=120.0,
        finance_lease_liability=40.0, nci=10.0, preferred=0.0,
        current_assets=None, current_liabilities=None, deferred_revenue=0.0,
    )
    facts = CompanyFacts({"cik": 1, "entityName": "Warranted Inc", "facts": {}}, "TGT")
    bridge = build_ev_bridge(fin, price, assumptions)
    metrics = compute_peer_metrics(fin, facts, price, assumptions, is_target=True)
    metrics.revenue_growth = growth
    metrics.ebitda_margin = margin
    return fin, bridge, metrics


def test_noiseless_data_recovers_the_coefficients_exactly(assumptions):
    """The whole method rests on this: OLS on a plane returns that plane."""
    from techval.comps import fit_warranted_multiple

    fit = fit_warranted_multiple(_on_the_line(12), _synthetic_target(), assumptions)

    assert fit is not None
    assert fit.intercept == pytest.approx(_A, abs=1e-9)
    assert fit.coefficients["revenue_growth"] == pytest.approx(_B1, abs=1e-9)
    assert fit.coefficients["ebitda_margin"] == pytest.approx(_B2, abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.adj_r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.standard_error == pytest.approx(0.0, abs=1e-9)
    assert fit.n_observations == 12
    # A coefficient measured with no residual variance is infinitely precise.
    assert all(abs(t) > 1e3 for t in fit.t_stats.values())


def test_warranted_multiple_reads_the_line_at_the_target(assumptions):
    from techval.comps import fit_warranted_multiple

    target = _synthetic_target(growth=0.25, margin=0.10, ev_revenue=9.0)
    fit = fit_warranted_multiple(_on_the_line(12), target, assumptions)

    expected = _A + _B1 * 0.25 + _B2 * 0.10
    assert fit.warranted_multiple == pytest.approx(expected, abs=1e-9)
    assert fit.actual_multiple == pytest.approx(9.0)
    assert fit.residual == pytest.approx(9.0 - expected, abs=1e-9)


def test_residual_is_actual_less_warranted_in_both_directions(assumptions):
    """Positive residual means the market pays above what the drivers explain."""
    from techval.comps import fit_warranted_multiple

    peers = _on_the_line(12)
    warranted = _A + _B1 * 0.25 + _B2 * 0.10

    rich = fit_warranted_multiple(peers, _synthetic_target(ev_revenue=warranted + 3.0), assumptions)
    cheap = fit_warranted_multiple(peers, _synthetic_target(ev_revenue=warranted - 3.0), assumptions)

    assert rich.residual == pytest.approx(3.0, abs=1e-9)
    assert cheap.residual == pytest.approx(-3.0, abs=1e-9)


def test_adjusted_r_squared_is_below_the_raw_figure(assumptions):
    """Raw R-squared cannot fall when a regressor is added; adjusted can."""
    from techval.comps import fit_warranted_multiple

    fit = fit_warranted_multiple(_on_the_line(12, noise=True), _synthetic_target(), assumptions)

    assert fit is not None
    assert 0.0 < fit.r_squared < 1.0
    assert fit.adj_r_squared < fit.r_squared
    n, k = fit.n_observations, len(fit.drivers)
    assert fit.adj_r_squared == pytest.approx(
        1.0 - (1.0 - fit.r_squared) * (n - 1) / (n - k - 1)
    )
    assert fit.standard_error > 0.0
    assert set(fit.t_stats) == set(fit.drivers) | {"intercept"}


def test_the_shipped_six_peer_set_is_refused(assumptions):
    """Two parameters on six points memorises the set. It must say so, not fit."""
    from techval.comps import fit_warranted_multiple

    notes: list[str] = []
    fit = fit_warranted_multiple(
        _on_the_line(6), _synthetic_target(), assumptions, notes=notes
    )

    assert fit is None
    assert any("not available" in n for n in notes)
    assert any("comps.peers" in n for n in notes)
    assert any("8" in n for n in notes)


def test_fixture_peer_set_is_refused_and_says_why(assumptions, market):
    """Three fixture peers cannot carry a two-driver fit, and the run says so."""
    assumptions.comps.peers = ["CRWD", "MDB", "ZS"]
    assumptions.comps.regression.enabled = True
    r = run_comps("DDOG", assumptions, FixtureClient(), market)

    assert r.regression is None
    assert any("Warranted EV/Revenue regression is not available" in n for n in r.notes)
    assert any("comps.peers" in n for n in r.notes)


def test_regression_is_off_unless_asked_for(assumptions, market):
    assumptions.comps.peers = ["CRWD", "MDB", "ZS"]
    assert assumptions.comps.regression.enabled is False
    r = run_comps("DDOG", assumptions, FixtureClient(), market)
    assert r.regression is None
    assert not any("regression" in n.lower() for n in r.notes)


def test_three_points_per_parameter_is_the_floor(assumptions):
    """Even with the minimum lowered, five points for two drivers is refused."""
    from techval.comps import fit_warranted_multiple

    assumptions.comps.regression.min_observations = 3
    notes: list[str] = []
    fit = fit_warranted_multiple(
        _on_the_line(5), _synthetic_target(), assumptions, notes=notes
    )

    assert fit is None
    assert any("three points per regressor" in n for n in notes)


def test_singular_design_is_refused_not_pseudo_inverted(assumptions):
    """A driver that does not move cannot be told apart from the intercept."""
    from techval.comps import fit_warranted_multiple

    peers = [
        _peer(f"S{i:02d}", ev_revenue=3.0 + 20.0 * (0.10 + 0.02 * i),
              growth=0.10 + 0.02 * i, margin=0.20)
        for i in range(10)
    ]
    notes: list[str] = []
    fit = fit_warranted_multiple(peers, _synthetic_target(), assumptions, notes=notes)

    assert fit is None
    assert any("singular" in n for n in notes)


def test_collinear_drivers_are_refused(assumptions):
    """Margin defined as twice growth carries no separate information."""
    from techval.comps import fit_warranted_multiple

    peers = [
        _peer(f"C{i:02d}", ev_revenue=3.0 + 5.0 * i,
              growth=0.10 + 0.02 * i, margin=2.0 * (0.10 + 0.02 * i))
        for i in range(10)
    ]
    fit = fit_warranted_multiple(peers, _synthetic_target(), assumptions)
    assert fit is None


def test_a_peer_missing_a_driver_is_dropped_and_named(assumptions):
    """Never imputed: a peer given the sample mean sits on the line by construction."""
    from techval.comps import fit_warranted_multiple

    peers = _on_the_line(12)
    peers[3].ebitda_margin = None
    peers[7].ev_revenue = None

    fit = fit_warranted_multiple(peers, _synthetic_target(), assumptions)

    assert fit is not None
    assert fit.n_observations == 10
    assert any(peers[3].ticker in n and "ebitda margin" in n for n in fit.notes)
    assert any(peers[7].ticker in n and "EV/Revenue" in n for n in fit.notes)
    # The ten survivors still sit on the original plane, so the fit is unchanged.
    assert fit.coefficients["revenue_growth"] == pytest.approx(_B1, abs=1e-9)


def test_dropped_peers_can_push_the_sample_under_the_minimum(assumptions):
    from techval.comps import fit_warranted_multiple

    peers = _on_the_line(9)
    for p in peers[:3]:
        p.revenue_growth = None
    notes: list[str] = []

    assert fit_warranted_multiple(peers, _synthetic_target(), assumptions, notes=notes) is None
    assert any("6 of 9 peers" in n for n in notes)


def test_a_backwards_coefficient_is_flagged(assumptions):
    """Growth that reduces the multiple is a symptom, not a finding."""
    from techval.comps import fit_warranted_multiple

    # A high base so the backwards slope still warrants a positive multiple at
    # the target: the point under test is the sign flag, not the refusal.
    peers = _on_the_line(12, a=12.0, b1=-20.0)
    fit = fit_warranted_multiple(peers, _synthetic_target(), assumptions)

    assert fit is not None
    assert fit.coefficients["revenue_growth"] < 0
    assert any("Sign check" in n and "revenue growth" in n for n in fit.notes)


def test_a_target_missing_a_driver_gets_no_warranted_multiple(assumptions):
    from techval.comps import fit_warranted_multiple

    target = _synthetic_target()
    target.ebitda_margin = None
    notes: list[str] = []

    assert fit_warranted_multiple(_on_the_line(12), target, assumptions, notes=notes) is None
    assert any("TGT" in n and "ebitda margin" in n for n in notes)


def test_a_negative_warranted_multiple_is_refused(assumptions):
    """A positive multiple times positive revenue is a value; a negative one is not."""
    from techval.comps import fit_warranted_multiple

    notes: list[str] = []
    # Far below every peer on both drivers, so the fitted line runs below zero.
    target = _synthetic_target(growth=-1.0, margin=-1.0)
    assert fit_warranted_multiple(_on_the_line(12), target, assumptions, notes=notes) is None
    assert any("is not a valuation" in n for n in notes)


def test_the_target_outside_the_peer_range_is_flagged_not_refused(assumptions):
    from techval.comps import fit_warranted_multiple

    fit = fit_warranted_multiple(_on_the_line(12), _synthetic_target(growth=0.90), assumptions)

    assert fit is not None
    assert any("extrapolates" in n for n in fit.notes)


def test_implied_value_uses_the_same_bridge_walk_as_the_percentile_range(assumptions):
    """A warranted price and a p25 price must subtract identical debt."""
    from techval.comps import fit_warranted_multiple
    from techval.ev_bridge import equity_value_from_ev

    fin, bridge, target = _real_target(assumptions)
    fit = fit_warranted_multiple(
        _on_the_line(12), target, assumptions, target_fin=fin, bridge=bridge
    )

    assert fit is not None
    assert fit.implied_ev == pytest.approx(fit.warranted_multiple * fin.revenue)
    expected_equity = equity_value_from_ev(fit.implied_ev, fin, bridge)
    assert fit.implied_price == pytest.approx(expected_equity / fin.diluted_shares)
    # Same debt definition as the percentile walk, so the two prices sit on the
    # same scale and their difference is a view, not a convention mismatch.
    assert fit.implied_price < fit.implied_ev / fin.diluted_shares


def test_the_walk_is_the_same_with_or_without_the_bridge(assumptions):
    """EV less market capitalisation is the bridge walk in reduced form."""
    from techval.comps import fit_warranted_multiple

    fin, bridge, target = _real_target(assumptions)
    peers = _on_the_line(12)

    with_bridge = fit_warranted_multiple(
        peers, target, assumptions, target_fin=fin, bridge=bridge
    )
    without = fit_warranted_multiple(peers, target, assumptions)

    assert without.implied_price == pytest.approx(with_bridge.implied_price)


def test_rows_render_without_a_none(assumptions):
    from techval.comps import fit_warranted_multiple

    target = _synthetic_target()
    target.ev_revenue = None
    fit = fit_warranted_multiple(_on_the_line(12), target, assumptions)

    assert fit.actual_multiple is None and fit.residual is None
    rows = fit.rows()
    assert all(isinstance(v, float) for _, v in rows)
    assert any(label.startswith("Warranted") for label, _ in rows)
    assert not any(label.startswith("Residual,") for label, _ in rows)
    assert any("suppressed" in n for n in fit.notes)


def test_an_unknown_dependent_or_driver_is_a_config_error(assumptions):
    from techval.comps import fit_warranted_multiple
    from techval.errors import ConfigError

    with pytest.raises(ConfigError):
        fit_warranted_multiple(
            _on_the_line(12), _synthetic_target(), assumptions, dependent="ev_sales"
        )

    assumptions.comps.regression.drivers = ["net_retention"]
    with pytest.raises(ConfigError):
        fit_warranted_multiple(_on_the_line(12), _synthetic_target(), assumptions)


def test_a_single_driver_still_fits(assumptions):
    """One regressor needs only three points per parameter, plus the minimum."""
    from techval.comps import fit_warranted_multiple

    assumptions.comps.regression.drivers = ["revenue_growth"]
    peers = [
        _peer(f"G{i:02d}", ev_revenue=_A + _B1 * (0.10 + 0.03 * i),
              growth=0.10 + 0.03 * i, margin=0.10)
        for i in range(9)
    ]
    fit = fit_warranted_multiple(peers, _synthetic_target(growth=0.25), assumptions)

    assert fit is not None
    assert fit.drivers == ["revenue_growth"]
    assert fit.coefficients["revenue_growth"] == pytest.approx(_B1, abs=1e-9)
    assert fit.warranted_multiple == pytest.approx(_A + _B1 * 0.25, abs=1e-9)
