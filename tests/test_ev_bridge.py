"""The bridge, and the two ways it is normally got wrong."""

from __future__ import annotations

import pytest

from techval.ev_bridge import build_ev_bridge, equity_value_from_ev


def test_bridge_is_the_stated_identity(ddog, ddog_bridge):
    b = ddog_bridge
    assert b.equity_value == pytest.approx(b.price * b.diluted_shares)
    assert b.enterprise_value == pytest.approx(
        b.equity_value
        + b.straight_debt
        + b.convertible_in_debt
        + b.finance_lease
        + b.operating_lease_in_debt
        + b.preferred
        + b.nci
        - b.cash
        - b.short_term_investments
    )


def test_net_debt_is_ev_less_equity(ddog_bridge):
    assert ddog_bridge.net_debt == pytest.approx(
        ddog_bridge.enterprise_value - ddog_bridge.equity_value
    )


def test_walking_back_from_ev_returns_equity_value(ddog, ddog_bridge):
    """The two directions must reconcile, or a DCF and a comp cannot be compared."""
    recovered = equity_value_from_ev(ddog_bridge.enterprise_value, ddog, ddog_bridge)
    assert recovered == pytest.approx(ddog_bridge.equity_value)


# --------------------------------------------------------------------------- #
# operating leases under ASC 842
# --------------------------------------------------------------------------- #


def test_capitalising_leases_raises_ev_by_the_liability(ddog, assumptions, market):
    price = market.spot("DDOG")
    excluded = build_ev_bridge(ddog, price, assumptions)

    assumptions.leases.capitalize_operating_leases = True
    included = build_ev_bridge(ddog, price, assumptions)

    assert included.enterprise_value == pytest.approx(
        excluded.enterprise_value + ddog.operating_lease_liability
    )
    assert ddog.operating_lease_liability > 0


def test_both_enterprise_values_are_reported_under_either_convention(
    ddog, assumptions, market
):
    price = market.spot("DDOG")
    a = build_ev_bridge(ddog, price, assumptions)
    assumptions.leases.capitalize_operating_leases = True
    b = build_ev_bridge(ddog, price, assumptions)

    assert a.ev_excluding_leases == pytest.approx(b.ev_excluding_leases)
    assert a.ev_including_leases == pytest.approx(b.ev_including_leases)
    assert a.enterprise_value == a.ev_excluding_leases
    assert b.enterprise_value == b.ev_including_leases


def test_denominator_switches_to_ebitdar_when_leases_are_debt(ddog, assumptions, market):
    """The guard against dividing a lease-inclusive EV by a post-rent EBITDA.

    Under ASC 842 the rent charge stays inside operating income, so US-GAAP
    EBITDA is already after rent. Counting the liability as debt and dividing by
    that same EBITDA charges the lease twice.
    """
    price = market.spot("DDOG")

    excluded = build_ev_bridge(ddog, price, assumptions)
    value, label = excluded.multiple_denominator(ddog)
    assert value == pytest.approx(ddog.ebitda)
    assert "after operating lease cost" in label

    assumptions.leases.capitalize_operating_leases = True
    included = build_ev_bridge(ddog, price, assumptions)
    value, label = included.multiple_denominator(ddog)
    assert value == pytest.approx(ddog.ebitdar)
    assert "EBITDAR" in label
    assert value > ddog.ebitda


def test_finance_leases_are_debt_under_both_conventions(mdb, assumptions, market):
    """Their interest and amortisation already sit outside EBITDA."""
    assert mdb.finance_lease_liability > 0
    price = market.spot("MDB")

    a = build_ev_bridge(mdb, price, assumptions)
    assumptions.leases.capitalize_operating_leases = True
    b = build_ev_bridge(mdb, price, assumptions)

    assert a.finance_lease == pytest.approx(mdb.finance_lease_liability)
    assert b.finance_lease == pytest.approx(mdb.finance_lease_liability)


def test_finance_leases_can_be_excluded_on_request(mdb, assumptions, market):
    assumptions.leases.include_finance_leases_in_debt = False
    b = build_ev_bridge(mdb, market.spot("MDB"), assumptions)
    assert b.finance_lease == 0.0


# --------------------------------------------------------------------------- #
# convertibles under ASU 2020-06
# --------------------------------------------------------------------------- #


def test_in_the_money_convertible_is_equity_not_debt(ddog, assumptions, market):
    """Its conversion shares are already inside diluted WASO."""
    assumptions.convertibles.treatment = "auto"
    assumptions.convertibles.conversion_price = 148.15
    b = build_ev_bridge(ddog, market.spot("DDOG"), assumptions)

    assert b.convertible_treatment == "if_converted"
    assert b.convertible_in_debt == 0.0
    assert b.convertible_debt > 0
    assert any("in the money" in n for n in b.notes)


def test_out_of_the_money_convertible_is_debt(ddog, assumptions, market):
    assumptions.convertibles.treatment = "auto"
    assumptions.convertibles.conversion_price = 10_000.0
    b = build_ev_bridge(ddog, market.spot("DDOG"), assumptions)

    assert b.convertible_treatment == "debt"
    assert b.convertible_in_debt == pytest.approx(ddog.convertible_debt)


def test_double_count_is_exactly_the_convertible_balance(ddog, assumptions, market):
    """The size of the error the default is there to prevent."""
    price = market.spot("DDOG")
    assumptions.convertibles.treatment = "if_converted"
    correct = build_ev_bridge(ddog, price, assumptions)
    assumptions.convertibles.treatment = "debt"
    doubled = build_ev_bridge(ddog, price, assumptions)

    assert doubled.enterprise_value - correct.enterprise_value == pytest.approx(
        ddog.convertible_debt
    )
    assert ddog.convertible_debt > 900


def test_missing_conversion_price_warns_loudly(ddog, assumptions, market):
    assumptions.convertibles.treatment = "auto"
    assumptions.convertibles.conversion_price = None
    b = build_ev_bridge(ddog, market.spot("DDOG"), assumptions)

    assert b.convertible_treatment == "debt"
    assert any("overstated" in n for n in b.notes)


def test_company_without_convertibles_is_left_alone(mdb, assumptions, market):
    assert mdb.convertible_debt == 0.0
    b = build_ev_bridge(mdb, market.spot("MDB"), assumptions)
    assert b.convertible_treatment == "none"
    assert b.convertible_in_debt == 0.0
