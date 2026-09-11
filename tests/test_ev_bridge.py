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


def test_conversion_shares_are_confirmed_present_in_diluted_waso(
    ddog, assumptions, market
):
    """The if-converted treatment rests on a testable claim, so it is tested.

    Datadog's diluted WASO runs 14.6mm above basic and conversion accounts for
    6.7mm of that, so the shares really are in the count the equity value is
    built on.
    """
    assumptions.convertibles.treatment = "auto"
    assumptions.convertibles.conversion_price = 148.15
    b = build_ev_bridge(ddog, market.spot("DDOG"), assumptions)

    implied = ddog.convertible_debt / 148.15
    gap = ddog.diluted_shares - ddog.basic_shares
    assert gap > implied
    assert any(n.startswith("Checked:") for n in b.notes)


def test_conversion_shares_missing_from_diluted_waso_is_flagged(
    ddog, assumptions, market
):
    """If the gap cannot hold the conversion shares, the treatment is unsound.

    Forced here by naming a conversion price low enough that conversion would
    create more shares than the whole diluted-to-basic gap.
    """
    assumptions.convertibles.treatment = "auto"
    assumptions.convertibles.conversion_price = 20.0
    b = build_ev_bridge(ddog, market.spot("DDOG"), assumptions)

    assert b.convertible_treatment == "if_converted"
    assert any(n.startswith("Warning:") and "understated" in n for n in b.notes)


# --------------------------------------------------------------------------- #
# the treasury stock count and the convertible, which must not fall between them
# --------------------------------------------------------------------------- #


def test_bridge_prices_on_the_valuation_share_count(ddog, assumptions, market):
    """Equity value divides by whatever count has been settled, not always WASO."""
    price = market.spot("DDOG")
    assumptions.convertibles.conversion_price = None
    assumptions.convertibles.treatment = "debt"

    waso = build_ev_bridge(ddog, price, assumptions)
    assert waso.equity_value == pytest.approx(price * ddog.diluted_shares)

    ddog.valuation_shares = 377.73
    tsm = build_ev_bridge(ddog, price, assumptions)
    assert tsm.equity_value == pytest.approx(price * 377.73)
    assert tsm.equity_value > waso.equity_value


def test_conversion_shares_are_added_to_a_treasury_stock_count(
    ddog, assumptions, market
):
    """The instrument has to appear on one side of the bridge or the other.

    Diluted WASO already contains an in-the-money convertible's shares, because
    ASU 2020-06 makes if-converted mandatory. A treasury stock count does not:
    it is built from shares outstanding and the award tables, and a convertible
    is neither. So pairing that count with a bridge that also carries the note
    as equity rather than debt would drop the instrument out of BOTH sides, and
    understate the share count by the conversion shares.

    Datadog: 986mm of notes at a 148.15 conversion price is 6.65mm shares.
    """
    price = market.spot("DDOG")
    assumptions.convertibles.treatment = "auto"
    assumptions.convertibles.conversion_price = 148.15
    ddog.valuation_shares = 377.73

    b = build_ev_bridge(ddog, price, assumptions)
    expected = 377.73 + ddog.convertible_debt / 148.15

    assert b.convertible_treatment == "if_converted"
    assert b.convertible_in_debt == 0.0
    assert b.diluted_shares == pytest.approx(expected)
    assert any("conversion shares to the treasury" in n for n in b.notes)


def test_no_conversion_shares_are_added_when_the_note_is_debt(
    ddog, assumptions, market
):
    """Carried as debt, the note is already on the bridge and must not be twice."""
    price = market.spot("DDOG")
    assumptions.convertibles.treatment = "debt"
    assumptions.convertibles.conversion_price = 148.15
    ddog.valuation_shares = 377.73

    b = build_ev_bridge(ddog, price, assumptions)
    assert b.convertible_in_debt == pytest.approx(ddog.convertible_debt)
    assert b.diluted_shares == pytest.approx(377.73)


def test_waso_path_never_adds_conversion_shares(ddog, assumptions, market):
    """Diluted WASO already contains them; adding again would double count."""
    price = market.spot("DDOG")
    assumptions.convertibles.treatment = "auto"
    assumptions.convertibles.conversion_price = 148.15
    assert ddog.valuation_shares is None

    b = build_ev_bridge(ddog, price, assumptions)
    assert b.diluted_shares == pytest.approx(ddog.diluted_shares)


# --------------------------------------------------------------------------- #
# a debt figure that carries finance leases inside it
#
# Verizon's balance sheet has one line for long-term debt and no line for
# finance leases, because the concept it tags is "long-term debt AND capital
# lease obligations". Everything below is about saying so on the page rather
# than presenting a debt row and a zero lease row and letting the reader draw
# the obvious wrong conclusion.
#
# Priced at 49.97, the Nasdaq close for 2026-09-11, passed in rather than read
# from a fixture so the bridge arithmetic is the only thing under test.
# --------------------------------------------------------------------------- #

VZ_CLOSE = 49.97


def test_the_debt_row_names_what_is_inside_it(vz, assumptions):
    """"Straight debt 165,231" beside "Finance leases 0.0" reads as a company
    with no finance leases. What is true is that they are in the row above."""
    b = build_ev_bridge(vz, VZ_CLOSE, assumptions)
    labels = dict(b.rows())

    assert "+ Debt and finance leases" in labels
    assert "+ Straight debt" not in labels
    assert labels["+ Debt and finance leases"] == pytest.approx(165_231.0)


def test_the_bridge_states_the_concepts_the_debt_came_from(vz, assumptions):
    """The engine's rule is that a number nobody can trace does not get printed.

    A debt figure is the line item most likely to be silently wrong, so the
    bridge names the two concepts it was built out of.
    """
    b = build_ev_bridge(vz, VZ_CLOSE, assumptions)
    note = " ".join(b.notes)
    assert "LongTermDebtAndCapitalLeaseObligations" in note
    assert "LongTermDebtCurrent" in note


def test_the_telecom_enterprise_value_reconciles_to_the_balance_sheet(vz, assumptions):
    """The whole finding, in one assertion.

    4,212.65mm diluted shares at 49.97 is 210,506mm of equity value. Verizon's
    Q2 2026 balance sheet carries 143,448mm of long-term debt, 21,783mm maturing
    within one year, 1,752mm of cash and 1,276mm of non-controlling interest, so
    net debt is 164,755mm and enterprise value is 375,261mm. The engine printed
    231,813mm before this ladder was rewritten, because it resolved the 21,783mm
    and nothing else.
    """
    b = build_ev_bridge(vz, VZ_CLOSE, assumptions)

    assert b.equity_value == pytest.approx(210_506.0, rel=1e-4)
    assert b.net_debt == pytest.approx(164_755.0, rel=1e-4)
    assert b.enterprise_value == pytest.approx(375_261.0, rel=1e-4)
    assert b.enterprise_value > b.equity_value


def test_a_net_cash_bridge_says_why_it_is_below_the_market_capitalisation(
    ddog, ddog_bridge
):
    """An enterprise value below market cap is arithmetic, and is also what a
    debt figure read as zero looks like. The bridge says which.

    Comcast's peer row printed an enterprise value 7.7 billion dollars below its
    own market capitalisation and said nothing at all, and the reason was
    90 billion dollars of debt under a concept no ladder carried.
    """
    assert ddog_bridge.enterprise_value < ddog_bridge.equity_value
    note = " ".join(ddog_bridge.notes)
    assert "below" in note and "market capitalisation" in note
    assert "a debt figure that failed to resolve looks exactly like this" in note


def test_a_bridge_with_net_debt_carries_no_net_cash_note(vz, assumptions):
    """The note has to stay off the company it does not apply to."""
    b = build_ev_bridge(vz, VZ_CLOSE, assumptions)
    assert not [n for n in b.notes if "market capitalisation" in n]


def test_a_lease_inside_debt_is_never_added_on_the_lease_row(vz, assumptions):
    """The identity the bridge is tested on cannot be made to close twice.

    Whatever the lease convention, the finance lease row carries only what the
    filer tags separately, so switching the convention can move operating leases
    and can never move the same finance lease in twice.
    """
    for capitalize in (False, True):
        assumptions.leases.capitalize_operating_leases = capitalize
        b = build_ev_bridge(vz, VZ_CLOSE, assumptions)
        assert b.finance_lease == pytest.approx(vz.finance_lease_liability)
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
