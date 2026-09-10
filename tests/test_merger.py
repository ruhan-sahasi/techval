"""Accretion/(dilution), checked against arithmetic done by hand."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from techval.config import Assumptions
from techval.errors import ConfigError
from techval.ev_bridge import build_ev_bridge
from techval.financials import Financials
from techval.merger import run_merger


def _company(
    ticker: str,
    *,
    revenue: float,
    ebit: float,
    da: float,
    net_income: float,
    shares: float,
    cash: float = 0.0,
    debt: float = 0.0,
) -> Financials:
    """A clean synthetic filer, so the hand arithmetic below is checkable."""
    return Financials(
        ticker=ticker,
        entity_name=ticker,
        cik=1,
        as_of=date(2026, 6, 30),
        revenue=revenue,
        gross_profit=revenue * 0.75,
        ebit=ebit,
        da=da,
        sbc=0.0,
        net_income=net_income,
        pretax_income=net_income / 0.76,
        tax_expense=net_income / 0.76 * 0.24,
        interest_expense=0.0,
        operating_lease_cost=0.0,
        capex=0.0,
        cfo=None,
        diluted_shares=shares,
        basic_shares=shares,
        cash=cash,
        short_term_investments=0.0,
        straight_debt=debt,
        convertible_debt=0.0,
        operating_lease_liability=0.0,
        finance_lease_liability=0.0,
        nci=0.0,
        preferred=0.0,
        current_assets=None,
        current_liabilities=None,
        deferred_revenue=0.0,
    )


@pytest.fixture
def hand_case():
    """The worked example the assertions below check against.

    Acquirer: 1,000mm net income, 500mm shares, so standalone EPS is 2.00.
              Share price 60.00, i.e. a 30.0x P/E.
    Target:   200mm net income, 100mm shares, unaffected price 25.00.

    Offer 32.50 a share, a 30% premium. Equity purchase price is
    32.50 x 100 = 3,250mm. Half cash, half stock.

        cash consideration            1,625.0
        stock consideration           1,625.0
        new shares  1,625 / 60.00  =     27.0833mm
        pro forma shares                527.0833mm
        new debt (no balance cash)    1,625.0
        interest 1,625 x 6.0%         =   97.5 pre-tax, 74.1 after tax at 24%
        synergies 150 pre-tax, fully phased, 114.0 after tax

        pro forma net income = 1,000 + 200 + 114.0 - 74.1 = 1,239.9
        pro forma EPS        = 1,239.9 / 527.083333... = 74,394 / 31,625
                             = 2.35237944664...
        accretion            = 0.35237944664..., or +17.61897%

    The EPS is asserted as the exact fraction rather than a rounded decimal, so
    the test measures the engine rather than how many digits were typed here.

    Breakeven pre-tax synergies solve
        2.00 x 527.0833 = 1,000 + 200 + S x 0.76 - 74.1
        1,054.1667 = 1,125.9 + 0.76 S
        S = -94.386mm
    Negative, meaning the deal is accretive with no synergies at all, which is
    what a 30.0x acquirer paying 16.25x for a target funded at 6% should be.
    """
    acq = _company(
        "ACQ", revenue=8000, ebit=1400, da=200, net_income=1000, shares=500, cash=3000
    )
    tgt = _company("TGT", revenue=1500, ebit=280, da=60, net_income=200, shares=100)

    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    a.tax.marginal_tax_rate = 0.24
    a.merger.offer_premium = 0.30
    a.merger.pct_cash = 0.50
    a.merger.cost_of_new_debt = 0.06
    a.merger.foregone_cash_yield = 0.04
    a.merger.balance_sheet_cash_used = 0.0
    a.merger.deal_fees = 0.0
    a.merger.financing_fees = 0.0
    a.merger.synergies.pretax_cost_synergies = 150.0
    a.merger.synergies.phase_in_year_one = 1.0

    acq_price, tgt_price = 60.0, 25.0
    acq_bridge = build_ev_bridge(acq, acq_price, a)
    tgt_bridge = build_ev_bridge(tgt, tgt_price, a)
    result = run_merger(acq, tgt, acq_bridge, tgt_bridge, acq_price, tgt_price, a)
    return result, a, acq, tgt


# --------------------------------------------------------------------------- #
# the hand-computed case
# --------------------------------------------------------------------------- #


def test_consideration_matches_hand_arithmetic(hand_case):
    r, _, _, _ = hand_case
    c = r.consideration
    assert c.offer_price == pytest.approx(32.50)
    assert c.premium == pytest.approx(0.30)
    assert c.equity_purchase_price == pytest.approx(3250.0)
    assert c.cash_consideration == pytest.approx(1625.0)
    assert c.stock_consideration == pytest.approx(1625.0)
    assert c.new_shares_issued == pytest.approx(1625.0 / 60.0)
    assert c.exchange_ratio == pytest.approx(32.50 / 60.0)
    assert c.new_debt == pytest.approx(1625.0)


def test_pro_forma_earnings_match_hand_arithmetic(hand_case):
    r, _, _, _ = hand_case
    a = r.accretion
    assert a.pro_forma_shares == pytest.approx(527.08333, rel=1e-6)
    assert a.pro_forma_net_income == pytest.approx(1239.9, rel=1e-6)
    assert a.acquirer_eps_standalone == pytest.approx(2.00)
    exact_eps = 74_394 / 31_625
    assert a.pro_forma_eps == pytest.approx(exact_eps, rel=1e-12)
    assert a.accretion_dollars == pytest.approx(exact_eps - 2.0, rel=1e-12)
    assert a.accretion_pct == pytest.approx((exact_eps - 2.0) / 2.0, rel=1e-12)


def test_components_sum_to_pro_forma_net_income(hand_case):
    r, _, _, _ = hand_case
    assert sum(v for _, v in r.accretion.components) == pytest.approx(
        r.accretion.pro_forma_net_income
    )


def test_breakeven_synergies_match_hand_arithmetic(hand_case):
    r, _, _, _ = hand_case
    assert r.accretion.breakeven_synergies == pytest.approx(-94.386, rel=1e-3)


def test_breakeven_synergies_actually_break_even(hand_case):
    """Substituting the solved figure must return pro forma EPS to standalone."""
    r, a, acq, tgt = hand_case
    a.merger.synergies.pretax_cost_synergies = r.accretion.breakeven_synergies
    acq_bridge = build_ev_bridge(acq, 60.0, a)
    tgt_bridge = build_ev_bridge(tgt, 25.0, a)
    rerun = run_merger(acq, tgt, acq_bridge, tgt_bridge, 60.0, 25.0, a)
    assert rerun.accretion.pro_forma_eps == pytest.approx(
        rerun.accretion.acquirer_eps_standalone, rel=1e-9
    )


def test_sources_and_uses_balance(hand_case):
    r, _, _, _ = hand_case
    uses = sum(v for side, _, v in r.sources_uses if side == "Use")
    sources = sum(v for side, _, v in r.sources_uses if side == "Source")
    assert uses == pytest.approx(sources)


# --------------------------------------------------------------------------- #
# the rule of thumb
# --------------------------------------------------------------------------- #


def test_all_stock_deal_is_accretive_when_the_acquirer_multiple_is_higher(hand_case):
    """A 30.0x acquirer paying 16.25x in paper must be accretive before synergies."""
    _, a, acq, tgt = hand_case
    a.merger.pct_cash = 0.0
    a.merger.synergies.pretax_cost_synergies = 0.0
    acq_bridge = build_ev_bridge(acq, 60.0, a)
    tgt_bridge = build_ev_bridge(tgt, 25.0, a)
    r = run_merger(acq, tgt, acq_bridge, tgt_bridge, 60.0, 25.0, a)

    assert r.accretion.accretion_dollars > 0
    assert any("stock" in c.lower() and "accretive" in c.lower() for c in r.checks)


def test_all_stock_deal_is_dilutive_when_the_acquirer_multiple_is_lower(hand_case):
    """Reverse the multiples and the sign of the answer must reverse with them."""
    _, a, acq, tgt = hand_case
    a.merger.pct_cash = 0.0
    a.merger.synergies.pretax_cost_synergies = 0.0
    a.merger.offer_premium = 0.30

    # Acquirer now trades at 10.0x, paying 16.25x for the target.
    low_price = 20.0
    acq_bridge = build_ev_bridge(acq, low_price, a)
    tgt_bridge = build_ev_bridge(tgt, 25.0, a)
    r = run_merger(acq, tgt, acq_bridge, tgt_bridge, low_price, 25.0, a)
    assert r.accretion.accretion_dollars < 0


def test_cash_deal_turns_on_the_after_tax_cost_of_the_cash(hand_case):
    """Accretive while the target's earnings yield beats the after-tax coupon."""
    _, a, acq, tgt = hand_case
    a.merger.pct_cash = 1.0
    a.merger.synergies.pretax_cost_synergies = 0.0

    # Target yields 200 / 3,250 = 6.15%. Debt at 4% costs 3.04% after tax.
    a.merger.cost_of_new_debt = 0.04
    cheap = run_merger(
        acq, tgt, build_ev_bridge(acq, 60.0, a), build_ev_bridge(tgt, 25.0, a),
        60.0, 25.0, a,
    )
    assert cheap.accretion.accretion_dollars > 0

    # At 12% the after-tax coupon is 9.12%, well above the target's yield.
    a.merger.cost_of_new_debt = 0.12
    dear = run_merger(
        acq, tgt, build_ev_bridge(acq, 60.0, a), build_ev_bridge(tgt, 25.0, a),
        60.0, 25.0, a,
    )
    assert dear.accretion.accretion_dollars < 0


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #


def test_percentage_is_withheld_when_the_base_is_near_zero(hand_case):
    """Four cents of dilution on four cents of EPS is not minus eight hundred percent."""
    _, a, acq, tgt = hand_case
    thin = replace(acq, net_income=20.0)
    r = run_merger(
        thin, tgt, build_ev_bridge(thin, 60.0, a), build_ev_bridge(tgt, 25.0, a),
        60.0, 25.0, a,
    )
    assert r.accretion.acquirer_eps_standalone < 0.10
    assert r.accretion.accretion_pct is None
    assert r.accretion.accretion_dollars is not None
    assert any("cents" in n or "floor" in n for n in r.accretion.notes)


def test_percentage_is_withheld_when_the_acquirer_loses_money(hand_case):
    _, a, acq, tgt = hand_case
    loss = replace(acq, net_income=-300.0)
    r = run_merger(
        loss, tgt, build_ev_bridge(loss, 60.0, a), build_ev_bridge(tgt, 25.0, a),
        60.0, 25.0, a,
    )
    assert r.accretion.accretion_pct is None


def test_contribution_percentages_are_withheld_on_a_negative_metric(hand_case):
    """A share of a combined loss ranks the bigger loss as the bigger contributor."""
    _, a, acq, tgt = hand_case
    loss = replace(acq, ebit=-400.0)
    r = run_merger(
        loss, tgt, build_ev_bridge(loss, 60.0, a), build_ev_bridge(tgt, 25.0, a),
        60.0, 25.0, a,
    )
    ebit_row = next(row for row in r.contribution if row.metric == "EBIT")
    assert ebit_row.acquirer_pct is None
    assert ebit_row.target_pct is None
    assert ebit_row.acquirer == pytest.approx(-400.0)


def test_contribution_is_shown_where_both_sides_are_positive(hand_case):
    r, _, _, _ = hand_case
    rev = next(row for row in r.contribution if row.metric == "Revenue")
    assert rev.acquirer_pct == pytest.approx(8000 / 9500)
    assert rev.target_pct == pytest.approx(1500 / 9500)
    assert rev.acquirer_pct + rev.target_pct == pytest.approx(1.0)


def test_ownership_reflects_the_shares_actually_issued(hand_case):
    r, _, _, _ = hand_case
    c, a = r.consideration, r.accretion
    assert r.pro_forma_ownership["target"] == pytest.approx(
        c.new_shares_issued / a.pro_forma_shares
    )
    assert sum(r.pro_forma_ownership.values()) == pytest.approx(1.0)


def test_offer_price_and_premium_are_mutually_exclusive(hand_case):
    _, a, acq, tgt = hand_case
    a.merger.offer_price_per_share = 32.50
    a.merger.offer_premium = 0.30
    with pytest.raises(ConfigError):
        run_merger(
            acq, tgt, build_ev_bridge(acq, 60.0, a), build_ev_bridge(tgt, 25.0, a),
            60.0, 25.0, a,
        )


def test_one_of_offer_price_or_premium_is_required(hand_case):
    _, a, acq, tgt = hand_case
    a.merger.offer_price_per_share = None
    a.merger.offer_premium = None
    with pytest.raises(ConfigError):
        run_merger(
            acq, tgt, build_ev_bridge(acq, 60.0, a), build_ev_bridge(tgt, 25.0, a),
            60.0, 25.0, a,
        )


def test_cash_the_acquirer_does_not_have_is_refused(hand_case):
    """Funding is a real constraint, not a slider.

    A sources and uses table that spends more cash than the balance sheet holds
    balances arithmetically and is fiction.
    """
    _, a, acq, tgt = hand_case
    broke = replace(acq, cash=50.0, short_term_investments=0.0)
    a.merger.balance_sheet_cash_used = 1000.0
    with pytest.raises(ConfigError) as exc:
        run_merger(
            broke, tgt, build_ev_bridge(broke, 60.0, a), build_ev_bridge(tgt, 25.0, a),
            60.0, 25.0, a,
        )
    assert "1,000" in str(exc.value)


def test_balance_sheet_cash_reduces_new_debt_and_costs_the_foregone_yield(hand_case):
    _, a, acq, tgt = hand_case
    a.merger.balance_sheet_cash_used = 1000.0
    r = run_merger(
        acq, tgt, build_ev_bridge(acq, 60.0, a), build_ev_bridge(tgt, 25.0, a),
        60.0, 25.0, a,
    )
    assert r.consideration.new_debt == pytest.approx(625.0)
    assert r.consideration.cash_used == pytest.approx(1000.0)
    labels = [lab for lab, _ in r.accretion.components]
    assert any("foregone" in lab.lower() for lab in labels)
