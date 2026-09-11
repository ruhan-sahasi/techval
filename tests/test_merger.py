"""Accretion/(dilution), checked against arithmetic done by hand."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from techval.config import Assumptions
from techval.errors import ConfigError, MissingDataError
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


# --------------------------------------------------------------------------- #
# pro forma leverage
# --------------------------------------------------------------------------- #


def test_cash_spent_on_the_deal_raises_pro_forma_net_debt(hand_case):
    """Cash that leaves the balance sheet is not a reduction in net debt.

    Post-deal debt is the two companies' debt plus new borrowing; post-deal cash
    is their cash less what was spent. Net debt is the difference, which puts
    cash used on the same side as new debt. Netting it off instead understates
    leverage by twice the cash spent, and understates it most on exactly the
    deals that should worry a reader.

    Isolation: an acquirer holding 3,000mm of cash and no debt buys a debt-free,
    cash-free target for 1,000mm of that cash. The combined company must hold
    2,000mm of net cash.
    """
    _, a, _, _ = hand_case
    acq = _company(
        "ACQ", revenue=8000, ebit=1400, da=200, net_income=1000, shares=500, cash=3000
    )
    tgt = _company("TGT", revenue=1500, ebit=280, da=60, net_income=200, shares=100)

    a.merger.offer_premium = None
    a.merger.offer_price_per_share = 10.0        # 100mm shares -> 1,000mm price
    a.merger.pct_cash = 1.0
    a.merger.balance_sheet_cash_used = 1000.0
    a.merger.deal_fees = 0.0
    a.merger.financing_fees = 0.0
    a.merger.synergies.pretax_cost_synergies = 0.0

    r = run_merger(
        acq, tgt, build_ev_bridge(acq, 60.0, a), build_ev_bridge(tgt, 25.0, a),
        60.0, 25.0, a,
    )
    assert r.consideration.new_debt == pytest.approx(0.0)
    assert r.consideration.cash_used == pytest.approx(1000.0)

    leverage = next(c for c in r.checks if "Pro forma" in c and "EBITDA" in c)
    assert "net cash of 2,000mm" in leverage


def test_zero_cost_cash_does_not_crash_the_run(hand_case):
    """A cash leg assumed to cost nothing has no crossover offer to quote."""
    _, a, acq, tgt = hand_case
    a.merger.pct_cash = 1.0
    a.merger.cost_of_new_debt = 0.0
    a.merger.foregone_cash_yield = 0.0
    a.merger.balance_sheet_cash_used = 0.0

    r = run_merger(
        acq, tgt, build_ev_bridge(acq, 60.0, a), build_ev_bridge(tgt, 25.0, a),
        60.0, 25.0, a,
    )
    assert any("cost nothing" in c for c in r.checks)


# --------------------------------------------------------------------------- #
# purchase accounting and the multi-year pro forma
# --------------------------------------------------------------------------- #


def _rerun(acq, tgt, a):
    return run_merger(
        acq, tgt, build_ev_bridge(acq, 60.0, a), build_ev_bridge(tgt, 25.0, a),
        60.0, 25.0, a,
    )


@pytest.fixture
def pa_case(hand_case):
    """Full purchase accounting on the same hand-computed deal.

    The target gets a balance sheet: 1,000mm of current assets against 400mm of
    current liabilities, no debt, no preferred and no minority interest, so the
    book equity acquired is 600mm.

        equity purchase price                       3,250.0
        less book equity acquired                     600.0
        excess over book                            2,650.0
        intangibles        40% x 2,650.0    =       1,060.0
        deferred tax       24% x 1,060.0    =         254.4
        goodwill   2,650.0 - 1,060.0 + 254.4 =      1,844.4

    Reconciliation: 600.0 + 1,060.0 + 1,844.4 - 254.4 = 3,250.0, the price paid.

    Intangible amortisation is 1,060.0 / 7 = 151.428571 a year.

    Year one carries no growth, so it is the trailing twelve months combined:

        revenue                              9,500.0
        EBITDA  1,940.0 + 150.0 synergies    2,090.0
        less D&A                               260.0
        less intangible amortisation           151.428571
        EBIT                                 1,678.571429
        non-operating (1,200/0.76 - 1,680)    -101.052632
        interest  1,625.0 x 6.0%               -97.5
        pre-tax                              1,480.018797
        tax at 24%                             355.204511
        net income                           1,124.814286

    Which is the headline pro forma net income of 1,239.9 less the after-tax
    amortisation of 151.428571 x 0.76 = 115.085714. The two tie because these
    synthetic filers are taxed at exactly the marginal rate; a real filer's
    effective rate will not reproduce it.

        free cash flow  1,124.814286 + 115.085714 + 260.0 = 1,499.9
        swept at 50%                                          749.95
        acquisition debt at end of year one                   875.05
        interest in year two  875.05 x 6.0%                    52.503
    """
    _, a, acq, tgt = hand_case
    tgt = replace(tgt, current_assets=1000.0, current_liabilities=400.0)
    pa = a.merger.purchase_accounting
    pa.enabled = True
    pa.intangible_pct_of_excess = 0.40
    pa.intangible_life_years = 7
    pa.deferred_revenue_haircut = 0.40
    pa.projection_years = 3
    pa.debt_repayment_pct_of_fcf = 0.50
    result = _rerun(acq, tgt, a)
    assert result.purchase_accounting is not None
    return result, a, acq, tgt


def test_opening_balance_sheet_matches_hand_arithmetic(pa_case):
    r, _, _, _ = pa_case
    o = r.purchase_accounting.opening
    assert o.equity_purchase_price == pytest.approx(3250.0)
    assert o.target_book_equity == pytest.approx(600.0)
    assert o.excess_purchase_price == pytest.approx(2650.0)
    assert o.intangibles_created == pytest.approx(1060.0)
    assert o.deferred_tax_liability == pytest.approx(254.4)
    assert o.goodwill_created == pytest.approx(1844.4)


def test_assets_acquired_reconcile_to_consideration_paid(pa_case):
    """The identity purchase accounting exists to satisfy.

    Book equity plus the write-up plus goodwill, less the deferred tax liability
    assumed against the write-up, is the price paid. Nothing else balances.
    """
    r, _, _, _ = pa_case
    o = r.purchase_accounting.opening
    acquired = (
        o.target_book_equity
        + o.intangibles_created
        + o.goodwill_created
        - o.deferred_tax_liability
    )
    assert acquired == pytest.approx(o.equity_purchase_price, rel=1e-12)


def test_the_deferred_tax_liability_increases_goodwill(pa_case):
    """The step most models miss, asserted in the direction people get wrong.

    A stock deal carries over the seller's tax basis, so the write-up is a book
    entry with no tax deduction behind it. The liability that represents the
    difference is assumed at close, which takes identifiable net assets down and
    pushes goodwill up. Booking it the other way, or omitting it, leaves goodwill
    short by the tax on the write-up and the opening balance sheet does not
    reconcile.
    """
    r, a, _, _ = pa_case
    o = r.purchase_accounting.opening
    rate = a.tax.marginal_tax_rate

    assert o.deferred_tax_liability == pytest.approx(o.intangibles_created * rate)
    assert o.goodwill_created == pytest.approx(
        o.excess_purchase_price - o.intangibles_created + o.deferred_tax_liability
    )

    naive_goodwill = o.excess_purchase_price - o.intangibles_created
    assert o.goodwill_created > naive_goodwill
    naive_acquired = (
        o.target_book_equity
        + o.intangibles_created
        + naive_goodwill
        - o.deferred_tax_liability
    )
    assert naive_acquired == pytest.approx(
        o.equity_purchase_price - o.deferred_tax_liability
    )


def test_goodwill_is_not_amortised_but_intangibles_are(pa_case):
    """ASC 350: goodwill is impairment tested, never amortised.

    The charge in every year is the write-up over its life and nothing else, so
    no part of the 1,844.4mm of goodwill reaches EPS.
    """
    r, _, _, _ = pa_case
    pa = r.purchase_accounting
    annual = pa.opening.intangibles_created / 7
    assert annual == pytest.approx(1060.0 / 7)
    for year in pa.years:
        assert year.intangible_amortisation == pytest.approx(annual)
        assert year.intangible_amortisation < pa.opening.goodwill_created / 7


def test_amortisation_stops_when_the_asset_is_written_off(pa_case):
    """A two year life on a four year projection leaves two clean years."""
    _, a, acq, tgt = pa_case
    a.merger.purchase_accounting.intangible_life_years = 2
    a.merger.purchase_accounting.projection_years = 4
    pa = _rerun(acq, tgt, a).purchase_accounting

    charges = [y.intangible_amortisation for y in pa.years]
    assert charges[0] == pytest.approx(pa.opening.intangibles_created / 2)
    assert charges[1] == pytest.approx(pa.opening.intangibles_created / 2)
    assert charges[2] == 0.0
    assert charges[3] == 0.0
    assert sum(charges) == pytest.approx(pa.opening.intangibles_created)


def test_year_one_is_the_headline_less_the_after_tax_amortisation(pa_case):
    """The pro forma and the screen differ by exactly the purchase accounting.

    With no deferred revenue to haircut, the only thing the opening balance sheet
    adds to year one is the amortisation, and it is a book charge carrying a full
    deferred tax benefit as the liability unwinds. So it costs ``amortisation x
    (1 - t)`` of net income and nothing more.
    """
    r, a, _, _ = pa_case
    first = r.purchase_accounting.years[0]
    after_tax_charge = first.intangible_amortisation * (1 - a.tax.marginal_tax_rate)

    assert first.revenue == pytest.approx(9500.0)
    assert first.ebitda == pytest.approx(2090.0)
    assert first.interest == pytest.approx(97.5)
    assert first.net_income == pytest.approx(
        r.accretion.pro_forma_net_income - after_tax_charge, rel=1e-12
    )
    assert first.eps == pytest.approx(
        r.accretion.pro_forma_eps - after_tax_charge / first.shares, rel=1e-12
    )


def test_free_cash_flow_adds_back_only_the_after_tax_amortisation(pa_case):
    """The deferred tax benefit inside net income is not cash.

    In a stock deal the amortisation is never deducted on a tax return, so the
    full add-back overstates cash generation by the tax on it, and overstates how
    fast the acquisition debt comes down.
    """
    r, a, _, _ = pa_case
    rate = a.tax.marginal_tax_rate
    first = r.purchase_accounting.years[0]

    assert first.cash_flow == pytest.approx(1499.9)
    full_addback = first.net_income + first.intangible_amortisation + 260.0
    assert first.cash_flow == pytest.approx(
        full_addback - rate * first.intangible_amortisation
    )
    assert first.cash_flow < full_addback


def test_deferred_revenue_haircut_destroys_year_one_revenue(pa_case):
    """Written down to fair value, and the difference is never recognised.

    500mm of acquired deferred revenue at a 40% haircut removes 200mm from the
    first year and its margin with it. Year two is clean: the balance is written
    down once, at close.
    """
    _, a, acq, tgt = pa_case
    billed = replace(tgt, deferred_revenue=500.0)

    a.merger.purchase_accounting.deferred_revenue_haircut = 0.0
    clean = _rerun(acq, billed, a).purchase_accounting
    a.merger.purchase_accounting.deferred_revenue_haircut = 0.40
    cut = _rerun(acq, billed, a).purchase_accounting

    assert cut.opening.deferred_revenue_haircut == pytest.approx(200.0)
    assert cut.years[0].revenue == pytest.approx(clean.years[0].revenue - 200.0)
    assert cut.years[1].revenue == pytest.approx(clean.years[1].revenue)

    margin = 1940.0 / 9500.0
    assert cut.years[0].ebitda == pytest.approx(
        clean.years[0].ebitda - 200.0 * margin
    )
    assert cut.years[0].eps < clean.years[0].eps


def test_the_sweep_pays_down_debt_and_interest_falls_with_it(pa_case):
    """Interest is charged on the balance at the start of each year.

    Sweeping out of cash the year has not generated yet would credit the deal
    with a repayment before the money arrives.
    """
    r, a, _, _ = pa_case
    years = r.purchase_accounting.years
    rate = a.merger.cost_of_new_debt

    assert years[0].interest == pytest.approx(1625.0 * rate)
    assert years[0].debt_repaid == pytest.approx(0.5 * years[0].cash_flow)
    assert years[0].debt_balance == pytest.approx(1625.0 - years[0].debt_repaid)
    assert years[1].interest == pytest.approx(years[0].debt_balance * rate)
    assert years[1].interest < years[0].interest

    balances = [y.debt_balance for y in years]
    assert balances == sorted(balances, reverse=True)
    assert min(balances) >= 0.0


def test_the_sweep_never_repays_more_than_is_outstanding(pa_case):
    """A full sweep retires the facility and then stops.

    Year one generates 1,499.9 against a 1,625.0 facility, so all of it goes to
    the debt and 125.1 is left. Year two throws off far more than that and repays
    only the 125.1 outstanding, after which there is nothing left to service.
    """
    _, a, acq, tgt = pa_case
    a.merger.purchase_accounting.debt_repayment_pct_of_fcf = 1.0
    pa = _rerun(acq, tgt, a).purchase_accounting

    assert pa.years[0].debt_repaid == pytest.approx(1499.9)
    assert pa.years[0].debt_balance == pytest.approx(125.1)
    assert pa.years[1].cash_flow > 125.1
    assert pa.years[1].debt_repaid == pytest.approx(125.1)
    assert pa.years[1].debt_balance == pytest.approx(0.0)
    assert pa.years[2].interest == pytest.approx(0.0)
    assert pa.years[2].debt_repaid == pytest.approx(0.0)


def test_the_crossover_year_is_found_and_named(pa_case):
    """A one year life makes year one dilutive and year two clean.

    The whole excess is written up and amortised inside the first year, so the
    deal is heavily dilutive on it and the charge is gone thereafter. That is the
    clearest case of a crossover there is, and the check has to name the year.
    """
    _, a, acq, tgt = pa_case
    a.merger.purchase_accounting.intangible_pct_of_excess = 1.0
    a.merger.purchase_accounting.intangible_life_years = 1
    pa = _rerun(acq, tgt, a).purchase_accounting

    assert pa.years[0].intangible_amortisation == pytest.approx(2650.0)
    assert pa.years[1].intangible_amortisation == 0.0

    by_year = pa.accretion_by_year
    assert by_year[0][1] < 0
    assert by_year[1][1] > 0
    assert any("year 2" in c for c in pa.checks)


def test_accretion_by_year_carries_the_headline_unit_test(pa_case):
    """Cents always, percent only over a base worth dividing by.

    The headline refuses a percentage below ten cents of standalone EPS. The
    yearly figures apply the same test, so the two cannot disagree about what
    they are quoting.
    """
    r, a, acq, tgt = pa_case
    assert all(pct is not None for _, _, pct in r.purchase_accounting.accretion_by_year)

    thin = replace(acq, net_income=20.0)
    pa = _rerun(thin, tgt, a).purchase_accounting
    assert all(pct is None for _, _, pct in pa.accretion_by_year)
    assert all(isinstance(dollars, float) for _, dollars, _ in pa.accretion_by_year)


def test_net_leverage_is_reported_at_close_and_at_the_horizon(pa_case):
    """Cash spent and cash generated both land in net debt.

    The acquirer holds 3,000mm of cash and borrows 1,625mm, so the combined
    company opens at 1,375mm of net cash. Every dollar of free cash flow takes
    net debt down whether it repays the facility or sits in the bank, which is
    why the sweep percentage moves interest expense and not leverage.
    """
    r, _, _, _ = pa_case
    pa = r.purchase_accounting
    leverage = next(c for c in pa.checks if "Pro forma net" in c)

    assert "net cash of 1,375mm at close" in leverage
    assert "-0.7x EBITDA" in leverage
    expected_horizon = -1375.0 - sum(y.cash_flow for y in pa.years)
    assert f"net cash of {-expected_horizon:,.0f}mm" in leverage
    assert "falling" in leverage


def test_book_equity_that_cannot_be_formed_is_refused(pa_case):
    """No balance sheet, no goodwill. The whole price is not a plug."""
    _, a, acq, tgt = pa_case
    blind = replace(tgt, current_assets=None)
    with pytest.raises(MissingDataError) as exc:
        _rerun(acq, blind, a)
    assert "current assets" in str(exc.value)
    assert "book equity" in str(exc.value)


def test_a_price_below_book_is_a_bargain_purchase_not_negative_amortisation(pa_case):
    """Nothing is written up when there is no excess, and ASC 805 is named."""
    _, a, acq, tgt = pa_case
    rich = replace(tgt, current_assets=10_000.0, current_liabilities=400.0)
    pa = _rerun(acq, rich, a).purchase_accounting

    o = pa.opening
    assert o.target_book_equity == pytest.approx(9600.0)
    assert o.excess_purchase_price == pytest.approx(-6350.0)
    assert o.intangibles_created == 0.0
    assert o.deferred_tax_liability == 0.0
    assert o.goodwill_created == pytest.approx(-6350.0)
    assert all(y.intangible_amortisation == 0.0 for y in pa.years)
    assert any("805" in n for n in pa.notes)


def test_purchase_accounting_off_changes_nothing(hand_case):
    """The switch adds a result. It must not restate one.

    Everything the screen produces is compared line by line between a run with
    purchase accounting off and the same run with it on: the consideration, the
    earnings walk, the contribution, the ownership split, sources and uses, the
    sensitivity grid and every check. Only the closing note differs, and only
    because it is the note that says none of this is modelled.
    """
    off, a, acq, tgt = hand_case
    tgt = replace(tgt, current_assets=1000.0, current_liabilities=400.0)
    off = _rerun(acq, tgt, a)
    assert off.purchase_accounting is None

    a.merger.purchase_accounting.enabled = True
    on = _rerun(acq, tgt, a)
    assert on.purchase_accounting is not None

    assert on.consideration == off.consideration
    assert on.accretion == off.accretion
    assert on.contribution == off.contribution
    assert on.pro_forma_ownership == off.pro_forma_ownership
    assert on.sources_uses == off.sources_uses
    assert on.checks == off.checks
    pd.testing.assert_frame_equal(on.sensitivity, off.sensitivity)

    assert on.notes[:-1] == off.notes[:-1]
    assert off.notes[-1].startswith("Not modelled")
    assert on.notes[-1].startswith("Purchase accounting is on")
