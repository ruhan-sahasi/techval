"""Tests for the point-in-time feature store.

Everything here runs against the four committed fact payloads and their price
CSVs, so no test touches the network.

Two properties of the fixtures shape what can be asserted. The payloads were
pruned to periods ending on or after 2023-01-01 and, for each period, to the
version of the fact that was filed most recently. A knowledge date in early 2025
therefore sees a shorter history than it really would have, because the 2024
comparatives on file at the time were superseded by 10-Qs filed in 2025 and only
the later version survives in the fixture. The effect is conservative: the engine
under test sees less than it would in production, never more, so a test that
passes here cannot be hiding a leak. It does mean the three-year growth features
are legitimately unavailable at most fixture dates, and the tests assert that
they come back None rather than pretending otherwise.

The load-bearing tests in this file are the three that try to break the
point-in-time guarantee: the reporting-lag case, the seeded provenance violation
where a fact set claims a knowledge date it does not honour, and the uncapped
price feed. The rest is arithmetic. Those three are the reason anything fitted on
this store counts as evidence.
"""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from techval.config import Assumptions
from techval.edgar import CompanyFacts, HttpCache
from techval.errors import ConfigError, MissingDataError
from techval.financials import build_financials
from techval.market import CsvSource, MarketData
from techval.ml.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    GROUP_OF,
    MATRIX_COLUMNS,
    MISSING_INDICATORS,
    WINSOR_MIN_OBSERVATIONS,
    FeaturePanel,
    FeatureRow,
    LookaheadError,
    assert_point_in_time,
    build_features,
    build_panel,
    winsorize,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
PRICES = FIXTURES / "prices"

# Dates chosen inside the window the fixtures cover. The price CSVs run
# 2023-09-01 to 2026-09-09 and the earliest filing in any payload is 2024-03-06,
# so a knowledge date before that presents as a company that had not filed.
D_2025_JAN = date(2025, 1, 15)
D_2025_MAR = date(2025, 3, 3)
D_2025_SEP = date(2025, 9, 2)
D_2026_MAR = date(2026, 3, 2)
D_2026_SEP = date(2026, 9, 10)

TICKERS = ["DDOG", "CRWD", "MDB", "ZS"]


def _payload(ticker: str) -> dict:
    return json.loads((FIXTURES / f"companyfacts_{ticker.upper()}.json").read_text())


class PinnedClient:
    """A fixture client that honours a knowledge date, as ``EdgarClient`` does."""

    def __init__(self, knowledge_date: date) -> None:
        self.knowledge_date = knowledge_date

    def company_facts(self, ticker: str) -> CompanyFacts:
        return CompanyFacts(
            _payload(ticker), ticker, knowledge_date=self.knowledge_date
        )


class UnpinnedClient:
    """The mistake this store exists to refuse: a client with no knowledge date."""

    def company_facts(self, ticker: str) -> CompanyFacts:
        return CompanyFacts(_payload(ticker), ticker)


class _LyingFacts(CompanyFacts):
    """Claims a knowledge date and then ignores it.

    This is the seeded violation. It passes every configuration check the store
    makes, because the attribute is set and is not in the future, and it then
    serves facts filed years later. Only the walk over the provenance of what
    actually reached the row can catch it, which is the point of having one.
    """

    def facts(self, tag: str):
        pinned, self.knowledge_date = self.knowledge_date, None
        try:
            return super().facts(tag)
        finally:
            self.knowledge_date = pinned


class LyingClient:
    def __init__(self, knowledge_date: date) -> None:
        self.knowledge_date = knowledge_date

    def company_facts(self, ticker: str) -> CompanyFacts:
        return _LyingFacts(
            _payload(ticker), ticker, knowledge_date=self.knowledge_date
        )


def _market(when: date, history_years: float = 3.0) -> MarketData:
    return MarketData(
        CsvSource(PRICES),
        HttpCache(enabled=False),
        today=when,
        history_years=history_years,
    )


@pytest.fixture
def ml_assumptions() -> Assumptions:
    """Defaults with the risk-free rate pinned so nothing reaches the Treasury."""
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    return a


def _row(ticker: str, when: date, assumptions: Assumptions, **kw) -> FeatureRow:
    return build_features(
        ticker, when, PinnedClient(when), _market(when), assumptions, **kw
    )


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


def test_feature_names_are_the_flattened_groups_in_order():
    flattened = tuple(n for g in FEATURE_GROUPS.values() for n in g)
    assert FEATURE_NAMES == flattened
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert 40 <= len(FEATURE_NAMES) <= 50


def test_every_feature_belongs_to_exactly_one_group():
    assert set(GROUP_OF) == set(FEATURE_NAMES)
    for group, names in FEATURE_GROUPS.items():
        for name in names:
            assert GROUP_OF[name] == group
            # The name carries its group, so an ablation table read on its own
            # still says which block a feature came from.
            assert name.startswith(group + "_")


def test_matrix_columns_are_the_features_then_one_indicator_per_group():
    assert MATRIX_COLUMNS == FEATURE_NAMES + MISSING_INDICATORS
    assert len(MISSING_INDICATORS) == len(FEATURE_GROUPS)


def test_a_row_carries_every_declared_feature_and_nothing_else(ml_assumptions):
    row = _row("DDOG", D_2026_SEP, ml_assumptions)
    assert list(row.values) == list(FEATURE_NAMES)


def test_groups_are_populated_on_a_well_covered_filer(ml_assumptions):
    """Every group should produce at least one real number for a clean filer."""
    row = _row("DDOG", D_2026_SEP, ml_assumptions)
    for group, names in FEATURE_GROUPS.items():
        observed = [n for n in names if row.values[n] is not None]
        assert observed, f"{group} produced nothing for DDOG at {D_2026_SEP}"


# --------------------------------------------------------------------------- #
# Point in time: the reporting lag
# --------------------------------------------------------------------------- #


def test_a_january_row_does_not_contain_the_december_quarter(ml_assumptions):
    """The lag test. DDOG runs a calendar fiscal year.

    On 15 January 2025 the December 2024 quarter had happened but had not been
    filed, so nobody could see it. The row must anchor on September. By 3 March
    the annual report carrying December is on file and the row moves.
    """
    january = _row("DDOG", D_2025_JAN, ml_assumptions)
    assert january.statement_date == date(2024, 9, 30)
    assert january.statement_date < date(2024, 12, 31)

    march = _row("DDOG", D_2025_MAR, ml_assumptions)
    assert march.statement_date == date(2024, 12, 31)


def test_the_january_row_carries_the_september_revenue_not_the_december_one(
    ml_assumptions,
):
    """Not just a different date label: a different number.

    Trailing revenue to September 2024 is 2,536mm and to December 2024 is
    2,684mm. A row dated 15 January that reported the second figure would be
    trading on an unfiled quarter, and the log-scale size term is where that
    would show up first.
    """
    january = _row("DDOG", D_2025_JAN, ml_assumptions)
    march = _row("DDOG", D_2025_MAR, ml_assumptions)

    sep_ttm = np.exp(january.values["scale_log_revenue"])
    dec_ttm = np.exp(march.values["scale_log_revenue"])
    assert sep_ttm == pytest.approx(2536.2, abs=0.5)
    assert dec_ttm == pytest.approx(2684.3, abs=0.5)
    assert sep_ttm < dec_ttm


def test_the_knowledge_date_is_recorded_on_the_row(ml_assumptions):
    row = _row("ZS", D_2026_SEP, ml_assumptions)
    assert row.knowledge_date == D_2026_SEP
    assert row.as_of == D_2026_SEP


def test_no_provenance_on_a_clean_row_postdates_the_row(ml_assumptions):
    """The guarantee, checked directly rather than through the helper."""
    row = _row("CRWD", D_2025_SEP, ml_assumptions)
    filed = [
        date.fromisoformat(p.filed) for p in row.provenance.values() if p.filed
    ]
    assert filed, "the row recorded no filing dates at all, so it proves nothing"
    assert max(filed) <= D_2025_SEP


# --------------------------------------------------------------------------- #
# Point in time: the guards
# --------------------------------------------------------------------------- #


def test_an_unpinned_client_is_refused(ml_assumptions):
    with pytest.raises(LookaheadError, match="not pinned"):
        build_features(
            "DDOG", D_2025_JAN, UnpinnedClient(), _market(D_2025_JAN), ml_assumptions
        )


def test_a_client_pinned_after_the_row_date_is_refused(ml_assumptions):
    with pytest.raises(LookaheadError, match="days later"):
        build_features(
            "DDOG",
            D_2025_JAN,
            PinnedClient(D_2026_SEP),
            _market(D_2025_JAN),
            ml_assumptions,
        )


def test_a_client_pinned_earlier_is_allowed_but_says_so(ml_assumptions):
    """Conservative, not leaking. The row is built and the gap is on the record."""
    earlier = D_2025_SEP - timedelta(days=30)
    row = build_features(
        "DDOG", D_2025_SEP, PinnedClient(earlier), _market(earlier), ml_assumptions
    )
    assert row.knowledge_date == earlier
    assert any("conservative rather than leaking" in n for n in row.notes)


def test_an_uncapped_market_feed_is_refused(ml_assumptions):
    with pytest.raises(LookaheadError, match="today="):
        build_features(
            "DDOG",
            D_2025_JAN,
            PinnedClient(D_2025_JAN),
            _market(D_2026_SEP),
            ml_assumptions,
        )


def test_the_provenance_assertion_catches_a_seeded_violation(ml_assumptions):
    """A fact set that claims a knowledge date and serves later filings anyway.

    Every configuration check passes. The statements come back looking entirely
    ordinary. Only walking the filing dates behind what was actually used catches
    it, and this is the test that says the walk is real.
    """
    with pytest.raises(LookaheadError) as exc:
        build_features(
            "DDOG",
            D_2025_JAN,
            LyingClient(D_2025_JAN),
            _market(D_2025_JAN),
            ml_assumptions,
        )
    assert "after the valuation date" in str(exc.value)


def test_the_assertion_walks_this_modules_own_historical_provenance(ml_assumptions):
    """The statements alone are not the whole row.

    Growth and margin change come from trailing-twelve-month resolutions this
    module makes at earlier anchors, separately from ``build_financials``. A
    check that only walked ``Financials.provenance`` would leave those unproven,
    so the violation is seeded on one of them and must still be caught.
    """
    row = _row("DDOG", D_2026_SEP, ml_assumptions)
    historical = [k for k in row.provenance if k.endswith("t-1y")]
    assert historical, "no historical resolutions were recorded"

    seeded = copy.deepcopy(row)
    seeded.provenance[historical[0]].filed = "2027-01-04"
    with pytest.raises(LookaheadError, match="after the row date"):
        assert_point_in_time(seeded)


def test_the_assertion_catches_a_price_series_running_past_the_row(ml_assumptions):
    row = _row("DDOG", D_2025_JAN, ml_assumptions)
    late = _market(D_2026_SEP).prices("DDOG")
    with pytest.raises(LookaheadError, match="price series running to"):
        assert_point_in_time(row, prices=late)


def test_a_clean_row_passes_the_assertion(ml_assumptions):
    row = _row("MDB", D_2026_SEP, ml_assumptions)
    fin = build_financials(
        "MDB", facts=PinnedClient(D_2026_SEP).company_facts("MDB")
    )
    assert_point_in_time(row, fin, prices=_market(D_2026_SEP).prices("MDB"))


# --------------------------------------------------------------------------- #
# Missing is a value, not a zero
# --------------------------------------------------------------------------- #


def test_missing_features_stay_none_and_are_listed(ml_assumptions):
    """CrowdStrike tags amortisation of intangibles and no depreciation line.

    The engine therefore cannot build its EBITDA, and every ratio over EBITDA is
    unavailable. Zero would say the company converts no cash and carries no
    leverage, which is a specific claim and a false one.
    """
    row = _row("CRWD", D_2026_SEP, ml_assumptions)
    assert row.values["margin_ebitda"] is None
    assert row.values["returns_cash_conversion"] is None
    assert "margin_ebitda" in row.missing
    assert all(row.values[n] is None for n in row.missing)
    assert all(row.values[n] is not None for n in FEATURE_NAMES if n not in row.missing)


def test_missing_is_none_in_values_and_nan_only_in_the_matrix(ml_assumptions):
    row = _row("CRWD", D_2026_SEP, ml_assumptions)
    assert row.values["margin_ebitda"] is None

    panel = FeaturePanel(rows=[row])
    X, names, _ = panel.to_matrix()
    assert np.isnan(X[0, names.index("margin_ebitda")])
    # The row itself is untouched by having been rendered.
    assert row.values["margin_ebitda"] is None


def test_the_missing_indicator_is_the_share_of_its_group(ml_assumptions):
    row = _row("CRWD", D_2026_SEP, ml_assumptions)
    panel = FeaturePanel(rows=[row])
    X, names, _ = panel.to_matrix()
    for group, members in FEATURE_GROUPS.items():
        absent = sum(1 for n in members if row.values[n] is None)
        column = names.index(f"missing_share_{group}")
        assert X[0, column] == pytest.approx(absent / len(members))


def test_a_denominator_with_the_wrong_sign_is_nm_with_a_reason(ml_assumptions):
    """Not missing: not meaningful, which is a different fact and says so.

    CrowdStrike's cash and securities exceed its debt plus book equity, so
    invested capital is negative. A return on it would carry the sign of the
    denominator, and the note records that rather than leaving a reader to
    wonder why the field is blank.
    """
    row = _row("CRWD", D_2025_SEP, ml_assumptions)
    assert row.values["returns_roic_proxy"] is None
    assert any("returns_roic_proxy: NM" in n for n in row.notes)


def test_three_year_growth_is_none_rather_than_extrapolated(ml_assumptions):
    """The fixtures do not reach back three years, so the feature is unavailable.

    The alternative, annualising whatever history there is and labelling it a
    three-year rate, is the failure this whole store is built to avoid.
    """
    row = _row("DDOG", D_2026_SEP, ml_assumptions)
    assert row.values["growth_revenue_cagr_3y"] is None
    assert row.values["growth_revenue_1y"] is not None


def test_a_missing_market_feed_costs_the_market_group_and_nothing_else(
    ml_assumptions,
):
    row = build_features(
        "ZS", D_2026_SEP, PinnedClient(D_2026_SEP), None, ml_assumptions
    )
    for name in FEATURE_GROUPS["market"]:
        assert row.values[name] is None
    assert row.values["margin_gross"] is not None
    assert row.values["scale_log_revenue"] is not None
    # Ratios over market capitalisation go with the price, and are reported gone.
    assert row.values["scale_log_market_cap"] is None
    assert any("no market feed" in n for n in row.notes)


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


def test_margins_match_the_statements_they_came_from(ml_assumptions):
    row = _row("DDOG", D_2026_SEP, ml_assumptions)
    fin = build_financials(
        "DDOG", facts=PinnedClient(D_2026_SEP).company_facts("DDOG")
    )
    assert row.values["margin_gross"] == pytest.approx(fin.gross_margin)
    assert row.values["margin_ebit"] == pytest.approx(fin.ebit_margin)
    assert row.values["margin_net"] == pytest.approx(fin.net_income / fin.revenue)
    assert row.values["scale_log_revenue"] == pytest.approx(np.log(fin.revenue))


def test_growth_is_this_trailing_year_over_the_last_one(ml_assumptions):
    """Trailing twelve months against trailing twelve months a year earlier.

    Both built through the same knowledge date, so the comparison is against the
    prior year as it was reported then rather than as it was later restated.
    """
    row = _row("ZS", D_2026_SEP, ml_assumptions)
    fin = build_financials("ZS", facts=PinnedClient(D_2026_SEP).company_facts("ZS"))

    # Trailing revenue to 31 July 2026 is 3,352.5mm and to 31 July 2025 is
    # 2,673.1mm. The growth rate is those two and nothing else.
    assert fin.as_of == date(2026, 7, 31)
    assert fin.revenue == pytest.approx(3352.5, abs=0.5)
    implied_prior = fin.revenue / (1.0 + row.values["growth_revenue_1y"])
    assert implied_prior == pytest.approx(2673.1, abs=0.5)
    assert row.values["growth_revenue_1y"] == pytest.approx(0.254163, abs=1e-5)


def test_rule_of_40_is_growth_plus_free_cash_flow_margin(ml_assumptions):
    row = _row("DDOG", D_2026_SEP, ml_assumptions)
    assert row.values["returns_rule_of_40"] == pytest.approx(
        row.values["growth_revenue_1y"] + row.values["margin_fcf"]
    )


def test_acceleration_is_the_one_year_rate_less_the_two_year_rate(ml_assumptions):
    row = _row("ZS", D_2026_SEP, ml_assumptions)
    assert row.values["growth_acceleration"] == pytest.approx(
        row.values["growth_revenue_1y"] - row.values["growth_revenue_cagr_2y"]
    )


def _close_a_year_before(series) -> float:
    target = series.last_date - timedelta(days=round(365.25))
    last = None
    for d, c in zip(series.dates, series.closes):
        if d <= target:
            last = float(c)
    return last


def test_market_features_are_in_a_sane_range(ml_assumptions):
    row = _row("DDOG", D_2026_SEP, ml_assumptions)
    assert 0.0 <= row.values["market_52w_range_position"] <= 1.0
    assert 0.0 < row.values["market_realised_volatility_1y"] < 2.0
    assert 0.2 < row.values["market_beta"] < 4.0


def test_relative_momentum_is_the_stock_less_the_index(ml_assumptions):
    """The feature is excess return against SPY, not a second price return."""
    row = _row("DDOG", D_2026_SEP, ml_assumptions)
    spy = _market(D_2026_SEP).prices("SPY")
    index_return = spy.last / _close_a_year_before(spy) - 1.0

    assert row.values["market_relative_momentum_12m"] == pytest.approx(
        row.values["market_momentum_12m"] - index_return
    )
    assert row.values["market_relative_momentum_12m"] != pytest.approx(
        row.values["market_momentum_12m"]
    )


def test_capex_intensity_is_a_spend_rate_whatever_the_filed_sign(ml_assumptions):
    row = _row("DDOG", D_2026_SEP, ml_assumptions)
    assert row.values["efficiency_capex_to_revenue"] > 0.0


def test_the_two_volatility_features_need_a_real_quarterly_history(ml_assumptions):
    """Built from discrete quarters recovered by subtraction, not from TTM points."""
    row = _row("DDOG", D_2026_SEP, ml_assumptions)
    assert 0.0 < row.values["quality_revenue_volatility_3y"] < 0.5
    assert row.values["quality_margin_volatility_3y"] is not None


# --------------------------------------------------------------------------- #
# Tag ladders this module owns
# --------------------------------------------------------------------------- #


def test_the_research_and_sales_ladders_resolve_when_the_filer_reports_them():
    """The pruned fixtures carry no operating expense lines, so one is injected.

    Cost of revenue is cloned onto the research tag, which gives a real period
    structure and a known answer: research intensity must then equal one less the
    gross margin. That tests the ladder rather than the arithmetic of a stub.
    """
    payload = _payload("DDOG")
    gaap = payload["facts"]["us-gaap"]
    gaap["ResearchAndDevelopmentExpense"] = copy.deepcopy(
        gaap["CostOfGoodsAndServicesSold"]
    )

    class Injected:
        knowledge_date = D_2026_SEP

        def company_facts(self, ticker: str) -> CompanyFacts:
            return CompanyFacts(payload, ticker, knowledge_date=D_2026_SEP)

    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    row = build_features("DDOG", D_2026_SEP, Injected(), _market(D_2026_SEP), a)

    assert row.values["efficiency_rnd_to_revenue"] is not None
    assert row.values["efficiency_rnd_to_revenue"] == pytest.approx(
        1.0 - row.values["margin_gross"], abs=1e-9
    )


# --------------------------------------------------------------------------- #
# Winsorization
# --------------------------------------------------------------------------- #


def _synthetic_rows(when: date, values_by_ticker: dict[str, float]) -> list[FeatureRow]:
    """Rows carrying one known number in every feature, for testing the trim."""
    return [
        FeatureRow(
            ticker=t,
            as_of=when,
            knowledge_date=when,
            values={n: v for n in FEATURE_NAMES},
        )
        for t, v in values_by_ticker.items()
    ]


def test_winsorization_pulls_the_tails_and_counts_what_it_touched():
    rows = _synthetic_rows(
        date(2026, 1, 1), {f"T{i}": float(i) for i in range(100)}
    )
    trimmed, reports = winsorize(rows)

    applied = [r for r in reports if r.applied]
    assert len(applied) == len(FEATURE_NAMES)

    one = next(r for r in applied if r.feature == "margin_gross")
    assert one.n_observed == 100
    # With a hundred evenly spaced values the 1st and 99th percentiles are the
    # second and second-to-last, so exactly one name is pulled at each end.
    assert one.n_pulled_low == 1
    assert one.n_pulled_high == 1
    assert one.n_touched == 2
    assert trimmed[0].values["margin_gross"] == pytest.approx(one.lower)
    assert trimmed[-1].values["margin_gross"] == pytest.approx(one.upper)
    # Everything between the bounds is left exactly as it was.
    assert trimmed[50].values["margin_gross"] == rows[50].values["margin_gross"]


def test_winsorization_never_drops_a_row():
    rows = _synthetic_rows(date(2026, 1, 1), {f"T{i}": float(i) for i in range(100)})
    trimmed, _ = winsorize(rows)
    assert len(trimmed) == len(rows)
    assert [r.ticker for r in trimmed] == [r.ticker for r in rows]


def test_winsorization_leaves_the_original_rows_alone():
    rows = _synthetic_rows(date(2026, 1, 1), {f"T{i}": float(i) for i in range(100)})
    before = rows[0].values["margin_gross"]
    winsorize(rows)
    assert rows[0].values["margin_gross"] == before


def test_winsorization_is_cross_sectional_by_date_not_pooled():
    """Two dates on different scales must get their own bounds.

    Pooled bounds would send every name on the low date to the pooled 1st
    percentile, which is the failure this is arranged to expose: the trim would
    be reporting on the difference between two eras rather than on outliers
    inside either of them.
    """
    early = _synthetic_rows(date(2025, 1, 1), {f"A{i}": float(i) for i in range(100)})
    late = _synthetic_rows(
        date(2026, 1, 1), {f"B{i}": 1000.0 + i for i in range(100)}
    )
    trimmed, reports = winsorize(early + late)

    bounds = {
        (r.as_of, r.feature): (r.lower, r.upper)
        for r in reports
        if r.applied and r.feature == "margin_gross"
    }
    lo_early, hi_early = bounds[(date(2025, 1, 1), "margin_gross")]
    lo_late, hi_late = bounds[(date(2026, 1, 1), "margin_gross")]
    assert hi_early < lo_late
    # Nothing on the early date was pulled toward the late date's level.
    assert max(r.values["margin_gross"] for r in trimmed[:100]) < 100.0


def test_a_thin_cross_section_is_reported_as_skipped_not_trimmed():
    """Below the threshold a 1st percentile describes the sample size, not the data."""
    rows = _synthetic_rows(date(2026, 1, 1), {"A": 1.0, "B": 2.0, "C": 3.0, "D": 400.0})
    trimmed, reports = winsorize(rows)

    assert all(not r.applied for r in reports)
    assert all(str(WINSOR_MIN_OBSERVATIONS) in (r.note or "") for r in reports)
    assert trimmed[3].values["margin_gross"] == 400.0


def test_winsorization_leaves_missing_values_missing():
    rows = _synthetic_rows(date(2026, 1, 1), {f"T{i}": float(i) for i in range(100)})
    for r in rows[:10]:
        r.values["margin_gross"] = None
    trimmed, reports = winsorize(rows)

    assert all(t.values["margin_gross"] is None for t in trimmed[:10])
    one = next(r for r in reports if r.feature == "margin_gross")
    assert one.n_observed == 90


def test_winsorization_rejects_incoherent_bounds():
    rows = _synthetic_rows(date(2026, 1, 1), {"A": 1.0})
    with pytest.raises(ConfigError, match="lower_pct"):
        winsorize(rows, lower_pct=99.0, upper_pct=1.0)


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #


def _two_date_panel() -> FeaturePanel:
    """Four ordinary names in the training period, one extreme one after it."""
    train = _synthetic_rows(date(2025, 1, 1), {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    test = _synthetic_rows(date(2026, 1, 1), {"A": 1000.0, "B": 2000.0})
    return FeaturePanel(rows=train + test)


def test_standardize_refuses_to_run_without_a_fit_set():
    with pytest.raises(ConfigError, match="requires fit_rows"):
        _two_date_panel().standardize()


def test_standardize_fits_on_the_training_rows_only():
    """The leak test.

    If the transform were fitted on the whole panel the mean of this column
    would be about 501. Fitted on the training rows it is 2.5, and the test rows
    come out at the enormous z-scores they deserve, which is what tells a reader
    the distribution moved.
    """
    panel = _two_date_panel()
    Z, mu, sd = panel.standardize(date(2025, 1, 1))

    names = list(MATRIX_COLUMNS)
    j = names.index("margin_gross")
    assert mu[j] == pytest.approx(2.5)
    assert sd[j] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))

    assert Z[0, j] == pytest.approx((1.0 - 2.5) / sd[j])
    assert Z[4, j] > 500.0


def test_the_three_spellings_of_a_fit_set_agree():
    panel = _two_date_panel()
    by_date, mu_a, sd_a = panel.standardize(date(2025, 1, 1))
    by_rows, mu_b, sd_b = panel.standardize(panel.rows[:4])
    by_mask, mu_c, sd_c = panel.standardize([True] * 4 + [False] * 2)

    np.testing.assert_allclose(mu_a, mu_b)
    np.testing.assert_allclose(mu_a, mu_c)
    np.testing.assert_allclose(sd_a, sd_b)
    np.testing.assert_allclose(by_date, by_rows, equal_nan=True)
    np.testing.assert_allclose(by_date, by_mask, equal_nan=True)


def test_standardize_is_exactly_invertible():
    panel = _two_date_panel()
    Z, mu, sd = panel.standardize(date(2025, 1, 1))
    X, _, _ = panel.to_matrix()
    np.testing.assert_allclose(Z * sd + mu, X, equal_nan=True)


def test_a_column_with_no_variation_is_centred_but_not_scaled():
    """Dividing by zero would put infinities into the design matrix."""
    rows = _synthetic_rows(date(2025, 1, 1), {"A": 5.0, "B": 5.0, "C": 5.0})
    panel = FeaturePanel(rows=rows)
    Z, mu, sd = panel.standardize(date(2025, 1, 1))

    j = list(MATRIX_COLUMNS).index("margin_gross")
    assert sd[j] == 1.0
    assert Z[0, j] == pytest.approx(0.0)
    assert np.isfinite(Z[:, j]).all()
    assert any("centred but not scaled" in n for n in panel.notes)


def test_standardize_rejects_a_fit_set_that_matches_nothing():
    panel = _two_date_panel()
    with pytest.raises(ConfigError, match="nothing to fit"):
        panel.standardize(date(2020, 1, 1))


def test_standardize_rejects_a_mask_of_the_wrong_length():
    panel = _two_date_panel()
    with pytest.raises(ConfigError, match="boolean mask"):
        panel.standardize([True, False])


def test_missing_values_survive_standardisation_as_missing():
    panel = _two_date_panel()
    panel.rows[0].values["margin_gross"] = None
    Z, _, _ = panel.standardize(date(2025, 1, 1))
    assert np.isnan(Z[0, list(MATRIX_COLUMNS).index("margin_gross")])


# --------------------------------------------------------------------------- #
# The panel
# --------------------------------------------------------------------------- #


def _factories():
    return (
        lambda when: PinnedClient(when),
        lambda when: _market(when),
    )


def test_build_panel_records_a_failure_rather_than_dropping_it(ml_assumptions):
    """MongoDB's diluted share count cannot be tiled at the earlier dates.

    That is a real parsing failure on a real payload, and it is exactly the case
    that must not vanish. Dropping it would make the panel's coverage look better
    than it is, and a model's training set would silently exclude every filer
    whose tagging is awkward, which is not a random subset of anything.
    """
    client_factory, market_factory = _factories()
    panel = build_panel(
        TICKERS,
        [D_2025_JAN, D_2026_SEP],
        client_factory,
        market_factory,
        ml_assumptions,
    )

    assert len(panel.rows) == len(TICKERS) * 2
    failures = panel.failures()
    assert [r.ticker for r in failures] == ["MDB"]
    assert failures[0].as_of == D_2025_JAN
    assert "MissingDataError" in failures[0].error
    assert all(v is None for v in failures[0].values.values())
    assert len(failures[0].missing) == len(FEATURE_NAMES)
    assert any("could not be built" in n for n in panel.notes)


def test_a_failed_row_is_all_nan_in_the_matrix_and_is_not_imputed(ml_assumptions):
    client_factory, market_factory = _factories()
    panel = build_panel(
        TICKERS, [D_2025_JAN], client_factory, market_factory, ml_assumptions
    )
    X, names, dates = panel.to_matrix()
    failed = panel.rows.index(panel.failures()[0])
    assert np.isnan(X[failed, : len(FEATURE_NAMES)]).all()
    # The indicator says the whole row is missing rather than that it is zero.
    for group in FEATURE_GROUPS:
        assert X[failed, names.index(f"missing_share_{group}")] == pytest.approx(1.0)


def test_the_panel_shape_and_ordering_are_the_contract(ml_assumptions):
    client_factory, market_factory = _factories()
    dates = [D_2025_SEP, D_2026_MAR, D_2026_SEP]
    panel = build_panel(
        TICKERS, dates, client_factory, market_factory, ml_assumptions
    )
    X, names, row_dates = panel.to_matrix()

    assert X.shape == (len(TICKERS) * len(dates), len(MATRIX_COLUMNS))
    assert names == list(MATRIX_COLUMNS)
    assert row_dates == [d for d in dates for _ in TICKERS]
    assert panel.dates == dates
    assert panel.tickers == TICKERS


def test_the_panel_frame_renders_missing_as_nan_without_touching_the_rows(
    ml_assumptions,
):
    client_factory, market_factory = _factories()
    panel = build_panel(
        TICKERS, [D_2026_SEP], client_factory, market_factory, ml_assumptions
    )
    frame = panel.to_frame()

    assert len(frame) == len(TICKERS)
    assert list(frame["ticker"]) == TICKERS
    assert set(FEATURE_NAMES).issubset(frame.columns)
    assert frame["margin_ebitda"].isna().sum() >= 1
    crwd = next(r for r in panel.rows if r.ticker == "CRWD")
    assert crwd.values["margin_ebitda"] is None


def test_a_thin_panel_is_not_winsorized_and_says_so(ml_assumptions):
    """Four names per date is far below the count a percentile means anything at."""
    client_factory, market_factory = _factories()
    panel = build_panel(
        TICKERS, [D_2026_SEP], client_factory, market_factory, ml_assumptions
    )
    assert panel.n_winsorized == 0
    assert panel.winsorization
    assert all(not w.applied for w in panel.winsorization)


def test_the_panel_records_the_seed_even_though_it_uses_no_randomness(
    ml_assumptions,
):
    ml_assumptions.ml.random_seed = 31337
    client_factory, market_factory = _factories()
    panel = build_panel(
        ["DDOG"], [D_2026_SEP], client_factory, market_factory, ml_assumptions
    )
    assert panel.random_seed == 31337


def test_build_panel_does_not_swallow_a_lookahead(ml_assumptions):
    """A parsing failure costs a row. A leak invalidates the panel, so it escapes."""
    with pytest.raises(LookaheadError):
        build_panel(
            ["DDOG"],
            [D_2025_JAN],
            lambda when: LyingClient(when),
            lambda when: _market(when),
            ml_assumptions,
        )


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_two_panels_from_the_same_inputs_agree_exactly(ml_assumptions):
    """Exactly, not nearly. Nothing here is sampled, so nothing here drifts."""
    client_factory, market_factory = _factories()
    dates = [D_2025_SEP, D_2026_MAR, D_2026_SEP]

    first = build_panel(TICKERS, dates, client_factory, market_factory, ml_assumptions)
    second = build_panel(TICKERS, dates, client_factory, market_factory, ml_assumptions)

    a, names_a, dates_a = first.to_matrix()
    b, names_b, dates_b = second.to_matrix()

    assert names_a == names_b
    assert dates_a == dates_b
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])

    assert [r.notes for r in first.rows] == [r.notes for r in second.rows]
    assert [r.missing for r in first.rows] == [r.missing for r in second.rows]
    assert [r.error for r in first.rows] == [r.error for r in second.rows]


def test_a_single_row_is_reproducible(ml_assumptions):
    a = _row("ZS", D_2026_MAR, ml_assumptions)
    b = _row("ZS", D_2026_MAR, ml_assumptions)
    assert a.values == b.values
    assert a.statement_date == b.statement_date
    assert a.notes == b.notes


def test_standardisation_is_reproducible():
    first = _two_date_panel().standardize(date(2025, 1, 1))
    second = _two_date_panel().standardize(date(2025, 1, 1))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])


# --------------------------------------------------------------------------- #
# Coverage across the fixture universe
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ticker", TICKERS)
def test_every_fixture_filer_builds_at_the_latest_date(ticker, ml_assumptions):
    row = _row(ticker, D_2026_SEP, ml_assumptions)
    assert row.ok
    assert row.n_observed >= 35
    assert row.statement_date is not None
    assert row.statement_date <= D_2026_SEP


@pytest.mark.parametrize(
    "when", [D_2025_JAN, D_2025_MAR, D_2025_SEP, D_2026_MAR, D_2026_SEP]
)
def test_the_statement_date_never_runs_past_the_row_date(when, ml_assumptions):
    row = _row("DDOG", when, ml_assumptions)
    assert row.statement_date <= when


def test_a_date_before_any_filing_fails_rather_than_inventing_a_row(ml_assumptions):
    """Nothing was on file, so there is nothing to describe."""
    before = date(2023, 6, 1)
    with pytest.raises(MissingDataError):
        build_features(
            "DDOG", before, PinnedClient(before), _market(before), ml_assumptions
        )


def test_a_stale_price_feed_is_reported_rather_than_hidden(ml_assumptions):
    """Capped correctly but ending weeks early, which the leak check cannot see.

    Momentum, volatility and the range position are all measured back from the
    last close. A feed that stopped a month ago dates every one of them to then,
    and nothing about that is wrong until it goes unsaid.
    """
    stale = D_2026_SEP - timedelta(days=45)
    row = build_features(
        "DDOG", D_2026_SEP, PinnedClient(D_2026_SEP), _market(stale), ml_assumptions
    )
    assert any("the price series ends" in n for n in row.notes)
    assert row.values["market_momentum_12m"] is not None


def test_an_empty_fit_set_says_there_is_nothing_to_fit():
    panel = _two_date_panel()
    with pytest.raises(ConfigError, match="nothing to fit"):
        panel.standardize([])


def test_the_panel_totals_what_winsorization_touched():
    rows = _synthetic_rows(date(2026, 1, 1), {f"T{i}": float(i) for i in range(100)})
    trimmed, reports = winsorize(rows)
    panel = FeaturePanel(rows=trimmed, winsorization=reports)
    # One name pulled at each end of every feature.
    assert panel.n_winsorized == 2 * len(FEATURE_NAMES)
