"""Tests for the revenue fade curve.

Everything runs against ``tests/fixtures/fade_companyfacts.json.gz``, 224 TMT
filers pruned to the annual facts this module reads, so no test touches the
network. The fixture keeps every filed version of every period, which is what
makes the point-in-time construction testable at all: a fixture holding only the
latest restatement could not tell a panel that respects filing dates from one
that ignores them.

Three groups of tests carry the weight.

The first is point in time. The observation date is the filing date and not the
fiscal year end, the features are the figures that filing stated, the label is
the growth a later filing stated, and a seeded provenance violation raises rather
than scoring well. Those four together are the reason anything fitted here counts
as evidence.

The second is the baselines. The one-year fit does NOT beat persistence outside
the fold noise and the test asserts that it does not, because the day it starts
to, somebody should have to change this file and explain why.

The third is the revenue ladder, where a real bug lived. Charter's revenue
resolved 62 times too small through ``tags.REVENUE`` before the component guard,
and the test that proves it fails without the guard is the first one below.
"""

from __future__ import annotations

import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from techval import tags
from techval.config import Assumptions
from techval.dcf import project, run_dcf
from techval.edgar import CompanyFacts, HttpCache, Provenance
from techval.errors import ConfigError, NotMeaningfulError
from techval.ev_bridge import build_ev_bridge
from techval.financials import build_financials
from techval.market import CsvSource, MarketData
from techval.ml.features import LookaheadError
from techval.ml.forecast import (
    DELISTED_CIKS,
    DELISTED_UNIVERSE,
    MATRIX_COLUMNS,
    PERSISTENCE_BASELINE,
    FadeObservation,
    FadePanel,
    _growth,
    _Scaler,
    build_fade_panel,
    compare_fade,
    fade_universe,
    fit_fade,
    growth_path_for_dcf,
    is_delisted,
)
from techval.ml.protocol import EvalResult, ModelCard
from techval.wacc import compute_wacc

FIXTURES = Path(__file__).parent.parent / "fixtures"
PRICES = FIXTURES / "prices"
AS_OF = date(2026, 9, 10)


def _blob() -> dict:
    with gzip.open(FIXTURES / "fade_companyfacts.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


BLOB = _blob()


class FadeFixtureClient:
    """Serves the pruned payloads, unpinned, exactly as a live client would.

    Unpinned on purpose: ``build_fade_panel`` pins the fact set once per fiscal
    year, and a client that arrived pre-pinned would hide whether it does.
    """

    def company_facts(self, ticker: str) -> CompanyFacts:
        payload = BLOB["payloads"][ticker.upper()]
        return CompanyFacts(payload, ticker)


@pytest.fixture(scope="module")
def universe() -> dict[str, str]:
    return {t: meta["sub_vertical"] for t, meta in BLOB["meta"].items()}


@pytest.fixture(scope="module")
def panel(universe) -> FadePanel:
    return build_fade_panel(universe, FadeFixtureClient())


@pytest.fixture(scope="module")
def fade_assumptions() -> Assumptions:
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    a.ml.forecast.enabled = True
    return a


@pytest.fixture(scope="module")
def model(panel, fade_assumptions):
    return fit_fade(panel, fade_assumptions)


def _observation(panel: FadePanel, ticker: str, year: int) -> FadeObservation:
    for o in panel.observations:
        if o.ticker == ticker and o.fiscal_year_end.year == year:
            return o
    raise AssertionError(f"{ticker} has no fiscal year ending in {year}")


# --------------------------------------------------------------------------- #
# The bug: a revenue ladder that resolves a disaggregation component
# --------------------------------------------------------------------------- #


def test_charter_revenue_without_the_guard_is_a_disaggregation_component():
    """The failing case. Fails before the component guard, passes after.

    Charter tags ``RevenueFromContractWithCustomerIncludingAssessedTax`` at
    889mm for fiscal 2025 and ``Revenues`` at 54,774mm. ``tags.REVENUE`` ranks
    the first above the second, so the unguarded ladder returns a number 62
    times too small, and every multiple struck on it is 62 times too high.
    """
    facts = CompanyFacts(BLOB["payloads"]["CHTR"], "CHTR")
    unguarded, provenance = facts.resolve_ttm(
        "revenue", tags.REVENUE, date(2025, 12, 31)
    )
    assert provenance.tag == "RevenueFromContractWithCustomerIncludingAssessedTax"
    assert unguarded == pytest.approx(889_000_000)

    guarded, provenance = facts.resolve_ttm(
        "revenue", tags.REVENUE, date(2025, 12, 31), component_guard=10.0
    )
    assert provenance.tag == "Revenues"
    assert guarded == pytest.approx(54_774_000_000)
    assert "disaggregation component" in (provenance.note or "")
    assert guarded / unguarded > 60


def test_the_guard_leaves_an_ordinary_ladder_alone():
    """Datadog reports revenue under one tag and the guard changes nothing.

    The guard must be inert on the overwhelming majority of filers, or it is not
    a guard, it is a different resolution rule.
    """
    facts = CompanyFacts(json.loads((FIXTURES / "companyfacts_DDOG.json").read_text()), "DDOG")
    plain, plain_prov = facts.resolve_ttm("revenue", tags.REVENUE, date(2026, 6, 30))
    guarded, guarded_prov = facts.resolve_ttm(
        "revenue", tags.REVENUE, date(2026, 6, 30), component_guard=10.0
    )
    assert plain == guarded
    assert plain_prov.tag == guarded_prov.tag
    assert guarded_prov.note is None


def test_a_negative_or_zero_incumbent_is_never_swapped():
    """The guard is a ratio, so it may not fire on a figure that reaches zero.

    This is why the guard is opt-in per concept rather than always on: EBIT
    passes through zero every cycle and a ratio test against zero fires at
    random.
    """
    facts = CompanyFacts(BLOB["payloads"]["CHTR"], "CHTR")
    value, provenance = facts.resolve_ttm(
        "EBIT", tags.EBIT, date(2025, 12, 31), component_guard=10.0, required=False
    )
    assert provenance.note is None


def test_the_panel_carries_charters_real_revenue(panel):
    charter = _observation(panel, "CHTR", 2024)
    assert charter.revenue == pytest.approx(55_085_000_000)


# --------------------------------------------------------------------------- #
# Point in time
# --------------------------------------------------------------------------- #


def test_the_observation_is_dated_to_the_filing_not_the_year_end(panel):
    """A December year end is not public in January.

    Datadog's fiscal 2025 ended on 31 December 2025 and its 10-K was filed on
    18 February 2026. An observation dated to the year end would hand the model
    seven weeks of the future for nothing.
    """
    ddog = _observation(panel, "DDOG", 2025)
    assert ddog.fiscal_year_end == date(2025, 12, 31)
    assert ddog.as_of == date(2026, 2, 18)
    assert (ddog.as_of - ddog.fiscal_year_end).days > 40


def test_the_features_are_the_figures_that_filing_stated(panel):
    """Revenue on the row is the revenue in the 10-K that first reported it."""
    ddog = _observation(panel, "DDOG", 2025)
    assert ddog.revenue == pytest.approx(3_427_158_000, rel=1e-9)
    assert ddog.growth == pytest.approx(0.2769, abs=5e-4)


def test_the_label_is_the_growth_the_later_filing_printed(panel):
    """Forward growth is another observation's own growth, not a cross-vintage ratio."""
    earlier = _observation(panel, "DDOG", 2023)
    later = _observation(panel, "DDOG", 2024)
    assert earlier.labels[1] == later.growth
    assert earlier.label_dates[1] == later.as_of
    # And the two-year label is the year after that, not the two-year CAGR.
    two_out = _observation(panel, "DDOG", 2025)
    assert earlier.labels[2] == two_out.growth


def test_a_filer_that_stopped_filing_has_no_label_for_its_last_year(panel):
    """The survivorship channel, which is an absence rather than a number.

    Splunk's final 10-K is fiscal 2023; Cisco closed the acquisition in March
    2024 and there is no fiscal 2024. That missing label is not noise, it is the
    outcome the sample most needs and cannot have.
    """
    splunk = sorted(
        (o for o in panel.observations if o.ticker == "SPLK"),
        key=lambda o: o.fiscal_year_end,
    )
    assert splunk, "the panel should carry Splunk, which no ticker lookup can find"
    assert splunk[-1].labels == {}
    assert splunk[-2].labels.get(1) is not None


def test_a_provenance_violation_raises_rather_than_scoring_well(panel):
    """Seed one figure filed after the observation date and the panel is not evidence."""
    from techval.ml.features import assert_point_in_time

    row = _observation(panel, "DDOG", 2023).feature_row()
    assert_point_in_time(row)  # clean to start with

    row.provenance["revenue"] = Provenance(
        concept="revenue",
        tag="RevenueFromContractWithCustomerExcludingAssessedTax",
        method="annual period",
        filed=str(row.as_of + timedelta(days=400)),
    )
    with pytest.raises(LookaheadError) as exc:
        assert_point_in_time(row)
    assert "400 days after the row date" in str(exc.value)


def test_the_panel_refuses_to_read_a_later_restatement(panel):
    """The same fiscal year, read through the filing and read through today.

    Microsoft's fiscal 2017 revenue is 7.4% higher in today's fact set than in
    the 10-K that reported it, because ASC 606 was adopted with full
    retrospective restatement. The panel carries the filed figure; the restated
    one is kept beside it and used for nothing.
    """
    msft = _observation(panel, "MSFT", 2017)
    assert msft.restated_revenue is not None
    assert msft.restatement_gap == pytest.approx(0.074, abs=0.005)
    assert msft.revenue < msft.restated_revenue


# --------------------------------------------------------------------------- #
# What the panel measures about itself
# --------------------------------------------------------------------------- #


def test_the_depth_is_measured_rather_than_assumed(panel):
    """XBRL reaches back to 2009 here, not to 2007, and the median filer has 12 years."""
    depth = panel.depth_summary()
    assert depth["max"] == 19
    assert depth["median"] == 12
    assert depth["at_least_10"] == 154
    assert min(o.as_of for o in panel.observations).year == 2009


def test_restatement_is_reported_both_ways(panel):
    """Holding the tag fixed separates restatement from tag migration.

    Through the ladder 1.75% of fiscal years move by more than a percent.
    Holding the tag fixed only 0.71% do, so more than half of what looks like
    restatement is the filer moving revenue to a tag with a different scope.
    """
    report = panel.restatement_report()
    assert report["ladder_above_1pct"] == pytest.approx(0.0175, abs=0.002)
    assert report["same_tag_above_1pct"] == pytest.approx(0.0071, abs=0.002)
    assert report["same_tag_above_1pct"] < report["ladder_above_1pct"]


def test_acquisitive_years_grow_faster_the_year_after_the_deal(panel):
    """The measurement that decides acquisitive years are kept, not excluded."""
    report = panel.acquisition_report()
    assert report["heavy"] > 300
    same_year = report["same_year_growth_heavy"] - report["same_year_growth_quiet"]
    next_year = report["next_year_growth_heavy"] - report["next_year_growth_quiet"]
    assert next_year > same_year * 2
    assert next_year == pytest.approx(0.059, abs=0.01)


def test_the_leavers_are_in_the_panel_and_are_not_a_rounding_error(panel):
    delisted = [o for o in panel.observations if is_delisted(o.ticker)]
    assert len(delisted) / len(panel.observations) > 0.40
    assert set(DELISTED_UNIVERSE) == set(DELISTED_CIKS)
    assert "SPLK" in DELISTED_UNIVERSE and "DDOG" not in DELISTED_UNIVERSE
    assert set(fade_universe()) >= set(DELISTED_UNIVERSE)


def test_the_leavers_fade_before_they_leave(panel):
    """Their last observed year grows well below the filers still quoted."""
    report = panel.survivorship_report()
    assert report["final_year_growth_mean"] < report["panel_growth_mean"]
    assert report["observations_without_a_next_year_label"] > 200


def test_the_reversion_table_needs_no_functional_form(panel):
    """Top decile fades hard, bottom decile reverts up, and they cross in the middle."""
    table = panel.reversion_table()
    top = table.iloc[-1]
    bottom = table.iloc[0]
    assert top["Trailing growth"] > 0.80
    assert top["Forward growth"] < top["Trailing growth"] / 1.5
    assert bottom["Trailing growth"] < -0.10
    assert bottom["Forward growth"] > bottom["Trailing growth"]
    assert list(table["Trailing growth"]) == sorted(table["Trailing growth"])


# --------------------------------------------------------------------------- #
# The guards inside the panel
# --------------------------------------------------------------------------- #


def test_a_thousandfold_jump_is_a_change_of_units_not_growth():
    """Early XBRL filings carry scale errors, and Groupon's is exactly 1,000x."""
    assert _growth(312_941_000, 312_941) is None
    assert _growth(312_941, 312_941_000) is None
    # A genuine hypergrowth year is kept: Groupon really did grow 21x in 2010.
    assert _growth(312_941, 14_540) == pytest.approx(20.52, abs=0.01)


def test_no_observation_in_the_panel_carries_an_impossible_growth_rate(panel):
    growths = [o.growth for o in panel.observations if o.growth is not None]
    assert min(growths) > -1.0
    assert max(growths) < 99.0


def test_winsorization_is_inside_each_filing_year(panel, universe):
    """A pooled trim would bake the level of a whole era into the bound."""
    untrimmed = build_fade_panel(
        universe, FadeFixtureClient(), winsorize_cross_section=False
    )
    assert panel.winsorized > 500
    assert untrimmed.winsorized == 0
    raw = max(o.values["growth_1y"] for o in untrimmed.observations if o.growth is not None)
    trimmed = max(o.values["growth_1y"] for o in panel.observations if o.growth is not None)
    assert trimmed < raw


def test_the_scaler_standardises_on_observed_values_not_the_imputed_column():
    """The 66-sigma trap, in eight rows.

    One column is observed twice in ten and missing eight times. Impute first
    and the column's dispersion collapses, so a test value two real standard
    deviations out arrives at the model as an extreme it has never seen and a
    small coefficient turns into a forecast of several hundred percent growth.
    """
    observed = [-1.5, -0.5, 0.0, 0.0, 0.5, 1.5]
    column = np.array([[v] for v in observed] + [[np.nan]] * 24)
    scaler = _Scaler.fit(column)
    assert scaler.centre[0] == pytest.approx(0.0)
    assert scaler.scale[0] == pytest.approx(float(np.std(observed)))

    # What imputing first would have used: 24 copies of the median flatten the
    # dispersion to a third of the real one, and everything real then reads as
    # three times as extreme as it is.
    naive = float(np.std(np.where(np.isfinite(column), column, 0.0)))
    assert naive < scaler.scale[0] / 2

    transformed = scaler.transform(np.array([[1.5], [np.nan]]))
    assert transformed[0, 0] == pytest.approx(1.5 / float(np.std(observed)))
    assert transformed[1, 0] == pytest.approx(0.0)


def test_the_design_matrix_is_clipped_rather_than_left_to_extrapolate():
    scaler = _Scaler.fit(np.array([[0.0], [1.0], [-1.0], [0.5], [-0.5]]))
    far = scaler.transform(np.array([[500.0]]))
    assert far[0, 0] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# The fit, and the baselines it has to beat
# --------------------------------------------------------------------------- #


def test_every_horizon_reports_three_baselines_before_the_model(model):
    for horizon, fit in model.fits.items():
        assert set(fit.baselines) == {
            "persistence",
            "training_mean",
            "sub_vertical_mean",
        }
        assert fit.evaluation.baseline_name
        assert fit.evaluation.metric == "mae"
        assert fit.n_train > 1_000


def test_the_one_year_fit_does_not_beat_persistence_outside_the_noise(model):
    """The publishable negative result, pinned so it cannot drift away quietly.

    At one year out the ridge scores 0.1474 against 0.1510 for carrying last
    year's growth forward. That is a lift of 0.0035 against a fold standard
    deviation of 0.0242, which is to say the two are the same number. Growth is
    sticky one year out and there is almost nothing for a model to add.

    If a change to this module ever makes this test fail, the right response is
    not to delete the test. It is to check whether the improvement is real
    across folds and then rewrite this docstring with the new numbers.
    """
    fit = model.fits[1]
    assert fit.evaluation.score == pytest.approx(0.1474, abs=0.004)
    assert fit.baselines["persistence"] == pytest.approx(0.1510, abs=0.004)
    assert abs(fit.evaluation.lift) < fit.evaluation.fold_sd
    assert "inside the fold-to-fold noise" in fit.evaluation.verdict()


def test_the_verdict_names_persistence_as_the_baseline_it_was_scored_against(model):
    """The comparison is only readable if the sentence says what it compares.

    ``evaluate_regression`` cannot know what an array handed to it as
    ``baseline_pred`` means, so it calls it "supplied baseline". Left alone,
    that is the phrase the model card and the dashboard print: 0.1510 for
    supplied baseline, a number with no referent. The baseline here is this
    year's growth carried forward, and the verdict has to say so at every
    horizon.
    """
    for horizon, fit in model.fits.items():
        verdict = fit.evaluation.verdict()
        assert "supplied baseline" not in verdict, f"h={horizon}: {verdict}"
        assert f"for {PERSISTENCE_BASELINE}" in verdict, f"h={horizon}: {verdict}"
        assert fit.evaluation.baseline_score == pytest.approx(
            fit.baselines["persistence"], abs=1e-12
        )
    summary = model.card.summary()
    assert f"for {PERSISTENCE_BASELINE}" in summary
    # The verdict follows the card's own full stop, so it opens a sentence.
    assert " features. MAE of " in summary and ". mae of" not in summary


@pytest.mark.parametrize(
    "metric, opens",
    [
        ("mae", "MAE of"),
        ("rmse", "RMSE of"),
        ("auc", "AUC of"),
        ("ndcg@10", "NDCG@10 of"),
        ("ic", "IC of"),
        ("spearman", "Spearman of"),
        ("mean information coefficient", "Mean information coefficient of"),
    ],
)
def test_the_card_opens_the_verdict_as_a_sentence_with_initialisms_in_capitals(metric, opens):
    """``ModelCard.summary`` joins ``EvalResult.verdict`` after a full stop.

    The verdict opens on the metric in lower case, which is right where it is
    printed alone and wrong mid-paragraph: "across 25 features. mae of 0.1230".
    The join sentence-cases it, an initialism in capitals rather than "Mae", and
    leaves ``EvalResult.verdict`` itself exactly as it was.
    """
    evaluation = EvalResult(
        metric=metric, score=0.123, baseline_name="the baseline", baseline_score=0.151,
        n_observations=400, higher_is_better=False,
    )
    card = ModelCard(
        name="m", task="a task", trained_through=date(2025, 6, 30), n_train=900,
        features=["a", "b"], evaluation=evaluation,
    )
    line = evaluation.verdict()
    assert line.startswith(f"{metric} of ")
    assert card.summary() == (
        "m: a task, fitted on 900 observations through 2025-06-30 across 2 features. "
        f"{opens} {line.removeprefix(f'{metric} of ')}"
    )


def test_the_two_and_three_year_fits_beat_persistence_outside_the_noise(model):
    """Mean reversion is where a fade curve earns its keep.

    Persistence gets worse as the horizon lengthens (0.1510, 0.1884, 0.1921)
    while the model gets better (0.1474, 0.1476, 0.1391), which is the shape of
    the phenomenon rather than a property of the estimator.
    """
    for horizon in (2, 3):
        fit = model.fits[horizon]
        assert fit.evaluation.beat_baseline
        assert fit.evaluation.lift > fit.evaluation.fold_sd
        assert fit.evaluation.score < fit.baselines["persistence"]
    assert (
        model.fits[1].baselines["persistence"]
        < model.fits[2].baselines["persistence"]
        < model.fits[3].baselines["persistence"]
    )


def test_the_model_also_beats_the_sub_vertical_mean(model):
    """The baseline persistence cannot supply: sector means, which do revert."""
    for horizon, fit in model.fits.items():
        assert fit.evaluation.score < fit.baselines["sub_vertical_mean"]
        assert fit.evaluation.score < fit.baselines["training_mean"]


def test_the_embargo_scales_with_the_horizon(model):
    """A three-year label is not known until three years later.

    Without the scaling, a fold trained on observations filed a year before the
    test block has seen two thirds of its own answer.
    """
    for horizon, fit in model.fits.items():
        folds = fit.evaluation.folds
        assert len(folds) == 5
    assert model.card.hyperparameters["embargo_days_per_horizon"] == 365


def test_the_fade_curve_is_two_numbers_and_they_are_the_economics(model):
    """Slope 0.47, intercept 6.9 points: half the gap closed a year, toward 13%."""
    assert model.persistence_slope == pytest.approx(0.472, abs=0.02)
    assert model.persistence_intercept == pytest.approx(0.069, abs=0.02)
    assert 0.45 < model.fade_per_year < 0.60
    assert 0.10 < model.reversion_level < 0.16
    assert 0.8 < model.half_life_years < 1.1
    assert "fades" in model.summary()


def test_the_model_card_says_what_it_cannot_do(model):
    assert model.card.evaluation is model.fits[1].evaluation
    assert len(model.card.limitations) >= 5
    assert any("Survivorship" in line for line in model.card.limitations)
    assert any("retention" in line for line in model.card.limitations)
    assert model.card.features == list(MATRIX_COLUMNS)
    assert model.card.trained_through <= AS_OF


def test_two_fits_from_one_panel_agree_bit_for_bit(panel, fade_assumptions):
    """Determinism, asserted rather than assumed."""
    first = fit_fade(panel, fade_assumptions, horizon_years=1)
    second = fit_fade(panel, fade_assumptions, horizon_years=1)
    assert first.fits[1].evaluation.score == second.fits[1].evaluation.score
    assert first.persistence_slope == second.persistence_slope
    assert first.path("DDOG", 3).growth == second.path("DDOG", 3).growth


def test_each_model_kind_fits_and_the_default_is_the_ridge(panel, fade_assumptions):
    """All three offered models run, and none of them is distinguishable."""
    scores = {}
    for kind in ("ridge", "gradient_boosting", "linear"):
        fitted = fit_fade(panel, fade_assumptions, model=kind, horizon_years=1)
        scores[kind] = fitted.fits[1].evaluation.score
    assert scores["ridge"] == pytest.approx(0.1474, abs=0.004)
    # Unpenalised least squares is worse on the thin first fold, which is the
    # whole argument for a penalty chosen on the training data.
    assert scores["linear"] > scores["ridge"]
    with pytest.raises(ConfigError):
        fit_fade(panel, fade_assumptions, model="random_forest", horizon_years=1)


def test_a_panel_below_the_floor_reports_unavailable(fade_assumptions):
    """``min_train_observations`` refuses rather than fitting forty company-years."""
    small = FadePanel(
        observations=[],
        depth={},
    )
    with pytest.raises(NotMeaningfulError) as exc:
        fit_fade(small, fade_assumptions)
    assert "min_train_observations" in str(exc.value)


def test_the_drift_report_would_have_caught_a_clock(model):
    """Every feature's train-to-test shift, in training standard deviations."""
    drift = model.fits[1].drift
    assert set(drift) == set(MATRIX_COLUMNS)
    assert all(np.isfinite(v) for v in drift.values())
    assert max(abs(v) for v in drift.values()) < 5.0


# --------------------------------------------------------------------------- #
# The path
# --------------------------------------------------------------------------- #


def test_the_path_says_which_years_are_fitted_and_which_assumed(model):
    path = model.path("DDOG", 5)
    assert len(path) == 5
    assert path.basis == ["fitted", "fitted", "fitted", "assumed", "assumed"]
    assert path.growth[-1] == pytest.approx(model.terminal_growth)
    assert any("not fitted" in note for note in path.notes)


def test_the_band_converges_onto_the_terminal_assumption(model):
    """An assumption has no sampling error, and the band should not pretend it does."""
    path = model.path("DDOG", 5)
    assert path.lower[-1] == pytest.approx(model.terminal_growth)
    assert path.upper[-1] == pytest.approx(model.terminal_growth)
    for point, low, high in zip(path.growth, path.lower, path.upper):
        assert low <= point + 1e-9 and point <= high + 1e-9


def test_the_interval_is_wide_enough_to_be_embarrassing(model):
    """Forty growth points at one year out. That is the honest output."""
    path = model.path("DDOG", 3)
    width = path.upper[0] - path.lower[0]
    assert width > 0.40
    low, high = model.fits[1].residual_quantiles
    assert low < 0 < high


def test_a_growing_company_fades_and_a_shrinking_one_does_not(model, panel):
    """The direction of the curve, on two real companies rather than in the abstract."""
    ddog = _observation(panel, "DDOG", 2025)
    point, low, high = model.predict(ddog, 1)
    assert low < point < high
    assert point < ddog.growth  # 28% does not stay 28%
    assert point > 0.10


def test_the_path_refuses_a_company_the_panel_never_saw(model):
    with pytest.raises(ConfigError) as exc:
        model.path("NOTATICKER", 3)
    assert "not in the panel" in str(exc.value)
    with pytest.raises(ConfigError):
        model.path("DDOG", 0)


def test_a_horizon_beyond_the_fit_is_refused_rather_than_extrapolated(model, panel):
    ddog = _observation(panel, "DDOG", 2025)
    with pytest.raises(ConfigError) as exc:
        model.predict(ddog, 9)
    assert "horizon_years" in str(exc.value)


# --------------------------------------------------------------------------- #
# Survivorship, in dollars
# --------------------------------------------------------------------------- #


def test_dropping_the_leavers_lifts_growth_at_every_horizon(model):
    gaps = [model.survivorship[f"gap_{h}y"] for h in (1, 2, 3)]
    assert all(gap > 0 for gap in gaps)
    assert gaps == sorted(gaps), "the bias should compound with the horizon"
    assert gaps[2] > 0.01
    assert (
        model.survivorship["final_year_growth_leavers"]
        < model.survivorship["final_year_growth_survivors"]
    )


def test_a_survivors_only_fit_fades_more_slowly(panel, fade_assumptions):
    """The bias, refitted rather than inferred, and it points the same way."""
    survivors = FadePanel(
        observations=[o for o in panel.observations if not is_delisted(o.ticker)],
        depth={t: n for t, n in panel.depth.items() if not is_delisted(t)},
    )
    biased = fit_fade(survivors, fade_assumptions)
    full = fit_fade(panel, fade_assumptions)
    assert biased.reversion_level > full.reversion_level
    assert biased.path("DDOG", 3).growth[0] > full.path("DDOG", 3).growth[0]


# --------------------------------------------------------------------------- #
# The seam into the valuation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ddog_valuation(fade_assumptions):
    """Datadog priced off the committed fixtures, ready for two growth paths."""
    facts = CompanyFacts(
        json.loads((FIXTURES / "companyfacts_DDOG.json").read_text()), "DDOG"
    )
    fin = build_financials("DDOG", facts=facts)
    market = MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=AS_OF)
    bridge = build_ev_bridge(fin, market.spot("DDOG"), fade_assumptions)
    wacc = compute_wacc(fin, bridge, market, fade_assumptions)
    return fin, bridge, wacc


def test_the_dcf_is_unchanged_when_no_path_is_supplied(ddog_valuation, fade_assumptions):
    """Nothing about a run without a fitted path may move."""
    fin, bridge, wacc = ddog_valuation
    plain = run_dcf(fin, bridge, wacc, fade_assumptions)
    assumed = [round(p.growth, 12) for p in plain.projections]
    assert assumed == [0.20, 0.1625, 0.125, 0.0875, 0.05]
    explicit = run_dcf(fin, bridge, wacc, fade_assumptions, growth_path=assumed)
    assert explicit.enterprise_value == pytest.approx(plain.enterprise_value)


def test_a_supplied_path_is_used_year_by_year_and_recorded(ddog_valuation, fade_assumptions):
    fin, bridge, wacc = ddog_valuation
    path = [0.30, 0.25, 0.20, 0.15, 0.10]
    result = run_dcf(fin, bridge, wacc, fade_assumptions, growth_path=path)
    assert [round(p.growth, 12) for p in result.projections] == path
    assert any("supplied by the caller" in note for note in result.notes)
    # Revenue compounds off the trailing figure, not off the assumption.
    assert result.projections[0].revenue == pytest.approx(fin.revenue * 1.30)


def test_a_path_of_the_wrong_length_is_refused(ddog_valuation, fade_assumptions):
    """Padding to fit would value a different company from the one described."""
    fin, bridge, wacc = ddog_valuation
    with pytest.raises(ConfigError) as exc:
        project(fin, fade_assumptions, growth_path=[0.2, 0.1])
    assert "projection runs 5" in str(exc.value)
    with pytest.raises(ConfigError):
        project(fin, fade_assumptions, growth_path=[0.2, 0.1, -1.4, 0.1, 0.05])


def test_the_switch_is_off_by_default_and_the_report_is_unchanged(model):
    """``ml.forecast.enabled`` defaults false, so nothing reaches a base run."""
    off = Assumptions()
    assert off.ml.forecast.enabled is False
    assert growth_path_for_dcf(model, "DDOG", off) is None
    on = Assumptions()
    on.ml.forecast.enabled = True
    path = growth_path_for_dcf(model, "DDOG", on)
    assert path is not None
    assert len(path) == on.dcf.projection_years


def test_compare_fade_refuses_to_run_behind_the_switch(ddog_valuation, model):
    fin, bridge, wacc = ddog_valuation
    off = Assumptions()
    off.market.risk_free_rate = 0.0483
    with pytest.raises(ConfigError) as exc:
        compare_fade(fin, bridge, wacc, off, model)
    assert "ml.forecast.enabled" in str(exc.value)


def test_datadog_valued_on_the_assumed_fade_and_on_the_fitted_one(
    ddog_valuation, fade_assumptions, model
):
    """The number an analyst can argue with.

    The typed fade takes Datadog from the default 20% growth to 5% in a straight
    line over five years, starting below the 28% its fiscal 2025 10-K reported,
    and values the enterprise at 9,447mm, 36.65 a share. The fitted fade holds
    growth near 20% for three years before handing over to the same terminal
    assumption, and values it at 10,901mm, 40.61 a share, 15% higher.

    The band is the part worth reading. The 10th and 90th percentile paths value
    the same company at 26.97 and 66.69 a share, which is not a forecast, it is
    the honest width of what this data supports.
    """
    fin, bridge, wacc = ddog_valuation
    comparison = compare_fade(fin, bridge, wacc, fade_assumptions, model)

    assert comparison.assumed_path[0] == pytest.approx(0.20)
    assert comparison.assumed_path[-1] == pytest.approx(0.05)
    assert comparison.fitted_path.growth[0] == pytest.approx(0.213, abs=0.02)

    assert comparison.assumed.per_share == pytest.approx(36.65, abs=0.5)
    assert comparison.fitted.per_share == pytest.approx(40.61, abs=1.0)
    assert comparison.enterprise_value_gap > 0
    assert comparison.per_share_gap == pytest.approx(3.96, abs=1.0)

    assert comparison.low.per_share == pytest.approx(26.97, abs=2.0)
    assert comparison.high.per_share == pytest.approx(66.69, abs=4.0)
    assert comparison.low.per_share < comparison.assumed.per_share
    assert comparison.high.per_share > 1.5 * comparison.assumed.per_share
    assert "a share" in comparison.summary()
    assert list(comparison.to_frame().index) == [
        "Assumed fade",
        "Fitted fade",
        "Fitted, low band",
        "Fitted, high band",
    ]


def test_survivorship_is_worth_nearly_as_much_as_the_whole_model(
    ddog_valuation, fade_assumptions, panel, model
):
    """The sample decision against the modelling decision, both in dollars.

    Replacing the typed fade with the fitted one moves Datadog by about 4 a
    share. Fitting the same model on the filers still quoted today, which is
    what any panel built from the SEC ticker file would be, moves it by almost
    as much again and in the same direction. The invisible decision is as large
    as the visible one.
    """
    fin, bridge, wacc = ddog_valuation
    survivors = FadePanel(
        observations=[o for o in panel.observations if not is_delisted(o.ticker)],
        depth={t: n for t, n in panel.depth.items() if not is_delisted(t)},
    )
    biased = fit_fade(survivors, fade_assumptions)

    full_path = model.path("DDOG", fade_assumptions.dcf.projection_years)
    biased_path = biased.path("DDOG", fade_assumptions.dcf.projection_years)
    assumed = run_dcf(fin, bridge, wacc, fade_assumptions)
    fitted = run_dcf(fin, bridge, wacc, fade_assumptions, growth_path=full_path.growth)
    survivors_only = run_dcf(
        fin, bridge, wacc, fade_assumptions, growth_path=biased_path.growth
    )

    modelling = fitted.per_share - assumed.per_share
    survivorship = survivors_only.per_share - fitted.per_share
    assert modelling > 0 and survivorship > 0
    assert survivorship > 0.5 * modelling


def test_the_survivorship_refit_is_reported_as_a_pair(panel):
    """Both fits, side by side, so the bias is a number rather than a warning."""
    from techval.ml.forecast import survivorship_bias

    one_year = Assumptions()
    one_year.ml.forecast.enabled = True
    one_year.ml.forecast.horizon_years = 1
    report = survivorship_bias(panel, one_year)
    assert report["survivor_observations"] < report["full_observations"]
    assert report["survivor_reversion_level"] > report["full_reversion_level"]
    assert report["gap_1y"] > 0


def test_a_path_with_no_assumed_tail_says_it_ends_on_a_cliff(panel, fade_assumptions):
    """Fitting every projection year hands a 20% grower to a 2.5% perpetuity."""
    five = Assumptions()
    five.market.risk_free_rate = 0.0483
    five.ml.forecast.enabled = True
    five.ml.forecast.horizon_years = 5
    model = fit_fade(panel, five)
    path = model.path("DDOG", 5)
    assert set(path.basis) == {"fitted"}
    assert any("cliff" in note for note in path.notes)
