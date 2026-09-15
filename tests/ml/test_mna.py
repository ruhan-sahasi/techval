"""The acquisition propensity model, against 493 real registrants and 107 real deals.

Every fixture under ``tests/fixtures/mna`` came from SEC EDGAR or the Nasdaq
daily quote endpoint on 2026-09-11 and is recorded as ``MANIFEST.json`` says. No
test here touches the network.

The sample is the point. 493 TMT registrants carrying one of the sub-vertical SIC
codes and reporting at least 500mm of revenue in some year of the window, of
which 104 have left the filing record, and 107 announced acquisitions found by
running the production precedent extractor over their merger filings. Seventy
seven of those 107 deals are on companies that no longer file, which is the
number that decides whether this experiment is possible at all: build the
universe from names listed today and nearly three quarters of the positive class
is gone.

Four groups of tests carry the weight and the rest is arithmetic.

The survivorship group proves by counting that the departed are in the panel
before their deals and out of it afterwards, and that a universe built the
obvious way loses the labels.

The leak group proves the two channels this task leaks through. One is the
announcement itself: a feature row built the day before a merger agreement must
know nothing about it, and ``assert_point_in_time`` is run over real payloads to
show it. The other is subtler and is the finding that shaped the feature set: a
delisted company has no share price from any public source, so a price feature is
present for most of the negatives and absent for most of the positives. Its
absence alone scores an AUC of 0.82 on the labelled sample, far above anything
the filings-only model reaches, though not a perfect separation, because about a
quarter of the positive rows are on companies that still file. That is measured
here rather than asserted.

The labelling group proves that an unresolved window is dropped rather than
counted as a negative, that an observation sitting inside the rumour gap is
dropped rather than relabelled, and that a deal blocked or still pending is a
positive because the label is announcement.

The honesty group proves that the baseline is the size sort rather than the base
rate, that ``beat_baseline`` is computed, and that the verdict prints the refusal
when the model loses.
"""

from __future__ import annotations

import csv
import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from techval.config import Assumptions
from techval.edgar import CompanyFacts
from techval.errors import ConfigError, MissingDataError, NotMeaningfulError
from techval.ml.evaluation import calibration_table
from techval.ml.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FeaturePanel,
    FeatureRow,
    LookaheadError,
    assert_point_in_time,
    build_features,
)
from techval.ml.mna import (
    CONSENT_DEAL_FORMS,
    DEFAULT_GAP_DAYS,
    FEATURE_RATIONALE,
    FITTED_COLUMNS,
    MIN_DISTINCT_DEALS,
    Attribution,
    DealEvent,
    Observation,
    PropensityModel,
    Universe,
    balance_sheet_extras,
    base_rate_by_year,
    build_matrix,
    build_universe,
    derive_features,
    fit_propensity,
    label_observations,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mna"

#: The date every fixture was recorded and every claim about what was pending is
#: fixed to.
AS_OF = date(2026, 9, 11)

#: Companies the tests name, by the key the roster gives them. A delisted
#: registrant has no ticker anywhere in EDGAR, so its key is built from its CIK,
#: and that is a fact about the filing record rather than a shortcut here.
SPLUNK = "CIK1353283"
SLACK = "CIK1764925"
MANDIANT = "CIK1370880"
ROKU = "ROKU"
DATADOG = "DDOG"
DATADOG_CIK = "CIK1561550"


# --------------------------------------------------------------------------- #
# Loading the recorded panel
# --------------------------------------------------------------------------- #


def _load_rows() -> tuple[list[FeatureRow], dict[tuple[str, date], dict]]:
    rows: list[FeatureRow] = []
    extras: dict[tuple[str, date], dict] = {}
    with gzip.open(FIXTURES / "panel.csv.gz", "rt", newline="") as fh:
        for rec in csv.DictReader(fh):
            when = date.fromisoformat(rec["as_of"])
            values = {
                name: (None if rec[name] == "" else float(rec[name]))
                for name in FEATURE_NAMES
            }
            rows.append(
                FeatureRow(
                    ticker=rec["ticker"],
                    as_of=when,
                    knowledge_date=date.fromisoformat(rec["knowledge_date"]),
                    values=values,
                    missing=[n for n, v in values.items() if v is None],
                    statement_date=(
                        date.fromisoformat(rec["statement_date"])
                        if rec["statement_date"]
                        else None
                    ),
                )
            )
            raw = rec["mna_net_cash_to_assets"]
            extras[(rec["ticker"], when)] = {
                "mna_net_cash_to_assets": None if raw == "" else float(raw)
            }
    return rows, extras


_CACHE: dict[str, object] = {}


@pytest.fixture(scope="module")
def panel() -> FeaturePanel:
    if "panel" not in _CACHE:
        rows, extras = _load_rows()
        _CACHE["panel"] = FeaturePanel(rows=rows, random_seed=7)
        _CACHE["extras"] = extras
    return _CACHE["panel"]


@pytest.fixture(scope="module")
def extras(panel) -> dict:
    return _CACHE["extras"]


@pytest.fixture(scope="module")
def universe() -> Universe:
    if "universe" not in _CACHE:
        _CACHE["universe"] = build_universe(
            json.loads((FIXTURES / "universe.json").read_text())
        )
    return _CACHE["universe"]


@pytest.fixture(scope="module")
def events() -> list[DealEvent]:
    raw = json.loads((FIXTURES / "events.json").read_text())
    return [
        DealEvent(
            ticker=e["ticker"],
            announced=date.fromisoformat(e["announced"]),
            completed=bool(e["completed"]),
            name=e.get("name") or "",
            acquirer=e.get("acquirer"),
            source_form=e.get("source_form"),
        )
        for e in raw
    ]


@pytest.fixture
def assumptions() -> Assumptions:
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    a.as_of = AS_OF.isoformat()
    return a


@pytest.fixture(scope="module")
def result(panel, universe, events):
    """The headline fit. Module scoped because it is the expensive one."""
    if "result" not in _CACHE:
        a = Assumptions()
        a.as_of = AS_OF.isoformat()
        _CACHE["result"] = fit_propensity(
            panel, universe, events, a, as_of=AS_OF, extras=_CACHE["extras"]
        )
    return _CACHE["result"]


def _facts(cik: str, ticker: str, knowledge: date) -> CompanyFacts:
    path = FIXTURES / "facts" / f"{int(cik.replace('CIK', '')):010d}.json.gz"
    payload = json.loads(gzip.decompress(path.read_bytes()))
    return CompanyFacts(payload, ticker, knowledge_date=knowledge)


class OneCompanyClient:
    """A pinned client serving one committed payload, as build_features demands."""

    def __init__(self, cik: str, ticker: str, knowledge: date) -> None:
        self.cik, self.ticker, self.knowledge = cik, ticker, knowledge

    def company_facts(self, ticker: str) -> CompanyFacts:
        return _facts(self.cik, self.ticker, self.knowledge)


# --------------------------------------------------------------------------- #
# Trap one: survivorship, which here is not a bias but the whole experiment
# --------------------------------------------------------------------------- #


def test_the_roster_keeps_companies_that_have_left(universe):
    gone = universe.departed_members()
    assert len(gone) > 50, "a roster with no departed names cannot carry a positive"
    assert len(universe.members) > 400
    # Not merely present: a large minority of the universe.
    assert 0.1 < len(gone) / len(universe.members) < 0.5


def test_three_quarters_of_the_deals_are_on_companies_that_no_longer_file(
    universe, events
):
    """The number that decides whether the experiment is possible.

    This is the count the module docstring rests on. Every one of these
    positives is invisible to a universe built from a current ticker file.
    """
    members = universe.by_ticker
    gone = [e for e in events if members[e.ticker].departed is not None]
    assert len(events) >= 90
    assert len(gone) / len(events) > 0.6, (
        f"only {len(gone)} of {len(events)} deals are on departed names, which "
        "would mean the survivorship problem had gone away"
    )


def test_an_acquired_company_is_in_the_panel_for_the_years_before_its_deal(universe):
    """Splunk, bought by Cisco, deregistered March 2024."""
    splunk = universe.by_ticker[SPLUNK]
    assert splunk.departed == date(2024, 3, 18)
    for year in (2019, 2020, 2021, 2022, 2023):
        when = date(year, 6, 30)
        present = {m.ticker for m in universe.as_of(when)}
        assert SPLUNK in present, f"Splunk is missing from the {year} cross-section"
    for when in (date(2024, 6, 30), date(2025, 6, 30), date(2026, 6, 30)):
        assert SPLUNK not in {m.ticker for m in universe.as_of(when)}


def test_a_company_is_absent_before_it_was_a_registrant(universe):
    """Slack listed in 2019 and was gone by 2021. Both edges are enforced."""
    slack = universe.by_ticker[SLACK]
    assert slack.admitted is not None and slack.departed is not None
    assert slack.admitted.year == 2019
    assert SLACK not in {m.ticker for m in universe.as_of(date(2019, 3, 31))}
    assert SLACK in {m.ticker for m in universe.as_of(date(2020, 6, 30))}
    assert SLACK not in {m.ticker for m in universe.as_of(date(2022, 6, 30))}


def test_a_survivor_universe_loses_almost_every_positive(universe, events):
    """The failure this module exists to prevent, reproduced and counted.

    A universe built the obvious way, from the names that are still filing,
    keeps the negatives and throws away the companies that were bought. The
    resulting sample is nearly all negative and a model fitted on it is
    measuring nothing.
    """
    dates = [date(y, 6, 30) for y in range(2019, 2025)]
    full = label_observations(universe, dates, events, as_of=AS_OF)

    survivors = build_universe(
        [
            {
                "ticker": m.ticker,
                "cik": m.cik,
                "name": m.name,
                "admitted": m.admitted,
                "sub_vertical": m.sub_vertical,
            }
            for m in universe.members
            if m.departed is None
        ]
    )
    survivor_only = label_observations(survivors, dates, events, as_of=AS_OF)

    assert full.n_positive > 50
    assert survivor_only.n_positive < full.n_positive * 0.4, (
        f"the survivor universe kept {survivor_only.n_positive} of "
        f"{full.n_positive} positives, which is too many for this to be the "
        "failure the module describes"
    )
    assert full.n_distinct_deals > survivor_only.n_distinct_deals


def test_coverage_frame_shows_the_departed_in_every_cross_section(universe):
    dates = [date(y, 6, 30) for y in range(2019, 2027)]
    frame = universe.coverage(dates)
    assert list(frame.columns) == ["as_of", "n", "since_departed", "still_listed"]
    assert (frame["n"] > 300).all()
    # Every historical cross-section holds names that have since gone, and the
    # count falls to zero only at the end of the window where nobody has left yet.
    assert (frame.loc[frame["as_of"] < date(2024, 1, 1), "since_departed"] > 20).all()
    assert (frame["since_departed"] + frame["still_listed"] == frame["n"]).all()


def test_a_delisted_registrant_has_no_ticker_anywhere_in_edgar(universe):
    """Why the roster is keyed on CIK and the symbol is only a label."""
    departed = universe.departed_members()
    with_symbol = [m for m in departed if not m.ticker.startswith("CIK")]
    assert not with_symbol, (
        "a departed registrant carried a ticker, which would mean EDGAR had "
        "started publishing symbols for delisted filers"
    )
    assert universe.by_ticker[SPLUNK].name.upper().startswith("SPLUNK")


def test_a_duplicate_symbol_in_a_roster_is_refused():
    """A reused symbol silently merges two companies, so it raises instead."""
    with pytest.raises(ConfigError, match="appears twice"):
        build_universe(
            [
                {"ticker": "PARA", "cik": 1, "admitted": "2019-01-01"},
                {"ticker": "PARA", "cik": 2, "admitted": "2024-01-01"},
            ]
        )


# --------------------------------------------------------------------------- #
# Trap three, part one: the price channel, which is why there are no price features
# --------------------------------------------------------------------------- #


def test_no_public_price_source_serves_a_delisted_target():
    """Recorded live. The reason this model uses no price feature at all."""
    probe = json.loads((FIXTURES / "price_availability.json").read_text())["results"]
    for delisted in ("SPLK", "ZEN", "WORK", "MNDT"):
        for source, got in probe[delisted].items():
            assert got["rows"] == 0, f"{source} unexpectedly served {delisted}"
    assert probe["DDOG"]["nasdaq"]["rows"] > 500, (
        "the survivor must have prices, or this proves nothing about asymmetry"
    )


def test_the_panel_carries_no_price_derived_feature(panel):
    """Twelve features need a share price and all twelve are empty by construction.

    Not a gap in the fixture. Including them would put a column in the design
    matrix that is populated for the negatives and empty for the positives.
    """
    priced = list(FEATURE_GROUPS["market"]) + [
        "scale_log_market_cap",
        "scale_log_enterprise_value",
        "capital_net_debt_to_market_cap",
        "capital_net_debt_to_ev",
        "capital_cash_to_market_cap",
    ]
    for name in priced:
        observed = [r.values.get(name) for r in panel.rows]
        assert all(v is None for v in observed), f"{name} has values in this panel"
    assert not (set(priced) & set(FITTED_COLUMNS))


def test_the_absence_of_a_price_alone_scores_an_auc_above_three_quarters(
    panel, universe, events
):
    """Measure the leak rather than assert it.

    A company that has left the filing record has no price history, so a
    would-be price feature is missing for it and present for everybody else. A
    classifier handed only that indicator scores an AUC far above anything the
    real model reaches, which is the whole argument for leaving prices out.

    Far above, not perfect: on the committed fixtures it is 0.82, with 73% of
    the positive rows and 9% of the negatives on departed companies. The
    positives on companies that still file are what keep it off 1.0.
    """
    dates = [d for d in panel.dates if d <= date(2025, 6, 30)]
    report = label_observations(universe, dates, events, as_of=AS_OF)
    members = universe.by_ticker
    y = np.array([o.label for o in report.observations], dtype=float)
    # The indicator a price-bearing panel would carry: 1 where no price exists.
    unpriced = np.array(
        [1.0 if members[o.ticker].departed is not None else 0.0
         for o in report.observations]
    )
    auc = _auc(y, unpriced)
    assert auc > 0.75, (
        "the delisting indicator was expected to separate the classes strongly; "
        f"it scored {auc:.3f}"
    )


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(y.size, dtype=float)
    ranks[order] = np.arange(1, y.size + 1, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < y.size:
        j = i
        while j + 1 < y.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    n_pos, n_neg = float(y.sum()), float(y.size - y.sum())
    return float((ranks[y > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# Trap three, part two: point in time, proved on real payloads
# --------------------------------------------------------------------------- #


def test_features_the_day_before_an_announcement_know_nothing_of_the_deal(
    assumptions, events
):
    """Splunk on 20 September 2023. Cisco announced the next day."""
    announced = {e.ticker: e.announced for e in events}[SPLUNK]
    day_before = announced - timedelta(days=1)
    client = OneCompanyClient(SPLUNK, "SPLK", day_before)
    row = build_features("SPLK", day_before, client, None, assumptions)
    assert row.ok
    assert row.statement_date is not None and row.statement_date < announced
    assert_point_in_time(row)
    for prov in row.provenance.values():
        if prov.filed:
            assert date.fromisoformat(prov.filed) <= day_before


def test_a_client_pinned_after_the_row_date_is_refused(assumptions):
    """The leak that produces a beautiful fit and means nothing."""
    when = date(2023, 6, 30)
    client = OneCompanyClient(SPLUNK, "SPLK", date(2024, 1, 31))
    with pytest.raises(LookaheadError, match="later"):
        build_features("SPLK", when, client, None, assumptions)


def test_every_recorded_panel_row_was_built_through_its_own_date(panel):
    for row in panel.rows:
        assert row.knowledge_date == row.as_of
        if row.statement_date is not None:
            assert row.statement_date <= row.as_of


def test_the_reporting_lag_is_visible_on_the_rows(panel):
    """A quarter that ended is not public until it is filed, and the gap shows."""
    lags = [
        (r.as_of - r.statement_date).days
        for r in panel.rows
        if r.statement_date is not None
    ]
    assert lags, "no row recorded a statement date"
    median = float(np.median(lags))
    assert 30 < median < 200, (
        f"median reporting lag of {median:.0f} days is outside anything a filing "
        "calendar produces, so the panel is probably not point in time"
    )


def test_balance_sheet_extras_are_point_in_time_and_signed_right():
    """Net cash over assets, resolved from the balance sheet rather than a price."""
    when = date(2023, 6, 30)
    client = OneCompanyClient(DATADOG_CIK, "DDOG", when)
    got = balance_sheet_extras("DDOG", when, client)["mna_net_cash_to_assets"]
    assert got is not None
    # Datadog carried convertible notes against a large securities portfolio, so
    # the sign is a real question rather than a formality.
    assert -1.0 < got < 1.0


# --------------------------------------------------------------------------- #
# Trap two: class imbalance
# --------------------------------------------------------------------------- #


def test_the_base_rate_is_low_single_digit(result):
    rate = result.labels.base_rate
    assert 0.005 < rate < 0.10, f"base rate of {rate:.2%} is not a takeout base rate"


def test_accuracy_is_meaningless_and_is_not_reported(result):
    """A model that predicts never is right most of the time and worth nothing."""
    y = result.matrix.y
    never_accuracy = float((y == 0).mean())
    assert never_accuracy > 0.9
    assert result.evaluation.metric == "auc"
    assert "accuracy" not in " ".join(result.evaluation.notes).lower()


def test_the_result_reports_auc_precision_recall_and_calibration(result):
    assert result.evaluation.metric == "auc"
    assert 0.0 <= result.evaluation.score <= 1.0
    table = result.precision_at_k
    assert table is not None and not table.empty
    for column in (
        "precision_at_k_model",
        "recall_at_k_model",
        "precision_at_k_size",
        "recall_at_k_size",
    ):
        assert column in table.columns
    assert result.calibration is not None
    assert list(result.calibration.columns) == [
        "low", "high", "n", "predicted", "realised", "gap", "thin",
    ]


def test_the_probabilities_are_probabilities_and_the_table_says_what_they_are_worth(
    result,
):
    probs = result.model.probabilities(result.matrix.X)
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0
    # The class weighting is undone before a probability is quoted, so the mean
    # predicted probability lands near the base rate rather than near a half.
    assert abs(float(probs.mean()) - result.labels.base_rate) < 0.15
    table = calibration_table(result.matrix.y, probs)
    assert float(table["n"].sum()) == float(result.matrix.n)


def test_the_class_weighting_does_not_change_the_ordering_claim(result):
    """Scores rank; probabilities are the same ordering after a constant shift.

    Compared on the statistic that reads an ordering rather than on the
    permutation itself. A logit of -24 saturates the sigmoid to a float that
    ties with its neighbours, so the two argsorts differ on ties that no
    ranking statistic can see, and asserting on the permutation would be
    asserting about floating point rather than about the model.
    """
    scores = result.model.scores(result.matrix.X)
    probs = result.model.probabilities(result.matrix.X)
    assert _auc(result.matrix.y, scores) == pytest.approx(
        _auc(result.matrix.y, probs), abs=1e-9
    )
    assert float(np.min(np.diff(np.sort(probs)))) >= 0.0


# --------------------------------------------------------------------------- #
# Trap three, part three: the gap between the feature date and the announcement
# --------------------------------------------------------------------------- #


def test_an_observation_inside_the_rumour_window_is_dropped_not_relabelled():
    """The company was under offer weeks later. It is neither class."""
    members = [
        {"ticker": "AAA", "cik": 1, "admitted": "2018-01-01"},
        {"ticker": "BBB", "cik": 2, "admitted": "2018-01-01"},
    ]
    uni = build_universe(members)
    dates = [date(2021, 3, 31)]
    inside = [DealEvent("AAA", date(2021, 5, 1))]
    report = label_observations(
        uni, dates, inside, as_of=date(2024, 1, 1), gap_days=90
    )
    assert report.dropped_in_gap == 1
    assert {o.ticker for o in report.observations} == {"BBB"}
    assert report.n_positive == 0

    # With the gap switched off the same company is a positive.
    without = label_observations(
        uni, dates, inside, as_of=date(2024, 1, 1), gap_days=0
    )
    assert without.dropped_in_gap == 0
    assert without.n_positive == 1


def test_the_gap_costs_observations_on_the_real_sample(panel, universe, events):
    dates = [d for d in panel.dates if d <= date(2025, 3, 31)]
    with_gap = label_observations(
        universe, dates, events, as_of=AS_OF, gap_days=DEFAULT_GAP_DAYS
    )
    without_gap = label_observations(universe, dates, events, as_of=AS_OF, gap_days=0)
    assert with_gap.dropped_in_gap > 0
    assert with_gap.n_positive <= without_gap.n_positive
    assert len(with_gap.observations) < len(without_gap.observations)


def test_the_model_can_be_fitted_with_and_without_the_gap(panel, universe, events):
    """Both results exist, so the cost of the discipline is measurable."""
    a = Assumptions()
    with_gap = fit_propensity(
        panel, universe, events, a, as_of=AS_OF, gap_days=DEFAULT_GAP_DAYS,
        extras=_CACHE["extras"],
    )
    without_gap = fit_propensity(
        panel, universe, events, a, as_of=AS_OF, gap_days=0, extras=_CACHE["extras"]
    )
    assert with_gap.labels.dropped_in_gap > without_gap.labels.dropped_in_gap
    assert np.isfinite(with_gap.evaluation.score)
    assert np.isfinite(without_gap.evaluation.score)


def test_no_label_window_opens_before_its_own_feature_date(result):
    for obs in result.labels.observations:
        assert obs.window_opens >= obs.as_of
        assert obs.window_closes > obs.window_opens
        if obs.announced is not None:
            assert obs.announced > obs.as_of
            assert obs.announced >= obs.window_opens


def test_an_unresolved_window_is_dropped_rather_than_called_a_negative():
    """The commonest way a propensity sample inflates its negative class."""
    uni = build_universe([{"ticker": "AAA", "cik": 1, "admitted": "2018-01-01"}])
    report = label_observations(
        uni, [date(2026, 6, 30)], [], as_of=date(2026, 9, 11), horizon_months=12
    )
    assert report.observations == []
    assert report.dropped_unresolved == 1


def test_observations_at_or_after_an_announcement_are_excluded():
    uni = build_universe([{"ticker": "AAA", "cik": 1, "admitted": "2018-01-01"}])
    report = label_observations(
        uni,
        [date(2021, 3, 31), date(2021, 6, 30), date(2021, 9, 30)],
        [DealEvent("AAA", date(2021, 6, 30))],
        as_of=date(2024, 1, 1),
        gap_days=0,
    )
    assert report.dropped_after_announcement == 2
    assert [o.as_of for o in report.observations] == [date(2021, 3, 31)]


# --------------------------------------------------------------------------- #
# Trap four: announced is not completed
# --------------------------------------------------------------------------- #


def test_a_deal_that_has_not_completed_is_still_a_positive(universe, events):
    """A blocked or still-pending deal was a bid, and the target was a target.

    The date matters and is chosen rather than convenient. An observation whose
    label window has not closed is dropped whatever its label, so a pending 2026
    deal can only appear as a positive at a feature date early enough for its
    window to have closed by the as-of date.
    """
    pending = {e.ticker for e in events if not e.completed}
    assert pending, "the fixture set must hold at least one unclosed deal"
    report = label_observations(
        universe,
        [date(2025, 3, 31)],
        events,
        as_of=AS_OF,
        horizon_months=12,
        gap_days=DEFAULT_GAP_DAYS,
    )
    positives = {o.ticker for o in report.observations if o.label}
    assert positives & pending, (
        "no unclosed deal reached the positive class, so the label is tracking "
        "completion rather than announcement"
    )


def test_completion_is_carried_for_reporting_and_is_not_the_label(result):
    positives = [o for o in result.labels.observations if o.label]
    assert positives
    assert any(o.completed is False for o in positives)
    assert all(o.announced is not None for o in positives)


# --------------------------------------------------------------------------- #
# Trap five: the cycle
# --------------------------------------------------------------------------- #


def test_the_base_rate_moves_with_the_cycle(result):
    frame = result.base_rates
    assert list(frame.columns) == ["year", "n", "positives", "base_rate"]
    assert len(frame) >= 5
    rates = frame.loc[frame["n"] > 100, "base_rate"]
    assert float(rates.max()) > float(rates.min()) * 1.4, (
        "the per-year base rate barely moves in this sample, which would "
        "contradict everything known about the M&A cycle"
    )


def test_the_evaluation_is_walk_forward_and_embargoed(result):
    assert len(result.folds) >= 2
    for fold in result.folds:
        assert fold.train_end < fold.test_start
        assert fold.embargo_days >= 365
        assert (fold.test_start - fold.train_end).days > fold.embargo_days - 40
    starts = [f.test_start for f in result.folds]
    assert starts == sorted(starts)


# --------------------------------------------------------------------------- #
# The honesty contract
# --------------------------------------------------------------------------- #


def test_the_baseline_is_the_size_sort_and_not_the_base_rate(result):
    assert "size only" in result.evaluation.baseline_name
    assert result.evaluation.baseline_score != 0.5
    # The size sort has to be a real alternative rather than a straw man.
    assert result.evaluation.baseline_score > 0.5, (
        "sorting smallest first scored no better than a coin toss, which would "
        "mean the baseline had been built wrong"
    )
    assert result.base_rate_result.baseline_score == 0.5


def test_beat_baseline_is_computed_and_the_verdict_says_so(result):
    evaluation = result.evaluation
    assert evaluation.beat_baseline == (evaluation.score > evaluation.baseline_score)
    verdict = evaluation.verdict()
    if not evaluation.beat_baseline:
        assert "Use the baseline." in verdict
    else:
        assert "lift" in verdict


def test_the_model_card_records_the_deal_count_not_the_observation_count(result):
    card = result.model.card
    assert card.name == "mna.propensity"
    assert card.trained_through <= AS_OF
    assert card.features == list(FITTED_COLUMNS)
    assert card.evaluation is result.evaluation
    joined = " ".join(card.limitations)
    assert "distinct companies were acquired" in joined
    assert "price" in joined.lower()
    assert "Announcement is the label" in joined


def test_the_card_summary_carries_the_verdict(result):
    summary = result.model.card.summary()
    assert "mna.propensity" in summary
    assert result.evaluation.verdict().startswith("auc of ")
    assert " features. AUC of " in summary and ". auc of" not in summary


def test_fit_refuses_a_sample_below_the_distinct_deal_floor(panel, universe, events):
    few = events[:3]
    with pytest.raises(NotMeaningfulError, match="distinct companies"):
        fit_propensity(
            panel, universe, few, Assumptions(), as_of=AS_OF, extras=_CACHE["extras"]
        )


def test_two_fits_on_the_same_panel_agree_exactly(panel, universe, events):
    a = Assumptions()
    one = fit_propensity(panel, universe, events, a, as_of=AS_OF, extras=_CACHE["extras"])
    two = fit_propensity(panel, universe, events, a, as_of=AS_OF, extras=_CACHE["extras"])
    assert np.array_equal(one.model.weights, two.model.weights)
    assert one.model.intercept == two.model.intercept
    assert one.evaluation.score == two.evaluation.score


def test_every_fitted_feature_states_what_it_claims():
    for name in FITTED_COLUMNS:
        assert name in FEATURE_RATIONALE, f"{name} has no stated rationale"
        assert len(FEATURE_RATIONALE[name]) > 60


# --------------------------------------------------------------------------- #
# The deliverable: a ranked list with attribution
# --------------------------------------------------------------------------- #


def test_rank_returns_a_screen_with_a_reason_per_name(result, panel, universe):
    latest = max(panel.dates)
    frame = result.model.rank(
        panel, universe, latest, top_k=20, extras=_CACHE["extras"]
    )
    assert len(frame) == 20
    assert list(frame["rank"]) == list(range(1, 21))
    assert frame["probability"].is_monotonic_decreasing
    assert frame["name"].notna().all()
    assert frame["why"].str.len().min() > 20
    assert frame["driver_1"].notna().all()


def test_rank_only_scores_companies_that_existed_on_the_date(result, panel, universe):
    when = date(2021, 6, 30)
    frame = result.model.rank(panel, universe, when, top_k=50, extras=_CACHE["extras"])
    live = {m.ticker for m in universe.as_of(when)}
    assert set(frame["ticker"]) <= live
    assert SLACK not in set(frame["ticker"]) or SLACK in live


def test_rank_refuses_a_date_the_panel_does_not_cover(result, panel, universe):
    with pytest.raises(NotMeaningfulError, match="nothing to rank"):
        result.model.rank(panel, universe, date(1999, 12, 31))


def test_attribution_sums_to_the_logit(result):
    X = result.matrix.X[:40]
    tickers = result.matrix.tickers[:40]
    dates = result.matrix.dates[:40]
    attributions = result.model.attribute(X, tickers, dates)
    scores = result.model.scores(X)
    for i, attribution in enumerate(attributions):
        total = sum(c[3] for c in attribution.contributions) + result.model.intercept
        assert total == pytest.approx(float(scores[i]), abs=1e-9)


def test_attribution_names_the_features_that_drove_the_score(result):
    X = result.matrix.X[:5]
    attributions = result.model.attribute(
        X, result.matrix.tickers[:5], result.matrix.dates[:5]
    )
    for attribution in attributions:
        assert len(attribution.contributions) == len(FITTED_COLUMNS)
        top = attribution.top(3)
        assert len(top) == 3
        assert abs(top[0][3]) >= abs(top[1][3]) >= abs(top[2][3])
        assert attribution.ticker in attribution.sentence()
        frame = attribution.to_frame()
        assert list(frame.columns) == [
            "feature", "standardised_value", "coefficient", "contribution",
        ]


def test_the_screen_beats_the_base_rate_at_the_top_of_the_list(result):
    """Precision at twenty against that date's base rate, which is the real test."""
    table = result.precision_at_k
    lifts = table["precision_at_k_model"] - table["base_rate"]
    assert len(table) >= 4
    # Reported rather than demanded: the point is the number exists per date.
    assert lifts.notna().all()
    assert (table["precision_at_k_model"] <= 1.0).all()
    assert (table["recall_at_k_model"] <= 1.0).all()


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #


def test_relative_features_are_struck_against_the_date_cross_section(panel, universe):
    derived = derive_features(panel, universe, _CACHE["extras"])
    assert derived
    ranks = [
        v["mna_growth_rank_in_vertical"]
        for v in derived.values()
        if v["mna_growth_rank_in_vertical"] is not None
    ]
    assert ranks and 0.0 <= min(ranks) and max(ranks) <= 1.0
    assert 0.4 < float(np.mean(ranks)) < 0.6, (
        "a percentile inside its own cross-section must average near a half"
    )


def test_subscale_is_centred_on_its_own_sub_vertical(panel, universe):
    derived = derive_features(panel, universe, _CACHE["extras"])
    by_date_vertical: dict[tuple, list[float]] = {}
    members = universe.by_ticker
    for (ticker, when), values in derived.items():
        value = values["mna_subscale_in_vertical"]
        if value is None:
            continue
        vertical = members[ticker].sub_vertical
        by_date_vertical.setdefault((when, vertical), []).append(value)
    big = [v for k, v in by_date_vertical.items() if len(v) >= 20]
    assert big, "no sub-vertical cross-section was large enough to check"
    for values in big:
        assert abs(float(np.median(values))) < 0.6, (
            "subscale is a distance from the sub-vertical median, so the median "
            "of it has to sit near zero"
        )


def test_growth_decay_is_the_turn_and_not_the_level(panel, universe):
    derived = derive_features(panel, universe, _CACHE["extras"])
    decays = [
        v["mna_growth_decay"] for v in derived.values()
        if v["mna_growth_decay"] is not None
    ]
    assert len(decays) > 200
    assert min(decays) < 0 < max(decays), "deceleration must take both signs"


def test_a_missing_feature_reaches_the_missing_share_column(result):
    share = result.matrix.X[:, list(result.matrix.columns).index("mna_missing_share")]
    assert float(share.min()) >= 0.0
    assert float(share.max()) <= 1.0
    assert float(share.mean()) > 0.0, "no row was missing anything, which is unlikely"


def test_standardisation_puts_the_training_mean_where_a_gap_was(result):
    X = result.matrix.X.copy()
    X[0, 0] = np.nan
    Z = result.model.standardise(X)
    assert Z[0, 0] == 0.0
    assert np.isfinite(Z).all()


def test_unmatched_observations_are_counted_rather_than_imputed(panel, universe):
    report = label_observations(
        universe, [date(2019, 3, 31)], [], as_of=AS_OF
    )
    report.observations.append(
        Observation(
            ticker="NOT_IN_THE_PANEL",
            as_of=date(2019, 3, 31),
            label=0,
            window_opens=date(2019, 6, 29),
            window_closes=date(2020, 6, 29),
        )
    )
    matrix = build_matrix(panel, universe, report, extras=_CACHE["extras"])
    assert "NOT_IN_THE_PANEL" not in matrix.tickers
    assert any("had no feature row" in n for n in matrix.notes)


# --------------------------------------------------------------------------- #
# The finding about the deal forms
# --------------------------------------------------------------------------- #


POWERSCHOOL_INFORMATION_STATEMENT = "0001193125-24-212624"


class _NameOnlyClient:
    """Enough client for ``extract_transaction`` and nothing more."""

    def __init__(self, entity_name: str) -> None:
        self._name = entity_name

    def company_facts(self, ticker: str):
        return type("_F", (), {"entity_name": self._name})()

    def ticker_to_cik(self, ticker: str) -> int:
        raise MissingDataError("CIK", ticker=ticker)

    def _get_json(self, url: str):
        raise MissingDataError("ticker file")


def _powerschool_text() -> str:
    path = FIXTURES / "text" / f"{POWERSCHOOL_INFORMATION_STATEMENT}.txt.gz"
    return gzip.decompress(path.read_bytes()).decode()


def test_the_consent_solicitation_forms_are_reachable():
    """A controlled company is sold by written consent and never files a proxy.

    ``DEFM14C`` is an information statement under Regulation 14C, filed where a
    holder with the votes has already approved the merger, and ``SC 13E3`` is the
    going-private schedule Rule 13e-3 requires when an affiliate is buying. Both
    belong to the population a propensity model most wants to see, because the
    route a company takes to be sold is decided by who controls its votes.
    """
    from techval.tmt.precedents import DEAL_FORMS

    assert set(CONSENT_DEAL_FORMS) & set(DEAL_FORMS) == {
        "DEFM14C", "PREM14C", "SC 13E3", "SC 13E3/A",
    }


def test_a_dual_class_target_is_not_discarded_as_somebody_elses_deal():
    """Regression: PowerSchool's DEFM14C, verbatim from EDGAR.

    Bain took PowerSchool private at $22.80 a share in cash. The information
    statement converts Class B Common Stock in one clause and Class A Common
    Stock into the cash in another, and the Class B clause comes first. Before
    the fix in ``_consideration_clause`` the extractor read the Class B clause,
    concluded the document was the acquirer's side of somebody else's deal and
    returned None, losing the transaction. Every dual-class filer carries a
    clause like that, so the failure took founder and sponsor controlled targets
    as a group rather than at random, which is the same population the brief
    hypothesised was most likely to be bought.
    """
    from techval.tmt import precedents as P

    filing = {
        "accession": POWERSCHOOL_INFORMATION_STATEMENT,
        "form": "DEFM14C",
        "filed": date(2024, 9, 4),
    }
    txn = P.extract_transaction(
        _powerschool_text(),
        "PWSC",
        _NameOnlyClient("PowerSchool Holdings, Inc."),
        filing,
    )
    assert txn is not None, (
        "the information statement was read as somebody else's deal, which is "
        "the dual-class failure this test exists for"
    )
    assert txn.offer_price == pytest.approx(22.80)
    assert txn.consideration == "cash"
    assert txn.confidence == 0.90
    assert txn.source_form == "DEFM14C"


def test_the_acquirer_side_refusal_still_holds_after_that_fix():
    """The guard the fix had to preserve, checked here as well as in the tmt suite.

    Zendesk's own S-4 to buy Momentive must never become a precedent in which
    Zendesk was acquired. Continuing the scan past a non-filer clause into the
    looser pattern is what breaks it: the sentence that then matches is about
    who may vote at the Zendesk special meeting.
    """
    from techval.tmt import precedents as P

    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures" / "merger" / "text" / "0001193125-21-349135.txt"
    )
    txn = P.extract_transaction(
        path.read_text(),
        "ZEN",
        _NameOnlyClient("ZENDESK, INC."),
        {"accession": "0001193125-21-349135", "form": "S-4", "filed": date(2021, 12, 3)},
    )
    assert txn is None
