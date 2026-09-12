"""The harness is only worth what its own tests are worth.

The central test in this file is the leakage test. It builds a panel whose
label is a forward twelve month average, which is the shape of every label this
package cares about: next year's revenue growth, the return over the year after
a screen, the probability of being acquired within twelve months. Two
observations one month apart then share eleven twelfths of their label window,
so a model allowed to remember the most recent training label is being handed
most of the answer. The test scores that model twice on identical test windows,
once with the training cut sitting right against the test window and once with
a 365 day embargo, and the whole result is the difference: skill without the
embargo, worse than a constant with it. That is what an unembargoed backtest
buys you, and it is why the number in a pitch book has to say which one it was.

Everything else here checks that a figure is refused rather than guessed. NDCG
is checked against the arithmetic written out in its own docstring, calibration
against a model built to be well calibrated and a model built to be
overconfident in a way that leaves its ordering untouched, and ablation against
features whose true contributions are known by construction.

No test in this file touches the network or reads a fixture. Every panel is
generated from a seeded numpy Generator, so the numbers are the same on every
machine.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from techval.errors import ConfigError, NotMeaningfulError
from techval.ml.evaluation import (
    Fold,
    ablation,
    calibration_table,
    evaluate_classification,
    evaluate_ranking,
    evaluate_regression,
    ndcg_at_k,
    precision_recall_at_k,
    walk_forward_folds,
)
from techval.ml.protocol import EvalResult

# The label window in months. Twelve is the horizon the growth model forecasts
# over and the window the takeout model asks about, and it is the number that
# makes an unembargoed split leak.
FORWARD_MONTHS = 12


def month_starts(n: int, start_year: int = 2016) -> list[date]:
    """Monthly observation dates, which is how a filing panel actually arrives."""
    out, year, month = [], start_year, 1
    for _ in range(n):
        out.append(date(year, month, 1))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def forward_panel(
    seed: int = 4, n_months: int = 128, tickers: tuple[str, ...] = ("AAA", "BBB", "CCC")
) -> tuple[list[date], list[str], np.ndarray]:
    """A panel whose label for date t is the mean shock over (t, t+12 months].

    The label is unknowable on the observation date and only closes a year
    later, which is the property that makes a same-day train/test cut a
    fiction. Consecutive labels overlap in eleven of twelve monthly shocks and
    labels twelve months apart share none, so the leak has a known shape:
    strong at short lag, exactly zero at a lag of a year.
    """
    rng = np.random.default_rng(seed)
    timeline = month_starts(n_months)
    shocks = {t: rng.normal(size=n_months + FORWARD_MONTHS) for t in tickers}
    dates: list[date] = []
    names: list[str] = []
    labels: list[float] = []
    for i, day in enumerate(timeline):
        for ticker in tickers:
            dates.append(day)
            names.append(ticker)
            labels.append(float(np.mean(shocks[ticker][i + 1 : i + 1 + FORWARD_MONTHS])))
    return dates, names, np.asarray(labels, dtype=float)


def same_windows_without_embargo(folds: list[Fold], dates: list[date]) -> list[Fold]:
    """The same test windows with the training cut pushed up against them.

    Holding the test windows fixed is what makes the comparison fair. Both
    versions are scored on identical observations, so the only thing that
    changes is how recent the last training label is allowed to be.
    """
    out = []
    for fold in folds:
        train = [d for d in dates if d < fold.test_start]
        out.append(
            Fold(
                index=fold.index,
                train_end=max(train),
                test_start=fold.test_start,
                test_end=fold.test_end,
                n_train=len(train),
                n_test=fold.n_test,
                embargo_days=0,
                n_embargoed=0,
            )
        )
    return out


def carry_last_known_label(
    dates: list[date], names: list[str], y: np.ndarray, folds: list[Fold]
) -> np.ndarray:
    """Predict each test row with that company's most recent training label.

    This is not a straw man. It is what any fitted model does implicitly when
    the sample holds a persistent company-level effect and the cut sits one
    month before the test window: the freshest training label for a name is
    almost the same number as the label being predicted, because the two
    windows overlap.
    """
    pred = np.full(y.size, np.nan)
    for fold in folds:
        for ticker in set(names):
            prior = [
                i
                for i, (d, t) in enumerate(zip(dates, names))
                if t == ticker and d <= fold.train_end
            ]
            latest = y[max(prior, key=lambda i: dates[i])]
            for i, (d, t) in enumerate(zip(dates, names)):
                if t == ticker and fold.test_start <= d <= fold.test_end:
                    pred[i] = latest
    return pred


# ---------------------------------------------------------------- walk forward


def test_folds_march_forward_and_never_overlap():
    """Each fold tests a window strictly after its own training cut."""
    dates, _, _ = forward_panel()
    folds = walk_forward_folds(dates, 5, min_train=200)

    assert [f.index for f in folds] == [0, 1, 2, 3, 4]
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert fold.test_start <= fold.test_end
        assert fold.n_test > 0
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end < later.test_start
        # Expanding, not rolling: a model run today has all of the past.
        assert later.n_train > earlier.n_train


def test_embargo_opens_a_dated_gap_and_counts_what_it_cost():
    """Under an embargo the last training row predates the test window by the horizon."""
    dates, _, _ = forward_panel()
    embargoed = walk_forward_folds(dates, 5, min_train=300, embargo_days=365)
    adjacent = walk_forward_folds(dates, 5, min_train=300, embargo_days=0)

    for fold in embargoed:
        assert fold.train_end < fold.test_start - timedelta(days=365)
        assert fold.n_embargoed > 0
        # Nothing is double counted: the gap holds rows used by neither side.
        in_gap = [
            d
            for d in dates
            if fold.test_start - timedelta(days=365) <= d < fold.test_start
        ]
        assert fold.n_embargoed == len(in_gap)
    for fold in adjacent:
        assert fold.n_embargoed == 0

    # The embargo costs a year of testable sample at the front, and on the same
    # window it costs a year of training rows at the back.
    assert embargoed[0].test_start > adjacent[0].test_start
    twin = same_windows_without_embargo(embargoed, dates)[0]
    assert embargoed[0].n_train < twin.n_train
    assert embargoed[0].train_end < twin.train_end


def test_declining_the_embargo_is_recorded_rather_than_assumed():
    """A zero embargo is a choice, and the result has to say it was made."""
    dates, _, y = forward_panel()
    pred = y + 0.01
    result = evaluate_regression(y, pred, None, dates=dates, n_folds=4, min_train=200)

    assert any("No embargo was applied" in note for note in result.notes)
    assert any("leakage" in note for note in result.notes)

    protected = evaluate_regression(
        y, pred, None, dates=dates, n_folds=4, min_train=200, embargo_days=365
    )
    assert any("365-day embargo" in note for note in protected.notes)


def test_walk_forward_refuses_what_it_cannot_split():
    """Every refusal names what was asked for against what the sample holds."""
    dates, _, _ = forward_panel()

    with pytest.raises(NotMeaningfulError, match="below the floor"):
        walk_forward_folds(dates[:20], 3)
    with pytest.raises(ConfigError, match="at least 2 folds"):
        walk_forward_folds(dates, 1)
    with pytest.raises(ConfigError, match="embargo_days must be"):
        walk_forward_folds(dates, 3, embargo_days=-1)
    with pytest.raises(ConfigError, match="distinct"):
        # Forty observations on three dates cannot make five dated blocks.
        walk_forward_folds([date(2026, 1, 1)] * 20 + [date(2026, 2, 1)] * 20, 5)
    with pytest.raises(ConfigError, match="no walk-forward split"):
        # A ten year embargo on an eleven year panel leaves nothing to train on.
        walk_forward_folds(dates, 5, min_train=200, embargo_days=3650)


# -------------------------------------------------------------------- leakage


def test_leakage_looks_like_skill_until_the_embargo_removes_it():
    """The same model, the same test rows, and opposite verdicts.

    Without the embargo the model carries a label whose window overlaps the
    test window by up to eleven months and beats a constant comfortably. With
    365 days between the cut and the test window that same rule is reaching
    back past the end of its own label window, the overlap is zero, and it
    loses to the training mean. Nothing about the model changed.
    """
    dates, names, y = forward_panel()
    embargoed = walk_forward_folds(dates, 5, min_train=300, embargo_days=365)
    adjacent = same_windows_without_embargo(embargoed, dates)
    assert [f.test_start for f in adjacent] == [f.test_start for f in embargoed]

    leaked = evaluate_regression(
        y,
        carry_last_known_label(dates, names, y, adjacent),
        None,
        dates=dates,
        folds=adjacent,
    )
    honest = evaluate_regression(
        y,
        carry_last_known_label(dates, names, y, embargoed),
        None,
        dates=dates,
        folds=embargoed,
    )

    # Same test rows on both sides, so the two scores are comparable.
    assert leaked.n_observations == honest.n_observations

    assert leaked.beat_baseline
    assert leaked.score < leaked.baseline_score * 0.75

    assert not honest.beat_baseline
    assert honest.score > honest.baseline_score
    assert "does NOT beat the baseline" in honest.verdict()

    # The leak is worth more than the model: error roughly doubles without it.
    assert honest.score > leaked.score * 1.5


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 9])
def test_the_leak_is_a_property_of_the_split_not_of_one_lucky_panel(seed: int):
    """Closing the embargo raises the error on every panel, not just the one above.

    Whether the leaked model also clears its baseline depends on how far the
    training mean sits from the test period, which drifts panel to panel. What
    does not vary is the direction and the size: the same prediction rule is
    between one and a half and three times worse once it can no longer reach a
    label whose window was still open.
    """
    dates, names, y = forward_panel(seed=seed)
    embargoed = walk_forward_folds(dates, 5, min_train=300, embargo_days=365)
    adjacent = same_windows_without_embargo(embargoed, dates)

    leaked = evaluate_regression(
        y, carry_last_known_label(dates, names, y, adjacent), None, dates=dates, folds=adjacent
    )
    honest = evaluate_regression(
        y, carry_last_known_label(dates, names, y, embargoed), None, dates=dates, folds=embargoed
    )
    assert honest.score > leaked.score * 1.5


def test_embargoed_folds_never_train_on_an_open_label_window():
    """The training cut is strict, so a window closing on the first test date is out."""
    dates, _, _ = forward_panel()
    folds = walk_forward_folds(dates, 4, min_train=250, embargo_days=365)
    for fold in folds:
        train = [d for d in dates if d < fold.test_start - timedelta(days=fold.embargo_days)]
        assert max(train) == fold.train_end
        # The label of the last training row closed before the test window opened.
        assert fold.train_end + timedelta(days=365) <= fold.test_start


# ----------------------------------------------------------------- regression


def test_regression_invents_the_obvious_baseline_and_names_it():
    """A caller who supplies no baseline gets one, and is told which one."""
    dates, _, y = forward_panel()
    pred = y * 0.5

    folded = evaluate_regression(y, pred, None, dates=dates, n_folds=4, min_train=200)
    assert folded.baseline_name == "training-mean constant"
    assert folded.baseline_score > 0

    pooled = evaluate_regression(y, pred, None)
    assert "sample-mean" in pooled.baseline_name
    assert "in sample" in pooled.baseline_name
    assert any("no fold structure" in note for note in pooled.notes)
    assert pooled.folds == []


def test_regression_reports_fold_dispersion_beside_the_lift():
    """A lift inside the fold noise has to be visible as such, not rounded away."""
    dates, _, y = forward_panel()
    rng = np.random.default_rng(11)
    pred = y + rng.normal(scale=0.05, size=y.size)

    result = evaluate_regression(
        y, pred, None, dates=dates, n_folds=5, min_train=200, embargo_days=365
    )
    assert len(result.folds) == 5
    assert result.fold_sd is not None
    assert result.beat_baseline
    assert "Fold standard deviation" in result.verdict()
    assert any("beat the baseline in" in note for note in result.notes)


def test_regression_refuses_rather_than_scoring_around_a_gap():
    """Missing rows are dropped by the caller with a reason, never silently."""
    dates, _, y = forward_panel()
    folds = walk_forward_folds(dates, 4, min_train=200)
    scored = next(
        i for i, d in enumerate(dates) if folds[0].test_start <= d <= folds[0].test_end
    )

    pred = y.copy()
    pred[scored] = np.nan
    with pytest.raises(ConfigError, match="missing or infinite"):
        evaluate_regression(y, pred, None, dates=dates, folds=folds)

    infinite = y.copy()
    infinite[scored] = np.inf
    with pytest.raises(ConfigError, match="missing or infinite"):
        evaluate_regression(y, infinite, None, dates=dates, folds=folds)


def test_regression_rejects_misaligned_and_unscorable_inputs():
    dates, _, y = forward_panel()

    with pytest.raises(ConfigError, match="aligned row for row"):
        evaluate_regression(y, y[:-1], None)
    with pytest.raises(ConfigError, match="unknown regression metric"):
        evaluate_regression(y, y, None, metric="mape")
    with pytest.raises(ConfigError, match="folds were supplied without dates"):
        evaluate_regression(y, y, None, folds=walk_forward_folds(dates, 3, min_train=200))
    with pytest.raises(NotMeaningfulError, match="below the floor"):
        evaluate_regression(y[:12], y[:12], None)
    with pytest.raises(ConfigError, match="against"):
        evaluate_regression(y, y, y[:-1])


def test_spearman_metric_is_scored_the_right_way_round():
    """Rank correlation is higher is better, absolute error is not.

    The pair matters. A model whose ordering is perfect and whose level is
    nonsense is useful for a screen and useless for a price, and only reporting
    both metrics shows which of the two you have.
    """
    dates, _, y = forward_panel()
    pred = y * 3.0 + 2.0  # Order preserved exactly, level completely wrong.

    ranked = evaluate_regression(
        y, pred, None, dates=dates, n_folds=4, min_train=200, metric="spearman"
    )
    assert ranked.higher_is_better
    assert ranked.score == pytest.approx(1.0, abs=1e-9)
    # A constant has no ordering, so the naive alternative is zero correlation.
    assert ranked.baseline_score == 0.0
    assert "no ordering" in ranked.baseline_name
    assert ranked.lift == pytest.approx(1.0, abs=1e-9)
    assert ranked.folds == pytest.approx([1.0] * 4, abs=1e-9)

    absolute = evaluate_regression(
        y, pred, None, dates=dates, n_folds=4, min_train=200, metric="mae"
    )
    assert not absolute.higher_is_better
    assert not absolute.beat_baseline


# -------------------------------------------------------------------- ranking


DOC_GRADES = {"A": 3.0, "B": 2.0, "C": 0.0, "D": 1.0}


def test_ndcg_matches_the_worked_example_in_its_own_docstring():
    """Grades A=3 B=2 C=0 D=1 and the ranking [B, A, D, C] give 0.9225 at k=4."""
    dcg = 2 / np.log2(2) + 3 / np.log2(3) + 1 / np.log2(4)
    idcg = 3 / np.log2(2) + 2 / np.log2(3) + 1 / np.log2(4)
    assert dcg == pytest.approx(4.3928, abs=5e-5)
    assert idcg == pytest.approx(4.7619, abs=5e-5)

    assert ndcg_at_k(DOC_GRADES, ["B", "A", "D", "C"], 4) == pytest.approx(
        dcg / idcg, rel=1e-12
    )
    assert ndcg_at_k(DOC_GRADES, ["B", "A", "D", "C"], 4) == pytest.approx(0.9225, abs=5e-5)


def test_ndcg_is_one_for_the_ideal_order_and_falls_for_the_inverted_one():
    """Perfect ordering scores 1.0 by construction, and the reverse is hand checked."""
    assert ndcg_at_k(DOC_GRADES, ["A", "B", "D", "C"], 4) == pytest.approx(1.0, abs=1e-12)

    inverted = (0 / np.log2(2) + 1 / np.log2(3) + 2 / np.log2(4) + 3 / np.log2(5)) / (
        3 / np.log2(2) + 2 / np.log2(3) + 1 / np.log2(4)
    )
    assert ndcg_at_k(DOC_GRADES, ["C", "D", "B", "A"], 4) == pytest.approx(
        inverted, rel=1e-12
    )
    assert ndcg_at_k(DOC_GRADES, ["C", "D", "B", "A"], 4) == pytest.approx(0.6138, abs=5e-5)

    # Graded, not binary: the best name first beats the second best first.
    assert ndcg_at_k(DOC_GRADES, ["A", "B", "D", "C"], 4) > ndcg_at_k(
        DOC_GRADES, ["B", "A", "D", "C"], 4
    )


def test_ndcg_normalises_against_what_was_actually_ranked():
    """A relevant name that never entered the universe is a screening failure.

    Hiding it inside NDCG would blame the ranker for a name the ranker never
    saw, so the ideal ordering is taken over the candidates present. The miss
    shows up in recall instead.
    """
    grades = dict(DOC_GRADES, E=5.0)
    assert ndcg_at_k(grades, ["A", "B", "D", "C"], 4) == pytest.approx(1.0, abs=1e-12)
    _, recall = precision_recall_at_k({"A", "B", "D", "E"}, ["A", "B", "D", "C"], 4)
    assert recall == pytest.approx(0.75)


def test_ndcg_refuses_an_undefined_ratio():
    with pytest.raises(NotMeaningfulError, match="ideal gain is zero"):
        ndcg_at_k(DOC_GRADES, ["C", "Z"], 2)
    with pytest.raises(ConfigError, match="k must be at least 1"):
        ndcg_at_k(DOC_GRADES, ["A"], 0)
    with pytest.raises(ConfigError, match="cannot be negative"):
        ndcg_at_k({"A": -1.0}, ["A"], 1)


def test_precision_and_recall_answer_different_questions():
    """Precision is over what was shown, recall over the whole disclosed set."""
    relevant = {f"P{i}" for i in range(20)}
    ranking = ["P0", "P1", "P2", "P3", "P4"] + [f"X{i}" for i in range(95)]

    precision, recall = precision_recall_at_k(relevant, ranking, 10)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.25)

    # A universe of six is not four wasted slots.
    short = ["P0", "P1", "X0", "X1", "X2", "X3"]
    precision, recall = precision_recall_at_k(relevant, short, 10)
    assert precision == pytest.approx(2 / 6)
    assert recall == pytest.approx(0.1)

    with pytest.raises(NotMeaningfulError, match="relevant set is empty"):
        precision_recall_at_k(set(), ranking, 10)
    with pytest.raises(NotMeaningfulError, match="ranking is empty"):
        precision_recall_at_k(relevant, [], 10)


def peer_task(seed: int = 1, n_queries: int = 8, universe: int = 100, n_peers: int = 20):
    """Eight targets, a hundred candidates each, twenty of them disclosed peers.

    Grades are 2 for a name the company itself discloses and 1 for a name a
    banker would add, which is the graded structure NDCG exists to score.
    """
    rng = np.random.default_rng(seed)
    relevance, ranked, shuffled = {}, {}, {}
    for q in range(n_queries):
        names = [f"C{q}_{i}" for i in range(universe)]
        peers = list(rng.choice(names, size=n_peers, replace=False))
        grades = {name: 2.0 for name in peers[: n_peers // 2]}
        grades.update({name: 1.0 for name in peers[n_peers // 2 :]})
        relevance[f"T{q}"] = grades

        # A model with real but imperfect signal: peers get a head start, and
        # the noise is large enough that it does not simply sort them to the top.
        score = {name: rng.normal() + (1.6 if name in grades else 0.0) for name in names}
        ranked[f"T{q}"] = sorted(names, key=lambda n: -score[n])
        shuffled[f"T{q}"] = list(rng.permutation(names))
    return relevance, ranked, shuffled


def test_ranking_beats_the_random_baseline_and_says_where_the_bar_is():
    relevance, ranked, _ = peer_task()
    result = evaluate_ranking(relevance, ranked, k=10)

    assert result.metric == "ndcg@10"
    assert result.beat_baseline
    assert "random order" in result.baseline_name
    assert result.n_observations == 8
    # Per query values, so the dispersion is across targets rather than periods.
    assert len(result.folds) == 8
    assert result.fold_sd is not None
    assert any("across 8 targets" in note for note in result.notes)
    assert any("precision@10" in note for note in result.notes)
    assert any("sits near 0.20" in note for note in result.notes)


def test_a_shuffled_ranking_lands_on_its_random_expectation():
    """The closed form baseline is the number a coin toss actually earns."""
    relevance, _, shuffled = peer_task()
    result = evaluate_ranking(relevance, shuffled, k=10)

    assert result.score == pytest.approx(result.baseline_score, abs=0.15)

    precisions = [
        precision_recall_at_k(
            {n for n, g in relevance[q].items() if g > 0}, shuffled[q], 10
        )[0]
        for q in relevance
    ]
    # Twenty peers in a hundred names: random precision@10 sits near 0.20.
    assert 0.08 < float(np.mean(precisions)) < 0.32


def test_ranking_accepts_a_harder_baseline_but_insists_it_be_comparable():
    relevance, ranked, shuffled = peer_task()
    result = evaluate_ranking(
        relevance, ranked, k=10, baseline=shuffled, baseline_name="popularity order"
    )
    assert result.baseline_name == "popularity order"
    assert result.beat_baseline

    missing = {q: order for q, order in list(shuffled.items())[:-1]}
    with pytest.raises(ConfigError, match="no ordering for query"):
        evaluate_ranking(relevance, ranked, k=10, baseline=missing)

    trimmed = {q: order[:-5] for q, order in shuffled.items()}
    with pytest.raises(ConfigError, match="different candidate set"):
        evaluate_ranking(relevance, ranked, k=10, baseline=trimmed)


def test_ranking_refuses_a_handful_of_targets_and_flags_a_thin_one():
    relevance, ranked, _ = peer_task(n_queries=4)
    with pytest.raises(NotMeaningfulError, match="below the floor"):
        evaluate_ranking(relevance, ranked, k=10)

    relevance, ranked, _ = peer_task(n_queries=6)
    result = evaluate_ranking(relevance, ranked, k=10)
    assert any("is thin" in note for note in result.notes)


def test_ranking_counts_the_queries_it_could_not_score():
    """A target with no peer in its own candidate list is reported, not dropped."""
    relevance, ranked, _ = peer_task(n_queries=7)
    relevance["T7"] = {"nowhere_near": 2.0}
    ranked["T7"] = [f"C7_{i}" for i in range(100)]

    result = evaluate_ranking(relevance, ranked, k=10)
    assert result.n_observations == 7
    assert any("no relevant candidate" in note for note in result.notes)

    ranked.pop("T7")
    result = evaluate_ranking(relevance, ranked, k=10)
    assert any("no ranking and were not scored" in note for note in result.notes)


def test_a_ranking_result_declares_its_folds_are_queries_not_folds():
    """The README quoted a fold-noise sentence about numbers that are not folds.

    ``evaluate_ranking`` fills ``folds`` with per-query NDCG, so a verdict that
    judged the lift inside or outside "the fold-to-fold noise" was claiming
    fold dispersion about across-target dispersion: 1.25 sigma of the wrong
    quantity on the committed scoreboard, and the row a sceptical reader would
    attack. The result now says what its folds are, the verdict names the
    spread as across-query and refuses the inside-or-outside judgment, and the
    statistic it cites instead is the paired per-query difference, which is the
    error bar the lift actually has.
    """
    relevance, ranked, shuffled = peer_task()
    result = evaluate_ranking(
        relevance, ranked, k=10, baseline=shuffled, baseline_name="popularity order"
    )
    assert result.beat_baseline
    assert result.fold_unit == "query"

    text = result.verdict()
    assert "fold-to-fold noise" not in text.replace("not fold-to-fold noise", "")
    assert "Fold standard deviation" not in text
    assert f"spread across {result.n_observations:,} queries" in text
    assert "Paired on identical queries" in text

    # rows() must not label per-query scores as folds either.
    labels = dict(result.rows())
    assert "Folds" not in labels
    assert labels["Per-query scores"] == result.n_observations


def test_the_paired_delta_is_the_same_queries_paired_by_construction():
    """Mean of per-query differences equals the lift, and the shape is sane.

    Scored on identical queries, mean(model - baseline) is exactly
    mean(model) - mean(baseline), so ``paired.mean`` must reproduce ``lift`` to
    the last decimal. The rest is arithmetic on the same vector: the standard
    error is sd over root n, t is mean over the standard error, and the win
    rate lives in [0, 1].
    """
    relevance, ranked, shuffled = peer_task()
    result = evaluate_ranking(
        relevance, ranked, k=10, baseline=shuffled, baseline_name="popularity order"
    )
    p = result.paired
    assert p is not None
    assert p.n == result.n_observations
    assert p.mean == pytest.approx(result.lift)
    assert p.standard_error == pytest.approx(p.sd / np.sqrt(p.n))
    assert p.t == pytest.approx(p.mean / p.standard_error)
    assert 0.0 <= p.win_rate <= 1.0
    # A model with real signal against a shuffle should win nearly everywhere.
    assert p.t > 2.0


def test_a_walk_forward_verdict_still_judges_the_lift_against_fold_noise():
    """The fold sentence is the right one where the folds are folds.

    Regression and classification fill ``folds`` with walk-forward fold scores,
    ``fold_unit`` stays at its default, and the one-line verdict keeps the
    inside-or-outside judgment it has always made. This pins the default so the
    ranking fix cannot silently take the honest sentence away from the results
    it was true for.
    """
    dates, _, y = forward_panel()
    rng = np.random.default_rng(11)
    pred = y + rng.normal(scale=0.05, size=y.size)
    result = evaluate_regression(
        y, pred, None, dates=dates, n_folds=5, min_train=200, embargo_days=365
    )
    assert result.fold_unit == "fold"
    assert result.paired is None
    assert "Fold standard deviation" in result.verdict()
    assert "the fold-to-fold noise" in result.verdict()
    assert dict(result.rows())["Folds"] == 5


def test_a_result_cached_before_fold_unit_existed_still_answers():
    """An old joblib cache entry must not crash the first verdict it is asked for.

    Fitted bundles are cached with joblib, which restores a dataclass from its
    saved attribute dict without calling ``__init__``. A result pickled before
    ``fold_unit`` and ``paired`` existed therefore arrives without either
    attribute, and without the ``__setstate__`` backfill the first ``verdict()``
    or ``rows()`` would raise AttributeError deep inside a command. The old
    entry keeps its old meaning: unit ``fold``, no paired statistic.
    """
    result = EvalResult(
        metric="mae",
        score=1.0,
        baseline_name="training-mean constant",
        baseline_score=2.0,
        n_observations=40,
        higher_is_better=False,
        folds=[1.1, 0.9, 1.0],
    )
    state = dict(result.__dict__)
    del state["fold_unit"]
    del state["paired"]

    revived = EvalResult.__new__(EvalResult)
    revived.__setstate__(state)
    assert revived.fold_unit == "fold"
    assert revived.paired is None
    assert "Fold standard deviation" in revived.verdict()
    assert dict(revived.rows())["Folds"] == 3


# ------------------------------------------------------------- classification


def takeout_panel(seed: int = 3, n: int = 600):
    """A takeout propensity sample: a low base rate and a score that half works."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.0, 0.30, size=n)
    y = rng.binomial(1, truth)
    scores = truth + rng.normal(scale=0.05, size=n)
    dates = [d for d in month_starts(n // 4) for _ in range(4)]
    return dates, y, scores, truth


def test_classification_scores_ordering_against_a_base_rate_baseline():
    dates, y, scores, _ = takeout_panel()
    result = evaluate_classification(y, scores, dates=dates, n_folds=4, embargo_days=365)

    assert result.metric == "auc"
    assert result.baseline_score == 0.5
    assert "base rate" in result.baseline_name
    assert result.beat_baseline
    assert result.score > 0.65
    assert len(result.folds) >= 3
    assert any("365-day embargo" in note for note in result.notes)


def test_classification_says_whether_the_scores_are_probabilities_at_all():
    dates, y, scores, truth = takeout_panel()

    as_probabilities = evaluate_classification(y, np.clip(scores, 0.0, 1.0))
    assert any("Brier score" in note for note in as_probabilities.notes)
    assert any("calibration_table" in note for note in as_probabilities.notes)

    as_z_scores = evaluate_classification(y, (scores - scores.mean()) / scores.std())
    assert any("not probabilities" in note for note in as_z_scores.notes)
    # No Brier score is quoted on numbers that are not probabilities.
    assert not any(note.startswith("Brier score") for note in as_z_scores.notes)

    # AUC is a ranking statistic, so standardising the scores cannot move it.
    assert as_z_scores.score == pytest.approx(
        evaluate_classification(y, scores).score, abs=1e-12
    )


def test_classification_refuses_a_sample_that_cannot_support_an_auc():
    _, y, scores, _ = takeout_panel()

    with pytest.raises(NotMeaningfulError, match="below the floor"):
        evaluate_classification(y[:20], scores[:20])

    # Five events in six hundred names is an anecdote, not a base rate.
    rare = np.zeros(400, dtype=int)
    rare[:5] = 1
    with pytest.raises(NotMeaningfulError, match="positives and"):
        evaluate_classification(rare, np.linspace(0, 1, 400))

    with pytest.raises(ConfigError, match="must be 0 or 1"):
        evaluate_classification(np.full(100, 2), np.linspace(0, 1, 100))
    with pytest.raises(ConfigError, match="baseline_rate must lie"):
        evaluate_classification(y, scores, baseline_rate=1.4)


# -------------------------------------------------------------- calibration


def test_calibration_table_passes_a_model_that_is_actually_calibrated():
    """When a well calibrated model says 70%, the realised frequency is near 70%."""
    rng = np.random.default_rng(21)
    p = rng.uniform(0.02, 0.98, size=4000)
    y = rng.binomial(1, p)

    table = calibration_table(y, p, bins=10)
    assert list(table.columns) == [
        "low",
        "high",
        "n",
        "predicted",
        "realised",
        "gap",
        "thin",
    ]
    assert len(table) == 10
    assert int(table["n"].sum()) == 4000

    populated = table[table["n"] >= 100]
    assert len(populated) >= 8
    assert populated["gap"].abs().max() < 0.08
    assert not populated["thin"].any()


def test_calibration_table_catches_a_model_that_ranks_well_and_lies_about_odds():
    """Same ordering, same AUC, and a stated probability nobody should quote.

    This is the model fitted on a balanced sample and then applied to a
    population where three percent of names get taken out. Multiplying every
    probability by five leaves the ranking untouched, so discrimination is
    identical and only the calibration table shows the damage.
    """
    rng = np.random.default_rng(22)
    truth = rng.uniform(0.0, 0.12, size=4000)
    y = rng.binomial(1, truth)
    overconfident = np.clip(truth * 5.0, 0.0, 0.95)

    honest_auc = evaluate_classification(y, truth).score
    inflated_auc = evaluate_classification(y, overconfident).score
    assert inflated_auc == pytest.approx(honest_auc, abs=1e-9)

    good = calibration_table(y, truth, bins=10)
    bad = calibration_table(y, overconfident, bins=10)

    good_gap = good.loc[good["n"] >= 100, "gap"].abs().max()
    bad_gap = bad.loc[bad["n"] >= 100, "gap"].min()
    assert good_gap < 0.05
    # Negative gap is overconfidence: promised more than the world delivered.
    assert bad_gap < -0.2
    assert bad.loc[bad["n"] >= 100, "realised"].max() < 0.2


def test_calibration_table_keeps_empty_buckets_and_flags_thin_ones():
    """A model that never speaks above 0.1 is telling you something about itself."""
    rng = np.random.default_rng(23)
    p = np.concatenate([rng.uniform(0.0, 0.1, size=195), np.full(5, 0.95)])
    y = rng.binomial(1, p)

    table = calibration_table(y, p, bins=10)
    silent = table[table["n"] == 0]
    assert len(silent) == 8
    assert silent["predicted"].isna().all()
    assert silent["realised"].isna().all()
    assert not silent["thin"].any()

    top = table.iloc[-1]
    assert top["n"] == 5
    assert bool(top["thin"])


def test_calibration_table_refuses_anything_that_is_not_a_probability():
    rng = np.random.default_rng(24)
    y = rng.binomial(1, 0.3, size=200)

    with pytest.raises(ConfigError, match="defined for probabilities"):
        calibration_table(y, rng.normal(size=200))
    with pytest.raises(ConfigError, match="bins must be at least 2"):
        calibration_table(y, rng.uniform(size=200), bins=1)
    with pytest.raises(NotMeaningfulError, match="below the floor"):
        calibration_table(y[:10], rng.uniform(size=10))


# --------------------------------------------------------------------ablation


def ols(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame):
    """Least squares with an intercept, refitted inside every fold."""
    A = np.column_stack([np.ones(len(X_train)), X_train.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(A, np.asarray(y_train, dtype=float), rcond=None)
    B = np.column_stack([np.ones(len(X_test)), X_test.to_numpy(dtype=float)])
    return B @ beta


def feature_panel(seed: int = 31, n_months: int = 60, tickers: int = 4):
    """A panel where the truth is known: size carries the model and noise does not."""
    rng = np.random.default_rng(seed)
    timeline = month_starts(n_months)
    dates = [d for d in timeline for _ in range(tickers)]
    n = len(dates)
    X = pd.DataFrame(
        {
            "log_size": rng.normal(size=n),
            "ntm_growth": rng.normal(size=n),
            "noise_a": rng.normal(size=n),
            "noise_b": rng.normal(size=n),
        }
    )
    y = 3.0 * X["log_size"] + 1.0 * X["ntm_growth"] + rng.normal(scale=0.5, size=n)
    return dates, X, y.to_numpy()


def test_ablation_ranks_the_features_that_are_doing_the_work():
    dates, X, y = feature_panel()
    groups = {
        "size": ["log_size"],
        "growth": ["ntm_growth"],
        "noise": ["noise_a", "noise_b"],
    }
    table = ablation(ols, groups, X, y, dates, n_folds=4, min_train=80, embargo_days=365)

    assert list(table.columns) == [
        "group",
        "n_columns",
        "score",
        "damage",
        "fold_sd",
        "n_test",
    ]
    assert table.iloc[0]["group"] == "all features"
    assert table.iloc[0]["damage"] == 0.0
    assert table.iloc[0]["n_columns"] == 4

    damage = dict(zip(table["group"], table["damage"]))
    assert damage["without size"] > damage["without growth"] > damage["without noise"]
    assert damage["without size"] > 1.0
    assert damage["without growth"] > 0.2
    # Two pure noise columns are not load bearing, whatever the card implies.
    assert abs(damage["without noise"]) < 0.1

    ordered = table.iloc[1:]["damage"].tolist()
    assert ordered == sorted(ordered, reverse=True)
    assert (table["n_test"] == table.iloc[0]["n_test"]).all()


def test_ablation_is_scored_on_embargoed_folds_like_everything_else():
    """The drop has to be measured out of sample or it measures fit, not value."""
    dates, X, y = feature_panel()
    folds = walk_forward_folds(dates, 4, min_train=80, embargo_days=365)
    table = ablation(ols, {"size": ["log_size"], "rest": ["ntm_growth", "noise_a", "noise_b"]},
                     X, y, dates, folds=folds)

    scored = sum(
        1
        for d in dates
        if any(f.test_start <= d <= f.test_end for f in folds)
    )
    assert int(table.iloc[0]["n_test"]) == scored
    assert table.iloc[0]["fold_sd"] > 0


def test_ablation_refuses_a_group_it_cannot_drop():
    dates, X, y = feature_panel()

    with pytest.raises(ConfigError, match="columns absent from X"):
        ablation(ols, {"ghost": ["not_a_column"]}, X, y, dates, n_folds=4, min_train=80)
    with pytest.raises(ConfigError, match="no model"):
        # Dropping the only group leaves nothing to refit.
        ablation(ols, {"everything": list(X.columns)}, X, y, dates, n_folds=4, min_train=80)
    with pytest.raises(ConfigError, match="rows against"):
        ablation(ols, ["log_size", "ntm_growth"], X, y[:-1], dates[:-1], n_folds=4, min_train=80)


def test_ablation_accepts_a_plain_list_as_one_group_per_column():
    dates, X, y = feature_panel()
    table = ablation(
        ols, ["log_size", "ntm_growth", "noise_a"], X, y, dates, n_folds=4, min_train=80
    )
    assert set(table["group"]) == {
        "all features",
        "without log_size",
        "without ntm_growth",
        "without noise_a",
    }
    assert table.iloc[1]["group"] == "without log_size"
