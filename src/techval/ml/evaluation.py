"""Scoring a model honestly, which is mostly a question of what it was not shown.

Split by date, never at random. A random split lets a model train on 2026 and
test on 2024, which is the cleanest way to manufacture a result nobody can
repeat. Every fold in this module trains on everything up to a cut and tests on
the window after it, and the cuts march forward through the sample.

**The embargo is the detail that separates a real backtest from a plausible
one.** Where the label is a forward return or a forward growth rate, the label
of an observation dated t is not known until t plus the horizon. Train on an
observation dated one month before the test window opens and its label window
overlaps that test window by eleven months out of twelve, so the model is
shown most of the answer before it is asked the question. The fit looks
excellent and the failure is silent. Every fold here therefore takes an embargo
in days and drops from training everything dated later than the test start less
the embargo. For a one-year forward return the embargo is 365. For a label that
is known on the observation date, such as whether a company had already been
acquired, it is zero, and passing zero is recorded in the notes so a reader can
see that the protection was declined rather than forgotten.

**Nothing is scored without a baseline.** A mean absolute error of 8 points on
revenue growth means nothing until you know that carrying last year's growth
forward gives 7. Where the caller supplies no baseline this module computes the
obvious one and names it in ``baseline_name``: the training-period mean for a
regression, the base rate for a classifier, the expected value of a random
ordering for a ranking. Where a computed baseline has to peek at the answer key
to exist, it is named as in-sample, which makes it a harder baseline to beat and
therefore the conservative choice.

**Ranking is the flagship use.** The peer model proposes a comparable company
set, and it is scored on the disclosed peer group with NDCG and precision at k.
A target picks from a universe of roughly 100 candidates of which perhaps 20 are
the disclosed comparables, so precision@10 under a random ordering sits near
0.20 and recall@10 near 0.10. Clearing 0.20 is the bar. A precision@10 of 0.35
is a result worth showing a banker, and a precision@10 of 0.20 is a coin toss in
a suit.

**Ranking well and being calibrated are different claims.** A model can order
companies by takeout likelihood beautifully and still say 70% where the realised
frequency is 20%. Ordering is what a screen needs; a stated probability is what
a memo needs, and only ``calibration_table`` can tell you whether the second one
is earned. A model that ranks well and calibrates badly may be used to sort and
must not be used to state a number.

**Small samples are the normal case here, not the exception.** A few hundred
companies over a handful of years is what the filing record offers, so every
function refuses below the point where the statistic is noise and flags the
range where it is merely thin. Fold to fold dispersion is reported on every
result, because a lift of 0.02 with a fold standard deviation of 0.09 is not a
lift, and ``EvalResult.verdict`` says so in words.

Rendering belongs in the command line layer. Everything here returns an
``EvalResult`` or a frame.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..errors import ConfigError, NotMeaningfulError
from .protocol import EvalResult, PairedDelta, spearman

# Mirrors assumptions.ml.walk_forward_folds. A caller holding an Assumptions
# object should pass its value; either way the number actually used is written
# into the result notes so a reader never has to guess it.
DEFAULT_FOLDS = 5

# Refusal thresholds. These are statistical floors rather than valuation
# assumptions: below them the statistic is not a weak answer, it is noise with a
# decimal point. Thirty observations is already a small sample for anything
# fitted, and it is the point below which a mean absolute error moves more with
# one company than with the model.
MIN_OBSERVATIONS = 30
MIN_TRAIN_OBSERVATIONS = 30
MIN_QUERIES = 5
MIN_CLASS_EVENTS = 10

# Flag thresholds. Above the refusal floor and below these the number is
# reportable but thin, and the note says so rather than the reader assuming
# otherwise.
THIN_OBSERVATIONS = 100
THIN_FOLD_OBSERVATIONS = 10
THIN_QUERIES = 20
THIN_CLASS_EVENTS = 30
THIN_BIN_OBSERVATIONS = 10

_REGRESSION_METRICS = {"mae": False, "rmse": False, "spearman": True}

# Metrics that score an ordering rather than a level. They need a different
# naive alternative: a constant beats every error metric by sitting at the
# centre of the distribution, and has no ordering at all to be correlated with.
_RANK_METRICS = {"spearman"}
_NO_ORDERING_BASELINE = (
    "no ordering (rank correlation of zero, which is what an uninformative ranking earns)"
)


@dataclass
class Fold:
    """One walk-forward split: train on the past, test on the window after it.

    ``train_end`` is the date of the last observation actually used for
    training, which under an embargo is strictly earlier than
    ``test_start`` less ``embargo_days``. ``n_embargoed`` counts the
    observations that fell into the gap and were used for neither, and it is
    the number that makes the cost of the embargo visible: an embargo that
    throws away a third of the sample is a decision, not a formality.
    """

    index: int
    train_end: date
    test_start: date
    test_end: date
    n_train: int
    n_test: int
    embargo_days: int = 0
    n_embargoed: int = 0

    def rows(self) -> list[tuple[str, Any]]:
        return [
            ("Fold", self.index),
            ("Train through", str(self.train_end)),
            ("Test from", str(self.test_start)),
            ("Test to", str(self.test_end)),
            ("Train observations", self.n_train),
            ("Test observations", self.n_test),
            ("Embargo days", self.embargo_days),
            ("Embargoed observations", self.n_embargoed),
        ]


def walk_forward_folds(
    dates: Sequence[Any],
    n_folds: int,
    min_train: int | None = None,
    *,
    embargo_days: int = 0,
) -> list[Fold]:
    """Cut the sample into expanding walk-forward folds, split by date.

    The timeline is the sorted distinct observation dates. The earliest stretch
    of it is reserved as the initial training window, large enough to hold
    ``min_train`` observations after the embargo has taken its share, and
    everything after that is divided into ``n_folds`` contiguous test blocks of
    roughly equal numbers of dates. Blocks are cut on dates rather than on
    calendar spans because filings arrive in clusters, and equal calendar
    windows would hand one fold a reporting season and another an empty August.

    Training is strictly before ``test_start`` less ``embargo_days``. The cutoff
    is strict, so an observation whose one-year label window closes exactly on
    the first test date is excluded. Where the embargo matters at all, being one
    day conservative costs nothing.

    Each fold trains on everything before its cut, so fold 0 is fitted on the
    least history and fold 4 on the most. That is the shape of the real problem:
    a model run today has all of the past, and a model run in 2019 had 2019.

    Raises ConfigError where no split satisfies the request, naming what was
    asked for and what the sample holds, and NotMeaningfulError below
    MIN_OBSERVATIONS.
    """
    d = _as_dates(dates)
    if n_folds < 2:
        raise ConfigError(
            f"walk-forward evaluation needs at least 2 folds, got {n_folds}: a single "
            "fold is one arbitrary split with no dispersion to report"
        )
    if embargo_days < 0:
        raise ConfigError(f"embargo_days must be zero or positive, got {embargo_days}")
    if d.size < MIN_OBSERVATIONS:
        raise NotMeaningfulError(
            f"{d.size} observations is below the floor of {MIN_OBSERVATIONS} for a "
            "walk-forward evaluation: the fold scores would move more with one "
            "company than with the model"
        )

    floor = MIN_TRAIN_OBSERVATIONS if min_train is None else int(min_train)
    if floor < 1:
        raise ConfigError(f"min_train must be at least 1, got {min_train}")

    timeline = sorted(set(d.tolist()))
    n_dates = len(timeline)
    if n_dates < n_folds + 1:
        raise ConfigError(
            f"{n_folds} walk-forward folds need at least {n_folds + 1} distinct "
            f"observation dates, one for training and one per test block, but the "
            f"sample has {n_dates} ({timeline[0]} to {timeline[-1]})"
        )

    first_test = None
    for candidate in range(1, n_dates - n_folds + 1):
        cutoff = timeline[candidate] - timedelta(days=embargo_days)
        if int(np.sum(d < cutoff)) >= floor:
            first_test = candidate
            break
    if first_test is None:
        raise ConfigError(
            f"no walk-forward split leaves {floor} training observations before the "
            f"first of {n_folds} test blocks under a {embargo_days}-day embargo: "
            f"{d.size} observations across {n_dates} dates from {timeline[0]} to "
            f"{timeline[-1]}. Shorten the embargo, lower min_train, or accept that "
            "the sample does not support this evaluation"
        )

    folds: list[Fold] = []
    for i, block in enumerate(np.array_split(np.arange(first_test, n_dates), n_folds)):
        test_start = timeline[int(block[0])]
        test_end = timeline[int(block[-1])]
        cutoff = test_start - timedelta(days=embargo_days)
        train_mask = d < cutoff
        n_train = int(train_mask.sum())
        if n_train == 0:
            raise ConfigError(
                f"a {embargo_days}-day embargo leaves fold {i} (test from {test_start}) "
                "with no training observations at all"
            )
        folds.append(
            Fold(
                index=i,
                train_end=max(d[train_mask].tolist()),
                test_start=test_start,
                test_end=test_end,
                n_train=n_train,
                n_test=int(((d >= test_start) & (d <= test_end)).sum()),
                embargo_days=embargo_days,
                n_embargoed=int(((d >= cutoff) & (d < test_start)).sum()),
            )
        )
    return folds


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    baseline_pred: Sequence[float] | None,
    dates: Sequence[Any] | None = None,
    folds: Sequence[Fold] | None = None,
    metric: str = "mae",
    *,
    n_folds: int = DEFAULT_FOLDS,
    embargo_days: int = 0,
    min_train: int | None = None,
) -> EvalResult:
    """Score out-of-sample predictions of a continuous quantity against a baseline.

    Metrics are ``mae`` (the default, and the one to quote for a growth rate
    because it is in the units of the thing forecast), ``rmse`` where large
    misses should be punished more than proportionally, and ``spearman`` where
    only the ordering is traded. Mean absolute percentage error is deliberately
    not offered: a growth rate near zero sends it to infinity, and the companies
    whose growth is near zero are exactly the ones a TMT model most often has to
    call.

    The headline score is pooled over every test observation, so a fold holding
    thirty names does not weigh the same as a fold holding three. The per-fold
    scores go into ``EvalResult.folds``, which is what makes ``fold_sd`` an
    error bar rather than a decoration, and the notes record how many folds the
    model actually won. Winning on the pooled number while losing three folds
    out of five is a warning, not a result.

    ``baseline_pred`` of None is not silently accepted. With folds available the
    baseline becomes the mean of each fold's own training labels, which is the
    honest naive forecast: the average of what had already happened. Without
    folds it becomes the sample mean, which peeks at the test period and so
    flatters the baseline, and the name says as much. Under ``spearman`` that
    constant is no baseline at all, since it has no ordering to correlate with,
    so the comparison becomes a rank correlation of zero: the score an
    uninformative ranking earns. Pass last year's value as ``baseline_pred``
    wherever one exists, because it is a much harder number to beat than zero.

    Observations before the first test block are never scored. They are the
    initial training window, and the count that fell there is in the notes.
    """
    y = _as_floats(y_true, "y_true")
    p = _as_floats(y_pred, "y_pred")
    if y.size != p.size:
        raise ConfigError(
            f"y_true has {y.size} observations and y_pred has {p.size}: predictions and "
            "outcomes must be aligned row for row"
        )
    if metric not in _REGRESSION_METRICS:
        raise ConfigError(
            f"unknown regression metric {metric!r}: one of {sorted(_REGRESSION_METRICS)}"
        )
    if y.size < MIN_OBSERVATIONS:
        raise NotMeaningfulError(
            f"{y.size} observations is below the floor of {MIN_OBSERVATIONS} for a "
            f"{metric} worth reporting"
        )

    base = None if baseline_pred is None else _as_floats(baseline_pred, "baseline_pred")
    if base is not None and base.size != y.size:
        raise ConfigError(
            f"baseline_pred has {base.size} observations against {y.size} outcomes"
        )

    notes: list[str] = []
    higher = _REGRESSION_METRICS[metric]

    if folds is not None and dates is None:
        raise ConfigError(
            "folds were supplied without dates, so there is no way to tell which "
            "observation belongs to which fold"
        )

    if dates is None:
        # No dates means no fold structure, and no fold structure means this
        # function cannot verify the thing it exists to verify.
        notes.append(
            "No dates supplied, so this is one pooled score with no fold structure: "
            "nothing here demonstrates that the predictions were made out of sample."
        )
        score = _regression_score(metric, y, p)
        fold_scores: list[float] = []
        n_scored = y.size
        if base is None and metric in _RANK_METRICS:
            baseline_name = _NO_ORDERING_BASELINE
            baseline_score = 0.0
        elif base is None:
            baseline_name = "sample-mean constant (in sample, which favours the baseline)"
            baseline_score = _regression_score(
                metric, y, np.full(y.size, float(np.mean(y)))
            )
        else:
            baseline_name = "supplied baseline"
            baseline_score = _regression_score(metric, y, base)
    else:
        d = _as_dates(dates)
        if d.size != y.size:
            raise ConfigError(
                f"dates has {d.size} entries against {y.size} outcomes"
            )
        if folds is None:
            folds = walk_forward_folds(
                d, n_folds, min_train=min_train, embargo_days=embargo_days
            )
        slices = _fold_slices(d, folds)

        # A constant carries no ordering, so under a rank metric the naive
        # alternative is not the training mean but the absence of information:
        # zero rank correlation, which is what an uninformative ranking earns.
        unranked = base is None and metric in _RANK_METRICS
        baseline_name = "supplied baseline"
        if unranked:
            baseline_name = _NO_ORDERING_BASELINE
        elif base is None:
            base = np.full(y.size, np.nan)
            for fold, _ in slices:
                train = _train_index(d, fold)
                if train.size == 0:
                    raise ConfigError(
                        f"fold {fold.index} has no training observations, so no "
                        "training-mean baseline exists for it"
                    )
                test = _test_index(d, fold)
                base[test] = float(np.mean(y[train]))
            baseline_name = "training-mean constant"

        def baseline_at(idx: np.ndarray) -> float:
            return 0.0 if unranked else _regression_score(metric, y[idx], base[idx])

        test_idx = np.concatenate([idx for _, idx in slices])
        n_scored = int(test_idx.size)
        score = _regression_score(metric, y[test_idx], p[test_idx])
        baseline_score = baseline_at(test_idx)

        fold_scores = []
        fold_lifts = []
        wins = 0
        thin = []
        for fold, idx in slices:
            fold_score = _regression_score(metric, y[idx], p[idx])
            fold_base = baseline_at(idx)
            fold_scores.append(fold_score)
            lift = (fold_score - fold_base) if higher else (fold_base - fold_score)
            fold_lifts.append(lift)
            if lift > 0:
                wins += 1
            if idx.size < THIN_FOLD_OBSERVATIONS:
                thin.append(f"{fold.index} ({idx.size})")
        notes.extend(
            _fold_notes(folds, wins, fold_lifts, thin, int(y.size - n_scored), y.size)
        )

    if n_scored < THIN_OBSERVATIONS:
        notes.append(
            f"{n_scored:,} scored observations is a thin sample: read the lift beside "
            "the fold standard deviation, not on its own."
        )

    return EvalResult(
        metric=metric,
        score=score,
        baseline_name=baseline_name,
        baseline_score=baseline_score,
        n_observations=n_scored,
        higher_is_better=higher,
        folds=fold_scores,
        notes=notes,
    )


def evaluate_ranking(
    relevance_by_query: Mapping[Any, Mapping[Any, float]],
    ranked_by_query: Mapping[Any, Sequence[Any]],
    k: int = 10,
    baseline: Mapping[Any, Sequence[Any]] | None = None,
    *,
    baseline_name: str = "supplied baseline order",
) -> EvalResult:
    """Score a proposed ordering against a known relevant set, query by query.

    The query is a target company, the ranking is the candidate universe in the
    model's preferred order, and the relevance map grades each candidate:
    graded rather than binary because a peer set is not flat. Scoring the
    company's own disclosed comparables at 2 and the ones a banker would add at
    1 says more than counting hits, and NDCG is built for exactly that.

    ``EvalResult.folds`` carries the per-query NDCG rather than per-period fold
    scores, and the result says so: ``fold_unit`` is ``"query"``, so ``fold_sd``
    is the dispersion across targets and ``verdict`` reports it as that rather
    than as fold-to-fold noise. Across-target spread is worth printing because
    one target with an obvious peer set can carry a mean across five, but it is
    not an error bar on the lift, since targets differ mostly in ways the model
    and the baseline share. The error bar on the lift is the per-query paired
    difference between the two, computed here from the same aligned scores and
    carried in ``EvalResult.paired``.

    With no baseline supplied the comparison is the expected NDCG of a uniform
    random ordering of the same candidate list, computed in closed form rather
    than simulated so the number is exact and needs no seed. Under a random
    permutation every candidate is equally likely at every position, so the
    expected discounted gain at each position is the mean relevance of the
    universe, and the expectation of the whole discounted sum follows. Pass an
    explicit ordering as ``baseline`` where a harder comparison exists, such as
    ranking by size or by the popularity of each candidate across all targets.

    Queries with no relevant candidate in the list have no defined NDCG and are
    counted in the notes rather than dropped in silence.
    """
    if k < 1:
        raise ConfigError(f"k must be at least 1, got {k}")

    queries = [q for q in relevance_by_query if q in ranked_by_query]
    missing = [q for q in relevance_by_query if q not in ranked_by_query]
    notes: list[str] = []
    if missing:
        notes.append(
            f"{len(missing)} of {len(relevance_by_query)} queries have a relevant set "
            f"but no ranking and were not scored: {_sample_names(missing)}."
        )

    scores: list[float] = []
    base_scores: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    random_precisions: list[float] = []
    undefined: list[Any] = []
    scored_queries: list[Any] = []

    for q in queries:
        relevance = relevance_by_query[q]
        ranking = list(ranked_by_query[q])
        if any(float(v) < 0 for v in relevance.values()):
            raise ConfigError(
                f"query {q!r} has a negative relevance grade: grades are gains, so the "
                "least relevant candidate scores zero"
            )
        if not ranking:
            undefined.append(q)
            continue
        if not any(float(relevance.get(item, 0.0)) > 0 for item in ranking):
            undefined.append(q)
            continue

        scores.append(ndcg_at_k(relevance, ranking, k))
        relevant = {item for item, grade in relevance.items() if float(grade) > 0}
        precision, recall = precision_recall_at_k(relevant, ranking, k)
        precisions.append(precision)
        recalls.append(recall)
        random_precisions.append(
            len(relevant & set(ranking)) / len(ranking) if ranking else 0.0
        )
        scored_queries.append(q)

        if baseline is None:
            base_scores.append(_expected_random_ndcg(relevance, ranking, k))
        else:
            if q not in baseline:
                raise ConfigError(
                    f"the baseline has no ordering for query {q!r}, so the model and "
                    "the baseline would be scored on different queries"
                )
            base_ranking = list(baseline[q])
            if set(base_ranking) != set(ranking):
                raise ConfigError(
                    f"the baseline ordering for query {q!r} covers a different candidate "
                    "set from the model's, so the two scores are not comparable"
                )
            base_scores.append(ndcg_at_k(relevance, base_ranking, k))

    if undefined:
        notes.append(
            f"{len(undefined)} queries had no relevant candidate in the ranked list, "
            f"where NDCG is undefined, and were excluded: {_sample_names(undefined)}."
        )
    if len(scores) < MIN_QUERIES:
        raise NotMeaningfulError(
            f"{len(scores)} scorable queries is below the floor of {MIN_QUERIES}: a mean "
            "NDCG over a handful of targets is an anecdote"
        )
    if len(scores) < THIN_QUERIES:
        notes.append(
            f"{len(scores)} queries is thin. Read the mean beside the across-query "
            "standard deviation, which is what fold_sd reports here."
        )

    name = (
        "random order (expected value under a uniform permutation)"
        if baseline is None
        else baseline_name
    )
    universe = float(np.mean([len(ranked_by_query[q]) for q in scored_queries]))
    notes.append(
        f"Per-query values in folds are NDCG by target, not by period: dispersion is "
        f"across {len(scores)} targets."
    )
    notes.append(
        f"precision@{k} {float(np.mean(precisions)):.3f} against a random expectation of "
        f"{float(np.mean(random_precisions)):.3f}, recall@{k} "
        f"{float(np.mean(recalls)):.3f}, over a mean candidate universe of "
        f"{universe:.0f} names."
    )
    notes.append(
        f"With around 20 disclosed peers in a universe of 100, precision@{k} under a "
        "random ordering sits near 0.20. Clearing it is the bar, and not clearing it "
        "means the ordering carries no information about comparability."
    )

    # The model and the baseline were scored on identical queries, so the
    # per-query difference is paired by construction and its own dispersion is
    # the error bar on the lift. The across-target sd in fold_sd is not: targets
    # differ mostly in ways both methods share.
    paired: PairedDelta | None = None
    differences = np.asarray(scores, dtype=float) - np.asarray(base_scores, dtype=float)
    if differences.size >= 2:
        sd_diff = float(np.std(differences, ddof=1))
        standard_error = sd_diff / float(np.sqrt(differences.size))
        paired = PairedDelta(
            mean=float(differences.mean()),
            sd=sd_diff,
            standard_error=standard_error,
            t=float(differences.mean() / standard_error)
            if standard_error
            else float("inf"),
            win_rate=float((differences > 0).mean()),
            n=int(differences.size),
        )

    return EvalResult(
        metric=f"ndcg@{k}",
        score=float(np.mean(scores)),
        baseline_name=name,
        baseline_score=float(np.mean(base_scores)),
        n_observations=len(scores),
        higher_is_better=True,
        folds=scores,
        notes=notes,
        fold_unit="query",
        paired=paired,
    )


def evaluate_classification(
    y_true: Sequence[Any],
    scores: Sequence[float],
    baseline_rate: float | None = None,
    dates: Sequence[Any] | None = None,
    *,
    folds: Sequence[Fold] | None = None,
    n_folds: int = DEFAULT_FOLDS,
    embargo_days: int = 0,
    min_train: int | None = None,
) -> EvalResult:
    """Score a binary classifier on discrimination, and say whether it is calibrated.

    The headline is the area under the ROC curve, computed from ranks so that
    ``scores`` need not be probabilities: for a takeout propensity model the
    question a banker asks first is whether the names at the top of the list get
    bought more often than the names at the bottom, which is an ordering
    question. The baseline is the constant prediction at the base rate, whose
    AUC is 0.5 by construction and not by assumption, so any model below 0.5 is
    ordering the wrong way round.

    Where the scores do lie in [0, 1] the Brier score of the model and of the
    constant base rate go into the notes, because discrimination and calibration
    fail independently. A model with an AUC of 0.72 and a Brier score worse than
    the base rate is a screen, not a probability, and the notes say so. Run
    ``calibration_table`` before quoting any single number as a likelihood.

    Class imbalance is the norm for this task. Roughly 3% of listed software
    companies are acquired in a given year, so a 500-name sample holds perhaps
    15 events, and the fold-level AUC of a window containing two of them is
    noise. Folds with no event in the test window are excluded from the fold
    scores and counted in the notes.
    """
    y = _as_binary(y_true)
    s = _as_floats(scores, "scores")
    if y.size != s.size:
        raise ConfigError(
            f"y_true has {y.size} observations and scores has {s.size}"
        )
    if y.size < MIN_OBSERVATIONS:
        raise NotMeaningfulError(
            f"{y.size} observations is below the floor of {MIN_OBSERVATIONS} for a "
            "classifier worth reporting"
        )
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos < MIN_CLASS_EVENTS or n_neg < MIN_CLASS_EVENTS:
        raise NotMeaningfulError(
            f"{n_pos} positives and {n_neg} negatives: below {MIN_CLASS_EVENTS} of "
            "either class an AUC moves with one company and means nothing"
        )

    notes: list[str] = []
    rate = float(np.mean(y)) if baseline_rate is None else float(baseline_rate)
    if baseline_rate is None:
        notes.append(
            f"Base rate of {rate:.2%} computed in sample over {y.size:,} observations, "
            "which favours the baseline. Pass the known prior where there is one."
        )
    elif not 0.0 < rate < 1.0:
        raise ConfigError(f"baseline_rate must lie strictly between 0 and 1, got {rate}")

    fold_scores: list[float] = []
    if dates is None:
        if folds is not None:
            raise ConfigError(
                "folds were supplied without dates, so there is no way to tell which "
                "observation belongs to which fold"
            )
        notes.append(
            "No dates supplied, so this is one pooled AUC with no fold structure: "
            "nothing here demonstrates that the scores were produced out of sample."
        )
        n_scored = int(y.size)
        auc = _auc(y, s)
    else:
        d = _as_dates(dates)
        if d.size != y.size:
            raise ConfigError(f"dates has {d.size} entries against {y.size} outcomes")
        if folds is None:
            folds = walk_forward_folds(
                d, n_folds, min_train=min_train, embargo_days=embargo_days
            )
        slices = _fold_slices(d, folds)
        test_idx = np.concatenate([idx for _, idx in slices])
        n_scored = int(test_idx.size)
        auc = _auc(y[test_idx], s[test_idx])

        wins = 0
        lifts = []
        thin: list[str] = []
        eventless = 0
        for fold, idx in slices:
            if y[idx].sum() == 0 or y[idx].sum() == idx.size:
                eventless += 1
                continue
            fold_auc = _auc(y[idx], s[idx])
            fold_scores.append(fold_auc)
            lifts.append(fold_auc - 0.5)
            if fold_auc > 0.5:
                wins += 1
            if int(y[idx].sum()) < THIN_FOLD_OBSERVATIONS:
                thin.append(f"{fold.index} ({int(y[idx].sum())} events)")
        notes.extend(
            _fold_notes(folds, wins, lifts, thin, int(y.size - n_scored), y.size)
        )
        if eventless:
            notes.append(
                f"{eventless} of {len(folds)} folds had no event in the test window, "
                "where AUC is undefined, and are excluded from the fold scores."
            )

    if n_pos < THIN_CLASS_EVENTS:
        notes.append(
            f"{n_pos} events in total. Under {THIN_CLASS_EVENTS} the AUC carries a wide "
            "interval that this harness does not attempt to estimate."
        )

    if float(s.min()) >= 0.0 and float(s.max()) <= 1.0:
        brier = float(np.mean((s - y) ** 2))
        base_brier = float(np.mean((rate - y) ** 2))
        verdict = "better than" if brier < base_brier else "no better than"
        notes.append(
            f"Brier score {brier:.4f} against {base_brier:.4f} for the constant base "
            f"rate, so the stated probabilities are {verdict} the base rate. Ordering "
            "and calibration fail independently: run calibration_table before quoting "
            "any of these numbers as a likelihood."
        )
    else:
        notes.append(
            f"Scores range {float(s.min()):.3f} to {float(s.max()):.3f}, so they are not "
            "probabilities. AUC is a ranking statistic and survives that; no Brier "
            "score or calibration table is meaningful without a probability."
        )

    return EvalResult(
        metric="auc",
        score=auc,
        baseline_name=f"base rate {rate:.2%} (constant prediction, AUC 0.5 by construction)",
        baseline_score=0.5,
        n_observations=n_scored,
        higher_is_better=True,
        folds=fold_scores,
        notes=notes,
    )


def ndcg_at_k(relevance: Mapping[Any, float], ranking: Sequence[Any], k: int) -> float:
    """Normalised discounted cumulative gain at k, with graded relevance.

    Each candidate contributes its relevance grade divided by the base-2
    logarithm of one plus its position, so the first position is undiscounted,
    the second is worth 1/log2(3) of its grade, and so on. The sum is then
    divided by the same sum over the ideal ordering, which is the grades sorted
    from best to worst, so the result is 1.0 for a perfect ordering and
    comparable across queries whose relevant sets differ in size.

    Worked example, which the tests check against this docstring. Grades
    A=3, B=2, C=0, D=1, and the model returns [B, A, D, C] at k=4:

        DCG  = 2/log2(2) + 3/log2(3) + 1/log2(4) + 0/log2(5)
             = 2.0000 + 1.8928 + 0.5000 + 0
             = 4.3928
        IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5)
             = 3.0000 + 1.2619 + 0.5000 + 0
             = 4.7619
        NDCG@4 = 4.3928 / 4.7619 = 0.9225

    Putting the second best name first rather than the best costs eight points,
    which is the property that matters: a peer set is judged on whether the
    right names are near the top, not on whether the order within them is exact.

    The ideal ordering is taken over the candidates actually ranked. A relevant
    company that never entered the universe is a recall failure of whatever
    built the universe, not a ranking failure, and hiding it inside NDCG would
    put the blame in the wrong place. Use ``precision_recall_at_k`` against the
    full relevant set to see it.

    Raises NotMeaningfulError where no ranked candidate is relevant, because the
    ideal gain is then zero and the ratio is undefined.
    """
    if k < 1:
        raise ConfigError(f"k must be at least 1, got {k}")
    items = list(ranking)
    grades = [float(relevance.get(item, 0.0)) for item in items]
    if any(g < 0 for g in grades):
        raise ConfigError("relevance grades are gains and cannot be negative")

    dcg = sum(g / np.log2(i + 2.0) for i, g in enumerate(grades[:k]))
    ideal = sorted(grades, reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2.0) for i, g in enumerate(ideal))
    if idcg <= 0:
        raise NotMeaningfulError(
            f"no relevant candidate appears in a ranking of {len(items)} names, so the "
            "ideal gain is zero and NDCG is undefined"
        )
    return float(dcg / idcg)


def precision_recall_at_k(
    relevant_set: set[Any] | Sequence[Any], ranking: Sequence[Any], k: int
) -> tuple[float, float]:
    """Hit rate in the top k, and share of the relevant set the top k recovered.

    Precision is divided by the number of names actually shown, which is k
    unless the candidate universe is smaller than k. A universe of six is not a
    ranking failure and should not be scored as though four slots were wasted.

    Recall is divided by the whole relevant set including names that never
    entered the universe, which is where a screen that filters too hard shows
    up. Precision can be 1.0 while recall is 0.15, and that pair describes a
    model that is right about the few names it puts forward and is missing most
    of the comparables.

    Raises NotMeaningfulError on an empty relevant set, where recall is 0/0.
    """
    if k < 1:
        raise ConfigError(f"k must be at least 1, got {k}")
    relevant = set(relevant_set)
    if not relevant:
        raise NotMeaningfulError(
            "the relevant set is empty, so recall is undefined and precision is "
            "trivially zero"
        )
    items = list(ranking)
    if not items:
        raise NotMeaningfulError("the ranking is empty, so precision is undefined")
    top = items[:k]
    hits = sum(1 for item in top if item in relevant)
    return float(hits / len(top)), float(hits / len(relevant))


def calibration_table(
    y_true: Sequence[Any], probabilities: Sequence[float], bins: int = 10
) -> pd.DataFrame:
    """Predicted probability against realised frequency, bucket by bucket.

    Buckets are equal width over [0, 1] rather than equal population, because
    the question a reader is asking is absolute: when this model says 70%, how
    often does it happen? Quantile buckets would answer a different question and
    would hide the fact that a model never says 70% at all. Empty buckets are
    therefore kept in the frame with a count of zero, since a model that only
    ever speaks between 0.02 and 0.08 is telling you something about itself.

    ``gap`` is realised less predicted, so a negative gap is overconfidence: the
    model promised more than the world delivered. Systematic negative gaps in
    the upper buckets are the signature of a model fitted on an imbalanced
    sample and then read as though its output were a probability. The ``thin``
    column marks buckets holding fewer than ten observations, where the realised
    frequency moves by ten points or more with one company and no conclusion
    should be drawn from the gap.

    Columns: low, high, n, predicted, realised, gap, thin.
    """
    y = _as_binary(y_true)
    p = _as_floats(probabilities, "probabilities")
    if y.size != p.size:
        raise ConfigError(
            f"y_true has {y.size} observations and probabilities has {p.size}"
        )
    if bins < 2:
        raise ConfigError(f"bins must be at least 2, got {bins}")
    if float(p.min()) < 0.0 or float(p.max()) > 1.0:
        raise ConfigError(
            f"calibration is defined for probabilities, but these values range "
            f"{float(p.min()):.3f} to {float(p.max()):.3f}. Map the scores to "
            "probabilities first, or report AUC alone"
        )
    if y.size < MIN_OBSERVATIONS:
        raise NotMeaningfulError(
            f"{y.size} observations is below the floor of {MIN_OBSERVATIONS} for a "
            "calibration table: every bucket would be a rounding error"
        )

    edges = np.linspace(0.0, 1.0, bins + 1)
    # searchsorted rather than digitize so that a probability of exactly 1.0
    # lands in the top bucket instead of falling off the end.
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = which == b
        n = int(mask.sum())
        rows.append(
            {
                "low": float(edges[b]),
                "high": float(edges[b + 1]),
                "n": n,
                "predicted": float(np.mean(p[mask])) if n else float("nan"),
                "realised": float(np.mean(y[mask])) if n else float("nan"),
                "gap": float(np.mean(y[mask]) - np.mean(p[mask])) if n else float("nan"),
                "thin": bool(0 < n < THIN_BIN_OBSERVATIONS),
            }
        )
    return pd.DataFrame(rows, columns=["low", "high", "n", "predicted", "realised", "gap", "thin"])


def ablation(
    fit_predict: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], Sequence[float]],
    features: Mapping[str, Sequence[str]] | Sequence[str],
    X: pd.DataFrame,
    y: Sequence[float],
    dates: Sequence[Any],
    *,
    folds: Sequence[Fold] | None = None,
    n_folds: int = DEFAULT_FOLDS,
    embargo_days: int = 0,
    min_train: int | None = None,
    metric: str = "mae",
) -> pd.DataFrame:
    """Refit without each feature group in turn, and report what its absence costs.

    A model card listing fourteen features invites the reader to assume all
    fourteen are earning their place. Usually two are, and one of those two is
    size. This refits the same model on the same walk-forward folds with one
    group of columns removed at a time and reports the damage: how much worse
    the out-of-sample score gets without that group, signed so that positive
    always means the group was doing work. A group with damage at or below zero
    is not helping, and a group whose damage is smaller than the fold standard
    deviation beside it has not been shown to help.

    ``features`` is a mapping of group name to column names, because features
    arrive in families and dropping one dummy from a set of eleven sub-vertical
    dummies measures nothing. A plain list of column names is accepted and
    treated as one group per column.

    ``fit_predict`` is called as ``fit_predict(X_train, y_train, X_test)`` and
    returns one prediction per test row. It is fitted separately inside every
    fold, which is the point: refitting is what makes the drop attributable to
    the feature rather than to a model that still carries the feature's
    coefficient from a full-sample fit. Any seeding inside it is the caller's
    responsibility and should come from assumptions.ml.random_seed.

    Returns a frame whose first row is the full model and whose remaining rows
    are the groups sorted by damage, most load-bearing first. Columns: group,
    n_columns, score, damage, fold_sd, n_test.
    """
    if not isinstance(X, pd.DataFrame):
        raise ConfigError("X must be a pandas DataFrame whose columns are the features")
    target = _as_floats(y, "y")
    if len(X) != target.size:
        raise ConfigError(f"X has {len(X)} rows against {target.size} outcomes")
    if metric not in _REGRESSION_METRICS:
        raise ConfigError(
            f"unknown regression metric {metric!r}: one of {sorted(_REGRESSION_METRICS)}"
        )
    d = _as_dates(dates)
    if d.size != target.size:
        raise ConfigError(f"dates has {d.size} entries against {target.size} outcomes")

    groups = _feature_groups(features)
    known = set(X.columns)
    for name, columns in groups.items():
        unknown = [c for c in columns if c not in known]
        if unknown:
            raise ConfigError(
                f"feature group {name!r} names columns absent from X: {unknown}"
            )
    used = sorted({c for columns in groups.values() for c in columns})
    if not used:
        raise ConfigError("no feature columns were named, so there is nothing to ablate")

    if folds is None:
        folds = walk_forward_folds(
            d, n_folds, min_train=min_train, embargo_days=embargo_days
        )
    slices = _fold_slices(d, folds)
    higher = _REGRESSION_METRICS[metric]

    def run(columns: list[str]) -> tuple[float, float, int]:
        if not columns:
            raise ConfigError(
                "dropping this group leaves no features at all, so there is no model "
                "left to refit"
            )
        preds = np.full(target.size, np.nan)
        fold_scores = []
        for fold, test in slices:
            train = _train_index(d, fold)
            if train.size == 0:
                raise ConfigError(
                    f"fold {fold.index} has no training rows under a "
                    f"{fold.embargo_days}-day embargo"
                )
            out = np.asarray(
                fit_predict(X.iloc[train][columns], target[train], X.iloc[test][columns]),
                dtype=float,
            )
            if out.size != test.size:
                raise ConfigError(
                    f"fit_predict returned {out.size} predictions for {test.size} test "
                    f"rows in fold {fold.index}"
                )
            preds[test] = out
            fold_scores.append(_regression_score(metric, target[test], out))
        idx = np.concatenate([test for _, test in slices])
        pooled = _regression_score(metric, target[idx], preds[idx])
        sd = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else float("nan")
        return pooled, sd, int(idx.size)

    full_score, full_sd, n_test = run(used)
    rows = [
        {
            "group": "all features",
            "n_columns": len(used),
            "score": full_score,
            "damage": 0.0,
            "fold_sd": full_sd,
            "n_test": n_test,
        }
    ]
    for name, columns in groups.items():
        remaining = [c for c in used if c not in set(columns)]
        score, sd, n = run(remaining)
        damage = (full_score - score) if higher else (score - full_score)
        rows.append(
            {
                "group": f"without {name}",
                "n_columns": len(remaining),
                "score": score,
                "damage": damage,
                "fold_sd": sd,
                "n_test": n,
            }
        )

    frame = pd.DataFrame(rows)
    ordered = pd.concat(
        [frame.iloc[[0]], frame.iloc[1:].sort_values("damage", ascending=False)],
        ignore_index=True,
    )
    return ordered


def _feature_groups(
    features: Mapping[str, Sequence[str]] | Sequence[str]
) -> dict[str, list[str]]:
    """One group per column where the caller passed a plain list of column names."""
    if isinstance(features, Mapping):
        return {str(name): list(columns) for name, columns in features.items()}
    return {str(column): [column] for column in features}


def _fold_notes(
    folds: Sequence[Fold],
    wins: int,
    lifts: Sequence[float],
    thin: Sequence[str],
    n_unscored: int,
    n_total: int,
) -> list[str]:
    """The lines every fold-structured result carries, including the awkward ones."""
    embargo = folds[0].embargo_days if folds else 0
    embargoed = sum(f.n_embargoed for f in folds)
    notes = [
        f"{len(folds)} walk-forward folds, split by date, testing "
        f"{folds[0].test_start} to {folds[-1].test_end}."
    ]
    if embargo <= 0:
        notes.append(
            "No embargo was applied. If the label is a forward return or a forward "
            "growth rate, training observations near each cut had label windows still "
            "open into the test window and this result is not protected against "
            "leakage."
        )
    else:
        notes.append(
            f"{embargo}-day embargo between train and test, which dropped "
            f"{embargoed:,} observations into the gap so that no training label window "
            "was still open when its test window began."
        )
    if lifts:
        sd = float(np.std(lifts, ddof=1)) if len(lifts) > 1 else float("nan")
        notes.append(
            f"The model beat the baseline in {wins} of {len(lifts)} folds, mean fold "
            f"lift {float(np.mean(lifts)):+.4f} with a fold-to-fold standard deviation "
            f"of {sd:.4f}."
        )
    if thin:
        notes.append(
            f"Thin test folds, under {THIN_FOLD_OBSERVATIONS} observations: "
            f"{', '.join(thin)}."
        )
    if n_unscored:
        notes.append(
            f"{n_unscored:,} of {n_total:,} observations fall before the first test "
            "window and were never scored: they are the initial training history."
        )
    return notes


def _regression_score(metric: str, y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.all():
        raise ConfigError(
            f"{int((~ok).sum())} of {y.size} rows carry a missing or infinite value. "
            "Drop them with a stated reason rather than scoring around them"
        )
    if metric == "mae":
        return float(np.mean(np.abs(y - p)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y - p) ** 2)))
    value = spearman(y, p)
    if value is None:
        raise NotMeaningfulError(
            f"a rank correlation is undefined on {y.size} rows where one side has no "
            "variation: a constant prediction cannot be rank correlated with anything"
        )
    return value


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve as the Mann-Whitney statistic, with ties halved.

    The rank form rather than a trapezoid over thresholds, because it handles
    tied scores exactly: two names the model cannot separate contribute half a
    point, which is what a coin toss between them earns.
    """
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise NotMeaningfulError(
            "AUC is undefined where every observation belongs to one class"
        )
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _expected_random_ndcg(
    relevance: Mapping[Any, float], ranking: Sequence[Any], k: int
) -> float:
    """NDCG a uniform random ordering earns in expectation, in closed form.

    Under a random permutation each candidate is equally likely in each slot, so
    the expected gain at every position is the mean relevance of the universe,
    and the expected DCG is that mean times the sum of the discounts over the
    first k positions. Exact, deterministic, and no seed to argue about.
    """
    grades = [float(relevance.get(item, 0.0)) for item in ranking]
    n = len(grades)
    if n == 0:
        raise NotMeaningfulError("an empty ranking has no expected NDCG")
    depth = min(k, n)
    discount = sum(1.0 / np.log2(i + 2.0) for i in range(depth))
    expected_dcg = discount * (sum(grades) / n)
    ideal = sorted(grades, reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2.0) for i, g in enumerate(ideal))
    if idcg <= 0:
        raise NotMeaningfulError(
            "no relevant candidate appears in the ranking, so NDCG is undefined"
        )
    return float(expected_dcg / idcg)


def _fold_slices(d: np.ndarray, folds: Sequence[Fold]) -> list[tuple[Fold, np.ndarray]]:
    if not folds:
        raise ConfigError("no folds were supplied")
    out = []
    for fold in folds:
        idx = _test_index(d, fold)
        if idx.size == 0:
            raise ConfigError(
                f"fold {fold.index} tests {fold.test_start} to {fold.test_end}, where "
                "the supplied dates hold no observations"
            )
        out.append((fold, idx))
    return out


def _test_index(d: np.ndarray, fold: Fold) -> np.ndarray:
    return np.flatnonzero((d >= fold.test_start) & (d <= fold.test_end))


def _train_index(d: np.ndarray, fold: Fold) -> np.ndarray:
    """Every observation strictly before the cut, which is the test start less embargo."""
    return np.flatnonzero(d < fold.test_start - timedelta(days=fold.embargo_days))


def _as_dates(values: Sequence[Any]) -> np.ndarray:
    out = []
    for value in values:
        if isinstance(value, datetime):
            out.append(value.date())
        elif isinstance(value, date):
            out.append(value)
        else:
            try:
                out.append(pd.Timestamp(value).date())
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"{value!r} is not a date, and a walk-forward split has no meaning "
                    "without one"
                ) from exc
    if not out:
        raise ConfigError("no observation dates were supplied")
    return np.array(out, dtype=object)


def _as_floats(values: Sequence[Any], name: str) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be numeric: {exc}") from exc
    if out.size == 0:
        raise ConfigError(f"{name} is empty")
    return out


def _as_binary(values: Sequence[Any]) -> np.ndarray:
    out = np.asarray(values)
    if out.dtype == bool:
        return out.astype(float).ravel()
    out = _as_floats(out, "y_true")
    bad = set(np.unique(out)) - {0.0, 1.0}
    if bad:
        raise ConfigError(
            f"the outcome must be 0 or 1 for a classification metric, found {sorted(bad)}"
        )
    return out


def _sample_names(items: Sequence[Any], limit: int = 5) -> str:
    shown = [str(item) for item in items[:limit]]
    if len(items) > limit:
        shown.append(f"and {len(items) - limit} more")
    return ", ".join(shown)
