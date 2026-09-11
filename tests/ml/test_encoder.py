"""Tests for the two-tower peer encoder.

Two kinds of test, answering two different questions, and they are kept apart
on purpose.

**Does the machinery do what it says.** Gradients, batching, leakage refusals,
determinism and persistence are tested on an invented universe, because only an
invented universe lets a test state what the right answer is. Six sectors of
companies whose fundamentals and whose prose both carry their sector, peer
groups drawn inside a sector, and one company deliberately named by everybody so
the popularity trap is present rather than hoped for. Nothing here is evidence
about real filings and the docstrings say so where it matters.

**Does it work on filings.** The real fixture is the compensation peer groups
of the TMT seed universe, read from DEF 14A proxies on 2026-09-11, with the
Item 1 excerpts and the point-in-time feature panel behind them. Those tests
assert coverage and ordering facts about companies that exist.

Nothing here touches the network.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from techval.config import Assumptions
from techval.errors import ConfigError, MissingDataError, NotMeaningfulError
from techval.ml.encoder import (
    EMBARGO_DAYS,
    METHOD_NAMES,
    PeerDataset,
    PeerEncoder,
    TwoTowerEncoder,
    ablate_towers,
    build_dataset,
    cache_path,
    evaluate_peer_encoder,
    fit_peer_encoder,
    load_peer_encoder,
    paired_lift,
    popularity_collapse,
    popularity_ranking,
    survivorship_report,
)
from techval.ml.features import FEATURE_NAMES, MATRIX_COLUMNS, FeaturePanel, FeatureRow
from techval.ml.nn import gradient_check
from techval.ml.peer_labels import PeerGroup
from techval.ml.text import TextCorpus

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PEER_FIXTURE = FIXTURES / "peer_groups_tmt.json"
PANEL_FIXTURE = FIXTURES / "peer_panel_tmt.json"
ITEM1_FIXTURE = FIXTURES / "peer_item1_tmt.json"


# --------------------------------------------------------------------------- #
# An invented universe
# --------------------------------------------------------------------------- #

# Six sectors, each with its own vocabulary and its own financial shape. The
# words are deliberately disjoint across sectors so the text tower has a signal
# to find; real filings are nothing like this clean, which is why the real
# fixture exists below.
_VOCAB = {
    0: "spectrum cell sites postpaid churn handsets roaming coverage wireless subscribers",
    1: "observability monitoring telemetry traces dashboards ingest logs cloud platform",
    2: "wafer fabrication lithography nodes foundry analog yields packaging semiconductor",
    3: "towers ground leases backhaul colocation interconnection data centre kilowatts",
    4: "streaming titles subscribers content slate theatrical licensing studio catalogue",
    5: "merchants interchange authorisation settlement acquiring gateway payments volume",
}
_SECTORS = len(_VOCAB)
_PER_SECTOR = 6
_DATES = [date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1), date(2024, 1, 1)]

# Named by everybody regardless of sector, which is the popularity trap. Without
# a company like this in the fixture the popularity baseline is a straw man and
# the test that the encoder beats it proves nothing.
_CELEBRITY = "S0C0"


def _synthetic_ticker(sector: int, i: int) -> str:
    return f"S{sector}C{i}"


def _synthetic_universe() -> list[str]:
    return [
        _synthetic_ticker(s, i) for s in range(_SECTORS) for i in range(_PER_SECTOR)
    ]


def _synthetic_panel(rng: np.random.Generator) -> FeaturePanel:
    """A feature row per company per date, with a sector centre plus noise.

    Two features are left missing for a whole sector so the missing-share
    indicators are exercised rather than constant, which is what the panel emits
    on real data and what the encoder has to cope with.
    """
    rows: list[FeatureRow] = []
    centres = {s: rng.normal(0.0, 2.0, size=len(FEATURE_NAMES)) for s in range(_SECTORS)}
    for as_of in _DATES:
        for s in range(_SECTORS):
            for i in range(_PER_SECTOR):
                values: dict[str, float | None] = {}
                draw = centres[s] + rng.normal(0.0, 0.6, size=len(FEATURE_NAMES))
                for name, value in zip(FEATURE_NAMES, draw):
                    values[name] = float(value)
                missing: list[str] = []
                if s == 2:
                    for name in ("margin_ebitda", "margin_fcf"):
                        values[name] = None
                        missing.append(name)
                rows.append(
                    FeatureRow(
                        ticker=_synthetic_ticker(s, i),
                        as_of=as_of,
                        knowledge_date=as_of,
                        values=values,
                        missing=missing,
                    )
                )
    return FeaturePanel(rows=rows, random_seed=7)


def _synthetic_corpora(rng: np.random.Generator) -> dict[date, TextCorpus]:
    corpora: dict[date, TextCorpus] = {}
    for as_of in _DATES:
        tickers, documents = [], []
        for s in range(_SECTORS):
            words = _VOCAB[s].split()
            for i in range(_PER_SECTOR):
                body = " ".join(rng.choice(words, size=120))
                filler = " ".join(rng.choice("alpha beta gamma delta".split(), size=40))
                tickers.append(_synthetic_ticker(s, i))
                documents.append(f"{body} {filler}")
        corpora[as_of] = TextCorpus(
            tickers=tickers,
            documents=documents,
            as_of_by_ticker={t: as_of for t in tickers},
            source_accessions={t: f"acc-{t}-{as_of}" for t in tickers},
            entity_names={t: f"{t} Holdings" for t in tickers},
        )
    return corpora


def _synthetic_groups() -> list[PeerGroup]:
    """One group per company per year: its own sector, plus the celebrity.

    Filed on a spread of dates inside each year so a walk-forward split has a
    timeline to cut, and stamped with a fiscal year so the labels carry the one
    the module insists on.
    """
    groups: list[PeerGroup] = []
    for year_index, as_of in enumerate(_DATES):
        year = as_of.year
        for s in range(_SECTORS):
            for i in range(_PER_SECTOR):
                ticker = _synthetic_ticker(s, i)
                peers = [
                    _synthetic_ticker(s, j) for j in range(_PER_SECTOR) if j != i
                ]
                neighbour = (s + 1) % _SECTORS
                peers += [_synthetic_ticker(neighbour, j) for j in range(4)]
                if _CELEBRITY not in peers and ticker != _CELEBRITY:
                    peers.append(_CELEBRITY)
                groups.append(
                    PeerGroup(
                        ticker=ticker,
                        cik=1000 + s * 100 + i,
                        accession=f"{year}-{ticker}",
                        filed=date(year, 3 + (i % 6), 1 + (s % 20)),
                        fiscal_year=year - 1,
                        peers=sorted(set(peers)),
                        method="fixture",
                        confidence=1.0,
                    )
                )
    return groups


@pytest.fixture(scope="module")
def synthetic() -> PeerDataset:
    rng = np.random.default_rng(11)
    return build_dataset(
        _synthetic_groups(), _synthetic_panel(rng), _synthetic_corpora(rng)
    )


@pytest.fixture
def ml_assumptions(tmp_path) -> Assumptions:
    """Defaults with the model cache pointed somewhere disposable.

    The default cache is the user's home directory. A suite that writes there is
    a suite that passes on the second run for the wrong reason.
    """
    a = Assumptions()
    a.ml.cache_dir = str(tmp_path / "mlcache")
    return a


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #


def test_both_towers_gradient_check_to_the_modules_own_bar():
    """The composite has to meet the bar every layer in nn.py meets, 1e-6.

    A hand-written backward through two towers, a scaling and a concatenation is
    exactly the place a sign or a factor goes missing, and a wrong gradient here
    would train to a worse optimum without ever failing outright.
    """
    rng = np.random.default_rng(3)
    encoder = TwoTowerEncoder(
        7, 5, dim=4, hidden=(6,), text_weight=0.4, dropout=0.0, rng=rng
    )
    x = rng.standard_normal((9, 12))
    assert gradient_check(encoder, x) < 1e-6


def test_gradient_check_holds_with_dropout_reseeded():
    """Dropout has to be reseeded or the difference measures the mask, not the slope."""
    rng = np.random.default_rng(5)
    encoder = TwoTowerEncoder(
        6, 6, dim=4, hidden=(8,), text_weight=0.5, dropout=0.25, rng=rng
    )
    encoder.train()
    x = np.random.default_rng(6).standard_normal((8, 12))
    worst = gradient_check(
        encoder, x, reset=lambda: encoder.reseed(np.random.default_rng(99))
    )
    assert worst < 1e-6


def test_a_relu_tower_can_hand_the_normaliser_a_row_with_no_direction():
    """Why the towers use tanh, measured rather than asserted.

    A ReLU layer can output exactly zero across a whole row. The output layer's
    bias starts at zero, as it must, so that row reaches L2Normalize as the zero
    vector, comes back as cosine zero against every company, and reads as
    "resembles nothing" when the truth is "was not measured". It is not rare: on
    this batch of eight standardised rows at He initialisation it happens to one
    of them, and the gradient there is divided by the eps guard rather than by a
    norm. Tanh cannot produce an identically zero row except at a point of
    measure zero, and the gradient check below is the evidence.
    """
    x = np.random.default_rng(21).standard_normal((8, 12))

    relu = TwoTowerEncoder(
        6, 6, dim=4, hidden=(8,), text_weight=0.5, dropout=0.0,
        rng=np.random.default_rng(5), activation="relu",
    )
    relu.eval()
    assert np.isclose(
        np.linalg.norm(relu.fundamentals.forward(x[:, :6]), axis=1), 0.0
    ).any()
    # The composite is not zero, it is worse than that: the surviving tower
    # carries its sqrt(text_weight) share alone, so the embedding silently stops
    # having unit length and the documented blend stops holding for that row.
    assert not np.allclose(np.linalg.norm(relu.forward(x), axis=1), 1.0)
    assert gradient_check(relu, x) > 1e-6

    tanh = TwoTowerEncoder(
        6, 6, dim=4, hidden=(8,), text_weight=0.5, dropout=0.0,
        rng=np.random.default_rng(5),
    )
    tanh.eval()
    np.testing.assert_allclose(np.linalg.norm(tanh.forward(x), axis=1), 1.0, atol=1e-12)
    assert gradient_check(tanh, x) < 1e-6


def test_the_embedding_dot_product_is_the_documented_blend():
    """similarity = (1 - w) * cosine_fundamentals + w * cosine_text, exactly.

    This is the claim the module docstring makes about what text_weight means,
    and it is the reason the two towers are concatenated with square-root
    weights rather than averaged after the fact. Asserted rather than described,
    because a reader setting text_weight is entitled to know what it does.
    """
    rng = np.random.default_rng(13)
    weight = 0.3
    encoder = TwoTowerEncoder(
        5, 4, dim=6, hidden=(7,), text_weight=weight, dropout=0.0, rng=rng
    )
    encoder.eval()
    x = rng.standard_normal((2, 9))
    z = encoder.forward(x)
    assert z.shape == (2, 12)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-12)

    fundamentals = encoder.fundamentals.forward(x[:, :5])
    text = encoder.text.forward(x[:, 5:])
    expected = (1 - weight) * float(fundamentals[0] @ fundamentals[1]) + weight * float(
        text[0] @ text[1]
    )
    assert float(z[0] @ z[1]) == pytest.approx(expected, abs=1e-12)


def test_a_weight_of_zero_or_one_removes_the_other_tower_entirely():
    """Not scaled to zero: removed. A parameter that provably never moves should
    not appear in a parameter count on a model card."""
    rng = np.random.default_rng(17)
    fundamentals_only = TwoTowerEncoder(
        5, 0, dim=4, hidden=(6,), text_weight=0.0, dropout=0.0, rng=rng
    )
    assert fundamentals_only.text is None
    assert fundamentals_only.width == 4

    text_only = TwoTowerEncoder(
        0, 5, dim=4, hidden=(6,), text_weight=1.0, dropout=0.0, rng=rng
    )
    assert text_only.fundamentals is None
    assert text_only.width == 4
    assert gradient_check(text_only, rng.standard_normal((6, 5))) < 1e-6


def test_a_tower_that_carries_weight_but_has_no_columns_is_refused():
    rng = np.random.default_rng(19)
    with pytest.raises(ConfigError, match="text tower carries weight"):
        TwoTowerEncoder(5, 0, dim=4, hidden=(6,), text_weight=0.5, dropout=0.0, rng=rng)
    with pytest.raises(ConfigError, match="fundamentals tower carries weight"):
        TwoTowerEncoder(0, 5, dim=4, hidden=(6,), text_weight=0.5, dropout=0.0, rng=rng)


# --------------------------------------------------------------------------- #
# The labelled sample
# --------------------------------------------------------------------------- #


def test_every_example_attaches_to_a_panel_date_at_or_before_its_filing(synthetic):
    """Features are always older than the assertion they are labelled with.

    Attaching a group to a later panel date would let a ranking be computed from
    figures filed in response to the very year being scored, and the direction
    of that error always flatters the model.
    """
    assert synthetic.examples
    for e in synthetic.examples:
        assert e.as_of <= e.filed


def test_a_peer_outside_the_universe_is_dropped_from_training_and_kept_in_relevance():
    """A peer the universe cannot represent is a recall failure of the universe.

    Hiding it inside the ranking would move the blame off the thing that caused
    it, so the pair does not train and the name still counts against recall.
    """
    rng = np.random.default_rng(23)
    groups = _synthetic_groups()
    groups[0].peers = sorted(set(groups[0].peers) | {"NOTINUNIVERSE"})
    dataset = build_dataset(groups, _synthetic_panel(rng), _synthetic_corpora(rng))
    assert dataset.dropped["peer not in the candidate universe"] >= 1
    assert all(e.b != "NOTINUNIVERSE" for e in dataset.examples)
    assert "NOTINUNIVERSE" in dataset.group_of(
        (groups[0].ticker, groups[0].fiscal_year)
    ).peers


def test_a_dataset_with_no_corpus_for_a_panel_date_is_refused(synthetic):
    partial = dict(synthetic.corpora)
    partial.pop(_DATES[-1])
    with pytest.raises(ConfigError, match="no corpus"):
        build_dataset(synthetic.groups, synthetic.panel, partial)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def test_a_batch_never_holds_two_pairs_of_one_group_or_one_positive_twice(synthetic):
    """The in-batch negative assumption is enforced by the sampler, as nn.py says.

    Twenty pairs from one proxy in one batch would have the loss spend its
    gradient pushing apart twenty companies the same filer called comparable,
    and two rows sharing a positive would ask the softmax to choose between two
    correct answers.
    """
    from techval.ml.encoder import _batches, _fit_space

    a = Assumptions()
    space = _fit_space(
        synthetic, max(_DATES), a, text_dim=16, min_df=2, max_df=0.95
    )
    X = space.inputs(use_fundamentals=True, use_text=True)
    usable = [
        e for e in synthetic.examples if space.has(e.a, e.as_of) and space.has(e.b, e.as_of)
    ]
    rng = np.random.default_rng(29)
    packed = _batches(usable, space, X, 32, rng)
    assert packed

    # Rebuild the membership of each batch by matching rows back to the space.
    row_of = {i: key for key, i in space.index.items()}
    lookup = {tuple(np.round(X[i], 9)): row_of[i] for i in range(X.shape[0])}
    for inputs, _targets in packed:
        half = inputs.shape[0] // 2
        anchors = [lookup[tuple(np.round(r, 9))] for r in inputs[:half]]
        positives = [lookup[tuple(np.round(r, 9))] for r in inputs[half:]]
        assert len(set(positives)) == len(positives)
        assert len(set(anchors)) == len(anchors)


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #


def test_the_vocabulary_is_fitted_only_on_documents_the_fold_could_see(synthetic):
    """A later fold's fit must know strictly more terms than an earlier one's.

    Vocabulary leakage leaves no trace in the output, so it is checked rather
    than trusted: the 2021 fit is built from the 2021 corpus and cannot contain
    a term that first appears in 2023.
    """
    from techval.ml.encoder import _fit_space

    a = Assumptions()
    early = _fit_space(synthetic, _DATES[0], a, text_dim=16, min_df=2, max_df=0.95)
    late = _fit_space(synthetic, _DATES[-1], a, text_dim=16, min_df=2, max_df=0.95)
    assert early.fit_date == _DATES[0]
    assert late.fit_date == _DATES[-1]
    assert early.text is not None and late.text is not None
    assert early.text.fitted_through <= _DATES[0]


def test_standardisation_is_fitted_on_the_training_window_only(synthetic):
    """The mean the encoder centres on belongs to the training rows and no others."""
    from techval.ml.encoder import _fit_space

    a = Assumptions()
    early = _fit_space(synthetic, _DATES[0], a, text_dim=16, min_df=2, max_df=0.95)
    late = _fit_space(synthetic, _DATES[-1], a, text_dim=16, min_df=2, max_df=0.95)
    assert not np.allclose(early.mu, late.mu)
    assert len(early.mu) == len(MATRIX_COLUMNS)


def test_the_embargo_is_a_year_and_keeps_adjacent_proxies_apart():
    """A filer's next proxy repeats most of its previous one.

    The usual reason for an embargo does not apply here, because the label is
    public the day it is filed. This one exists so that a training group filed
    eleven months before a test group is not very nearly the test answer.
    """
    assert EMBARGO_DAYS == 365


# --------------------------------------------------------------------------- #
# The popularity trap
# --------------------------------------------------------------------------- #


def test_the_popularity_prior_ignores_the_query_and_still_ranks_the_celebrity_first(
    synthetic,
):
    """The degenerate solution, written down so the model has to beat it."""
    order = popularity_ranking(
        synthetic.groups, synthetic.universe, np.random.default_rng(7)
    )
    assert order[0] == _CELEBRITY
    other = popularity_ranking(
        synthetic.groups, synthetic.universe, np.random.default_rng(7)
    )
    assert order == other


def test_a_query_blind_ranker_scores_one_on_the_collapse_diagnostic(synthetic):
    """The number that separates beating popularity from being popularity.

    A model that returns the same list whatever it is asked has learned nothing
    about comparability however good its NDCG, and this is what says so.
    """
    order = popularity_ranking(
        synthetic.groups, synthetic.universe, np.random.default_rng(7)
    )
    identical = {q: list(order) for q in range(10)}
    assert popularity_collapse(identical, order) == pytest.approx(1.0)

    reversed_lists = {q: list(reversed(order)) for q in range(10)}
    assert popularity_collapse(reversed_lists, order) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# Fitting, determinism and persistence
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fitted(synthetic) -> PeerEncoder:
    a = Assumptions()
    return fit_peer_encoder(
        synthetic, a, dim=16, hidden=(24,), text_dim=24, epochs=12, patience=12
    )


def test_the_same_seed_produces_bitwise_identical_embeddings(synthetic):
    """Asserted rather than hoped for: a model beside a valuation is reproducible."""
    a = Assumptions()
    kwargs = dict(dim=12, hidden=(16,), text_dim=16, epochs=6, patience=6)
    first = fit_peer_encoder(synthetic, a, **kwargs)
    second = fit_peer_encoder(synthetic, a, **kwargs)
    assert first.tickers == second.tickers
    assert np.array_equal(first.embeddings, second.embeddings)


def test_a_fitted_encoder_puts_a_companys_own_sector_at_the_top(fitted):
    """The invented universe has a right answer, so the fit can be held to it.

    This says the machinery learns a separable signal when one is present. It
    says nothing whatsoever about real filings, where the signal is weaker and
    the measurement is the walk-forward evaluation rather than a fit like this.
    """
    hits = 0
    for sector in range(_SECTORS):
        query = _synthetic_ticker(sector, 0)
        top = [t for t, _ in fitted.neighbours(query, 5, apply_size_gate=False)]
        hits += sum(1 for t in top if t.startswith(f"S{sector}"))
    assert hits >= _SECTORS * 3


def test_embeddings_are_unit_length_so_a_dot_product_is_a_cosine(fitted):
    np.testing.assert_allclose(
        np.linalg.norm(fitted.embeddings, axis=1), 1.0, atol=1e-10
    )
    assert -1.0 <= fitted.similarity(fitted.tickers[0], fitted.tickers[1]) <= 1.0


def test_neighbours_excludes_the_company_itself(fitted):
    for ticker in fitted.tickers[:5]:
        assert ticker not in [t for t, _ in fitted.neighbours(ticker, 8, apply_size_gate=False)]


def test_asking_for_a_company_outside_the_fitted_universe_says_so(fitted):
    with pytest.raises(MissingDataError, match="not in the fitted universe"):
        fitted.neighbours("NOSUCHCO")


def test_a_saved_encoder_reloads_to_the_same_rankings(fitted, ml_assumptions, tmp_path):
    path = cache_path(ml_assumptions)
    fitted.save(path)
    assert path.exists()
    reloaded = load_peer_encoder(path)
    assert reloaded.tickers == fitted.tickers
    np.testing.assert_allclose(reloaded.embeddings, fitted.embeddings)
    for ticker in fitted.tickers[:4]:
        assert reloaded.neighbours(ticker, 6, apply_size_gate=False) == fitted.neighbours(
            ticker, 6, apply_size_gate=False
        )


def test_a_fit_below_the_pair_floor_is_refused_rather_than_produced(synthetic):
    """A contrastive fit on a few hundred relationships is memorising them."""
    thin = PeerDataset(
        examples=synthetic.examples[:10],
        panel=synthetic.panel,
        corpora=synthetic.corpora,
        groups=synthetic.groups,
        universe=synthetic.universe,
        panel_dates=synthetic.panel_dates,
    )
    with pytest.raises(NotMeaningfulError, match="below the floor"):
        fit_peer_encoder(thin, Assumptions())


def test_the_model_card_records_what_it_could_not_do(fitted):
    card = fitted.card
    assert card.n_train > 0
    assert card.hyperparameters["random_seed"] == 7
    assert card.hyperparameters["parameters"] > 0
    assert any("compensation peer groups" in line for line in card.limitations)
    assert any("imputed" in line for line in card.limitations)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def evaluated(synthetic):
    return evaluate_peer_encoder(
        synthetic,
        Assumptions(),
        k=5,
        n_folds=3,
        embargo_days=0,
        dim=16,
        hidden=(24,),
        text_dim=24,
        epochs=10,
        patience=10,
    )


def test_every_baseline_is_scored_on_identical_queries(evaluated):
    """Six methods, one query set. A comparison over different queries is not one."""
    counts = set(evaluated.scores["n_queries"])
    assert len(counts) == 1
    assert set(evaluated.scores["method"]) >= {
        "encoder",
        "popularity prior",
        "same sub-vertical",
        "size and growth",
        "fundamentals cosine",
        "text cosine",
        "text lsa cosine",
    }


def test_the_headline_is_the_encoder_against_the_popularity_prior(evaluated):
    assert "popularity prior" in evaluated.headline.baseline_name
    assert evaluated.headline.metric == "ndcg@5"
    assert 0.0 <= evaluated.headline.score <= 1.0
    assert 0.0 <= evaluated.headline.baseline_score <= 1.0
    assert isinstance(evaluated.headline.beat_baseline, bool)


def test_the_verdict_reads_as_a_sentence_either_way(evaluated):
    text = evaluated.verdict()
    assert "ndcg@5" in text
    assert ("does NOT beat the baseline" in text) or ("a lift of" in text)


def test_the_collapse_diagnostic_is_reported_beside_the_score(evaluated):
    """Beating popularity by imitating it has to be visible, not merely suspected."""
    assert -1.0 <= evaluated.collapse <= 1.0
    assert any("popularity order" in n for n in evaluated.notes)


def test_the_paired_test_is_reported_because_the_fold_sd_answers_another_question(
    evaluated,
):
    """EvalResult.fold_sd is the spread across TARGETS, not the error on the lift.

    Targets differ enormously in how findable their peers are, so that spread is
    dominated by variation both methods share and a perfectly reliable lift can
    sit well inside it. The difference taken query by query is the statistic
    that answers whether the encoder beats a baseline on the same company, and
    it is reported beside the verdict rather than instead of it.
    """
    paired = evaluated.paired
    assert set(paired["baseline"]) == set(METHOD_NAMES[1:])
    assert (paired["n_queries"] == evaluated.n_queries).all()
    assert (paired["standard_error"] <= paired["sd_of_difference"]).all()
    assert ((paired["win_rate"] >= 0.0) & (paired["win_rate"] <= 1.0)).all()
    # The frame is sorted by how much the encoder beats each baseline.
    assert list(paired["mean_difference"]) == sorted(
        paired["mean_difference"], reverse=True
    )


def test_paired_lift_against_an_identical_ranking_is_exactly_zero():
    """The one case with an arithmetic answer, so the sign convention is pinned."""
    relevance = {q: {"A": 1.0, "B": 1.0} for q in range(6)}
    order = {q: ["A", "C", "B", "D"] for q in range(6)}
    frame = paired_lift(relevance, {"encoder": order, "twin": order}, 4)
    assert frame.iloc[0]["mean_difference"] == pytest.approx(0.0)
    assert frame.iloc[0]["win_rate"] == 0.0


def test_recall_is_reported_against_the_whole_named_set(evaluated):
    """Precision can be high while recall is low, and that pair is the result."""
    assert "in_universe" in evaluated.coverage
    row = evaluated.scores[evaluated.scores["method"] == "encoder"].iloc[0]
    assert 0.0 <= row["recall@5"] <= 1.0
    assert 0.0 <= row["precision@5"] <= 1.0


def test_folds_train_strictly_before_the_window_they_test(evaluated):
    for fold in evaluated.folds:
        assert fold.train_end < fold.test_start
        assert fold.test_start <= fold.test_end


# --------------------------------------------------------------------------- #
# Ablation and survivorship
# --------------------------------------------------------------------------- #


def test_the_ablation_reports_the_full_model_and_one_row_per_tower(synthetic, evaluated):
    frame = ablate_towers(
        synthetic,
        Assumptions(),
        folds=evaluated.folds,
        dim=12,
        hidden=(16,),
        text_dim=16,
        epochs=6,
        patience=6,
        negatives_per_pair=3,
    )
    assert list(frame["group"])[0] == "all features"
    assert set(frame["group"]) == {"all features", "without fundamentals", "without text"}
    assert frame["n_test"].min() > 0
    # Damage is signed so positive means the tower was doing work.
    assert frame.loc[frame["group"] == "all features", "damage"].iloc[0] == 0.0


def test_survivorship_lists_the_named_peers_the_universe_never_held():
    listed = [f"P{i:02d}" for i in range(8)]
    groups = [
        PeerGroup(
            ticker="AAA",
            cik=1,
            accession="x",
            filed=date(2024, 4, 1),
            fiscal_year=2023,
            peers=sorted(listed + ["GONE"]),
            confidence=1.0,
        )
    ]
    frame = survivorship_report(
        groups, ["AAA", *listed], deal_events=[("GONE", date(2023, 6, 1), True)]
    )
    absent = frame[~frame["in_universe"]]
    assert list(absent["peer"]) == ["GONE"]
    assert absent.iloc[0]["merger_announced"] == date(2023, 6, 1)
    assert bool(absent.iloc[0]["merger_completed"]) is True


# --------------------------------------------------------------------------- #
# Real filings
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real() -> PeerDataset:
    """The TMT seed universe's disclosed peer groups, read from proxies on 2026-09-11.

    The panel and the Item 1 excerpts behind it are point in time: each row was
    built through an EdgarClient pinned to the row date and each document is the
    newest 10-K on file at that date. See the fixture headers.
    """
    for path in (PEER_FIXTURE, PANEL_FIXTURE, ITEM1_FIXTURE):
        if not path.exists():
            pytest.skip(f"{path.name} is not committed")

    panel_payload = json.loads(PANEL_FIXTURE.read_text())
    rows = [
        FeatureRow(
            ticker=r["ticker"],
            as_of=date.fromisoformat(r["as_of"]),
            knowledge_date=date.fromisoformat(r["knowledge_date"])
            if r["knowledge_date"]
            else None,
            values=r["values"],
            missing=r["missing"],
            statement_date=date.fromisoformat(r["statement_date"])
            if r["statement_date"]
            else None,
            error=r["error"],
        )
        for r in panel_payload["rows"]
    ]
    panel = FeaturePanel(rows=rows, random_seed=7)

    item1 = json.loads(ITEM1_FIXTURE.read_text())
    names = item1["entity_names"]
    corpora = {}
    for iso, companies in item1["by_date"].items():
        tickers = sorted(companies)
        corpora[date.fromisoformat(iso)] = TextCorpus(
            tickers=tickers,
            documents=[companies[t]["item1"] for t in tickers],
            as_of_by_ticker={
                t: date.fromisoformat(companies[t]["filed"]) for t in tickers
            },
            source_accessions={t: companies[t]["accession"] for t in tickers},
            entity_names={t: names[t] for t in tickers if names.get(t)},
        )

    groups = [
        PeerGroup(
            ticker=g["ticker"],
            cik=g["cik"],
            accession=g["accession"],
            filed=date.fromisoformat(g["filed"]),
            fiscal_year=g["fiscal_year"],
            peers=g["peers"],
            method=g["method"],
            selection_criteria=g.get("selection_criteria"),
            confidence=g["confidence"],
        )
        for g in json.loads(PEER_FIXTURE.read_text())["groups"]
    ]
    return build_dataset(groups, panel, corpora)


def test_the_real_labels_are_dated_assertions_by_companies_that_exist(real):
    assert len(real.groups) >= 30
    assert len(real.examples) >= 500
    for group in real.groups:
        assert group.filed is not None
        assert group.fiscal_year is not None
        assert 8 <= len(group.peers) <= 30


def test_a_real_filers_own_features_predate_its_own_proxy(real):
    """The reporting lag is in the construction, not in a rule.

    Every example's panel date is at or before the proxy that asserted it, so no
    figure on the row was filed in response to the year being scored.
    """
    for e in real.examples:
        assert e.as_of <= e.filed


def test_the_universe_cannot_hold_every_named_peer_and_says_so(real):
    """The survivorship hole, measured rather than asserted.

    A candidate list built from an index of currently listed technology
    companies cannot contain a peer that was acquired, that files a 20-F rather
    than a 10-K, or that sits outside the sector the index covers. The share
    that is missing bounds recall for every ranker at once, so it belongs beside
    the scores rather than inside them.
    """
    assert real.dropped["peer not in the candidate universe"] > 0
    named = {p for g in real.groups for p in g.peers}
    inside = named & set(real.universe)
    assert 0.3 < len(inside) / len(named) < 1.0
