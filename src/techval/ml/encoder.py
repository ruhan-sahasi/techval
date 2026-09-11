"""The learned peer encoder: which technology companies are comparable, measured.

The engine's own limitations section has always conceded that its comp sets are
hand-picked and that comparability is the user's judgment. This module is the
attempt to stop conceding it. Two encoders, one over what a company's financial
profile looks like and one over what the company says it does, are trained to
put a filer and the peers its own compensation committee disclosed near each
other on the unit sphere. Nothing about the label is the author's opinion:
``peer_labels.py`` reads the peer tables out of DEF 14A proxies, so every
training pair is a dated assertion of comparability that a board of directors
signed.

**The architecture, and why the blend is the architecture rather than a
post-processing step.** Each tower ends in ``L2Normalize``, so each returns a
point on a unit sphere and a dot product there is a cosine. A company's
embedding is the two towers' outputs concatenated, the fundamentals half scaled
by ``sqrt(1 - text_weight)`` and the text half by ``sqrt(text_weight)``. The
scaling looks arbitrary until you take the dot product of two such vectors:

    similarity(a, b) = (1 - w) * cos_fundamentals(a, b) + w * cos_text(a, b)

which is exactly what ``assumptions.ml.peers.text_weight`` is documented to
mean, and the concatenated vector still has unit length, so the whole embedding
is itself a cosine space and InfoNCE can be run on it directly. The blend is
therefore inside the model and inside the gradient: each tower is trained
knowing how much of the final similarity it is responsible for, rather than
being trained alone and averaged afterwards by a caller.

**What the model can learn that the features cannot say on their own.** Raw
cosine on standardised fundamentals treats all fifty features as equally
important, so two companies that happen to share a capital structure score as
highly as two that share a business. The towers are free to discover that
gross margin and revenue growth separate software peers while subscriber
metrics and capital intensity separate carriers, and to discard the columns
that separate nobody. Whether it actually does is the question this module
exists to answer, and the answer is reported against five baselines below
rather than asserted.

---

## The four traps, each of which is real, and one of which is worse than stated

**1. Popularity is a degenerate solution and it is very strong.** A handful of
companies appear in almost everybody's proxy table. A model can therefore score
respectably at retrieval by learning a single number per candidate, "how often
is this company named by anyone", and ignoring the query entirely. That model
has learned nothing about comparability and would return the same list for a
tower REIT as for a payments processor. ``popularity_ranking`` is the baseline
that catches it: it ranks every candidate by how many groups named it inside
the training window and never looks at the query. It is reported first because
it is the one that matters, and ``popularity_collapse`` additionally reports how
close the model's own orderings sit to it, so a model that beat the baseline by
imitating it is visible rather than merely suspected.

**2. Pair leakage runs through symmetry, through transitivity, and through
time, and the third is the one that costs the most.** ``peer_pairs`` emits both
(A, B) and (B, A), so a random split puts one fact on each side of it. Worse,
the groups are near-cliques: a proxy that names twenty companies asserts one
relationship, and splitting the hundreds of pairs it generates at random leaves
the test set a paraphrase of the training set. The split here is therefore by
DATE, on the proxy's filing date, with a 365-day embargo. The embargo is not
the usual forward-label embargo, because a compensation peer group is known on
the day it is filed and there is nothing in the future to overlap. It is there
for a different reason: a filer's next proxy repeats most of its previous one,
so without a year of separation the test question is whether the model can
recall a list it was shown eleven months earlier.

That still leaves the filer itself. ``evaluate_peer_encoder`` therefore reports
two numbers and names them, because they answer two different questions a bank
would actually ask:

    warm start   the query company had groups in the training window
    cold start   the query company appears in no training group at all

Cold start is the harder and more honest number and it is the one to quote for
a new coverage name. Warm start is the honest number for the case a bank is
usually in, which is re-running a model over companies it has covered for
years. Reporting only one of them would be choosing the flattering half of the
result.

**3. Vocabulary, projection and scaling are all fitted inside the fold.** The
TF-IDF vocabulary, the SVD basis and the feature mean and standard deviation
are fitted on the training window and applied to the test window through
``text.transform`` and a stored ``(mu, sd)``. Fitting any of the three on the
full sample leaves no trace whatsoever in the output and lifts every fold. This
module never calls ``fit_text_features`` on a corpus that postdates its fold,
which is enforced twice over: once here, and once by that function's own
refusal to fit a corpus containing a late document.

**4. Survivorship is in the candidate universe, not in the labels.** The labels
are dated and safe. The universe is not: a candidate list built from companies
listed today silently drops every company acquired mid-sample, and being
acquired is correlated with being a good peer, because a company gets bought
for looking like the companies around it. ``survivorship_report`` counts the
named peers that never enter the universe and, given ``tmt.precedents.deal_events``,
separates the ones that left through an acquisition from the ones that were
never in scope. The measured cost is in the pull request and it is larger than
it looks, because it lands entirely in recall.

---

## What is imputed, and the one place this module overrules a sibling

``features.py`` refuses to impute and says why: filling a missing gross margin
with zero asserts that the company broke even, and filling it with the
cross-sectional mean leaks the cross-section. Both objections are right, and a
neural network still cannot consume a NaN.

The resolution taken here is to impute to the TRAINING FOLD's mean, which after
in-fold standardisation is exactly zero, and to feed the encoder the
missing-share indicators the panel already emits beside the design matrix. That
answers the second objection outright, since the mean being borrowed belongs to
the training window and nothing else. It does not answer the first, and nothing
does: the encoder is told "this company sits at the training average on this
feature" alongside "this share of this feature group was unavailable", and it is
left to learn what the pair means together. Every row's imputation count is
carried on the model card.

## Determinism

Weight initialisation, dropout masks, batch order and every tie-break in every
baseline draw from one ``numpy.random.Generator`` seeded with
``assumptions.ml.random_seed``. Two fits from one seed produce bitwise identical
parameters and bitwise identical scores, and ``tests/ml/test_encoder.py`` asserts
that rather than hoping for it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Assumptions
from ..errors import ConfigError, MissingDataError, NotMeaningfulError
from .evaluation import (
    Fold,
    ablation,
    evaluate_ranking,
    walk_forward_folds,
)
from .features import MATRIX_COLUMNS, FeaturePanel, FeatureRow
from .nn import Adam, Dropout, L2Normalize, Layer, Linear, ReLU, Sequential, Tanh, train
from .nn import info_nce_in_batch
from .peer_labels import PeerGroup
from .protocol import EvalResult, ModelCard, spearman
from .text import TextCorpus, TextFeatures, fit_text_features, transform

__all__ = [
    "EMBARGO_DAYS",
    "PeerExample",
    "PeerDataset",
    "TwoTowerEncoder",
    "PeerEncoder",
    "build_dataset",
    "fit_peer_encoder",
    "evaluate_peer_encoder",
    "ablate_towers",
    "popularity_ranking",
    "popularity_collapse",
    "paired_lift",
    "survivorship_report",
    "load_peer_encoder",
]

# A filer's proxies land roughly a year apart and consecutive ones repeat most of
# the same peer table, so a training group filed eleven months before a test
# group is very nearly the test answer. A year of separation is the smallest
# embargo that guarantees no filer's adjacent proxies straddle a fold boundary.
# The usual reason for an embargo, a forward label window overlapping the test
# period, does not apply here: the label is public on the day it is filed.
EMBARGO_DAYS = 365

# Below this many training pairs a contrastive fit is memorising a few hundred
# relationships rather than learning a space. Reported rather than fatal at the
# model level, because the honest answer to a thin sample is a thin result with
# the sample size beside it, but a fold thinner than this is not fitted at all.
MIN_TRAIN_PAIRS = 200

# In-batch negatives assume the off-diagonal pairs really are negatives. At this
# batch size or below the assumption barely matters because the softmax has
# almost nothing to discriminate against, and the loss carries little gradient.
MIN_BATCH = 8


# --------------------------------------------------------------------------- #
# The labelled sample
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PeerExample:
    """One disclosed pair, stamped with the date it became public.

    ``as_of`` is the panel date the features were built through, and it is
    always at or before ``filed``. The two are kept apart because they can
    differ by months: the panel is built on a handful of dates and a proxy filed
    in April attaches to the panel date before it, so the features are strictly
    older than the assertion they are labelled with rather than contemporaneous
    with it. That is the conservative direction and it is the only one that is
    safe.
    """

    a: str
    b: str
    fiscal_year: int
    filed: date
    as_of: date
    group_key: tuple[str, int]


@dataclass
class PeerDataset:
    """Labelled pairs, the panel behind them, and everything that did not make it.

    ``dropped`` is the part a reader should look at first. A pair is dropped
    when either leg has no feature row or no business description at the pair's
    panel date, and the count of those is the coverage of the whole exercise. It
    is carried rather than logged because a recall number computed over a
    universe that quietly lost a third of the named peers is not a recall
    number.
    """

    examples: list[PeerExample]
    panel: FeaturePanel
    corpora: dict[date, TextCorpus]
    groups: list[PeerGroup]
    universe: list[str]
    panel_dates: list[date]
    dropped: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def dates(self) -> list[date]:
        """The filing date of each example, which is what the split runs on."""
        return [e.filed for e in self.examples]

    @property
    def queries(self) -> list[tuple[str, int]]:
        seen: dict[tuple[str, int], None] = {}
        for e in self.examples:
            seen.setdefault(e.group_key, None)
        return list(seen)

    def group_of(self, key: tuple[str, int]) -> PeerGroup:
        for g in self.groups:
            if (g.ticker, g.fiscal_year) == key:
                return g
        raise MissingDataError("peer group", ticker=key[0], hint=f"fiscal year {key[1]}")

    def rows(self) -> list[tuple[str, Any]]:
        filed = [e.filed for e in self.examples]
        return [
            ("Disclosed pairs", len(self.examples)),
            ("Distinct queries", len(self.queries)),
            ("Candidate universe", len(self.universe)),
            ("Panel dates", len(self.panel_dates)),
            ("Earliest proxy", min(filed) if filed else None),
            ("Latest proxy", max(filed) if filed else None),
            ("Pairs dropped", sum(self.dropped.values())),
        ]


def build_dataset(
    groups: Sequence[PeerGroup],
    panel: FeaturePanel,
    corpora: Mapping[date, TextCorpus],
    *,
    universe: Sequence[str] | None = None,
) -> PeerDataset:
    """Attach every usable disclosed group to the newest panel date before it.

    A group filed on 29 April 2026 is attached to the panel date at or before
    that day, never after it. Attaching forward would let a ranking be computed
    from figures that were filed in response to the very year being scored, and
    the direction of that error is always flattering.

    A group whose filer has no feature row at its panel date, or no business
    description there, contributes nothing: the query cannot be encoded, so
    there is no ranking to score. A peer in the same position is dropped from
    the training pairs but KEPT in the relevance set, because a peer the
    universe could not represent is a recall failure of the universe and hiding
    it inside the ranking would move the blame.
    """
    if not panel.rows:
        raise ConfigError("the feature panel is empty, so no example can be encoded")
    if not corpora:
        raise ConfigError(
            "no text corpora were supplied. Pass a mapping of panel date to the "
            "TextCorpus built as of that date; the text tower needs one corpus "
            "per date so its vocabulary can be fitted inside each fold"
        )

    panel_dates = sorted({r.as_of for r in panel.rows})
    missing_corpus = [d for d in panel_dates if d not in corpora]
    if missing_corpus:
        raise ConfigError(
            f"{len(missing_corpus)} panel date(s) have no corpus: "
            f"{', '.join(str(d) for d in missing_corpus)}. A date the text tower "
            "cannot see is a date at which half the model does not exist"
        )

    ok: dict[tuple[str, date], FeatureRow] = {
        (r.ticker, r.as_of): r for r in panel.rows if r.ok
    }
    texted: dict[date, set[str]] = {
        d: set(corpora[d].tickers) for d in panel_dates
    }
    encodable = {
        key for key in ok if key[0] in texted[key[1]]
    }

    pool = sorted({t for t, _ in encodable}) if universe is None else [
        str(t).upper() for t in universe
    ]

    examples: list[PeerExample] = []
    kept_groups: list[PeerGroup] = []
    dropped = {
        "group not usable": 0,
        "group filed before every panel date": 0,
        "filer not encodable at its panel date": 0,
        "peer not in the candidate universe": 0,
        "peer not encodable at the panel date": 0,
        "self-reference": 0,
    }

    for g in sorted(groups, key=lambda g: (g.filed or date.max, g.ticker)):
        if not g.usable or g.filed is None or g.fiscal_year is None:
            dropped["group not usable"] += 1
            continue
        earlier = [d for d in panel_dates if d <= g.filed]
        if not earlier:
            dropped["group filed before every panel date"] += 1
            continue
        as_of = earlier[-1]
        if (g.ticker, as_of) not in encodable:
            dropped["filer not encodable at its panel date"] += 1
            continue

        key = (g.ticker, g.fiscal_year)
        made_any = False
        for peer in g.peers:
            if peer == g.ticker:
                dropped["self-reference"] += 1
                continue
            if peer not in pool:
                dropped["peer not in the candidate universe"] += 1
                continue
            if (peer, as_of) not in encodable:
                dropped["peer not encodable at the panel date"] += 1
                continue
            examples.append(
                PeerExample(
                    a=g.ticker,
                    b=peer,
                    fiscal_year=g.fiscal_year,
                    filed=g.filed,
                    as_of=as_of,
                    group_key=key,
                )
            )
            made_any = True
        if made_any:
            kept_groups.append(g)

    notes = [
        f"{len(examples)} directed training pairs from {len(kept_groups)} disclosed "
        f"groups over {len(panel_dates)} panel dates."
    ]
    if dropped["peer not in the candidate universe"]:
        notes.append(
            f"{dropped['peer not in the candidate universe']} named peers never "
            "entered the candidate universe. They stay in the relevance set, so "
            "the loss lands in recall where it belongs; see survivorship_report "
            "for how many of them left through an acquisition."
        )

    return PeerDataset(
        examples=examples,
        panel=panel,
        corpora=dict(corpora),
        groups=kept_groups,
        universe=sorted(pool),
        panel_dates=panel_dates,
        dropped=dropped,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class TwoTowerEncoder(Layer):
    """Two encoders into one cosine space, blended by ``text_weight``.

    The input is one matrix whose first ``n_fundamental`` columns are the
    standardised feature vector and whose remaining ``n_text`` columns are the
    LSA vector, so a batch is a single array and a single forward pass, which is
    what ``nn.train`` needs: a layer caches exactly one forward for its backward
    and an encoder called twice before its backward computes the wrong gradient.

    The output is the two unit vectors side by side, scaled so that the
    concatenation is itself a unit vector and its dot product with another is
    the weighted average of the two cosines. Setting ``text_weight`` to zero or
    one removes the other tower entirely rather than multiplying it by zero: a
    tower scaled to nothing still holds parameters that receive no gradient, and
    a model card listing parameters that provably never moved is a model card
    that misleads about the size of the model.
    """

    def __init__(
        self,
        n_fundamental: int,
        n_text: int,
        *,
        dim: int,
        hidden: Sequence[int],
        text_weight: float,
        dropout: float,
        rng: np.random.Generator,
        activation: str = "tanh",
    ) -> None:
        w = float(text_weight)
        if not 0.0 <= w <= 1.0:
            raise ConfigError(f"text_weight must lie in [0, 1], got {text_weight}")
        if dim < 2:
            raise ConfigError(f"the embedding needs at least two dimensions, got {dim}")
        use_fund = w < 1.0
        use_text = w > 0.0
        if use_fund and n_fundamental < 1:
            raise ConfigError(
                "the fundamentals tower carries weight but was given no feature "
                "columns; set text_weight to 1.0 to run text alone"
            )
        if use_text and n_text < 1:
            raise ConfigError(
                "the text tower carries weight but was given no text columns; set "
                "text_weight to 0.0 to run fundamentals alone"
            )

        self.n_fundamental = int(n_fundamental)
        self.n_text = int(n_text)
        self.dim = int(dim)
        self.text_weight = w
        self.fund_scale = float(np.sqrt(1.0 - w))
        self.text_scale = float(np.sqrt(w))
        self.activation = str(activation).lower()
        self.fundamentals = (
            _tower(self.n_fundamental, dim, hidden, dropout, rng, self.activation)
            if use_fund
            else None
        )
        self.text = (
            _tower(self.n_text, dim, hidden, dropout, rng, self.activation)
            if use_text
            else None
        )
        self._split: int | None = None

    @property
    def width(self) -> int:
        """Columns of the embedding this encoder returns."""
        return self.dim * ((self.fundamentals is not None) + (self.text is not None))

    @property
    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.params()))

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise ConfigError(
                f"TwoTowerEncoder input must be two dimensional, got {arr.shape}"
            )
        expected = self.n_fundamental + self.n_text
        if arr.shape[1] != expected:
            raise ConfigError(
                f"TwoTowerEncoder was given {arr.shape[1]} columns where "
                f"{self.n_fundamental} fundamental plus {self.n_text} text "
                f"columns were expected"
            )
        parts: list[np.ndarray] = []
        if self.fundamentals is not None:
            parts.append(self.fundamentals.forward(arr[:, : self.n_fundamental]) * self.fund_scale)
        if self.text is not None:
            parts.append(self.text.forward(arr[:, self.n_fundamental :]) * self.text_scale)
        self._split = parts[0].shape[1] if len(parts) > 1 else None
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        g = np.asarray(grad_out, dtype=float)
        pieces: list[np.ndarray] = []
        if self.fundamentals is not None and self.text is not None:
            gf, gt = g[:, : self._split], g[:, self._split :]
            pieces.append(self.fundamentals.backward(gf * self.fund_scale))
            pieces.append(self.text.backward(gt * self.text_scale))
        elif self.fundamentals is not None:
            pieces.append(self.fundamentals.backward(g * self.fund_scale))
        else:
            pieces.append(self.text.backward(g * self.text_scale))
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)

    def params(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for tower in (self.fundamentals, self.text):
            if tower is not None:
                out.extend(tower.params())
        return out

    def grads(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for tower in (self.fundamentals, self.text):
            if tower is not None:
                out.extend(tower.grads())
        return out

    def train(self, mode: bool = True) -> "TwoTowerEncoder":
        self.training = bool(mode)
        for tower in (self.fundamentals, self.text):
            if tower is not None:
                tower.train(mode)
        return self

    def reseed(self, rng: np.random.Generator) -> None:
        """Put every dropout mask back to a known state, for a gradient check."""
        for tower in (self.fundamentals, self.text):
            if tower is None:
                continue
            for layer in tower.layers:
                if isinstance(layer, Dropout):
                    layer.reseed(np.random.default_rng(rng.integers(0, 2**31 - 1)))


def _tower(
    n_in: int,
    dim: int,
    hidden: Sequence[int],
    dropout: float,
    rng: np.random.Generator,
    activation: str = "tanh",
) -> Sequential:
    """One modality's encoder: a small MLP onto the unit sphere.

    **Tanh rather than ReLU, and the reason is the layer above.** A ReLU layer
    can output exactly zero across an entire row, and with the output layer's
    bias initialised at zero, as it must be, that row then arrives at
    ``L2Normalize`` as the zero vector. The zero vector has no direction. The
    eps guard keeps the arithmetic finite and what comes out has cosine zero
    with every other company, which reads as "resembles nothing" when the truth
    is "was not measured": exactly the outcome ``text.py`` refuses to return
    when a document projects onto the origin. It is not hypothetical. At He
    initialisation on a batch of eight standardised rows it happens to roughly
    one row in eight, and the gradient at that point is divided by the eps
    rather than by a norm, so the fit starts by taking one enormous and
    meaningless step. Tanh is bounded away from saturating both ways and cannot
    produce an identically zero row except at a point of measure zero.

    The case for ReLU is real and is not dismissed: it is cheaper, it does not
    saturate, and on a deep stack it trains faster. These towers are one hidden
    layer over standardised inputs, where none of those advantages is worth a
    representation that can silently collapse, so ``activation="relu"`` is
    available and is not the default.

    No layer normalisation either way. Both inputs arrive standardised, the
    fundamentals by an in-fold mean and standard deviation and the text by the
    L2 normalisation the LSA projection already applies, so there is nothing for
    a normalising layer to fix and it would only add parameters this sample size
    cannot pay for.
    """
    key = str(activation).lower()
    if key not in {"tanh", "relu"}:
        raise ConfigError(
            f"unknown activation {activation!r}: use 'tanh' or 'relu'"
        )
    layers: list[Layer] = []
    width = int(n_in)
    for size in hidden:
        layers.append(
            Linear(width, int(size), rng=rng, init="he" if key == "relu" else "xavier")
        )
        layers.append(ReLU() if key == "relu" else Tanh())
        if dropout > 0.0:
            layers.append(Dropout(dropout, rng=rng))
        width = int(size)
    layers.append(Linear(width, int(dim), rng=rng, init="xavier"))
    layers.append(L2Normalize())
    return Sequential(*layers)


# --------------------------------------------------------------------------- #
# Turning the panel and the corpus into tower inputs
# --------------------------------------------------------------------------- #


@dataclass
class _Space:
    """One fold's fitted representation of every company at every panel date.

    Everything here was fitted on the training window and nothing on the test
    window, which is the whole reason this object exists instead of a few loose
    arrays: keeping the transform together with the matrix it produced makes it
    hard to apply one fold's scaling to another fold's rows by accident.

    Both blocks are always built, even when the encoder is about to use only one
    of them. The baselines the encoder has to beat include a raw cosine on each
    block separately, and refitting the discarded half to score a baseline would
    refit it on a different window from the one the model saw.
    """

    index: dict[tuple[str, date], int]
    fundamentals: np.ndarray
    lsa: np.ndarray
    tfidf: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    text: TextFeatures | None
    n_imputed: int
    fit_date: date
    notes: list[str] = field(default_factory=list)

    @property
    def n_fundamental(self) -> int:
        return int(self.fundamentals.shape[1])

    @property
    def n_text(self) -> int:
        return int(self.lsa.shape[1])

    def has(self, ticker: str, as_of: date) -> bool:
        return (ticker, as_of) in self.index

    def inputs(self, *, use_fundamentals: bool, use_text: bool) -> np.ndarray:
        """The encoder's design matrix for the towers that are switched on."""
        blocks = []
        if use_fundamentals:
            blocks.append(self.fundamentals)
        if use_text:
            blocks.append(self.lsa)
        if not blocks:
            raise ConfigError("an encoder with neither tower has no input")
        return np.hstack(blocks) if len(blocks) > 1 else blocks[0]

    def rows_of(self, matrix: np.ndarray, keys: Sequence[tuple[str, date]]) -> np.ndarray:
        return matrix[[self.index[k] for k in keys]]


def _fit_space(
    dataset: PeerDataset,
    train_through: date,
    assumptions: Assumptions,
    *,
    text_dim: int,
    min_df: int,
    max_df: float,
) -> _Space:
    """Fit the standardisation and the vocabulary on the training window only.

    ``train_through`` is the last date whose data the fold is allowed to have
    seen. Three things are fitted from it and nothing else: the feature mean and
    standard deviation, the TF-IDF vocabulary with its inverse document
    frequencies, and the SVD basis. Every later panel date is then TRANSFORMED
    through them, which is why a 2019 fold cannot see the word "generative" and
    should not be able to.
    """
    panel = dataset.panel
    fit_mask = np.asarray([r.as_of <= train_through for r in panel.rows], dtype=bool)
    if not fit_mask.any():
        raise ConfigError(
            f"no panel row is dated on or before {train_through}, so this fold "
            "has nothing to fit its standardisation on"
        )
    Z, mu, sd = panel.standardize(fit_mask, include_missing_indicators=True)

    # Imputation to zero AFTER standardising is imputation to the training-window
    # mean, which is the only mean that is not a leak. The missing-share columns
    # the panel emits travel beside it so the encoder can tell an imputed
    # average from a measured one.
    n_imputed = int(np.count_nonzero(~np.isfinite(Z)))
    Z = np.where(np.isfinite(Z), Z, 0.0)

    fundamental_by_key: dict[tuple[str, date], np.ndarray] = {}
    for row, vector in zip(panel.rows, Z):
        if row.ok:
            fundamental_by_key[(row.ticker, row.as_of)] = vector

    corpus_dates = [d for d in dataset.panel_dates if d <= train_through]
    if not corpus_dates:
        raise ConfigError(
            f"no corpus is dated on or before {train_through}, so the text tower "
            "has no vocabulary it is allowed to fit"
        )
    fit_date = corpus_dates[-1]
    fit_corpus, text_features, excluded = _fit_text(
        dataset.corpora[fit_date], assumptions, fit_date, text_dim, min_df, max_df
    )
    notes = [
        f"text vocabulary fitted on the {len(fit_corpus.tickers)} documents on file "
        f"at {fit_date}: {text_features.vocabulary_size:,} terms reduced to "
        f"{text_features.dim} dimensions holding "
        f"{text_features.explained_variance:.1%} of the variance"
    ]
    if excluded:
        notes.append(
            f"{len(excluded)} document(s) were excluded from the fit because they "
            f"share no surviving vocabulary with the rest of the corpus: "
            f"{', '.join(excluded)}"
        )

    lsa_by_key: dict[tuple[str, date], np.ndarray] = {}
    tfidf_by_key: dict[tuple[str, date], np.ndarray] = {}
    unprojectable: list[str] = []
    for as_of in dataset.panel_dates:
        corpus = dataset.corpora[as_of]
        if not corpus.tickers:
            continue
        # The sparse term vectors behind the projection, which is the baseline a
        # learned text tower has to beat. TfidfVectorizer L2-normalises its rows,
        # so a dot product here is already a cosine.
        prepared = _scrubbed(corpus)
        dense = np.asarray(
            text_features.vectorizer.transform(prepared).todense(), dtype=float
        )
        for ticker, document, term_vector in zip(corpus.tickers, corpus.documents, dense):
            try:
                vector = transform(
                    text_features,
                    [document],
                    names=[corpus.entity_names.get(ticker)],
                    tickers=[ticker],
                )[0]
            except MissingDataError:
                # This document shares no surviving vocabulary with the training
                # window's corpus, so it projects onto the origin. text.transform
                # refuses to return that, and it is right to: a zero vector has
                # cosine zero with everything, which reads as "resembles nothing"
                # when the truth is "was not measured". The company is dropped
                # from this fold's space and named, not given a zero.
                unprojectable.append(f"{ticker} at {as_of}")
                continue
            lsa_by_key[(ticker, as_of)] = vector
            tfidf_by_key[(ticker, as_of)] = term_vector

    if unprojectable:
        notes.append(
            f"{len(unprojectable)} company-date(s) share no vocabulary with the "
            f"training corpus and were dropped from this fold rather than given a "
            f"zero vector: {', '.join(sorted(unprojectable)[:6])}"
            + (" and others" if len(unprojectable) > 6 else "")
        )

    keys = sorted(k for k in fundamental_by_key if k in lsa_by_key)
    if not keys:
        raise ConfigError(
            "no company-date has both a feature row and a business description, "
            "so neither tower has an input"
        )

    return _Space(
        index={k: i for i, k in enumerate(keys)},
        fundamentals=np.vstack([fundamental_by_key[k] for k in keys]),
        lsa=np.vstack([lsa_by_key[k] for k in keys]),
        tfidf=np.vstack([tfidf_by_key[k] for k in keys]),
        mu=mu,
        sd=sd,
        text=text_features,
        n_imputed=n_imputed,
        fit_date=fit_date,
        notes=notes,
    )


def _fit_text(
    corpus: TextCorpus,
    assumptions: Assumptions,
    fit_date: date,
    text_dim: int,
    min_df: int,
    max_df: float,
) -> tuple[TextCorpus, TextFeatures, list[str]]:
    """Fit the vocabulary, dropping any document that projects onto the origin.

    ``fit_text_features`` refuses a corpus containing a document that survives
    pruning with no terms left, and it is right to refuse: the alternative is a
    zero row whose cosine against every company is zero, which reads as
    "resembles nothing" rather than "was not measured". It refuses the whole
    fit, though, and on a real universe one filer in a hundred is enough to
    stop a fold.

    Alphabet is the case that forced this. ``nlp.sections`` returns twenty-seven
    characters for its Item 1, because Alphabet's 10-K answers Item 1 by cross
    reference rather than inline, and the splitter flags the section as too
    short to be prose rather than pretending otherwise. Twenty-seven characters
    share no vocabulary with anything, and without this the whole walk-forward
    stops on one filer's formatting convention.

    So the offending documents are removed and NAMED, one at a time, rather than
    given a zero. A company that is not in the fitted corpus is simply not a
    candidate at that date, which is an honest coverage gap and shows up in
    recall.
    """
    working = corpus
    excluded: list[str] = []
    for _attempt in range(len(corpus.tickers)):
        try:
            features = fit_text_features(
                working,
                assumptions,
                dim=text_dim,
                fit_through=fit_date,
                min_df=min_df,
                max_df=max_df,
                cache=False,
            )
        except MissingDataError as exc:
            dead = exc.ticker
            if dead is None or dead not in working.tickers:
                raise
            excluded.append(dead)
            keep = [t for t in working.tickers if t != dead]
            working = TextCorpus(
                tickers=keep,
                documents=[working.document(t) for t in keep],
                as_of_by_ticker={t: working.as_of_by_ticker[t] for t in keep},
                source_accessions={
                    t: working.source_accessions.get(t, "") for t in keep
                },
                notes=list(working.notes),
                entity_names={
                    t: n for t, n in working.entity_names.items() if t in set(keep)
                },
            )
            continue
        return working, features, excluded
    raise ConfigError(
        f"every document in the corpus at {fit_date} projects onto the origin, so "
        "there is no text space to fit"
    )


def _scrubbed(corpus: TextCorpus) -> list[str]:
    """Documents with each filer's own name and ticker removed, as the fit did.

    ``text.scrub`` is applied here rather than left to ``transform`` because the
    raw term vectors are wanted for the text baseline as well as the projection,
    and the two have to be built from the identical preparation or the baseline
    is scoring a different document from the model.
    """
    from .text import scrub

    return [
        scrub(doc, name=corpus.entity_names.get(t), ticker=t)
        for t, doc in zip(corpus.tickers, corpus.documents)
    ]


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def _batches(
    examples: Sequence[PeerExample],
    space: _Space,
    X: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, None]]:
    """Pairs packed into batches that do not contain their own false negatives.

    ``info_nce_in_batch`` treats every off-diagonal pair in a batch as a
    negative, which is what makes contrastive training cheap and is also an
    assumption that this dataset violates in two specific ways. A proxy names
    twenty companies, so twenty pairs from one table put twenty genuine peers of
    the same filer in one batch and the loss spends its gradient pushing them
    apart. And a company named by several filers can appear as the positive of
    two rows at once, where the softmax is being asked to pick between two
    correct answers.

    Both are fixed by the sampler rather than by the loss, exactly as
    ``nn.info_nce_in_batch`` says they should be: a batch takes at most one pair
    from any one disclosed group and uses any one company as its positive at
    most once.

    There is a cost and it is worth naming, because it is also a partial defence
    against the popularity trap. A company named by eighty filers can appear as
    a positive at most once per batch, so across an epoch it is seen once per
    batch rather than eighty times. The sampler therefore flattens the
    popularity distribution the model is trained on, which is a thumb on the
    scale against the degenerate solution and is declared here rather than
    discovered later.
    """
    order = rng.permutation(len(examples))
    pending = [examples[int(i)] for i in order]
    batches: list[tuple[np.ndarray, None]] = []

    while pending:
        chosen: list[PeerExample] = []
        used_groups: set[tuple[str, int]] = set()
        used_positives: set[str] = set()
        leftover: list[PeerExample] = []
        for example in pending:
            if (
                len(chosen) < batch_size
                and example.group_key not in used_groups
                and example.b not in used_positives
            ):
                chosen.append(example)
                used_groups.add(example.group_key)
                used_positives.add(example.b)
            else:
                leftover.append(example)
        if len(chosen) < MIN_BATCH:
            break
        anchors = space.rows_of(X, [(e.a, e.as_of) for e in chosen])
        positives = space.rows_of(X, [(e.b, e.as_of) for e in chosen])
        batches.append((np.vstack([anchors, positives]), None))
        pending = leftover

    if not batches:
        raise NotMeaningfulError(
            f"no batch of at least {MIN_BATCH} pairs could be packed from "
            f"{len(examples)} examples without putting two pairs of one disclosed "
            "group, or two copies of one positive company, side by side. The "
            "sample is too narrow for in-batch negatives"
        )
    return batches


def _contrastive_loss(temperature: float):
    """Split a stacked forward pass down the middle and score the two halves.

    The stacking is the pattern ``nn.train`` documents for exactly this model.
    The encoder sees anchors and positives as one array of 2B rows in one
    forward pass, so it caches one input and its backward is well defined; this
    closure splits the 2B embeddings into the two halves the loss compares and
    stacks the two gradients back into the (2B, D) array the backward pass
    expects.
    """

    def loss_fn(prediction: np.ndarray, _targets: Any) -> tuple[float, np.ndarray]:
        half = prediction.shape[0] // 2
        anchors, positives = prediction[:half], prediction[half:]
        loss, (ga, gp) = info_nce_in_batch(anchors, positives, temperature)
        return loss, np.vstack([ga, gp])

    return loss_fn


# --------------------------------------------------------------------------- #
# The fitted model
# --------------------------------------------------------------------------- #


@dataclass
class PeerEncoder:
    """A fitted peer encoder and everything needed to defend a ranking it produces.

    ``embeddings`` holds every company in the universe as it stood at
    ``fit_date``, already unit length, so a neighbour query is one matrix
    multiply. ``card`` carries the honest evaluation beside it, and
    ``card.evaluation.verdict()`` is the sentence to read first: when the model
    lost to its baseline it says so in those words and the right answer is to
    use the baseline.
    """

    tickers: list[str]
    embeddings: np.ndarray
    fit_date: date
    text_weight: float
    dim: int
    encoder: TwoTowerEncoder | None
    mu: np.ndarray
    sd: np.ndarray
    text: TextFeatures | None
    log_market_cap: dict[str, float]
    card: ModelCard
    assumptions_peers: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def _index(self, ticker: str) -> int:
        try:
            return self.tickers.index(str(ticker).upper())
        except ValueError:
            raise MissingDataError(
                "peer embedding",
                ticker=ticker,
                hint=(
                    "this company was not in the fitted universe. The encoder can "
                    "only place a company it has both a feature row and a business "
                    "description for at the fit date"
                ),
            ) from None

    def similarity(self, a: str, b: str) -> float:
        """Cosine between two fitted companies, which is what the loss optimised."""
        dot = float(self.embeddings[self._index(a)] @ self.embeddings[self._index(b)])
        return float(np.clip(dot, -1.0, 1.0))

    def neighbours(
        self,
        ticker: str,
        k: int | None = None,
        *,
        apply_size_gate: bool = True,
    ) -> list[tuple[str, float]]:
        """The k most comparable companies, highest similarity first, excluding self.

        ``apply_size_gate`` is on by default and it is the one place this class
        does something the evaluation deliberately does not. A banker does not
        put a 2bn company in a 200bn company's comp set however similar the
        prose, so ``min_market_cap`` and ``max_size_ratio`` from
        ``assumptions.ml.peers`` gate the candidate list here.

        The evaluation runs UNGATED, and the difference is not an inconsistency.
        A size gate lifts every ranker's score at once, because a compensation
        peer group is selected inside a revenue and market capitalisation band
        to begin with, so scoring through one measures the band rather than the
        model. The gate belongs in the product and not in the measurement, and
        ``evaluate_peer_encoder`` reports the gated score separately so the size
        of that free lift is on the record.
        """
        want = int(self.assumptions_peers.get("n_peers", 8) if k is None else k)
        if want < 1:
            raise ConfigError(f"k must be at least 1, got {want}")
        i = self._index(ticker)
        sims = self.embeddings @ self.embeddings[i]
        order = np.argsort(-sims, kind="stable")
        out: list[tuple[str, float]] = []
        for j in order:
            j = int(j)
            if j == i:
                continue
            if apply_size_gate and not self._passes_size_gate(self.tickers[i], self.tickers[j]):
                continue
            out.append((self.tickers[j], float(np.clip(sims[j], -1.0, 1.0))))
            if len(out) >= want:
                break
        return out

    def _passes_size_gate(self, target: str, candidate: str) -> bool:
        """Both bands, worked in logs because the panel carries sizes that way.

        ``scale_log_market_cap`` is the natural log of market capitalisation in
        USD millions, and it has been through the cross-sectional winsorization,
        so a company at the very top or bottom of a date's distribution has been
        pulled to the boundary before this comparison sees it. That widens the
        gate slightly at the extremes and it is the price of using the same
        numbers the model was fitted on rather than a second, unwinsorized copy.
        """
        floor = float(self.assumptions_peers.get("min_market_cap", 0.0))
        ratio = float(self.assumptions_peers.get("max_size_ratio", 0.0))
        here = self.log_market_cap.get(target)
        there = self.log_market_cap.get(candidate)
        if here is None or there is None:
            # Size unknown is not size out of band. A candidate whose market
            # capitalisation could not be sourced is passed through and the gap
            # is visible in the panel's missing-share column rather than being
            # silently resolved against it.
            return True
        if floor > 0.0 and there < np.log(floor):
            return False
        if ratio > 0.0 and abs(here - there) > np.log(ratio):
            return False
        return True

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.embeddings,
            index=pd.Index(self.tickers, name="ticker"),
            columns=[f"peer_{i:03d}" for i in range(self.embeddings.shape[1])],
        )

    def rows(self) -> list[tuple[str, Any]]:
        return [
            ("Companies", len(self.tickers)),
            ("Embedding width", int(self.embeddings.shape[1])),
            ("Text weight", self.text_weight),
            ("Fitted through", str(self.fit_date)),
            ("Parameters", self.encoder.n_parameters if self.encoder else 0),
        ]

    # -- persistence ------------------------------------------------------- #

    def save(self, path: str | Path) -> Path:
        """Write the fit to disk, weights, transforms, card and all.

        joblib rather than JSON because the fitted vectorizer and SVD basis are
        scikit-learn objects with no honest text representation, and splitting
        the model across two formats would let the two halves drift apart.
        """
        import joblib

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "tickers": self.tickers,
            "embeddings": self.embeddings,
            "fit_date": self.fit_date,
            "text_weight": self.text_weight,
            "dim": self.dim,
            "mu": self.mu,
            "sd": self.sd,
            "text": self.text,
            "log_market_cap": self.log_market_cap,
            "card": self.card,
            "assumptions_peers": self.assumptions_peers,
            "notes": self.notes,
            "weights": None
            if self.encoder is None
            else [p.copy() for p in self.encoder.params()],
            "shape": None
            if self.encoder is None
            else {
                "n_fundamental": self.encoder.n_fundamental,
                "n_text": self.encoder.n_text,
                "dim": self.encoder.dim,
                "hidden": [
                    layer.out_dim
                    for tower in (self.encoder.fundamentals, self.encoder.text)
                    if tower is not None
                    for layer in tower.layers
                    if isinstance(layer, Linear)
                ][:-1],
            },
        }
        joblib.dump(payload, target, compress=3)
        return target


def cache_path(assumptions: Assumptions | None, name: str = "peer_encoder") -> Path:
    """Where a fitted encoder is kept, from ``assumptions.ml.cache_dir``."""
    configured = assumptions.ml.cache_dir if assumptions is not None else None
    root = Path(configured).expanduser() if configured else Path.home() / ".techval" / "ml"
    return root / f"{name}.joblib"


def load_peer_encoder(path: str | Path) -> PeerEncoder:
    """Read back a saved fit, or say what is wrong with the file.

    The towers are not rebuilt. Everything a caller does with a loaded encoder,
    ranking neighbours and reading the card, runs off the stored embeddings, and
    rebuilding a network to answer a question a matrix multiply already answers
    would be a second place for the architecture to drift out of step with the
    weights. ``encoder`` is None on a loaded model and its parameter count lives
    on the card.
    """
    import joblib

    payload = joblib.load(Path(path))
    if payload.get("schema") != 1:
        raise ConfigError(
            f"{path} was written by a different version of this module "
            f"(schema {payload.get('schema')!r} against 1). Refit rather than "
            "reading weights whose meaning has changed"
        )
    return PeerEncoder(
        tickers=list(payload["tickers"]),
        embeddings=np.asarray(payload["embeddings"], dtype=float),
        fit_date=payload["fit_date"],
        text_weight=float(payload["text_weight"]),
        dim=int(payload["dim"]),
        encoder=None,
        mu=np.asarray(payload["mu"], dtype=float),
        sd=np.asarray(payload["sd"], dtype=float),
        text=payload["text"],
        log_market_cap=dict(payload["log_market_cap"]),
        card=payload["card"],
        assumptions_peers=dict(payload.get("assumptions_peers", {})),
        notes=list(payload.get("notes", [])) + [f"loaded from {path}"],
    )


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def fit_peer_encoder(
    dataset: PeerDataset,
    assumptions: Assumptions,
    *,
    fit_through: date | None = None,
    dim: int = 64,
    hidden: Sequence[int] = (128,),
    text_dim: int = 128,
    text_weight: float | None = None,
    dropout: float = 0.1,
    temperature: float = 0.07,
    batch_size: int = 128,
    epochs: int = 60,
    patience: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_share: float = 0.15,
    min_df: int = 2,
    max_df: float = 0.85,
    evaluation: EvalResult | None = None,
) -> PeerEncoder:
    """Fit both towers on every pair disclosed on or before ``fit_through``.

    This is the production fit and it is deliberately NOT where the score comes
    from. It trains on everything available, which is the right thing for a
    model about to rank live comparables and the wrong thing to measure, so the
    ``EvalResult`` on the card is the walk-forward one computed by
    ``evaluate_peer_encoder`` and passed in. Fitting and scoring the same rows
    would produce a number in the high nineties that means nothing, and keeping
    the two functions apart is what stops that happening by accident.

    ``validation_share`` holds out the most recent slice of the training pairs,
    by date, for early stopping. By date rather than at random for the reason
    the whole module exists: a random holdout of pairs drawn from the same proxy
    tables as the training set will stop improving long after the model has
    started memorising them.

    ``text_weight`` defaults to ``assumptions.ml.peers.text_weight``. Zero fits
    the fundamentals tower alone and one fits the text tower alone, which is
    what ``ablate_towers`` uses to say which tower carries the signal.
    """
    if not dataset.examples:
        raise ConfigError("the dataset holds no labelled pairs, so there is nothing to fit")
    weight = (
        float(assumptions.ml.peers.text_weight) if text_weight is None else float(text_weight)
    )
    seed = int(assumptions.ml.random_seed)
    rng = np.random.default_rng(seed)

    cut = fit_through or max(e.filed for e in dataset.examples)
    usable = [e for e in dataset.examples if e.filed <= cut]
    if len(usable) < MIN_TRAIN_PAIRS:
        raise NotMeaningfulError(
            f"{len(usable)} training pairs on or before {cut} is below the floor of "
            f"{MIN_TRAIN_PAIRS}. A contrastive fit on fewer than that is memorising "
            "a few hundred disclosed relationships, not learning a space"
        )

    space = _fit_space(
        dataset, cut, assumptions, text_dim=text_dim, min_df=min_df, max_df=max_df
    )
    encoder, X, history, train_pairs = _train_encoder(
        dataset,
        space,
        [e for e in usable if space.has(e.a, e.as_of) and space.has(e.b, e.as_of)],
        text_weight=weight,
        dim=dim,
        hidden=hidden,
        dropout=dropout,
        temperature=temperature,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        validation_share=validation_share,
        seed=seed,
        rng=rng,
    )

    embed_date = max(d for _, d in space.index)
    present_tickers = [t for t in dataset.universe if space.has(t, embed_date)]
    embeddings = encoder.forward(
        space.rows_of(X, [(t, embed_date) for t in present_tickers])
    )

    log_cap = _log_sizes(dataset.panel, embed_date)
    card = ModelCard(
        name="peer encoder (two-tower contrastive, InfoNCE)",
        task=(
            "rank a candidate universe by comparability to a target company, "
            "trained on compensation peer groups the filers disclosed themselves"
        ),
        trained_through=cut,
        n_train=len(train_pairs),
        features=(list(MATRIX_COLUMNS) if weight < 1.0 else [])
        + ([f"text_lsa_{i:03d}" for i in range(space.n_text)] if weight > 0.0 else []),
        hyperparameters={
            "dim": dim,
            "hidden": list(hidden),
            "text_dim": space.n_text,
            "text_weight": weight,
            "dropout": dropout,
            "temperature": temperature,
            "batch_size": batch_size,
            "epochs_run": history.epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "random_seed": seed,
            "parameters": encoder.n_parameters,
        },
        evaluation=evaluation,
        limitations=[
            "The labels are compensation peer groups, not trading comparables. A "
            "committee picks partly for competition for executive talent, so the "
            "label skews toward the filer's own labour market and toward companies "
            "of similar size, and the model inherits both skews.",
            f"{len(dataset.groups)} disclosed groups over "
            f"{len(dataset.panel_dates)} dates is a small sample by any standard. "
            "Read every score beside the across-query dispersion.",
            f"{space.n_imputed:,} standardised feature cells were imputed to the "
            "training-window mean of zero. The missing-share indicators travel "
            "beside them but an imputed average is not a measurement.",
            "The candidate universe is the seed universe, so a company acquired "
            "mid-sample is absent from it. See survivorship_report for the size "
            "of that hole and note that it lands entirely in recall.",
        ],
        notes=list(history.notes) + list(space.notes) + list(dataset.notes),
    )

    return PeerEncoder(
        tickers=present_tickers,
        embeddings=embeddings,
        fit_date=embed_date,
        text_weight=weight,
        dim=dim,
        encoder=encoder,
        mu=space.mu,
        sd=space.sd,
        text=space.text,
        log_market_cap=log_cap,
        card=card,
        assumptions_peers={
            "n_peers": assumptions.ml.peers.n_peers,
            "text_weight": assumptions.ml.peers.text_weight,
            "min_market_cap": assumptions.ml.peers.min_market_cap,
            "max_size_ratio": assumptions.ml.peers.max_size_ratio,
        },
        notes=list(space.notes),
    )


def _train_encoder(
    dataset: PeerDataset,
    space: _Space,
    examples: Sequence[PeerExample],
    *,
    text_weight: float,
    dim: int,
    hidden: Sequence[int],
    dropout: float,
    temperature: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    validation_share: float,
    seed: int,
    rng: np.random.Generator,
):
    """Build the towers, pack the batches and run the fit. Shared by fit and evaluate.

    The validation cut is by DATE, so the held-out pairs are the most recent
    ones rather than a random sample of the same proxy tables. A random holdout
    here would stop improving long after the model had started memorising the
    tables it was drawn from, and early stopping would fire at the wrong epoch
    in the direction that flatters the fit.
    """
    use_fund = text_weight < 1.0
    use_text = text_weight > 0.0
    X = space.inputs(use_fundamentals=use_fund, use_text=use_text)

    by_date = sorted(examples, key=lambda e: (e.filed, e.a, e.b))
    if not by_date:
        raise NotMeaningfulError(
            "no training pair survives the panel and the corpus at its own date, "
            "so there is nothing to fit"
        )
    split = max(1, int(len(by_date) * (1.0 - float(validation_share))))
    boundary = by_date[split - 1].filed
    train_pairs = [e for e in by_date if e.filed <= boundary]
    val_pairs = [e for e in by_date if e.filed > boundary]

    encoder = TwoTowerEncoder(
        space.n_fundamental if use_fund else 0,
        space.n_text if use_text else 0,
        dim=dim,
        hidden=hidden,
        text_weight=text_weight,
        dropout=dropout,
        rng=rng,
    )
    batches = _batches(train_pairs, space, X, batch_size, rng)
    val_batches = None
    if val_pairs:
        try:
            val_batches = _batches(val_pairs, space, X, batch_size, rng)
        except NotMeaningfulError:
            # Too few recent pairs to pack a clean batch. Training without a
            # holdout is reported by nn.train in its own notes rather than
            # papered over with a random split that would leak.
            val_batches = None

    history = train(
        encoder,
        batches,
        _contrastive_loss(temperature),
        Adam(lr=learning_rate, weight_decay=weight_decay),
        val_batches=val_batches,
        epochs=epochs,
        patience=patience,
        seed=seed,
    )
    encoder.eval()
    return encoder, X, history, train_pairs


def _log_sizes(panel: FeaturePanel, as_of: date) -> dict[str, float]:
    """Log size at one date for the size gate: market capitalisation, or assets.

    Market capitalisation is the measure ``assumptions.ml.peers.max_size_ratio``
    is written in terms of, and it is the right one. It is also unavailable
    whenever the panel was built without a price feed, which on this data is
    every date before roughly 2023, because the keyless quote source this
    package uses serves about three years of history and no more. Log total
    assets stands in where that happens, and it is a different quantity: a
    ten-times band on assets is not a ten-times band on market value, most
    obviously for a software company whose value is almost entirely off its
    balance sheet. The substitution is recorded on the encoder's notes so a
    reader of a gated comp set knows which band was applied.
    """
    out: dict[str, float] = {}
    for row in panel.rows:
        if row.as_of != as_of or not row.ok:
            continue
        for name in ("scale_log_market_cap", "scale_log_total_assets"):
            value = row.values.get(name)
            if value is not None and np.isfinite(value):
                out[row.ticker] = float(value)
                break
    return out


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def popularity_ranking(
    groups: Sequence[PeerGroup],
    universe: Sequence[str],
    rng: np.random.Generator,
) -> list[str]:
    """Every candidate ordered by how often anybody named it, query ignored.

    The degenerate solution written down deliberately, so that a model claiming
    to have learned comparability has to beat a model that does not know which
    company it was asked about. On this data it is a strong baseline rather than
    a straw man, because a compensation peer table is built from a small pool of
    widely benchmarked names and a handful of companies really do appear in most
    of them.

    Ties are broken by a draw from ``rng`` rather than alphabetically. Whole
    blocks of this ranking are tied, so an alphabetical tie-break would let the
    ticker alphabet decide a large part of the score, and a number that moves
    when a company renames itself is not a measurement.
    """
    counts: dict[str, int] = {t: 0 for t in universe}
    for g in groups:
        if not g.usable:
            continue
        for peer in set(g.peers):
            if peer in counts:
                counts[peer] += 1
    jitter = rng.random(len(universe))
    order = sorted(
        range(len(universe)),
        key=lambda i: (-counts[universe[i]], jitter[i]),
    )
    return [universe[i] for i in order]


def popularity_collapse(
    ranked_by_query: Mapping[Any, Sequence[str]],
    popularity: Sequence[str],
) -> float:
    """Mean rank correlation between the model's orderings and the popularity order.

    The number that separates "beat the popularity baseline" from "beat the
    popularity baseline by being it". A model whose every query returns the same
    list scores 1.0 here and should be read as having learned nothing about
    comparability regardless of what its NDCG says; a model that genuinely
    conditions on the query sits well below it.

    Returns NaN where fewer than two queries were supplied, because a mean over
    one ordering is that ordering.
    """
    rank_of = {t: i for i, t in enumerate(popularity)}
    values: list[float] = []
    for ranking in ranked_by_query.values():
        items = [t for t in ranking if t in rank_of]
        if len(items) < 3:
            continue
        theirs = np.arange(len(items), dtype=float)
        mine = np.asarray([rank_of[t] for t in items], dtype=float)
        rho = spearman(theirs, mine)
        if rho is not None:
            values.append(rho)
    if len(values) < 2:
        return float("nan")
    return float(np.mean(values))


def _same_sector_ranking(
    query: str,
    universe: Sequence[str],
    sector_of: Mapping[str, Any],
    rng: np.random.Generator,
) -> list[str]:
    """Same sub-vertical first, then everything else, each block shuffled.

    The SIC-code baseline, which is what a screen without a model does. It is
    almost entirely ties: a block of perhaps a dozen companies share the
    target's sub-vertical and the other ninety share nothing, so the ordering
    inside each block is arbitrary and is drawn rather than assumed.
    """
    mine = sector_of.get(query)
    jitter = rng.random(len(universe))
    order = sorted(
        range(len(universe)),
        key=lambda i: (
            0 if (mine is not None and sector_of.get(universe[i]) == mine) else 1,
            jitter[i],
        ),
    )
    return [universe[i] for i in order]


def _cosine_ranking(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    universe: Sequence[str],
) -> list[str]:
    """Order a universe by cosine against one vector, ties broken by position.

    ``np.argsort`` with a stable kind, so two candidates with an identical
    similarity keep the order the universe was given in and the ranking does not
    depend on the sort implementation.
    """
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    q = query_vector / (np.linalg.norm(query_vector) or 1.0)
    sims = (matrix @ q) / safe
    order = np.argsort(-sims, kind="stable")
    return [universe[int(i)] for i in order]


def _distance_ranking(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    universe: Sequence[str],
) -> list[str]:
    """Order a universe by euclidean distance, nearest first."""
    gaps = np.linalg.norm(matrix - query_vector[None, :], axis=1)
    order = np.argsort(gaps, kind="stable")
    return [universe[int(i)] for i in order]


# --------------------------------------------------------------------------- #
# Walk-forward evaluation
# --------------------------------------------------------------------------- #

# The methods scored side by side. "encoder" is the model; everything else is an
# alternative it has to beat, and the last two are the ones that decide whether
# the LEARNING added anything over the FEATURES.
METHOD_NAMES: tuple[str, ...] = (
    "encoder",
    "popularity prior",
    "same sub-vertical",
    "size and growth",
    "fundamentals cosine",
    "text cosine",
    "text lsa cosine",
)

# Size and growth, and nothing else. Two size columns because market
# capitalisation is unavailable wherever the panel was built without prices, and
# a baseline that silently becomes a growth-only baseline at some dates is not
# the baseline it claims to be. Both are standardised in fold, so a column that
# is entirely missing contributes a constant and drops out of the distance.
SIZE_GROWTH_COLUMNS: tuple[str, ...] = (
    "scale_log_market_cap",
    "scale_log_total_assets",
    "growth_revenue_1y",
)


@dataclass
class PeerEvaluation:
    """Every method's ranking score, and the model's verdict against each baseline.

    ``headline`` is the encoder against the popularity prior, because that is
    the comparison that decides whether the model learned comparability or
    learned which companies get named a lot. ``against`` carries the same
    encoder against each of the other baselines.

    ``warm`` and ``cold`` split the queries by whether the target company
    appeared in any training group. They are not two views of one number: the
    cold-start score is what a model is worth on a company the bank has not
    covered before, and it is usually materially worse.
    """

    headline: EvalResult
    against: dict[str, EvalResult]
    scores: pd.DataFrame
    paired: pd.DataFrame
    warm: EvalResult | None
    cold: EvalResult | None
    collapse: float
    folds: list[Fold]
    k: int
    n_queries: int
    coverage: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def rows(self) -> list[tuple[str, Any]]:
        return [
            ("Queries scored", self.n_queries),
            ("Folds", len(self.folds)),
            ("k", self.k),
            ("Encoder NDCG@%d" % self.k, self.headline.score),
            ("Popularity prior NDCG@%d" % self.k, self.headline.baseline_score),
            ("Beats the popularity prior", self.headline.beat_baseline),
            ("Correlation with the popularity order", self.collapse),
            ("Recall ceiling from universe coverage", self.coverage.get("in_universe")),
        ]

    def verdict(self) -> str:
        lines = [self.headline.verdict()]
        for name, result in self.against.items():
            lines.append(f"against {name}: {result.verdict()}")
        return "\n".join(lines)


def evaluate_peer_encoder(
    dataset: PeerDataset,
    assumptions: Assumptions,
    *,
    k: int = 10,
    n_folds: int | None = None,
    embargo_days: int = EMBARGO_DAYS,
    sic_by_ticker: Mapping[str, str | None] | None = None,
    dim: int = 64,
    hidden: Sequence[int] = (128,),
    text_dim: int = 128,
    text_weight: float | None = None,
    dropout: float = 0.1,
    temperature: float = 0.07,
    batch_size: int = 128,
    epochs: int = 60,
    patience: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    min_df: int = 2,
    max_df: float = 0.85,
    progress: Callable[[str], None] | None = None,
) -> PeerEvaluation:
    """Walk forward by date, refit everything inside each fold, score every method.

    One fold does five things in this order, and the order is the guarantee:
    cut the labels at the fold boundary less the embargo, fit the
    standardisation and the vocabulary on what is left, fit the encoder on it,
    embed the test window through those fitted transforms, and rank. Nothing
    fitted in a fold ever sees a document or a filing from that fold's test
    window.

    The candidate universe at a test date is every company that has both a
    feature row and a business description on that date, less the target. It is
    NOT filtered by size. A compensation peer group is chosen inside a revenue
    and market capitalisation band to begin with, so gating the candidate list
    on size hands every method a large part of the answer and measures the band
    rather than the ranker. ``PeerEncoder.neighbours`` applies the gate because
    a comp set on a banker's screen should have it; the measurement does not,
    and the gated score is reported separately in the notes so the size of that
    free lift is on the record.

    Raises ``NotMeaningfulError`` below the floors ``evaluation.py`` sets, which
    is the right behaviour: a mean NDCG over four targets is an anecdote.
    """
    weight = (
        float(assumptions.ml.peers.text_weight) if text_weight is None else float(text_weight)
    )
    seed = int(assumptions.ml.random_seed)
    folds_wanted = int(assumptions.ml.walk_forward_folds if n_folds is None else n_folds)
    say = progress or (lambda _msg: None)

    by_key: dict[tuple[str, int], list[PeerExample]] = {}
    for e in dataset.examples:
        by_key.setdefault(e.group_key, []).append(e)
    if not by_key:
        raise ConfigError("the dataset holds no labelled pairs, so nothing can be scored")

    # One observation per disclosed group, not per pair. A proxy naming twenty
    # companies is one assertion made on one day, and letting it put twenty
    # entries on the timeline would hand the fold cutter a weight it has no
    # business carrying.
    group_dates = [by_key[key][0].filed for key in by_key]
    folds = walk_forward_folds(group_dates, folds_wanted, embargo_days=embargo_days)

    relevance: dict[tuple[str, int], dict[str, float]] = {}
    rankings: dict[str, dict[tuple[str, int], list[str]]] = {m: {} for m in METHOD_NAMES}
    gated: dict[tuple[str, int], list[str]] = {}
    warm_keys: set[tuple[str, int]] = set()
    cold_keys: set[tuple[str, int]] = set()
    notes: list[str] = []

    for fold in folds:
        cut = fold.test_start - timedelta(days=fold.embargo_days)
        train_examples = [e for e in dataset.examples if e.filed < cut]
        test_keys = [
            key
            for key, items in by_key.items()
            if fold.test_start <= items[0].filed <= fold.test_end
        ]
        if not test_keys or len(train_examples) < MIN_TRAIN_PAIRS:
            notes.append(
                f"fold {fold.index} ({fold.test_start} to {fold.test_end}) was "
                f"skipped: {len(train_examples)} training pairs and "
                f"{len(test_keys)} test queries"
            )
            continue

        say(
            f"fold {fold.index}: train through {fold.train_end}, test "
            f"{fold.test_start} to {fold.test_end}, {len(train_examples)} pairs, "
            f"{len(test_keys)} queries"
        )
        space = _fit_space(
            dataset,
            fold.train_end,
            assumptions,
            text_dim=text_dim,
            min_df=min_df,
            max_df=max_df,
        )
        rng = np.random.default_rng(seed + fold.index)
        encoder, X, _history, _pairs = _train_encoder(
            dataset,
            space,
            [e for e in train_examples if space.has(e.a, e.as_of) and space.has(e.b, e.as_of)],
            text_weight=weight,
            dim=dim,
            hidden=hidden,
            dropout=dropout,
            temperature=temperature,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_share=0.15,
            seed=seed + fold.index,
            rng=rng,
        )

        trained_filers = {e.a for e in train_examples}
        train_groups = _groups_before(dataset, cut)
        sectors = _sectors(dataset, space.fit_date, assumptions, sic_by_ticker)
        # Embed every company-date once for the fold rather than once per query.
        # The encoder is in evaluation mode and the space does not change inside
        # a fold, so a forward per query would be the same arithmetic repeated a
        # hundred times over.
        embedded = encoder.forward(X)
        sizes_by_date = {
            d: _log_sizes(dataset.panel, d) for d in dataset.panel_dates
        }

        for key in test_keys:
            items = by_key[key]
            target, as_of = key[0], items[0].as_of
            universe = [
                t for t in dataset.universe if t != target and space.has(t, as_of)
            ]
            if len(universe) < k:
                continue
            keys_at_date = [(t, as_of) for t in universe]

            group = dataset.group_of(key)
            relevance[key] = {p: 1.0 for p in group.peers if p != target}
            (warm_keys if target in trained_filers else cold_keys).add(key)

            embeddings = space.rows_of(embedded, keys_at_date)
            query = space.rows_of(embedded, [(target, as_of)])[0]
            rankings["encoder"][key] = _cosine_ranking(query, embeddings, universe)

            rankings["popularity prior"][key] = popularity_ranking(
                train_groups, universe, np.random.default_rng(seed + fold.index)
            )
            rankings["same sub-vertical"][key] = _same_sector_ranking(
                target, universe, sectors, np.random.default_rng(seed + fold.index)
            )

            size_growth = _size_growth_block(space, keys_at_date)
            size_growth_q = _size_growth_block(space, [(target, as_of)])[0]
            rankings["size and growth"][key] = _distance_ranking(
                size_growth_q, size_growth, universe
            )

            fundamentals = space.rows_of(space.fundamentals, keys_at_date)
            rankings["fundamentals cosine"][key] = _cosine_ranking(
                space.rows_of(space.fundamentals, [(target, as_of)])[0],
                fundamentals,
                universe,
            )

            terms = space.rows_of(space.tfidf, keys_at_date)
            rankings["text cosine"][key] = _cosine_ranking(
                space.rows_of(space.tfidf, [(target, as_of)])[0], terms, universe
            )

            # The same projection the text tower is built on, with no learning
            # on top of it. This is the baseline that separates what the SVD did
            # from what the contrastive fit did, and without it a lift over raw
            # TF-IDF could be entirely the dimensionality reduction.
            lsa = space.rows_of(space.lsa, keys_at_date)
            rankings["text lsa cosine"][key] = _cosine_ranking(
                space.rows_of(space.lsa, [(target, as_of)])[0], lsa, universe
            )

            gated[key] = [
                t
                for t in rankings["encoder"][key]
                if _within_band(sizes_by_date[as_of], target, t, assumptions)
            ]

    if not relevance:
        raise NotMeaningfulError(
            "no fold produced a scorable query. Either every fold fell below the "
            f"{MIN_TRAIN_PAIRS}-pair training floor or the candidate universe was "
            f"smaller than k={k} at every test date"
        )

    scores = _score_frame(relevance, rankings, k)
    headline = evaluate_ranking(
        relevance,
        rankings["encoder"],
        k,
        baseline=rankings["popularity prior"],
        baseline_name="popularity prior (named-by-anybody count, query ignored)",
    )
    against = {
        name: evaluate_ranking(
            relevance, rankings["encoder"], k, baseline=rankings[name], baseline_name=name
        )
        for name in METHOD_NAMES[2:]
    }
    against["random order"] = evaluate_ranking(relevance, rankings["encoder"], k)
    paired = paired_lift(relevance, rankings, k)

    collapse = popularity_collapse(
        rankings["encoder"],
        popularity_ranking(
            _groups_before(dataset, max(e.filed for e in dataset.examples)),
            dataset.universe,
            np.random.default_rng(seed),
        ),
    )

    warm = _subset_result(relevance, rankings, k, warm_keys)
    cold = _subset_result(relevance, rankings, k, cold_keys)

    named = sum(len(r) for r in relevance.values())
    in_universe = sum(
        1 for r in relevance.values() for p in r if p in set(dataset.universe)
    )
    coverage = {
        "named peers": float(named),
        "in_universe": in_universe / named if named else float("nan"),
    }
    notes.append(
        f"{in_universe:,} of {named:,} named peers ({in_universe / named:.1%}) are in "
        "the candidate universe. Recall at any k is capped by that share, and the "
        "shortfall is a property of the universe rather than of any ranker."
    )
    if gated:
        gated_ndcg = _mean_ndcg(relevance, gated, k)
        notes.append(
            f"with the size gate from assumptions.ml.peers applied to the encoder's "
            f"candidate list, NDCG@{k} is {gated_ndcg:.4f} against "
            f"{headline.score:.4f} ungated. The gate is in the product and not in "
            "the measurement, and this is the size of the difference."
        )
    notes.append(
        f"correlation between the encoder's orderings and the popularity order is "
        f"{collapse:.3f}. At 1.0 the model returns the same list for every query "
        "and has learned popularity rather than comparability."
    )

    return PeerEvaluation(
        headline=headline,
        against=against,
        scores=scores,
        paired=paired,
        warm=warm,
        cold=cold,
        collapse=collapse,
        folds=folds,
        k=k,
        n_queries=len(relevance),
        coverage=coverage,
        notes=notes,
    )


def _groups_before(dataset: PeerDataset, cut: date) -> list[PeerGroup]:
    return [g for g in dataset.groups if g.filed is not None and g.filed < cut]


def _sectors(
    dataset: PeerDataset,
    fit_date: date,
    assumptions: Assumptions,
    sic_by_ticker: Mapping[str, str | None] | None,
) -> dict[str, Any]:
    """Classify the universe from the training window's own business descriptions.

    ``taxonomy.classify`` reconciles the SEC's SIC code with what the company
    says it does, and it is given the Item 1 text on file at the fold's fit
    date rather than the newest one. That matters less than it does for the
    towers, since a company rarely changes sub-vertical, but running the
    baseline on a later document than the model saw would make the comparison
    unfair in the baseline's favour and there is no reason to accept that.
    """
    from ..tmt.taxonomy import classify

    corpus = dataset.corpora.get(fit_date)
    out: dict[str, Any] = {}
    for ticker in dataset.universe:
        sic = None if sic_by_ticker is None else sic_by_ticker.get(ticker)
        text = None
        name = ""
        if corpus is not None and ticker in corpus.tickers:
            text = corpus.document(ticker)
            name = corpus.entity_names.get(ticker, "")
        vertical, _confidence, _source = classify(sic, name, text, assumptions)
        out[ticker] = vertical
    return out


def _size_growth_block(space: _Space, keys: Sequence[tuple[str, date]]) -> np.ndarray:
    """The size and growth columns alone, already standardised by the fold."""
    columns = [list(MATRIX_COLUMNS).index(c) for c in SIZE_GROWTH_COLUMNS]
    return space.rows_of(space.fundamentals, keys)[:, columns]


def _within_band(
    log_size: Mapping[str, float], target: str, candidate: str, assumptions: Assumptions
) -> bool:
    here, there = log_size.get(target), log_size.get(candidate)
    if here is None or there is None:
        return True
    floor = float(assumptions.ml.peers.min_market_cap)
    ratio = float(assumptions.ml.peers.max_size_ratio)
    if floor > 0.0 and there < float(np.log(floor)):
        return False
    if ratio > 0.0 and abs(here - there) > float(np.log(ratio)):
        return False
    return True


def _mean_ndcg(
    relevance: Mapping[Any, Mapping[str, float]],
    rankings: Mapping[Any, Sequence[str]],
    k: int,
) -> float:
    from .evaluation import ndcg_at_k

    values = []
    for key, ranking in rankings.items():
        grades = relevance.get(key, {})
        if not ranking or not any(grades.get(t, 0.0) > 0 for t in ranking):
            continue
        values.append(ndcg_at_k(grades, list(ranking), k))
    return float(np.mean(values)) if values else float("nan")


def paired_lift(
    relevance: Mapping[Any, Mapping[str, float]],
    rankings: Mapping[str, Mapping[Any, Sequence[str]]],
    k: int,
    *,
    against: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-query difference between the encoder and each baseline, and its dispersion.

    The statistic ``EvalResult.fold_sd`` reports for a ranking task is the
    spread of NDCG ACROSS TARGETS, and ``verdict`` compares the lift to it. That
    is a deliberately brutal bar and it answers the wrong question. Targets
    differ enormously in how findable their peers are, so the across-target
    spread is dominated by variation both methods share, and a lift can be
    perfectly reliable while sitting well inside it.

    The question is whether the encoder beats the baseline ON THE SAME TARGET,
    so the difference is taken query by query and its own mean and standard
    deviation are reported, together with the standard error of that mean and
    the share of targets where the encoder won. Paired because the two methods
    are scored on identical queries, which is the one thing that makes the
    difference an estimate of anything.

    ``win_rate`` is reported beside the mean because a mean difference can be
    carried by a handful of targets. A lift with a win rate near a half is a few
    big wins, not a better ranker.
    """
    from .evaluation import ndcg_at_k

    names = list(against) if against is not None else [
        m for m in rankings if m != "encoder"
    ]
    records = []
    for name in names:
        differences: list[float] = []
        for key, ranking in rankings["encoder"].items():
            grades = relevance.get(key, {})
            other = rankings[name].get(key)
            if other is None or not ranking:
                continue
            if not any(grades.get(t, 0.0) > 0 for t in ranking):
                continue
            differences.append(
                ndcg_at_k(grades, list(ranking), k) - ndcg_at_k(grades, list(other), k)
            )
        if len(differences) < 2:
            continue
        values = np.asarray(differences, dtype=float)
        sd = float(np.std(values, ddof=1))
        records.append(
            {
                "baseline": name,
                "mean_difference": float(values.mean()),
                "sd_of_difference": sd,
                "standard_error": sd / np.sqrt(values.size),
                "t": float(values.mean() / (sd / np.sqrt(values.size))) if sd else np.inf,
                "win_rate": float((values > 0).mean()),
                "n_queries": int(values.size),
            }
        )
    return pd.DataFrame(records).sort_values(
        "mean_difference", ascending=False, ignore_index=True
    )


def _score_frame(
    relevance: Mapping[Any, Mapping[str, float]],
    rankings: Mapping[str, Mapping[Any, Sequence[str]]],
    k: int,
) -> pd.DataFrame:
    """Absolute NDCG, precision and recall for every method on the same queries.

    Computed directly rather than through ``evaluate_ranking``, which pairs one
    ordering against one baseline. A reader comparing six methods wants them in
    one table on identical queries, and the EvalResults beside it carry the
    formal verdict.
    """
    from .evaluation import ndcg_at_k, precision_recall_at_k

    records = []
    for method, by_query in rankings.items():
        ndcgs, precisions, recalls = [], [], []
        for key, ranking in by_query.items():
            grades = relevance.get(key, {})
            relevant = {t for t, g in grades.items() if g > 0}
            if not relevant or not ranking:
                continue
            if not any(grades.get(t, 0.0) > 0 for t in ranking):
                continue
            ndcgs.append(ndcg_at_k(grades, list(ranking), k))
            precision, recall = precision_recall_at_k(relevant, list(ranking), k)
            precisions.append(precision)
            recalls.append(recall)
        records.append(
            {
                "method": method,
                f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else float("nan"),
                "ndcg_sd": float(np.std(ndcgs, ddof=1)) if len(ndcgs) > 1 else float("nan"),
                f"precision@{k}": float(np.mean(precisions)) if precisions else float("nan"),
                f"recall@{k}": float(np.mean(recalls)) if recalls else float("nan"),
                "n_queries": len(ndcgs),
            }
        )
    frame = pd.DataFrame(records)
    return frame.sort_values(f"ndcg@{k}", ascending=False, ignore_index=True)


def _subset_result(
    relevance: Mapping[Any, Mapping[str, float]],
    rankings: Mapping[str, Mapping[Any, Sequence[str]]],
    k: int,
    keys: set,
) -> EvalResult | None:
    """The encoder against the popularity prior on one slice of the queries.

    Returns None rather than raising where the slice is below the floor
    ``evaluate_ranking`` enforces. A cold-start sample of four targets is not a
    weak result, it is not a result, and the caller is told it is absent rather
    than handed a number.
    """
    if not keys:
        return None
    sub_rel = {key: relevance[key] for key in keys if key in relevance}
    model = {key: rankings["encoder"][key] for key in keys if key in rankings["encoder"]}
    base = {
        key: rankings["popularity prior"][key]
        for key in keys
        if key in rankings["popularity prior"]
    }
    try:
        return evaluate_ranking(
            sub_rel,
            model,
            k,
            baseline=base,
            baseline_name="popularity prior (named-by-anybody count, query ignored)",
        )
    except NotMeaningfulError:
        return None


# --------------------------------------------------------------------------- #
# Ablation: which tower carries the signal
# --------------------------------------------------------------------------- #


def ablate_towers(
    dataset: PeerDataset,
    assumptions: Assumptions,
    *,
    folds: Sequence[Fold] | None = None,
    n_folds: int | None = None,
    embargo_days: int = EMBARGO_DAYS,
    negatives_per_pair: int = 5,
    dim: int = 64,
    hidden: Sequence[int] = (128,),
    text_dim: int = 128,
    dropout: float = 0.1,
    temperature: float = 0.07,
    batch_size: int = 128,
    epochs: int = 60,
    patience: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    min_df: int = 2,
    max_df: float = 0.85,
) -> pd.DataFrame:
    """Refit with each tower removed in turn and report what its absence costs.

    Two feature groups, so ``evaluation.ablation`` produces exactly the three
    configurations the question needs: the full model, the model without the
    text group, which is fundamentals alone, and the model without the
    fundamentals group, which is text alone. ``damage`` is signed so positive
    means the tower was doing work, and a tower whose damage is smaller than the
    fold standard deviation beside it has not been shown to help.

    **The observation is a pair, and the score is a rank correlation.**
    ``ablation`` is written for a per-row regression and a peer encoder is not
    one, so the adaptation is stated here rather than left to be discovered. A
    row is a (target, candidate, date) pair, the outcome is 1 for a candidate
    the target's proxy named and 0 for one it did not, and the prediction is the
    cosine the model gives that pair. Spearman between the two is a rank measure
    of exactly the thing the ranking metrics measure, one monotone transform
    away from the area under the ROC curve, and unlike NDCG it is defined per
    row, which is what the ablation machinery needs.

    **What the columns are, precisely.** The frame's index carries the pair and
    its date and its columns carry the raw point-in-time feature values of both
    legs, so it is a real design matrix a reader can inspect. The encoder does
    not read its inputs from it. It refits the standardisation, the TF-IDF
    vocabulary and the SVD basis inside the fold through ``_fit_space``, because
    a projected text vector cannot be put in a column without having fitted the
    projection on the whole sample first, which is exactly the leak trap three
    is about. What the ablation therefore varies is which towers exist, which is
    the question being asked, and the two ``doc`` columns are the pointer the
    text group is named by.

    ``negatives_per_pair`` non-peers are drawn per disclosed pair from the same
    date's universe, seeded from ``assumptions.ml.random_seed``. Five rather
    than the whole universe because a rank correlation over a 1-in-100 positive
    rate is dominated by the ordering among the negatives, which is not the
    question.
    """
    seed = int(assumptions.ml.random_seed)
    rng = np.random.default_rng(seed)
    weight = float(assumptions.ml.peers.text_weight)

    peers_by_key: dict[tuple[str, int], set[str]] = {}
    for e in dataset.examples:
        peers_by_key.setdefault(e.group_key, set()).add(e.b)

    by_date_universe: dict[date, list[str]] = {}
    ok_keys = {(r.ticker, r.as_of) for r in dataset.panel.rows if r.ok}
    for d in dataset.panel_dates:
        available = set(dataset.corpora[d].tickers)
        by_date_universe[d] = [
            t for t in dataset.universe if t in available and (t, d) in ok_keys
        ]

    index_rows: list[tuple[str, str, date, int]] = []
    outcomes: list[float] = []
    dates: list[date] = []
    for e in dataset.examples:
        index_rows.append((e.a, e.b, e.as_of, e.fiscal_year))
        outcomes.append(1.0)
        dates.append(e.filed)
        pool = [
            t
            for t in by_date_universe[e.as_of]
            if t != e.a and t not in peers_by_key[e.group_key]
        ]
        if not pool:
            continue
        draw = rng.choice(len(pool), size=min(negatives_per_pair, len(pool)), replace=False)
        for j in draw:
            index_rows.append((e.a, pool[int(j)], e.as_of, e.fiscal_year))
            outcomes.append(0.0)
            dates.append(e.filed)

    frame = _pair_frame(dataset, index_rows)
    if folds is None:
        folds = walk_forward_folds(
            [d for d in dates],
            int(assumptions.ml.walk_forward_folds if n_folds is None else n_folds),
            embargo_days=embargo_days,
        )

    settings = dict(
        dim=dim,
        hidden=hidden,
        text_dim=text_dim,
        dropout=dropout,
        temperature=temperature,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        min_df=min_df,
        max_df=max_df,
    )

    def fit_predict(
        X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame
    ) -> np.ndarray:
        columns = set(X_train.columns)
        use_text = "a_doc" in columns
        use_fund = any(c.startswith("a_feat_") for c in columns)
        if use_fund and use_text:
            this_weight = weight
        elif use_fund:
            this_weight = 0.0
        else:
            this_weight = 1.0

        train_through = max(d for _a, _b, d, _y in X_train.index)
        space = _fit_space(
            dataset,
            train_through,
            assumptions,
            text_dim=settings["text_dim"],
            min_df=settings["min_df"],
            max_df=settings["max_df"],
        )
        positives = [
            PeerExample(a=a, b=b, fiscal_year=fy, filed=train_through, as_of=d, group_key=(a, fy))
            for (a, b, d, fy), y in zip(X_train.index, y_train)
            if y > 0 and space.has(a, d) and space.has(b, d)
        ]
        encoder, X, _history, _pairs = _train_encoder(
            dataset,
            space,
            positives,
            text_weight=this_weight,
            dim=settings["dim"],
            hidden=settings["hidden"],
            dropout=settings["dropout"],
            temperature=settings["temperature"],
            batch_size=settings["batch_size"],
            epochs=settings["epochs"],
            patience=settings["patience"],
            learning_rate=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            validation_share=0.15,
            seed=seed,
            rng=np.random.default_rng(seed),
        )
        out = np.zeros(len(X_test), dtype=float)
        wanted = [
            (a, b, d)
            for (a, b, d, _fy) in X_test.index
        ]
        encodable = [
            i
            for i, (a, b, d) in enumerate(wanted)
            if space.has(a, d) and space.has(b, d)
        ]
        if encodable:
            left = encoder.forward(
                space.rows_of(X, [(wanted[i][0], wanted[i][2]) for i in encodable])
            )
            right = encoder.forward(
                space.rows_of(X, [(wanted[i][1], wanted[i][2]) for i in encodable])
            )
            out[encodable] = np.sum(left * right, axis=1)
        return out

    groups = {
        "fundamentals": [c for c in frame.columns if "_feat_" in c],
        "text": ["a_doc", "b_doc"],
    }
    return ablation(
        fit_predict,
        groups,
        frame,
        np.asarray(outcomes, dtype=float),
        dates,
        folds=folds,
        metric="spearman",
    )


def _pair_frame(
    dataset: PeerDataset, index_rows: Sequence[tuple[str, str, date, int]]
) -> pd.DataFrame:
    """A real design matrix over pairs: both legs' raw features and their documents.

    The values are the point-in-time feature row of each leg as the panel built
    it, before standardisation, so a reader can read a row of this frame and see
    the two companies being compared. The ``doc`` columns hold each leg's
    position in its own date's corpus, which is what the text group is named by;
    see ``ablate_towers`` for why the projected text vector cannot be a column.
    """
    value_by_key = {
        (r.ticker, r.as_of): r for r in dataset.panel.rows if r.ok
    }
    doc_index = {
        (t, d): i
        for d in dataset.panel_dates
        for i, t in enumerate(dataset.corpora[d].tickers)
    }
    names = list(MATRIX_COLUMNS)
    records: list[dict[str, float]] = []
    for a, b, d, _fy in index_rows:
        record: dict[str, float] = {}
        for side, ticker in (("a", a), ("b", b)):
            row = value_by_key.get((ticker, d))
            for name in names:
                value = None if row is None else row.values.get(name)
                record[f"{side}_feat_{name}"] = (
                    np.nan if value is None else float(value)
                )
            record[f"{side}_doc"] = float(doc_index.get((ticker, d), -1))
        records.append(record)
    index = pd.MultiIndex.from_tuples(
        [tuple(r) for r in index_rows], names=["a", "b", "as_of", "fiscal_year"]
    )
    return pd.DataFrame.from_records(records, index=index)


# --------------------------------------------------------------------------- #
# Survivorship
# --------------------------------------------------------------------------- #


def survivorship_report(
    groups: Sequence[PeerGroup],
    universe: Sequence[str],
    deal_events: Sequence[tuple[str, date, bool]] = (),
) -> pd.DataFrame:
    """Which named peers never entered the universe, and which of them were acquired.

    The labels are dated and safe. The universe is where survivorship lives: a
    candidate list assembled from companies listed today has already lost every
    company that was acquired during the sample, and the loss is not random.
    A company gets bought partly for resembling the companies around it, so the
    names that vanish are disproportionately the good peers, and their absence
    shows up as a recall shortfall that looks like a weakness of the ranker.

    One row per named peer that is absent from the universe, with the count of
    groups that named it, the last year it was named, and the announcement date
    of a merger agreement where ``deal_events`` carries one. Pass the output of
    ``tmt.precedents.deal_events`` over the named peers to fill that column; the
    default of nothing returns the absence census alone, which is still the
    number that bounds recall.
    """
    pool = {str(t).upper() for t in universe}
    deal_by_ticker = {t: (when, done) for t, when, done in deal_events}

    counts: dict[str, int] = {}
    last_year: dict[str, int] = {}
    for g in groups:
        if not g.usable or g.fiscal_year is None:
            continue
        for peer in set(g.peers):
            counts[peer] = counts.get(peer, 0) + 1
            last_year[peer] = max(last_year.get(peer, 0), g.fiscal_year)

    records = []
    for peer, count in counts.items():
        deal = deal_by_ticker.get(peer)
        records.append(
            {
                "peer": peer,
                "times_named": count,
                "last_named_fiscal_year": last_year[peer],
                "in_universe": peer in pool,
                "merger_announced": None if deal is None else deal[0],
                "merger_completed": None if deal is None else deal[1],
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["in_universe", "times_named"], ascending=[True, False], ignore_index=True
    )
