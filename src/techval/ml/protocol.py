"""The contract every model in this package satisfies.

A model that lands next to a discounted cash flow has to meet the standard the
rest of the engine is held to: every number traceable, every judgment stated,
nothing silently substituted. Three things make that possible for a fitted
model, and they are defined here so that no model can skip one.

**The baseline.** A score in isolation says nothing. An R-squared of 0.4 on
revenue growth sounds respectable until you learn that simply carrying last
year's growth forward scores 0.45, at which point the model is worse than doing
nothing and should be reported as such. Every ``EvalResult`` carries the naive
alternative beside the model, and ``beat_baseline`` is computed, not asserted.

**The model card.** What it was trained on, over what period, with what
features, and how it scored out of sample. A number nobody can audit does not
belong beside numbers that trace to filings.

**Point in time.** Every observation carries the knowledge date it was built
through. A model fitted on facts that postdate its own label is measuring
hindsight, and the failure is silent because the fit looks excellent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np


@dataclass
class EvalResult:
    """How a model scored, against what, and whether that was any good.

    ``metric`` names the score (``spearman``, ``ndcg@10``, ``auc``, ``mae``).
    ``baseline_name`` names what it is being compared against, and it must be
    the honest alternative rather than a straw man: for a growth model that is
    last year's growth, for a ranking model the sector median, for a classifier
    the base rate.
    """

    metric: str
    score: float
    baseline_name: str
    baseline_score: float
    n_observations: int
    higher_is_better: bool = True
    folds: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def beat_baseline(self) -> bool:
        if self.higher_is_better:
            return self.score > self.baseline_score
        return self.score < self.baseline_score

    @property
    def lift(self) -> float:
        """Score less baseline, signed so that positive always means better."""
        raw = self.score - self.baseline_score
        return raw if self.higher_is_better else -raw

    @property
    def fold_sd(self) -> float | None:
        """Dispersion across folds, which is the honest error bar on the score."""
        if len(self.folds) < 2:
            return None
        return float(np.std(self.folds, ddof=1))

    def verdict(self) -> str:
        """One line a reader can act on, including when the answer is no."""
        if not self.beat_baseline:
            return (
                f"{self.metric} of {self.score:.4f} against {self.baseline_score:.4f} "
                f"for {self.baseline_name}: the model does NOT beat the baseline on "
                f"{self.n_observations:,} observations. Use the baseline."
            )
        sd = self.fold_sd
        spread = (
            f" Fold standard deviation {sd:.4f}, so the lift is "
            + ("inside" if sd and abs(self.lift) < sd else "outside")
            + " the fold-to-fold noise."
            if sd is not None
            else ""
        )
        return (
            f"{self.metric} of {self.score:.4f} against {self.baseline_score:.4f} "
            f"for {self.baseline_name}, a lift of {self.lift:+.4f} on "
            f"{self.n_observations:,} observations.{spread}"
        )

    def rows(self) -> list[tuple[str, Any]]:
        return [
            ("Metric", self.metric),
            ("Model score", self.score),
            (f"Baseline ({self.baseline_name})", self.baseline_score),
            ("Lift", self.lift),
            ("Observations", self.n_observations),
            ("Folds", len(self.folds)),
            ("Beats baseline", self.beat_baseline),
        ]


@dataclass
class ModelCard:
    """What a fitted model is, so that it can be argued with.

    ``limitations`` is not optional decoration. Every model here is fitted on a
    few hundred companies over a handful of years, which is a small sample by
    any standard, and the card is where that is said out loud.
    """

    name: str
    task: str
    trained_through: date
    n_train: int
    features: list[str]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    evaluation: EvalResult | None = None
    limitations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def rows(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = [
            ("Model", self.name),
            ("Task", self.task),
            ("Trained through", str(self.trained_through)),
            ("Training observations", self.n_train),
            ("Features", len(self.features)),
        ]
        if self.evaluation is not None:
            out.extend(self.evaluation.rows())
        return out

    def summary(self) -> str:
        head = (
            f"{self.name}: {self.task}, fitted on {self.n_train:,} observations "
            f"through {self.trained_through} across {len(self.features)} features."
        )
        if self.evaluation is None:
            return head + " Not evaluated."
        return head + " " + self.evaluation.verdict()


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Rank correlation, implemented here so the package keeps one dependency less.

    Rank rather than Pearson wherever a relationship is monotonic at best, which
    covers essentially everything in this package: a handful of extreme values
    would otherwise drive the whole coefficient.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not distort the correlation."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(a.size, dtype=float)
    # Average the ranks inside each tied group.
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks
