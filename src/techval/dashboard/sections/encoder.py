"""Peer encoder: the learned comp set against every rule that needs no learning.

The collector runs the encoder's own walk-forward evaluation on the committed
compensation peer groups, feature panel and Item 1 excerpts, built exactly as
``techval peers`` builds them, then the tower ablation on two fold cuts and the
evaluation again on shorter excerpts. Nothing on the page is typed in: every
number, every figure title that states a number and the takeaway are built
from what those runs return.

The work is split in two so the page can be tested without a fit.
``gather(ctx)`` runs the models and returns plain values, recording provenance
as it goes. ``shape(results)`` is pure: it turns those values into the figures,
the headline, the takeaway and the refusals.

**A figure exists exactly when its provenance does.** ``ctx.record`` appends a
row only when its block completes, and the snapshot schema refuses a row that
names a missing figure. So every decision to drop a figure is taken inside the
block that records it, by raising a ``TechvalError`` there. ``gather`` catches
it outside the block and hands the reason to ``shape`` under ``refused``, and
``shape`` draws a figure only when its key is not in ``refused``. A part of a
figure that failed (one fold cut, one excerpt length) is caught inside the
block instead, and becomes a refusal note under the figure it belongs to.

**Where the timing goes.** The evaluation is timed once, on the first figure
that reads it. The figures derived from the same evaluation record their own
short blocks, so the seconds on this section's provenance sum to the time the
section took rather than counting one fit several times.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

from ...commands_peers import _load_corpora, _load_groups, _load_panel
from ...errors import NotMeaningfulError, TechvalError
from ...ml.encoder import (
    EMBARGO_DAYS,
    METHOD_NAMES,
    MIN_TRAIN_PAIRS,
    ablate_towers,
    ablation_folds,
    build_dataset,
    evaluate_peer_encoder,
)
from ...ml.features import FEATURE_NAMES

ID = "encoder"
TITLE = "Peer encoder"

GROUPS = "peer_groups_tmt.json"
PANEL = "peer_panel_tmt.json"
ITEM1 = "peer_item1_tmt.json"
INPUTS: list[str] = [GROUPS, PANEL, ITEM1]

K = 10

# Excerpt lengths the truncation sweep scores, longest first. The first is the
# length the committed corpus was pruned to; the figure's subtitle says how many
# documents each cut actually shortened, which is none at that length.
CUTS: tuple[int, ...] = (2500, 1500, 800)

EVALUATE = "techval.ml.encoder.evaluate_peer_encoder"
ABLATE = "techval.ml.encoder.ablate_towers"

# The truncation sweep draws three lines and tables every method: the model, one
# baseline that reads the excerpts, one that does not.
SWEEP_DRAWN = (("encoder", "model"), ("text cosine", "alt"), ("fundamentals cosine", "baseline"))

# How each method is named on the page. A method the model module adds without a
# label here is a KeyError, which is a bug to fix rather than a row to hide.
LABELS = {
    "encoder": "Peer encoder",
    "popularity prior": "Popularity prior",
    "same sub-vertical": "Same sub-vertical",
    "size and growth": "Size and growth",
    "fundamentals cosine": "Fundamentals cosine",
    "text cosine": "Text cosine",
    "text lsa cosine": "Text after SVD",
}

# The two ways the ablation is cut into folds, in the order they are drawn.
CUT_OWN = "ablation"
CUT_HEADLINE = "headline"

TOWERS = {"without text": "text", "without fundamentals": "fundamentals"}

# A margin over the fold standard deviation smaller than this share of that
# deviation is described as "by a hair". It is a choice of words, not a verdict:
# the number is printed beside it either way.
HAIR = 0.05

# Correlation with the popularity order at or above which the encoder's lists
# are described as leaning on popularity. Again words, with the value beside them.
LEANS = 0.5

# The paired t the headline has to clear. Stated once and used by the rule.
T_BAR = 2.0

def _label(method: str) -> str:
    return LABELS[method]


def _phrase(method: str) -> str:
    """The label as it reads mid-sentence."""
    text = LABELS[method]
    return text[:1].lower() + text[1:]


def _f(value: Any) -> float:
    return float(value)


_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _count(n: int, noun: str) -> str:
    """A count as it reads before a verb: three baselines do, one baseline does."""
    word = _WORDS[n] if 0 <= n < len(_WORDS) else f"{n:,}"
    return f"{word} {noun} does" if n == 1 else f"{word} {noun}s do"


# --------------------------------------------------------------------------- #
# gather: run the models and return plain values
# --------------------------------------------------------------------------- #


def _eval_result(r) -> dict[str, Any]:
    p = r.paired
    return {
        "metric": r.metric,
        "score": _f(r.score),
        "baseline_name": r.baseline_name,
        "baseline_score": _f(r.baseline_score),
        "lift": _f(r.lift),
        "n": int(r.n_observations),
        "verdict": r.verdict(),
        "paired": None
        if p is None
        else {
            "mean": _f(p.mean),
            "se": _f(p.standard_error),
            "t": _f(p.t),
            "win_rate": _f(p.win_rate),
            "n": int(p.n),
        },
    }


def method_scores(evaluation) -> list[dict[str, Any]]:
    """NDCG@k for every method on identical queries, best first, from the score frame."""
    column = f"ndcg@{evaluation.k}"
    return [
        {"method": row["method"], "ndcg": _f(row[column]), "n": int(row["n_queries"])}
        for _, row in evaluation.scores.iterrows()
    ]


def _paired_rows(evaluation) -> list[dict[str, Any]]:
    return [
        {
            "baseline": row["baseline"],
            "mean": _f(row["mean_difference"]),
            "se": _f(row["standard_error"]),
            "t": _f(row["t"]),
            "win_rate": _f(row["win_rate"]),
            "n": int(row["n_queries"]),
        }
        for _, row in evaluation.paired.iterrows()
    ]


def _ablation_rows(frame) -> list[dict[str, Any]]:
    return [
        {
            "group": row["group"],
            "score": _f(row["score"]),
            "damage": _f(row["damage"]),
            "fold_sd": _f(row["fold_sd"]),
            "n_test": int(row["n_test"]),
        }
        for _, row in frame.iterrows()
    ]


def _cut_corpora(corpora, chars: int):
    """The corpora with every document cut to ``chars`` characters, and how many changed."""
    out = {}
    shortened = 0
    for when, corpus in corpora.items():
        docs = [d[:chars] for d in corpus.documents]
        shortened += sum(1 for before, after in zip(corpus.documents, docs) if before != after)
        out[when] = dataclasses.replace(corpus, documents=docs)
    return out, shortened


def _load(ctx):
    """The three recorded inputs, read by the loaders ``techval peers`` reads them with."""
    return (
        _load_groups(ctx.input(GROUPS)),
        _load_panel(ctx.input(PANEL)),
        _load_corpora(ctx.input(ITEM1)),
    )


def gather(ctx) -> dict[str, Any]:
    """Run the evaluation, the ablation and the sweep, recording each figure's provenance."""
    a = ctx.assumptions
    refused: dict[str, str] = {}
    out: dict[str, Any] = {"refused": refused}

    # A failure here is a TechvalError that refuses the whole section: every
    # figure and the headline read this evaluation.
    with ctx.record("ndcg_by_method", EVALUATE, INPUTS):
        groups, panel, corpora = _load(ctx)
        dataset = build_dataset(groups, panel, corpora)
        evaluation = evaluate_peer_encoder(dataset, a, k=K)
        out["k"] = evaluation.k
        out["n_queries"] = int(evaluation.n_queries)
        out["methods"] = method_scores(evaluation)
        out["random_order"] = _f(evaluation.against["random order"].baseline_score)
        out["headline"] = _eval_result(evaluation.headline)
        out["popularity"] = METHOD_NAMES[1]

    with ctx.record("paired_lift", EVALUATE, INPUTS):
        out["paired"] = _paired_rows(evaluation)

    with ctx.record("diagnostics", EVALUATE, INPUTS):
        out["diagnostics"] = {
            "collapse": _f(evaluation.collapse),
            "in_universe": _f(evaluation.coverage["in_universe"]),
            "n_folds": len(evaluation.folds),
            "embargo_days": EMBARGO_DAYS,
        }

    try:
        with ctx.record("warm_cold", EVALUATE, INPUTS):
            warm, cold = evaluation.warm, evaluation.cold
            if warm is None and cold is None:
                raise NotMeaningfulError(
                    "neither the warm-start nor the cold-start slice holds enough "
                    "scorable queries for evaluate_ranking's floor"
                )
            out["warm_cold"] = {
                "warm": None if warm is None else _eval_result(warm),
                "cold": None if cold is None else _eval_result(cold),
            }
    except TechvalError as exc:
        refused["warm_cold"] = str(exc)

    try:
        with ctx.record("tower_ablation", ABLATE, INPUTS):
            cuts: dict[str, Any] = {}
            plans = (
                (CUT_OWN, None, len(ablation_folds(dataset, a))),
                (CUT_HEADLINE, evaluation.folds, len(evaluation.folds)),
            )
            for name, folds, n_folds in plans:
                try:
                    frame = ablate_towers(dataset, a, folds=folds)
                except TechvalError as exc:
                    cuts[name] = {"refused": str(exc), "n_folds": n_folds}
                    continue
                cuts[name] = {"rows": _ablation_rows(frame), "n_folds": n_folds}
            if all("refused" in c for c in cuts.values()):
                raise NotMeaningfulError(
                    "the ablation refused on both fold cuts: "
                    + "; ".join(f"{k}: {c['refused']}" for k, c in cuts.items())
                )
            empty = [
                name
                for name in FEATURE_NAMES
                if not any(
                    r.values.get(name) is not None and math.isfinite(r.values[name])
                    for r in panel.rows
                )
            ]
            out["ablation"] = {
                "cuts": cuts,
                "min_train_pairs": MIN_TRAIN_PAIRS,
                "panel": {
                    "features": len(FEATURE_NAMES),
                    "empty": empty,
                    "market": [n for n in FEATURE_NAMES if n.startswith("market_")],
                    "rows": len(panel.rows),
                },
            }
    except TechvalError as exc:
        refused["tower_ablation"] = str(exc)

    try:
        with ctx.record("truncation", EVALUATE, INPUTS):
            sweep = []
            for chars in CUTS:
                cut, shortened = _cut_corpora(corpora, chars)
                entry: dict[str, Any] = {"chars": chars, "shortened": shortened}
                if shortened == 0:
                    # Not one document changed, so the inputs are the bytes the
                    # evaluation above already scored under the same seed, and
                    # rerunning it would repeat identical arithmetic.
                    entry["methods"] = out["methods"]
                else:
                    try:
                        entry["methods"] = method_scores(
                            evaluate_peer_encoder(build_dataset(groups, panel, cut), a, k=K)
                        )
                    except TechvalError as exc:
                        entry["refused"] = str(exc)
                sweep.append(entry)
            if sum(1 for e in sweep if "methods" in e) < 2:
                raise NotMeaningfulError(
                    "fewer than two excerpt lengths could be scored, so there is no "
                    "sweep to draw: "
                    + "; ".join(f"{e['chars']:,} characters: {e['refused']}" for e in sweep if "refused" in e)
                )
            out["truncation"] = {
                "documents": sum(len(c.documents) for c in corpora.values()),
                "longest": max(len(d) for c in corpora.values() for d in c.documents),
                "cuts": sweep,
            }
    except TechvalError as exc:
        refused["truncation"] = str(exc)

    return out


# --------------------------------------------------------------------------- #
# shape: plain values to figures, headline, takeaway and refusals
# --------------------------------------------------------------------------- #


def verdict_status(lift: float, t: float | None) -> str:
    """The headline chip, from the paired t against the popularity prior.

    beats            t above T_BAR
    loses            t below minus T_BAR
    ties             otherwise, where the lift is exactly zero
    not_significant  otherwise, including when no paired t could be computed

    The fold-noise status (inside_noise) is not used: a ranking result's spread
    is across queries, and the paired difference is what judges the lift. An
    infinite t, which is every query moving by the same nonzero amount, falls
    on the side its sign puts it; a NaN t clears neither bar.
    """
    if t is not None and t > T_BAR:
        return "beats"
    if t is not None and t < -T_BAR:
        return "loses"
    if lift == 0:
        return "ties"
    return "not_significant"


def _figure(kind: str, title: str, subtitle: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "title": title, "subtitle": subtitle, "data": data}


def _headline(results: Mapping[str, Any]) -> dict[str, Any]:
    h = results["headline"]
    paired = h["paired"]
    return {
        "metric": h["metric"].upper(),
        "score": h["score"],
        "baseline_name": _phrase(results["popularity"]),
        "baseline_score": h["baseline_score"],
        "lift": h["lift"],
        "n": h["n"],
        "higher_is_better": True,
        "verdict_status": verdict_status(h["lift"], None if paired is None else paired["t"]),
        # The EvalResult's own sentence for the encoder against the popularity
        # prior, verbatim. It is the first line of PeerEvaluation.verdict(); the
        # other lines are the per-baseline verdicts the paired figure draws.
        "verdict_text": h["verdict"],
    }


def _diagnostics(results: Mapping[str, Any]) -> dict[str, Any]:
    d = results["diagnostics"]
    leans = d["collapse"] >= LEANS
    title = (
        "The encoder's lists lean on the popular names"
        if leans
        else "The encoder does not just return the popular names"
    )
    return _figure(
        "tiles",
        title,
        "",
        {
            "tiles": [
                {
                    "label": "Correlation with the popularity order",
                    "value": d["collapse"],
                    "format": "num:3",
                    "sub": "Mean rank correlation per query; 1.0 is one list for every company",
                },
                {
                    "label": "Named peers inside the candidate universe",
                    "value": d["in_universe"],
                    "format": "pct:1",
                    "sub": "The ceiling on recall, the same for every method",
                },
                {
                    "label": "Walk-forward folds",
                    "value": d["n_folds"],
                    "format": "int",
                    "sub": f"Split by proxy filing date, {d['embargo_days']}-day embargo, "
                    f"{results['n_queries']} queries",
                },
            ]
        },
    )


def _ndcg_by_method(results: Mapping[str, Any]) -> dict[str, Any]:
    k = results["k"]
    methods = results["methods"]
    baselines = [m for m in methods if m["method"] != "encoder"]
    encoder = next(m for m in methods if m["method"] == "encoder")
    best = baselines[0]
    if best["ndcg"] >= encoder["ndcg"]:
        title = f"{_label(best['method'])} ranks peers at least as well as the encoder"
    else:
        title = f"{_label(best['method'])} comes closest to the encoder, with no learning at all"
    rows = [
        {
            "label": _label(m["method"]),
            "method": m["method"],
            "value": m["ndcg"],
            "role": "model" if m["method"] == "encoder" else "baseline",
        }
        for m in methods
    ]
    return _figure(
        "hbar",
        title,
        f"NDCG@{k} on the same {encoder['n']} queries, walk-forward by proxy filing date; "
        "the rule is a random ordering's expected score",
        {
            "rows": rows,
            "format": "num:3",
            "valueLabel": f"NDCG@{k}",
            "labelHeader": "Method",
            "roleLabels": {"model": "Peer encoder", "baseline": "Baselines"},
            "reference": [{"value": results["random_order"], "label": "Random order"}],
        },
    )


def _paired_lift(results: Mapping[str, Any]) -> dict[str, Any]:
    k = results["k"]
    paired = results["paired"]
    n = paired[0]["n"] if paired else 0
    # A NaN t clears nothing; an infinite one, every query moving alike, clears.
    clear = [p for p in paired if p["t"] > T_BAR]
    narrowest = min(paired, key=lambda p: p["mean"])
    if len(clear) == len(paired):
        title = (
            f"The encoder wins query by query against every baseline, narrowest against "
            f"{_phrase(narrowest['baseline'])}"
        )
    else:
        behind = ", ".join(_phrase(p["baseline"]) for p in paired if p not in clear)
        title = f"The paired lift does not clear two standard errors against {behind}"
    rows = [
        {
            "label": _label(p["baseline"]),
            "method": p["baseline"],
            "values": {"lift": p["mean"]},
            "lo": p["mean"] - p["se"],
            "hi": p["mean"] + p["se"],
            "text": f"wins {p['win_rate']:.0%}",
            "tip": [
                {"label": "Standard error", "value": p["se"], "format": "num:4"},
                {"label": "t", "value": p["t"], "format": "num:1"},
                {"label": "Encoder wins", "value": p["win_rate"], "format": "pct:1"},
            ],
            "mean": p["mean"],
            "se": p["se"],
            "t": p["t"],
            "win_rate": p["win_rate"],
            "n": p["n"],
        }
        for p in paired
    ]
    return _figure(
        "dot",
        title,
        f"Encoder minus baseline, NDCG@{k} per query, mean over the same {n} queries; "
        "whiskers are one standard error, labels the share of queries the encoder wins",
        {
            "rows": rows,
            "series": [{"key": "lift", "name": "Paired lift", "role": "model"}],
            "whiskers": True,
            "format": "signed:3",
            "valueLabel": "Paired lift",
            "intervalLabel": "One standard error",
            "labelHeader": "Baseline",
            "reference": [{"value": 0, "label": "No lift"}],
            "zero": True,
            "table": {
                "columns": [
                    {"key": "label", "label": "Baseline"},
                    {"key": "mean", "label": "Paired lift", "align": "right", "format": "signed:4"},
                    {"key": "se", "label": "Standard error", "align": "right", "format": "num:4"},
                    {"key": "t", "label": "t", "align": "right", "format": "num:1"},
                    {"key": "win_rate", "label": "Encoder wins", "align": "right", "format": "pct:1"},
                    {"key": "n", "label": "Queries", "align": "right", "format": "int"},
                ],
                "rows": rows,
            },
        },
    )


def _warm_cold(results: Mapping[str, Any], refusals: list[dict[str, str]]) -> dict[str, Any]:
    k = results["k"]
    split = results["warm_cold"]
    warm, cold = split["warm"], split["cold"]
    rows = []
    for key, name, part in (("warm", "Warm start", warm), ("cold", "Cold start", cold)):
        if part is None:
            continue
        rows.append(
            {
                "label": f"{name}, {part['n']} queries",
                "slice": key,
                "values": {"encoder": part["score"], "prior": part["baseline_score"]},
                "n": part["n"],
            }
        )
    if warm is not None and cold is not None:
        title = (
            f"On a company it never trained on, the encoder falls from "
            f"{warm['score']:.2f} to {cold['score']:.2f}"
        )
    else:
        present = warm or cold
        title = f"Only the {'warm' if warm else 'cold'}-start slice can be scored, at {present['score']:.2f}"
    figure = _figure(
        "dot",
        title,
        f"NDCG@{k}, encoder against the popularity prior; warm means the target filed a peer "
        "group inside the training window, cold that it filed none",
        {
            "rows": rows,
            "series": [
                {"key": "encoder", "name": "Peer encoder", "role": "model"},
                {"key": "prior", "name": "Popularity prior", "role": "baseline"},
            ],
            "format": "num:3",
            "gapLabel": "Lift",
            "gapDp": 3,
            "labelHeader": "Queries",
            "zero": True,
            "wide": True,
        },
    )
    for name, part in (("warm-start", warm), ("cold-start", cold)):
        if part is None:
            refusals.append(
                {
                    "what": figure["title"],
                    "why": f"The {name} slice holds too few scorable queries for "
                    "evaluate_ranking's floor, so it is not drawn.",
                }
            )
    return figure


def _margin_text(margin: float, sd: float) -> str:
    if margin > 0:
        if margin < HAIR * sd:
            return f"clears by a hair, {margin:.4f}"
        return f"clears by {margin:.4f}"
    return f"inside by {-margin:.4f}"


def _tower_ablation(results: Mapping[str, Any], refusals: list[dict[str, str]]) -> dict[str, Any]:
    ab = results["ablation"]
    names = {
        CUT_OWN: "Ablation's own cut, {n} folds",
        CUT_HEADLINE: "Headline evaluation's cut, {n} folds",
    }
    rows = []
    table = []
    cleared: dict[str, list[bool]] = {t: [] for t in TOWERS.values()}
    hairs: dict[str, list[bool]] = {t: [] for t in TOWERS.values()}
    for key in (CUT_OWN, CUT_HEADLINE):
        cut = ab["cuts"].get(key)
        if cut is None or "refused" in cut:
            continue
        group = names[key].format(n=cut["n_folds"])
        full = next(r for r in cut["rows"] if r["group"] == "all features")
        table.append(
            {
                "cut": group,
                "cut_key": key,
                "tower": None,
                "label": "All features",
                "score": full["score"],
                "damage": full["damage"],
                "fold_sd": full["fold_sd"],
                "margin": None,
                "n_test": full["n_test"],
            }
        )
        for r in cut["rows"]:
            if r["group"] not in TOWERS:
                continue
            tower = TOWERS[r["group"]]
            margin = r["damage"] - r["fold_sd"]
            cleared[tower].append(margin > 0)
            hairs[tower].append(0 < margin < HAIR * r["fold_sd"])
            label = f"Without {tower}"
            row = {
                "label": label,
                "group": group,
                "cut": key,
                "tower": tower,
                "values": {"damage": r["damage"]},
                "lo": r["damage"] - r["fold_sd"],
                "hi": r["damage"] + r["fold_sd"],
                "text": _margin_text(margin, r["fold_sd"]),
                "tip": [
                    {"label": "Score without it", "value": r["score"], "format": "num:4"},
                    {"label": "Fold standard deviation", "value": r["fold_sd"], "format": "num:4"},
                    {"label": "Damage less deviation", "value": margin, "format": "signed:4"},
                ],
                "score": r["score"],
                "damage": r["damage"],
                "fold_sd": r["fold_sd"],
                "n_test": r["n_test"],
            }
            rows.append(row)
            table.append(
                {
                    "cut": group,
                    "cut_key": key,
                    "tower": tower,
                    "label": label,
                    "score": r["score"],
                    "damage": r["damage"],
                    "fold_sd": r["fold_sd"],
                    "margin": margin,
                    "n_test": r["n_test"],
                }
            )
    every = {t: bool(v) and all(v) for t, v in cleared.items()}
    never = {t: bool(v) and not any(v) for t, v in cleared.items()}
    winners = [t for t in every if every[t]]
    if len(winners) == 1 and all(never[t] for t in every if t not in winners):
        tower = winners[0]
        title = f"Only the {tower} tower clears its fold noise"
        if all(hairs[tower]):
            title += ", and only by a hair"
    elif len(winners) == len(every):
        title = "Both towers clear their fold noise on every cut"
    elif not winners and all(never.values()):
        title = "Neither tower clears its fold noise"
    else:
        title = "Whether a tower clears its fold noise depends on the fold cut"

    panel = ab["panel"]
    empty = set(panel["empty"])
    market_empty = [n for n in panel["market"] if n in empty]
    note = (
        f"{len(panel['empty'])} of the panel's {panel['features']} features are empty on every "
        f"one of its {panel['rows']:,} rows"
        + (
            f", including all {len(market_empty)} market features, "
            if market_empty and len(market_empty) == len(panel["market"])
            else ", "
        )
        + "so the fundamentals tower is judged on a degraded input."
        if panel["empty"]
        else None
    )
    figure = _figure(
        "dot",
        title,
        "Spearman of pair cosine against the disclosed-peer flag; damage is the full model's "
        "score less the score refitted without the tower, whisker one fold standard deviation "
        f"either side. The ablation's own cut starts testing after {ab['min_train_pairs']} "
        "disclosed pairs; the headline cut is the one the NDCG figures use",
        {
            "rows": rows,
            "series": [{"key": "damage", "name": "Damage", "role": "model"}],
            "whiskers": True,
            "format": "signed:3",
            "valueLabel": "Damage",
            "intervalLabel": "One fold standard deviation",
            "labelHeader": "Tower removed",
            "reference": [{"value": 0, "label": "No damage"}],
            "zero": True,
            "note": note,
            "table": {
                "columns": [
                    {"key": "cut", "label": "Fold cut"},
                    {"key": "label", "label": "Configuration"},
                    {"key": "score", "label": "Spearman", "align": "right", "format": "num:4"},
                    {"key": "damage", "label": "Damage", "align": "right", "format": "signed:4"},
                    {"key": "fold_sd", "label": "Fold sd", "align": "right", "format": "num:4"},
                    {"key": "margin", "label": "Damage less sd", "align": "right", "format": "signed:4"},
                    {"key": "n_test", "label": "Test pairs", "align": "right", "format": "int"},
                ],
                "rows": table,
            },
        },
    )
    for key, text in ((CUT_OWN, "the ablation's own fold cut"), (CUT_HEADLINE, "the headline evaluation's fold cut")):
        cut = ab["cuts"].get(key)
        if cut is not None and "refused" in cut:
            refusals.append({"what": figure["title"], "why": f"On {text}: {cut['refused']}"})
    return figure


def _truncation(results: Mapping[str, Any], refusals: list[dict[str, str]]) -> dict[str, Any]:
    k = results["k"]
    tr = results["truncation"]
    scored = [c for c in tr["cuts"] if "methods" in c]
    base = scored[0]
    base_by = {m["method"]: m for m in base["methods"]}
    by_cut = [(c, {m["method"]: m for m in c["methods"]}) for c in scored]
    shortest, shortest_by = by_cut[-1]

    unmoved = [
        m for m in METHOD_NAMES if m in base_by and all(by[m]["ndcg"] == base_by[m]["ndcg"] for _, by in by_cut)
    ]
    enc_drop = base_by["encoder"]["ndcg"] - shortest_by["encoder"]["ndcg"]
    if unmoved:
        title = (
            f"Cutting the excerpts to {shortest['chars']:,} characters costs the encoder "
            f"{enc_drop:.3f}, and {_count(len(unmoved), 'baseline')} not move at all"
        )
    else:
        title = f"Cutting the excerpts to {shortest['chars']:,} characters costs the encoder {enc_drop:.3f}"

    def x_label(c):
        return f"{c['chars']:,}"

    series = [
        {
            "name": _label(method),
            "role": role,
            "values": [
                {"x": x_label(c), "y": by[method]["ndcg"] - base_by[method]["ndcg"]} for c, by in by_cut
            ],
        }
        for method, role in SWEEP_DRAWN
        if method in base_by
    ]
    columns = [{"key": "label", "label": "Method"}]
    for c, _ in by_cut:
        columns.append({"key": f"s{c['chars']}", "label": f"{c['chars']:,} chars", "align": "right", "format": "num:4"})
    columns.append({"key": "change", "label": f"Change at {shortest['chars']:,}", "align": "right", "format": "signed:4"})
    table_rows = []
    for method in METHOD_NAMES:
        if method not in base_by:
            continue
        row: dict[str, Any] = {"label": _label(method), "method": method}
        for c, by in by_cut:
            row[f"s{c['chars']}"] = by[method]["ndcg"]
        row["change"] = shortest_by[method]["ndcg"] - base_by[method]["ndcg"]
        table_rows.append(row)
    counts = sorted({by["encoder"]["n"] for _, by in by_cut})
    queries = (
        f"{counts[0]} queries at every length"
        if len(counts) == 1
        else f"{counts[0]} to {counts[-1]} queries depending on the length"
    )
    origin = (
        f"the committed excerpts, which stop at {tr['longest']:,} characters"
        if base["shortened"] == 0 and base["chars"] >= tr["longest"]
        else f"excerpts cut to {base['chars']:,} characters"
    )
    cutting = [c for c, _ in by_cut if c["shortened"]]
    if cutting and all(c["shortened"] == tr["documents"] for c in cutting):
        shortened = f"every shorter cut shortens all {tr['documents']:,} documents"
    else:
        shortened = "; ".join(
            f"{c['chars']:,} characters shortens {c['shortened']:,} of {tr['documents']:,} documents"
            for c, _ in by_cut
        )
    same = (
        f". Identical at every length: {', '.join(_phrase(m) for m in unmoved)}"
        if unmoved
        else ""
    )
    figure = _figure(
        "line",
        title,
        f"Change in NDCG@{k} from {origin}, on {queries}; {shortened}{same}",
        {
            "series": series,
            "x": {"label": "Excerpt length, characters"},
            "format": "signed:3",
            "zero": True,
            "unmoved": unmoved,
            "table": {"columns": columns, "rows": table_rows},
        },
    )
    for c in tr["cuts"]:
        if "refused" in c:
            refusals.append(
                {"what": figure["title"], "why": f"At {c['chars']:,} characters: {c['refused']}"}
            )
    return figure


REFUSED_NAMES = {
    "warm_cold": "Warm against cold start",
    "tower_ablation": "Tower ablation",
    "truncation": "Truncation sweep",
}


def _takeaway(results: Mapping[str, Any]) -> str:
    h = results["headline"]
    k = results["k"]
    paired = h["paired"]
    baselines = [m for m in results["methods"] if m["method"] != "encoder"]
    best = baselines[0]
    parts = [
        f"The encoder scores {h['score']:.4f} NDCG@{k} against {h['baseline_score']:.4f} for the "
        f"popularity prior"
        + (
            f", and ranks the disclosed peers higher on {paired['win_rate']:.0%} of the "
            f"{paired['n']} test queries."
            if paired
            else "."
        ),
        f"The closest baseline is {_phrase(best['method'])}, at {best['ndcg']:.4f}.",
    ]
    split = results.get("warm_cold")
    if "warm_cold" not in results["refused"] and split and split["warm"] and split["cold"]:
        parts.append(
            f"On a company it never trained on it scores {split['cold']['score']:.4f}, "
            f"against {split['warm']['score']:.4f} on one it did."
        )
    return " ".join(parts)


def shape(results: Mapping[str, Any]) -> dict[str, Any]:
    """The section as ``collect`` returns it, from ``gather``'s plain values."""
    refused = results["refused"]
    refusals: list[dict[str, str]] = []
    figures: dict[str, Any] = {
        "diagnostics": _diagnostics(results),
        "ndcg_by_method": _ndcg_by_method(results),
        "paired_lift": _paired_lift(results),
    }
    builders = {
        "warm_cold": _warm_cold,
        "tower_ablation": _tower_ablation,
        "truncation": _truncation,
    }
    for fid, build in builders.items():
        if fid in refused:
            refusals.append({"what": REFUSED_NAMES[fid], "why": refused[fid]})
        else:
            figures[fid] = build(results, refusals)
    return {
        "status": "ok",
        "takeaway": _takeaway(results),
        "refusals": refusals,
        "headline": _headline(results),
        "figures": figures,
    }


def collect(ctx) -> dict:
    return shape(gather(ctx))
