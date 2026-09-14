"""The encoder section: what the page says, held to the values it was given.

Three kinds of test, and none of them fits a model.

**Shaping.** ``shape`` is pure, so it is fed small invented results whose
numbers are deliberately unlike the real ones. The figures must be kinds the
kit draws, carry finite values and labels, pass the snapshot schema, and say
what the values say: a takeaway or a title that did not move when its inputs
moved would be a typed-in sentence.

**Collection with the models stubbed.** ``gather`` records provenance per
figure and drops a figure by raising inside the block that records it. That
contract only holds if the two stay in step, and the schema check in
``collect_section`` is what catches a drift, so the section is collected end to
end with the fits replaced by fakes, including the paths where a part refuses.

**Pinned truths.** Once ``docs/dashboard/snapshot.json`` is committed, the
section's key numbers in it are held to the values reproduced from the
committed fixtures. Until then those tests skip.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from techval.config import Assumptions
from techval.dashboard.collect import CollectContext, collect_section
from techval.dashboard.sections import encoder as S
from techval.dashboard.snapshot import (
    HEADLINE_KEYS,
    VERDICT_STATUSES,
    to_jsonable,
    validate_section,
)
from techval.errors import NotMeaningfulError
from techval.ml.evaluation import Fold
from techval.ml.protocol import EvalResult, PairedDelta
from techval.ml.text import TextCorpus

from dashboard.test_frontend import KIT, _object_keys

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"


# --------------------------------------------------------------------------- #
# invented results, in the shape gather returns
# --------------------------------------------------------------------------- #


def _part(score, prior, n, t=6.0, win=0.7):
    return {
        "metric": "ndcg@10",
        "score": score,
        "baseline_name": "popularity prior (named-by-anybody count, query ignored)",
        "baseline_score": prior,
        "lift": score - prior,
        "n": n,
        "verdict": f"ndcg@10 of {score:.4f} against {prior:.4f}, an invented sentence.",
        "paired": {"mean": score - prior, "se": 0.03, "t": t, "win_rate": win, "n": n},
    }


def _methods(scale=1.0):
    base = {
        "encoder": 0.6123,
        "text cosine": 0.4411,
        "fundamentals cosine": 0.3902,
        "text lsa cosine": 0.3001,
        "size and growth": 0.2702,
        "popularity prior": 0.2001,
        "same sub-vertical": 0.1500,
    }
    moves = {"encoder", "text cosine", "text lsa cosine", "same sub-vertical"}
    rows = [
        {"method": m, "ndcg": v * (scale if m in moves else 1.0), "n": 90}
        for m, v in base.items()
    ]
    return sorted(rows, key=lambda r: -r["ndcg"])


def _ablation_cut(text_damage, text_sd, fund_damage, fund_sd, n_folds=4):
    return {
        "n_folds": n_folds,
        "rows": [
            {"group": "all features", "score": 0.5, "damage": 0.0, "fold_sd": 0.09, "n_test": 900},
            {"group": "without text", "score": 0.5 - text_damage, "damage": text_damage, "fold_sd": text_sd, "n_test": 900},
            {"group": "without fundamentals", "score": 0.5 - fund_damage, "damage": fund_damage, "fold_sd": fund_sd, "n_test": 900},
        ],
    }


def results(**changes):
    out = {
        "refused": {},
        "k": 10,
        "n_queries": 90,
        "methods": _methods(),
        "random_order": 0.0911,
        "headline": _part(0.6123, 0.2001, 90, t=9.1, win=0.9),
        "popularity": "popularity prior",
        "paired": [
            {"baseline": b, "mean": m, "se": 0.02, "t": m / 0.02, "win_rate": w, "n": 90}
            for b, m, w in (
                ("same sub-vertical", 0.46, 0.93),
                ("popularity prior", 0.41, 0.9),
                ("size and growth", 0.34, 0.88),
                ("text lsa cosine", 0.31, 0.81),
                ("fundamentals cosine", 0.22, 0.79),
                ("text cosine", 0.17, 0.71),
            )
        ],
        "diagnostics": {"collapse": 0.2222, "in_universe": 0.77, "n_folds": 4, "embargo_days": 365},
        "warm_cold": {"warm": _part(0.7111, 0.3, 60), "cold": _part(0.3333, 0.12, 30)},
        "ablation": {
            "cuts": {
                "ablation": _ablation_cut(0.05, 0.049, 0.02, 0.06),
                "headline": _ablation_cut(0.06, 0.0595, 0.01, 0.07),
            },
            "min_train_pairs": 200,
            "panel": {
                "features": 50,
                "empty": ["market_beta", "market_momentum_3m", "capital_debt_to_capital"],
                "market": ["market_beta", "market_momentum_3m"],
                "rows": 400,
            },
        },
        "truncation": {
            "documents": 300,
            "longest": 2500,
            "cuts": [
                {"chars": 2500, "shortened": 0, "methods": _methods()},
                {"chars": 1500, "shortened": 300, "methods": _methods(0.99)},
                {"chars": 800, "shortened": 300, "methods": _methods(0.95)},
            ],
        },
    }
    out.update(changes)
    return out


def _schema_checked(section: dict) -> dict:
    """The section as collect would write it, with one provenance row per figure."""
    full = to_jsonable(
        {
            "id": S.ID,
            "title": S.TITLE,
            **section,
            "provenance": [
                {"figure": fid, "entry_point": S.EVALUATE, "inputs": [], "seconds": 0.0}
                for fid in section["figures"]
            ],
        }
    )
    validate_section(full)
    return full


def _numbers(value, path="data"):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _numbers(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _numbers(v, f"{path}[{i}]")


# --------------------------------------------------------------------------- #
# shaping
# --------------------------------------------------------------------------- #


def test_every_figure_is_a_kit_chart_with_finite_values_and_labels():
    kinds = _object_keys(KIT.read_text(encoding="utf-8"), "TV.charts")
    raw = S.shape(results())
    section = _schema_checked(copy.deepcopy(raw))
    assert set(section["figures"]) == {
        "diagnostics", "ndcg_by_method", "paired_lift", "warm_cold", "tower_ablation", "truncation",
    }
    for fid, fig in section["figures"].items():
        assert fig["kind"] in kinds, fid
        assert fig["title"].strip(), fid
        if fig["kind"] != "tiles":
            assert fig["subtitle"].strip(), fid
        # Before serialisation, which would quietly write a NaN as null.
        for path, number in _numbers(raw["figures"][fid]["data"]):
            assert math.isfinite(number), f"{fid} {path}"
        for row in fig["data"].get("rows", []):
            assert str(row["label"]).strip(), fid
        for series in fig["data"].get("series", []):
            assert series["name"] and series["role"] in ("model", "baseline", "alt"), fid
        for text in (fig["title"], fig["subtitle"]):
            assert "\u2014" not in text


def test_the_fitted_model_is_the_only_blue_mark_and_the_rule_is_the_random_floor():
    figs = S.shape(results())["figures"]
    rows = figs["ndcg_by_method"]["data"]["rows"]
    assert [r["role"] for r in rows if r["method"] == "encoder"] == ["model"]
    assert {r["role"] for r in rows if r["method"] != "encoder"} == {"baseline"}
    assert figs["ndcg_by_method"]["data"]["reference"] == [{"value": 0.0911, "label": "Random order"}]
    roles = {s["name"]: s["role"] for s in figs["truncation"]["data"]["series"]}
    assert roles == {"Peer encoder": "model", "Text cosine": "alt", "Fundamentals cosine": "baseline"}
    assert len(figs["truncation"]["data"]["series"]) <= 4
    # The table view carries every method, not only the drawn lines.
    assert len(figs["truncation"]["data"]["table"]["rows"]) == len(S.METHOD_NAMES)


def test_the_takeaway_and_titles_are_built_from_the_values_they_describe():
    section = S.shape(results())
    takeaway = section["takeaway"]
    for text in ("0.6123", "0.2001", "90%", "text cosine, at 0.4411", "0.3333", "0.7111"):
        assert text in takeaway
    figs = section["figures"]
    assert figs["ndcg_by_method"]["title"].startswith("Text cosine comes closest")
    assert figs["paired_lift"]["title"].endswith("narrowest against text cosine")
    assert "from 0.71 to 0.33" in figs["warm_cold"]["title"]
    assert "800 characters costs the encoder 0.031" in figs["truncation"]["title"]
    assert "three baselines do not move" in figs["truncation"]["title"]

    moved = results(methods=_methods(), headline=_part(0.7777, 0.1234, 90, t=9.0, win=0.55))
    moved["methods"][0]["ndcg"] = 0.7777
    again = S.shape(moved)
    assert "0.7777" in again["takeaway"] and "55%" in again["takeaway"]
    assert "0.6123" not in again["takeaway"]


def test_the_headline_follows_the_schema_and_quotes_the_models_verdict():
    headline = S.shape(results())["headline"]
    assert tuple(headline) == HEADLINE_KEYS
    assert headline["verdict_text"] == results()["headline"]["verdict"]
    assert headline["metric"] == "NDCG@10"
    assert headline["baseline_name"] == "popularity prior"
    assert headline["verdict_status"] == "beats"
    assert headline["lift"] == pytest.approx(0.6123 - 0.2001)


@pytest.mark.parametrize(
    "lift, t, status",
    [
        (0.3, 13.6, "beats"),
        (0.3, float("inf"), "beats"),
        (0.02, 2.0, "not_significant"),
        (0.02, 1.2, "not_significant"),
        (0.02, None, "not_significant"),
        (0.02, float("nan"), "not_significant"),
        (0.0, 0.0, "ties"),
        (-0.1, -2.5, "loses"),
    ],
)
def test_the_verdict_rule_reads_the_paired_t(lift, t, status):
    assert S.verdict_status(lift, t) == status
    assert status in VERDICT_STATUSES


def test_a_refused_figure_is_a_refusal_with_its_reason_and_not_a_figure():
    r = results()
    r["refused"]["truncation"] = "fewer than two excerpt lengths could be scored"
    del r["truncation"]
    section = _schema_checked(S.shape(r))
    assert "truncation" not in section["figures"]
    assert {"what": "Truncation sweep", "why": "fewer than two excerpt lengths could be scored"} in section["refusals"]


def test_one_refused_fold_cut_leaves_the_other_drawn_with_a_note_under_it():
    r = results()
    r["ablation"]["cuts"]["headline"] = {"refused": "fold 0 has 9 disclosed pairs", "n_folds": 4}
    section = _schema_checked(S.shape(r))
    fig = section["figures"]["tower_ablation"]
    assert {row["cut"] for row in fig["data"]["rows"]} == {"ablation"}
    notes = [x for x in section["refusals"] if x["what"] == fig["title"]]
    assert len(notes) == 1 and "fold 0 has 9 disclosed pairs" in notes[0]["why"]


def test_a_missing_start_slice_is_noted_and_the_other_is_still_drawn():
    r = results()
    r["warm_cold"]["cold"] = None
    section = _schema_checked(S.shape(r))
    fig = section["figures"]["warm_cold"]
    assert [row["slice"] for row in fig["data"]["rows"]] == ["warm"]
    assert any(x["what"] == fig["title"] and "cold-start" in x["why"] for x in section["refusals"])
    assert "never trained on" not in section["takeaway"]


def test_the_ablation_says_by_a_hair_only_when_the_margin_is_a_hair():
    fig = S.shape(results())["figures"]["tower_ablation"]
    assert fig["title"] == "Only the text tower clears its fold noise, and only by a hair"
    texts = {(row["cut"], row["tower"]): row["text"] for row in fig["data"]["rows"]}
    assert texts[("ablation", "text")] == "clears by a hair, 0.0010"
    assert texts[("headline", "text")] == "clears by a hair, 0.0005"
    assert texts[("ablation", "fundamentals")] == "inside by 0.0400"
    # The whisker is the damage plus and minus the fold sd, so it clears zero exactly when the text says so.
    for row in fig["data"]["rows"]:
        assert row["lo"] == pytest.approx(row["damage"] - row["fold_sd"])
        assert (row["lo"] > 0) == row["text"].startswith("clears")
    assert fig["data"]["note"].startswith("3 of the panel's 50 features are empty")
    assert "all 2 market features" in fig["data"]["note"]

    wide = results()
    wide["ablation"]["cuts"]["ablation"] = _ablation_cut(0.09, 0.03, 0.02, 0.06)
    wide["ablation"]["cuts"]["headline"] = _ablation_cut(0.09, 0.04, 0.01, 0.07)
    fig = S.shape(wide)["figures"]["tower_ablation"]
    assert fig["title"] == "Only the text tower clears its fold noise"

    split = results()
    split["ablation"]["cuts"]["headline"] = _ablation_cut(0.01, 0.04, 0.01, 0.07)
    assert S.shape(split)["figures"]["tower_ablation"]["title"] == (
        "Whether a tower clears its fold noise depends on the fold cut"
    )


def test_the_sweep_names_exactly_the_methods_that_did_not_move():
    fig = S.shape(results())["figures"]["truncation"]
    assert fig["data"]["unmoved"] == ["popularity prior", "size and growth", "fundamentals cosine"]
    assert "Identical at every length: popularity prior, size and growth, fundamentals cosine" in fig["subtitle"]
    flat = next(s for s in fig["data"]["series"] if s["role"] == "baseline")
    assert [p["y"] for p in flat["values"]] == [0.0, 0.0, 0.0]
    assert [p["x"] for p in flat["values"]] == ["2,500", "1,500", "800"]


# --------------------------------------------------------------------------- #
# collection with the models stubbed
# --------------------------------------------------------------------------- #


def _fake_evaluation(shortest: int):
    """A PeerEvaluation-shaped object whose encoder score moves with document length."""
    k = 10
    encoder = 0.5 + shortest / 100_000
    methods = [
        ("encoder", encoder), ("fundamentals cosine", 0.37), ("text cosine", 0.2 + shortest / 200_000),
        ("text lsa cosine", 0.28), ("size and growth", 0.25), ("popularity prior", 0.22),
        ("same sub-vertical", 0.2),
    ]
    scores = pd.DataFrame(
        [{"method": m, f"ndcg@{k}": v, "ndcg_sd": 0.2, "n_queries": 40} for m, v in methods]
    ).sort_values(f"ndcg@{k}", ascending=False, ignore_index=True)
    paired = pd.DataFrame(
        [
            {"baseline": m, "mean_difference": encoder - v, "sd_of_difference": 0.1,
             "standard_error": 0.016, "t": (encoder - v) / 0.016, "win_rate": 0.7, "n_queries": 40}
            for m, v in methods[1:]
        ]
    ).sort_values("mean_difference", ascending=False, ignore_index=True)

    def result(score, base, n):
        return EvalResult(
            metric=f"ndcg@{k}", score=score, baseline_name="popularity prior", baseline_score=base,
            n_observations=n, folds=[score] * n, fold_unit="query",
            paired=PairedDelta(score - base, 0.1, 0.016, (score - base) / 0.016, 0.7, n),
        )

    fold = Fold(0, date(2022, 1, 1), date(2023, 1, 1), date(2023, 6, 1), 300, 40, 365, 0)
    return SimpleNamespace(
        k=k, n_queries=40, scores=scores, paired=paired, folds=[fold, fold],
        against={"random order": result(encoder, 0.08, 40)},
        headline=result(encoder, 0.22, 40), warm=result(0.6, 0.27, 25), cold=result(0.35, 0.13, 15),
        collapse=0.15, coverage={"in_universe": 0.85},
    )


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    root = tmp_path / "fixtures"
    root.mkdir()
    for name in S.INPUTS:
        (root / name).write_text("{}")
    corpora = {
        date(2024, 1, 1): TextCorpus(
            tickers=["AAA", "BBB"],
            documents=["a" * 2500, "b" * 2400],
            as_of_by_ticker={"AAA": date(2023, 3, 1), "BBB": date(2023, 3, 1)},
            source_accessions={},
        )
    }
    panel = SimpleNamespace(rows=[SimpleNamespace(values={"margin_gross": 0.5, "market_beta": None})])
    monkeypatch.setattr(S, "_load", lambda ctx: ([], panel, corpora))
    monkeypatch.setattr(
        S, "build_dataset", lambda groups, panel, corpora: SimpleNamespace(corpora=corpora)
    )
    monkeypatch.setattr(
        S,
        "evaluate_peer_encoder",
        lambda ds, a, k: _fake_evaluation(min(len(d) for c in ds.corpora.values() for d in c.documents)),
    )
    monkeypatch.setattr(S, "ablation_folds", lambda ds, a: [1, 2, 3])
    frame = pd.DataFrame(_ablation_cut(0.05, 0.049, 0.02, 0.06)["rows"])
    monkeypatch.setattr(S, "ablate_towers", lambda ds, a, folds=None: frame.copy())
    assumptions = Assumptions()
    assumptions.ml.cache_dir = str(tmp_path / "cache")

    def run():
        ctx = CollectContext(root, tmp_path, assumptions)
        section, _ = collect_section(S, ctx, use_cache=False)
        return section

    return SimpleNamespace(run=run, monkeypatch=monkeypatch)


def test_collecting_with_the_models_stubbed_keeps_figures_and_provenance_in_step(stubbed):
    section = stubbed.run()
    assert section["status"] == "ok"
    assert section["refusals"] == []
    sourced = {row["figure"]: row["entry_point"] for row in section["provenance"]}
    assert set(sourced) == set(section["figures"])
    assert sourced["tower_ablation"] == S.ABLATE
    assert sourced["truncation"] == S.EVALUATE
    for row in section["provenance"]:
        assert row["inputs"] == [f"fixtures/{name}" for name in S.INPUTS]
    sweep = section["figures"]["truncation"]["data"]["table"]["rows"]
    encoder = next(r for r in sweep if r["method"] == "encoder")
    # 2,500 shortens nothing and reuses the evaluation; the shorter cuts refit on cut text.
    assert (encoder["s2500"], encoder["s1500"], encoder["s800"]) == pytest.approx((0.524, 0.515, 0.508))
    assert section["figures"]["tower_ablation"]["data"]["note"].startswith(
        f"{len(S.FEATURE_NAMES) - 1} of the panel's {len(S.FEATURE_NAMES)} features are empty"
    )


def test_a_refusal_inside_gather_drops_the_figure_and_its_provenance_together(stubbed):
    def refuse(ds, a, folds=None):
        raise NotMeaningfulError("too few disclosed pairs to fit")

    stubbed.monkeypatch.setattr(S, "ablate_towers", refuse)
    real = S.evaluate_peer_encoder

    def refuse_short(ds, a, k):
        if min(len(d) for c in ds.corpora.values() for d in c.documents) < 2400:
            raise NotMeaningfulError("no fold produced a scorable query")
        return real(ds, a, k)

    stubbed.monkeypatch.setattr(S, "evaluate_peer_encoder", refuse_short)
    section = stubbed.run()
    assert section["status"] == "ok"
    assert "tower_ablation" not in section["figures"]
    assert "truncation" not in section["figures"]
    assert {row["figure"] for row in section["provenance"]} == set(section["figures"])
    whats = {r["what"]: r["why"] for r in section["refusals"]}
    assert "too few disclosed pairs to fit" in whats["Tower ablation"]
    assert "no fold produced a scorable query" in whats["Truncation sweep"]


def test_a_refused_evaluation_refuses_the_whole_section(stubbed):
    def refuse(ds, a, k):
        raise NotMeaningfulError("no fold produced a scorable query")

    stubbed.monkeypatch.setattr(S, "evaluate_peer_encoder", refuse)
    section = stubbed.run()
    assert section["status"] == "refused"
    assert section["figures"] == {} and section["provenance"] == []
    assert "no fold produced a scorable query" in section["refusals"][0]["why"]


# --------------------------------------------------------------------------- #
# pinned truths, against the committed snapshot
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def committed() -> dict:
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    section = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"]["encoder"]
    assert section["status"] == "ok", section["refusals"]
    return copy.deepcopy(section)


def _by(rows, key, value):
    return next(r for r in rows if r.get(key) == value)


def test_pinned_headline(committed):
    h = committed["headline"]
    assert h["score"] == pytest.approx(0.5407, abs=5e-5)
    assert h["baseline_score"] == pytest.approx(0.2213, abs=5e-5)
    assert h["n"] == 162
    assert h["verdict_status"] == "beats"
    assert "(t 13.6)" in h["verdict_text"] and "wins 84% of 162 queries" in h["verdict_text"]


def test_pinned_every_method_on_one_axis(committed):
    data = committed["figures"]["ndcg_by_method"]["data"]
    expected = {
        "encoder": 0.5407,
        "fundamentals cosine": 0.3729,
        "text cosine": 0.3108,
        "text lsa cosine": 0.2824,
        "size and growth": 0.2486,
        "popularity prior": 0.2213,
        "same sub-vertical": 0.2015,
    }
    got = {r["method"]: r["value"] for r in data["rows"]}
    assert got == pytest.approx(expected, abs=5e-5)
    assert data["reference"][0]["value"] == pytest.approx(0.080, abs=5e-4)


def test_pinned_paired_lift_and_diagnostics(committed):
    rows = committed["figures"]["paired_lift"]["data"]["rows"]
    pop = _by(rows, "method", "popularity prior")
    assert pop["mean"] == pytest.approx(0.32, abs=5e-3)
    assert pop["t"] == pytest.approx(13.6, abs=0.05)
    assert pop["win_rate"] == pytest.approx(0.84, abs=5e-3)
    fund = _by(rows, "method", "fundamentals cosine")
    assert fund["mean"] == pytest.approx(0.168, abs=5e-4)
    assert fund["t"] == pytest.approx(7.9, abs=0.05)
    assert fund["win_rate"] == pytest.approx(0.741, abs=5e-4)
    collapse = committed["figures"]["diagnostics"]["data"]["tiles"][0]
    assert collapse["value"] == pytest.approx(0.146, abs=5e-4)


def test_pinned_warm_and_cold_start(committed):
    rows = committed["figures"]["warm_cold"]["data"]["rows"]
    warm, cold = _by(rows, "slice", "warm"), _by(rows, "slice", "cold")
    assert (warm["n"], cold["n"]) == (106, 56)
    assert warm["values"]["encoder"] == pytest.approx(0.6307, abs=5e-5)
    assert cold["values"]["encoder"] == pytest.approx(0.3702, abs=5e-5)
    assert cold["values"]["prior"] == pytest.approx(0.1360, abs=5e-5)


def test_pinned_tower_ablation_on_both_cuts(committed):
    rows = committed["figures"]["tower_ablation"]["data"]["table"]["rows"]

    def row(cut, tower):
        return next(r for r in rows if r["cut_key"] == cut and r["tower"] == tower)

    full = row("ablation", None)
    assert (full["score"], full["fold_sd"]) == pytest.approx((0.4241, 0.0850), abs=5e-5)
    text = row("ablation", "text")
    assert (text["score"], text["damage"], text["fold_sd"]) == pytest.approx((0.3598, 0.0643, 0.0636), abs=5e-5)
    assert 0 < text["margin"] < 0.001
    fund = row("ablation", "fundamentals")
    assert (fund["score"], fund["damage"], fund["fold_sd"]) == pytest.approx((0.3854, 0.0387, 0.0690), abs=5e-5)
    assert fund["margin"] < 0
    assert row("headline", "text")["damage"] == pytest.approx(0.0694, abs=5e-5)
    assert row("headline", "fundamentals")["damage"] == pytest.approx(0.0274, abs=5e-5)


def test_pinned_truncation_sweep(committed):
    rows = committed["figures"]["truncation"]["data"]["table"]["rows"]
    encoder = _by(rows, "method", "encoder")
    assert (encoder["s2500"], encoder["s1500"], encoder["s800"]) == pytest.approx(
        (0.5407, 0.5342, 0.5127), abs=5e-5
    )
    for method in ("popularity prior", "size and growth", "fundamentals cosine"):
        r = _by(rows, "method", method)
        assert r["s2500"] == r["s1500"] == r["s800"], method
