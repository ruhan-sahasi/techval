"""The M&A propensity section: its shaping on small fakes, and its pinned truths.

Nothing here fits a model. ``tests/ml/test_mna.py`` pins the model; these tests
pin the page. The first group hands ``shape`` invented numbers, chosen to be
unlike the real ones so a title that ignored its inputs would be caught, and
checks that the figures are drawable, that the words follow the numbers, and
that a component which could not be computed becomes a refusal instead of a
figure. The second group reads the committed snapshot, once it exists, and holds
the section to the values audited runs reproduced from the fixtures.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from techval.dashboard.sections import propensity as P
from techval.dashboard.snapshot import to_jsonable, validate_section
from techval.errors import MissingDataError

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "src" / "techval" / "dashboard" / "assets" / "kit.js"
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"
MNA = ROOT / "tests" / "fixtures" / "mna"


def _kit_kinds() -> set[str]:
    source = KIT.read_text(encoding="utf-8")
    body = source[source.index("TV.charts = {") : ]
    body = body[: body.index("};")]
    return set(re.findall(r"^\s*(\w+)\s*:", body, flags=re.M))


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def _evaluation(**over) -> P.Evaluation:
    base = dict(
        metric="walk-forward AUC",
        score=0.61,
        baseline_name="size only (log revenue, smallest first)",
        baseline_score=0.55,
        lift=0.06,
        fold_sd=0.02,
        n_scored=512,
        n_folds=3,
        higher_is_better=True,
        beat_baseline=True,
        verdict_text="The model's own sentence, carried verbatim.",
        chance=0.5,
    )
    base.update(over)
    return P.Evaluation(**base)


def _refit(events: int, score: float, baseline: float, sd: float = 0.03) -> P.Refit:
    return P.Refit(
        events=events,
        score=score,
        baseline_score=baseline,
        lift=score - baseline,
        fold_sd=sd,
        beat_baseline=score > baseline,
    )


def _facts(**over) -> P.Facts:
    base = dict(
        evaluation=_evaluation(),
        sample=P.Sample(
            n_labelled=1200,
            positives=48,
            distinct_deals=19,
            base_rate=0.04,
            scored_positives=22,
            scored_deals=11,
        ),
        folds=(
            P.FoldScore(0, date(2020, 3, 31), date(2020, 12, 31), 200, 8, 0.52, 0.58),
            P.FoldScore(1, date(2021, 3, 31), date(2021, 12, 31), 180, 6, 0.66, 0.54),
            P.FoldScore(2, date(2022, 3, 31), date(2022, 12, 31), 132, 8, 0.63, 0.57),
        ),
        recovered=P.Recovered(
            committed=_refit(60, 0.61, 0.55),
            without_doubted=_refit(58, 0.60, 0.57),
            without_all=_refit(55, 0.59, 0.62),
            n_recovered=5,
            n_doubted=2,
        ),
        coefficients=P.Coefficients(
            fold_labels=("Fold 1", "Fold 2", "Fold 3"),
            fold_train=(90, 240, 400),
            rows=(
                P.Coefficient("alpha_feature", (1.2, -0.3, 0.05), 0.32, 0.76, True),
                P.Coefficient("beta_feature", (-0.4, -0.2, -0.1), -0.23, 0.15, False),
                P.Coefficient("gamma_feature", (0.2, 0.0, -0.6), -0.13, 0.40, True),
            ),
        ),
        base_rates=(
            P.YearRate(2021, 400, 8, 0.02, 4),
            P.YearRate(2022, 400, 20, 0.05, 4),
            P.YearRate(2023, 100, 9, 0.09, 1),
        ),
        calibration=P.Calibration(
            buckets=(
                P.Bucket(0.0, 0.1, 400, 0.04, 0.05, 0.01, False),
                P.Bucket(0.1, 0.2, 60, 0.14, 0.08, -0.06, False),
                P.Bucket(0.3, 0.4, 40, 0.35, 0.10, -0.25, False),
                P.Bucket(0.6, 0.7, 4, 0.65, 0.0, -0.65, True),
            ),
            thin_below=10,
            n=504,
        ),
        top_k=P.TopK(
            k=10,
            dates=6,
            precision_model=0.10,
            precision_size=0.05,
            base_rate=0.04,
            recall_model=0.20,
            recall_size=0.03,
            random_recall=0.05,
        ),
        screen=P.Screen(
            as_of=date(2024, 6, 30),
            n_ranked=210,
            rows=(
                P.ScreenRow(1, "AAA", "Alpha Inc.", "internet", 0.31, ("alpha_feature", "beta_feature")),
                P.ScreenRow(2, "BBB", "Beta Corp", "telecom", 0.22, ("gamma_feature",)),
                P.ScreenRow(3, "CCC", "Gamma plc", "internet", 0.19, ("beta_feature",)),
            ),
        ),
        price=P.PriceLeak(
            auc=0.77,
            positives_departed=0.7,
            negatives_departed=0.1,
            served=(("XYZ", "nasdaq", 400),),
            unserved=("OLD1", "OLD2"),
            fitted_columns=12,
        ),
    )
    base.update(over)
    return P.Facts(**base)


def _numbers(value, path="data"):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _numbers(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _numbers(v, f"{path}[{i}]")


def _as_section(returned: dict) -> dict:
    """What the collector would store, with a provenance row per figure."""
    return to_jsonable(
        {
            "id": P.ID,
            "title": P.TITLE,
            "provenance": [
                {"figure": fid, "entry_point": "techval.ml.mna.fit_propensity", "inputs": [], "seconds": 0.0}
                for fid in returned["figures"]
            ],
            **returned,
        }
    )


# --------------------------------------------------------------------------- #
# Shaping
# --------------------------------------------------------------------------- #


def test_every_figure_is_drawable_and_the_section_meets_the_schema():
    returned = P.shape(_facts())
    kinds = _kit_kinds()
    assert set(returned["figures"]) == set(P.FIGURES)
    for fid, fig in returned["figures"].items():
        assert fig["kind"] in kinds, f"{fid} has kind {fig['kind']!r}"
        assert fig["title"].strip(), fid
        for where, v in _numbers(fig["data"]):
            assert math.isfinite(v), f"{fid} {where} is {v}"
        data = fig["data"]
        if fig["kind"] in ("dot", "hbar", "column"):
            assert data["rows"] and all(str(r["label"]).strip() for r in data["rows"]), fid
        if fig["kind"] == "dot":
            keys = [s["key"] for s in data["series"]]
            assert all(set(keys) <= set(r["values"]) for r in data["rows"]), fid
            assert all("gap" in r and "labelled" in r for r in data["rows"]), fid
        if fig["kind"] == "heat":
            assert len(data["values"]) == len(data["rows"]) == len(data["flips"])
            assert all(len(row) == len(data["cols"]) for row in data["values"])
            assert sum(g["count"] for g in data["groups"]) == len(data["rows"])
        if fig["kind"] == "tiles":
            assert all(t["label"].strip() for t in data["tiles"])
    section = _as_section(returned)
    validate_section(section)
    assert "\u2014" not in json.dumps(section, ensure_ascii=False)


def test_the_headline_is_the_verdict_verbatim_with_its_rule_applied():
    head = P.shape(_facts())["headline"]
    assert head["verdict_text"] == "The model's own sentence, carried verbatim."
    assert head["score"] == 0.61 and head["baseline_score"] == 0.55 and head["n"] == 512
    assert head["verdict_status"] == "beats"
    noisy = P.shape(_facts(evaluation=_evaluation(fold_sd=0.09)))["headline"]
    assert noisy["verdict_status"] == "inside_noise"


@pytest.mark.parametrize(
    "lift, sd, status",
    [
        (0.0212, 0.0910, "inside_noise"),
        (0.2, 0.0910, "beats"),
        (0.05, None, "beats"),
        (-0.001, 0.0910, "loses"),
        (-0.5, 0.01, "loses"),
        (0.0, 0.05, "ties"),
    ],
)
def test_verdict_status_rule(lift, sd, status):
    assert P.verdict_status(lift, sd) == status


def test_the_takeaway_is_built_from_the_values_it_was_given():
    text = P.shape(_facts())["takeaway"]
    assert "0.6100" in text and "0.5500" in text and "+0.0600" in text
    assert "outside a fold standard deviation of 0.0200" in text
    assert "11 acquired companies" in text
    assert "2 of its 3 coefficients change sign" in text
    assert "removing the 5 deals" in text and "flips the verdict" in text

    quiet = P.shape(
        _facts(
            evaluation=_evaluation(fold_sd=0.2),
            recovered=P.Recovered(
                committed=_refit(60, 0.61, 0.55),
                without_doubted=_refit(58, 0.60, 0.57),
                without_all=_refit(55, 0.60, 0.58),
                n_recovered=5,
                n_doubted=2,
            ),
        )
    )["takeaway"]
    assert "inside a fold standard deviation of 0.2000" in quiet
    assert "flips" not in quiet

    losing = P.shape(_facts(evaluation=_evaluation(score=0.5, lift=-0.05, beat_baseline=False)))
    assert "does not beat" in losing["takeaway"]
    assert losing["headline"]["verdict_status"] == "loses"


def test_titles_follow_the_numbers():
    figs = P.shape(_facts())["figures"]
    assert figs["auc_by_fold"]["title"] == "The model beats the size sort in 2 of 3 test folds"
    assert figs["coefficient_signs"]["title"] == "2 of 3 coefficients change sign between folds"
    # The partial year is higher and is not allowed to set the range.
    assert figs["base_rate_by_year"]["title"] == (
        "The base rate more than doubles from 2021 (2.00%) to 2022 (5.00%)"
    )
    assert "2023 holds 1 screen date of 4" in figs["base_rate_by_year"]["subtitle"]
    assert figs["calibration"]["title"] == (
        "Every stated probability above 10% over-promises, by up to 25 points"
    )
    assert figs["precision_at_k"]["title"].startswith("One in 10 on the model's top 10")
    assert figs["recall_at_k"]["title"] == (
        "The size sort's top 10 catch fewer of each date's targets than 10 random names would"
    )
    assert figs["target_list"]["title"] == "2 of the top 3 on 30 June 2024 are internet"
    assert "flips" in figs["recovered_deals"]["title"]
    assert [r["gap"] for r in figs["recovered_deals"]["data"]["rows"]] == pytest.approx(
        [0.06, 0.03, -0.03]
    )


def test_labels_are_selective():
    figs = P.shape(_facts())["figures"]
    folds = figs["auc_by_fold"]["data"]["rows"]
    # The widest fold each way: -0.06 on fold 1 and +0.12 on fold 2.
    assert [r["labelled"] for r in folds] == [True, True, False]
    calibration = figs["calibration"]["data"]["rows"]
    assert [r["labelled"] for r in calibration] == [False, False, True, False]
    assert calibration[2]["gap"] == pytest.approx(0.25)  # stated less realised
    assert calibration[2]["model_gap"] == pytest.approx(-0.25)


def test_the_coefficient_grid_puts_the_sign_changes_first():
    data = P.shape(_facts())["figures"]["coefficient_signs"]["data"]
    assert data["rows"] == ["alpha_feature", "gamma_feature", "beta_feature"]
    assert data["flips"] == [True, True, False]
    assert data["annotate"] == "alpha_feature"
    assert data["n_flipped"] == 2


def test_a_component_that_could_not_be_computed_is_refused_not_drawn():
    returned = P.shape(
        _facts(
            calibration=P.Refused("Calibration", "the scored sample is below the harness floor"),
            top_k=P.Refused("Precision and recall at k", "no screen date held a positive"),
            screen=P.Refused("The target list", "nothing to rank on 2024-06-30"),
            price=None,
        )
    )
    for gone in ("calibration", "precision_at_k", "recall_at_k", "target_list"):
        assert gone not in returned["figures"]
    assert {r["what"] for r in returned["refusals"]} == {
        "Calibration",
        "Precision and recall at k",
        "The target list",
    }
    assert all(r["why"].strip() for r in returned["refusals"])
    validate_section(_as_section(returned))


def test_a_refused_price_probe_leaves_the_list_and_refuses_the_note():
    returned = P.shape(_facts(price=P.Refused("What the target list cannot see", "no probe")))
    assert returned["figures"]["target_list"]["data"]["notes"] == []
    assert returned["refusals"] == [{"what": "What the target list cannot see", "why": "no probe"}]


# --------------------------------------------------------------------------- #
# Identifying the recovered deals
# --------------------------------------------------------------------------- #


def _events(names):
    return [SimpleNamespace(name=n) for n in names]


def test_the_committed_manifest_still_names_the_recovered_deals():
    manifest = json.loads((MNA / "MANIFEST.json").read_text(encoding="utf-8"))
    events = _events(e["name"] for e in json.loads((MNA / "events.json").read_text(encoding="utf-8")))
    seven, doubted = P._recovered_events(manifest, events)
    assert len(seven) == 7 and len(doubted) == 3
    assert len({id(e) for e in seven}) == 7


def test_the_recovered_deals_are_refused_when_the_audit_or_events_disagree():
    manifest = json.loads((MNA / "MANIFEST.json").read_text(encoding="utf-8"))
    names = [e["name"] for e in json.loads((MNA / "events.json").read_text(encoding="utf-8"))]
    with pytest.raises(MissingDataError):
        P._recovered_events({"files": {}}, _events(names))
    recount = json.loads(json.dumps(manifest))
    audit = recount["files"]["events.json"]["audit"]
    recount["files"]["events.json"]["audit"] = audit.replace("Of the 7", "Of the 8")
    with pytest.raises(MissingDataError):
        P._recovered_events(recount, _events(names))
    with pytest.raises(MissingDataError):
        P._recovered_events(manifest, _events(n for n in names if not n.upper().startswith("DISH")))
    with pytest.raises(MissingDataError):
        P._recovered_events(manifest, _events(names + ["DISH Network CORP"]))


# --------------------------------------------------------------------------- #
# Pinned truths, once the snapshot is committed
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def section() -> dict:
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    got = snap["sections"].get(P.ID)
    if got is None or got["status"] != "ok":
        pytest.fail(f"the committed snapshot holds no collected {P.ID} section")
    return got


def test_pinned_headline(section):
    head = section["headline"]
    assert round(head["score"], 4) == 0.5685
    assert round(head["baseline_score"], 4) == 0.5474
    assert round(head["lift"], 4) == 0.0212
    assert head["n"] == 5881
    assert head["verdict_status"] == "inside_noise"
    assert round(section["figures"]["auc_by_fold"]["data"]["fold_sd"], 4) == 0.0910


def test_pinned_sample_and_base_rates(section):
    counts = section["figures"]["sample"]["data"]["counts"]
    assert (counts["n_labelled"], counts["positives"], counts["distinct_deals"]) == (9400, 336, 97)
    assert round(counts["base_rate"] * 100, 2) == 3.57
    years = {r["label"]: r for r in section["figures"]["base_rate_by_year"]["data"]["rows"]}
    assert round(years["2022"]["value"] * 100, 2) == 2.05
    assert round(years["2023"]["value"] * 100, 2) == 4.53
    assert round(years["2025"]["value"] * 100, 2) == 4.36
    assert (years["2025"]["positives"], years["2025"]["n"]) == (16, 367)


def test_pinned_precision_and_recall_at_twenty(section):
    def by_role(fid):
        data = section["figures"][fid]["data"]
        assert data["k"] == 20 and data["aggregation"] == "per-date mean"
        return {r["role"]: r["value"] for r in data["rows"]}, data["reference"][0]["value"]

    precision, base_rate = by_role("precision_at_k")
    assert round(precision["model"], 4) == 0.0658
    assert round(precision["baseline"], 4) == 0.0289
    # 0.03775, which the audited runs quoted half-up as 0.0378; round() on the
    # float lands on the other side of the half, so it is pinned to six places.
    assert base_rate == pytest.approx(0.03775, abs=1e-6)
    recall, _ = by_role("recall_at_k")
    assert round(recall["model"], 4) == 0.0972
    assert round(recall["baseline"], 4) == 0.0385


def test_pinned_coefficient_instability(section):
    data = section["figures"]["coefficient_signs"]["data"]
    assert data["n_flipped"] == 9 and len(data["rows"]) == 15
    i = data["rows"].index("mna_opex_load")
    assert data["flips"][i]
    assert round(data["mean"][i], 2) == 0.65
    assert round(data["sd"][i], 2) == 1.28


def test_pinned_calibration_overconfidence(section):
    rows = section["figures"]["calibration"]["data"]["rows"]
    loud = [r for r in rows if not r["thin"] and r["label"] in ("30% to 40%", "40% to 50%")]
    assert [round(r["model_gap"] * 100, 1) for r in loud] == [-30.5, -36.6]


def test_pinned_recovered_deals_flip(section):
    rows = section["figures"]["recovered_deals"]["data"]["rows"]
    assert rows[0]["beats"] and not rows[-1]["beats"]
    assert rows[0]["label"].endswith("107 deals") and rows[-1]["label"].endswith("100 deals")
