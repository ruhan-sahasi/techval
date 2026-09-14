"""The warranted-multiple section: the shaping on small fakes, and the page's pinned numbers.

Nothing here fits the model. ``tests/ml/test_warranted.py`` pins the fit; these
tests pin what the page makes of a fit. The first group feeds ``shape`` hand-built
``Inputs`` and checks that every figure is well formed, that the takeaway and the
chip follow the values they were given, and that a figure the values cannot
support becomes a refusal with a reason. The second group reads the committed
snapshot and holds the section to the numbers the audited runs reproduced from
the committed panel, so a change that moves one of them has to move this file too.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest

from techval.dashboard.collect import entry_module
from techval.dashboard.sections import warranted as S
from techval.dashboard.snapshot import to_jsonable, validate_section

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "src" / "techval" / "dashboard" / "assets" / "kit.js"
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"

CARD = "comps.py OLS refit on the date, sub-vertical peers"
UNIVERSE = "comps.py OLS refit on the date, whole TMT cross-section"
MEDIAN = "sub-vertical median multiple on the date (the comps convention)"
PERSISTENCE = (
    "the company's own multiple last quarter (persistence: NOT a warranted "
    "multiple, it reads the answer)"
)


def _kit_charts() -> set[str]:
    source = KIT.read_text(encoding="utf-8")
    body = re.search(r"TV\.charts = \{(.*?)\};", source, flags=re.S).group(1)
    return set(re.findall(r"(\w+)\s*:", body))


def _folds(pairs):
    starts = [date(2022, 12, 31), date(2023, 12, 31), date(2024, 12, 31), date(2025, 12, 31)]
    return tuple(
        S.FoldResult(
            index=i,
            test_start=starts[i],
            test_end=date(starts[i].year + 1, 9, 30),
            n=100 + i,
            model_score=m,
            baseline_score=b,
        )
        for i, (m, b) in enumerate(pairs)
    )


def _inputs(**overrides) -> S.Inputs:
    values = dict(
        target_label="EV/Revenue",
        model_kind="mlp",
        metric="spearman",
        higher_is_better=True,
        comparisons=(
            S.Comparison(CARD, 0.70, 0.60, 400, True),
            S.Comparison(MEDIAN, 0.80, 0.55, 700, False),
            S.Comparison(PERSISTENCE, 0.82, 0.95, 750, False),
        ),
        verdict_text="The model's own verdict sentence, carried verbatim.",
        fold_score_sd=0.05,
        folds=_folds([(0.80, 0.50), (0.75, 0.70), (0.60, 0.65)]),
        change=0.03,
        n_changes=750,
        pooled_on_change_rows=0.82,
        screen_date=date(2026, 6, 30),
        screen_names=40,
        screen=(
            S.ScreenRow("AAA", "semiconductors", 20.0, 8.0, 0.9, 1.0),
            S.ScreenRow("BBB", "semiconductors", 15.0, 9.0, 0.5, 0.6),
            S.ScreenRow("CCC", "application_software", 3.0, 9.0, -1.1, -1.2),
            S.ScreenRow("DDD", "application_software", 2.0, 7.0, -1.3, -1.4),
        ),
        date_means=((date(2021, 12, 31), 1.85), (date(2022, 12, 31), 1.35), (date(2023, 12, 31), 1.65)),
        between_date_share=0.021,
        within_date_sd=0.95,
        n_observations=300,
        n_companies=30,
        n_dates=3,
        n_refused=12,
    )
    values.update(overrides)
    return S.Inputs(**values)


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


# --------------------------------------------------------------------------- #
# Shaping, on fakes
# --------------------------------------------------------------------------- #


def test_every_figure_has_a_kind_the_kit_draws_finite_values_and_labels():
    section = S.shape(_inputs())
    charts = _kit_charts()
    assert section["status"] == "ok"
    assert section["refusals"] == []
    assert list(section["figures"]) == [fid for fid, _, _ in S.FIGURES]
    for fid, fig in section["figures"].items():
        assert fig["kind"] in charts, fid
        assert fig["title"].strip(), fid
        if fig["kind"] != "tiles":
            assert fig["subtitle"].strip(), fid
        for where, number in _numbers(fig["data"]):
            assert math.isfinite(number), f"{fid} {where}"
        data = fig["data"]
        for row in data.get("rows", []):
            assert str(row["label"]).strip(), fid
        for series in data.get("series", []):
            assert series["name"].strip(), fid
        for tile in data.get("tiles", []):
            assert tile["label"].strip(), fid
        table = data.get("table")
        if table:
            keys = {c["key"] for c in table["columns"]}
            assert table["rows"] and all(keys <= set(r) for r in table["rows"]), fid


def test_the_shaped_section_passes_the_snapshot_schema_and_carries_no_em_dash():
    section = S.shape(_inputs())
    full = to_jsonable(
        {
            "id": S.ID,
            "title": S.TITLE,
            **section,
            "provenance": [
                {"figure": fid, "entry_point": entry, "inputs": [], "seconds": 0.0}
                for fid, entry, _ in S.FIGURES
            ],
        }
    )
    validate_section(full)
    assert "\N{EM DASH}" not in json.dumps(full, ensure_ascii=False)


def test_every_entry_point_the_section_cites_resolves():
    for _, entry, _ in S.FIGURES:
        entry_module(entry)


def test_the_takeaway_is_built_from_the_values_it_was_given():
    text = S.takeaway(_inputs())
    assert "0.7000 against 0.6000 on 400 observations" in text
    assert "+0.0300 on 750" in text
    assert "almost all of the ranking is company identity" in text

    changed = S.takeaway(_inputs(change=0.25, n_changes=612))
    assert "+0.2500 on 612" in changed
    assert "survives differencing" in changed
    assert "almost all" not in changed

    losing = _inputs(comparisons=(S.Comparison(CARD, 0.55, 0.60, 400, True),))
    assert "does not rank" in S.takeaway(losing)


def test_the_chip_follows_the_stricter_noise_test():
    # Fold lifts +0.30, +0.05, -0.05: mean 0.10 against a spread of 0.18.
    assert S.verdict_status(_inputs()) == "inside_noise"
    steady = _inputs(folds=_folds([(0.80, 0.60), (0.78, 0.60), (0.79, 0.60)]))
    assert S.verdict_status(steady) == "beats"
    losing = _inputs(comparisons=(S.Comparison(CARD, 0.55, 0.60, 400, True),))
    assert S.verdict_status(losing) == "loses"
    one_fold = _inputs(folds=_folds([(0.80, 0.60)]))
    assert S.verdict_status(one_fold) == "not_significant"


def test_the_headline_is_the_card_and_the_verdict_is_verbatim():
    head = S.headline(_inputs())
    assert head["score"] == 0.70 and head["baseline_score"] == 0.60 and head["n"] == 400
    assert head["baseline_name"] == CARD
    assert head["lift"] == pytest.approx(0.10)
    assert head["verdict_text"] == "The model's own verdict sentence, carried verbatim."


def test_a_stronger_baseline_left_off_the_card_is_marked_and_explained():
    fig = S.build_baselines(_inputs())
    labels = {row["key"]: row["label"] for row in fig["data"]["rows"]}
    assert "the card's baseline" in labels[CARD]
    assert "left off the card" in labels[PERSISTENCE]
    assert "left off the card" not in labels[MEDIAN]
    assert "carried forward" in fig["subtitle"]
    assert [row["key"] for row in fig["data"]["rows"]][0] == CARD
    assert [row["key"] for row in fig["data"]["rows"]][-1] == PERSISTENCE


def test_the_noise_bars_are_each_lift_plus_and_minus_its_own_yardstick():
    fig = S.build_noise(_inputs())
    shipped, strict = fig["data"]["rows"]
    assert shipped["mid"] == pytest.approx(0.10)
    assert shipped["hi"] - shipped["mid"] == pytest.approx(0.05)
    assert strict["mid"] == pytest.approx(0.10)
    assert strict["hi"] - strict["mid"] == pytest.approx(math.sqrt(0.0325))
    calls = [r["call"] for r in fig["data"]["table"]["rows"]]
    assert calls == ["outside the noise, above zero", "inside the noise"]
    assert fig["title"].startswith("Outside the noise by the verdict sentence")


def test_the_screen_is_diverging_and_richest_first():
    fig = S.build_screen(_inputs())
    rows = fig["data"]["rows"]
    assert [r["label"] for r in rows] == ["AAA", "BBB", "CCC", "DDD"]
    assert [r["role"] for r in rows] == ["pos", "pos", "neg", "neg"]
    assert "never that the market is wrong" in fig["subtitle"]
    assert fig["title"].startswith("Semiconductors crowd the rich end")


def test_a_figure_the_values_cannot_support_is_refused_with_a_reason():
    section = S.shape(_inputs(change=None, pooled_on_change_rows=None, n_changes=12, screen=(), folds=_folds([(0.8, 0.6)])))
    refused = {r["what"]: r["why"] for r in section["refusals"]}
    assert set(refused) == {"deflation", "folds", "noise", "screen"}
    assert all(why.strip() for why in refused.values())
    assert "12" in refused["deflation"]
    for fid in refused:
        assert fid not in section["figures"]
    assert {"baselines", "rerating", "panel"} <= set(section["figures"])
    assert "could not be computed" in section["takeaway"]


def test_provenance_is_recorded_only_for_the_figures_that_were_built():
    recorded = []

    @contextmanager
    def record(fid, entry):
        yield
        recorded.append((fid, entry))

    section = S.shape(_inputs(screen=()), record=record)
    assert [fid for fid, _ in recorded] == list(section["figures"])
    assert "screen" not in dict(recorded)


# --------------------------------------------------------------------------- #
# Pinned truths, against the committed snapshot
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def committed() -> dict:
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    section = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"][S.ID]
    assert section["status"] == "ok", section["refusals"]
    return section


def _at4(value, expected):
    assert round(value, 4) == pytest.approx(expected, abs=1e-9), (value, expected)


def test_the_headline_is_the_card_against_the_incumbent_ols(committed):
    head = committed["headline"]
    _at4(head["score"], 0.7651)
    _at4(head["baseline_score"], 0.6313)
    _at4(head["lift"], 0.1339)
    assert head["n"] == 888
    assert head["baseline_name"] == CARD
    assert head["verdict_status"] == "inside_noise"
    assert "outside the fold-to-fold noise" in head["verdict_text"]


def test_every_baseline_comparison_is_pinned(committed):
    rows = {r["key"]: r for r in committed["figures"]["baselines"]["data"]["rows"]}
    expected = {
        CARD: (0.7651, 0.6313, 888),
        UNIVERSE: (0.8391, 0.4480, 1500),
        MEDIAN: (0.8294, 0.6180, 1441),
        PERSISTENCE: (0.8412, 0.9722, 1548),
    }
    assert set(rows) == set(expected)
    for name, (model, baseline, n) in expected.items():
        _at4(rows[name]["values"]["model"], model)
        _at4(rows[name]["values"]["baseline"], baseline)
        assert rows[name]["n"] == n


def test_the_differenced_score_is_pinned_beside_the_pooled_one(committed):
    rows = {r["key"]: r for r in committed["figures"]["deflation"]["data"]["rows"]}
    _at4(rows["differenced"]["value"], 0.0420)
    _at4(rows["pooled"]["value"], 0.8412)
    assert rows["differenced"]["n"] == rows["pooled"]["n"] == 1548
    assert "+0.0420 on 1,548" in committed["takeaway"]


def test_the_folds_and_both_noise_tests_are_pinned(committed):
    folds = committed["figures"]["folds"]["data"]["rows"]
    for row, expected in zip(folds, (0.8086, 0.7981, 0.7720, 0.8515, 0.6306)):
        _at4(row["values"]["model"], expected)
    lifts = [r["values"]["model"] - r["values"]["baseline"] for r in folds]
    assert len(folds) == 5 and sum(1 for v in lifts if v > 0) == 4

    noise = {r["key"]: r for r in committed["figures"]["noise"]["data"]["rows"]}
    _at4(noise["shipped"]["mid"], 0.1339)
    _at4(noise["shipped"]["hi"] - noise["shipped"]["mid"], 0.0841)
    _at4(noise["strict"]["mid"], 0.1438)
    _at4(noise["strict"]["hi"] - noise["strict"]["mid"], 0.1736)


def test_the_panel_and_the_rerating_are_pinned(committed):
    tiles = {t["key"]: t["value"] for t in committed["figures"]["panel"]["data"]["tiles"]}
    assert tiles == {"observations": 1764, "companies": 94, "dates": 22, "refused": 546}
    rerating = committed["figures"]["rerating"]["data"]
    assert round(rerating["between_date_share"], 3) == pytest.approx(0.021)
    assert len(rerating["series"][0]["values"]) == 22
