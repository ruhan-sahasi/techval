"""The scoreboard: the page it shapes from other sections' results, and what it must keep saying.

Three groups. The first hands the three computations and ``shape`` small fake
results, so nothing is fitted and every case runs in milliseconds: the verdict
counts drive the takeaway, a model section with no score yields a state tile and
never a number, and the section table counts figures and refusals. The second
runs the collector the way ``collect_snapshot`` does, so the figures it records
and the refusal of the whole section are held to the schema. The third reads the
committed snapshot and checks the scoreboard still agrees with the sections it
summarises; until the snapshot is committed those tests skip.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.dashboard.collect import CollectContext, collect_section, entry_module
from techval.dashboard.sections import MODEL_SECTIONS, SECTION_IDS, overview
from techval.dashboard.sections.overview import (
    RETURNS_MODEL,
    _name,
    assumption_overrides,
    scoreboard,
    section_states,
    shape,
    takeaway,
)
from techval.dashboard.snapshot import validate_section
from techval.errors import NotMeaningfulError

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"
RENDERER = ROOT / "src" / "techval" / "dashboard" / "assets" / "sections" / "overview.js"

TITLES = {
    "signal": "The value signal",
    "encoder": "Peer encoder",
    "warranted": "Warranted multiple",
    "fade": "Revenue fade",
    "propensity": "M&A propensity",
    "engine": "Valuation engine",
    "tmt": "TMT layer",
    "reading": "Reading filings",
    "datalayer": "Data layer",
    "sample": "What the sample can support",
}


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


def _headline(verdict: str, score: float = 0.55, baseline: float = 0.50, lift: float | None = None, **extra):
    return {
        "metric": extra.get("metric", "AUC"),
        "score": score,
        "baseline_name": extra.get("baseline_name", "a size sort"),
        "baseline_score": baseline,
        "lift": score - baseline if lift is None else lift,
        "n": extra.get("n", 400),
        "higher_is_better": extra.get("higher_is_better", True),
        "verdict_status": verdict,
        "verdict_text": "the model's own sentence",
    }


def _section(sid: str, status: str = "ok", headline=None, figures: int = 2, refusals: int = 0, seconds: float = 1.0):
    figs = {f"f{i}": {"kind": "hbar", "title": f"Figure {i}", "data": {}} for i in range(figures if status == "ok" else 0)}
    reasons = [{"what": f"Thing {i}", "why": f"Because {i}."} for i in range(refusals)]
    if status == "refused" and not reasons:
        reasons = [{"what": TITLES[sid], "why": "declared input(s) not found under the ml-data root"}]
    return {
        "id": sid,
        "title": TITLES[sid],
        "takeaway": "",
        "status": status,
        "refusals": reasons,
        "headline": headline if status == "ok" else None,
        "figures": figs,
        "provenance": [
            {"figure": fid, "entry_point": "techval.ml.mna.fit_propensity", "inputs": [], "seconds": seconds}
            for fid in figs
        ],
    }


VERDICTS = {
    "signal": _headline("not_significant", score=-0.098405, baseline=0.000178, metric="mean IC", n=35),
    "encoder": _headline("beats", score=0.540665, baseline=0.221331, metric="NDCG@10", n=162),
    "warranted": _headline("inside_noise", score=0.765125, baseline=0.631267),
    "fade": _headline("ties", score=0.147436, baseline=0.15098, lift=0.003544, higher_is_better=False, metric="MAE"),
    "propensity": _headline("inside_noise", score=0.568543, baseline=0.547352),
}


def _results(**overrides) -> dict:
    """Every section but the scoreboard, as the collector hands them over."""
    out = {}
    for sid in SECTION_IDS:
        if sid == overview.ID:
            continue
        if sid in VERDICTS:
            out[sid] = _section(sid, headline=VERDICTS[sid])
        elif sid == "sample":
            out[sid] = _section(sid, status="not_built")
        else:
            out[sid] = _section(sid, figures=3, refusals={"engine": 3, "tmt": 10, "reading": 0, "datalayer": 2}[sid])
    for sid, section in overrides.items():
        if section is None:
            out.pop(sid, None)
        else:
            out[sid] = section
    return out


def _with_verdicts(**verdicts) -> dict:
    return _results(**{sid: _section(sid, headline=_headline(v)) for sid, v in verdicts.items()})


def _numbers(value, path="tile"):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _numbers(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _numbers(v, f"{path}[{i}]")


def _shape(results: dict, assumptions: Assumptions | None = None) -> dict:
    return shape(
        scoreboard(results),
        section_states(results),
        assumption_overrides(assumptions or Assumptions()),
    )


# --------------------------------------------------------------------------- #
# the scoreboard tiles
# --------------------------------------------------------------------------- #


def test_every_model_section_gets_one_tile_in_page_order_linking_to_its_section():
    tiles = scoreboard(_results())
    assert [t["section"] for t in tiles] == list(MODEL_SECTIONS)
    assert [t["href"] for t in tiles] == [f"#{sid}" for sid in MODEL_SECTIONS]
    assert [t["label"] for t in tiles] == [TITLES[sid] for sid in MODEL_SECTIONS]


def test_a_scored_tile_carries_the_headline_as_its_section_wrote_it():
    tile = scoreboard(_results())[3]
    head = VERDICTS["fade"]
    assert tile["state"] == "scored"
    for key in ("metric", "score", "baseline_name", "baseline_score", "lift", "n", "higher_is_better", "verdict_status"):
        assert tile[key] == head[key], key
    # The kit's own tile keys, so the default drawing of a tiles figure is still true.
    assert tile["value"] == head["score"] and tile["status"] == "ties"
    assert tile["delta"]["value"] == head["lift"]
    assert "0.1510 for a size sort" in tile["sub"]


def test_a_refused_model_section_yields_a_state_tile_and_no_number():
    why = "declared input(s) not found under the ml-data root tests/fixtures: fundamentals/panel.csv.gz"
    refused = _section("fade", status="refused")
    refused["refusals"] = [{"what": "Revenue fade", "why": why}]
    tile = scoreboard(_results(fade=refused))[3]
    assert tile["state"] == "refused"
    assert tile["status"] == "refused"
    assert tile["reason"] == {"what": "Revenue fade", "why": why}
    assert isinstance(tile["value"], str) and tile["value"] == "Refused"
    assert list(_numbers(tile)) == []
    assert not {"score", "baseline_score", "lift", "n", "delta", "format"} & set(tile)


@pytest.mark.parametrize("state, results", [
    ("not_built", _results(propensity=_section("propensity", status="not_built"))),
    ("not_built", _results(propensity=None)),
    ("no_headline", _results(propensity=_section("propensity", headline=None))),
])
def test_a_model_section_without_a_score_says_which_state_it_is_in_and_prints_no_number(state, results):
    tile = scoreboard(results)[4]
    assert tile["state"] == state
    assert tile["label"] == "M&A propensity"
    assert list(_numbers(tile)) == []
    assert "status" not in tile


def test_a_non_finite_headline_is_a_bug_not_a_tile():
    broken = _section("encoder", headline=_headline("beats", score=float("nan")))
    with pytest.raises(ValueError):
        scoreboard(_results(encoder=broken))


# --------------------------------------------------------------------------- #
# the takeaway and the titles, from the verdict counts
# --------------------------------------------------------------------------- #


def test_the_takeaway_leads_with_the_model_scored_against_returns_and_then_counts_verdicts():
    out = _shape(_results())
    assert out["takeaway"] == (
        "The value signal, the one model scored against realised returns, came back behind "
        "its baseline: −0.0984 on mean IC against 0.0002, a lift of −0.0986 that is "
        "not significant. Five models were each held to a baseline, and only the peer encoder "
        "beats it outside the noise; two sit inside the noise, one ties it and one is not "
        "significant."
    )
    assert out["figures"]["scoreboard"]["title"] == (
        "One of five models beats its baseline outside the noise: the peer encoder"
    )
    assert out["figures"]["scoreboard"]["data"]["counts"] == {
        "beats": 1, "inside_noise": 2, "ties": 1, "not_significant": 1, "loses": 0,
    }


@pytest.mark.parametrize("verdicts, sentence, title", [
    (
        dict(signal="beats", encoder="beats", warranted="beats", fade="beats", propensity="beats"),
        "Five models were each held to a baseline, and all five beat it outside the noise.",
        "All five scored models beat their baselines outside the noise",
    ),
    (
        dict(signal="loses", encoder="ties", warranted="ties", fade="loses", propensity="inside_noise"),
        "Five models were each held to a baseline, and none beats it outside the noise; one "
        "sits inside the noise, two tie it and two lose to it.",
        "No model beats its baseline outside the noise",
    ),
    (
        dict(signal="not_significant", encoder="beats", warranted="beats", fade="loses", propensity="loses"),
        "Five models were each held to a baseline, and the peer encoder and the warranted "
        "multiple beat it outside the noise; one is not significant and two lose to it.",
        "Two of five models beat their baselines outside the noise: the peer encoder and the "
        "warranted multiple",
    ),
])
def test_the_verdict_counts_drive_the_takeaway_and_the_title(verdicts, sentence, title):
    out = _shape(_with_verdicts(**verdicts))
    assert sentence in out["takeaway"]
    assert out["figures"]["scoreboard"]["title"] == title
    counts = out["figures"]["scoreboard"]["data"]["counts"]
    assert sum(counts.values()) == 5
    for verdict, n in counts.items():
        assert n == sum(v == verdict for v in verdicts.values())


def test_the_direction_of_the_returns_sentence_follows_the_sign_of_the_lift():
    ahead = _results(signal=_section("signal", headline=_headline("beats", score=0.09, baseline=0.0, metric="mean IC")))
    assert "came back ahead of its baseline: 0.0900 on mean IC against 0.0000, a lift of +0.0900 that clears the noise" in _shape(ahead)["takeaway"]


def test_models_without_a_score_are_named_and_the_count_says_how_many_were_held_to_a_baseline():
    results = _results(
        fade=_section("fade", status="refused"),
        propensity=_section("propensity", status="not_built"),
    )
    text = _shape(results)["takeaway"]
    assert "Three of the five models were held to a baseline, and only the peer encoder beats it" in text
    assert text.endswith("The revenue fade refused its score and the M&A propensity has not been collected.")


def test_when_the_returns_model_has_no_score_the_takeaway_says_nothing_was_scored_against_returns():
    text = _shape(_results(signal=_section("signal", status="refused")))["takeaway"]
    assert "the one model scored against realised returns" not in text
    assert text.startswith("Four of the five models were held to a baseline")
    assert text.endswith(
        "The value signal refused its score, so no model on this page is scored against realised returns."
    )


def test_a_section_name_inside_a_sentence_keeps_its_initialisms():
    assert _name("The value signal") == "the value signal"
    assert _name("Peer encoder") == "the peer encoder"
    assert _name("M&A propensity") == "the M&A propensity"
    assert _name("TMT layer") == "the TMT layer"


# --------------------------------------------------------------------------- #
# the section table
# --------------------------------------------------------------------------- #


def test_the_section_table_counts_figures_and_refusals_for_every_other_section():
    results = _results(fade=_section("fade", status="refused"), sample=None)
    rows = section_states(results)
    assert [r["id"] for r in rows] == [s for s in SECTION_IDS if s != "overview"]
    by = {r["id"]: r for r in rows}
    assert (by["tmt"]["figures"], by["tmt"]["refusals"]) == (3, 10)
    assert (by["fade"]["status"], by["fade"]["figures"], by["fade"]["refusals"]) == ("refused", 0, 1)
    assert by["encoder"]["verdict_status"] == "beats" and by["engine"]["verdict_status"] is None
    # A section the results do not hold is not collected, and its title comes from its module.
    assert by["sample"]["status"] == "not_built"
    assert by["sample"]["title"] == "What the sample can support"
    assert (by["sample"]["figures"], by["sample"]["refusals"]) == (0, 0)

    figure = _shape(results)["figures"]["sections"]
    totals = figure["data"]["totals"]
    assert totals == {
        "sections": 10,
        "collected": 8,
        "figures": sum(r["figures"] for r in rows),
        "refusals": 16,
        "refusing_sections": 4,
    }
    assert figure["title"] == f"{totals['figures']} figures drawn and 16 refusals stated, 10 of them in the TMT layer"
    assert [r["value"] for r in figure["data"]["rows"]] == [r["refusals"] for r in rows]
    assert {r["role"] for r in figure["data"]["rows"]} == {"baseline"}


def test_a_table_with_no_refusals_says_so():
    quiet = {sid: _section(sid, figures=1, headline=VERDICTS.get(sid)) for sid in TITLES}
    assert _shape(quiet)["figures"]["sections"]["title"] == "10 figures drawn and nothing refused"


# --------------------------------------------------------------------------- #
# the assumptions
# --------------------------------------------------------------------------- #


def test_the_assumptions_figure_lists_what_differs_from_the_defaults_and_not_the_cache_dir():
    applied = Assumptions.model_validate({"market": {"risk_free_rate": 0.0483}, "ml": {"cache_dir": "/tmp/elsewhere"}})
    overrides = assumption_overrides(applied)
    assert overrides == [{"key": "market.risk_free_rate", "value": 0.0483, "default": None}]
    figure = _shape(_results(), applied)["figures"]["collection"]
    assert figure["title"] == "Every assumption but one is the engine default"
    assert figure["data"]["tiles"][0]["value"] == 1
    assert figure["data"]["tiles"][0]["sub"] == "market.risk_free_rate = 0.0483; the default is unset"

    assert assumption_overrides(Assumptions()) == []
    assert _shape(_results())["figures"]["collection"]["title"] == "Every assumption is the engine default"


# --------------------------------------------------------------------------- #
# the collector
# --------------------------------------------------------------------------- #


def _collect(results: dict, assumptions: Assumptions | None = None) -> dict:
    ctx = CollectContext(ROOT / "tests" / "fixtures", ROOT, assumptions or Assumptions(), results)
    section, _ = collect_section(overview, ctx, use_cache=False)
    return section


def test_the_collected_section_holds_to_the_schema_with_every_figure_recorded():
    section = _collect(_results())
    validate_section(section)
    assert section["status"] == "ok"
    assert section["headline"] is None
    assert list(section["figures"]) == list(overview.FIGURES)
    assert {p["figure"] for p in section["provenance"]} == set(overview.FIGURES)
    for row in section["provenance"]:
        assert row["inputs"] == []
        entry_module(row["entry_point"])
    # The defaults the overrides are measured against are cited, so a moved default misses the cache.
    cited = [p["entry_point"] for p in section["provenance"] if p["figure"] == "collection"]
    assert cited == [overview.ASSUMPTIONS_ENTRY, "techval.config.Assumptions"]
    text = json.dumps(section, ensure_ascii=False)
    assert chr(0x2014) not in text
    for where, number in _numbers(section["figures"]):
        assert math.isfinite(number), where


def test_the_section_is_refused_only_when_no_model_section_produced_a_headline():
    none_scored = _results(**{sid: _section(sid, status="refused") for sid in MODEL_SECTIONS})
    with pytest.raises(NotMeaningfulError):
        _shape(none_scored)
    section = _collect(none_scored)
    assert section["status"] == "refused"
    assert section["figures"] == {} and section["provenance"] == []
    assert "no model section produced a headline" in section["refusals"][0]["why"]

    one_scored = _results(**{sid: _section(sid, status="refused") for sid in MODEL_SECTIONS if sid != "fade"})
    assert _collect(one_scored)["status"] == "ok"


def test_the_returns_model_is_a_model_section():
    assert RETURNS_MODEL in MODEL_SECTIONS
    assert overview.INPUTS == []


# --------------------------------------------------------------------------- #
# the renderer's own sums, run under node
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_collection_tiles_sum_the_provenance_the_page_prints():
    snapshot = {
        "techval_commit": "abc1234",
        "collected_at": "2026-09-14",
        "fixtures": {"a": "0", "b": "1", "c": "2"},
        "sections": {
            "encoder": _section("encoder", headline=VERDICTS["encoder"], figures=2, seconds=100.0),
            "tmt": _section("tmt", figures=3, seconds=0.5),
            "sample": _section("sample", status="not_built"),
        },
    }
    script = (
        "const vm = require('vm');"
        "const fs = require('fs');"
        "const window = {TV: {el: function () {}, sections: {register: function () {}}}};"
        "vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {window: window});"
        "const got = window.TV.overviewCollection(JSON.parse(process.argv[2]));"
        "process.stdout.write(JSON.stringify(got));"
    )
    out = subprocess.run(
        ["node", "-e", script, str(RENDERER), json.dumps(snapshot)],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["commit"] == "abc1234" and got["collected"] == "2026-09-14"
    assert got["fixtures"] == 3
    assert got["rows"] == 5
    assert got["seconds"] == pytest.approx(201.5)
    assert got["top"] == {"title": "Peer encoder", "seconds": 200.0}


# --------------------------------------------------------------------------- #
# against the committed snapshot
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def committed() -> dict:
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    sections = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"]
    if "overview" not in sections:
        pytest.skip("the committed snapshot carries no scoreboard")
    assert sections["overview"]["status"] == "ok", sections["overview"]["refusals"]
    return sections


def test_committed_verdicts_are_the_ones_the_sections_reached(committed):
    tiles = committed["overview"]["figures"]["scoreboard"]["data"]["tiles"]
    assert {t["section"]: t["verdict_status"] for t in tiles} == {
        "signal": "not_significant",
        "encoder": "beats",
        "warranted": "inside_noise",
        "fade": "ties",
        "propensity": "inside_noise",
    }


def test_committed_tiles_agree_with_every_model_headline(committed):
    tiles = {t["section"]: t for t in committed["overview"]["figures"]["scoreboard"]["data"]["tiles"]}
    for sid in MODEL_SECTIONS:
        headline = committed[sid]["headline"]
        for key in ("metric", "score", "baseline_name", "baseline_score", "lift", "n", "verdict_status"):
            assert tiles[sid][key] == headline[key], (sid, key)


def test_committed_takeaway_reads_the_negative_result_first(committed):
    text = committed["overview"]["takeaway"]
    assert text.startswith(
        "The value signal, the one model scored against realised returns, came back behind its baseline"
    )
    assert "only the peer encoder beats it outside the noise" in text
    # The held identity still matches what the signal section says it tests.
    assert "returns" in committed[RETURNS_MODEL]["takeaway"]


def test_committed_section_table_matches_the_sections_it_counts(committed):
    rows = {r["id"]: r for r in committed["overview"]["figures"]["sections"]["data"]["table"]}
    assert set(rows) == set(SECTION_IDS) - {"overview"}
    for sid, row in rows.items():
        section = committed[sid]
        assert row["status"] == section["status"], sid
        assert row["figures"] == len(section["figures"]), sid
        assert row["refusals"] == len(section["refusals"]), sid
