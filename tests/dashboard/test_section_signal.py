"""The value signal section: the page it shapes, and the numbers it must keep showing.

Two kinds of test, and neither runs the signal harness.

The shaping tests hand ``shape`` a small ``SignalFacts`` built by hand and hold
what comes back to the page's rules: every figure a kind the kit draws, every
number finite, every mark labelled, the takeaway and the titles following the
values they were given, and a figure that cannot be drawn becoming a refusal
with a reason.

The pinned tests read the committed ``docs/dashboard/snapshot.json`` and check
the section's key numbers against the values the recorded run reproduces. They
skip until that file is committed; after that they are what stops the page
drifting from the model without anybody seeing it. The model itself is pinned
in ``tests/ml/test_signals.py``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import pytest

from techval.dashboard.collect import entry_module
from techval.dashboard.sections import signal as S
from techval.dashboard.snapshot import HEADLINE_KEYS, to_jsonable, validate_section

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "src" / "techval" / "dashboard" / "assets" / "kit.js"
RENDERER = ROOT / "src" / "techval" / "dashboard" / "assets" / "sections" / "signal.js"
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"
FIXTURES = ROOT / "tests" / "fixtures"

MINUS = "\u2212"


def _kit_charts() -> set[str]:
    source = KIT.read_text(encoding="utf-8")
    body = re.search(r"TV\.charts = \{(.*?)\};", source, flags=re.S)
    assert body, "kit.js no longer assigns TV.charts"
    return set(re.findall(r"(\w+)\s*:", body.group(1)))


def _facts(**changes: Any) -> S.SignalFacts:
    ic = (-0.30, -0.25, -0.20, -0.10, 0.05, 0.12, -0.08, -0.15, 0.02, -0.04, 0.09, -0.06)
    mean = sum(ic) / len(ic)
    base = S.SignalFacts(
        label="cheapness (negative trailing EV/Revenue)",
        horizon_months=12,
        companies=40,
        n_scored=420,
        dates=tuple(date(2018 + i // 4, 3 * (i % 4) + 3, 15) for i in range(len(ic))),
        ic=ic,
        counts=tuple(30 + i for i in range(len(ic))),
        mean_ic=mean,
        share_positive=4 / 12,
        t_naive=-2.31,
        t_newey_west=-1.40,
        p_newey_west=0.1615,
        lag=3,
        spacing_months=3,
        inflation=1.65,
        effective_n=4.4,
        autocorrelations=(0.70, 0.45, 0.20),
        permutation_p=1 / 401,
        baseline_draws=400,
        baseline_name="a random score with the same cross-sectional shape",
        baseline_score=0.001,
        lift=mean - 0.001,
        bucket_returns=(0.21, 0.19, 0.17, 0.16, 0.12),
        bucket_dates=12,
        spread=0.12 - 0.21,
        spread_t=-0.9,
        monotonicity=-1.0,
        acquired=0,
        delisted=0,
        absent=("EA", "JNPR"),
        verdict="The model's own verdict sentence.",
    )
    return replace(base, **changes)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _numbers(value: Any) -> Iterator[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _numbers(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _numbers(v)


def _as_snapshot_section(shaped: dict[str, Any]) -> dict[str, Any]:
    """What the collector would store, with one provenance row per figure."""
    return to_jsonable(
        {
            "id": S.ID,
            "title": S.TITLE,
            "provenance": [
                {"figure": fid, "entry_point": S.ENTRY_POINT, "inputs": S.INPUTS, "seconds": 0.0}
                for fid in shaped["figures"]
            ],
            **shaped,
        }
    )


# --------------------------------------------------------------------------- #
# shaping, with no model run
# --------------------------------------------------------------------------- #


def test_every_figure_is_a_kind_the_kit_draws_with_finite_values_and_labels():
    shaped = S.shape(_facts())
    charts = _kit_charts()
    assert set(shaped["figures"]) == set(S.FIGURE_IDS)
    for fid, fig in shaped["figures"].items():
        assert fig["kind"] in charts, fid
        assert fig["title"].strip(), fid
        assert all(math.isfinite(v) for v in _numbers(fig["data"])), fid
        data = fig["data"]
        for row in data.get("rows", []):
            assert isinstance(row["label"], str) and row["label"].strip(), fid
        for series in data.get("series", []):
            assert series["name"].strip() and series["role"] in ("model", "baseline"), fid
        for tile in data.get("tiles", []):
            assert tile["label"].strip() and tile["sub"].strip(), fid
        if fig["kind"] != "tiles":
            assert fig["subtitle"].strip(), fid


def test_the_shaped_section_passes_the_snapshot_schema():
    shaped = S.shape(_facts())
    validate_section(_as_snapshot_section(shaped))
    assert set(shaped["headline"]) == set(HEADLINE_KEYS)
    assert shaped["headline"]["verdict_text"] == "The model's own verdict sentence."
    assert shaped["headline"]["n"] == 12, "n is the number of dates, not company-dates"


def test_no_copy_on_the_page_carries_an_em_dash():
    for facts in (_facts(), _facts(lag=0, autocorrelations=(), bucket_dates=0)):
        assert not [s for s in _strings(S.shape(facts)) if "\u2014" in s]


def test_the_takeaway_is_built_from_the_values_it_was_given():
    text = S.shape(_facts())["takeaway"]
    assert "the wrong sign" in text
    assert f"{MINUS}2.31" in text and f"{MINUS}1.40" in text
    assert "p = 0.16" in text
    assert "40 TMT names" in text and "12 quarterly dates" in text
    assert "not significant" in text

    flipped = S.shape(
        _facts(mean_ic=0.051, t_naive=3.9, t_newey_west=2.6, p_newey_west=0.0093, lift=0.05)
    )["takeaway"]
    assert "the wrong sign" not in flipped
    assert "+0.051" in flipped and "survives the overlap correction" in flipped
    assert "p = 0.009" in flipped


def test_titles_follow_the_signs_and_counts():
    figures = S.shape(_facts())["figures"]
    assert figures["t_statistics"]["title"] == "The naive t clears the 5% line and the corrected t does not"
    assert figures["ic_by_date"]["title"] == "The IC was below zero on 8 of 12 dates, 4 of them in a row"
    assert figures["evidence"]["title"] == "About 4 effective observations, not 420"
    assert figures["bucket_returns"]["title"] == (
        "The cheapest fifth trailed the dearest by 9.0 percentage points"
    )

    neither = S.shape(_facts(t_naive=-1.5))["figures"]["t_statistics"]["title"]
    assert neither == "Neither t-statistic reaches the 5% line"


@pytest.mark.parametrize(
    ("p", "lift", "status"),
    [
        (0.106, -0.0986, "not_significant"),
        (0.106, 0.0986, "not_significant"),
        (0.05, 0.02, "not_significant"),
        (float("nan"), 0.02, "not_significant"),
        (0.01, 0.02, "beats"),
        (0.01, -0.02, "loses"),
    ],
)
def test_the_verdict_status_is_the_newey_west_p_value_then_the_sign(p, lift, status):
    assert S.verdict_status(p, lift) == status


def test_the_buckets_run_from_cheapest_to_dearest():
    data = S.shape(_facts())["figures"]["bucket_returns"]["data"]
    # The module's last bucket holds the highest score, which is the cheapest name.
    assert [r["value"] for r in data["rows"]] == [0.12, 0.16, 0.17, 0.19, 0.21]
    assert data["rows"][0]["label"] == "Cheapest" and data["rows"][-1]["label"] == "Dearest"
    subtitle = S.shape(_facts())["figures"]["bucket_returns"]["subtitle"]
    assert f"monotonicity {MINUS}1.00, so returns fall as cheapness rises" in subtitle


def test_the_window_overlap_is_the_share_of_the_horizon_two_dates_have_in_common():
    rows = S.shape(_facts())["figures"]["ic_autocorrelation"]["data"]["rows"]
    assert [r["values"]["overlap"] for r in rows] == [0.75, 0.5, 0.25]
    assert [r["values"]["measured"] for r in rows] == [0.70, 0.45, 0.20]
    assert [r["label"] for r in rows] == ["3 months apart", "6 months apart", "9 months apart"]


def test_the_two_t_statistics_share_one_axis_with_the_five_per_cent_lines():
    data = S.shape(_facts())["figures"]["t_statistics"]["data"]
    assert data["rows"][0]["values"] == {"newey_west": -1.40, "naive": -2.31}
    roles = {s["key"]: s["role"] for s in data["series"]}
    assert roles == {"newey_west": "model", "naive": "baseline"}
    assert [r["value"] for r in data["reference"]] == [-S.SIGNIFICANT_T, S.SIGNIFICANT_T]
    assert all("5% line" in r["label"] for r in data["reference"])


def test_the_permutation_p_value_is_shown_as_the_wrong_null_and_its_floor_named():
    tiles = S.shape(_facts())["figures"]["evidence"]["data"]["tiles"]
    sub = next(t["sub"] for t in tiles if t["label"] == "Newey-West p")
    assert "0.0025" in sub and "floor of 400 draws" in sub and "independent" in sub
    off_floor = S.shape(_facts(permutation_p=0.02))["figures"]["evidence"]["data"]["tiles"]
    assert "floor" not in next(t["sub"] for t in off_floor if t["label"] == "Newey-West p")


def test_a_non_overlapping_design_refuses_the_autocorrelation_figure_with_a_reason():
    shaped = S.shape(_facts(lag=0, autocorrelations=(), spacing_months=12))
    assert "ic_autocorrelation" not in shaped["figures"]
    refusal = next(r for r in shaped["refusals"] if r["what"] == "IC autocorrelation by lag")
    assert "no two windows overlap" in refusal["why"]
    validate_section(_as_snapshot_section(shaped))


def test_a_bucket_table_that_could_not_be_built_is_refused_rather_than_drawn():
    nan = float("nan")
    shaped = S.shape(_facts(bucket_dates=0, bucket_returns=(nan,) * 5))
    assert "bucket_returns" not in shaped["figures"]
    refusal = next(r for r in shaped["refusals"] if r["what"] == "Returns by cheapness bucket")
    assert refusal["why"].strip()
    validate_section(_as_snapshot_section(shaped))


def test_survivorship_is_a_refusal_that_names_the_missing_names_from_the_facts():
    refusal = next(r for r in S.shape(_facts())["refusals"] if r["what"] == "Survivorship correction")
    assert "Two names in the seed universe (EA, JNPR)" in refusal["why"]
    assert "420 holding periods" in refusal["why"]
    assert "measured here, not corrected" in refusal["why"]
    with_exits = S.shape(_facts(acquired=3))["refusals"]
    assert not any(r["what"] == "Survivorship correction" for r in with_exits)


def test_the_renderer_orders_every_figure_the_section_emits():
    source = RENDERER.read_text(encoding="utf-8")
    order = re.search(r"var ORDER = \[(.*?)\];", source, flags=re.S)
    assert order, "signal.js no longer declares its figure order"
    assert re.findall(r'"(\w+)"', order.group(1)) == list(S.FIGURE_IDS)


def test_the_significance_line_is_the_modules_own():
    from techval.ml.signals import SIGNIFICANT_T

    assert S.SIGNIFICANT_T == SIGNIFICANT_T


def test_the_entry_point_resolves_and_the_inputs_are_committed():
    entry_module(S.ENTRY_POINT)
    for name in S.INPUTS:
        assert (FIXTURES / name).is_file(), name


# --------------------------------------------------------------------------- #
# pinned truths, against the committed snapshot
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pinned() -> dict[str, Any]:
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    section = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"][S.ID]
    assert section["status"] == "ok", section["refusals"]
    return section


def _tile(section: dict[str, Any], label: str) -> dict[str, Any]:
    return next(t for t in section["figures"]["evidence"]["data"]["tiles"] if t["label"] == label)


def test_pinned_headline(pinned):
    h = pinned["headline"]
    assert h["metric"] == "mean IC"
    assert h["score"] == pytest.approx(-0.0984, abs=5e-4)
    assert h["baseline_score"] == pytest.approx(0.0002, abs=5e-4)
    assert h["n"] == 35
    assert h["verdict_status"] == "not_significant"
    assert "NOT SIGNIFICANT" in h["verdict_text"]
    assert "wrong sign" in pinned["takeaway"]


def test_pinned_t_statistics(pinned):
    fig = pinned["figures"]["t_statistics"]
    values = fig["data"]["rows"][0]["values"]
    assert values["naive"] == pytest.approx(-2.651, abs=5e-3)
    assert values["newey_west"] == pytest.approx(-1.617, abs=5e-3)
    assert "lag 3" in fig["subtitle"] and "1.64x" in fig["subtitle"]


def test_pinned_evidence_tiles(pinned):
    assert _tile(pinned, "Newey-West p")["value"] == pytest.approx(0.106, abs=5e-3)
    assert _tile(pinned, "Dates with a positive IC")["value"] == pytest.approx(0.4286, abs=5e-4)
    assert _tile(pinned, "Effective observations")["value"] == pytest.approx(13.0, abs=0.2)
    assert "2,604 company-dates" in _tile(pinned, "Effective observations")["sub"]
    sub = _tile(pinned, "Newey-West p")["sub"]
    assert "0.0025" in sub and "floor of 400 draws" in sub


def test_pinned_ic_series_and_autocorrelation(pinned):
    rows = pinned["figures"]["ic_by_date"]["data"]["rows"]
    assert len(rows) == 35
    assert sum(r["value"] for r in rows) / len(rows) == pytest.approx(-0.0984, abs=5e-4)
    lags = pinned["figures"]["ic_autocorrelation"]["data"]["rows"]
    assert lags[0]["values"]["measured"] == pytest.approx(0.76, abs=0.02)
    assert lags[0]["values"]["overlap"] == pytest.approx(0.75)


def test_pinned_bucket_returns(pinned):
    fig = pinned["figures"]["bucket_returns"]
    rows = fig["data"]["rows"]
    assert rows[0]["value"] - rows[-1]["value"] == pytest.approx(-0.0538, abs=5e-4)
    assert f"monotonicity {MINUS}0.90" in fig["subtitle"]
    assert f"Newey-West t {MINUS}0.79" in fig["subtitle"]


def test_pinned_survivorship_note(pinned):
    refusal = next(r for r in pinned["refusals"] if r["what"] == "Survivorship correction")
    assert "(EA, FI, FYBR, IPG, JNPR)" in refusal["why"]
