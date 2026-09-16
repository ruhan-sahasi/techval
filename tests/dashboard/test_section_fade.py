"""The revenue fade section: the page it shapes, and the numbers it must keep showing.

Two groups. The first calls ``shape`` on small fakes, so it fits nothing and runs
in milliseconds: every figure is one the kit can draw, every number on it is
finite, the takeaway and titles move with the values they are given, and a part
that cannot be computed becomes a refusal with its reason and no figure.

The second reads the committed snapshot and pins the numbers the section was
audited at. The snapshot is committed after every section lands, so until then
those tests skip. The model itself is pinned in ``tests/ml/test_forecast.py``; what
is pinned here is that the page still says what the model says.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from techval.dashboard.collect import entry_module
from techval.dashboard.sections import fade
from techval.dashboard.sections.fade import (
    DCF_REFUSAL,
    CompanyPaths,
    CompanyRevenue,
    Curve,
    Depth,
    HorizonScore,
    Refusal,
    Results,
    Survivorship,
    Valuation,
    shape,
    verdict_status,
)
from techval.dashboard.snapshot import HEADLINE_KEYS, to_jsonable, validate_section

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"
KIT_CHARTS = {"hbar", "column", "dot", "line", "heat", "hist", "range", "waterfall", "tiles"}


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


def _horizons(model_1: float = 0.2000, persistence_1: float = 0.2100, sd_1: float | None = 0.0500):
    return (
        HorizonScore(1, model_1, persistence_1, 0.2600, 0.2500, sd_1, 900, "the model's own sentence, one year"),
        HorizonScore(2, 0.1800, 0.2700, 0.2000, 0.1900, 0.0300, 800, "two years"),
        HorizonScore(3, 0.1700, 0.2900, 0.1900, 0.1750, 0.0300, 700, "three years"),
    )


CURVE = Curve(
    start=0.30,
    slope=0.50,
    intercept=0.05,
    reversion_level=0.10,
    half_life_years=1.0,
    typed=(0.30, 0.2375, 0.175, 0.1125, 0.05),
    n=1000,
)
SURVIVORSHIP = Survivorship(leavers=40, filers=100, gaps=((1, 0.01), (2, 0.02), (3, 0.03)))
DEPTH = Depth(
    years={0: 1, 3: 4, 7: 10, 9: 6},
    observations=150,
    filers_with_rows=20,
    unbuilt=("OLD",),
    first_filed=2011,
    last_filed=2025,
)
PATHS = CompanyPaths(
    ticker="DDOG",
    name="Datadog",
    trailing=0.33,
    fiscal_year_end="2025-12-31",
    filed="2026-02-18",
    typed=CURVE.typed,
    fitted=(0.28, 0.24, 0.22, 0.16, 0.10),
    lower=(0.08, 0.04, 0.05, 0.075, 0.10),
    upper=(0.50, 0.44, 0.45, 0.275, 0.10),
    basis=("fitted", "fitted", "fitted", "assumed", "assumed"),
    terminal=0.10,
)
REVENUE = CompanyRevenue(trailing=1000.0, trailing_as_of="2026-06-30", typed=2000.0, fitted=2500.0, low=1500.0, high=4000.0)
VALUATION = Valuation(
    per_share_typed=30.0,
    per_share_fitted=36.0,
    per_share_low=20.0,
    per_share_high=60.0,
    ev_typed=5000.0,
    ev_fitted=6000.0,
    price_date="2026-09-09",
    wacc=0.1128,
    beta=1.30,
    risk_free_rate=0.0483,
)


def _results(**changes) -> Results:
    base = Results(
        horizons=_horizons(),
        curve=CURVE,
        survivorship=SURVIVORSHIP,
        depth=DEPTH,
        paths=PATHS,
        revenue=REVENUE,
        valuation=VALUATION,
    )
    return replace(base, **changes)


def _numbers(value, path="data"):
    """Every number anywhere inside a figure, with where it sits."""
    if isinstance(value, bool):
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
    """What the collector would write, with a provenance row per figure."""
    section = to_jsonable(
        {
            "id": fade.ID,
            "title": fade.TITLE,
            **returned,
            "provenance": [
                {"figure": fid, "entry_point": "techval.ml.forecast.fit_fade", "inputs": [], "seconds": 0.0}
                for fid in returned["figures"]
            ],
        }
    )
    validate_section(section)
    return section


# --------------------------------------------------------------------------- #
# the verdict rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "lift, sd, expected",
    [
        (0.0035, 0.0242, "ties"),
        (-0.0100, 0.0242, "ties"),
        (0.0407, 0.0162, "beats"),
        (-0.0500, 0.0200, "loses"),
        (0.0100, None, "beats"),
        (-0.0100, None, "loses"),
    ],
)
def test_a_lift_inside_the_fold_spread_is_a_tie_whichever_way_it_points(lift, sd, expected):
    assert verdict_status(lift, sd) == expected


# --------------------------------------------------------------------------- #
# shaping
# --------------------------------------------------------------------------- #


def test_every_figure_is_one_the_kit_draws_with_finite_numbers_and_labels():
    out = shape(_results())
    section = _as_section(out)
    assert set(section["figures"]) == {
        "error_by_horizon",
        "fade_curve",
        "ddog_paths",
        "ddog_revenue",
        "ddog_dcf",
        "depth",
    }
    for fid, fig in section["figures"].items():
        assert fig["kind"] in KIT_CHARTS, fid
        assert fig["title"].strip(), fid
        assert chr(0x2014) not in json.dumps(fig, ensure_ascii=False), fid
        for where, number in _numbers(fig["data"]):
            assert math.isfinite(number), f"{fid} {where}"
        data = fig["data"]
        if fig["kind"] == "line":
            assert 1 < len(data["series"]) <= 4
            for s in data["series"]:
                assert s["name"] and s["role"] in {"model", "alt", "baseline"}
                assert all(isinstance(p["x"], str) and p["x"] for p in s["values"])
        elif fig["kind"] == "column":
            assert all(r["label"] for r in data["rows"])
        elif fig["kind"] == "tiles":
            assert all(t["label"] for t in data["tiles"])
    assert section["status"] == "ok"
    assert section["refusals"] == []


def test_the_fitted_model_is_blue_and_the_thing_it_replaces_is_named_in_orange():
    figures = shape(_results())["figures"]
    for fid in ("error_by_horizon", "fade_curve", "ddog_paths"):
        roles = {s["name"]: s["role"] for s in figures[fid]["data"]["series"]}
        assert roles["Fitted fade"] == "model"
        assert sum(1 for r in roles.values() if r == "alt") == 1
    baselines = figures["error_by_horizon"]["data"]["series"]
    assert {s["name"] for s in baselines if s["role"] == "baseline"} == {"Sub-vertical mean", "Training mean"}


def test_the_band_on_the_error_chart_is_one_fold_standard_deviation():
    horizons = _horizons()
    model = shape(_results())["figures"]["error_by_horizon"]["data"]["series"][0]
    for h, point in zip(horizons, model["values"]):
        assert point["y"] == h.model
        assert point["hi"] - point["y"] == pytest.approx(h.fold_sd)
        assert point["y"] - point["lo"] == pytest.approx(h.fold_sd)


def test_the_headline_is_the_one_year_score_against_persistence():
    out = shape(_results())
    headline = out["headline"]
    assert set(headline) == set(HEADLINE_KEYS)
    assert headline["score"] == 0.2000
    assert headline["baseline_score"] == 0.2100
    assert headline["lift"] == pytest.approx(0.0100)
    assert headline["higher_is_better"] is False
    assert headline["n"] == 900
    assert headline["verdict_status"] == "ties"
    assert headline["verdict_text"] == "the model's own sentence, one year"


def test_the_takeaway_and_titles_are_built_from_the_values_they_describe():
    out = shape(_results())
    takeaway = out["takeaway"]
    for text in ("ties this year's growth", "0.2000", "0.2100", "+0.0100", "0.0500", "+0.0900", "+0.1200"):
        assert text in takeaway, text
    assert "36.00 a share" in takeaway and "30.00" in takeaway
    figures = out["figures"]
    assert figures["error_by_horizon"]["title"] == (
        "The fitted fade ties this year's growth at one year and beats it at two and three years"
    )
    assert figures["fade_curve"]["title"] == "The filings level off near 10%; the typed schedule falls to 5%"
    assert "25% more revenue" in figures["ddog_revenue"]["title"]
    assert "+6.00 a share" in figures["ddog_dcf"]["title"]
    depth = figures["depth"]
    assert depth["title"] == "The median filer supports 7 fiscal years of revenue; 6 reach the full 9"
    assert [r["value"] for r in depth["data"]["rows"]] == [1, 0, 0, 4, 0, 0, 0, 10, 0, 6]
    assert "OLD has no fiscal year" in depth["subtitle"]


def test_a_one_year_win_outside_the_noise_changes_the_verdict_and_the_words():
    out = shape(_results(horizons=_horizons(model_1=0.1000, persistence_1=0.2100, sd_1=0.0200)))
    assert out["headline"]["verdict_status"] == "beats"
    assert "beats this year's growth" in out["takeaway"]
    assert "outside a fold standard deviation of 0.0200" in out["takeaway"]
    assert out["figures"]["error_by_horizon"]["title"].startswith(
        "The fitted fade beats this year's growth at one, two and three years"
    )

    lost = shape(_results(horizons=_horizons(model_1=0.3000, persistence_1=0.2100, sd_1=0.0200)))
    assert lost["headline"]["verdict_status"] == "loses"
    assert "loses to this year's growth" in lost["takeaway"]


def test_the_survivorship_note_states_the_direction_and_the_measured_gaps():
    note = shape(_results())["figures"]["fade_curve"]["note"]
    assert note["what"] == "Survivorship flatters a fade curve"
    assert "fades too slowly" in note["why"] and "too optimistic" in note["why"]
    assert "1.0, 2.0 and 3.0 points at one, two and three years" in note["why"]
    assert "40 of its 100 filers" in note["why"]


def test_an_unvalued_path_is_a_refusal_with_its_reason_and_no_figure():
    why = "The DCF needs a risk-free rate and the committed fixtures record none."
    out = shape(_results(valuation=Refusal(DCF_REFUSAL, why)))
    assert "ddog_dcf" not in out["figures"]
    assert out["refusals"] == [{"what": DCF_REFUSAL, "why": why}]
    # The renderer hangs the refusal beside the revenue tiles by this name.
    assert out["figures"]["ddog_revenue"]["data"]["refusalSlot"] == DCF_REFUSAL
    assert "a share" not in out["takeaway"]
    _as_section(out)


def test_a_company_that_cannot_be_pathed_is_one_refusal_not_three():
    out = shape(_results(paths=Refusal("Datadog, assumed against fitted", "DDOG is not in the panel."), revenue=None, valuation=None))
    assert not {"ddog_paths", "ddog_revenue", "ddog_dcf"} & set(out["figures"])
    assert out["refusals"] == [{"what": "Datadog, assumed against fitted", "why": "DDOG is not in the panel."}]
    _as_section(out)


def test_a_curve_that_does_not_revert_is_refused():
    out = shape(_results(curve=Refusal("The fade curve", "The one-year persistence slope is 1.0200.")))
    assert "fade_curve" not in out["figures"]
    assert out["refusals"][0]["what"] == "The fade curve"
    _as_section(out)


def test_a_non_finite_score_is_a_bug_not_a_refusal():
    broken = _horizons()
    broken = (replace(broken[0], model=float("nan")),) + broken[1:]
    with pytest.raises(ValueError):
        shape(_results(horizons=broken))


# --------------------------------------------------------------------------- #
# the collector's declarations
# --------------------------------------------------------------------------- #


def test_every_declared_input_is_committed():
    for name in fade.INPUTS:
        assert (FIXTURES / name).exists(), name


def test_every_recorded_entry_point_resolves_and_names_a_figure_shape_draws():
    source = Path(fade.__file__).read_text(encoding="utf-8")
    recorded = re.findall(r'ctx\.record\(\s*"(\w+)",\s*"([\w.]+)"', source)
    assert {fid for fid, _ in recorded} == set(shape(_results())["figures"])
    for _, entry in recorded:
        entry_module(entry)


# --------------------------------------------------------------------------- #
# pinned truths, against the committed snapshot
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def committed() -> dict:
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    section = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"].get("fade")
    if section is None:
        pytest.skip("the committed snapshot carries no fade section")
    assert section["status"] == "ok", section["refusals"]
    return section


def test_committed_one_year_fade_ties_persistence(committed):
    headline = committed["headline"]
    assert headline["score"] == pytest.approx(0.1474, abs=5e-5)
    assert headline["baseline_score"] == pytest.approx(0.1510, abs=5e-5)
    assert headline["lift"] == pytest.approx(0.0035, abs=5e-5)
    assert headline["verdict_status"] == "ties"
    assert "inside the fold-to-fold noise" in headline["verdict_text"]


def test_committed_lifts_over_persistence_by_horizon(committed):
    series = {s["name"]: s["values"] for s in committed["figures"]["error_by_horizon"]["data"]["series"]}
    model = series["Fitted fade"]
    persistence = series["Persistence"]
    lifts = [p["y"] - m["y"] for m, p in zip(model, persistence)]
    assert lifts == pytest.approx([0.0035, 0.0407, 0.0530], abs=5e-5)
    statuses = [verdict_status(lift, m["hi"] - m["y"]) for lift, m in zip(lifts, model)]
    assert statuses == ["ties", "beats", "beats"]


def test_committed_lift_over_the_strongest_baseline_is_inside_the_noise_at_every_horizon(committed):
    series = committed["figures"]["error_by_horizon"]["data"]["series"]
    model = series[0]["values"]
    others = [s["values"] for s in series[1:]]
    for i, m in enumerate(model):
        strongest = min(values[i]["y"] for values in others)
        assert abs(strongest - m["y"]) < m["hi"] - m["y"]


def test_committed_panel_and_filing_depth(committed):
    stats = committed["figures"]["depth"]["data"]["stats"]
    assert stats["observations"] == 2798
    assert stats["companies"] == 224
    assert stats["filers_with_rows"] == 223
    assert (stats["first_filed"], stats["last_filed"]) == (2009, 2026)
    assert stats["median_years"] == 12
    assert stats["max_years"] == 19
    assert stats["at_max"] == 34



def test_the_dcf_states_its_discount_rate_and_never_calls_itself_the_engine_dcf():
    """The page carried two Datadog DCFs, 31.89 in the engine section and 36.65 here.

    They differ because this one is discounted on Datadog's own regression beta and
    the engine's on a peer median. The figure once called itself "the same DCF" and
    named no rate, so a reader could only see a contradiction. It now states the
    rate, the beta and that the risk-free rate is a pinned assumption.
    """
    section = shape(_results())
    fig = section["figures"]["ddog_dcf"]
    assert "same DCF" not in fig["subtitle"] and "same DCF" not in section["takeaway"]
    assert "11.28%" in fig["subtitle"] and "beta of 1.30" in fig["subtitle"]
    assert "4.83% risk-free rate that is an assumption" in fig["subtitle"]
    assert "peer-median beta" in fig["subtitle"]
    assert "11.28% on its own beta" in section["takeaway"]


def test_the_chip_judges_the_noise_on_the_lift_across_folds():
    """The same test the warranted multiple's chip applies, so the scoreboard compares like with like.

    A pooled lift can clear the spread of the model's own fold errors while the
    lift itself swings from fold to fold. That lift is inside the noise, and the
    older comparison against the score's spread would have called it a win.
    """
    swinging = (0.09, -0.07, 0.08, -0.06, 0.05)
    assert verdict_status(0.04, 0.02) == "beats"
    assert verdict_status(0.04, 0.02, swinging) == "ties"
    steady = (0.04, 0.05, 0.035, 0.045, 0.04)
    assert verdict_status(0.04, 0.02, steady) == "beats"
    assert verdict_status(-0.04, 0.02, tuple(-x for x in steady)) == "loses"
