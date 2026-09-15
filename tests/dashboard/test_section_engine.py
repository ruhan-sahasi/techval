"""The engine section: its figures from small fakes, and its numbers against the snapshot.

Nothing here fits or values anything. The engine's own tests pin the bridge,
the DCF, the simulation and APV; these pin the page. ``shape`` is exercised on
hand-built ``Measured`` records, so a figure that is malformed, a title that
does not move with its values, or a refusal that is drawn anyway fails here in
milliseconds. The last group reads the committed snapshot, once it exists, and
holds the section's key numbers to what the collector reproduced from the
fixtures on the 2026-09-09 close.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from techval.dashboard.sections import engine as E
from techval.dashboard.snapshot import to_jsonable, validate_section

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
ASSETS = ROOT / "src" / "techval" / "dashboard" / "assets"
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


def _cost(**over) -> E.CostOfCapital:
    base = dict(
        risk_free_rate=0.04,
        risk_free_pinned=True,
        erp=0.05,
        peers=["AAA", "BBB"],
        beta_method="median",
        beta=1.5,
        alternative_method="vasicek",
        alternative_beta=1.4,
        alternative_wacc=0.105,
        cost_of_equity=0.115,
        wacc=0.11,
        weight_equity=0.9,
        weight_debt=0.1,
        after_tax_cost_of_debt=0.06,
        debt_to_equity=0.11,
    )
    base.update(over)
    return E.CostOfCapital(**base)


def _bridge(**over) -> E.Bridge:
    base = dict(
        equity_value=1000.0,
        diluted_shares=10.0,
        steps=[("Convertible notes", 100.0), ("Cash", -300.0)],
        enterprise_value=800.0,
        convertible_debt=100.0,
        convertible_treatment="debt",
        convertible_note="Convertible notes of 100.0mm are carried as debt.",
        lease_convention="operating leases excluded, EBITDA basis (ASC 842)",
        operating_lease=20.0,
        operating_lease_in_debt=0.0,
    )
    base.update(over)
    return E.Bridge(**base)


def _sensitivity(**over) -> E.Sensitivity:
    base = dict(
        wacc_labels=["10.0%", "11.0%", "12.0%"],
        growth_labels=["2.0%", "2.5%", "3.0%"],
        values=[[22.0, 24.0, 26.0], [18.0, 20.0, 21.5], [15.0, 16.5, None]],
        wacc_low=0.10,
        wacc_high=0.12,
        growth_low=0.02,
        growth_high=0.03,
        base_wacc=0.11,
        base_growth=0.025,
        base_value=20.0,
    )
    base.update(over)
    return E.Sensitivity(**base)


def _simulation(sd_correlated=6.0, sd_independent=5.0, **over) -> E.Simulation:
    base = dict(
        draws=12,
        drivers=4,
        correlated=[12.0, 15.5, 18.0, 19.0, 20.0, 20.5, 21.0, 22.0, 24.0, 27.0, 31.0, 40.0],
        independent=[14.0, 16.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 22.5, 24.0, 26.0, 33.0],
        sd_correlated=sd_correlated,
        sd_independent=sd_independent,
        central=20.0,
        terminal_method="gordon",
    )
    base.update(over)
    return E.Simulation(**base)


def _apv(**over) -> E.APV:
    base = dict(
        terminal_method="gordon",
        wacc=0.11,
        wacc_enterprise_value=500.0,
        unlevered_value=490.0,
        ku=0.112,
        kd=0.07,
        at_cost_of_debt=530.0,
        at_unlevered_cost_of_equity=515.0,
        modelled_interest=7.0,
        filed_interest=3.5,
    )
    base.update(over)
    return E.APV(**base)


def _measured(**over) -> E.Measured:
    base = dict(
        ticker="TEST",
        price=100.0,
        price_date="2026-01-02",
        filings_through="2025-12-31",
        cost_of_capital=_cost(),
        bridge=_bridge(),
        bands=[
            E.Band("52-week trading range", "market", 70.0, 130.0),
            E.Band("Comps, EV/Revenue", "market", 60.0, 140.0),
            E.Band("DCF, Gordon growth", "dcf", 15.0, 26.0, 20.0),
            E.Band("DCF, value driver", "dcf", 14.0, 19.0, 17.0),
        ],
        comps_quantiles=(0.25, 0.75),
        peers_requested=["AAA", "BBB"],
        per_share_gordon=20.0,
        per_share_value_driver=17.0,
        sensitivity=_sensitivity(),
        simulation=_simulation(),
        apv=_apv(),
        refusals=[],
    )
    base.update(over)
    return E.Measured(**base)


def _kit_charts() -> set[str]:
    source = (ASSETS / "kit.js").read_text(encoding="utf-8")
    body = source[source.index("TV.charts = {") :]
    body = body[: body.index("}")]
    return set(re.findall(r"(\w+)\s*:", body))


def _numbers(value, path="data"):
    """Every number in a figure's data, with where it sits."""
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


def _labels(fig) -> list[str]:
    d = fig["data"]
    kind = fig["kind"]
    if kind == "tiles":
        return [t["label"] for t in d["tiles"]]
    if kind in ("range", "dot"):
        return [r["label"] for r in d["rows"]]
    if kind == "waterfall":
        return [d["start"]["label"], d["total"]["label"]] + [s["label"] for s in d["steps"]]
    if kind == "heat":
        return list(d["rows"]) + list(d["cols"])
    if kind == "hist":
        return [s["name"] for s in d["series"]]
    raise AssertionError(f"no label rule for {kind}")


# --------------------------------------------------------------------------- #
# shaping
# --------------------------------------------------------------------------- #


def test_every_figure_is_a_kit_chart_with_finite_values_and_labels():
    section = E.shape(_measured())
    charts = _kit_charts()
    assert set(E.KINDS) <= charts
    assert set(section["figures"]) == {
        "cost_of_capital", "football", "bridge", "sensitivity", "montecarlo", "apv"
    }
    for fid, fig in section["figures"].items():
        assert fig["kind"] in charts, fid
        assert fig["title"].strip(), fid
        for where, number in _numbers(fig["data"]):
            assert math.isfinite(number), f"{fid} {where} is {number}"
        labels = _labels(fig)
        assert labels and all(isinstance(x, str) and x.strip() for x in labels), fid
    assert section["headline"] is None
    assert section["status"] == "ok"


def test_the_shaped_section_holds_to_the_snapshot_schema():
    returned = E.shape(_measured())
    section = to_jsonable(
        {
            "id": E.ID,
            "title": E.TITLE,
            **returned,
            "provenance": [
                {
                    "figure": fid,
                    "entry_point": "techval.dcf.run_dcf",
                    "inputs": [],
                    "seconds": 0.0,
                }
                for fid in returned["figures"]
            ],
        }
    )
    validate_section(section)


def test_no_copy_carries_an_em_dash():
    text = json.dumps(E.shape(_measured()), ensure_ascii=False)
    assert chr(0x2014) not in text


def test_the_takeaway_and_titles_are_built_from_the_values():
    m = _measured()
    section = E.shape(m)
    takeaway = section["takeaway"]
    for piece in (
        "2026-01-02", "100.00", "4.00% risk-free", "11.00% WACC", "20.00", "17.00",
        "20% and 17% of the price", "four drivers widens", "by 20%",
    ):
        assert piece in takeaway, piece
    assert "exit-multiple" not in takeaway

    titles = {fid: fig["title"] for fid, fig in section["figures"].items()}
    assert titles["montecarlo"] == "Correlating the drivers widens the spread by 20%"
    assert titles["football"].startswith("Every DCF band tops out below 27% of the price")
    assert titles["sensitivity"] == "The richest cell, 26.00 a share, is 26% of the price"
    assert titles["bridge"].endswith("sits 200mm below equity value")
    assert titles["apv"] == "APV sits above the WACC DCF only once the shield is added"

    moved = E.shape(
        _measured(
            price=24.0,
            simulation=_simulation(sd_correlated=4.0, sd_independent=5.0),
            apv=_apv(unlevered_value=510.0),
        )
    )
    titles = {fid: fig["title"] for fid, fig in moved["figures"].items()}
    assert titles["montecarlo"] == "Correlating the drivers narrows the spread by 20%"
    assert "narrows" in moved["takeaway"] and "24.00" in moved["takeaway"]
    assert titles["sensitivity"] == "2 of 8 cells clear the 24.00 price"
    assert titles["football"] == "1 of 4 methods bracket the price"
    assert titles["apv"] == "APV clears the WACC DCF before any tax shield is added"


def test_a_refused_row_is_noted_under_its_figure_and_not_drawn():
    m = _measured(
        bands=[b for b in _measured().bands if b.label != "DCF, value driver"],
        refusals=[
            E.Refusal("DCF, value driver", "Steady-state NOPAT is not positive.", "football")
        ],
    )
    section = E.shape(m)
    football = section["figures"]["football"]
    assert [r["label"] for r in football["data"]["rows"]] == [
        "52-week trading range", "Comps, EV/Revenue", "DCF, Gordon growth"
    ]
    assert football["refused"] == ["DCF, value driver"]
    assert section["refusals"] == [
        {"what": "DCF, value driver", "why": "Steady-state NOPAT is not positive."}
    ]


def test_a_refused_exit_multiple_is_named_in_the_takeaway():
    m = _measured(refusals=[E.Refusal(E.EXIT_ROW, "no peer median", "football")])
    section = E.shape(m)
    assert "No exit-multiple value is drawn" in section["takeaway"]
    assert section["figures"]["football"]["refused"] == [E.EXIT_ROW]


def test_a_band_that_collapses_to_one_value_is_refused_not_drawn():
    bands = _measured().bands + [E.Band("Comps, EV/Gross Profit", "market", 90.0, 90.0)]
    section = E.shape(_measured(bands=bands))
    football = section["figures"]["football"]
    assert "Comps, EV/Gross Profit" not in [r["label"] for r in football["data"]["rows"]]
    assert football["refused"] == ["Comps, EV/Gross Profit"]
    assert "90.00" in section["refusals"][0]["why"]


def test_a_refused_figure_is_absent_and_its_reason_is_section_wide():
    m = _measured(
        per_share_gordon=None,
        per_share_value_driver=None,
        bands=[E.Band("52-week trading range", "market", 70.0, 130.0)],
        sensitivity=None,
        simulation=None,
        apv=None,
        refusals=[E.Refusal("Discounted cash flow", "terminal growth is not below the WACC")],
    )
    section = E.shape(m)
    assert set(section["figures"]) == {"cost_of_capital", "football", "bridge"}
    assert section["refusals"] == [
        {"what": "Discounted cash flow", "why": "terminal growth is not below the WACC"}
    ]
    assert not any("refused" in fig for fig in section["figures"].values())
    assert "no DCF could be formed" in section["takeaway"]


def test_the_risk_free_tile_says_where_the_rate_came_from():
    def tiles(m):
        return E.shape(m)["figures"]["cost_of_capital"]["data"]["tiles"]

    pinned = tiles(_measured())
    configured = tiles(_measured(cost_of_capital=_cost(risk_free_pinned=False)))
    rate = next(t for t in pinned if t["label"] == "Risk-free rate")
    assert "test suite" in rate["sub"]
    rate = next(t for t in configured if t["label"] == "Risk-free rate")
    assert "market.risk_free_rate" in rate["sub"]
    assert pinned[0]["label"] == "Valuation date" and pinned[0]["value"] == "2026-01-02"


def test_bin_draws_counts_every_draw_on_shared_round_edges():
    s = _simulation()
    edges, a, b = E.bin_draws(s.correlated, s.independent, max_bins=10)
    assert len(edges) - 1 == len(a) == len(b) <= 10
    assert sum(a) == len(s.correlated) and sum(b) == len(s.independent)
    step = edges[1] - edges[0]
    assert all(math.isclose(y - x, step) for x, y in zip(edges, edges[1:]))
    assert step in (1.0, 2.0, 2.5, 5.0, 10.0)
    assert edges[0] <= min(s.correlated + s.independent)
    assert edges[-1] >= max(s.correlated + s.independent)
    with pytest.raises(ValueError):
        E.bin_draws([], [1.0])


# --------------------------------------------------------------------------- #
# the module's contract with the fixtures and the renderer
# --------------------------------------------------------------------------- #


def test_every_declared_input_is_committed():
    for name in E.INPUTS:
        assert (FIXTURES / name).is_file(), name


def test_the_suite_constants_are_the_conftest_fixture(assumptions):
    """The rate and the peers the section fills in are the ones the engine's tests pin."""
    assert E.SUITE_RISK_FREE_RATE == assumptions.market.risk_free_rate
    assert list(E.SUITE_PEERS) == list(assumptions.comps.peers)


def test_the_collector_runs_offline_on_the_fixtures_and_holds_to_the_schema(tmp_path):
    """One real collection, under a second: provenance resolves, refusals are named."""
    from techval.config import Assumptions
    from techval.dashboard.collect import CollectContext, collect_section

    a = Assumptions()
    a.ml.cache_dir = str(tmp_path)
    section, run = collect_section(E, CollectContext(FIXTURES, ROOT, a), use_cache=False)
    assert run.status == "ok"
    assert set(section["figures"]) == {
        "cost_of_capital", "football", "bridge", "sensitivity", "montecarlo", "apv"
    }
    assert {r["what"] for r in section["refusals"]} == {
        "Comps, EV/EBITDA", E.EXIT_ROW, "APV, Miles-Ezzell"
    }
    steps = {s["label"]: s["value"] for s in section["figures"]["bridge"]["data"]["steps"]}
    assert steps["Convertible notes"] == pytest.approx(985.545, abs=1e-3)
    assert 1.15 < section["figures"]["montecarlo"]["data"]["sd"]["ratio"] < 1.30
    assert "4.83% risk-free rate" in section["takeaway"]
    # The engine stopped filing Miles-Ezzell under Harris-Pringle (see apv.py);
    # the refusal has to say what the engine does now, not what it once claimed.
    miles = next(r for r in section["refusals"] if r["what"] == "APV, Miles-Ezzell")
    assert "does not compute Miles-Ezzell" in miles["why"]
    assert "files Miles-Ezzell" not in miles["why"]


def test_the_renderer_orders_every_figure_the_collector_returns():
    source = (ASSETS / "sections" / "engine.js").read_text(encoding="utf-8")
    order = re.findall(r'"(\w+)"', source[source.index("var ORDER") : source.index("];")])
    assert order == list(E.shape(_measured())["figures"])


# --------------------------------------------------------------------------- #
# pinned truths, against the committed snapshot
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def engine_section():
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    section = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"].get("engine")
    if section is None or section["status"] == "not_built":
        pytest.skip("the committed snapshot does not carry the engine section")
    return section


def test_snapshot_date_price_and_cost_of_capital(engine_section):
    assert engine_section["status"] == "ok"
    data = engine_section["figures"]["cost_of_capital"]["data"]
    tiles = {t["label"]: t["value"] for t in data["tiles"]}
    assert tiles["Valuation date"] == "2026-09-09"
    assert tiles["Risk-free rate"] == pytest.approx(0.0483, abs=1e-9)
    assert tiles["Beta, peer median"] == pytest.approx(1.638817, abs=1e-6)
    assert tiles["Beta, peer Vasicek"] == pytest.approx(1.512171, abs=1e-6)
    assert tiles["Cost of equity"] == pytest.approx(0.130241, abs=1e-6)
    assert tiles["WACC"] == pytest.approx(0.129715, abs=1e-6)


def test_snapshot_bridge_carries_the_convertibles_as_debt(engine_section):
    data = engine_section["figures"]["bridge"]["data"]
    steps = {s["label"]: s["value"] for s in data["steps"]}
    assert steps["Convertible notes"] == pytest.approx(985.545, abs=1e-3)
    assert data["start"]["value"] == pytest.approx(82659.291304, abs=1e-3)
    assert data["total"]["value"] == pytest.approx(78659.398304, abs=1e-3)
    walked = data["start"]["value"] + sum(steps.values())
    assert walked == pytest.approx(data["total"]["value"], abs=1e-3)


def test_snapshot_football_field_and_grid(engine_section):
    football = engine_section["figures"]["football"]
    rows = {r["label"]: r for r in football["data"]["rows"]}
    assert football["data"]["reference"][0]["value"] == pytest.approx(225.27)
    assert rows["DCF, Gordon growth"]["mid"] == pytest.approx(31.892171, abs=1e-5)
    assert rows["DCF, Gordon growth"]["lo"] == pytest.approx(27.655319, abs=1e-5)
    assert rows["DCF, Gordon growth"]["hi"] == pytest.approx(38.792483, abs=1e-5)
    assert rows["DCF, value driver"]["mid"] == pytest.approx(28.970311, abs=1e-5)
    assert rows["52-week trading range"]["lo"] == pytest.approx(102.615, abs=1e-6)
    assert rows["52-week trading range"]["hi"] == pytest.approx(288.15, abs=1e-6)
    assert "DCF, exit multiple" not in rows
    assert set(football["refused"]) == {"Comps, EV/EBITDA", E.EXIT_ROW}

    grid = engine_section["figures"]["sensitivity"]["data"]["values"]
    assert grid[2][2] == pytest.approx(31.892171, abs=1e-5)
    assert grid[0][4] == pytest.approx(38.792483, abs=1e-5)
    assert grid[4][0] == pytest.approx(27.655319, abs=1e-5)


def test_snapshot_simulation_widens_by_about_a_fifth(engine_section):
    data = engine_section["figures"]["montecarlo"]["data"]
    assert data["sd"]["ratio"] == pytest.approx(1.2138, abs=1e-4)
    assert data["sd"]["correlated"] == pytest.approx(6.581271, abs=1e-5)
    assert data["sd"]["independent"] == pytest.approx(5.422099, abs=1e-5)
    for series in data["series"]:
        assert sum(series["counts"]) == 10_000


def test_snapshot_apv_and_its_refusals(engine_section):
    rows = [r["values"] for r in engine_section["figures"]["apv"]["data"]["rows"]]
    assert rows[0]["wacc"] == pytest.approx(7702.438513, abs=1e-3)
    assert rows[0]["apv"] == pytest.approx(7720.642603, abs=1e-3)
    assert rows[1]["apv"] == pytest.approx(7971.556067, abs=1e-3)
    assert rows[2]["apv"] == pytest.approx(8016.544754, abs=1e-3)
    assert {r["what"] for r in engine_section["refusals"]} == {
        "Comps, EV/EBITDA", "DCF, exit multiple", "APV, Miles-Ezzell"
    }


def test_the_sensitivity_grid_is_plain_with_the_base_case_boxed_and_wacc_to_two_decimals():
    s = _sensitivity(wacc_low=0.1047, wacc_high=0.1247, base_wacc=0.1147)
    data = E._heat(_measured(sensitivity=s))["data"]
    assert data["scale"] == "plain"
    assert data["rows"] == ["10.47%", "11.47%", "12.47%"]
    assert data["base"] == [1, 1]

    moved = _sensitivity(values=[[22.0, 24.0, 26.0], [18.0, 20.5, 21.5], [15.0, 16.5, None]])
    assert "base" not in E._heat(_measured(sensitivity=moved))["data"]
