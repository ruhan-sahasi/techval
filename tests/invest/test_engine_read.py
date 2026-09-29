"""The engine read: fade path and warranted residual for held names, offline.

Both models fit once from committed panels and answer for every holding, so
twelve positions cost two fits. A name outside a panel gets a refusal that
says which pane is missing and why, never a blank.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.invest.engine_read import EngineRead, fade_model, read_holdings, warranted_model

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def assumptions() -> Assumptions:
    return Assumptions()


@pytest.fixture(scope="module")
def reads(assumptions) -> dict[str, EngineRead]:
    return read_holdings(["DDOG", "JPM"], fixtures=FIXTURES, assumptions=assumptions)


def test_a_covered_name_gets_a_fade_path(reads, assumptions):
    ddog = reads["DDOG"]
    assert ddog.covered and ddog.sub_vertical == "infrastructure_software"
    fade = ddog.fade
    years = assumptions.dcf.projection_years
    assert len(fade["fitted"]) == len(fade["assumed"]) == len(fade["basis"]) == years
    assert fade["basis"][0] == "fitted" and fade["basis"][-1] == "assumed"
    assert fade["assumed"][0] == pytest.approx(assumptions.dcf.revenue_growth_start)
    assert fade["assumed"][-1] == pytest.approx(assumptions.dcf.revenue_growth_terminal)
    assert 0 < fade["trailing"] < 1
    assert "verdict" in fade and "persistence" in fade["verdict"]


def test_a_covered_name_gets_a_warranted_residual(reads):
    import math

    w = reads["DDOG"].warranted
    assert w["traded"] > 0 and w["warranted"] > 0
    assert w["residual_log"] == pytest.approx(math.log(w["traded"] / w["warranted"]), abs=1e-6)
    assert w["as_of"] >= "2026-01-01"
    assert w["call"] in ("rich", "cheap")
    assert "verdict" in w


def test_an_uncovered_name_refuses_each_pane_by_name(reads):
    jpm = reads["JPM"]
    assert not jpm.covered and jpm.sub_vertical is None
    assert jpm.fade is None and jpm.warranted is None
    whats = [r["what"] for r in jpm.refusals]
    assert "Fade path" in whats and "Warranted multiple" in whats
    assert all("JPM" in r["why"] for r in jpm.refusals)


def test_the_models_fit_once_and_the_reads_are_deterministic(assumptions):
    assert fade_model(FIXTURES) is fade_model(FIXTURES)
    assert warranted_model(FIXTURES) is warranted_model(FIXTURES)
    again = read_holdings(["DDOG", "JPM"], fixtures=FIXTURES, assumptions=assumptions)
    first = read_holdings(["DDOG", "JPM"], fixtures=FIXTURES, assumptions=assumptions)
    assert first["DDOG"].fade == again["DDOG"].fade
    assert first["DDOG"].warranted == again["DDOG"].warranted


def test_a_cached_fit_is_built_once_and_loaded_after(tmp_path):
    from techval.invest.engine_read import cached_fit

    panel = tmp_path / "panel.bin"
    panel.write_bytes(b"one")
    built = []

    def build():
        built.append(1)
        return {"fit": len(built)}

    modules = ("techval.invest.engine_read",)
    first = cached_fit("probe", [panel], modules, build, tmp_path / "cache")
    second = cached_fit("probe", [panel], modules, build, tmp_path / "cache")
    assert first == second == {"fit": 1}
    assert len(built) == 1
    assert len(list((tmp_path / "cache").glob("invest_probe_*.joblib"))) == 1


def test_changed_panel_bytes_are_a_cache_miss(tmp_path):
    from techval.invest.engine_read import cached_fit, fit_key

    panel = tmp_path / "panel.bin"
    modules = ("techval.invest.engine_read",)
    panel.write_bytes(b"one")
    before = fit_key("probe", [panel], modules)
    panel.write_bytes(b"two")
    assert fit_key("probe", [panel], modules) != before
    built = []
    cached_fit("probe", [panel], modules, lambda: built.append(1) or "a", tmp_path / "cache")
    panel.write_bytes(b"three")
    cached_fit("probe", [panel], modules, lambda: built.append(1) or "b", tmp_path / "cache")
    assert len(built) == 2


def test_an_unreadable_entry_is_a_miss_and_refit_rebuilds(tmp_path):
    from techval.invest.engine_read import cached_fit, fit_key

    panel = tmp_path / "panel.bin"
    panel.write_bytes(b"one")
    modules = ("techval.invest.engine_read",)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"invest_probe_{fit_key('probe', [panel], modules)}.joblib").write_bytes(b"not a pickle")
    assert cached_fit("probe", [panel], modules, lambda: "rebuilt", cache) == "rebuilt"
    assert cached_fit("probe", [panel], modules, lambda: "refit", cache, refit=True) == "refit"
    assert cached_fit("probe", [panel], modules, lambda: "never", cache) == "refit"


def test_no_cache_root_means_no_disk_cache(tmp_path):
    from techval.invest.engine_read import cached_fit

    calls = []
    for _ in range(2):
        cached_fit("probe", [], (), lambda: calls.append(1), None)
    assert len(calls) == 2
