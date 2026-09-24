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
