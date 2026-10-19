"""techval compare: the same engine across a set of companies."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import date
from pathlib import Path

import pytest

from techval.compare import measure_all
from techval.config import Assumptions
from techval.dashboard.sections import engine as E
from techval.market import CsvSource
from techval.tearsheet import RecordedFacts

FIXTURES = Path(__file__).parent / "fixtures"
AS_OF = date(2026, 9, 9)


@pytest.fixture(scope="module")
def panel():
    from techval.invest.engine_read import fade_panel

    return fade_panel(FIXTURES).observations


def sources(observations=None):
    return E.Sources(
        client=RecordedFacts(FIXTURES),
        prices=CsvSource(FIXTURES / "prices"),
        record=lambda *args, **kwargs: nullcontext(),
        facts_name=str,
        prices_name=str,
        panel_name="fade panel",
        observations=observations,
    )


def assumptions():
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    return a


@pytest.fixture(scope="module")
def compared(panel):
    return measure_all(["DDOG", "CRWD", "DIS", "NOPE"], assumptions(), sources(panel), today=AS_OF)


def test_every_company_keeps_its_place_and_one_with_no_filings_is_refused(compared):
    assert [c.ticker for c in compared] == ["DDOG", "CRWD", "DIS", "NOPE"]
    nope = compared[3]
    assert nope.base is None and "companyfacts_NOPE.json" in nope.refused


def test_each_company_is_measured_on_the_assumptions_and_on_its_own_record(compared):
    dis = compared[2]
    assert dis.base.anchors == [] and dis.record.anchors
    assert dis.record.per_share_gordon < dis.base.per_share_gordon
    # Neither carries the quarterly history, which no comparison draws.
    assert all(c.base.history is None and c.record.history is None for c in compared[:3])


def test_the_set_can_be_its_own_peers(panel):
    rows = measure_all(["DDOG", "CRWD", "ZS"], assumptions(), sources(panel), today=AS_OF, peers_from_set=True)
    assert set(rows[0].base.cost_of_capital.peers) <= {"CRWD", "ZS"} and rows[0].base.cost_of_capital.peers


def test_the_value_figure_puts_each_dcf_over_its_close(compared):
    from techval.compare import value_figure

    fig = value_figure(compared)
    rows = {r["label"]: r["values"] for r in fig["data"]["rows"]}
    assert set(rows) == {"DDOG", "CRWD", "DIS"}
    dis = compared[2]
    assert rows["DIS"]["base"] == pytest.approx(dis.base.per_share_gordon / dis.base.price)
    assert fig["data"]["reference"][0]["value"] == 1.0
    # Disney's base case clears its close and its record case does not.
    assert fig["title"] == "1 of 3 base cases reach the close, and 0 do on the companies' own records"


def test_the_returns_figure_names_the_close_that_earns_least(compared):
    from techval.compare import returns_figure

    fig = returns_figure(compared)
    rows = {r["label"]: r["values"] for r in fig["data"]["rows"]}
    lowest = min(rows, key=lambda t: rows[t]["implied"])
    assert fig["title"].startswith(f"{lowest}'s close earns least")
    assert all(0 < v["wacc"] < 0.25 for v in rows.values())
