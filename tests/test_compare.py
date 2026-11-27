"""techval compare: the same engine across a set of companies."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import date
from pathlib import Path

import pytest

from techval.compare import (
    build_comparison,
    measure_all,
    returns_figure,
    table_figure,
    takeaway,
    value_figure,
)
from techval.dashboard.render import render_dashboard
from techval.dashboard.sections import engine as E
from techval.dashboard.snapshot import Snapshot, load_snapshot
from techval.market import CsvSource
from techval.tearsheet import RecordedFacts

from conftest import FIXTURES, PRICES, pinned_assumptions

AS_OF = date(2026, 9, 9)


def sources(observations=None):
    return E.Sources(
        client=RecordedFacts(FIXTURES),
        prices=CsvSource(PRICES),
        record=lambda *args, **kwargs: nullcontext(),
        facts_name=str,
        prices_name=str,
        panel_name="fade panel",
        observations=observations,
    )


@pytest.fixture(scope="module")
def compared(fade_observations):
    return measure_all(["DDOG", "CRWD", "DIS", "NOPE"], pinned_assumptions(), sources(fade_observations), today=AS_OF)


def test_a_price_no_growth_reaches_is_the_one_that_asks_the_most(fade_observations):
    """At an 11% terminal margin no first-year growth up to 300% reaches CrowdStrike's close.

    That price needs more than any company-year compounded, which the screen
    ranks first, but its growth solve found no root, so the takeaway passed
    over it and named Datadog's 209%.
    """
    a = pinned_assumptions()
    a.dcf.ebit_margin_terminal = 0.11
    rows = measure_all(["DDOG", "CRWD"], a, sources(fade_observations), today=AS_OF)
    assert takeaway(rows, AS_OF.isoformat()).endswith(
        " CRWD's price asks the most: no first-year growth up to 300% reaches it."
    )
    assert rows[1].base.expectations.growth_missed == "above"


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


def test_the_set_can_be_its_own_peers(fade_observations):
    rows = measure_all(
        ["DDOG", "CRWD", "ZS"], pinned_assumptions(), sources(fade_observations), today=AS_OF, peers_from_set=True
    )
    assert set(rows[0].base.cost_of_capital.peers) <= {"CRWD", "ZS"} and rows[0].base.cost_of_capital.peers


def test_the_value_figure_puts_each_dcf_over_its_close(compared):
    fig = value_figure(compared)
    rows = {r["label"]: r["values"] for r in fig["data"]["rows"]}
    assert set(rows) == {"DDOG", "CRWD", "DIS"}
    dis = compared[2]
    assert rows["DIS"]["base"] == pytest.approx(dis.base.per_share_gordon / dis.base.price)
    assert fig["data"]["reference"][0]["value"] == 1.0
    # Disney's base case clears its close and its record case does not.
    assert fig["title"] == "1 of 3 base cases reach the close, and 0 do on the companies' own records"


def test_the_returns_figure_names_the_close_that_earns_least(compared):
    fig = returns_figure(compared)
    rows = {r["label"]: r["values"] for r in fig["data"]["rows"]}
    lowest = min(rows, key=lambda t: rows[t]["implied"])
    assert fig["title"].startswith(f"{lowest}'s close earns least")
    assert all(0 < v["wacc"] < 0.25 for v in rows.values())


def test_the_table_carries_every_reading_for_every_company_valued(compared):
    fig = table_figure(compared)
    rows = {r["company"]: r for r in fig["data"]["rows"]}
    assert list(rows) == ["DDOG", "CRWD", "DIS"]
    ddog = rows["DDOG"]
    assert ddog["base"] == pytest.approx(compared[0].base.per_share_gordon)
    assert ddog["record"] == pytest.approx(compared[0].record.per_share_gordon)
    assert "of 1,662" in ddog["did"] and ddog["against"].endswith("the bull")
    assert [c["key"] for c in fig["data"]["columns"]][:4] == ["company", "close", "base", "record"]


def test_the_takeaway_names_the_price_that_asks_the_most(compared):
    text = takeaway(compared, "2026-09-09")
    assert text.startswith("At the closes of 2026-09-09, the base case reaches the price of 1 of the 3")
    assert "CRWD's price asks the most" in text


def test_the_standfirst_names_what_a_record_case_left_at_the_assumption():
    snap = build_comparison(
        ["CRWD", "DDOG"], pinned_assumptions(), client=RecordedFacts(FIXTURES), prices=CsvSource(PRICES),
        price_feed="csv", observations=None, panel="x", today=AS_OF, collected_at="2026-09-09", commit="abc1234",
    )
    assert "set from its filings, except CRWD's D&A, which stays at the assumption." in snap.page["standfirst"]


def test_the_comparison_is_a_one_section_page_that_validates(fade_observations):
    snap = build_comparison(
        ["DDOG", "DIS", "NOPE"], pinned_assumptions(), client=RecordedFacts(FIXTURES), prices=CsvSource(PRICES),
        price_feed="csv", observations=fade_observations, panel="tests/fixtures/fade_companyfacts.json.gz",
        today=AS_OF, collected_at="2026-09-09", commit="abc1234",
    )
    back = Snapshot.from_dict(snap.to_dict())
    section = back.sections["compare"]
    assert back.title == "DDOG, DIS, NOPE, compared" and back.fixtures == {}
    assert set(section["figures"]) == {"value", "returns", "table"}
    assert [section["figures"][f]["order"] for f in ("value", "returns", "table")] == [1, 2, 3]
    assert [r["what"] for r in section["refusals"]] == ["NOPE"]
    assert "2 could be valued" in back.page["standfirst"]
    # NOPE has no filings to read; the sources name what was.
    assert back.page["sources"][:2] == [
        "SEC EDGAR companyfacts for DDOG, DIS",
        "csv daily closes for DDOG, DIS, through 2026-09-09",
    ]


def test_the_example_comparison_is_exactly_what_its_committed_snapshot_renders():
    docs = Path(__file__).parents[1] / "docs" / "compare"
    snap = load_snapshot(docs / "snapshot.json")
    assert snap.title == "CRWD, DDOG, MDB, ZS, DIS, compared"
    assert render_dashboard(snap) == (docs / "index.html").read_text(encoding="utf-8")
