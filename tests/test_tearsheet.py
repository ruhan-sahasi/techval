"""A tearsheet is the results page's valuation section, measured for any company."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.dashboard.snapshot import Snapshot
from techval.market import CsvSource
from techval.tearsheet import RecordedFacts, build_tearsheet

FIXTURES = Path(__file__).parent / "fixtures"
AS_OF = date(2026, 9, 9)


@pytest.fixture(scope="module")
def panel():
    from techval.ml.panels import fade_panel

    return fade_panel(FIXTURES).observations


def sheet(ticker, peers=(), observations=None):
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    a.comps.peers = list(peers)
    return build_tearsheet(
        ticker, a, client=RecordedFacts(FIXTURES), prices=CsvSource(FIXTURES / "prices"), price_feed="csv",
        observations=observations, panel="tests/fixtures/fade_companyfacts.json.gz",
        today=AS_OF, collected_at="2026-10-11", commit="abc1234",
    )


def test_datadogs_tearsheet_draws_the_results_pages_figures(panel):
    from techval.dashboard.collect import CollectContext, collect_section
    from techval.dashboard.sections import engine as E

    snap = sheet("DDOG", ("CRWD", "MDB", "ZS"), panel)
    page = snap.sections["engine"]
    results, _ = collect_section(E, CollectContext(FIXTURES, Path(__file__).parents[1], Assumptions()), use_cache=False)
    assert set(page["figures"]) == set(results["figures"])
    for fid, fig in results["figures"].items():
        if fid == "cost_of_capital":
            continue  # its risk-free tile says the rate was configured, not pinned
        assert page["figures"][fid] == fig, fid
    assert page["takeaway"] == results["takeaway"]


def test_the_page_names_its_live_sources_and_validates(panel):
    snap = sheet("DDOG", ("CRWD", "MDB", "ZS"), panel)
    back = Snapshot.from_dict(snap.to_dict())
    assert back.fixtures == {} and list(back.sections) == ["engine"]
    assert back.title == "DDOG, valued by the engine"
    sources = back.page["sources"]
    assert sources[0].startswith("SEC EDGAR companyfacts for DDOG, CRWD, MDB, ZS")
    assert "market.risk_free_rate in the assumptions" in sources[2]
    assert "trading comps against CRWD, MDB, ZS" in back.page["standfirst"]
    inputs = {i for row in back.sections["engine"]["provenance"] for i in row["inputs"]}
    assert "SEC EDGAR companyfacts, DDOG" in inputs and "csv daily closes, DDOG" in inputs


def test_a_company_with_no_peers_or_panel_still_gets_a_page_that_says_what_is_missing():
    snap = sheet("CRWD")
    page = snap.sections["engine"]
    assert page["status"] == "ok"
    refused = {r["what"] for r in page["refusals"]}
    assert {"Trading comps", "What the price assumes", "Bear, base and bull"} <= refused
    assert "no trading comps" in snap.page["standfirst"] and "no fade panel" in snap.page["standfirst"]
    assert not any("Fade panel" in s for s in snap.page["sources"])


@pytest.mark.parametrize("ticker", ["CRWD", "DIS"])
def test_each_example_page_is_exactly_what_its_committed_snapshot_renders(ticker):
    from techval.dashboard.render import render_dashboard
    from techval.dashboard.snapshot import load_snapshot

    docs = Path(__file__).parents[1] / "docs" / "tearsheet"
    snap = load_snapshot(docs / f"{ticker}.json")
    assert snap.title == f"{ticker}, valued by the engine" and snap.page is not None
    assert render_dashboard(snap) == (docs / f"{ticker}.html").read_text(encoding="utf-8")


def test_the_disney_example_is_on_its_own_record():
    from techval.dashboard.snapshot import load_snapshot

    snap = load_snapshot(Path(__file__).parents[1] / "docs" / "tearsheet" / "DIS.json")
    assert "record" in snap.sections["engine"]["figures"]
    assert "The DCF is the record case" in snap.page["standfirst"]
