"""Rebuild the example tearsheets at docs/tearsheet, offline.

Run from the repository root:

    .venv/bin/python tests/fixtures/tearsheet/record_example.py

Two pages, from the committed companyfacts and closes, the suite's pinned
0.0483 risk-free rate and the fade panel. CrowdStrike against Datadog, MongoDB
and Zscaler on the assumptions file; Disney on its own record, dcf.from_record,
the case the assumptions file's software defaults were not written for. The
provenance run times are zeroed, since they are the only thing two rebuilds of
an unchanged tree would disagree on.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from techval.config import Assumptions
from techval.dashboard.collect import techval_commit
from techval.dashboard.render import write_dashboard
from techval.dashboard.snapshot import dump_snapshot
from techval.invest.engine_read import fade_panel
from techval.market import CsvSource
from techval.tearsheet import RecordedFacts, build_tearsheet

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures"
OUT = ROOT / "docs" / "tearsheet"
TODAY = date(2026, 9, 9)
# Each page: the ticker, its peers, and whether the DCF is the record case.
PAGES = (("CRWD", ["DDOG", "MDB", "ZS"], False), ("DIS", [], True))


def build(ticker: str, peers: list[str], from_record: bool) -> None:
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    a.comps.peers = peers
    a.dcf.from_record = from_record
    snap = build_tearsheet(
        ticker,
        a,
        client=RecordedFacts(FIXTURES),
        prices=CsvSource(FIXTURES / "prices"),
        price_feed="Committed fixture",
        observations=fade_panel(FIXTURES).observations,
        panel="tests/fixtures/fade_companyfacts.json.gz",
        today=TODAY,
        collected_at=TODAY.isoformat(),
        commit=techval_commit(ROOT),
    )
    for row in snap.sections["engine"]["provenance"]:
        row["seconds"] = 0.0
    dump_snapshot(snap, OUT / f"{ticker}.json")
    print(write_dashboard(snap, OUT / f"{ticker}.html"))


def main() -> None:
    for page in PAGES:
        build(*page)


if __name__ == "__main__":
    main()
