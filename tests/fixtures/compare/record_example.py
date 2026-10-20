"""Rebuild the example comparison at docs/compare, offline.

Run from the repository root:

    .venv/bin/python tests/fixtures/compare/record_example.py

The five companies the fixtures carry, from the committed companyfacts and
closes, the suite's pinned 0.0483 risk-free rate and the fade panel, each
valued on the assumptions file and on its own record. The provenance run times
are zeroed, since they are the only thing two rebuilds of an unchanged tree
would disagree on.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from techval.compare import build_comparison
from techval.config import Assumptions
from techval.dashboard.collect import techval_commit
from techval.dashboard.render import write_dashboard
from techval.dashboard.snapshot import dump_snapshot
from techval.ml.panels import fade_panel
from techval.market import CsvSource
from techval.tearsheet import RecordedFacts

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures"
OUT = ROOT / "docs" / "compare"
TICKERS = ["CRWD", "DDOG", "MDB", "ZS", "DIS"]
TODAY = date(2026, 9, 9)


def main() -> None:
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    snap = build_comparison(
        TICKERS,
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
    for row in snap.sections["compare"]["provenance"]:
        row["seconds"] = 0.0
    dump_snapshot(snap, OUT / "snapshot.json")
    print(write_dashboard(snap, OUT / "index.html"))


if __name__ == "__main__":
    main()
