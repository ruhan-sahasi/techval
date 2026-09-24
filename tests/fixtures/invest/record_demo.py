"""Rebuild the demo investing snapshot at docs/invest/snapshot.json, offline.

Run from the repository root:

    .venv/bin/python tests/fixtures/invest/record_demo.py

Everything comes from committed inputs: the fictional portfolio beside this
script, the daily closes under tests/fixtures/prices, the fade and warranted
panels, and DDOG's companyfacts fixture for the one DCF the demo can run. The
risk-free rate is the suite's pinned 0.0483, not a Treasury quote, for the
reason docs/dashboard/assumptions.yaml gives. TODAY is pinned to the last date
the committed closes share, so a rebuild of an unchanged tree is byte
identical.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from techval.config import Assumptions  # noqa: E402
from techval.edgar import CompanyFacts, HttpCache  # noqa: E402
from techval.invest.ledger import Ledger  # noqa: E402
from techval.invest.quotes import Quotes  # noqa: E402
from techval.invest.snapshot import build_snapshot, write_snapshot  # noqa: E402
from techval.market import CsvSource, MarketData  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
OUT = ROOT / "docs" / "invest" / "snapshot.json"
TODAY = date(2026, 9, 9)


def facts_for(symbol: str) -> CompanyFacts | None:
    path = FIXTURES / f"companyfacts_{symbol}.json"
    if not path.is_file():
        return None
    return CompanyFacts(json.loads(path.read_text()), symbol)


def build() -> dict:
    ledger = Ledger.load(FIXTURES / "invest" / "portfolio.yaml")
    quotes = Quotes(CsvSource(FIXTURES / "prices"), start=ledger.first_date, today=TODAY)
    assumptions = Assumptions()
    assumptions.market.risk_free_rate = 0.0483
    market = MarketData(CsvSource(FIXTURES / "prices"), HttpCache(enabled=False), today=TODAY)
    return build_snapshot(
        ledger,
        quotes,
        fixtures=FIXTURES,
        assumptions=assumptions,
        today=TODAY,
        facts_for=facts_for,
        market=market,
    )


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_snapshot(build(), OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
