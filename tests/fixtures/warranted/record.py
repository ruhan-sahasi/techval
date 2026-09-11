"""Record the warranted-multiple observation panel from live SEC and price data.

Committed beside the fixture it writes, because that fixture is derived rather
than raw. The other fixtures in this tree are SEC payloads a reader can
recognise; this one is two thousand rows of computed multiples and features, and
the only way to audit it is to be able to rebuild it.

    python tests/fixtures/warranted/record.py <price-csv-dir> <out.json.gz>

``price-csv-dir`` holds one ``<TICKER>.csv`` of Date,Close per name, pulled once
from the configured price source and reused across every date, because a panel
of a hundred names over twenty-one quarter ends would otherwise be two thousand
price requests for twenty-one distinct answers.

The same argument applies to the filings and is the reason ``MemoisedFacts``
exists. A companyfacts payload runs to tens of megabytes, and a plain
``EdgarClient`` reads and parses it from the HTTP cache on every call: a hundred
names over twenty-one dates, twice each because the split basis needs the current
fact set as well, is four and a half thousand full parses of a large document to
answer four and a half thousand questions about a hundred of them. The payload is
parsed once per ticker and rewrapped in a ``CompanyFacts`` per date, which is the
only object that has to be per-date, because its knowledge date decides which
facts it will admit and which splits it has heard of.

Two properties of the run matter and both are recorded in the output:

*It is point in time per date.* A fact view is pinned to each date and a fresh
``MarketData`` is capped at it, and ``build_observations`` runs
``assert_point_in_time`` on every row, so a payload that leaked would raise here
rather than reach the fixture. The memoisation above changes none of that: the
parsed payload is the raw document, and it is the ``CompanyFacts`` wrapper that
decides which of its facts exist at a date, which is why a new wrapper is built
per date and only the parse is shared.

*Names with no price history are filtered before the panel, not inside it, and
are recorded under ``no_price_history``.* Handing one to ``build_observations``
would produce twenty-one identical skips saying the same thing about the same
company, which buries the twenty-one different skips that are worth reading. The
loss is a property of the universe rather than of a date, so it is recorded once.
It is not a small loss and it is the survivorship bias of methodology 11.3 in the
flesh: five of the seed universe are gone from the SEC's own
``company_tickers.json`` as well as from the price feed, so a panel built today
cannot see them at ANY historical date, including the dates on which they were
live and cheap.

*The price CSVs are an input and are not committed.* They are daily closes for
the seed universe from 2018 to the run date, pulled once from the configured
price source into ``<dir>/<TICKER>.csv``. They are not in the repository because a
hundred names of daily history is larger than the fixture it feeds, and because
the one property of them the panel depends on, that they are split adjusted to
the run date, is exactly the property ``share_basis_factor`` exists to reconcile.

*It is not reproducible from the network alone, and the fixture is the record.*
SEC restates, the price vendor revises, and a company that delists loses its
history entirely, so a rerun in six months will not produce the same file. That
is precisely why the output is committed and the tests read it.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import date
from pathlib import Path

from techval.config import Assumptions
from techval.edgar import CompanyFacts, EdgarClient, HttpCache
from techval.market import CsvSource, MarketData
from techval.ml.warranted import WARRANTED_FEATURES, build_observations
from techval.tmt.taxonomy import SEED, tmt_universe

FIRST = date(2021, 3, 31)
LAST = date(2026, 6, 30)


class MemoisedFacts:
    """An ``EdgarClient`` that parses each payload once and rewraps it per date.

    Identical to ``EdgarClient`` from the panel's point of view, and deliberately
    not a new caching layer inside ``edgar.py``: the engine's own cache is an HTTP
    cache and is right to be, since a valuation touches a handful of filers once.
    A panel touches a hundred filers twenty-one times, and that is a property of
    this script rather than of the engine.
    """

    def __init__(self, client: EdgarClient, knowledge_date: date | None) -> None:
        self.client = client
        self.knowledge_date = knowledge_date
        self.raw: dict[str, dict] = {}

    def payload(self, ticker: str) -> dict:
        key = ticker.upper()
        if key not in self.raw:
            self.raw[key] = self.client.company_facts(key).raw
        return self.raw[key]

    def company_facts(self, ticker: str) -> CompanyFacts:
        return CompanyFacts(
            self.payload(ticker), ticker, knowledge_date=self.knowledge_date
        )

    def ticker_to_cik(self, ticker: str) -> int:
        return int(self.payload(ticker)["cik"])

    def pinned(self, knowledge_date: date) -> "MemoisedFacts":
        """A view on the same parsed payloads pinned to another date."""
        view = MemoisedFacts(self.client, knowledge_date)
        view.raw = self.raw
        return view


def quarter_ends(first: date, last: date) -> list[date]:
    out = []
    for year in range(first.year, last.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            when = date(year, month, day)
            if first <= when <= last:
                out.append(when)
    return out


def main(price_dir: Path, out: Path) -> None:
    assumptions = Assumptions()
    # Pinned so the run never reaches the Treasury. Nothing in this panel uses a
    # discount rate, but ``build_features`` estimates a beta and a beta needs a
    # market feed rather than a rate, so this is belt and braces.
    assumptions.market.risk_free_rate = 0.0430

    cache = HttpCache()
    tickers = sorted(t for t in SEED if (price_dir / f"{t}.csv").exists())
    missing = sorted(set(SEED) - set(tickers))

    # Classified once, at the end of the window, and applied to every date. A
    # company's sub-vertical is not a quarterly variable, and re-resolving it at
    # each date would cost a submissions request per name per date to answer a
    # question whose answer does not move.
    universe = tmt_universe(EdgarClient(cache=cache), assumptions, as_of=LAST)
    verticals = {
        company.ticker: company.sub_vertical.value
        for company in universe
        if company.sub_vertical is not None
    }

    current = MemoisedFacts(EdgarClient(cache=cache), None)
    for position, ticker in enumerate(tickers, 1):
        try:
            current.payload(ticker)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"  {ticker}: {type(exc).__name__}: {exc}", flush=True)
        if position % 20 == 0:
            print(f"  parsed {position}/{len(tickers)} payloads", flush=True)

    def market_factory(when: date) -> MarketData:
        return MarketData(CsvSource(price_dir), HttpCache(enabled=False), today=when)

    panel = build_observations(
        [t for t in tickers if t in current.raw],
        quarter_ends(FIRST, LAST),
        current.pinned,
        market_factory,
        verticals,
        assumptions,
        current_client=current,
        target="ev_revenue",
    )

    payload = {
        "recorded": date.today().isoformat(),
        "target": panel.target,
        "first_date": FIRST.isoformat(),
        "last_date": LAST.isoformat(),
        "features": list(WARRANTED_FEATURES),
        "sub_verticals": verticals,
        "no_price_history": missing,
        "notes": panel.notes,
        "observations": [
            {
                "ticker": o.ticker,
                "as_of": o.as_of.isoformat(),
                "sub_vertical": o.sub_vertical,
                "multiple": round(o.multiple, 6),
                "enterprise_value": round(o.enterprise_value, 3),
                "equity_value": round(o.equity_value, 3),
                "denominator": round(o.denominator, 3),
                "statement_date": None if o.statement_date is None else o.statement_date.isoformat(),
                "basis_factor": o.basis_factor,
                "features": {
                    name: None if o.features.get(name) is None else round(float(o.features[name]), 8)
                    for name in WARRANTED_FEATURES
                },
            }
            for o in panel.observations
        ],
        "skips": [
            {
                "ticker": s.ticker,
                "as_of": s.as_of.isoformat(),
                "category": s.category,
                "reason": s.reason,
            }
            for s in panel.skips
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as handle:
        json.dump(payload, handle, sort_keys=True)
    print(
        f"{len(payload['observations'])} observations, {len(payload['skips'])} skips, "
        f"{len(tickers)} tickers, {len(missing)} without price history -> {out}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
