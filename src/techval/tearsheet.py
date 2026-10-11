"""techval tearsheet: one company's valuation as a single page.

The page is the results dashboard's valuation engine section measured for the
company asked about instead of Datadog: the same ``measure`` and the same
``shape``, so every figure, refusal and sentence is the one the results page
would draw for it. What differs is where the inputs come from. The results page
reads committed fixtures and digests them; a tearsheet reads EDGAR and the
configured price feed, so its provenance names those sources and the dates they
ran to, and its basis of preparation lists them in place of fixture digests.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import Assumptions
from .dashboard.collect import entry_module
from .dashboard.sections import engine as E
from .dashboard.snapshot import Snapshot, to_jsonable, validate_section


class RecordedFacts:
    """companyfacts payloads recorded as ``companyfacts_<TICKER>.json``, served as EdgarClient serves them."""

    def __init__(self, directory: Path, knowledge_date: date | None = None) -> None:
        self.directory = Path(directory)
        self.knowledge_date = knowledge_date

    def company_facts(self, ticker: str):
        from .edgar import CompanyFacts
        from .errors import MissingDataError

        path = self.directory / f"companyfacts_{ticker.upper()}.json"
        if not path.is_file():
            raise MissingDataError("companyfacts", ticker=ticker.upper(), hint=f"no recorded payload at {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CompanyFacts(payload, ticker, knowledge_date=self.knowledge_date)

    def ticker_to_cik(self, ticker: str) -> int:
        return self.company_facts(ticker).cik


class _Recorder:
    """Provenance rows for a live run: the entry point, and the sources it read, in words."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    @contextmanager
    def record(self, figure_id: str, entry_point: str, inputs: Sequence[str] = ()) -> Iterator[None]:
        entry_module(entry_point)
        start = time.perf_counter()
        yield
        self.rows.append(
            {
                "figure": figure_id,
                "entry_point": entry_point,
                "inputs": sorted(set(inputs)),
                "seconds": round(time.perf_counter() - start, 2),
            }
        )


def build_tearsheet(
    ticker: str,
    assumptions: Assumptions,
    *,
    client: Any,
    prices: Any,
    price_feed: str,
    observations: Sequence | None,
    panel: str,
    today: date,
    collected_at: str,
    commit: str,
) -> Snapshot:
    """Measure one company through the engine section and wrap it as a one-section page.

    ``client`` answers ``company_facts`` and ``ticker_to_cik`` as ``EdgarClient``
    does; ``prices`` is a price source; ``price_feed`` names it for the reader.
    ``observations`` is the fade panel, or None, in which case the base rates
    and the scenarios are refused rather than drawn.
    """
    symbol = ticker.upper()
    a = assumptions.model_copy(deep=True)
    peers = [p.upper() for p in a.comps.peers]
    a.comps.peers = peers
    recorder = _Recorder()
    sources = E.Sources(
        client=client,
        prices=prices,
        record=recorder.record,
        facts_name=lambda t: f"SEC EDGAR companyfacts, {t}",
        prices_name=lambda t: f"{price_feed} daily closes, {t}",
        panel_name=f"fade panel, {panel}",
        observations=observations,
    )
    measured = E.measure(symbol, peers, a, sources, today=today)
    shaped = E.shape(measured)
    section = to_jsonable(
        {
            "id": E.ID,
            "title": E.TITLE,
            "takeaway": shaped.get("takeaway", ""),
            "status": shaped["status"],
            "refusals": shaped.get("refusals", []),
            "headline": None,
            "figures": shaped.get("figures", {}),
            "provenance": recorder.rows,
        }
    )
    validate_section(section)
    return Snapshot(
        title=f"{symbol}, valued by the engine",
        techval_commit=commit,
        collected_at=collected_at,
        sections={E.ID: section},
        page={
            "standfirst": _standfirst(measured, peers, observations is not None),
            "sources": _sources(measured, peers, price_feed, panel, observations is not None),
        },
    )


def _standfirst(m: E.Measured, peers: list[str], panel: bool) -> str:
    comps = f"trading comps against {', '.join(peers)}" if peers else "no trading comps, since no peers are set"
    reads = (
        "what the price assumes and bear, base and bull from what similar company-years did"
        if panel
        else "no base rates or scenarios, since no fade panel was supplied"
    )
    return (
        f"{m.ticker} through the same engine as the techval results page: the cost of capital, "
        f"a DCF with its bridge and sensitivity, {comps}, a Monte Carlo of value, APV, and {reads}. "
        f"Its filings run through {m.filings_through} and its closes through {m.price_date}. Every "
        "figure is the engine's own, and where it could not form one a note says so in its place. "
        "Nothing here is advice."
    )


def _sources(m: E.Measured, peers: list[str], price_feed: str, panel: str, has_panel: bool) -> list[str]:
    names = ", ".join([m.ticker, *peers])
    out = [
        f"SEC EDGAR companyfacts for {names}, {m.ticker}'s filings through {m.filings_through}",
        f"{price_feed} daily closes for {names}, through {m.price_date}",
        f"Risk-free rate: {_rate_source(m.cost_of_capital.risk_free_source)}",
    ]
    if has_panel:
        out.append(f"Fade panel of point-in-time TMT company-years: {panel}")
    return out


def _rate_source(source: str) -> str:
    return "market.risk_free_rate in the assumptions" if source == "assumptions file override" else source
