"""Fixtures. No test in this suite touches the network.

Company facts are frozen SEC `companyfacts` payloads retrieved on 2026-09-10,
pruned to the us-gaap concepts techval reads and to facts ending on or after
2023-01-01. Prices are frozen daily closes from the same date. Between them the
whole engine runs offline and deterministically.

The companies are chosen for what they break:

    DDOG  calendar fiscal year, deep in-the-money convertibles, thin GAAP EBITDA
    CRWD  January fiscal year end, a four-for-one split mid-2026, no combined
          D&A tag, non-controlling interest
    MDB   January fiscal year end, finance leases as well as operating leases
    ZS    January fiscal year end, marketable securities under a tag outside the
          obvious ladder
    VZ    debt under a concept that bundles finance leases with it, a current
          portion under a third concept, and a LongTermDebtNoncurrent whose
          newest value is dated 2013
    DIS   segment reporting, and a media pack the TMT commands read
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path

# No test reads or writes the real ~/.techval. The session gets a home of its
# own before anything from techval is imported, since the HTTP cache root is
# fixed at import, and the directory goes when the session ends. matplotlib's
# font list is not techval's and takes twenty seconds to rebuild, so it stays
# where it was.
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".matplotlib"))
_HOME = tempfile.mkdtemp(prefix="techval-tests-home-")
os.environ["HOME"] = _HOME
os.environ["TECHVAL_CACHE"] = str(Path(_HOME) / ".techval" / "cache")
atexit.register(shutil.rmtree, _HOME, True)

import pytest  # noqa: E402

from techval.config import Assumptions  # noqa: E402
from techval.edgar import CompanyFacts, DimensionedFact, HttpCache
from techval.ev_bridge import build_ev_bridge
from techval.financials import Financials, build_financials
from techval.market import CsvSource, MarketData
from techval.wacc import compute_wacc

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = FIXTURES / "prices"

# The date the fixtures were retrieved. Pinning it keeps the 52-week window and
# the beta lookback fixed, so every assertion below stays true indefinitely.
AS_OF = date(2026, 9, 10)


def load_payload(ticker: str) -> dict:
    """A committed companyfacts payload as plain JSON, for a test that edits it."""
    return json.loads((FIXTURES / f"companyfacts_{ticker.upper()}.json").read_text())


def load_facts(ticker: str, knowledge_date: date | None = None) -> CompanyFacts:
    return CompanyFacts(load_payload(ticker), ticker, knowledge_date=knowledge_date)


def load_instance_file(path: Path) -> tuple[list[DimensionedFact], str, date]:
    """A pruned XBRL instance document: its facts, accession number and filing date."""
    payload = json.loads(path.read_text())
    facts = [
        DimensionedFact(
            tag=row["tag"],
            value=row["value"],
            unit=row["unit"],
            start=date.fromisoformat(row["start"]) if row["start"] else None,
            end=date.fromisoformat(row["end"]),
            dimensions=row["dimensions"],
        )
        for row in payload["facts"]
    ]
    return facts, payload["_accession"], date.fromisoformat(payload["_filed"])


def load_instance(ticker: str) -> tuple[list[DimensionedFact], str, date]:
    return load_instance_file(FIXTURES / f"instance_facts_{ticker}.json")


def facts_payload(series: dict[str, list[tuple[str, str, float]]]) -> dict:
    """A companyfacts payload written by hand: (start, end, value) rows in USD per tag.

    Each row is filed on its own end date, as a 10-K when it spans a year and as
    a 10-Q otherwise.
    """

    def row(start: str, end: str, value: float) -> dict:
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days
        return {"start": start, "end": end, "val": value, "form": "10-K" if days > 300 else "10-Q", "filed": end}

    us_gaap = {tag: {"units": {"USD": [row(*r) for r in rows]}} for tag, rows in series.items()}
    return {"cik": 1, "entityName": "Hand Written Inc", "facts": {"us-gaap": us_gaap}}


# A hand-written filer's lines that its example leaves out: None where the
# statements allow it, zero on the balance sheet.
_ABSENT_LINES = dict(
    gross_profit=None,
    da=None,
    sbc=None,
    pretax_income=None,
    tax_expense=None,
    interest_expense=None,
    operating_lease_cost=None,
    capex=None,
    cfo=None,
    basic_shares=None,
    current_assets=None,
    current_liabilities=None,
    cash=0.0,
    short_term_investments=0.0,
    straight_debt=0.0,
    convertible_debt=0.0,
    operating_lease_liability=0.0,
    finance_lease_liability=0.0,
    nci=0.0,
    preferred=0.0,
    deferred_revenue=0.0,
)


def hand_financials(**lines) -> Financials:
    """Statements written by hand, naming only the lines a worked example uses."""
    return Financials(**{**_ABSENT_LINES, **lines})


def offline_config(tmp_path: Path, extra: str = "") -> Path:
    """An assumptions file that keeps a command offline and its caches in ``tmp_path``.

    The committed closes for prices, the pinned risk-free rate, and the retrieval
    date as ``as_of``. ``extra`` is appended as YAML.
    """
    path = tmp_path / "assumptions.yaml"
    path.write_text(
        "market:\n  risk_free_rate: 0.0483\n"
        f"price_source: csv\nprice_csv_dir: {PRICES}\n"
        f"as_of: '{AS_OF.isoformat()}'\n"
        f"ml:\n  cache_dir: {tmp_path / 'ml'}\n" + extra,
        encoding="utf-8",
    )
    return path


def flat(text: str) -> str:
    """Command output with its colour codes stripped and its whitespace collapsed.

    Rich wraps at the console width, so a sentence a test cares about can be
    split across two lines with the break falling anywhere. Matching on the
    collapsed text asserts the sentence rather than the wrapping.
    """
    import re

    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())


def usage_error(output: str) -> str:
    """Typer's usage error as one line of plain text.

    Typer draws the error in a box, so a long message wraps between border
    characters, and under GITHUB_ACTIONS it forces colour on, so the message
    carries escape codes that ``flat`` strips.
    """
    return flat(output.replace("\u2502", " "))


class FixtureClient:
    """Stands in for EdgarClient, reading the committed payloads."""

    def company_facts(self, ticker: str) -> CompanyFacts:
        return load_facts(ticker)

    def ticker_to_cik(self, ticker: str) -> int:
        return load_facts(ticker).cik


@pytest.fixture
def client() -> FixtureClient:
    return FixtureClient()


def market_at(today: date) -> MarketData:
    """Market data from the committed closes, as it stood on ``today``."""
    return MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=today)


@pytest.fixture
def market() -> MarketData:
    return market_at(AS_OF)


def pinned_assumptions() -> Assumptions:
    """Defaults, with a risk-free rate pinned so no test reaches the Treasury."""
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    return a


@pytest.fixture
def assumptions() -> Assumptions:
    """The pinned defaults, with the three comparables the engine tests value against."""
    a = pinned_assumptions()
    a.comps.peers = ["CRWD", "MDB", "ZS"]
    return a


@pytest.fixture(scope="module")
def fade_observations():
    """Every observation in the committed fade panel, loaded once per test module."""
    from techval.ml.panels import fade_panel

    return fade_panel(FIXTURES).observations


@pytest.fixture(scope="module")
def real_dataset():
    """The TMT seed universe's disclosed peer groups, read from proxies on 2026-09-11.

    Read with the peers command's own readers. The panel and the Item 1 excerpts
    behind it are point in time: each row was built through an EdgarClient pinned
    to the row date and each document is the newest 10-K on file at that date.
    See the fixture headers.
    """
    from techval.commands_peers import _load_corpora, _load_groups, _load_panel
    from techval.ml.encoder import build_dataset

    groups, panel, item1 = (
        FIXTURES / f"peer_{part}_tmt.json" for part in ("groups", "panel", "item1")
    )
    for path in (groups, panel, item1):
        if not path.exists():
            pytest.skip(f"{path.name} is not committed")
    return build_dataset(_load_groups(groups), _load_panel(panel), _load_corpora(item1))


@pytest.fixture
def ddog():
    return build_financials("DDOG", facts=load_facts("DDOG"))


@pytest.fixture
def crwd():
    return build_financials("CRWD", facts=load_facts("CRWD"))


@pytest.fixture
def mdb():
    return build_financials("MDB", facts=load_facts("MDB"))


@pytest.fixture
def zs():
    return build_financials("ZS", facts=load_facts("ZS"))


@pytest.fixture
def vz():
    """Verizon: the debt ladder's worst case, and the reason it was rewritten.

    Long-term debt under ``LongTermDebtAndCapitalLeaseObligations`` with finance
    leases inside it, a current portion under a different concept again, and a
    ``LongTermDebtNoncurrent`` whose newest value is dated 2013. The fixture
    keeps the debt and lease concepts back to the first filing rather than
    trimming them at 2023, because the thirteen-year-old value IS the test.
    """
    return build_financials("VZ", facts=load_facts("VZ"))


@pytest.fixture
def ddog_bridge(ddog, assumptions, market):
    return build_ev_bridge(ddog, market.spot("DDOG"), assumptions)


@pytest.fixture
def wacc_result(ddog, ddog_bridge, market, assumptions):
    """Datadog's WACC on the shared assumptions, which the DCF tests start from."""
    return compute_wacc(ddog, ddog_bridge, market, assumptions)
