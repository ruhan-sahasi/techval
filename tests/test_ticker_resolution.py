"""Resolving a ticker that no longer trades, which is most of M&A.

``techval precedents`` is a command about completed transactions, and before
this it could not reach one. Every path into it goes through
``EdgarClient.ticker_to_cik``, which read ``company_tickers.json`` and nothing
else, and that file lists registrants that still trade: SPLK, ZEN, MNDT, WORK
and TWTR are absent from it by construction, because each of those companies was
bought. The command's own help text offered ``SPLK,ZEN,MNDT,WORK`` as its
example and all four failed, with a hint blaming foreign filers.

Nothing here touches the network. The current ticker file is served from
``tests/fixtures/merger/company_tickers.json``, which is a pruned copy of the
real one and, like the real one, carries the acquirers and not the delisted
targets. The former-ticker index is a committed table read off the targets' own
cover pages, and the CIKs it returns are cross-checked here against two
fixtures recorded independently of it: the merger submissions payloads and the
M&A universe.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from techval.edgar import SEC_TICKERS_URL, EdgarClient, HttpCache
from techval.errors import MissingDataError
from techval.former_tickers import FORMER_TICKERS, FormerListing, resolve

FIXTURES = Path(__file__).parent / "fixtures"
TICKER_FILE = FIXTURES / "merger" / "company_tickers.json"
INDEX = FIXTURES / "former_tickers" / "index.json"

#: The four the precedents help text offers as its example, with the CIK each
#: one's own submissions payload carries in tests/fixtures/merger.
DELISTED_EXAMPLES = {"SPLK": 1353283, "ZEN": 1463172, "MNDT": 1370880, "WORK": 1764925}


class OfflineClient(EdgarClient):
    """The production client with the SEC's ticker file served from disk.

    Subclassed rather than reimplemented, so what these tests exercise is the
    resolution the engine runs and not a second copy of it.
    """

    def __init__(self, knowledge_date: date | None = None, extra: dict | None = None):
        super().__init__(cache=HttpCache(enabled=False), knowledge_date=knowledge_date)
        self.extra = extra or {}

    def _get_json(self, url: str) -> dict:
        if url != SEC_TICKERS_URL:
            raise AssertionError(f"a test reached for {url}, which is not a fixture")
        raw = {k: v for k, v in json.loads(TICKER_FILE.read_text()).items() if k != "_fixture"}
        raw.update(self.extra)
        return raw


# --------------------------------------------------------------------------- #
# The delisted targets
# --------------------------------------------------------------------------- #


def test_the_committed_ticker_file_really_does_lack_the_delisted_targets():
    """The premise of everything below, asserted rather than assumed."""
    payload = json.loads(TICKER_FILE.read_text())
    live = {v["ticker"].upper() for k, v in payload.items() if k != "_fixture"}
    assert not (live & set(DELISTED_EXAMPLES))
    assert "TXN" in live and "ROKU" in live


@pytest.mark.parametrize("ticker,cik", sorted(DELISTED_EXAMPLES.items()))
def test_a_delisted_target_resolves_to_the_cik_its_own_filings_carry(ticker, cik):
    """SPLK, ZEN, MNDT and WORK: the four the command's help text names.

    The expected CIK is not taken from the index under test. It is the one in
    ``tests/fixtures/merger/submissions_<TICKER>.json``, which was recorded from
    the submissions API months before this table existed.
    """
    recorded = json.loads((FIXTURES / "merger" / f"submissions_{ticker}.json").read_text())
    assert int(recorded["cik"]) == cik

    resolution = OfflineClient().resolve_ticker(ticker)
    assert resolution.cik == cik
    assert resolution.source == "former-ticker index"
    assert "delisting rather than an absence" in (resolution.note or "")


def test_every_row_agrees_with_the_mna_universe_recorded_separately():
    """The index is cross-checked against a fixture built without it.

    ``tests/fixtures/mna/universe.json`` carries the CIK and the last filing
    date of every departed registrant, recorded by a different script from a
    different endpoint. Where the two name the same company, the CIK has to
    match, and the date the symbol was last proven has to fall inside the filing
    history the universe records.
    """
    universe = {
        int(row["cik"]): row
        for row in json.loads((FIXTURES / "mna" / "universe.json").read_text())
        if row.get("departed")
    }
    checked = 0
    for symbol, claims in FORMER_TICKERS.items():
        for claim in claims:
            row = universe.get(claim.cik)
            if row is None:
                continue
            checked += 1
            assert claim.through <= date.fromisoformat(row["last_filing"]), symbol
            assert claim.since <= claim.through, symbol
    assert checked >= 90


def test_the_table_matches_the_filings_it_was_read_from():
    """The module and its evidence cannot drift apart without this turning red.

    ``tests/fixtures/former_tickers/index.json`` carries the accession and the
    document each symbol was read out of. The literal in the module is generated
    from it, and a generated literal that nobody checks is a literal somebody
    will hand-edit.
    """
    payload = json.loads(INDEX.read_text())
    expected = {
        symbol: tuple(
            FormerListing(
                int(c["cik"]),
                date.fromisoformat(c["since"]),
                date.fromisoformat(c["through"]),
                c["name"],
            )
            for c in claims
        )
        for symbol, claims in payload["entries"].items()
    }
    assert FORMER_TICKERS == expected
    for claims in payload["entries"].values():
        for claim in claims:
            assert claim["read_from"] in ("dei:TradingSymbol", "cover-page label")
            assert claim["accession"]


def test_the_registrants_it_could_not_read_are_recorded_rather_than_guessed():
    """Cover-page tagging began in 2019, so the older departures are unreachable.

    They are in the file under ``unresolved`` with the reason, and the resolver
    refuses them rather than reaching for a third-party symbol list. EMC and
    Twenty-First Century Fox are in that group.
    """
    payload = json.loads(INDEX.read_text())
    assert payload["unresolved"]
    for row in payload["unresolved"]:
        assert row["why"]
        assert row["cik"]


# --------------------------------------------------------------------------- #
# The CIK escape hatch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("form", ["CIK0000790070", "CIK790070", "cik-790070", "790070"])
def test_a_cik_can_be_passed_where_a_ticker_is_expected(form):
    """The identifier the SEC's own APIs are keyed on always works.

    It is the answer for a company this package has never heard of, and it is
    how ``tests/fixtures/mna/universe.json`` already keys a departed registrant:
    that fixture's own manifest says a delisted ticker was not recoverable when
    it was recorded, which is what made the escape hatch necessary.
    """
    resolution = OfflineClient().resolve_ticker(form)
    assert resolution.cik == 790070
    assert resolution.source == "explicit CIK"


def test_an_unknown_symbol_is_refused_with_the_cause_and_the_way_round_it():
    """The old hint blamed foreign filers for what is nearly always a delisting."""
    with pytest.raises(MissingDataError) as exc:
        OfflineClient().resolve_ticker("NOSUCHCO")
    message = str(exc.value)
    assert "still trade" in message
    assert "delisting" in message
    assert "CIK0001353283" in message
    # The old hint read "the engine covers US filers only", which sends a reader
    # looking for a foreign registrant when the company was American and bought.
    assert "US filers only" not in message


# --------------------------------------------------------------------------- #
# A ticker is not a durable identifier
# --------------------------------------------------------------------------- #


def test_a_reassigned_symbol_resolves_by_knowledge_date():
    """S was Sprint's until January 2020 and is SentinelOne's now.

    Both companies tag that one letter on their own cover pages, five years
    apart, so the symbol alone cannot decide between them and the knowledge date
    has to. A live run means today's holder; a run pinned inside Sprint's filing
    history means Sprint. This is the case that makes the whole table a list per
    symbol rather than one row.
    """
    sentinelone = {"9999": {"cik_str": 1583708, "ticker": "S", "title": "SentinelOne, Inc."}}

    live = OfflineClient(extra=sentinelone).resolve_ticker("S")
    assert live.cik == 1583708
    assert live.source == "SEC ticker file"

    then = OfflineClient(date(2019, 6, 30), extra=sentinelone).resolve_ticker("S")
    assert then.cik == 101830
    assert then.source == "former-ticker index"
    assert "SPRINT" in (then.note or "").upper()


def test_a_date_before_the_registrant_existed_is_refused_rather_than_answered():
    """ALTR was Altera's until 2015 and Altair Engineering's from 2017.

    Only Altair is in this table, because Altera left before cover pages were
    tagged. Answering a 2014 question with Altair would be a confident wrong
    company, so the claim is bounded below by the registrant's oldest filing and
    a date in front of that resolves to nothing.
    """
    assert resolve("ALTR", date(2021, 1, 1)) is not None
    assert resolve("ALTR", date(2014, 1, 1)) is None
    with pytest.raises(MissingDataError):
        OfflineClient(date(2014, 1, 1)).resolve_ticker("ALTR")


def test_one_symbol_two_registrants_of_the_same_name():
    """AZPN is CIK 929940's through April 2022 and CIK 1897982's afterwards."""
    claims = FORMER_TICKERS["AZPN"]
    assert len(claims) == 2
    assert [c.cik for c in claims] == [929940, 1897982]
    assert resolve("AZPN", date(2021, 1, 1)).cik == 929940
    assert resolve("AZPN", date(2024, 1, 1)).cik == 1897982
    assert resolve("AZPN").cik == 1897982


def test_the_index_agrees_with_the_fade_panels_hand_listed_table():
    """Two tables built by different means, checked against each other.

    ``ml.forecast.DELISTED_CIKS`` was written by hand for the fade panel. This
    one was read off cover pages. They overlap on 36 symbols and agree on 35.
    The exception is ALTR and it is not an error in either: the hand table means
    Altera, bought in 2015, and this one means Altair Engineering, bought in
    2025. That is the reassignment trap in the two tables rather than in a
    docstring, and it is why resolution here takes a date.
    """
    from techval.ml.forecast import DELISTED_CIKS

    overlap = sorted(set(FORMER_TICKERS) & set(DELISTED_CIKS))
    assert len(overlap) >= 30
    disagree = [
        t for t in overlap if DELISTED_CIKS[t] not in [c.cik for c in FORMER_TICKERS[t]]
    ]
    assert disagree == ["ALTR"]
