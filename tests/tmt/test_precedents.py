"""Precedent transactions, against nine real TMT merger filings.

Every filing, fact set and price series under ``tests/fixtures/merger`` came
from SEC EDGAR or the Nasdaq daily quote endpoint on 2026-09-11 and is pruned as
``MANIFEST.json`` records. No test here touches the network.

The nine deals are chosen for the phrasings and the failures they carry:

    SLAB  Texas Instruments, $231.00 all cash, and a fact set whose FY2024 net
          income is simply absent from companyfacts, so the multiples cannot be
          built and have to say so
    RAMP  Publicis through a bidco, $38.50 all cash, with the ultimate parent
          named as a separate party in the same sentence
    PAYO  Nuvei through a Canadian holding company, $7.40 all cash, and a stock
          that had already run a month before the announcement
    ROKU  Fox, 0.9693 Class A shares plus $96.00 in cash, a dual-class buyer,
          and a leak that moved the stock a full trading day before the
          agreement was even signed
    IRDM  Rocket Lab, $27.00 in cash plus a collared exchange ratio, which is
          the case where no offer price exists at announcement
    SPLK  Cisco, $157.00 all cash, phrased without the words "per share", and a
          near-zero GAAP EBITDA that pushes EV/EBITDA past the NM cut-off
    MNDT  Google LLC, $23.00 all cash, a buyer that is not in the SEC ticker
          file, and an SIC code that files a security software company under
          computer peripherals
    ZEN   Hellman and Friedman with Permira through Zoro BidCo, $77.50 all cash,
          negative EBITDA, and a 2021 S-4 in which Zendesk is the acquirer
    WORK  Salesforce, 0.0776 shares plus $26.79 in cash, which marks out at the
          $45.86 the deal was reported at, and a share count the engine cannot
          build

Two documents are here to be refused: Splunk's June 2021 Item 1.01 8-K, which is
a convertible note sale carrying the phrase "an initial conversion price of
$160.00 per share", and Zendesk's December 2021 S-4 to buy Momentive.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from techval.config import Assumptions
from techval.edgar import (
    SEC_FACTS_URL,
    SEC_SUBMISSIONS_URL,
    SEC_TICKERS_URL,
    EdgarClient,
    HttpCache,
)
from techval.market import CsvSource
from techval.tmt import precedents as P

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "merger"
PRICES = FIXTURES / "prices"

#: The date the merger fixtures were retrieved. Every assertion about what was
#: pending and what had closed is fixed to it.
MERGER_AS_OF = date(2026, 9, 11)

DEAL_TICKERS = ["SLAB", "RAMP", "PAYO", "ROKU", "IRDM", "SPLK", "MNDT", "ZEN", "WORK"]

# The two documents that must not become precedents, by accession.
SPLUNK_CONVERTIBLE_NOTES = "0001104659-21-084180"
ZENDESK_AS_ACQUIRER = "0001193125-21-349135"


# Parsed fixtures, kept across clients. The production client has HttpCache for
# exactly this; without it the point-in-time test reparses a hundred kilobytes
# of company facts once per ticker per knowledge date.
_JSON_CACHE: dict[str, dict] = {}
_TEXT_CACHE: dict[str, str] = {}


def _read_json(path: Path) -> dict:
    key = str(path)
    if key not in _JSON_CACHE:
        _JSON_CACHE[key] = json.loads(path.read_text())
    return _JSON_CACHE[key]


class MergerFixtureClient(EdgarClient):
    """The real EdgarClient with its three JSON endpoints served from disk.

    Subclassing rather than reimplementing means ``filings``, ``ticker_to_cik``
    and the knowledge-date filtering under test are the production code paths
    and not a second implementation that could agree with the tests while
    disagreeing with the engine.
    """

    def __init__(self, knowledge_date: date | None = None) -> None:
        super().__init__(cache=HttpCache(enabled=False), knowledge_date=knowledge_date)
        self._by_cik: dict[int, str] = {}
        for ticker in DEAL_TICKERS:
            payload = _read_json(FIXTURES / f"submissions_{ticker}.json")
            self._by_cik[int(payload["cik"])] = ticker

    def _get_json(self, url: str) -> dict:
        if url == SEC_TICKERS_URL:
            raw = _read_json(FIXTURES / "company_tickers.json")
            return {k: v for k, v in raw.items() if k != "_fixture"}
        for cik, ticker in self._by_cik.items():
            if url == SEC_SUBMISSIONS_URL.format(cik=cik):
                return _read_json(FIXTURES / f"submissions_{ticker}.json")
            if url == SEC_FACTS_URL.format(cik=cik):
                return _read_json(FIXTURES / f"companyfacts_{ticker}.json")
        raise AssertionError(f"a test reached for {url}, which is not a fixture")

    def ticker_to_cik(self, ticker: str) -> int:
        for cik, known in self._by_cik.items():
            if known == ticker.upper():
                return cik
        return super().ticker_to_cik(ticker)

    def filing_text(self, ticker: str, filing: dict) -> str:
        # Most filings in the pruned submissions index have no committed text,
        # which stands in for a document that carries no merger agreement.
        return load_text(filing["accession"], missing_ok=True)


def load_text(accession: str, *, missing_ok: bool = False) -> str:
    if accession not in _TEXT_CACHE:
        path = FIXTURES / "text" / f"{accession}.txt"
        if not path.exists():
            if missing_ok:
                return ""
            raise AssertionError(f"no committed text for {accession}")
        _TEXT_CACHE[accession] = path.read_text()
    return _TEXT_CACHE[accession]


@pytest.fixture
def merger_client() -> MergerFixtureClient:
    return MergerFixtureClient(knowledge_date=MERGER_AS_OF)


@pytest.fixture
def merger_assumptions() -> Assumptions:
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    a.as_of = MERGER_AS_OF.isoformat()
    # The default 250mm floor would be exercised separately; the fixture set is
    # about extraction and arithmetic, so nothing is dropped for size here.
    a.ml.mna.min_deal_size = 0.0
    return a


@pytest.fixture
def merger_prices() -> CsvSource:
    return CsvSource(PRICES)


@pytest.fixture
def precedents(merger_client, merger_assumptions, merger_prices) -> P.PrecedentSet:
    return P.build_precedents(
        DEAL_TICKERS, merger_client, merger_assumptions, prices=merger_prices
    )


def by_ticker(result: P.PrecedentSet, ticker: str) -> P.Transaction:
    found = [t for t in result.transactions if t.target_ticker == ticker]
    assert found, f"{ticker} produced no transaction"
    return found[0]


# --------------------------------------------------------------------------- #
# Finding the deal documents
# --------------------------------------------------------------------------- #


def test_find_merger_filings_returns_only_deal_forms(merger_client):
    found = P.find_merger_filings("SPLK", merger_client)
    assert found, "Splunk filed a merger proxy and an 8-K inside ten years"
    assert {f["form"] for f in found} <= set(P.DEAL_FORMS)
    filed = [f["filed"] for f in found]
    assert filed == sorted(filed, reverse=True), "newest first"


def test_find_merger_filings_honours_the_lookback(merger_client):
    long_window = P.find_merger_filings("SPLK", merger_client, lookback_years=10)
    short_window = P.find_merger_filings("SPLK", merger_client, lookback_years=1)
    assert len(short_window) < len(long_window)
    assert all(f["filed"] >= date(2025, 9, 11) for f in short_window)


def test_find_merger_filings_prefers_the_8k_on_a_shared_filing_date(merger_client):
    found = P.find_merger_filings("ZEN", merger_client)
    same_day = [f for f in found if f["filed"] == date(2022, 6, 24)]
    assert len(same_day) > 1, "Zendesk filed an 8-K and proxy material that day"
    assert same_day[0]["form"] == "8-K"


# --------------------------------------------------------------------------- #
# Parsing the price, one phrasing at a time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "ticker,accession,price,acquirer_fragment",
    [
        # "converted into the right to receive $157.00 in cash", with no
        # "per share" anywhere near the figure.
        ("SPLK", "0001104659-23-102594", 157.00, "Cisco"),
        # "$23.00 in cash, without interest", and a buyer that is an LLC.
        ("MNDT", "0001104659-22-031786", 23.00, "Google"),
        # "$77.50 in cash, without interest" behind a three-clause exclusion.
        ("ZEN", "0001193125-22-181655", 77.50, "Zoro BidCo"),
        # "$231.00 in cash, without interest".
        ("SLAB", "0001193125-26-036712", 231.00, "Texas Instruments"),
        # "$38.50 in cash, without interest" through a bidco.
        ("RAMP", "0001104659-26-062908", 38.50, "MMS USA Holdings"),
        # "$7.40 in cash, without interest".
        ("PAYO", "0000950103-26-008945", 7.40, "Neon Maple"),
    ],
)
def test_all_cash_prices_parse(
    merger_client, ticker, accession, price, acquirer_fragment
):
    filing = {"accession": accession, "form": "8-K", "filed": date(2026, 1, 1)}
    txn = P.extract_transaction(load_text(accession), ticker, merger_client, filing)
    assert txn is not None
    assert txn.consideration == "cash"
    assert txn.offer_price == pytest.approx(price)
    assert txn.cash_per_share == pytest.approx(price)
    assert txn.pct_cash == 1.0
    assert txn.exchange_ratio is None
    assert txn.confidence == pytest.approx(0.90)
    assert acquirer_fragment in (txn.acquirer_name or "")


def test_par_value_is_never_read_as_an_offer_price(merger_client):
    """Every one of these filings says "par value $0.001 per share" nearby."""
    for accession in (
        "0001104659-23-102594",
        "0001193125-22-181655",
        "0001193125-26-036712",
    ):
        text = P._normalise(load_text(accession))
        assert "par value $0.0" in text
    filing = {"accession": "0001193125-22-181655", "form": "8-K", "filed": date(2022, 6, 24)}
    txn = P.extract_transaction(load_text(filing["accession"]), "ZEN", merger_client, filing)
    assert txn is not None
    assert txn.offer_price == pytest.approx(77.50)


def test_mixed_consideration_carries_both_legs(merger_client):
    """Slack: 0.0776 Salesforce shares and $26.79 in cash for each share."""
    accession = "0001193125-20-307385"
    filing = {"accession": accession, "form": "8-K", "filed": date(2020, 12, 1)}
    txn = P.extract_transaction(load_text(accession), "WORK", merger_client, filing)
    assert txn is not None
    assert txn.consideration == "mixed"
    assert txn.cash_per_share == pytest.approx(26.79)
    assert txn.exchange_ratio == pytest.approx(0.0776)
    # Not yet priced: a stock leg has no value until the buyer's close is read.
    assert txn.offer_price is None
    assert txn.acquirer_ticker == "CRM", "salesforce.com, inc. resolves to CRM"
    assert txn.agreement_date == date(2020, 12, 1)


def test_stock_only_consideration_is_not_read_as_cash(merger_client):
    """Roku's clause states an exchange ratio and a cash amount together."""
    accession = "0001140361-26-025115"
    filing = {"accession": accession, "form": "8-K", "filed": date(2026, 6, 15)}
    txn = P.extract_transaction(load_text(accession), "ROKU", merger_client, filing)
    assert txn is not None
    assert txn.consideration == "mixed"
    assert txn.exchange_ratio == pytest.approx(0.9693)
    assert txn.cash_per_share == pytest.approx(96.00)
    assert txn.offer_price is None
    assert txn.acquirer_ticker == "FOXA", "the class A ticker, not FOX"
    assert any("FOX, FOXA" in n for n in txn.notes)


def test_a_collared_exchange_ratio_produces_no_offer_price(merger_client):
    """Iridium pays $27.00 plus a ratio the buyer's closing price decides.

    Reporting the $27.00 cash leg as the offer price would be the worst kind of
    wrong number: a real figure standing in the wrong role. The transaction is
    kept, the cash leg is kept, and the price is refused.
    """
    accession = "0001104659-26-078482"
    filing = {"accession": accession, "form": "8-K", "filed": date(2026, 6, 29)}
    txn = P.extract_transaction(load_text(accession), "IRDM", merger_client, filing)
    assert txn is not None
    assert txn.consideration == "mixed"
    assert txn.cash_per_share == pytest.approx(27.00)
    assert txn.exchange_ratio is None
    assert txn.offer_price is None
    assert txn.confidence == pytest.approx(0.50)
    assert any("collar" in n for n in txn.notes)
    assert "Rocket Lab" in (txn.acquirer_name or "")


# --------------------------------------------------------------------------- #
# What must be refused
# --------------------------------------------------------------------------- #


def test_a_convertible_note_8k_is_not_a_merger(merger_client):
    """Item 1.01, a definitive agreement, and a "$160.00 per share" in it."""
    text = load_text(SPLUNK_CONVERTIBLE_NOTES)
    assert "conversion price of $160.00 per share" in P._normalise(text)
    filing = {
        "accession": SPLUNK_CONVERTIBLE_NOTES,
        "form": "8-K",
        "filed": date(2021, 6, 22),
    }
    assert P.extract_transaction(text, "SPLK", merger_client, filing) is None


def test_an_acquirer_side_registration_is_not_a_precedent(merger_client):
    """Zendesk's own S-4 to buy Momentive converts Momentive's shares."""
    text = P._normalise(load_text(ZENDESK_AS_ACQUIRER))
    assert "converted into the right to receive 0.225" in text
    filing = {"accession": ZENDESK_AS_ACQUIRER, "form": "S-4", "filed": date(2021, 12, 6)}
    assert P.extract_transaction(text, "ZEN", merger_client, filing) is None


def test_a_failed_parse_is_recorded_rather_than_guessed(merger_client):
    """A merger agreement whose consideration clause is unreadable.

    The document keeps its merger language and its party list and loses the
    numbers. What comes back is a transaction with no price, a low confidence
    and a note saying so, which is the difference between an engine that admits
    a gap and one that fills it.
    """
    text = load_text("0001104659-23-102594")
    broken = text.replace("$157.00 in cash", "the Merger Consideration")
    filing = {"accession": "x", "form": "8-K", "filed": date(2023, 9, 21)}
    txn = P.extract_transaction(broken, "SPLK", merger_client, filing)
    assert txn is not None
    assert txn.offer_price is None
    assert txn.consideration is None
    assert txn.confidence == pytest.approx(0.25)
    assert any("no per-share cash amount" in n for n in txn.notes)
    assert "Cisco" in (txn.acquirer_name or ""), "the parties still parse"


def test_a_buyer_absent_from_the_ticker_file_resolves_to_nothing(merger_client):
    """Google LLC is a subsidiary of Alphabet and is not a registrant.

    A fuzzy match would map it onto Alphabet and price a stock leg in the wrong
    security. The right answer is no ticker at all.
    """
    ticker, notes = P.resolve_name_to_ticker("Google LLC", merger_client)
    assert ticker is None
    assert notes == []
    assert P.resolve_name_to_ticker("Cisco Systems, Inc.", merger_client)[0] == "CSCO"
    assert P.resolve_name_to_ticker("salesforce.com, inc.", merger_client)[0] == "CRM"


def test_a_dual_class_buyer_without_a_stated_class_is_refused(merger_client):
    ticker, notes = P.resolve_name_to_ticker("Fox Corporation", merger_client)
    assert ticker is None
    assert any("more than one ticker" in n for n in notes)
    assert P.resolve_name_to_ticker("Fox Corporation", merger_client, share_class="A")[0] == "FOXA"


# --------------------------------------------------------------------------- #
# The announcement date
# --------------------------------------------------------------------------- #


def test_the_announcement_date_is_the_first_filing_not_the_signing(precedents):
    """Splunk signed on a Wednesday evening and announced on Thursday morning.

    Taking the 8-K event date would measure the premium against the close of the
    20th minus one, which is a day before the market could have known anything.
    """
    splk = by_ticker(precedents, "SPLK")
    assert splk.agreement_date == date(2023, 9, 20)
    assert splk.announced == date(2023, 9, 21)

    # Mandiant is the other direction: the agreement was signed on the 7th, the
    # 8-K was not filed until the 9th, and the press release reached EDGAR as
    # additional proxy material on the 8th.
    mndt = by_ticker(precedents, "MNDT")
    assert mndt.agreement_date == date(2022, 3, 7)
    assert mndt.announced == date(2022, 3, 8)


def test_the_announcement_date_is_never_before_the_agreement(precedents):
    for txn in precedents.transactions:
        if txn.agreement_date is not None:
            assert txn.announced >= txn.agreement_date


# --------------------------------------------------------------------------- #
# The unaffected price, both conventions
# --------------------------------------------------------------------------- #


def test_both_unaffected_conventions_are_computed_and_the_headline_is_named(precedents):
    """Silicon Laboratories closed at 136.62 on the session before the deal."""
    slab = by_ticker(precedents, "SLAB")
    assert slab.announced == date(2026, 2, 4)
    assert slab.unaffected_1d == pytest.approx(136.62)
    assert slab.unaffected_30d is not None
    assert slab.unaffected_30d != pytest.approx(slab.unaffected_1d)
    assert slab.unaffected_basis == "1-day close"
    assert slab.unaffected_price == slab.unaffected_1d
    assert slab.premium == slab.premium_1d
    assert slab.premium_1d == pytest.approx(231.00 / 136.62 - 1.0)
    assert slab.premium_30d == pytest.approx(231.00 / slab.unaffected_30d - 1.0)


def test_the_thirty_day_reference_is_the_mean_of_the_closes_in_the_window(
    merger_prices, precedents
):
    slab = by_ticker(precedents, "SLAB")
    series = merger_prices.fetch("SLAB", date(2025, 1, 1), date(2026, 9, 11))
    window = [
        c
        for d, c in zip(series.dates, series.closes)
        if date(2026, 1, 5) <= d < date(2026, 2, 4)
    ]
    assert len(window) > 15
    assert slab.unaffected_30d == pytest.approx(sum(window) / len(window))


def test_the_unaffected_price_never_sees_the_announcement_day_close(
    merger_prices, precedents
):
    """Silicon Laboratories jumped from 136.62 to 203.41 on the announcement."""
    series = merger_prices.fetch("SLAB", date(2025, 1, 1), date(2026, 9, 11))
    on_the_day = dict(zip(series.dates, series.closes))[date(2026, 2, 4)]
    assert on_the_day == pytest.approx(203.41)
    slab = by_ticker(precedents, "SLAB")
    assert slab.unaffected_1d < on_the_day
    assert slab.unaffected_30d < on_the_day


def test_a_leak_is_flagged_when_the_two_conventions_disagree(precedents):
    """Roku moved 20 percent on the Friday before the agreement was signed.

    Measured against that close the premium is 11 percent; measured against the
    month before it is 27. The gap is the market having already moved, and a
    reader who is only shown the one-day figure is being told the buyer paid
    eleven points for control of Roku.
    """
    roku = by_ticker(precedents, "ROKU")
    assert roku.premium_1d is not None and roku.premium_30d is not None
    assert roku.premium_1d < roku.premium_30d
    gap = roku.premium_gap_points
    assert gap is not None and gap < -P.LEAK_FLAG_POINTS
    assert any("leak looks like" in f for f in roku.flags)

    payo = by_ticker(precedents, "PAYO")
    assert payo.premium_gap_points < -P.LEAK_FLAG_POINTS
    assert any("leak looks like" in f for f in payo.flags)


def test_a_small_disagreement_is_not_flagged(precedents):
    """LiveRamp's two conventions differ by less than a point."""
    ramp = by_ticker(precedents, "RAMP")
    assert abs(ramp.premium_gap_points) < P.LEAK_FLAG_POINTS
    assert not any("leak looks like" in f for f in ramp.flags)


def test_no_price_series_means_no_premium_and_a_flag(precedents):
    """Nasdaq serves no history for a delisted symbol, which every target is."""
    splk = by_ticker(precedents, "SPLK")
    assert splk.unaffected_price is None
    assert splk.premium is None
    assert splk.premium_30d is None
    assert any("no price series" in f for f in splk.flags)


def test_a_series_carrying_volume_is_used_as_a_true_vwap(merger_prices):
    """The thirty-day figure weights by volume when the source publishes it."""
    series = merger_prices.fetch("SLAB", date(2025, 1, 1), date(2026, 9, 11))
    announced = date(2026, 2, 4)
    _, plain, _, plain_notes = P._unaffected(series, announced)
    assert any("not a volume weighted average" in n for n in plain_notes)

    # Weight the last session in the window far above the rest and the answer
    # has to move toward that session's close.
    series.volumes = [1.0] * len(series.closes)
    last_in_window = max(d for d in series.dates if d < announced)
    series.volumes[series.dates.index(last_in_window)] = 1_000_000.0
    _, weighted, _, vwap_notes = P._unaffected(series, announced)
    assert any("true volume weighted average" in n for n in vwap_notes)
    assert weighted == pytest.approx(136.62, abs=0.05)
    assert weighted != pytest.approx(plain)


# --------------------------------------------------------------------------- #
# The multiple arithmetic
# --------------------------------------------------------------------------- #


def test_multiples_are_the_offer_price_through_the_standard_bridge(
    merger_client, merger_assumptions, precedents
):
    """EV/Revenue is rebuilt by hand from the engine's own bridge.

    The point is that the precedent multiple is the offer price times the share
    count plus the net debt the buyer assumed, not a market capitalisation.
    """
    from techval.edgar import CompanyFacts
    from techval.ev_bridge import build_ev_bridge
    from techval.financials import build_financials

    ramp = by_ticker(precedents, "RAMP")
    raw = merger_client.company_facts("RAMP").raw
    facts = CompanyFacts(raw, "RAMP", knowledge_date=ramp.announced)
    fin = build_financials("RAMP", facts=facts)
    bridge = build_ev_bridge(fin, 38.50, merger_assumptions)

    assert ramp.equity_value == pytest.approx(bridge.equity_value)
    assert ramp.enterprise_value == pytest.approx(bridge.enterprise_value)
    assert ramp.target_revenue_ttm == pytest.approx(fin.revenue)
    assert ramp.ev_revenue == pytest.approx(bridge.enterprise_value / fin.revenue)
    assert ramp.ev_ebitda == pytest.approx(bridge.enterprise_value / fin.ebitda)
    # Net debt assumed is inside the enterprise value, so the two differ.
    assert ramp.enterprise_value != pytest.approx(ramp.equity_value)
    assert ramp.enterprise_value == pytest.approx(
        ramp.equity_value + bridge.net_debt, rel=1e-9
    )


def test_the_trailing_twelve_months_are_the_ones_filed_at_announcement(
    merger_client, precedents
):
    """A knowledge-dated fact set, not the restated figures that came later."""
    from techval.edgar import CompanyFacts
    from techval.financials import build_financials

    zen = by_ticker(precedents, "ZEN")
    assert zen.announced == date(2022, 6, 24)
    dated = build_financials(
        "ZEN",
        facts=CompanyFacts(merger_client.company_facts("ZEN").raw, "ZEN",
                           knowledge_date=zen.announced),
    )
    undated = build_financials(
        "ZEN", facts=CompanyFacts(merger_client.company_facts("ZEN").raw, "ZEN")
    )
    assert dated.as_of < undated.as_of, "the fixture carries later filings too"
    assert zen.target_revenue_ttm == pytest.approx(dated.revenue)
    assert zen.target_revenue_ttm != pytest.approx(undated.revenue)


def test_a_mixed_deal_is_priced_at_the_buyers_unaffected_close(precedents):
    """Slack marks out at the $45.86 the deal was reported at.

    0.0776 Salesforce shares at its 30 November 2020 close of $245.80, plus
    $26.79 in cash. Marking the stock leg at any later price would put the
    market's reaction to the deal inside the price the buyer agreed to pay.
    """
    work = by_ticker(precedents, "WORK")
    assert work.acquirer_price == pytest.approx(245.80)
    assert work.offer_price == pytest.approx(26.79 + 0.0776 * 245.80)
    assert work.offer_price == pytest.approx(45.86, abs=0.01)
    assert work.pct_cash == pytest.approx(26.79 / work.offer_price)
    assert work.consideration == "mixed"
    assert work.confidence == pytest.approx(0.75)


def test_negative_ebitda_is_nm_and_not_a_multiple(precedents):
    """Mandiant lost money at the EBITDA line in the year before it was sold."""
    mndt = by_ticker(precedents, "MNDT")
    assert mndt.target_ebitda_ttm is not None and mndt.target_ebitda_ttm < 0
    assert mndt.ev_ebitda is None
    assert mndt.ev_revenue is not None and mndt.ev_revenue > 1.0
    assert any("is NM" in f for f in mndt.flags)


def test_an_enormous_ebitda_multiple_is_suppressed(precedents, merger_assumptions):
    """Splunk's trailing GAAP EBITDA was 70mm against a 26bn enterprise value."""
    splk = by_ticker(precedents, "SPLK")
    assert splk.target_ebitda_ttm is not None and 0 < splk.target_ebitda_ttm < 200
    assert splk.ev_ebitda is None
    assert any(
        f"above the {merger_assumptions.comps.ev_ebitda_nm_threshold:,.0f}x" in f
        for f in splk.flags
    )


def test_financials_that_cannot_be_built_are_recorded_not_filled(precedents):
    """Silicon Laboratories has no FY2024 net income in companyfacts at all."""
    slab = by_ticker(precedents, "SLAB")
    assert slab.offer_price == pytest.approx(231.00)
    assert slab.equity_value is None
    assert slab.enterprise_value is None
    assert slab.ev_revenue is None
    assert any("could not be built" in f for f in slab.flags)
    assert any("net income" in f for f in slab.flags)


def test_no_offer_price_means_no_deal_size(precedents):
    irdm = by_ticker(precedents, "IRDM")
    assert irdm.offer_price is None
    assert irdm.equity_value is None
    assert irdm.ev_revenue is None
    assert any("no offer price could be fixed" in f for f in irdm.flags)


# --------------------------------------------------------------------------- #
# The set, its statistics and its refusals
# --------------------------------------------------------------------------- #


def test_every_deal_in_the_fixture_set_is_found(precedents):
    assert {t.target_ticker for t in precedents.transactions} == set(DEAL_TICKERS)
    assert precedents.as_of == MERGER_AS_OF


def test_transactions_are_ordered_newest_first_and_deterministically(
    merger_client, merger_assumptions, merger_prices
):
    first = P.build_precedents(
        DEAL_TICKERS, merger_client, merger_assumptions, prices=merger_prices
    )
    second = P.build_precedents(
        list(reversed(DEAL_TICKERS)),
        MergerFixtureClient(knowledge_date=MERGER_AS_OF),
        merger_assumptions,
        prices=merger_prices,
    )
    assert [t.target_ticker for t in first.transactions] == [
        t.target_ticker for t in second.transactions
    ]
    announced = [t.announced for t in first.transactions]
    assert announced == sorted(announced, reverse=True)
    assert first.to_frame().equals(second.to_frame())


def test_a_repeated_ticker_is_read_once(
    merger_client, merger_assumptions, merger_prices
):
    result = P.build_precedents(
        ["SPLK", "splk", "SPLK"], merger_client, merger_assumptions, prices=merger_prices
    )
    assert [t.target_ticker for t in result.transactions] == ["SPLK"]
    assert P.deal_events(["ZEN", "zen"], merger_client, merger_assumptions) == [
        ("ZEN", date(2022, 6, 24), True)
    ]


def test_the_notes_say_precedents_are_not_trading_comps(precedents):
    joined = " ".join(precedents.notes)
    assert "not trading comparables" in joined
    assert "control premium" in joined
    assert "25" in joined and "40 percent" in joined
    assert "by construction" in joined


def test_statistics_carry_their_n_and_refuse_below_five_deals(precedents):
    stats = precedents.stats
    assert list(stats.index) == ["n", "Min", "p25", "Median", "p75", "Max"]
    assert "EV/Revenue" in stats.columns and "EV/EBITDA" in stats.columns

    revenue_n = stats.loc["n", "EV/Revenue"]
    assert revenue_n >= P.MIN_DEALS_FOR_STATS
    assert stats.loc["Median", "EV/Revenue"] > 0

    # Only two targets carried a positive EBITDA the bridge would divide by.
    ebitda_n = stats.loc["n", "EV/EBITDA"]
    assert 0 < ebitda_n < P.MIN_DEALS_FOR_STATS
    assert pd.isna(stats.loc["Median", "EV/EBITDA"]), "refused, not computed"
    assert any("EV/EBITDA" in f and "below the 5-deal floor" in f for f in precedents.flags)


def test_no_sub_vertical_in_this_set_reaches_five_deals(precedents):
    assert precedents.sub_vertical_stats == {}
    buckets = {t.sub_vertical or "unclassified" for t in precedents.transactions}
    for bucket in buckets:
        assert any(f.startswith(f"{bucket}: ") for f in precedents.flags)
    assert any("three transactions is three transactions" in f for f in precedents.flags)


def test_a_sub_vertical_with_enough_deals_gets_statistics(
    merger_client, merger_assumptions, merger_prices
):
    """The classifier is injected, which is also how taxonomy.py would plug in."""
    result = P.build_precedents(
        DEAL_TICKERS,
        merger_client,
        merger_assumptions,
        prices=merger_prices,
        classify=lambda ticker: "infrastructure_software",
    )
    assert set(result.sub_vertical_stats) == {"infrastructure_software"}
    frame = result.sub_vertical_stats["infrastructure_software"]
    assert frame.loc["n", "EV/Revenue"] >= P.MIN_DEALS_FOR_STATS
    assert frame.loc["Median", "EV/Revenue"] == pytest.approx(
        result.stats.loc["Median", "EV/Revenue"]
    )
    assert all(t.sub_vertical == "infrastructure_software" for t in result.transactions)


def test_the_sub_vertical_floor_can_be_lowered_at_the_call_site(
    merger_client, merger_assumptions, merger_prices
):
    result = P.build_precedents(
        DEAL_TICKERS,
        merger_client,
        merger_assumptions,
        prices=merger_prices,
        min_deals_for_stats=2,
    )
    assert "application_software" in result.sub_vertical_stats
    assert result.sub_vertical_stats["application_software"].loc["n", "EV/Revenue"] >= 1


def test_sic_classification_is_used_and_flagged_as_weak(precedents):
    """Mandiant sold incident response under SIC 3577, computer peripherals."""
    assert by_ticker(precedents, "MNDT").sub_vertical == "hardware"
    assert by_ticker(precedents, "SLAB").sub_vertical == "semiconductors"
    assert by_ticker(precedents, "ROKU").sub_vertical == "media_entertainment"
    assert by_ticker(precedents, "PAYO").sub_vertical is None
    assert any("weak classifier" in f for f in precedents.flags)
    assert any("does not map to a TMT sub-vertical" in f for f in precedents.flags)


def test_the_assumptions_override_beats_every_classifier(
    merger_client, merger_prices, merger_assumptions
):
    merger_assumptions.tmt.sub_vertical = "towers_fiber"
    result = P.build_precedents(
        DEAL_TICKERS, merger_client, merger_assumptions, prices=merger_prices
    )
    assert {t.sub_vertical for t in result.transactions} == {"towers_fiber"}


def test_to_frame_gives_one_row_per_transaction(precedents):
    frame = precedents.to_frame()
    assert len(frame) == len(precedents.transactions)
    assert list(frame["Target"]) == [t.target_ticker for t in precedents.transactions]
    assert "Premium to 30-day" in frame.columns
    assert "Enterprise value" in frame.columns


def test_filter_narrows_the_set_and_recomputes_the_statistics(precedents):
    recent = precedents.filter(since=date(2026, 1, 1))
    assert {t.target_ticker for t in recent.transactions} == {
        "SLAB", "RAMP", "PAYO", "ROKU", "IRDM"
    }
    assert recent.stats.loc["n", "EV/Revenue"] < precedents.stats.loc["n", "EV/Revenue"]
    assert recent.notes == precedents.notes

    media = precedents.filter(sub_vertical="media_entertainment")
    assert [t.target_ticker for t in media.transactions] == ["ROKU"]
    # The refusals belong to the set they were computed over, not to its parent.
    assert not any(f.startswith("semiconductors: ") for f in media.flags)
    assert any(f.startswith("media_entertainment: ") for f in media.flags)
    # What was said while building the set travels with it either way.
    assert any("weak classifier" in f for f in media.flags)


def test_a_size_floor_drops_what_cannot_be_shown_to_clear_it(precedents):
    big = precedents.filter(min_size=5_000.0)
    assert {t.target_ticker for t in big.transactions} == {"SPLK", "ZEN", "ROKU", "MNDT"}
    assert all(t.equity_value >= 5_000.0 for t in big.transactions)
    assert any("not because they were small" in f for f in big.flags)


def test_the_configured_size_floor_excludes_small_deals(
    merger_client, merger_prices, merger_assumptions
):
    merger_assumptions.ml.mna.min_deal_size = 3_000.0
    result = P.build_precedents(
        DEAL_TICKERS, merger_client, merger_assumptions, prices=merger_prices
    )
    priced = {t.target_ticker for t in result.transactions if t.equity_value is not None}
    assert "PAYO" not in priced, "2.7bn of equity value is under the 3bn floor"
    assert "SPLK" in priced
    assert any("below the 3,000mm floor" in f for f in result.flags)


# --------------------------------------------------------------------------- #
# Point in time, and the propensity labels
# --------------------------------------------------------------------------- #


def test_a_knowledge_date_before_the_deal_finds_nothing(
    merger_assumptions, merger_prices
):
    """Zendesk's only earlier deal document is the S-4 in which it was the buyer."""
    early = MergerFixtureClient(knowledge_date=date(2022, 5, 1))
    merger_assumptions.as_of = "2022-05-01"
    result = P.build_precedents(["ZEN", "SPLK"], early, merger_assumptions, prices=merger_prices)
    assert result.transactions == []


def test_deal_events_are_point_in_time_labels(merger_client, merger_assumptions):
    events = P.deal_events(DEAL_TICKERS, merger_client, merger_assumptions)
    assert len(events) == len(DEAL_TICKERS)
    assert events == sorted(events, key=lambda e: (e[1], e[0])), "oldest first"

    as_map = {ticker: (when, done) for ticker, when, done in events}
    assert as_map["SPLK"] == (date(2023, 9, 21), True)
    assert as_map["MNDT"] == (date(2022, 3, 8), True)
    assert as_map["ZEN"] == (date(2022, 6, 24), True)
    assert as_map["WORK"] == (date(2020, 12, 1), True)
    # The 2026 deals had not deregistered by the as-of date.
    for pending in ("SLAB", "RAMP", "PAYO", "ROKU", "IRDM"):
        assert as_map[pending][1] is False
        assert as_map[pending][0] >= date(2026, 2, 4)


def test_deal_event_dates_match_the_precedent_announcements(precedents, merger_client,
                                                            merger_assumptions):
    events = {ticker: when for ticker, when, _ in
              P.deal_events(DEAL_TICKERS, merger_client, merger_assumptions)}
    for txn in precedents.transactions:
        assert events[txn.target_ticker] == txn.announced


def test_the_deal_is_invisible_from_the_day_before_its_label(
    merger_assumptions, merger_prices
):
    """Rerun each deal through a knowledge date one day earlier and it vanishes.

    This is the property a propensity model depends on, and it is the one that
    fails silently. If any document about the merger were readable before the
    label date, a model given that knowledge date would read it and would learn
    to predict an acquisition from the acquisition.
    """
    known = P.deal_events(
        DEAL_TICKERS, MergerFixtureClient(knowledge_date=MERGER_AS_OF), merger_assumptions
    )
    assert len(known) == len(DEAL_TICKERS)
    for ticker, announced, _completed in known:
        eve = announced - timedelta(days=1)
        blind = MergerFixtureClient(knowledge_date=eve)
        merger_assumptions.as_of = eve.isoformat()
        result = P.build_precedents(
            [ticker], blind, merger_assumptions, prices=merger_prices
        )
        assert result.transactions == [], (
            f"{ticker} was already visible on {eve}, one day before its own "
            "announcement, so a model trained on that label could see the deal"
        )
        assert P.deal_events([ticker], blind, merger_assumptions) == []


def test_the_source_document_never_predates_the_announcement(precedents):
    for txn in precedents.transactions:
        assert txn.source_accession is not None
        assert txn.announced is not None
        assert txn.agreement_date is None or txn.agreement_date <= txn.announced


def test_completion_is_evidenced_by_deregistration(precedents):
    splk = by_ticker(precedents, "SPLK")
    assert splk.status == "completed"
    assert splk.completed is True
    assert splk.closed == date(2024, 3, 18), "the Form 25 that delisted it"

    slab = by_ticker(precedents, "SLAB")
    assert slab.status == "pending"
    assert slab.completed is False
    assert slab.closed is None


def test_pending_is_not_reported_as_failed(precedents):
    """A false in deal_events covers pending as well as broken."""
    pending = [t for t in precedents.transactions if t.status == "pending"]
    assert pending
    assert all(t.completed is False for t in pending)
    assert all(t.closed is None for t in pending)
    assert {t.status for t in precedents.transactions} <= {
        "completed", "pending", "unresolved"
    }


def test_no_transaction_reports_a_premium_without_an_unaffected_price(precedents):
    for txn in precedents.transactions:
        if txn.premium is not None:
            assert txn.unaffected_price is not None
            assert txn.offer_price is not None
        if txn.unaffected_price is None:
            assert txn.premium is None and txn.premium_30d is None


def test_no_transaction_reports_a_multiple_without_its_denominator(precedents):
    for txn in precedents.transactions:
        if txn.ev_revenue is not None:
            assert txn.enterprise_value is not None
            assert txn.target_revenue_ttm is not None and txn.target_revenue_ttm > 0
        if txn.ev_ebitda is not None:
            assert txn.target_ebitda_ttm is not None and txn.target_ebitda_ttm > 0


def test_an_empty_set_still_has_a_shaped_frame_and_its_notes(
    merger_client, merger_assumptions
):
    result = P.build_precedents([], merger_client, merger_assumptions)
    assert result.transactions == []
    assert result.to_frame().empty
    assert "Target" in result.to_frame().columns
    assert result.stats.loc["n", "EV/Revenue"] == 0
    assert any("not trading comparables" in n for n in result.notes)
