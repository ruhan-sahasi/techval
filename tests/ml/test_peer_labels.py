"""Peer-label extraction, tested against four real proxies held offline.

The fixtures are markup-stripped excerpts of filings that exist, cut to the
region around each compensation peer group, with the accession, the filing date
and the retrieval date recorded inside the file. Nothing inside a window is
edited, so an assertion here is an assertion about what a company actually told
the SEC. The name index is the Commission's own ``company_tickers.json`` payload
pruned to the registrants these tests name, in the original shape.

Every case that matters is a case where a plausible parser is wrong rather than
empty: the suffix word that bridges two names, the parenthesised ticker that is
also a registrant name, the list that runs into prose with no punctuation at the
seam, and the fragment that looks like a small peer group.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.edgar import SEC_TICKERS_URL
from techval.errors import DataSourceError
from techval.ml.peer_labels import (
    MAX_PLAUSIBLE_PEERS,
    MIN_PLAUSIBLE_PEERS,
    NameIndex,
    PeerGroup,
    build_name_index,
    cache_dir,
    collect_peer_groups,
    extract_peer_group,
    groups_to_frame,
    normalise_name,
    pairs_to_frame,
    peer_pairs,
    segment_names,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TICKER_FIXTURE = FIXTURES / "sec_tickers_tmt.json"
PROXY_FIXTURE = FIXTURES / "proxy_peer_groups.json"

# The universe the fixtures cover. Passing it as the collection universe is what
# lets a collision be settled by preference rather than by a guess.
UNIVERSE = ("DDOG", "ZS", "CRWD", "MDB")


class FixtureProxyClient:
    """Stands in for EdgarClient over the committed proxy excerpts.

    Any call that would reach the network raises, so a test that drifts into
    fetching fails loudly instead of passing slowly.
    """

    def __init__(self) -> None:
        self._tickers = json.loads(TICKER_FIXTURE.read_text())["data"]
        self._proxies = json.loads(PROXY_FIXTURE.read_text())["filings"]
        self.text_calls = 0

    def _get_json(self, url: str) -> dict:
        if url != SEC_TICKERS_URL:
            raise AssertionError(f"unexpected network read of {url}")
        return self._tickers

    def ticker_to_cik(self, ticker: str) -> int:
        for row in self._tickers.values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"])
        raise KeyError(ticker)

    def submissions(self, ticker: str) -> dict:
        return {"formerNames": []}

    def filings(self, ticker, forms=("10-K",), since=None, limit=20) -> list[dict]:
        row = self._proxies.get(ticker.upper())
        if row is None or "DEF 14A" not in forms:
            return []
        return [
            {
                "accession": row["accession"],
                "filed": date.fromisoformat(row["filed"]),
                "form": row["form"],
                "document": row["document"],
                "period": row["period"],
            }
        ][:limit]

    def filing_text(self, ticker: str, filing: dict) -> str:
        self.text_calls += 1
        return self._proxies[ticker.upper()]["text"]


@pytest.fixture
def client() -> FixtureProxyClient:
    return FixtureProxyClient()


@pytest.fixture
def index(client) -> NameIndex:
    return build_name_index(client)


def _proxy(client, ticker: str):
    filing = client.filings(ticker, ("DEF 14A",))[0]
    return client.filing_text(ticker, filing), filing


# --------------------------------------------------------------------------- #
# Normalisation and the index
# --------------------------------------------------------------------------- #


def test_normalise_keeps_both_the_written_and_the_stripped_form():
    """Both forms are needed, and the pair is what closes the bridging hole."""
    assert normalise_name("Veeva Systems") == ("veevasystems", "veeva")
    assert normalise_name("VEEVA SYSTEMS INC") == ("veevasystemsinc", "veeva")
    assert normalise_name("The Trade Desk") == ("thetradedesk", "tradedesk")
    assert normalise_name("Trade Desk, Inc.") == ("tradedeskinc", "tradedesk")
    assert normalise_name("Palo Alto Networks") == ("paloaltonetworks", "paloalto")


def test_normalise_strips_a_share_class_only_as_a_pair():
    """A bare trailing letter is never a suffix.

    Stripping one would let a span reach a character into the next company and
    still match, which is how "BILL Holdings" becomes "ILL Holdings".
    """
    assert normalise_name("Alphabet Inc. Class A")[1] == "alphabet"
    # The B belongs to BILL, not to the company before it.
    assert normalise_name("SnowflakeB")[1] == "snowflakeb"


def test_normalise_reads_a_dotted_acronym_as_one_token():
    assert normalise_name("Elastic N.V.")[1] == "elastic"


def test_index_maps_both_forms_and_hides_its_collisions(index):
    assert index["veevasystemsinc"] == "VEEV"
    assert index["veeva"] == "VEEV"
    assert index["tradedesk"] == "TTD"
    # Graham Holdings and Graham Corp are two CIKs on one normalised name, so
    # the plain mapping must not offer either of them.
    assert "graham" not in index
    assert index.ambiguous["graham"] == ("GHC", "GHM")


def test_index_treats_share_classes_of_one_cik_as_one_company(index):
    """GOOG and GOOGL are not an ambiguity: the CIK already settled the company.

    The symbol returned is GOOGL rather than GOOG because the index now takes the
    Commission's own listing order rather than the shorter symbol. Which of
    Alphabet's two classes carries the label was always arbitrary and this test
    never cared; what the ordering rule fixes is the case below, where the two
    symbols of one CIK are not share classes at all.
    """
    assert index["alphabet"] == "GOOGL"
    assert "alphabet" not in index.ambiguous
    assert index.cik_of["GOOG"] == index.cik_of["GOOGL"]


def test_the_state_of_incorporation_marker_is_not_part_of_the_name(client):
    """The SEC's ticker file appends "/DE" to a title and no proxy ever writes it.

    Left in, the marker tokenises into a trailing "de", the suffix stripper
    stops on it before it reaches "inc", and the core form of "APPLIED MATERIALS
    INC /DE" is "appliedmaterialsincde". The proxy's "Applied Materials" then
    matches nothing, and the span lands in unresolved rather than becoming a
    label. Measured over the seed universe this accounted for 121 of the 1,205
    spans that could not be resolved, Applied Materials at 29 and Qualcomm at
    24 of them.

    It cannot be fixed by adding "de" to the suffix list: those are ordinary
    word fragments and stripping them would eat the tail of a real name. The
    marker is removed by its slash, at the end of the string, and nowhere else.
    """
    assert normalise_name("APPLIED MATERIALS INC /DE") == (
        "appliedmaterialsinc",
        "appliedmaterials",
    )
    assert normalise_name("QUALCOMM INC/DE") == ("qualcomminc", "qualcomm")
    assert normalise_name("CHARTER COMMUNICATIONS, INC. /MO/") == (
        "chartercommunicationsinc",
        "chartercommunications",
    )
    # A name that merely ends in two letters keeps them: there is no slash.
    assert normalise_name("Elastic N.V.") == normalise_name("Elastic NV")
    assert normalise_name("F5, Inc.") == ("f5inc", "f5")

    class WithMarker(FixtureProxyClient):
        def _get_json(self, url):
            data = dict(super()._get_json(url))
            data["910"] = {
                "cik_str": 6951,
                "ticker": "AMAT",
                "title": "APPLIED MATERIALS INC /DE",
            }
            return data

    index = build_name_index(WithMarker())
    assert index["appliedmaterials"] == "AMAT"
    assert segment_names("Applied Materials, Inc.Netflix, Inc.", index)[0] == [
        "AMAT",
        "NFLX",
    ]


def test_a_registrants_symbols_are_ranked_by_the_commissions_own_order(client):
    """Two symbols on one CIK are not always two share classes, and the short one loses.

    Comcast files its common stock as CMCSA and an exchangeable debenture as CCZ,
    both under CIK 1166691 and both titled COMCAST CORP in the Commission's
    ticker file. The rule this replaced took the shorter symbol, so every proxy
    that named Comcast produced a label pointing at a debt security rather than
    at the company, and because no equity universe contains CCZ the pair was then
    dropped as a peer outside the universe. One of the largest filers in the
    sector disappeared from the training set and nothing said so.

    Prudential lost PRU to the preferred PFH the same way, and DTE Energy lost
    DTE to the debenture DTB. Across the live file the shortest-symbol rule
    disagrees with the Commission's ordering for 224 of the 1,441 registrants
    carrying more than one symbol.

    The rows below are the real ones, in the real order, with CMCSA listed first
    exactly as the Commission lists it.
    """

    class TwoSymbols(FixtureProxyClient):
        def _get_json(self, url):
            data = dict(super()._get_json(url))
            data["900"] = {"cik_str": 1166691, "ticker": "CMCSA", "title": "COMCAST CORP"}
            data["901"] = {"cik_str": 1166691, "ticker": "CCZ", "title": "COMCAST CORP"}
            return data

    index = build_name_index(TwoSymbols())
    assert index.cik_of["CMCSA"] == index.cik_of["CCZ"]
    assert "comcast" not in index.ambiguous
    assert index["comcast"] == "CMCSA"
    assert segment_names("ComcastNetflix, Inc.", index)[0] == ["CMCSA", "NFLX"]


def test_index_survives_a_malformed_row_without_losing_the_rest(client):
    class Bent(FixtureProxyClient):
        def _get_json(self, url):
            rows = dict(super()._get_json(url))
            rows["bad"] = {"ticker": "X"}  # no title, no cik
            return rows

    index = build_name_index(Bent())
    assert index["datadog"] == "DDOG"
    assert any("malformed" in n for n in index.notes)


def test_index_refuses_an_empty_ticker_file(client):
    class Empty(FixtureProxyClient):
        def _get_json(self, url):
            return {}

    with pytest.raises(DataSourceError):
        build_name_index(Empty())


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def test_the_suffix_word_never_bridges_two_names(index):
    """The bug this parser exists to avoid.

    Lowercasing the blob and stripping suffix words globally leaves "systems"
    unconsumed between Veeva and CrowdStrike, and "stem" inside it is the
    ticker STEM. Requiring a match to start where a name can start, and
    preferring the longest match, removes both halves of the trap.
    """
    tickers, leftovers = segment_names("Veeva SystemsCrowdStrike HoldingsOkta", index)
    assert tickers == ["VEEV", "CRWD", "OKTA"]
    assert leftovers == []
    assert "STEM" not in tickers
    # STEM really is in the index, so the absence above is the rule working
    # rather than the trap being unreachable.
    assert index["stem"] == "STEM"


def test_a_suffix_word_that_opens_the_next_name_is_not_swallowed(index):
    """"NV" and "SA" begin NVIDIA and SAP SE and are also corporate suffixes.

    The bridging trap in its second and worse form. In the first form a suffix
    word is left stranded and the match simply fails, which is visible. Here the
    bridge resolves. "Netflix, Inc.NV" normalises to "netflix", because the core
    form strips the trailing "nv", so it matches NFLX across fifteen characters
    where the correct reading matches thirteen. Longest match prefers it and
    eats the first two letters of NVIDIA, and nothing looks wrong afterwards:
    every ticker returned is real, and only the count is quietly short by two.

    The boundary rules cannot catch this one, because "NV" is capitalised and a
    name is allowed to end there. What settles it is that giving the characters
    back opens a seam where a registrant actually starts.
    """
    tickers, leftovers = segment_names(
        "Netflix, Inc.NVIDIA CorporationOracle CorporationSalesforce, Inc.SAP SE",
        index,
    )
    assert tickers == ["NFLX", "NVDA", "ORCL", "CRM", "SAP"]
    assert leftovers == []


def test_a_parenthesised_ticker_is_read_as_a_symbol_not_as_a_name(index):
    """"Atlassian (TEAM)" must not also yield Team, Inc.

    TEAM is Atlassian's symbol and it is also the registrant name of Team,
    Inc., an industrial services company. Reading the annotation as a name
    produces a confident wrong label, which is the worst output this module has.
    """
    blob = (
        "Atlassian (TEAM)MongoDB (MDB)Snowflake (SNOW)Datadog (DDOG)"
        "Okta (OKTA)Zscaler (ZS)"
    )
    tickers, leftovers = segment_names(blob, index)
    assert tickers == ["TEAM", "MDB", "SNOW", "DDOG", "OKTA", "ZS"]
    assert "TISI" not in tickers
    assert leftovers == []
    # And the registrant that the trap points at is genuinely in the index.
    assert index["team"] == "TISI"


def test_a_symbol_that_no_longer_exists_is_left_unresolved(index):
    """A delisted peer is a missing label, never a label pointing at nothing."""
    blob = (
        "Datadog (DDOG)Splunk (SPLK)MongoDB (MDB)Snowflake (SNOW)Okta (OKTA)"
    )
    tickers, leftovers = segment_names(blob, index)
    assert tickers == ["DDOG", "MDB", "SNOW", "OKTA"]
    assert len(leftovers) == 1
    assert "SPLK" in leftovers[0] and "absent" in leftovers[0]


def test_a_lowercase_initial_brand_is_not_split_down_the_middle(index):
    """"AdobeeBay" is two companies, and neither is "Adobee"."""
    tickers, _ = segment_names("AdobeeBayOracleSnowflake", index)
    assert tickers == ["ADBE", "EBAY", "ORCL", "SNOW"]


def test_a_renamed_company_is_reached_through_its_current_name(index):
    """The proxy says "Zoom Video Communications"; the ticker file says "Zoom
    Communications, Inc.". The registrant name sits inside the disclosed one,
    and that containment is allowed in one direction only."""
    tickers, leftovers = segment_names("Zoom Video CommunicationsDynatrace", index)
    assert tickers == ["ZM", "DT"]
    assert leftovers == []


def test_containment_is_one_directional(index):
    """A shortened disclosed name may not reach a longer registrant name.

    Veeva's proxy writes "Zoom" where the ticker file says "Zoom
    Communications, Inc.". The rename rule runs the other way round only, so
    this stays unresolved. Allowing it would let any first word pull in a
    registrant nobody named, and one such label is worse than the missing one.
    """
    tickers, leftovers = segment_names("Veeva SystemsZoomOkta", index)
    assert tickers == ["VEEV", "OKTA"]
    assert leftovers == ["Zoom"]


def test_an_ambiguous_name_goes_to_unresolved_with_both_candidates(index):
    """Two CIKs on one name is not a coin toss.

    Graham Holdings is a media company and Graham Corp makes vacuum equipment.
    Nothing in the span says which, so both are reported and neither is used.
    """
    tickers, leftovers = segment_names("DatadogGrahamSnowflake", index)
    assert tickers == ["DDOG", "SNOW"]
    assert len(leftovers) == 1
    assert "GHC" in leftovers[0] and "GHM" in leftovers[0]
    assert "ambiguous" in leftovers[0]


def test_a_collision_is_settled_by_the_company_already_in_play(index):
    """Preference resolves what the span alone cannot, and only that far."""
    tickers, leftovers = segment_names(
        "DatadogGrahamSnowflake", index, prefer_tickers=("GHC",)
    )
    assert tickers == ["DDOG", "GHC", "SNOW"]
    assert leftovers == []


def test_a_short_span_inside_a_run_is_refused(index):
    """Three characters inside a run of joined names is noise, not a company."""
    tickers, _ = segment_names("SnowflakeTheDatadog", index)
    assert tickers == ["SNOW", "DDOG"]


def test_segmentation_is_order_preserving_and_repeatable(index):
    blob = "AtlassianHubSpotThe Trade DeskCloudflare"
    first = segment_names(blob, index)
    second = segment_names(blob, index)
    assert first == second == (["TEAM", "HUBS", "TTD", "NET"], [])


# --------------------------------------------------------------------------- #
# Locating the list inside a real proxy
# --------------------------------------------------------------------------- #


def test_datadog_2025_yields_exactly_the_seventeen_disclosed_peers(client, index):
    """The whole pipeline against the concatenated form, end to end.

    Datadog's 2026 proxy names seventeen companies as its 2025 compensation
    peer group, joined into one run with no separators and running straight into
    the paragraph that follows. All seventeen resolve and nothing else appears.
    """
    text, filing = _proxy(client, "DDOG")
    group = extract_peer_group(
        text, index, "DDOG", accession=filing["accession"], filed=filing["filed"]
    )
    assert group is not None
    assert group.peers == [
        "TEAM",  # Atlassian
        "HUBS",  # HubSpot
        "TTD",  # The Trade Desk
        "NET",  # Cloudflare
        "MDB",  # MongoDB
        "VEEV",  # Veeva Systems
        "CRWD",  # CrowdStrike Holdings
        "OKTA",  # Okta
        "WDAY",  # Workday
        "DOCU",  # DocuSign
        "PLTR",  # Palantir Technologies
        "ZM",  # Zoom Video Communications
        "DT",  # Dynatrace
        "PANW",  # Palo Alto Networks
        "ZS",  # Zscaler
        "FTNT",  # Fortinet
        "SNOW",  # Snowflake
    ]
    assert len(group.peers) == 17
    assert group.unresolved == []
    assert group.confidence == 1.0
    assert group.usable


def test_datadog_stops_the_list_where_the_prose_starts_again(client, index):
    """The table runs into "The compensation committee reviews" with no seam.

    Snowflake is the last peer and "The" is the first word of the next
    sentence, so a parser that does not detect the end either loses Snowflake or
    feeds a paragraph of prose into the segmenter.
    """
    text, _ = _proxy(client, "DDOG")
    group = extract_peer_group(text, index, "DDOG")
    assert group is not None
    assert group.peers[-1] == "SNOW"
    assert "NOW" not in group.peers  # ServiceNow is not in this table


def test_datadog_keeps_the_stated_size_bands(client, index):
    """What the label means travels with the label."""
    text, _ = _proxy(client, "DDOG")
    group = extract_peer_group(text, index, "DDOG")
    assert group is not None
    assert group.selection_criteria is not None
    assert "0.5x to 2.5x" in group.selection_criteria
    assert "0.3x to 3.0x" in group.selection_criteria


def test_datadog_is_stamped_with_the_year_the_group_applied_to(client, index):
    """Not the filing year. The 2026 proxy discloses the 2025 group."""
    text, filing = _proxy(client, "DDOG")
    group = extract_peer_group(text, index, "DDOG", filed=filing["filed"])
    assert group is not None
    assert filing["filed"].year == 2026
    assert group.fiscal_year == 2025
    assert any("fiscal year 2025" in n for n in group.notes)


def test_zscaler_segments_twenty_one_names_and_says_which_two_are_gone(client, index):
    """Zscaler's fiscal 2025 group, in the "consisted of the following" form.

    Twenty-one names are segmented. Nineteen resolve; ANSYS and Bill.com
    Holdings do not, because one was acquired and the other renamed, and the
    ticker file has no memory of either. Both are named in ``unresolved`` rather
    than dropped, so the reader can see the parse was complete even though the
    resolution was not.
    """
    text, filing = _proxy(client, "ZS")
    group = extract_peer_group(
        text, index, "ZS", accession=filing["accession"], filed=filing["filed"]
    )
    assert group is not None
    assert group.n_spans == 21
    assert group.peers == [
        "DT",  # Dynatrace
        "PAYC",  # Paycom Software
        "ANET",  # Arista Networks
        "FTNT",  # Fortinet
        "SNOW",  # Snowflake
        "HUBS",  # HubSpot
        "TTD",  # The Trade Desk
        "NET",  # Cloudflare
        "MDB",  # MongoDB
        "TWLO",  # Twilio
        "CRWD",  # CrowdStrike Holdings
        "OKTA",  # Okta
        "U",  # Unity Software
        "DDOG",  # Datadog
        "PLTR",  # Palantir Technologies
        "VEEV",  # Veeva Systems
        "DOCU",  # DocuSign
        "PANW",  # Palo Alto Networks
        "GTM",  # Zoominfo Technologies
    ]
    assert group.unresolved == ["ANSYS", "Bill.com Holdings"]
    assert group.confidence == pytest.approx(19 / 21, abs=1e-4)
    assert group.fiscal_year == 2025
    assert group.usable


def test_zscaler_keeps_the_market_capitalisation_band_verbatim(client, index):
    text, _ = _proxy(client, "ZS")
    group = extract_peer_group(text, index, "ZS")
    assert group is not None
    assert group.selection_criteria is not None
    assert "0.33x to 3.0x" in group.selection_criteria
    assert "30-day average market capitalization" in group.selection_criteria


def test_crowdstrike_reads_the_one_name_per_line_form(client, index):
    """The third rendered shape: the renderer kept the table's row breaks."""
    text, filing = _proxy(client, "CRWD")
    group = extract_peer_group(
        text, index, "CRWD", accession=filing["accession"], filed=filing["filed"]
    )
    assert group is not None
    assert group.fiscal_year == 2026
    assert group.unresolved == []
    assert set(group.peers) >= {"TEAM", "MDB", "SNAP", "ANET", "NOW", "SHOP", "XYZ"}
    assert group.confidence == 1.0
    assert "CRWD" not in group.peers


def test_mongodb_takes_the_ticker_the_filer_printed(client, index):
    """MongoDB prints "DocuSign [DOCU]", so the symbol is the filer's own answer."""
    text, filing = _proxy(client, "MDB")
    group = extract_peer_group(
        text, index, "MDB", accession=filing["accession"], filed=filing["filed"]
    )
    assert group is not None
    assert "annotated-ticker" in group.method
    assert group.fiscal_year == 2026
    assert group.peers == [
        "DOCU", "IOT", "NET", "DT", "SNOW", "ESTC", "TTD",
        "CRWD", "HUBS", "U", "DDOG", "OKTA", "ZS",
    ]
    # ANSYS and Confluent were acquired, so their symbols no longer resolve.
    assert len(group.unresolved) == 2
    assert all("absent from the SEC ticker file" in u for u in group.unresolved)


def test_the_filer_is_never_its_own_peer(client, index):
    """Several filers print themselves in the table to show where they sit."""
    text, _ = _proxy(client, "CRWD")
    group = extract_peer_group(text, index, "CRWD")
    assert group is not None
    assert "CRWD" not in group.peers
    assert len(group.peers) == len(set(group.peers))


def test_a_proxy_with_no_peer_group_returns_none(index):
    text = (
        "The compensation committee met four times during the year. It reviewed "
        "the performance of each named executive officer and set base salaries "
        "accordingly. No peer group was used."
    )
    assert extract_peer_group(text, index, "DDOG") is None


def test_a_table_of_dollar_amounts_is_not_a_peer_group(index):
    """The pay-versus-performance disclosure also ends in "Peer Group ...:"."""
    text = (
        "Value of Initial Fixed $100 Investment Based On: Total Shareholder "
        "Return Peer Group Total Shareholder Return Net Income Company-Selected "
        "Measure: Revenue 2025$29,753 $47,444,526 $18,480,646 $24,954,394 $220 "
        "$281 ($41,478,000)$2,673,115,000 2024$25,162 $9,529,728 $14,418,278"
    )
    assert extract_peer_group(text, index, "ZS") is None


def test_adobe_keeps_nvidia_and_sap_through_the_bridge(client, index):
    """The bridging trap end to end, in the filing it was found in.

    Adobe's table joins "Netflix, Inc." to "NVIDIA Corporation" and
    "Salesforce, Inc." to "SAP SE", which is both bridges in one list. All
    twenty peers resolve and neither company is lost at the seam.
    """
    text, filing = _proxy(client, "ADBE")
    group = extract_peer_group(
        text, index, "ADBE", accession=filing["accession"], filed=filing["filed"]
    )
    assert group is not None
    assert "NVDA" in group.peers
    assert "SAP" in group.peers
    assert len(group.peers) == 20
    assert group.unresolved == []
    assert group.confidence == 1.0
    assert "ADBE" not in group.peers


def test_a_bare_year_heading_with_no_colon_anchors_the_list(client, index):
    """Adobe introduces its table with a heading carrying no punctuation at all.

    "Peer Group for Fiscal Year 2025" runs straight into "Alphabet Inc." with no
    colon, so an anchor set that insists on one finds nothing in a proxy that
    discloses its group perfectly plainly. The year in the heading is also the
    year the group applied to, which is the stamp the label has to carry.
    """
    text, filing = _proxy(client, "ADBE")
    group = extract_peer_group(
        text, index, "ADBE", accession=filing["accession"], filed=filing["filed"]
    )
    assert group is not None
    assert group.fiscal_year == 2025
    assert group.method.endswith("(anchor: peer group heading for a year)")
    # Alphabet, the first name after the heading. GOOGL rather than GOOG since
    # the index began honouring the Commission's listing order; see
    # test_a_registrants_symbols_are_ranked_by_the_commissions_own_order.
    assert group.peers[0] == "GOOGL"


def test_a_page_break_inside_the_table_does_not_hide_the_list(client, index):
    """Okta's renderer cuts the table at a page boundary.

    The running header lands between the introducing colon and the first peer:
    the page number, then "2026 Proxy Statement", then the registrant's own name
    and the section heading, and only then "Cloudflare". That breaks the parse
    twice over. The anchor's lookahead sees a digit where it expects a capital
    and never fires, and if it did fire, the filer's own name sitting inside the
    furniture would be read as the first peer.
    """
    text, filing = _proxy(client, "OKTA")
    assert "2026 Proxy Statement" in text  # the furniture really is in the fixture
    group = extract_peer_group(
        text, index, "OKTA", accession=filing["accession"], filed=filing["filed"]
    )
    assert group is not None
    assert len(group.peers) == 17
    assert group.peers[0] == "NET"  # Cloudflare, the first name after the break
    assert "OKTA" not in group.peers
    assert group.unresolved == []
    assert group.confidence == 1.0


# --------------------------------------------------------------------------- #
# A failed parse is flagged, not passed off as a peer group
# --------------------------------------------------------------------------- #


def test_a_three_name_fragment_is_flagged_rather_than_believed(index):
    """Three names is a parse that found a fragment, not a small peer group."""
    text = (
        "The compensation committee approved the following peer group for 2025: "
        "AtlassianHubSpotCloudflare. Each of these companies was reviewed."
    )
    group = extract_peer_group(text, index, "DDOG")
    assert group is not None
    assert group.peers == ["TEAM", "HUBS", "NET"]
    assert len(group.peers) < MIN_PLAUSIBLE_PEERS
    assert not group.count_plausible
    assert group.confidence == 0.0
    assert not group.usable
    assert any("outside the plausible" in n for n in group.notes)


def test_an_overrun_parse_is_flagged_the_same_way(index):
    """Too many names means the span ran past the end of the table."""
    names = [
        "Atlassian", "HubSpot", "Cloudflare", "MongoDB", "Okta", "Workday",
        "DocuSign", "Snowflake", "Fortinet", "Zscaler", "Dynatrace", "Twilio",
        "Adobe", "Oracle", "Netflix", "Salesforce", "Intuit", "Uber", "Airbnb",
        "Pinterest", "Roblox", "ServiceNow", "Shopify", "Snap", "Visa",
        "PayPal", "Arista Networks", "Paycom Software", "Unity Software",
        "Samsara", "Elastic", "AppLovin",
    ]
    text = "The 2025 compensation peer group was as follows:" + "".join(names)
    group = extract_peer_group(text, index, "DDOG")
    assert group is not None
    assert len(group.peers) > MAX_PLAUSIBLE_PEERS
    assert group.confidence == 0.0
    assert not group.usable


def test_a_flagged_group_contributes_no_training_pairs(index):
    text = (
        "The compensation committee approved the following peer group for 2025: "
        "AtlassianHubSpotCloudflare. Each of these companies was reviewed."
    )
    group = extract_peer_group(text, index, "DDOG")
    assert group is not None and not group.usable
    assert peer_pairs([group]) == []


def test_collect_leaves_a_failed_parse_out_unless_asked_for_it(client, index):
    class OneBadProxy(FixtureProxyClient):
        def filing_text(self, ticker, filing):
            return (
                "The compensation peer group for 2025 was as follows: "
                "AtlassianHubSpotCloudflare. The committee then met."
            )

    bad = OneBadProxy()
    assert collect_peer_groups(["DDOG"], bad, index, use_cache=False) == []
    kept = collect_peer_groups(
        ["DDOG"], bad, index, use_cache=False, keep_failed=True
    )
    assert len(kept) == 1
    assert not kept[0].usable
    assert any("outside the plausible" in n for n in kept[0].notes)


# --------------------------------------------------------------------------- #
# Pairs
# --------------------------------------------------------------------------- #


def _group(ticker: str, peers: list[str], year: int) -> PeerGroup:
    return PeerGroup(
        ticker=ticker,
        cik=None,
        accession=f"acc-{ticker}-{year}",
        filed=date(year + 1, 4, 1),
        fiscal_year=year,
        peers=peers,
        method="test",
        confidence=1.0,
    )


def test_pairs_are_symmetric_and_carry_the_year():
    peers = [f"P{i}" for i in range(MIN_PLAUSIBLE_PEERS)]
    pairs = peer_pairs([_group("DDOG", peers, 2025)])
    assert ("DDOG", "P0", 2025) in pairs
    assert ("P0", "DDOG", 2025) in pairs
    assert len(pairs) == 2 * MIN_PLAUSIBLE_PEERS
    assert all(len(p) == 3 and isinstance(p[2], int) for p in pairs)


def test_the_same_relationship_in_two_years_is_two_labels():
    """Training on the union across years would leak the future into the past."""
    peers = [f"P{i}" for i in range(MIN_PLAUSIBLE_PEERS)]
    pairs = peer_pairs([_group("DDOG", peers, 2021), _group("DDOG", peers, 2025)])
    assert ("DDOG", "P0", 2021) in pairs
    assert ("DDOG", "P0", 2025) in pairs
    assert {p[2] for p in pairs} == {2021, 2025}


def test_co_membership_is_off_by_default_because_nobody_asserted_it():
    peers = [f"P{i}" for i in range(MIN_PLAUSIBLE_PEERS)]
    plain = peer_pairs([_group("DDOG", peers, 2025)])
    derived = peer_pairs([_group("DDOG", peers, 2025)], include_co_members=True)
    assert ("P0", "P1", 2025) not in plain
    assert ("P0", "P1", 2025) in derived
    assert ("P1", "P0", 2025) in derived


def test_pairs_are_sorted_and_de_duplicated():
    peers = [f"P{i}" for i in range(MIN_PLAUSIBLE_PEERS)]
    groups = [_group("DDOG", peers, 2025), _group("DDOG", peers, 2025)]
    pairs = peer_pairs(groups)
    assert pairs == sorted(set(pairs))


def test_pairs_to_frame_is_ready_for_a_walk_forward_split():
    peers = [f"P{i}" for i in range(MIN_PLAUSIBLE_PEERS)]
    frame = pairs_to_frame(peer_pairs([_group("DDOG", peers, 2025)]))
    assert list(frame.columns) == ["a", "b", "fiscal_year"]
    assert len(frame) == 2 * MIN_PLAUSIBLE_PEERS


# --------------------------------------------------------------------------- #
# Collection, caching and determinism
# --------------------------------------------------------------------------- #


def test_collect_reads_every_proxy_in_the_universe(client, index, tmp_path):
    a = Assumptions()
    a.ml.cache_dir = str(tmp_path)
    groups = collect_peer_groups(UNIVERSE, client, index, assumptions=a)
    assert [g.ticker for g in groups] == ["CRWD", "DDOG", "MDB", "ZS"]
    assert all(g.usable for g in groups)
    assert all(g.accession for g in groups)
    pairs = peer_pairs(groups)
    # Every filer asserted the relationship, so every pair has its mirror.
    assert all((b, a_, y) in set(pairs) for a_, b, y in pairs)


def test_collect_filters_to_the_requested_years(client, index, tmp_path):
    a = Assumptions()
    a.ml.cache_dir = str(tmp_path)
    groups = collect_peer_groups(UNIVERSE, client, index, years=[2026], assumptions=a)
    assert {g.fiscal_year for g in groups} == {2026}
    assert {g.ticker for g in groups} == {"CRWD", "MDB"}


def test_collect_builds_its_own_index_when_none_is_given(client, tmp_path):
    a = Assumptions()
    a.ml.cache_dir = str(tmp_path)
    groups = collect_peer_groups(["DDOG"], client, assumptions=a)
    assert len(groups) == 1
    assert len(groups[0].peers) == 17


def test_the_extraction_cache_is_keyed_by_accession(client, index, tmp_path):
    """A universe build reads a megabyte of HTML per proxy, so it happens once."""
    a = Assumptions()
    a.ml.cache_dir = str(tmp_path)
    assert cache_dir(a) == tmp_path / "peer_groups"

    first = collect_peer_groups(["DDOG"], client, index, assumptions=a)
    reads = client.text_calls
    second = collect_peer_groups(["DDOG"], client, index, assumptions=a)

    assert client.text_calls == reads, "the second build re-read the filing"
    assert first[0].peers == second[0].peers
    assert first[0].fiscal_year == second[0].fiscal_year
    assert first[0].selection_criteria == second[0].selection_criteria
    assert any("extraction cache" in n for n in second[0].notes)
    cached = list((tmp_path / "peer_groups").glob("*.json"))
    assert [p.stem for p in cached] == [first[0].accession]


def test_a_cache_written_by_an_older_parser_is_re_read(client, index, tmp_path):
    a = Assumptions()
    a.ml.cache_dir = str(tmp_path)
    collect_peer_groups(["DDOG"], client, index, assumptions=a)
    path = next((tmp_path / "peer_groups").glob("*.json"))
    payload = json.loads(path.read_text())
    payload["parser_version"] = -1
    payload["peers"] = ["WRONG"]
    path.write_text(json.dumps(payload))

    reads = client.text_calls
    again = collect_peer_groups(["DDOG"], client, index, assumptions=a)
    assert client.text_calls == reads + 1
    assert again[0].peers != ["WRONG"]


def test_two_runs_agree_exactly(client, index, tmp_path):
    """Determinism is not optional beside a valuation."""
    a = Assumptions()
    a.ml.cache_dir = str(tmp_path / "one")
    b = Assumptions()
    b.ml.cache_dir = str(tmp_path / "two")
    first = collect_peer_groups(UNIVERSE, FixtureProxyClient(), index, assumptions=a)
    second = collect_peer_groups(UNIVERSE, FixtureProxyClient(), index, assumptions=b)
    assert [g.rows() for g in first] == [g.rows() for g in second]
    assert [g.peers for g in first] == [g.peers for g in second]
    assert peer_pairs(first) == peer_pairs(second)


def test_collect_raises_by_default_and_records_the_reason_when_told_to_skip(index):
    class Broken(FixtureProxyClient):
        def filing_text(self, ticker, filing):
            raise DataSourceError("the archive refused the document")

    with pytest.raises(DataSourceError):
        collect_peer_groups(["DDOG"], Broken(), index, use_cache=False)

    kept = collect_peer_groups(
        ["DDOG"], Broken(), index, use_cache=False, skip_errors=True, keep_failed=True
    )
    assert len(kept) == 1
    assert kept[0].method == "unreadable"
    assert kept[0].confidence == 0.0
    assert any("archive refused" in n for n in kept[0].notes)


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def test_rows_and_frames_expose_what_a_reader_needs(client, index):
    text, filing = _proxy(client, "DDOG")
    group = extract_peer_group(
        text, index, "DDOG", accession=filing["accession"], filed=filing["filed"]
    )
    assert group is not None
    labels = [label for label, _ in group.rows()]
    assert "Confidence" in labels and "Usable as a label" in labels

    frame = group.to_frame()
    assert len(frame) == 17
    assert set(frame.columns) == {
        "filer",
        "peer",
        "fiscal_year",
        "accession",
        "confidence",
    }
    assert (frame["filer"] == "DDOG").all()
    assert groups_to_frame([group]).equals(frame)


def test_groups_to_frame_of_nothing_still_has_its_columns():
    frame = groups_to_frame([])
    assert list(frame.columns) == [
        "filer",
        "peer",
        "fiscal_year",
        "accession",
        "confidence",
    ]
    assert frame.empty
