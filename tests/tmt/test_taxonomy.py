"""Taxonomy tests. Offline against a committed, pruned submissions fixture.

The fixture is real: 105 SEC submissions payloads retrieved on 2026-09-11 and cut
down to the fields this module reads. That matters more here than in most test
files, because the claims this module makes in its comments are claims about the
register. A synthetic fixture would let the comments say whatever they liked. The
block of tests under "the documented traps" exists to bind every named trap to the
filing that motivated it, so the day the SEC recodes Palo Alto Networks the test
goes red and the comment gets corrected rather than quietly becoming a lie.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from conftest import FIXTURES, load_facts
from techval.errors import ConfigError, MissingDataError
from techval.tmt.taxonomy import (
    CONFIDENCE_AGREE_BASE,
    CONFIDENCE_AMBIGUOUS,
    CONFIDENCE_CURATED,
    CONFIDENCE_OVERRIDE_BASE,
    CONFIDENCE_SIC_CONTESTED,
    CONFIDENCE_SIC_ONLY,
    SEED,
    SIC_NEEDS_EVIDENCE,
    SIC_TO_SUB_VERTICAL,
    SIC_UNVERIFIED,
    STRENGTH_BONUS,
    WEAK_SIC,
    Company,
    SubVertical,
    classify,
    normalize_sic,
    score_business_text,
    tmt_universe,
)

SUBMISSIONS = json.loads((FIXTURES / "submissions_tmt.json").read_text())["companies"]

# Synthetic business descriptions, written in the register's own vocabulary. They
# stand in for Item 1 of the 10-K, which the engine does not fetch for a screen.
AMT_TEXT = """
We are one of the largest global REITs and a leading independent owner and
operator of multitenant communications real estate. Our portfolio of macro towers
and tower sites is leased to wireless carriers under long term ground leases with
contractual escalators. We add tenants to existing tower sites through lease
amendments and colocation, and we operate interconnection and data centers.
"""

AMZN_TEXT = """
We serve consumers through our online stores. Our marketplace enables third party
sellers to reach buyers and sellers around the world, and we report gross
merchandise value alongside a take rate. We provide cloud infrastructure with
compute and storage for developers, and we earn advertising revenue from
impressions sold to vendors.
"""

PANW_TEXT = """
We provide cybersecurity with a next generation firewall, zero trust network
access, endpoint protection, threat detection and security operations delivered
from the cloud.
"""


class FixtureSubmissions:
    """Stands in for EdgarClient over the committed submissions payloads.

    Raises the same MissingDataError that ``EdgarClient.ticker_to_cik`` raises for
    a ticker the SEC file does not carry, because the point of the unresolved path
    is that it survives the real failure, not a bespoke one.
    """

    knowledge_date = None

    def __init__(self, facts_for: tuple[str, ...] = ()) -> None:
        self.facts_for = {t.upper() for t in facts_for}
        self.submissions_calls: list[str] = []

    def submissions(self, ticker: str) -> dict:
        self.submissions_calls.append(ticker.upper())
        try:
            return SUBMISSIONS[ticker.upper()]
        except KeyError:
            raise MissingDataError(
                "CIK",
                ticker=ticker,
                hint="not present in the SEC ticker file; the engine covers US filers only",
            ) from None

    def company_facts(self, ticker: str):
        if ticker.upper() not in self.facts_for:
            raise MissingDataError("company facts", ticker=ticker)
        return load_facts(ticker)


@pytest.fixture
def client() -> FixtureSubmissions:
    return FixtureSubmissions()


def sic_of(ticker: str) -> str:
    return normalize_sic(SUBMISSIONS[ticker]["sic"])


# --------------------------------------------------------------------------- #
# The map
# --------------------------------------------------------------------------- #


def test_sic_map_is_four_character_codes_and_known_verticals():
    for code, vertical in SIC_TO_SUB_VERTICAL.items():
        assert code == normalize_sic(code), f"{code} is not stored zero padded"
        assert isinstance(vertical, SubVertical)


def test_codes_needing_evidence_are_not_also_mapped():
    """A code cannot both decide and abstain. The whole point of the abstain list
    is that the lookup returns nothing for it."""
    assert not set(SIC_NEEDS_EVIDENCE) & set(SIC_TO_SUB_VERTICAL)


def test_every_mapped_code_is_either_observed_in_a_filing_or_declared_unverified():
    """The map is allowed to rest on the SEC's code definition, but not silently.

    A code with a filer in the fixture has been checked against the register. A
    code without one is a reasonable reading of what the code means and nothing
    more, and it has to say so. This test is the thing that keeps the two apart as
    the seed list changes: add a name under 4899 and the code stops being
    unverified, and this fails until the declaration is updated.
    """
    observed = {normalize_sic(e["sic"]) for e in SUBMISSIONS.values()}
    for code in SIC_TO_SUB_VERTICAL:
        checked = code in observed
        declared = code in SIC_UNVERIFIED
        problem = "observed in a filing and also" if checked else "neither observed nor"
        assert checked != declared, f"SIC {code} is {problem} declared unverified"


def test_a_code_outside_every_list_falls_through_to_the_prior(client, assumptions):
    """Zebra files under 3560, General Industrial Machinery, which is mostly not
    TMT and so is deliberately absent from both lists. The name still has to land
    somewhere, and the only thing left standing is the curated prior."""
    assert sic_of("ZBRA") == "3560"
    assert "3560" not in SIC_TO_SUB_VERTICAL
    assert "3560" not in SIC_NEEDS_EVIDENCE
    zbra = {c.ticker: c for c in tmt_universe(client, assumptions)}["ZBRA"]
    assert zbra.sub_vertical is SubVertical.HARDWARE
    assert zbra.confidence == CONFIDENCE_CURATED
    assert "curated prior hardware" in zbra.source
    assert "unclassified" in zbra.source


def test_weak_codes_are_mapped_codes():
    """WEAK_SIC lowers the bar for overturning a mapped code, so a code that maps
    to nothing has no business being in it."""
    assert WEAK_SIC <= set(SIC_TO_SUB_VERTICAL)


def test_no_sic_code_claims_to_identify_payments():
    """The SEC has no payments code, and the map must not invent one.

    Visa, Mastercard, PayPal, FIS and Global Payments all file under 7389, the
    services catch-all, and Toast files under 7374 with an HR software vendor and
    an IT services firm. A map that answered "payments" for either code would be
    right about Toast and wrong about Workday, Accenture, Uber and eBay.
    """
    assert SubVertical.PAYMENTS not in SIC_TO_SUB_VERTICAL.values()
    for ticker in ("V", "MA", "PYPL", "FIS", "GPN"):
        assert sic_of(ticker) in SIC_NEEDS_EVIDENCE


@pytest.mark.parametrize(
    "ticker,code,vertical",
    [
        ("NVDA", "3674", SubVertical.SEMICONDUCTORS),
        ("LRCX", "3559", SubVertical.SEMICONDUCTORS),
        ("KLAC", "3827", SubVertical.SEMICONDUCTORS),
        ("AAPL", "3571", SubVertical.HARDWARE),
        ("CSCO", "3576", SubVertical.HARDWARE),
        ("GOOGL", "7370", SubVertical.INTERNET),
        ("MSFT", "7372", SubVertical.APPLICATION_SOFTWARE),
        ("CTSH", "7371", SubVertical.IT_SERVICES),
        ("G", "8742", SubVertical.IT_SERVICES),
        ("TMUS", "4812", SubVertical.TELECOM),
        ("VZ", "4813", SubVertical.TELECOM),
        ("CMCSA", "4841", SubVertical.TELECOM),
        ("FOX", "4833", SubVertical.MEDIA_ENTERTAINMENT),
        ("NFLX", "7841", SubVertical.MEDIA_ENTERTAINMENT),
        ("NYT", "2711", SubVertical.MEDIA_ENTERTAINMENT),
        ("OMC", "7311", SubVertical.MEDIA_ENTERTAINMENT),
    ],
)
def test_sic_spine_maps_the_filing_code_to_the_expected_vertical(ticker, code, vertical):
    """The filer really does carry that code, and the map really does answer."""
    assert sic_of(ticker) == code
    assert SIC_TO_SUB_VERTICAL[code] is vertical


# --------------------------------------------------------------------------- #
# The documented traps, bound to the filings that motivated them
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "ticker,code,why",
    [
        ("AMT", "6798", "a tower company files as a REIT"),
        ("CCI", "6798", "so does the second tower company"),
        ("EQIX", "6798", "and the data centre operator"),
        ("AMZN", "5961", "the largest cloud business files as a mail-order catalogue"),
        ("ABNB", "7340", "Airbnb files under services to dwellings and buildings"),
        ("BKNG", "4700", "online travel files as transportation services"),
        ("ACN", "7389", "the largest IT consultancy files under services NEC"),
    ],
)
def test_codes_that_cannot_decide_are_declared_rather_than_guessed(ticker, code, why):
    assert sic_of(ticker) == code, why
    assert code in SIC_NEEDS_EVIDENCE
    assert code not in SIC_TO_SUB_VERTICAL


@pytest.mark.parametrize(
    "ticker,code,mapped,truth",
    [
        # The security software vendors that still carry an appliance code.
        ("PANW", "3577", SubVertical.HARDWARE, SubVertical.INFRASTRUCTURE_SOFTWARE),
        ("FTNT", "3577", SubVertical.HARDWARE, SubVertical.INFRASTRUCTURE_SOFTWARE),
        # Zscaler files with the offshore project shops.
        ("ZS", "7371", SubVertical.IT_SERVICES, SubVertical.INFRASTRUCTURE_SOFTWARE),
        # IBM carries a typewriter-era hardware code.
        ("IBM", "3570", SubVertical.HARDWARE, SubVertical.IT_SERVICES),
        # Qualcomm designs mobile silicon under a broadcasting equipment code.
        ("QCOM", "3663", SubVertical.HARDWARE, SubVertical.SEMICONDUCTORS),
        # Warner Bros Discovery and Roku sit in the cable access code.
        ("WBD", "4841", SubVertical.TELECOM, SubVertical.MEDIA_ENTERTAINMENT),
        ("ROKU", "4841", SubVertical.TELECOM, SubVertical.MEDIA_ENTERTAINMENT),
        # Take-Two and Roblox file next to enterprise software.
        ("TTWO", "7372", SubVertical.APPLICATION_SOFTWARE, SubVertical.GAMING),
        ("RBLX", "7372", SubVertical.APPLICATION_SOFTWARE, SubVertical.GAMING),
        # Workday files with the data processing bureaus.
        ("WDAY", "7374", None, SubVertical.APPLICATION_SOFTWARE),
    ],
)
def test_named_traps_are_real_and_the_code_alone_gets_them_wrong(
    ticker, code, mapped, truth
):
    """Each of these is named in a comment. If the register changes, this fails."""
    assert sic_of(ticker) == code
    assert SIC_TO_SUB_VERTICAL.get(code) is mapped
    assert mapped is not truth
    assert SEED[ticker] is truth


def test_the_traps_that_are_wide_codes_are_marked_weak():
    """A code that is wrong about a large filer should be cheaper to overturn."""
    for code in ("3577", "7371", "7372", "4841", "7370"):
        assert code in WEAK_SIC


# --------------------------------------------------------------------------- #
# Evidence scoring
# --------------------------------------------------------------------------- #


def test_evidence_counts_distinct_phrases_not_occurrences():
    """Ninety mentions of "wafer" is one point about the business, not ninety."""
    once = score_business_text("We ship a wafer.")
    many = score_business_text("wafer " * 90)
    assert once.hits[SubVertical.SEMICONDUCTORS] == 1
    assert many.hits[SubVertical.SEMICONDUCTORS] == 1


def test_evidence_matches_across_the_line_breaks_a_stripped_filing_carries():
    assert score_business_text("macro\n   towers").hits[SubVertical.TOWERS_FIBER] >= 1


def test_evidence_does_not_match_inside_a_longer_word():
    """"api" must not fire on "capital", or every filing becomes infrastructure."""
    assert SubVertical.INFRASTRUCTURE_SOFTWARE not in score_business_text(
        "capital rapidly"
    ).hits


def test_empty_text_scores_nothing():
    assert score_business_text(None).total == 0
    assert score_business_text("   ").total == 0


# --------------------------------------------------------------------------- #
# classify: the override cases
# --------------------------------------------------------------------------- #


def test_american_tower_is_admitted_from_a_reit_code_on_evidence(assumptions):
    """6798 is a tax election, not an industry. The evidence has to carry it, and
    the source string has to show the code abstaining rather than agreeing."""
    vertical, confidence, source = classify(
        sic_of("AMT"), SUBMISSIONS["AMT"]["name"], AMT_TEXT, assumptions
    )
    assert vertical is SubVertical.TOWERS_FIBER
    assert confidence >= CONFIDENCE_OVERRIDE_BASE
    assert "6798" in source
    assert "REIT election" in source
    assert "business text alone" in source


def test_amazon_is_admitted_from_a_catalogue_retail_code_on_evidence(assumptions):
    vertical, confidence, source = classify(
        sic_of("AMZN"), SUBMISSIONS["AMZN"]["name"], AMZN_TEXT, assumptions
    )
    assert vertical is SubVertical.INTERNET
    assert confidence >= CONFIDENCE_OVERRIDE_BASE
    assert "5961" in source
    assert "business text alone" in source


def test_evidence_overrides_a_mapped_code_and_names_what_it_beat(assumptions):
    """Palo Alto Networks: the code says hardware and the business says security
    software. The override must be stated, with the losing answer named, because a
    peer set built on a silent relabel is one nobody can defend."""
    vertical, confidence, source = classify(
        sic_of("PANW"), SUBMISSIONS["PANW"]["name"], PANW_TEXT, assumptions
    )
    assert vertical is SubVertical.INFRASTRUCTURE_SOFTWARE
    assert "overrides" in source
    assert "3577" in source
    assert "hardware" in source
    assert (
        CONFIDENCE_OVERRIDE_BASE
        <= confidence
        <= CONFIDENCE_OVERRIDE_BASE + STRENGTH_BONUS
    )


def test_an_override_starts_out_less_confident_than_accepting_the_code(assumptions):
    """Overturning the registrant's own filing code is a bigger claim than taking
    it, so the override floor sits below the SIC-only rung."""
    assert CONFIDENCE_OVERRIDE_BASE < CONFIDENCE_SIC_ONLY


def test_code_and_evidence_agreeing_outranks_either_alone(assumptions):
    vertical, confidence, source = classify(
        "3674",
        "NVIDIA CORP",
        "We are a fabless semiconductor company shipping integrated circuits from a "
        "foundry at an advanced process technology node.",
        assumptions,
    )
    assert vertical is SubVertical.SEMICONDUCTORS
    assert confidence >= CONFIDENCE_AGREE_BASE
    assert "confirmed by business text" in source


# --------------------------------------------------------------------------- #
# classify: margins, floors and ties
# --------------------------------------------------------------------------- #


def test_a_single_stray_phrase_is_not_a_business_description(assumptions):
    """One matched phrase sits under the evidence floor, so the code keeps the
    name and the source says the text was too thin rather than that it agreed."""
    vertical, confidence, source = classify(
        "3674", "Some Semi Inc", "Our board discussed a marketplace.", assumptions
    )
    assert vertical is SubVertical.SEMICONDUCTORS
    assert confidence == CONFIDENCE_SIC_ONLY
    assert "below the evidence floor" in source


def test_evidence_under_the_margin_leaves_the_code_in_place_and_records_the_contest(
    assumptions,
):
    """A narrow lean is not an override. The code keeps the name, at a confidence
    below SIC-only, and the source names the answer that nearly won."""
    # Four internet phrases against three semiconductor ones is a 14% margin, under
    # the 15% a specific code demands.
    text = (
        "We run a marketplace with a take rate, monthly active users and online "
        "advertising. We also sell semiconductors, wafers and silicon."
    )
    vertical, confidence, source = classify("3674", "Mixed Co", text, assumptions)
    assert vertical is SubVertical.SEMICONDUCTORS
    assert confidence == CONFIDENCE_SIC_CONTESTED
    assert confidence < CONFIDENCE_SIC_ONLY
    assert "internet" in source
    assert "under the" in source


def test_a_wide_code_is_cheaper_to_overturn_than_a_specific_one(assumptions):
    """The same evidence should move a filer off "all software" more readily than
    off "semiconductors", because 7372 is one code for the whole industry."""
    # A 14% margin: over the 5% a wide code demands, under the 15% a specific one
    # demands. The same evidence therefore lands differently, which is the point.
    text = (
        "We provide observability, log management, kubernetes and cloud "
        "infrastructure. We also sell semiconductors, wafers and silicon."
    )
    weak_pick, _, _ = classify("7372", "Ambiguous Inc", text, assumptions)
    strong_pick, _, _ = classify("3674", "Ambiguous Inc", text, assumptions)
    assert "7372" in WEAK_SIC and "3674" not in WEAK_SIC
    assert weak_pick is SubVertical.INFRASTRUCTURE_SOFTWARE
    assert strong_pick is SubVertical.SEMICONDUCTORS


def test_confidence_rises_with_the_margin_inside_a_rung(assumptions):
    narrow = (
        "We provide observability and cloud infrastructure. We also run a "
        "marketplace with a take rate and monthly active users."
    )
    wide = (
        "We provide observability, cloud infrastructure, log management, "
        "kubernetes, containers and a developer platform."
    )
    _, narrow_conf, _ = classify("6798", "Infra Co", narrow, assumptions)
    _, wide_conf, _ = classify("6798", "Infra Co", wide, assumptions)
    assert wide_conf > narrow_conf


def test_a_tie_is_reported_as_a_tie_and_not_resolved_by_dictionary_order(assumptions):
    """Two buckets level on the evidence and no code to break it. Answering
    "internet" because i sorts before s would be inventing a decision."""
    text = "We sell wafers and silicon. We run a marketplace with a take rate."
    vertical, confidence, source = classify(None, "Tied Corp", text, assumptions)
    assert vertical is None
    assert confidence == CONFIDENCE_AMBIGUOUS
    assert "ambiguous" in source
    assert "internet" in source and "semiconductors" in source


def test_a_tie_names_every_candidate_not_just_two(assumptions):
    text = (
        "We sell wafers and silicon. We run a marketplace with a take rate. We "
        "provide observability and a developer platform."
    )
    vertical, _, source = classify(None, "Three Way", text, assumptions)
    assert vertical is None
    for name in ("internet", "semiconductors", "infrastructure_software"):
        assert name in source


def test_a_code_that_is_one_of_the_tied_candidates_breaks_the_tie(assumptions):
    """This is the one job SIC can still do when the evidence is level."""
    text = "We sell wafers and silicon. We run a marketplace with a take rate."
    vertical, confidence, source = classify("3674", "Tied Corp", text, assumptions)
    assert vertical is SubVertical.SEMICONDUCTORS
    assert confidence == CONFIDENCE_AGREE_BASE
    assert "broke a tie" in source


def test_a_tie_a_mapped_code_cannot_resolve_keeps_the_code_and_records_it(assumptions):
    text = "We sell wafers and silicon. We run a marketplace with a take rate."
    vertical, confidence, source = classify("7841", "Tied Corp", text, assumptions)
    assert vertical is SubVertical.MEDIA_ENTERTAINMENT
    assert confidence == CONFIDENCE_SIC_CONTESTED
    assert "does not resolve it" in source


def test_a_company_that_is_not_tmt_is_returned_unclassified(assumptions):
    """No code, no evidence, no answer. Returning a bucket anyway would put a
    cement company in a software comp set."""
    vertical, confidence, source = classify("3241", "Cement Co", None, assumptions)
    assert vertical is None
    assert confidence == 0.0
    assert "unclassified" in source


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [("7372", "7372"), (7372, "7372"), (" 7372 ", "7372"), (100, "0100"),
     (None, None), ("", None), ("n/a", None)],
)
def test_normalize_sic(raw, expected):
    assert normalize_sic(raw) == expected


def test_out_of_range_margin_is_a_config_error(assumptions):
    """A margin of 1.0 can never be met, so every company would silently pin to
    its code while the output claimed the evidence had been weighed."""

    class Knobs:
        keyword_margin = 1.0

    class TMT:
        sub_vertical = Knobs()

    assumptions.__dict__["tmt"] = TMT()
    with pytest.raises(ConfigError, match="never be met"):
        classify("7372", "X", None, assumptions)


def test_zero_evidence_floor_is_a_config_error(assumptions):
    class Knobs:
        min_keyword_hits = 0

    class TMT:
        sub_vertical = Knobs()

    assumptions.__dict__["tmt"] = TMT()
    with pytest.raises(ConfigError, match="at least 1"):
        classify("7372", "X", None, assumptions)


def test_company_row_is_flat_and_keeps_the_audit_trail():
    company = Company(
        ticker="AMT",
        cik=1053507,
        name="AMERICAN TOWER CORP /MA/",
        sic="6798",
        sub_vertical=SubVertical.TOWERS_FIBER,
        confidence=0.65,
        source="classified from business text alone",
        notes="",
    )
    row = company.row()
    assert row["sub_vertical"] == "towers_fiber"
    assert row["cik"] == 1053507
    assert row["source"]


# --------------------------------------------------------------------------- #
# The seed list
# --------------------------------------------------------------------------- #


def test_seed_is_the_size_of_a_candidate_pool():
    assert 60 <= len(SEED) <= 120


def test_seed_spans_every_sub_vertical():
    """Breadth is the product. A TMT platform that can only see software is a
    software platform."""
    assert set(SEED.values()) == set(SubVertical)


@pytest.mark.parametrize(
    "letter,verticals,examples",
    [
        (
            "technology",
            {
                SubVertical.SEMICONDUCTORS,
                SubVertical.HARDWARE,
                SubVertical.IT_SERVICES,
                SubVertical.INFRASTRUCTURE_SOFTWARE,
                SubVertical.APPLICATION_SOFTWARE,
                SubVertical.INTERNET,
                SubVertical.PAYMENTS,
            },
            ("NVDA", "AAPL", "MSFT", "CRM", "GOOGL", "V", "ACN"),
        ),
        (
            "media",
            {SubVertical.MEDIA_ENTERTAINMENT, SubVertical.GAMING},
            ("DIS", "NFLX", "WBD", "LYV", "TTWO", "RBLX"),
        ),
        (
            "telecom",
            {SubVertical.TELECOM, SubVertical.TOWERS_FIBER},
            ("T", "VZ", "TMUS", "AMT", "CCI", "EQIX"),
        ),
    ],
)
def test_seed_covers_all_three_letters_of_tmt(letter, verticals, examples):
    covered = set(SEED.values())
    assert verticals <= covered, f"{letter} is not covered"
    for ticker in examples:
        assert ticker in SEED, f"{ticker} missing from the {letter} pool"
        assert SEED[ticker] in verticals


def test_seed_holds_the_semis_media_and_carriers_the_brief_named():
    for ticker in ("NVDA", "AVGO", "AMD", "TXN", "MU"):
        assert SEED[ticker] is SubVertical.SEMICONDUCTORS
    for ticker in ("DIS", "NFLX", "WBD", "LYV"):
        assert SEED[ticker] is SubVertical.MEDIA_ENTERTAINMENT
    for ticker in ("T", "VZ", "TMUS", "LUMN"):
        assert SEED[ticker] is SubVertical.TELECOM
    for ticker in ("AMT", "CCI", "SBAC", "EQIX", "DLR"):
        assert SEED[ticker] is SubVertical.TOWERS_FIBER


def test_seed_does_not_carry_a_reused_ticker():
    """PARA meant Paramount for years and today resolves to an unrelated small
    cap. Paramount files as PSKY. A universe keyed on the stale string would pull
    the wrong company's filings and never say so."""
    assert "PARA" not in SEED
    assert SEED["PSKY"] is SubVertical.MEDIA_ENTERTAINMENT


# --------------------------------------------------------------------------- #
# tmt_universe
# --------------------------------------------------------------------------- #


def test_universe_resolves_the_seed_and_carries_the_cik(client, assumptions):
    universe = tmt_universe(client, assumptions)
    assert len(universe) == len(SEED)
    amt = next(c for c in universe if c.ticker == "AMT")
    assert amt.cik == int(SUBMISSIONS["AMT"]["cik"])
    assert amt.name == SUBMISSIONS["AMT"]["name"]
    assert amt.sub_vertical is SubVertical.TOWERS_FIBER


def test_an_unresolvable_ticker_is_recorded_and_never_dropped(client, assumptions):
    """EA was taken private and Fiserv is still listed under a symbol the SEC file
    has not updated. Both vanish from a today's ticker file. A universe that
    silently shrank would look stable while losing exactly the names whose absence
    is the finding."""
    universe = tmt_universe(client, assumptions)
    tickers = [c.ticker for c in universe]
    assert set(SEED) <= set(tickers)

    unresolved = {c.ticker: c for c in universe if c.sub_vertical is None}
    assert {"EA", "FI", "JNPR", "IPG", "FYBR"} <= set(unresolved)
    for company in unresolved.values():
        assert company.confidence == 0.0
        assert company.cik is None
        assert "unresolved" in company.source
        assert "ticker file" in company.notes


def test_the_curated_prior_overrides_the_code_and_says_so(client, assumptions):
    """No business text was read, so the only thing standing against a named
    analyst's call is a box the registrant ticked once at registration."""
    universe = {c.ticker: c for c in tmt_universe(client, assumptions)}

    qcom = universe["QCOM"]
    assert qcom.sic == "3663"
    assert qcom.sub_vertical is SubVertical.SEMICONDUCTORS
    assert qcom.confidence == CONFIDENCE_CURATED
    assert "curated prior semiconductors overrides" in qcom.source
    assert "3663" in qcom.source


def test_the_code_keeps_the_name_where_the_prior_agrees_with_it(client, assumptions):
    universe = {c.ticker: c for c in tmt_universe(client, assumptions)}
    nflx = universe["NFLX"]
    assert nflx.sic == "7841"
    assert nflx.sub_vertical is SubVertical.MEDIA_ENTERTAINMENT
    assert nflx.confidence == CONFIDENCE_SIC_ONLY
    assert "curated prior agrees" in nflx.notes


def test_business_text_is_not_read_unless_a_reader_is_given(client, assumptions):
    """A hundred candidates would be a hundred 10-K fetches for a screen that
    mostly does not need them."""
    universe = {c.ticker: c for c in tmt_universe(client, assumptions)}
    assert "no business text was read" in universe["PANW"].source


def test_a_reader_turns_a_code_only_call_into_a_reconciled_one(client, assumptions):
    texts = {"PANW": PANW_TEXT, "AMT": AMT_TEXT, "AMZN": AMZN_TEXT}
    universe = {
        c.ticker: c
        for c in tmt_universe(client, assumptions, business_text=texts.get)
    }
    panw = universe["PANW"]
    assert panw.sub_vertical is SubVertical.INFRASTRUCTURE_SOFTWARE
    assert "overrides SIC 3577" in panw.source
    assert panw.confidence > CONFIDENCE_CURATED

    amt = universe["AMT"]
    assert amt.sub_vertical is SubVertical.TOWERS_FIBER
    assert "business text alone" in amt.source


def test_a_reader_that_fails_is_recorded_and_the_code_still_decides(client, assumptions):
    def reader(ticker: str) -> str:
        raise MissingDataError("10-K text", ticker=ticker)

    universe = {c.ticker: c for c in tmt_universe(client, assumptions, business_text=reader)}
    nflx = universe["NFLX"]
    assert nflx.sub_vertical is SubVertical.MEDIA_ENTERTAINMENT
    assert "business text unavailable" in nflx.notes


# --------------------------------------------------------------------------- #
# Point in time
# --------------------------------------------------------------------------- #


def test_a_past_universe_excludes_a_filer_that_was_not_yet_reporting(client, assumptions):
    """Zscaler listed in 2018 and Paramount Skydance filed its first periodic
    report in 2025. Neither belongs in a 2017 universe, and a 2017 universe that
    held them would be measuring hindsight."""
    early = {c.ticker for c in tmt_universe(client, assumptions, as_of=date(2017, 1, 1))}
    assert "ZS" not in early
    assert "PSKY" not in early
    assert "PLTK" not in early
    assert "AAPL" in early

    today = {c.ticker for c in tmt_universe(client, assumptions, as_of=date(2026, 9, 10))}
    assert {"ZS", "PSKY", "PLTK"} <= today


def test_the_universe_only_grows_as_the_knowledge_date_moves_forward(client, assumptions):
    sizes = [
        len(tmt_universe(client, assumptions, as_of=as_of))
        for as_of in (date(2017, 1, 1), date(2020, 1, 1), date(2026, 9, 10))
    ]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_admission_uses_the_first_periodic_report_not_the_first_filing(client, assumptions):
    """Zscaler's CIK carries a draft registration statement from 2017 and its
    first 10-Q from June 2018. A company filing to go public is not yet a company
    you can buy, so the 2018 date is the one that admits it."""
    assert SUBMISSIONS["ZS"]["filings"]["recent"]["form"][0] == "DRS"
    assert "2017" in SUBMISSIONS["ZS"]["filings"]["recent"]["filingDate"][0]

    before = {c.ticker for c in tmt_universe(client, assumptions, as_of=date(2018, 1, 1))}
    after = {c.ticker for c in tmt_universe(client, assumptions, as_of=date(2018, 12, 31))}
    assert "ZS" not in before
    assert "ZS" in after


def test_a_shard_dated_admission_declares_that_it_is_the_record_start(client, assumptions):
    """Roblox's CIK carries submissions from 2005 against a 2021 listing, because
    a private company files Form D notices. The shards hold no form types, so the
    date cannot be refined without fetching every shard, and the record says so
    rather than passing a 2005 date off as a first report."""
    assert SUBMISSIONS["RBLX"]["filings"]["files"][0]["filingFrom"].startswith("2005")
    universe = {
        c.ticker: c
        for c in tmt_universe(client, assumptions, as_of=date(2026, 9, 10))
    }
    assert "start of the EDGAR record" in universe["RBLX"].notes
    # A filer whose whole history fits inside the recent window needs no caveat.
    assert "start of the EDGAR record" not in universe["PSKY"].notes


def test_the_knowledge_date_comes_from_the_client_then_the_assumptions(assumptions):
    """A client pinned to a date is already refusing facts filed after it, so a
    universe reaching past it would hold companies the rest of the run cannot
    price."""
    pinned = FixtureSubmissions()
    pinned.knowledge_date = date(2017, 1, 1)
    assert "ZS" not in {c.ticker for c in tmt_universe(pinned, assumptions)}

    assumptions.as_of = "2017-01-01"
    assert "ZS" not in {c.ticker for c in tmt_universe(FixtureSubmissions(), assumptions)}


def test_a_bad_as_of_is_a_config_error(client, assumptions):
    assumptions.as_of = "last Tuesday"
    with pytest.raises(ConfigError, match="ISO date"):
        tmt_universe(client, assumptions)


# --------------------------------------------------------------------------- #
# The market cap screen
# --------------------------------------------------------------------------- #


def test_a_market_cap_floor_without_a_price_source_is_a_config_error(client, assumptions):
    with pytest.raises(ConfigError, match="MarketData"):
        tmt_universe(client, assumptions, min_market_cap=1000.0)


def test_a_name_that_cannot_be_priced_is_kept_with_the_reason(assumptions, market):
    """The screen failing is not evidence that the company is small. Cutting on a
    failed lookup would quietly shrink the universe toward whatever happens to be
    easy to price."""
    client = FixtureSubmissions(facts_for=("DDOG", "CRWD", "MDB", "ZS"))
    screened = {
        c.ticker: c
        for c in tmt_universe(
            client, assumptions, market=market, min_market_cap=1.0
        )
    }
    assert "AAPL" in screened
    assert "market cap screen" in screened["AAPL"].notes
    assert "DDOG" in screened
    assert "market cap screen" not in screened["DDOG"].notes


def test_the_floor_cuts_a_priced_name_that_is_too_small(assumptions, market):
    client = FixtureSubmissions(facts_for=("DDOG", "CRWD", "MDB", "ZS"))
    kept = {
        c.ticker
        for c in tmt_universe(client, assumptions, market=market, min_market_cap=1.0)
    }
    cut = {
        c.ticker
        for c in tmt_universe(client, assumptions, market=market, min_market_cap=1e9)
    }
    assert {"DDOG", "CRWD", "MDB", "ZS"} <= kept
    assert not {"DDOG", "CRWD", "MDB", "ZS"} & cut


# --------------------------------------------------------------------------- #
# Config-driven extension
# --------------------------------------------------------------------------- #


def test_extra_tickers_and_overrides_come_from_the_assumptions(client, assumptions):
    class Knobs:
        extra_tickers = ["nvda", "SHOP"]
        overrides = {"shop": "internet"}

    class TMT:
        sub_vertical = Knobs()

    assumptions.__dict__["tmt"] = TMT()
    universe = {c.ticker: c for c in tmt_universe(client, assumptions)}
    assert universe["SHOP"].sub_vertical is SubVertical.INTERNET
    assert "curated prior internet overrides" in universe["SHOP"].source
    assert universe["NVDA"].sub_vertical is SubVertical.SEMICONDUCTORS


def test_an_override_naming_a_bucket_that_does_not_exist_is_a_config_error(
    client, assumptions
):
    class Knobs:
        overrides = {"SHOP": "biotech"}

    class TMT:
        sub_vertical = Knobs()

    assumptions.__dict__["tmt"] = TMT()
    with pytest.raises(ConfigError, match="not a sub-vertical"):
        tmt_universe(client, assumptions)
