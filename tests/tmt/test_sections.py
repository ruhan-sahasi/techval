"""Item segmentation, tested against the traps rather than against a happy path.

Two real 10-Ks are committed in full, gzipped, under ``tests/fixtures``. Pruning
them to a few thousand characters around each boundary was the alternative, and it
would have deleted the whole problem: the contents table sits 11,000 characters
before the real Item 1, the eight self-references that defeat a last-match split
are scattered through the body, and the running head that glues "Table of
ContentsItem 7." together only appears because the filer repeats it on every page.
A fixture that removed those would pass whatever the splitter did.

Datadog is the calendar-year filer with title-case headers. CrowdStrike is the
January filer that shouts its headers in capitals, repeats a running head, and
puts its exhibit index after Item 16. No test here touches the network.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path

import pytest

from techval.errors import ConfigError, MissingDataError
from techval.nlp import sections as sections_module
from techval.nlp.sections import FilingSections, Section, load_sections, split_items

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# Items the platform actually reads, plus the ones the assignment requires be
# covered. Item 1 feeds the peer model, 1A the risk diff, 7 the KPI extractor.
REQUIRED = ("1", "1A", "1B", "2", "3", "5", "7", "7A", "8", "9A")


def load_filing(name: str) -> dict:
    with gzip.open(FIXTURES / name, "rt", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def ddog_filing() -> dict:
    return load_filing("filing_text_DDOG_2025.json.gz")


@pytest.fixture(scope="module")
def crwd_filing() -> dict:
    return load_filing("filing_text_CRWD_2026.json.gz")


@pytest.fixture(scope="module")
def ddog(ddog_filing) -> FilingSections:
    return split_items(ddog_filing["text"])


@pytest.fixture(scope="module")
def crwd(crwd_filing) -> FilingSections:
    return split_items(crwd_filing["text"])


class FilingTextClient:
    """Stands in for EdgarClient over one frozen filing, and counts the fetches.

    The count is the point of the cache test: reading Item 1 and Item 7 of the same
    10-K must strip the document once, not twice.
    """

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.text_calls = 0
        self.list_calls = 0

    def filings(self, ticker: str, forms=("10-K",), since=None, limit=20) -> list[dict]:
        self.list_calls += 1
        if self.payload["form"] not in forms:
            return []
        return [
            {
                "accession": self.payload["accession"],
                "filed": date.fromisoformat(self.payload["filed"]),
                "form": self.payload["form"],
                "document": "primary.htm",
                "period": self.payload["period"],
            }
        ]

    def filing_text(self, ticker: str, filing: dict) -> str:
        self.text_calls += 1
        return self.payload["text"]


# --------------------------------------------------------------------------- #
# The table of contents trap, on a document built to contain nothing else
# --------------------------------------------------------------------------- #

PROSE = (
    "The company sells software to enterprises under subscription contracts that "
    "renew annually and are billed in advance. " * 40
)
RISKS = (
    "We have a history of operating losses and may not achieve profitability. "
    "Our business depends on retaining customers whose contracts renew annually. " * 40
)


def synthetic_filing(
    *,
    toc: bool = True,
    cross_reference: bool = False,
    item_1_body: str = PROSE,
    item_1a_body: str = RISKS,
    preamble: str = "",
) -> str:
    """A miniature 10-K with the same shape as a real one.

    The contents table is written the way markup stripping leaves one: each entry
    is a title run straight into its page number, with no space anywhere.
    """
    head = "ANNUAL REPORT ON FORM 10-K" + preamble
    if toc:
        head += (
            "TABLE OF CONTENTSPagePART I.Item 1.Business5Item 1A.Risk Factors13"
            "Item 1B.Unresolved Staff Comments42Item 2.Properties44"
            "Item 3.Legal Proceedings44Item 5.Market for Registrant's Common Equity45"
            "Item 7.Management's Discussion and Analysis46"
            "Item 7A.Quantitative and Qualitative Disclosures About Market Risk57"
            "Item 8.Financial Statements and Supplementary Data59"
            "Item 9A.Controls and Procedures94"
        )
    mdna = "Revenue grew on the strength of new customer additions. " * 40
    if cross_reference:
        mdna += (
            "Concentrations of our revenue are described under Part I - Item 1. "
            "Business in this Annual Report on Form 10-K. "
        )
    return (
        head
        + "4PART IItem 1. Business" + item_1_body
        + "Item 1A. Risk Factors" + item_1a_body
        + "Item 1B. Unresolved Staff CommentsNone."
        + "Item 2. PropertiesWe lease our head office. " * 1
        + "Item 3. Legal ProceedingsWe are party to no material proceedings. "
        + "Item 5. Market for Registrant's Common EquityOur Class A stock trades. "
        + "Item 7. Management's Discussion and Analysis" + mdna
        + "Item 7A. Quantitative and Qualitative Disclosures About Market RiskWe hold cash. "
        + "Item 8. Financial Statements and Supplementary DataSee the statements. "
        + "Item 9A. Controls and ProceduresDisclosure controls were effective. "
    )


def test_table_of_contents_is_not_returned_as_item_1():
    """The whole point. A first-match split hands back the contents table."""
    text = synthetic_filing()
    parts = split_items(text)

    business = parts.business
    assert business is not None
    assert business.start_char > text.index("TABLE OF CONTENTS")
    assert "Risk Factors13" not in business.text
    assert business.text.startswith("Item 1. Business")
    assert business.word_count > 200


def test_the_contents_table_is_named_in_the_notes():
    parts = split_items(synthetic_filing())
    assert any("contents table at characters" in n for n in parts.notes)


def test_a_document_without_a_contents_table_still_splits():
    """A filing that has no table must not be penalised for not having one."""
    parts = split_items(synthetic_filing(toc=False))
    assert set(REQUIRED) <= set(parts.sections)
    assert any("no contents table found" in n for n in parts.notes)


def test_a_cross_reference_does_not_claim_item_1():
    """Last-match would put Item 1 inside Item 7. Order and punctuation stop it."""
    text = synthetic_filing(cross_reference=True)
    parts = split_items(text)

    reference = text.index("described under Part I - Item 1.")
    assert parts.business is not None
    assert parts.business.end_char < reference
    assert parts.mdna is not None
    assert parts.mdna.start_char < reference < parts.mdna.end_char
    assert any("rejected as cross-references" in n for n in parts.notes)


@pytest.mark.parametrize(
    "header",
    [
        "Item 1. Business",
        "Item 1: Business",
        "ITEM 1 - BUSINESS",
        "Item 1 \u2014 Business",
        "Item\u00a01.\u00a0Business",
        "ITEM 1. BUSINESS",
    ],
)
def test_header_punctuation_variants_all_parse(header: str):
    text = "Cover page." + header + PROSE + "Item 1A. Risk Factors" + RISKS
    parts = split_items(text)

    assert parts.business is not None
    assert parts.business.word_count > 200
    assert parts.risk_factors is not None
    assert "Risk Factors" in parts.risk_factors.text[:40]


def test_item_1a_is_never_swallowed_by_item_1():
    """Item 1 runs straight into Item 1A with no separator. Order of match matters."""
    text = "Cover page.Item 1. Business" + PROSE + "Item 1A. Risk Factors" + RISKS
    parts = split_items(text)

    assert parts.business is not None and parts.risk_factors is not None
    assert "Item 1A." not in parts.business.text
    assert parts.business.end_char == parts.risk_factors.start_char
    assert parts.risk_factors.text.startswith("Item 1A.")


# --------------------------------------------------------------------------- #
# The real Datadog 10-K
# --------------------------------------------------------------------------- #


def test_ddog_item_1_is_the_business_description(ddog, ddog_filing):
    """Thousands of words of prose, not a list of section names and page numbers."""
    business = ddog.business
    assert business is not None
    assert business.title == "Business"
    assert business.start_char == 40_836
    assert business.word_count == 5_397
    assert "observability" in business.text.lower()
    # The tell-tale of a contents-table hit: an entry and its page number.
    assert "Risk Factors13" not in business.text
    assert "TABLE OF CONTENTS" not in business.text


def test_ddog_finds_every_item_the_platform_reads(ddog):
    assert set(REQUIRED) <= set(ddog.sections)
    assert not [f for f in ddog.flags if "no header found" in f]


def test_ddog_risk_factors_dwarf_the_business_description(ddog):
    """True of every modern filer, and the sanity check that says so must stay quiet."""
    assert ddog.risk_factors is not None and ddog.business is not None
    assert ddog.risk_factors.word_count > 4 * ddog.business.word_count
    assert not [f for f in ddog.flags if "shorter than Item 1" in f]


def test_ddog_mdna_is_the_discussion_and_not_the_statements(ddog):
    mdna = ddog.mdna
    assert mdna is not None
    assert mdna.title.lower().startswith("management")
    assert "Results of Operations" in mdna.text
    # The audited statements are Item 8, and feeding them to a KPI reader as MD&A
    # is exactly the failure this module exists to prevent.
    assert "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM" not in mdna.text.upper()


def test_ddog_sections_tile_the_document_in_order(ddog):
    ordered = sorted(ddog.sections.values(), key=lambda s: s.start_char)
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.end_char == later.start_char
    assert ddog.covered_fraction > 0.85
    assert not [f for f in ddog.flags if "cover" in f and "under" in f]


def test_ddog_section_text_is_exactly_the_slice_it_claims(ddog, ddog_filing):
    """Auditability. A section that cannot be checked against the filing is a claim."""
    raw = ddog_filing["text"]
    assert ddog.raw_length == len(raw)
    for sec in ddog.sections.values():
        assert sec.text == raw[sec.start_char : sec.end_char]


def test_ddog_contents_table_falls_in_no_section(ddog, ddog_filing):
    toc_at = ddog_filing["text"].index("TABLE OF CONTENTSPage")
    assert all(
        not (s.start_char <= toc_at < s.end_char) for s in ddog.sections.values()
    )


def test_ddog_rows_render_in_filing_order(ddog):
    rows = ddog.rows()
    assert [r[0] for r in rows][:4] == ["1", "1A", "1B", "1C"]
    item, title, words, share = rows[0]
    assert (item, title) == ("1", "Business")
    assert words == 5_397
    assert 0.0 < share < 1.0


# --------------------------------------------------------------------------- #
# The real CrowdStrike 10-K: capitals, a running head, and a trailing index
# --------------------------------------------------------------------------- #


def test_crwd_uppercase_headers_parse(crwd):
    business = crwd.business
    assert business is not None
    assert business.title == "BUSINESS"
    assert business.word_count > 5_000
    assert set(REQUIRED) <= set(crwd.sections)


def test_crwd_running_head_does_not_hide_a_section(crwd, crwd_filing):
    """CrowdStrike repeats a running head on every page, glued to the header.

    Every one of its section headers is preceded by "Table of Contents" with no
    space, so a splitter that requires a word boundary in front of "Item" finds
    none of them, and a splitter that reads the preceding lowercase "s" as prose
    rejects all of them as cross-references.
    """
    raw = crwd_filing["text"]
    for item in ("1", "1A", "7", "8", "9A"):
        section = crwd.sections[item]
        # Directly before the header, or a part label before that.
        assert "Table of Contents" in raw[section.start_char - 24 : section.start_char]

    assert crwd.mdna is not None and crwd.mdna.word_count > 5_000
    assert crwd.get("8") is not None and crwd.get("8").word_count > 10_000


def test_crwd_exhibit_index_after_item_16_is_not_silently_dropped(crwd):
    """A known limitation, asserted so that it stays known.

    Nothing marks the start of an exhibit index, so CrowdStrike's lands inside
    Item 16 rather than in a section of its own. The text is not lost, but Item 16
    is not one word of "None." either, and anyone reading it should know that.
    """
    summary = crwd.get("16")
    assert summary is not None
    assert summary.word_count > 500
    assert "EXHIBIT INDEX" in summary.text.upper()


# --------------------------------------------------------------------------- #
# What the split says when it cannot do the job
# --------------------------------------------------------------------------- #


def test_a_missing_item_is_flagged_and_never_fabricated():
    text = (
        "Cover page.Item 1. Business" + PROSE
        + "Item 1A. Risk Factors" + RISKS
        + "Item 7. Management's Discussion and AnalysisRevenue grew. "
    )
    parts = split_items(text)

    assert parts.get("3") is None
    assert any(f.startswith("Item 3 (Legal Proceedings)") for f in parts.flags)
    assert all("Not substituted." in f for f in parts.flags if "no header found" in f)


def test_a_section_too_short_to_be_prose_is_flagged():
    text = (
        "Cover page.Item 1. BusinessWe sell software."
        + "Item 1A. Risk Factors" + RISKS
    )
    parts = split_items(text)

    short = [f for f in parts.flags if f.startswith("Item 1 is")]
    assert short and "contents table" in short[0]


def test_risk_factors_shorter_than_the_business_description_is_flagged():
    text = (
        "Cover page.Item 1. Business" + PROSE
        + "Item 1A. Risk FactorsWe may not become profitable. " * 1
        + "Item 2. PropertiesWe lease an office. "
    )
    parts = split_items(text)
    assert any("unusual for a modern filer" in f for f in parts.flags)


def test_a_split_that_missed_most_of_the_document_is_flagged():
    """Coverage is the check that catches a split which found the wrong headers."""
    text = synthetic_filing(
        preamble="Forward looking statements. " * 900,
        item_1_body="We sell software. ",
        item_1a_body="We may not become profitable. ",
    )
    parts = split_items(text)

    assert parts.covered_fraction < 0.5
    assert any("the split missed most of the filing" in f for f in parts.flags)


def test_sanity_checks_flag_and_do_not_raise():
    """A filing with one line per Item is odd, not fatal. It must still return."""
    text = "Cover.Item 1. BusinessWe sell software.Item 1A. Risk FactorsThings happen."
    parts = split_items(text)
    assert isinstance(parts, FilingSections)
    assert parts.flags
    assert isinstance(parts.business, Section)


# --------------------------------------------------------------------------- #
# Forms, lookup and caching
# --------------------------------------------------------------------------- #


def test_a_form_with_no_known_item_sequence_raises():
    with pytest.raises(ConfigError) as exc:
        split_items("Item 1. Financial Statements", form="10-Q")
    assert "part-qualified" in str(exc.value)


def test_get_tolerates_how_a_caller_names_an_item(ddog):
    for spelling in ("1a", "1A", " Item 1A ", "item 1a.", "1A"):
        assert ddog.get(spelling) is ddog.risk_factors


def test_load_sections_carries_the_filing_identity(ddog_filing):
    client = FilingTextClient(ddog_filing)
    parts = load_sections("ddog", client)

    assert parts.ticker == "DDOG"
    assert parts.accession == "0001628280-26-008819"
    assert parts.form == "10-K"
    assert parts.filed == date(2026, 2, 18)
    assert parts.period == "2025-12-31"
    assert parts.business is not None and parts.business.word_count == 5_397


def test_load_sections_reads_and_parses_a_filing_once(ddog_filing, monkeypatch):
    """Four hundred thousand characters is not something to parse twice."""
    monkeypatch.setattr(sections_module, "_LOAD_CACHE", {})
    monkeypatch.setattr(sections_module, "_SPLIT_CACHE", {})
    client = FilingTextClient(ddog_filing)

    first = load_sections("DDOG", client)
    second = load_sections("DDOG", client)

    assert client.text_calls == 1
    assert first.sections.keys() == second.sections.keys()
    assert first.business.text == second.business.text
    # Each caller gets its own lists, so one of them annotating a filing cannot
    # write into the copy the next caller reads.
    assert first.notes is not second.notes
    second.notes.append("scribbled on by a caller")
    assert "scribbled on by a caller" not in load_sections("DDOG", client).notes


def test_split_items_does_not_rescan_text_it_has_already_seen(ddog_filing, monkeypatch):
    monkeypatch.setattr(sections_module, "_SPLIT_CACHE", {})
    calls = []
    real = sections_module._candidates
    monkeypatch.setattr(
        sections_module,
        "_candidates",
        lambda text: (calls.append(len(text)), real(text))[1],
    )

    split_items(ddog_filing["text"])
    split_items(ddog_filing["text"])

    assert len(calls) == 1


def test_one_callers_notes_cannot_reach_another(ddog_filing):
    first = split_items(ddog_filing["text"])
    first.notes.append("scribbled on by a caller")
    second = split_items(ddog_filing["text"])
    assert "scribbled on by a caller" not in second.notes


def test_a_filer_with_no_such_form_raises_rather_than_substituting(ddog_filing):
    client = FilingTextClient(ddog_filing)
    with pytest.raises(MissingDataError) as exc:
        load_sections("DDOG", client, form="10-K/A")
    assert "10-K/A" in str(exc.value)
    assert client.text_calls == 0
