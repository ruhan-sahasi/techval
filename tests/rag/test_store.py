"""The filing store: dated from the filers' own records, and blind to anything filed later."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from techval.errors import MissingDataError
from techval.rag.store import FilingStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DDOG_TEXT = "filing_text_DDOG_2025.json.gz"


@pytest.fixture(scope="module")
def store() -> FilingStore:
    return FilingStore.from_fixtures(FIXTURES, annual_reports=(DDOG_TEXT, "rag/not_committed.json.gz"))


def test_a_merger_document_is_dated_from_its_filer_s_submissions(store):
    doc = store.get("0000950103-26-008945")
    assert (doc.ticker, doc.form, doc.filed) == ("PAYO", "8-K", date(2026, 6, 15))
    assert doc.url == "https://www.sec.gov/Archives/edgar/data/1845815/000095010326008945/dp248400_8k.htm"
    assert doc.entity_name == "Payoneer Global Inc."
    assert doc.path == "merger/text/0000950103-26-008945.txt"


def test_nothing_filed_after_the_date_asked_about_is_served(store):
    assert [d.accession for d in store.documents("SLAB", date(2026, 3, 1))] == ["0001193125-26-036712"]
    assert [d.form for d in store.documents("slab", date(2026, 12, 31))] == ["DEFM14A", "8-K"]
    assert store.documents("SLAB", date(2026, 2, 3)) == []
    assert [d.form for d in store.documents("SLAB", date(2026, 12, 31), forms=("8-K",))] == ["8-K"]


def test_a_compressed_merger_text_reads_like_a_plain_one(store):
    doc = store.get("0001193125-26-128959")
    assert doc.path.endswith(".txt.gz")
    assert "merger" in store.text(doc).lower()


def test_an_annual_report_is_split_into_its_items(store):
    doc = store.by_path(DDOG_TEXT)
    assert (doc.ticker, doc.form, doc.accession, doc.filed) == ("DDOG", "10-K", "0001628280-26-008819", date(2026, 2, 18))
    assert doc.url == "https://www.sec.gov/Archives/edgar/data/1561550/000162828026008819/"
    text, base = store.section(doc, "7")
    assert store.text(doc)[base : base + len(text)] == text
    assert "dollar-based net retention rate" in text


def test_an_annual_report_that_is_not_committed_is_not_served(store):
    assert store.by_path("rag/not_committed.json.gz") is None


def test_an_item_the_splitter_cannot_find_is_refused(store):
    with pytest.raises(MissingDataError):
        store.section(store.by_path(DDOG_TEXT), "99")
