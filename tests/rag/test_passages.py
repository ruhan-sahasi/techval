"""Passages: every one is the document at its own range, and together they cover it."""

from __future__ import annotations

import pytest

from techval.rag.passages import chunk

TEXT = ("Alpha beta gamma. " * 30 + "Delta epsilon! " * 30).strip()


def test_every_passage_is_the_document_at_its_range():
    head = "HEADER " * 10
    document = head + TEXT
    for p in chunk(TEXT, accession="A", base_offset=len(head), size=200, overlap=40):
        assert document[p.start_char : p.end_char] == p.text
        assert p.id == f"A:{p.start_char}"


def test_windows_end_on_a_sentence_when_one_is_near():
    passages = chunk(TEXT, accession="A", size=200, overlap=40, snap=60)
    assert len(passages) > 2
    for p in passages[:-1]:
        assert p.text.rstrip()[-1] in ".!"


def test_the_text_is_covered_and_neighbours_overlap():
    passages = chunk(TEXT, accession="A", size=200, overlap=40)
    assert passages[0].start_char == 0 and passages[-1].end_char == len(TEXT)
    for a, b in zip(passages, passages[1:]):
        assert a.start_char < b.start_char <= a.end_char


def test_whitespace_only_windows_are_skipped():
    text = "Words here. " + " " * 500 + "More words."
    passages = chunk(text, accession="A", size=200, overlap=0, snap=50)
    assert passages and all(p.text.strip() for p in passages)
    assert passages[-1].text.strip().endswith("More words.")


@pytest.mark.parametrize("size,overlap", [(200, 200), (200, -1), (150, 10)])
def test_windows_that_cannot_advance_are_refused(size, overlap):
    with pytest.raises(ValueError):
        chunk(TEXT, accession="A", size=size, overlap=overlap)
