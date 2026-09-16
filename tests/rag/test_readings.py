"""The reading value every reader returns, and the settings that shape a reading."""

from __future__ import annotations

import pytest

from techval.config import Assumptions
from techval.errors import ConfigError
from techval.rag.readings import READERS, STATUSES, UNITS, Reading


def test_the_rag_settings_default_to_the_spec():
    rag = Assumptions().ml.rag
    assert (rag.model, rag.passage_chars, rag.overlap_chars, rag.top_k, rag.max_tokens) == (
        "claude-opus-5",
        1200,
        200,
        6,
        16000,
    )


def test_an_unknown_rag_setting_is_refused(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("ml:\n  rag:\n    temperature: 0.2\n")
    with pytest.raises(ConfigError):
        Assumptions.load(path)


def test_a_reading_names_a_known_reader_and_status():
    with pytest.raises(ValueError):
        Reading("T", "gpt", "stated")
    with pytest.raises(ValueError):
        Reading("T", "claude", "maybe")


def test_refusing_keeps_the_reading_and_states_why():
    reading = Reading("T", "claude", "stated", value=1.0, unit="count", quote="1 customer")
    refused = reading.refuse("the quote is not in the passage it cites")
    assert refused.status == "refused" and refused.value == 1.0
    assert refused.reason == "the quote is not in the passage it cites"
    assert not refused.answered and reading.answered


def test_the_vocabularies_are_the_spec_s():
    assert READERS == ("claude", "regex_retrieved", "regex_native")
    assert STATUSES == ("stated", "not_stated", "ambiguous", "refused", "missing")
    assert "usd_per_share" in UNITS and "percent" in UNITS
