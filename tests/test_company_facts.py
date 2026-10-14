"""CompanyFacts.facts, read once per tag and never shared mutably."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from techval.edgar import CompanyFacts

FIXTURES = Path(__file__).parent / "fixtures"
PAYLOAD = json.loads((FIXTURES / "companyfacts_DDOG.json").read_text(encoding="utf-8"))
TAG = "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_asking_twice_gives_the_same_facts_and_a_fresh_list():
    facts = CompanyFacts(PAYLOAD, "DDOG")
    first = facts.facts(TAG)
    assert first and facts.facts(TAG) == first
    first.clear()
    # A caller that sorts or filters the list it was handed cannot change the next answer.
    assert facts.facts(TAG) and facts.facts(TAG) is not facts.facts(TAG)


def test_a_knowledge_date_still_drops_what_was_filed_later():
    full = CompanyFacts(PAYLOAD, "DDOG").facts(TAG)
    dated = CompanyFacts(PAYLOAD, "DDOG", knowledge_date=date(2024, 1, 1)).facts(TAG)
    assert dated and len(dated) < len(full)
    assert all(f.filed <= date(2024, 1, 1) for f in dated)


def test_an_unknown_tag_is_an_empty_list_every_time():
    facts = CompanyFacts(PAYLOAD, "DDOG")
    assert facts.facts("NoSuchTag") == [] and facts.facts("NoSuchTag") == []
