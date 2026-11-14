"""CompanyFacts.facts, read once per tag and never shared mutably."""

from __future__ import annotations

import json
from datetime import date

from techval.edgar import CompanyFacts

from conftest import FIXTURES

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


def _revenue_payload(series: dict[str, list[tuple[str, str, float]]]) -> dict:
    """Company facts with one (start, end, value) list per revenue tag."""
    us_gaap = {
        tag: {"units": {"USD": [{"start": s, "end": e, "val": v, "form": "10-K", "filed": e} for s, e, v in rows]}}
        for tag, rows in series.items()
    }
    return {"cik": 1, "entityName": "Revenue Tags Inc", "facts": {"us-gaap": us_gaap}}


def test_the_duration_series_follows_a_filer_that_changed_revenue_tags():
    """Nvidia's shape: the first tag on the ladder retired, the live one further down.

    Read from the first tag with any facts at all, every period stops in 2022,
    so the latest annual revenue a segment note falls back to is years stale and
    the consolidated total for a recent period is not found.
    """
    from techval import tags

    facts = CompanyFacts(
        _revenue_payload(
            {
                "RevenueFromContractWithCustomerExcludingAssessedTax": [("2021-02-01", "2022-01-30", 26.9e9)],
                "Revenues": [("2025-01-27", "2026-01-25", 130.5e9)],
            }
        ),
        "NVDA",
    )
    tag, series, _ = facts.resolve_duration_series(tags.REVENUE)
    assert tag == "Revenues"
    assert max(f.end for f in series) == date(2026, 1, 25)


def test_two_tags_running_to_the_same_end_keep_the_ladders_order():
    from techval import tags

    facts = CompanyFacts(
        _revenue_payload(
            {
                "RevenueFromContractWithCustomerExcludingAssessedTax": [("2025-01-01", "2025-12-31", 90.0)],
                "Revenues": [("2025-01-01", "2025-12-31", 100.0)],
            }
        ),
        "X",
    )
    tag, _, _ = facts.resolve_duration_series(tags.REVENUE)
    assert tag == "RevenueFromContractWithCustomerExcludingAssessedTax"
