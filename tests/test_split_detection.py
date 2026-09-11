"""One split, seen from six filings, must stay one split.

A quarterly report shows the current period and its prior-year comparative and
nothing else, so a single four-for-one split is first reflected in one period at
the next 10-Q, in two more at the one after that, and in more again at the 10-K.
Every one of those filings looks like a fresh corporate action to a detector that
dates a split by the filing that introduced the new value, and the adjustment
factor is then multiplied by four once per filing rather than once per split.

Nvidia is the case that shows the damage, and it is real data rather than a
constructed one. Three filings first-restate periods across the 2021
four-for-one and three more across the 2024 ten-for-one, so counting filings
applies 4 cubed times 10 cubed and the trailing diluted share count comes out at
2.5 trillion against a true 24.9 billion. A market capitalisation built on that
is out by a factor of a thousand, in the same direction, at every date.

The fixture is the Nvidia fact set cut to the two weighted-average share tags
that the detector reads. Nothing else is needed to reproduce it, and cutting it
that far makes the file small enough to read.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from techval.edgar import CompanyFacts

FIXTURE = Path(__file__).parent / "fixtures" / "companyfacts_NVDA_shares.json"


@pytest.fixture
def nvda_shares() -> dict:
    return json.loads(FIXTURE.read_text())


def test_one_split_is_detected_once_however_many_filings_restate_it(nvda_shares):
    """Two corporate actions, not six sightings of two corporate actions."""
    facts = CompanyFacts(nvda_shares, "NVDA")
    assert facts._split_factors() == [
        (date(2021, 8, 20), 4.0),
        (date(2024, 8, 28), 10.0),
    ]


def test_the_note_names_both_splits_and_only_both(nvda_shares):
    note = CompanyFacts(nvda_shares, "NVDA").split_note()
    assert note is not None
    assert note.count("-for-1") == 2
    assert "4-for-1" in note and "10-for-1" in note


def test_the_share_count_lands_in_billions_not_trillions(nvda_shares):
    """The number the error is visible in, at a date inside the sample.

    On 2024-03-15 the latest statements on file end 2024-01-28 and report 2.49
    billion diluted shares in the pre-split units of the day. Today's price feed
    is retroactively split-adjusted, so the count has to be carried into today's
    units to be paired with it, which is one factor of ten and not three. The
    check is on the order of magnitude because that is where the failure lives.
    """
    facts = CompanyFacts(nvda_shares, "NVDA", knowledge_date=date(2024, 3, 15))
    rows = facts.facts("WeightedAverageNumberOfDilutedSharesOutstanding")
    quarter = max(
        (f for f in rows if f.end <= date(2024, 1, 28) and not f.is_instant),
        key=lambda f: (f.end, f.days),
    )
    assert 24.0e9 < quarter.val < 25.5e9


def test_split_units_are_deliberately_not_knowledge_dated(nvda_shares):
    """A split is a unit convention, and both sides of a ratio must share it.

    Detection reads the whole fact set rather than only the filings visible at
    the knowledge date, and that is the right behaviour rather than a leak. The
    price series a point-in-time valuation prices against is retroactively
    split-adjusted by the vendor, so a share count left in the units of its own
    day would be paired with a price in today's units and the market
    capitalisation would be out by the split factor. A split carries no
    information about a return either: it multiplies the count and divides the
    price by the same number, and the product is unchanged. The unit has to be
    consistent; only the facts have to be point in time.

    The test is that the same quarter reads the same however the client is
    pinned, which is what "unit, not information" means in practice.
    """
    early = CompanyFacts(nvda_shares, "NVDA", knowledge_date=date(2024, 3, 15))
    late = CompanyFacts(nvda_shares, "NVDA", knowledge_date=date(2026, 6, 15))

    def january_2024(facts: CompanyFacts) -> float:
        rows = facts.facts("WeightedAverageNumberOfDilutedSharesOutstanding")
        hit = [f for f in rows if f.end == date(2024, 1, 28) and not f.is_instant]
        return max(f.val for f in hit)

    assert january_2024(early) == pytest.approx(january_2024(late))


def test_a_clean_fact_set_with_no_split_reports_none():
    """The detector must not find a split in a filer that never had one."""
    payload = {
        "cik": 1,
        "entityName": "No Splits Inc",
        "facts": {
            "us-gaap": {
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "start": "2024-01-01",
                                "end": "2024-03-31",
                                "val": 100_000_000,
                                "filed": "2024-05-01",
                                "form": "10-Q",
                                "fy": 2024,
                                "fp": "Q1",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-03-31",
                                "val": 100_400_000,
                                "filed": "2025-05-01",
                                "form": "10-Q",
                                "fy": 2025,
                                "fp": "Q1",
                            },
                        ]
                    }
                }
            }
        },
    }
    assert CompanyFacts(payload, "NOSPL")._split_factors() == []


def test_two_genuine_splits_at_the_same_ratio_stay_two():
    """Clustering must not merge two real actions that happen to share a ratio.

    Their intervals are years apart and cannot intersect, which is the property
    the clustering rests on. Built by hand rather than found in a filer, because
    no company in the seed universe split two-for-one twice inside the window.
    """
    rows = []

    def quarter(start: str, end: str, val: float, filed: str) -> dict:
        return {"start": start, "end": end, "val": val, "filed": filed, "form": "10-Q"}

    # Two periods carried at 50mm, restated to 100mm in the filing of 2019-05-01,
    # then to 200mm in the filing of 2023-05-01. Two two-for-one splits.
    for start, end in (("2018-01-01", "2018-03-31"), ("2018-04-01", "2018-06-30")):
        rows.append(quarter(start, end, 50_000_000, "2018-08-01"))
        rows.append(quarter(start, end, 100_000_000, "2019-05-01"))
        rows.append(quarter(start, end, 200_000_000, "2023-05-01"))
    payload = {
        "cik": 2,
        "entityName": "Twice Split Inc",
        "facts": {
            "us-gaap": {
                "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": rows}}
            }
        },
    }
    assert CompanyFacts(payload, "TWICE")._split_factors() == [
        (date(2019, 5, 1), 2.0),
        (date(2023, 5, 1), 2.0),
    ]
