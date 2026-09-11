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

The second half of this file guards the *dating* of a split rather than the
counting of it. Collapsing the sightings correctly still leaves each action
bracketed inside an interval, and neither end of that interval is the ex date.
Every filing that lands inside one is then a coin toss. The oracle for this needs
no external truth at all and is at the bottom of the file: after adjustment,
every filing of the same period must agree, because they are all supposed to be
quoting the same quantity in the same unit.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import pytest

from techval.edgar import CompanyFacts

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "companyfacts_NVDA_shares.json"

SHARE_TAGS = CompanyFacts._SHARE_COUNT_TAGS


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


# --------------------------------------------------------------------------- #
# Dating a split, and the oracle that needs no external truth
# --------------------------------------------------------------------------- #
#
# Collapsing six sightings into two actions is only half the problem. Each action
# is still only known to lie inside an interval, bracketed by the last filing
# carrying old units and the first carrying new ones, and that interval is wide:
# eight months for Palo Alto, a year for CrowdStrike. A filing landing inside one
# reports whichever basis was current on the day it was made, so a threshold
# pinned to either end of the interval falls on the wrong side of some filings.
#
# Measured over these fixtures, dating at the late end misdates three of Palo
# Alto's diluted periods and dating at the early end misdates five of
# CrowdStrike's. Neither end is a fix. The threshold has to be narrowed by
# evidence instead, and these tests are how that is checked.


@pytest.fixture(scope="module")
def splitters() -> dict[str, dict]:
    """Every filer in the fixtures with a share basis worth checking.

    ``share_basis_facts.json`` carries the three weighted-average share tags for
    four filers; ``companyfacts_CRWD.json`` is a full payload. Between them:
    Nvidia's four-for-one and ten-for-one, Arista's two four-for-ones, Palo Alto's
    three-for-one and two-for-one, CrowdStrike's four-for-one, and Datadog, which
    has never split and is the control.
    """
    payloads = json.loads((FIXTURES / "share_basis_facts.json").read_text())
    payloads["CRWD"] = json.loads((FIXTURES / "companyfacts_CRWD.json").read_text())
    return payloads


def _iso(s: str) -> date:
    return date.fromisoformat(s)


def _versions_by_period(facts: CompanyFacts):
    """Every period a share tag reports more than once, with its raw filed rows.

    Raw rather than through ``facts()``, because ``facts()`` keeps one version per
    period by design and the disagreement between versions is exactly what is
    being measured.
    """
    for tag in SHARE_TAGS:
        found = facts._units(tag)
        if not found:
            continue
        unit, rows = found
        groups = defaultdict(list)
        for row in rows:
            if row.get("val") in (None, 0):
                continue
            if not row.get("end") or not row.get("filed"):
                continue
            groups[(row.get("start"), row["end"])].append(row)
        for key, versions in groups.items():
            if len(versions) >= 2:
                yield tag, unit, key, versions


def _disagreements(facts: CompanyFacts) -> tuple[int, list[str]]:
    """Periods whose filings do not agree once adjusted, and how many were checked.

    The oracle. Two filings of one quarter's share count are two statements about
    the same quantity, so once both are carried into current units they must
    match. A one percent band absorbs genuine restatement: the largest honest one
    across these filers moves 200,000 shares on 578 million, which is three
    hundredths of a percent. A misdated split moves a period by two, three, four
    or ten times, so nothing lands near the boundary and the band never needed
    tuning.

    Only the weighted-average share counts are checked, and that is deliberate.
    Diluted earnings per share moves by the same ratio and would widen the oracle,
    but it is reported to two decimals: CrowdStrike's fiscal 2026 half-year loss
    is 17 cents restated from 19, a 12 percent gap that is pure rounding. Share
    counts are large enough that a split and a restatement are three orders of
    magnitude apart, which is what makes a fixed band safe here and unsafe there.
    """
    checked = 0
    bad: list[str] = []
    for tag, unit, key, versions in _versions_by_period(facts):
        checked += 1
        adjusted = [
            facts._split_adjust(float(r["val"]), _iso(r["filed"]), unit)
            for r in versions
        ]
        if max(adjusted) / min(adjusted) > 1.01:
            trail = ", ".join(
                f"{r['filed']}: {r['val']:,.0f} -> {a:,.0f}"
                for r, a in sorted(zip(versions, adjusted), key=lambda p: p[0]["filed"])
            )
            bad.append(f"{facts.ticker} {tag} {key}: {trail}")
    return checked, bad


def test_every_filing_of_a_period_agrees_once_adjusted(splitters):
    """The whole point of the change, stated as one invariant over real data.

    Nothing external is consulted: no ex-date calendar, no vendor split file, no
    hand-entered truth. The claim is internal to ``companyfacts`` and it is the
    strongest one available. Where two filings report the same fiscal quarter's
    diluted share count, they are two statements about one quantity, and a unit
    conversion that is right for one of them is right for both.

    Before the per-filing basis map this stood at three inconsistent periods out
    of 149 on the diluted tag alone, all of them Palo Alto's, and six out of 298
    across all three weighted-average tags. It is now zero.
    """
    checked = 0
    failures: list[str] = []
    for ticker, payload in sorted(splitters.items()):
        n, bad = _disagreements(CompanyFacts(payload, ticker))
        checked += n
        failures.extend(bad)

    assert checked > 250, "the fixtures should offer a few hundred periods to check"
    assert failures == [], "\n".join(failures)


def test_the_oracle_would_have_caught_the_defect_it_was_written_for(splitters):
    """A test that cannot fail is worth nothing, so prove this one has teeth.

    The old rule dated each split at the late end of its interval. Applying that
    rule by hand to Palo Alto reproduces the inconsistent periods the change was
    written to remove, which is what makes the assertion above a measurement
    rather than a tautology.
    """
    facts = CompanyFacts(splitters["PANW"], "PANW")
    late_end = [(_iso(hi), factor) for _, hi, factor in facts._split_brackets()]
    assert late_end == [(date(2023, 5, 24), 3.0), (date(2025, 2, 14), 2.0)]

    def old_rule(value: float, filed: date, unit: str) -> float:
        factor = 1.0
        for threshold, f in late_end:
            if filed < threshold:
                factor *= f
        return value * factor if unit == "shares" else value / factor

    bad = 0
    for _, unit, _, versions in _versions_by_period(facts):
        adjusted = [old_rule(float(r["val"]), _iso(r["filed"]), unit) for r in versions]
        if max(adjusted) / min(adjusted) > 1.01:
            bad += 1
    assert bad == 6, "three diluted periods and the same three on the basic tag"


def test_narrowing_moves_palo_alto_and_leaves_everyone_else_alone(splitters):
    """Narrowing can only tighten, so a filer with no new evidence must not move.

    This is the guard against the previous attempt's failure mode. Re-dating at
    the early end of the interval repaired Palo Alto and broke CrowdStrike,
    because it changed an answer that was already right. Evidence-led narrowing
    cannot do that: a bound moves only when a filing proves it.
    """
    detected = {
        ticker: [
            (_iso(hi), factor)
            for _, hi, factor in CompanyFacts(p, ticker)._split_brackets()
        ]
        for ticker, p in splitters.items()
    }
    final = {
        ticker: CompanyFacts(p, ticker)._split_factors()
        for ticker, p in splitters.items()
    }

    assert {t for t in detected if detected[t] != final[t]} == {"PANW"}
    assert final["PANW"] == [(date(2022, 11, 18), 3.0), (date(2025, 2, 14), 2.0)]

    # CrowdStrike's interval is a year wide and its late end is the right answer.
    # Nothing in the fact set proves otherwise, so nothing moves it.
    assert final["CRWD"] == [(date(2026, 8, 27), 4.0)]
    assert final["NVDA"] == [(date(2021, 8, 20), 4.0), (date(2024, 8, 28), 10.0)]
    assert final["ANET"] == [(date(2022, 2, 15), 4.0), (date(2025, 2, 19), 4.0)]
    assert final["DDOG"] == []


def test_the_palo_alto_quarter_that_no_later_filing_repeats(splitters):
    """The worst period, and the reason anchoring per period is not enough.

    Palo Alto's quarter ended 2021-10-31 appears exactly once in the whole fact
    set, as the prior-year comparative in the 10-Q of 2022-11-18. There is no
    second version of it, so there is no ratio to read off it and nothing to
    anchor it against. Its unit can only be settled by settling the unit of the
    filing it arrived in, which a *different* period in that same filing settles.

    292.9mm shares, already on the post-three-for-one basis, times the two-for-one
    that came afterwards, is 585.8mm. The old rule multiplied by three as well and
    returned 1,757.4mm, for a company that has never had more than about 700mm
    shares outstanding.
    """
    facts = CompanyFacts(splitters["PANW"], "PANW")
    tag = "WeightedAverageNumberOfDilutedSharesOutstanding"

    raw = [
        r
        for r in facts._units(tag)[1]
        if r["end"] == "2021-10-31" and r.get("start") == "2021-08-01"
    ]
    assert len(raw) == 1, "the quarter is reported once and only once"
    assert raw[0]["filed"] == "2022-11-18"
    assert raw[0]["val"] == 292_900_000

    quarter = [
        f
        for f in facts.facts(tag)
        if f.end == date(2021, 10, 31) and f.start == date(2021, 8, 1)
    ]
    assert len(quarter) == 1
    assert quarter[0].val == pytest.approx(585_800_000)


def test_pinning_the_fact_set_does_not_move_the_unit(splitters):
    """A pinned run and a current one must agree on every share count they share.

    A split is a unit rather than information, so knowledge dating must not reach
    it. The quarter above is the sharpest case: a fact set pinned to 2022-12-31
    contains the 10-Q that reported it and none of the later filings whose
    restatements settle the surrounding periods. If narrowing depended on filings
    the pinned set cannot see, this is where it would show.

    The assertion is on the *unit* and not on the value, which is the distinction
    the whole module turns on. The two sets are allowed to disagree about a
    figure, because the pinned one is meant to: the quarter ended 2022-07-31
    reads 578.4mm pinned and 578.2mm current, which is Palo Alto restating itself
    by 200,000 shares and is exactly the point-in-time behaviour a backtest needs.
    They are not allowed to disagree by a factor of two, three or six, because
    that is a unit and a unit is not knowledge.
    """
    tag = "WeightedAverageNumberOfDilutedSharesOutstanding"
    current = CompanyFacts(splitters["PANW"], "PANW")
    pinned = CompanyFacts(splitters["PANW"], "PANW", knowledge_date=date(2022, 12, 31))

    assert pinned._split_factors() == current._split_factors()

    shared = {(f.start, f.end): f.val for f in current.facts(tag)}
    compared = 0
    restated = 0
    for fact in pinned.facts(tag):
        if (fact.start, fact.end) not in shared:
            continue
        compared += 1
        ratio = shared[(fact.start, fact.end)] / fact.val
        assert ratio == pytest.approx(1.0, rel=0.01), (fact.start, fact.end)
        if ratio != pytest.approx(1.0, rel=1e-9):
            restated += 1
    assert compared >= 10, "the two sets should share a decent number of periods"
    assert restated, "a restatement should survive, or the test proves nothing"

    # And the quarter the defect lived in is identical either way, to the share.
    def quarter(facts: CompanyFacts) -> float:
        hit = [
            f
            for f in facts.facts(tag)
            if f.end == date(2021, 10, 31) and f.start == date(2021, 8, 1)
        ]
        assert len(hit) == 1
        return hit[0].val

    assert quarter(pinned) == pytest.approx(quarter(current))
    assert quarter(pinned) == pytest.approx(585_800_000)


def test_a_filing_no_later_report_restates_falls_back_to_the_interval(splitters):
    """The honest limit of the method, asserted rather than hidden.

    Narrowing needs a later filing that repeats one of a filing's periods. The
    most recent few reports in any fact set have no such successor yet, so their
    basis cannot be read and the late end of the interval is used for them, which
    is the old behaviour. That is sound where it applies, since a split too recent
    to have been restated anywhere is also too recent to have been detected, but
    it is a fallback rather than a measurement, and it is why ``split_note`` words
    its date as a bound.

    CrowdStrike is the filer it matters for. Its basis map has to come out as one
    clean step even though three of its filings sit inside the interval with no
    evidence of their own.
    """
    facts = CompanyFacts(splitters["CRWD"], "CRWD")
    basis = facts._basis_by_filing()
    assert basis, "CrowdStrike split, so it has a basis map"

    for filed, factor in sorted(basis.items()):
        assert factor == (4.0 if _iso(filed) < date(2026, 8, 27) else 1.0), filed


def test_a_filer_with_no_split_has_no_basis_map(splitters):
    """Datadog is the control. No action, no interval, nothing to narrow."""
    facts = CompanyFacts(splitters["DDOG"], "DDOG")
    assert facts._split_brackets() == []
    assert facts._basis_by_filing() == {}
    assert facts._split_factors() == []
    assert facts.split_note() is None
    checked, bad = _disagreements(facts)
    assert checked > 0 and bad == []


def test_a_restatement_is_never_read_as_a_split():
    """The snap must refuse as readily as it accepts, or it invents corporate actions.

    Two periods restated by a genuine four-for-one, and a third that moves by
    three hundredths of a percent between two filings because the filer corrected
    itself. The split must be found and the correction must not be swept into it:
    the later filing of the corrected period survives, unmultiplied.
    """

    def row(start: str, end: str, val: float, filed: str) -> dict:
        return {"start": start, "end": end, "val": val, "filed": filed, "form": "10-Q"}

    rows = []
    for start, end in (("2023-01-01", "2023-03-31"), ("2023-04-01", "2023-06-30")):
        rows.append(row(start, end, 100_000_000, "2023-08-01"))
        rows.append(row(start, end, 400_000_000, "2024-08-01"))
    # Filed after the split and restated by a fraction of a percent.
    rows.append(row("2024-01-01", "2024-03-31", 402_000_000, "2024-11-01"))
    rows.append(row("2024-01-01", "2024-03-31", 402_120_000, "2025-11-01"))

    payload = {
        "cik": 3,
        "entityName": "One Split One Correction Inc",
        "facts": {
            "us-gaap": {
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {"shares": rows}
                }
            }
        },
    }
    facts = CompanyFacts(payload, "ONECOR")
    assert facts._split_factors() == [(date(2024, 8, 1), 4.0)]

    corrected = [
        f
        for f in facts.facts("WeightedAverageNumberOfDilutedSharesOutstanding")
        if f.end == date(2024, 3, 31)
    ]
    assert len(corrected) == 1
    assert corrected[0].val == pytest.approx(402_120_000)


def test_the_snap_is_exposed_for_callers_outside_the_fact_set(splitters):
    """``snap_to_declared_splits`` is the rounding judgement, lifted out to be reused.

    ``ml.warranted.share_basis_factor`` makes the identical decision on a ratio
    between two fact sets and currently rebuilds the suffix products itself. Both
    sides use a tolerance of 0.01, so the duplicate can delegate here without
    changing a single answer. That edit belongs to whoever owns ``warranted.py``;
    this test pins the behaviour it would be delegating to.

    Palo Alto is the case that fixes the tolerance at both ends. Its counts are
    rounded to thousands, so a genuine six-for-one arrives as 5.998 and must be
    read as six; an ordinary restatement arrives as 0.9997 and must be read as
    one, not snapped to anything else.
    """
    facts = CompanyFacts(splitters["PANW"], "PANW")
    assert facts.snap_to_declared_splits(5.998) == pytest.approx(6.0)
    assert facts.snap_to_declared_splits(0.9997) == pytest.approx(1.0)
    assert facts.snap_to_declared_splits(2.0) == pytest.approx(2.0)

    # A third is not a product of anything Palo Alto declared, and neither is
    # seven. Refusing is the behaviour the caller relies on to drop a row.
    assert facts.snap_to_declared_splits(1.0 / 3.0) is None
    assert facts.snap_to_declared_splits(7.0) is None

    # A filer that never split admits exactly one basis.
    ddog = CompanyFacts(splitters["DDOG"], "DDOG")
    assert ddog.snap_to_declared_splits(1.0) == pytest.approx(1.0)
    assert ddog.snap_to_declared_splits(4.0) is None
