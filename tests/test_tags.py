"""The ladders themselves, tested against a census of what filers still use.

Every other test in this suite asks whether the engine computes the right answer
from the tags it resolved. None of them can ask the question that actually broke
Verizon, which is whether the tags it asked for still exist.

    Verizon stopped tagging ``LongTermDebtNoncurrent`` after 2013. The ladder
    went on asking for it until 2026. ``resolve_instant`` correctly refused the
    thirteen-year-old value as stale, fell through to the default of zero, and
    the bridge printed an enterprise value of 231,813mm against 375,261mm with
    no warning anywhere, because every line of it was behaving exactly as
    written.

A retired tag does not raise. It resolves to the default, and the default for a
balance-sheet concept is zero because for most filers zero is the truth. So the
failure is invisible by construction and no amount of arithmetic testing finds
it. The only thing that finds it is going and looking at what filers tag.

``tests/fixtures/tag_census.json`` is that look, recorded over the 110-name TMT
seed universe by ``tests/fixtures/record_tag_census.py``. It holds two views of
one sweep. Per concept: how many filers report it at all, how many report it
within twenty days of their own balance-sheet date, and the newest period end
anyone carries. Per filer: which concepts it reports with a material balance at
its own balance-sheet date, so a test can intersect that against the ladders as
they stand rather than against a conclusion baked in when the census was taken.

No value, no company financials, nothing that could reach a valuation. It is a
census of tags, and the tests below are the whole point of it.
"""

from __future__ import annotations

import json

import pytest

from techval import tags

from conftest import FIXTURES

CENSUS = json.loads((FIXTURES / "tag_census.json").read_text())
CONCEPTS = CENSUS["concepts"]

# The ladders whose rot prints a wrong number rather than a missing row. A
# retired concept in the revenue ladder makes the engine refuse; a retired
# concept in a debt ladder makes it print an enterprise value that is equity less
# cash and looks entirely normal. These take no exemptions.
DEBT_RESOLUTION_LADDERS = ("DEBT_NONCURRENT", "DEBT_CURRENT", "DEBT_COMBINED")

# Concepts that mean "this filer publishes a non-current borrowings line", and
# the same for a current one. Written out here rather than read off the ladders,
# because a guard that asks the ladder both halves of its own question cannot
# fail: every entry in the old non-current ladder was in live use on the day
# Verizon's long-term debt read as zero. These are the concepts the seed universe
# is observed to use for those lines, and the ladder has to reach them.
REPORTS_NONCURRENT_DEBT = frozenset(
    {
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
        "LongTermNotesPayable",
        "SeniorNotes",
        "SecuredDebt",
    }
)
REPORTS_CURRENT_DEBT = frozenset(
    {
        "DebtCurrent",
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "ShortTermBorrowings",
        "NotesPayableCurrent",
        "LinesOfCreditCurrent",
        "OtherShortTermBorrowings",
    }
)


def ladder_concepts() -> dict[str, list[str]]:
    """Every ladder in ``tags``, flattened, read off the module rather than the
    census, so a ladder added since the census was recorded shows up as missing
    rather than as absent."""
    out: dict[str, list[str]] = {}
    for name in dir(tags):
        if name.startswith("_") or not name.isupper():
            continue
        value = getattr(tags, name)
        if not isinstance(value, (list, tuple, frozenset)):
            continue
        concepts: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                concepts.append(entry)
            elif isinstance(entry, tuple):
                concepts.extend(entry)
        if concepts:
            out[name] = concepts
    return out


def every_concept() -> set[str]:
    return {c for group in ladder_concepts().values() for c in group}


# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #


def test_the_census_covers_the_universe_it_claims_to():
    """A census of four companies would pass every test below and mean nothing."""
    assert CENSUS["universe_size"] >= 100
    assert len(CENSUS["covered"]) >= 90, (
        "the census reached too few filers to say anything about which concepts "
        f"are in live use: {CENSUS['not_covered']}"
    )


def test_every_ladder_concept_appears_in_the_census():
    """A concept added to a ladder without re-recording the census is untested.

    Rerun ``tests/fixtures/record_tag_census.py`` after touching a ladder. The
    census is the evidence for the entry; adding the entry without it is the
    same guess the ladders were full of before.
    """
    unknown = sorted(every_concept() - set(CONCEPTS))
    assert not unknown, (
        "these concepts are in a ladder but not in tests/fixtures/tag_census.json, "
        f"so nothing knows whether any filer uses them: {unknown}"
    )


def test_no_ladder_names_a_concept_no_filer_has_ever_used():
    """A concept the whole universe has never reported is a guess, not a ladder.

    Distinct from the retired-tag test below. A retired concept was real once and
    is the right answer for a date in the past. A concept with no filers at all
    has never been the right answer for anybody, and its only effect is to make
    the ladder longer to read.
    """
    never = sorted(c for c in every_concept() if CONCEPTS[c]["filers"] == 0)
    assert not never, (
        "no filer in the TMT seed universe has ever reported these, so they "
        f"cannot resolve for anyone: {never}"
    )


@pytest.mark.parametrize("ladder", DEBT_RESOLUTION_LADDERS)
def test_the_debt_ladders_carry_no_retired_concept(ladder):
    """The headline guard, and the test this whole file exists for.

    Every concept in the three ladders that resolve straight debt has to be
    reported by at least one filer AT its own latest balance-sheet date. Not
    "exists in someone's fact set", which is what ``LongTermDebtNoncurrent`` did
    at Verizon for thirteen years after it stopped meaning anything: reported
    now, on a balance sheet an analyst could open.

    A failure here does not mean the concept is wrong. It means the evidence for
    it has expired, and the fix is to go and look at what the filers who used to
    use it tag instead, which is how ``LongTermDebtAndCapitalLeaseObligations``
    and ``DebtCurrent`` got here.
    """
    dead = [
        f"{c} (last reported {CONCEPTS[c]['newest_end']}, "
        f"{CONCEPTS[c]['filers']} filers ever)"
        for c in ladder_concepts()[ladder]
        if CONCEPTS[c]["live_filers"] == 0
    ]
    assert not dead, (
        f"{ladder} names concepts no filer reports at its own balance-sheet date "
        "any more. A debt ladder that resolves nothing does not refuse, it reads "
        f"zero, and the enterprise value built on it is equity less cash: {dead}"
    )


def test_the_debt_ladders_reach_the_companies_that_have_debt():
    """The coverage guard, and the one the per-concept tests cannot give.

    Every entry in the old non-current ladder except two was in live use when
    Verizon's debt read as zero: ``LongTermDebtNoncurrent`` was being reported by
    fifty filers on that same day. The entries had not rotted. The LADDER had, by
    not covering the concept Verizon had moved to, and no count of any single
    concept can see that.

    So the census also records, per filer, every concept it reports with a
    material balance at its own balance-sheet date, and this test intersects that
    against the ladders AS THEY STAND when it runs. Taking a concept out of a
    ladder therefore changes the answer here without re-recording anything, which
    is the only form of the guard that can fail on the bug it was written for.

    Measured over the census recorded on 2026-09-11: 80 filers publish a
    non-current borrowings line at their own balance-sheet date and these
    ladders reach 79 of them, and 58 publish a current one and all 58 are
    reached. The ladders as they stood before the Verizon fix reach 67 and 48
    by the same intersection. The one filer not reached is Digital Realty,
    which publishes 16,014mm of ``SeniorNotes`` and 842mm of ``SecuredDebt`` and
    no total concept of any kind, so a single-tag ladder cannot add its debt
    stack up without assuming the pieces do not overlap. It refuses instead,
    which is the right outcome and is tested in ``test_financials``.
    """
    combined = set(tags.DEBT_COMBINED)
    for leg, reported in (
        ("DEBT_NONCURRENT", REPORTS_NONCURRENT_DEBT),
        ("DEBT_CURRENT", REPORTS_CURRENT_DEBT),
    ):
        # A filer whose TOTAL resolves does not need its legs to, since the
        # combined figure replaces the split rather than joining it.
        reach = set(ladder_concepts()[leg]) | combined
        reports, unreached = 0, []
        for ticker, live in CENSUS["live_by_filer"].items():
            if not reported & set(live):
                continue
            reports += 1
            if not reach & set(live):
                unreached.append(ticker)

        assert reports >= 40, f"too few filers publish a {leg} line to measure"
        assert (reports - len(unreached)) / reports >= 0.95, (
            f"{leg} reaches too few of the filers who publish that line. These "
            "report it at their own balance-sheet date under a concept the ladder "
            f"does not carry: {sorted(unreached)}"
        )


def test_every_retired_concept_elsewhere_is_kept_on_purpose():
    """Outside the debt ladders a retired concept may be deliberate, but not silent.

    ``--as-of`` pushes a knowledge date through the engine, so a valuation dated
    2018 needs ``SalesRevenueNet`` and will find nothing else. Those entries are
    right to keep and each one carries the reason it is kept.
    """
    debt = {c for name in DEBT_RESOLUTION_LADDERS for c in ladder_concepts()[name]}
    unexplained = sorted(
        c
        for c in every_concept() - debt
        if CONCEPTS[c]["live_filers"] == 0 and c not in tags.RETIRED_BUT_KEPT
    )
    assert not unexplained, (
        "no filer reports these at its own balance-sheet date and nothing says "
        "why they are still in a ladder. Either take them out, or add them to "
        f"tags.RETIRED_BUT_KEPT with the reason: {unexplained}"
    )


def test_the_exemption_list_cannot_rot_either():
    """An exemption for a concept that is back in live use is a stale exemption.

    The list is a record of judgments, and a judgment nobody revisits is how the
    ladders got into this state in the first place.
    """
    for concept, reason in tags.RETIRED_BUT_KEPT.items():
        assert concept in every_concept(), (
            f"{concept} is exempted in tags.RETIRED_BUT_KEPT but is in no ladder"
        )
        assert len(reason) > 20, f"{concept} is exempted without a reason"
        assert CONCEPTS[concept]["live_filers"] == 0, (
            f"{concept} is back in live use at {CONCEPTS[concept]['live_filers']} "
            "filers, so its exemption is stale and should be removed"
        )


# --------------------------------------------------------------------------- #
# internal consistency of the debt lists
# --------------------------------------------------------------------------- #


def test_the_lease_inclusive_set_names_only_concepts_a_debt_ladder_resolves():
    """A typo in ``INCLUDES_FINANCE_LEASES`` disables the suppression silently.

    The set is matched against the winning tag by string equality, so a concept
    spelled wrong or dropped from the ladder never matches, the finance lease is
    added on top of a debt figure that already contains it, and the bridge
    reports the same obligation twice with no sign of it.
    """
    resolvable = {
        c for name in DEBT_RESOLUTION_LADDERS for c in ladder_concepts()[name]
    }
    stranded = sorted(tags.INCLUDES_FINANCE_LEASES - resolvable)
    assert not stranded, (
        "these are marked as carrying finance leases inside them but no debt "
        f"ladder can resolve them, so the mark can never fire: {stranded}"
    )
    stranded = sorted(tags.MAY_INCLUDE_FINANCE_LEASES - resolvable)
    assert not stranded, (
        f"unresolvable concepts in MAY_INCLUDE_FINANCE_LEASES: {stranded}"
    )


def test_the_two_lease_sets_do_not_overlap():
    """A concept is either definitively lease-inclusive or conditionally so."""
    assert not (tags.INCLUDES_FINANCE_LEASES & tags.MAY_INCLUDE_FINANCE_LEASES)


def test_the_split_ladders_share_nothing_with_each_other():
    """A concept in both the current and the non-current ladder is added twice.

    ``_straight_debt`` sums the two legs, so a concept reachable from both would
    resolve on both and double the balance it reports.
    """
    overlap = set(tags.DEBT_NONCURRENT) & set(tags.DEBT_CURRENT)
    assert not overlap, f"resolvable from both legs and so summed twice: {overlap}"


def test_only_the_ambiguous_total_sits_in_two_debt_ladders():
    """``LongTermDebt`` is deliberately in two ladders; nothing else may be.

    It is a non-current line for Adobe and Qualcomm and a total for Microsoft and
    Dell, and the two ladders read it as the two different things. Any OTHER
    concept appearing in both the non-current ladder and the combined one would
    be an accident, and would let a figure be taken as a part and as a whole
    depending on which leg happened to fail.
    """
    both = (set(tags.DEBT_NONCURRENT) | set(tags.DEBT_CURRENT)) & set(
        tags.DEBT_COMBINED
    )
    assert both == set(tags.DEBT_TOTAL_BY_DEFINITION) == {"LongTermDebt"}


def test_the_crosscheck_never_resolves_a_number():
    """The control's concepts are a net, not a source, with one deliberate overlap.

    ``DEBT_CROSSCHECK`` is read only to test a resolved figure from below. Where
    it shares a concept with a resolution ladder that is fine and expected, since
    a filer's own non-current line is a lower bound on its borrowings. What would
    not be fine is the control reading a concept whose scope it cannot trust,
    which is why ``DebtInstrumentCarryingAmount`` is in neither list.
    """
    assert "DebtInstrumentCarryingAmount" not in tags.DEBT_CROSSCHECK
    assert "DebtInstrumentCarryingAmount" not in tags.DEBT_COMBINED
    assert set(tags.DEBT_TOTALS) <= set(tags.DEBT_CROSSCHECK)
    # LongTermDebt is a total for many filers and a long-term-debt-only figure
    # for Cisco, whose commercial paper sits outside it. It can bound a
    # resolution from below and it cannot be a ceiling.
    assert "LongTermDebt" in tags.DEBT_CROSSCHECK
    assert "LongTermDebt" not in tags.DEBT_TOTALS


def test_no_ladder_repeats_a_concept():
    """A repeat inside one ladder is dead code at best and a double count at worst."""
    for name, concepts in ladder_concepts().items():
        assert len(concepts) == len(set(concepts)), f"{name} repeats a concept"
