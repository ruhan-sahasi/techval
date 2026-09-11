"""Record which us-gaap concepts the TMT universe still uses, and when.

    TECHVAL_SEC_EMAIL=you@example.com python tests/fixtures/record_tag_census.py \\
        tests/fixtures/tag_census.json

Committed beside the fixture it writes, on the same rule as
``tests/fixtures/warranted/record.py``: the file is derived rather than raw, and
the only way to audit a derived fixture is to be able to rebuild it.

**What it is for.** Every ladder in ``techval.tags`` is a bet that real filers
still use those concepts, and the bet decays. Verizon stopped tagging
``LongTermDebtNoncurrent`` in 2013 and the ladder kept asking for it until 2026,
so Verizon's long-term debt read as zero and its enterprise value printed
143 billion dollars light with nothing on the page to say so. Nothing in the
suite could have caught that, because a retired tag does not raise: it resolves
to the default and the valuation looks entirely normal.

This census is what makes it catchable. For each concept in each ladder it
records how many of the seed universe report it at all, how many report it AT
their own latest balance-sheet date, and the newest period end anyone carries.
``tests/test_tags.py`` then asserts that every concept in the ladders is still in
live use somewhere in the universe, so a concept that rots is a red test rather
than a silent zero.

**Why a census rather than the payloads.** The alternative is committing 110
companyfacts payloads, which is several gigabytes for a question that needs
three integers and a date per concept. The cost is that the census cannot be
recomputed offline, which is why the recorder is here and why the fixture
carries the date it was taken.

The census is about the CONCEPTS, not the companies: no figure in it reaches a
valuation, and no test asserts a value from it.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# The census has to describe THIS tree's ladders. ``pyproject.toml`` puts src on
# the path for pytest, but a plain ``python tests/fixtures/record_tag_census.py``
# gets whatever ``techval`` is installed or first on sys.path, which in a
# worktree is a different checkout with different ladders. Recording a census of
# somebody else's tags.py against this tree's test is worse than not recording
# one, because it passes.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from techval import tags  # noqa: E402
from techval.edgar import EdgarClient  # noqa: E402
from techval.tmt.taxonomy import SEED  # noqa: E402

# A balance-sheet fact this far from the filer's own balance-sheet date is the
# same tolerance ``CompanyFacts.resolve_instant`` applies before it calls a tag
# stale, so "live" here means exactly what "resolvable" means in the engine.
TOLERANCE_DAYS = 20

# A balance below this, in USD, is a rounding artefact rather than a line the
# ladders need to reach. Matches ``financials._MATERIAL_DEBT``.
MATERIAL = 100e6


def ladders() -> dict[str, list[str]]:
    """Every ladder in ``techval.tags``, flattened to plain concept names."""
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
            out[name] = sorted(set(concepts))
    return out


def main(out_path: str) -> None:
    client = EdgarClient()
    every = sorted({c for group in ladders().values() for c in group})

    seen: dict[str, dict] = {
        c: {"filers": 0, "live_filers": 0, "newest_end": None, "newest_filer": None}
        for c in every
    }
    covered: list[str] = []
    missing: dict[str, str] = {}
    # Per-concept counts say whether a ladder ENTRY has rotted. They cannot say
    # whether the ladder as a whole still reaches the companies, which is the
    # question Verizon actually failed: LongTermDebtNoncurrent was in live use at
    # fifty filers on the day Verizon's own long-term debt was unreachable, so
    # every entry in the ladder looked healthy and the ladder was not.
    #
    # So the per-filer view is recorded too: for each filer, which concepts it
    # reports with a material balance AT its own balance-sheet date. The test
    # intersects that against the ladders as they stand when it runs, which means
    # taking a concept OUT of a ladder changes the answer without re-recording
    # anything. That is the only form of this guard that can fail on the bug it
    # was written for.
    live_by_filer: dict[str, list[str]] = {}

    for ticker in sorted(SEED):
        try:
            facts = client.company_facts(ticker)
        except Exception as exc:  # noqa: BLE001
            missing[ticker] = f"{type(exc).__name__}: {exc}".splitlines()[0][:120]
            continue
        covered.append(ticker)

        # The filer's own balance-sheet date, on the same anchor build_financials
        # uses: the end of the most recent period for which it reported revenue.
        ends = [
            f.end
            for tag in tags.REVENUE
            for f in facts.facts(tag)
            if not f.is_instant
        ]
        as_of = max(ends) if ends else None

        live_here: set[str] = set()
        for concept in every:
            got = facts.facts(concept)
            if not got:
                continue
            row = seen[concept]
            row["filers"] += 1
            newest = max(f.end for f in got)
            if row["newest_end"] is None or newest > date.fromisoformat(
                row["newest_end"]
            ):
                row["newest_end"] = str(newest)
                row["newest_filer"] = ticker
            if as_of is not None and 0 <= (as_of - newest).days <= TOLERANCE_DAYS:
                row["live_filers"] += 1
                if max(abs(f.val) for f in got if f.end == newest) >= MATERIAL:
                    live_here.add(concept)

        live_by_filer[ticker] = sorted(live_here)

    by_ladder = defaultdict(list)
    for name, concepts in ladders().items():
        by_ladder[name] = concepts

    payload = {
        "_techval_fixture": {
            "recorded": str(date.today()),
            "recorder": "tests/fixtures/record_tag_census.py",
            "endpoint": "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
            "universe": "techval.tmt.taxonomy.SEED",
            "tolerance_days": TOLERANCE_DAYS,
            "material_usd": MATERIAL,
            "what": (
                "Two views of the same sweep. 'concepts': for each us-gaap "
                "concept named in a ladder in techval.tags, how many of the seed "
                "universe report it at all, how many report it within "
                "tolerance_days of their own balance-sheet date, and the newest "
                "period end anyone carries. 'live_by_filer': for each filer, the "
                "concepts it reports with a balance of at least material_usd at "
                "its own balance-sheet date. No figure is recorded and no test "
                "asserts one; this is a census of tags, not of companies."
            ),
        },
        "universe_size": len(SEED),
        "covered": covered,
        "not_covered": missing,
        "live_by_filer": dict(sorted(live_by_filer.items())),
        "ladders": dict(sorted(by_ladder.items())),
        "concepts": dict(sorted(seen.items())),
    }
    Path(out_path).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out_path}: {len(covered)} filers, {len(every)} concepts")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/tag_census.json")
