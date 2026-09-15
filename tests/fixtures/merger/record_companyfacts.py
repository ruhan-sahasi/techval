"""Re-prune the merger companyfacts fixtures against this tree's tag ladders.

    TECHVAL_SEC_EMAIL=you@example.com \\
        python tests/fixtures/merger/record_companyfacts.py --check
    TECHVAL_SEC_EMAIL=you@example.com \\
        python tests/fixtures/merger/record_companyfacts.py SPLK

Committed beside the files it writes, on the rule
``tests/fixtures/warranted/record.py`` sets: a pruned fixture is only evidence
while somebody can rebuild it from the source by the rule it claims.

**Why it exists.** Each ``companyfacts_<TICKER>.json`` here is the SEC's payload
reduced to the concepts in ``techval.tags``, and that set is not fixed. Splunk's
file was pruned before ``DebtCurrent`` joined the current-debt ladder, so it
lacked the 776.456mm of current debt Splunk tagged at 2023-07-31 in the 10-Q it
filed on 2023-08-24, four weeks before Cisco's offer. The precedent built from
the fixture printed 26,510 of enterprise value and 6.9x revenue where the same
code against the live payload prints 27,287 and 7.1x, and nothing failed,
because a concept a fixture does not carry resolves to the default exactly as a
retired one does. A ladder change is therefore a reason to run ``--check``.

**The rule.** The one ``MANIFEST.json`` records for each file, applied to
whatever ``techval.tags`` names when this runs:

* ``us-gaap`` reduced to the concepts named anywhere in ``techval.tags``;
  ``dei`` kept whole, since the payload carries only the two cover facts;
* every fact whose period ends before the entry's floor, or that was filed after
  its ceiling, dropped. The dates are read back out of the entry's own
  ``pruned`` sentence, so the manifest stays the one place they are written;
* every row key techval does not read dropped, leaving ``start``, ``end``,
  ``val``, ``filed`` and ``form``, and each concept's ``label`` and
  ``description`` with them.

Concept, unit and row order are the payload's own. ``--check`` writes nothing: it
reports, per file, the concepts the rule would add or drop and whether any row
the file already carries has changed at the source. Naming tickers rewrites
those files, their ``_fixture`` block and their manifest entries.

**What it cannot promise.** The SEC restates, so a rebuild months later may move
a figure for a reason that has nothing to do with the ladders. ``--check`` says
so separately, as rows that differ inside concepts both versions carry, so the
two causes are never mistaken for each other.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST.json"

# The ladders have to be THIS tree's. A plain ``python`` run otherwise imports
# whichever ``techval`` is installed, which in a worktree is another checkout's
# tags.py, and a fixture pruned to somebody else's ladders passes this tree's
# tests for the wrong reason.
sys.path.insert(0, str(HERE.parents[2] / "src"))

from techval import tags  # noqa: E402

ROW_KEYS = ("start", "end", "val", "filed", "form")

_WINDOW = re.compile(
    r"period ending before (?P<floor>\d{4}-\d{2}-\d{2}) or filed after (?P<ceiling>\d{4}-\d{2}-\d{2})"
)
_CIK = re.compile(r"CIK(?P<cik>\d{10})\.json$")


def concepts(module=tags) -> frozenset[str]:
    """Every concept named in ``techval.tags``, composites and exemptions included."""
    out: set[str] = set()
    for name in dir(module):
        if name.startswith("_") or not name.isupper():
            continue
        value = getattr(module, name)
        if isinstance(value, dict):
            out.update(key for key in value if isinstance(key, str))
            continue
        if not isinstance(value, (list, tuple, frozenset, set)):
            continue
        for entry in value:
            if isinstance(entry, str):
                out.add(entry)
            elif isinstance(entry, tuple):
                out.update(part for part in entry if isinstance(part, str))
    return frozenset(out)


def prune(payload: dict, wanted: frozenset[str], floor: str, ceiling: str) -> dict:
    """The payload reduced by the manifest's rule, in the payload's own order."""
    facts: dict[str, dict] = {}
    for taxonomy in ("us-gaap", "dei"):
        kept: dict[str, dict] = {}
        for concept, node in (payload.get("facts", {}).get(taxonomy) or {}).items():
            if taxonomy == "us-gaap" and concept not in wanted:
                continue
            units: dict[str, list] = {}
            for unit, rows in (node.get("units") or {}).items():
                trimmed = [
                    {key: row[key] for key in ROW_KEYS if key in row}
                    for row in rows
                    if row.get("end", "") >= floor and row.get("filed", "") <= ceiling
                ]
                if trimmed:
                    units[unit] = trimmed
            if units:
                kept[concept] = {"units": units}
        if kept:
            facts[taxonomy] = kept
    return {"cik": payload["cik"], "entityName": payload["entityName"], "facts": facts}


def _fetch(cik: str) -> dict:
    email = os.environ.get("TECHVAL_SEC_EMAIL", "").strip()
    if not email:
        raise SystemExit("set TECHVAL_SEC_EMAIL; the SEC requires a contact address")
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    request = urllib.request.Request(url, headers={"User-Agent": f"techval fixture recorder {email}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429, 500, 502, 503) and attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except OSError:
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def _rows(pruned: dict) -> dict[str, set[str]]:
    return {
        concept: {json.dumps(row, sort_keys=True) for rows in node["units"].values() for row in rows}
        for taxonomy in pruned["facts"].values()
        for concept, node in taxonomy.items()
    }


def _compare(ticker: str, committed: dict, rebuilt: dict) -> list[str]:
    old, new = _rows(committed), _rows(rebuilt)
    lines = [f"{ticker}: {len(old)} us-gaap and dei concepts committed, {len(new)} by today's rule"]
    for concept in sorted(set(new) - set(old)):
        lines.append(f"  missing  {concept} ({len(new[concept])} rows)")
    for concept in sorted(set(old) - set(new)):
        lines.append(f"  surplus  {concept} ({len(old[concept])} rows)")
    for concept in sorted(set(old) & set(new)):
        if old[concept] != new[concept]:
            lines.append(
                f"  restated {concept}: {len(new[concept] - old[concept])} rows new at the "
                f"source, {len(old[concept] - new[concept])} gone"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*", help="rewrite these files")
    parser.add_argument("--check", action="store_true", help="report every file, write nothing")
    args = parser.parse_args(argv)
    if not args.check and not args.tickers:
        parser.error("name the tickers to rewrite, or pass --check")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = {
        name[len("companyfacts_") : -len(".json")]: (name, entry)
        for name, entry in manifest["files"].items()
        if name.startswith("companyfacts_")
    }
    targets = sorted(entries) if args.check else [t.upper() for t in args.tickers]
    unknown = [t for t in targets if t not in entries]
    if unknown:
        parser.error(f"no companyfacts entry in {MANIFEST.name} for {', '.join(unknown)}")

    wanted = concepts()
    today = date.today().isoformat()
    for ticker in targets:
        name, entry = entries[ticker]
        window = _WINDOW.search(entry["pruned"])
        cik = _CIK.search(entry["source"])
        if window is None or cik is None:
            raise SystemExit(f"{name}: the manifest entry does not state a CIK and a date window")
        payload = _fetch(cik.group("cik"))
        rebuilt = prune(payload, wanted, window.group("floor"), window.group("ceiling"))
        path = HERE / name
        committed = json.loads(path.read_text(encoding="utf-8"))
        print("\n".join(_compare(ticker, committed, rebuilt)), flush=True)
        if not args.check:
            document = {
                "_fixture": {"source": entry["source"], "retrieved": today, "pruned": entry["pruned"]},
                **rebuilt,
            }
            body = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
            path.write_text(body, encoding="utf-8")
            entry["retrieved"] = today
            entry["tags"] = len(rebuilt["facts"].get("us-gaap", {}))
            entry["bytes"] = len(body.encode("utf-8"))
            print(f"  wrote {name}: {entry['tags']} us-gaap concepts, {entry['bytes']:,} bytes")
        time.sleep(0.2)

    if not args.check:
        MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
