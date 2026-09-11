"""Record the former-ticker index from the filings of registrants that left.

Committed beside the file it writes, for the same reason
``tests/fixtures/warranted/record.py`` is: the fixture is derived rather than
raw, and a derived fixture nobody can rebuild is an assertion rather than
evidence.

    TECHVAL_SEC_EMAIL=you@example.com \
        python tests/fixtures/former_tickers/record.py tests/fixtures/former_tickers/index.json

**What it is for.** ``https://www.sec.gov/files/company_tickers.json`` lists
companies that exist today. Every completed acquisition target is out of it by
construction, so ``EdgarClient.ticker_to_cik`` refuses SPLK, ZEN, MNDT, WORK and
TWTR, and ``techval precedents`` cannot reach a single completed deal: the
command whose entire subject is closed transactions can see none of them. The
submissions API is keyed on CIK and a delisted registrant keeps its filing
history there forever, so the only thing missing is the mapping, and the mapping
is recoverable from the filings themselves.

**Where the symbol comes from.** Since 2019 every registrant tags its own cover
page in inline XBRL, and ``dei:TradingSymbol`` on that cover page is the company
stating its own ticker in a document it signed. That is a better authority than
any third-party symbol list: it is the filer, in the filing, on the date. The
recorder reads the last periodic report each departed registrant filed and takes
the symbol out of its cover page.

**What it cannot reach, and why that is left visible.** Inline XBRL cover-page
tagging phased in over 2019 to 2021 by filer size. A registrant that stopped
filing before its first tagged cover page has no symbol to read, and this
recorder records it under ``unresolved`` with the reason rather than reaching
for a guess. Seven of the eight it misses filed their last report between
February 2019 and August 2020, which is exactly that window; the eighth,
Audacy, was filing in 2024 with no listed security, and a company with nothing
listed tags no trading symbol. The second pass looks for a literal "Trading
Symbol" label in the stripped text, which catches some of the transition filers,
and everything it still cannot read stays unresolved. A reader who needs one of
those names passes the CIK instead, which ``EdgarClient.ticker_to_cik`` accepts
as ``CIK0000790070``.

**Why two dates travel with the symbol.** A ticker is not a durable identifier.
S was Sprint's until it stopped filing in January 2020 and is SentinelOne's
today; ALTR was Altera's until 2015 and is Altair Engineering's now. The index
therefore records, per symbol, the filing that proved it and that filing's date,
which is ``through``. It proves the symbol belonged to that registrant on that
date and claims nothing about any other date, which is what lets the resolver
prefer the current ticker file for a live run and the former registrant for a
knowledge date inside its filing history.

``since`` is the other end and it is a bound rather than a sighting: the oldest
periodic report the registrant has on file, which it cannot have traded under
this symbol before. Without it the table would answer a 2014 question about ALTR
with Altair Engineering, which did not list until 2017, and be confidently
wrong about a company that existed. Where ``filings.recent`` truncates, the
bound is later than the truth and ``since_is_a_bound`` says so; the error then
costs a refusal rather than a wrong company, which is the direction to be wrong
in.

For the same reason a symbol maps to a LIST of claimants ordered by the date
each was proven, rather than to one. Aspen Technology is the case inside this
seed universe: AZPN was CIK 929940's through April 2022 and CIK 1897982's
afterwards, two registrants of the same name, and an index that kept one of them
would answer a 2021 question with a 2024 company.

**The seed universe.** The departed registrants of
``tests/fixtures/mna/universe.json``, which is the TMT universe the M&A module
was built on, plus the handful of CIKs the committed merger fixtures name that
are not in it. It is deliberately not every delisted filer on EDGAR: this engine
covers TMT, and a table of forty thousand symbols nobody in this repository will
ask for is a liability rather than an asset.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UNIVERSE = ROOT / "tests" / "fixtures" / "mna" / "universe.json"

#: CIKs the merger fixtures read that the M&A universe does not carry. Zendesk
#: is in tests/fixtures/merger as submissions_ZEN.json and companyfacts_ZEN.json
#: and is one of the four names the precedents help text offers as an example,
#: so an index that cannot resolve it leaves the documented example failing.
EXTRA: tuple[tuple[int, str], ...] = (
    (1463172, "Zendesk, Inc."),
)

#: The cover-page fact. ``ix:nonNumeric`` wraps it in the document the filer
#: actually publishes; the attribute order and the quoting vary by preparer, so
#: the pattern matches on the attribute rather than on the element's shape.
_IX_SYMBOL = re.compile(
    r"name\s*=\s*[\"']dei:TradingSymbol[\"'][^>]*>(?P<body>.{0,400}?)</\s*ix:nonNumeric\s*>",
    re.I | re.S,
)
#: The pre-inline-XBRL cover page, where the symbol is a table cell beside a
#: label rather than a tagged fact.
_LABELLED_SYMBOL = re.compile(
    r"Trading\s+Symbol\s*\(?s?\)?\s*:?\s*(?P<symbol>[A-Z][A-Z.\-]{0,5})\b"
)
_TAGS = re.compile(r"<[^>]+>")


def _user_agent() -> str:
    email = os.environ.get("TECHVAL_SEC_EMAIL")
    if not email:
        raise SystemExit("set TECHVAL_SEC_EMAIL; the SEC requires a contact address")
    return f"techval former-ticker recorder {email}"


def _get(url: str, *, first_bytes: int | None = None) -> bytes:
    headers = {"User-Agent": _user_agent()}
    if first_bytes is not None:
        headers["Range"] = f"bytes=0-{first_bytes}"
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
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


def _periodic_filings(cik: int) -> tuple[str, list[tuple[str, str, str, str]], bool]:
    """Name, periodic reports newest first, and whether the index was truncated.

    ``filings.recent`` holds the most recent thousand filings and the rest live
    in ``filings.files``. Where that happens the oldest periodic report here is
    not the oldest the registrant filed, which matters because it is what dates
    the lower bound of the symbol. The flag travels with the row so the bound
    can say it is a bound rather than a first sighting.
    """
    payload = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    filings = payload.get("filings") or {}
    recent = filings.get("recent") or {}
    rows = list(
        zip(
            recent.get("accessionNumber", []),
            recent.get("filingDate", []),
            recent.get("form", []),
            recent.get("primaryDocument", []),
        )
    )
    periodic = [r for r in rows if r[2] in ("10-K", "10-Q", "20-F", "40-F")]
    return str(payload.get("name") or ""), periodic, bool(filings.get("files"))


def _symbols(cik: int, accession: str, document: str) -> tuple[list[str], str]:
    """Trading symbols on one filing's cover page, and how they were read."""
    bare = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{bare}/{document}"
    body = _get(url, first_bytes=600_000).decode("utf-8", "replace")
    found = [
        _TAGS.sub("", m.group("body")).strip().upper()
        for m in _IX_SYMBOL.finditer(body)
    ]
    clean = [s for s in dict.fromkeys(found) if re.fullmatch(r"[A-Z][A-Z.\-]{0,5}", s)]
    if clean:
        return clean, "dei:TradingSymbol"
    flat = re.sub(r"\s+", " ", _TAGS.sub(" ", body))
    labelled = [m.group("symbol") for m in _LABELLED_SYMBOL.finditer(flat)]
    clean = list(dict.fromkeys(labelled))
    if clean:
        return clean, "cover-page label"
    return [], ""


def main(out: Path) -> None:
    universe = json.loads(UNIVERSE.read_text())
    seeds: list[tuple[int, str]] = [
        (int(row["cik"]), str(row["name"]))
        for row in universe
        if row.get("departed")
    ]
    seen = {cik for cik, _ in seeds}
    seeds.extend((cik, name) for cik, name in EXTRA if cik not in seen)
    seeds.sort()

    entries: dict[str, list[dict]] = {}
    unresolved: list[dict] = []

    for index, (cik, name) in enumerate(seeds, start=1):
        try:
            filed_name, periodic, truncated = _periodic_filings(cik)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            unresolved.append({"cik": cik, "name": name, "why": f"submissions: {exc}"})
            continue
        if not periodic:
            unresolved.append({"cik": cik, "name": name, "why": "no periodic report on file"})
            continue
        symbols: list[str] = []
        how = ""
        used: tuple[str, str, str, str] | None = None
        for row in periodic[:3]:
            try:
                symbols, how = _symbols(cik, row[0], row[3])
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                unresolved.append({"cik": cik, "name": name, "why": f"{row[0]}: {exc}"})
                symbols = []
                continue
            if symbols:
                used = row
                break
        if not symbols or used is None:
            unresolved.append(
                {
                    "cik": cik,
                    "name": name,
                    "why": "no trading symbol on the cover page of the last three "
                    f"periodic reports (newest {periodic[0][2]} {periodic[0][1]})",
                }
            )
            continue
        for symbol in symbols:
            entries.setdefault(symbol, []).append(
                {
                    "cik": cik,
                    "name": filed_name or name,
                    "since": periodic[-1][1],
                    "since_is_a_bound": truncated,
                    "through": used[1],
                    "form": used[2],
                    "accession": used[0],
                    "document": used[3],
                    "read_from": how,
                }
            )
        print(f"{index}/{len(seeds)} {cik} {name} -> {symbols} ({how})", flush=True)
        time.sleep(0.15)

    ordered = {
        symbol: sorted(claims, key=lambda claim: (claim["through"], claim["cik"]))
        for symbol, claims in sorted(entries.items())
    }
    shared = {s: c for s, c in ordered.items() if len({x["cik"] for x in c}) > 1}
    payload = {
        "_recorded": time.strftime("%Y-%m-%d"),
        "_source": "SEC EDGAR submissions API and the primary document of each "
        "registrant's last periodic report",
        "_how": "tests/fixtures/former_tickers/record.py, whose docstring carries "
        "the method and its limits",
        "_seeds": len(seeds),
        "_shape": "entries maps a symbol to the registrants proven to have traded "
        "under it, oldest proof first. 'through' is the filing date of the "
        "document the symbol was read from and is the only date the entry "
        "asserts anything about.",
        "entries": ordered,
        "unresolved": sorted(unresolved, key=lambda row: row["cik"]),
    }
    out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    print(
        f"wrote {out}: {len(ordered)} symbols over {len(seeds)} registrants, "
        f"{len(payload['unresolved'])} unresolved, "
        f"{len(shared)} symbols claimed by more than one registrant {sorted(shared)}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
