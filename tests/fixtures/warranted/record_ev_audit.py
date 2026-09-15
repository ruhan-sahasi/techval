"""Rebuild every recorded warranted observation's enterprise value with today's code.

    TECHVAL_SEC_EMAIL=you@example.com python tests/fixtures/warranted/record_ev_audit.py \\
        [payload-cache-dir]

Writes ``tests/fixtures/warranted/ev_audit.json.gz`` beside this file.

**Why it exists.** ``observations.json.gz`` was recorded before commit 3243655
fixed the debt ladder that read Verizon's long-term debt as zero, so every
enterprise value in the panel was bridged with the old ladders. Re-recording the
panel rewrites results the documents publish, and that is a decision for the
owner rather than for a script. This audit is the evidence the decision rests on:
for each recorded observation, what today's code would have bridged the same
equity value to, and whether today's code would have admitted the row at all.

**The method.** For each company-date in the panel, the companyfacts payload is
pinned to that date with ``CompanyFacts(..., knowledge_date=as_of)``, rebuilt with
``build_financials`` and bridged with ``build_ev_bridge``. The rebuilt enterprise
value is the RECORDED equity value plus today's net debt. That is the right
comparison rather than a shortcut: the equity value is a price times a share
count and neither went through a debt ladder, and net debt under the default
assumptions does not depend on the price, which the recorder checks by bridging
at two prices and refusing to write if they differ. A row ``build_financials``
refuses (today that is the debt cross-check, raising on a straight-debt figure it
cannot reconcile) is recorded as ``not_built`` with its reason. A row that builds
is then put through the panel's three row checks as they stand today, and the
first that fires is recorded under the panel's own category name. The rows the
panel skipped as ``debt_outside_the_ladder`` are rebuilt the same way and marked
``admitted`` when neither the revenue check nor the debt check fires today.

**What it does not cover.** The panel's features are not rebuilt, and
``capital_debt_to_capital`` came from the same old ladder, so a refit that swaps
only the enterprise values understates what re-recording would change. No refit
is part of this file. The float check cannot run on a skipped row, because the
panel recorded no equity value for it, so ``admitted`` means admitted by the two
checks that need no price. The revenue denominators are rebuilt and compared,
and the count that moved is recorded, but nothing else in ``build_features`` is.

**Why the payloads are optional.** A companyfacts payload runs to tens of
megabytes and the panel spans ninety-six filers, so a cache directory saves the
second run from the network. Pass one and each payload is read from
``<dir>/<TICKER>.json.gz`` when present and written there when fetched. The
payloads are not committed for the reason ``record.py`` gives for its prices.

**It is not reproducible from the network alone.** The SEC restates, and a filer
that delists loses its payload, so a rerun later may not produce the same file.
The output is committed for that reason, with the commit whose code did the
rebuild.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# The rebuild has to be THIS tree's ladders, for the reason
# ``tests/fixtures/record_tag_census.py`` gives: a plain ``python`` run otherwise
# imports whichever ``techval`` is installed, which in a worktree is another
# checkout's code.
sys.path.insert(0, str(REPO / "src"))

from techval.config import Assumptions  # noqa: E402
from techval.edgar import CompanyFacts, EdgarClient, HttpCache  # noqa: E402
from techval.errors import MissingDataError, TechvalError  # noqa: E402
from techval.ev_bridge import build_ev_bridge  # noqa: E402
from techval.financials import build_financials  # noqa: E402
from techval.ml.warranted import (  # noqa: E402
    debt_is_outside_the_ladder,
    price_disagrees_with_the_public_float,
    revenue_is_a_component,
)

PANEL = HERE / "observations.json.gz"
OUT = HERE / "ev_audit.json.gz"

# Two prices far enough apart that a price-dependent net debt (a convertible
# counted as if converted) could not hide between them.
PROBE_PRICES = (100.0, 250.0)

# The recorded figures are rounded to a thousandth of a million; so is the rebuild.
DP = 3


def _reason(exc: TechvalError) -> str:
    first = str(exc).splitlines()[0]
    hint = getattr(exc, "hint", None) if isinstance(exc, MissingDataError) else None
    return f"{type(exc).__name__}: {first}" + (f". {hint}" if hint else "")


def _commit() -> tuple[str, bool]:
    """HEAD, and whether ``src`` matches it, so the header can say whose code ran."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return head, not dirty


class Payloads:
    """Companyfacts payloads, parsed once per ticker, optionally cached on disk."""

    def __init__(self, cache_dir: Path | None) -> None:
        self.cache_dir = cache_dir
        self.client = EdgarClient(cache=HttpCache(enabled=False))
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, ticker: str) -> dict:
        path = None if self.cache_dir is None else self.cache_dir / f"{ticker}.json.gz"
        if path is not None and path.exists():
            with gzip.open(path, "rt") as handle:
                return json.load(handle)
        raw = self.client.company_facts(ticker).raw
        if path is not None:
            with gzip.open(path, "wt") as handle:
                json.dump(raw, handle)
        time.sleep(0.15)
        return raw


def main(cache_dir: Path | None) -> None:
    with gzip.open(PANEL, "rt") as handle:
        panel = json.load(handle)
    verticals = panel["sub_verticals"]
    skipped = [s for s in panel["skips"] if s["category"] == "debt_outside_the_ladder"]
    tickers = sorted({o["ticker"] for o in panel["observations"]} | {s["ticker"] for s in skipped})

    assumptions = Assumptions()
    # Pinned so the run never reaches the Treasury, as ``record.py`` pins it.
    assumptions.market.risk_free_rate = 0.0430
    commit, clean = _commit()
    payloads = Payloads(cache_dir)

    observations: list[dict] = []
    readmitted: list[dict] = []
    denominators_moved = 0
    price_dependent: list[str] = []

    for position, ticker in enumerate(tickers, 1):
        raw = payloads.get(ticker)
        mine = [o for o in panel["observations"] if o["ticker"] == ticker]
        for o in mine:
            row = {
                "ticker": ticker,
                "as_of": o["as_of"],
                "sub_vertical": o["sub_vertical"],
                "ev_recorded": o["enterprise_value"],
                "ev_rebuilt": None,
                "refused": None,
                "reason": None,
            }
            try:
                facts = CompanyFacts(raw, ticker, knowledge_date=date.fromisoformat(o["as_of"]))
                fin = build_financials(ticker, facts=facts)
                low, high = (build_ev_bridge(fin, p, assumptions) for p in PROBE_PRICES)
            except TechvalError as exc:
                row["refused"], row["reason"] = "not_built", _reason(exc)
                observations.append(row)
                continue
            if abs(low.net_debt - high.net_debt) > 1e-6:
                price_dependent.append(f"{ticker} {o['as_of']}")
            row["ev_rebuilt"] = round(o["equity_value"] + low.net_debt, DP)
            if abs(fin.revenue - o["denominator"]) > 10 ** -DP:
                denominators_moved += 1
            for reason, category in (
                (revenue_is_a_component(facts, fin), "revenue_is_a_component"),
                (debt_is_outside_the_ladder(facts, fin), "debt_outside_the_ladder"),
                (
                    price_disagrees_with_the_public_float(facts, fin, o["equity_value"]),
                    "price_disagrees_with_float",
                ),
            ):
                if reason is not None:
                    row["refused"], row["reason"] = category, reason
                    break
            observations.append(row)

        for s in (s for s in skipped if s["ticker"] == ticker):
            row = {
                "ticker": ticker,
                "as_of": s["as_of"],
                "sub_vertical": verticals.get(ticker),
                "recorded_as": s["category"],
                "today": "admitted",
                "reason": None,
            }
            try:
                facts = CompanyFacts(raw, ticker, knowledge_date=date.fromisoformat(s["as_of"]))
                fin = build_financials(ticker, facts=facts)
            except TechvalError as exc:
                row["today"], row["reason"] = "not_built", _reason(exc)
                readmitted.append(row)
                continue
            for reason, category in (
                (revenue_is_a_component(facts, fin), "revenue_is_a_component"),
                (debt_is_outside_the_ladder(facts, fin), "debt_outside_the_ladder"),
            ):
                if reason is not None:
                    row["today"], row["reason"] = category, reason
                    break
            readmitted.append(row)
        print(f"  {position}/{len(tickers)} {ticker}: {len(mine)} observations", flush=True)

    if price_dependent:
        raise SystemExit(
            "net debt depends on the price for "
            + ", ".join(price_dependent)
            + "; equity value plus net debt is not the rebuilt enterprise value for "
            "those rows, and the audit is not written"
        )

    payload = {
        "_techval_fixture": {
            "recorded": date.today().isoformat(),
            "recorder": "tests/fixtures/warranted/record_ev_audit.py",
            "endpoint": "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
            "panel": "tests/fixtures/warranted/observations.json.gz",
            "panel_recorded": panel["recorded"],
            "code_commit": commit,
            "code_matches_commit": clean,
            "method": (
                "For each observation in the panel, the filer's companyfacts payload is "
                "pinned to the observation date, rebuilt with build_financials and "
                "bridged with build_ev_bridge as they stand at code_commit, and the "
                "rebuilt enterprise value is the recorded equity value plus today's net "
                "debt. Net debt was checked not to depend on the price by bridging at "
                "two prices. A row build_financials refuses is recorded as not_built "
                "with its reason; a row that builds is put through the panel's three "
                "row checks and the first that fires is recorded under the panel's own "
                "category. The rows the panel skipped as debt_outside_the_ladder are "
                "rebuilt the same way, and today is admitted when neither the revenue "
                "check nor the debt check fires."
            ),
            "not_covered": (
                "The features are not rebuilt, and capital_debt_to_capital came from "
                "the same old debt ladder, so swapping only the enterprise values "
                "understates what re-recording the panel would change. No refit is part "
                "of this file. The float check needs an equity value the panel did not "
                "record for a skipped row, so admitted means admitted by the two checks "
                "that need no price."
            ),
        },
        "denominators_moved": denominators_moved,
        "observations": observations,
        "skipped": readmitted,
    }
    with gzip.GzipFile(OUT, "wb", mtime=0) as handle:
        handle.write(json.dumps(payload, sort_keys=True).encode("utf-8"))
    refused = sum(1 for r in observations if r["refused"])
    admitted = sum(1 for r in readmitted if r["today"] == "admitted")
    print(
        f"{len(observations)} observations ({refused} refused today), "
        f"{len(readmitted)} skipped rows ({admitted} admitted today), "
        f"{denominators_moved} denominators moved -> {OUT.relative_to(REPO)}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
