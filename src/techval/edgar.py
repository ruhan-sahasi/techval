"""SEC EDGAR client and XBRL fact algebra.

Two jobs. First, fetch company facts politely: the SEC's fair-access policy asks
for a declared identity and a request rate no higher than ten per second, and it
enforces both. Second, turn the ``companyfacts`` blob into something a statement
can be built from, which is harder than it looks and is where most naive
implementations go quietly wrong.

Three traps this module exists to avoid, each observed in real filer data:

*Retired tags.* A tag that appears anywhere in a company's history keeps
appearing in ``companyfacts`` forever. CrowdStrike's ``ShortTermInvestments``
last carried a value in early 2025 and now reads zero; HubSpot's
``ConvertibleLongTermNotesPayable`` stops in 2021. Resolving a balance-sheet
concept by "first tag present" hands back a years-stale number with no warning.
Balance-sheet resolution here is therefore *as of a date*: a tag only wins if it
carries a fact at the date being asked about.

*Dimensioned facts are absent.* ``companyfacts`` publishes only undimensioned
facts. A dual-class issuer tags shares outstanding by class, so
``dei:EntityCommonStockSharesOutstanding`` simply does not exist for them --
Datadog is one. Period-end share counts cannot be relied on from this endpoint.

*Quarters are not always reported.* Q4 is never a filed period; it is the annual
figure less nine months. Snowflake tags depreciation only cumulatively, so no
discrete quarter exists at all. CrowdStrike files a year-to-date half and a
discrete Q2, leaving Q1 to be recovered by subtraction. Summing "the last four
quarterly values" finds nothing for these companies. The fact algebra below
instead derives every period it can by subtracting nested periods that share an
endpoint, then covers the trailing-twelve-month window exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import requests

from .errors import DataSourceError, MissingDataError, StaleDataError

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

CACHE_ROOT = Path(os.environ.get("TECHVAL_CACHE", Path.home() / ".techval" / "cache"))

# The SEC permits ten requests a second. Eight leaves headroom and still warms a
# full peer set in well under a minute.
_MAX_REQUESTS_PER_SEC = 8.0


def _user_agent() -> str:
    """SEC fair access requires a declared identity. Refuse to run without one."""
    email = os.environ.get("TECHVAL_SEC_EMAIL", "").strip()
    if not email or "@" not in email:
        raise DataSourceError(
            "TECHVAL_SEC_EMAIL is not set to a real address.\n"
            "  The SEC's fair-access policy requires every automated request to "
            "declare a contact, and blocks clients that do not.\n"
            "  Set it before running, e.g.  export TECHVAL_SEC_EMAIL=you@example.com"
        )
    return f"techval/0.1 ({email})"


class _Throttle:
    """Process-wide minimum spacing between outbound SEC requests."""

    def __init__(self, per_second: float) -> None:
        self._min_gap = 1.0 / per_second
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self._min_gap:
                time.sleep(self._min_gap - gap)
            self._last = time.monotonic()


_THROTTLE = _Throttle(_MAX_REQUESTS_PER_SEC)


# --------------------------------------------------------------------------- #
# HTTP with an on-disk cache keyed by URL
# --------------------------------------------------------------------------- #


class HttpCache:
    """Content cache keyed by request URL.

    Determinism is the point: the same ticker against the same cached filings
    must produce byte-identical valuations, so that a number in a memo can be
    reproduced later.
    """

    def __init__(self, root: Path = CACHE_ROOT, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.root / f"{digest}.cache"

    def get(self, url: str) -> bytes | None:
        if not self.enabled:
            return None
        p = self._path(url)
        return p.read_bytes() if p.exists() else None

    def put(self, url: str, body: bytes) -> None:
        if not self.enabled:
            return
        tmp = self._path(url).with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(self._path(url))


def http_get(
    url: str,
    *,
    cache: HttpCache,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
    max_retries: int = 5,
) -> bytes:
    """GET with cache, throttling and exponential backoff on 429/503."""
    cached = cache.get(url)
    if cached is not None:
        return cached

    hdrs = {"Accept-Encoding": "gzip, deflate"}
    hdrs.update(headers or {})

    delay = 1.0
    last_status: int | None = None
    for attempt in range(max_retries):
        if "sec.gov" in url:
            _THROTTLE.wait()
        try:
            resp = requests.get(url, headers=hdrs, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise DataSourceError(f"network failure for {url}: {exc}") from exc
            time.sleep(delay)
            delay *= 2
            continue

        last_status = resp.status_code
        if resp.status_code == 200:
            cache.put(url, resp.content)
            return resp.content
        # 429 rate limited, 503 briefly unavailable: both are worth waiting out.
        if resp.status_code in (429, 503) and attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        break

    raise DataSourceError(
        f"request for {url} failed with HTTP {last_status} after {max_retries} attempts"
    )


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Fact:
    """One XBRL fact, already reduced to what a statement needs."""

    tag: str
    val: float
    end: date
    start: date | None
    form: str
    filed: date
    unit: str
    derived_from: str | None = None

    @property
    def is_instant(self) -> bool:
        return self.start is None

    @property
    def days(self) -> int:
        if self.start is None:
            return 0
        return (self.end - self.start).days + 1


def _d(s: str) -> date:
    y, m, dd = s.split("-")
    return date(int(y), int(m), int(dd))


def _neg_date(iso: str) -> tuple[int, ...]:
    """Sort key that orders ISO dates newest-first inside an ascending compare."""
    try:
        return tuple(-int(part) for part in iso.split("-"))
    except ValueError:
        return (0, 0, 0)


@dataclass
class Provenance:
    """Where a single reported figure came from.

    Carried alongside every line item so that any number in the output can be
    walked back to the tag, the filing and the arithmetic that produced it.
    """

    concept: str
    tag: str | None
    method: str
    periods: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)
    filed: str | None = None
    note: str | None = None
    tags_tried: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, str]:
        return {
            "concept": self.concept,
            "tag": self.tag or "-",
            "method": self.method,
            "period": ", ".join(self.periods) if self.periods else "-",
            "form": ", ".join(sorted(set(self.forms))) if self.forms else "-",
            "filed": self.filed or "-",
            "note": self.note or "",
        }


class CompanyFacts:
    """The ``companyfacts`` payload for one filer, indexed for lookup."""

    def __init__(self, payload: dict[str, Any], ticker: str) -> None:
        self.raw = payload
        self.ticker = ticker.upper()
        self.entity_name: str = payload.get("entityName", ticker)
        self.cik: int = int(payload.get("cik", 0))
        self._gaap: dict[str, Any] = payload.get("facts", {}).get("us-gaap", {})
        self._dei: dict[str, Any] = payload.get("facts", {}).get("dei", {})
        self._splits: list[tuple[date, float]] | None = None

    # -- raw tag access ---------------------------------------------------- #

    def has(self, tag: str) -> bool:
        return tag in self._gaap or tag in self._dei

    def _units(self, tag: str) -> tuple[str, list[dict]] | None:
        node = self._gaap.get(tag) or self._dei.get(tag)
        if not node:
            return None
        units = node.get("units", {})
        for preferred in ("USD", "shares", "USD/shares", "pure"):
            if preferred in units:
                return preferred, units[preferred]
        if not units:
            return None
        first = next(iter(units))
        return first, units[first]

    # -- stock splits ------------------------------------------------------ #

    # Ratios a board actually declares. A restatement moves a number by a few
    # percent; a split moves it by exactly one of these.
    _SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 1.5, 2.5, 0.5, 0.1, 0.05)

    def _split_factors(self) -> list[tuple[date, float]]:
        """Detect stock splits from the filings themselves.

        A split makes every historical share count and per-share figure in
        ``companyfacts`` inconsistent with every later one. Filings after the
        split restate the comparatives they happen to show, but earlier periods
        that no later filing repeats keep their pre-split values forever. A
        trailing twelve months built across that boundary silently mixes units.
        CrowdStrike split four-for-one in mid-2026 and its fact set shows exactly
        this: the same quarter carries 249.9mm shares as filed in 2025 and
        999.6mm as restated in 2026.

        Detection needs no external data. Where one period carries two filed
        values whose ratio is within half a percent of a ratio a board would
        actually declare, that is a split, and the filing that introduced the new
        value dates it. Ordinary restatements are excluded by the tightness of
        that test.

        Returns ``(filed_on_or_after, factor)`` pairs: any fact filed strictly
        before that date must be multiplied by ``factor`` to be comparable.
        """
        if self._splits is not None:
            return self._splits

        votes: dict[tuple[str, float], set[tuple[str | None, str]]] = {}
        for tag in (
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
            "WeightedAverageNumberOfSharesOutstanding",
        ):
            found = self._units(tag)
            if not found:
                continue
            _, rows = found
            groups: dict[tuple[str | None, str], list[dict]] = {}
            for r in rows:
                if r.get("val") in (None, 0) or not r.get("end"):
                    continue
                groups.setdefault((r.get("start"), r["end"]), []).append(r)

            for versions in groups.values():
                versions.sort(key=lambda r: r.get("filed") or "")
                for older, newer in zip(versions, versions[1:]):
                    ratio = float(newer["val"]) / float(older["val"])
                    for nice in self._SPLIT_RATIOS:
                        if abs(ratio - nice) < 0.005 * nice:
                            key = (newer["filed"], nice)
                            # Keyed by period, not by observation. Diluted and
                            # basic shares both report the same quarter, so
                            # counting observations would let a single period
                            # clear a safeguard meant to require two.
                            votes.setdefault(key, set()).add(
                                (newer.get("start"), newer["end"])
                            )
                            break

        # One agreeing period could be a typo in a single tagged fact. Two or
        # more distinct periods moving by the identical ratio in the identical
        # filing is a corporate action.
        self._splits = sorted(
            (_d(filed), factor)
            for (filed, factor), periods in votes.items()
            if len(periods) >= 2
        )
        return self._splits

    def split_note(self) -> str | None:
        splits = self._split_factors()
        if not splits:
            return None
        return "; ".join(
            f"{f:g}-for-1 share split first reflected in the filing of {d}"
            for d, f in splits
        )

    def _split_adjust(self, value: float, filed: date, unit: str) -> float:
        factor = 1.0
        for threshold, f in self._split_factors():
            if filed < threshold:
                factor *= f
        if factor == 1.0:
            return value
        # Share counts scale up; per-share figures scale down by the same ratio.
        return value * factor if unit == "shares" else value / factor

    # A period can be reported by an audited financial statement and, later, by a
    # proxy. They disagree: CrowdStrike's fiscal 2024 net income is 72.2mm in the
    # 10-K that restated it and 73.4mm in the proxy filed six weeks afterwards.
    # The periodic report is the audited statement and wins on class, before
    # filing date is consulted at all.
    _FORM_RANK = {
        "10-K": 0, "10-K/A": 0, "10-KT": 0,
        "10-Q": 0, "10-Q/A": 0, "10-QT": 0,
        "20-F": 0, "20-F/A": 0, "40-F": 0,
        "8-K": 1, "8-K/A": 1,
    }

    @classmethod
    def _rank(cls, form: str) -> int:
        # Everything else (DEF 14A, PRE 14A, S-1, ARS) ranks last.
        return cls._FORM_RANK.get(form or "", 2)

    def facts(self, tag: str) -> list[Fact]:
        """Deduplicated, split-adjusted facts for one tag.

        A period is reported many times: once when filed, again as the prior-year
        comparative in later filings, again in an amendment, and sometimes again
        in a proxy statement. Those values differ, so two rules pick between them.

        First, an audited periodic report beats anything else, whenever it was
        filed. A proxy summarises the financials, it does not restate them, and
        letting one override a 10-K because it happened to be filed later swaps an
        audited figure for a summary of it.

        Second, within the same class the most recently filed value wins, because
        that is the genuine restatement and the number the company now stands
        behind. CrowdStrike's fiscal 2024 net income moved from 89.3mm to 72.2mm
        between two 10-Ks, and the later one is right.

        Share and per-share facts are restated into current units first, so that
        deduplication compares like with like across a split.
        """
        found = self._units(tag)
        if not found:
            return []
        unit, rows = found
        adjust = unit in ("shares", "USD/shares")

        best: dict[tuple[str | None, str], dict] = {}
        for r in rows:
            end = r.get("end")
            if not end or r.get("val") is None:
                continue
            key = (r.get("start"), end)
            prev = best.get(key)
            if prev is None or (
                self._rank(r.get("form", "")),
                # Negated so that a lower tuple wins on rank first, then on the
                # latest filing date within that rank.
                _neg_date(r.get("filed") or ""),
            ) < (
                self._rank(prev.get("form", "")),
                _neg_date(prev.get("filed") or ""),
            ):
                best[key] = r

        out = []
        for r in best.values():
            filed = _d(r["filed"]) if r.get("filed") else _d(r["end"])
            val = float(r["val"])
            if adjust:
                val = self._split_adjust(val, filed, unit)
            out.append(
                Fact(
                    tag=tag,
                    val=val,
                    end=_d(r["end"]),
                    start=_d(r["start"]) if r.get("start") else None,
                    form=r.get("form", "?"),
                    filed=filed,
                    unit=unit,
                )
            )
        out.sort(key=lambda f: (f.end, f.days))
        return out

    # -- resolution through a fallback ladder ------------------------------ #

    def resolve_instant(
        self,
        concept: str,
        ladder: Iterable[str],
        as_of: date,
        *,
        tolerance_days: int = 20,
        required: bool = True,
        default_when_absent: float | None = None,
    ) -> tuple[float | None, Provenance]:
        """Resolve a balance-sheet concept *at a date*.

        A tag only wins if it carries a fact within ``tolerance_days`` of the
        balance-sheet date. This is the guard against retired tags: a ladder entry
        whose newest fact is three years old is skipped rather than believed.

        ``default_when_absent`` distinguishes the two reasons a concept can be
        missing. Non-controlling interest is absent from a filer's fact set
        because the filer has none, and zero is the correct reading. Revenue is
        never legitimately absent, so its absence is an error. Only concepts of
        the first kind pass a default.
        """
        ladder = list(ladder)
        stale: list[str] = []
        zero_hit: tuple[float, Provenance] | None = None
        for tag in ladder:
            facts = [f for f in self.facts(tag) if f.is_instant and f.end <= as_of]
            if not facts:
                continue
            newest = facts[-1]
            if (as_of - newest.end).days > tolerance_days:
                stale.append(f"{tag} (newest {newest.end})")
                continue
            prov = Provenance(
                concept=concept,
                tag=tag,
                method="balance sheet, latest instant",
                periods=[str(newest.end)],
                forms=[newest.form],
                filed=str(newest.filed),
                tags_tried=ladder,
            )
            # An explicit zero is kept as a candidate but does not end the
            # search. A filer can report zero under the ladder's first tag while
            # carrying the real balance under a later one, and stopping at the
            # zero would defeat the retired-tag defence in the one case it is
            # for. If every current tag reads zero, zero is the answer, sourced
            # from the first of them.
            if newest.val == 0 and zero_hit is None:
                zero_hit = (newest.val, prov)
                continue
            if newest.val != 0:
                if zero_hit is not None:
                    prov.note = (
                        f"{zero_hit[1].tag} reports an explicit zero at the same "
                        "date; the non-zero balance below it in the ladder is taken"
                    )
                return newest.val, prov
        if zero_hit is not None:
            return zero_hit

        if default_when_absent is not None:
            note = "no tag reports this; read as zero"
            if stale:
                note = f"only stale tags found ({'; '.join(stale)}); read as zero"
            return default_when_absent, Provenance(
                concept=concept,
                tag=None,
                method="absent, defaulted",
                note=note,
                tags_tried=ladder,
            )

        if not required:
            return None, Provenance(
                concept=concept, tag=None, method="unavailable", tags_tried=ladder
            )

        if stale:
            raise StaleDataError(
                concept,
                ticker=self.ticker,
                tags_tried=ladder,
                period=f"as of {as_of}",
                hint=(
                    "tags exist but carry only stale values: "
                    + "; ".join(stale)
                    + f". Nothing is reported within {tolerance_days} days of the "
                    "balance-sheet date, so the filer has retired these tags."
                ),
            )
        raise MissingDataError(
            concept,
            ticker=self.ticker,
            tags_tried=ladder,
            period=f"as of {as_of}",
            hint="no us-gaap tag in the ladder appears in this filer's company facts",
        )

    def resolve_duration_series(
        self, concept: str, ladder: Iterable[str]
    ) -> tuple[str | None, list[Fact], list[str]]:
        """First ladder tag that carries any duration facts, with its series."""
        ladder = list(ladder)
        for tag in ladder:
            facts = [f for f in self.facts(tag) if not f.is_instant]
            if facts:
                return tag, facts, ladder
        return None, [], ladder

    def _composite_series(self, parts: tuple[str, ...]) -> list[Fact]:
        """Sum several tags over identical periods.

        Some filers never publish a combined line and tag the components instead.
        A composite ladder entry adds them, but only across periods where every
        component is present, so a partial sum is never passed off as a total.

        That restraint means the composite can decline to fire. CrowdStrike tags
        amortisation of intangibles and no depreciation line whatever, so nothing
        can be summed and its EBITDA is simply unavailable. Reporting that is the
        correct outcome; the alternative is an EBITDA missing its depreciation.
        """
        series = [
            {(f.start, f.end): f for f in self.facts(p) if not f.is_instant}
            for p in parts
        ]
        if not series or any(not s for s in series):
            return []
        shared = set(series[0])
        for s in series[1:]:
            shared &= set(s)
        out = []
        for key in shared:
            first = series[0][key]
            out.append(
                Fact(
                    tag=" + ".join(parts),
                    val=sum(s[key].val for s in series),
                    start=first.start,
                    end=first.end,
                    form=first.form,
                    filed=max(s[key].filed for s in series),
                    unit=first.unit,
                )
            )
        return sorted(out, key=lambda f: (f.start or f.end, f.end))

    def resolve_ttm(
        self,
        concept: str,
        ladder: Iterable[str | tuple[str, ...]],
        as_of: date,
        *,
        kind: str = "flow",
        required: bool = True,
        allow_annual_fallback: bool = False,
        default_when_absent: float | None = None,
    ) -> tuple[float | None, Provenance]:
        """Resolve a flow concept over the trailing twelve months.

        A ladder entry only wins if it can actually cover the window. A tag that
        exists but whose facts stop two years ago is skipped, exactly as for
        balance-sheet concepts -- otherwise the first tag with any history at all
        wins and the concept silently fails. Entries may be a single tag or a
        tuple of tags to be summed.
        """
        ladder = list(ladder)
        names = [t if isinstance(t, str) else " + ".join(t) for t in ladder]
        partial: list[str] = []

        for entry, name in zip(ladder, names):
            facts = (
                [f for f in self.facts(entry) if not f.is_instant]
                if isinstance(entry, str)
                else self._composite_series(entry)
            )
            if not facts:
                continue
            got = trailing_twelve_months(facts, as_of, kind=kind)
            if got is not None:
                value, tiles, method = got
                return value, Provenance(
                    concept=concept,
                    tag=name,
                    method=method,
                    periods=[f"{t.start}..{t.end}" for t in tiles],
                    forms=[t.form for t in tiles],
                    filed=str(max(t.filed for t in tiles)),
                    tags_tried=names,
                    note=(
                        "day-weighted average, not a sum" if kind == "average" else None
                    ),
                )
            partial.append(name)

        if allow_annual_fallback:
            for entry, name in zip(ladder, names):
                facts = (
                    [f for f in self.facts(entry) if not f.is_instant]
                    if isinstance(entry, str)
                    else self._composite_series(entry)
                )
                fy = latest_annual(facts, as_of) if facts else None
                if fy is not None:
                    return fy.val, Provenance(
                        concept=concept,
                        tag=name,
                        method="latest filed fiscal year, not trailing twelve months",
                        periods=[f"{fy.start}..{fy.end}"],
                        forms=[fy.form],
                        filed=str(fy.filed),
                        tags_tried=names,
                        note=(
                            "the filer reports this only annually, so the twelve "
                            "months ending on the latest balance-sheet date cannot "
                            "be constructed; this is the most recent full year"
                        ),
                    )

        if default_when_absent is not None:
            return default_when_absent, Provenance(
                concept=concept,
                tag=None,
                method="absent, defaulted",
                note="no tag reports this; read as zero",
                tags_tried=names,
            )
        if not required:
            return None, Provenance(
                concept=concept, tag=None, method="unavailable", tags_tried=names
            )

        hint = "no us-gaap tag in the ladder appears in this filer's company facts"
        if partial:
            hint = (
                "these tags exist but do not report enough periods to tile a "
                f"trailing twelve months: {', '.join(partial)}"
            )
        raise MissingDataError(
            concept,
            ticker=self.ticker,
            tags_tried=names,
            period=f"twelve months ended {as_of}",
            hint=hint,
        )


# --------------------------------------------------------------------------- #
# Period algebra
# --------------------------------------------------------------------------- #


def _to_integral(facts: list[Fact]) -> list[Fact]:
    """Convert period averages into period integrals so they can be added.

    A weighted-average share count is not additive: subtracting a first-quarter
    average from a half-year average is meaningless arithmetic. The *integral* --
    average times days, the sum of the daily share count over the window -- is
    additive, and reduces back to an average by dividing through by total days at
    the end. Every period rule below then applies to averages unchanged.
    """
    return [
        Fact(
            tag=f.tag,
            val=f.val * f.days,
            start=f.start,
            end=f.end,
            form=f.form,
            filed=f.filed,
            unit=f.unit,
            derived_from=f.derived_from,
        )
        for f in facts
        if f.start is not None
    ]


def derive_periods(facts: list[Fact], passes: int = 3) -> list[Fact]:
    """Expand a series with every period recoverable by subtraction.

    Filers report overlapping windows. If a nine-month and a six-month figure
    share a start date, their difference is the third quarter. If a half-year and
    a discrete Q2 share an end date, their difference is Q1. Applying both rules
    repeatedly recovers the discrete quarters that were never filed, including
    the fourth, which is by construction never a reported period.

    Derived values inherit the filing date of the wider period, since that is the
    filing whose arithmetic produced them.
    """
    known: dict[tuple[date, date], Fact] = {}
    for f in facts:
        if f.start is None:
            continue
        known.setdefault((f.start, f.end), f)

    for _ in range(passes):
        added = False
        current = list(known.values())
        for big in current:
            for small in current:
                if big is small or small.start is None or big.start is None:
                    continue
                if small.days >= big.days:
                    continue
                # Shared start: the tail is the remainder.
                if small.start == big.start and small.end < big.end:
                    key = (small.end + timedelta(days=1), big.end)
                # Shared end: the head is the remainder.
                elif small.end == big.end and small.start > big.start:
                    key = (big.start, small.start - timedelta(days=1))
                else:
                    continue
                if key in known:
                    continue
                known[key] = Fact(
                    tag=big.tag,
                    val=big.val - small.val,
                    start=key[0],
                    end=key[1],
                    form=big.form,
                    filed=max(big.filed, small.filed),
                    unit=big.unit,
                    derived_from=(
                        f"{big.start}..{big.end} less {small.start}..{small.end}"
                    ),
                )
                added = True
        if not added:
            break

    return sorted(known.values(), key=lambda f: (f.start or f.end, f.end))


def cover_window(
    facts: list[Fact], window_start: date, window_end: date, *, slack_days: int = 8
) -> list[Fact] | None:
    """Tile ``[window_start, window_end]`` with contiguous, non-overlapping periods.

    Returns the tiling, longest pieces first where there is a choice, or None if
    the window cannot be covered. Tiles are contiguous by construction: each one
    begins the day after the last ended.

    ``slack_days`` absorbs the fact that a fiscal quarter is thirteen weeks
    rather than a calendar quarter, so four of them do not land precisely on the
    anniversary of the start date. Eight days rather than four, because a 53-week
    fiscal year inserts a whole extra week and the quarters that follow one are
    offset by seven days, not by the day or two a 52-week year drifts.
    """
    by_start: dict[date, list[Fact]] = {}
    for f in facts:
        if f.start is None:
            continue
        by_start.setdefault(f.start, []).append(f)
    for lst in by_start.values():
        lst.sort(key=lambda f: -f.days)

    starts = sorted(by_start)

    def _walk(cursor: date, depth: int) -> list[Fact] | None:
        if abs((cursor - window_end).days) <= 1:
            return []
        if cursor > window_end or depth > 8:
            return None
        # Allow the first tile to begin a few days either side of the requested
        # start, so a 13-week fiscal quarter lines up.
        candidates: list[Fact] = []
        for s in starts:
            if abs((s - cursor).days) <= (slack_days if depth == 0 else 1):
                candidates.extend(by_start[s])
        for f in candidates:
            if f.end > window_end + timedelta(days=slack_days):
                continue
            rest = _walk(f.end + timedelta(days=1), depth + 1)
            if rest is not None:
                return [f, *rest]
        return None

    return _walk(window_start, 0)


def trailing_twelve_months(
    facts: list[Fact], as_of: date, *, kind: str = "flow"
) -> tuple[float, list[Fact], str] | None:
    """Aggregate a concept over the twelve months ending ``as_of``.

    ``kind`` is ``"flow"`` for anything that accumulates -- revenue, expense,
    capital spending -- and ``"average"`` for a period-average such as weighted
    average shares outstanding, which is day-weighted rather than summed.

    Returns the value, the periods it was built from, and a description of the
    method, or None when the window cannot be tiled from what the filer reported.
    """
    working = _to_integral(facts) if kind == "average" else facts
    expanded = derive_periods(working)

    def _finish(tiles: list[Fact], method: str) -> tuple[float, list[Fact], str]:
        total = sum(t.val for t in tiles)
        if kind == "average":
            days = sum(t.days for t in tiles)
            return (total / days if days else 0.0), tiles, method
        return total, tiles, method

    # A filed annual period ending exactly here is the cleanest answer available.
    for f in expanded:
        if f.end == as_of and 350 <= f.days <= 380 and f.derived_from is None:
            return _finish([f], "single filed annual period")

    window_start = as_of - timedelta(days=364)
    tiles = cover_window(expanded, window_start, as_of)
    if tiles is None:
        return None
    derived = any(t.derived_from for t in tiles)
    method = (
        f"{len(tiles)} periods tiled to twelve months"
        + (", some recovered by subtraction" if derived else "")
    )
    return _finish(tiles, method)


def latest_annual(facts: list[Fact], as_of: date) -> Fact | None:
    """Most recent filed annual period ending on or before ``as_of``."""
    annuals = [
        f
        for f in facts
        if f.start is not None and 350 <= f.days <= 380 and f.end <= as_of
    ]
    return max(annuals, key=lambda f: f.end) if annuals else None


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class EdgarClient:
    def __init__(self, cache: HttpCache | None = None) -> None:
        self.cache = cache if cache is not None else HttpCache()
        self._ticker_map: dict[str, int] | None = None

    def _get_json(self, url: str) -> dict:
        body = http_get(url, cache=self.cache, headers={"User-Agent": _user_agent()})
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise DataSourceError(f"{url} did not return JSON: {exc}") from exc

    def ticker_to_cik(self, ticker: str) -> int:
        if self._ticker_map is None:
            payload = self._get_json(SEC_TICKERS_URL)
            self._ticker_map = {
                v["ticker"].upper(): int(v["cik_str"]) for v in payload.values()
            }
        cik = self._ticker_map.get(ticker.upper())
        if cik is None:
            raise MissingDataError(
                "CIK",
                ticker=ticker,
                hint="not present in the SEC ticker file; the engine covers US filers only",
            )
        return cik

    def company_facts(self, ticker: str) -> CompanyFacts:
        cik = self.ticker_to_cik(ticker)
        payload = self._get_json(SEC_FACTS_URL.format(cik=cik))
        return CompanyFacts(payload, ticker)
