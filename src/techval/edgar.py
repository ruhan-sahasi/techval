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
from typing import Any, Iterable, Sequence

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


def _cluster_sightings(
    seen: set[tuple[str, str, str | None, str]],
) -> list[str]:
    """Collapse repeated sightings of one split into one effective date each.

    Each sighting is ``(last filing in old units, first filing in new units,
    period start, period end)`` and brackets the split inside the half-open
    interval ``(older, newer]``. Sightings of the same corporate action all
    bracket the same day, so their intervals share at least that day; two real
    splits at the same ratio are separated by years and share nothing.

    Walking the sightings in order of the filing that first showed the new
    units, a cluster stays open while the running intersection is non-empty and
    closes the moment a sighting starts at or after the earliest new-units
    filing already in it.

    The cluster is dated at that earliest new-units filing, which is the first
    date on which the new units are known to be on the record.

    **The bracket is not always tight, and this is a known limitation.** The
    true ex date lies somewhere inside ``(lo, hi]`` and neither end is it. Where
    a filing falls between the two ends it reports whichever basis was current
    when it was made, and the threshold can be on the wrong side of it. Palo
    Alto split three for one in September 2022 and the first filing to restate a
    comparative by three is dated May 2023, eight months later, so the 10-Q
    filed in November 2022 already reported post-split shares and is multiplied
    by three a second time. Dating at ``lo`` instead is not a fix but a trade:
    measured across every splitter in the fixtures it repairs Palo Alto's three
    affected periods and breaks five of CrowdStrike's, whose own bracket is a
    year wide. The real fix anchors each period on its own latest filing and
    reads the restatement ratio off the filer, the way
    ``techval.ml.warranted.share_basis_factor`` already does across two fact
    sets, rather than deciding from a date at all.

    Nothing silently consumes a figure this affects: ``share_basis_factor``
    compares a pinned fact set against a current one, returns a ratio that is
    not a product of any declared split, and the caller drops the row with the
    reason attached rather than training on it.

    A cluster resting on a single period is discarded. One period moving by
    exactly four could be a typo in one tagged fact; two could not.
    """
    effective: list[str] = []
    cluster: list[tuple[str, str, str | None, str]] = []
    lo = hi = ""

    def close() -> None:
        if len({(s, e) for _, _, s, e in cluster}) >= 2:
            effective.append(hi)

    for older, newer, start, end in sorted(seen):
        if cluster:
            nlo, nhi = max(lo, older), min(hi, newer)
            if nlo >= nhi:
                close()
                cluster, nlo, nhi = [], older, newer
        else:
            nlo, nhi = older, newer
        cluster.append((older, newer, start, end))
        lo, hi = nlo, nhi
    if cluster:
        close()
    return effective


def _neg_date(iso: str) -> tuple[int, ...]:
    """Sort key that orders ISO dates newest-first inside an ascending compare."""
    try:
        return tuple(-int(part) for part in iso.split("-"))
    except ValueError:
        return (0, 0, 0)


def _collapse_to_actions(
    detections: list[tuple[date, float, date]],
) -> list[tuple[date, float]]:
    """Reduce split detections to one threshold per corporate action.

    ``detections`` is ``(filed, factor, latest_period_end_restated)``, sorted.
    A company reflects one split across several filings, because each report
    restates only the comparative periods it happens to show, so the same
    four-for-one is detected three or four times over the following year. Every
    one of those detections is the same event and a pre-split fact must be
    multiplied by four once, not once per filing.

    The test that separates a repeat from a second split is the period end. A
    filing reflecting split S can only restate periods that closed before S,
    since anything that closed afterwards was first reported on the new basis.
    So a detection whose latest restated period ends after the open action's own
    filing date cannot belong to that action, and opens a new one.

    Returns ``(filed_on_or_after, factor)``, which is what ``_split_adjust``
    multiplies through.
    """
    thresholds: list[tuple[date, float]] = []
    open_action: dict[float, date] = {}
    for filed, factor, latest_end in detections:
        opened = open_action.get(factor)
        if opened is None or latest_end > opened:
            open_action[factor] = filed
            thresholds.append((filed, factor))
    return sorted(thresholds)


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

    def __init__(
        self,
        payload: dict[str, Any],
        ticker: str,
        knowledge_date: date | None = None,
    ) -> None:
        """``knowledge_date`` pins the run to what was filed by that date.

        Every fact filed later is discarded, so a valuation dated in the past uses
        only information that existed then. Without it a historical run silently
        reads restatements, later comparatives and split adjustments that nobody
        could have seen, and any backtest built on that measures hindsight rather
        than the model.
        """
        self.raw = payload
        self.ticker = ticker.upper()
        self.entity_name: str = payload.get("entityName", ticker)
        self.cik: int = int(payload.get("cik", 0))
        self._gaap: dict[str, Any] = payload.get("facts", {}).get("us-gaap", {})
        self._dei: dict[str, Any] = payload.get("facts", {}).get("dei", {})
        self._splits: list[tuple[date, float]] | None = None
        self.knowledge_date = knowledge_date

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
        actually declare, that is a split. Ordinary restatements are excluded by
        the tightness of that test.

        **One split is seen many times, and counting the sightings compounds the
        adjustment.** A quarterly report shows the current period and its
        prior-year comparative and nothing else, so a single four-for-one split
        is first reflected in one period at the next 10-Q, in two more periods
        at the one after that, and in more again at the 10-K. Treating each of
        those filings as its own corporate action multiplies the factor by four
        once per filing. Nvidia is the case that shows the damage: three filings
        first-restate periods across the 2021 four-for-one and three more across
        the 2024 ten-for-one, so a naive count applies 4 cubed times 10 cubed,
        and the diluted share count comes out at 2.5 trillion against a true
        24.9 billion.

        What each sighting actually pins down is an interval. A period whose
        value moves between a filing on A and the next filing on B says the
        split took effect somewhere in (A, B]. Every sighting of the same split
        brackets the same date, so their intervals intersect, and two genuinely
        separate splits at the same ratio are years apart and cannot. Sightings
        of one ratio are therefore clustered by intersecting interval, and each
        cluster is one split dated at the earliest filing that showed the new
        units.

        **A split is a unit, not information, so it is NOT knowledge dated.**
        This is the opposite of how every other fact here is treated and the
        reason is arithmetic rather than principle. The vendor restates its whole
        price history for a split, which is checkable in the committed close
        series: Nvidia split four for one in 2021 and ten for one in 2024, and
        the largest single-day move across 2,513 trading days is 1.298. Pair a
        share count left on the basis of its own day with a price already divided
        by ten and the market capitalisation is out by ten. Both sides of a ratio
        have to be quoted in the same unit; only the facts have to be point in
        time. A split carries no information about value either, since it
        multiplies the count and divides the price by the same number.

        Returns ``(filed_on_or_after, factor)`` pairs: any fact filed strictly
        before that date must be multiplied by ``factor`` to be comparable.
        """
        if self._splits is not None:
            return self._splits

        # ratio -> sightings of (last filing in old units, first in new units,
        # the period that moved). Keyed on the period rather than on the
        # observation because diluted and basic shares report the same quarter,
        # and counting observations would let one period clear a safeguard
        # meant to require two.
        sightings: dict[float, set[tuple[str, str, str | None, str]]] = {}
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
                if r.get("val") in (None, 0) or not r.get("end") or not r.get("filed"):
                    continue
                groups.setdefault((r.get("start"), r["end"]), []).append(r)

            for versions in groups.values():
                versions.sort(key=lambda r: r["filed"])
                for older, newer in zip(versions, versions[1:]):
                    ratio = float(newer["val"]) / float(older["val"])
                    for nice in self._SPLIT_RATIOS:
                        if abs(ratio - nice) < 0.005 * nice:
                            sightings.setdefault(nice, set()).add(
                                (
                                    older["filed"],
                                    newer["filed"],
                                    newer.get("start"),
                                    newer["end"],
                                )
                            )
                            break

        out: list[tuple[date, float]] = []
        for factor, seen in sightings.items():
            out.extend(
                (_d(effective), factor) for effective in _cluster_sightings(seen)
            )
        self._splits = sorted(out)
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
            # Filtered here as well as below: if a later filing won selection and
            # were only dropped afterwards, the period would vanish rather than
            # falling back to the version that was actually on file at the time.
            if self.knowledge_date is not None and r.get("filed"):
                if _d(r["filed"]) > self.knowledge_date:
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
            if self.knowledge_date is not None and filed > self.knowledge_date:
                continue
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

    def _largest_covering(
        self,
        entries: Sequence[str | tuple[str, ...]],
        names: Sequence[str],
        as_of: date,
        kind: str,
        incumbent: float,
        guard: float,
    ) -> tuple[str, tuple[float, list["Fact"], str], float] | None:
        """The lower-ranked ladder entry that dwarfs the incumbent, if any.

        Returns the winning name, its trailing-twelve-months resolution and the
        incumbent it beat. Only a strictly positive incumbent can be beaten: a
        ratio against zero or a negative measures nothing, and no income
        statement total this guard applies to is negative.
        """
        if incumbent <= 0:
            return None
        best: tuple[str, tuple[float, list[Fact], str], float] | None = None
        for entry, name in zip(entries, names):
            facts = (
                [f for f in self.facts(entry) if not f.is_instant]
                if isinstance(entry, str)
                else self._composite_series(entry)
            )
            if not facts:
                continue
            got = trailing_twelve_months(facts, as_of, kind=kind)
            if got is None or got[0] <= incumbent * guard:
                continue
            if best is None or got[0] > best[1][0]:
                best = (name, got, incumbent)
        return best

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
        component_guard: float | None = None,
    ) -> tuple[float | None, Provenance]:
        """Resolve a flow concept over the trailing twelve months.

        A ladder entry only wins if it can actually cover the window. A tag that
        exists but whose facts stop two years ago is skipped, exactly as for
        balance-sheet concepts -- otherwise the first tag with any history at all
        wins and the concept silently fails. Entries may be a single tag or a
        tuple of tags to be summed.

        ``component_guard`` handles the one failure the ladder cannot: a higher
        ranked tag that is not the total at all. Set it and every remaining
        entry is also resolved; where one covers the same window with a value
        more than this multiple of the winner's, it replaces the winner and the
        provenance says which tag it beat and by how much. An order of magnitude
        apart is not a disagreement about scope, it is one of the two being a
        disaggregation component.

        Charter Communications is the case that earned it. Charter tags
        ``RevenueFromContractWithCustomerIncludingAssessedTax`` at 889mm for
        fiscal 2025, the revenue ladder ranks that tag above ``Revenues``, and
        Charter's actual revenue that year was 54,774mm. Without the guard every
        Charter multiple this engine prints is 62 times too high, and nothing on
        the page says so.

        Left None, which is every caller that does not ask for it, nothing about
        the resolution changes. It is opt-in per concept because the test is a
        ratio, and a ratio conveys nothing about a concept that legitimately
        passes through zero: on EBIT or net income an order of magnitude is an
        ordinary year.
        """
        ladder = list(ladder)
        names = [t if isinstance(t, str) else " + ".join(t) for t in ladder]
        partial: list[str] = []

        for index, (entry, name) in enumerate(zip(ladder, names)):
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
                note = (
                    "day-weighted average, not a sum" if kind == "average" else None
                )
                if component_guard is not None:
                    swapped = self._largest_covering(
                        ladder[index + 1 :],
                        names[index + 1 :],
                        as_of,
                        kind,
                        value,
                        component_guard,
                    )
                    if swapped is not None:
                        beaten, name = name, swapped[0]
                        value, tiles, method = swapped[1]
                        note = (
                            f"{beaten} also covers this window at "
                            f"{swapped[2]:,.0f}, more than {component_guard:g} "
                            "times smaller, so it is a disaggregation component "
                            f"rather than the total and {name} was taken instead"
                        )
                return value, Provenance(
                    concept=concept,
                    tag=name,
                    method=method,
                    periods=[f"{t.start}..{t.end}" for t in tiles],
                    forms=[t.form for t in tiles],
                    filed=str(max(t.filed for t in tiles)),
                    tags_tried=names,
                    note=note,
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


@dataclass(frozen=True)
class DimensionedFact:
    """One XBRL fact together with the axes it was reported along.

    ``companyfacts`` publishes only undimensioned facts, so everything reported
    by class, by award type or by exercise-price band is invisible there. Those
    are exactly the facts a treasury-stock share count needs: option counts and
    strikes sit under the award-type axis, and a dual-class issuer's shares
    outstanding sit under the class-of-stock axis.
    """

    tag: str
    value: float
    unit: str | None
    start: date | None
    end: date
    dimensions: dict[str, str]

    @property
    def is_instant(self) -> bool:
        return self.start is None

    def axis(self, name: str) -> str | None:
        """Member reported along an axis, matched on the local name."""
        for axis, member in self.dimensions.items():
            if axis.split(":")[-1].lower() == name.split(":")[-1].lower():
                return member
        return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def strip_markup(body: bytes) -> str:
    """HTML or inline XBRL to readable text.

    Uses lxml where available and falls back to a regex strip, because a filing
    that cannot be parsed should still yield something a keyword search can run
    over rather than raising and taking a whole screen down with it.
    """
    try:
        from lxml import html as lxml_html

        tree = lxml_html.fromstring(body)
        for bad in tree.xpath("//script | //style | //ix:header", namespaces={
            "ix": "http://www.xbrl.org/2013/inlineXBRL"
        }):
            bad.getparent().remove(bad)
        text = tree.text_content()
    except Exception:
        import re

        raw = body.decode("utf-8", "replace")
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        text = re.sub(r"(?s)<[^>]+>", " ", raw)

    import html as _html
    import re as _re

    text = _html.unescape(text)
    text = text.replace("\xa0", " ")
    return _re.sub(r"[ \t]*\n[ \t]*", "\n", _re.sub(r"[ \t]+", " ", text)).strip()


def parse_instance(xml_bytes: bytes) -> list[DimensionedFact]:
    """Read an XBRL instance document into dimensioned facts.

    Deliberately namespace-agnostic: instance documents differ in prefixes and
    in which taxonomy versions they bind, and matching on local names survives
    that where a namespace map does not. Facts whose value is not numeric, and
    footnote or schema-reference elements, are skipped rather than coerced.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)

    contexts: dict[str, tuple[date | None, date | None, dict[str, str]]] = {}
    for node in root.iter():
        if _local(node.tag) != "context":
            continue
        cid = node.get("id")
        if not cid:
            continue
        start = end = None
        dims: dict[str, str] = {}
        for sub in node.iter():
            name = _local(sub.tag)
            text = (sub.text or "").strip()
            if name == "instant" and text:
                end = _d(text)
            elif name == "startDate" and text:
                start = _d(text)
            elif name == "endDate" and text:
                end = _d(text)
            elif name == "explicitMember":
                axis = sub.get("dimension")
                if axis and text:
                    dims[axis] = text
        if end is not None:
            contexts[cid] = (start, end, dims)

    units: dict[str, str] = {}
    for node in root.iter():
        if _local(node.tag) != "unit":
            continue
        uid = node.get("id")
        measures = [
            (m.text or "").strip().split(":")[-1]
            for m in node.iter()
            if _local(m.tag) == "measure"
        ]
        if uid and measures:
            units[uid] = "/".join(measures)

    out: list[DimensionedFact] = []
    for node in root.iter():
        ctx = node.get("contextRef")
        if not ctx or ctx not in contexts:
            continue
        text = (node.text or "").strip().replace(",", "")
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        sign = -1.0 if (node.get("sign") == "-") else 1.0
        start, end, dims = contexts[ctx]
        out.append(
            DimensionedFact(
                tag=_local(node.tag),
                value=value * sign,
                unit=units.get(node.get("unitRef") or ""),
                start=start,
                end=end,
                dimensions=dims,
            )
        )
    return out


class EdgarClient:
    def __init__(
        self, cache: HttpCache | None = None, knowledge_date: date | None = None
    ) -> None:
        self.cache = cache if cache is not None else HttpCache()
        self.knowledge_date = knowledge_date
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
        return CompanyFacts(payload, ticker, knowledge_date=self.knowledge_date)

    # -- filing documents, for the facts companyfacts leaves out ----------- #

    def submissions(self, ticker: str) -> dict:
        cik = self.ticker_to_cik(ticker)
        return self._get_json(SEC_SUBMISSIONS_URL.format(cik=cik))

    def latest_filing(
        self, ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q")
    ) -> tuple[str, date, str]:
        """Accession, filing date and form of the most recent periodic report.

        Honours the client's knowledge date, so a historical run reaches for the
        filing that was current then rather than the newest one on file.
        """
        recent = (self.submissions(ticker).get("filings") or {}).get("recent") or {}
        rows = list(
            zip(
                recent.get("accessionNumber", []),
                recent.get("filingDate", []),
                recent.get("form", []),
            )
        )
        for accn, filed, form in rows:
            if form not in forms:
                continue
            when = _d(filed)
            if self.knowledge_date is not None and when > self.knowledge_date:
                continue
            return accn, when, form
        raise MissingDataError(
            "periodic filing",
            ticker=ticker,
            hint=f"no {' or '.join(forms)} on file"
            + (f" by {self.knowledge_date}" if self.knowledge_date else ""),
        )

    def filings(
        self,
        ticker: str,
        forms: tuple[str, ...] = ("10-K",),
        since: date | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Recent filings of the given forms, newest first.

        Each entry carries accession, form, filing date and the primary
        document name, which is what ``filing_text`` needs. The client's
        knowledge date applies, so a historical run never sees a filing that had
        not happened.
        """
        recent = (self.submissions(ticker).get("filings") or {}).get("recent") or {}
        rows = zip(
            recent.get("accessionNumber", []),
            recent.get("filingDate", []),
            recent.get("form", []),
            recent.get("primaryDocument", []),
            recent.get("reportDate", []),
        )
        out: list[dict] = []
        for accn, filed, form, doc, period in rows:
            if form not in forms:
                continue
            when = _d(filed)
            if self.knowledge_date is not None and when > self.knowledge_date:
                continue
            if since is not None and when < since:
                continue
            out.append(
                {
                    "accession": accn,
                    "filed": when,
                    "form": form,
                    "document": doc,
                    "period": period,
                }
            )
            if len(out) >= limit:
                break
        return out

    def filing_text(self, ticker: str, filing: dict) -> str:
        """Plain text of a filing's primary document.

        Modern filings are inline XBRL, meaning the human-readable HTML and the
        tagged facts are the same file, so the document has to be stripped of
        markup before anything can read it. Script and style content is dropped
        rather than flattened, because an inline-XBRL document carries a good
        deal of both and it would otherwise end up in the text as noise.
        """
        cik = self.ticker_to_cik(ticker)
        bare = filing["accession"].replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{bare}/{filing['document']}"
        )
        body = http_get(url, cache=self.cache, headers={"User-Agent": _user_agent()})
        return strip_markup(body)

    def instance_facts(
        self, ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q")
    ) -> tuple[list[DimensionedFact], str, date]:
        """Dimensioned facts from the latest periodic filing's instance document.

        Returns the facts, the accession they came from and its filing date, so
        anything built on them can be sourced as precisely as a companyfacts
        figure. Raises ``MissingDataError`` when the filing carries no separable
        instance document, which happens with older filings that predate inline
        XBRL; callers are expected to fall back rather than guess.
        """
        cik = self.ticker_to_cik(ticker)
        accn, filed, _form = self.latest_filing(ticker, forms)
        bare = accn.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{bare}"
        listing = self._get_json(f"{base}/index.json")

        names = [
            item.get("name", "")
            for item in (listing.get("directory") or {}).get("item", [])
        ]
        instance = next(
            (n for n in names if n.endswith("_htm.xml")),
            next((n for n in names if n.endswith(".xml") and "cal" not in n
                  and "def" not in n and "lab" not in n and "pre" not in n), None),
        )
        if instance is None:
            raise MissingDataError(
                "XBRL instance document",
                ticker=ticker,
                hint=f"filing {accn} exposes no instance document to parse",
            )
        body = http_get(
            f"{base}/{instance}",
            cache=self.cache,
            headers={"User-Agent": _user_agent()},
        )
        return parse_instance(body), accn, filed
