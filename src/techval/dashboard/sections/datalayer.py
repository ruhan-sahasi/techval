"""Data layer: the places where the filings are not what a reader assumes they are.

Every figure in this section is a check on the engineering underneath the
finance, and every one is computed here from a committed fixture. None of them
scores a model, so the section carries no headline.

Five checks, each one a way the data has already been wrong in this repository:

**The debt ladders.** Verizon stopped tagging ``LongTermDebtNoncurrent`` in 2013
and the ladder asked for it until 2026, so its long-term debt read as zero. The
tag census (``tag_census.json``) records, per filer, every ladder concept it
reports with a material balance at its own balance-sheet date, and
``debt_ladder_reach`` intersects that against the ladders in ``techval.tags`` as
they stand when the collector runs. The ladders from before the fix are in no
fixture, so that comparison is refused rather than read off a docstring.

**Stock splits.** ``split_consistency`` runs the oracle from
``tests/test_split_detection.py`` over the same payloads: two filings of one
period are two statements about one quantity, so after adjustment they must
agree. It also scores the two dating rules the per-filing basis replaced, which
is what makes the zero a measurement.

**Split-adjusted prices.** ``nvda_split_evidence`` reads Nvidia's committed
closes across the filing windows that bracket each split. A vendor that did not
restate history would show a one-day fall of three quarters and nine tenths.

**Sub-vertical taxonomy and peer labels.** Each runs the package's own entry
point (``tmt_universe``, ``build_dataset``) against the committed fixtures and
counts what survives and what does not.

**Ticker resolution.** ``ticker_reach`` runs ``EdgarClient.resolve_ticker`` with
the SEC ticker file served from ``sec_tickers_tmt.json``. That file is a pruned
copy, so only the answers the pruning cannot change are drawn: the departed
registrants, resolved at the date of the filing that proved each symbol, and
the seed tickers the live file no longer carried. The split of the whole seed
universe by rung is computed too, and refused, with the numbers that show why.

The collector is split in two. ``collect`` loads fixtures and runs the checks
inside ``ctx.record``; ``shape`` turns their results into figures, the takeaway
and the refusals, and is pure, so it is tested on small fakes without running
anything expensive.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from datetime import date
from typing import Any, Callable, Iterable, Iterator, Mapping

ID = "datalayer"
TITLE = "Data layer"

CENSUS = "tag_census.json"
SHARE_BASIS = "share_basis_facts.json"
CRWD_FACTS = "companyfacts_CRWD.json"
CLOSES = "signals/closes.csv.gz"
SUBMISSIONS = "submissions_tmt.json"
PEER_GROUPS = "peer_groups_tmt.json"
PEER_PANEL = "peer_panel_tmt.json"
PEER_TEXT = "peer_item1_tmt.json"
TICKER_FILE = "sec_tickers_tmt.json"
FORMER_INDEX = "former_tickers/index.json"
MNA_UNIVERSE = "mna/universe.json"

INPUTS: list[str] = [
    CENSUS,
    SHARE_BASIS,
    CRWD_FACTS,
    CLOSES,
    SUBMISSIONS,
    PEER_GROUPS,
    PEER_PANEL,
    PEER_TEXT,
    TICKER_FILE,
    FORMER_INDEX,
    MNA_UNIVERSE,
]

_HERE = __name__

# Which figures each result becomes, in page order. ``collect`` records
# provenance against these ids before ``shape`` runs, so the two must agree;
# ``shape`` checks that they do, and the renderer's order is tested against it.
FIGURES_BY_RESULT: dict[str, tuple[str, ...]] = {
    "debt": ("debt_reach",),
    "splits": ("split_counts", "split_oracle"),
    "closes": ("nvda_closes",),
    "taxonomy": ("taxonomy_by_vertical", "taxonomy_how"),
    "peers": ("peer_label_counts", "peer_groups_lost", "peer_label_funnel"),
    "tickers": ("ticker_reach",),
}
FIGURE_IDS: tuple[str, ...] = tuple(f for figs in FIGURES_BY_RESULT.values() for f in figs)

REFUSED_DEBT_BEFORE = "Debt ladder coverage before the Verizon fix"
REFUSED_SEED_RUNGS = "Ticker resolution by rung across the seed universe"


class _Unavailable(Exception):
    """A check that cannot be computed from what is committed.

    Raised inside ``ctx.record`` so the block appends no provenance row, and
    caught by ``collect`` so the figure becomes a refusal. Deliberately not a
    ``TechvalError``: one missing series refuses one figure, not the section.
    """


# --------------------------------------------------------------------------- #
# Small text helpers
# --------------------------------------------------------------------------- #

_LEGAL_SUFFIX = re.compile(
    r"[,\s]+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|n\.?v|s\.?a)\.?$",
    re.IGNORECASE,
)
_SMALL_WORDS = {"THE", "AND", "OF", "FOR"}


def display_name(name: str | None, fallback: str) -> str:
    """A registrant's name as a person writes it: no legal suffix, no shouting.

    The SEC stores many names in capitals (``DIGITAL REALTY TRUST, INC.``). A
    word of four letters or more is recased; a short one is kept as filed, since
    it is more often an acronym than a word.
    """
    text = str(name or "").strip()
    while True:
        stripped = _LEGAL_SUFFIX.sub("", text).strip()
        if stripped == text or not stripped:
            break
        text = stripped
    if not text:
        return fallback
    if text.upper() == text:
        words = []
        for i, word in enumerate(text.split()):
            if word.isalpha() and len(word) >= 4:
                words.append(word.capitalize())
            elif word in _SMALL_WORDS:
                words.append(word.capitalize() if i == 0 else word.lower())
            else:
                words.append(word)
        text = " ".join(words)
    return text


def _join(items: Iterable[str], last: str = "and") -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {last} " + items[-1]


def _pct(share: float, dp: int = 1) -> str:
    return f"{share * 100:+.{dp}f}%".replace("-", "−")


def _factor(x: float) -> str:
    return f"{x:g}"


def _plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or one + "s")


# --------------------------------------------------------------------------- #
# The debt ladders
# --------------------------------------------------------------------------- #

# Concepts that mean "this filer publishes a borrowings line", copied from
# tests/test_tags.py, which states why they are written out rather than read off
# the ladders: a guard that asks the ladder both halves of its own question
# cannot fail. The dashboard test holds these two sets equal to the guard's.
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
# A total covers both legs, so a filer that publishes only a total still carries
# debt on each leg. The guard in test_tags.py counts the total on the reaching
# side only, which leaves a total-only filer (Super Micro, on the census) out of
# the denominator; counting it on both sides is what "carries debt" means, and it
# is the count the guard's own docstring states (81 filers, 80 reached).
REPORTS_TOTAL_DEBT = frozenset(
    {
        "DebtLongtermAndShorttermCombinedAmount",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
        "DebtAndCapitalLeaseObligations",
        "NotesPayable",
    }
)

_DEBT_LADDERS = ("DEBT_NONCURRENT", "DEBT_CURRENT", "DEBT_COMBINED")


def _concepts(ladder: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for entry in ladder:
        if isinstance(entry, str):
            out.add(entry)
        elif isinstance(entry, (tuple, list)):
            out.update(entry)
    return out


def debt_ladder_reach(census: Mapping[str, Any]) -> dict[str, Any]:
    """Of the filers publishing a material borrowings line, how many the ladders reach.

    Two legs, as the engine resolves them. A filer counts on a leg when its
    census entry holds a concept that means that line, or a total; it is reached
    when the leg's ladder, or the combined ladder, names one of the concepts it
    reports. A filer counts as reached overall only when every leg it reports is
    reached. The ladders are read from ``techval.tags`` at run time, so taking a
    concept out of one changes this figure without touching the census.

    An unreached filer is not necessarily a silent zero: ``financials`` refuses
    the debt figure when a concept in ``DEBT_CROSSCHECK`` reports more than the
    ladder resolved, so ``caught_by`` names the cross-check concepts each one
    carries. An unreached filer with none is the Verizon failure.
    """
    from techval import tags

    combined = _concepts(tags.DEBT_COMBINED)
    crosscheck = _concepts(tags.DEBT_CROSSCHECK)
    live_by_filer = census["live_by_filer"]
    legs = []
    unreached_any: set[str] = set()
    reporting_any: set[str] = set()
    for key, reported, ladder in (
        ("long_term", REPORTS_NONCURRENT_DEBT, tags.DEBT_NONCURRENT),
        ("current", REPORTS_CURRENT_DEBT, tags.DEBT_CURRENT),
    ):
        reach = _concepts(ladder) | combined
        reporting, unreached = [], []
        for ticker, live in sorted(live_by_filer.items()):
            held = set(live)
            if not held & (reported | REPORTS_TOTAL_DEBT):
                continue
            reporting.append(ticker)
            if not held & reach:
                unreached.append(ticker)
        reporting_any.update(reporting)
        unreached_any.update(unreached)
        legs.append(
            {
                "leg": key,
                "reporting": len(reporting),
                "reached": len(reporting) - len(unreached),
                "unreached": unreached,
            }
        )

    every_debt_concept = REPORTS_NONCURRENT_DEBT | REPORTS_CURRENT_DEBT | REPORTS_TOTAL_DEBT
    header = census.get("_techval_fixture") or {}
    recorded_ladders = census.get("ladders") or {}
    census_matches_tags = all(
        sorted(recorded_ladders.get(name, [])) == sorted(_concepts(getattr(tags, name)))
        for name in _DEBT_LADDERS
    )
    return {
        "legs": legs,
        "any": {"reporting": len(reporting_any), "reached": len(reporting_any - unreached_any)},
        "unreached": [
            {
                "ticker": t,
                "concepts": sorted(set(live_by_filer[t]) & every_debt_concept),
                "has_total": bool(set(live_by_filer[t]) & REPORTS_TOTAL_DEBT),
                "caught_by": sorted(set(live_by_filer[t]) & crosscheck),
            }
            for t in sorted(unreached_any)
        ],
        "filers_covered": len(census["covered"]),
        "universe_size": census["universe_size"],
        "recorded": header.get("recorded"),
        "material_usd": header.get("material_usd"),
        "tolerance_days": header.get("tolerance_days"),
        "census_ladders_match_tags": census_matches_tags,
    }


# --------------------------------------------------------------------------- #
# Stock splits
# --------------------------------------------------------------------------- #

# The band the oracle allows, from tests/test_split_detection.py: the largest
# honest restatement across these filers is three hundredths of a percent, and a
# misdated split moves a period by a factor of two or more.
AGREEMENT_BAND = 1.01

# The quarter test_split_detection.py singles out: reported once, as a prior-year
# comparative, so only the per-filing basis can settle its unit.
PANW_QUARTER = ("PANW", "WeightedAverageNumberOfDilutedSharesOutstanding", "2021-08-01", "2021-10-31")


def _versions_by_period(facts) -> Iterator[tuple[str, str, tuple, list[dict]]]:
    """Every period a share tag reports more than once, with its raw filed rows."""
    for tag in facts._SHARE_COUNT_TAGS:
        found = facts._units(tag)
        if not found:
            continue
        unit, rows = found
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            if row.get("val") in (None, 0) or not row.get("end") or not row.get("filed"):
                continue
            groups[(row.get("start"), row["end"])].append(row)
        for key, versions in groups.items():
            if len(versions) >= 2:
                yield tag, unit, key, versions


def _threshold_rule(thresholds: list[tuple[str, float]], inclusive: bool) -> Callable:
    """A split basis dated at one fixed end of each window, for comparison."""

    def adjust(value: float, filed: str, unit: str) -> float:
        factor = 1.0
        for when, f in thresholds:
            if filed <= when if inclusive else filed < when:
                factor *= f
        return value * factor if unit == "shares" else value / factor

    return adjust


def split_consistency(payloads: Mapping[str, dict]) -> dict[str, Any]:
    """The oracle over every splitter and the control, under three dating rules.

    ``current`` is ``CompanyFacts._split_adjust``, the per-filing basis. ``late``
    dates each split at the end of its window, the rule the basis replaced;
    ``early`` dates it at the start, the attempt before that. A period is
    inconsistent when its filings still disagree by more than the band once
    adjusted.
    """
    from techval.edgar import CompanyFacts

    filers = []
    tags_seen: set[str] = set()
    for ticker in sorted(payloads):
        facts = CompanyFacts(payloads[ticker], ticker)
        brackets = facts._split_brackets()
        rules = {
            "current": lambda v, filed, unit, f=facts: f._split_adjust(
                v, date.fromisoformat(filed), unit
            ),
            "late": _threshold_rule([(hi, x) for _, hi, x in brackets], inclusive=False),
            "early": _threshold_rule([(lo, x) for lo, _, x in brackets], inclusive=True),
        }
        checked = 0
        bad: Counter = Counter()
        for tag, unit, _, versions in _versions_by_period(facts):
            checked += 1
            tags_seen.add(tag)
            for name, rule in rules.items():
                adjusted = [rule(float(r["val"]), r["filed"], unit) for r in versions]
                if max(adjusted) / min(adjusted) > AGREEMENT_BAND:
                    bad[name] += 1
        filers.append(
            {
                "ticker": ticker,
                "splits": [factor for _, factor in facts._split_factors()],
                "checked": checked,
                "inconsistent": bad["current"],
                "late_end": bad["late"],
                "early_end": bad["early"],
            }
        )

    quarter = None
    ticker, tag, start, end = PANW_QUARTER
    if ticker in payloads:
        facts = CompanyFacts(payloads[ticker], ticker)
        found = facts._units(tag)
        raw = [
            r
            for r in (found[1] if found else [])
            if r.get("start") == start and r.get("end") == end
        ]
        adjusted = [
            f
            for f in facts.facts(tag)
            if f.start == date.fromisoformat(start) and f.end == date.fromisoformat(end)
        ]
        if found and len(raw) == 1 and len(adjusted) == 1:
            late = _threshold_rule(
                [(hi, x) for _, hi, x in facts._split_brackets()], inclusive=False
            )
            as_filed = float(raw[0]["val"])
            quarter = {
                "ticker": ticker,
                "start": start,
                "end": end,
                "filed": raw[0]["filed"],
                "as_filed": as_filed,
                "adjusted": adjusted[0].val,
                "late_end": late(as_filed, raw[0]["filed"], found[0]),
            }
    return {
        "filers": filers,
        "quarter": quarter,
        "band": AGREEMENT_BAND,
        "tags": len(tags_seen),
    }


# --------------------------------------------------------------------------- #
# Split-adjusted prices
# --------------------------------------------------------------------------- #


def read_closes(path, ticker: str) -> list[tuple[date, float]]:
    """One ticker's daily closes from the committed signal price series, oldest first."""
    rows = []
    with gzip.open(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["ticker"] == ticker:
                rows.append((date.fromisoformat(row["date"]), float(row["close"])))
    rows.sort()
    return rows


def nvda_split_evidence(
    closes: list[tuple[date, float]], brackets: list[tuple[str, str, float]]
) -> dict[str, Any]:
    """Whether Nvidia's closes step down where its filings say a split happened.

    ``brackets`` are the windows ``CompanyFacts._split_brackets`` reads off the
    share filings: the last filing on the old basis and the first on the new.
    Each ex date lies inside its window, so a series the vendor did not restate
    would show one day inside it falling by ``1 - 1/factor``. The drawn series
    is the last close of each week, which keeps the snapshot readable; every
    move is measured on the daily series.
    """
    if len(closes) < 2:
        raise _Unavailable(
            "the committed price series carries no Nvidia closes, so there is no "
            "series to read a split in"
        )
    if not brackets:
        raise _Unavailable(
            "the Nvidia share facts show no split, so there is no window to read the "
            "price series across"
        )
    moves = [
        (closes[i][1] / closes[i - 1][1] - 1.0, closes[i][0])
        for i in range(1, len(closes))
    ]
    largest = max(moves, key=lambda m: abs(m[0]))
    windows = []
    for lo, hi, factor in brackets:
        start, end = date.fromisoformat(lo), date.fromisoformat(hi)
        inside = [m for m in moves if start < m[1] <= end]
        if not inside:
            raise _Unavailable(
                f"no close in the committed series falls inside the split window "
                f"{lo} to {hi}, so the series cannot show whether it was restated"
            )
        worst = min(inside, key=lambda m: m[0])
        windows.append(
            {
                "from": lo,
                "to": hi,
                "factor": factor,
                "worst_day": worst[0],
                "worst_date": worst[1],
                "unadjusted_move": 1.0 / factor - 1.0,
                "days": len(inside),
            }
        )

    weekly: dict[tuple[int, int], tuple[date, float]] = {}
    for d, c in closes:
        iso = d.isocalendar()
        weekly[(iso[0], iso[1])] = (d, c)
    drawn = dict(weekly.values())
    drawn[largest[1]] = dict(closes)[largest[1]]
    return {
        "ticker": "NVDA",
        "first": closes[0][0],
        "last": closes[-1][0],
        "days": len(closes),
        "weekly": sorted(drawn.items()),
        "largest": {"move": largest[0], "date": largest[1], "close": drawn[largest[1]]},
        "windows": windows,
    }


# --------------------------------------------------------------------------- #
# Sub-vertical taxonomy
# --------------------------------------------------------------------------- #

VERTICAL_LABELS = {
    "infrastructure_software": "Infrastructure software",
    "application_software": "Application software",
    "internet": "Internet",
    "semiconductors": "Semiconductors",
    "hardware": "Hardware",
    "it_services": "IT services",
    "payments": "Payments",
    "media_entertainment": "Media and entertainment",
    "telecom": "Telecom",
    "towers_fiber": "Towers and fibre",
    "gaming": "Gaming",
}


class _FixtureSubmissions:
    """``tmt_universe``'s client, over the committed submissions payloads.

    A ticker with no payload raises the ``MissingDataError`` the real client
    raises for a symbol the SEC file does not carry, which is how the fixture
    records the seeds that did not resolve on its retrieval date.
    """

    knowledge_date = None

    def __init__(self, companies: Mapping[str, dict]) -> None:
        self.companies = companies

    def submissions(self, ticker: str) -> dict:
        from techval.errors import MissingDataError

        try:
            return self.companies[ticker.upper()]
        except KeyError:
            raise MissingDataError(
                "CIK",
                ticker=ticker,
                hint="absent from the SEC ticker file on the fixture's retrieval date",
            ) from None


def taxonomy_placement(companies: Mapping[str, dict], assumptions) -> dict[str, Any]:
    """How each seed filer got its sub-vertical.

    ``tmt_universe`` gives the final call. Running ``classify`` again on the same
    code and name, with no business text as the universe had none, gives what
    the code alone would have said, so the two can be compared without parsing
    the source string.
    """
    from techval.tmt.taxonomy import classify, tmt_universe

    universe = tmt_universe(_FixtureSubmissions(companies), assumptions)
    how: Counter = Counter()
    by_vertical: dict[str, Counter] = defaultdict(Counter)
    unresolved = []
    for company in universe:
        if company.sub_vertical is None:
            how["unresolved"] += 1
            unresolved.append(company.ticker)
            continue
        derived, _, _ = classify(company.sic, company.name, None, assumptions)
        if derived is company.sub_vertical:
            kind = "code_agrees"
        elif derived is None:
            kind = "prior_fills"
        else:
            kind = "prior_overrides"
        how[kind] += 1
        by_vertical[company.sub_vertical.value]["classified"] += 1
        by_vertical[company.sub_vertical.value]["by_code"] += kind == "code_agrees"
    return {
        "seed": len(universe),
        "payloads": len(companies),
        "classified": len(universe) - len(unresolved),
        "how": {k: how[k] for k in ("code_agrees", "prior_fills", "prior_overrides", "unresolved")},
        "by_vertical": {
            v: {"classified": c["classified"], "by_code": c["by_code"]}
            for v, c in sorted(by_vertical.items())
        },
        "unresolved": sorted(unresolved),
    }


# --------------------------------------------------------------------------- #
# Peer labels
# --------------------------------------------------------------------------- #


def peer_label_losses(groups, panel, corpora) -> dict[str, Any]:
    """Disclosed peer groups as recorded, and what ``build_dataset`` can encode.

    A named peer becomes a training pair only when both legs have a feature row
    and a business description at the group's panel date. A group is kept when
    at least one of its peers does. ``build_dataset`` counts a group that fails
    a group-level check once, under the first check it fails, and a group that
    passes them all but keeps no peer under no heading, so that last count is
    what is left over. The peers lost with a whole group are likewise what the
    groups named less what the kept and peer-level counts account for, so every
    named peer lands in exactly one row.
    """
    from techval.ml.encoder import build_dataset

    dataset = build_dataset(groups, panel, corpora)
    named = sum(len(g.peers) for g in groups)
    dropped = dataset.dropped
    peer_level = (
        dropped["self-reference"]
        + dropped["peer not in the candidate universe"]
        + dropped["peer not encodable at the panel date"]
    )
    pairs = len(dataset.examples)
    group_level = {
        "not_usable": dropped["group not usable"],
        "before_panel": dropped["group filed before every panel date"],
        "filer_not_encodable": dropped["filer not encodable at its panel date"],
    }
    lost_groups = len(groups) - len(dataset.groups)
    group_level["no_peer_kept"] = lost_groups - sum(group_level.values())
    return {
        "groups_recorded": len(groups),
        "filers_recorded": len({g.ticker for g in groups}),
        "groups_kept": len(dataset.groups),
        "filers_kept": len({g.ticker for g in dataset.groups}),
        "named_peers": named,
        "pairs": pairs,
        "lost": {
            "group": named - pairs - peer_level,
            "universe": dropped["peer not in the candidate universe"],
            "encodable": dropped["peer not encodable at the panel date"],
            "self": dropped["self-reference"],
        },
        "groups_dropped": group_level,
        "panel_dates": len(dataset.panel_dates),
        "universe": len(dataset.universe),
    }


# --------------------------------------------------------------------------- #
# Ticker resolution
# --------------------------------------------------------------------------- #


def _offline_client(ticker_file: Mapping[str, Any], knowledge_date: date | None):
    from techval.edgar import SEC_TICKERS_URL, EdgarClient, HttpCache

    class OfflineClient(EdgarClient):
        """The production resolver with the SEC ticker file served from the fixture."""

        def _get_json(self, url: str) -> dict:
            if url != SEC_TICKERS_URL:
                raise AssertionError(f"resolution reached for {url}, which is not a fixture")
            return dict(ticker_file)

    return OfflineClient(cache=HttpCache(enabled=False), knowledge_date=knowledge_date)


def ticker_reach(
    index: Mapping[str, Any],
    universe: list[Mapping[str, Any]],
    ticker_file: Mapping[str, Any],
    seed: Iterable[str],
    submissions: Mapping[str, Any],
) -> dict[str, Any]:
    """Which rung of ``resolve_ticker`` reaches the registrants a ticker file misses.

    Three views, each separating what the pruned ticker file can change from
    what it cannot.

    *Departed registrants.* The seeds are the index's own: the departed
    registrants of the M&A universe and the CIKs the index names beyond it. Each
    symbol is resolved at the date of the filing that proved it, where a
    departed registrant's CIK is by definition not the live file's answer for
    that symbol, so the pruning cannot move the rung. A registrant no symbol
    reaches is resolved by its CIK, the rung that always works.

    *Seed tickers the live file had dropped.* A seed with no submissions payload
    was absent from the SEC ticker file on the fixture's retrieval date. Where
    the pruned copy lacks it too, the offline resolution is the live one.

    *The whole seed universe.* Resolved offline against the pruned copy. Kept so
    the refusal can say exactly how far the pruning moves the answer.
    """
    from techval import former_tickers
    from techval.errors import MissingDataError

    by_cik: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for symbol, claims in index["entries"].items():
        for claim in claims:
            by_cik[int(claim["cik"])].append((symbol, claim))
    departed = {int(r["cik"]) for r in universe if r.get("departed")}
    unresolved = {int(r["cik"]): r for r in index.get("unresolved", [])}
    seeds = departed | set(by_cik) | set(unresolved)

    undated = _offline_client(ticker_file, None)
    rungs: Counter = Counter()
    cik_only = []
    for cik in sorted(seeds):
        hit = None
        for symbol, claim in by_cik.get(cik, []):
            pinned = _offline_client(ticker_file, date.fromisoformat(claim["through"]))
            dated = pinned.resolve_ticker(symbol)
            if dated.cik == cik:
                hit = dated
                break
        if hit is not None:
            rungs[hit.source] += 1
            continue
        resolution = undated.resolve_ticker(f"CIK{cik:010d}")
        rungs[resolution.source] += 1
        cik_only.append(cik)

    shared = []
    for symbol, claims in sorted(index["entries"].items()):
        if len(claims) < 2:
            continue
        latest = former_tickers.resolve(symbol)
        shared.append(
            {
                "symbol": symbol,
                "registrants": len(claims),
                "undated_cik": latest.cik if latest else None,
                "reached_only_dated": sum(
                    1 for c in claims if latest is None or int(c["cik"]) != latest.cik
                ),
            }
        )

    carried = {str(v["ticker"]).upper() for v in ticker_file.values()}
    seed = sorted({str(t).upper() for t in seed})
    absent = [t for t in seed if t not in submissions]
    absent_rows = []
    for t in absent:
        if t in carried:
            # The pruned copy carries a symbol the live file did not: offline and
            # live would disagree, so this row cannot stand for the live answer.
            absent_rows = None
            break
        try:
            r = undated.resolve_ticker(t)
        except MissingDataError:
            absent_rows.append({"ticker": t, "source": "unresolved", "cik": None, "name": None})
        else:
            absent_rows.append({"ticker": t, "source": r.source, "cik": r.cik, "name": r.name})

    whole: Counter = Counter()
    unresolved_with_payload = 0
    for t in seed:
        try:
            whole[undated.resolve_ticker(t).source] += 1
        except MissingDataError:
            whole["unresolved"] += 1
            unresolved_with_payload += t in submissions

    return {
        "seeds": len(seeds),
        "symbols": len(index["entries"]),
        "registrants": len(by_cik),
        "rungs": dict(sorted(rungs.items())),
        "cik_only": len(cik_only),
        "cik_only_reasons": sorted(
            {unresolved[c]["why"].split(" (")[0] for c in cik_only if c in unresolved}
        ),
        "shared_symbols": shared,
        "seed_universe": len(seed),
        "seed_absent": absent_rows,
        "seed_offline": {
            "ticker_file_registrants": len(carried),
            "seed_in_ticker_file": len(set(seed) & carried),
            "rungs": dict(sorted(whole.items())),
            "unresolved_with_payload": unresolved_with_payload,
        },
    }


# --------------------------------------------------------------------------- #
# Shaping
# --------------------------------------------------------------------------- #


def _figure(kind: str, title: str, subtitle: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "title": title, "subtitle": subtitle, "data": data}


Built = tuple[dict[str, Any], list[dict[str, str]]]


def _debt_figures(r: dict[str, Any], names: Mapping[str, str]) -> Built:
    legs = {leg["leg"]: leg for leg in r["legs"]}
    lt, cur, every = legs["long_term"], legs["current"], r["any"]
    material = f"{(r['material_usd'] or 0) / 1e6:,.0f}mm"
    misses = r["unreached"]

    def who(m: dict[str, Any]) -> str:
        return f"{display_name(names.get(m['ticker']), m['ticker'])} ({m['ticker']})"

    def why(m: dict[str, Any]) -> str:
        held = _join(m["concepts"]) or "no ladder concept"
        text = f"{who(m)} reports {held}" + ("" if m["has_total"] else " and no total")
        if m["caught_by"]:
            text += (
                f", which the debt cross-check sees, so its debt is refused rather "
                "than read as zero"
            )
        else:
            text += ", which no check sees, so its debt reads as zero"
        return text

    caught = [display_name(names.get(m["ticker"]), m["ticker"]) for m in misses if m["caught_by"]]
    silent = [display_name(names.get(m["ticker"]), m["ticker"]) for m in misses if not m["caught_by"]]
    title = (
        f"The debt ladders reach {lt['reached']} of the {lt['reporting']} filers with "
        "material long-term debt"
    )
    if silent:
        title += f", and {_join(silent)} {'reads' if len(silent) == 1 else 'read'} as zero"
    elif caught:
        title += (
            f"; the cross-check refuses the {'one' if len(caught) == 1 else len(caught)} "
            f"they miss, {_join(caught)}"
        )
    tiles = [
        {
            "label": "Long-term debt line or total",
            "value": lt["reached"],
            "format": "int",
            "sub": (
                f"reached, of {lt['reporting']} filers reporting at least {material} "
                f"within {r['tolerance_days']} days of their own balance-sheet date"
            ),
        },
        {
            "label": "Current debt line or total",
            "value": cur["reached"],
            "format": "int",
            "sub": f"reached, of {cur['reporting']} filers reporting one",
        },
        {
            "label": "Any debt line",
            "value": every["reached"],
            "format": "int",
            "sub": (
                f"reached on every leg they report, of {every['reporting']} filers; "
                f"census of {r['filers_covered']} of {r['universe_size']} seed filers, "
                f"recorded {r['recorded']}"
            ),
        },
        {
            "label": "Not reached",
            "value": len(misses),
            "format": "int",
            "sub": "; ".join(why(m) for m in misses) or "every filer with a borrowings line is reached",
        },
    ]
    if r["census_ladders_match_tags"]:
        before = (
            f"The census was recorded on {r['recorded']}, and the debt ladders it "
            "records are the ones techval.tags carries today, so no committed fixture "
            "holds a ladder from before the fix to count against. The earlier "
            "ladders survive only in git history, which the snapshot does not digest."
        )
    else:
        before = (
            f"The debt ladders the census recorded on {r['recorded']} differ from the "
            "ones techval.tags carries today, and no committed fixture says which "
            "ladder stood before the fix. The earlier ladders survive only in git "
            "history, which the snapshot does not digest."
        )
    figure = _figure(
        "tiles",
        title,
        "",
        {
            "tiles": tiles,
            "counts": {
                "long_term": lt,
                "current": cur,
                "any": every,
                "unreached": [m["ticker"] for m in misses],
                "silent_zero": [m["ticker"] for m in misses if not m["caught_by"]],
                "filers_covered": r["filers_covered"],
                "universe_size": r["universe_size"],
                "census_recorded": r["recorded"],
            },
            "near": [REFUSED_DEBT_BEFORE],
        },
    )
    return {"debt_reach": figure}, [{"what": REFUSED_DEBT_BEFORE, "why": before}]


def _splits_label(ticker: str, splits: list[float]) -> str:
    if not splits:
        return f"{ticker}, never split (control)"
    return f"{ticker}, " + " then ".join(f"{_factor(s)}-for-1" for s in splits)


def _split_figures(r: dict[str, Any]) -> dict[str, Any]:
    filers = r["filers"]
    checked = sum(f["checked"] for f in filers)
    bad = sum(f["inconsistent"] for f in filers)
    splitters = [f for f in filers if f["splits"]]
    controls = [f for f in filers if not f["splits"]]
    ordered = splitters + controls
    band = f"{(r['band'] - 1) * 100:g}%"
    title = (
        f"All {checked} share-count periods filed more than once agree once split-adjusted"
        if bad == 0
        else f"{bad} of {checked} share-count periods filed more than once disagree once split-adjusted"
    )
    tiles = [
        {
            "label": "Periods filed more than once",
            "value": checked,
            "format": "int",
            "sub": (
                f"across {len(splitters)} {_plural(len(splitters), 'splitter')} and "
                f"{len(controls)} {_plural(len(controls), 'control')}, "
                f"{r['tags']} weighted-average share {_plural(r['tags'], 'tag')}"
            ),
        },
        {
            "label": "Disagree after adjustment",
            "value": bad,
            "format": "int",
            "sub": f"by more than {band}, a band a genuine restatement stays inside",
        },
    ]
    q = r.get("quarter")
    if q:
        tiles.append(
            {
                "label": f"{q['ticker']} diluted shares, quarter ended {q['end']}",
                "value": q["adjusted"],
                "format": "int",
                "sub": (
                    f"filed once, as {q['as_filed']:,.0f} on {q['filed']}; dating each "
                    f"split at the late end of its window reads {q['late_end']:,.0f}"
                ),
            }
        )
    late_total = sum(f["late_end"] for f in ordered)
    early_total = sum(f["early_end"] for f in ordered)
    worst_late = max(ordered, key=lambda f: f["late_end"]) if ordered else None
    worst_early = max(ordered, key=lambda f: f["early_end"]) if ordered else None
    if bad == 0 and late_total and early_total:
        oracle_title = (
            f"Dating a split at either end of its window breaks {late_total} or "
            f"{early_total} periods; the per-filing basis breaks none"
        )
    elif bad == 0:
        oracle_title = "Every dating rule leaves every filer consistent"
    else:
        oracle_title = title
    rows = [
        {
            "label": _splits_label(f["ticker"], f["splits"]),
            "values": {
                "current": f["inconsistent"],
                "late": f["late_end"],
                "early": f["early_end"],
            },
        }
        for f in ordered
    ]
    subtitle = (
        "Periods whose filings still disagree by more than "
        f"{band} after adjustment, by how each split is dated. The oracle needs no "
        "outside truth: every filing of one period must report the same quantity. "
        f"{checked} periods, share_basis_facts.json and companyfacts_CRWD.json"
    )
    if worst_late and worst_late["late_end"]:
        subtitle += f"; late-end errors fall on {worst_late['ticker']}"
    if worst_early and worst_early["early_end"]:
        subtitle += f", early-end errors on {worst_early['ticker']}"
    return {
        "split_counts": _figure(
            "tiles",
            title,
            "",
            {
                "tiles": tiles,
                "counts": {
                    "periods_checked": checked,
                    "inconsistent": bad,
                    "by_filer": {f["ticker"]: f["inconsistent"] for f in filers},
                    "quarter": q,
                },
            },
        ),
        "split_oracle": _figure(
            "dot",
            oracle_title,
            subtitle + ".",
            {
                "rows": rows,
                "series": [
                    {"key": "current", "name": "Per-filing basis, as shipped", "role": "model"},
                    {"key": "late", "name": "Dated at the window's late end", "role": "baseline"},
                    {"key": "early", "name": "Dated at the window's early end", "role": "alt"},
                ],
                "format": "int",
                "zero": True,
                "labelHeader": "Filer",
            },
        ),
    }


def _closes_figures(r: dict[str, Any]) -> dict[str, Any]:
    windows = r["windows"]
    restated = all(w["worst_day"] > w["unadjusted_move"] / 2 for w in windows)
    deepest = min(windows, key=lambda w: w["worst_day"])
    if restated:
        title = (
            f"Nvidia's closes carry no split step: no day inside either split window "
            f"falls more than {_pct(-deepest['worst_day']).lstrip('+')}"
        )
    else:
        title = (
            f"Nvidia's closes fall {_pct(deepest['worst_day'])} in a day inside the "
            f"{_factor(deepest['factor'])}-for-1 window, as an unrestated series would"
        )
    largest = r["largest"]
    unrestated = _join(
        f"{_pct(w['unadjusted_move'], 0)} at the {_factor(w['factor'])}-for-1" for w in windows
    )
    return {
        "nvda_closes": _figure(
            "line",
            title,
            (
                f"Close, USD, last close of each week, {r['first']} to {r['last']}, "
                f"{r['days']:,} trading days, signals/closes.csv.gz. An unrestated "
                f"series would move {unrestated}."
            ),
            {
                "series": [
                    {
                        "name": "Close",
                        "role": "model",
                        "values": [{"x": d, "y": c} for d, c in r["weekly"]],
                    }
                ],
                "x": {"label": "Date"},
                "format": "num:2",
                "valueLabel": "Close, USD",
                "windows": [
                    {
                        "from": w["from"],
                        "to": w["to"],
                        "label": f"{_factor(w['factor'])}-for-1 window",
                        "worst_day": w["worst_day"],
                        "worst_date": w["worst_date"],
                        "unadjusted_move": w["unadjusted_move"],
                    }
                    for w in windows
                ],
                "mark": {
                    "x": largest["date"],
                    "y": largest["close"],
                    "move": largest["move"],
                    "label": f"Largest day, {_pct(largest['move'])} on {largest['date']}",
                },
                "counts": {
                    "days": r["days"],
                    "largest_move": largest["move"],
                    "largest_move_date": largest["date"],
                    "restated": restated,
                },
            },
        )
    }


def _taxonomy_figures(r: dict[str, Any], tickers: Mapping[str, Any] | None) -> dict[str, Any]:
    how = r["how"]
    unresolved = r["unresolved"]
    prior = how["prior_fills"] + how["prior_overrides"]
    reachable = []
    for row in (tickers or {}).get("seed_absent") or []:
        if row["ticker"] in unresolved and row["source"] == "former-ticker index":
            reachable.append(row["ticker"])
    note = ""
    if reachable:
        note = (
            f" {_join(reachable)} {'resolves' if len(reachable) == 1 else 'resolve'} "
            "through the former-ticker index, but no submissions payload is committed "
            f"for {'it' if len(reachable) == 1 else 'them'}."
        )
    rows = [
        {"label": "SIC code gives the analyst's answer", "value": how["code_agrees"], "role": "model"},
        {"label": "Code maps to no TMT bucket; the prior places it", "value": how["prior_fills"], "role": "model"},
        {"label": "The prior overrules the SIC code", "value": how["prior_overrides"], "role": "model"},
        {
            "label": f"Unresolved: {_join(unresolved)}" if unresolved else "Unresolved",
            "value": how["unresolved"],
            "role": "baseline",
        },
    ]
    verticals = sorted(r["by_vertical"].items(), key=lambda kv: (-kv[1]["classified"], kv[0]))
    never = [VERTICAL_LABELS.get(v, v).lower().replace("it services", "IT services") for v, c in verticals if c["by_code"] == 0]
    if never:
        vertical_title = f"The SIC code alone places no filer in {_join(never, 'or')}"
    else:
        vertical_title = "The SIC code places at least one filer in every sub-vertical"
    return {
        "taxonomy_by_vertical": _figure(
            "dot",
            vertical_title,
            (
                f"Filers per sub-vertical: {r['classified']} of {r['seed']} seed filers "
                f"classified by tmt_universe, {how['unresolved']} unresolved. Each row "
                "gives the classified count and how many of those the SIC code alone "
                f"would have placed there; submissions_tmt.json, {r['payloads']} payloads."
            ),
            {
                "rows": [
                    {
                        "label": VERTICAL_LABELS.get(v, v),
                        "values": {"classified": c["classified"], "by_code": c["by_code"]},
                    }
                    for v, c in verticals
                ],
                "series": [
                    {"key": "classified", "name": "Classified", "role": "model"},
                    {"key": "by_code", "name": "SIC code alone agrees", "role": "baseline"},
                ],
                "format": "int",
                "gapLabel": "Placed by the prior",
                "gapFormat": "int",
                "zero": True,
                "labelHeader": "Sub-vertical",
                "counts": {
                    "seed": r["seed"],
                    "classified": r["classified"],
                    "unresolved": unresolved,
                },
            },
        ),
        "taxonomy_how": _figure(
            "hbar",
            (
                f"The analyst's prior, not the SIC code, places {prior} of the "
                f"{r['classified']} classified filers"
            ),
            (
                f"Seed filers by how they got a sub-vertical, {r['seed']} in all, with "
                "no business text read; submissions_tmt.json." + note
            ),
            {
                "rows": rows,
                "format": "int",
                "valueLabel": "Filers",
                "labelHeader": "How it was placed",
                "labels": "all",
                "roleLabels": {"model": "Classified", "baseline": "Unresolved"},
                "counts": {
                    "seed": r["seed"],
                    "classified": r["classified"],
                    "unresolved_tickers": unresolved,
                    **how,
                },
            },
        ),
    }


_LOSS_LABELS = {
    "group": "Went down with a group the encoder cannot use",
    "universe": "Peer outside the candidate universe",
    "encodable": "Peer has no features or text at the panel date",
    "self": "Filer named itself",
}
_LOSS_TITLES = {
    "group": "Most lost peers go down with a whole group, not one at a time",
    "universe": "Most lost peers never entered the candidate universe",
    "encodable": "Most lost peers had no features or text at the panel date",
    "self": "Most lost peers are filers naming themselves",
}
_GROUP_LOSS_LABELS = {
    "filer_not_encodable": "Filer has no features or text at its panel date",
    "before_panel": "Filed before the first panel date",
    "not_usable": "No filing date or fiscal year",
    "no_peer_kept": "No named peer could be encoded",
}
_GROUP_LOSS_CLAUSES = {
    "filer_not_encodable": "the filer could not be encoded",
    "before_panel": "the group predates the panel",
    "not_usable": "the group has no filing date or fiscal year",
    "no_peer_kept": "no named peer could be encoded",
}

_GROUP_NONE = {
    "filer_not_encodable": "has a filer that cannot be encoded",
    "before_panel": "predates the panel",
    "not_usable": "lacks a filing date or fiscal year",
    "no_peer_kept": "keeps no encodable peer",
}


def _peer_figures(r: dict[str, Any]) -> dict[str, Any]:
    lost_groups = r["groups_recorded"] - r["groups_kept"]
    losses = {k: v for k, v in r["lost"].items() if v}
    largest = max(losses, key=losses.get) if losses else None
    rows = [{"label": "Became a training pair", "value": r["pairs"], "role": "model"}]
    rows += [
        {"label": _LOSS_LABELS[k], "value": v, "role": "baseline"}
        for k, v in sorted(losses.items(), key=lambda kv: -kv[1])
    ]
    group_rows = sorted(
        ((k, v) for k, v in r["groups_dropped"].items() if v), key=lambda kv: (-kv[1], kv[0])
    )
    biggest = group_rows[:2]
    never = [_GROUP_NONE[k] for k, v in r["groups_dropped"].items() if not v]
    if not lost_groups:
        groups_title = f"All {r['groups_recorded']} disclosed peer groups reach the encoder"
    else:
        groups_title = (
            f"{lost_groups} of {r['groups_recorded']} peer groups are lost: "
            + _join(f"{v} because {_GROUP_LOSS_CLAUSES[k]}" for k, v in biggest)
        )
    return {
        "peer_label_counts": _figure(
            "tiles",
            (
                f"The encoder learns from {r['groups_kept']} of {r['groups_recorded']} "
                f"disclosed peer groups, filed by {r['filers_kept']} of "
                f"{r['filers_recorded']} filers"
            ),
            "",
            {
                "tiles": [
                    {
                        "label": "Peer groups kept",
                        "value": r["groups_kept"],
                        "format": "int",
                        "sub": f"of {r['groups_recorded']} recorded from proxy filings",
                    },
                    {
                        "label": "Filers kept",
                        "value": r["filers_kept"],
                        "format": "int",
                        "sub": f"of {r['filers_recorded']} that disclosed a group",
                    },
                    {
                        "label": "Training pairs",
                        "value": r["pairs"],
                        "format": "int",
                        "sub": (
                            f"of {r['named_peers']:,} named peers, both legs encodable "
                            f"at one of {r['panel_dates']} panel dates"
                        ),
                    },
                ],
                "counts": {
                    k: r[k]
                    for k in (
                        "groups_recorded",
                        "filers_recorded",
                        "groups_kept",
                        "filers_kept",
                        "named_peers",
                        "pairs",
                    )
                },
            },
        ),
        "peer_groups_lost": _figure(
            "hbar",
            groups_title,
            (
                f"Disclosed peer groups build_dataset drops, by the first check each "
                f"fails; {r['groups_recorded']} recorded, peer_groups_tmt.json against "
                f"{r['panel_dates']} panel dates."
                + (f" No group is lost because it {_join(never, 'or')}." if never else "")
            ),
            {
                "rows": [
                    {"label": _GROUP_LOSS_LABELS[k], "value": v, "role": "baseline"}
                    for k, v in group_rows
                ],
                "format": "int",
                "valueLabel": "Peer groups",
                "labelHeader": "Why the group is lost",
                "labels": "all",
                "counts": {"lost": lost_groups, **r["groups_dropped"]},
            },
        ),
        "peer_label_funnel": _figure(
            "hbar",
            _LOSS_TITLES[largest] if largest else "No named peer is lost",
            (
                f"Where the {r['named_peers']:,} peers named in {r['groups_recorded']} "
                "disclosed groups end up, each counted once; peer_groups_tmt.json"
            ),
            {
                "rows": rows,
                "format": "int",
                "valueLabel": "Named peers",
                "labelHeader": "Outcome",
                "labels": "all",
                "roleLabels": {"model": "Kept", "baseline": "Lost"},
                "counts": {"pairs": r["pairs"], **r["lost"]},
            },
        ),
    }


def _ticker_figures(r: dict[str, Any]) -> Built:
    index_rung = r["rungs"].get("former-ticker index", 0)
    tiles = [
        {
            "label": "Departed registrants reached by old ticker",
            "value": index_rung,
            "format": "int",
            "sub": (
                f"of {r['seeds']}, through the former-ticker index's {r['symbols']} "
                f"symbols over {r['registrants']} registrants, each read off the "
                "registrant's own cover page"
            ),
        },
        {
            "label": "Reachable only by CIK",
            "value": r["cik_only"],
            "format": "int",
            "sub": (
                f"of {r['seeds']}: " + _join(r["cik_only_reasons"])
                if r["cik_only_reasons"]
                else "every departed registrant has a symbol"
            ),
        },
    ]
    absent = r.get("seed_absent")
    hopeless: list[str] = []
    if absent:
        by_source: dict[str, list[dict]] = defaultdict(list)
        for row in absent:
            by_source[row["source"]].append(row)
        parts = []
        for row in by_source.get("former-ticker index", []):
            parts.append(
                f"{row['ticker']} resolves through the former-ticker index to "
                f"{display_name(row['name'], row['ticker'])}"
            )
        for source in sorted(s for s in by_source if s not in ("former-ticker index", "unresolved")):
            parts.append(f"{_join(x['ticker'] for x in by_source[source])} through the {source}")
        hopeless = [x["ticker"] for x in by_source.get("unresolved", [])]
        if hopeless:
            parts.append(f"{_join(hopeless)} through no rung but an explicit CIK")
        tiles.append(
            {
                "label": "Seed tickers the SEC file had dropped",
                "value": len(absent),
                "format": "int",
                "sub": f"of {r['seed_universe']}: " + "; ".join(parts),
            }
        )
    shared = r["shared_symbols"]
    if shared:
        tiles.append(
            {
                "label": "Symbols the index gives to two registrants",
                "value": len(shared),
                "format": "int",
                "sub": "; ".join(
                    f"{s['symbol']}: an undated lookup reaches the later registrant, so "
                    + (
                        "the earlier one needs a run dated inside its history"
                        if s["reached_only_dated"] == 1
                        else f"the {s['reached_only_dated']} earlier ones need a run dated inside their history"
                    )
                    for s in shared
                ),
            }
        )

    title = f"{index_rung} of {r['seeds']} departed registrants resolve by their old ticker"
    if hopeless:
        title += f", and {len(hopeless)} seed {_plural(len(hopeless), 'ticker')} by nothing but a CIK"

    offline = r["seed_offline"]
    split = _join(
        f"{n} {'unresolved' if source == 'unresolved' else 'by the ' + source}"
        for source, n in sorted(offline["rungs"].items(), key=lambda kv: -kv[1])
    )
    why = (
        f"The committed SEC ticker file is a pruned copy of {offline['ticker_file_registrants']} "
        f"registrants, trimmed for the peer-label tests, and carries "
        f"{offline['seed_in_ticker_file']} of the {r['seed_universe']} seed tickers. "
        f"Resolved against it, the seed universe splits {split}, but "
        f"{offline['unresolved_with_payload']} of the unresolved seeds carry a "
        "submissions payload that was fetched through the live file, so that split "
        "measures the pruning and not resolve_ticker. The live file cannot be read "
        "offline. The seeds the live file had dropped are resolved above."
    )
    figure = _figure(
        "tiles",
        title,
        "",
        {
            "tiles": tiles,
            "counts": {
                "seeds": r["seeds"],
                "symbols": r["symbols"],
                "registrants": r["registrants"],
                "rungs": r["rungs"],
                "cik_only": r["cik_only"],
                "shared_symbols": [s["symbol"] for s in shared],
                "seed_absent": absent,
                "seed_offline": offline,
            },
            "near": [REFUSED_SEED_RUNGS],
        },
    )
    return {"ticker_reach": figure}, [{"what": REFUSED_SEED_RUNGS, "why": why}]


_TOPIC_NAMES = {
    "debt": "Debt ladder coverage",
    "splits": "Stock split consistency",
    "closes": "Nvidia's split-adjusted closes",
    "taxonomy": "Sub-vertical taxonomy",
    "peers": "Where peer labels are lost",
    "tickers": "Ticker resolution",
}


def _usable(result: Any) -> bool:
    return isinstance(result, Mapping) and "refused" not in result


def shape(
    results: Mapping[str, Any],
    names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Figures, takeaway and refusals from the checks' results. Pure.

    Each result is a dict, or ``{"refused": reason}`` for a check that could
    not be computed, which becomes a refusal naming the check rather than a
    figure. A check that was not run at all is refused the same way. Figures
    come back in page order.
    """
    names = names or {}
    made: dict[str, Any] = {}
    refusals: list[dict[str, str]] = []
    builders: dict[str, Callable[[dict[str, Any]], Any]] = {
        "debt": lambda r: _debt_figures(r, names),
        "splits": _split_figures,
        "closes": _closes_figures,
        "taxonomy": lambda r: _taxonomy_figures(
            r, results.get("tickers") if _usable(results.get("tickers")) else None
        ),
        "peers": _peer_figures,
        "tickers": _ticker_figures,
    }
    for key, build in builders.items():
        result = results.get(key)
        if not _usable(result):
            why = (result or {}).get("refused") if isinstance(result, Mapping) else None
            refusals.append({"what": _TOPIC_NAMES[key], "why": why or "the check was not run"})
            continue
        out = build(result)
        figures, extra = out if isinstance(out, tuple) else (out, [])
        if tuple(figures) != FIGURES_BY_RESULT[key]:
            raise ValueError(f"{key} built figures {sorted(figures)}, not {FIGURES_BY_RESULT[key]}")
        made.update(figures)
        refusals.extend(extra)
    figures = {fid: made[fid] for fid in FIGURE_IDS if fid in made}

    clauses = []
    debt = results.get("debt")
    if _usable(debt):
        lt = next(leg for leg in debt["legs"] if leg["leg"] == "long_term")
        clauses.append(
            f"the debt ladders reach {lt['reached']} of the {lt['reporting']} filers "
            "with material long-term debt"
        )
    splits = results.get("splits")
    if _usable(splits):
        checked = sum(f["checked"] for f in splits["filers"])
        bad = sum(f["inconsistent"] for f in splits["filers"])
        clauses.append(
            f"every one of {checked} share-count periods filed more than once agrees "
            "after split adjustment"
            if bad == 0
            else f"{bad} of {checked} share-count periods filed more than once still "
            "disagree after split adjustment"
        )
    peers = results.get("peers")
    if _usable(peers):
        clauses.append(
            f"{peers['groups_recorded'] - peers['groups_kept']} of "
            f"{peers['groups_recorded']} disclosed peer groups never reach the encoder"
        )
    if clauses:
        body = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + ", and " + clauses[-1]
        takeaway = body[0].upper() + body[1:] + "."
    else:
        takeaway = "No data-layer check could be computed from the committed fixtures."
    return {
        "status": "ok",
        "takeaway": takeaway,
        "refusals": refusals,
        "headline": None,
        "figures": figures,
    }


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


@contextmanager
def _sourced(ctx, figures: tuple[str, ...], entry_points: list[str], inputs: list[str]):
    """Record every figure a result becomes against every entry point behind it.

    The first entry point is the one the figure's footer names. The rest are
    the model code the figure measures, recorded so the result cache is keyed
    on their source as well: a change to ``techval.tags`` must miss the cached
    debt figure even though the counting lives in this module.
    """
    with ExitStack() as stack:
        for figure in figures:
            for entry in reversed(entry_points):
                stack.enter_context(ctx.record(figure, entry, inputs))
        yield


def _load_json(ctx, name: str) -> Any:
    return json.loads(ctx.input(name).read_text(encoding="utf-8"))


def collect(ctx) -> dict:
    from techval.edgar import CompanyFacts
    from techval.tmt.taxonomy import SEED

    results: dict[str, Any] = {}

    census = _load_json(ctx, CENSUS)
    submissions = _load_json(ctx, SUBMISSIONS)["companies"]
    names = {t: str(p.get("name") or t) for t, p in submissions.items()}

    with _sourced(
        ctx,
        ("debt_reach",),
        [
            f"{_HERE}.debt_ladder_reach",
            "techval.tags.DEBT_NONCURRENT",
            "techval.tags.DEBT_CURRENT",
            "techval.tags.DEBT_COMBINED",
            "techval.tags.DEBT_CROSSCHECK",
        ],
        [CENSUS, SUBMISSIONS],
    ):
        results["debt"] = debt_ladder_reach(census)

    payloads = _load_json(ctx, SHARE_BASIS)
    payloads = {k: v for k, v in payloads.items() if not k.startswith("_")}
    payloads["CRWD"] = _load_json(ctx, CRWD_FACTS)
    with _sourced(
        ctx,
        ("split_counts", "split_oracle"),
        [f"{_HERE}.split_consistency", "techval.edgar.CompanyFacts._split_adjust"],
        [SHARE_BASIS, CRWD_FACTS],
    ):
        results["splits"] = split_consistency(payloads)

    try:
        with _sourced(
            ctx,
            ("nvda_closes",),
            [f"{_HERE}.nvda_split_evidence", "techval.edgar.CompanyFacts._split_brackets"],
            [CLOSES, SHARE_BASIS],
        ):
            brackets = (
                CompanyFacts(payloads["NVDA"], "NVDA")._split_brackets()
                if "NVDA" in payloads
                else []
            )
            results["closes"] = nvda_split_evidence(
                read_closes(ctx.input(CLOSES), "NVDA"), brackets
            )
    except _Unavailable as exc:
        results["closes"] = {"refused": str(exc)}

    with _sourced(
        ctx,
        ("taxonomy_by_vertical", "taxonomy_how"),
        ["techval.tmt.taxonomy.tmt_universe", "techval.tmt.taxonomy.classify"],
        [SUBMISSIONS],
    ):
        results["taxonomy"] = taxonomy_placement(submissions, ctx.assumptions)

    with _sourced(
        ctx,
        ("peer_label_counts", "peer_groups_lost", "peer_label_funnel"),
        [
            "techval.ml.encoder.build_dataset",
            "techval.commands_peers._load_groups",
            "techval.commands_peers._load_panel",
            "techval.commands_peers._load_corpora",
        ],
        [PEER_GROUPS, PEER_PANEL, PEER_TEXT],
    ):
        from techval.commands_peers import _load_corpora, _load_groups, _load_panel

        results["peers"] = peer_label_losses(
            _load_groups(ctx.input(PEER_GROUPS)),
            _load_panel(ctx.input(PEER_PANEL)),
            _load_corpora(ctx.input(PEER_TEXT)),
        )

    with _sourced(
        ctx,
        ("ticker_reach",),
        [
            "techval.edgar.EdgarClient.resolve_ticker",
            "techval.former_tickers.resolve",
            "techval.tmt.taxonomy.SEED",
        ],
        [FORMER_INDEX, MNA_UNIVERSE, TICKER_FILE, SUBMISSIONS],
    ):
        results["tickers"] = ticker_reach(
            _load_json(ctx, FORMER_INDEX),
            _load_json(ctx, MNA_UNIVERSE),
            _load_json(ctx, TICKER_FILE)["data"],
            SEED,
            submissions,
        )

    return shape(results, names)
