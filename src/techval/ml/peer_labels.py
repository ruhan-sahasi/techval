"""Compensation peer groups, read out of DEF 14A proxy statements.

Every learned peer model needs ground truth, and the weakest thing a peer model
can be built on is the author's own opinion of which companies are comparable:
it makes the model a restatement of the prior, and the backtest a tautology.
This module takes the labels from the filings instead. A registrant's proxy
discloses the companies its compensation committee benchmarked pay against,
chosen by an independent consultant under stated revenue and market
capitalisation bands. That is a disclosed, dated, auditable assertion of
comparability by the people with the most to lose from getting it wrong, and it
is the only peer set in the public record that carries a date.

**What a label means, exactly.** A pair ``(filer, peer, fiscal_year)`` means the
filer told the SEC that its board used ``peer`` when setting pay for that fiscal
year. It does not mean the two trade on the same multiple, and a compensation
peer group is selected partly for competition for executive talent, so it skews
toward companies of similar size and toward the filer's own labour market. Those
biases are real and they are the reason ``selection_criteria`` is captured and
carried: a reader who knows the bands knows what the label is worth.

**Why the parse is fiddly.** The SEC's rendered proxy is HTML, and the peer list
is a table. Stripping markup joins the cells with no separator, so the list
arrives as one run of proper nouns:

    AtlassianHubSpotThe Trade DeskCloudflareMongoDBVeeva SystemsCrowdStrike...

Three formats occur in practice and all three are handled: that concatenated
run, one name per line where the renderer kept the row breaks, and a name
followed by its own ticker, ``ANSYS [ANSS]DocuSign [DOCU]`` or ``Atlassian
(TEAM)``. The annotated form is taken at face value once the symbol is confirmed
to exist; the other two are segmented against the SEC's own name index.

**The trap in the segmentation, and how it is avoided.** The obvious approach
lowercases the blob, strips corporate suffix words, and runs a greedy dictionary
match. It produces silent garbage. Stripping suffixes from the whole blob at
once lets a suffix word from one name bridge into the next, and the leftover
letters spell other companies: ``Veeva Systems`` followed by ``CrowdStrike``
leaves the unconsumed word ``systems``, inside which ``stem`` is a real ticker.
Three rules kill it. A match may only begin where a name can begin, which is a
capital letter or a delimiter, and ``stem`` starts at the lowercase ``t`` inside
``Systems``. Every candidate span is normalised in two forms, as written and
with its corporate suffix removed, and both are looked up, so ``Veeva Systems``
matches the registrant ``VEEVA SYSTEMS INC`` across its full length instead of
leaving a tail behind. And the longest match at a boundary wins, so a thirteen
character match is preferred to the five character one inside it.

**Ambiguity is never guessed.** Roughly eight thousand distinct registrant names
normalise into a space where collisions happen. Where a span matches more than
one company the resolver prefers a candidate already named elsewhere in the same
proxy, then one inside the universe being collected, and if the tie survives it
records the span in ``unresolved`` together with both candidates. A wrong label
is worse than a missing one: a missing label costs one training row, a wrong one
teaches the model a relationship nobody asserted.

**Determinism.** Nothing here is random, so nothing consumes
``assumptions.ml.random_seed``. Every tie is broken by an explicit rule stated
in the code, and every list the module returns is sorted, so two runs over the
same filings agree byte for byte.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from techval.config import Assumptions
from techval.errors import DataSourceError, MissingDataError

# Bumped when the parser changes in a way that would alter a cached result, so a
# stale cache from an older parser is re-read rather than trusted.
PARSER_VERSION = 4

# Plausibility band for a disclosed compensation peer group. Consultants build
# these to give a defensible median, which needs enough names to be stable and
# few enough to stay comparable; in practice every one observed sits inside it.
# A parse returning three names or eighty has found a fragment or run past the
# end of the table, and either way it is a parse failure, not a small peer group.
MIN_PLAUSIBLE_PEERS = 8
MAX_PLAUSIBLE_PEERS = 30


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

# Words a registrant appends to its trade name. Removing them from the end of a
# name is what lets the proxy's "Veeva Systems" meet the SEC file's "VEEVA
# SYSTEMS INC", and the proxy's "Palo Alto Networks" meet "Palo Alto Networks
# Inc". They are only ever removed from the end, never from the middle, because
# "Cognizant Technology Solutions" is not "Cognizant Solutions".
_SUFFIX_TOKENS = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "holdings",
        "holding",
        "technologies",
        "technology",
        "systems",
        "software",
        "networks",
        "plc",
        "ltd",
        "limited",
        "llc",
        "lp",
        "nv",
        "sa",
        "ag",
        "se",
        "group",
        "common",
        "stock",
        "shares",
        "sponsored",
        "adr",
        "ads",
    }
)

# Only ever removed from the front. "The Trade Desk" in a proxy is "Trade Desk,
# Inc." in the SEC file, and the article is the whole difference.
_ARTICLE_TOKENS = frozenset({"the"})

# "Class A" and "Cl B" are removed as a pair, never a letter on its own.
_CLASS_TOKENS = frozenset({"class", "cl", "ser", "series"})
_CLASS_LETTERS = frozenset({"a", "b", "c", "d"})

# English words that open a sentence and are never a company name. When the list
# runs straight into the prose that follows it, as in "SnowflakeThe compensation
# committee reviews", the trailing fragment is one of these, and dropping it is
# what leaves "Snowflake" whole.
_SENTENCE_OPENERS = frozenset(
    {
        "a",
        "additionally",
        "after",
        "all",
        "although",
        "an",
        "and",
        "approximately",
        "as",
        "at",
        "based",
        "because",
        "both",
        "but",
        "by",
        "during",
        "each",
        "for",
        "from",
        "however",
        "if",
        "in",
        "it",
        "its",
        "no",
        "none",
        "of",
        "or",
        "our",
        "please",
        "set",
        "since",
        "source",
        "such",
        "that",
        "the",
        "their",
        "then",
        "these",
        "they",
        "this",
        "those",
        "to",
        "under",
        "using",
        "we",
        "when",
        "which",
        "while",
        "who",
        "with",
    }
)

# Lowercase fragments that genuinely occur inside a company name, so that seeing
# one does not mean the prose has resumed. Deliberately tiny: every addition
# widens the window in which a run of prose can be mistaken for peer names.
_NAME_PARTICLES = frozenset({"com", "ai", "io"})

_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿­]")
_PAGE_BREAK = re.compile(r"\s*\d{0,4}\s*table\s+of\s+contents\s*", re.I)
_PAGE_MARKER = re.compile(r"\s*page\s+\d{1,4}\s*\|", re.I)
# The running header and footer a proxy repeats at every page break, which the
# renderer drops into the middle of whatever sentence or table straddles the
# break: "...for our NEOs:" then "34" and "2026 Proxy Statement" and the
# registrant name and the section heading, and only then the first peer.
_RUNNING_HEADER = re.compile(
    r"\s*\d{0,4}\s*(?:19|20)\d{2}\s+proxy\s+statement\s*\|?"
    r"|\s*proxy\s+statement\s*\|"
    r"|\s*compensation\s+discussion\s+and\s+analysis",
    re.I,
)
_WORD = re.compile(r"[0-9a-z]+", re.I)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_DOTTED = re.compile(r"\b(?:[A-Za-z]\.){2,}")

# A ticker printed beside its company, which several filers do: "ANSYS [ANSS]",
# "Atlassian (TEAM)". Parentheses need two letters or more, because a single
# parenthesised capital is a footnote marker far more often than a symbol.
_TICKER_ANNOTATION = re.compile(
    r"\[\s*([A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)\s*\]|\(\s*([A-Z]{2,5}(?:[.\-][A-Z]{1,2})?)\s*\)"
)

# Where a company name is allowed to begin inside a run of joined table cells:
# at a capital, or immediately after a delimiter. The cell text itself never
# starts lowercase, which is exactly what makes the rule safe.
_DELIMITERS = " \t\r\n,;:|/()[]\u2013\u2014\u2022\u25aa\u25cf\u00b7"


def _clean(text: str) -> str:
    """Strip the artefacts that a rendered proxy leaves in its own prose.

    Zero-width characters sit between table cells, and the page furniture
    ("Table of Contents", "Page 34 |", a bare page number followed by "2026
    Proxy Statement" and the running section heading) is injected mid-sentence
    at every page break, so a peer list can arrive split by a running header.
    Okta's table is cut exactly there, and the furniture lands between the
    introducing colon and the first company name, where it both hides the
    anchor and would be segmented as though it were a peer. Removing it before
    anything is located means a name broken across a page boundary is rejoined
    rather than lost.
    """
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\xa0", " ")
    for dash in ("‘", "’", "ʼ"):
        text = text.replace(dash, "'")
    for quote in ("“", "”"):
        text = text.replace(quote, '"')
    for thin in (" ", " ", " ", " ", " "):
        text = text.replace(thin, " ")
    text = _PAGE_MARKER.sub(" ", text)
    text = _PAGE_BREAK.sub(" ", text)
    text = _RUNNING_HEADER.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n[ \t]*", "\n", text)


def _tokens(raw: str) -> list[str]:
    """Lowercase word tokens, splitting camel case only when there is no space.

    "Veeva Systems" already carries its own word break. "CrowdStrikeHoldings",
    arriving from joined cells, does not, and its suffix cannot be recognised
    without one. Splitting on the lower-to-upper transition supplies it, and
    restricting that to spaceless spans keeps "CrowdStrike" intact wherever the
    renderer preserved the spacing.
    """
    if not raw.strip():
        return []
    # "Elastic N.V." has to reach one token "nv" for the suffix rule to see it,
    # and "U.S." has to stay one token so it is not read as two initials.
    raw = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), raw)
    source = raw if " " in raw.strip() else _CAMEL.sub(" ", raw)
    return [m.group(0).lower() for m in _WORD.finditer(source)]


def _core(tokens: Sequence[str]) -> list[str]:
    """Tokens with the leading article and the trailing corporate suffix removed.

    A share-class letter is stripped only as part of the pair "class a", never
    on its own. Stripping a bare trailing letter looks harmless and is not: in a
    run of joined cells it lets a candidate span reach one character into the
    next company and still match, so "...BILL Holdings" is read as a name ending
    in "B" plus the fragment "ILL Holdings", and a real peer is lost to a name
    that never appeared.
    """
    out = list(tokens)
    while out and out[0] in _ARTICLE_TOKENS:
        out = out[1:]
    while len(out) > 1:
        if len(out) > 2 and out[-2] in _CLASS_TOKENS and out[-1] in _CLASS_LETTERS:
            out = out[:-2]
            continue
        if out[-1] in _SUFFIX_TOKENS:
            out = out[:-1]
            continue
        break
    return out


# The Commission's ticker file appends the state of incorporation to a
# registrant's title, as a slash and a two-letter code, sometimes closed by a
# second slash: "APPLIED MATERIALS INC /DE", "QUALCOMM INC/DE", "CORNING INC
# /NY", "CHARTER COMMUNICATIONS, INC. /MO/". It is not part of the name and no
# proxy ever writes it. Matched at the end of the string only, and only after a
# slash, so a real name ending in two letters is untouched.
_STATE_OF_INCORPORATION = re.compile(r"/\s*[A-Za-z]{2}\s*/?\s*$")


def normalise_name(raw: str) -> tuple[str, str]:
    """A company name in the two forms every lookup is tried against.

    Returns ``(full, core)``: the name as written with punctuation and spacing
    removed, and the same name with its article and corporate suffix stripped.
    Both are needed on both sides of the match. The proxy writes "Veeva
    Systems" and the SEC file says "VEEVA SYSTEMS INC", which agree only on the
    core; the proxy writes "The Trade Desk" and the file says "Trade Desk,
    Inc.", which again agree only on the core. Keeping the full form as well is
    what stops the stripped form from being the only thing on offer, because a
    stripped form alone leaves the discarded suffix word sitting unconsumed in
    the blob for a shorter ticker to match inside.
    """
    # The state-of-incorporation marker is removed before tokenising rather
    # than added to the suffix list, because "de" and "ny" are ordinary word
    # fragments and a suffix rule that stripped them would also strip the tail
    # of a real name. Left in, it is fatal rather than cosmetic: the marker
    # tokenises into a trailing "de", the suffix stripper stops on it before it
    # reaches "inc", and the core form of "APPLIED MATERIALS INC /DE" comes out
    # as "appliedmaterialsincde", which no proxy's "Applied Materials" can ever
    # meet. Across the live file 291 of 10,407 registrants carry one, and they
    # account for 121 of the 1,205 peer spans this module could not resolve
    # over the seed universe, Applied Materials and Qualcomm among them.
    toks = _tokens(_STATE_OF_INCORPORATION.sub("", raw))
    full = "".join(toks)
    core = "".join(_core(toks))
    # Two characters is the floor, not three, because "F5, Inc." and "F5
    # Networks Inc" only meet at "f5" and a three-character floor would send
    # both to the full form and leave them apart.
    return full, (core if len(core) >= 2 else full)


# --------------------------------------------------------------------------- #
# The name index
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NameCandidate:
    """One registrant a normalised name could refer to.

    ``listing_rank`` is the symbol's position in the Commission's own ticker
    file, which orders a registrant's symbols with the primary listing first.
    It is what decides between two symbols of one registrant; see
    ``_pick_symbol`` for why guessing from the symbol itself does not work. The
    default is a sentinel above any real position, so a candidate built from a
    former name, which the file does not rank, sorts after every ranked one.
    """

    ticker: str
    cik: int
    title: str
    from_full_form: bool
    listing_rank: int = 1 << 30


class NameIndex(dict):
    """Normalised company name to ticker, with the collisions kept visible.

    This is a plain ``dict[str, str]`` for every name that resolves to exactly
    one registrant, and that is all most callers need. Names that collide are
    deliberately absent from the mapping and live in ``ambiguous`` instead, so
    a caller who subscripts the dict can never be handed a guess. Share classes
    are not a collision: ``GOOG`` and ``GOOGL`` are one CIK, so the index picks
    a class by an explicit rule and says so, because the company is what a peer
    label is about.
    """

    def __init__(self) -> None:
        super().__init__()
        self.candidates: dict[str, tuple[NameCandidate, ...]] = {}
        self.ambiguous: dict[str, tuple[str, ...]] = {}
        self.cik_of: dict[str, int] = {}
        self.title_of: dict[str, str] = {}
        self.tickers: frozenset[str] = frozenset()
        self.by_first_token: dict[str, tuple[tuple[frozenset[str], str], ...]] = {}
        self.notes: list[str] = []

    def lookup(self, key: str) -> tuple[NameCandidate, ...]:
        return self.candidates.get(key, ())

    def fuzzy(self, tokens: Sequence[str]) -> tuple[tuple[str, int], ...]:
        """Registrants whose own name sits inside this one, with the extra word count.

        A company renames and the SEC's ticker file carries only the current
        name, so the proxy's "Zoom Video Communications" has to reach the file's
        "Zoom Communications, Inc." The containment is one-directional on
        purpose: the registrant's name must be a subset of the disclosed one,
        never the reverse, because a one-word registrant name is contained in
        half the file and "Apple" would swallow "Apple Hospitality REIT". The
        registrant name must also carry at least two words, for the same reason.
        """
        core = _core(tokens)
        if len(core) < 2:
            return ()
        want = set(core)
        out: list[tuple[str, int]] = []
        for cand_tokens, ticker in self.by_first_token.get(core[0], ()):
            if len(cand_tokens) < 2 or not cand_tokens <= want:
                continue
            extra = len(want) - len(cand_tokens)
            # More than two inserted words is a different company, not a rename.
            if extra > 2:
                continue
            out.append((ticker, extra))
        return tuple(sorted(out, key=lambda p: (p[1], p[0])))


def _pick_symbol(cands: Sequence[NameCandidate]) -> str:
    """One ticker for one registrant that files under several symbols.

    Not a guess about which company is meant: the CIK already settled that.
    The question is only which of the registrant's symbols carries the label,
    and the Commission's own file answers it. ``company_tickers.json`` lists a
    registrant's symbols with the primary listing first, so the first one is
    taken and the symbol's own text decides nothing.

    **This replaced a shortest-symbol rule, and the difference is not
    cosmetic.** "Shortest, then alphabetical" disagrees with the Commission's
    ordering for 224 of the 1,441 registrants that carry more than one symbol,
    and the disagreements are not share classes at all. Comcast files its common
    stock as CMCSA and an exchangeable debenture as CCZ under one CIK, so the
    shorter symbol won and every proxy naming Comcast produced a label pointing
    at a debt security. Prudential lost PRU to the preferred PFH and DTE Energy
    lost DTE to the debenture DTB the same way. The failure is silent twice
    over: the label still resolves, and the ticker it resolves to is absent from
    any equity universe it is then matched against, so a major company simply
    stops appearing as a peer.

    Where no candidate carries a rank, which is the case for a name recovered
    from a filer's former names, the old rule still decides, because a stable
    arbitrary answer is better than an unstable one.
    """
    return sorted(cands, key=lambda c: (c.listing_rank, len(c.ticker), c.ticker))[0].ticker


def build_name_index(
    client: Any,
    former_names_for: Sequence[str] | None = None,
) -> dict[str, str]:
    """The SEC's own ticker file, turned into a company-name resolver.

    ``company_tickers.json`` carries every symbol the Commission knows about,
    which is around ten thousand symbols over eight thousand distinct
    registrant names. Each name is indexed in both normalised forms, so a proxy
    that writes the trade name and a file that writes the legal name meet.

    ``former_names_for`` adds each named ticker's previous registrant names,
    read from its submissions payload. Peer lists are historical by
    construction, so a 2021 proxy names companies under the names they had in
    2021, and the current ticker file has no memory of those. The call is free
    where ``filings`` has already been run for the same ticker, since the
    submissions payload is cached.

    Raises ``DataSourceError`` if the file cannot be read, rather than returning
    a partial index that would silently drop peers.
    """
    from techval.edgar import SEC_TICKERS_URL

    payload = client._get_json(SEC_TICKERS_URL)
    if not payload:
        raise DataSourceError(
            f"{SEC_TICKERS_URL} returned no entries, so no company name can be "
            "resolved to a ticker"
        )

    index = NameIndex()
    by_key: dict[str, dict[str, NameCandidate]] = {}
    tickers: set[str] = set()

    def _add(title: str, ticker: str, cik: int, rank: int = 1 << 30) -> None:
        ticker = ticker.upper().strip()
        if not ticker or not title.strip():
            return
        tickers.add(ticker)
        index.cik_of[ticker] = cik
        index.title_of.setdefault(ticker, title.strip())
        full, core = normalise_name(title)
        for key, is_full in ((full, True), (core, False)):
            if len(key) < 2:
                continue
            slot = by_key.setdefault(key, {})
            prior = slot.get(ticker)
            # A ticker reaching one key through both forms counts as the full
            # form, which is the stronger claim on the name.
            if prior is None or (is_full and not prior.from_full_form):
                slot[ticker] = NameCandidate(ticker, cik, title.strip(), is_full, rank)

    # The Commission lists a registrant's symbols primary listing first, and
    # that order is the only evidence in the file about which symbol is the
    # company's ordinary equity. It is captured here and nowhere else.
    for rank, row in enumerate(payload.values()):
        try:
            _add(str(row["title"]), str(row["ticker"]), int(row["cik_str"]), rank)
        except (KeyError, TypeError, ValueError):
            # One malformed row in the Commission's file is not a reason to
            # refuse the other ten thousand; it is recorded instead.
            index.notes.append(f"skipped a malformed ticker-file row: {row!r}")

    for ticker in sorted(set(former_names_for or ())):
        for title, cik in _former_names(client, ticker):
            _add(title, ticker, cik)

    index.tickers = frozenset(tickers)

    for key in sorted(by_key):
        cands = tuple(sorted(by_key[key].values(), key=lambda c: c.ticker))
        index.candidates[key] = cands
        by_cik = {c.cik for c in cands}
        if len(by_cik) == 1:
            index[key] = _pick_symbol(cands)
        else:
            # Prefer a candidate that owns the name outright over one that only
            # arrives at it after its suffix is stripped.
            full_only = [c for c in cands if c.from_full_form]
            full_ciks = {c.cik for c in full_only}
            if len(full_ciks) == 1:
                index[key] = _pick_symbol(full_only)
            else:
                index.ambiguous[key] = tuple(sorted(c.ticker for c in cands))

    first: dict[str, list[tuple[frozenset[str], str]]] = {}
    for ticker in sorted(tickers):
        title = index.title_of.get(ticker)
        if not title:
            continue
        core = _core(_tokens(title))
        if len(core) < 2:
            continue
        first.setdefault(core[0], []).append((frozenset(core), ticker))
    index.by_first_token = {k: tuple(sorted(v, key=lambda p: p[1])) for k, v in first.items()}
    return index


def _former_names(client: Any, ticker: str) -> list[tuple[str, int]]:
    """Previous registrant names for one ticker, empty if the payload has none."""
    try:
        sub = client.submissions(ticker)
        cik = int(client.ticker_to_cik(ticker))
    except (AttributeError, MissingDataError, DataSourceError, KeyError, TypeError, ValueError):
        return []
    out: list[tuple[str, int]] = []
    for row in sub.get("formerNames") or ():
        name = (row or {}).get("name")
        if isinstance(name, str) and name.strip():
            out.append((name, cik))
    return out


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

# No registrant name in the SEC file is anywhere near this long, and capping the
# candidate span keeps the boundary search linear in practice.
_MAX_NAME_CHARS = 80

# A span inferred from capitalisation alone has to earn its match. Three letters
# inside a run of joined names is noise; at a hard delimiter on both sides the
# filing itself has drawn the boundary and a short name is safe.
_MIN_INFERRED_CHARS = 4
_MIN_DELIMITED_CHARS = 2


def _boundaries(text: str) -> list[int]:
    """Positions where a company name is allowed to start.

    A capital letter, anything just past a delimiter, and the one lowercase case
    that matters: a single lowercase letter in front of a capital, which is how
    "eBay", "iRobot" and "nCino" spell themselves. Without that third rule
    "AdobeeBay" splits into "Adobee" and "Bay" and loses both companies.

    This is also the rule that makes the "stem" inside "Systems" unreachable. It
    begins at a lowercase letter that is followed by another lowercase letter,
    in the middle of a word, so no candidate span can start there.
    """
    out = [0]
    last = len(text) - 1
    for i, ch in enumerate(text):
        if i == 0:
            continue
        if ch.isupper() or ch.isdigit():
            out.append(i)
        elif text[i - 1] in _DELIMITERS and ch not in _DELIMITERS:
            out.append(i)
        elif (
            ch.islower()
            and i < last
            and text[i + 1].isupper()
            and text[i - 1].islower()
        ):
            out.append(i)
    seen: set[int] = set()
    uniq = []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def _resolve_span(
    text: str,
    index: dict[str, str],
    rich: "NameIndex | None",
    prefer_ciks: set[int],
    start: int,
    end: int,
) -> tuple[str | None, tuple[str, ...]]:
    """Resolve one candidate span to a ticker, or report the tie it cannot break.

    Returns ``(ticker, collision)``. A span shorter than the floor for its
    context is not tried at all: inside a run of joined names a three letter
    span is noise, while at a hard delimiter the filing has drawn the boundary
    itself and a short name is safe there.
    """
    raw = text[start:end]
    trimmed = raw.strip(_DELIMITERS)
    floor = _MIN_DELIMITED_CHARS if _delimited(text, start, end) else _MIN_INFERRED_CHARS
    if len(trimmed) < floor:
        return None, ()
    full, core = normalise_name(raw)
    if rich is None:
        return (index.get(full) or index.get(core)), ()
    cands: list[NameCandidate] = []
    for key in (full, core):
        for c in rich.lookup(key):
            if c not in cands:
                cands.append(c)
    return _resolve_candidates(cands, prefer_ciks)


def _name_starts_at(
    text: str,
    index: dict[str, str],
    rich: "NameIndex | None",
    prefer_ciks: set[int],
    bounds: Sequence[int],
    n: int,
    start: int,
) -> bool:
    """Whether any candidate span beginning at ``start`` resolves to a registrant.

    One step of lookahead, used to decide whether a shorter reading of the
    preceding span is the right one. It answers the only question that
    separates a suffix word belonging to the name just read from the opening
    letters of the name that follows it.
    """
    ends = [e for e in bounds if start < e <= start + _MAX_NAME_CHARS]
    if n - start <= _MAX_NAME_CHARS:
        ends.append(n)
    for e in sorted(set(ends), reverse=True):
        ticker, _ = _resolve_span(text, index, rich, prefer_ciks, start, e)
        if ticker is not None:
            return True
    return False


def _delimited(text: str, start: int, end: int) -> bool:
    before = start == 0 or text[start - 1] in _DELIMITERS
    after = end >= len(text) or text[end] in _DELIMITERS
    return before and after


def _resolve_candidates(
    cands: Sequence[NameCandidate],
    prefer_ciks: set[int],
) -> tuple[str | None, tuple[str, ...]]:
    """One ticker, or the colliding tickers when the tie cannot be broken honestly."""
    if not cands:
        return None, ()
    ciks = {c.cik for c in cands}
    if len(ciks) == 1:
        return _pick_symbol(cands), ()
    preferred = [c for c in cands if c.cik in prefer_ciks]
    if len({c.cik for c in preferred}) == 1:
        return _pick_symbol(preferred), ()
    full_only = [c for c in cands if c.from_full_form]
    if len({c.cik for c in full_only}) == 1:
        return _pick_symbol(full_only), ()
    return None, tuple(sorted(c.ticker for c in cands))


@dataclass
class _Segmentation:
    """One reading of a blob: what resolved, what did not, and how it was read."""

    tickers: list[str]
    leftovers: list[str]
    spans: list[str]
    annotated: bool


def _segment(
    blob: str,
    index: dict[str, str],
    prefer_ciks: set[int],
) -> _Segmentation:
    """One pass of longest-match segmentation over a run of joined names."""
    text = blob.strip()
    if not text:
        return _Segmentation([], [], [], False)

    rich = index if isinstance(index, NameIndex) else None
    bounds = _boundaries(text)
    bound_set = set(bounds) | {len(text)}
    n = len(text)

    tickers: list[str] = []
    leftovers: list[str] = []
    spans: list[str] = []
    gap_start: int | None = None

    def _flush_gap(upto: int) -> None:
        nonlocal gap_start
        if gap_start is None:
            return
        raw = text[gap_start:upto].strip(_DELIMITERS + ".")
        gap_start = None
        toks = _tokens(raw)
        if not toks:
            return
        # A trailing sentence opener is the prose running into the last name,
        # not a company, and a span made only of suffix words or of a year is
        # table furniture. None of it belongs in the unresolved list, where it
        # would depress the confidence of a group that parsed correctly.
        if all(
            t in _SENTENCE_OPENERS or t in _SUFFIX_TOKENS or t.isdigit() for t in toks
        ):
            return
        if len(raw) < _MIN_DELIMITED_CHARS:
            return
        spans.append(raw)
        leftovers.append(raw)

    i = 0
    bi = 0
    while i < n:
        while bi < len(bounds) and bounds[bi] < i:
            bi += 1
        if bi >= len(bounds):
            break
        p = bounds[bi]
        if p > i and gap_start is None:
            gap_start = i

        ends = [e for e in bounds[bi + 1 :] if e - p <= _MAX_NAME_CHARS]
        if len(text) - p <= _MAX_NAME_CHARS:
            ends.append(n)
        ends = sorted(set(ends), reverse=True)

        hit_end: int | None = None
        hit_ticker: str | None = None
        hit_collision: tuple[str, ...] = ()

        for e in ends:
            ticker, collision = _resolve_span(text, index, rich, prefer_ciks, p, e)
            if ticker is not None:
                hit_end, hit_ticker = e, ticker
                break
            if collision and hit_collision == ():
                hit_end, hit_collision = e, collision
                # Keep looking: a longer span may still resolve cleanly.

        if hit_ticker is not None and hit_end is not None:
            # A longer span landing on the same registrant won nothing by being
            # longer: those characters were absorbed by suffix stripping rather
            # than matched, and they belong to the next company.
            # "Netflix, Inc.NV" normalises to "netflix", because the "NV" that
            # opens NVIDIA is itself a suffix word, so it resolves to NFLX
            # exactly as "Netflix, Inc." does while eating two letters of
            # NVIDIA. The boundary rules cannot see it, since "NV" is
            # capitalised and a name may legally end there. Giving the
            # characters back is right only when a registrant actually starts at
            # the seam that opens, which is what separates this from "Veeva
            # Systems": there "Systems" begins nothing and belongs to Veeva.
            for e in sorted(x for x in ends if x < hit_end):
                shrunk, _ = _resolve_span(text, index, rich, prefer_ciks, p, e)
                if shrunk != hit_ticker:
                    continue
                if _name_starts_at(text, index, rich, prefer_ciks, bounds, n, e):
                    hit_end = e
                    break

        if hit_ticker is None and rich is not None and hit_collision == ():
            best: tuple[int, int, str, int] | None = None
            for e in ends:
                raw = text[p:e]
                trimmed = raw.strip(_DELIMITERS)
                floor = (
                    _MIN_DELIMITED_CHARS
                    if _delimited(text, p, e)
                    else _MIN_INFERRED_CHARS
                )
                if len(trimmed) < floor:
                    continue
                hits = rich.fuzzy(_tokens(raw))
                if len(hits) != 1:
                    continue
                ticker, extra = hits[0]
                key = (extra, -(e - p), ticker, e)
                if best is None or key < best:
                    best = key
            if best is not None:
                hit_end, hit_ticker = best[3], best[2]

        if hit_ticker is not None and hit_end is not None:
            _flush_gap(p)
            span = text[p:hit_end].strip(_DELIMITERS)
            spans.append(span)
            tickers.append(hit_ticker)
            i = hit_end
            continue
        if hit_collision and hit_end is not None:
            _flush_gap(p)
            span = text[p:hit_end].strip(_DELIMITERS)
            spans.append(span)
            leftovers.append(f"{span} (ambiguous: {', '.join(hit_collision)})")
            i = hit_end
            continue

        if gap_start is None:
            gap_start = p
        bi += 1
        i = bounds[bi] if bi < len(bounds) else n

    _flush_gap(n)
    return _Segmentation(tickers, leftovers, spans, False)


def _annotations(blob: str, known: frozenset[str] | None) -> list[tuple[int, int, str]]:
    """Ticker symbols printed beside a company name, with their extents."""
    out: list[tuple[int, int, str]] = []
    for m in _TICKER_ANNOTATION.finditer(blob):
        symbol = (m.group(1) or m.group(2) or "").upper()
        if not symbol:
            continue
        out.append((m.start(), m.end(), symbol))
    if known is None:
        return out
    # At least two symbols must be real for the list to be read this way. One
    # stray parenthesised capital word is a footnote, not a convention.
    if sum(1 for _, _, s in out if s in known) < 2:
        return []
    return out


def _segment_annotated(
    blob: str,
    index: dict[str, str],
    prefer_ciks: set[int],
) -> _Segmentation | None:
    """Read a list that prints each peer's own ticker beside its name.

    Some proxies render "ANSYS [ANSS]DocuSign [DOCU]", others "Atlassian
    (TEAM)". The symbol is the registrant's own statement of who it means, so it
    is taken in preference to matching the name, but only after the symbol is
    confirmed to exist in the SEC ticker file. A symbol that is not a live
    registrant is a delisted or renamed peer and goes to ``unresolved`` rather
    than becoming a label pointing at nothing.

    Reading the annotation is not a convenience, it is a correctness
    requirement. Left in the text, "Atlassian (TEAM)" segments into Atlassian,
    whose ticker is TEAM, and then into the word TEAM, which is the registrant
    name of Team, Inc., an industrial services company. That is the worst thing
    this module can produce: a confident, wrong label.
    """
    known = index.tickers if isinstance(index, NameIndex) else None
    marks = _annotations(blob, known)
    if len(marks) < 3:
        return None

    tickers: list[str] = []
    leftovers: list[str] = []
    spans: list[str] = []
    cursor = 0
    for start, end, symbol in marks:
        name = blob[cursor:start].strip(_DELIMITERS + ".")
        cursor = end
        label = f"{name} ({symbol})".strip() if name else f"({symbol})"
        spans.append(label)
        if known is not None and symbol in known:
            tickers.append(symbol)
            continue
        inner = _segment(name, index, prefer_ciks)
        if len(inner.tickers) == 1:
            tickers.append(inner.tickers[0])
        elif known is not None:
            leftovers.append(
                f"{label}: symbol absent from the SEC ticker file, so the peer has "
                "since been acquired, renamed or delisted"
            )
        else:
            leftovers.append(label)
    tail = blob[cursor:].strip(_DELIMITERS + ".")
    if tail:
        inner = _segment(tail, index, prefer_ciks)
        spans.extend(inner.spans)
        tickers.extend(inner.tickers)
        leftovers.extend(inner.leftovers)
    return _Segmentation(tickers, leftovers, spans, True)


def segment_names(
    blob: str,
    index: dict[str, str],
    *,
    prefer_tickers: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Split a run of joined company names and resolve each to a ticker.

    Returns ``(tickers, leftovers)``. ``tickers`` is in the order the names
    appear, with duplicates kept so a caller can see the raw reading;
    ``leftovers`` holds the spans that did not resolve, in readable form, with
    any collision spelled out so the reader can settle it by hand.

    ``prefer_tickers`` breaks ties toward companies already known to be in play,
    which in practice means the other peers in the same proxy and the universe
    being collected. It never invents a resolution: a collision that survives
    the preference is reported, not decided.
    """
    seg = _segment_blob(blob, index, prefer_tickers)
    return seg.tickers, seg.leftovers


def _segment_blob(
    blob: str,
    index: dict[str, str],
    prefer_tickers: Iterable[str] = (),
) -> _Segmentation:
    """``segment_names`` with the reading kept, so callers can report the method."""
    prefer_ciks: set[int] = set()
    if isinstance(index, NameIndex):
        for t in prefer_tickers:
            cik = index.cik_of.get(str(t).upper())
            if cik is not None:
                prefer_ciks.add(cik)

    annotated = _segment_annotated(blob, index, prefer_ciks)
    if annotated is not None:
        return annotated

    seg = _segment(blob, index, prefer_ciks)
    if prefer_ciks and isinstance(index, NameIndex):
        # Second pass: the first pass may have resolved names further down the
        # table that settle a collision seen earlier in it.
        found = {index.cik_of[t] for t in seg.tickers if t in index.cik_of}
        if found - prefer_ciks:
            seg = _segment(blob, index, prefer_ciks | found)
    return seg


# --------------------------------------------------------------------------- #
# Locating the list inside a proxy
# --------------------------------------------------------------------------- #

# Ordered by how specific the phrase is. The last pattern will match almost any
# colon after the words "peer group", so it is tried last and every candidate it
# produces still has to survive the plausibility checks below.
_ANCHORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "was as follows",
        re.compile(
            r"(?:compensation\s+)?peer\s+group[^.\n]{0,140}?was\s+as\s+follows\s*:?",
            re.I,
        ),
    ),
    (
        "consisted of the following",
        re.compile(
            r"(?:compensation\s+)?peer\s+group[^.\n]{0,140}?"
            r"(?:consisted|consists|comprised|comprises|composed|made\s+up)\s+of\s+"
            r"the\s+following[^.:\n]{0,40}:?",
            re.I,
        ),
    ),
    (
        "peer group for year",
        re.compile(
            r"(?:compensation\s+)?peer\s+group\s+(?:used\s+)?for\s+"
            r"(?:purposes\s+of\s+)?(?:our\s+)?(?:fiscal\s+)?(?:year\s+)?"
            r"(?:19|20)\d{2}[^.\n]{0,80}?:",
            re.I,
        ),
    ),
    (
        "peer group heading for a year",
        re.compile(
            r"peer\s+group\s+for\s+(?:the\s+)?(?:fiscal\s+)?(?:year\s+)?"
            r"(?:ended\s+)?(?:19|20)\d{2}\s*:?(?=\s*[A-Z•▪●])",
            re.I,
        ),
    ),
    (
        "year peer group heading",
        re.compile(
            r"(?:fiscal\s+(?:year\s+)?)?(?:19|20)\d{2}\s+(?:compensation\s+)?"
            # The parenthetical is part of the heading, not the list: ServiceNow
            # prints "2026 Peer Group (which was also the 2025 Peer Group)" and
            # leaving it in the span would end the list at its first lowercase
            # word before a single company name had been read.
            r"peer\s+(?:group|companies)\s*(?:\([^)\n]{0,90}\))?\s*:?"
            r"(?=\s*\n|\s*[A-Z])",
            re.I,
        ),
    ),
    (
        "the following companies",
        re.compile(
            r"the\s+following\s+(?:\d{1,2}\s+)?(?:peer\s+)?companies"
            r"(?:\s+(?:were|are|comprised|constituted|made\s+up))?[^.\n]{0,170}?:",
            re.I,
        ),
    ),
    (
        "compensation peer group",
        re.compile(
            r"(?:compensation\s+)?peer\s+group[^.\n]{0,190}?:(?=\s*\n|\s*[0-9A-Z•▪])",
            re.I,
        ),
    ),
)

_YEAR = re.compile(r"(?:19|20)\d{2}")
_PROSE_ONSET = re.compile(r"\s([a-z]+)")

_CRITERIA_KEYWORDS = (
    "revenue",
    "market cap",
    "market capitali",
    "capitali",
    "headcount",
    "growth rate",
    "criteria",
    "selection",
    "industry",
    "headquarter",
    "similar",
    "range of",
)
_CRITERIA_BAND = re.compile(r"\d+(?:\.\d+)?\s*x|\$\s?\d|percentile", re.I)
_CRITERIA_CHARS = 900

# Sentence boundaries in a rendered proxy, which include the seam a table leaves
# behind: "...market capitalization.The independent consultant also..." has no
# space after the full stop, because the two sentences came from two cells.
_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z])|(?<=[a-z])\.(?=[A-Z])|[•▪●]|\n")


def _list_end(span: str) -> int:
    """Where the run of company names stops and ordinary prose starts again.

    Detecting the end matters as much as detecting the start, because the table
    runs straight into the paragraph after it with no punctuation at the seam:
    "...SnowflakeThe compensation committee reviews". Company names in these
    lists are title case, so every internal word break is followed by a capital.
    The first word break followed by a lowercase letter is therefore the moment
    prose resumes, and it is the one signal that survives all three rendered
    formats.

    The cost of this rule is a name containing a genuine lowercase word, such as
    "Bank of America", which would truncate the list early. That is the right
    way to be wrong: a short list is flagged by the plausibility check, while
    running past the end feeds paragraphs of prose into the segmenter.
    """
    for m in _PROSE_ONSET.finditer(span):
        word = m.group(1)
        if word in _NAME_PARTICLES:
            continue
        return m.start()
    return len(span)


def _trim_trailing_opener(span: str) -> str:
    """Drop the first word of the resuming sentence, left glued to the last name."""
    stripped = span.rstrip()
    m = re.search(r"([A-Z][a-z]{0,14})$", stripped)
    if m and m.group(1).lower() in _SENTENCE_OPENERS:
        return stripped[: m.start()]
    return stripped


def _selection_criteria(before: str) -> str | None:
    """The stated revenue and market capitalisation bands, verbatim.

    Worth keeping because it is what the label means. A group screened to 0.5x
    to 2.5x the filer's revenue is a statement about size as much as about
    business model, and a reader who cannot see the bands cannot tell how much
    of the learned similarity is just scale.
    """
    chunks = [
        c.strip(" \n\t;:•▪")
        for c in re.split(_SENTENCE_SPLIT, before)
    ]
    keep = [
        c
        for c in chunks
        if len(c) > 25
        and any(k in c.lower() for k in _CRITERIA_KEYWORDS)
        and (_CRITERIA_BAND.search(c) or "criteria" in c.lower())
    ]
    if not keep:
        return None
    # The sentence nearest the list is the one that states the bands, so it is
    # taken first and kept even when it alone fills the budget; earlier
    # sentences are added only while they fit.
    out: list[str] = [_truncate(keep[-1], _CRITERIA_CHARS)]
    total = len(out[0])
    for chunk in reversed(keep[-4:-1]):
        if total + len(chunk) + 2 > _CRITERIA_CHARS:
            break
        out.append(chunk)
        total += len(chunk) + 2
    return "; ".join(reversed(out)) or None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > limit // 2 else limit].rstrip(" ,;") + " ..."


def _fiscal_year(anchor_text: str, before: str, filed: date | None) -> tuple[int | None, str]:
    """The fiscal year the disclosed group applied to, and how it was determined.

    Not the filing year. A proxy filed in 2026 discloses the group used for 2025
    pay decisions, and stamping the label with the filing year would move every
    relationship a year into the future, which is exactly the leak the
    point-in-time rule exists to prevent.
    """
    years = _YEAR.findall(anchor_text)
    if years:
        return int(years[-1]), "stated in the introducing phrase"
    tail = before[-260:]
    near = _YEAR.findall(tail)
    if near:
        return int(near[-1]), "read from the sentence introducing the list"
    if filed is not None:
        return filed.year - 1, (
            "no year stated beside the list, so the year before the filing date "
            "is used; the group a proxy discloses is the one used for the prior "
            "year's pay decisions"
        )
    return None, "no year stated beside the list and no filing date supplied"


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #


@dataclass
class PeerGroup:
    """One disclosed compensation peer group, resolved to tickers.

    ``confidence`` is the share of segmented spans that resolved to a ticker,
    set to zero outright when the number of peers falls outside the plausible
    band, because a count of three or of eighty is a parse that found a fragment
    or overran the table rather than a small or large peer group. ``usable``
    reads that decision back: a group that is not usable is still returned so it
    can be audited, but it never becomes a training pair.
    """

    ticker: str
    cik: int | None
    accession: str | None
    filed: date | None
    fiscal_year: int | None
    peers: list[str]
    unresolved: list[str] = field(default_factory=list)
    method: str = ""
    selection_criteria: str | None = None
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def n_spans(self) -> int:
        return len(self.peers) + len(self.unresolved)

    @property
    def resolved_share(self) -> float:
        return len(self.peers) / self.n_spans if self.n_spans else 0.0

    @property
    def count_plausible(self) -> bool:
        return MIN_PLAUSIBLE_PEERS <= len(self.peers) <= MAX_PLAUSIBLE_PEERS

    @property
    def usable(self) -> bool:
        """Whether this group may become a training label."""
        return self.count_plausible and self.confidence > 0.0 and self.fiscal_year is not None

    def rows(self) -> list[tuple[str, Any]]:
        return [
            ("Filer", self.ticker),
            ("Fiscal year", self.fiscal_year),
            ("Accession", self.accession),
            ("Filed", str(self.filed) if self.filed else None),
            ("Peers resolved", len(self.peers)),
            ("Spans unresolved", len(self.unresolved)),
            ("Confidence", self.confidence),
            ("Usable as a label", self.usable),
            ("Method", self.method),
        ]

    def to_frame(self) -> pd.DataFrame:
        """One row per peer, which is the shape a training set wants."""
        return pd.DataFrame(
            {
                "filer": [self.ticker] * len(self.peers),
                "peer": list(self.peers),
                "fiscal_year": [self.fiscal_year] * len(self.peers),
                "accession": [self.accession] * len(self.peers),
                "confidence": [self.confidence] * len(self.peers),
            }
        )


def groups_to_frame(groups: Sequence[PeerGroup]) -> pd.DataFrame:
    """Every group's peers stacked, for eyeballing a universe-wide build."""
    frames = [g.to_frame() for g in groups if g.peers]
    if not frames:
        return pd.DataFrame(
            columns=["filer", "peer", "fiscal_year", "accession", "confidence"]
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_peer_group(
    text: str,
    index: dict[str, str],
    ticker: str,
    *,
    accession: str | None = None,
    filed: date | None = None,
    prefer_tickers: Iterable[str] = (),
) -> PeerGroup | None:
    """Find the disclosed compensation peer group in one proxy and resolve it.

    Returns ``None`` when no introducing phrase in the document is followed by
    anything that looks like a list of company names, which is the honest answer
    for a proxy that discloses no peer group or describes it only in prose.

    The list is located, never keyword-searched. A proxy says "peer group" a
    dozen times, and the occurrences that matter are the ones that introduce a
    table: the phrase is matched, the span after it is taken up to the point
    where prose resumes, and every candidate is then scored on what it actually
    segments into. That scoring is what rejects the pay-versus-performance
    table, whose header also ends in a colon after the words "Peer Group" but
    whose contents are dollar amounts.
    """
    cleaned = _clean(text)
    filer = str(ticker).upper()
    prefer = list(dict.fromkeys([filer, *[str(t).upper() for t in prefer_tickers]]))

    best: tuple[tuple[int, int, float, int], PeerGroup] | None = None
    for label, pattern in _ANCHORS:
        for m in pattern.finditer(cleaned):
            group = _candidate_from_anchor(
                cleaned, m, label, index, filer, prefer, accession, filed
            )
            if group is None:
                continue
            # A candidate inside the plausible band outranks a longer one
            # outside it, so a span that overran the table does not displace the
            # clean read of the same table. Then most peers, then the best
            # resolution rate, then the earliest position in the document.
            score = (
                int(group.count_plausible),
                len(group.peers),
                group.resolved_share,
                -m.start(),
            )
            if best is None or score > best[0]:
                best = (score, group)
    return best[1] if best is not None else None


def _candidate_from_anchor(
    cleaned: str,
    match: re.Match[str],
    anchor_label: str,
    index: dict[str, str],
    filer: str,
    prefer: Sequence[str],
    accession: str | None,
    filed: date | None,
) -> PeerGroup | None:
    """Build and vet one candidate peer group from one introducing phrase."""
    after = cleaned[match.end() : match.end() + 4000]
    span = after[: _list_end(after)]
    span = _trim_trailing_opener(span)
    if len(span.strip()) < 12:
        return None

    # A table of dollar amounts is not a list of company names. The
    # pay-versus-performance disclosure carries a "Peer Group ...:" header and
    # is nothing but figures, and this is what keeps it out.
    digits = sum(c.isdigit() for c in span)
    if digits / max(len(span), 1) > 0.10 and not _TICKER_ANNOTATION.search(span):
        return None

    seg = _segment_blob(span, index, prefer)
    tickers, leftovers = seg.tickers, seg.leftovers

    notes: list[str] = []
    seen: set[str] = set()
    peers: list[str] = []
    self_seen = False
    for t in tickers:
        if t == filer:
            self_seen = True
            continue
        if t in seen:
            continue
        seen.add(t)
        peers.append(t)
    if self_seen:
        # Several filers print themselves inside the table to show where they
        # sit in it. The filer is not its own peer and the pair would be a
        # degenerate training row.
        notes.append(f"{filer} names itself inside its own table; dropped")

    n_dupes = len(tickers) - len(set(tickers))
    if n_dupes:
        notes.append(f"{n_dupes} duplicate name(s) in the table, de-duplicated")

    if not peers:
        return None

    year, how = _fiscal_year(match.group(0), cleaned[max(0, match.start() - 400) : match.start()], filed)
    notes.append(f"fiscal year {year}: {how}")

    n_spans = len(peers) + len(leftovers)
    resolved_share = len(peers) / n_spans if n_spans else 0.0
    count_ok = MIN_PLAUSIBLE_PEERS <= len(peers) <= MAX_PLAUSIBLE_PEERS
    if count_ok:
        confidence = round(resolved_share, 4)
    else:
        confidence = 0.0
        notes.append(
            f"{len(peers)} peers is outside the plausible {MIN_PLAUSIBLE_PEERS} to "
            f"{MAX_PLAUSIBLE_PEERS} band for a disclosed compensation peer group, "
            "so this is treated as a failed parse rather than a peer group; the "
            f"share of spans that did resolve was {resolved_share:.2f}"
        )
    if leftovers:
        notes.append(
            f"{len(leftovers)} span(s) did not resolve to a ticker: "
            + "; ".join(leftovers[:8])
        )

    criteria = _selection_criteria(cleaned[max(0, match.start() - 2600) : match.start()])
    method = (
        "def14a/annotated-ticker"
        if seg.annotated
        else "def14a/longest-match-segmentation"
    )

    cik = None
    if isinstance(index, NameIndex):
        cik = index.cik_of.get(filer)

    return PeerGroup(
        ticker=filer,
        cik=cik,
        accession=accession,
        filed=filed,
        fiscal_year=year,
        peers=peers,
        unresolved=list(leftovers),
        method=f"{method} (anchor: {anchor_label})",
        selection_criteria=criteria,
        confidence=confidence,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def cache_dir(assumptions: Assumptions | None = None) -> Path:
    """Where extracted groups are kept, from ``assumptions.ml.cache_dir``.

    A universe-wide build reads several proxies per company and each is a
    megabyte of HTML, so the parse is done once and kept. The HTTP layer already
    caches the filing bytes; this caches the far smaller result of reading them.
    """
    configured = assumptions.ml.cache_dir if assumptions is not None else None
    root = Path(configured).expanduser() if configured else Path.home() / ".techval" / "ml"
    return root / "peer_groups"


def _cache_path(accession: str, assumptions: Assumptions | None) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", accession)
    return cache_dir(assumptions) / f"{safe}.json"


def _cache_read(accession: str, assumptions: Assumptions | None) -> PeerGroup | None:
    path = _cache_path(accession, assumptions)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("parser_version") != PARSER_VERSION:
        return None
    if payload.get("no_peer_group"):
        return None
    filed = payload.get("filed")
    return PeerGroup(
        ticker=payload["ticker"],
        cik=payload.get("cik"),
        accession=payload.get("accession"),
        filed=date.fromisoformat(filed) if filed else None,
        fiscal_year=payload.get("fiscal_year"),
        peers=list(payload.get("peers", [])),
        unresolved=list(payload.get("unresolved", [])),
        method=payload.get("method", ""),
        selection_criteria=payload.get("selection_criteria"),
        confidence=float(payload.get("confidence", 0.0)),
        notes=list(payload.get("notes", [])),
    )


def _cache_write(
    accession: str, group: PeerGroup | None, assumptions: Assumptions | None
) -> None:
    path = _cache_path(accession, assumptions)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"parser_version": PARSER_VERSION, "accession": accession}
        if group is None:
            payload["no_peer_group"] = True
        else:
            payload.update(
                {
                    "ticker": group.ticker,
                    "cik": group.cik,
                    "filed": group.filed.isoformat() if group.filed else None,
                    "fiscal_year": group.fiscal_year,
                    "peers": group.peers,
                    "unresolved": group.unresolved,
                    "method": group.method,
                    "selection_criteria": group.selection_criteria,
                    "confidence": group.confidence,
                    "notes": group.notes,
                }
            )
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)
    except OSError:
        # An unwritable cache slows the next build down. It is not a reason to
        # fail a parse that already succeeded.
        pass


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


def collect_peer_groups(
    tickers: Sequence[str],
    client: Any,
    index: dict[str, str] | None = None,
    years: Iterable[int] | None = None,
    *,
    assumptions: Assumptions | None = None,
    proxies_per_ticker: int = 6,
    use_cache: bool = True,
    keep_failed: bool = False,
    skip_errors: bool = False,
) -> list[PeerGroup]:
    """Read every company's proxies and return the peer groups they disclose.

    Each company has four to six proxies on file, so a few hundred TMT filers
    yield several thousand dated groups and tens of thousands of labelled pairs.
    Groups whose parse failed the plausibility check are left out by default,
    since they are what a training set must not contain; pass ``keep_failed`` to
    see them and audit the failures.

    ``years`` restricts the result to groups that applied to those fiscal years.
    Where two proxies disclose a group for the same fiscal year, the one with
    the higher confidence wins and the earlier filing breaks the tie, so the
    result does not depend on the order the filings came back in.

    Errors are raised by default, because a silent gap in a training set is
    indistinguishable from a company that discloses nothing. A universe build
    can pass ``skip_errors`` to turn each failure into a zero-confidence record
    carrying the reason, which ``keep_failed`` will then show.
    """
    universe = [str(t).upper() for t in tickers]
    if index is None:
        index = build_name_index(client, former_names_for=universe)
    wanted = set(years) if years is not None else None

    collected: list[PeerGroup] = []
    for filer in universe:
        try:
            filings = client.filings(filer, ("DEF 14A",), limit=proxies_per_ticker)
        except Exception as exc:
            if not skip_errors:
                raise
            collected.append(_unreadable(filer, None, None, index, f"{type(exc).__name__}: {exc}"))
            continue

        for filing in filings:
            accession = filing.get("accession")
            group: PeerGroup | None = None
            cached = False
            if use_cache and accession:
                group = _cache_read(accession, assumptions)
                cached = group is not None
            if group is None:
                try:
                    text = client.filing_text(filer, filing)
                    group = extract_peer_group(
                        text,
                        index,
                        filer,
                        accession=accession,
                        filed=filing.get("filed"),
                        prefer_tickers=universe,
                    )
                except Exception as exc:
                    if not skip_errors:
                        raise
                    collected.append(
                        _unreadable(
                            filer,
                            accession,
                            filing.get("filed"),
                            index,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                if use_cache and accession:
                    _cache_write(accession, group, assumptions)
            if group is None:
                continue
            if cached:
                group.notes.append(f"read from the extraction cache at {cache_dir(assumptions)}")
            collected.append(group)

    by_year: dict[tuple[str, int | None], PeerGroup] = {}
    loose: list[PeerGroup] = []
    for g in collected:
        if wanted is not None and g.fiscal_year not in wanted:
            continue
        if not g.usable:
            loose.append(g)
            continue
        key = (g.ticker, g.fiscal_year)
        prior = by_year.get(key)
        if prior is None or _better(g, prior):
            by_year[key] = g

    out = list(by_year.values())
    if keep_failed:
        out.extend(loose)
    return sorted(
        out,
        key=lambda g: (g.ticker, g.fiscal_year or 0, g.accession or ""),
    )


def _better(candidate: PeerGroup, incumbent: PeerGroup) -> bool:
    """Which of two readings of the same filer and year to keep.

    Higher confidence first. Where they tie, the earlier filing wins, because
    the proxy nearest the year in question is the one that disclosed the group
    while it was in force; a later proxy repeating it is a restatement.
    """
    if candidate.confidence != incumbent.confidence:
        return candidate.confidence > incumbent.confidence
    return (candidate.filed or date.max) < (incumbent.filed or date.max)


def _unreadable(
    filer: str,
    accession: str | None,
    filed: date | None,
    index: dict[str, str],
    reason: str,
) -> PeerGroup:
    cik = index.cik_of.get(filer) if isinstance(index, NameIndex) else None
    return PeerGroup(
        ticker=filer,
        cik=cik,
        accession=accession,
        filed=filed,
        fiscal_year=None,
        peers=[],
        unresolved=[],
        method="unreadable",
        selection_criteria=None,
        confidence=0.0,
        notes=[f"could not be read: {reason}"],
    )


# --------------------------------------------------------------------------- #
# Training pairs
# --------------------------------------------------------------------------- #


def peer_pairs(
    groups: Sequence[PeerGroup],
    *,
    include_co_members: bool = False,
) -> list[tuple[str, str, int]]:
    """Year-stamped symmetric pairs, which is what a similarity model trains on.

    Symmetric because comparability is a property of the pair, not a direction:
    the filer asserted it, and a model that learned to rank A near B but not B
    near A would have learned the filing convention rather than the similarity.

    Year-stamped because the assertion was made about one fiscal year and about
    no other. Datadog's 2021 group is not its 2025 group, companies enter and
    leave as they are acquired or outgrow the size bands, and training on the
    union across years would let a 2025 relationship justify a 2021 ranking.
    That leak is invisible in the fit and fatal out of sample, so the year
    travels with the label and a walk-forward split can respect it.

    Groups whose parse failed the plausibility check contribute nothing, whether
    or not the caller filtered them out already.

    ``include_co_members`` additionally pairs the peers with each other. It is
    off by default because no filer asserted it: co-membership of one table is
    a derived signal, weaker than the disclosure, and mixing the two would put
    labels of two different strengths in one training set without a way to tell
    them apart.
    """
    seen: set[tuple[str, str, int]] = set()
    for g in groups:
        if not g.usable or g.fiscal_year is None:
            continue
        year = g.fiscal_year
        for peer in g.peers:
            if peer == g.ticker:
                continue
            seen.add((g.ticker, peer, year))
            seen.add((peer, g.ticker, year))
        if include_co_members:
            members = sorted(set(g.peers))
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    seen.add((a, b, year))
                    seen.add((b, a, year))
    return sorted(seen)


def pairs_to_frame(pairs: Sequence[tuple[str, str, int]]) -> pd.DataFrame:
    """The pair list as a frame, ready to be split by date for walk-forward."""
    return pd.DataFrame(list(pairs), columns=["a", "b", "fiscal_year"])
