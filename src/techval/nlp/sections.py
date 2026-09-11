"""Cutting a 10-K into its Items, so that everything downstream reads the right part.

A 10-K is not one document. Item 1 is what the company sells and to whom, Item 1A
is what management is required to admit could go wrong, Item 7 is the only place a
software filer writes annual recurring revenue and net retention down in words.
Those three readers want three different texts. Handing any of them the whole
405,000-character filing is the difference between a signal and a word cloud: a
peer model fed the entire document scores Datadog against every other company that
also has a legal proceedings section.

**The table of contents is the trap.** Every 10-K names each Item twice, once in
the contents table near the front and once at the section itself. Datadog's FY2025
10-K carries "Item 1.Business5Item 1A.Risk Factors13" at character 29,363 and the
real "Item 1. BusinessOverviewDatadog is the AI-powered observability..." at
character 40,836. A first-match split returns a list of section names and page
numbers as the entire business description, and nothing downstream notices,
because a list of section names is still text.

**The obvious fix is also wrong.** Taking the last occurrence of each header fails
on the same filing, because a 10-K cross-references itself. At character 85,583
Datadog writes "described under Part I - Item 1. Business in this Annual Report",
which is the last "Item 1. Business" in the document and sits in the middle of the
risk factors. Last-match would hand back the back half of the filing as Item 1.

So the split uses three rules together, each of which earns its place on real
filings:

1. *Page numbers, not density.* Contents entries cluster, but so do the Items at
   the back of a 10-K that each answer in one line by pointing at the proxy
   statement. Across six filings those back-matter runs sit a median of 159 to
   205 characters apart, which is inside any window wide enough to catch a
   contents table, so clustering alone would discard six real sections in
   CrowdStrike's filing and six more in Zscaler's. What separates the two is what
   ends the entry. A contents line ends in a page number; a section ends in a
   sentence. Measured over the same six filings the split is total: 22 of 22
   entries page-numbered in every contents table, none or one of four to six in
   every back-matter run. So a run of five or more headers whose entries are at
   least three-quarters page-numbered is a contents table, and every header inside
   its span is discarded.

2. *Sentence boundary.* A real header begins a block. In stripped filing text that
   shows up as a preceding sentence terminator, a closing bracket, or a page
   number, once the page furniture that survives markup stripping is set aside: a
   trailing "PART II", a repeated "Table of Contents" running head, a page number.
   A cross-reference instead sits inside a sentence, so it is preceded by a
   lowercase word, a comma, a dash or an opening quote. This one test rejects all
   eight self-references in the Datadog filing and accepts all twenty-three real
   headers.

3. *Order.* Items appear in a 10-K in the order Regulation S-K prescribes. The
   scan therefore walks that sequence and takes, for each Item, the first surviving
   candidate after the header already accepted. A reference that points forward
   can only mislead the scan if it also survives rules 1 and 2.

What this module will not do is guess. An Item whose header is not found is absent
from the result and named in ``flags`` with the reason; it is never filled with the
nearest text. The sanity checks flag rather than raise, because a filer with two
lines of risk factors exists and is interesting, and an exception would stop a
whole peer run over one odd document.

Two limitations to know about before building on this. Nothing marks the start of
the signature page or the exhibit index, so whatever follows the last Item header
is attributed to that Item: CrowdStrike puts its exhibit index after Item 16, and
Item 16 accordingly comes back at 1,129 words rather than the one word its "None."
deserves. And a filer whose contents table lost its page numbers in stripping
would defeat rule 1, in which case Item 1 comes back as the contents table itself,
several hundred words long, and says so in ``flags``. Six 10-Ks measured here all
kept their page numbers.

One caveat on word counts. Markup stripping joins the last word of one HTML block
to the first word of the next, so "BusinessOverviewDatadog" counts as one word
where a reader sees three. Counts here run a little under a human count, by roughly
the number of block boundaries, and the thresholds below are set knowing that.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from ..errors import ConfigError, MissingDataError

# The Items of a 10-K in the order Regulation S-K prescribes, with the title the
# SEC gives each. The order is load-bearing: it is rule 3 of the split.
_TEN_K_ITEMS: tuple[tuple[str, str], ...] = (
    ("1", "Business"),
    ("1A", "Risk Factors"),
    ("1B", "Unresolved Staff Comments"),
    ("1C", "Cybersecurity"),
    ("2", "Properties"),
    ("3", "Legal Proceedings"),
    ("4", "Mine Safety Disclosures"),
    ("5", "Market for Registrant's Common Equity, Related Stockholder Matters and "
          "Issuer Purchases of Equity Securities"),
    ("6", "[Reserved]"),
    ("7", "Management's Discussion and Analysis of Financial Condition and Results "
          "of Operations"),
    ("7A", "Quantitative and Qualitative Disclosures About Market Risk"),
    ("8", "Financial Statements and Supplementary Data"),
    ("9", "Changes in and Disagreements with Accountants on Accounting and "
          "Financial Disclosure"),
    ("9A", "Controls and Procedures"),
    ("9B", "Other Information"),
    ("9C", "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections"),
    ("10", "Directors, Executive Officers and Corporate Governance"),
    ("11", "Executive Compensation"),
    ("12", "Security Ownership of Certain Beneficial Owners and Management and "
           "Related Stockholder Matters"),
    ("13", "Certain Relationships and Related Transactions, and Director "
           "Independence"),
    ("14", "Principal Accounting Fees and Services"),
    ("15", "Exhibits, Financial Statement Schedules"),
    ("16", "Form 10-K Summary"),
)

_FORMS: dict[str, tuple[tuple[str, str], ...]] = {
    "10-K": _TEN_K_ITEMS,
    "10-K/A": _TEN_K_ITEMS,
    "10-KT": _TEN_K_ITEMS,
}

# The Items that are prose, and that something downstream reads as prose. A short
# section here is a parse failure until proved otherwise. Everywhere else short is
# ordinary: Mine Safety Disclosures is "not applicable", Properties for a software
# filer is two sentences about a leased head office, and Items 10 through 14 are
# answered by incorporating the proxy statement. Both cases are flagged, because
# the caller is entitled to know either way, but only one of them is alarming.
_PROSE_ITEMS = frozenset({"1", "1A", "1C", "7", "8", "9A"})

# Spacing inside a header. The non-breaking space is written as an escape rather
# than left as an invisible byte in this file, and it does turn up: inline XBRL
# filings put them between "Item" and its number often enough to matter.
_SP = r"[\s\u00a0]"

# Header forms seen in the wild: "Item 1." and "Item 1:", "ITEM 1" followed by a
# hyphen or a dash of any width, "Item 1A." with no space at all, and non-breaking
# spaces wherever a space can go. The dashes are written as escapes so that none of
# them ends up loose in this file.
#
# The suffix letter is captured in the same pass as the number, which is what stops
# "Item 1A" being read as "Item 1": there is no per-Item pattern that could run in
# the wrong order.
#
# Deliberately no word boundary in front. Markup stripping glues a header to
# whatever preceded it on the page, and the two things that most often precede one
# are the part label and the running head that MongoDB, CrowdStrike and Zscaler
# repeat at the top of every page: "PART IItem 1." and "Table of ContentsItem 7."
# both have to match, and the second of them ends in a lowercase letter. What a
# boundary test would have bought is bought instead by the sentence-boundary rule,
# which sees "subitem 1" and "line item 1" for what they are.
_ITEM_RE = re.compile(
    rf"(?i:item){_SP}{{0,3}}(\d{{1,2}}){_SP}?([A-Ca-c])?"
    rf"(?![A-Za-z0-9]){_SP}*[.:)\u2014\u2013\u2010-]?{_SP}*"
)

# Page furniture that survives markup stripping and lands between the end of one
# section and the header of the next: a part label, and the running head that some
# filers repeat at the top of every page. Stripped one piece at a time rather than
# with a single repeated group, because a repeated alternation anchored at the end
# of a string backtracks catastrophically on the digit runs these documents are
# full of. Page numbers are deliberately not stripped: a page number sitting
# directly in front of a header is itself evidence that a section ended there.
_FURNITURE_PARTS = (
    re.compile(rf"{_SP}*(?i:part){_SP}+[IVXivx\d]{{1,4}}[.,:]?{_SP}*$"),
    re.compile(rf"{_SP}*(?i:table{_SP}+of{_SP}+contents){_SP}*$"),
)

# How far back to look for the end of the previous section. Page furniture is a
# line or two; anything further back cannot change the verdict, and scanning the
# whole preceding document for every candidate turns a millisecond into a minute.
_LOOKBEHIND = 200

# A contents entry ends in its page number. Anything else ends in prose.
_PAGE_NUMBER_RE = re.compile(rf"\d{{1,4}}(?:{_SP}*(?i:part){_SP}+[IVXivx]{{1,4}}[.,:]?)?{_SP}*$")

# A real header is preceded by the end of something. A cross-reference is preceded
# by the middle of a sentence.
_SECTION_BREAK = frozenset(".:;!?]}")

# Closing quotation marks come off before that test, because a sentence that ends
# inside a quotation still ended: MongoDB closes Item 9B with "...equity trading
# arrangement." plus a closing quotation mark, and the header follows straight
# after. Opening marks are left alone on purpose: a header preceded by one is
# being quoted, not declared.
_CLOSING_QUOTES = "\"'\u201d\u2019"

# Two headers closer together than this are neighbours in a list, not two sections
# with prose between them.
_TOC_MAX_GAP = 600
# A contents table lists most of the filing. Five entries is above what a run of
# genuinely empty Items produces and far below what any real table contains.
_TOC_MIN_ENTRIES = 5
# Share of a run's entries that must end in a page number. Measured across six
# 10-Ks the separation is total: every contents table scored 22 of 22, and every
# back-matter run of Items that merely incorporate the proxy statement scored zero
# or one out of four to six.
_TOC_MIN_PAGED = 0.75

# Sanity thresholds. These describe SEC document typography, not a valuation
# judgment, which is why they live here rather than in Assumptions.
_MIN_SECTION_WORDS = 200
_MIN_COVERAGE = 0.5

# How far past a header to look for the prescribed title before giving up on it.
_TITLE_WINDOW = 60


@dataclass
class Section:
    """One Item of a filing, and exactly where it came from.

    ``text`` is the literal slice ``raw[start_char:end_char]`` of the document,
    header included. Keeping that identity exact is the point: anything built on a
    section can quote a character range back to the filing, and a reader can check
    it without trusting this module's idea of where a title ends.
    """

    item: str
    title: str
    text: str
    start_char: int
    end_char: int
    word_count: int


@dataclass
class FilingSections:
    """Every Item found in one filing, with what was not found and why.

    ``notes`` records how the split was made, so the segmentation is auditable the
    way a number is. ``flags`` records what looks wrong: an Item with no header, a
    section too short to be prose, a split that left most of the document
    unassigned. Neither list is decoration. A caller that ignores them can publish
    a peer comparison built on a table of contents.
    """

    ticker: str = ""
    accession: str = ""
    form: str = "10-K"
    filed: date | None = None
    period: str | None = None
    sections: dict[str, Section] = field(default_factory=dict)
    raw_length: int = 0
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def get(self, item: str) -> Section | None:
        """Look up an Item, tolerant of case and of a stray "Item " prefix."""
        key = str(item).strip().upper().removeprefix("ITEM").strip().rstrip(".:")
        return self.sections.get(key)

    @property
    def business(self) -> Section | None:
        return self.get("1")

    @property
    def risk_factors(self) -> Section | None:
        return self.get("1A")

    @property
    def mdna(self) -> Section | None:
        return self.get("7")

    @property
    def covered_fraction(self) -> float:
        """Share of the document that landed in some section.

        The front matter ahead of Item 1, being the cover page, the contents table
        and the forward-looking statement, belongs to no Item, so a clean split on
        a modern 10-K covers roughly nine tenths rather than all of it.
        """
        if not self.raw_length:
            return 0.0
        spanned = sum(s.end_char - s.start_char for s in self.sections.values())
        return spanned / self.raw_length

    def rows(self) -> list[tuple[str, str, int, float]]:
        """Item, title, words and share of the document, in filing order."""
        ordered = sorted(self.sections.values(), key=lambda s: s.start_char)
        return [
            (
                s.item,
                s.title,
                s.word_count,
                (s.end_char - s.start_char) / self.raw_length if self.raw_length else 0.0,
            )
            for s in ordered
        ]


# Parsed splits, keyed by form and a digest of the text. A 10-K is read once per
# consumer and there are several consumers; re-running the scan for each is work
# done twice for an answer that cannot have changed.
_SPLIT_CACHE: dict[tuple[str, str], FilingSections] = {}
# Loaded filings, keyed by ticker, accession and form. This is the cache that
# matters in practice: it also avoids re-fetching and re-stripping three megabytes
# of inline XBRL when the peer model and the KPI extractor want different Items of
# the same filing.
_LOAD_CACHE: dict[tuple[str, str, str], FilingSections] = {}


def _title_pattern(title: str) -> re.Pattern[str]:
    """Match a prescribed title as the filer actually typed it.

    Filers vary the apostrophe, the spacing and the capitalisation of a title
    without varying the title, and half of them drop the square brackets the SEC
    puts around "[Reserved]". Matching word by word absorbs all of that, and the
    span that matches is what gets reported, so the title shown is the filing's own
    characters rather than the SEC's.
    """
    words = []
    for word in title.split():
        escaped = re.escape(word).replace("'", "['\\u2019]")
        words.append(escaped.replace("\\[", "\\[?").replace("\\]", "\\]?"))
    return re.compile(rf"{_SP}*".join(words), re.IGNORECASE)


_TITLE_PATTERNS: dict[str, re.Pattern[str]] = {
    item: _title_pattern(title) for item, title in _TEN_K_ITEMS
}


def _detach(parts: FilingSections) -> FilingSections:
    """A copy a caller can annotate without writing into the shared cache.

    The split is memoised, so without this a consumer that appended to ``notes``
    would be appending to every other consumer's copy of the same filing.
    """
    return replace(
        parts,
        sections=dict(parts.sections),
        notes=list(parts.notes),
        flags=list(parts.flags),
    )


def _trim_furniture(text: str) -> str:
    """Take the page furniture off the end of a run of text."""
    trimmed = text.rstrip()
    for _ in range(len(_FURNITURE_PARTS)):
        shorter = trimmed
        for pattern in _FURNITURE_PARTS:
            shorter = pattern.sub("", shorter)
        if shorter == trimmed:
            break
        trimmed = shorter
    return trimmed


def _candidates(text: str) -> list[tuple[str, int, int]]:
    """Every place the text names an Item: (item key, header start, header end)."""
    out: list[tuple[str, int, int]] = []
    for m in _ITEM_RE.finditer(text):
        suffix = (m.group(2) or "").upper()
        out.append((f"{m.group(1)}{suffix}", m.start(), m.end()))
    return out


def _toc_spans(text: str, found: list[tuple[str, int, int]]) -> list[tuple[int, int]]:
    """Character spans occupied by a contents table or a similar index.

    Rule 1 of the split. Proximity groups the headers into runs; what marks a run
    as a contents table is that its entries end in page numbers instead of
    sentences. Proximity on its own is not enough, and that is not a theoretical
    worry: in CrowdStrike's 10-K the six back-matter Items that each incorporate
    the proxy statement by reference sit a median of 184 characters apart, inside
    the same 600-character window that a contents table would be. Discarding them
    as a table would throw away six real sections. None of them ends in a page
    number, and every entry of every contents table measured does.

    The page-number test survives formatting differences. It does not care whether
    the filer writes "Item 1." or "ITEM 1 -", whether the table sits at the front
    or is repeated, or whether the numbers were right-aligned in a column that
    stripping has flattened onto one line.
    """
    order = {item: i for i, (item, _title) in enumerate(_TEN_K_ITEMS)}
    spans: list[tuple[int, int]] = []
    run: list[tuple[str, int, int]] = []

    def close(run: list[tuple[str, int, int]]) -> None:
        if len({r[0] for r in run}) < _TOC_MIN_ENTRIES:
            return
        entries = len(run) - 1
        paged = sum(
            1
            for i in range(entries)
            if _PAGE_NUMBER_RE.search(text[run[i][2] : run[i + 1][1]])
        )
        if paged / entries >= _TOC_MIN_PAGED:
            spans.append((run[0][1], run[-1][2]))

    for cand in found:
        # A contents table lists each Item once and in order, so a repeat or a step
        # backwards ends it. This is what stops the table running on into the real
        # Item 1 when a filer puts no cover matter between the two: the last entry
        # is Item 9A or Item 16, and the header that follows is Item 1.
        if run and (
            cand[1] - run[-1][1] > _TOC_MAX_GAP
            or order.get(cand[0], -1) <= order.get(run[-1][0], -1)
        ):
            close(run)
            run = []
        run.append(cand)
    if run:
        close(run)
    return spans


def _starts_a_section(text: str, at: int) -> bool:
    """Rule 2: does the text before position ``at`` end, or does it continue?

    Page furniture comes off first, because "...without notice.4PART IItem 1."
    ended a sentence four characters and one part label before the header. What is
    left is judged on its final character: a terminator, a closing bracket or a
    digit ends something, while a lowercase letter, a comma, a dash or an opening
    quote means the header is a cross-reference inside somebody's sentence.

    The digit is there on purpose. "...New YorkFebruary 18, 202695Item 9B." ends in
    a page number that was a table cell before the markup came off, and stripping
    it would expose the comma of a date and lose a real header.
    """
    before = _trim_furniture(text[max(0, at - _LOOKBEHIND) : at]).rstrip(_CLOSING_QUOTES)
    if not before:
        return True
    last = before[-1]
    return last in _SECTION_BREAK or last.isdigit()


def split_items(text: str, form: str = "10-K") -> FilingSections:
    """Split filing text into its Items.

    Returns a ``FilingSections`` with the identity fields blank, so the value can
    be used as it is or completed by ``load_sections`` once the filing it came from
    is known. The three rules are set out at the top of this module; the short
    version is that contents tables are found by their page numbers,
    cross-references by the punctuation in front of them, and what survives is
    assigned by walking the Items in the order the SEC prescribes.

    Raises ``ConfigError`` for a form whose Item sequence this does not know. The
    10-Q is the one that matters: it restarts its numbering in Part II, so "Item 1"
    is both the financial statements and the legal proceedings, and the keys would
    have to be part-qualified before they meant anything.
    """
    sequence = _FORMS.get(form.upper())
    if sequence is None:
        raise ConfigError(
            f"no Item sequence defined for form {form!r}; "
            f"known forms are {', '.join(sorted(_FORMS))}. A 10-Q would need "
            "part-qualified keys because it restarts its Item numbering in Part II."
        )

    key = (form.upper(), hashlib.sha256(text.encode("utf-8", "replace")).hexdigest())
    cached = _SPLIT_CACHE.get(key)
    if cached is not None:
        return _detach(cached)

    notes: list[str] = []
    flags: list[str] = []
    found = _candidates(text)
    notes.append(
        f"{len(found)} Item headers named anywhere in {len(text):,} characters"
    )

    toc = _toc_spans(text, found)
    in_toc = {
        i for i, c in enumerate(found) if any(a <= c[1] < b for a, b in toc)
    }
    for start, end in toc:
        listed = sum(1 for c in found if start <= c[1] < end)
        notes.append(
            f"contents table at characters {start:,}-{end:,} skipped: "
            f"{listed} headers listed there, each followed by a page number"
        )
    if not toc:
        notes.append(
            "no contents table found: no run of five or more Item headers here ends "
            "in page numbers. Every header was judged on its own"
        )

    survivors: list[tuple[str, int, int]] = []
    cross_refs = 0
    for i, cand in enumerate(found):
        if i in in_toc:
            continue
        if not _starts_a_section(text, cand[1]):
            cross_refs += 1
            continue
        survivors.append(cand)
    if cross_refs:
        notes.append(
            f"{cross_refs} headers rejected as cross-references: the filing names the "
            "Item inside a sentence rather than starting one with it"
        )

    # Rule 3. Walk the prescribed order and take the first surviving header at or
    # after the end of the one already accepted, so a reference pointing forward
    # cannot claim a section the scan has not reached.
    accepted: list[tuple[str, str, int, int]] = []
    cursor = 0
    for item, canonical in sequence:
        hit = next((c for c in survivors if c[0] == item and c[1] >= cursor), None)
        if hit is None:
            flags.append(
                f"Item {item} ({canonical[:44]}): no header found outside the contents "
                "table. The filing may omit it, or may format it in a way this split "
                "does not recognise. Not substituted."
            )
            continue
        accepted.append((item, canonical, hit[1], hit[2]))
        cursor = hit[2]

    sections: dict[str, Section] = {}
    for i, (item, canonical, start, header_end) in enumerate(accepted):
        end = accepted[i + 1][2] if i + 1 < len(accepted) else len(text)
        body = text[start:end]
        window = text[header_end : min(header_end + len(canonical) + _TITLE_WINDOW, end)]
        match = _TITLE_PATTERNS[item].match(window)
        if match:
            title = re.sub(rf"{_SP}+", " ", match.group(0)).strip()
        else:
            # The filing's own opening words, truncated, rather than the SEC's
            # title: a title not read out of this document is not sourced from it.
            title = re.sub(rf"{_SP}+", " ", window[:_TITLE_WINDOW]).strip()
            notes.append(
                f"Item {item}: the header is not followed by the prescribed title, so "
                "the title shown is the filing's own opening words"
            )
        sections[item] = Section(
            item=item,
            title=title,
            text=body,
            start_char=start,
            end_char=end,
            word_count=len(body.split()),
        )

    result = FilingSections(
        form=form.upper(),
        sections=sections,
        raw_length=len(text),
        notes=notes,
        flags=flags + _sanity_flags(sections, len(text)),
    )
    _SPLIT_CACHE[key] = result
    return _detach(result)


def _sanity_flags(sections: dict[str, Section], raw_length: int) -> list[str]:
    """What a reader must be told before trusting the split.

    Every one of these flags rather than raises. A short risk factors section is a
    parse failure nine times in ten and a genuinely terse filer the tenth, and this
    module is not in a position to tell which, so it says what it sees.
    """
    flags: list[str] = []

    for item, sec in sorted(sections.items(), key=lambda kv: kv[1].start_char):
        if sec.word_count >= _MIN_SECTION_WORDS:
            continue
        verdict = (
            " A section this short is usually a contents table that the split "
            "mistook for the section itself. Read it before using it."
            if item in _PROSE_ITEMS
            else " Short is ordinary for this Item, which is often answered in a line "
            "or by incorporating the proxy statement by reference."
        )
        flags.append(
            f"Item {item} is {sec.word_count} words, under the {_MIN_SECTION_WORDS} "
            f"a section of prose would run to.{verdict}"
        )

    business = sections.get("1")
    risk = sections.get("1A")
    if business is not None and risk is not None and risk.word_count < business.word_count:
        flags.append(
            f"Item 1A ({risk.word_count:,} words) is shorter than Item 1 "
            f"({business.word_count:,} words), which is unusual for a modern filer: "
            "risk factors normally run longer than the business description. Check "
            "that the Item 1A header was not matched at a cross-reference."
        )

    spanned = sum(s.end_char - s.start_char for s in sections.values())
    covered = spanned / raw_length if raw_length else 0.0
    if covered < _MIN_COVERAGE:
        flags.append(
            f"the Items found cover {covered:.0%} of the document, under "
            f"{_MIN_COVERAGE:.0%}: the split missed most of the filing, and nothing "
            "built on these sections should be published."
        )

    return flags


def load_sections(
    ticker: str,
    client: Any,
    form: str = "10-K",
    filing: dict | None = None,
) -> FilingSections:
    """Fetch a filing and split it, once per filing per process.

    ``filing`` is a row from ``EdgarClient.filings``. Pass one to pin the split to a
    particular year, as a risk diff does when it reads this year against last year;
    leave it out for the most recent filing of that form, which the client's
    knowledge date already constrains, so a point-in-time run reads the document
    that existed then rather than the newest one on file.

    Raises ``MissingDataError`` when the filer has no such form, rather than
    falling back to an adjacent one: Item 1A of a 10-K and Item 1A of a 10-Q are
    different documents with the same name, and quietly swapping them would make a
    risk diff read as a year of dramatic change.
    """
    if filing is None:
        available = client.filings(ticker, forms=(form,), limit=1)
        if not available:
            raise MissingDataError(
                f"{form} filing",
                ticker=ticker,
                hint="no filing of this form is on file for this company by the "
                "client's knowledge date",
            )
        filing = available[0]

    key = (ticker.upper(), str(filing.get("accession", "")), form.upper())
    cached = _LOAD_CACHE.get(key)
    if cached is not None:
        return _detach(cached)

    result = replace(
        split_items(client.filing_text(ticker, filing), form=form),
        ticker=ticker.upper(),
        accession=str(filing.get("accession", "")),
        form=str(filing.get("form", form)),
        filed=filing.get("filed"),
        period=filing.get("period"),
    )
    _LOAD_CACHE[key] = result
    return _detach(result)
