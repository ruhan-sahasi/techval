"""Business descriptions as dense vectors: the text tower of the similarity model.

A comp set is a claim that two companies are the same kind of business. The
engine has always taken that claim from a hand-written ticker list. This module
is the first half of measuring it instead: it turns what a company says it does,
in its own 10-K, into a vector whose cosine against another company's vector is
an estimate of how alike the two businesses are.

**The document is Item 1, not the 10-K.** Datadog's latest annual report is
405,446 characters. Item 1 (Business) is 38,928 of them. The other 366,518 are
risk factors, MD&A, statements, footnotes and exhibits, and they are written to
a template that every filer in the country shares. TF-IDF over the whole filing
measures how closely two companies follow that template, which is a fact about
their securities counsel. Item 1 is the part where a company describes its
business in its own words, and it is the only part worth vectorising for this.

**The company's own name is the leak.** Left in, the rarest and therefore
highest-weighted term in Datadog's document is "Datadog", and the nearest
neighbour of any company becomes whichever other company happens to mention it.
Similarity then collapses into a lookup: the model identifies companies rather
than describing them, scores beautifully on anything that is really a
name-matching task, and is worthless for the one question being asked, which is
who resembles a company it has never seen. Stripping the registrant's own name
and ticker from its own document is the single most important preprocessing step
here, and ``test_text.py`` tests that it changes the neighbour ranking rather
than assuming it does. This is also why ``transform`` asks for ``names`` or
``tickers`` rather than defaulting to leaving them in: a caller whose documents
genuinely carry no identity to remove says so with ``names=[None] * n``, and one
who forgot gets an error instead of a vector that quietly indexes companies.

**Vocabulary is fitted point in time.** ``fit_text_features`` refuses a corpus
containing a document filed after the fit date. The refusal looks pedantic until
you notice what the alternative does: fitting the vectorizer on the full history
and then backtesting means the 2019 fold's idf weights already know which words
mattered in 2026, the fold scores lift, and nothing about the lift is real.
Vocabulary leakage leaves no trace in the output, which is why it is checked
here rather than trusted.

**Latent semantic analysis, not an embedding model.** TF-IDF is sparse, high
dimensional and blind to the fact that "observability" and "monitoring" are the
same idea. Truncated SVD of the term-document matrix is the classical fix:
project onto the few hundred directions that carry the covariance, which puts
synonyms near each other because they co-occur with the same neighbours. It is
several decades old, it takes seconds on a few hundred filings, and a reader can
audit every step of it. ``explained_variance`` reports how much of the original
matrix survived the projection, because a reduction that kept 12% of the
variance and one that kept 70% are different objects and should not both be
called a vector.

**What this is not.** These vectors are counts of words, weighted. They do not
understand negation, they cannot tell a company entering a market from one
leaving it, and two filings that describe the same business in different
vocabulary will sit apart. The learned similarity model is meant to improve on
raw TF-IDF cosine, so raw TF-IDF cosine is kept and exposed as the baseline
through ``baseline_neighbours``: a model that does not beat it has not earned
its place.

**Dependency.** ``nlp/sections.py`` is the module that properly splits a filing
into its Items, and where it is available it should be passed as
``section_loader``, a callable ``(ticker, client) -> str`` returning Item 1.
This module does not import it, and falls back to ``item1_window`` below, which
is a deliberately small heuristic with a documented failure mode.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from ..config import Assumptions
from ..errors import ConfigError, MissingDataError

# Bumped whenever a change here would make a cached matrix mean something
# different from a freshly fitted one. It is part of the cache key, so an old
# cache is ignored rather than silently returned against new code.
SCHEMA_VERSION = 1

DEFAULT_CACHE_DIR = Path.home() / ".techval" / "ml"

# What a redacted company name is replaced by. It has to be a token rather than
# a space: dropping "Datadog" out of "the Datadog platform" would leave "the"
# and "platform" adjacent and manufacture a bigram the filing never contained.
# The placeholder lands in essentially every document, so max_df would remove it
# anyway, but it is also in the stop list below so that it can never become a
# feature meaning "this filer mentioned itself".
REDACTION = "redactedcompanyname"

# Shortest Item 1 this module will accept. A table-of-contents line and a
# cross-reference in the risk factors both look like an Item 1 heading and are
# both far shorter than the section itself; see item1_window.
MIN_ITEM1_CHARS = 2_000

# How many 10-K index entries to read per filer. Annual filings, so this is a
# lookback in years. See build_corpus: the client truncates newest first and the
# point-in-time filter runs after it, so this number is what decides whether a
# fold dated years ago can see its own filing at all.
DEFAULT_FILING_LIMIT = 25


# --------------------------------------------------------------------------- #
# What the caller has to supply
# --------------------------------------------------------------------------- #


class FilingClient(Protocol):
    """The part of ``EdgarClient`` this module needs.

    Stated as a protocol rather than imported as a type so that a test, a cache
    layer or a fixture reader can stand in for the real client without
    inheriting from anything.
    """

    def filings(
        self,
        ticker: str,
        forms: tuple[str, ...] = ("10-K",),
        since: date | None = None,
        limit: int = 20,
    ) -> list[dict]: ...

    def filing_text(self, ticker: str, filing: dict) -> str: ...


# ``nlp/sections.py`` is expected to provide something of this shape. It is
# accepted as an argument rather than imported so the two modules can be built
# and reviewed independently.
SectionLoader = Callable[[str, Any], str]


# --------------------------------------------------------------------------- #
# Stop words: the vocabulary every 10-K shares
# --------------------------------------------------------------------------- #

# Legal, structural and forward-looking-statement vocabulary. Every annual
# report on file carries it, so it describes the genre and not the company.
# ``max_df`` removes the terms that appear in nearly every document of a given
# corpus, but a corpus of forty filings is small enough that a boilerplate term
# can miss the threshold by one filer and survive with a high idf weight, which
# is the worst of both worlds. These are removed by name instead.
#
# The test applied to every word below: could it ever distinguish a tower REIT
# from a database company? "Subscribers" could, and is absent. "Pursuant" could
# not, and is present. Sector vocabulary is deliberately left in even where it
# is common, because the whole point of the exercise is to weigh it.
FILING_STOP_WORDS = frozenset(
    {
        # Structure of the document itself.
        "item", "items", "part", "annual", "report", "form", "10k", "table",
        "contents", "page", "pages", "note", "notes", "accompanying", "herein",
        "hereof", "thereof", "hereby", "above", "below", "following", "see",
        "included", "reference", "described", "set", "forth",
        # Registrant and filing vocabulary.
        "registrant", "company", "companies", "inc", "corp", "corporation",
        "incorporated", "holdings", "group", "llc", "ltd", "plc", "nv", "sa",
        "securities", "exchange", "commission", "sec", "filed", "filing",
        "filings", "disclosure", "disclosures", "act", "rule", "rules",
        "regulation", "regulations", "pursuant", "applicable", "respect",
        "respectively", "compliance", "comply", "subject", "laws", "law",
        "legal", "governmental", "jurisdictions", "jurisdiction",
        # Fiscal and calendar scaffolding.
        "fiscal", "year", "years", "quarter", "quarterly", "annually", "ended",
        "ending", "period", "periods", "date", "dates", "january", "february",
        "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december",
        # Forward-looking-statements boilerplate.
        "forward", "looking", "statement", "statements", "believe", "believes",
        "believed", "expect", "expects", "expected", "anticipate", "anticipates",
        "anticipated", "intend", "intends", "estimate", "estimates", "estimated",
        "project", "projects", "projected", "assume", "assumptions", "actual",
        "results", "differ", "materially", "material", "adverse", "adversely",
        "affect", "affected", "uncertainties", "uncertainty", "risk", "risks",
        "factors", "predict", "outcome", "outcomes", "could", "would", "should",
        "will", "might", "cannot",
        # Accounting scaffolding that is not a description of a business.
        "gaap", "consolidated", "financial", "accounting", "audited",
        "million", "millions", "billion", "billions", "thousand", "thousands",
        "approximately", "percent", "aggregate", "total", "amount", "amounts",
        # Connective filler the tokenizer keeps but that carries nothing.
        "addition", "additional", "additionally", "also", "however", "further",
        "furthermore", "therefore", "thus", "certain", "various", "including",
        "include", "includes", "well", "based", "make", "makes", "made", "use",
        "used", "uses", "using", "new", "current", "currently", "generally",
        "primarily", "significant", "significantly", "substantially", "ability",
        "able", "continue", "continued", "continues", "one", "two", "three",
        REDACTION,
    }
)


def _stop_words(token_pattern: str) -> list[str]:
    """English plus filing boilerplate, filtered to what the tokenizer can see.

    scikit-learn compares the stop list against its own tokenisation and warns
    when the two disagree, which they would here: the English list contains "a"
    and "i" and this module's token pattern requires two characters. Filtering
    first makes the warning impossible rather than suppressing it.
    """
    matcher = re.compile(f"^(?:{token_pattern})$")
    words = set(ENGLISH_STOP_WORDS) | set(FILING_STOP_WORDS)
    return sorted(w for w in words if matcher.match(w))


# A token is two or more alphanumerics containing at least one letter. Pure
# numbers are dates, page references and dollar figures, none of which describe
# a business; the lookahead keeps "5g", "4k" and "iot", which do.
TOKEN_PATTERN = r"\b(?=[a-z0-9]*[a-z])[a-z0-9]{2,}\b"


# --------------------------------------------------------------------------- #
# Finding Item 1 without a section parser
# --------------------------------------------------------------------------- #

# The item markers. No word boundary is asserted before "item", because filings
# concatenate table cells and the heading commonly arrives as "PART IItem 1.
# Business" with nothing between the two. The lookahead after the "1" is what
# separates Item 1 from Item 1A, Item 1B, Item 1C and Item 10 through 16.
#
# The dash characters are spelled as escapes rather than written literally to
# keep the house rule against dashes in source text enforceable by grep. U+2013
# is the en dash and U+2014 the em dash; filers use both, and the plain hyphen,
# between an item number and its title.
_DASHES = "\\-\u2013\u2014"
_ITEM1_HEADING = re.compile(
    rf"item\s*1(?![0-9a-z])\s*[.:{_DASHES}]?\s*(?:our\s+)?business", re.I
)
_ITEM1_BARE = re.compile(r"item\s*1(?![0-9a-z])", re.I)
_ITEM1A = re.compile(r"item\s*1a(?![0-9a-z])", re.I)


def item1_window(
    text: str, *, ticker: str | None = None, min_chars: int = MIN_ITEM1_CHARS
) -> str:
    """Item 1 (Business) out of a filing's plain text, heuristically.

    The fallback for when ``nlp/sections.py`` is not available. It is a
    heuristic and the failure mode is named rather than hidden.

    The obvious rule, take the window after the last "Item 1." up to the next
    "Item 1A", is wrong, and Datadog's 2025 10-K shows why. The string "Item 1."
    occurs in the table of contents at character 29,363, as the real heading at
    40,836, and again at 85,583 inside the risk factors, where the text reads
    "described under Part I: Item 1. Business in this Annual Report". That last
    cross-reference is followed 147,463 characters later by another mention of
    Item 1A, so the last-occurrence rule returns a 147,000 character window that
    is almost entirely risk factors, which is the precise content this module
    exists to exclude.

    What works instead, on all six filings it was checked against: take the
    first candidate whose window is longer than ``min_chars``. The table of
    contents entry is thrown out because its window is the dozen characters
    before the next contents line, cross-references are thrown out because they
    all sit after the real heading, and the real heading is what is left.

    Candidates are restricted to the heading form, an item marker followed by
    the word "business", because a bare "Item 1" also appears in prose. Where no
    heading-form candidate survives, bare markers are tried before giving up.
    That is the weaker match of the two and the section it returns may begin a
    paragraph or two early.

    Raises ``MissingDataError`` when nothing survives. It never returns the
    whole filing as a consolation: a document that is 90% risk factors would
    poison every vector fitted alongside it, and silence is the failure this
    engine does not do.
    """
    ends = [m.start() for m in _ITEM1A.finditer(text)]
    for pattern in (_ITEM1_HEADING, _ITEM1_BARE):
        for match in pattern.finditer(text):
            start = match.start()
            end = next((e for e in ends if e > start), None)
            if end is not None and end - start >= min_chars:
                return text[start:end].strip()
    raise MissingDataError(
        "Item 1 (Business)",
        ticker=ticker,
        hint=(
            f"no Item 1 heading in {len(text):,} characters of filing text is "
            f"followed by an Item 1A marker at least {min_chars:,} characters "
            "later; the document may be an older filing that predates the "
            "current item numbering, or the markup may not have stripped cleanly"
        ),
    )


# --------------------------------------------------------------------------- #
# Removing the company's identity from its own description
# --------------------------------------------------------------------------- #

# Dropped from a registered name before its remaining words are treated as the
# company's identity. "Holdings" in "CrowdStrike Holdings, Inc." names a legal
# structure, not a business.
CORPORATE_SUFFIXES = frozenset(
    {
        "inc", "inc.", "incorporated", "corp", "corp.", "corporation", "co",
        "co.", "company", "holding", "holdings", "group", "llc", "l.l.c.",
        "lp", "l.p.", "plc", "ltd", "ltd.", "limited", "nv", "n.v.", "sa",
        "s.a.", "ag", "the", "&",
    }
)

# Name words that describe an industry rather than identify a firm. Removing
# "Verizon" from Verizon's filing is the point of the exercise; removing
# "Communications" from Verizon's filing and leaving it in T-Mobile's would
# manufacture exactly the artificial difference the step exists to prevent, so
# these survive.
GENERIC_NAME_WORDS = frozenset(
    {
        "communications", "communication", "technologies", "technology",
        "systems", "networks", "network", "software", "solutions", "services",
        "media", "entertainment", "international", "industries", "global",
        "digital", "data", "cloud", "wireless", "broadcasting", "electronics",
        "semiconductor", "semiconductors", "interactive", "information",
        "enterprises", "partners", "platforms", "labs", "worldwide", "america",
        "american", "national", "united", "states",
    }
)

_WORD_SPLIT = re.compile(r"[^A-Za-z0-9&]+")

# Where a lower-case run runs straight into a capitalised word. Stripping markup
# out of an inline-XBRL filing joins adjacent table cells and headings with
# nothing between them, so Datadog's Item 1 opens "Item 1. BusinessOverviewDatadog
# is the AI-powered observability platform" as one run of characters.
#
# The right-hand side demands an upper-case letter followed by a lower-case one,
# which is what keeps this from vandalising the sector's own vocabulary: "SaaS",
# "IoT", "MongoDB" and "API" all end in a capital that starts no word, so none of
# them is split. "BusinessOverviewDatadog" is three words and all three are.
_GLUED = re.compile(r"(?<=[a-z0-9])(?=[A-Z][a-z])")


def deglue(text: str) -> str:
    """Separate words that markup stripping ran together.

    Worth doing for two reasons. The obvious one is vocabulary: a token like
    "BusinessOverviewDatadog" occurs exactly once in the corpus, is discarded by
    min_df, and takes three real words down with it. The one that matters more
    is that no word boundary exists in front of a glued name, so the company's
    own name survives every attempt to remove it, which is the leak this module
    cares most about closing.
    """
    return _GLUED.sub(" ", text)


def name_variants(name: str | None) -> list[str]:
    """The strings that identify this filer and must leave its own document.

    Returns the full registered name, the name with its corporate suffixes
    dropped, and each remaining distinctive word. Generic industry words are
    held back deliberately, for the reason given on ``GENERIC_NAME_WORDS``.

    The ticker is deliberately not here. ``scrub`` matches it case sensitively
    and these case insensitively, because lower-casing a ticker first would
    delete the word "all" from every document belonging to the ticker ALL.
    """
    out: list[str] = []
    if name:
        cleaned = " ".join(w for w in _WORD_SPLIT.split(name) if w)
        if cleaned:
            out.append(cleaned)
        words = [
            w
            for w in cleaned.split(" ")
            if w.lower().strip(".") not in CORPORATE_SUFFIXES
        ]
        short = " ".join(words)
        if short and short != cleaned:
            out.append(short)
        for word in words:
            if len(word) > 2 and word.lower() not in GENERIC_NAME_WORDS:
                out.append(word)
    # Longest first, so "Warner Bros Discovery" is removed as a phrase before
    # its parts are matched individually.
    return sorted(set(out), key=len, reverse=True)


def scrub(document: str, *, name: str | None = None, ticker: str | None = None) -> str:
    """Replace the filer's own name and ticker with a placeholder token.

    Runs ``deglue`` first, because a name with no word boundary in front of it
    cannot be matched and is exactly the occurrence that would otherwise survive.

    Case insensitive for the name, because filings mix "Datadog", "DATADOG" and
    "datadog" freely. Case sensitive for the ticker, because a three-letter
    ticker is very often an ordinary English word and deleting every occurrence
    of "all", "key" or "now" from one company's document would do more damage
    than the leak being closed.
    """
    out = deglue(document)
    for variant in name_variants(name):
        out = re.sub(rf"\b{re.escape(variant)}\b", REDACTION, out, flags=re.I)
    if ticker:
        out = re.sub(rf"\b{re.escape(ticker.upper())}\b", REDACTION, out)
    return out


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #


@dataclass
class TextCorpus:
    """Business descriptions, each stamped with the date it became knowable.

    ``as_of_by_ticker`` carries the FILING date of the 10-K the text came from,
    not the fiscal period it covers. A trainer proving that a model dated D saw
    nothing it could not have seen needs the date the document became public,
    and a December-2025 fiscal year is not public until the filing lands in
    February 2026.

    A ticker whose document could not be built is absent from ``tickers`` and
    the reason is in ``notes``. Nothing is dropped without saying so, and
    nothing is filled in.
    """

    tickers: list[str]
    documents: list[str]
    as_of_by_ticker: dict[str, date]
    source_accessions: dict[str, str]
    notes: list[str] = field(default_factory=list)
    entity_names: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.tickers) != len(self.documents):
            raise ConfigError(
                f"corpus has {len(self.tickers)} tickers and "
                f"{len(self.documents)} documents; they are parallel lists"
            )
        undated = [t for t in self.tickers if t not in self.as_of_by_ticker]
        if undated:
            raise ConfigError(
                "every document needs the date it became knowable, and "
                f"{', '.join(undated)} has none. A corpus that cannot say when "
                "each document was filed cannot be fitted point in time, which "
                "is the only way any of this is worth fitting."
            )

    def document(self, ticker: str) -> str:
        try:
            return self.documents[self.tickers.index(ticker.upper())]
        except ValueError:
            raise MissingDataError(
                "business description",
                ticker=ticker,
                hint="not in this corpus; see corpus.notes for filers that were skipped",
            ) from None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": self.tickers,
                "entity_name": [self.entity_names.get(t, "") for t in self.tickers],
                "filed": [self.as_of_by_ticker[t] for t in self.tickers],
                "accession": [self.source_accessions.get(t, "") for t in self.tickers],
                "characters": [len(d) for d in self.documents],
            }
        ).set_index("ticker")

    def rows(self) -> list[tuple[str, Any]]:
        chars = [len(d) for d in self.documents]
        filed = list(self.as_of_by_ticker.values())
        return [
            ("Companies", len(self.tickers)),
            ("Earliest filing", min(filed) if filed else None),
            ("Latest filing", max(filed) if filed else None),
            ("Median characters", int(np.median(chars)) if chars else 0),
            ("Skipped", len(self.notes)),
        ]


def _entity_name(client: Any, ticker: str) -> str | None:
    """The filer's registered name, from whichever of the two sources answers.

    Tried in cost order. ``submissions`` was already fetched and cached by the
    ``filings`` call that precedes this, so it costs nothing; ``company_facts``
    is a second document and is only reached when the first has no name. A
    client that offers neither is not an error: the ticker alone still strips
    something, and the gap is recorded in the corpus notes.
    """
    for attr, key in (("submissions", "name"), ("company_facts", None)):
        getter = getattr(client, attr, None)
        if getter is None:
            continue
        try:
            got = getter(ticker)
        except Exception:
            continue
        name = got.get(key) if key else getattr(got, "entity_name", None)
        if name:
            return str(name)
    return None


def build_corpus(
    tickers: Sequence[str],
    client: FilingClient,
    section_loader: SectionLoader | None = None,
    as_of: date | None = None,
    *,
    filing_limit: int = DEFAULT_FILING_LIMIT,
) -> TextCorpus:
    """The latest 10-K business description for each ticker, as of a date.

    Point in time twice over. The filing is the newest 10-K filed on or before
    ``as_of``, and ``as_of`` of ``None`` means no cut beyond whatever knowledge
    date the client itself was built with, so a caller who has already passed a
    ``knowledge_date`` to ``EdgarClient`` does not have to repeat it. Passing
    both is harmless; the tighter of the two wins.

    ``filing_limit`` is how far back the filing index is read, and it is not a
    performance knob. ``EdgarClient.filings`` returns newest first and cuts the
    list at its limit, while the ``as_of`` filter here runs afterwards, so a
    limit of eight asks a walk-forward fold dated 2015 to find its 10-K among
    filings from 2019 through 2026 and conclude there is none. The default of
    twenty-five covers a quarter century of annual filings, which is longer than
    XBRL has existed. Where even that does not reach, the corpus note says the
    window was exhausted rather than claiming the filer never filed.

    ``section_loader`` is the hook for ``nlp/sections.py``, called as
    ``loader(ticker, client)`` and expected to return Item 1 for that ticker
    honouring the same knowledge date. The accession recorded against the
    document is the one selected here, so a loader that reaches for a different
    filing than this module would will mislabel the source; that is the price of
    keeping the two modules independent, and it is noted in the corpus.

    A filer with no 10-K in the window, or whose Item 1 cannot be found, is
    skipped with a reason rather than given an empty document. An empty document
    would become an all-zero vector, and an all-zero vector has cosine zero with
    everything, which reads as "similar to nothing" rather than "unknown".
    """
    docs: list[str] = []
    kept: list[str] = []
    filed_by: dict[str, date] = {}
    accession_by: dict[str, str] = {}
    names: dict[str, str] = {}
    notes: list[str] = []
    if section_loader is not None:
        notes.append(
            "Item 1 text came from the supplied section loader; the accession "
            "recorded against each document is the filing this module selected."
        )

    for raw in tickers:
        ticker = raw.upper()
        try:
            filings = client.filings(ticker, forms=("10-K",), limit=filing_limit)
        except Exception as exc:
            notes.append(f"{ticker}: skipped, could not list filings ({exc})")
            continue
        available = [f for f in filings if as_of is None or f["filed"] <= as_of]
        if not available:
            # The client truncates newest first, and this filter runs after it,
            # so a full page of results that all postdate as_of does not mean
            # there is no earlier 10-K. It means the window did not reach back
            # far enough to show one, and saying otherwise would assert an
            # absence that was never checked.
            if len(filings) >= filing_limit:
                notes.append(
                    f"{ticker}: skipped, the {filing_limit} most recent 10-Ks all "
                    f"postdate {as_of}; an older one may exist outside that "
                    "window, so raise filing_limit rather than read this as an "
                    "absence of filings"
                )
            else:
                notes.append(
                    f"{ticker}: skipped, no 10-K filed on or before "
                    f"{as_of if as_of is not None else 'the client knowledge date'}"
                )
            continue
        filing = max(available, key=lambda f: f["filed"])

        try:
            if section_loader is not None:
                document = section_loader(ticker, client)
            else:
                document = item1_window(client.filing_text(ticker, filing), ticker=ticker)
        except Exception as exc:
            notes.append(f"{ticker}: skipped, no business description ({exc})")
            continue
        if not document or not document.strip():
            notes.append(f"{ticker}: skipped, business description is empty")
            continue

        name = _entity_name(client, ticker)
        if name is None:
            notes.append(
                f"{ticker}: registered name unavailable, so only the ticker "
                "itself can be stripped from this document"
            )
        else:
            names[ticker] = name
        kept.append(ticker)
        docs.append(document)
        filed_by[ticker] = filing["filed"]
        accession_by[ticker] = filing["accession"]

    return TextCorpus(
        tickers=kept,
        documents=docs,
        as_of_by_ticker=filed_by,
        source_accessions=accession_by,
        notes=notes,
        entity_names=names,
    )


# --------------------------------------------------------------------------- #
# The features
# --------------------------------------------------------------------------- #


@dataclass
class TextFeatures:
    """Fitted vectorizer, fitted projection, and the matrix they produced.

    ``matrix`` rows are L2-normalised, so the cosine between two companies is
    their dot product and nothing has to remember to normalise later.

    ``vectorizer`` and ``reducer`` are carried rather than discarded because
    ``transform`` needs them to place an unseen company in the same space, and
    because a reader who wants to know which terms drove a similarity can read
    them straight off the fitted objects.
    """

    tickers: list[str]
    matrix: np.ndarray
    dim: int
    vocabulary_size: int
    explained_variance: float
    vectorizer: Any
    reducer: Any
    fitted_through: date
    notes: list[str] = field(default_factory=list)
    strip_names: bool = True
    tfidf_matrix: Any = None

    def _index(self, ticker: str) -> int:
        try:
            return self.tickers.index(ticker.upper())
        except ValueError:
            raise MissingDataError(
                "text feature vector",
                ticker=ticker,
                hint=(
                    "this company was not in the fitted corpus; use transform() "
                    "with its business description to place it in the same space"
                ),
            ) from None

    def vector(self, ticker: str) -> np.ndarray:
        """This company's row. A copy, so a caller cannot mutate the fit."""
        return self.matrix[self._index(ticker)].copy()

    def similarity(self, a: str, b: str) -> float:
        """Cosine between two fitted companies, in the reduced space.

        A dot product because the rows are already unit length. Clipped only to
        absorb floating point drift a hair outside [-1, 1], never to hide a
        value that genuinely fell outside it.
        """
        dot = float(self.matrix[self._index(a)] @ self.matrix[self._index(b)])
        return float(np.clip(dot, -1.0, 1.0))

    def neighbours(
        self, ticker: str, k: int = 8, *, space: str = "lsa"
    ) -> list[tuple[str, float]]:
        return nearest_neighbours(self, ticker, k, space=space)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.matrix,
            index=pd.Index(self.tickers, name="ticker"),
            columns=[f"lsa_{i:03d}" for i in range(self.matrix.shape[1])],
        )

    def similarity_frame(self) -> pd.DataFrame:
        """Every pairwise cosine, for a reader who wants to see the whole block."""
        index = pd.Index(self.tickers, name="ticker")
        return pd.DataFrame(
            np.clip(self.matrix @ self.matrix.T, -1.0, 1.0), index=index, columns=index
        )

    def rows(self) -> list[tuple[str, Any]]:
        return [
            ("Companies", len(self.tickers)),
            ("Vocabulary terms", self.vocabulary_size),
            ("Reduced dimensions", self.dim),
            ("Variance retained", self.explained_variance),
            ("Fitted through", str(self.fitted_through)),
            ("Own name stripped", self.strip_names),
        ]


def _fit_date(assumptions: Assumptions, override: date | None) -> date:
    """The date the vocabulary is allowed to know about.

    ``assumptions.as_of`` is the engine-wide valuation date and is the right
    default: a model fitted for a valuation dated D must not have read anything
    filed after D. With neither an override nor an as_of there is no cut, and
    the latest filing in the corpus is used, which is the honest reading of
    "fit on everything I have".
    """
    if override is not None:
        return override
    raw = getattr(assumptions, "as_of", None)
    if raw is None:
        return date.max
    try:
        return date.fromisoformat(str(raw))
    except ValueError as exc:
        raise ConfigError(f"as_of {raw!r} is not a YYYY-MM-DD date: {exc}") from exc


def _cache_dir(assumptions: Assumptions) -> Path:
    configured = getattr(getattr(assumptions, "ml", None), "cache_dir", None)
    return Path(configured).expanduser() if configured else DEFAULT_CACHE_DIR


def _cache_key(corpus: TextCorpus, fitted_through: date, params: dict[str, Any]) -> str:
    """A hash of everything that would change the answer.

    The documents themselves are hashed, not just the tickers and the date. Two
    corpora over the same tickers on the same date can hold different text, if
    one was built with a section loader and the other with the fallback, and
    returning the first one's matrix for the second would be a silent wrong
    answer of exactly the kind this engine refuses to give.
    """
    h = hashlib.sha256()
    h.update(f"v{SCHEMA_VERSION}|{fitted_through.isoformat()}".encode())
    for key in sorted(params):
        h.update(f"|{key}={params[key]}".encode())
    for ticker, document in sorted(zip(corpus.tickers, corpus.documents)):
        h.update(f"|{ticker}:{corpus.source_accessions.get(ticker, '')}:".encode())
        h.update(hashlib.sha256(document.encode("utf-8", "replace")).digest())
    return h.hexdigest()[:32]


def fit_text_features(
    corpus: TextCorpus,
    assumptions: Assumptions,
    dim: int = 256,
    *,
    fit_through: date | None = None,
    strip_names: bool = True,
    min_df: int = 2,
    max_df: float = 0.85,
    cache: bool = True,
) -> TextFeatures:
    """TF-IDF over Item 1, projected to ``dim`` dimensions by truncated SVD.

    Parameters that are judgment calls rather than arithmetic:

    ``min_df`` of 2 discards a term that occurs in one document only. Such a
    term cannot contribute to any similarity between two companies by
    construction, and in a corpus of filings most of them are a single filer's
    product names, which is the leak the name stripping already fights.

    ``max_df`` of 0.85 discards a term present in more than 85% of documents. A
    term that nearly everyone uses separates nobody, and in a sector corpus that
    covers most of the residual legal boilerplate that survived the stop list.

    Bigrams are kept because the sector signal is in them. "Data center", "net
    retention", "content delivery" and "wireless subscribers" each mean
    something specific; split into unigrams, "data", "net", "content" and
    "wireless" scatter across every filer in the corpus and the signal is gone.

    ``strip_names`` exists to be turned off in one test and nowhere else. The
    experiment that shows what stripping is worth has to be able to run both
    ways, and a claim about preprocessing that is asserted rather than measured
    is not worth making. Fitting with it off is fitting a company identifier.

    Raises ``ConfigError`` when a document in the corpus was filed after the fit
    date. That is not a defensive check against a mistake nobody makes: fitting
    the vocabulary on the whole history and then evaluating fold by fold is the
    normal way to build one of these, it lifts every fold's score, and it leaves
    no trace in the output. Later documents belong in ``transform``, which is
    what the refusal message says.
    """
    if not corpus.tickers:
        raise ConfigError(
            "cannot fit text features on an empty corpus; "
            f"{len(corpus.notes)} filers were skipped, see corpus.notes"
        )
    if len(corpus.tickers) < 3:
        # Two documents span one direction after the mean is accounted for, and
        # min_df of 2 on two documents keeps only the terms they already share.
        # There is no projection to fit; the caller wants more documents.
        raise ConfigError(
            f"latent semantic analysis over {len(corpus.tickers)} document(s) has "
            "nothing to project onto: with fewer than three documents every "
            "surviving term is one both filings already share. Widen the corpus."
        )
    if min_df < 2:
        raise ConfigError(
            f"min_df of {min_df} keeps terms that occur in a single document, "
            "which cannot contribute to a similarity between two companies and "
            "in a filing corpus is mostly one filer's product names"
        )
    if not 0.0 < max_df <= 1.0:
        raise ConfigError(f"max_df must lie in (0, 1]; got {max_df}")

    fitted_through = _fit_date(assumptions, fit_through)
    late = {
        t: d for t, d in corpus.as_of_by_ticker.items()
        if t in corpus.tickers and d > fitted_through
    }
    if late:
        offenders = ", ".join(f"{t} filed {d}" for t, d in sorted(late.items()))
        raise ConfigError(
            f"vocabulary leakage: {len(late)} document(s) in this corpus postdate "
            f"the fit date of {fitted_through} ({offenders}). Fitting the "
            "vectorizer on documents that did not exist at the fit date gives "
            "every earlier fold idf weights computed with hindsight, which lifts "
            "the score and means nothing. Rebuild the corpus with "
            f"as_of={fitted_through}, or pass these documents to transform()."
        )
    if fitted_through == date.max:
        fitted_through = max(corpus.as_of_by_ticker[t] for t in corpus.tickers)

    seed = int(getattr(getattr(assumptions, "ml", None), "random_seed", 7))
    params = {
        "dim": dim, "min_df": min_df, "max_df": max_df,
        "strip_names": strip_names, "seed": seed,
    }
    key = _cache_key(corpus, fitted_through, params)
    cache_path = _cache_dir(assumptions) / f"text_features_{key}.joblib"
    if cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    notes: list[str] = []
    if strip_names:
        prepared = [
            scrub(doc, name=corpus.entity_names.get(t), ticker=t)
            for t, doc in zip(corpus.tickers, corpus.documents)
        ]
        missing = [t for t in corpus.tickers if t not in corpus.entity_names]
        if missing:
            notes.append(
                "registered name unknown for " + ", ".join(missing)
                + "; only the ticker was stripped from those documents"
            )
    else:
        prepared = [deglue(doc) for doc in corpus.documents]
        notes.append(
            "FITTED WITH THE COMPANY NAME LEFT IN. The highest-weighted term in "
            "each document is the filer's own name, so these vectors identify "
            "companies rather than describe them. Not for use in a valuation."
        )

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        stop_words=_stop_words(TOKEN_PATTERN),
        token_pattern=TOKEN_PATTERN,
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float64,
    )
    try:
        tfidf = vectorizer.fit_transform(prepared)
    except ValueError as exc:
        raise ConfigError(
            f"no vocabulary survived min_df={min_df} and max_df={max_df} across "
            f"{len(prepared)} documents ({exc}). With a corpus this small every "
            "term is either in one document or in nearly all of them; widen the "
            "corpus rather than the thresholds."
        ) from exc

    n_docs, n_terms = tfidf.shape
    # The term-document matrix has rank at most min(documents, terms), and the
    # last component of a rank-deficient matrix is numerical noise, so the cap
    # is one below that rather than at it.
    k = min(dim, n_terms - 1, n_docs - 1)
    if k < 2:
        raise ConfigError(
            f"cannot reduce a {n_docs} by {n_terms} term-document matrix to two "
            "or more dimensions. After pruning, this corpus has too few surviving "
            "terms to project onto anything; widen the corpus."
        )
    if k < dim:
        notes.append(
            f"requested {dim} dimensions, fitted {k}: a corpus of {n_docs} "
            f"documents over {n_terms:,} terms has no more directions than that "
            "to project onto."
        )

    reducer = TruncatedSVD(
        n_components=k, algorithm="randomized", n_iter=7, random_state=seed
    )
    reduced = reducer.fit_transform(tfidf)
    matrix = _l2_normalise(reduced, corpus.tickers)
    explained = float(reducer.explained_variance_ratio_.sum())
    if explained < 0.30:
        notes.append(
            f"the projection kept {explained:.1%} of the variance in the "
            "term-document matrix, so most of what distinguished these filings "
            "was discarded; read the cosines as coarse."
        )

    features = TextFeatures(
        tickers=list(corpus.tickers),
        matrix=matrix,
        dim=k,
        vocabulary_size=n_terms,
        explained_variance=explained,
        vectorizer=vectorizer,
        reducer=reducer,
        fitted_through=fitted_through,
        notes=notes,
        strip_names=strip_names,
        tfidf_matrix=tfidf,
    )
    if cache:
        features.notes.extend(_save_cache(cache_path, features))
    return features


def _l2_normalise(reduced: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Unit-length rows, or an error naming the company that has no length.

    A zero row happens when a document shares no surviving vocabulary with the
    rest of the corpus. Normalising it would divide by zero; leaving it would
    give it cosine zero against everything, which reads as "resembles nothing"
    when the truth is "was not measured". Neither is acceptable, so it raises.
    """
    norms = np.linalg.norm(reduced, axis=1)
    dead = [labels[i] for i in np.flatnonzero(norms == 0.0)]
    if dead:
        raise MissingDataError(
            "text feature vector",
            ticker=dead[0],
            hint=(
                f"{len(dead)} document(s) ({', '.join(map(str, dead))}) share no "
                "vocabulary with the rest of the corpus after pruning, so they "
                "project onto the origin. A zero vector is not a similarity of "
                "zero, it is an absence of measurement, so it is not returned."
            ),
        )
    return reduced / norms[:, None]


def transform(
    features: TextFeatures,
    documents: Sequence[str],
    *,
    names: Sequence[str | None] | None = None,
    tickers: Sequence[str | None] | None = None,
) -> np.ndarray:
    """Place unseen business descriptions in the fitted space.

    The vocabulary, the idf weights and the projection all come from the fit, so
    a term this company uses that nobody in the training corpus used is simply
    absent. That is the correct behaviour for a point-in-time model and it is
    also its limit: a 2019 fit cannot see the word "generative".

    ``names`` and ``tickers`` are parallel to ``documents``. When the fit
    stripped company identities, a transform that does not strip them puts each
    company's own name back into its vector, and the space stops being the same
    space. Rather than let that happen quietly this raises, and a caller who
    genuinely has documents with no identity to remove says so by passing a
    sequence of ``None``.
    """
    if features.strip_names and names is None and tickers is None:
        raise ConfigError(
            "these features were fitted with each filer's own name stripped out, "
            "so transform() needs names= or tickers= to do the same to these "
            "documents; leaving the name in puts the single highest-weighted "
            "term in the document back into the vector. Pass names=[None] * n "
            "to state that these documents carry no identity to strip."
        )
    n = len(documents)
    name_list = list(names) if names is not None else [None] * n
    ticker_list = list(tickers) if tickers is not None else [None] * n
    if len(name_list) != n or len(ticker_list) != n:
        raise ConfigError(
            f"names and tickers must be parallel to documents; got {n} documents, "
            f"{len(name_list)} names and {len(ticker_list)} tickers"
        )
    prepared = [
        scrub(doc, name=nm, ticker=tk) if features.strip_names else deglue(doc)
        for doc, nm, tk in zip(documents, name_list, ticker_list)
    ]
    reduced = features.reducer.transform(features.vectorizer.transform(prepared))
    labels = [t or f"document {i}" for i, t in enumerate(ticker_list)]
    return _l2_normalise(reduced, labels)


# --------------------------------------------------------------------------- #
# Neighbours, and the baseline a learned model has to beat
# --------------------------------------------------------------------------- #


def nearest_neighbours(
    features: TextFeatures, ticker: str, k: int = 8, *, space: str = "lsa"
) -> list[tuple[str, float]]:
    """The k most similar companies, highest cosine first, excluding self.

    ``space`` picks which representation is ranked. "lsa" uses the reduced
    vectors. "tfidf" uses the sparse term vectors the projection was built from,
    which is the baseline, and is kept on the features object precisely so that
    producing it costs one argument rather than a second fit.
    """
    if space not in {"lsa", "tfidf"}:
        raise ConfigError(f"space must be 'lsa' or 'tfidf'; got {space!r}")
    if k < 1:
        raise ConfigError(f"k must be at least 1; got {k}")
    i = features._index(ticker)
    if space == "lsa":
        sims = features.matrix @ features.matrix[i]
    else:
        if features.tfidf_matrix is None:
            raise ConfigError(
                "these features carry no term matrix, so the raw TF-IDF baseline "
                "cannot be produced from them; refit rather than guess"
            )
        # TfidfVectorizer L2-normalises its rows by default, so this dot product
        # is already a cosine.
        sims = np.asarray(
            (features.tfidf_matrix @ features.tfidf_matrix[i].T).todense()
        ).ravel()
    order = np.argsort(-sims, kind="stable")
    out: list[tuple[str, float]] = []
    for j in order:
        if int(j) == i:
            continue
        out.append((features.tickers[int(j)], float(np.clip(sims[int(j)], -1.0, 1.0))))
        if len(out) >= k:
            break
    return out


def baseline_neighbours(
    features: TextFeatures, ticker: str, k: int = 8
) -> list[tuple[str, float]]:
    """Raw TF-IDF cosine neighbours: the comparison a learned model must win.

    One call, deliberately. A baseline that takes effort to produce does not get
    produced, and a similarity model scored without one is a number with nothing
    behind it. Pass the ranking this returns as the ``baseline_name`` of
    "raw tf-idf cosine" side of an ``EvalResult``.

    It is a real baseline rather than a straw man. On a sector corpus, raw
    TF-IDF over a stripped Item 1 already puts the security software companies
    next to each other, and any learned model worth shipping has to do better
    than a word count.
    """
    return nearest_neighbours(features, ticker, k, space="tfidf")


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def _load_cache(path: Path) -> TextFeatures | None:
    """A previously fitted object, or None. Never a partially valid one.

    Any failure here means a refit, which costs seconds. Returning something
    stale or half-read to avoid that would be the wrong trade by a wide margin.
    """
    if not path.exists():
        return None
    try:
        import joblib

        got = joblib.load(path)
    except Exception:
        return None
    return got if isinstance(got, TextFeatures) else None


def _save_cache(path: Path, features: TextFeatures) -> list[str]:
    """Write the fit, and say so if it could not be written.

    A cache that cannot be written is an inconvenience, not a wrong answer, so
    it is reported through the notes rather than raised.
    """
    try:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(features, path)
    except Exception as exc:
        return [f"fitted features were not cached to {path}: {exc}"]
    return []
