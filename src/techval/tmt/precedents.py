"""Precedent transactions, built from the merger filings themselves.

A precedent set answers a different question from a comp table. Trading
comparables say what the market pays for a minority stake in a similar business
on an ordinary Tuesday. Precedents say what an acquirer paid to own the whole
thing, and that price contains two premia a trading multiple does not: control,
because a buyer who takes the board and the cash flows will pay for the right to
direct them, and synergies, because the buyer has told itself the combined firm
is worth more than the two apart. Across TMT the median control premium has
historically run roughly 25 to 40 percent over the unaffected share price. A
precedent multiple therefore sits above a trading multiple for the same company
by construction, not by coincidence, and the two are never averaged, never
blended and never printed in the same column. Every ``PrecedentSet`` carries
that sentence in its notes so it survives the journey to a page.

What this module reads, and what it refuses to invent:

**The deal documents.** A merger agreement reaches EDGAR through a small set of
forms: an 8-K under Item 1.01 within four business days of signing, a merger
proxy (PREM14A then DEFM14A), a tender offer recommendation (SC 14D9), and an
S-4 where the buyer is paying in its own stock. A target that is acquired then
stops filing, so the last documents before the record goes quiet are the deal
documents. ``find_merger_filings`` asks for those forms; it does not try to
guess a deal from a 10-K. Rule 425 communications and additional proxy material
are deliberately not on that list, because both are also used for routine annual
meeting solicitations and neither establishes that a deal exists. They are read
afterwards, once a deal is established, and only to date it.

**Item 1.01 is not a merger.** Splunk filed an Item 1.01 8-K in June 2021 for a
convertible note sale, and that document contains the phrase "an initial
conversion price of $160.00 per share". A price regex pointed at Item 1.01 will
read that as an offer. Extraction here requires the language of a merger
agreement and the language of share conversion before it will look for a number.

**The filer is not always the target.** Zendesk's own S-4 in December 2021 was
filed to buy Momentive, and it says each share "will be converted into the right
to receive 0.225" of a Zendesk share. Read carelessly that becomes a precedent
in which Zendesk was acquired at an exchange ratio of 0.225. The extractor reads
which company's shares are being converted and discards the document when the
answer is somebody else.

**Par value is not an offer price.** Nearly every merger 8-K contains the phrase
"par value $0.001 per share" within a few words of the real consideration. The
price is taken only from the consideration clause and only where it is tied to
cash payable for a share.

**A price that cannot be fixed is not fixed.** Iridium's June 2026 agreement
with Rocket Lab pays $27.00 in cash plus a number of Rocket Lab shares set by a
collar: one ratio below a reference price, a floating ratio inside the band, a
third ratio above it. No single exchange ratio exists at announcement, so none
is reported. The transaction is kept, the cash leg is kept, and the offer price
is ``None`` with the reason in ``notes``.

**The unaffected price is the judgment call, and here is ours.** A premium is
measured against the price before the market knew, and the market usually knew
something. The house convention, and the headline here, is the last close
strictly before the announcement date. It is the convention because it is the
one every reader can reconstruct, and because a longer window quietly imports
the market's whole opinion of the sector over a month. The engine also computes
the mean close over the thirty calendar days before the announcement and reports
both. Where the two premia differ by more than ``LEAK_FLAG_POINTS`` points, the
transaction is flagged: that gap is the market having already moved, and on a
leaked deal the one-day premium understates what the buyer actually paid over
the standalone value. Roku is the case the fixtures carry. The stock rose twenty
percent on Friday 12 June 2026, the merger agreement was signed over that
weekend and the 8-K came on the Monday, so the premium to the last close before
the announcement is eleven percent while the premium to the month before it is
twenty seven. Only one of those is a control premium.

The thirty-day figure is an unweighted mean of daily closes, not a volume
weighted average price, because ``PriceSeries`` carries closes and no volume:
Nasdaq's endpoint publishes a volume column and the engine discards it, the CSV
source never had one, and a figure labelled VWAP that is not one is exactly the
sort of quiet substitution this package refuses. Where the series passed in does
expose a ``volumes`` attribute aligned with ``closes``, it is used and the basis
says so.

The practical limit is worth stating plainly: no public price source serves
history for a delisted symbol, and a completed target is delisted by definition.
Splunk, Mandiant, Zendesk and Slack therefore come back from a live run with
their multiples intact and no premium at all, flagged rather than filled. Real
premia over closed deals need ``price_source: csv`` and the closes supplied.

**The announcement date is not the filing date and is not the signing date.**
An 8-K reports the date of the event, which is the date the merger agreement was
signed, often after the close. The press release goes out the next morning. The
8-K itself may not be filed for four business days. The date used here is the
earliest EDGAR filing date across the announcement cluster, which is the day the
target's own press release was filed under Rule 425 or as additional proxy
material, floored at the agreement date. On Splunk that gives 21 September 2023
rather than the 20 September signing date, and the resulting unaffected close is
the one the street quoted.

**Multiples are struck on what was public then.** Target financials are rebuilt
through a ``CompanyFacts`` pinned to the announcement date, so the trailing
twelve months are the ones a bidder could have seen, not the restated figures
that arrived later. Enterprise value runs through the standard bridge, which
means the target's net debt assumed in the deal is inside every multiple, and
the EBITDA line is whichever one the bridge's lease convention permits.

Money is USD millions, prices are dollars per share, premia and ``pct_cash`` are
decimal fractions. Nothing in this module draws a random number, so
``assumptions.ml.random_seed`` has nothing to seed here; ordering is by
announcement date and then ticker, so two runs over the same filings produce
byte-identical output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

from ..config import Assumptions
from ..edgar import SEC_TICKERS_URL, CompanyFacts
from ..errors import DataSourceError, NotMeaningfulError, TechvalError
from ..ev_bridge import build_ev_bridge
from ..financials import build_financials

# --------------------------------------------------------------------------- #
# Method constants
#
# These are rules of the method rather than valuation judgments, which is why
# they are named module constants and not fields on Assumptions: changing one
# does not change a view of a company, it changes what the engine is willing to
# call a statistic. Each is overridable at the call site.
# --------------------------------------------------------------------------- #

#: A median of three transactions is three transactions. Below this count a
#: sub-vertical gets no statistics at all, and the refusal is recorded with the
#: count so the reader knows a bucket existed and was declined.
MIN_DEALS_FOR_STATS = 5

#: Premium points of disagreement between the one-day and thirty-day unaffected
#: conventions above which the transaction is flagged as a suspected leak. Five
#: points is roughly a sixth of a typical control premium: large enough that the
#: choice of convention changes the conclusion, small enough to catch a drift
#: that a reader eyeballing a chart would miss.
LEAK_FLAG_POINTS = 5.0

#: Calendar days in the longer unaffected window. Thirty is the sell-side
#: convention for a "one month prior" reference price.
UNAFFECTED_WINDOW_DAYS = 30

#: Days after the agreement date inside which the announcement cluster is
#: gathered. A signing is announced the same day or the next morning; four
#: business days is the 8-K deadline; thirty covers a signing announced late on
#: a Friday before a holiday without reaching the proxy that follows weeks later.
CLUSTER_DAYS = 30

#: Days after an announcement inside which a deal with no deregistration on file
#: is called pending rather than unresolved. Eighteen months is long by the
#: standards of a software deal and short by the standards of one the antitrust
#: agencies have asked a second set of questions about, which is the point: past
#: it, silence stops being evidence of a deal still working its way through.
PENDING_WINDOW_DAYS = 540

#: Forms that can carry a merger agreement. DEFA14A and 425 are deliberately not
#: here: both are also used for routine annual meeting material, so neither
#: establishes a deal. They are read only once a deal is established, to date it.
DEAL_FORMS: tuple[str, ...] = (
    "8-K",
    "DEFM14A",
    "PREM14A",
    "SC 14D9",
    "SC 14D9/A",
    "S-4",
    "S-4/A",
)

#: Filed the day a deal is announced, and useless before that. Used only to pull
#: the announcement date earlier than the first document that proves the deal.
ANNOUNCEMENT_FORMS: tuple[str, ...] = ("425", "DEFA14A", "8-K")

#: Delisting and deregistration. A Form 25 removes the security from the
#: exchange and a Form 15 suspends the reporting duty. Either one filed after an
#: announcement is filed evidence that the deal closed.
COMPLETION_FORMS: tuple[str, ...] = ("25", "25-NSE", "15-12B", "15-12G")

#: The columns a precedent table quotes, as (Transaction attribute, label).
MULTIPLES: tuple[tuple[str, str], ...] = (
    ("ev_revenue", "EV/Revenue"),
    ("ev_ebitda", "EV/EBITDA"),
    ("premium_1d", "Premium to 1-day"),
    ("premium_30d", "Premium to 30-day"),
)

_STAT_ROWS = ["n", "Min", "p25", "Median", "p75", "Max"]


# --------------------------------------------------------------------------- #
# Optional collaborators
#
# tmt/taxonomy.py is being written in parallel and is not imported here. The
# shape this module expects from a classifier is one call taking a ticker and
# returning one of the sub-vertical names in TMTAssumptions.sub_vertical, or
# None where it will not guess.
# --------------------------------------------------------------------------- #


class SubVerticalClassifier(Protocol):
    """What ``build_precedents(classify=...)`` expects.

    ``__call__(ticker)`` returns a sub-vertical name or ``None``. Returning
    ``None`` is the correct answer for a filer whose business cannot be placed;
    it puts the deal in the unclassified bucket rather than in a wrong one.
    """

    def __call__(self, ticker: str) -> str | None: ...


class PriceLookup(Protocol):
    """What ``build_precedents(prices=...)`` expects.

    Satisfied by every source in ``techval.market``: ``CsvSource``,
    ``NasdaqSource`` and ``StooqSource`` all implement ``fetch``.
    """

    def fetch(self, symbol: str, start: date, end: date): ...


# SIC is a weak stand-in for a coverage banker's sector map and is used only
# when no classifier is supplied. It cannot separate infrastructure software
# from application software, and it misfiles businesses that changed shape after
# they registered: Mandiant sat under 3577, computer peripheral equipment,
# while selling incident response. Every deal classified this way is flagged.
_SIC_SUB_VERTICAL: dict[str, str] = {
    "3571": "hardware",
    "3572": "hardware",
    "3575": "hardware",
    "3576": "hardware",
    "3577": "hardware",
    "3578": "hardware",
    "3651": "hardware",
    "3661": "hardware",
    "3663": "hardware",
    "3669": "hardware",
    "3670": "hardware",
    "3672": "hardware",
    # 3674 is the semiconductor code proper. The neighbouring 367x codes are
    # electronic components, which is a different business at a different
    # multiple, so they are hardware rather than semiconductors.
    "3674": "semiconductors",
    "3559": "semiconductors",
    "7371": "application_software",
    "7372": "application_software",
    "7373": "application_software",
    "7370": "internet",
    "7374": "internet",
    "7375": "internet",
    "7379": "internet",
    "4812": "telecom",
    "4813": "telecom",
    "4899": "telecom",
    "2711": "media_entertainment",
    "2721": "media_entertainment",
    "2731": "media_entertainment",
    "4822": "media_entertainment",
    "4832": "media_entertainment",
    "4833": "media_entertainment",
    "4841": "media_entertainment",
    "7812": "media_entertainment",
    "7819": "media_entertainment",
    "7822": "media_entertainment",
    "7841": "media_entertainment",
}


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class Transaction:
    """One deal, as the filings state it and as the arithmetic follows.

    Money is USD millions, ``offer_price`` and the unaffected prices are dollars
    per share, and every premium is a decimal fraction. ``premium`` and
    ``unaffected_price`` repeat the headline convention, the one-day close, so a
    caller reading only the required fields still gets a stated basis rather
    than a silent choice; ``unaffected_basis`` names it in words.

    ``confidence`` is a coarse ladder over how much of the deal the filing text
    actually pinned down, not a probability:

    ``0.90``  an all-cash price, one unambiguous figure, from a primary form
    ``0.75``  a mixed or all-stock price completed with the acquirer's own close
    ``0.50``  consideration identified but no per-share value can be fixed
    ``0.25``  a merger agreement identified with no consideration parsed at all
    """

    target_ticker: str
    target_name: str | None = None
    target_cik: int | None = None
    acquirer_name: str | None = None
    acquirer_ticker: str | None = None
    announced: date | None = None
    closed: date | None = None
    equity_value: float | None = None
    enterprise_value: float | None = None
    offer_price: float | None = None
    unaffected_price: float | None = None
    premium: float | None = None
    consideration: str | None = None
    pct_cash: float | None = None
    ev_revenue: float | None = None
    ev_ebitda: float | None = None
    target_revenue_ttm: float | None = None
    target_ebitda_ttm: float | None = None
    sub_vertical: str | None = None
    source_form: str | None = None
    source_accession: str | None = None
    confidence: float = 0.25
    notes: list[str] = field(default_factory=list)

    # Beyond the required surface, and load-bearing for the premium argument.
    agreement_date: date | None = None
    cash_per_share: float | None = None
    exchange_ratio: float | None = None
    acquirer_price: float | None = None
    unaffected_1d: float | None = None
    unaffected_30d: float | None = None
    premium_1d: float | None = None
    premium_30d: float | None = None
    unaffected_basis: str | None = None
    ebitda_label: str | None = None
    status: str = "unresolved"
    completed: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def premium_gap_points(self) -> float | None:
        """Points of premium between the two unaffected conventions.

        The number the leak flag is struck on. A deal nobody saw coming sits
        near zero, because the last close and the month before it agree.
        Strongly negative means the shares had already run into the
        announcement, which is what a leak looks like. Positive means they had
        fallen into it, and the one-day premium is flattering the buyer.
        """
        if self.premium_1d is None or self.premium_30d is None:
            return None
        return (self.premium_1d - self.premium_30d) * 100.0

    def row(self) -> dict[str, Any]:
        return {
            "Target": self.target_ticker,
            "Target name": self.target_name,
            "Acquirer": self.acquirer_name,
            "Announced": self.announced,
            "Closed": self.closed,
            "Status": self.status,
            "Consideration": self.consideration,
            "% cash": self.pct_cash,
            "Offer price": self.offer_price,
            "Unaffected (1-day)": self.unaffected_1d,
            "Unaffected (30-day)": self.unaffected_30d,
            "Premium to 1-day": self.premium_1d,
            "Premium to 30-day": self.premium_30d,
            "Equity value": self.equity_value,
            "Enterprise value": self.enterprise_value,
            "Revenue (TTM)": self.target_revenue_ttm,
            "EBITDA (TTM)": self.target_ebitda_ttm,
            "EV/Revenue": self.ev_revenue,
            "EV/EBITDA": self.ev_ebitda,
            "Sub-vertical": self.sub_vertical,
            "Form": self.source_form,
            "Accession": self.source_accession,
            "Confidence": self.confidence,
        }

    def rows(self) -> list[tuple[str, Any]]:
        return list(self.row().items())


@dataclass
class PrecedentSet:
    """A set of precedents, its statistics, and what it refused to compute.

    ``stats`` runs over every transaction that carried a value for the column,
    with ``n`` as its first row for the same reason the comp table carries one:
    a median over two deals is two deals wearing a statistic's clothing.
    ``sub_vertical_stats`` holds the same frame per sub-vertical, and only for
    those with at least ``min_deals_for_stats`` transactions. Buckets that were
    declined appear in ``flags`` with their count.
    """

    transactions: list[Transaction]
    as_of: date
    stats: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    sub_vertical_stats: dict[str, pd.DataFrame] = field(default_factory=dict)
    min_deals_for_stats: int = MIN_DEALS_FOR_STATS
    # What was said about building the set, as distinct from what was said about
    # summarising it. ``filter`` keeps the first and recomputes the second, so a
    # narrowed set does not carry refusals about buckets it no longer contains.
    discovery_flags: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """One row per transaction, newest announcement first."""
        if not self.transactions:
            return pd.DataFrame(columns=list(Transaction("").row()))
        return pd.DataFrame([t.row() for t in self.transactions])

    def filter(
        self,
        *,
        sub_vertical: str | None = None,
        since: date | None = None,
        min_size: float | None = None,
    ) -> "PrecedentSet":
        """A narrower set, with the statistics recomputed over what is left.

        ``min_size`` is a floor on equity purchase price in USD millions. A
        transaction whose equity value could not be built is excluded by a size
        floor rather than waved through, because an unknown size cannot be shown
        to clear one; the exclusion is recorded in the new set's flags.
        """
        kept: list[Transaction] = []
        dropped_unsized = 0
        for t in self.transactions:
            if sub_vertical is not None and t.sub_vertical != sub_vertical:
                continue
            if since is not None and (t.announced is None or t.announced < since):
                continue
            if min_size is not None:
                if t.equity_value is None:
                    dropped_unsized += 1
                    continue
                if t.equity_value < min_size:
                    continue
            kept.append(t)

        # The discovery flags travel with the set whether or not the transaction
        # they describe survived the filter: a reader narrowing to one
        # sub-vertical still needs to know that two other names were dropped
        # because their financials could not be built.
        flags = list(self.discovery_flags)
        if dropped_unsized:
            flags.append(
                f"{dropped_unsized} transaction(s) dropped by the "
                f"{min_size:,.0f}mm size floor because no equity purchase price "
                "could be built for them, not because they were small."
            )
        stats, per_vertical, stat_flags = _build_stats(kept, self.min_deals_for_stats)
        return PrecedentSet(
            transactions=kept,
            as_of=self.as_of,
            stats=stats,
            notes=list(self.notes),
            flags=flags + stat_flags,
            sub_vertical_stats=per_vertical,
            min_deals_for_stats=self.min_deals_for_stats,
            discovery_flags=flags,
        )


# --------------------------------------------------------------------------- #
# Text normalisation and the consideration grammar
# --------------------------------------------------------------------------- #

# Filings arrive with typographic quotes, and every defined term in a merger
# agreement is wrapped in them. Folding them to ASCII once means no pattern
# below has to carry both forms.
_SMART = {
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u2018": "'",   # left single quotation mark
    "\u2019": "'",   # right single quotation mark, which is also the apostrophe
    "\u00a0": " ",   # non-breaking space
}

_NUM = r"[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,4})?"

# The principal conversion clause: "each ... share ... converted into the right
# to receive ...". The gap is bounded because a merger 8-K puts a long
# parenthetical about appraisal rights between the two halves and nothing
# useful more than a few hundred characters away.
_CONVERTED = re.compile(
    r"\beach\b(?P<pre>.{0,800}?)"
    r"\bconvert(?:ed|s)?\b\s*"
    r"(?:in\s+the\s+[A-Za-z ]{0,30}?merger\s+)?"
    r"into\s+(?:the\s+)?(?:contingent\s+)?right\s+to\s+receive"
    r"(?P<post>.{0,800})",
    re.I | re.S,
)

# The tender-offer and stock-for-stock phrasings of the same sentence.
_EXCHANGED = re.compile(
    r"\beach\b(?P<pre>.{0,800}?)"
    r"\b(?:be\s+)?(?:exchanged\s+for|entitled\s+to\s+receive)"
    r"(?P<post>.{0,800})",
    re.I | re.S,
)

# Award clauses share the grammar of the share clause and must not be read as a
# price. The test is which noun the sentence is about, so these are matched only
# against the head of the clause.
_AWARD_HEAD = re.compile(
    r"restricted\s+stock\s+unit|stock\s+option|option\s+to\s+purchase|"
    r"stock\s+appreciation|purchase\s+right|deferred\s+stock|\bRSU\b|\bPSU\b|"
    r"\bESPP\b|warrant|phantom",
    re.I,
)
_HEAD_CHARS = 90

# Cash consideration. The number has to be tied to cash payable, which is what
# keeps "par value $0.001 per share" out of the answer.
_CASH = re.compile(
    rf"\$\s?(?P<amt>{_NUM})\s*"
    r"(?:\([^)]{0,60}\)\s*)?"
    r"(?:per\s+share\s+)?"
    r"in\s+cash",
    re.I,
)
_CASH_TRAILING = re.compile(
    rf"\$\s?(?P<amt>{_NUM})\s+per\s+share\s*,?\s+(?:in\s+cash|without\s+interest)",
    re.I,
)

# Par value, so it can be struck out of the clause before any price is read.
# Nearly every merger 8-K puts "par value $0.001 per share" a few words from the
# real consideration, and a per-share dollar figure is exactly what is being
# looked for.
_PAR_VALUE = re.compile(rf"par\s+value\s+(?:of\s+)?\$\s?{_NUM}", re.I)

# Evidence that part of the consideration is buyer stock. Its purpose is not to
# price the stock leg but to stop an all-cash reading of a deal that pays cash
# plus shares: reporting Iridium's $27.00 cash leg as the offer price would be a
# fabrication of the worst kind, a real number in the wrong role.
_STOCK_LEG = re.compile(
    r"\bshares?\s+of\b|\bexchange\s+ratio\b|\bstock\s+consideration\b", re.I
)

# End of the sentence that grants the consideration. Awards, listings and
# aggregate deal values follow it and none of them sets the price. The
# lookbehind keeps "$0.0001" and "salesforce.com" from ending a sentence.
_SENTENCE_END = re.compile(r"(?<=[a-z\)\"])\.\s+(?=[A-Z])")

# Exchange ratios. The first form is the ratio written in front of the shares it
# buys; the second is the ratio named as a defined term.
_RATIO_INLINE = re.compile(
    r"(?<![\d.])(?P<ratio>0?\.[0-9]{2,6}|[1-9][0-9]?\.[0-9]{2,6})\s*"
    r"(?:\([^)]{0,60}\)\s*)?"
    r"(?:validly\s+issued\s+|fully\s+paid\s+|newly\s+issued\s+)*"
    r"(?:shares?|of\s+a\s+share)\s+of\b",
    re.I,
)
_RATIO_NAMED = re.compile(
    r"exchange\s+ratio[\"',]?\s*(?:will\s+be|shall\s+be|is|of|equal\s+to)\s*"
    r"(?P<ratio>0?\.[0-9]{2,6}|[1-9][0-9]?\.[0-9]{2,6})",
    re.I,
)

# The parties sentence, and the two ways a party is introduced in it.
_MERGER_AGREEMENT = re.compile(
    r"(?:entered\s+into|executed)\s+(?:an?\s+)?"
    r"(?:Agreement\s+and\s+Plan\s+of\s+Merger|Merger\s+Agreement)",
    re.I,
)
# The comma before the article is mandatory. Without it "by and among the
# Company, Cisco Systems, Inc., a Delaware corporation" matches with a name of
# "by and among" and an article of "the", which is how a party list turns into
# a company called "by and among".
_PARTY_WITH_KIND = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9.,&'\- ]{2,70}?)\s*,\s+"
    r"(?:an?|the)\s+(?P<kind>[^(]{3,140}?)"
    r"\(\s*(?P<labels>(?:\"[^\"]{1,60}\"(?:\s*(?:or|and|,)\s*)?)+)\s*\)",
    re.I,
)
_PARTY_BARE = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9.,&'\- ]{2,70}?)\s*"
    r"\(\s*\"(?P<label>[^\"]{1,60})\"\s*\)"
)
_SUBSIDIARY_OF = re.compile(
    r"(?:wholly[\s-]owned|direct|indirect)[^.]{0,60}?subsidiary\s+of\s+"
    r"(?P<parent>[A-Za-z0-9][A-Za-z0-9.,&'\- ]{1,60}?)\s*(?:\(|,|\.|$)",
    re.I,
)
# "On <date>, <Company> entered into an Agreement and Plan of Merger". The last
# clause is required: the same 8-K carries "On November 30, 2020, Stewart
# Butterfield ... entered into a written agreement", which is a voting
# undertaking by the chief executive and not the merger.
_AGREEMENT_DATE = re.compile(
    r"\bOn\s+(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4})\s*,?\s*"
    r".{0,220}?(?:entered\s+into|executed)\s+.{0,90}?"
    r"(?:Agreement\s+and\s+Plan\s+of\s+Merger|Merger\s+Agreement)",
    re.I,
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

# Words that end a company name when the regex has swallowed the connector in
# front of it.
_NAME_PREFIXES = re.compile(
    r"^(?:by\s+and\s+among|and\s+among|among|by\s+and\s+between|between|with|and|the|,)\s+",
    re.I,
)
# Labels that never name an acquirer.
_NOT_A_PARTY = re.compile(
    r"merger\s+agreement|merger\s+sub|merger\s+subsidiary|purchaser\s+sub|"
    r"^company$|^the\s+company$|effective\s+time|^merger$|^mergers$|^dgcl$|"
    r"^exchange\s+act|^securities\s+act|^board$|surviving|^transactions?$",
    re.I,
)
# Corporate suffixes, dropped before a name is matched to the SEC ticker file.
_SUFFIXES = re.compile(
    r"\b(?:incorporated|corporation|company|holdings?|group|technologies|"
    r"international|systems|inc|corp|co|llc|lp|plc|ltd|limited|sa|nv|ag|ab|"
    r"as|oyj|spa)\b",
    re.I,
)


def _normalise(text: str) -> str:
    """Typographic quotes to ASCII and whitespace to single spaces.

    Filings break lines mid-sentence, so every pattern here would otherwise
    need to tolerate a newline between any two words.
    """
    for bad, good in _SMART.items():
        text = text.replace(bad, good)
    return re.sub(r"\s+", " ", text)


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


# Lowercase words a company name is allowed to contain. Everything else in
# lower case is prose that the party regex swallowed on its way to the name:
# "by and among the Company, Cisco Systems, Inc." and "are affiliates of funds
# advised by Hellman & Friedman LLC" both end in a real name with prose in front
# of it. The domain rule keeps "salesforce.com, inc.", which is genuinely all
# lower case, from being trimmed away to nothing.
_NAME_PARTICLES = frozenset(
    {
        "and", "of", "the", "de", "der", "van", "von", "la", "le", "du", "und",
        "inc", "corp", "co", "llc", "lp", "llp", "plc", "ltd", "limited", "sa",
        "nv", "ag", "ab", "as", "spa", "gmbh", "kk", "pte", "bv", "oyj", "asa",
        "incorporated", "corporation", "company", "holdings", "holding",
        "group", "technologies", "systems", "international",
    }
)
_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9-]*\.[a-z]{2,4}$")


def _trim_to_name(raw: str) -> str:
    """Cut a captured run back to the company name at the end of it.

    A party regex that reaches leftwards through a sentence picks up the
    connective tissue in front of the name. The name is what follows the last
    word that a company name would not contain.
    """
    tokens = raw.split()
    last_prose = -1
    for i, token in enumerate(tokens):
        word = token.strip(",.;:()\"'&").lower()
        if not word or not token[:1].islower():
            continue
        if word in _NAME_PARTICLES or _DOMAIN.match(word):
            continue
        last_prose = i
    return " ".join(tokens[last_prose + 1 :]).strip(" ,;:")


def _drop_leading_chunks(name: str) -> str:
    """Drop comma-separated leading chunks that are not part of the name.

    "the Company, Cisco Systems, Inc." is three chunks, and the first of them is
    the counterparty this sentence is listing the buyer against. Chunks are kept
    from the right for as long as each one reads as part of a company name.
    """
    chunks = [c.strip() for c in name.split(",")]
    kept: list[str] = []
    for chunk in reversed(chunks):
        residue = _NAME_PREFIXES.sub("", chunk).strip()
        if not residue or _NOT_A_PARTY.search(residue):
            break
        kept.insert(0, chunk)
    if kept:
        kept[0] = _NAME_PREFIXES.sub("", kept[0]).strip()
    return ", ".join(c for c in kept if c).strip(" ,;:")


def _clean_name(raw: str) -> str:
    name = _NAME_PREFIXES.sub("", raw.strip()).strip(" ,;:")
    name = _trim_to_name(name) or name
    name = _drop_leading_chunks(name)
    return re.sub(r"\s+", " ", name).strip(" ,;:")


def _name_key(name: str) -> str:
    """A company name reduced to what is distinctive about it.

    Punctuation, case and corporate suffixes carry no information and differ
    between the filing and the SEC ticker file for the same company:
    "salesforce.com, inc." in a 2020 8-K against "Salesforce, Inc." in today's
    ticker file. Both reduce to "salesforce".
    """
    key = name.lower().replace("&", " and ")
    key = key.replace(".com", " ")
    key = re.sub(r"[^a-z0-9 ]", " ", key)
    key = _SUFFIXES.sub(" ", key)
    return re.sub(r"\s+", " ", key).strip()


def _filer_tokens(name: str) -> set[str]:
    """The words of a filer's name that identify it inside a sentence."""
    return {w for w in _name_key(name).split() if len(w) >= 3}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def _agreement_date(text: str) -> date | None:
    """The date the merger agreement was signed, as the document states it."""
    m = _AGREEMENT_DATE.search(text)
    if not m:
        return None
    month = _MONTHS.get(m.group("month").lower())
    if month is None:
        return None
    try:
        return date(int(m.group("year")), month, int(m.group("day")))
    except ValueError:
        return None


def _parties_sentence(text: str) -> str | None:
    """The clause that names who signed, bounded at the first operative verb."""
    m = _MERGER_AGREEMENT.search(text)
    if not m:
        return None
    window = text[m.start() : m.start() + 900]
    cut = re.search(r"\bpursuant\s+to\s+which\b|\bprovides\s+that\b|\bPursuant\s+to\s+the\s+Merger",
                    window, re.I)
    return window[: cut.start()] if cut else window


def _acquirer_name(text: str, filer_tokens: set[str]) -> tuple[str | None, list[str]]:
    """Who is buying, read from the party list rather than guessed.

    The party list of a merger agreement always contains the target, the buyer
    and at least one merger subsidiary formed to disappear into the target. The
    subsidiaries identify themselves as subsidiaries and are discarded, and the
    entity they say they are a subsidiary of is the buyer. Where the buyer is a
    holding company formed for the deal, that holding company is what is
    reported, because it is what the agreement says; any other named party is
    listed in the notes, which is where a sponsor such as Hellman and Friedman
    or an ultimate parent such as Publicis Groupe will appear.
    """
    sentence = _parties_sentence(text)
    if sentence is None:
        return None, []

    candidates: list[tuple[str, list[str]]] = []
    parent_refs: list[str] = []
    spans: list[tuple[int, int]] = []

    for m in _PARTY_WITH_KIND.finditer(sentence):
        spans.append(m.span())
        name = _clean_name(m.group("name"))
        kind = m.group("kind")
        labels = [lab.strip() for lab in re.findall(r"\"([^\"]{1,60})\"", m.group("labels"))]
        sub = _SUBSIDIARY_OF.search(kind)
        if sub:
            parent_refs.append(_clean_name(sub.group("parent")))
            continue
        if any(_NOT_A_PARTY.search(lab) for lab in labels):
            continue
        if _filer_tokens(name) & filer_tokens:
            continue
        candidates.append((name, labels))

    # Parties introduced without ", a Delaware corporation": salesforce.com,
    # inc. is named that way in the Slack agreement.
    for m in _PARTY_BARE.finditer(sentence):
        if any(a <= m.start() < b for a, b in spans):
            continue
        name = _clean_name(m.group("name"))
        label = m.group("label").strip()
        if _NOT_A_PARTY.search(label) or not name:
            continue
        if _filer_tokens(name) & filer_tokens:
            continue
        if any(name == c for c, _ in candidates):
            continue
        candidates.append((name, [label]))

    if not candidates:
        return None, []

    others = [name for name, _ in candidates]

    # A party the merger subs call their parent is the buyer, whatever it chose
    # to be called.
    for ref in parent_refs:
        for name, labels in candidates:
            if ref in labels or _name_key(ref) == _name_key(name):
                return name, [o for o in others if o != name]
    for name, labels in candidates:
        if any(lab.lower() in ("parent", "purchaser", "acquiror", "acquirer", "buyer")
               for lab in labels):
            return name, [o for o in others if o != name]
    return candidates[0][0], others[1:]


def _target_side(pre: str, filer_tokens: set[str]) -> tuple[bool, str]:
    """Is it the filer's own stock being converted, or somebody else's.

    The test reads the security clause, the words immediately after "share of",
    and asks whose stock it describes. A filer that names itself or says "the
    Company" is the target. A filer that names another company there is the
    acquirer, and its document is not a precedent about itself.
    """
    m = re.search(r"\bshares?\s+of\b", pre, re.I)
    clause = pre[m.start() : m.start() + 170] if m else pre[:170]
    if re.search(r"\bthe\s+Company\b|\bCompany\s+Common\s+Stock\b|\bour\s+common\s+stock\b",
                 clause, re.I):
        return True, "the clause converts the Company's own stock"
    tokens = {w for w in re.sub(r"[^A-Za-z0-9 ]", " ", clause).lower().split() if len(w) >= 3}
    if tokens & filer_tokens:
        return True, "the clause names the filer as the converted security"
    other = re.search(r"\bof\s+(?P<who>[A-Z][A-Za-z0-9.&'\-]*(?:\s+[A-Z][A-Za-z0-9.&'\-]*){0,3})",
                      clause)
    if other:
        return False, (
            f"the converted security belongs to {_clean_name(other.group('who'))}, not to "
            "the filer, so this document is the acquirer's side of somebody else's deal"
        )
    return False, "the converted security could not be tied to the filer"


def _consideration_clause(text: str, filer_tokens: set[str]) -> tuple[str, str] | None:
    """The clause that states what a share of the target is worth.

    Returns the text after the conversion verb and the reason it was accepted.
    Award clauses are rejected on the noun the sentence is about: "each option
    to purchase shares of Common Stock" converts an option, not a share, and it
    pays the merger consideration rather than setting it.
    """
    for pattern in (_CONVERTED, _EXCHANGED):
        for m in pattern.finditer(text):
            pre = m.group("pre")
            head = pre[:_HEAD_CHARS]
            share = re.search(r"\bshares?\b", head, re.I)
            award = _AWARD_HEAD.search(head)
            if share is None:
                continue
            if award is not None and award.start() < share.start():
                continue
            ok, why = _target_side(pre, filer_tokens)
            if not ok:
                return None, why
            return m.group("post"), why
    return None


def _parse_cash(clause: str) -> tuple[float | None, list[str]]:
    """Cash payable per share, or nothing, and never a par value."""
    scrubbed = _PAR_VALUE.sub(" ", clause)
    found: list[float] = []
    for pattern in (_CASH, _CASH_TRAILING):
        for m in pattern.finditer(scrubbed):
            found.append(_to_float(m.group("amt")))
    distinct = sorted(set(found))
    if not distinct:
        return None, []
    if len(distinct) > 1:
        return None, [
            "the consideration clause states "
            + " and ".join(f"${v:,.2f}" for v in distinct)
            + " in cash, so no single cash figure per share exists in it. "
            "No price is recorded rather than one of them chosen."
        ]
    return distinct[0], []


def _parse_ratio(clause: str, tail: str) -> tuple[float | None, bool, list[str]]:
    """Shares of the buyer per share of the target, and whether there are any.

    The second return value is whether the consideration includes buyer stock
    at all, which is a different question from what the ratio is. Iridium's
    agreement pays cash plus a number of Rocket Lab shares set by a collar: one
    ratio below a reference price, a floating ratio inside the band, a third
    above it. No ratio exists at announcement, but the stock leg certainly does,
    and calling that deal all-cash at its $27.00 cash leg would be worse than
    reporting no price at all.

    The tail is searched as well as the clause because a collar is defined in
    the sentences after the one that grants the consideration.
    """
    scrubbed = _PAR_VALUE.sub(" ", clause)
    present = bool(_STOCK_LEG.search(scrubbed))
    found = [_to_float(m.group("ratio")) for m in _RATIO_INLINE.finditer(scrubbed)]
    found += [_to_float(m.group("ratio")) for m in _RATIO_NAMED.finditer(scrubbed)]
    wider = [_to_float(m.group("ratio")) for m in _RATIO_NAMED.finditer(_PAR_VALUE.sub(" ", tail))]
    all_seen = sorted(set(found) | set(wider))
    if not all_seen:
        if present:
            return None, True, [
                "the consideration includes buyer stock but the clause states no "
                "exchange ratio, so none is recorded"
            ]
        return None, False, []
    if len(all_seen) > 1:
        return None, True, [
            "the agreement states "
            + ", ".join(f"{v:.4f}" for v in all_seen)
            + " as alternative exchange ratios, which is a collar: the ratio is "
            "fixed from the buyer's share price at closing and does not exist at "
            "announcement. No exchange ratio is recorded."
        ]
    return all_seen[0], True, []


def resolve_name_to_ticker(
    name: str, client, *, share_class: str | None = None
) -> tuple[str | None, list[str]]:
    """Match a company name to a ticker through the SEC's own ticker file.

    The file carries every ticker the SEC knows with the registrant name beside
    it, which makes it the one authority available offline for this mapping.
    Matching is on the name reduced by ``_name_key``, so punctuation and
    corporate suffixes do not defeat it, and only an exact match on that reduced
    form counts. A near match would happily put "Google LLC" onto "Alphabet
    Inc." and price a stock leg in the wrong security; Google LLC is not in the
    ticker file at all, and ``None`` is the honest answer.

    A dual-class issuer files one registrant name against two tickers, so Fox
    Corp is both FOX and FOXA. Where the merger agreement names the class it is
    paying in, the ticker carrying that class letter is taken. Where it does not,
    the ambiguity is reported rather than resolved by picking one.
    """
    if not name:
        return None, []
    try:
        payload = client._get_json(SEC_TICKERS_URL)
    except (DataSourceError, AttributeError, OSError):
        return None, []
    want = _name_key(name)
    if not want:
        return None, []

    matches = sorted(
        {
            str(entry["ticker"]).upper()
            for entry in payload.values()
            if isinstance(entry, dict)
            and "title" in entry
            and _name_key(entry["title"]) == want
        }
    )
    if not matches:
        return None, []
    if len(matches) == 1:
        return matches[0], []
    if share_class:
        classed = [t for t in matches if t.endswith(share_class.upper())]
        if len(classed) == 1:
            return classed[0], [
                f"{name} lists as {', '.join(matches)}; the agreement pays in class "
                f"{share_class.upper()} stock, so {classed[0]} is the series marked"
            ]
    return None, [
        f"{name} lists under more than one ticker ({', '.join(matches)}) and the "
        "agreement does not say which class it pays in, so no price series is "
        "chosen for the stock leg"
    ]


def find_merger_filings(
    ticker: str,
    client,
    lookback_years: int = 10,
    *,
    as_of: date | None = None,
) -> list[dict]:
    """Filings that could carry a merger agreement, newest first.

    Returns the raw dictionaries ``EdgarClient.filings`` produces, so the caller
    keeps the accession, the filing date, the form and the primary document.
    The list is a candidate list and nothing more: an 8-K reaches it on its form
    alone, and the Item 1.01 8-Ks a software company files are mostly credit
    agreements and note issuances. ``extract_transaction`` is what decides.

    ``as_of`` bounds the lookback window. It defaults to the client's own
    knowledge date where it has one, which keeps a historical run from reaching
    for filings that had not happened, and to today otherwise.
    """
    anchor = as_of or getattr(client, "knowledge_date", None) or date.today()
    since = anchor - timedelta(days=int(365.25 * lookback_years))
    out = client.filings(ticker, forms=DEAL_FORMS, since=since, limit=200)
    # Newest first, and among documents filed the same day the 8-K first: it is
    # the contemporaneous primary document, it states the agreement date in
    # words, and it is thirty kilobytes against a four megabyte proxy.
    return sorted(
        out,
        key=lambda f: (f["filed"], f["form"] == "8-K", f["accession"]),
        reverse=True,
    )


def extract_transaction(
    text: str,
    ticker: str,
    client,
    filing: dict,
    *,
    target_name: str | None = None,
    target_cik: int | None = None,
) -> Transaction | None:
    """Read one filing into a transaction, or decline to.

    Returns ``None`` when the document is not a merger agreement for this filer:
    a debt issuance under the same Item 1.01, a proxy about an annual meeting, a
    registration statement in which the filer is the buyer. Returns a
    ``Transaction`` with ``offer_price`` of ``None`` and the reason in ``notes``
    when the document is a merger agreement whose per-share consideration cannot
    be fixed from the text alone. It never returns a guessed price.

    ``announced`` is set to the best date this single document supports: the
    agreement date it states, falling back to its own filing date.
    ``build_precedents`` narrows it afterwards using the rest of the cluster.
    """
    flat = _normalise(text)
    if not _MERGER_AGREEMENT.search(flat):
        return None

    name = target_name
    if name is None:
        try:
            name = client.company_facts(ticker).entity_name
        except (TechvalError, AttributeError, OSError):
            name = None
    filer_tokens = _filer_tokens(name or ticker)

    found = _consideration_clause(flat, filer_tokens)
    if found is None:
        return None
    clause, why = found
    if clause is None:
        # A merger agreement for somebody else. Not this filer's precedent.
        return None

    notes: list[str] = [why]
    # Only the sentence that grants the consideration sets the price. What
    # follows it treats the equity awards, and those clauses pay the merger
    # consideration rather than defining it.
    sentence = _SENTENCE_END.split(clause, maxsplit=1)[0]
    tail_start = flat.find(clause)
    tail = flat[tail_start : tail_start + 3000] if tail_start >= 0 else clause
    cash, cash_notes = _parse_cash(sentence)
    ratio, stock_leg, ratio_notes = _parse_ratio(sentence, tail)
    notes.extend(cash_notes)
    notes.extend(ratio_notes)

    acquirer, others = _acquirer_name(flat, filer_tokens)
    if others:
        notes.append(
            "other parties named in the agreement: " + ", ".join(others[:4])
            + ". A buyer that is a holding company formed for the deal is what "
            "the agreement names; the sponsor or ultimate parent is in this list."
        )
    share_class = None
    class_match = re.search(r"\bClass\s+([A-Z])\b", sentence)
    if class_match:
        share_class = class_match.group(1)
    acquirer_ticker, ticker_notes = resolve_name_to_ticker(
        acquirer or "", client, share_class=share_class
    )
    notes.extend(ticker_notes)

    if stock_leg and cash is not None:
        consideration = "mixed"
    elif stock_leg:
        consideration = "stock"
    elif cash is not None:
        consideration = "cash"
    else:
        consideration = None
        notes.append(
            "the document states a merger agreement but no per-share cash amount "
            "or exchange ratio could be read out of its consideration clause"
        )

    offer_price = cash if consideration == "cash" else None
    pct_cash = 1.0 if consideration == "cash" else (0.0 if consideration == "stock" else None)
    if consideration == "cash":
        confidence = 0.90
    elif consideration in ("mixed", "stock"):
        confidence = 0.50
        notes.append(
            "the consideration includes buyer stock, so the offer price is only "
            "fixed once that leg is marked at the buyer's own unaffected close"
        )
    else:
        confidence = 0.25

    agreement = _agreement_date(flat)
    announced = agreement or filing.get("filed")

    return Transaction(
        target_ticker=ticker.upper(),
        target_name=name,
        target_cik=target_cik,
        acquirer_name=acquirer,
        acquirer_ticker=acquirer_ticker,
        announced=announced,
        offer_price=offer_price,
        consideration=consideration,
        pct_cash=pct_cash,
        source_form=filing.get("form"),
        source_accession=filing.get("accession"),
        confidence=confidence,
        notes=notes,
        agreement_date=agreement,
        cash_per_share=cash,
        exchange_ratio=ratio,
    )


# --------------------------------------------------------------------------- #
# Dates, prices and the premium
# --------------------------------------------------------------------------- #


def _announcement_date(ticker: str, client, txn: Transaction, filing: dict) -> tuple[date, list[str]]:
    """The day the market could first have known, from the filing record.

    An 8-K reports the date of the event and is filed up to four business days
    later; a signing after the close is announced the following morning. Neither
    of those two dates is reliably the day the news broke, but the earliest
    document in the announcement cluster is: the press release reaches EDGAR the
    same day it goes out, filed under Rule 425 or as additional proxy material.
    """
    notes: list[str] = []
    floor = txn.agreement_date or filing["filed"]
    try:
        cluster = client.filings(
            ticker,
            forms=tuple(sorted(set(ANNOUNCEMENT_FORMS + DEAL_FORMS))),
            since=floor,
            limit=200,
        )
    except (TechvalError, OSError):
        cluster = [filing]
    dates = [f["filed"] for f in cluster if floor <= f["filed"] <= floor + timedelta(days=CLUSTER_DAYS)]
    dates.append(filing["filed"])
    announced = max(min(dates), floor)
    if txn.agreement_date and announced > txn.agreement_date:
        notes.append(
            f"the merger agreement is dated {txn.agreement_date} and the first "
            f"document in the announcement cluster reached EDGAR on {announced}, "
            "which is the date the premium is measured against"
        )
    return announced, notes


def _completion(ticker: str, client, announced: date, as_of: date) -> tuple[bool, date | None, str]:
    """Whether the deal closed, on filed evidence only.

    A Form 25 delists the security and a Form 15 suspends the reporting duty.
    Either one after an announcement is the filing record saying the company
    stopped being public, which is what completion means for a target. Absence
    of one is not evidence of termination: a deal can be pending for a year
    while regulators look at it, so the third state is named rather than
    collapsed into a false.
    """
    try:
        closing = client.filings(ticker, forms=COMPLETION_FORMS, since=announced, limit=20)
    except (TechvalError, OSError):
        closing = []
    if closing:
        closed = min(f["filed"] for f in closing)
        return True, closed, "completed"
    if as_of - announced <= timedelta(days=PENDING_WINDOW_DAYS):
        return False, None, "pending"
    return False, None, "unresolved"


def _closes_before(series, when: date) -> tuple[list[date], np.ndarray]:
    idx = [i for i, d in enumerate(series.dates) if d < when]
    return [series.dates[i] for i in idx], series.closes[idx]


def _unaffected(series, announced: date) -> tuple[float | None, float | None, str, list[str]]:
    """Both unaffected conventions, and which one is the headline.

    The one-day close is the headline because it is the figure every reader can
    reconstruct from a single quote, and because a thirty-day window imports a
    month of sector news into a number meant to isolate the deal. The thirty-day
    figure is computed anyway and reported beside it: on a leaked deal it is the
    more honest denominator, and the disagreement between the two is itself the
    evidence that a leak happened.
    """
    notes: list[str] = []
    dates, closes = _closes_before(series, announced)
    if len(closes) == 0:
        return None, None, "unavailable", [
            f"no close before {announced} in the price series for {series.symbol}, "
            "so neither unaffected price could be built"
        ]
    one_day = float(closes[-1])
    if dates[-1] < announced - timedelta(days=7):
        notes.append(
            f"the last close before the announcement is {dates[-1]}, more than a "
            f"week before {announced}. The series is short of the announcement "
            "rather than the market being closed, and the one-day figure is that "
            "stale close."
        )

    start = announced - timedelta(days=UNAFFECTED_WINDOW_DAYS)
    window = [(d, c) for d, c in zip(dates, closes) if d >= start]
    if not window:
        return one_day, None, "1-day close", notes + [
            f"no closes inside the {UNAFFECTED_WINDOW_DAYS} days before "
            f"{announced}, so no longer-window reference price was built"
        ]

    volumes = getattr(series, "volumes", None)
    if volumes is not None and len(volumes) == len(series.closes):
        by_date = dict(zip(series.dates, volumes))
        weights = np.array([float(by_date.get(d, 0.0)) for d, _ in window])
        if weights.sum() > 0:
            longer = float(np.average([c for _, c in window], weights=weights))
            return one_day, longer, "1-day close", notes + [
                f"the {UNAFFECTED_WINDOW_DAYS}-day reference price is a true "
                f"volume weighted average over {len(window)} sessions, because "
                "the price series carried volume beside its closes"
            ]
    longer = float(np.mean([c for _, c in window]))
    notes.append(
        f"the {UNAFFECTED_WINDOW_DAYS}-day reference price is the unweighted mean "
        f"of {len(window)} daily closes, not a volume weighted average price: this "
        "series carries no usable volume, and a figure labelled VWAP that is not "
        "one would be worse than the plain mean"
    )
    return one_day, longer, "1-day close", notes


def _premium(offer: float | None, reference: float | None) -> float | None:
    if offer is None or reference is None or reference <= 0:
        return None
    return offer / reference - 1.0


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _stats_frame(
    transactions: Sequence[Transaction], minimum: int, label_prefix: str = ""
) -> tuple[pd.DataFrame, list[str]]:
    """Quartiles per column over the transactions that carried that column.

    ``n`` is the first row and not an afterthought. A precedent table is small
    by nature, and the difference between a median of eleven deals and a median
    of two is the whole of what the reader needs before quoting one.

    The floor bites per column, not only per table. A set of nine deals can hold
    nine revenue multiples and two EBITDA multiples, because most of the targets
    lost money, and the column with two in it is refused while the column with
    nine is reported. ``n`` stays visible in both cases so the refusal is legible
    rather than a blank row nobody can account for.
    """
    data: dict[str, list[float]] = {}
    refusals: list[str] = []
    for attr, label in MULTIPLES:
        values = np.array(
            [v for v in (getattr(t, attr) for t in transactions) if v is not None],
            dtype=float,
        )
        if values.size < minimum:
            data[label] = [float(values.size)] + [float("nan")] * 5
            if values.size:
                refusals.append(
                    f"{label_prefix}{label}: {values.size} of "
                    f"{len(transactions)} transactions carried a value, below the "
                    f"{minimum}-deal floor, so no statistic is reported for the column."
                )
            continue
        data[label] = [
            float(values.size),
            float(values.min()),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 50)),
            float(np.percentile(values, 75)),
            float(values.max()),
        ]
    return pd.DataFrame(data, index=_STAT_ROWS), refusals


def _build_stats(
    transactions: Sequence[Transaction], minimum: int
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    overall, overall_flags = _stats_frame(transactions, minimum)
    buckets: dict[str, list[Transaction]] = {}
    for t in transactions:
        buckets.setdefault(t.sub_vertical or "unclassified", []).append(t)

    per_vertical: dict[str, pd.DataFrame] = {}
    flags: list[str] = list(overall_flags)
    for name in sorted(buckets):
        group = buckets[name]
        if len(group) < minimum:
            flags.append(
                f"{name}: {len(group)} transaction(s), below the {minimum}-deal "
                "floor, so no statistics are reported for it. A median of three "
                "transactions is three transactions."
            )
            continue
        frame, bucket_flags = _stats_frame(group, minimum, label_prefix=f"{name}: ")
        per_vertical[name] = frame
        flags.extend(bucket_flags)
    return overall, per_vertical, flags


def _standing_notes() -> list[str]:
    return [
        "Precedent transactions are not trading comparables and the two are "
        "never mixed on a page. A precedent price contains a control premium "
        "and whatever synergies the buyer underwrote; a trading multiple "
        "contains neither.",
        "The median control premium across TMT has historically run roughly 25 "
        "to 40 percent over the unaffected price, so a precedent multiple sits "
        "above a trading multiple for the same company by construction. Reading "
        "the gap as evidence that the company is cheap today is reading the "
        "control premium twice.",
        "Every multiple here is struck on the offer price against the target's "
        "trailing twelve months as filed at the announcement date, with the "
        "target's net debt inside enterprise value, so it is what the buyer "
        "paid for the whole enterprise and not what the equity was quoted at.",
    ]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


@dataclass
class _Deal:
    """One resolved deal for a ticker, before prices and financials."""

    transaction: Transaction
    filing: dict


def _classify(
    ticker: str,
    client,
    assumptions: Assumptions,
    classify: SubVerticalClassifier | None,
) -> tuple[str | None, list[str]]:
    """Sub-vertical, from the caller's classifier where one was supplied.

    The order is deliberate. An explicit override in the assumptions wins,
    because a user who has stated what a company is has stated it. A classifier
    passed in wins next, because it was built for the job. The SIC stand-in is
    last and says so: SIC is a registration-time self-description that filers
    rarely revisit, and it cannot tell infrastructure software from application
    software at all.
    """
    if assumptions.tmt.sub_vertical:
        return assumptions.tmt.sub_vertical, []
    if classify is not None:
        return classify(ticker), []
    try:
        sic = str((client.submissions(ticker) or {}).get("sic") or "")
    except (TechvalError, AttributeError, OSError):
        return None, [f"{ticker}: no SIC code on file, so the deal is unclassified"]
    mapped = _SIC_SUB_VERTICAL.get(sic)
    if mapped is None:
        return None, [
            f"{ticker}: SIC {sic or 'absent'} does not map to a TMT sub-vertical, "
            "so the deal is unclassified"
        ]
    return mapped, [
        f"{ticker}: sub-vertical {mapped} comes from SIC {sic}, which is the "
        "filer's own registration-time description and is a weak classifier. "
        "Pass classify= to override it."
    ]


def _discover(
    ticker: str,
    client,
    assumptions: Assumptions,
    *,
    as_of: date,
) -> tuple[_Deal | None, list[str]]:
    """The most recent merger agreement on file for one ticker.

    Candidates are walked newest first and the first document that reads as a
    merger agreement for this filer wins, so a company that has been through
    more than one process is represented by the most recent one. A candidate
    that was read and rejected raises nothing: an Item 1.01 8-K for a note
    issuance is ordinary corporate housekeeping, not an anomaly worth a flag.
    """
    flags: list[str] = []
    try:
        candidates = find_merger_filings(
            ticker, client, assumptions.ml.mna.lookback_years, as_of=as_of
        )
    except (TechvalError, OSError) as exc:
        return None, [f"{ticker}: no filing index could be read ({exc})"]

    for filing in candidates:
        try:
            text = client.filing_text(ticker, filing)
        except (TechvalError, OSError) as exc:
            flags.append(
                f"{ticker}: {filing['form']} {filing['accession']} could not be "
                f"fetched ({exc}), so it was not read"
            )
            continue
        txn = extract_transaction(text, ticker, client, filing)
        if txn is not None:
            return _Deal(transaction=txn, filing=filing), flags
    return None, flags


def _price_series(prices: PriceLookup | None, symbol: str, announced: date, as_of: date):
    """Closes around an announcement, or ``None`` when the source has none.

    The window is asked for from a year before the announcement to the as-of
    date rather than to the announcement itself, because Nasdaq's daily endpoint
    answers an entirely past window with no rows at all. Nothing on or after the
    announcement is ever read: the unaffected price is built strictly from
    closes before it, so the wider fetch cannot leak the deal into its own
    benchmark.
    """
    if prices is None:
        return None
    try:
        return prices.fetch(symbol, announced - timedelta(days=365), max(as_of, announced))
    except (TechvalError, OSError):
        return None


def build_precedents(
    tickers: Iterable[str],
    client,
    assumptions: Assumptions,
    *,
    prices: PriceLookup | None = None,
    classify: SubVerticalClassifier | None = None,
    min_deals_for_stats: int = MIN_DEALS_FOR_STATS,
) -> PrecedentSet:
    """A precedent set for the named targets, built from their own filings.

    Each ticker contributes at most one transaction, its most recent merger
    agreement inside ``assumptions.ml.mna.lookback_years``, and a repeated
    ticker is read once. A ticker with no merger agreement on file contributes
    nothing and is not an error: most companies are not acquired, which is the
    whole reason a propensity model has a negative class.

    ``prices`` takes any object with ``fetch(symbol, start, end)``, which is
    every source in ``techval.market``. Without it the multiples and the deal
    size still build, because those come from the offer price and the filings,
    and the premium does not, because a premium without a reference price is not
    a premium. The gap is flagged rather than filled. Be warned that the live
    Nasdaq source serves no history at all for a delisted symbol, which is every
    completed target: a precedent set with real premia over closed deals needs
    ``price_source: csv`` and the closes supplied.
    """
    as_of = (
        date.fromisoformat(assumptions.as_of)
        if assumptions.as_of
        else (getattr(client, "knowledge_date", None) or date.today())
    )
    mna = assumptions.ml.mna

    transactions: list[Transaction] = []
    flags: list[str] = []
    seen: set[str] = set()

    for raw in tickers:
        ticker = str(raw).upper()
        if ticker in seen:
            continue
        seen.add(ticker)
        deal, found_flags = _discover(ticker, client, assumptions, as_of=as_of)
        flags.extend(found_flags)
        if deal is None:
            continue

        txn = deal.transaction
        announced, date_notes = _announcement_date(ticker, client, txn, deal.filing)
        txn.announced = announced
        txn.notes.extend(date_notes)

        completed, closed, status = _completion(ticker, client, announced, as_of)
        txn.completed, txn.closed, txn.status = completed, closed, status

        sub_vertical, class_flags = _classify(ticker, client, assumptions, classify)
        txn.sub_vertical = sub_vertical
        flags.extend(class_flags)

        _price_the_deal(txn, client, assumptions, prices, as_of)
        _size_the_deal(txn, client, assumptions)

        if mna.min_deal_size:
            if txn.equity_value is None:
                flags.append(
                    f"{ticker}: kept without being checked against the "
                    f"{mna.min_deal_size:,.0f}mm floor in ml.mna.min_deal_size, "
                    "because no equity purchase price could be built for it. Use "
                    "PrecedentSet.filter(min_size=...) to drop the unsized as well."
                )
            elif txn.equity_value < mna.min_deal_size:
                flags.append(
                    f"{ticker}: equity purchase price of {txn.equity_value:,.0f}mm is "
                    f"below the {mna.min_deal_size:,.0f}mm floor in "
                    "ml.mna.min_deal_size and is excluded"
                )
                continue
        transactions.append(txn)

    transactions.sort(
        key=lambda t: (-(t.announced.toordinal() if t.announced else 0), t.target_ticker)
    )
    stats, per_vertical, stat_flags = _build_stats(transactions, min_deals_for_stats)
    return PrecedentSet(
        transactions=transactions,
        as_of=as_of,
        stats=stats,
        notes=_standing_notes(),
        flags=flags + stat_flags,
        sub_vertical_stats=per_vertical,
        min_deals_for_stats=min_deals_for_stats,
        discovery_flags=list(flags),
    )


def _price_the_deal(
    txn: Transaction,
    client,
    assumptions: Assumptions,
    prices: PriceLookup | None,
    as_of: date,
) -> None:
    """Fix the offer price where stock is part of it, then the two premia."""
    announced = txn.announced
    if announced is None:
        return

    # A stock leg is worth the buyer's own unaffected close times the ratio.
    # Marking it at any later price would put the market's reaction to the deal
    # inside the price the buyer agreed to pay.
    if txn.offer_price is None and txn.exchange_ratio is not None:
        if txn.acquirer_ticker is None:
            txn.notes.append(
                "the stock leg cannot be marked: the buyer's name in the agreement "
                "does not match any registrant in the SEC ticker file, so no price "
                "series can be identified for it"
            )
        else:
            buyer = _price_series(prices, txn.acquirer_ticker, announced, as_of)
            if buyer is None:
                txn.notes.append(
                    f"the stock leg cannot be marked: no closes for "
                    f"{txn.acquirer_ticker} were available from the price source"
                )
            else:
                _, closes = _closes_before(buyer, announced)
                if len(closes):
                    txn.acquirer_price = float(closes[-1])
                    cash = txn.cash_per_share or 0.0
                    txn.offer_price = cash + txn.exchange_ratio * txn.acquirer_price
                    txn.pct_cash = cash / txn.offer_price if txn.offer_price else None
                    txn.confidence = 0.75
                    txn.notes.append(
                        f"the stock leg is marked at {txn.acquirer_ticker}'s close of "
                        f"{txn.acquirer_price:,.2f} on the last session before "
                        f"{announced}, giving {txn.exchange_ratio:.4f} shares worth "
                        f"{txn.exchange_ratio * txn.acquirer_price:,.2f} against "
                        f"{cash:,.2f} in cash. The value of the stock leg moved with "
                        "the buyer from the next morning onward."
                    )

    series = _price_series(prices, txn.target_ticker, announced, as_of)
    if series is None:
        txn.flags.append(
            f"{txn.target_ticker}: no price series, so no unaffected price and no "
            "premium. A premium measured against nothing is not reported as zero."
        )
        return

    one_day, longer, basis, notes = _unaffected(series, announced)
    txn.unaffected_1d = one_day
    txn.unaffected_30d = longer
    txn.unaffected_basis = basis
    txn.unaffected_price = one_day
    txn.notes.extend(notes)

    txn.premium_1d = _premium(txn.offer_price, one_day)
    txn.premium_30d = _premium(txn.offer_price, longer)
    txn.premium = txn.premium_1d

    gap = txn.premium_gap_points
    if gap is not None and abs(gap) > LEAK_FLAG_POINTS:
        direction = (
            "the shares had already run into the announcement, which is what a "
            "leak looks like. A press report of talks moves a stock days before "
            "any document exists to read, so the one-day close can already carry "
            "part of the premium and understate what was paid over the standalone "
            "value"
            if gap < 0
            else "the shares had fallen into the announcement, so the one-day "
            "premium flatters the deal against where the stock had been trading"
        )
        txn.flags.append(
            f"{txn.target_ticker}: the premium to the one-day close is "
            f"{txn.premium_1d:.1%} against {txn.premium_30d:.1%} to the "
            f"{UNAFFECTED_WINDOW_DAYS}-day reference, a gap of {gap:+.1f} points. "
            + direction
        )


def _size_the_deal(txn: Transaction, client, assumptions: Assumptions) -> None:
    """Equity and enterprise value at the offer, and the multiples they give.

    The target's financials are rebuilt through a ``CompanyFacts`` pinned to the
    announcement date, so the trailing twelve months are the ones on file when
    the price was agreed rather than the restated figures that arrived with the
    next 10-K. The bridge is the engine's own, so the lease convention, the
    convertible treatment and the share count are whatever the assumptions say
    they are everywhere else, and the earnings line under EV/EBITDA is whichever
    one that convention permits.
    """
    if txn.offer_price is None or txn.announced is None:
        if txn.offer_price is None:
            txn.flags.append(
                f"{txn.target_ticker}: no offer price could be fixed, so no "
                "deal size and no multiple were computed"
            )
        return
    try:
        raw = client.company_facts(txn.target_ticker).raw
        facts = CompanyFacts(raw, txn.target_ticker, knowledge_date=txn.announced)
        fin = build_financials(txn.target_ticker, facts=facts)
    except (TechvalError, AttributeError, OSError) as exc:
        txn.flags.append(
            f"{txn.target_ticker}: the target's financials as at {txn.announced} "
            f"could not be built ({exc}), so the deal size and the multiples are "
            "not reported"
        )
        return

    txn.target_cik = txn.target_cik or fin.cik
    bridge = build_ev_bridge(fin, txn.offer_price, assumptions)
    txn.equity_value = bridge.equity_value
    txn.enterprise_value = bridge.enterprise_value
    txn.target_revenue_ttm = fin.revenue

    denominator, label = bridge.multiple_denominator(fin)
    txn.target_ebitda_ttm = denominator
    txn.ebitda_label = label

    try:
        txn.ev_revenue = _ratio(bridge.enterprise_value, fin.revenue)
    except NotMeaningfulError as exc:
        txn.flags.append(f"{txn.target_ticker}: EV/Revenue is NM, {exc}")
    try:
        txn.ev_ebitda = _ratio(bridge.enterprise_value, denominator)
    except NotMeaningfulError as exc:
        line = label.split(" ")[0]
        tail = (
            " A buyer paying a control premium for a business with no positive "
            "EBITDA is paying for revenue and for what it intends to do with it."
            if denominator is not None
            else ""
        )
        txn.flags.append(f"{txn.target_ticker}: EV/{line} is NM, {exc}.{tail}")

    threshold = assumptions.comps.ev_ebitda_nm_threshold
    if txn.ev_ebitda is not None and txn.ev_ebitda > threshold:
        txn.flags.append(
            f"{txn.target_ticker}: EV/EBITDA of {txn.ev_ebitda:,.0f}x is above the "
            f"{threshold:,.0f}x cut-off and is suppressed. It measures how near the "
            "denominator sits to zero, not what the buyer thought it was buying."
        )
        txn.ev_ebitda = None

    if fin.as_of < txn.announced - timedelta(days=200):
        txn.notes.append(
            f"the trailing twelve months end {fin.as_of}, more than six months "
            f"before the {txn.announced} announcement. The buyer saw a fresher "
            "quarter than the filing record carries at this date."
        )


def _ratio(numerator: float, denominator: float | None) -> float:
    if denominator is None:
        raise NotMeaningfulError("the denominator is not reported")
    if denominator <= 0:
        raise NotMeaningfulError(f"the denominator of {denominator:,.0f}mm is not positive")
    return numerator / denominator


def deal_events(
    tickers: Iterable[str],
    client,
    assumptions: Assumptions,
) -> list[tuple[str, date, bool]]:
    """Labels for an M&A propensity model: who was bid for, when, and did it close.

    One tuple per ticker that has a merger agreement on file inside the lookback
    window. A ticker that returns nothing is the negative class, and a propensity
    model needs both: the universe it scores minus the tickers here is the set of
    companies nobody bid for.

    The date is the label date and it is the announcement date, not the closing
    date and not the filing date of the 8-K. That matters more than it sounds.
    A model trained on these labels must build every feature from facts filed
    strictly before this date, because the target's own filings in the weeks
    after it are about the deal: a merger proxy carries management's forecast, an
    8-K carries the change of control, and a model that sees either has learned
    to predict an acquisition from the acquisition. ``CompanyFacts`` takes a
    ``knowledge_date`` for exactly this, and it should be set to the day before
    the label date rather than to the label date itself.

    The boolean is completion, on filed evidence: a Form 25 or Form 15 after the
    announcement. ``False`` means not completed as at the as-of date, which
    covers a deal still working through regulators as well as one that broke.
    ``build_precedents`` keeps the distinction in ``Transaction.status``; this
    signature cannot, so do not read a ``False`` here as a failed deal.
    """
    as_of = (
        date.fromisoformat(assumptions.as_of)
        if assumptions.as_of
        else (getattr(client, "knowledge_date", None) or date.today())
    )
    events: list[tuple[str, date, bool]] = []
    seen: set[str] = set()
    for raw in tickers:
        ticker = str(raw).upper()
        if ticker in seen:
            continue
        seen.add(ticker)
        deal, _flags = _discover(ticker, client, assumptions, as_of=as_of)
        if deal is None:
            continue
        announced, _notes = _announcement_date(ticker, client, deal.transaction, deal.filing)
        completed, _closed, _status = _completion(ticker, client, announced, as_of)
        events.append((ticker, announced, completed))
    events.sort(key=lambda e: (e[1], e[0]))
    return events
