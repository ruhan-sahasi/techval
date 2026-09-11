"""The TMT universe and its sub-verticals.

Everything downstream picks its candidates through this module, so a name landing
in the wrong bucket does not merely get a wrong label. It gets ranked against the
wrong peers, priced off the wrong multiple, and quoted in the wrong section of the
memo. The classification is therefore built to be auditable rather than clever,
and it says out loud which of its two sources of evidence won.

**SIC is the spine.** The SEC assigns every filer exactly one Standard Industrial
Classification code, publishes it in the submissions payload as ``sic`` and
``sicDescription``, and it is the only industry label in this pipeline that comes
from the filing rather than from a vendor with a licence fee. Most codes in
``SIC_TO_SUB_VERTICAL`` below are annotated with the filers actually observed
under them in ``tests/fixtures/submissions_tmt.json``, so the map can be checked
against the register rather than against intuition. The codes for which no filer
in this universe supplies that check are listed in ``SIC_UNVERIFIED`` rather than
left looking as though they had one.

**SIC is necessary and not sufficient, and that gap is the substance of the
module.** Four failures, each of them visible in that fixture:

*The code describes the filing entity, not the business.* American Tower, Crown
Castle, SBA, Equinix and Digital Realty all file under 6798, Real Estate
Investment Trusts, because that is their tax election. So do shopping malls and
apartment landlords. Mapping 6798 to towers would be worse than useless, so 6798
is deliberately absent from the map and listed instead as a code that requires
evidence before it admits anything at all.

*The code is assigned once and the business moves.* Amazon files under 5961,
Retail-Catalog and Mail-Order Houses, which was accurate in 1997 and now sits on
top of the largest cloud business in the world. Netflix still files under 7841,
Services-Video Tape Rental, the code for renting cassettes. IBM files under 3570,
Computer and Office Equipment, with most of its gross profit coming from software
and consulting.

*One code covers businesses that trade nothing alike.* SIC 7372, Prepackaged
Software, is the single code for Microsoft, Datadog, Take-Two Interactive, Roblox,
Synopsys and Salesforce. An EV/Revenue drawn from that peer set would average a
consumption-priced database company with a hit-driven console publisher. Worse,
7372 is not even where the largest security names sit: Palo Alto Networks and
Fortinet both file under 3577, Computer Peripheral Equipment, because they once
shipped appliances.

*Some sub-verticals have no code at all.* There is no SIC code for payments. Visa,
Mastercard, PayPal, FIS and Global Payments all file under 7389, Services-Business
Services NEC, the widest code in the register, alongside Accenture, Uber, eBay and
DoorDash. Toast files under 7374 with Workday and DXC. So every payments name in
this universe is admitted on evidence or on a stated human prior, never on its
code, and ``SIC_TO_SUB_VERTICAL`` has no payments entry to pretend otherwise.

So classification runs SIC first, then scores keyword evidence out of the business
description, then reconciles the two. Where evidence overturns the code, the
``source`` string says so and names the code it beat. Nothing is relabelled
quietly.

**Point in time, and the limits of it.** A universe built as of a past date
excludes filers that were not yet reporting then, and the admission date is the
earliest periodic report (10-K or 10-Q) on record rather than the earliest filing
of any kind. The distinction matters: Roblox's CIK carries submissions from 2005
against a 2021 listing, because a private company files Form D notices for its
exempt rounds. Admitting a name on its first Form D would put companies into a
historical universe years before they had a price.

Two limits survive that, and both are declared on the affected record rather than
buried here. The first is the recent-submission window. For a prolific filer the
payload truncates ``filings.recent`` and pushes older submissions into shards that
carry only date boundaries, no form types, so for those names the admission date
falls back to the earliest shard boundary and the Company record says that the
date is the start of the EDGAR record rather than a proven first report.

The second is survivorship, and it is the larger one. The SEC
``company_tickers.json`` file that resolves a ticker to a CIK is today's list.
Every TMT name delisted, taken private, or absorbed between the valuation date and
now is missing from it, and those are disproportionately the losers. The seed list
below is deliberately left carrying several of them so the gap is visible in the
output instead of invisible in the sample. Worse than absence, a ticker can be
reused: PARA resolved to Paramount for years and today resolves to an unrelated
small-cap, because Paramount became Paramount Skydance and files as PSKY. A
string that means one company today and another company in 2023 is not an
identifier, which is why every Company record carries its CIK. This universe is
therefore survivorship biased in the same way and for the same reason that
``techval.backtest`` already declares for its own ticker list. The honest fix is a
point-in-time ticker file or a delisting-complete vendor file, and this engine has
neither. No breadth statistic computed over this universe should be read as though
it did.

Units follow the rest of the engine: market capitalisation in USD millions, which
is what ``build_ev_bridge`` returns as equity value.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from ..errors import ConfigError, DataSourceError, MissingDataError
from ..ev_bridge import build_ev_bridge
from ..financials import build_financials


class SubVertical(str, Enum):
    """The eleven buckets the engine is willing to compare companies within.

    The split between infrastructure and application software is the one that
    moves a multiple most. Infrastructure software is consumption priced and its
    gross margin is constrained by the cloud bill underneath it; application
    software is seat priced and grows by landing and expanding. They do not trade
    at the same EV/Revenue for the same growth rate, and pooling them is the
    commonest way a software comp set goes quietly wrong.

    Towers and fibre are separated from telecom for the mirror-image reason. A
    tower company is a landlord with escalating multi-decade leases and three
    tenants; a carrier is an operating business with churn, spectrum and capex per
    home passed. One is valued on AFFO and lease-up, the other on EBITDA and
    subscriber economics.
    """

    INFRASTRUCTURE_SOFTWARE = "infrastructure_software"
    APPLICATION_SOFTWARE = "application_software"
    INTERNET = "internet"
    SEMICONDUCTORS = "semiconductors"
    HARDWARE = "hardware"
    IT_SERVICES = "it_services"
    PAYMENTS = "payments"
    MEDIA_ENTERTAINMENT = "media_entertainment"
    TELECOM = "telecom"
    TOWERS_FIBER = "towers_fiber"
    GAMING = "gaming"


# --------------------------------------------------------------------------- #
# SIC codes
# --------------------------------------------------------------------------- #

# Keys are the four-character SEC code as a string, because that is how the
# submissions payload carries it and because codes below 1000 need their leading
# zero. Filers named in the comments were read out of the committed submissions
# fixture, so each block can be checked rather than taken on trust.
#
# Note what is absent and why: 6798 (REITs), 7389 (services NEC), 5961 (catalogue
# retail), 7340 (building services), 4700 (transportation services), 7374 (data
# processing) and 6199 (finance services) all carry large TMT filers and large
# non-TMT filers, so they appear in SIC_NEEDS_EVIDENCE instead. There is also no
# payments entry anywhere in this map, because the SEC has no payments code.
SIC_TO_SUB_VERTICAL: dict[str, SubVertical] = {
    # -- semiconductors ---------------------------------------------------- #
    # 3674 Semiconductors and Related Devices. The core code, and the cleanest
    # block in the map: Nvidia, Broadcom, AMD, Texas Instruments, Micron, Intel,
    # Analog Devices, Microchip, Marvell, ON and NXP all sit here. So does Applied
    # Materials, which sells the deposition and etch tools rather than the chips,
    # and which trades on the same capex cycle as the fabs that buy them.
    "3674": SubVertical.SEMICONDUCTORS,
    # 3559 Special Industry Machinery NEC. Lam Research files here rather than
    # under 3674. Same cycle as its customers, different code.
    "3559": SubVertical.SEMICONDUCTORS,
    # 3827 Optical Instruments and Lenses. KLA files here, because process control
    # is metrology. The code says nothing about semiconductors and the business is
    # entirely levered to wafer starts.
    "3827": SubVertical.SEMICONDUCTORS,
    # -- hardware ---------------------------------------------------------- #
    # 3570 Computer and Office Equipment. HP Inc and Hewlett Packard Enterprise,
    # and the trap in the block: IBM files here too. A hardware code on a company
    # whose profit is software and consulting is the reason SEED carries a prior
    # for IBM and the reason the source string has to name what it overturned.
    "3570": SubVertical.HARDWARE,
    # 3571 Electronic Computers. Apple, Dell, Super Micro. Apple filing as a
    # computer manufacturer is its own reminder of how little the code tracks the
    # business it is attached to.
    "3571": SubVertical.HARDWARE,
    # 3572 Computer Storage Devices. NetApp, Seagate, Western Digital.
    "3572": SubVertical.HARDWARE,
    # 3576 Computer Communications Equipment. Cisco and Arista.
    "3576": SubVertical.HARDWARE,
    # 3577 Computer Peripheral Equipment NEC. Palo Alto Networks and Fortinet both
    # file here, from the years when a firewall was a box you racked. Both are now
    # subscription security software and neither belongs in a hardware comp set,
    # which is why 3577 is in WEAK_SIC and why the keyword layer is expected to
    # take these two off it.
    "3577": SubVertical.HARDWARE,
    # 3651 Household Audio and Video Equipment, 3661 Telephone and Telegraph
    # Apparatus, 3663 Radio and TV Broadcasting and Communications Equipment, 3669
    # Communications Equipment NEC. The equipment vendors. Motorola Solutions files
    # under 3663 and so, unhelpfully, does Qualcomm: the largest mobile chip
    # designer in the world carries a broadcasting equipment code, which is why
    # SEED holds a semiconductor prior for it.
    "3651": SubVertical.HARDWARE,
    "3661": SubVertical.HARDWARE,
    "3663": SubVertical.HARDWARE,
    "3669": SubVertical.HARDWARE,
    # 3670/3672/3678/3679 Electronic Components and Accessories, printed circuit
    # boards, connectors, components NEC. Passive and interconnect suppliers, whose
    # economics are assembly and bill of materials, not silicon design.
    "3670": SubVertical.HARDWARE,
    "3672": SubVertical.HARDWARE,
    "3678": SubVertical.HARDWARE,
    "3679": SubVertical.HARDWARE,
    # -- internet ---------------------------------------------------------- #
    # 7370 Services-Computer Programming, Data Processing, Etc. Alphabet, Meta,
    # Snap, Pinterest, The Trade Desk and AppLovin file here, which makes internet
    # the right prior. Zoom files here too and is application software, so the code
    # is wide enough to belong in WEAK_SIC.
    "7370": SubVertical.INTERNET,
    # 7375 Services-Information Retrieval Services. The older search and online
    # information code.
    "7375": SubVertical.INTERNET,
    # -- software ---------------------------------------------------------- #
    # 7372 Services-Prepackaged Software. The weakest node in the map and the one
    # carrying the most names: one code for Microsoft, Oracle, Salesforce,
    # ServiceNow, Adobe, Intuit, Autodesk, Cadence, Synopsys, Snowflake, Datadog,
    # MongoDB, Cloudflare, CrowdStrike, Okta, Palantir, Shopify, Take-Two, Unity
    # and Roblox. Application software is the modal answer among those, so it is
    # the prior, and keyword evidence routinely and correctly moves a filer off it
    # into infrastructure software or gaming.
    "7372": SubVertical.APPLICATION_SOFTWARE,
    # -- IT services ------------------------------------------------------- #
    # 7371 Services-Computer Programming Services. Cognizant and EPAM, the project
    # and headcount businesses valued on utilisation and bill rates rather than on
    # recurring revenue. Zscaler also files here and is infrastructure software,
    # so this code sits in WEAK_SIC as well.
    "7371": SubVertical.IT_SERVICES,
    # 7373 Services-Computer Integrated Systems Design. Systems integration by
    # name. Jack Henry files here and runs core banking processing, so the code
    # loses to evidence readily and is weak.
    "7373": SubVertical.IT_SERVICES,
    # 7379 Services-Computer Rental, Leasing and Services NEC.
    "7379": SubVertical.IT_SERVICES,
    # 8742 Services-Management Consulting. Genpact files here, alongside every
    # generalist consultancy in the market. Accenture, despite the obvious fit,
    # does not: it files under 7389 with everything else.
    "8742": SubVertical.IT_SERVICES,
    # -- telecom ----------------------------------------------------------- #
    # 4812 Radiotelephone Communications (T-Mobile) and 4813 Telephone
    # Communications Except Radiotelephone (AT&T, Verizon, Lumen, and Uniti, which
    # is fibre rather than a carrier and carries a prior for it). Access networks
    # sold by subscription: ARPU, churn, capex per home or per site. 4899
    # Communications Services NEC is the satellite operators' code and is mapped
    # on the code definition alone.
    "4812": SubVertical.TELECOM,
    "4813": SubVertical.TELECOM,
    "4899": SubVertical.TELECOM,
    # 4841 Cable and Other Pay Television Services. Mapped to telecom because the
    # cable systems that dominate it by market value, Comcast and Charter, are
    # broadband access businesses measured in homes passed, penetration and ARPU,
    # which is the telecom model rather than the content model. The trap is that
    # the same code catches Warner Bros Discovery and Roku, which are content and
    # advertising businesses with no access network at all, so 4841 is weak and
    # the two of them carry media priors. Comcast itself is both, which no single
    # label can fix: that is a sum-of-the-parts question, not a taxonomy one.
    "4841": SubVertical.TELECOM,
    # -- media and entertainment ------------------------------------------- #
    # 4833 Television Broadcasting Stations holds Fox, Nexstar and Paramount
    # Skydance: advertising and retransmission revenue. 4832 Radio Broadcasting
    # Stations is its audio twin and also, less obviously, where the audio
    # streaming services file.
    "4832": SubVertical.MEDIA_ENTERTAINMENT,
    "4833": SubVertical.MEDIA_ENTERTAINMENT,
    # 7812 Services-Motion Picture and Video Tape Production, 7822 Motion Picture
    # and Video Distribution, 7841 Services-Video Tape Rental. Netflix still files
    # under 7841, the Blockbuster-era code, which is the single cleanest
    # illustration in the register that a SIC code records what a company was when
    # it registered and is never revisited.
    "7812": SubVertical.MEDIA_ENTERTAINMENT,
    "7822": SubVertical.MEDIA_ENTERTAINMENT,
    "7841": SubVertical.MEDIA_ENTERTAINMENT,
    # 7900 Services-Amusement and Recreation (Live Nation, Warner Music) and 7990
    # Services-Miscellaneous Amusement and Recreation (Disney). 7990 is weak: it
    # also holds the sports betting operators, whose economics are a book, not a
    # content library.
    "7900": SubVertical.MEDIA_ENTERTAINMENT,
    "7990": SubVertical.MEDIA_ENTERTAINMENT,
    # 2711 Newspapers, 2721 Periodicals. Publishing, now sold as digital
    # subscriptions. The New York Times files under 2711.
    "2711": SubVertical.MEDIA_ENTERTAINMENT,
    "2721": SubVertical.MEDIA_ENTERTAINMENT,
    # 7311 Services-Advertising Agencies. Omnicom sits between the advertiser and
    # the media owner and is conventionally covered with media.
    "7311": SubVertical.MEDIA_ENTERTAINMENT,
    # -- gaming ------------------------------------------------------------ #
    # 7993 Services-Coin Operated Amusement Devices. The arcade and slot machine
    # code, where the gambling equipment makers file. Note what is not here: every
    # video game publisher in the fixture, Take-Two, Roblox and Unity, files under
    # 7372 next to enterprise software, and Playtika files under 7374 next to
    # payroll processing. Gaming is resolved from evidence far more often than
    # from the code.
    "7993": SubVertical.GAMING,
}


# Codes whose filers are frequently TMT and frequently not, so the code alone must
# neither admit a company to the universe nor assign it a bucket. These are kept
# out of SIC_TO_SUB_VERTICAL on purpose: a lookup that returns a plausible answer
# for a shopping mall is more dangerous than one that returns nothing. The reason
# string is printed in the source line, so a reader can see what the code failed
# to settle.
SIC_NEEDS_EVIDENCE: dict[str, str] = {
    "6798": (
        "REIT election, shared by American Tower, Crown Castle, SBA, Equinix and "
        "Digital Realty with every shopping mall and apartment landlord"
    ),
    "5961": "retail catalogue and mail order, the code Amazon has filed under since 1997",
    "7389": (
        "services not elsewhere classified, the widest code in the register: Visa, "
        "Mastercard, PayPal, FIS, Global Payments, Accenture, Uber, eBay and "
        "DoorDash all file under it"
    ),
    "7374": (
        "data processing and preparation, which in the fixture holds one payments "
        "company, one IT services firm, one HR software vendor and one mobile game "
        "publisher"
    ),
    "7340": "services to dwellings and other buildings, the code Airbnb files under",
    "4700": "transportation services, shared by Booking and Expedia with actual carriers",
    "6199": "finance services, a lending code that says nothing about a network",
}


# Mapped codes that no filer in the seed universe carries, so the fixture offers
# no check on them. They are in the map on the strength of the SEC's own code
# definition, which is a weaker warrant than a filing, and they are named here so
# that the difference is visible and stays visible: a test asserts that every
# mapped code is either observed in the fixture or declared in this set. The list
# is mostly the component and equipment tail, plus the film and periodical codes
# and 7993, the coin-operated amusement code that the video game publishers
# conspicuously do not use.
SIC_UNVERIFIED: frozenset[str] = frozenset(
    {
        "2721", "3651", "3661", "3669", "3670", "3672", "3678", "3679", "4832",
        "4899", "7375", "7379", "7812", "7822", "7993",
    }
)


# Codes that map to something, but weakly enough that evidence may overturn them
# on a smaller margin. Each is here because the fixture shows it holding filers
# from more than one sub-vertical: 7372 all software, 7370 internet and Zoom,
# 7371 IT services and Zscaler, 7373 integration and Jack Henry, 3577 peripherals
# and the two largest security software vendors, 4841 cable and Warner Bros
# Discovery, 7990 Disney and the sports books, 8742 every consultancy.
WEAK_SIC: frozenset[str] = frozenset(
    {"7370", "7371", "7372", "7373", "7379", "3577", "4841", "7990", "8742"}
)


# --------------------------------------------------------------------------- #
# Keyword evidence
# --------------------------------------------------------------------------- #

# Phrases chosen to discriminate rather than to cover: each should be language a
# company in that bucket uses about itself and companies in other buckets mostly
# do not. Generic software vocabulary such as "platform", "solutions" and
# "customers" is deliberately absent, because a phrase that matches everything
# separates nothing. A phrase may appear under more than one heading where the
# ambiguity is real, and the margin test below is what settles those cases.
PHRASES: dict[SubVertical, tuple[str, ...]] = {
    SubVertical.INFRASTRUCTURE_SOFTWARE: (
        "observability", "cloud infrastructure", "developer platform", "database",
        "data warehouse", "kubernetes", "containers", "devops", "apis",
        "cybersecurity", "endpoint protection", "zero trust", "threat detection",
        "security operations", "identity and access management", "firewall",
        "content delivery network", "log management", "infrastructure software",
        "consumption based pricing", "compute and storage", "next generation firewall",
    ),
    SubVertical.APPLICATION_SOFTWARE: (
        "customer relationship management", "enterprise resource planning",
        "human capital management", "software as a service", "saas",
        "subscription software", "workflow", "crm", "erp", "seats",
        "collaboration software", "electronic design automation",
        "marketing automation", "electronic signature", "back office",
        "per seat", "business applications", "vertical software", "payroll",
    ),
    SubVertical.INTERNET: (
        "marketplace", "online advertising", "digital advertising", "search engine",
        "social network", "e-commerce", "gross merchandise value", "take rate",
        "monthly active users", "third party sellers", "online travel", "listings",
        "ride sharing", "food delivery", "buyers and sellers", "impressions",
        "advertising revenue", "network effects", "hosts and guests",
    ),
    SubVertical.SEMICONDUCTORS: (
        "semiconductor", "semiconductors", "wafer", "wafers", "foundry", "fabless",
        "integrated circuits", "nanometer", "lithography", "tape out", "design wins",
        "assembly and test", "analog", "process technology", "chips", "silicon",
        "packaging and test", "wafer fabrication",
    ),
    SubVertical.HARDWARE: (
        "servers", "personal computers", "notebooks", "printers", "peripherals",
        "networking equipment", "switches and routers", "storage systems",
        "smartphones", "wearables", "original equipment manufacturer",
        "bill of materials", "channel partners", "manufacturing capacity",
        "component costs", "hardware appliances",
    ),
    SubVertical.IT_SERVICES: (
        "consulting", "systems integration", "managed services", "outsourcing",
        "application development", "billable", "utilization rate",
        "delivery centers", "staff augmentation", "offshore", "professional services",
        "digital transformation", "engagements", "billable headcount",
    ),
    SubVertical.PAYMENTS: (
        "payment processing", "merchant acquiring", "interchange", "payment volume",
        "card network", "issuers", "acquirers", "point of sale", "settlement",
        "chargebacks", "transactions processed", "money transfer",
        "payment gateway", "authorization", "payments network", "cardholders",
    ),
    SubVertical.MEDIA_ENTERTAINMENT: (
        "streaming", "box office", "theatrical", "studios", "content library",
        "advertising supported", "broadcast", "affiliate fees", "retransmission",
        "linear television", "film and television", "recorded music", "live events",
        "programming", "theme parks", "syndication", "content amortization",
    ),
    SubVertical.TELECOM: (
        "wireless", "postpaid", "prepaid", "churn", "arpu", "spectrum", "broadband",
        "fiber to the home", "homes passed", "roaming", "backhaul", "5g",
        "access lines", "network deployment", "universal service", "subscriber lines",
    ),
    SubVertical.TOWERS_FIBER: (
        "towers", "macro towers", "small cells", "colocation", "tenants",
        "ground leases", "site leasing", "interconnection", "dark fiber",
        "carrier neutral", "data centers", "distributed antenna", "escalators",
        "real estate investment trust", "tower sites", "lease amendments",
    ),
    SubVertical.GAMING: (
        "video game", "video games", "interactive entertainment", "in game",
        "live services", "game engine", "franchises", "microtransactions",
        "console", "mobile games", "user generated content", "bookings", "titles",
        "game development", "downloadable content", "players",
    ),
}


def _compile(phrase: str) -> re.Pattern[str]:
    """Whole-phrase match, tolerant of the line breaks a stripped 10-K is full of."""
    body = r"\s+".join(re.escape(word) for word in phrase.split())
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


_PATTERNS: dict[SubVertical, tuple[tuple[str, re.Pattern[str]], ...]] = {
    vertical: tuple((phrase, _compile(phrase)) for phrase in phrases)
    for vertical, phrases in PHRASES.items()
}


# --------------------------------------------------------------------------- #
# The confidence ladder
# --------------------------------------------------------------------------- #

# Flat rungs, so a confidence in a printed table reads back to a reason rather
# than to an opaque function. Within a rung the number moves with how decisively
# the evidence won, which is what STRENGTH_BONUS is for.
CONFIDENCE_SIC_ONLY = 0.70
"""SIC maps to a sub-vertical and there was no usable business text to check it."""

CONFIDENCE_AGREE_BASE = 0.80
"""Code and evidence point the same way. The only rung that can reach 0.95."""

CONFIDENCE_OVERRIDE_BASE = 0.55
"""Evidence beat a mapped SIC code by the required margin. Deliberately below
CONFIDENCE_SIC_ONLY at its floor: overturning the registrant's own filing code is
a claim that should not start out more confident than accepting it."""

CONFIDENCE_TEXT_ONLY_BASE = 0.50
"""SIC was absent or could not decide, and the evidence decided alone."""

CONFIDENCE_SIC_CONTESTED = 0.45
"""SIC maps, the evidence points elsewhere but under the margin, so the code keeps
the name and the contest is recorded."""

CONFIDENCE_AMBIGUOUS = 0.20
"""Two sub-verticals tied and nothing available broke the tie."""

CONFIDENCE_CURATED = 0.55
"""A curated seed prior stood against a classification drawn from the code alone.
The same rung as an evidence override, because it is the same kind of claim, made
by a person who can be asked to defend it."""

STRENGTH_BONUS = 0.15
"""Added on a sliding scale within a rung: zero when the winning margin is exactly
at the threshold, full at twice the threshold or better."""

DEFAULT_KEYWORD_MARGIN = 0.15
"""Share of total evidence the winner must hold over the runner-up before it may
overturn a mapped SIC code."""

DEFAULT_WEAK_SIC_MARGIN = 0.05
"""The margin demanded against a code in WEAK_SIC. Overturning "all software"
should not take the same weight of evidence as overturning "semiconductors"."""

DEFAULT_MIN_KEYWORD_HITS = 2
"""Distinct phrases the winner must match before evidence counts at all. One
stray word in a risk factor is not a business description."""

# The three knobs above are read from ``assumptions.tmt.sub_vertical`` under the
# names ``keyword_margin``, ``weak_sic_margin`` and ``min_keyword_hits`` when that
# block is present in the schema, and fall back to the documented default when it
# is not. The fallback is deliberate rather than lazy: the TMT assumption block is
# owned by config.py, this module has to stay usable against a bare Assumptions(),
# and a default written down beside the value it defends is not a hidden number.

# Forms that prove a filer was a reporting company. Form D, S-1 and the rest of
# the registration and exempt-offering paperwork do not, which is the whole point
# of testing on this tuple rather than on any filing at all.
PERIODIC_FORMS = ("10-K", "10-Q")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Company:
    """One resolved member of the universe, or one that could not be resolved.

    ``confidence`` is not a probability and does not come out of a fitted model.
    It is the rung of the ladder above that the classification landed on, which is
    why ``source`` always travels with it: the string is the audit trail and the
    number is only a sort key. A company that failed to resolve is kept, with
    ``sub_vertical`` None and the reason in ``notes``, because a candidate pool
    that quietly loses names is a candidate pool nobody can check.

    ``cik`` is the identifier that means something. The ticker is a label the
    exchange can reassign to an unrelated company, and in this universe already
    has.
    """

    ticker: str
    cik: int | None
    name: str
    sic: str | None
    sub_vertical: SubVertical | None
    confidence: float
    source: str
    notes: str = ""

    def row(self) -> dict[str, object]:
        """One flat record. Values stay raw; rendering and NM live in cli.py."""
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "name": self.name,
            "sic": self.sic,
            "sub_vertical": self.sub_vertical.value if self.sub_vertical else None,
            "confidence": self.confidence,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Evidence:
    """What the business text said, kept separable so a call can be argued with."""

    hits: dict[SubVertical, int] = field(default_factory=dict)
    matched: dict[SubVertical, tuple[str, ...]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.hits.values())

    def ranked(self) -> list[tuple[SubVertical, int]]:
        """Highest count first.

        The secondary sort is alphabetical purely so the output is deterministic
        under an equal count. It never decides an outcome, because an exact tie at
        the top is reported as a tie rather than resolved by it.
        """
        return sorted(self.hits.items(), key=lambda kv: (-kv[1], kv[0].value))

    def share(self, vertical: SubVertical | None) -> float:
        if vertical is None or not self.total:
            return 0.0
        return self.hits.get(vertical, 0) / self.total

    def tied_at_top(self) -> list[SubVertical]:
        """Every sub-vertical holding the top count, which is more than two when
        a short description matches three buckets once each."""
        ranked = self.ranked()
        if not ranked:
            return []
        best = ranked[0][1]
        return sorted((v for v, n in ranked if n == best), key=lambda v: v.value)


def score_business_text(text: str | None, name: str = "") -> Evidence:
    """Count distinct matched phrases per sub-vertical.

    Distinct phrases, not occurrences. A 10-K that says "wafer" ninety times is
    making one point about itself, and counting it ninety times would let a single
    word swamp a genuinely broader description. The registrant name is scanned in
    the same pass because it sometimes carries the cleanest signal in the
    document, and a name match is worth exactly one phrase like any other.
    """
    corpus = " ".join(part for part in (name, text) if part)
    if not corpus.strip():
        return Evidence()

    hits: dict[SubVertical, int] = {}
    matched: dict[SubVertical, tuple[str, ...]] = {}
    for vertical, patterns in _PATTERNS.items():
        found = tuple(phrase for phrase, rx in patterns if rx.search(corpus))
        if found:
            hits[vertical] = len(found)
            matched[vertical] = found
    return Evidence(hits=hits, matched=matched)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def _sub_vertical_config(assumptions):
    return getattr(getattr(assumptions, "tmt", None), "sub_vertical", None)


def _tuning(assumptions) -> tuple[float, float, int]:
    """Margins and the evidence floor, from ``assumptions.tmt.sub_vertical``.

    Read by name and defensively, for the reason set out beside the defaults. A
    knob that is present and out of range is a different matter and raises: a
    margin of 1.0 can never be met, which would pin every company to its SIC code
    while appearing to reconcile it against evidence.
    """
    cfg = _sub_vertical_config(assumptions)
    margin = float(getattr(cfg, "keyword_margin", DEFAULT_KEYWORD_MARGIN))
    weak_margin = float(getattr(cfg, "weak_sic_margin", DEFAULT_WEAK_SIC_MARGIN))
    min_hits = int(getattr(cfg, "min_keyword_hits", DEFAULT_MIN_KEYWORD_HITS))

    if not 0.0 <= margin < 1.0 or not 0.0 <= weak_margin < 1.0:
        raise ConfigError(
            f"tmt.sub_vertical keyword margins must sit in [0, 1); got {margin} and "
            f"{weak_margin}. A margin of 1.0 or more can never be met, which would "
            "pin every company to its SIC code without saying so."
        )
    if min_hits < 1:
        raise ConfigError(
            f"tmt.sub_vertical.min_keyword_hits must be at least 1; got {min_hits}. "
            "Zero would let a company with no matched phrase at all be classified "
            "from its business text."
        )
    return margin, weak_margin, min_hits


def _strength(margin: float, threshold: float) -> float:
    """Where inside a rung a win sits: 0 at the threshold, 1 at twice it."""
    if threshold <= 0.0:
        return 1.0
    return max(0.0, min(1.0, (margin - threshold) / threshold))


def normalize_sic(sic: str | int | None) -> str | None:
    """Four characters, zero padded.

    The submissions payload sends ``"7372"``, other SEC endpoints send ``7372`` as
    an integer, and codes below 1000 lose their leading zero in transit either
    way. Anything non-numeric is treated as no code rather than as a code that
    fails to match, because the two are different and only one of them is a data
    problem worth reporting.
    """
    if sic is None:
        return None
    raw = str(sic).strip()
    if not raw or not raw.isdigit():
        return None
    return raw.zfill(4)


def classify(
    sic: str | int | None,
    name: str,
    business_text: str | None,
    assumptions,
) -> tuple[SubVertical | None, float, str]:
    """Reconcile the SEC's code with the company's own description of itself.

    Returns the sub-vertical, a confidence on the ladder above, and a source
    string naming which evidence won and what it beat. The source string is as
    much the deliverable as the label: a peer set built on an override nobody can
    see is a peer set nobody can defend.

    The order of resolution:

    1. Score the business text. If the winner matches fewer than the evidence
       floor of distinct phrases, the text is treated as absent rather than as
       weak, because one phrase in a risk factor is noise.
    2. If the evidence ties at the top and the code is not one of the tied
       candidates, return None with every tied candidate named. Breaking that tie
       on dictionary order would manufacture a decision out of nothing.
    3. If the code maps and the evidence agrees, take it at the highest rung.
    4. If the code maps and the evidence disagrees, the evidence wins only if its
       margin over the runner-up clears the threshold, which is lower for the wide
       codes in WEAK_SIC than for the specific ones. Either way the loser is named
       in the source string.
    5. If the code is absent, unmapped, or one of the codes in SIC_NEEDS_EVIDENCE,
       the evidence decides alone at a lower rung. This is the path that admits
       American Tower from 6798 and Amazon from 5961.
    """
    margin_floor, weak_floor, min_hits = _tuning(assumptions)

    code = normalize_sic(sic)
    sic_pick = SIC_TO_SUB_VERTICAL.get(code) if code else None
    if code is None:
        code_label = "no SIC on file"
    elif code in SIC_NEEDS_EVIDENCE:
        code_label = f"SIC {code} ({SIC_NEEDS_EVIDENCE[code]})"
    else:
        code_label = f"SIC {code}"

    evidence = score_business_text(business_text, name)
    ranked = evidence.ranked()
    top, top_hits = ranked[0] if ranked else (None, 0)
    runner = ranked[1][0] if len(ranked) > 1 else None

    if top is None or top_hits < min_hits:
        why = (
            f"business text matched {top_hits} phrase(s), below the evidence floor "
            f"of {min_hits}"
            if top is not None
            else "no business text was read"
        )
        if sic_pick is not None:
            return sic_pick, CONFIDENCE_SIC_ONLY, f"{code_label} alone; {why}"
        return (
            None,
            0.0,
            f"unclassified: {code_label} maps to no TMT sub-vertical and {why}",
        )

    threshold = weak_floor if code in WEAK_SIC else margin_floor
    margin = evidence.share(top) - evidence.share(runner)
    matched_note = ", ".join(evidence.matched[top][:3])

    tied = evidence.tied_at_top()
    if len(tied) > 1:
        names = ", ".join(v.value for v in tied)
        if sic_pick in tied:
            return (
                sic_pick,
                CONFIDENCE_AGREE_BASE,
                f"{code_label} broke a tie in the business text between {names} at "
                f"{top_hits} matched phrases each",
            )
        detail = (
            f"ambiguous: business text tied between {names} at {top_hits} matched "
            f"phrases each, and {code_label} does not resolve it"
        )
        if sic_pick is not None:
            return sic_pick, CONFIDENCE_SIC_CONTESTED, detail
        return None, CONFIDENCE_AMBIGUOUS, detail

    if sic_pick is not None and sic_pick is top:
        conf = CONFIDENCE_AGREE_BASE + STRENGTH_BONUS * _strength(margin, threshold)
        return (
            top,
            round(conf, 4),
            f"{code_label} confirmed by business text ({top_hits} matched phrases: "
            f"{matched_note})",
        )

    if sic_pick is not None:
        if margin < threshold:
            return (
                sic_pick,
                CONFIDENCE_SIC_CONTESTED,
                f"{code_label} kept: business text leaned {top.value} ({top_hits} "
                f"phrases) but its margin of {margin:.0%} is under the "
                f"{threshold:.0%} needed to overturn the code",
            )
        conf = CONFIDENCE_OVERRIDE_BASE + STRENGTH_BONUS * _strength(margin, threshold)
        return (
            top,
            round(conf, 4),
            f"business text overrides {code_label}, which would have said "
            f"{sic_pick.value}: {top_hits} matched phrases ({matched_note}) at a "
            f"{margin:.0%} margin over {runner.value if runner else 'nothing else'}",
        )

    if margin < threshold:
        return (
            top,
            CONFIDENCE_AMBIGUOUS,
            f"{code_label}; business text leaned {top.value} ({top_hits} phrases) "
            f"but only {margin:.0%} clear of "
            f"{runner.value if runner else 'nothing else'}",
        )
    conf = CONFIDENCE_TEXT_ONLY_BASE + STRENGTH_BONUS * _strength(margin, threshold)
    return (
        top,
        round(conf, 4),
        f"{code_label}; classified from business text alone: {top_hits} matched "
        f"phrases ({matched_note}) at a {margin:.0%} margin",
    )


# --------------------------------------------------------------------------- #
# The seed universe
# --------------------------------------------------------------------------- #

# The candidate pool, with the covering analyst's prior for each name. The prior
# is not a shortcut around the classifier: classify() runs on every name and any
# disagreement is written into the Company record rather than hidden. The prior
# earns its keep in the case the code cannot handle and no business text was read
# to settle, which is most of the list: American Tower is not a shopping mall, and
# a universe that said it was would send the peer model looking for REIT comps.
#
# Breadth across all three letters is the product here. A TMT platform that can
# only see software is a software platform. So the pool spans semiconductors and
# the equipment that makes them, the hardware and networking supply chain, IT
# services, payments, consumer internet, both software buckets, media and
# broadcasting, the carriers, the tower and data centre REITs, and the game
# publishers.
#
# Five names are left in that did not resolve against the SEC ticker file on the
# fixture retrieval date, and they are the most honest thing in this module: EA,
# FI, JNPR, IPG and FYBR. Four of them left the file by the ordinary routes a
# large-cap leaves it, which the file itself does not record, so no cause is
# asserted for them here. The fifth is the instructive one and its cause is
# visible in the data: Fiserv is still listed and still fails, because the SEC
# file carries it under its pre-2023 symbol FISV. All five are kept and recorded
# as unresolved rather than deleted, so a reader can see the shape of what a
# today's ticker file cannot tell them about a past date.
SEED: dict[str, SubVertical] = {
    # Semiconductors and semiconductor equipment
    "NVDA": SubVertical.SEMICONDUCTORS,
    "AVGO": SubVertical.SEMICONDUCTORS,
    "AMD": SubVertical.SEMICONDUCTORS,
    "TXN": SubVertical.SEMICONDUCTORS,
    "MU": SubVertical.SEMICONDUCTORS,
    "INTC": SubVertical.SEMICONDUCTORS,
    "QCOM": SubVertical.SEMICONDUCTORS,
    "ADI": SubVertical.SEMICONDUCTORS,
    "MRVL": SubVertical.SEMICONDUCTORS,
    "MCHP": SubVertical.SEMICONDUCTORS,
    "ON": SubVertical.SEMICONDUCTORS,
    "NXPI": SubVertical.SEMICONDUCTORS,
    "AMAT": SubVertical.SEMICONDUCTORS,
    "LRCX": SubVertical.SEMICONDUCTORS,
    "KLAC": SubVertical.SEMICONDUCTORS,
    # Hardware, networking and storage
    "AAPL": SubVertical.HARDWARE,
    "DELL": SubVertical.HARDWARE,
    "HPQ": SubVertical.HARDWARE,
    "HPE": SubVertical.HARDWARE,
    "ANET": SubVertical.HARDWARE,
    "CSCO": SubVertical.HARDWARE,
    "NTAP": SubVertical.HARDWARE,
    "STX": SubVertical.HARDWARE,
    "WDC": SubVertical.HARDWARE,
    "SMCI": SubVertical.HARDWARE,
    "MSI": SubVertical.HARDWARE,
    "ZBRA": SubVertical.HARDWARE,
    "JNPR": SubVertical.HARDWARE,
    # IT services
    "ACN": SubVertical.IT_SERVICES,
    "IBM": SubVertical.IT_SERVICES,
    "CTSH": SubVertical.IT_SERVICES,
    "EPAM": SubVertical.IT_SERVICES,
    "DXC": SubVertical.IT_SERVICES,
    "G": SubVertical.IT_SERVICES,
    # Payments and processing
    "V": SubVertical.PAYMENTS,
    "MA": SubVertical.PAYMENTS,
    "PYPL": SubVertical.PAYMENTS,
    "FI": SubVertical.PAYMENTS,
    "FIS": SubVertical.PAYMENTS,
    "GPN": SubVertical.PAYMENTS,
    "TOST": SubVertical.PAYMENTS,
    "JKHY": SubVertical.PAYMENTS,
    # Consumer internet and marketplaces
    "GOOGL": SubVertical.INTERNET,
    "META": SubVertical.INTERNET,
    "AMZN": SubVertical.INTERNET,
    "UBER": SubVertical.INTERNET,
    "ABNB": SubVertical.INTERNET,
    "DASH": SubVertical.INTERNET,
    "BKNG": SubVertical.INTERNET,
    "EBAY": SubVertical.INTERNET,
    "EXPE": SubVertical.INTERNET,
    "PINS": SubVertical.INTERNET,
    "SNAP": SubVertical.INTERNET,
    "TTD": SubVertical.INTERNET,
    "APP": SubVertical.INTERNET,
    # Infrastructure software: data platforms, cloud infrastructure, security
    "MSFT": SubVertical.INFRASTRUCTURE_SOFTWARE,
    "ORCL": SubVertical.INFRASTRUCTURE_SOFTWARE,
    "SNOW": SubVertical.INFRASTRUCTURE_SOFTWARE,
    "DDOG": SubVertical.INFRASTRUCTURE_SOFTWARE,
    "MDB": SubVertical.INFRASTRUCTURE_SOFTWARE,
    "NET": SubVertical.INFRASTRUCTURE_SOFTWARE,
    "CRWD": SubVertical.INFRASTRUCTURE_SOFTWARE,
    "PANW": SubVertical.INFRASTRUCTURE_SOFTWARE,
    "ZS": SubVertical.INFRASTRUCTURE_SOFTWARE,
    "FTNT": SubVertical.INFRASTRUCTURE_SOFTWARE,
    "OKTA": SubVertical.INFRASTRUCTURE_SOFTWARE,
    # Application software
    "CRM": SubVertical.APPLICATION_SOFTWARE,
    "ADBE": SubVertical.APPLICATION_SOFTWARE,
    "INTU": SubVertical.APPLICATION_SOFTWARE,
    "NOW": SubVertical.APPLICATION_SOFTWARE,
    "WDAY": SubVertical.APPLICATION_SOFTWARE,
    "HUBS": SubVertical.APPLICATION_SOFTWARE,
    "TEAM": SubVertical.APPLICATION_SOFTWARE,
    "VEEV": SubVertical.APPLICATION_SOFTWARE,
    "DOCU": SubVertical.APPLICATION_SOFTWARE,
    "ADSK": SubVertical.APPLICATION_SOFTWARE,
    "CDNS": SubVertical.APPLICATION_SOFTWARE,
    "SNPS": SubVertical.APPLICATION_SOFTWARE,
    "PLTR": SubVertical.APPLICATION_SOFTWARE,
    "ZM": SubVertical.APPLICATION_SOFTWARE,
    "SHOP": SubVertical.APPLICATION_SOFTWARE,
    # Media and entertainment
    "DIS": SubVertical.MEDIA_ENTERTAINMENT,
    "NFLX": SubVertical.MEDIA_ENTERTAINMENT,
    "WBD": SubVertical.MEDIA_ENTERTAINMENT,
    "PSKY": SubVertical.MEDIA_ENTERTAINMENT,
    "FOX": SubVertical.MEDIA_ENTERTAINMENT,
    "NXST": SubVertical.MEDIA_ENTERTAINMENT,
    "LYV": SubVertical.MEDIA_ENTERTAINMENT,
    "NYT": SubVertical.MEDIA_ENTERTAINMENT,
    "OMC": SubVertical.MEDIA_ENTERTAINMENT,
    "IPG": SubVertical.MEDIA_ENTERTAINMENT,
    "WMG": SubVertical.MEDIA_ENTERTAINMENT,
    "ROKU": SubVertical.MEDIA_ENTERTAINMENT,
    # Telecom and cable
    "T": SubVertical.TELECOM,
    "VZ": SubVertical.TELECOM,
    "TMUS": SubVertical.TELECOM,
    "LUMN": SubVertical.TELECOM,
    "FYBR": SubVertical.TELECOM,
    "CMCSA": SubVertical.TELECOM,
    "CHTR": SubVertical.TELECOM,
    # Towers, fibre and data centres
    "AMT": SubVertical.TOWERS_FIBER,
    "CCI": SubVertical.TOWERS_FIBER,
    "SBAC": SubVertical.TOWERS_FIBER,
    "EQIX": SubVertical.TOWERS_FIBER,
    "DLR": SubVertical.TOWERS_FIBER,
    "UNIT": SubVertical.TOWERS_FIBER,
    # Gaming
    "EA": SubVertical.GAMING,
    "TTWO": SubVertical.GAMING,
    "RBLX": SubVertical.GAMING,
    "PLTK": SubVertical.GAMING,
}


# --------------------------------------------------------------------------- #
# Building the universe
# --------------------------------------------------------------------------- #


def _knowledge_date(client, assumptions, as_of: date | None) -> date | None:
    """Explicit argument, then the client's own knowledge date, then assumptions.

    The client outranks the assumptions file because a client built with a
    knowledge date is already refusing to hand back facts filed after it. A
    universe that reached past that date would hold companies whose fundamentals
    the rest of the run cannot see.
    """
    if as_of is not None:
        return as_of
    from_client = getattr(client, "knowledge_date", None)
    if from_client is not None:
        return from_client
    raw = getattr(assumptions, "as_of", None)
    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError as exc:
            raise ConfigError(f"as_of must be an ISO date, got {raw!r}: {exc}") from exc
    return None


def _admission_date(payload: dict) -> tuple[date | None, str | None]:
    """When this filer became a reporting company, and how sure we are of it.

    Returns the date and, when the date is weaker than it looks, a caveat to be
    recorded on the Company.

    The first trap is the form type. A CIK exists from the first piece of paper
    filed under it, and for a venture-backed company that is a Form D exempt
    offering notice filed years before the listing. Roblox's CIK carries
    submissions from 2005 against a 2021 IPO. So the admission date is the
    earliest 10-K or 10-Q, which only a reporting company files.

    The second trap is the window. ``filings.recent`` holds roughly the last
    thousand submissions, which for a prolific filer such as IBM or AT&T is a few
    years, and the rest are pushed into shards under ``filings.files`` that carry
    date boundaries but no form types. Taking the earliest periodic report visible
    in the window would date Apple to 2015 and drop it out of any earlier
    universe. So where shards exist the earliest shard boundary is used instead,
    and the caveat says what it is: the start of the EDGAR record, which for a
    company that spent years private before listing is earlier than its first
    report and cannot be separated from it without fetching every shard.
    """
    filings = payload.get("filings") or {}

    shard_starts: list[date] = []
    for shard in filings.get("files") or []:
        parsed = _as_date(shard.get("filingFrom"))
        if parsed is not None:
            shard_starts.append(parsed)
    if shard_starts:
        return min(shard_starts), (
            "admission dated from the start of the EDGAR record, because older "
            "submissions sit in shards that carry no form types; for a company "
            "that was private for years before listing this is earlier than its "
            "first periodic report"
        )

    recent = filings.get("recent") or {}
    periodic = [
        parsed
        for stamp, form in zip(
            recent.get("filingDate") or [], recent.get("form") or []
        )
        if str(form).upper().startswith(PERIODIC_FORMS)
        and (parsed := _as_date(stamp)) is not None
    ]
    if periodic:
        return min(periodic), None
    return None, None


def _as_date(stamp) -> date | None:
    if not stamp:
        return None
    try:
        return date.fromisoformat(str(stamp)[:10])
    except ValueError:
        return None


def _market_cap(ticker: str, client, assumptions, market) -> float:
    """Equity value in USD millions, on the engine's own share count convention.

    Goes through build_ev_bridge rather than multiplying a price by a raw share
    count, so the screen uses the same denominator as every other number in the
    run. Under the treasury stock method that is a point-in-time diluted count and
    under the weighted-average method a trailing one, and a screen that disagreed
    with the valuation about how many shares exist would cut names the valuation
    kept.
    """
    fin = build_financials(ticker, facts=client.company_facts(ticker))
    return build_ev_bridge(fin, market.spot(ticker), assumptions).equity_value


def tmt_universe(
    client,
    assumptions,
    as_of: date | None = None,
    min_market_cap: float | None = None,
    market=None,
    business_text: Callable[[str], str | None] | None = None,
) -> list[Company]:
    """Resolve the seed universe against EDGAR, point in time, losing nothing quietly.

    Each candidate is resolved through ``client.submissions`` for its CIK,
    registered name and SIC code, classified, and reconciled against the curated
    prior in SEED. Three things can happen to a candidate and all three are
    visible in the result:

    *It resolves.* The Company carries the sub-vertical, the confidence rung and a
    source string naming the winning evidence.

    *It does not resolve.* A ticker the SEC file does not know, or a submissions
    request that failed, is returned anyway with ``sub_vertical`` None, confidence
    zero and the error text in ``notes``. Dropping it would make a shrinking
    universe look like a stable one.

    *It was not yet a reporting company.* Where a knowledge date is in force and
    the filer's admission date postdates it, the name is excluded outright. This
    is the only case in which a candidate leaves the list on its own merits, and
    it is the point of the exercise: a 2019 universe holding a 2021 listing is not
    a universe, it is hindsight.

    ``business_text`` is an optional reader called once per ticker. Pass one when
    the caller has the Item 1 text and wants the evidence layer to run, which is
    what turns a SIC-only classification into a reconciled one. It is not called
    by default, because reading a 10-K for every candidate costs a request and a
    hundred candidates is a hundred requests for a screen that mostly does not
    need them. A reader that raises is not fatal: the reason is recorded and the
    name is classified on its code alone.

    ``min_market_cap`` is in USD millions and needs a ``market``. A name whose
    market capitalisation cannot be computed is kept rather than cut, with the
    reason in ``notes``, because the screen failing is not evidence that the
    company is small.

    The survivorship limitation in the module docstring applies to everything
    returned here.
    """
    if min_market_cap is not None and market is None:
        raise ConfigError(
            "min_market_cap needs a MarketData source to price the screen against. "
            "Pass market=, or drop the floor."
        )

    knowledge = _knowledge_date(client, assumptions, as_of)
    cfg = _sub_vertical_config(assumptions)
    overrides: dict[str, SubVertical] = {}
    for raw_ticker, raw_vertical in (getattr(cfg, "overrides", None) or {}).items():
        try:
            overrides[str(raw_ticker).upper()] = SubVertical(raw_vertical)
        except ValueError as exc:
            raise ConfigError(
                f"tmt.sub_vertical.overrides[{raw_ticker!r}] is {raw_vertical!r}, "
                "which is not a sub-vertical. Valid values: "
                + ", ".join(v.value for v in SubVertical)
            ) from exc
    extra = [str(t).upper() for t in (getattr(cfg, "extra_tickers", None) or [])]

    universe: list[Company] = []
    for ticker in sorted(set(SEED) | set(extra) | set(overrides)):
        try:
            payload = client.submissions(ticker)
        except (MissingDataError, DataSourceError) as exc:
            universe.append(
                Company(
                    ticker=ticker,
                    cik=None,
                    name=ticker,
                    sic=None,
                    sub_vertical=None,
                    confidence=0.0,
                    source="unresolved against EDGAR",
                    notes=str(exc),
                )
            )
            continue

        notes: list[str] = []
        admitted, caveat = _admission_date(payload)
        if knowledge is not None:
            if admitted is None:
                notes.append(
                    "the submissions payload carries no periodic report, so the "
                    f"point-in-time test against {knowledge} could not be applied"
                )
            elif admitted > knowledge:
                continue
            elif caveat:
                notes.append(caveat)

        code = normalize_sic(payload.get("sic"))
        description = (payload.get("sicDescription") or "").strip()
        name = (payload.get("name") or ticker).strip()
        raw_cik = payload.get("cik")
        try:
            cik = int(raw_cik) if raw_cik is not None else None
        except (TypeError, ValueError):
            cik = None

        text = None
        if business_text is not None:
            try:
                text = business_text(ticker)
            except (MissingDataError, DataSourceError) as exc:
                notes.append(f"business text unavailable, classified on code alone: {exc}")

        derived, confidence, source = classify(code, name, text, assumptions)
        if description:
            source = f"{source} [{description}]"

        prior = overrides.get(ticker, SEED.get(ticker))
        sub_vertical = derived
        if prior is not None and derived is None:
            sub_vertical = prior
            confidence = CONFIDENCE_CURATED
            source = f"curated prior {prior.value}; {source}"
        elif prior is not None and derived is not prior:
            # The prior is a named person's call and the code is a box the
            # registrant ticked once. The prior takes the name, at the override
            # rung rather than at its own, and the string keeps what it beat.
            sub_vertical = prior
            confidence = CONFIDENCE_CURATED
            source = f"curated prior {prior.value} overrides {source}"
        elif prior is not None:
            notes.append("curated prior agrees with the filing")

        universe.append(
            Company(
                ticker=ticker,
                cik=cik,
                name=name,
                sic=code,
                sub_vertical=sub_vertical,
                confidence=confidence,
                source=source,
                notes="; ".join(notes),
            )
        )

    if min_market_cap is None:
        return universe

    screened: list[Company] = []
    for company in universe:
        if company.sub_vertical is None:
            screened.append(company)
            continue
        try:
            cap = _market_cap(company.ticker, client, assumptions, market)
        except (MissingDataError, DataSourceError) as exc:
            note = (
                f"kept without the {min_market_cap:,.0f}mm market cap screen: equity "
                f"value could not be computed ({exc})"
            )
            screened.append(
                Company(
                    ticker=company.ticker,
                    cik=company.cik,
                    name=company.name,
                    sic=company.sic,
                    sub_vertical=company.sub_vertical,
                    confidence=company.confidence,
                    source=company.source,
                    notes="; ".join(n for n in (company.notes, note) if n),
                )
            )
            continue
        if cap >= min_market_cap:
            screened.append(company)
    return screened
