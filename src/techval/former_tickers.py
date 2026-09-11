"""Tickers that no longer trade, and the CIK each one belonged to.

``https://www.sec.gov/files/company_tickers.json`` is a list of companies that
exist today. It carried 10,407 symbols when this table was recorded and it
carried none of SPLK, ZEN, MNDT, WORK or TWTR, because every completed
acquisition target is out of it by construction: the security is delisted, the
registrant stops filing, and the symbol row goes with it. ``EdgarClient``
resolved every ticker through that file and nothing else, so the one command
whose entire subject is completed transactions, ``techval precedents``, could
not reach a single completed transaction. Its own help text offered
``SPLK,ZEN,MNDT,WORK`` as the example and all four failed, with a hint that
blamed foreign filers for what was a delisting.

The submissions API is keyed on CIK, and a delisted registrant keeps its whole
filing history under that CIK forever. So nothing about the filings is missing.
What is missing is one mapping, and the mapping is recoverable from the filings
themselves: since 2019 every registrant tags its own cover page in inline XBRL,
and ``dei:TradingSymbol`` there is the company stating its own ticker in a
document it signed. That is a better authority than any third-party symbol list,
and it is where every row below comes from.

**A ticker is not a durable identifier, so each row carries the dates it was
proven between.** ``S`` was Sprint's until it stopped filing in January 2020 and
is SentinelOne's today; both tag that one letter on their own cover pages, five
years apart. ``ALTR`` was Altera's until 2015 and Altair Engineering's until
2025. ``AZPN`` was CIK 929940's until April 2022 and CIK 1897982's afterwards,
two registrants of the same name. A table that answered "ALTR" with one number
would be wrong for half the history of the symbol, so a symbol maps to the list
of registrants proven to have traded under it, oldest first, and the resolution
is made against a knowledge date.

``through`` is the filing date of the document the symbol was read from. It
asserts that the symbol belonged to that registrant on that day and nothing
about any other day, which is the strongest claim the evidence supports.
``since`` is the registrant's oldest periodic report on file, and it is a bound
rather than a sighting: a company cannot have traded under a symbol before it
existed. Without it this table would answer a 2014 question about ALTR with
Altair Engineering, which did not list until 2017. Where EDGAR's recent-filing
index truncates, the bound lands later than the truth and the resolution refuses
a date in front of it, which costs an answer rather than giving a wrong one.

**What this deliberately is not.** It is not every delisted symbol on EDGAR, and
it is not a point-in-time ticker file, which is a commercial product this
engine does not have. It is the departed registrants of the TMT universe in
``tests/fixtures/mna/universe.json``, which is the population this engine's M&A
and peer work asks about: 100 symbols over 97 registrants, out of 105 seeds.

Eight seeds resolve to nothing and the reasons are worth knowing, because they
are the shape of what this cannot do. Seven filed their last report between
February 2019 and August 2020, while inline cover-page tagging was still phasing
in by filer size, so their cover pages are plain HTML with nothing to read:
Twenty-First Century Fox, Electronics for Imaging, Ultimate Software, L3,
Finisar, Windstream and McClatchy. The eighth, Audacy, was still filing in 2024
and tags no trading symbol at all, because by then it had no listed security to
name. Anything that left before 2019, EMC and Altera among them, is outside the
table for the same reason as the seven. For those, and for any symbol outside
TMT, ``EdgarClient.ticker_to_cik`` takes the CIK directly as ``CIK0000790070``,
which is also how the recorded M&A fixtures key a departed registrant.

The table is generated from ``tests/fixtures/former_tickers/index.json``, which
carries the accession and document each symbol was read out of, by
``tests/fixtures/former_tickers/record.py``. The two are pinned together by a
test, so a row here cannot drift away from the filing that proves it.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple


class FormerListing(NamedTuple):
    """One registrant proven to have traded under one symbol, on one date.

    ``through`` is evidence rather than an expiry: it is the filing date of the
    document whose cover page carried the symbol, which is usually the last
    periodic report before the registrant left. Nothing here says the symbol was
    the registrant's for the whole of its life, and nothing here says it was
    free afterwards. ``since`` is the registrant's oldest periodic report, which
    bounds the claim below rather than dating the listing.
    """

    cik: int
    since: date
    through: date
    name: str


def resolve(ticker: str, knowledge_date: date | None = None) -> FormerListing | None:
    """The registrant that traded under ``ticker`` as at ``knowledge_date``.

    With no knowledge date the most recent claimant is returned, which is the
    answer a live run wants: the symbol's last owner before it went quiet. With
    one, claimants that did not yet exist at that date are dropped, and the
    earliest of what is left whose evidence is dated on or after it wins, which
    is the answer a point-in-time run wants: a 2019 question about ``S`` is a
    question about Sprint. A knowledge date after every claim still returns the
    most recent one, because ``through`` is the last proof and not an expiry.
    A knowledge date in front of every claim returns nothing, because a table
    built from later filings is not evidence about an earlier symbol.
    """
    claims = FORMER_TICKERS.get(str(ticker).upper())
    if not claims:
        return None
    if knowledge_date is None:
        return claims[-1]
    live = [claim for claim in claims if claim.since <= knowledge_date]
    if not live:
        return None
    for claim in live:
        if claim.through >= knowledge_date:
            return claim
    return live[-1]


#: Symbol to the registrants proven to have traded under it, oldest proof first.
FORMER_TICKERS: dict[str, tuple[FormerListing, ...]] = {
    "ACIA": (
        FormerListing(1651235, date(2016, 8, 11), date(2021, 3, 1), "Acacia Communications, Inc."),
    ),
    "ALTR": (
        FormerListing(1701732, date(2020, 5, 7), date(2025, 2, 20), "Altair Engineering Inc."),
    ),
    "ANSS": (
        FormerListing(1013462, date(2014, 2, 27), date(2025, 4, 30), "ANSYS INC"),
    ),
    "ATCX": (
        FormerListing(1751143, date(2019, 3, 26), date(2023, 3, 15), "ATLAS TECHNICAL CONSULTANTS, INC."),
    ),
    "ATVI": (
        FormerListing(718877, date(2012, 5, 9), date(2023, 7, 31), "Activision Blizzard, Inc."),
    ),
    "AVLR": (
        FormerListing(1348036, date(2018, 8, 10), date(2022, 8, 9), "AVALARA, INC."),
    ),
    "AVX": (
        FormerListing(859163, date(2015, 2, 5), date(2020, 2, 5), "AVX Corp"),
    ),
    "AYX": (
        FormerListing(1689923, date(2017, 5, 11), date(2024, 2, 6), "Alteryx, Inc."),
    ),
    "AZPN": (
        FormerListing(929940, date(2010, 9, 2), date(2022, 4, 27), "ASPENTECH Corp"),
        FormerListing(1897982, date(2022, 5, 16), date(2025, 2, 4), "Aspen Technology, Inc."),
    ),
    "BKI": (
        FormerListing(1627014, date(2015, 8, 4), date(2023, 8, 3), "Black Knight, Inc."),
    ),
    "CCMP": (
        FormerListing(1102934, date(2013, 2, 8), date(2022, 5, 5), "CMC Materials, Inc."),
    ),
    "CERN": (
        FormerListing(804753, date(2011, 4, 29), date(2022, 5, 3), "CERNER Corp"),
    ),
    "CETV": (
        FormerListing(925645, date(2005, 5, 10), date(2020, 7, 21), "CENTRAL EUROPEAN MEDIA ENTERPRISES LTD"),
    ),
    "CLDR": (
        FormerListing(1535379, date(2017, 6, 9), date(2021, 9, 2), "Cloudera, Inc."),
    ),
    "CNSL": (
        FormerListing(1304421, date(2007, 8, 9), date(2024, 11, 5), "Consolidated Communications Holdings, Inc."),
    ),
    "COUP": (
        FormerListing(1385867, date(2016, 12, 9), date(2022, 12, 12), "Coupa Software Inc"),
    ),
    "CSOD": (
        FormerListing(1401680, date(2011, 5, 13), date(2021, 8, 6), "Cornerstone OnDemand Inc"),
    ),
    "CTXS": (
        FormerListing(877890, date(2016, 11, 4), date(2022, 7, 26), "CITRIX SYSTEMS INC"),
    ),
    "CVT": (
        FormerListing(1827075, date(2021, 3, 31), date(2023, 5, 5), "CVENT HOLDING CORP."),
    ),
    "CY": (
        FormerListing(791915, date(2011, 2, 25), date(2020, 2, 21), "CYPRESS SEMICONDUCTOR CORP /DE/"),
    ),
    "CYXT": (
        FormerListing(1794905, date(2020, 11, 16), date(2023, 5, 5), "Cyxtera Technologies, Inc."),
    ),
    "DATA": (
        FormerListing(1303652, date(2013, 8, 9), date(2019, 7, 31), "Tableau Software Inc"),
    ),
    "DKNG": (
        FormerListing(1772757, date(2019, 8, 12), date(2022, 2, 18), "DraftKings Holdings Inc."),
    ),
    "EBIX": (
        FormerListing(814549, date(2002, 11, 13), date(2023, 11, 14), "EBIX INC"),
    ),
    "EDR": (
        FormerListing(1766363, date(2021, 6, 2), date(2025, 2, 27), "Endeavor Group Holdings, Inc."),
    ),
    "EIGI": (
        FormerListing(1237746, date(2013, 12, 6), date(2020, 11, 9), "Endurance International Group Holdings, Inc."),
    ),
    "ENT": (
        FormerListing(1512077, date(2011, 6, 27), date(2020, 8, 14), "Global Eagle Entertainment Inc."),
    ),
    "EVRI": (
        FormerListing(1318568, date(2015, 3, 16), date(2025, 5, 12), "Everi Holdings Inc."),
    ),
    "FIT": (
        FormerListing(1447599, date(2015, 8, 7), date(2020, 11, 4), "FITBIT, INC."),
    ),
    "FUN": (
        FormerListing(811532, date(2011, 11, 4), date(2024, 5, 9), "CEDAR FAIR L P"),
    ),
    "GCI": (
        FormerListing(1635718, date(2015, 8, 6), date(2019, 11, 5), "Gannett Media Corp."),
    ),
    "GLIBA": (
        FormerListing(808461, date(2011, 8, 9), date(2020, 11, 5), "GRIZZLY MERGER SUB 1, LLC"),
    ),
    "GLIBP": (
        FormerListing(808461, date(2011, 8, 9), date(2020, 11, 5), "GRIZZLY MERGER SUB 1, LLC"),
    ),
    "GLUU": (
        FormerListing(1366246, date(2008, 11, 14), date(2021, 2, 26), "GLU MOBILE INC"),
    ),
    "GTT": (
        FormerListing(1315255, date(2012, 3, 27), date(2020, 5, 8), "GTT Communications, Inc."),
    ),
    "HCP": (
        FormerListing(1720671, date(2022, 3, 25), date(2024, 12, 5), "HashiCorp, Inc."),
    ),
    "INFN": (
        FormerListing(1138639, date(2014, 8, 1), date(2025, 2, 28), "Infinera Corp"),
    ),
    "INFO": (
        FormerListing(1598014, date(2015, 3, 10), date(2022, 1, 24), "IHS Markit Ltd."),
    ),
    "INST": (
        FormerListing(1841804, date(2021, 8, 18), date(2024, 11, 8), "INSTRUCTURE HOLDINGS, INC."),
    ),
    "IPHI": (
        FormerListing(1160958, date(2011, 3, 7), date(2021, 2, 25), "INPHI Corp"),
    ),
    "JNPR": (
        FormerListing(1043604, date(2015, 5, 8), date(2025, 5, 9), "JUNIPER NETWORKS INC"),
    ),
    "KEM": (
        FormerListing(887730, date(2009, 8, 10), date(2020, 5, 28), "KEMET CORP"),
    ),
    "KEYW": (
        FormerListing(1487101, date(2010, 11, 10), date(2019, 5, 7), "KEYW HOLDING CORP"),
    ),
    "LOGM": (
        FormerListing(1420302, date(2010, 10, 28), date(2020, 7, 29), "LogMeIn, Inc."),
    ),
    "MAXR": (
        FormerListing(1121142, date(2018, 3, 29), date(2023, 5, 3), "Maxar Technologies Inc."),
    ),
    "MCFE": (
        FormerListing(1783317, date(2020, 11, 19), date(2022, 2, 23), "McAfee Corp."),
    ),
    "MCRN": (
        FormerListing(1637913, date(2015, 8, 11), date(2019, 11, 12), "Milacron Holdings Corp."),
    ),
    "MDP": (
        FormerListing(65011, date(2010, 4, 28), date(2021, 11, 12), "Hawkeye Acquisition, Inc."),
    ),
    "MDSO": (
        FormerListing(1453814, date(2009, 8, 14), date(2019, 8, 6), "Medidata Solutions, Inc."),
    ),
    "MNDT": (
        FormerListing(1370880, date(2013, 11, 14), date(2022, 8, 4), "Mandiant, Inc."),
    ),
    "MSGN": (
        FormerListing(1469372, date(2010, 3, 18), date(2021, 5, 7), "MSG NETWORKS INC."),
    ),
    "MSP": (
        FormerListing(1724570, date(2020, 11, 23), date(2022, 5, 9), "DATTO HOLDING CORP."),
    ),
    "MTCH": (
        FormerListing(1575189, date(2016, 3, 28), date(2020, 5, 8), "Match Group Holdings II, LLC"),
    ),
    "MXIM": (
        FormerListing(743316, date(2014, 8, 18), date(2021, 8, 20), "MAXIM INTEGRATED PRODUCTS INC"),
    ),
    "NATI": (
        FormerListing(935494, date(2013, 5, 2), date(2023, 7, 28), "NATIONAL INSTRUMENTS CORP"),
    ),
    "NCI": (
        FormerListing(1019737, date(1999, 8, 16), date(2019, 8, 2), "NAVIGANT CONSULTING INC"),
    ),
    "NEWR": (
        FormerListing(1448056, date(2016, 5, 26), date(2023, 10, 27), "NEW RELIC, INC."),
    ),
    "NUAN": (
        FormerListing(1002517, date(2013, 8, 8), date(2022, 2, 7), "Nuance Communications, Inc."),
    ),
    "NXGN": (
        FormerListing(708818, date(2008, 11, 5), date(2023, 10, 24), "NEXTGEN HEALTHCARE, INC."),
    ),
    "PARA": (
        FormerListing(813828, date(2018, 8, 2), date(2025, 7, 31), "Paramount Global"),
    ),
    "PARAA": (
        FormerListing(813828, date(2018, 8, 2), date(2025, 7, 31), "Paramount Global"),
    ),
    "PLAN": (
        FormerListing(1540755, date(2018, 12, 10), date(2022, 6, 2), "Anaplan, Inc."),
    ),
    "POLY": (
        FormerListing(914025, date(2012, 5, 25), date(2022, 8, 11), "PLANTRONICS INC /CA/"),
    ),
    "PRFT": (
        FormerListing(1085869, date(2007, 3, 5), date(2024, 8, 8), "PERFICIENT INC"),
    ),
    "PVTL": (
        FormerListing(1574135, date(2018, 6, 14), date(2019, 12, 6), "Pivotal Software, Inc."),
    ),
    "PWSC": (
        FormerListing(1835681, date(2021, 9, 9), date(2024, 8, 9), "POWERSCHOOL HOLDINGS, INC."),
    ),
    "PYCR": (
        FormerListing(1839439, date(2021, 9, 2), date(2025, 2, 6), "PAYCOR HCM, INC."),
    ),
    "RBT": (
        FormerListing(1862068, date(2021, 11, 29), date(2024, 11, 22), "Rubicon Technologies, Inc."),
    ),
    "RDBX": (
        FormerListing(1820201, date(2021, 1, 8), date(2022, 8, 12), "Redbox Entertainment Inc."),
    ),
    "RDBXW": (
        FormerListing(1820201, date(2021, 1, 8), date(2022, 8, 12), "Redbox Entertainment Inc."),
    ),
    "RHT": (
        FormerListing(1087423, date(2011, 4, 29), date(2019, 4, 24), "RED HAT INC"),
    ),
    "RP": (
        FormerListing(1286225, date(2013, 8, 6), date(2021, 3, 1), "RealPage, Inc."),
    ),
    "S": (
        FormerListing(101830, date(2013, 8, 5), date(2020, 1, 27), "SPRINT LLC"),
    ),
    "SCWX": (
        FormerListing(1468666, date(2016, 6, 1), date(2024, 12, 4), "SecureWorks Corp"),
    ),
    "SIX": (
        FormerListing(701374, date(2011, 8, 8), date(2024, 5, 9), "Six Flags Entertainment Corp/OLD"),
    ),
    "SMAR": (
        FormerListing(1366561, date(2019, 4, 1), date(2024, 12, 5), "SMARTSHEET INC"),
    ),
    "SNPO": (
        FormerListing(1856430, date(2021, 8, 27), date(2024, 5, 8), "Snap One Holdings Corp."),
    ),
    "SPLK": (
        FormerListing(1353283, date(2014, 12, 10), date(2023, 11, 28), "SPLUNK INC"),
    ),
    "SPWR": (
        FormerListing(867773, date(2013, 8, 2), date(2023, 12, 18), "SUNPOWER CORP"),
    ),
    "SQSP": (
        FormerListing(1496963, date(2021, 8, 9), date(2024, 8, 2), "Squarespace, Inc."),
    ),
    "SWCH": (
        FormerListing(1710583, date(2017, 11, 14), date(2022, 11, 9), "Switch, Inc."),
    ),
    "SWI": (
        FormerListing(1739942, date(2018, 11, 27), date(2025, 2, 19), "SolarWinds Corp"),
    ),
    "SYKE": (
        FormerListing(1010612, date(2013, 11, 5), date(2021, 8, 9), "SYKES ENTERPRISES INC"),
    ),
    "TPCO": (
        FormerListing(1593195, date(2014, 8, 21), date(2021, 5, 6), "Tribune Publishing Co"),
    ),
    "TRCO": (
        FormerListing(726513, date(2004, 7, 30), date(2019, 8, 9), "TRIBUNE MEDIA CO"),
    ),
    "TWKS": (
        FormerListing(1866550, date(2021, 11, 15), date(2024, 11, 12), "Thoughtworks Holding, Inc."),
    ),
    "TWOU": (
        FormerListing(1459417, date(2014, 5, 12), date(2024, 8, 9), "2U, LLC"),
    ),
    "TWTR": (
        FormerListing(1418091, date(2015, 11, 6), date(2022, 7, 26), "TWITTER, INC."),
    ),
    "VG": (
        FormerListing(1272830, date(2014, 2, 13), date(2022, 5, 5), "VONAGE HOLDINGS CORP"),
    ),
    "VIA": (
        FormerListing(1339947, date(2012, 2, 2), date(2019, 11, 14), "Viacom Inc."),
    ),
    "VIAB": (
        FormerListing(1339947, date(2012, 2, 2), date(2019, 11, 14), "Viacom Inc."),
    ),
    "VMW": (
        FormerListing(1124610, date(2015, 8, 5), date(2023, 9, 7), "VMWARE LLC"),
    ),
    "VRTU": (
        FormerListing(1207074, date(2011, 8, 3), date(2021, 2, 9), "VIRTUSA CORP"),
    ),
    "VZIO": (
        FormerListing(1835591, date(2021, 5, 12), date(2024, 11, 6), "Vizio Holding Corp."),
    ),
    "WORK": (
        FormerListing(1764925, date(2019, 9, 5), date(2021, 6, 3), "Slack Technologies, Inc."),
    ),
    "WWE": (
        FormerListing(1091907, date(2013, 8, 2), date(2023, 8, 2), "WORLD WRESTLING ENTERTAINMENT, LLC"),
    ),
    "XLNX": (
        FormerListing(743988, date(2011, 11, 8), date(2022, 1, 27), "XILINX INC"),
    ),
    "XM": (
        FormerListing(1747748, date(2021, 3, 9), date(2023, 5, 2), "Qualtrics International Inc."),
    ),
    "ZAYO": (
        FormerListing(1608249, date(2014, 11, 10), date(2020, 2, 4), "Zayo Group Holdings, Inc."),
    ),
    "ZEN": (
        FormerListing(1463172, date(2019, 2, 14), date(2022, 10, 28), "Zendesk, Inc."),
    ),
}
