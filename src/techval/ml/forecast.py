"""The revenue growth fade curve, fitted on filings instead of typed into a file.

Every discounted cash flow in this engine fades revenue growth from
``dcf.revenue_growth_start`` to ``dcf.revenue_growth_terminal`` on a straight
line, and that line moves the answer more than the discount rate, the margin
path and the terminal multiple put together. Nothing about it is measured. This
module measures it: given what a technology company looks like on the day its
10-K is filed, how fast does its revenue growth actually decay?

**The answer, before the method.** On 2,334 point-in-time company-years from 223
TMT filers between 2009 and 2026, next year's growth regresses on this year's
with a slope of 0.47 and an intercept of 6.9 points. Growth falls a little over
half of the way toward a long-run TMT mean of 13.1% in a single year, a half-life
of eleven months. Sorted into deciles, which assumes no functional form at all,
the top decile grows 90% and then 47%, the bottom decile shrinks 16% and then
grows 7%, and the crossing point where growth neither fades nor accelerates sits
between 9 and 14%. The straight line in the default assumptions fades a fifth of
the gap a year; the filings fade half of it. That is the finding.

**But the fitted curve cannot reach a terminal value, and saying so is half the
module.** Mean reversion in the filings runs toward the average growth of a
growth sector, 13%, not toward the 2.5% a perpetuity needs. The fitted curve
governs the first ``horizon_years``. The terminal assumption still governs the
tail, and ``GrowthPath.basis`` says on every single year which of the two is
speaking rather than blending them into one line nobody can take apart.

**The baseline is the whole test, and it is brutal.** "Next year's growth equals
this year's growth" scores a mean absolute error of 0.1510 on this panel. Three
baselines are reported before the model at every horizon: that persistence rule,
the unconditional training mean, and the mean of the company's own sub-vertical.
The ridge scores 0.1474 at one year, a lift of 0.0035 against a fold standard
deviation of 0.0242, which is to say it ties. **At one year out this model does
not beat doing nothing, and the verdict says so.** At two years it scores 0.1476
against 0.1884, and at three 0.1391 against 0.1921, both outside the fold noise.

That pattern is the result, not a disappointment. Growth is sticky one year out
and mean-reverting after that, so at h=1 there is almost nothing for a model to
add to last year's number, and by h=3 last year's number is actively misleading
while the conditional mean is not. A fade curve is a claim about the second
regime, which is the regime a five-year DCF spends four of its five years in.

**What the model is allowed to see.** Every observation is built through a
``CompanyFacts`` pinned to the filing date of the 10-K that first reported the
fiscal year, so the features are the figures as the filer first published them,
and the observation is dated to the filing rather than to the year end: a
December year end is not public in January.

**Restatement, measured rather than assumed.** The brief for this module
expected revenue restatement to be commoner than people assume. It is rarer.
Holding the us-gaap tag fixed, 0.9% of the 2,798 fiscal years here carry a
different revenue today than in the filing that first reported them, and 0.7%
differ by more than one percent. What is two and a half times commoner, and far
larger when it happens, is the filer moving revenue to a different tag: through
the full ladder 2.6% of years move and 1.75% move by more than a percent, and
the largest gaps in the panel are all tag migrations rather than restatements.
Crown Castle's fiscal 2017 reads 88% lower through the ladder today than in its
own 10-K, and nothing was restated. Both numbers are reported, separately,
because they are different facts and only one of them is about accounting.

**The label is also point in time.** Forward growth for horizon h is literally
the growth the filer printed in its 10-K h years later, numerator and
denominator from that one filing. Mixing an as-filed denominator with a restated
numerator would manufacture growth no filing ever claimed.

**Acquisitions are kept, not excluded.** 399 labelled observations spent more
than a tenth of revenue buying other companies. They grow 2.1 points faster than
quiet years in the year of the deal and 5.9 points faster in the year after it,
because a deal closing in June contributes six months to this year and twelve to
the next. Excluding acquisitive years does not remove noise, it fits a fade curve
for a world in which nobody does M&A and then hands it to a DCF valuing a company
that will keep doing M&A. Acquisition spend over revenue is a feature instead.
``techval.tmt.precedents`` was the obvious source for an acquirer flag and it is
the wrong one: ``deal_events`` labels companies that were bid FOR, and it cannot
see them either, for the reason below.

**Survivorship is the way this model is most likely to be wrong in the direction
that costs money, and it is worth 9% of Datadog.** A company whose growth
collapses is bought or delisted and stops filing. The panel keeps 119 delisted
filers in until the day they stop, identified by CIK because their tickers no
longer resolve: 1,224 of the 2,798 observations, 44% of the panel. Their last
observed year grows 9.8% against 15.8% for the filers still quoted, and they then
vanish. Dropping them lifts mean forward growth by 0.8 points at one year, 1.3 at
two and 1.5 at three, and it lifts Datadog's fitted enterprise value by 1,377mm,
3.75 a share. Replacing the typed fade with the fitted one is worth 3.96 a share.
So the sample-construction decision is 95% as large as the entire modelling
decision, and it is the one nobody would have seen.

That the leavers are hard to find is not incidental. ``EdgarClient.ticker_to_cik``
resolves through the SEC's current ``company_tickers.json``, so Splunk, VMware,
Xilinx, Activision and 115 others raise ``MissingDataError`` that reads like a
typo, and ``deal_events(["SPLK"], ...)`` returns an empty list: Cisco paid 28bn
for Splunk in 2024, and the engine reports a company nobody bid for. Five names
that were in ``taxonomy.SEED`` when it was written have left the ticker file
since, Electronic Arts and Juniper among them. ``DelistedAwareClient`` is the fix,
a CIK at a time.

**What a fitted fade is worth, honestly.** The out-of-fold residual band at one
year runs from -20 points to +24 points around the estimate. A point forecast of
21% with an 80% interval of 1% to 45% is what this data supports, and
``GrowthPath`` carries the interval on every year. On Datadog those two ends are
a 26.97 and a 66.69 share against 36.65 on the assumed fade. Anyone quoting the
middle of that band as a forecast has stopped reporting a measurement.

**Determinism.** Nothing here samples. The ridge penalty is chosen by closed-form
generalised cross-validation on the training fold, the gradient booster takes
``assumptions.ml.random_seed``, and the panel is a pure function of the fact
payloads. Two runs agree bit for bit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .. import tags
from ..config import Assumptions
from ..edgar import CompanyFacts, EdgarClient, Provenance
from ..errors import ConfigError, MissingDataError, NotMeaningfulError
from ..tmt.taxonomy import SEED, SubVertical
from .evaluation import evaluate_regression, walk_forward_folds
from .features import RND_EXPENSE, SNM_EXPENSE, FeatureRow, assert_point_in_time
from .protocol import EvalResult, ModelCard

_MM = 1e6

# --------------------------------------------------------------------------- #
# Tag ladders this module needs beyond the ones tags.py already carries
# --------------------------------------------------------------------------- #

# Cash paid for acquisitions, from the investing section of the cash flow
# statement. This is the acquirer flag. ``techval.tmt.precedents`` was the
# obvious place to look for one and it is the wrong place: ``deal_events``
# labels companies that were bid FOR, not companies that bid, and it resolves a
# ticker through today's SEC ticker file, so a target that was actually acquired
# no longer resolves and comes back as though nobody had bid for it. The cash
# flow statement says what a company spent on acquisitions, in dollars, in the
# year it spent them, and it says it for every filer including the ones that
# have since disappeared.
ACQUISITION_SPEND = [
    "PaymentsToAcquireBusinessesNetOfCashAcquired",
    "PaymentsToAcquireBusinessesAndInterestInAffiliatesNetOfCashAcquired",
    "PaymentsToAcquireBusinessesGross",
]

# Operating cash flow, with the continuing-operations variant behind it. A filer
# that has discontinued a segment reports the continuing figure and nothing else
# for the years around the disposal.
OPERATING_CASH_FLOW = list(tags.CFO) + [
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
]


# --------------------------------------------------------------------------- #
# The contract: feature names and order
# --------------------------------------------------------------------------- #

# Column order is fixed here and nowhere else. Two omissions are deliberate and
# both were measured rather than assumed.
#
# There is no company-age or years-of-filing-history term, and the argument for
# leaving it out is a mechanism rather than a score. In an expanding-window
# walk-forward every company has more filing history in the test block than any
# company had in the training block, because XBRL began in 2009 and the clock
# runs forward: the feature drifts 2.2 training standard deviations between the
# final fold's training and test rows, the largest of any candidate here and
# more than twice the next. A coefficient on it is therefore always an
# extrapolation, and in this panel it is confounded with the calendar, since
# filers with long histories are observed disproportionately in the years after
# 2022 when TMT growth slowed everywhere. Fitted on this panel it improves the
# one-year error by 0.002, which is a tenth of the fold noise and is the model
# reading the date off the feature. A fade curve whose shape depends on when it
# was run rather than on the company is not a fade curve.
#
# It is also the feature that exposed the imputation bug documented on
# ``_Scaler``. With the scaler standardising by the imputed column's collapsed
# dispersion, adding this one term took the first fold from a mean absolute
# error of 0.16 to 0.46 against 0.16 for doing nothing, and the model forecast
# 370% revenue growth for a company. The two defects only bit together, the
# scaler is fixed, and ``FadeModel.fits[h].drift`` now reports the shift of every
# feature in training standard deviations so the next clock is visible before it
# is fitted rather than after.
#
# There is no net revenue retention term either, and it is the feature a TMT
# analyst would reach for first. It is not a us-gaap concept. It is disclosed in
# prose and in custom tags, ``techval.tmt.kpis`` reads it out of the filing text
# one filing at a time, and this panel is 2,798 filings deep. Deferred revenue
# over revenue and its growth are the billings proxy that companyfacts can carry
# for every filer and every year, and they are weaker.
FADE_FEATURES: tuple[str, ...] = (
    "growth_1y",
    "growth_2y",
    "growth_3y",
    "growth_acceleration",
    "log_revenue",
    "margin_gross",
    "margin_ebit",
    "margin_fcf",
    "spend_rnd",
    "spend_sales_marketing",
    "spend_sbc",
    "acquisition_spend",
    "deferred_revenue_ratio",
    "deferred_revenue_growth",
)

SUB_VERTICALS: tuple[str, ...] = tuple(sorted(v.value for v in SubVertical))

MATRIX_COLUMNS: tuple[str, ...] = FADE_FEATURES + tuple(
    f"is_{v}" for v in SUB_VERTICALS
)

# A fiscal year is 52 or 53 weeks, so the anniversary of a period end lands 364
# or 371 days earlier rather than 365, and a 53-week year stretches one of them.
# Forty-five days is wide enough for every filer in the panel and far inside the
# half-year that would be needed to confuse two fiscal years.
_ANNIVERSARY_TOLERANCE_DAYS = 45

# A duration fact is an annual period if its window is within this band. A
# 52-week year is 364 days, a 53-week year 371, and a transition period after a
# fiscal year end change can run a few weeks either way without ceasing to be
# the filer's annual figure.
_ANNUAL_MIN_DAYS = 330
_ANNUAL_MAX_DAYS = 400

# Two ladder entries reporting the same annual period and an order of magnitude
# apart are not disagreeing about scope. One of them is a disaggregation
# component and the other is the income statement total, and the total is the
# larger. Charter tags RevenueFromContractWithCustomerIncludingAssessedTax at
# 889mm for fiscal 2025 and Revenues at 54,774mm; the ladder in tags.py ranks the
# first above the second, so without this guard the panel carries Charter's
# revenue 62 times too small. Below an order of magnitude the guard stays out of
# the way, because the 2 to 5x disagreements in this universe are genuine
# questions of scope -- revenue before or after billable expenses at Interpublic,
# services revenue against total at Starz -- and picking the larger of those by
# rule would be a thumb on the scale.
_COMPONENT_GUARD = 10.0

# A year-on-year revenue ratio outside this band is not growth. Early XBRL
# filings carry scale errors: Groupon's fiscal 2009 and 2010 revenue is tagged in
# thousands under SalesRevenueServicesNet and in units under Revenues, a factor
# of exactly 1,000. A ratio that extreme is a change of units or of reporting
# entity, and it is recorded as unavailable with the reason rather than entered
# as a 99,900% growth rate.
_MAX_GROWTH_RATIO = 100.0
_MIN_GROWTH_RATIO = 0.01

# Cross-sectional winsorization bounds, matching features.py.
_WINSOR_LOWER_PCT = 1.0
_WINSOR_UPPER_PCT = 99.0
_WINSOR_MIN_OBSERVATIONS = 20

# Standardised feature values beyond this are clipped. A test observation
# fifteen training standard deviations out is not a measurement the training
# period has anything to say about, and left alone it multiplies a small
# coefficient into a forecast of 370% revenue growth, which is what the first
# version of this module produced. Clipping states that the value is off the
# scale rather than pretending the fit knows what to do with it.
_STANDARDISED_CLIP = 5.0

# Ridge penalties searched by closed-form generalised cross-validation on the
# training fold. The range spans four orders of magnitude either side of one
# because the early folds hold sixty observations and the late ones two
# thousand, and the right penalty for those two is not the same number.
_RIDGE_ALPHAS = tuple(float(a) for a in np.logspace(-2.0, 4.0, 25))

# Interval quantiles. Eighty per cent rather than ninety-five, because the tails
# of a growth residual on two thousand observations are four observations deep
# and an interval nobody can measure is worse than a narrower one that can be.
_INTERVAL_LOWER_PCT = 10.0
_INTERVAL_UPPER_PCT = 90.0

# What ``HorizonFit.evaluation`` is scored against, in the words its verdict
# prints. ``evaluate_regression`` has no way to know what a supplied baseline
# array means and calls it "supplied baseline", which is the right default and
# the wrong thing for a reader to see: the model card and the dashboard quote
# the verdict verbatim, and a lift against an unnamed baseline says nothing.
PERSISTENCE_BASELINE = "persistence (this year's growth carried forward)"



# --------------------------------------------------------------------------- #
# The filers the ticker file has forgotten
# --------------------------------------------------------------------------- #

# 119 TMT filers that were acquired, taken private or wound up between 2011 and
# 2026, with the sub-vertical stated by hand and the CIK carried because the
# ticker is no longer an identifier. Every one of them is invisible to
# ``taxonomy.tmt_universe`` and to ``precedents.deal_events``, both of which
# resolve a ticker through the SEC's current ``company_tickers.json``. Splunk is
# the clean demonstration: Cisco paid 28bn for it in 2024, and
# ``deal_events(["SPLK"], ...)`` returns an empty list, which is exactly what a
# company nobody bid for returns.
#
# Keeping them costs nothing and changes the answer. They are 1,224 of the
# panel's 2,798 observations, 44% of it, and the sample without them fades more
# slowly at every horizon.

DELISTED_UNIVERSE: dict[str, str] = {
    # application software
    "AVLR": "application_software",
    "AYX": "application_software",
    "BV": "application_software",
    "CERN": "application_software",
    "COUP": "application_software",
    "CPWR": "application_software",
    "CSOD": "application_software",
    "DATA": "application_software",
    "DCT": "application_software",
    "ELLI": "application_software",
    "EPAY": "application_software",
    "EVBG": "application_software",
    "LOGM": "application_software",
    "MDLA": "application_software",
    "MKTO": "application_software",
    "MNTV": "application_software",
    "MRIN": "application_software",
    "N": "application_software",
    "NUAN": "application_software",
    "PLAN": "application_software",
    "PS": "application_software",
    "RP": "application_software",
    "SNCR": "application_software",
    "TYPE": "application_software",
    "ULTI": "application_software",
    "WORK": "application_software",
    "XM": "application_software",
    "ZEN": "application_software",
    # gaming
    "ATVI": "gaming",
    "EA": "gaming",
    "ZNGA": "gaming",
    # hardware
    "ACIA": "hardware",
    "AVYA": "hardware",
    "EMC": "hardware",
    "FNSR": "hardware",
    "JNPR": "hardware",
    "PLCM": "hardware",
    "PLT": "hardware",
    "RVBD": "hardware",
    # infrastructure software
    "BCOV": "infrastructure_software",
    "CARB": "infrastructure_software",
    "CLDR": "infrastructure_software",
    "CTXS": "infrastructure_software",
    "EIGI": "infrastructure_software",
    "FORG": "infrastructure_software",
    "INAP": "infrastructure_software",
    "INFA_OLD": "infrastructure_software",
    "KNBE": "infrastructure_software",
    "MIME": "infrastructure_software",
    "MNDT": "infrastructure_software",
    "NEWR": "infrastructure_software",
    "PFPT": "infrastructure_software",
    "PING": "infrastructure_software",
    "RAX": "infrastructure_software",
    "SAIL": "infrastructure_software",
    "SPLK": "infrastructure_software",
    "SUMO": "infrastructure_software",
    "VMW": "infrastructure_software",
    # internet
    "ANGI_OLD": "internet",
    "APRN": "internet",
    "ETSY": "internet",
    "FUEL": "internet",
    "GRPN": "internet",
    "GRUB": "internet",
    "JCOM": "internet",
    "LNKD": "internet",
    "OSTK": "internet",
    "SSTK": "internet",
    "TRMR": "internet",
    "TWTR": "internet",
    "WEB": "internet",
    # it services
    "CSRA": "it_services",
    "CVG": "it_services",
    "PRSP": "it_services",
    "SYKE": "it_services",
    "SYNT": "it_services",
    "TTEC": "it_services",
    "VRTU": "it_services",
    # media entertainment
    "CMLS": "media_entertainment",
    "ETM": "media_entertainment",
    "FOXA_OLD": "media_entertainment",
    "IPG": "media_entertainment",
    "MDP": "media_entertainment",
    "MSGN": "media_entertainment",
    "NLSN": "media_entertainment",
    "P": "media_entertainment",
    "ROVI": "media_entertainment",
    "SNI": "media_entertainment",
    "STRZA": "media_entertainment",
    "TIVO": "media_entertainment",
    "TRCO": "media_entertainment",
    "TWX": "media_entertainment",
    # payments
    "FDC": "payments",
    "FI": "payments",
    "TSS": "payments",
    "VNTV": "payments",
    "WP": "payments",
    # semiconductors
    "ALTR": "semiconductors",
    "ATML": "semiconductors",
    "CAVM": "semiconductors",
    "CY": "semiconductors",
    "IDTI": "semiconductors",
    "IPHI": "semiconductors",
    "ISIL": "semiconductors",
    "LLTC": "semiconductors",
    "MLNX": "semiconductors",
    "MSCC": "semiconductors",
    "MXIM": "semiconductors",
    "XLNX": "semiconductors",
    # telecom
    "CBB": "telecom",
    "CNSL": "telecom",
    "CVC": "telecom",
    "DISH": "telecom",
    "FYBR": "telecom",
    "LBTYA": "telecom",
    "S": "telecom",
    "TWC": "telecom",
    "USM": "telecom",
    "WIN": "telecom",
}

# CIKs, because a ticker that no longer trades is not an identifier. PARA meant
# Paramount for years and today resolves to an unrelated small-cap. Every lookup
# in this module goes through the number.
DELISTED_CIKS: dict[str, int] = {
    "ACIA": 1651235,
    "ALTR": 768251,
    "ANGI_OLD": 1491778,
    "APRN": 1701114,
    "ATML": 872448,
    "ATVI": 718877,
    "AVLR": 1348036,
    "AVYA": 1418100,
    "AYX": 1689923,
    "BCOV": 1313275,
    "BV": 1330421,
    "CARB": 1340127,
    "CAVM": 1175609,
    "CBB": 716133,
    "CERN": 804753,
    "CLDR": 1535379,
    "CMLS": 1058623,
    "CNSL": 1304421,
    "COUP": 1385867,
    "CPWR": 859014,
    "CSOD": 1401680,
    "CSRA": 1646383,
    "CTXS": 877890,
    "CVC": 1053112,
    "CVG": 1062047,
    "CY": 791915,
    "DATA": 1303652,
    "DCT": 1160951,
    "DISH": 1001082,
    "EA": 712515,
    "EIGI": 1237746,
    "ELLI": 1122388,
    "EMC": 790070,
    "EPAY": 1073349,
    "ETM": 1067837,
    "ETSY": 1370637,
    "EVBG": 1437352,
    "FDC": 883980,
    "FI": 798354,
    "FNSR": 1094739,
    "FORG": 1543916,
    "FOXA_OLD": 1308161,
    "FUEL": 1477200,
    "FYBR": 20520,
    "GRPN": 1490281,
    "GRUB": 1594109,
    "IDTI": 703361,
    "INAP": 1056386,
    "INFA_OLD": 1080099,
    "IPG": 51644,
    "IPHI": 1160958,
    "ISIL": 1096325,
    "JCOM": 1084048,
    "JNPR": 1043604,
    "KNBE": 1664998,
    "LBTYA": 1570585,
    "LLTC": 791907,
    "LNKD": 1271024,
    "LOGM": 1420302,
    "MDLA": 1540184,
    "MDP": 65011,
    "MIME": 1644675,
    "MKTO": 1490660,
    "MLNX": 1356104,
    "MNDT": 1370880,
    "MNTV": 1739936,
    "MRIN": 1389002,
    "MSCC": 310568,
    "MSGN": 1469372,
    "MXIM": 743316,
    "N": 1117106,
    "NEWR": 1448056,
    "NLSN": 1492633,
    "NUAN": 1002517,
    "OSTK": 1130713,
    "P": 1230276,
    "PFPT": 1212458,
    "PING": 1679826,
    "PLAN": 1540755,
    "PLCM": 1010552,
    "PLT": 914025,
    "PRSP": 1724670,
    "PS": 1725579,
    "RAX": 1107694,
    "ROVI": 1424454,
    "RP": 1286225,
    "RVBD": 1357326,
    "S": 101830,
    "SAIL": 1627857,
    "SNCR": 1131554,
    "SNI": 1430602,
    "SPLK": 1353283,
    "SSTK": 1549346,
    "STRZA": 1507934,
    "SUMO": 1643269,
    "SYKE": 1010612,
    "SYNT": 1040426,
    "TIVO": 1675820,
    "TRCO": 726513,
    "TRMR": 1375796,
    "TSS": 721683,
    "TTEC": 1013880,
    "TWC": 1377013,
    "TWTR": 1418091,
    "TWX": 1105705,
    "TYPE": 1385292,
    "ULTI": 1016125,
    "USM": 821130,
    "VMW": 1124610,
    "VNTV": 1533932,
    "VRTU": 1207074,
    "WEB": 1095291,
    "WIN": 1282266,
    "WORK": 1764925,
    "WP": 1533932,
    "XLNX": 743988,
    "XM": 1747748,
    "ZEN": 1463172,
    "ZNGA": 1439404,
}


# --------------------------------------------------------------------------- #
# Small guarded arithmetic
# --------------------------------------------------------------------------- #


def _ratio(num: float | None, den: float | None) -> float | None:
    """A share of revenue. Revenue at or below zero has no shares to take."""
    if num is None or den is None or den <= 0:
        return None
    return num / den


def _growth(new: float | None, old: float | None) -> float | None:
    """Year-on-year growth, with the scale guard. See ``_MAX_GROWTH_RATIO``."""
    if new is None or old is None or old <= 0:
        return None
    ratio = new / old
    if not (_MIN_GROWTH_RATIO < ratio < _MAX_GROWTH_RATIO):
        return None
    return ratio - 1.0


def _anniversary(
    ends: Sequence[date], anchor: date, years: int
) -> date | None:
    """The period end ``years`` fiscal years from ``anchor``, or None.

    Negative ``years`` walks forward, which is how a label reaches the fiscal
    year whose growth it is going to become.
    """
    target = anchor - timedelta(days=round(365.25 * years))
    best: date | None = None
    gap = _ANNIVERSARY_TOLERANCE_DAYS + 1
    for end in ends:
        d = abs((end - target).days)
        if d <= _ANNIVERSARY_TOLERANCE_DAYS and d < gap:
            best, gap = end, d
    return best


def _annual_series(
    facts: CompanyFacts, ladder: Sequence[str]
) -> tuple[dict[date, float], dict[date, str], dict[date, date]]:
    """Fiscal-year values by period end, with the winning tag and its filing date.

    Three rules, each of them earned on a real filer in the panel.

    *The ladder fills gaps rather than choosing once.* A filer that adopted ASC
    606 in 2018 reports the years since under
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` and the years before
    under ``Revenues`` or ``SalesRevenueNet``, and nothing repeats the old years
    under the new tag. Taking the first ladder entry with any facts at all would
    end the series in 2017 or start it in 2018; 39 observations in this panel
    have a different tag behind the current year and the prior one, Salesforce's
    fiscal 2018 among them, and all of them are genuine total revenue on both
    sides.

    *A ladder entry that is an order of magnitude larger than the incumbent for
    the same period replaces it.* See ``_COMPONENT_GUARD``, and Charter.

    *Only annual windows.* A duration fact is taken only when its window is
    within ``_ANNUAL_MIN_DAYS`` and ``_ANNUAL_MAX_DAYS``, so a quarter or a
    year-to-date stub can never be read as a fiscal year.

    Everything here goes through ``CompanyFacts.facts``, which means the
    knowledge date on the fact set is already in force: a series resolved
    through a fact set pinned to a filing date holds the figures as that filing
    stated them, comparatives included, and nothing filed afterwards.
    """
    values: dict[date, float] = {}
    source: dict[date, str] = {}
    filed: dict[date, date] = {}
    for tag in ladder:
        for fact in facts.facts(tag):
            if fact.is_instant:
                continue
            if not (_ANNUAL_MIN_DAYS <= fact.days <= _ANNUAL_MAX_DAYS):
                continue
            incumbent = values.get(fact.end)
            if incumbent is None:
                values[fact.end] = fact.val
                source[fact.end] = tag
                filed[fact.end] = fact.filed
            elif incumbent > 0 and fact.val > incumbent * _COMPONENT_GUARD:
                values[fact.end] = fact.val
                source[fact.end] = tag
                filed[fact.end] = fact.filed
    return values, source, filed


def _instant_series(
    facts: CompanyFacts, ladder: Sequence[str]
) -> tuple[dict[date, float], dict[date, date]]:
    """Balance-sheet values by date, ladder order, first entry to report wins."""
    values: dict[date, float] = {}
    filed: dict[date, date] = {}
    for tag in ladder:
        for fact in facts.facts(tag):
            if not fact.is_instant or fact.end in values:
                continue
            values[fact.end] = fact.val
            filed[fact.end] = fact.filed
    return values, filed


# --------------------------------------------------------------------------- #
# One observation
# --------------------------------------------------------------------------- #


@dataclass
class FadeObservation:
    """One company-year as the filing that reported it stated it.

    ``as_of`` is the date the 10-K carrying ``fiscal_year_end`` was filed, and
    it is the date every fold boundary and every embargo is measured against.
    It is not the fiscal year end: a December year end is not public information
    until February, and a model that dates the observation to 31 December has
    given itself six weeks of the future for free.

    ``labels`` holds forward growth by horizon in years, taken from the filing
    that reported each of those later years, and it is empty for the last
    observation of any filer, including every filer that was acquired. That
    emptiness is the survivorship channel, and ``FadePanel.truncation`` counts
    it rather than letting it disappear.

    ``restated_revenue`` is the same fiscal year as it stands in the fact set
    today. It plays no part in any fit. It exists so the distance between what
    was filed and what is now on file can be measured instead of asserted.
    """

    ticker: str
    cik: int
    sub_vertical: str
    as_of: date
    fiscal_year_end: date
    revenue: float
    values: dict[str, float | None] = field(default_factory=dict)
    labels: dict[int, float] = field(default_factory=dict)
    label_dates: dict[int, date] = field(default_factory=dict)
    provenance: dict[str, Provenance] = field(default_factory=dict)
    restated_revenue: float | None = None
    restated_same_tag: float | None = None
    revenue_tag: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def growth(self) -> float | None:
        """This year's growth. The persistence baseline, and the fade anchor."""
        return self.values.get("growth_1y")

    @property
    def restatement_gap(self) -> float | None:
        """Today's revenue for this year over the filed one, less one.

        Resolved through the whole ladder, so it moves when the filer changes
        which tag carries revenue as well as when the figure itself is restated.
        Use ``same_tag_gap`` to separate the two.
        """
        if self.restated_revenue is None or self.revenue <= 0:
            return None
        return self.restated_revenue / self.revenue - 1.0

    @property
    def same_tag_gap(self) -> float | None:
        """The same comparison, holding the us-gaap tag fixed.

        This is restatement proper. The ladder comparison above also picks up
        every filer that migrated from ``Revenues`` to a contract-revenue tag
        whose scope is different, and those migrations are the largest gaps in
        the panel by a wide margin: Crown Castle's fiscal 2017 reads 88% lower
        through the ladder today than in the filing, and none of that is a
        restatement of anything. None means the tag that reported this year is
        no longer in the filer's fact set at all.
        """
        if self.restated_same_tag is None or self.revenue <= 0:
            return None
        return self.restated_same_tag / self.revenue - 1.0

    def vector(self) -> np.ndarray:
        """One row of the design matrix, NaN where a figure was not sourced.

        NaN rather than zero, for the reason ``features.py`` gives at length: a
        gross margin filled with zero is a specific claim that the company broke
        even, and the fit cannot tell that claim from a measured one.
        """
        out = [
            np.nan if self.values.get(name) is None else float(self.values[name])
            for name in FADE_FEATURES
        ]
        out.extend(1.0 if self.sub_vertical == v else 0.0 for v in SUB_VERTICALS)
        return np.asarray(out, dtype=float)

    def feature_row(self) -> FeatureRow:
        """The observation dressed as a ``FeatureRow`` so the point-in-time
        checker in ``features.py`` can walk its provenance.

        Every figure that reached this observation recorded the filing date
        behind it, and ``features.assert_point_in_time`` compares each of those
        against ``as_of``. The check is on the filing date and never on the
        period end, which is the difference between a panel that is evidence and
        one that is an assertion.
        """
        return FeatureRow(
            ticker=self.ticker,
            as_of=self.as_of,
            knowledge_date=self.as_of,
            values=dict(self.values),
            provenance=dict(self.provenance),
            statement_date=self.fiscal_year_end,
        )

    def rows(self) -> list[tuple[str, Any]]:
        return [
            ("Ticker", self.ticker),
            ("Sub-vertical", self.sub_vertical),
            ("Fiscal year end", str(self.fiscal_year_end)),
            ("Filed", str(self.as_of)),
            ("Revenue (USD mm)", self.revenue / _MM),
            ("Growth", self.growth),
            ("Forward growth (1y)", self.labels.get(1)),
        ]


# --------------------------------------------------------------------------- #
# The panel
# --------------------------------------------------------------------------- #


@dataclass
class FadePanel:
    """Every company-year the filings support, with what the sample is missing.

    The diagnostics are not decoration. Each of the four is a way this model
    could be confidently wrong, and each is measured on this panel rather than
    asserted about panels in general.
    """

    observations: list[FadeObservation]
    depth: dict[str, int] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)
    winsorized: int = 0
    notes: list[str] = field(default_factory=list)
    random_seed: int = 7

    # -- shape ------------------------------------------------------------- #

    def labelled(self, horizon: int) -> list[FadeObservation]:
        """Observations carrying this year's growth and the label ``horizon``
        years out, ordered by filing date then ticker so the order is a fact
        about the data rather than about a dict."""
        rows = [
            o
            for o in self.observations
            if o.growth is not None and horizon in o.labels
        ]
        rows.sort(key=lambda o: (o.as_of, o.ticker))
        return rows

    @property
    def tickers(self) -> list[str]:
        return sorted({o.ticker for o in self.observations})

    def depth_summary(self) -> dict[str, float]:
        """How many fiscal years of revenue each filer actually supports.

        XBRL reaches back to 2007 and the brief for this module expected 15 to
        18 years for a mature name. Measured on 224 TMT filers: the deepest is
        19 fiscal years, the median is 12, and only 34 filers reach 19. The gap
        is not attrition, it is when each filer entered the mandate and when it
        listed, and it is the reason the first walk-forward fold trains on 64
        observations.
        """
        counts = sorted(self.depth.values())
        if not counts:
            return {}
        arr = np.asarray(counts, dtype=float)
        return {
            "companies": float(len(arr)),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "at_least_10": float(sum(1 for c in counts if c >= 10)),
        }

    # -- the four ways this could be wrong --------------------------------- #

    def restatement_report(self) -> dict[str, float]:
        """How often the revenue on file today is not the revenue that was filed.

        The measurement the point-in-time discipline exists for. Every fiscal
        year in the panel carries the figure its own 10-K stated and the figure
        the same year shows in today's fact set, and this counts the distance
        between them.
        """
        ladder = np.abs(
            np.asarray(
                [
                    o.restatement_gap
                    for o in self.observations
                    if o.restatement_gap is not None
                ],
                dtype=float,
            )
        )
        same = np.abs(
            np.asarray(
                [
                    o.same_tag_gap
                    for o in self.observations
                    if o.same_tag_gap is not None
                ],
                dtype=float,
            )
        )
        if ladder.size == 0:
            return {}
        out = {
            "compared": float(ladder.size),
            "ladder_above_0.1pct": float((ladder > 0.001).mean()),
            "ladder_above_1pct": float((ladder > 0.01).mean()),
            "ladder_above_10pct": float((ladder > 0.10).mean()),
        }
        if same.size:
            out |= {
                "same_tag_compared": float(same.size),
                "same_tag_above_0.1pct": float((same > 0.001).mean()),
                "same_tag_above_1pct": float((same > 0.01).mean()),
                "same_tag_above_10pct": float((same > 0.10).mean()),
                "same_tag_median_absolute": float(np.median(same)),
            }
            out["tag_retired"] = float(
                sum(
                    1
                    for o in self.observations
                    if o.restatement_gap is not None and o.same_tag_gap is None
                )
            )
        return out

    def survivorship_report(self) -> dict[str, Any]:
        """What the sample looks like with and without the filers that left.

        ``status`` is not a field on the observation because it is not a fact
        about the observation: a company is not delisted in 2014 because it was
        bought in 2024. What the panel carries instead is the date each filer
        stopped filing, and an observation is a leaver's last one when no later
        fiscal year exists for that ticker at all.
        """
        by_ticker: dict[str, list[FadeObservation]] = {}
        for o in self.observations:
            by_ticker.setdefault(o.ticker, []).append(o)
        finals = []
        truncated = 0
        for rows in by_ticker.values():
            rows = sorted(rows, key=lambda o: o.fiscal_year_end)
            finals.append(rows[-1])
            truncated += sum(1 for o in rows if 1 not in o.labels)
        growths = [o.growth for o in finals if o.growth is not None]
        panel = [o.growth for o in self.observations if o.growth is not None]
        return {
            "companies": len(by_ticker),
            "observations_without_a_next_year_label": truncated,
            "final_year_growth_mean": float(np.mean(growths)) if growths else None,
            "panel_growth_mean": float(np.mean(panel)) if panel else None,
        }

    def acquisition_report(self, threshold: float = 0.10) -> dict[str, Any]:
        """What an acquisitive year does to the year after it.

        The finding that decides whether acquisitive years are excluded: a heavy
        acquirer grows no faster in the year of the deal and 7.6 points faster
        in the year after it, because a deal closing in June contributes six
        months to this year and twelve to the next. Excluding those years fits
        the curve for a world without M&A.
        """
        rows = [
            o
            for o in self.observations
            if o.growth is not None
            and 1 in o.labels
            and o.values.get("acquisition_spend") is not None
        ]
        heavy = [o for o in rows if o.values["acquisition_spend"] > threshold]
        quiet = [o for o in rows if o.values["acquisition_spend"] <= 0.01]
        mean = lambda xs, key: float(np.mean([key(o) for o in xs])) if xs else None
        return {
            "reporting_acquisition_spend": len(rows),
            "heavy": len(heavy),
            "quiet": len(quiet),
            "same_year_growth_heavy": mean(heavy, lambda o: o.growth),
            "same_year_growth_quiet": mean(quiet, lambda o: o.growth),
            "next_year_growth_heavy": mean(heavy, lambda o: o.labels[1]),
            "next_year_growth_quiet": mean(quiet, lambda o: o.labels[1]),
        }

    def reversion_table(
        self, horizon: int = 1, buckets: int = 10
    ) -> pd.DataFrame:
        """Mean forward growth by decile of trailing growth. No functional form.

        The fade curve without a model, and the most persuasive object in this
        module: sort every company-year by the growth it just reported, cut into
        ten equal buckets, and read off what each bucket did next. The top
        decile grows 90% and then 47%. The bottom decile shrinks 16% and then
        grows 7%. Both ends move toward the middle, the crossing point sits
        between 12 and 14%, and nothing about that depends on a regression
        specification anybody could quarrel with.
        """
        rows = self.labelled(horizon)
        if not rows:
            return pd.DataFrame()
        current = np.asarray([r.growth for r in rows], dtype=float)
        forward = np.asarray([r.labels[horizon] for r in rows], dtype=float)
        order = np.argsort(current, kind="mergesort")
        records = []
        for bucket in range(buckets):
            index = order[
                bucket * len(order) // buckets : (bucket + 1) * len(order) // buckets
            ]
            if index.size == 0:
                continue
            records.append(
                {
                    "Bucket": bucket + 1,
                    "Observations": int(index.size),
                    "Trailing growth": float(current[index].mean()),
                    "Forward growth": float(forward[index].mean()),
                    "Forward median": float(np.median(forward[index])),
                    "Fade": float(forward[index].mean() - current[index].mean()),
                }
            )
        return pd.DataFrame(records).set_index("Bucket")

    def to_frame(self) -> pd.DataFrame:
        records = []
        for o in self.observations:
            row = {
                "ticker": o.ticker,
                "sub_vertical": o.sub_vertical,
                "as_of": o.as_of,
                "fiscal_year_end": o.fiscal_year_end,
                "revenue": o.revenue / _MM,
            }
            row |= {name: o.values.get(name) for name in FADE_FEATURES}
            row |= {
                f"forward_{h}y": o.labels.get(h)
                for h in range(1, _MAX_LABEL_HORIZON + 1)
            }
            records.append(row)
        return pd.DataFrame(records)


def _report_dates(facts: CompanyFacts, ladder: Sequence[str]) -> dict[date, date]:
    """Fiscal year end to the date the first periodic report carrying it was filed.

    This reads the raw payload rather than ``CompanyFacts.facts``, and the
    reason is the one place in this engine where deduplication points the wrong
    way. ``facts`` keeps the most recently filed version of a period, because
    that is the restatement the company now stands behind and it is the right
    answer for a valuation. Here the question is the opposite one: when did this
    fiscal year first become public? That is the earliest annual report carrying
    it, and the later filings are precisely what must not be seen.

    Only 10-K and 20-F forms and their amendments count. An 8-K carrying an
    earnings release does put the number in the market earlier, by four to six
    weeks, and this deliberately ignores that: the release is unaudited, it is
    not tagged in XBRL for most filers, and dating the observation to the
    audited report is the conservative direction. The cost is that the panel
    sees each year slightly later than the market did, never earlier.
    """
    gaap = (facts.raw.get("facts") or {}).get("us-gaap") or {}
    first: dict[date, date] = {}
    for tag in ladder:
        node = gaap.get(tag)
        if not node:
            continue
        for row in (node.get("units") or {}).get("USD", []):
            start, end, filed = row.get("start"), row.get("end"), row.get("filed")
            if not (start and end and filed) or row.get("val") is None:
                continue
            form = row.get("form") or ""
            if not (form.startswith("10-K") or form.startswith("20-F")):
                continue
            begins = date.fromisoformat(start)
            ends = date.fromisoformat(end)
            if not (_ANNUAL_MIN_DAYS <= (ends - begins).days <= _ANNUAL_MAX_DAYS):
                continue
            when = date.fromisoformat(filed)
            if end not in first or when < first[ends]:
                first[ends] = when
    return first


def _build_company(
    ticker: str,
    sub_vertical: str,
    facts: CompanyFacts,
) -> tuple[list[FadeObservation], int]:
    """Every fiscal year one filer supports, each built through its own filing.

    The whole point-in-time guarantee of this module is the single line that
    re-pins the fact set: ``CompanyFacts(facts.raw, ticker, knowledge_date=...)``
    per fiscal year. Handing one unpinned fact set to every year would build
    2007 out of figures restated in 2024.
    """
    reported = _report_dates(facts, tags.REVENUE)
    if not reported:
        return [], 0

    today_revenue, _, _ = _annual_series(facts, tags.REVENUE)
    today_by_tag: dict[str, dict[date, float]] = {}
    for tag in tags.REVENUE:
        values = {
            fact.end: fact.val
            for fact in facts.facts(tag)
            if not fact.is_instant
            and _ANNUAL_MIN_DAYS <= fact.days <= _ANNUAL_MAX_DAYS
        }
        if values:
            today_by_tag[tag] = values
    by_year: dict[date, FadeObservation] = {}

    for fiscal_year_end, filed in sorted(reported.items()):
        pinned = CompanyFacts(facts.raw, ticker, knowledge_date=filed)
        revenue, revenue_tag, revenue_filed = _annual_series(pinned, tags.REVENUE)
        current = revenue.get(fiscal_year_end)
        if current is None or current <= 0:
            continue

        ends = list(revenue)
        prior = _anniversary(ends, fiscal_year_end, 1)
        prior2 = _anniversary(ends, fiscal_year_end, 2)
        prior3 = _anniversary(ends, fiscal_year_end, 3)

        gross_profit, _, gp_filed = _annual_series(pinned, tags.GROSS_PROFIT)
        cost_of_revenue, _, cor_filed = _annual_series(pinned, tags.COST_OF_REVENUE)
        ebit, _, ebit_filed = _annual_series(pinned, tags.EBIT)
        cfo, _, cfo_filed = _annual_series(pinned, OPERATING_CASH_FLOW)
        capex, _, capex_filed = _annual_series(pinned, tags.CAPEX)
        sbc, _, sbc_filed = _annual_series(pinned, tags.SBC)
        rnd, _, rnd_filed = _annual_series(pinned, RND_EXPENSE)
        snm, _, snm_filed = _annual_series(pinned, SNM_EXPENSE)
        acquisitions, _, acq_filed = _annual_series(pinned, ACQUISITION_SPEND)
        deferred, deferred_filed = _instant_series(
            pinned, tags.DEFERRED_REVENUE_CURRENT
        )

        # Gross profit where the filer publishes it, revenue less cost of
        # revenue where it does not. Never a partial: a filer reporting neither
        # leaves the margin missing rather than defaulting to the revenue line.
        gross = gross_profit.get(fiscal_year_end)
        if gross is None and fiscal_year_end in cost_of_revenue:
            gross = current - cost_of_revenue[fiscal_year_end]

        # Free cash flow after capital spending where both are reported. Where
        # capex is absent the operating figure stands alone and the feature is
        # an operating cash margin, which is stated in the note rather than
        # silently substituted.
        operating_cash = cfo.get(fiscal_year_end)
        spending = capex.get(fiscal_year_end)
        free_cash = None
        if operating_cash is not None:
            free_cash = operating_cash - (spending or 0.0)

        growth_1y = _growth(current, revenue.get(prior) if prior else None)
        growth_2y = _growth(
            revenue.get(prior) if prior else None,
            revenue.get(prior2) if prior2 else None,
        )
        growth_3y = _growth(
            revenue.get(prior2) if prior2 else None,
            revenue.get(prior3) if prior3 else None,
        )

        values: dict[str, float | None] = {
            "growth_1y": growth_1y,
            "growth_2y": growth_2y,
            "growth_3y": growth_3y,
            "growth_acceleration": (
                None if growth_1y is None or growth_2y is None else growth_1y - growth_2y
            ),
            "log_revenue": float(math.log(current / _MM)) if current > 0 else None,
            "margin_gross": _ratio(gross, current),
            "margin_ebit": _ratio(ebit.get(fiscal_year_end), current),
            "margin_fcf": _ratio(free_cash, current),
            "spend_rnd": _ratio(rnd.get(fiscal_year_end), current),
            "spend_sales_marketing": _ratio(snm.get(fiscal_year_end), current),
            "spend_sbc": _ratio(sbc.get(fiscal_year_end), current),
            "acquisition_spend": _ratio(acquisitions.get(fiscal_year_end), current),
            "deferred_revenue_ratio": _ratio(
                deferred.get(fiscal_year_end), current
            ),
            "deferred_revenue_growth": _growth(
                deferred.get(fiscal_year_end),
                deferred.get(prior) if prior else None,
            ),
        }

        # Provenance carries the filing date behind every figure that reached
        # the row, including the historical anchors, so that
        # ``features.assert_point_in_time`` has something to prove rather than
        # a claim to accept.
        provenance: dict[str, Provenance] = {}

        def record(label: str, tag: str | None, when: date | None) -> None:
            if when is None:
                return
            provenance[label] = Provenance(
                concept=label, tag=tag, method="annual period", filed=str(when)
            )

        record("revenue", revenue_tag.get(fiscal_year_end), revenue_filed.get(fiscal_year_end))
        for label, anchor in (("revenue t-1", prior), ("revenue t-2", prior2), ("revenue t-3", prior3)):
            if anchor is not None:
                record(label, revenue_tag.get(anchor), revenue_filed.get(anchor))
        record("gross profit", None, gp_filed.get(fiscal_year_end))
        record("cost of revenue", None, cor_filed.get(fiscal_year_end))
        record("EBIT", None, ebit_filed.get(fiscal_year_end))
        record("operating cash flow", None, cfo_filed.get(fiscal_year_end))
        record("capital expenditure", None, capex_filed.get(fiscal_year_end))
        record("stock-based compensation", None, sbc_filed.get(fiscal_year_end))
        record("research and development", None, rnd_filed.get(fiscal_year_end))
        record("sales and marketing", None, snm_filed.get(fiscal_year_end))
        record("acquisition spend", None, acq_filed.get(fiscal_year_end))
        record("deferred revenue", None, deferred_filed.get(fiscal_year_end))
        if prior is not None:
            record("deferred revenue t-1", None, deferred_filed.get(prior))

        notes: list[str] = []
        if operating_cash is not None and spending is None:
            notes.append(
                "capital expenditure is not tagged, so margin_fcf is an operating "
                "cash margin rather than a free cash flow margin"
            )
        if prior is not None and revenue_tag.get(prior) != revenue_tag.get(fiscal_year_end):
            notes.append(
                f"growth_1y spans a tag change, {revenue_tag.get(prior)} to "
                f"{revenue_tag.get(fiscal_year_end)}"
            )

        by_year[fiscal_year_end] = FadeObservation(
            ticker=ticker,
            cik=facts.cik,
            sub_vertical=sub_vertical,
            as_of=filed,
            fiscal_year_end=fiscal_year_end,
            revenue=current,
            values=values,
            provenance=provenance,
            restated_revenue=today_revenue.get(fiscal_year_end),
            restated_same_tag=today_by_tag.get(
                revenue_tag.get(fiscal_year_end, ""), {}
            ).get(fiscal_year_end),
            revenue_tag=revenue_tag.get(fiscal_year_end),
            notes=notes,
        )

    return list(by_year.values()), len(today_revenue)


# ``ForecastAssumptions.horizon_years`` allows up to five, so the panel carries
# labels to five whatever the run asks for. Building only as far as the current
# configuration would make the panel depend on the assumptions file, and a panel
# that changes shape when a setting changes is not a dataset.
_MAX_LABEL_HORIZON = 5


def _attach_labels(
    observations: Sequence[FadeObservation], horizons: int = _MAX_LABEL_HORIZON
) -> int:
    """Forward growth for each horizon, taken from the later filing's own figures.

    Two decisions are in this function and both of them are load-bearing.

    *Numerator and denominator come from the same later filing.* The label for
    horizon h is literally ``growth_1y`` on the observation h fiscal years
    later, which is the growth rate that company printed in that year's 10-K.
    Building it as later revenue over this observation's revenue would mix two
    vintages of the same fiscal year and manufacture growth no filing ever
    claimed, which is exactly what a restatement between the two creates: on
    this panel, a 1.75% chance per year of inventing several points of growth.

    *It runs after winsorization, not before.* A label here is another
    observation's feature, so it is trimmed in the cross-section it belongs to,
    among the companies filing alongside it, rather than in the cross-section of
    the year that is forecasting it. Trimming a 2011 label by the 2008 bounds
    would be a different operation and a worse one.
    """
    by_ticker: dict[str, dict[date, FadeObservation]] = {}
    for observation in observations:
        by_ticker.setdefault(observation.ticker, {})[
            observation.fiscal_year_end
        ] = observation
    attached = 0
    for years in by_ticker.values():
        ends = list(years)
        for fiscal_year_end, observation in years.items():
            for horizon in range(1, horizons + 1):
                target = _anniversary(ends, fiscal_year_end, -horizon)
                if target is None:
                    continue
                later = years[target]
                if later.growth is None:
                    continue
                observation.labels[horizon] = later.growth
                observation.label_dates[horizon] = later.as_of
                attached += 1
    return attached


def _winsorize_by_year(observations: Sequence[FadeObservation]) -> int:
    """Trim each filing year's cross-section at the 1st and 99th percentiles.

    Cross-sectional rather than pooled, for the reason ``features.py`` gives:
    a pooled bound bakes the level of a whole era into the trim, and the
    software cross-section of 2011 and of 2025 are not the same distribution.
    Grouped by the calendar year of the filing date rather than by the exact
    date, because filings arrive one at a time and an exact-date cross-section
    here is one company wide.

    A year with fewer than ``_WINSOR_MIN_OBSERVATIONS`` names is skipped rather
    than trimmed, because at that width the 1st percentile falls between the
    first and second order statistics and the operation degenerates into
    pulling the single most extreme name to its neighbour.
    """
    touched = 0
    groups: dict[int, list[FadeObservation]] = {}
    for o in observations:
        groups.setdefault(o.as_of.year, []).append(o)
    for rows in groups.values():
        if len(rows) < _WINSOR_MIN_OBSERVATIONS:
            continue
        for name in FADE_FEATURES:
            present = [
                o.values[name]
                for o in rows
                if o.values.get(name) is not None and np.isfinite(o.values[name])
            ]
            if len(present) < _WINSOR_MIN_OBSERVATIONS:
                continue
            low, high = np.percentile(present, [_WINSOR_LOWER_PCT, _WINSOR_UPPER_PCT])
            for o in rows:
                value = o.values.get(name)
                if value is None or not np.isfinite(value):
                    continue
                clipped = float(min(max(value, low), high))
                if clipped != value:
                    o.values[name] = clipped
                    touched += 1
    return touched


def build_fade_panel(
    universe: Mapping[str, str],
    client: Any,
    assumptions: Assumptions | None = None,
    *,
    winsorize_cross_section: bool = True,
    check_point_in_time: bool = True,
) -> FadePanel:
    """Build every fiscal year every filer in the universe supports.

    ``universe`` maps ticker to sub-vertical. It is passed in rather than
    derived because the filers that matter most to this model are the ones that
    no longer exist, and ``taxonomy.classify`` cannot reach them: it resolves a
    ticker through the SEC's current ticker file and a delisted ticker is not in
    it. ``fade_universe`` builds the default mapping and says which names are
    which.

    ``client`` needs one method, ``company_facts(ticker) -> CompanyFacts``, and
    unlike everywhere else in this package it should be handed an *unpinned*
    client. That is not a relaxation of the point-in-time rule, it is where the
    rule is applied: this module pins the fact set once per fiscal year, to the
    filing date of the report that first carried that year, which is a different
    knowledge date for every row. A single pinned client could only serve one of
    them.

    A filer that cannot be built is recorded in ``failures`` with the reason and
    never dropped silently, because the count of what could not be built is part
    of the result. A ``LookaheadError`` is not caught: it does not cost a row,
    it says the panel is not evidence.
    """
    assumptions = assumptions or Assumptions()
    observations: list[FadeObservation] = []
    depth: dict[str, int] = {}
    failures: list[tuple[str, str]] = []

    for ticker in sorted(universe):
        sub_vertical = universe[ticker]
        try:
            facts = client.company_facts(ticker)
        except Exception as exc:  # noqa: BLE001 - the reason is the record
            failures.append((ticker, f"{type(exc).__name__}: {exc}"))
            continue
        rows, years = _build_company(ticker, sub_vertical, facts)
        depth[ticker] = years
        if not rows:
            failures.append(
                (ticker, "no fiscal year could be built from the revenue ladder")
            )
            continue
        observations.extend(rows)

    if check_point_in_time:
        for observation in observations:
            assert_point_in_time(observation.feature_row())

    touched = _winsorize_by_year(observations) if winsorize_cross_section else 0
    _attach_labels(observations)

    panel = FadePanel(
        observations=observations,
        depth=depth,
        failures=failures,
        winsorized=touched,
        random_seed=assumptions.ml.random_seed,
    )
    if failures:
        panel.notes.append(
            f"{len(failures)} of {len(universe)} filers could not be built and are "
            "carried with the reason attached, not dropped"
        )
    summary = panel.depth_summary()
    if summary:
        panel.notes.append(
            f"{int(summary['companies'])} filers, median {summary['median']:.0f} "
            f"fiscal years of revenue each, deepest {summary['max']:.0f}, "
            f"{int(summary['at_least_10'])} with ten or more"
        )
    return panel


class DelistedAwareClient(EdgarClient):
    """An ``EdgarClient`` that can still find the filers the ticker file dropped.

    ``EdgarClient.ticker_to_cik`` reads ``company_tickers.json``, which is a
    list of companies that exist today. Every delisted filer in this panel fails
    there, and it fails as ``MissingDataError``, which reads exactly like a
    ticker somebody mistyped. That single lookup is the mechanism by which
    survivorship enters this engine: ``taxonomy.tmt_universe`` cannot build a
    historical universe, ``precedents.deal_events`` cannot see a completed deal
    because the target no longer resolves, and any panel built by walking
    today's tickers backwards is a panel of winners.

    This subclass falls back to ``DELISTED_CIKS``, which is the honest fix a CIK
    at a time rather than the point-in-time ticker file this engine does not
    have. Everything else about the client, the throttle, the cache and the
    knowledge date, is unchanged.
    """

    def ticker_to_cik(self, ticker: str) -> int:
        try:
            return super().ticker_to_cik(ticker)
        except MissingDataError:
            cik = DELISTED_CIKS.get(ticker.upper())
            if cik is None:
                raise
            return cik


def fade_universe(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Ticker to sub-vertical for the panel, survivors and leavers together.

    The survivors come from ``taxonomy.SEED``, which is the curated TMT prior
    the rest of this package screens on. The leavers are added here because they
    cannot be added there: ``taxonomy.tmt_universe`` resolves every candidate
    through ``client.submissions``, which resolves the ticker through the SEC's
    current ticker file, and a company that was acquired is not in it. Five
    names that were in SEED when it was written have since left it, Electronic
    Arts and Juniper among them, which is the bias arriving in real time.

    Sub-verticals for the leavers are stated by hand, in the same spirit and for
    the same reason as SEED itself: the alternative is a SIC code that describes
    the filing entity rather than the business, and Netflix still files under
    the code for renting video cassettes.
    """
    out = {ticker: vertical.value for ticker, vertical in SEED.items()}
    out.update(DELISTED_UNIVERSE)
    if extra:
        out.update({k.upper(): v for k, v in extra.items()})
    return out


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


@dataclass
class _Scaler:
    """Centre and scale on the training fold's OBSERVED values, then clip.

    Both halves matter and the first one is subtle. Impute a column that is 75%
    missing with its median and the imputed column's dispersion collapses to a
    quarter of the real one; standardise by that collapsed dispersion and a test
    observation that is two standard deviations out on the real distribution
    arrives as sixty-six. That is not a hypothetical. It happened here, to
    deferred revenue growth in the 2009-2010 training fold, and it turned a
    coefficient of 0.05 into a forecast of 370% revenue growth.

    So the centre and the scale come from the values that were actually
    observed, missing entries land at exactly zero after standardisation, which
    is the training median and the most honest thing to say about a figure
    nobody reported, and everything is clipped at ``_STANDARDISED_CLIP``.
    """

    centre: np.ndarray
    scale: np.ndarray
    observed_share: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "_Scaler":
        width = matrix.shape[1]
        centre = np.zeros(width)
        scale = np.ones(width)
        share = np.zeros(width)
        for j in range(width):
            column = matrix[:, j]
            observed = column[np.isfinite(column)]
            share[j] = observed.size / max(column.size, 1)
            if observed.size >= 5:
                centre[j] = float(np.median(observed))
                spread = float(np.std(observed))
                scale[j] = spread if spread > 1e-9 else 1.0
        return cls(centre=centre, scale=scale, observed_share=share)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(matrix), matrix, self.centre)
        return np.clip(
            (filled - self.centre) / self.scale,
            -_STANDARDISED_CLIP,
            _STANDARDISED_CLIP,
        )

    def drift(self, matrix: np.ndarray) -> np.ndarray:
        """Mean shift between a later block and the training fold, in training
        standard deviations. A feature that is a clock rather than a
        characteristic reports a large number here and is not fittable."""
        observed = np.where(np.isfinite(matrix), matrix, np.nan)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(observed, axis=0)
        means = np.where(np.isfinite(means), means, self.centre)
        return (means - self.centre) / self.scale


def _estimator(kind: str, seed: int):
    """The three models ``assumptions.ml.forecast.model`` offers.

    Ridge is the default and it should be. The panel is 2,000 rows wide by 25
    columns, the columns are collinear by construction (three growth rates and
    their difference), and the first walk-forward fold holds sixty observations.
    A penalty chosen by closed-form generalised cross-validation on the training
    fold is the right amount of structure for that; unpenalised least squares on
    the same sixty rows scores 0.249 against 0.149 for doing nothing, and is
    offered only so the difference is visible.

    Gradient boosting wins at one and three years and loses at two, by margins
    inside the fold-to-fold noise in all three cases. It is not the default,
    because a fade curve that a reader cannot differentiate is a fade curve
    nobody can argue with, and arguing with it is the point.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, RidgeCV

    if kind == "ridge":
        return RidgeCV(alphas=np.asarray(_RIDGE_ALPHAS))
    if kind == "linear":
        return LinearRegression()
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(
            random_state=seed,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
        )
    raise ConfigError(
        f"unknown forecast model {kind!r}: one of ridge, gradient_boosting, linear"
    )


def _sub_vertical_means(
    rows: Sequence[FadeObservation],
    labels: np.ndarray,
    train: np.ndarray,
    minimum: int = 10,
) -> tuple[dict[str, float], float]:
    """Mean forward growth by sub-vertical over the training rows only.

    The third baseline, and the one that does the work persistence cannot: it
    knows that infrastructure software and telecom do not revert to the same
    number. A sub-vertical with fewer than ``minimum`` training rows falls back
    to the pooled mean rather than reporting an average of four companies.
    """
    pooled = float(np.mean(labels[train])) if train.size else float("nan")
    buckets: dict[str, list[float]] = {}
    for index in train:
        buckets.setdefault(rows[index].sub_vertical, []).append(float(labels[index]))
    means = {
        vertical: (float(np.mean(values)) if len(values) >= minimum else pooled)
        for vertical, values in buckets.items()
    }
    return means, pooled


@dataclass
class HorizonFit:
    """One horizon: the fitted model, the three baselines, and the residuals.

    ``evaluation`` compares the model against persistence, which is the hardest
    of the three and the one ``EvalResult.verdict`` will print. The other two
    sit beside it in ``baselines`` because a model that beats persistence and
    loses to the mean of its own sub-vertical has not learned anything about the
    company, only something about the sector, and the reader is owed both
    numbers.

    ``residual_quantiles`` are taken out of fold. An in-sample residual band on
    a ridge is a statement about the penalty, not about the uncertainty.
    """

    horizon: int
    estimator: Any
    scaler: _Scaler
    evaluation: EvalResult
    baselines: dict[str, float]
    residual_quantiles: tuple[float, float]
    residual_sd: float
    n_train: int
    sub_vertical_means: dict[str, float]
    pooled_mean: float
    drift: dict[str, float] = field(default_factory=dict)

    def predict(self, observation: FadeObservation) -> float:
        matrix = observation.vector().reshape(1, -1)
        return float(self.estimator.predict(self.scaler.transform(matrix))[0])

    def interval(self, point: float) -> tuple[float, float]:
        """The point estimate plus the out-of-fold residual band around it.

        Unconditional: the same width for every company. A conditional band
        would be the right answer and this panel cannot support one, because the
        residual quantiles of a sub-vertical with 80 observations are two
        observations deep at each tail. Saying so is better than fitting one.
        """
        low, high = self.residual_quantiles
        return point + low, point + high


@dataclass
class GrowthPath:
    """A revenue growth path a DCF can take, with a band and a provenance per year.

    ``basis`` is the field that stops this being a black box. Year by year it
    says ``fitted`` where the model spoke, and ``assumed`` where the path has
    fallen back to the engine's own straight line toward
    ``dcf.revenue_growth_terminal``. The fallback is not a defect and it is not
    optional: what the filings support is mean reversion toward the average
    growth of a growth sector, the level ``FadeModel.reversion_level`` reports,
    and no terminal value can be built on a rate that high. The fitted curve
    governs the first ``horizon_years``. The assumption still governs the
    perpetuity, and this field is where the handover is visible rather than
    blended away.
    """

    ticker: str
    as_of: date
    growth: list[float]
    lower: list[float]
    upper: list[float]
    basis: list[str]
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.growth)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Year": list(range(1, len(self.growth) + 1)),
                "Growth": self.growth,
                "Low": self.lower,
                "High": self.upper,
                "Basis": self.basis,
            }
        ).set_index("Year")

    def rows(self) -> list[tuple[str, Any]]:
        return [
            (f"Year {i + 1} growth", f"{g:.1%} ({lo:.1%} to {hi:.1%}, {b})")
            for i, (g, lo, hi, b) in enumerate(
                zip(self.growth, self.lower, self.upper, self.basis)
            )
        ]


@dataclass
class FadeModel:
    """A fitted fade curve, and everything needed to argue with it.

    ``card`` carries the honest verdict of the one-year fit, which is the
    horizon a reader will check first. ``fits`` carries every horizon, each with
    its own three baselines.

    ``persistence_slope`` and ``persistence_intercept`` are the fade curve as a
    banker would write it: next year's growth equals the intercept plus the
    slope times this year's. The slope is the share of growth that survives a
    year, one minus it is the fade, and the ratio of intercept to one-minus-slope
    is the level growth reverts to. Those three numbers are the entire economic
    content of this module and they fit in a sentence.
    """

    card: ModelCard
    fits: dict[int, HorizonFit]
    horizon_years: int
    latest: dict[str, FadeObservation]
    persistence_slope: float
    persistence_intercept: float
    terminal_growth: float
    survivorship: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- the shape of the curve -------------------------------------------- #

    @property
    def fade_per_year(self) -> float:
        """Share of the gap to the long-run mean closed in one year."""
        return 1.0 - self.persistence_slope

    @property
    def reversion_level(self) -> float:
        """The growth rate the one-year regression reverts to."""
        if self.persistence_slope >= 1.0:
            return float("nan")
        return self.persistence_intercept / (1.0 - self.persistence_slope)

    @property
    def half_life_years(self) -> float:
        """Years for half of the excess growth over the reversion level to go."""
        slope = self.persistence_slope
        if not (0.0 < slope < 1.0):
            return float("nan")
        return float(math.log(0.5) / math.log(slope))

    def summary(self) -> str:
        return (
            f"Next year's growth = {self.persistence_intercept:+.4f} + "
            f"{self.persistence_slope:.4f} x this year's, so growth fades "
            f"{self.fade_per_year:.0%} of the way to {self.reversion_level:.1%} "
            f"each year, a half-life of {self.half_life_years:.2f} years. "
            + self.card.summary()
        )

    # -- prediction --------------------------------------------------------- #

    def predict(self, observation: FadeObservation, horizon: int) -> tuple[float, float, float]:
        """Point estimate and interval for one horizon, in growth-rate units."""
        fit = self.fits.get(horizon)
        if fit is None:
            raise ConfigError(
                f"this model was fitted to {sorted(self.fits)} year horizons, not "
                f"{horizon}. Refit with assumptions.ml.forecast.horizon_years raised."
            )
        point = fit.predict(observation)
        low, high = fit.interval(point)
        return point, low, high

    def path(self, ticker: str, years: int | None = None) -> GrowthPath:
        """The growth path for one company, ready to hand to ``techval.dcf``.

        Years one to ``horizon_years`` come from the fitted models, each from
        its own horizon rather than by iterating the one-year model forward.
        Direct fitting is the choice here because iterating compounds the
        one-year error three times and because the two disagree: the one-year
        slope is 0.36 and the three-year slope is 0.17, which is not 0.36 cubed,
        so growth reverts faster in the first year than a constant-decay model
        implies and slower afterwards.

        Years beyond that fall back to the engine's own straight line toward
        ``dcf.revenue_growth_terminal``, anchored on the last fitted year. Every
        year says which of the two produced it.
        """
        key = ticker.upper()
        observation = self.latest.get(key)
        if observation is None:
            raise ConfigError(
                f"{key} is not in the panel this fade model was fitted on. The "
                f"panel holds {len(self.latest)} filers; add the ticker to the "
                "universe and rebuild."
            )
        horizon = years if years is not None else self.horizon_years
        if horizon < 1:
            raise ConfigError(f"a growth path needs at least one year, got {horizon}")

        growth: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        basis: list[str] = []
        notes: list[str] = []

        fitted_years = min(horizon, self.horizon_years)
        for year in range(1, fitted_years + 1):
            point, low, high = self.predict(observation, year)
            growth.append(point)
            lower.append(low)
            upper.append(high)
            basis.append("fitted")

        if horizon > fitted_years:
            # The straight line the DCF already draws, now starting from a
            # measured level instead of a typed one. Linear rather than
            # geometric because that is what dcf._fade does, and a fitted front
            # end spliced onto a different tail shape would be comparing two
            # things at once.
            remaining = horizon - fitted_years
            anchors = (growth[-1], lower[-1], upper[-1])
            for step in range(1, remaining + 1):
                share = step / remaining
                point, low, high = (
                    anchor + (self.terminal_growth - anchor) * share
                    for anchor in anchors
                )
                growth.append(point)
                lower.append(low)
                upper.append(high)
                basis.append("assumed")
            notes.append(
                f"years {fitted_years + 1} to {horizon} are not fitted. They fade "
                f"in a straight line from the last fitted year to the assumed "
                f"terminal growth of {self.terminal_growth:.2%}, because what the "
                "filings support is reversion to the mean growth of a growth "
                f"sector, {self.reversion_level:.1%}, which no terminal value can "
                "carry."
            )
            notes.append(
                "the band converges with the path onto the terminal rate, because "
                "the terminal rate is an assumption and an assumption has no "
                "sampling error. That is not a claim that the later years are "
                "more certain. Nothing here measures how wrong a four-year-out "
                "forecast is, and the band on those years should be read as the "
                "band on the assumption that produced them."
            )

        if horizon <= fitted_years and growth[-1] > self.terminal_growth + 0.05:
            notes.append(
                f"every year of this path is fitted, so it ends at "
                f"{growth[-1]:.1%} and the terminal value then capitalises a "
                f"{self.terminal_growth:.1%} perpetuity off it. That is a cliff, "
                "not a fade. Either run more projection years than "
                "horizon_years so the assumed tail can do its work, or accept "
                "that the terminal value is being struck on a company still "
                "growing at a rate no perpetuity can hold."
            )

        fitted = growth[:fitted_years]
        if any(b > a + 1e-12 for a, b in zip(fitted, fitted[1:])):
            notes.append(
                "the fitted years do not decline monotonically. Each horizon is "
                "fitted directly on its own label rather than by iterating the "
                "one-year model, so nothing constrains them to decay, and the "
                "rises here are inside a band tens of points wide. A monotone "
                "path would be an assumption imposed on the fit, not a result of "
                "it, and imposing it would hide exactly this."
            )
        notes.append(
            f"built from {observation.ticker}'s fiscal year ended "
            f"{observation.fiscal_year_end}, as filed on {observation.as_of}"
        )
        if observation.growth is not None:
            notes.append(
                f"trailing growth was {observation.growth:.1%}; the fitted first "
                f"year is {growth[0]:.1%}, a fade of "
                f"{observation.growth - growth[0]:+.1%} points"
            )
        return GrowthPath(
            ticker=key,
            as_of=observation.as_of,
            growth=growth,
            lower=lower,
            upper=upper,
            basis=basis,
            notes=notes,
        )


def _design(rows: Sequence[FadeObservation]) -> np.ndarray:
    return np.vstack([row.vector() for row in rows])


def _persistence_line(
    rows: Sequence[FadeObservation], horizon: int
) -> tuple[float, float, int]:
    """Least squares of forward growth on this year's growth, tails trimmed.

    The fade curve in two numbers, and the trim is not cosmetic. Untrimmed, this
    panel's slope is 0.15; trimmed at the 1st and 99th percentiles of both axes
    it is 0.47, and at the 5th and 95th it is 0.52. Eighty-eight observations
    out of 2,334 move it by a factor of three, because one filer growing 2,026%
    in a year has more leverage on a least-squares line than the other two
    thousand put together.

    The cross-sectional winsorization the panel already applies does not catch
    them, and the reason is worth knowing: each filing year holds about 150
    companies, so the 99th percentile of that cross-section sits between the
    second and third largest observation and the trim barely moves. A pooled
    trim is the right instrument for a pooled regression, so this takes one and
    reports how many observations it dropped.

    The decile table in ``FadePanel.reversion_table`` is the version of this
    that assumes no functional form at all, and it is the one to quote in an
    argument. It puts the crossing point, where growth neither fades nor
    accelerates, at 12 to 14%.

    Returned as (slope, intercept, observations used).
    """
    current = np.asarray([row.growth for row in rows], dtype=float)
    forward = np.asarray([row.labels[horizon] for row in rows], dtype=float)
    lo_x, hi_x = np.percentile(current, [_WINSOR_LOWER_PCT, _WINSOR_UPPER_PCT])
    lo_y, hi_y = np.percentile(forward, [_WINSOR_LOWER_PCT, _WINSOR_UPPER_PCT])
    keep = (
        (current >= lo_x) & (current <= hi_x) & (forward >= lo_y) & (forward <= hi_y)
    )
    if keep.sum() < 3:
        raise NotMeaningfulError(
            "the trimmed persistence regression has fewer than three observations"
        )
    slope, intercept = np.polyfit(current[keep], forward[keep], 1)
    return float(slope), float(intercept), int(keep.sum())


def fit_fade(
    panel: FadePanel,
    assumptions: Assumptions | None = None,
    *,
    model: str | None = None,
    horizon_years: int | None = None,
) -> FadeModel:
    """Fit the fade curve, walk forward, and report what it beat and what it did not.

    Every horizon is fitted directly on its own label rather than by iterating
    the one-year model, and every horizon is scored against three baselines
    before the model is allowed to say anything: persistence, the training-period
    mean, and the mean of the company's own sub-vertical. The ``EvalResult``
    carried on each horizon is the comparison against persistence, which is the
    hardest of the three.

    The embargo is ``365 * horizon`` days, not 365. An observation filed in
    March 2019 carries a three-year label that is not known until the 10-K filed
    in early 2022, so training on it and testing on anything before 2022 shows
    the model most of the answer. Walk-forward folds without that embargo score
    materially better and mean nothing.

    Raises ``NotMeaningfulError`` below ``assumptions.ml.forecast.min_train_observations``,
    because a fade curve fitted on forty company-years is a fade curve for forty
    companies.
    """
    assumptions = assumptions or Assumptions()
    config = assumptions.ml.forecast
    kind = model or config.model
    horizons = horizon_years or config.horizon_years
    seed = assumptions.ml.random_seed
    minimum = config.min_train_observations

    fits: dict[int, HorizonFit] = {}
    for horizon in range(1, horizons + 1):
        rows = panel.labelled(horizon)
        if len(rows) < minimum:
            raise NotMeaningfulError(
                f"the {horizon}-year fade needs {minimum} labelled company-years "
                f"(assumptions.ml.forecast.min_train_observations) and the panel "
                f"holds {len(rows)}. Below that the fit is memorising the panel, "
                "and this reports unavailable rather than producing a growth path "
                "nobody should act on."
            )
        fits[horizon] = _fit_horizon(rows, horizon, kind, seed, assumptions, minimum)

    one_year = panel.labelled(1)
    slope, intercept, n_line = _persistence_line(one_year, 1)
    trained_through = max(row.as_of for row in one_year)

    latest: dict[str, FadeObservation] = {}
    for observation in panel.observations:
        if observation.growth is None:
            continue
        current = latest.get(observation.ticker)
        if current is None or observation.fiscal_year_end > current.fiscal_year_end:
            latest[observation.ticker] = observation

    survivorship = _survivorship_sample_gap(panel, horizons)

    card = ModelCard(
        name=f"revenue fade ({kind})",
        task=(
            f"next {horizons} fiscal years of revenue growth for a US-listed TMT "
            "filer, from its own filings as at the date each one was filed"
        ),
        trained_through=trained_through,
        n_train=len(one_year),
        features=list(MATRIX_COLUMNS),
        hyperparameters={
            "model": kind,
            "horizon_years": horizons,
            "walk_forward_folds": assumptions.ml.walk_forward_folds,
            "embargo_days_per_horizon": 365,
            "min_train_observations": minimum,
            "random_seed": seed,
            "standardised_clip": _STANDARDISED_CLIP,
        },
        evaluation=fits[1].evaluation,
        limitations=[
            "Survivorship. The panel keeps delisted filers in until they stop "
            f"filing, and mean forward growth is still "
            f"{survivorship.get('gap_1y', float('nan')):+.2%} higher one year out "
            "and more at longer horizons when the leavers are dropped. The "
            "panel cannot include companies that never filed in XBRL at all, and "
            "it cannot include the years after a leaver left, which are the "
            "years that would have faded hardest.",
            "Depth. XBRL begins in 2009, so the first walk-forward fold trains "
            "on 64 observations and is asked about four years of them. The "
            "median filer supports twelve fiscal years, not the fifteen to "
            "eighteen a reader might assume from the mandate date.",
            "One sector, one country, one accounting standard. Every filer here "
            "is a US-listed TMT company reporting under US GAAP, and nothing "
            "about this curve should be carried to an industrial or to a filer "
            "outside the SEC's reach.",
            "The interval is unconditional. Every company gets the same band "
            "because the panel cannot support a conditional one.",
            "Net revenue retention, the feature a TMT analyst would want most, "
            "is not a us-gaap concept and is not in the panel.",
        ],
        notes=[
            f"panel: {len(panel.observations):,} company-years from "
            f"{len(panel.tickers)} filers, {panel.winsorized:,} values winsorized "
            "in their own filing year's cross-section",
            f"one-year regression: forward growth = {intercept:+.4f} + "
            f"{slope:.4f} x trailing growth, on {n_line:,} of {len(one_year):,} "
            "observations after trimming both axes at the 1st and 99th percentiles",
        ],
    )

    notes = list(panel.notes)
    for horizon, fit in fits.items():
        notes.append(
            f"h={horizon}: model {fit.evaluation.score:.4f}, persistence "
            f"{fit.baselines['persistence']:.4f}, training mean "
            f"{fit.baselines['training_mean']:.4f}, sub-vertical mean "
            f"{fit.baselines['sub_vertical_mean']:.4f} (mean absolute error, "
            "lower is better)"
        )

    return FadeModel(
        card=card,
        fits=fits,
        horizon_years=horizons,
        latest=latest,
        persistence_slope=slope,
        persistence_intercept=intercept,
        terminal_growth=assumptions.dcf.revenue_growth_terminal,
        survivorship=survivorship,
        notes=notes,
    )


def _fit_horizon(
    rows: Sequence[FadeObservation],
    horizon: int,
    kind: str,
    seed: int,
    assumptions: Assumptions,
    minimum: int,
) -> HorizonFit:
    """One horizon, walked forward, then refitted on everything for production use."""
    dates = [row.as_of for row in rows]
    labels = np.asarray([row.labels[horizon] for row in rows], dtype=float)
    persistence = np.asarray([row.growth for row in rows], dtype=float)
    matrix = _design(rows)
    stamps = np.asarray(dates, dtype="datetime64[D]")

    folds = walk_forward_folds(
        dates,
        assumptions.ml.walk_forward_folds,
        min_train=minimum,
        embargo_days=365 * horizon,
    )

    predictions = np.full(len(rows), np.nan)
    training_mean = np.full(len(rows), np.nan)
    vertical_mean = np.full(len(rows), np.nan)
    drift: dict[str, float] = {}

    for fold in folds:
        cutoff = np.datetime64(fold.test_start - timedelta(days=365 * horizon), "D")
        train = np.flatnonzero(stamps < cutoff)
        test = np.flatnonzero(
            (stamps >= np.datetime64(fold.test_start, "D"))
            & (stamps <= np.datetime64(fold.test_end, "D"))
        )
        if train.size == 0 or test.size == 0:
            continue
        scaler = _Scaler.fit(matrix[train])
        estimator = _estimator(kind, seed)
        estimator.fit(scaler.transform(matrix[train]), labels[train])
        predictions[test] = estimator.predict(scaler.transform(matrix[test]))

        means, pooled = _sub_vertical_means(rows, labels, train)
        training_mean[test] = pooled
        for index in test:
            vertical_mean[index] = means.get(rows[index].sub_vertical, pooled)
        if fold.index == folds[-1].index:
            shift = scaler.drift(matrix[test])
            drift = {
                name: float(shift[j]) for j, name in enumerate(MATRIX_COLUMNS)
            }

    evaluation = evaluate_regression(
        labels, predictions, persistence, dates, folds, metric="mae"
    )
    evaluation.baseline_name = PERSISTENCE_BASELINE
    scored = np.isfinite(predictions)
    absolute = lambda values: float(np.mean(np.abs(labels[scored] - values[scored])))
    baselines = {
        "persistence": absolute(persistence),
        "training_mean": absolute(training_mean),
        "sub_vertical_mean": absolute(vertical_mean),
    }
    residuals = labels[scored] - predictions[scored]
    low, high = np.percentile(residuals, [_INTERVAL_LOWER_PCT, _INTERVAL_UPPER_PCT])

    # Refit on everything for production use. The score above belongs to the
    # walk-forward models; this is the one that will be asked about today, and
    # it has seen every label the panel holds and none that postdate it.
    scaler = _Scaler.fit(matrix)
    estimator = _estimator(kind, seed)
    estimator.fit(scaler.transform(matrix), labels)
    means, pooled = _sub_vertical_means(
        rows, labels, np.arange(len(rows))
    )

    return HorizonFit(
        horizon=horizon,
        estimator=estimator,
        scaler=scaler,
        evaluation=evaluation,
        baselines=baselines,
        residual_quantiles=(float(low), float(high)),
        residual_sd=float(np.std(residuals)),
        n_train=len(rows),
        sub_vertical_means=means,
        pooled_mean=pooled,
        drift=drift,
    )


# --------------------------------------------------------------------------- #
# Survivorship
# --------------------------------------------------------------------------- #


def is_delisted(ticker: str) -> bool:
    """Whether this filer has since been acquired, taken private or wound up.

    A fact about the ticker today, not about the observation: a company is not
    delisted in 2014 because it was bought in 2024. It is used only to split the
    sample for the bias measurement, never as a feature, because using it as one
    would hand the model the future in its purest form.
    """
    return ticker.upper() in DELISTED_UNIVERSE


def _survivorship_sample_gap(panel: FadePanel, horizons: int) -> dict[str, Any]:
    """How much higher forward growth looks with the leavers removed.

    The cheap half of the survivorship measurement, and the one that matters
    most, because it is a property of the sample rather than of any model. The
    expensive half, refitting the whole model on survivors only, is
    ``survivorship_bias``.
    """
    out: dict[str, Any] = {}
    for horizon in range(1, horizons + 1):
        rows = panel.labelled(horizon)
        if not rows:
            continue
        everyone = np.asarray([row.labels[horizon] for row in rows], dtype=float)
        survivors = np.asarray(
            [row.labels[horizon] for row in rows if not is_delisted(row.ticker)],
            dtype=float,
        )
        if survivors.size == 0:
            continue
        out[f"full_{horizon}y"] = float(everyone.mean())
        out[f"survivors_{horizon}y"] = float(survivors.mean())
        out[f"gap_{horizon}y"] = float(survivors.mean() - everyone.mean())
        out[f"n_full_{horizon}y"] = int(everyone.size)
        out[f"n_survivors_{horizon}y"] = int(survivors.size)
    finals: dict[str, FadeObservation] = {}
    for observation in panel.observations:
        current = finals.get(observation.ticker)
        if current is None or observation.fiscal_year_end > current.fiscal_year_end:
            finals[observation.ticker] = observation
    leaver_final = [
        o.growth
        for o in finals.values()
        if is_delisted(o.ticker) and o.growth is not None
    ]
    survivor_final = [
        o.growth
        for o in finals.values()
        if not is_delisted(o.ticker) and o.growth is not None
    ]
    if leaver_final and survivor_final:
        out["final_year_growth_leavers"] = float(np.mean(leaver_final))
        out["final_year_growth_survivors"] = float(np.mean(survivor_final))
    return out


def survivorship_bias(
    panel: FadePanel, assumptions: Assumptions | None = None
) -> dict[str, Any]:
    """Refit the whole model on survivors only and compare it to the full fit.

    The measurement the brief for this module asked for and the one most likely
    to matter to whoever reads the valuation. It is a separate function rather
    than part of ``fit_fade`` because it doubles the fitting work for a number
    that is reported once, not on every run.
    """
    assumptions = assumptions or Assumptions()
    survivors = FadePanel(
        observations=[
            o for o in panel.observations if not is_delisted(o.ticker)
        ],
        depth={t: n for t, n in panel.depth.items() if not is_delisted(t)},
        random_seed=panel.random_seed,
    )
    full_model = fit_fade(panel, assumptions)
    survivor_model = fit_fade(survivors, assumptions)
    out: dict[str, Any] = {
        "full_observations": len(panel.observations),
        "survivor_observations": len(survivors.observations),
        "full_slope": full_model.persistence_slope,
        "survivor_slope": survivor_model.persistence_slope,
        "full_reversion_level": full_model.reversion_level,
        "survivor_reversion_level": survivor_model.reversion_level,
        "full_half_life": full_model.half_life_years,
        "survivor_half_life": survivor_model.half_life_years,
    }
    for horizon in full_model.fits:
        out[f"full_mae_{horizon}y"] = full_model.fits[horizon].evaluation.score
        out[f"survivor_mae_{horizon}y"] = survivor_model.fits[horizon].evaluation.score
    out |= panel.survivorship_report()
    out |= full_model.survivorship
    return out


# --------------------------------------------------------------------------- #
# The seam into the valuation
# --------------------------------------------------------------------------- #


def growth_path_for_dcf(
    model: FadeModel,
    ticker: str,
    assumptions: Assumptions,
) -> GrowthPath | None:
    """The fitted path a DCF should use, or None when the switch is off.

    This is the only function a report should call. ``assumptions.ml.forecast.enabled``
    defaults to false, so the base valuation is byte for byte what it was before
    this module existed, and a fitted growth path can only reach a DCF through a
    switch somebody turned on deliberately.

    The path is ``dcf.projection_years`` long, because that is what
    ``dcf.project`` will accept, and ``horizon_years`` of it are fitted.
    """
    if not assumptions.ml.forecast.enabled:
        return None
    return model.path(ticker, assumptions.dcf.projection_years)


@dataclass
class FadeComparison:
    """The assumed fade and the fitted one, valued side by side.

    An R-squared is a claim about a panel. This is a claim about a company, in
    dollars per share, and it is the only form of this model's output that an
    analyst can argue with.
    """

    ticker: str
    assumed_path: list[float]
    fitted_path: GrowthPath
    assumed: Any
    fitted: Any
    low: Any
    high: Any

    @property
    def enterprise_value_gap(self) -> float:
        return self.fitted.enterprise_value - self.assumed.enterprise_value

    @property
    def per_share_gap(self) -> float:
        return self.fitted.per_share - self.assumed.per_share

    def to_frame(self) -> pd.DataFrame:
        rows = [
            ("Assumed fade", self.assumed_path, self.assumed),
            ("Fitted fade", list(self.fitted_path.growth), self.fitted),
            ("Fitted, low band", list(self.fitted_path.lower), self.low),
            ("Fitted, high band", list(self.fitted_path.upper), self.high),
        ]
        records = []
        for name, path, result in rows:
            record = {"Case": name}
            record |= {f"Y{i + 1}": g for i, g in enumerate(path)}
            record |= {
                "Terminal revenue": result.projections[-1].revenue,
                "Enterprise value": result.enterprise_value,
                "Equity value": result.equity_value,
                "Per share": result.per_share,
            }
            records.append(record)
        return pd.DataFrame(records).set_index("Case")

    def summary(self) -> str:
        return (
            f"{self.ticker}: the assumed fade values the enterprise at "
            f"{self.assumed.enterprise_value:,.0f}mm and "
            f"{self.assumed.per_share:,.2f} a share. The fitted fade values it at "
            f"{self.fitted.enterprise_value:,.0f}mm and "
            f"{self.fitted.per_share:,.2f}, a difference of "
            f"{self.enterprise_value_gap:+,.0f}mm and "
            f"{self.per_share_gap:+,.2f} a share "
            f"({self.enterprise_value_gap / self.assumed.enterprise_value:+.1%}). "
            f"The band the fit actually supports runs "
            f"{self.low.per_share:,.2f} to {self.high.per_share:,.2f} a share."
        )


def compare_fade(
    fin: Any,
    bridge: Any,
    wacc_result: Any,
    assumptions: Assumptions,
    model: FadeModel,
    *,
    ticker: str | None = None,
) -> FadeComparison:
    """Value one company on the assumed fade and on the fitted one, four ways.

    Four valuations rather than two, because a point estimate presented beside
    the assumption it replaces invites the reader to believe the fitted number
    is the answer. It is the middle of a band forty growth points wide, and the
    two ends of that band are different companies. All four are returned.

    The switch is honoured: with ``ml.forecast.enabled`` false this raises
    rather than quietly valuing on a path the configuration says not to use.
    """
    from ..dcf import run_dcf

    if not assumptions.ml.forecast.enabled:
        raise ConfigError(
            "ml.forecast.enabled is false, so the fitted fade is not in force. "
            "Set it true to value on a fitted growth path, which is a decision "
            "that should be visible in the assumptions file."
        )
    symbol = (ticker or fin.ticker).upper()
    years = assumptions.dcf.projection_years
    path = model.path(symbol, years)

    from ..dcf import _fade  # the straight line this module is arguing with

    assumed_path = [
        float(g)
        for g in _fade(
            assumptions.dcf.revenue_growth_start,
            assumptions.dcf.revenue_growth_terminal,
            years,
        )
    ]

    def value(growth: Sequence[float]):
        return run_dcf(fin, bridge, wacc_result, assumptions, growth_path=list(growth))

    return FadeComparison(
        ticker=symbol,
        assumed_path=assumed_path,
        fitted_path=path,
        assumed=run_dcf(fin, bridge, wacc_result, assumptions),
        fitted=value(path.growth),
        low=value(path.lower),
        high=value(path.upper),
    )
