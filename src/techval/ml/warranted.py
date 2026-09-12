"""The multiple a company's characteristics warrant, and the gap to where it trades.

A comp table answers "what do similar companies trade at". This module answers
the sharper question an analyst actually argues about: **what does the market pay
for this bundle of characteristics, and does this company sit above or below that
line.** The observed multiple is regressed on fundamentals across the TMT universe
and across time; the fitted value is the warranted multiple and the residual is
the rich-or-cheap signal. ``comps.fit_warranted_multiple`` already does this on a
single comp set of eight names on one date. This does it on two thousand
company-quarters across five years, which changes what can be asked of it and
introduces four traps that a ten-name cross-section never meets.

**Reflexivity, first, because it decides what the output means.** The multiple is
the market's opinion. A model fitted on multiples learns the market's own pricing
rule, so the most it can ever say is that a company is priced unlike its
characteristics suggest the market prices characteristics. It cannot say the
market is wrong. If the whole sector is mispriced the model is fitted on the
mispricing and reports everything as fair. That is the difference between this and
a discounted cash flow: the DCF has an outside anchor in the cash and the
discount rate, and it can tell you the market is wrong; this cannot, ever. A
reader who confuses the two will read a positive residual as "sell" when all it
says is "the market is paying more for this than it pays for otherwise similar
names, and there may be an excellent reason it is not in the regression".

**The re-rating is real, it is not about the companies, and it is not most of the
variance. It is two percent.** The move itself is exactly as advertised: the
median software name in this panel trades at 17.1x revenue at the end of 2021,
8.7x a year later and 7.7x by the middle of 2026, on fundamentals that did not
move anything like that far, because the discount rate did. A model fitted across
both periods without handling it will load the rate cycle onto whatever company
characteristic happens to correlate with the calendar and call the result a
valuation. ``assumptions.ml.warranted.demean_by_date`` defaults true: each date's
cross-sectional mean of the log multiple is removed before fitting, so the model
only ever learns RELATIVE position inside a date.

What measurement does NOT support is the size of it. Pooled across the whole TMT
universe the calendar is 2.1% of the variance of the log multiple, because the
dispersion between sub-verticals swamps the dispersion through time: an IT
services name at 1.9x and an infrastructure software name at 13.2x are four times
further apart than either of them moved across the entire rate cycle. Inside a
single sub-vertical, which is where the claim actually lives, the share rises to
between 9% and 18%: application software 16.5%, infrastructure software 13.5%,
semiconductors 16.1%, telecom 10.8%. Still a minority, and still worth removing,
because it is a real effect that has nothing to do with any company.

So demeaning stays on by default, for the right reason rather than the expected
one. ``Rerating`` reports the decomposition, and the fit runs both ways, so the
claim can be checked rather than repeated.

The date mean is taken **leave-one-out**. It is a contemporaneous number, so it
is not lookahead, but a mean that includes the company being predicted hands back
1/N of its own answer, and at N near a hundred that is a small leak in a residual
that is itself small. ``(sum - y_i) / (n - 1)`` costs one line and removes it.

**Fit in logs, and say who is lost to the log.** Multiples are right-skewed: a
cross-section running 2x to 25x has a mean well above its median and a squared
error dominated by the top decile. Logs fix the skew and make the residual
readable as a percentage, since a residual of +0.30 in logs is "roughly 35% above
the line" at any level of the multiple. The cost is that the log needs a positive
argument, and that is a selection decision rather than a technicality. On
EV/Revenue almost nothing is lost, because revenue is positive for every filer
here and enterprise value is negative only for a company whose cash exceeds its
market capitalisation. On EV/EBITDA the loss is severe and structural: the
unprofitable filers are exactly the high-growth names the multiple is most often
quoted for, ``NotMeaningfulError`` is the engine's answer to a negative
denominator, and dropping them quietly restricts the model to profitable
companies and then reports a result as though it were about all of them.
``ObservationPanel.selection`` counts and names the losses so the restriction is
visible rather than assumed.

**A residual is only a residual out of sample.** An in-sample residual is partly
the model fitting that very company: with enough parameters every company is fair
value by construction. Every residual this module reports comes from a
walk-forward fold that did not train on it, including the ones in
``WarrantedModel.extremes``, which is the table someone would actually trade off.

**Persistence is the best predictor in the table and the worst warranted
multiple, and the distinction is not a dodge.** Carrying a company's own multiple
forward a quarter beats everything here at predicting next quarter's multiple,
because a multiple is close to a random walk: on this panel it scores a rank
correlation of 0.97 against the fitted model's 0.84. It is reported for exactly
that reason, since a reader deserves to know what the ceiling is. It is not what
the model card is scored against, because it is built from the company's own
price. A warranted multiple exists to be differenced against that price, so a
construction that reproduces it has a residual of zero by definition and no
signal whatever. The card is scored against the strongest baseline that answers
the SAME question: what do this company's characteristics, and its peers, say it
should trade at. Both numbers are in ``WarrantedModel.baselines``.

**And the number that matters most is neither of those.** A pooled rank
correlation over a panel of the same hundred companies quarter after quarter is
mostly the companies. A feature vector barely moves in three months and neither
does a relative multiple, so a model fitted on the past is rewarded for
recognising a name as much as for understanding it. That is not lookahead, the
training window is strictly earlier, and it is not fraud, but a reader shown 0.84
will believe far more than the model earned.
``WarrantedModel.change_rank_correlation`` differences both sides against the
company's own previous observation and asks whether the model predicted the
CHANGE in relative position. On this panel the network scores +0.04 and the ridge
+0.02. That is the honest size of what is being added on top of knowing which
company it is, and ``verdict()`` prints it beside the headline so the two cannot
be separated.

The consequence for how the output should be read: the residual is a description
of where a company sits relative to how the market prices this bundle of
characteristics, and it is not a forecast that the gap will close. Testing
whether it closes means regressing the residual on forward returns with the full
horizon as an embargo, which is what ``assumptions.ml.signals`` exists for and is
a different piece of work from this one.

What follows was not in the brief. All of it was found by measurement on the
real panel, and all of it would have been fatal left alone.

**A market-capitalisation feature is the answer written on the other side of the
page.** The target is log(EV) - log(revenue). Hand the model ``scale_log_market_cap``
or ``scale_log_enterprise_value`` and it will discover that log EV predicts log EV,
report an R-squared near 0.95, and mean nothing whatever. The same applies with
one step of laundering to every ratio with price in the denominator:
``capital_net_debt_to_ev``, ``capital_net_debt_to_market_cap`` and
``capital_cash_to_market_cap`` all move mechanically with the multiple. They are
named in ``BANNED_FEATURES`` with the reason, and ``fit_warranted`` raises rather
than accepting one. The size term the model does get is ``scale_log_revenue``,
which is on the denominator side and safe.

The price momentum features are excluded on a weaker argument, and the other camp
deserves stating. Twelve-month momentum genuinely predicts next quarter's
multiple, because a stock that has doubled is usually on a higher multiple than
it was. A model including it would score better. But the question a warranted
multiple exists to answer is what the FUNDAMENTALS support, and a fitted value
built partly on the price is no longer an independent read on the price. The
residual would stop being a rich-or-cheap signal and become a momentum residual.
Excluded, and a reader who wants the other answer should say so out loud rather
than find it inside this one.

**The features come from ``ml.features`` alone, and not from ``tmt.metrics``,
because the two modules do not speak the same language.** ``tmt.metrics.build_metrics``
routes on a sub-vertical string and refuses an unknown one outright, which is the
right call: a semiconductor company scored on the Rule of 40 is a worse answer
than no answer. But its vocabulary is ``software, saas, internet, media,
streaming, entertainment, telecom, wireless, cable, fiber, towers``, while
``taxonomy.SubVertical`` speaks in ``infrastructure_software,
application_software, internet, semiconductors, hardware, it_services, payments,
media_entertainment, telecom, towers_fiber, gaming``. Only ``internet`` and
``telecom`` appear in both. Handed the other nine values of the enum this engine
actually classifies with, ``build_metrics`` raises ``ConfigError``, so it cannot
be run across this universe at all. Reconciling the two vocabularies is a design
decision about which pack a bucket belongs to, and it is the repository owner's
to make rather than something to guess at inside a model. The metrics a metric
pack would have contributed here, the margins and the Rule of 40, are in the
feature store anyway and are taken from there.

**Split-adjusted prices do not pair with point-in-time share counts.** This one
is not a modelling choice, it is a data defect that silently multiplies an
enterprise value by ten. A price vendor restates its whole history for every
split, so Nvidia's close for 31 March 2022 comes back as 27.29 rather than the
272.86 that actually printed. The share count from a fact set pinned to that date
is the 2,535mm the company reported at the time, on the pre-split basis, because
the split had not happened. Multiplying the two gives an equity value a tenth of
the truth, and EV/Revenue of 2.5x where the real figure is 25x. The correction is
a pure unit conversion, and the honest way to get it is from the filings
themselves: the ratio between a period's diluted share count as reported TODAY
and the same period's count as reported at the row date is exactly the cumulative
split factor between the two, whatever the ex-dates were. ``share_basis_factor``
computes it that way and snaps the result to a product of detected splits, so a
one-percent restatement is not mistaken for a corporate action.

A threshold on filing dates is NOT good enough here and the tempting version of
this fix is wrong. Palo Alto split three-for-one in September 2022, but the first
filing that restated a comparative by three is dated May 2023, so any rule keyed
on the filing date treats the whole of that intervening period as pre-split and
scales it by three when the price feed had already adjusted. Reading the ratio off
the two fact sets has no such window, because the fact set says what basis it is
on rather than when someone found out.

This correction is applied here, where the panel is built, and it is deliberately
not pushed into ``ev_bridge``. It belongs there: every point-in-time enterprise
value in the engine has the same defect, and ``backtest.py`` values a splitter at
a tenth or a fortieth of its real size on any date before the split. Moving it
is a change to the signature of a function the whole engine calls, and that is
the repository owner's decision rather than this module's.

**A universe this wide breaks ladders a software comp set never tested.** The
rest of this engine is exercised on US software filers, where the tag ladders are
right. Run across semiconductors, carriers, agencies and tower REITs and three
separate resolutions come back confidently wrong, each silently, each big enough
to ruin a fit. They are checked before the division, in ``revenue_is_a_component``,
``debt_is_outside_the_ladder`` and ``price_disagrees_with_the_public_float``, and
each one refuses the row and says what it found rather than correcting it, because
every correction belongs in the resolution path where it would repair the DCF and
the comp tables too, and a model is the wrong place to make that change.

    A lessor tags only its services revenue under ASC 606, so Crown Castle
    resolves 210mm against the 6,568mm in ``Revenues`` and the towers bucket
    reads as 125x revenue.

    A filer reporting ``LongTermDebtAndCapitalLeaseObligations`` matches no debt
    ladder, so Lumen resolves zero straight debt against roughly twenty billion
    and prints an enterprise value of 996mm. Verizon is the same failure wearing
    a resolved number: its current portion is inside a ladder and its 143,448mm
    of long-term debt is not, so the check compares what resolved against what
    the filer reports rather than against zero.

    A price vendor returns the wrong security, so Booking's series runs 71 to 174
    where the share traded near 1,750 to 5,500, and every committed read of its
    EV/Revenue prints below a single turn. The filer's own
    ``dei:EntityPublicFloat`` is the only independent anchor free data offers,
    and on that series the 2025-03-31 price implies an equity value of 6,277mm
    against a public float of 133,100mm reported as of 2024-06-30.

The cost of refusing rather than repairing is stated rather than absorbed: it
deletes the towers and fibre sub-vertical from the panel entirely, along with
Lumen, DXC, Charter and Booking. ``ObservationPanel.selection`` carries the count
under each category, and any result here is about the sample that survived them.

**What this is fitted on, and what that sample is worth.** Around a hundred US
TMT filers at quarter ends from 2021 to 2026: roughly two thousand
company-quarters, which is a large number of observations over a small number of
genuinely independent events. Quarterly readings of the same company are not
independent draws, so the effective sample is far smaller than the count and no
standard error here is corrected for that.

The names are the ones that still exist, so the survivorship caveat in
methodology §11.3 applies in full, and it bit visibly. Five of the hundred and
eleven names in ``taxonomy.SEED`` are no longer in the SEC's own
``company_tickers.json`` and have no price history left to fetch: EA, FI, FYBR,
IPG and JNPR. They did not fail to resolve on some date, they are absent from
every date, including the dates on which they were live and, in several cases,
cheap enough to be bought. A panel assembled today cannot contain them, and no
statistic in this module is corrected for that. It is the one limitation here
that more care in the code cannot fix; it needs a point-in-time universe file
this repository does not have.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .. import tags
from ..comps import PeerMetrics, fit_warranted_multiple
from ..config import Assumptions
from ..errors import ConfigError, NotMeaningfulError, TechvalError
from ..ev_bridge import build_ev_bridge
from ..financials import build_financials
from ..market import MarketData
from ..tmt.taxonomy import SubVertical
from . import nn
from .evaluation import (
    MIN_OBSERVATIONS,
    Fold,
    evaluate_regression,
    walk_forward_folds,
)
from .features import FEATURE_NAMES, assert_point_in_time, build_features
from .protocol import EvalResult, ModelCard, spearman

__all__ = [
    "WARRANTED_FEATURES",
    "SUB_VERTICAL_COLUMNS",
    "BANNED_FEATURES",
    "DILUTED_SHARES_TAG",
    "Observation",
    "ObservationPanel",
    "Rerating",
    "WarrantedRead",
    "WarrantedModel",
    "share_basis_factor",
    "build_observations",
    "fit_warranted",
]

# The multiples this module will fit, and the ``Financials`` attribute each one
# divides the enterprise value by. Mirrors ``comps.MULTIPLES`` for the three that
# ``assumptions.ml.warranted.target`` accepts.
TARGETS: dict[str, tuple[str, str]] = {
    "ev_revenue": ("revenue", "EV/Revenue"),
    "ev_gross_profit": ("gross_profit", "EV/Gross Profit"),
    "ev_ebitda": ("ebitda", "EV/EBITDA"),
}

# Fundamental features, in a fixed order so the design matrix is reproducible.
# Every one of them is a ratio, a growth rate or a log size on the denominator
# side of the multiple. The brief asks for growth, gross margin, Rule of 40,
# scale, capital intensity and sub-vertical; the rest earn their place by being
# things a software analyst argues about in the same breath.
#
# ``returns_rule_of_40`` in this package is revenue growth plus FREE CASH FLOW
# margin, not the EBITDA-margin version ``comps.PeerMetrics.rule_of_40`` carries,
# and the two are different numbers for a company with heavy capitalised
# software or a large working-capital swing. The feature-store definition is
# used here because the rest of the design matrix comes from the feature store,
# and the choice is stated rather than left for a reader to discover.
WARRANTED_FEATURES: tuple[str, ...] = (
    "growth_revenue_1y",
    "growth_revenue_cagr_2y",
    "margin_gross",
    "margin_ebitda",
    "margin_fcf",
    "returns_rule_of_40",
    "scale_log_revenue",
    "efficiency_capex_to_revenue",
    "efficiency_rnd_to_revenue",
    "efficiency_sbc_to_revenue",
    "capital_debt_to_capital",
    "returns_deferred_revenue_to_revenue",
    "quality_revenue_volatility_3y",
)

# One column per sub-vertical, in enum order so the matrix has the same width on
# every panel. The order is fixed by ``SubVertical`` rather than by what happens
# to be in the sample, so a panel missing gaming still produces a design matrix a
# model fitted elsewhere could read.
SUB_VERTICAL_COLUMNS: tuple[str, ...] = tuple(
    f"is_{vertical.value}" for vertical in SubVertical
)


# Features that cannot appear on the right-hand side, with the reason, because a
# refusal without one invites someone to delete the check. The first five are the
# target itself in different clothes: two are its numerator and three are ratios
# whose denominator is that numerator. The last five are price, and a warranted
# multiple built on the price is not an independent read on the price.
BANNED_FEATURES: dict[str, str] = {
    "scale_log_market_cap": (
        "market capitalisation is the numerator of the multiple being predicted, "
        "so this regresses log EV on log EV"
    ),
    "scale_log_enterprise_value": (
        "enterprise value IS the numerator of the multiple being predicted"
    ),
    "capital_net_debt_to_ev": (
        "enterprise value sits in the denominator of this ratio, so it moves "
        "mechanically with the multiple"
    ),
    "capital_net_debt_to_market_cap": (
        "market capitalisation sits in the denominator of this ratio, so it moves "
        "mechanically with the multiple"
    ),
    "capital_cash_to_market_cap": (
        "market capitalisation sits in the denominator of this ratio, so it moves "
        "mechanically with the multiple"
    ),
    "market_momentum_12m": (
        "a fitted value built on the price is not an independent read on the price; "
        "the residual would become a momentum residual"
    ),
    "market_momentum_6m": "price, as for market_momentum_12m",
    "market_momentum_3m": "price, as for market_momentum_12m",
    "market_relative_momentum_12m": "price, as for market_momentum_12m",
    "market_52w_range_position": "price, as for market_momentum_12m",
}

# --------------------------------------------------------------------------- #
# Three guards against a resolved figure that is not the figure it claims to be.
# Each of them was written because this panel hit it, and each names what it
# found rather than quietly dropping a row.
# --------------------------------------------------------------------------- #

# Another entry in the revenue ladder covering the same window, this many times
# larger than the one that won, means the winner is a COMPONENT rather than the
# total. Three is far outside anything two definitions of revenue could disagree
# by: including or excluding sales tax, or netting agency revenue, moves a figure
# by tens of percent, not by a factor of three.
_COMPONENT_REVENUE_MULTIPLE = 3.0

# Debt concepts the engine's ladders do not carry. A filer reporting under one of
# them resolves to zero straight debt in silence, which is the worst possible
# failure mode for an enterprise value.
_DEBT_TAGS_OUTSIDE_THE_LADDER = (
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
)

# Below this, a gap between what the ladder resolved and what a concept outside
# it reports is a rounding artefact rather than a gap worth refusing a row over.
# USD millions.
_MATERIAL_DEBT = 100.0

# How much larger a concept outside the ladder may be than what the ladder
# resolved before the resolution is not believed. The outside concepts bundle
# capital lease obligations in with the debt and the ladder's do not, so a filer
# that resolved correctly can still show a few percent more under one of them.
# Ten percent is wide enough to cover that and nowhere near wide enough to cover
# a missing balance-sheet line: Verizon resolves its current portion alone and
# the outside concept is six and a half times it.
_DEBT_LADDER_TOLERANCE = 1.10

# How far the price-implied equity value may sit from the filer's own reported
# public float before the price series is not believed. Ten times either way is
# very wide and deliberately so: the float is reported once a year as of a date
# months before the filing, and it excludes insider holdings, so a founder-led
# company that has doubled since can legitimately sit six or seven times above
# it. It catches only gross errors, which is all it is for.
_FLOAT_TOLERANCE = 10.0

# XBRL reports in units and every statement object in this engine is in USD
# millions, so anything read straight off a fact set is divided by this.
_MM = 1e6

# The one tag whose two readings date a split. Diluted rather than basic because
# every filer reports it and ``build_financials`` values the company on it.
DILUTED_SHARES_TAG = "WeightedAverageNumberOfDilutedSharesOutstanding"

# A share count is restated by a percent or two for an ordinary correction and by
# fifty percent or more for a split. Anything inside this band of 1.0 is a
# restatement and the basis has not moved.
_RESTATEMENT_TOLERANCE = 0.02

# How close the observed basis ratio has to sit to a product of the splits this
# filer actually declared before it is accepted as that product.
_BASIS_TOLERANCE = 0.01

# Ridge penalty on the standardised design matrix. Small, because the point of
# the ridge is to survive the near-collinearity of Rule of 40 with its own two
# components rather than to shrink anything meaningfully. Not in Assumptions:
# it is a numerical guard on this fit, not a valuation judgment.
_RIDGE_LAMBDA = 1.0

# The multilayer perceptron. Two hidden layers of 32 units is roughly two
# thousand parameters against a training fold of several hundred to two thousand
# rows, which is already generous; anything wider memorises the panel.
_MLP_HIDDEN = (32, 32)
_MLP_DROPOUT = 0.1
_MLP_EPOCHS = 300
_MLP_PATIENCE = 25
_MLP_BATCH = 64
_MLP_LR = 3e-3

# Share of each training window held out, by date, to stop the network. Taken off
# the END of the training window so the stopping signal is the most recent data,
# which is the same direction the test window lies in.
_VALIDATION_SHARE = 0.2

# A date whose cross-section is thinner than this cannot carry a mean anyone
# should subtract, nor an information coefficient worth plotting. Such dates are
# named in the notes rather than dropped: they are usually the start of the panel,
# where half the universe had not yet filed anything this engine can resolve.
_MIN_NAMES_PER_DATE = 10

# Below this many names a sub-vertical median on one date is two or three
# companies, which is not a median.
_MIN_PEERS_FOR_MEDIAN = 4


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Observation:
    """One company priced on one date, with the features that date knew.

    ``multiple`` is in turns and ``log_multiple`` is its natural log, which is
    what everything downstream fits. ``basis_factor`` is the split unit
    conversion described in the module docstring: 1.0 for a company that has not
    split since the row date, 10.0 for Nvidia anywhere before mid-2024. It is
    kept on the record rather than folded away, because an enterprise value
    silently multiplied by ten is the sort of thing a reader is entitled to
    audit.

    ``features`` holds ``None`` where the feature store could not source a
    figure, following its own convention that missing is a value and not a zero.
    """

    ticker: str
    as_of: date
    sub_vertical: str
    multiple: float
    log_multiple: float
    enterprise_value: float
    equity_value: float
    denominator: float
    statement_date: date | None
    basis_factor: float
    features: dict[str, float | None] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def row(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "sub_vertical": self.sub_vertical,
            "multiple": self.multiple,
            "enterprise_value": self.enterprise_value,
            "denominator": self.denominator,
            "basis_factor": self.basis_factor,
        }


@dataclass(frozen=True)
class Skip:
    """A company-date that could not become an observation, and why.

    Never dropped silently. A panel that quietly shrank from a hundred names to
    sixty is the most common way a cross-sectional result gets quoted with more
    confidence than it earned, and the reasons are not random: they cluster on
    the filers whose tagging is hardest and on the periods before a company
    listed.
    """

    ticker: str
    as_of: date
    reason: str
    category: str


@dataclass
class ObservationPanel:
    """Every observation the universe yielded, and everything it did not.

    ``target`` names the multiple, ``observations`` are the usable rows and
    ``skips`` are the losses with their reasons. ``selection`` summarises the
    losses by category, which is the number that says whether the sample is the
    universe or a profitable corner of it.
    """

    target: str
    observations: list[Observation]
    skips: list[Skip] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    random_seed: int | None = None
    # A memo, not state. The comps.py baseline is a pure function of this panel
    # and of comps.regression, and it costs several thousand least-squares fits,
    # so refitting it once per model rather than once per panel is the difference
    # between a test suite that runs and one nobody waits for. Keyed on the
    # settings it depends on, excluded from equality and repr.
    _comps_memo: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def dates(self) -> list[date]:
        return sorted({o.as_of for o in self.observations})

    @property
    def tickers(self) -> list[str]:
        return sorted({o.ticker for o in self.observations})

    def on(self, when: date) -> list[Observation]:
        return [o for o in self.observations if o.as_of == when]

    def selection(self) -> pd.DataFrame:
        """What was lost, by reason, beside what survived.

        The row that matters on an EV/EBITDA fit is ``not_meaningful``: it is the
        count of unprofitable filers the log threw away, and it is the reason the
        result is about profitable companies rather than about the sector.
        """
        counts: dict[str, int] = {}
        for s in self.skips:
            counts[s.category] = counts.get(s.category, 0) + 1
        rows = [{"outcome": "observed", "n": len(self.observations)}]
        rows += [
            {"outcome": k, "n": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        ]
        return pd.DataFrame(rows)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([o.row() for o in self.observations])
        for name in WARRANTED_FEATURES:
            frame[name] = [
                np.nan if o.features.get(name) is None else float(o.features[name])
                for o in self.observations
            ]
        return frame


@dataclass
class Rerating:
    """How much of the raw dispersion in multiples was the calendar.

    ``between_date_share`` is the share of the total variance of the log multiple
    that sits between date means rather than inside them, and it is the whole of
    trap one in one number. ``calendar_r_squared`` is what a model that knows
    nothing but the date scores against the raw multiple: predict each date's own
    cross-sectional mean and nothing else.

    Quote them together with ``within_date_sd``, because the last one is the
    scale a residual is read against. A residual of 0.25 log points against a
    within-date standard deviation of 0.55 is half a standard deviation, which is
    an argument. Against 0.15 it would be a trade.
    """

    n_observations: int
    n_dates: int
    total_variance: float
    between_date_variance: float
    within_date_variance: float
    calendar_r_squared: float
    date_means: dict[date, float] = field(default_factory=dict)

    @property
    def between_date_share(self) -> float:
        return (
            0.0 if self.total_variance <= 0 else self.between_date_variance / self.total_variance
        )

    @property
    def within_date_sd(self) -> float:
        return float(math.sqrt(max(self.within_date_variance, 0.0)))

    def rows(self) -> list[tuple[str, Any]]:
        return [
            ("Observations", self.n_observations),
            ("Dates", self.n_dates),
            ("Variance of log multiple", self.total_variance),
            ("Between dates", self.between_date_variance),
            ("Within dates", self.within_date_variance),
            ("Share of variance that is the calendar", self.between_date_share),
            ("R-squared of a date-mean-only model", self.calendar_r_squared),
            ("Within-date standard deviation (log)", self.within_date_sd),
        ]

    def sentence(self) -> str:
        return (
            f"{self.between_date_share:.0%} of the variance in the log multiple sits "
            f"between the {self.n_dates} dates rather than inside them: a model given "
            "the date and nothing else scores an R-squared of "
            f"{self.calendar_r_squared:.2f} against the raw multiple. That is the "
            "re-rating, and it is not a statement about any company."
        )


@dataclass(frozen=True)
class WarrantedRead:
    """One company's warranted multiple, its residual, and a sentence about it.

    ``residual_log`` is the actual less the warranted in log points and is the
    number to compare against ``within_date_sd``. ``residual_turns`` is the same
    gap in multiple turns, which is what gets quoted, and it is the larger-looking
    of the two at a high multiple for purely arithmetic reasons.

    ``out_of_sample`` is True when the prediction came from a fold that did not
    train on this observation. It is False only for a read taken on an
    observation inside the initial training window, which is scored nowhere and
    should not be traded on.
    """

    ticker: str
    as_of: date
    sub_vertical: str
    actual_multiple: float
    warranted_multiple: float
    residual_log: float
    residual_turns: float
    within_date_sd: float
    out_of_sample: bool

    @property
    def z(self) -> float:
        """Residual in within-date standard deviations. The comparable number."""
        return 0.0 if self.within_date_sd <= 0 else self.residual_log / self.within_date_sd

    def sentence(self) -> str:
        direction = "above" if self.residual_log > 0 else "below"
        strength = (
            "in line with" if abs(self.z) < 0.5 else "materially " + direction
        )
        caveat = "" if self.out_of_sample else " (IN SAMPLE: not evidence)"
        return (
            f"{self.ticker} trades at {self.actual_multiple:.1f}x against a warranted "
            f"{self.warranted_multiple:.1f}x on its {self.sub_vertical.replace('_', ' ')} "
            f"characteristics at {self.as_of}, {abs(self.residual_turns):.1f} turns "
            f"{direction} the line and {abs(self.z):.1f} within-date standard "
            f"deviations, so it is priced {strength} what the market pays for this "
            f"bundle of fundamentals. That is a statement about relative pricing, "
            f"not about value.{caveat}"
        )


@dataclass
class WarrantedModel:
    """A fitted warranted multiple, its honest score, and the names at the edges.

    ``card`` carries the evaluation against the strongest baseline that answers
    the same question, not the most flattering one. ``baselines`` holds every
    comparison that was run, persistence included, so a reader can see the ones
    the model lost as easily as the ones it won, and the card's notes say which
    baseline was excluded from the card and why.

    ``information_coefficients`` is the per-date rank correlation, which is what
    a screen would have earned had it been run on that date. Read its dispersion
    as well as its mean: a mean of 0.3 built from dates running from minus 0.2 to
    plus 0.7 is not a strategy, it is four good quarters and three bad ones.
    """

    card: ModelCard
    target: str
    demeaned: bool
    rerating: Rerating
    reads: dict[tuple[str, date], WarrantedRead]
    baselines: dict[str, EvalResult]
    folds: list[Fold]
    coefficients: dict[str, float] = field(default_factory=dict)
    information_coefficients: dict[date, float] = field(default_factory=dict)
    # The headline's own caveat, promoted to a field so nothing can quote the
    # score without it. See ``_change_in_relative_position``.
    change_rank_correlation: float | None = None
    n_changes: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def latest(self) -> date:
        return max(when for _, when in self.reads)

    def warranted(self, ticker: str, when: date | None = None) -> WarrantedRead:
        """The warranted multiple and residual for one company, out of sample.

        Defaults to the latest date the panel carries. Raises rather than
        returning an empty read: a company with no observation on that date was
        not valued, and saying so is the answer.
        """
        tk = ticker.upper()
        when = when or self.latest
        read = self.reads.get((tk, when))
        if read is None:
            held = sorted({t for t, d in self.reads if d == when})
            raise NotMeaningfulError(
                f"no out-of-sample warranted multiple for {tk} on {when}. The panel "
                f"holds {len(held)} names on that date"
                + (f", among them {', '.join(held[:6])}" if held else "")
            )
        return read

    def extremes(self, n: int = 10, when: date | None = None) -> pd.DataFrame:
        """The most over and under valued names on one date, richest first.

        Every residual in the frame is out of sample. ``z`` is the residual in
        within-date standard deviations and is the column to sort a screen on,
        because turns are not comparable between a 2x telecom and a 20x security
        name.
        """
        when = when or self.latest
        reads = [r for (_, d), r in self.reads.items() if d == when]
        if not reads:
            raise NotMeaningfulError(f"the panel holds no observations on {when}")
        reads.sort(key=lambda r: r.residual_log, reverse=True)
        chosen = reads[:n] + reads[-n:] if len(reads) > 2 * n else reads
        return pd.DataFrame(
            [
                {
                    "ticker": r.ticker,
                    "sub_vertical": r.sub_vertical,
                    "actual": r.actual_multiple,
                    "warranted": r.warranted_multiple,
                    "residual_turns": r.residual_turns,
                    "residual_log": r.residual_log,
                    "z": r.z,
                    "verdict": "rich" if r.residual_log > 0 else "cheap",
                }
                for r in chosen
            ]
        )

    def verdict(self) -> str:
        """One paragraph: the re-rating, the score, its caveat, and what to do.

        The caveat is not optional decoration and it is not in the notes where it
        could be skipped. A pooled rank correlation on a panel of the same
        companies quarter after quarter is mostly the companies, and the sentence
        that says so travels with the sentence that gives the number.
        """
        lines = [self.rerating.sentence(), self.card.summary()]
        result = self.card.evaluation
        if result is not None and not result.beat_baseline:
            lines.append(
                f"The fitted model does not beat {result.baseline_name}. Use the "
                "baseline: it is cheaper, it is auditable, and it is better."
            )
        if self.change_rank_correlation is not None:
            lines.append(
                "Differenced against each company's own previous observation the "
                f"rank correlation is {self.change_rank_correlation:+.4f} on "
                f"{self.n_changes:,} observations, so almost all of the score above "
                "is the level inherited from history rather than a view about the "
                "change. Treat the residual as a description of where a company "
                "sits, not as a forecast of where it is going."
            )
        return " ".join(lines)


# --------------------------------------------------------------------------- #
# The split basis correction
# --------------------------------------------------------------------------- #


def revenue_is_a_component(facts, fin) -> str | None:
    """Whether the winning revenue tag is a piece of revenue rather than revenue.

    The engine's revenue ladder ranks the ASC 606 concept first, which is right
    for an operating company and wrong for a lessor. A tower REIT's tenant and
    ground lease income is lease revenue under ASC 842 and is not a contract with
    a customer, so only its site-services piece carries the 606 tag and the total
    sits in ``Revenues``. Resolving the first entry that has any value at all
    therefore hands back a fraction of the company.

    Measured on this panel, at 31 March 2025: Crown Castle resolves 210.0mm
    against 6,568.0mm reported under ``Revenues``, American Tower 774.6mm against
    10,127.2mm, and SBA Communications 152.9mm against 2,679.6mm. Every EV/Revenue
    for the towers and fibre sub-vertical is therefore between thirteen and
    thirty-one times too high, and the whole bucket reads as though it trades at
    125x revenue.

    **The fix belongs one level down and is not made here.** Ranking or guarding
    the ladder inside ``edgar.resolve_ttm`` would repair the figure rather than
    decline it, and it would repair it for the DCF and the comp tables too, which
    have the same number wrong today. That is a change to the resolution path
    every valuation in the engine runs through, and a model is the wrong place to
    make it. This refuses the row and says what it found, so the panel is not
    fitted on a thirteenfold error while the decision is somebody else's to take.
    """
    if fin.revenue is None or fin.revenue <= 0:
        return None
    for entry in tags.REVENUE:
        if not isinstance(entry, str):
            continue
        value, _ = facts.resolve_ttm("revenue", [entry], fin.as_of, required=False)
        if value is None:
            continue
        alternative = value / _MM
        if alternative > fin.revenue * _COMPONENT_REVENUE_MULTIPLE:
            return (
                f"revenue resolved to {fin.revenue:,.1f}mm but {entry} covers the "
                f"same window at {alternative:,.1f}mm, {alternative / fin.revenue:,.1f} "
                "times larger. The winning tag is a component of revenue and not "
                "revenue: a lessor reports tenant income under ASC 842 and tags "
                "only its services piece under the ASC 606 concept the ladder ranks "
                "first. The multiple built on it would be wrong by that factor"
            )
    return None


def debt_is_outside_the_ladder(facts, fin) -> str | None:
    """Whether what the debt ladders resolved is the whole of the debt.

    A filer that reports under ``LongTermDebtAndCapitalLeaseObligations`` matches
    nothing in ``tags.DEBT_NONCURRENT``, ``DEBT_CURRENT`` or ``DEBT_COMBINED``, so
    its debt resolves to zero and its enterprise value comes back as equity less
    cash. Lumen at 30 September 2023 resolves zero against roughly twenty billion
    dollars of it, and its enterprise value prints as 996mm.

    ``LongTermDebt`` does exist in Lumen's fact set, which is what makes this
    quiet: the tag is there and its newest value is dated 2021, so a resolution
    that asked for a value AT the balance sheet date correctly finds nothing and
    returns zero rather than raising. The retired-tag trap in the ``edgar``
    docstring, one ladder along.

    **A partial resolution is worse than a zero, and this test used to stop at
    the zero.** The guard was written as ``if straight_debt > 0: return None``,
    which reads as "the ladder found the debt" and means "the ladder found A
    debt". Verizon at 30 June 2026 reports 143,448mm of long-term debt under the
    outside concept and 21,783mm of it maturing within a year under
    ``LongTermDebtCurrent``, which IS in the ladder. The current portion alone
    resolved, the guard saw a positive number and passed the row, and the panel
    took Verizon at an enterprise value 143bn light: a screen then called it
    priced in line when its real multiple sits above the fitted line rather than
    below it. Every large carrier has this shape, so the threshold excluded
    exactly the population the check was written for. The comparison is now
    against what the ladder resolved rather than against zero, which catches the
    zero as the special case it is.

    The tolerance is a ratio because the outside concepts bundle capital lease
    obligations in with the debt while the ladder's own concepts do not, so a
    filer that resolved correctly can legitimately show a few percent more under
    one of them. ``_MATERIAL_DEBT`` floors the gap in dollars so a small filer is
    not refused over a rounding difference.

    Adding the tag to the ladder is NOT a one-line fix, which is why it is flagged
    rather than made. The concept includes capital lease obligations, and the
    bridge already adds finance leases from their own tags, so importing it
    wholesale would count every finance lease twice. Someone has to decide whether
    to net them off or to suppress the separate lease line for filers using it.
    """
    resolved = float(fin.straight_debt or 0.0)
    for tag in _DEBT_TAGS_OUTSIDE_THE_LADDER:
        value, _ = facts.resolve_instant("debt", [tag], fin.as_of, required=False)
        if value is None:
            continue
        outside = value / _MM
        if outside - resolved < _MATERIAL_DEBT:
            continue
        if outside <= resolved * _DEBT_LADDER_TOLERANCE:
            continue
        had = (
            "straight debt resolved to zero"
            if resolved <= 0
            else f"straight debt resolved to {resolved:,.1f}mm"
        )
        return (
            f"{had}, but {tag} reports {outside:,.1f}mm at {fin.as_of}. That "
            "concept is in none of the three debt ladders, so the enterprise "
            "value here understates the claim on the business by at least "
            f"{outside - resolved:,.1f}mm, and by more where the two cover "
            "different parts of the balance sheet"
        )
    return None


def price_disagrees_with_the_public_float(facts, fin, equity_value: float) -> str | None:
    """Whether the price series belongs to this filer at all.

    Nothing inside a valuation can validate a price feed against itself, and the
    failure is silent: a vendor that returns the wrong security for a symbol gives
    a clean, monotone, entirely plausible series. The one independent anchor free
    data offers is the filer's own cover page. ``dei:EntityPublicFloat`` is the
    market value of the shares held by non-affiliates, reported once a year as of
    a date inside the prior half year, and it is a number the company states
    rather than a number the vendor computes.

    Booking Holdings is the case. The vendor's series for BKNG runs 71.39 at the
    start of 2018 to 174.33 in September 2026, against a share that actually
    traded near 1,750 and 5,500 on those days: not a split adjustment, since the
    ratio is not constant, but a different instrument. Priced on it at
    2025-03-31, Booking's equity value comes out at 6,277mm against a public
    float of 133,100mm reported as of 2024-06-30, a factor of 0.05, and every
    committed read of its EV/Revenue prints below a single turn of revenue.

    The tolerance is ten times either way and is meant to be far too wide to fire
    on anything real. The float is stale by up to a year, it excludes insider
    holdings, and a founder-led name that has run hard since the last cover page
    can legitimately sit several times above it: across this universe Palantir is
    the largest honest offender at 6.6 times, and everything else lands between
    0.6 and 2.3. A check this loose catches a wrong security and nothing else,
    which is exactly the job.
    """
    if equity_value <= 0:
        return None
    floats = [f for f in facts.facts("EntityPublicFloat") if f.val and f.val > 0]
    if not floats:
        return None
    latest = max(floats, key=lambda f: (f.end, f.filed))
    reported = latest.val / _MM
    ratio = equity_value / reported
    if 1.0 / _FLOAT_TOLERANCE <= ratio <= _FLOAT_TOLERANCE:
        return None
    return (
        f"the price implies an equity value of {equity_value:,.0f}mm against a "
        f"public float of {reported:,.0f}mm reported as of {latest.end}, a factor "
        f"of {ratio:,.2f}. A float is stale and excludes insiders, so this check is "
        "deliberately loose; a gap this size means the price series does not belong "
        "to this filer"
    )


def share_basis_factor(pinned, current) -> tuple[float, str | None]:
    """The unit conversion between an as-filed share count and today's price basis.

    ``pinned`` is a ``CompanyFacts`` built with a knowledge date and ``current``
    is one built without. Both are fact sets for the same filer. The return is
    ``(factor, note)``: multiply the pinned share count, and anything built from
    it, by ``factor`` to put it on the basis the price series is quoted on.

    The whole construction is one ratio. Take any fiscal period both fact sets
    carry. The pinned set reports it on whatever basis was current at the
    knowledge date; the current set reports the same period restated for every
    split since. Their ratio IS the cumulative split factor, and it needs no
    ex-date, no vendor calendar and no assumption about how quickly a filer
    restates its comparatives. The most recent shared period is used, because an
    old period stops being restated once no filing shows it as a comparative any
    more, and a stale one would miss the later splits.

    The ratio is then snapped to a product of the splits this filer actually
    declared, which are the suffix products of ``_split_factors``: a fact set
    holding a four-for-one and a ten-for-one admits factors of 1, 10 and 40 and
    nothing else. Palo Alto's pair of share counts differ by 5.998 rather than 6
    because the counts themselves are rounded to thousands, and snapping is what
    turns that into the 6 it obviously is. A ratio inside two percent of one is
    an ordinary restatement and the basis has not moved. Anything else is
    returned unsnapped with a note saying so, for the caller to refuse.
    """
    now = {(f.start, f.end): f.val for f in current.facts(DILUTED_SHARES_TAG)}
    ratio: float | None = None
    for fact in pinned.facts(DILUTED_SHARES_TAG):
        if fact.val and (fact.start, fact.end) in now:
            ratio = now[(fact.start, fact.end)] / fact.val
    if ratio is None or ratio <= 0:
        return 1.0, (
            "no fiscal period is carried by both the pinned and the current fact "
            "set, so the share basis could not be checked and 1.0 was assumed"
        )
    if abs(ratio - 1.0) <= _RESTATEMENT_TOLERANCE:
        return 1.0, None

    declared = [factor for _, factor in current._split_factors()]
    candidates = {1.0}
    for i in range(len(declared)):
        product = 1.0
        for factor in declared[i:]:
            product *= factor
        candidates.add(product)
    for candidate in sorted(candidates):
        if abs(ratio - candidate) <= _BASIS_TOLERANCE * candidate:
            return candidate, None
    return ratio, (
        f"the diluted share count has moved by {ratio:.4f} since this date and that "
        "is neither a rounding restatement nor a product of any split detected in "
        "the fact set, so the price and the share count may be on different bases"
    )


# --------------------------------------------------------------------------- #
# Building the panel
# --------------------------------------------------------------------------- #

ClientFactory = Callable[[date], Any]
MarketFactory = Callable[[date], MarketData]


def build_observations(
    tickers: Sequence[str],
    dates: Sequence[date],
    client_factory: ClientFactory,
    market_factory: MarketFactory,
    sub_verticals: dict[str, SubVertical | str],
    assumptions: Assumptions | None = None,
    *,
    current_client: Any,
    target: str | None = None,
    check_point_in_time: bool = True,
) -> ObservationPanel:
    """Price every company at every date and reduce it to one multiple and its features.

    ``client_factory`` returns an ``EdgarClient`` pinned to the date it is handed
    and ``market_factory`` returns a ``MarketData`` capped at it, exactly as
    ``features.build_panel`` requires, because a single feed can only be pinned to
    one date and handing one in would let every row after the first see the wrong
    world.

    ``current_client`` is the one deliberate exception and it is read for exactly
    one thing: the split calendar behind ``share_basis_factor``. Nothing else on
    the row touches it, ``assert_point_in_time`` still walks the provenance of
    every figure that reached the features, and a split is a unit change rather
    than information: knowing a company will later split tells you nothing about
    what it is worth. Without it the price and the share count are on different
    bases and every pre-split observation is wrong by an order of magnitude,
    which is the worse failure by a wide margin.

    The enterprise value comes from ``ev_bridge.build_ev_bridge``, so net debt,
    operating leases and convertibles are treated here exactly as they are in the
    comp tables and the DCF, and the denominator that pairs with it comes from
    ``bridge.multiple_denominator`` on the EBITDA target, which is the engine's
    only sanctioned way to keep a lease-inclusive numerator off a post-rent
    earnings figure.

    Nothing is dropped silently. A company that had not listed, one whose tagging
    will not resolve, and one whose multiple is not meaningful are three different
    facts about the sample and each is recorded as its own category in ``skips``.

    A ``LookaheadError`` is not one of those and is not caught. It is an
    ``AssertionError`` rather than a ``TechvalError`` precisely so it travels past
    the per-observation handler: a filer this engine cannot parse costs one row,
    while a row built from a filing that did not exist yet means the panel is not
    evidence and the run should stop.
    """
    assumptions = assumptions or Assumptions()
    target = target or assumptions.ml.warranted.target
    if target not in TARGETS:
        raise ConfigError(
            f"{target!r} is not a multiple this module fits; expected one of "
            + ", ".join(TARGETS)
        )
    attribute, label = TARGETS[target]

    observations: list[Observation] = []
    skips: list[Skip] = []
    unchecked_basis = 0

    for when in dates:
        client = client_factory(when)
        market = market_factory(when)
        for raw_ticker in tickers:
            ticker = raw_ticker.upper()
            vertical = sub_verticals.get(ticker)
            if vertical is None:
                skips.append(
                    Skip(ticker, when, "no sub-vertical classification", "unclassified")
                )
                continue
            try:
                row = build_features(ticker, when, client, market, assumptions)
                facts = client.company_facts(ticker)
                fin = build_financials(ticker, facts=facts)
                price = market.spot(ticker)
                bridge = build_ev_bridge(fin, price, assumptions)
                if check_point_in_time:
                    assert_point_in_time(row, fin, prices=market.prices(ticker))
            except TechvalError as exc:
                skips.append(
                    Skip(ticker, when, f"{type(exc).__name__}: {exc}", "not_built")
                )
                continue

            # A note here means one of two different things. Either no period is
            # shared between the two fact sets, in which case the basis could not
            # be checked at all and 1.0 is the conservative assumption, counted
            # and reported; or the ratio is real and matches no split this filer
            # declared, in which case the price and the share count may be on
            # different bases and the row is refused rather than guessed at.
            factor, note = share_basis_factor(
                facts, current_client.company_facts(ticker)
            )
            if note is not None and "1.0 was assumed" not in note:
                skips.append(Skip(ticker, when, note, "share_basis_unresolved"))
                continue
            if note is not None:
                unchecked_basis += 1

            equity = bridge.equity_value * factor
            enterprise = equity + bridge.net_debt

            # Three checks on the inputs before the division. Each was written
            # because this universe hit it, each names what it found, and each
            # costs the row rather than silently correcting it, because the
            # corrections belong in the resolution path rather than in a model.
            # See the three functions above for the cases that earned them.
            failed = next(
                (
                    (reason, category)
                    for reason, category in (
                        (
                            revenue_is_a_component(facts, fin),
                            "revenue_is_a_component",
                        ),
                        (
                            debt_is_outside_the_ladder(facts, fin),
                            "debt_outside_the_ladder",
                        ),
                        (
                            price_disagrees_with_the_public_float(facts, fin, equity),
                            "price_disagrees_with_float",
                        ),
                    )
                    if reason is not None
                ),
                None,
            )
            if failed is not None:
                skips.append(Skip(ticker, when, failed[0], failed[1]))
                continue

            denominator = (
                bridge.multiple_denominator(fin)[0]
                if attribute == "ebitda"
                else getattr(fin, attribute)
            )
            if denominator is None:
                skips.append(
                    Skip(
                        ticker,
                        when,
                        f"{label} has no denominator: {attribute.replace('_', ' ')} is "
                        "not reported by this filer",
                        "denominator_missing",
                    )
                )
                continue
            if denominator <= 0 or enterprise <= 0:
                skips.append(
                    Skip(
                        ticker,
                        when,
                        f"{label} is not meaningful: enterprise value "
                        f"{enterprise:,.0f}mm over {attribute.replace('_', ' ')} of "
                        f"{denominator:,.0f}mm. A log needs a positive argument and a "
                        "negative multiple is not a valuation",
                        "not_meaningful",
                    )
                )
                continue

            multiple = enterprise / denominator
            observations.append(
                Observation(
                    ticker=ticker,
                    as_of=when,
                    sub_vertical=(
                        vertical.value if isinstance(vertical, SubVertical) else str(vertical)
                    ),
                    multiple=multiple,
                    log_multiple=math.log(multiple),
                    enterprise_value=enterprise,
                    equity_value=equity,
                    denominator=denominator,
                    statement_date=row.statement_date,
                    basis_factor=factor,
                    features={name: row.values.get(name) for name in FEATURE_NAMES},
                    notes=() if note is None else (note,),
                )
            )

    panel = ObservationPanel(
        target=target,
        observations=observations,
        skips=skips,
        random_seed=assumptions.ml.random_seed,
    )
    adjusted = sum(1 for o in observations if o.basis_factor != 1.0)
    if adjusted:
        panel.notes.append(
            f"{adjusted} of {len(observations)} observations carry a share basis "
            "factor other than 1.0: their filed share count is on a pre-split basis "
            "and the price series is not, so the equity value was converted onto the "
            "price's basis. Without that conversion each of them is understated by "
            "the split ratio."
        )
    if unchecked_basis:
        panel.notes.append(
            f"{unchecked_basis} observation(s) share no fiscal period between the "
            "pinned and the current fact set, so their share basis could not be "
            "checked and was assumed unchanged. Each carries the note on its own row."
        )
    if skips:
        panel.notes.append(
            f"{len(skips)} company-dates produced no observation and are carried in "
            "skips with the reason, not dropped"
        )
    return panel


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


class _Ridge:
    """Ordinary least squares with a small ridge on the standardised columns.

    A ridge rather than a plain inverse for one reason: Rule of 40 is revenue
    growth plus free cash flow margin, and both of those are in the same design
    matrix, so the three columns are close to linearly dependent by construction.
    A plain normal-equation solve on that is numerically unstable and produces two
    enormous coefficients of opposite sign that cancel. The penalty is not applied
    to the intercept, because shrinking the intercept shrinks the prediction
    toward zero rather than toward the mean.
    """

    def __init__(self, lam: float = _RIDGE_LAMBDA) -> None:
        self.lam = lam
        self.beta: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Ridge":
        n, k = X.shape
        design = np.column_stack([np.ones(n), X])
        penalty = np.eye(k + 1) * self.lam
        penalty[0, 0] = 0.0
        self.beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            raise ConfigError("the ridge was asked to predict before it was fitted")
        return np.column_stack([np.ones(X.shape[0]), X]) @ self.beta


class _Mlp:
    """A small multilayer perceptron on the package's own numpy primitives.

    Two hidden layers of thirty-two units with dropout, trained by Adam with
    early stopping on a held-out tail of the training window. The architecture is
    deliberately small: a training fold here is a few hundred to a couple of
    thousand rows of two dozen columns, and a wider network memorises it. If a
    network this size cannot beat the ridge, the honest conclusion is that the
    relationship between fundamentals and the multiple is close to linear, which
    is a finding rather than a failure.

    **No hyperparameter here was chosen by looking at the walk-forward score.**
    The width, the depth, the dropout, the learning rate and the batch size were
    fixed before the first fit and have not been moved since, because moving them
    until the out-of-sample number improved would be fitting the test set through
    a slower channel, and the resulting score would be an in-sample score wearing
    a walk-forward costume. The one thing the data chooses is when to stop, and it
    chooses that on a slice of the TRAINING window that the fold never trains on.
    If this network is badly configured for the problem, the number below is what
    a badly configured network scores, and that is the honest thing to publish.
    """

    def __init__(self, width: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        layers: list[nn.Layer] = []
        previous = width
        for size in _MLP_HIDDEN:
            layers.append(nn.Linear(previous, size, rng=rng))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(_MLP_DROPOUT, rng=rng))
            previous = size
        # Xavier on the head: it feeds no activation, so the He scale meant for a
        # ReLU would start the output twice as wide as the target.
        layers.append(nn.Linear(previous, 1, rng=rng, init="xavier"))
        self.model = nn.Sequential(*layers)
        self.seed = seed
        self.history: nn.TrainHistory | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val, y_val) -> "_Mlp":
        batches = _batches(X, y, _MLP_BATCH)
        holdout = None if X_val is None else _batches(X_val, y_val, _MLP_BATCH)
        self.history = nn.train(
            self.model,
            batches,
            nn.mse_loss,
            nn.Adam(lr=_MLP_LR),
            val_batches=holdout,
            epochs=_MLP_EPOCHS,
            patience=_MLP_PATIENCE,
            seed=self.seed,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # ``nn.train`` leaves the stack in evaluation mode, which is what turns
        # dropout off. Asserted here rather than assumed, because a network left
        # in training mode predicts through a random mask and the damage looks
        # like noise rather than like a bug.
        self.model.eval()
        return self.model.forward(X).ravel()


def _batches(X: np.ndarray, y: np.ndarray, size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed contiguous batches. No shuffling here: ``nn.train`` shuffles order."""
    return [
        (X[i : i + size], y[i : i + size].reshape(-1, 1))
        for i in range(0, X.shape[0], size)
    ]


def _design(
    observations: Sequence[Observation], names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous features and the sub-vertical indicator block, kept apart.

    Apart because they are treated differently downstream: the continuous columns
    are centred and scaled on the training rows and can be missing, while an
    indicator is one or zero and is never missing. Pooling them would put the
    dummies through a standardisation that does nothing for them and would let
    them into the missing-share column, which is supposed to measure how much of
    a company's FUNDAMENTALS the filer left untagged.

    The sub-vertical earns its columns. Infrastructure software and telecom do not
    trade at the same multiple for the same growth and the same margin, and a
    model without the bucket has to explain that gap with whatever continuous
    feature happens to correlate with being a carrier. ``taxonomy.SubVertical``
    exists precisely because the engine refuses to compare across those lines, and
    this is the same refusal expressed as a fixed effect.
    """
    X = np.asarray(
        [
            [
                np.nan if o.features.get(n) is None else float(o.features[n])
                for n in names
            ]
            for o in observations
        ],
        dtype=float,
    )
    index = {vertical.value: i for i, vertical in enumerate(SubVertical)}
    dummies = np.zeros((len(observations), len(index)), dtype=float)
    for row, obs in enumerate(observations):
        column = index.get(obs.sub_vertical)
        if column is not None:
            dummies[row, column] = 1.0
    return X, dummies


def _leave_one_out_means(
    y: np.ndarray, dates: np.ndarray
) -> tuple[np.ndarray, dict[date, float]]:
    """Each date's cross-sectional mean, computed without the row it is applied to.

    The plain date mean is contemporaneous information and so not lookahead, but
    it contains 1/N of the very observation it is about to be subtracted from,
    and a residual that small can be moved by that. The leave-one-out mean is
    ``(sum - y_i) / (n - 1)`` and removes the self-contribution exactly.

    The plain means are returned beside it because the variance decomposition
    needs them and must not use the leave-one-out version: the between-date and
    within-date pieces only add up to the total when the thing being subtracted
    is the actual mean.
    """
    loo = np.zeros_like(y)
    plain: dict[date, float] = {}
    for when in sorted(set(dates.tolist())):
        mask = dates == when
        values = y[mask]
        n = values.size
        plain[when] = float(values.mean())
        loo[mask] = (
            (values.sum() - values) / (n - 1) if n > 1 else values
        )
    return loo, plain


def _standardize(X: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre on the training median, scale by the training standard deviation.

    The median rather than the mean because the panel still carries fat tails
    after a cross-sectional trim, and a single company at the 99th percentile of
    net-debt-to-capital should not decide where the centre of that column is. The
    standard deviation is kept as the scale because it is the quantity the
    network's initialisation assumes and a robust scale would leave the inputs a
    factor of two off it.

    Fitted on the TRAINING rows alone. Fitting on the whole panel and splitting
    afterwards moves the test period's own distribution into the transform, which
    is a small leak that buys a large and entirely fake improvement.

    Columns never observed in training pass through at unit scale rather than
    being divided by zero.
    """
    with warnings.catch_warnings():
        # An all-missing column in the training window is a real outcome on this
        # panel, and numpy's "mean of empty slice" warning is not news.
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmedian(X[train], axis=0)
        spread = np.nanstd(X[train], axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
    return centre, spread


def _prepare(X: np.ndarray, dummies: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Standardise, fill what is still missing with the training median, add the block.

    Filling with the training median is a decision with a cost and it is stated
    rather than hidden: a company whose gross margin the filer never reported is
    told it has the median gross margin, which is a claim nobody made. The
    alternative is to drop every row with any gap, and on this universe that
    deletes media and telecom wholesale, because a cable company reports cost of
    revenues without a gross profit subtotal. Deleting two sub-verticals to avoid
    imputing one column is the worse trade, and the missing share travels beside
    the features as its own column so the model can learn that a row with gaps is
    a different kind of company.

    After standardisation the median sits at zero, so filling with the median is
    the same operation as filling with zero and is written that way.

    Column order is continuous features, then the missing share, then the
    sub-vertical indicators, which is the order the coefficient names are built in.
    """
    centre, spread = _standardize(X, train)
    Z = (X - centre) / spread
    missing = np.isnan(Z)
    Z = np.where(missing, 0.0, Z)
    return np.column_stack([Z, missing.mean(axis=1), dummies])


def _train_validation_split(dates: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the most recent fifth of the training window, by date.

    By date and off the end, never at random. A random holdout lets the network
    stop on a row whose own date it trained on, which is the same leak the whole
    walk-forward exists to prevent, one level down.
    """
    training_dates = sorted(set(dates[train].tolist()))
    if len(training_dates) < 5:
        return train, np.zeros_like(train)
    cut = training_dates[int(len(training_dates) * (1 - _VALIDATION_SHARE))]
    fit = train & (dates < cut)
    validate = train & (dates >= cut)
    if fit.sum() < 30 or validate.sum() < 10:
        return train, np.zeros_like(train)
    return fit, validate


def _rerating(y: np.ndarray, dates: np.ndarray) -> Rerating:
    """Decompose the variance of the log multiple into calendar and cross-section."""
    _, plain = _leave_one_out_means(y, dates)
    fitted = np.asarray([plain[d] for d in dates.tolist()], dtype=float)
    grand = float(y.mean())
    total = float(((y - grand) ** 2).mean())
    between = float(((fitted - grand) ** 2).mean())
    within = float(((y - fitted) ** 2).mean())
    return Rerating(
        n_observations=int(y.size),
        n_dates=len(plain),
        total_variance=total,
        between_date_variance=between,
        within_date_variance=within,
        calendar_r_squared=0.0 if total <= 0 else 1.0 - within / total,
        date_means=plain,
    )


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def _grouped_rank_correlation(
    y: np.ndarray,
    prediction: np.ndarray,
    dates: np.ndarray,
    verticals: np.ndarray,
    mask: np.ndarray,
    min_names: int,
) -> tuple[float | None, int]:
    """Mean rank correlation inside each date-and-sub-vertical cross-section.

    The pooled score over the whole panel flatters every model here, because
    most of the ordering it has to get right is the ordering of the buckets: a
    carrier trades at 3x and a security name at 15x, and saying so is not
    analysis. Inside one sub-vertical on one date the bucket is held constant and
    what is left is the question an analyst actually asks, which is why this is
    reported beside the headline rather than instead of it. It is a mean over
    cross-sections rather than a pooled figure so that a date with forty names
    does not drown one with twelve.
    """
    scores: list[float] = []
    for when in sorted(set(dates[mask].tolist())):
        for vertical in sorted(set(verticals[mask].tolist())):
            group = mask & (dates == when) & (verticals == vertical)
            if int(group.sum()) < min_names:
                continue
            value = spearman(y[group], prediction[group])
            if value is not None:
                scores.append(value)
    if not scores:
        return None, 0
    return float(np.mean(scores)), len(scores)


def _sub_vertical_median(
    observations: Sequence[Observation], y: np.ndarray, dates: np.ndarray
) -> np.ndarray:
    """The sub-vertical's median log multiple on that date, excluding the name itself.

    The actual comps convention and a strong baseline, which is why it is here.
    Leave-one-out because ``comps.CompsResult.table`` is explicit that a target
    inside its own comp set pulls the median toward the answer it is being tested
    against. NaN where the sub-vertical has too few names on that date to carry a
    median, and those observations are then scored out of this comparison rather
    than given a number.
    """
    out = np.full(y.shape, np.nan)
    verticals = np.asarray([o.sub_vertical for o in observations])
    for when in sorted(set(dates.tolist())):
        for vertical in sorted(set(verticals.tolist())):
            mask = (dates == when) & (verticals == vertical)
            idx = np.flatnonzero(mask)
            if idx.size < _MIN_PEERS_FOR_MEDIAN:
                continue
            values = y[idx]
            for position, row in enumerate(idx):
                others = np.delete(values, position)
                out[row] = float(np.median(others))
    return out


def _persistence(
    observations: Sequence[Observation], y: np.ndarray
) -> np.ndarray:
    """The company's own relative position last quarter. Usually brutal to beat.

    A multiple is close to a random walk over a quarter, so carrying the last
    observation forward is a genuinely hard benchmark and the one most likely to
    embarrass a fitted model. NaN for a company's first appearance, which is
    scored out of this comparison.

    "Last quarter" is the company's own previous OBSERVATION, which is the
    previous quarter unless the panel skipped it, in which case the carry is
    older and the baseline is being given a slightly harder job than its name
    suggests. Left that way deliberately: dropping those rows would quietly
    remove the companies whose filings are hardest to resolve, and they are not a
    random sample of the universe.
    """
    out = np.full(y.shape, np.nan)
    history: dict[str, tuple[date, float]] = {}
    order = sorted(range(len(observations)), key=lambda i: (observations[i].as_of, i))
    for i in order:
        obs = observations[i]
        previous = history.get(obs.ticker)
        if previous is not None and previous[0] < obs.as_of:
            out[i] = previous[1]
        history[obs.ticker] = (obs.as_of, float(y[i]))
    return out


def _change_in_relative_position(
    y: np.ndarray,
    prediction: np.ndarray,
    previous: np.ndarray,
    mask: np.ndarray,
) -> tuple[float | None, int]:
    """Whether the model knows anything persistence does not.

    A company's feature vector barely moves from one quarter to the next, and
    neither does its relative multiple, so a model fitted on the past and scored
    on the present is rewarded for recognising the company as much as for
    understanding it. That is not lookahead: the training window is strictly
    earlier. But it means a rank correlation of 0.8 can be almost entirely a
    company fixed effect learned from history and dressed as a view about
    fundamentals, and a reader shown 0.8 without this number will believe more
    than the model earned.

    The test differences both sides against the company's own last observation.
    ``y - previous`` is how its relative position actually moved this quarter and
    ``prediction - previous`` is how the model said it would. A correlation near
    zero means the level was inherited and nothing about the CHANGE was
    predicted, which is the honest reading of most of what a model like this
    does. Anything materially positive is the part that is genuinely about
    fundamentals rather than about identity.
    """
    usable = mask & np.isfinite(previous)
    if int(usable.sum()) < MIN_OBSERVATIONS:
        return None, int(usable.sum())
    realised = y[usable] - previous[usable]
    implied = prediction[usable] - previous[usable]
    return spearman(realised, implied), int(usable.sum())


def _comps_ols(
    observations: Sequence[Observation],
    assumptions: Assumptions,
    *,
    within_sub_vertical: bool,
) -> tuple[np.ndarray, int]:
    """The existing OLS from ``comps.py``, refit on each date's cross-section.

    This is the incumbent and it is not a straw man. It is refit on every date, so
    unlike the walk-forward model it sees the CURRENT date's pricing relationship
    rather than having to carry one forward from history, and it excludes the
    target from its own peer set by construction. A model that only ties it is
    therefore saying something real: that a relationship learned from the past
    holds up against one fitted on the answer's own cross-section.

    ``within_sub_vertical`` picks the peer set. True is the convention a banker
    uses and it is where the function most often refuses, because a sub-vertical
    on one date holds six or eight names against a floor of eight; the count of
    refusals is returned so that refusal rate can be reported rather than hidden
    inside a NaN. False hands it the whole TMT cross-section, which it will
    always fit and which is a weaker claim about comparability.

    Returns the warranted multiple in TURNS, with NaN where the fit was refused.
    """
    out = np.full(len(observations), np.nan)
    refusals = 0
    by_date: dict[date, list[int]] = {}
    for i, obs in enumerate(observations):
        by_date.setdefault(obs.as_of, []).append(i)

    for indices in by_date.values():
        peers = {i: _as_peer_metrics(observations[i]) for i in indices}
        for i in indices:
            target = observations[i]
            others = [
                peers[j]
                for j in indices
                if j != i
                and (
                    not within_sub_vertical
                    or observations[j].sub_vertical == target.sub_vertical
                )
            ]
            fit = fit_warranted_multiple(others, peers[i], assumptions)
            if fit is None or fit.warranted_multiple <= 0:
                refusals += 1
                continue
            out[i] = fit.warranted_multiple
    return out, refusals


def _comps_baselines(
    panel: ObservationPanel,
    observations: Sequence[Observation],
    assumptions: Assumptions,
) -> dict[str, tuple[np.ndarray, int]]:
    """Both comps.py variants, computed once per panel and per regression setting."""
    cfg = assumptions.comps.regression
    key = (cfg.min_observations, tuple(cfg.drivers))
    cached = panel._comps_memo.get(key)
    if cached is None:
        cached = {
            "sub_vertical": _comps_ols(
                observations, assumptions, within_sub_vertical=True
            ),
            "universe": _comps_ols(
                observations, assumptions, within_sub_vertical=False
            ),
        }
        panel._comps_memo[key] = cached
    return cached


def _as_peer_metrics(obs: Observation) -> PeerMetrics:
    """An Observation in the shape ``comps.fit_warranted_multiple`` reads.

    Only six fields are touched on this call path: ``ev_revenue`` as the
    dependent, ``revenue_growth`` and ``ebitda_margin`` as the default drivers,
    and ``revenue``, ``market_cap`` and ``price`` for the implied enterprise value
    and price walk that the fit reports and this module ignores. The rest are
    filled with the figures the panel does carry, or with zero where it does not,
    and none of them reaches the regression. Building the peer through the real
    dataclass rather than reimplementing the regression is the point: the
    baseline has to be the incumbent itself, not a reconstruction of it that
    could differ.
    """
    growth = obs.features.get("growth_revenue_1y")
    margin = obs.features.get("margin_ebitda")
    gross = obs.features.get("margin_gross")
    return PeerMetrics(
        ticker=obs.ticker,
        name=obs.ticker,
        price=1.0,
        market_cap=obs.equity_value,
        enterprise_value=obs.enterprise_value,
        gross_debt=0.0,
        revenue=obs.denominator,
        ebitda=None if margin is None else margin * obs.denominator,
        ebit=0.0,
        net_income=0.0,
        gross_profit=None if gross is None else gross * obs.denominator,
        revenue_growth=growth,
        ebitda_margin=margin,
        gross_margin=gross,
        rule_of_40=None if growth is None or margin is None else (growth + margin) * 100.0,
        ev_revenue=obs.multiple,
        ev_gross_profit=None,
        ev_ebitda=None,
        ev_ebit=None,
        pe=None,
    )


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #


def fit_warranted(
    panel: ObservationPanel,
    assumptions: Assumptions | None = None,
    *,
    model: str = "mlp",
    features: Sequence[str] = WARRANTED_FEATURES,
    metric: str = "spearman",
    n_folds: int | None = None,
) -> WarrantedModel:
    """Fit the warranted multiple walk-forward and score it against three baselines.

    ``model`` is ``"mlp"`` or ``"ridge"``. Both are fitted and the one named is
    the headline; the other travels in ``notes``, because the comparison between
    them is the interesting part. A network that ties a thirteen-variable ridge is
    saying the relationship is close to linear.

    **The embargo is zero and that is a decision, not an oversight.** The label
    here is the multiple observed ON the observation date, not a forward return,
    so an observation dated one day before the test window opens leaks nothing
    into it: its label was knowable that day. Where the label is a forward
    quantity the embargo has to be the horizon and ``evaluation.walk_forward_folds``
    takes one; here the correct value is zero and it is written into the notes so
    a reader can see the protection was considered and declined rather than
    forgotten.

    **Refuses below ``ml.warranted.min_train_observations``.** A hundred and fifty
    is not a statistical nicety: below it the cross-section cannot support a dozen
    fundamental coefficients and eleven sub-vertical fixed effects, and the
    residual that comes out is the model's own noise dressed as a valuation
    signal.
    """
    assumptions = assumptions or Assumptions()
    cfg = assumptions.ml.warranted
    n_folds = n_folds or assumptions.ml.walk_forward_folds
    if model not in ("mlp", "ridge"):
        raise ConfigError(f"model must be 'mlp' or 'ridge', got {model!r}")

    banned = [name for name in features if name in BANNED_FEATURES]
    if banned:
        raise ConfigError(
            "these features cannot appear on the right-hand side of a warranted "
            "multiple:\n  "
            + "\n  ".join(f"{name}: {BANNED_FEATURES[name]}" for name in banned)
        )
    unknown = [name for name in features if name not in FEATURE_NAMES]
    if unknown:
        raise ConfigError(
            f"{', '.join(unknown)} are not features the store builds; see "
            "techval.ml.features.FEATURE_NAMES"
        )

    observations = sorted(panel.observations, key=lambda o: (o.as_of, o.ticker))
    if len(observations) < cfg.min_train_observations:
        raise NotMeaningfulError(
            f"{len(observations)} observations against a floor of "
            f"{cfg.min_train_observations} in ml.warranted.min_train_observations. "
            "A warranted multiple fitted below that is the model's own noise with a "
            "decimal point on it, and this module reports nothing rather than that."
        )

    y_raw = np.asarray([o.log_multiple for o in observations], dtype=float)
    dates = np.asarray([o.as_of for o in observations], dtype=object)
    thin = [
        when
        for when in sorted(set(dates.tolist()))
        if int(np.sum(dates == when)) < _MIN_NAMES_PER_DATE
    ]
    rerating = _rerating(y_raw, dates)

    loo_means, _ = _leave_one_out_means(y_raw, dates)
    y = y_raw - loo_means if cfg.demean_by_date else y_raw
    add_back = loo_means if cfg.demean_by_date else np.zeros_like(y_raw)

    X, dummies = _design(observations, features)
    columns = ("intercept", *features, "missing_share", *SUB_VERTICAL_COLUMNS)
    folds = walk_forward_folds(
        dates, n_folds, min_train=cfg.min_train_observations, embargo_days=0
    )

    predictions: dict[str, np.ndarray] = {
        name: np.full(y.shape, np.nan) for name in ("mlp", "ridge")
    }
    # The last fold's, which is the one fitted on the most history. Reported for
    # sign inspection rather than for inference: they move fold to fold, and a
    # penalised fit on two dozen correlated columns does not carry a standard
    # error anyone should quote.
    coefficients: dict[str, float] = {}
    seed = assumptions.ml.random_seed

    for fold in folds:
        train = np.asarray([d <= fold.train_end for d in dates], dtype=bool)
        test = np.asarray(
            [fold.test_start <= d <= fold.test_end for d in dates], dtype=bool
        )
        if train.sum() < cfg.min_train_observations or test.sum() == 0:
            continue
        Z = _prepare(X, dummies, train)
        ridge = _Ridge().fit(Z[train], y[train])
        predictions["ridge"][test] = ridge.predict(Z[test])
        coefficients = {
            name: float(value) for name, value in zip(columns, ridge.beta)
        }
        fit_rows, validate = _train_validation_split(dates, train)
        network = _Mlp(Z.shape[1], seed + fold.index)
        network.fit(
            Z[fit_rows],
            y[fit_rows],
            Z[validate] if validate.any() else None,
            y[validate] if validate.any() else None,
        )
        predictions["mlp"][test] = network.predict(Z[test])

    scored = np.isfinite(predictions[model])
    if not scored.any():
        raise NotMeaningfulError(
            "no fold produced an out-of-sample prediction: every split fell below "
            f"the {cfg.min_train_observations} training observations the assumptions "
            "require"
        )

    sub_median = _sub_vertical_median(observations, y, dates)
    persistence = _persistence(observations, y)
    incumbent = _comps_baselines(panel, observations, assumptions)
    comps_sub, refused_sub = incumbent["sub_vertical"]
    comps_all, refused_all = incumbent["universe"]
    comps_sub_log = _to_relative(comps_sub, loo_means if cfg.demean_by_date else None)
    comps_all_log = _to_relative(comps_all, loo_means if cfg.demean_by_date else None)

    # The third column says whether a baseline is answering the same question. A
    # warranted multiple is built from things OTHER than the company's own price,
    # because its whole output is the gap to that price. Persistence is not built
    # that way: it is the company's own multiple carried forward, its residual is
    # "how much did the multiple move this quarter" rather than "is this rich",
    # and a warranted multiple that reproduced it would have no signal at all. It
    # is reported, loudly, because it is the strongest predictor in the table and
    # a reader is entitled to know that. It is not what the card is scored
    # against, because scoring a valuation tool against a tool that reads the
    # valuation would be comparing two different claims and calling it a contest.
    candidates: tuple[tuple[str, np.ndarray, bool], ...] = (
        (
            "sub-vertical median multiple on the date (the comps convention)",
            sub_median,
            True,
        ),
        ("comps.py OLS refit on the date, sub-vertical peers", comps_sub_log, True),
        (
            "comps.py OLS refit on the date, whole TMT cross-section",
            comps_all_log,
            True,
        ),
        (
            "the company's own multiple last quarter (persistence: NOT a warranted "
            "multiple, it reads the answer)",
            persistence,
            False,
        ),
    )
    comparable: dict[str, EvalResult] = {}

    baselines: dict[str, EvalResult] = {}
    unavailable: list[str] = []
    for name, values, is_comparable in candidates:
        mask = scored & np.isfinite(values)
        if int(mask.sum()) < 30:
            unavailable.append(
                f"{name}: only {int(mask.sum())} scored observations carry it, "
                "against the floor of 30"
            )
            continue
        try:
            result = evaluate_regression(
                y[mask],
                predictions[model][mask],
                values[mask],
                dates[mask],
                folds=folds,
                metric=metric,
            )
        except (ConfigError, NotMeaningfulError) as exc:
            # A baseline that does not exist across every fold cannot be scored on
            # the same split as the model, and scoring it on a different one would
            # not be a comparison. Recorded and dropped, never quietly widened.
            unavailable.append(f"{name}: {exc}")
            continue
        # ``evaluate_regression`` calls a supplied baseline "supplied baseline",
        # which is the right default and the wrong thing to print. The verdict is
        # meant to be one line a reader can act on, and it has to name what the
        # model is being compared against.
        result.baseline_name = name
        baselines[name] = result
        if is_comparable:
            comparable[name] = result

    if not comparable:
        raise NotMeaningfulError(
            "not one baseline answering the same question could be formed on 30 or "
            "more scored observations, so there is nothing honest to compare the "
            "model against"
        )

    # The hardest of the comparable baselines, not the most flattering. Signed by
    # the metric's own direction so this stays correct if the metric becomes an
    # error rather than a correlation.
    strongest = max(
        comparable.values(),
        key=lambda r: r.baseline_score if r.higher_is_better else -r.baseline_score,
    )
    toughest = next(k for k, v in comparable.items() if v is strongest)

    notes = list(panel.notes)
    for reason in unavailable:
        notes.append(f"Baseline not scored: {reason}")
    notes.append(
        "The embargo is zero days. The label is the multiple observed on the "
        "observation date rather than a forward return, so an observation dated "
        "the day before a test window opens leaks nothing into it. This is "
        "recorded rather than assumed: for a forward-looking label the embargo "
        "would have to be the full horizon."
    )
    notes.append(rerating.sentence())
    other = "ridge" if model == "mlp" else "mlp"
    if np.isfinite(predictions[other]).any():
        mask = scored & np.isfinite(predictions[other])
        rival = spearman(y[mask], predictions[other][mask])
        mine = spearman(y[mask], predictions[model][mask])
        notes.append(
            f"The {other} scores a rank correlation of "
            f"{'n/a' if rival is None else f'{rival:.4f}'} against "
            f"{'n/a' if mine is None else f'{mine:.4f}'} for the {model} on the same "
            f"{int(mask.sum()):,} observations."
        )
    for name, result in baselines.items():
        if name in comparable:
            continue
        better = "above" if result.lift < 0 else "below"
        notes.append(
            f"Reported and NOT the card's baseline: {name} scores "
            f"{result.baseline_score:.4f}, {better} the model's {result.score:.4f} on "
            f"{result.n_observations:,} observations. It is excluded from the card "
            "because it is built from the company's own multiple, which is the "
            "quantity a warranted multiple exists to be compared against. Read it as "
            "the ceiling on predicting the multiple, not as a rival valuation."
        )
    verticals = np.asarray([o.sub_vertical for o in observations])
    grouped_model, n_groups = _grouped_rank_correlation(
        y, predictions[model], dates, verticals, scored, 2 * _MIN_PEERS_FOR_MEDIAN
    )
    if grouped_model is not None:
        parts = [f"the {model} scores {grouped_model:+.4f}"]
        for name, values in ((n, v) for n, v, c in candidates if c):
            group_mask = scored & np.isfinite(values)
            value, _ = _grouped_rank_correlation(
                y, values, dates, verticals, group_mask, 2 * _MIN_PEERS_FOR_MEDIAN
            )
            if value is not None:
                parts.append(f"{name.split(' (')[0]} scores {value:+.4f}")
        notes.append(
            "Inside one sub-vertical on one date, averaged over the "
            f"{n_groups} cross-sections carrying at least "
            f"{2 * _MIN_PEERS_FOR_MEDIAN} names: " + "; ".join(parts) + ". This is "
            "the harder number and the one a screen lives on, because the pooled "
            "score above is mostly the ordering of the buckets. The sub-vertical "
            "median scores badly here for a structural reason rather than a "
            "measured one: a median assigns every member of its group the same "
            "value and so carries no ordering inside it at all, and what little it "
            "has comes from leaving each name out, which runs the wrong way by "
            "construction. That is the case against quoting one number for a whole "
            "comp set, stated as a number."
        )
    change, n_change = _change_in_relative_position(
        y, predictions[model], persistence, scored
    )
    if change is not None:
        notes.append(
            f"Differenced against the company's own previous observation, the "
            f"{model} scores a rank correlation of {change:+.4f} on {n_change:,} "
            "observations. Read the headline beside this one. A company's features "
            "barely move quarter to quarter and neither does its relative multiple, "
            "so most of a pooled score like the one above is the level the model "
            "inherited from history rather than a view about the change. This "
            "number is the part that is not."
        )
    notes.append(
        f"comps.py refused to fit on {refused_sub:,} of {len(observations):,} "
        f"sub-vertical cross-sections and on {refused_all:,} of the whole-universe "
        "ones. The sub-vertical refusals are the engine behaving as documented: a "
        "real comp set is six to ten names and the floor is eight."
    )
    if thin:
        notes.append(
            f"{len(thin)} date(s) carry fewer than {_MIN_NAMES_PER_DATE} names, so "
            "their cross-sectional mean is thin: " + ", ".join(str(d) for d in thin[:5])
        )

    reads: dict[tuple[str, date], WarrantedRead] = {}
    for i, obs in enumerate(observations):
        if not np.isfinite(predictions[model][i]):
            continue
        fitted_log = float(predictions[model][i] + add_back[i])
        warranted_multiple = float(math.exp(fitted_log))
        reads[(obs.ticker, obs.as_of)] = WarrantedRead(
            ticker=obs.ticker,
            as_of=obs.as_of,
            sub_vertical=obs.sub_vertical,
            actual_multiple=obs.multiple,
            warranted_multiple=warranted_multiple,
            residual_log=obs.log_multiple - fitted_log,
            residual_turns=obs.multiple - warranted_multiple,
            within_date_sd=rerating.within_date_sd,
            out_of_sample=True,
        )

    ics: dict[date, float] = {}
    for when in sorted(set(dates.tolist())):
        mask = scored & (dates == when)
        if int(mask.sum()) >= _MIN_NAMES_PER_DATE:
            value = spearman(y[mask], predictions[model][mask])
            if value is not None:
                ics[when] = value

    card = ModelCard(
        name=f"warranted {TARGETS[panel.target][1]} ({model})",
        task=(
            f"fitted {TARGETS[panel.target][1]} from fundamentals across the TMT "
            "universe, and the out-of-sample residual against it"
        ),
        trained_through=max(fold.train_end for fold in folds),
        # The last fold's training window, which is the largest of them and the
        # one the coefficients reported here came from. The count of observations
        # actually SCORED is a different number and lives on the EvalResult, where
        # a reader comparing a score against a sample size will look for it.
        n_train=max(fold.n_train for fold in folds),
        features=[*features, "missing_share", *SUB_VERTICAL_COLUMNS],
        hyperparameters={
            "model": model,
            "demean_by_date": cfg.demean_by_date,
            "hidden": list(_MLP_HIDDEN) if model == "mlp" else None,
            "ridge_lambda": _RIDGE_LAMBDA if model == "ridge" else None,
            "folds": len(folds),
            "embargo_days": 0,
            "random_seed": seed,
        },
        evaluation=strongest,
        limitations=[
            "Reflexive. Fitted on multiples, so it learns the market's own pricing "
            "rule and can say a company is priced unlike its characteristics. It "
            "cannot say the market is wrong. A DCF can; this is not one.",
            f"Survivorship. The universe is the {len(panel.tickers)} names that still "
            "exist and still have a price history, so the delisted and the acquired "
            "are absent and they were disproportionately the cheap ones.",
            f"{len(panel.skips):,} company-dates produced no observation. See "
            "ObservationPanel.selection for the reasons, and read any result as "
            "being about the sample that survived them.",
            "Quarterly observations of the same companies are not independent draws. "
            "The effective sample is far smaller than the observation count and no "
            "standard error here is corrected for it.",
            "Most of the headline score is company identity. A company's features "
            "barely move quarter to quarter and neither does its relative multiple, "
            "so a model fitted on the past is rewarded for recognising the company. "
            "WarrantedModel.change_rank_correlation is the part that is not, and it "
            "is small.",
        ],
        notes=list(notes),
    )
    card.notes.append(f"Strongest baseline: {toughest}.")

    return WarrantedModel(
        card=card,
        target=panel.target,
        demeaned=cfg.demean_by_date,
        rerating=rerating,
        reads=reads,
        baselines=baselines,
        folds=folds,
        coefficients=coefficients,
        information_coefficients=ics,
        change_rank_correlation=change,
        n_changes=n_change,
        notes=notes,
    )


def _to_relative(turns: np.ndarray, add_back: np.ndarray | None) -> np.ndarray:
    """A warranted multiple in turns, put on the same footing as the fitted target.

    Logged, then demeaned by the same leave-one-out date mean the model's own
    target was demeaned by, because comparing a level against a relative position
    would make the baseline look terrible for a reason that has nothing to do
    with the baseline.
    """
    out = np.full(turns.shape, np.nan)
    positive = np.isfinite(turns) & (turns > 0)
    out[positive] = np.log(turns[positive])
    if add_back is not None:
        out[positive] -= add_back[positive]
    return out
