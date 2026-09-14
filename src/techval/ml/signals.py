"""Does any score in this package predict anything? The adversary that finds out.

Every other model in this wave produces a number. A warranted-multiple residual,
a fade forecast error, an acquisition probability: each one is easy to believe
and none of them is yet evidence of anything. This module takes any score at all,
as ``(ticker, date, value)``, and asks the only question that settles it. Was the
score related to what the stock did next?

The generic triple is deliberate. Nothing here imports another model, so nothing
here can be tuned to flatter one, and every model in the package can be tested
through the same harness on the same conventions. A harness that knows what it is
scoring is a harness that can be made to agree with it.

**The answer arrives as an information coefficient, a bucket table and a turnover
figure, and the three say different things.** The information coefficient is the
cross-sectional rank correlation between the score and the forward return, one
per rebalance date, and it measures whether the whole ordering was right. The
bucket table sorts each date into quantiles and reports what each earned, which
separates a signal that is monotone across the cross-section from one that works
only in its extreme bucket: the first is a factor, the second is a screen with a
handful of names behind it and a much shorter life expectancy. Turnover says what
it costs to own. A small edge with high turnover is a costless-trading illusion,
and the break-even cost reported beside the spread is the number that pops it.

**The t-statistic on an overlapping-window information coefficient is roughly
double what it should be, and this is what most of the file exists to fix.**
Twelve month returns sampled quarterly share eleven months of their path. Two
adjacent coefficients are therefore near-repeats of one another rather than two
draws, the series is heavily autocorrelated, and dividing the mean by
``sd / sqrt(T)`` counts each observation about four times. ``newey_west_se``
computes the standard error that accounts for it, with a Bartlett kernel and a
lag matched to the overlap, and both numbers are always reported together with
the ratio between them. Where the naive statistic clears two and the corrected
one does not, ``SignalResult.verdict`` says the result is not significant, in
those words, and the naive figure is kept beside it so a reader can see exactly
how much of the original claim was arithmetic.

The lag is ``ceil(horizon / rebalance spacing) - 1``, which is three for a twelve
month horizon sampled quarterly: coefficients four steps apart no longer share
any of their path. Bartlett weights rather than the uniform weights of
Hansen-Hodrick, because Bartlett guarantees a non-negative variance estimate and
Hansen-Hodrick does not, and a standard error that comes out imaginary is worse
than one that is slightly conservative. The correction is not always an inflation
either. Where the coefficients alternate in sign the autocovariances are negative
and the corrected standard error is the smaller of the two, which is a real
result and is reported as found rather than assumed away.

**Survivorship and delisting are the mirror-image biases, and only one of them
is usually noticed.** A company acquired six months into a holding period has no
twelve month return in the usual sense, and dropping it deletes precisely the
outcome with the largest return in the sample: a takeout at a 40% premium. Every
such name is instead terminated at the deal, using the offer price from
``tmt.precedents`` where the filings pin one down and the last traded close where
they do not, and the remaining months are carried at the benchmark where a
benchmark is supplied and at cash where it is not. Leaving those months out
entirely would put a six month holding period into the same statistic as a twelve
month one, which is the thing ``backtest.forward_return`` refuses to do.

The mirror image is the bankruptcy and the plain delisting, and this harness does
not silently guess at either. A name whose price series stops while the rest of
the universe continues is recorded as delisted, counted, and excluded from the
statistics unless the caller states a delisting return. That default is itself a
bias and the flag says so out loud: excluding failures raises every number in the
table. The literature's usual figures are about -30% for a delisting from a major
exchange and about -55% from Nasdaq, from Shumway (1997) and Shumway and Warther
(1999), and they are offered as arguments rather than applied behind the reader's
back. The distinction this module does insist on is between a name that stopped
while others carried on, which is a delisting, and one that stops because the
whole sample stops, which is ordinary right-censoring at the edge of the data and
means nothing at all. Counting the second as the first would manufacture a wave
of failures every time the data ended.

**Point-in-time scores, refused rather than warned about.** A score computed with
any knowledge that postdates its own date is hindsight, and a harness that
accepts one will certify hindsight as skill: the fit looks superb and the failure
is silent, which is the worst combination available. Every score is passed
through ``features.assert_point_in_time`` and a score carrying a knowledge date
later than its own date is refused outright. This is also the one module in the
package where a price series running past its row date is required rather than
forbidden, because the forward return is the label. The two rules face opposite
directions and both are enforced here.

**The multiple-comparisons problem, which is where honest harnesses go to die.**
Run one signal against one horizon on one universe and a t-statistic of two means
what the textbook says. Run four models against three horizons on two universes
and you have run twenty-four tests, of which better than one is expected to clear
that bar on noise alone, and the one you will report is the one that did. Nothing
about the winner's own arithmetic reveals this; it is a property of the search,
not of the result, and it cannot be recovered from the result afterwards. So
``test_signal`` takes ``n_tests_run`` and reports the Sidak-adjusted p-value
beside the raw one, and the count is the caller's to state honestly. The only
real corrections are to pre-register the test before looking, or to discount the
best result by the number of attempts that produced it. Reporting the best of
twenty-four as though it were the only one is not a rounding error, it is the
single most common way a backtest lies.

**Sample size, plainly.** Roughly a hundred technology companies over ten years
of usable price history, rebalanced quarterly, is about forty cross-sections and
something like ten genuinely independent twelve month periods. It is not four
thousand observations, whatever the row count says, and the standard error must
reflect the number of independent periods rather than the number of company
dates. Both counts are on every result, the second one first. The cross-section
is not free either: a hundred technology names in one quarter load on one factor
and one rate cycle, so a single date is closer to one observation than to a
hundred. That is why the statistic of record here is the time series of
coefficients and not the pooled correlation across every company date, which
would divide by the square root of four thousand and print a t-statistic of six
on nothing.

**What beating the baseline means here.** The null is not zero. Zero is what a
coefficient would average over infinite data, and this is not infinite data: a
random score on forty cross-sections of a hundred names has a mean coefficient
that wanders, and the width of that wandering is the bar. The baseline is
therefore a random score with the same cross-sectional shape as the real one,
built by permuting the real scores within each date so the distribution, the
ties and the count are preserved exactly, and evaluated through the identical
code path. The permutation is repeated and the share of draws reaching the
observed coefficient is an empirical p-value that needs no distributional
assumption at all.

One caveat on that p-value, because it points the same way as the naive
t-statistic. Permuting within a date leaves each date's coefficient independent
of every other by construction, so the permutation distribution does not
reproduce the autocorrelation that overlapping windows create, and it is
anti-conservative for exactly the reason the naive standard error is. It is the
right null for whether any cross-sectional relation exists and the wrong one for
how precisely the mean is measured. The Newey-West interval is the statistic of
record, the moving-block bootstrap beside it is a second opinion that does
respect the autocorrelation, and all three are printed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import LookaheadError, assert_no_lookahead, forward_return
from ..config import Assumptions
from ..edgar import CompanyFacts, Provenance
from ..errors import ConfigError, NotMeaningfulError, TechvalError
from ..ev_bridge import build_ev_bridge
from ..financials import build_financials
from ..market import PriceSeries
from .features import FeatureRow, assert_point_in_time
from .protocol import EvalResult, ModelCard, spearman

# Calendar days per month, used to turn a horizon in months into the horizon in
# days that the price layer counts in. The average Gregorian month, so that four
# quarters land on a year rather than eleven days short of one.
DAYS_PER_MONTH = 365.25 / 12.0

# Refusal and flag thresholds on the number of DATES, which is the sample size
# that matters. Eight coefficients is the floor below which a Newey-West
# correction at a lag of three is estimating four autocovariances from eight
# points and is fiction. Twenty-four is six years of quarterly rebalancing and
# is still thin; above it the interval is worth reading.
MIN_DATES = 8
THIN_DATES = 24

# A price series that stops this many days before the rest of the panel stopped
# is a delisting, not the edge of the sample. Thirty days is a month of missing
# closes while a hundred other names carried on printing, which no live ticker
# does. Below it the gap is a stale feed or a holiday run, and calling that a
# delisting would invent a failure.
DELISTING_GAP_DAYS = 30

# Permutation draws for the baseline and resamples for the block bootstrap.
# Four hundred permutations resolve an empirical p-value to about half a point,
# which is finer than anything the rest of the sample supports.
DEFAULT_BASELINE_DRAWS = 400
DEFAULT_BOOTSTRAP_DRAWS = 1000

# Two-sided 5%, on the normal rather than the t. Every standard error here is
# asymptotic already, so pretending to a t distribution with an exact degrees of
# freedom would add a decimal of false precision to an interval whose real
# uncertainty is the lag choice.
SIGNIFICANT_T = 1.959963984540054

# Delisting returns from the literature, offered as arguments and never applied
# unless the caller names one. Shumway (1997) puts the average delisting return
# on the major exchanges near -30%, and Shumway and Warther (1999) put Nasdaq
# near -55%. Both are averages over performance delistings in the 1990s and
# neither is a fact about a 2020s technology company; they are here so that a
# sensitivity can be run against a number somebody published rather than a
# number somebody invented.
DELISTING_CONVENTIONS: dict[str, float] = {
    "total_loss": -1.0,
    "shumway_nyse": -0.30,
    "shumway_nasdaq": -0.55,
}

_EXIT_HELD = "held"
_EXIT_ACQUIRED = "acquired"
_EXIT_DELISTED = "delisted"
_EXIT_CENSORED = "censored"
_EXIT_NO_PRICE = "no price"


# --------------------------------------------------------------------------- #
# What goes in
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Score:
    """One model's opinion about one company on one date.

    ``value`` is on whatever scale the model produces. Nothing here reads its
    level, only its cross-sectional ordering, so a residual in log points and a
    probability in [0, 1] are tested by identical code.

    The sign convention is the caller's and the harness will not guess at it. A
    positive coefficient means high scores earned high returns, so a model whose
    high scores are meant to be sells must be handed in negated, and the label
    should say which way round it was handed in. Choosing the sign after seeing
    the coefficient is a second test masquerading as a convention.

    ``knowledge_date`` is the date the client that built the score was pinned to
    and ``provenance`` is the filing record behind it. Both are optional because
    a score can arrive from anywhere, and both are checked where present: a score
    that can prove it was point in time is worth more than one that asserts it.
    """

    ticker: str
    as_of: date
    value: float
    knowledge_date: date | None = None
    provenance: Mapping[str, Provenance] = field(default_factory=dict)
    note: str | None = None


@dataclass(frozen=True)
class DealTerm:
    """An acquisition that ends a holding period, with what the holder received.

    ``offer_price`` is dollars per share and is what the terminating return is
    struck on where the filings pin one down. It is None for a stock deal whose
    value moves with the acquirer, and for a deal whose consideration the parser
    could not fix, in which case the harness falls back to the last traded close.
    That fallback is close but not equal: a target trades a few points below the
    offer until the deal closes, so the fallback understates the realised return
    by the arbitrage spread, and the note on the outcome says which was used.
    """

    ticker: str
    announced: date
    offer_price: float | None = None
    completed: bool = True
    source: str = ""


def deal_terms_from_events(
    events: Iterable[tuple[str, date, bool]],
) -> list[DealTerm]:
    """Adapt ``tmt.precedents.deal_events`` output, which carries no price.

    ``deal_events`` answers who was bid for and when, which is what an M&A
    propensity label needs. It does not answer what the holder received, so every
    term built here has ``offer_price`` None and every terminating return falls
    back to the last traded close. Pass ``Transaction`` objects instead where the
    price matters, which is whenever the deal is in cash.
    """
    return [
        DealTerm(ticker=str(t).upper(), announced=d, completed=bool(c), source="deal_events")
        for t, d, c in events
    ]


def deal_terms_from_transactions(transactions: Iterable[Any]) -> list[DealTerm]:
    """Adapt ``tmt.precedents.Transaction`` objects, which do carry a price.

    Anything without an announcement date is skipped: a transaction the parser
    identified but could not date cannot terminate a holding period, because
    there is no date at which to terminate it.
    """
    out: list[DealTerm] = []
    for txn in transactions:
        announced = getattr(txn, "announced", None)
        if announced is None:
            continue
        out.append(
            DealTerm(
                ticker=str(getattr(txn, "target_ticker", "")).upper(),
                announced=announced,
                offer_price=getattr(txn, "offer_price", None),
                completed=bool(getattr(txn, "completed", False)),
                source=str(getattr(txn, "source_form", "") or "precedents"),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# What comes out
# --------------------------------------------------------------------------- #


@dataclass
class Outcome:
    """One score paired with what happened to the stock afterwards.

    ``exit_kind`` is the whole survivorship argument in one field. ``held`` is an
    ordinary holding period that ran its course. ``acquired`` was terminated at a
    deal and is the outcome whose silent removal biases every statistic upward.
    ``delisted`` stopped trading while the rest of the panel carried on and is
    the mirror image, biasing everything upward when it is dropped and downward
    when it is assumed to be a total loss. ``censored`` ran into the end of the
    price data and carries no information about anything.

    ``held_days`` is the realised holding period, which equals the horizon for an
    ordinary outcome and is shorter for a terminated one. It is on the record
    because a statistic pooling six and twelve month holding periods is measuring
    the calendar as much as the model.
    """

    ticker: str
    as_of: date
    score: float
    entry_price: float | None = None
    exit_price: float | None = None
    exit_date: date | None = None
    forward_return: float | None = None
    exit_kind: str = _EXIT_HELD
    held_days: int | None = None
    note: str | None = None

    @property
    def scored(self) -> bool:
        return self.forward_return is not None


@dataclass
class ICSeries:
    """The coefficient series and every error bar anyone could reasonably want.

    ``naive_se`` divides the dispersion by the square root of the number of
    dates, which is right only if the dates are independent draws. Under
    overlapping windows they are not, ``newey_west_se`` is the one to quote, and
    ``inflation`` is the ratio that says how much the naive figure was flattering
    the result.

    ``share_positive`` is the share of dates whose coefficient was above zero,
    and it is the more robust of the two headline numbers. A mean coefficient can
    be carried by one quarter in which everything worked; a share above a half
    over forty quarters cannot.
    """

    dates: list[date]
    values: list[float]
    counts: list[int]
    lag: int
    spacing_days: float
    autocorrelations: list[float] = field(default_factory=list)
    block_bootstrap_se: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else float("nan")

    @property
    def sd(self) -> float:
        if len(self.values) < 2:
            return float("nan")
        return float(np.std(self.values, ddof=1))

    @property
    def naive_se(self) -> float:
        return self.sd / math.sqrt(self.n) if self.n else float("nan")

    @property
    def newey_west_se(self) -> float:
        se, _ = newey_west_se(np.asarray(self.values, dtype=float), self.lag)
        return se

    @property
    def t_naive(self) -> float:
        se = self.naive_se
        return self.mean / se if se and math.isfinite(se) and se > 0 else float("nan")

    @property
    def t_newey_west(self) -> float:
        se = self.newey_west_se
        return self.mean / se if se and math.isfinite(se) and se > 0 else float("nan")

    @property
    def inflation(self) -> float:
        """How many times too large the naive t-statistic was.

        Above one the naive figure was flattering the result, which is the usual
        direction under overlapping windows. Below one the coefficients alternate
        in sign, the autocovariances are negative, and the naive figure was the
        conservative one. Both happen and neither is assumed.
        """
        nw, naive = self.newey_west_se, self.naive_se
        if not (math.isfinite(nw) and math.isfinite(naive)) or naive <= 0:
            return float("nan")
        return nw / naive

    @property
    def share_positive(self) -> float:
        if not self.values:
            return float("nan")
        return float(np.mean(np.asarray(self.values) > 0.0))

    @property
    def effective_n(self) -> float:
        """Dates the Newey-West variance says the series is really worth.

        The ratio of the naive variance to the corrected one, times the number of
        dates. Forty quarterly coefficients on a twelve month horizon typically
        come back worth about ten, which is the number of non-overlapping years
        in the sample, and seeing the two agree is the check that the lag was set
        sensibly.
        """
        infl = self.inflation
        if not math.isfinite(infl) or infl <= 0:
            return float("nan")
        return self.n / (infl * infl)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Names": self.counts, "IC": self.values},
            index=pd.Index(self.dates, name="Date"),
        )


_BUCKET_COLUMNS = [
    "Names per date",
    "Mean score",
    "Mean return",
    "Median of date means",
    "Share of dates positive",
]


@dataclass
class BucketTable:
    """Equal-weight returns by score bucket, and whether they line up.

    Bucket one holds the lowest scores. ``spread`` is the top bucket less the
    bottom one, per date, averaged, and it carries its own Newey-West standard
    error because a spread series built from overlapping windows is exactly as
    autocorrelated as the coefficient series is.

    ``monotonicity`` is the rank correlation between bucket number and mean
    return, which on five buckets can only take a handful of values and should be
    read as a direction rather than a measurement. ``monotone_steps`` is the
    share of adjacent pairs that rise, and between them they separate a factor
    from a screen: a signal that is monotone across the cross-section is a
    different animal from one that lives entirely in its extreme bucket, and the
    second is usually a handful of names and a shorter life expectancy.
    """

    frame: pd.DataFrame
    spread_by_date: list[float]
    spread_dates: list[date]
    lag: int
    monotonicity: float | None
    monotone_steps: float
    n_dates: int
    notes: list[str] = field(default_factory=list)

    @property
    def spread(self) -> float:
        return float(np.mean(self.spread_by_date)) if self.spread_by_date else float("nan")

    @property
    def spread_se(self) -> float:
        """Not a number where the table was too thin to build, rather than a raise.

        Reached from ``verdict``, which has to print something for a sample that
        carried a coefficient series and no bucket table. That happens when the
        caller asks for more buckets than the thinnest cross-section can fill.
        """
        if len(self.spread_by_date) <= self.lag or len(self.spread_by_date) < 2:
            return float("nan")
        se, _ = newey_west_se(np.asarray(self.spread_by_date, dtype=float), self.lag)
        return se

    @property
    def spread_t(self) -> float:
        se = self.spread_se
        return self.spread / se if se and math.isfinite(se) and se > 0 else float("nan")


@dataclass
class TurnoverReport:
    """How much of each extreme bucket is replaced at every rebalance.

    ``top`` and ``bottom`` are one-sided: the share of the bucket's names that
    were not in it at the previous rebalance. ``churn`` is the share of the whole
    scored universe that was not scored at the previous rebalance, and it has to
    be read beside the other two, because a name that left the universe rather
    than the bucket is not a trade anybody chose to make.

    ``breakeven_cost_bps`` is the one-way trading cost that would erase the
    bucket spread, under the plainest implementation: rebalance at every date,
    hold the long-short to the next one, pay the cost on both legs. It uses the
    spread measured over the full horizon as though it were the annual figure,
    which is exact when the horizon is a year and an approximation otherwise, and
    it is the number that decides whether an edge survives contact with a
    broker. It is None where the spread is not positive, because a signal that
    loses money before costs has no break-even cost, and printing a negative one
    invites it to be read as a threshold.
    """

    top: float
    bottom: float
    churn: float
    rebalances_per_year: float
    n_pairs: int
    breakeven_cost_bps: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return 0.5 * (self.top + self.bottom)


@dataclass
class SignalResult:
    """Everything the harness found, including what it refused to compute.

    ``evaluation`` is the honest comparison and ``verdict`` is the line to read
    first. ``checks`` holds one line per thing worth knowing about the sample,
    with the ones that failed a threshold prefixed ``FLAG:``, following the
    convention ``backtest.BacktestResult`` already uses.
    """

    label: str
    outcomes: list[Outcome]
    ic: ICSeries
    buckets: BucketTable
    turnover: TurnoverReport
    evaluation: EvalResult
    card: ModelCard
    horizon_months: int
    n_tests_run: int = 1
    permutation_p: float | None = None
    baseline_draws: int = 0
    notes: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    # -- counts ------------------------------------------------------------ #

    @property
    def n_scored(self) -> int:
        return sum(1 for o in self.outcomes if o.scored)

    def exit_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.outcomes:
            out[o.exit_kind] = out.get(o.exit_kind, 0) + 1
        return out

    @property
    def n_independent_periods(self) -> float:
        """Non-overlapping horizons the calendar span actually contains.

        The span of the rebalance dates divided by the horizon. This is the
        number the standard error should reflect, and it is usually one order of
        magnitude below the number of dates and two below the number of company
        dates.
        """
        if len(self.ic.dates) < 2:
            return 0.0
        span = (max(self.ic.dates) - min(self.ic.dates)).days
        horizon = self.horizon_months * DAYS_PER_MONTH
        return span / horizon if horizon else 0.0

    # -- the answer -------------------------------------------------------- #

    @property
    def p_naive(self) -> float:
        return two_sided_p(self.ic.t_naive)

    @property
    def p_newey_west(self) -> float:
        return two_sided_p(self.ic.t_newey_west)

    @property
    def p_adjusted(self) -> float:
        """Sidak-adjusted for the number of tests the caller says were run.

        ``1 - (1 - p) ** m``, which is Bonferroni's ``p * m`` without the
        overshoot at large m and is the same thing to three decimals at the
        counts anybody actually runs.
        """
        p = self.p_newey_west
        if not math.isfinite(p):
            return p
        return 1.0 - (1.0 - p) ** max(1, self.n_tests_run)

    @property
    def significant(self) -> bool:
        """Corrected for overlap and for the number of tests, both."""
        p = self.p_adjusted
        return math.isfinite(p) and p < 0.05

    def verdict(self) -> str:
        """The line a reader can act on, including when the answer is no."""
        ic = self.ic
        parts = [
            f"{self.label}: mean IC {ic.mean:+.4f} over {ic.n} dates "
            f"({self.n_scored:,} company-dates, about "
            f"{self.n_independent_periods:.0f} independent {self.horizon_months} "
            f"month periods). Share of dates positive {ic.share_positive:.0%}."
        ]
        parts.append(
            f"t = {ic.t_naive:+.2f} naive, {ic.t_newey_west:+.2f} Newey-West at "
            f"lag {ic.lag}; the correction moved the standard error by "
            f"{ic.inflation:.2f}x, leaving about {ic.effective_n:.0f} effective "
            f"observations."
        )
        if abs(ic.t_naive) >= SIGNIFICANT_T and abs(ic.t_newey_west) < SIGNIFICANT_T:
            parts.append(
                "The naive statistic clears two and the corrected one does not, "
                "so the honest answer is NOT SIGNIFICANT. The naive figure is "
                "printed above only so the size of the difference is visible."
            )
        elif abs(ic.t_newey_west) >= SIGNIFICANT_T:
            parts.append(
                f"Significant after the overlap correction at p = "
                f"{self.p_newey_west:.4f}."
            )
        else:
            parts.append(
                f"Not significant either way, at p = {self.p_newey_west:.3f} "
                "after the overlap correction."
            )
        if self.n_tests_run > 1:
            parts.append(
                f"{self.n_tests_run} tests were run; the Sidak-adjusted p-value "
                f"is {self.p_adjusted:.3f}, and the honest correction is to "
                "pre-register the test or to discount this result by the number "
                "of attempts behind it."
            )
        parts.append(
            f"Top-bottom spread {self.buckets.spread:+.2%} over "
            f"{self.horizon_months} months (t = {self.buckets.spread_t:+.2f}), "
            f"monotone across {self.buckets.monotone_steps:.0%} of the steps."
        )
        if self.turnover.breakeven_cost_bps is not None:
            if self.buckets.spread > 0:
                parts.append(
                    f"Turnover {self.turnover.mean:.0%} of each extreme bucket "
                    f"per rebalance, so a one-way cost above "
                    f"{self.turnover.breakeven_cost_bps:.0f}bp erases the spread."
                )
            else:
                parts.append(
                    f"Turnover {self.turnover.mean:.0%} of each extreme bucket "
                    "per rebalance, on a spread that is negative before any cost "
                    "at all, so there is nothing for a break-even cost to erase."
                )
        # ``EvalResult.verdict`` opens on the metric's name in lower case, which
        # is right for a line of its own and wrong after a full stop, so the
        # first letter is raised here rather than in the protocol.
        comparison = self.evaluation.verdict()
        parts.append(comparison[:1].upper() + comparison[1:])
        return " ".join(parts)

    def rows(self) -> list[tuple[str, Any]]:
        """Headline rows, counts before coefficients. Formatting lives in the CLI."""
        counts = self.exit_counts()
        out: list[tuple[str, Any]] = [
            ("Signal", self.label),
            ("Horizon (months)", self.horizon_months),
            ("Rebalance dates scored", self.ic.n),
            ("Company-dates scored", self.n_scored),
            ("Independent periods", round(self.n_independent_periods, 1)),
            ("Effective observations (Newey-West)", round(self.ic.effective_n, 1)),
            ("Held to horizon", counts.get(_EXIT_HELD, 0)),
            ("Terminated at a deal", counts.get(_EXIT_ACQUIRED, 0)),
            ("Delisted", counts.get(_EXIT_DELISTED, 0)),
            ("Censored at the data edge", counts.get(_EXIT_CENSORED, 0)),
            ("Mean IC", self.ic.mean),
            ("Share of dates positive", self.ic.share_positive),
            ("IC t, naive", self.ic.t_naive),
            ("IC t, Newey-West", self.ic.t_newey_west),
            ("Standard error inflation", self.ic.inflation),
            ("Top-bottom spread", self.buckets.spread),
            ("Spread t, Newey-West", self.buckets.spread_t),
            ("Monotone steps", self.buckets.monotone_steps),
            ("Turnover per rebalance", self.turnover.mean),
            ("Tests run", self.n_tests_run),
            ("p, Newey-West", self.p_newey_west),
            ("p, Sidak-adjusted", self.p_adjusted),
        ]
        if self.permutation_p is not None:
            out.append(("p, permutation", self.permutation_p))
        return out

    def frame(self) -> pd.DataFrame:
        """One row per attempted company-date, failures kept.

        Dropping the unscored rows would make the table agree with the statistics
        while disagreeing with the run, and the count of what could not be scored
        is part of the result.
        """
        return pd.DataFrame(
            [
                {
                    "Ticker": o.ticker,
                    "As of": o.as_of,
                    "Score": o.score,
                    "Entry": o.entry_price,
                    "Exit": o.exit_price,
                    "Exit date": o.exit_date,
                    "Held days": o.held_days,
                    "Forward return": o.forward_return,
                    "Exit kind": o.exit_kind,
                    "Note": o.note,
                }
                for o in self.outcomes
            ]
        )


# --------------------------------------------------------------------------- #
# The statistics
# --------------------------------------------------------------------------- #


def newey_west_se(x: np.ndarray, lag: int) -> tuple[float, list[float]]:
    """Standard error of a mean whose observations overlap, and the rho behind it.

    The variance of a sample mean is the sum of every autocovariance in the
    series, not just the variance. Ignoring the rest is exactly right when the
    observations are independent and exactly wrong when each one shares most of
    its path with its neighbour, which is what sampling a twelve month return
    quarterly does.

        S = g0 + 2 * sum_j w_j g_j,    w_j = 1 - j / (lag + 1)
        se = sqrt(S / T)

    ``g_j`` is the sample autocovariance at lag j, divided by T rather than by
    T - j, which is the convention that keeps the estimator non-negative. The
    Bartlett weights ``w_j`` taper to zero at the truncation point and guarantee
    a non-negative S; the uniform weights of Hansen-Hodrick do not, and a
    variance estimate that comes out negative leaves the caller with nothing.

    A lag of zero returns the ordinary standard error, so the naive figure and
    the corrected one come from one code path and cannot drift apart.

    The estimator is asymptotic. At the forty-odd dates this package can offer it
    is biased downward, which means the correction reported here is an
    understatement of the correction that is really needed, not an overstatement.
    The moving-block bootstrap in ``block_bootstrap_se`` is the second opinion.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        raise NotMeaningfulError(
            f"a standard error needs at least two observations, got {n}"
        )
    if lag < 0:
        raise ConfigError(f"the Newey-West lag must not be negative, got {lag}")
    if lag >= n:
        raise ConfigError(
            f"a Newey-West lag of {lag} on {n} observations estimates more "
            "autocovariances than there are gaps between the points. Shorten the "
            "horizon, widen the rebalance spacing, or collect more dates."
        )

    dev = x - x.mean()
    g0 = float(dev @ dev) / n
    total = g0
    rhos: list[float] = []
    for j in range(1, lag + 1):
        gj = float(dev[j:] @ dev[:-j]) / n
        rhos.append(gj / g0 if g0 > 0 else 0.0)
        total += 2.0 * (1.0 - j / (lag + 1)) * gj
    total = max(total, 0.0)
    return math.sqrt(total / n), rhos


def block_bootstrap_se(
    x: np.ndarray, block: int, draws: int, rng: np.random.Generator
) -> float | None:
    """Standard error of the mean from resampled contiguous blocks.

    Resampling single observations destroys the autocorrelation that is the whole
    problem and reproduces the naive standard error. Resampling contiguous blocks
    longer than the overlap keeps it: each block carries its own internal
    dependence, and blocks drawn from different parts of the sample are close
    enough to independent.

    This is a second opinion on Newey-West rather than a replacement. It makes no
    assumption about the kernel and it inherits the same small-sample problem
    from the other end, since forty observations in blocks of four are ten
    blocks, and ten blocks is not a bootstrap anybody should lean on alone.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    block = max(1, int(block))
    if n < 2 or block > n:
        return None
    n_blocks = int(math.ceil(n / block))
    last_start = n - block
    means = np.empty(draws, dtype=float)
    for d in range(draws):
        starts = rng.integers(0, last_start + 1, size=n_blocks)
        sample = np.concatenate([x[s : s + block] for s in starts])[:n]
        means[d] = sample.mean()
    return float(np.std(means, ddof=1))


def two_sided_p(t: float) -> float:
    """Two-sided normal p-value, from the error function rather than a dependency.

    The normal rather than the t distribution because every standard error in
    this module is asymptotic already. Quoting a t with an exact degrees of
    freedom would put a decimal of false precision on an interval whose real
    uncertainty is the choice of lag.
    """
    if not math.isfinite(t):
        return float("nan")
    return math.erfc(abs(t) / math.sqrt(2.0))


def overlap_lag(dates: Sequence[date], horizon_months: int) -> tuple[int, float]:
    """The Newey-West lag matched to the overlap, and the spacing behind it.

    Coefficients k rebalances apart share part of their path whenever k times the
    spacing is less than the horizon, so the last lag that still overlaps is
    ``ceil(horizon / spacing) - 1``: three for twelve months sampled quarterly,
    eleven for twelve months sampled monthly, zero for a horizon no longer than
    the spacing, which is the non-overlapping case where no correction is due.

    The spacing is measured in months from the median gap between consecutive
    dates and rounded, rather than taken in days. Quarterly dates pinned to the
    fifteenth are 90, 91 and 92 days apart, and a day-count formula flips between
    a lag of three and a lag of four depending on which quarters the sample
    happens to contain. The lag should not move because February was short.
    """
    unique = sorted(set(dates))
    if len(unique) < 2:
        return 0, float("nan")
    gaps = [(b - a).days for a, b in zip(unique, unique[1:])]
    spacing_days = float(np.median(gaps))
    spacing_months = max(1, int(round(spacing_days / DAYS_PER_MONTH)))
    lag = int(math.ceil(horizon_months / spacing_months)) - 1
    return max(0, lag), spacing_days


# --------------------------------------------------------------------------- #
# Pairing a score with what happened next
# --------------------------------------------------------------------------- #


def _as_scores(raw: Iterable[Any]) -> list[Score]:
    """Accept ``Score`` objects or bare ``(ticker, date, value)`` triples."""
    out: list[Score] = []
    for item in raw:
        if isinstance(item, Score):
            out.append(item)
            continue
        try:
            ticker, when, value = item
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "a score must be a Score or a (ticker, date, value) triple, got "
                f"{item!r}"
            ) from exc
        out.append(Score(ticker=str(ticker).upper(), as_of=when, value=float(value)))
    return out


def assert_scores_point_in_time(scores: Sequence[Score]) -> None:
    """Refuse a score series built with knowledge that postdates its own date.

    Two checks, because they catch different mistakes.

    The provenance of each score is walked by ``features.assert_point_in_time``,
    which compares the filing date behind every figure that reached the score
    against the score's own date. That is the check that catches a client built
    without a knowledge date, a cached payload reused from a later run, and a
    fixture loaded by the wrong helper, none of which any configuration check
    would notice.

    The knowledge date is then compared to the score date directly, which
    ``assert_point_in_time`` does not do. A score pinned to a date later than its
    own is a leak whether or not a late filing happened to reach it: it is
    correct today and wrong the moment the filer amends something, and a harness
    that certifies it is certifying a coin that has not landed yet. Pinning
    earlier than the score date is allowed, because using less than was knowable
    is conservative rather than dishonest.

    Refused, not warned about. A warning on a leak is read once and then lives in
    a log file while the number it invalidates goes into a memo.
    """
    for s in scores:
        if s.knowledge_date is not None and s.knowledge_date > s.as_of:
            raise LookaheadError(
                f"{s.ticker} carries a score dated {s.as_of} built through a "
                f"client pinned to {s.knowledge_date}, "
                f"{(s.knowledge_date - s.as_of).days} days later. Build the "
                "client with knowledge_date set to the score date."
            )
        if not s.provenance:
            continue
        assert_point_in_time(
            FeatureRow(
                ticker=s.ticker,
                as_of=s.as_of,
                knowledge_date=s.knowledge_date,
                provenance=dict(s.provenance),
            )
        )


def _price_lookup(
    prices: Mapping[str, PriceSeries] | Callable[[str], PriceSeries | None],
) -> Callable[[str], PriceSeries | None]:
    if callable(prices):
        return prices
    upper = {k.upper(): v for k, v in prices.items()}
    return lambda t: upper.get(t.upper())


def _close_at_or_after(series: PriceSeries, when: date) -> tuple[date, float] | None:
    for d, c in zip(series.dates, series.closes):
        if d >= when:
            return d, float(c)
    return None


def _resolve_outcome(
    score: Score,
    series: PriceSeries | None,
    horizon_days: int,
    deals: Mapping[str, list[DealTerm]],
    panel_last: date,
    delisting_return: float | None,
    benchmark: PriceSeries | None,
) -> Outcome:
    """Pair one score with the return that followed it, or say why there is none.

    The ordinary case runs through ``backtest.forward_return``, which resolves
    entry to the last close at or before the score date and exit to the first
    close at or after the horizon. Everything below is the handling of the cases
    where that produced no exit, and the whole survivorship argument lives in the
    order the branches are tried.
    """
    if series is None:
        return Outcome(
            ticker=score.ticker,
            as_of=score.as_of,
            score=score.value,
            exit_kind=_EXIT_NO_PRICE,
            note="no price series was supplied for this ticker",
        )

    fr = forward_return(score.ticker, score.as_of, series, horizon_days)
    if fr.entry_price is None:
        return Outcome(
            ticker=score.ticker,
            as_of=score.as_of,
            score=score.value,
            exit_kind=_EXIT_NO_PRICE,
            note=fr.note,
        )
    if fr.total_return is not None:
        return Outcome(
            ticker=score.ticker,
            as_of=score.as_of,
            score=score.value,
            entry_price=fr.entry_price,
            exit_price=fr.exit_price,
            exit_date=fr.exit_date,
            forward_return=fr.total_return,
            exit_kind=_EXIT_HELD,
            held_days=(fr.exit_date - fr.entry_date).days if fr.exit_date and fr.entry_date else None,
        )

    # No exit close inside the horizon. Either the company stopped trading or the
    # data did, and which of the two it was decides everything that follows.
    horizon_end = score.as_of + timedelta(days=horizon_days)
    deal = _deal_in_window(deals.get(score.ticker, []), score.as_of, horizon_end)
    if deal is not None:
        return _terminate_at_deal(score, series, fr.entry_price, deal, horizon_end, benchmark)

    stopped_early = (panel_last - series.last_date).days > DELISTING_GAP_DAYS
    if not stopped_early:
        return Outcome(
            ticker=score.ticker,
            as_of=score.as_of,
            score=score.value,
            entry_price=fr.entry_price,
            exit_kind=_EXIT_CENSORED,
            note=(
                f"the horizon ends {horizon_end}, past the {panel_last} end of the "
                "price data. Right-censored at the edge of the sample, which says "
                "nothing about the company."
            ),
        )

    note = (
        f"the price series stops {series.last_date}, "
        f"{(panel_last - series.last_date).days} days before the rest of the "
        "panel, with no merger agreement on record. A delisting or a failure."
    )
    if delisting_return is None:
        return Outcome(
            ticker=score.ticker,
            as_of=score.as_of,
            score=score.value,
            entry_price=fr.entry_price,
            exit_kind=_EXIT_DELISTED,
            note=note + " Excluded, because no delisting return was stated.",
        )
    return Outcome(
        ticker=score.ticker,
        as_of=score.as_of,
        score=score.value,
        entry_price=fr.entry_price,
        exit_price=None,
        exit_date=series.last_date,
        forward_return=float(delisting_return),
        exit_kind=_EXIT_DELISTED,
        held_days=(series.last_date - score.as_of).days,
        note=note + f" Scored at the stated delisting return of {delisting_return:+.0%}.",
    )


def _deal_in_window(
    terms: Sequence[DealTerm], start: date, end: date
) -> DealTerm | None:
    """The earliest announcement strictly after the score date and inside the horizon.

    Strictly after, because a deal announced on the score date is already in the
    entry price. Scoring the premium on a name the market had already repriced
    would credit the signal with a return no holder earned.
    """
    inside = [t for t in terms if start < t.announced <= end]
    return min(inside, key=lambda t: t.announced) if inside else None


def _terminate_at_deal(
    score: Score,
    series: PriceSeries,
    entry_price: float,
    deal: DealTerm,
    horizon_end: date,
    benchmark: PriceSeries | None,
) -> Outcome:
    """Terminate a holding period at the deal and carry the remainder forward.

    The offer price is used where the filings pinned one down, which is the price
    the holder of a cash deal actually received. Where they did not, the last
    traded close stands in: a target trades within a few points of the offer from
    announcement to close, so the substitute is close, and it understates the
    realised return by the arbitrage spread rather than overstating it.

    The months between the deal and the end of the horizon are then carried at
    the benchmark where one is supplied and at cash where one is not. Leaving
    them out would put a six month holding period into the same statistic as a
    twelve month one, and the whole point of a fixed horizon is that every
    observation is measured over the same clock.
    """
    if deal.offer_price is not None and deal.offer_price > 0:
        exit_price = float(deal.offer_price)
        basis = f"the {deal.offer_price:,.2f} offer price"
    else:
        exit_price = float(series.closes[-1])
        basis = (
            f"the last traded close of {exit_price:,.2f} on {series.last_date}, "
            "the filings carrying no per-share offer price"
        )
    deal_return = exit_price / entry_price - 1.0

    carry = 0.0
    carry_note = ""
    exit_date = deal.announced if deal.offer_price is not None else series.last_date
    if benchmark is not None and exit_date < horizon_end:
        start = _close_at_or_after(benchmark, exit_date)
        end = _close_at_or_after(benchmark, horizon_end)
        if start and end and start[1] > 0:
            carry = end[1] / start[1] - 1.0
            carry_note = (
                f" The {(horizon_end - exit_date).days} days to the end of the "
                f"horizon are carried at the benchmark, {carry:+.1%}."
            )
    if not carry_note and exit_date < horizon_end:
        carry_note = (
            f" The {(horizon_end - exit_date).days} days to the end of the "
            "horizon are carried at cash, which is zero here and understates a "
            "rising market."
        )

    return Outcome(
        ticker=score.ticker,
        as_of=score.as_of,
        score=score.value,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_date=exit_date,
        forward_return=(1.0 + deal_return) * (1.0 + carry) - 1.0,
        exit_kind=_EXIT_ACQUIRED,
        held_days=(horizon_end - score.as_of).days,
        note=(
            f"acquired, announced {deal.announced}. Terminated at {basis}, a "
            f"{deal_return:+.1%} return on the position." + carry_note
        ),
    )


# --------------------------------------------------------------------------- #
# Buckets and turnover
# --------------------------------------------------------------------------- #


def _bucket_of(values: np.ndarray, buckets: int) -> np.ndarray:
    """Bucket index from zero, cut on the ranked score and split as evenly as it goes.

    Cut on ranks rather than on equal-width score bands. Equal-width bands on a
    distribution with a long tail, which every valuation multiple has, put almost
    everything in one bucket and one name in another, which is a picture of the
    distribution rather than of the signal.

    The sort is stable, so tied scores land in the same buckets whatever order
    the run produced them in. Ties split across a bucket boundary are broken by
    arrival order rather than shared, which is the one place this differs from
    the rank correlation, and it matters only for a score with heavy ties.
    """
    order = np.argsort(values, kind="mergesort")
    out = np.empty(values.size, dtype=int)
    for b, idx in enumerate(np.array_split(order, buckets)):
        out[idx] = b
    return out


def _bucket_table(
    by_date: Mapping[date, list[Outcome]],
    dates: Sequence[date],
    buckets: int,
    lag: int,
) -> tuple[BucketTable, dict[date, dict[str, int]]]:
    """Equal-weight bucket returns, averaged across dates rather than pooled.

    Averaging the per-date bucket means gives every rebalance the same weight.
    Pooling every company-date instead would weight a quarter with a hundred
    names five times as heavily as one with twenty, which makes the table a
    statement about when the universe was large as much as about the signal.
    """
    per_date: dict[int, list[float]] = {b: [] for b in range(buckets)}
    per_date_score: dict[int, list[float]] = {b: [] for b in range(buckets)}
    per_date_n: dict[int, list[int]] = {b: [] for b in range(buckets)}
    membership: dict[date, dict[str, int]] = {}
    spreads: list[float] = []
    spread_dates: list[date] = []
    notes: list[str] = []
    skipped = 0

    for d in dates:
        rows = [o for o in by_date[d] if o.scored]
        if len(rows) < 2 * buckets:
            skipped += 1
            continue
        values = np.asarray([o.score for o in rows], dtype=float)
        rets = np.asarray([o.forward_return for o in rows], dtype=float)
        assign = _bucket_of(values, buckets)
        membership[d] = {o.ticker: int(b) for o, b in zip(rows, assign)}
        means: list[float] = []
        for b in range(buckets):
            sel = assign == b
            per_date[b].append(float(rets[sel].mean()))
            per_date_score[b].append(float(values[sel].mean()))
            per_date_n[b].append(int(sel.sum()))
            means.append(float(rets[sel].mean()))
        spreads.append(means[-1] - means[0])
        spread_dates.append(d)

    if skipped:
        notes.append(
            f"{skipped} date(s) carried fewer than {2 * buckets} scored names and "
            f"were left out of the bucket table: a quantile of one name is a "
            "name, not a quantile."
        )

    records = []
    for b in range(buckets):
        series = np.asarray(per_date[b], dtype=float)
        records.append(
            {
                "Names per date": float(np.mean(per_date_n[b])) if per_date_n[b] else float("nan"),
                "Mean score": float(np.mean(per_date_score[b])) if per_date_score[b] else float("nan"),
                "Mean return": float(series.mean()) if series.size else float("nan"),
                "Median of date means": float(np.median(series)) if series.size else float("nan"),
                "Share of dates positive": float(np.mean(series > 0)) if series.size else float("nan"),
            }
        )
    frame = pd.DataFrame(
        records, columns=_BUCKET_COLUMNS, index=pd.Index(range(1, buckets + 1), name="Bucket")
    )

    means = frame["Mean return"].to_numpy(dtype=float)
    mono = spearman(np.arange(1, buckets + 1, dtype=float), means)
    steps = np.diff(means)
    monotone_steps = float(np.mean(steps > 0)) if steps.size else float("nan")

    return (
        BucketTable(
            frame=frame,
            spread_by_date=spreads,
            spread_dates=spread_dates,
            lag=lag,
            monotonicity=mono,
            monotone_steps=monotone_steps,
            n_dates=len(spreads),
            notes=notes,
        ),
        membership,
    )


def _turnover(
    membership: Mapping[date, dict[str, int]],
    dates: Sequence[date],
    buckets: int,
    spacing_days: float,
    spread: float,
) -> TurnoverReport:
    """Share of each extreme bucket replaced at every rebalance, and what it costs.

    One-sided and measured on the bucket rather than on weights, because an
    equal-weight bucket has no weights worth tracking: the trade is the name
    entering or leaving. A name that left because it left the universe is counted
    separately in ``churn``, since nobody chose that trade and a harness that
    charges for it would penalise a signal for the shape of its own sample.
    """
    ordered = [d for d in dates if d in membership]
    top, bottom, churn = [], [], []
    for prev, now in zip(ordered, ordered[1:]):
        a, b = membership[prev], membership[now]
        top_now = {t for t, k in b.items() if k == buckets - 1}
        top_prev = {t for t, k in a.items() if k == buckets - 1}
        bot_now = {t for t, k in b.items() if k == 0}
        bot_prev = {t for t, k in a.items() if k == 0}
        if top_now:
            top.append(len(top_now - top_prev) / len(top_now))
        if bot_now:
            bottom.append(len(bot_now - bot_prev) / len(bot_now))
        if b:
            churn.append(len(set(b) - set(a)) / len(b))

    rebalances = 365.25 / spacing_days if spacing_days and math.isfinite(spacing_days) else float("nan")
    mean_top = float(np.mean(top)) if top else float("nan")
    mean_bottom = float(np.mean(bottom)) if bottom else float("nan")
    mean_turnover = 0.5 * (mean_top + mean_bottom)

    breakeven = None
    notes: list[str] = []
    if math.isfinite(mean_turnover) and mean_turnover > 0 and math.isfinite(rebalances) and spread > 0:
        # Two legs, each replacing mean_turnover of its names per rebalance, at
        # rebalances per year. One-way cost c is paid on each side of each
        # replacement, so annual cost is 2 * turnover * rebalances * c and the
        # break-even c is the spread divided by that.
        breakeven = spread / (2.0 * mean_turnover * rebalances) * 10_000.0
        notes.append(
            "The break-even cost treats the spread measured over the horizon as "
            "the annual figure and assumes the plainest implementation: "
            "rebalance at every date, hold the long-short to the next one, pay "
            "the cost on both legs. A twelve month signal rebalanced quarterly "
            "is really four overlapping sleeves and turns over less than this."
        )
    return TurnoverReport(
        top=mean_top,
        bottom=mean_bottom,
        churn=float(np.mean(churn)) if churn else float("nan"),
        rebalances_per_year=rebalances,
        n_pairs=len(top),
        breakeven_cost_bps=breakeven,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def test_signal(
    scores: Iterable[Any],
    prices: Mapping[str, PriceSeries] | Callable[[str], PriceSeries | None],
    *,
    assumptions: Assumptions | None = None,
    horizon_months: int | None = None,
    buckets: int | None = None,
    min_names_per_date: int | None = None,
    deals: Sequence[DealTerm] = (),
    delisting_return: float | str | None = None,
    benchmark: PriceSeries | None = None,
    n_tests_run: int = 1,
    baseline_draws: int = DEFAULT_BASELINE_DRAWS,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    lag: int | None = None,
    seed: int | None = None,
    label: str = "signal",
) -> SignalResult:
    """Did this score predict the next ``horizon_months`` of returns?

    ``scores`` is any iterable of ``Score`` objects or ``(ticker, date, value)``
    triples, and no part of this function knows or cares what produced them.
    ``prices`` is a mapping or a callable from ticker to ``PriceSeries``, which
    must run past the last score date by the horizon or the last dates come back
    censored.

    ``deals`` terminates holding periods at acquisitions, from
    ``deal_terms_from_transactions`` where the offer price is known and
    ``deal_terms_from_events`` where only the date is. ``delisting_return``
    states what a name that stopped trading with no deal on record earned: a
    float, a key of ``DELISTING_CONVENTIONS``, or None to exclude them and be
    told how many were excluded and which way that biases the answer.

    ``baseline_draws`` sets how many within-date permutations the null is built
    from, and it bounds the permutation p-value below at one over one more than
    that count. ``n_tests_run`` is the number of tests behind the one being
    reported and it is the caller's to state honestly. It changes no coefficient and only the
    p-value, and it is the difference between a result and a search result.

    Returns a ``SignalResult`` whose ``verdict`` is the line to read first,
    whether the answer is yes or no.
    """
    cfg = (assumptions or Assumptions()).ml
    sig = cfg.signals
    horizon_months = int(horizon_months if horizon_months is not None else sig.horizon_months)
    buckets = int(buckets if buckets is not None else sig.buckets)
    min_names = int(min_names_per_date if min_names_per_date is not None else sig.min_names_per_date)
    seed = int(seed if seed is not None else cfg.random_seed)
    if horizon_months < 1:
        raise ConfigError(f"the horizon must be at least one month, got {horizon_months}")
    if buckets < 2:
        raise ConfigError(f"a bucket table needs at least two buckets, got {buckets}")

    if isinstance(delisting_return, str):
        if delisting_return not in DELISTING_CONVENTIONS:
            raise ConfigError(
                f"unknown delisting convention {delisting_return!r}; the named "
                "ones are " + ", ".join(sorted(DELISTING_CONVENTIONS))
            )
        delisting_value: float | None = DELISTING_CONVENTIONS[delisting_return]
        delisting_label = delisting_return
    else:
        delisting_value = None if delisting_return is None else float(delisting_return)
        delisting_label = "none" if delisting_return is None else f"{delisting_value:+.0%}"

    parsed = _as_scores(scores)
    if not parsed:
        raise NotMeaningfulError("no scores were supplied, so there is nothing to test")
    seen: set[tuple[str, date]] = set()
    for s in parsed:
        key = (s.ticker.upper(), s.as_of)
        if key in seen:
            raise ConfigError(
                f"{s.ticker} carries two scores dated {s.as_of}. A company cannot "
                "hold two places in one cross-section; deduplicate before testing."
            )
        seen.add(key)
    assert_scores_point_in_time(parsed)

    lookup = _price_lookup(prices)
    series_by_ticker: dict[str, PriceSeries | None] = {}
    for s in parsed:
        if s.ticker not in series_by_ticker:
            series_by_ticker[s.ticker] = lookup(s.ticker)
    live = [p for p in series_by_ticker.values() if p is not None]
    if not live:
        raise NotMeaningfulError(
            "no price series was found for any scored ticker, so no forward "
            "return can be computed"
        )
    panel_last = max(p.last_date for p in live)

    deals_by_ticker: dict[str, list[DealTerm]] = {}
    for term in deals:
        deals_by_ticker.setdefault(term.ticker.upper(), []).append(term)

    horizon_days = int(round(horizon_months * DAYS_PER_MONTH))
    outcomes = [
        _resolve_outcome(
            s,
            series_by_ticker[s.ticker],
            horizon_days,
            deals_by_ticker,
            panel_last,
            delisting_value,
            benchmark,
        )
        for s in parsed
    ]

    by_date: dict[date, list[Outcome]] = {}
    for o in outcomes:
        by_date.setdefault(o.as_of, []).append(o)

    notes: list[str] = []
    checks: list[str] = []
    kept: list[date] = []
    thin = 0
    for d in sorted(by_date):
        n = sum(1 for o in by_date[d] if o.scored)
        if n >= min_names:
            kept.append(d)
        elif n:
            thin += 1
    if thin:
        notes.append(
            f"{thin} date(s) carried fewer than {min_names} scored names and were "
            "dropped from the coefficient series. A rank correlation across eight "
            "companies is not a result."
        )
    if len(kept) < MIN_DATES:
        raise NotMeaningfulError(
            f"only {len(kept)} rebalance date(s) carried at least {min_names} "
            f"scored names, and {MIN_DATES} is the floor below which a standard "
            "error corrected for overlap is estimating more autocovariances than "
            "the series contains."
        )

    ic_lag, spacing_days = overlap_lag(kept, horizon_months)
    if lag is not None:
        if lag < 0:
            raise ConfigError(f"the Newey-West lag must not be negative, got {lag}")
        notes.append(
            f"the Newey-West lag was set to {lag} by the caller, against the {ic_lag} "
            f"the {horizon_months} month horizon and the "
            f"{spacing_days:.0f} day rebalance spacing imply."
        )
        ic_lag = int(lag)

    ic_values: list[float] = []
    ic_dates: list[date] = []
    ic_counts: list[int] = []
    undefined = 0
    for d in kept:
        rows = [o for o in by_date[d] if o.scored]
        coef = spearman(
            np.asarray([o.score for o in rows], dtype=float),
            np.asarray([o.forward_return for o in rows], dtype=float),
        )
        if coef is None:
            undefined += 1
            continue
        ic_dates.append(d)
        ic_values.append(float(coef))
        ic_counts.append(len(rows))
    if undefined:
        notes.append(
            f"{undefined} date(s) produced no coefficient because one side was "
            "entirely tied, and a constant has no ordering to correlate with."
        )
    if len(ic_values) < MIN_DATES:
        raise NotMeaningfulError(
            f"only {len(ic_values)} date(s) produced a coefficient, below the "
            f"floor of {MIN_DATES}."
        )

    rng = np.random.default_rng(seed)
    ic_array = np.asarray(ic_values, dtype=float)
    _, rhos = newey_west_se(ic_array, ic_lag)
    ic = ICSeries(
        dates=ic_dates,
        values=ic_values,
        counts=ic_counts,
        lag=ic_lag,
        spacing_days=spacing_days,
        autocorrelations=rhos,
        block_bootstrap_se=block_bootstrap_se(ic_array, ic_lag + 1, bootstrap_draws, rng),
    )

    table, membership = _bucket_table(by_date, ic_dates, buckets, ic_lag)
    churn = _turnover(membership, ic_dates, buckets, spacing_days, table.spread)

    # -- the baseline, which is what noise scores on this sample ------------ #
    baseline_means = _permutation_means(by_date, ic_dates, min_names, baseline_draws, rng)
    baseline_mean = float(np.mean(baseline_means)) if baseline_means.size else 0.0
    # (hits + 1) / (draws + 1), not hits / draws. A finite number of
    # permutations cannot establish a p-value of zero, and printing one invites
    # a reader to treat four hundred draws as proof. The add-one form is the
    # standard correction and bounds the p-value below at 1 / (draws + 1), which
    # is what the draw count actually supports.
    permutation_p = (
        float((np.sum(np.abs(baseline_means) >= abs(ic.mean)) + 1) / (baseline_means.size + 1))
        if baseline_means.size
        else None
    )

    evaluation = EvalResult(
        metric="mean information coefficient",
        score=ic.mean,
        baseline_name=(
            f"a random score with the same cross-sectional shape, "
            f"{baseline_draws} permutations within date"
        ),
        baseline_score=baseline_mean,
        n_observations=sum(ic_counts),
        higher_is_better=True,
        folds=list(ic_values),
        notes=[
            "The folds are the per-date coefficients, so the fold standard "
            "deviation is the dispersion of the coefficient across rebalances "
            "and not an out-of-sample spread.",
            "The permutation null leaves each date independent of every other by "
            "construction, so its p-value does not carry the overlap correction "
            "and is anti-conservative for the same reason the naive t is.",
        ],
    )

    card = ModelCard(
        name=label,
        task=(
            f"cross-sectional {horizon_months} month forward-return signal test, "
            f"{buckets} buckets, rebalanced every {spacing_days:.0f} days"
        ),
        trained_through=max(ic_dates),
        n_train=0,
        features=[label],
        hyperparameters={
            "horizon_months": horizon_months,
            "buckets": buckets,
            "min_names_per_date": min_names,
            "newey_west_lag": ic_lag,
            "delisting_return": delisting_label,
            "random_seed": seed,
            "n_tests_run": n_tests_run,
        },
        evaluation=evaluation,
        limitations=[
            "Nothing is fitted here. The card records the test, not a training "
            "run, and n_train is zero because no parameter was estimated from "
            "the data.",
            "Returns are price returns. The price layer carries closes rather "
            "than a total-return index, so a dividend-paying name is understated "
            "by its yield.",
            "The universe is whatever the caller scored, and a universe drawn "
            "from the companies that exist today is survivorship-biased before "
            "this harness sees it.",
        ],
    )

    result = SignalResult(
        label=label,
        outcomes=outcomes,
        ic=ic,
        buckets=table,
        turnover=churn,
        evaluation=evaluation,
        card=card,
        horizon_months=horizon_months,
        n_tests_run=n_tests_run,
        permutation_p=permutation_p,
        baseline_draws=baseline_draws,
        notes=notes + table.notes + churn.notes,
        checks=[],
    )
    result.checks = _checks(result, delisting_label, min_names)
    return result


# Pytest collects any module-level callable whose name starts with ``test_`` and
# calls it with no arguments. This one is the module's entry point and not a
# test, so it says so, and a test module may import it by name without the suite
# trying to run it.
test_signal.__test__ = False


def _permutation_means(
    by_date: Mapping[date, list[Outcome]],
    dates: Sequence[date],
    min_names: int,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mean coefficient of a random score with the same cross-sectional shape.

    The scores on each date are permuted across the names on that date, so the
    distribution of scores, its ties and its count are all preserved exactly and
    only the pairing with the returns is destroyed. That is a tighter null than
    drawing from a normal, which would give the random score a shape the real one
    does not have and a slightly different coefficient in consequence.

    The arithmetic is done on standardised ranks rather than on the raw values.
    A Spearman coefficient is a Pearson coefficient of the two rank vectors, the
    returns' ranks do not move under a permutation of the scores, and permuting
    the scores and re-ranking gives the same rank vector as permuting the ranks.
    So each draw is one dot product of a shuffled vector with a fixed one, which
    is algebraically identical to re-running ``spearman`` and about ten times
    quicker, and four hundred draws stop being something a caller economises on.
    """
    if draws <= 0:
        return np.asarray([], dtype=float)
    panels: list[tuple[np.ndarray, np.ndarray]] = []
    for d in dates:
        rows = [o for o in by_date[d] if o.scored]
        if len(rows) < min_names:
            continue
        x = _standardised_ranks(np.asarray([o.score for o in rows], dtype=float))
        y = _standardised_ranks(np.asarray([o.forward_return for o in rows], dtype=float))
        if x is None or y is None:
            continue
        panels.append((x, y))
    if not panels:
        return np.asarray([], dtype=float)

    out = np.empty(draws, dtype=float)
    for k in range(draws):
        out[k] = float(np.mean([float(rng.permutation(x) @ y) for x, y in panels]))
    return out[np.isfinite(out)]


def _standardised_ranks(values: np.ndarray) -> np.ndarray | None:
    """Mid-ranks, centred and scaled so that a dot product is a correlation.

    None where every value is tied, which is the case ``spearman`` refuses for
    the same reason: a constant has no ordering to correlate with.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    ordered = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and ordered[j + 1] == ordered[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    centred = ranks - ranks.mean()
    norm = float(np.sqrt(centred @ centred))
    return None if norm == 0.0 else centred / norm


def _checks(result: SignalResult, delisting_label: str, min_names: int) -> list[str]:
    """One line per thing a reader needs to know, flags first in spirit.

    Lines beginning ``FLAG:`` failed a threshold. The rest are stated for the
    record, on the principle that a sample description nobody printed is a sample
    description nobody checked.
    """
    ic = result.ic
    counts = result.exit_counts()
    out: list[str] = []

    out.append(
        f"{ic.n} rebalance dates spanning {min(ic.dates)} to {max(ic.dates)}, "
        f"about {result.n_independent_periods:.1f} non-overlapping "
        f"{result.horizon_months} month periods. The standard error is quoted "
        f"against roughly {ic.effective_n:.0f} effective observations, not "
        f"against the {result.n_scored:,} company-dates."
    )
    if ic.n < THIN_DATES:
        out.append(
            f"FLAG: {ic.n} dates is below the {THIN_DATES} at which an interval "
            "corrected for overlap is worth reading. Every figure here is thin."
        )
    if ic.lag == 0:
        out.append(
            "The rebalance spacing is at least the horizon, so the windows do not "
            "overlap and no correction is due. The Newey-West and naive standard "
            "errors are the same number by construction."
        )
    else:
        out.append(
            f"Newey-West at lag {ic.lag} against a {ic.spacing_days:.0f} day "
            f"rebalance spacing. First-order autocorrelation of the coefficient "
            f"series {ic.autocorrelations[0]:+.2f}; the correction multiplied the "
            f"standard error by {ic.inflation:.2f}x."
        )
        if ic.inflation < 1.0:
            out.append(
                "The corrected standard error is SMALLER than the naive one. The "
                "coefficients alternate in sign, so the autocovariances are "
                "negative and the naive figure was the conservative one. This is "
                "a real result rather than a failure of the correction."
            )
    if ic.block_bootstrap_se is not None:
        nw = ic.newey_west_se
        line = (
            f"Moving-block bootstrap standard error {ic.block_bootstrap_se:.4f} "
            f"against {nw:.4f} from Newey-West. The two should agree to within a "
            "fifth; where they do not, the lag is the thing to doubt."
        )
        if nw > 0 and abs(ic.block_bootstrap_se / nw - 1.0) > 0.20:
            out.append("FLAG: " + line)
        else:
            out.append(line)

    acquired = counts.get(_EXIT_ACQUIRED, 0)
    delisted = counts.get(_EXIT_DELISTED, 0)
    censored = counts.get(_EXIT_CENSORED, 0)
    out.append(
        f"{counts.get(_EXIT_HELD, 0):,} holding periods ran their course, "
        f"{acquired} were terminated at a deal, {delisted} ended in a delisting "
        f"with no deal on record, {censored} were censored at the edge of the "
        f"price data, {counts.get(_EXIT_NO_PRICE, 0)} had no usable price."
    )
    if acquired == 0 and delisted == 0:
        out.append(
            "FLAG: not one name in this sample was acquired or delisted over the "
            "whole period. That is not what happens to a technology universe over "
            "a decade, so the universe is a list of today's survivors and every "
            "number here is biased upward by the outcomes it cannot see."
        )
    if delisted and delisting_label == "none":
        out.append(
            f"FLAG: {delisted} delisted company-date(s) were excluded because no "
            "delisting return was stated. Excluding failures raises every figure "
            "in this table. Re-run with delisting_return to see how much."
        )
    if result.turnover.n_pairs and math.isfinite(result.turnover.churn) and result.turnover.churn > 0.15:
        out.append(
            f"FLAG: {result.turnover.churn:.0%} of the scored universe changes "
            "between rebalances, so a large part of the measured turnover is the "
            "universe moving rather than the signal."
        )
    out.append(
        f"Dates needed {min_names} scored names to enter the series; the thinnest "
        f"that did had {min(ic.counts)} and the widest {max(ic.counts)}."
    )
    if result.n_tests_run > 1:
        out.append(
            f"{result.n_tests_run} tests were run. The p-value quoted in the "
            "verdict is Sidak-adjusted for that count; the unadjusted one is "
            f"{result.p_newey_west:.4f}."
        )
    else:
        out.append(
            "One test is claimed. If this signal is the best of several that were "
            "tried, pass n_tests_run and the p-value will be adjusted for the "
            "search; nothing in the arithmetic can recover that count afterwards."
        )
    return out


# --------------------------------------------------------------------------- #
# A real signal to point it at
# --------------------------------------------------------------------------- #


def ev_revenue_scores(
    tickers: Sequence[str],
    dates: Sequence[date],
    *,
    facts_for: Callable[[str, date], CompanyFacts | None],
    price_at: Callable[[str, date], float | None],
    assumptions: Assumptions,
    cheap_is_high: bool = True,
) -> tuple[list[Score], list[str]]:
    """Trailing EV/Revenue for a universe on a set of dates, point in time.

    The simplest real signal the engine can produce and the one worth testing
    first, because it has no dependency on anything fitted. ``facts_for`` must
    return a ``CompanyFacts`` pinned to the score date and ``price_at`` the close
    on or before it, and both are injected so the whole thing runs offline
    against a fixture.

    ``cheap_is_high`` negates the multiple, so a high score means a cheap company
    and a positive coefficient means cheap beat expensive. The sign is fixed
    before the test rather than after it, which is the only way a one-sided
    reading of the answer is honest.

    A negative enterprise value is kept rather than refused. A company whose net
    cash exceeds its market capitalisation is a real state of the world and the
    multiple is genuinely negative, which under this ranking makes it the
    cheapest name in the cross-section. That is a judgment, it is stated here,
    and the count of such names is returned in the notes so a reader can see how
    much of the bottom bucket it is.

    Every score is proved point in time with ``backtest.assert_no_lookahead``
    before it is returned, and carries the provenance so the harness can prove it
    again.
    """
    out: list[Score] = []
    notes: list[str] = []
    failures: dict[str, int] = {}
    negative_ev = 0

    for when in dates:
        for ticker in tickers:
            symbol = ticker.upper()
            try:
                facts = facts_for(symbol, when)
                if facts is None:
                    failures["no facts on file"] = failures.get("no facts on file", 0) + 1
                    continue
                price = price_at(symbol, when)
                if price is None or not math.isfinite(price) or price <= 0:
                    failures["no price"] = failures.get("no price", 0) + 1
                    continue
                fin = build_financials(symbol, facts=facts)
                assert_no_lookahead(fin, when)
                if fin.revenue is None or fin.revenue <= 0:
                    failures["no trailing revenue"] = failures.get("no trailing revenue", 0) + 1
                    continue
                bridge = build_ev_bridge(fin, float(price), assumptions)
                multiple = bridge.enterprise_value / fin.revenue
            except LookaheadError:
                raise
            except (TechvalError, ZeroDivisionError, ValueError, KeyError) as exc:
                key = type(exc).__name__
                failures[key] = failures.get(key, 0) + 1
                continue
            if multiple < 0:
                negative_ev += 1
            out.append(
                Score(
                    ticker=symbol,
                    as_of=when,
                    value=-multiple if cheap_is_high else multiple,
                    knowledge_date=when,
                    provenance=dict(fin.provenance),
                    note=f"EV/Revenue {multiple:.2f}x on the twelve months to {fin.as_of}",
                )
            )

    for reason, n in sorted(failures.items()):
        notes.append(f"{n} company-date(s) produced no multiple: {reason}.")
    if negative_ev:
        notes.append(
            f"{negative_ev} company-date(s) carried a negative enterprise value, "
            "net cash above market capitalisation. They are kept and rank as the "
            "cheapest names in their cross-section."
        )
    notes.append(
        "The sign is "
        + ("negated, so a high score is a cheap company" if cheap_is_high else "as reported, so a high score is an expensive company")
        + ", and it was fixed before the test rather than after it."
    )
    return out, notes
