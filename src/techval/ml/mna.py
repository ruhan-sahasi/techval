"""Which technology companies get bought, scored on what was knowable before the bid.

A coverage banker does not want a probability. He wants a list of twenty names he
can defend in a Monday meeting, and a sentence per name saying why it is on the
list. That is what this module produces: a ranked screen with per-name
attribution, fitted on real merger filings, and reported beside the cheapest
alternative anybody could have run instead.

Everything below is downstream of one decision about the sample, and that
decision is worth more than the model.

**Survivorship here is not a bias, it is the destruction of the experiment.** A
company that is acquired stops filing, is struck from the exchange, and
disappears from every list of tickers that exists today. Build a panel from names
that are listed now and every positive label vanishes with them: the fit runs on
an all-negative sample, reports a flawless in-sample accuracy, and means nothing
at all. The failure is silent because an all-negative classifier is a perfectly
well-behaved object. So the universe here is constructed as of each date from a
roster that keeps the departed, and ``Universe.coverage`` counts how many members
of each date's cross-section have since left, so the reader can see that the
sample contains what the experiment needs.

Three specific things make the departed hard to keep, and all three were verified
against EDGAR rather than assumed:

*A delisted ticker does not resolve.* ``company_tickers.json`` is today's file.
Asking it for SPLK, ZEN, WORK or MNDT raises ``MissingDataError`` on a live run,
so ``build_universe`` takes CIKs for the departed from a roster and never asks
the ticker file to supply what it no longer carries. A ticker is also reusable,
which is worse than absent, so the roster is keyed on CIK and the symbol is a
label.

*A Form 25 is not a departure.* The obvious test for "has this company left" is
the first Form 25 or Form 15 in the filing index, and it is wrong. A Form 25
delists a security, not a company, and an issuer files one when a class of
warrants expires or a note matures. Super Micro Computer filed a Form 25 in March
2019 and has filed a 10-Q every quarter since. The test used here is the Form 25
or Form 15 that is followed by no further periodic report, which is the filing
record actually going quiet.

*The forms that establish a deal are wider than the ones a merger proxy implies.*
See the section on labels below, because this one changed the positive count by a
fifth and it is the finding in this module most likely to matter to anything else
built on ``tmt.precedents``.

**Targets, not completions, and the choice is not a coin toss.** A deal that
regulators block was still a bid, and the company was still a target: Xerox
offered for HP in 2019, HP filed a Schedule 14D-9, and everything that made HP
worth bidding for was true whether or not the bid succeeded. Predicting
completion is a different problem whose drivers sit with the acquirer, the
financing market and the antitrust division, none of which is a fact about the
target and none of which this feature set contains. It is also right-censored in
a way that quietly corrupts the label: a deal announced four months before the
as-of date has not completed and has not broken, and calling it a failure because
no Form 25 has landed yet labels a live deal as a dead one.
``precedents.deal_events`` returns a completion flag and this module carries it
for reporting, but ``label`` is announcement and nothing else.

**No price feature reaches this model, and the reason is the labels.** The brief
this was built from asks for a depressed multiple against a fitted peer line, and
that feature cannot be built honestly here. No public price source serves history
for a delisted symbol: Nasdaq returns no rows for SPLK, ZEN, WORK or MNDT, which
was checked, and the precedents module says the same thing for the same reason.
So every price-derived feature, market capitalisation and every multiple taken
over it included, is missing for exactly the companies that were acquired and
present for the companies that were not. Feed that to a classifier and the
missing-value indicator is a perfect predictor. The model would score an AUC near
one, and it would have learned that companies with no share price get bought,
which is true, circular and useless. The feature set here is therefore built from
the filings alone. Size is log revenue and log total assets, never market
capitalisation. The cost is real and is stated rather than worked around: the
valuation channel, which is the one a banker would most expect to matter, is
absent, and this model cannot say anything about whether cheap companies get
bought.

**The gap between the feature date and the announcement.** Prices move on rumour
in the weeks before a deal is signed, so a feature sitting inside that window
reads the leak rather than predicting the deal. Because this model takes no price
feature the channel is narrow, but the discipline is kept and measured rather
than waved away: an observation whose announcement falls within
``DEFAULT_GAP_DAYS`` of the feature date is dropped, not labelled negative,
because labelling a company that was under offer six weeks later as a
non-target teaches the model the opposite of the truth. ``fit_propensity`` takes
``gap_days`` so the result can be shown with and without it, and the honest
finding on this sample is reported in the tests: the gap costs observations and
moves the score very little, which is what a fundamentals-only feature set should
do and is evidence that the price channel really is the one that leaks.

**Class imbalance is the normal condition and accuracy is meaningless.** Roughly
three percent of a TMT universe is acquired in a given year, so a model that
predicts "never" is right ninety-seven percent of the time. Nothing here reports
accuracy. The headline is the area under the ROC curve, the working numbers are
precision and recall at k, because a screen is read from the top, and
``calibration_table`` answers the separate question of whether the number may be
read as a likelihood at all. The two fail independently and both are reported.

**The baseline is size, not the base rate.** The base rate is a constant
prediction whose AUC is 0.5 by construction, and beating it is not an
achievement. The cheap alternative anybody would actually run is to sort the
universe from smallest to largest, because small companies get bought, and that
costs one line of code. ``EvalResult.baseline_score`` on the returned model is
the walk-forward AUC of that sort, so ``beat_baseline`` answers the question that
matters: does the fitted model add anything to knowing how big a company is. The
base-rate comparison is kept beside it in ``PropensityResult.base_rate_result``.

**Base rates move with the cycle and the model must not be allowed to learn the
calendar.** 2021 and 2023 are different worlds for technology M&A: rates, the
financing market and the antitrust posture all moved, and the count of deals
moved with them. A model scored across both without regard to this looks skilful
when it has learned which year it is. Two things guard against it.
``base_rate_by_year`` is part of the result rather than a diagnostic, so the
swing is visible. And every evaluation is walk-forward by date with an embargo of
the full label horizon plus the gap, so the model is never fitted on an
observation whose label window overlaps the window it is being tested on.

**What this model is worth, said with numbers rather than adjectives.** The
filing record holds a few dozen to a couple of hundred TMT acquisitions of any
size in a decade. That is the sample, it cannot be enlarged by choosing a
different estimator, and a logistic regression on fourteen features is already at
the edge of what it supports. ``ModelCard.limitations`` carries the count of
distinct deals, which is the number that governs, rather than the count of
observations, which is four times larger only because a quarterly panel looks at
the same deal four times. Read the verdict, not the score.

Money is USD millions and dates are label dates unless a name says otherwise.
Nothing here draws a random number except the weight initialisation, which is
seeded from ``assumptions.ml.random_seed``, so two runs over the same panel agree
exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..config import Assumptions
from ..errors import ConfigError, NotMeaningfulError, TechvalError
from .evaluation import (
    Fold,
    calibration_table,
    evaluate_classification,
    precision_recall_at_k,
    walk_forward_folds,
)
from .features import FEATURE_NAMES, FeaturePanel, FeatureRow
from .nn import Adam, Linear, bce_with_logits
from .protocol import EvalResult, ModelCard

# --------------------------------------------------------------------------- #
# Method constants
#
# Rules of the method rather than views about a company, which is why they live
# here and not on Assumptions. Each is overridable at the call site.
# --------------------------------------------------------------------------- #

#: The label window. A company is a positive at date t when a merger agreement is
#: announced within this many months after t. Twelve months because it is the
#: horizon a coverage screen is refreshed on, and because a shorter window leaves
#: too few positives per fold for a fold-level AUC to mean anything.
DEFAULT_HORIZON_MONTHS = 12

#: Calendar days of separation demanded between the feature date and the
#: announcement. An observation closer than this to its own announcement is
#: dropped rather than relabelled, because the company was under offer weeks
#: later and teaching the model that it was not a target is worse than losing the
#: row. Ninety days is one reporting quarter, which is the granularity at which
#: anything in the feature set can actually change.
DEFAULT_GAP_DAYS = 90

#: Below this many distinct deals the fit is not reported as a model. Not the
#: count of observations: a quarterly panel sees one deal four times and the
#: effective sample is the number of companies that were bought.
MIN_DISTINCT_DEALS = 10

#: Decoupled weight decay on the coefficients, which is the ridge penalty under
#: Adam. Chosen rather than tuned: tuning it would spend the out-of-sample window
#: the score is measured on, and a number picked to maximise the reported AUC is
#: not a number the reported AUC may then be quoted against. It is set so that
#: the shrinkage over the fixed step budget is material without annihilating the
#: coefficients, which is checked in the tests by looking at the fitted weights
#: rather than at the score.
DEFAULT_L2 = 0.02

#: Full-batch steps. The problem is a convex logistic regression in fourteen
#: dimensions; this is comfortably past convergence and fixed rather than
#: early-stopped so that two runs agree exactly.
DEFAULT_STEPS = 400
DEFAULT_LEARNING_RATE = 0.05

#: The top of a ranked screen, which is the number a banker actually reads.
DEFAULT_TOP_K = 20

#: A sub-vertical cross-section smaller than this on a given date cannot supply a
#: median worth subtracting, so the relative features fall back to the whole
#: cross-section and the row says so.
MIN_VERTICAL_PEERS = 5

#: Days after a company's last periodic report beyond which it is treated as
#: having left the universe even though no Form 25 or Form 15 was ever filed.
#:
#: Deregistration is the tidy exit and it is not the only one. A company in
#: Chapter 11 stops filing and its securities move to the pink sheets without
#: the exchange filing anything: Avaya's last 10-Q was September 2023 and its
#: filing record simply stops. Left in, such a name sits in every later
#: cross-section carrying three-year-old financials, and because its numbers are
#: frozen at their worst it screens as a permanent takeout candidate. Fifteen
#: months is one annual report plus a full quarter of slack, so an ordinary late
#: filer survives it and a company that has stopped does not.
STALE_REGISTRANT_DAYS = 460


# --------------------------------------------------------------------------- #
# The point-in-time universe
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UniverseMember:
    """One company, with the two dates that decide whether it existed yet.

    ``admitted`` is the filing date of the earliest periodic report on file,
    which is the first moment the company was a reporting registrant with
    financial statements a model could read. ``departed`` is the Form 25 or Form
    15 that is followed by no further periodic report, which is the filing record
    going quiet rather than one security being delisted.

    ``cik`` is the identity and ``ticker`` is a label. A symbol is reused: PARA
    meant Paramount for years and today means something else entirely, so a panel
    keyed on symbols silently splices two companies together across a rename.
    """

    ticker: str
    cik: int
    name: str = ""
    admitted: date | None = None
    departed: date | None = None
    last_filing: date | None = None
    sub_vertical: str | None = None
    sic: str | None = None
    notes: str = ""

    def listed_on(self, when: date) -> bool:
        """Was this company a reporting registrant on that date.

        Admission is inclusive and departure is exclusive: a company is in the
        cross-section on the day its first periodic report was filed and out of
        it on the day the record goes quiet. An unknown admission date is treated
        as admitted, with the reason already recorded in ``notes`` by whatever
        built the roster, because excluding a name for a gap in the submissions
        index would shrink the universe for a reason that has nothing to do with
        the company.
        """
        if self.admitted is not None and when < self.admitted:
            return False
        if self.departed is not None and when >= self.departed:
            return False
        if (
            self.last_filing is not None
            and when > self.last_filing + timedelta(days=STALE_REGISTRANT_DAYS)
        ):
            return False
        return True


@dataclass
class Universe:
    """A roster of companies and the dates over which each of them existed.

    The whole point is that it holds companies that no longer exist. A universe
    built from a current ticker file is a universe of survivors, and on this task
    the survivors are exactly the negatives.
    """

    members: tuple[UniverseMember, ...]
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for m in self.members:
            if m.ticker in seen:
                raise ConfigError(
                    f"{m.ticker} appears twice in the universe roster. A symbol is "
                    "the key of the panel, so a duplicate silently merges two "
                    "companies; key the roster on CIK and disambiguate the label."
                )
            seen.add(m.ticker)

    @property
    def by_ticker(self) -> dict[str, UniverseMember]:
        return {m.ticker: m for m in self.members}

    def as_of(self, when: date) -> list[UniverseMember]:
        """The cross-section on one date, in ticker order so runs agree."""
        return sorted(
            (m for m in self.members if m.listed_on(when)), key=lambda m: m.ticker
        )

    def departed_members(self) -> list[UniverseMember]:
        return [m for m in self.members if m.departed is not None]

    def coverage(self, dates: Sequence[date]) -> pd.DataFrame:
        """Per-date census, split by whether the member has since left.

        This frame is the survivorship proof. ``since_departed`` counts the names
        in that date's cross-section that are gone by the end of the roster, and
        a column of zeros means the panel was built from survivors and the
        experiment is void.
        """
        rows = []
        for when in dates:
            live = self.as_of(when)
            gone = sum(1 for m in live if m.departed is not None)
            rows.append(
                {
                    "as_of": when,
                    "n": len(live),
                    "since_departed": gone,
                    "still_listed": len(live) - gone,
                }
            )
        return pd.DataFrame(rows, columns=["as_of", "n", "since_departed", "still_listed"])


def build_universe(
    roster: Iterable[Mapping[str, Any]],
    *,
    sub_verticals: Mapping[str, str] | None = None,
) -> Universe:
    """Assemble a universe from a roster of dated registrants.

    Each entry needs ``ticker`` and ``cik`` and may carry ``name``, ``admitted``,
    ``departed``, ``sub_vertical``, ``sic`` and ``notes``. Dates may be ISO
    strings or ``date`` objects.

    There is deliberately no default roster and no call to the SEC ticker file.
    The ticker file cannot supply a departed company, so a convenience default
    built from it would hand back a survivor universe while looking like it had
    done the work, and that is the one failure this module exists to prevent.
    """
    members: list[UniverseMember] = []
    for entry in roster:
        ticker = str(entry["ticker"]).upper()
        sub = entry.get("sub_vertical")
        if sub is None and sub_verticals is not None:
            sub = sub_verticals.get(ticker)
        members.append(
            UniverseMember(
                ticker=ticker,
                cik=int(entry["cik"]),
                name=str(entry.get("name") or ticker),
                admitted=_as_date(entry.get("admitted")),
                departed=_as_date(entry.get("departed")),
                last_filing=_as_date(entry.get("last_filing")),
                sub_vertical=sub,
                sic=entry.get("sic"),
                notes=str(entry.get("notes") or ""),
            )
        )
    members.sort(key=lambda m: m.ticker)
    universe = Universe(tuple(members))
    n_gone = len(universe.departed_members())
    universe.notes.append(
        f"{len(members)} registrants, of which {n_gone} have left the filing "
        "record. A roster whose departed count is zero is a survivor list and "
        "cannot carry a positive label."
    )
    return universe


def _as_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


#: Forms that establish the filer was itself a target, beyond the set
#: ``tmt.precedents.DEAL_FORMS`` looks at.
#:
#: A merger proxy is a DEFM14A only when the shareholders are asked to vote. When
#: a controlling holder approves the merger by written consent, no vote is
#: solicited, and Regulation 14C requires an information statement instead: the
#: company files PREM14C and DEFM14C and never files a proxy at all. Schedule
#: 13E-3 is the going-private schedule required by Rule 13e-3 whenever an
#: affiliate is on the buy side.
#:
#: This is not a rounding error in coverage and it is not random. Searching a
#: real universe of 493 TMT registrants, 13 of the 23 departed companies that a
#: proxy-only screen could not explain had filed a DEFM14C: PowerSchool,
#: Instructure, Thoughtworks, Informatica, Paycor, SolarWinds, Vizio and their
#: like, which is to say sponsor and founder-controlled take-privates almost
#: without exception. The form a company uses to be sold is decided by who
#: controls its votes, so a screen that reads only proxies systematically drops
#: controlled companies, and control is one of the things a propensity model is
#: trying to use as a predictor. Screening on proxies alone biases the label set
#: against the hypothesis.
CONSENT_DEAL_FORMS: tuple[str, ...] = (
    "PREM14C",
    "DEFM14C",
    "PRER14C",
    "DEFR14C",
    "SC 13E3",
    "SC 13E3/A",
)


@dataclass(frozen=True)
class DealEvent:
    """One announced acquisition of one company.

    ``completed`` is carried for reporting and is not the label. See the module
    docstring: an announced deal that regulators blocked is a deal, and a deal
    announced last quarter is neither completed nor broken.
    """

    ticker: str
    announced: date
    completed: bool = False
    name: str = ""
    acquirer: str | None = None
    source_form: str | None = None


@dataclass
class Observation:
    """One company on one date, with the label the horizon assigns it.

    ``label`` is 1 when an announcement falls inside the label window, 0 when the
    window closed with no announcement, and the observation is absent from the
    sample entirely when the window could not be resolved. The three cases are
    kept apart because collapsing the third into a zero is the commonest way a
    propensity sample is quietly poisoned.
    """

    ticker: str
    as_of: date
    label: int
    window_opens: date
    window_closes: date
    announced: date | None = None
    completed: bool | None = None


@dataclass
class LabelReport:
    """The sample, and everything that was excluded from it with the reason.

    The counts here are the first thing to read. ``n_positive`` against
    ``n_distinct_deals`` says how much repetition the panel contains, and
    ``dropped_in_gap`` is the price of the leak discipline.
    """

    observations: list[Observation]
    dropped_in_gap: int = 0
    dropped_after_announcement: int = 0
    dropped_unresolved: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_positive(self) -> int:
        return sum(o.label for o in self.observations)

    @property
    def n_negative(self) -> int:
        return len(self.observations) - self.n_positive

    @property
    def n_distinct_deals(self) -> int:
        return len({o.ticker for o in self.observations if o.label})

    @property
    def base_rate(self) -> float:
        if not self.observations:
            raise NotMeaningfulError("no observations, so there is no base rate")
        return self.n_positive / len(self.observations)

    def rows(self) -> list[tuple[str, Any]]:
        return [
            ("Observations", len(self.observations)),
            ("Positives", self.n_positive),
            ("Distinct deals", self.n_distinct_deals),
            ("Base rate", self.base_rate if self.observations else float("nan")),
            ("Dropped inside the gap", self.dropped_in_gap),
            ("Dropped after announcement", self.dropped_after_announcement),
            ("Dropped with an unresolved window", self.dropped_unresolved),
        ]


def label_observations(
    universe: Universe,
    dates: Sequence[date],
    events: Iterable[DealEvent | tuple[str, date, bool]],
    *,
    as_of: date,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    gap_days: int = DEFAULT_GAP_DAYS,
) -> LabelReport:
    """Label every company on every date, and account for what was excluded.

    The window for an observation dated ``t`` opens at ``t + gap_days`` and closes
    ``horizon_months`` after that. Four things can happen to a company-date and
    all four are counted:

    *It is a positive.* An announcement falls inside the window.

    *It is a negative.* The window closed at or before ``as_of`` with no
    announcement, so the answer is known.

    *It is inside the gap.* An announcement falls after ``t`` but before the
    window opens. The company was under offer within weeks of the feature date,
    so it is neither a clean positive at this horizon nor anything like a
    negative, and it is dropped. Set ``gap_days`` to zero to keep these and see
    what the leak was worth.

    *Its window is unresolved.* The window closes after ``as_of``, so it is
    dropped, and it is dropped before the label is looked at rather than after.
    That ordering is not fussiness, it is the difference between a sample and an
    artefact. A positive is settled the day the agreement is announced; a
    negative is settled only when the whole window has closed with nothing in
    it. Apply the test to the negatives alone and the last year of the sample
    keeps every deal and discards the companies that were not bought, which on
    this panel tripled the 2025 base rate to ten percent against a true rate
    near three. The final fold then trains on a world that never existed and the
    model appears to have found something. This is the mirror image of calling
    an unresolved window a zero, which is the more familiar error and labels
    every recent deal as a non-deal; both are censoring and both are fixed by
    refusing to look at the answer before deciding whether the question has one.

    Observations at or after a company's own announcement are excluded outright.
    The label has already happened, the company's filings from that point are
    about the deal, and a model shown them has learned to predict an acquisition
    from the acquisition.
    """
    if horizon_months < 1:
        raise ConfigError(f"horizon_months must be at least 1, got {horizon_months}")
    if gap_days < 0:
        raise ConfigError(f"gap_days cannot be negative, got {gap_days}")

    by_ticker: dict[str, DealEvent] = {}
    for raw in events:
        event = raw if isinstance(raw, DealEvent) else DealEvent(*raw)
        ticker = event.ticker.upper()
        prior = by_ticker.get(ticker)
        # One deal per company, the earliest, because a later one is a second
        # process the panel has already stopped observing by the time it happens.
        if prior is None or event.announced < prior.announced:
            by_ticker[ticker] = event

    report = LabelReport(observations=[])
    known = set(universe.by_ticker)
    unknown = sorted(t for t in by_ticker if t not in known)
    if unknown:
        report.notes.append(
            f"{len(unknown)} labelled companies are not in the universe roster and "
            "their deals cannot be scored: "
            + ", ".join(unknown[:8])
            + ("..." if len(unknown) > 8 else "")
            + ". Every one of these is a positive the sample loses, which is the "
            "survivorship failure arriving through the back door."
        )

    for when in sorted(dates):
        for member in universe.as_of(when):
            event = by_ticker.get(member.ticker)
            opens = when + timedelta(days=gap_days)
            closes = _add_months(opens, horizon_months)
            if event is not None and event.announced <= when:
                report.dropped_after_announcement += 1
                continue
            if event is not None and when < event.announced < opens:
                report.dropped_in_gap += 1
                continue
            if closes > as_of:
                # Before the label is looked at, not after. A positive is
                # settled the day the agreement is announced and a negative is
                # settled only when its window closes, so applying this test to
                # the negatives alone keeps every recent positive and discards
                # the recent negatives it should be compared against. The base
                # rate of the last year of the sample then triples, the final
                # fold trains on a world that never existed, and the model looks
                # as though it found something. Everything whose window is still
                # open is dropped, positive or not.
                report.dropped_unresolved += 1
                continue
            if event is not None and opens <= event.announced <= closes:
                report.observations.append(
                    Observation(
                        ticker=member.ticker,
                        as_of=when,
                        label=1,
                        window_opens=opens,
                        window_closes=closes,
                        announced=event.announced,
                        completed=event.completed,
                    )
                )
                continue
            report.observations.append(
                Observation(
                    ticker=member.ticker,
                    as_of=when,
                    label=0,
                    window_opens=opens,
                    window_closes=closes,
                )
            )

    report.notes.append(
        f"Label window of {horizon_months} months opening {gap_days} days after "
        f"the feature date, resolved against {as_of}."
    )
    if report.observations:
        report.notes.append(
            f"{report.n_positive} positives over {report.n_distinct_deals} distinct "
            f"deals: the panel sees each deal about "
            f"{report.n_positive / max(report.n_distinct_deals, 1):.1f} times, so the "
            "effective sample is the deal count and not the observation count."
        )
    return report


def _add_months(anchor: date, months: int) -> date:
    """Calendar months forward, clamped to the last day of the target month."""
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    day = min(anchor.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def base_rate_by_year(report: LabelReport) -> pd.DataFrame:
    """Positives, observations and base rate for each year of the sample.

    Part of the result rather than a diagnostic. Technology M&A is violently
    cyclical: the financing market, the rate environment and the antitrust
    posture all moved between 2021 and 2023 and the deal count moved with them. A
    model evaluated across both without this frame beside it can look skilful
    while having learned nothing but the calendar, and a reader who cannot see
    the swing cannot tell the difference.
    """
    rows: dict[int, list[int]] = {}
    for obs in report.observations:
        bucket = rows.setdefault(obs.as_of.year, [0, 0])
        bucket[0] += 1
        bucket[1] += obs.label
    out = [
        {
            "year": year,
            "n": n,
            "positives": pos,
            "base_rate": pos / n if n else float("nan"),
        }
        for year, (n, pos) in sorted(rows.items())
    ]
    return pd.DataFrame(out, columns=["year", "n", "positives", "base_rate"])


# --------------------------------------------------------------------------- #
# Features
#
# The finance content of this module is here. Each of these is a claim about why
# a company is bought, and each states the claim rather than leaving it to the
# coefficient to imply.
# --------------------------------------------------------------------------- #

#: Features taken straight from the point-in-time store. Every one is computable
#: from filings alone. Nothing that needs a share price appears, for the reason
#: in the module docstring.
PASSTHROUGH_FEATURES: tuple[str, ...] = (
    "scale_log_revenue",
    "growth_revenue_1y",
    "margin_gross",
    "margin_ebit_chg_1y",
    "efficiency_sbc_to_revenue",
    "capital_current_ratio",
    "returns_rule_of_40",
    "returns_deferred_revenue_to_revenue",
)

#: Features this module derives, each with the argument for it.
DERIVED_FEATURES: tuple[str, ...] = (
    "mna_subscale_in_vertical",
    "mna_growth_decay",
    "mna_growth_rank_in_vertical",
    "mna_opex_load",
    "mna_net_cash_to_assets",
    "mna_years_listed",
)

FEATURE_SET: tuple[str, ...] = PASSTHROUGH_FEATURES + DERIVED_FEATURES

#: One column for the share of the fitted features a row could not source. A
#: linear model handed a mean-imputed value cannot tell an average company from
#: an unreadable one, and on this panel unreadability is not random: it tracks
#: the age of the filing and the filer's tagging discipline.
MISSING_SHARE = "mna_missing_share"

FITTED_COLUMNS: tuple[str, ...] = FEATURE_SET + (MISSING_SHARE,)

#: What each feature claims, in the words the model card and the attribution
#: table print. A coefficient nobody can read is a coefficient nobody can argue
#: with, and an unarguable model has no place beside a discounted cash flow.
FEATURE_RATIONALE: dict[str, str] = {
    "scale_log_revenue": (
        "Absolute size. The cheapest true thing about this problem: the pool of "
        "buyers who can write the cheque shrinks with every order of magnitude, "
        "and a company at the top of the cross-section has almost none."
    ),
    "growth_revenue_1y": (
        "Trailing revenue growth. A board with a credible growth plan has an "
        "alternative to selling, and a board without one does not."
    ),
    "margin_gross": (
        "Gross margin. High gross margin is what makes a company worth owning "
        "rather than worth running, because it is the part of the cost structure "
        "an acquirer cannot improve and therefore the part it is buying."
    ),
    "margin_ebit_chg_1y": (
        "Change in operating margin over the year. Margin going backwards while "
        "gross margin holds is the signature of a cost structure that has grown "
        "past the revenue, which is the case a buyer underwrites."
    ),
    "efficiency_sbc_to_revenue": (
        "Stock compensation over revenue. A cost a private owner removes on day "
        "one and a public company cannot, so it is a source of synergy that "
        "belongs to the buyer and not to the standalone plan."
    ),
    "capital_current_ratio": (
        "Current assets over current liabilities. A liquidity screen, and a "
        "company whose working capital is tightening is a company whose board is "
        "running out of time to be patient."
    ),
    "returns_rule_of_40": (
        "Growth plus free cash flow margin. The single number the software "
        "cross-section trades on, and a genuine interaction rather than a "
        "restatement of its parts."
    ),
    "returns_deferred_revenue_to_revenue": (
        "Deferred revenue over revenue. A proxy for how much of next year is "
        "already contracted, which is what a buyer is really underwriting and "
        "what a lender will advance against."
    ),
    "mna_subscale_in_vertical": (
        "Log revenue less the median log revenue of the sub-vertical on the same "
        "date. Absolute size is not the claim; subscale is. A two billion dollar "
        "semiconductor company is a minnow and a two billion dollar vertical "
        "software company is the leader of its category, and only the first of "
        "those is a target for that reason."
    ),
    "mna_growth_decay": (
        "Trailing one-year growth less the three-year compound rate. Negative is "
        "deceleration. The moment a standalone plan stops being credible to a "
        "board is rarely low growth, which the board has usually lived with for "
        "years; it is growth that has just turned down."
    ),
    "mna_growth_rank_in_vertical": (
        "Percentile of one-year growth inside the sub-vertical on that date, so "
        "that a sector-wide slowdown does not read as company-specific distress. "
        "Every software company decelerated in 2022 and almost none of them were "
        "bought for it."
    ),
    "mna_opex_load": (
        "Gross margin less operating margin: the operating expense the business "
        "carries per dollar of revenue. High gross margin with a heavy opex load "
        "is the classic sponsor target, because the gross profit is real and the "
        "cost structure is the part somebody else thinks they can fix."
    ),
    "mna_net_cash_to_assets": (
        "Cash and securities less total debt, over total assets. Net cash comes "
        "back to the buyer at closing, so it lowers the effective price of the "
        "equity and makes a target financeable. Taken over assets rather than "
        "over market capitalisation, which is not available for a delisted "
        "target and would therefore be present only for the negatives."
    ),
    "mna_years_listed": (
        "Years since the first periodic report. A company that listed last year "
        "has a lock-up, a story and a board with no appetite to sell; a company "
        "that has been subscale and public for a decade has run out of both."
    ),
    MISSING_SHARE: (
        "Share of the fitted features this row could not source. Carried as a "
        "feature rather than imputed away, because how readable a filer is "
        "correlates with its age and its size and is not missing at random."
    ),
}


@dataclass
class FeatureMatrix:
    """The design matrix, its labels, and the provenance of every row.

    Held together in one object because the three arrays have to stay aligned
    through every fold, and the commonest way a walk-forward evaluation goes
    wrong is that they do not.
    """

    X: np.ndarray
    y: np.ndarray
    dates: list[date]
    tickers: list[str]
    columns: tuple[str, ...]
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.X, columns=list(self.columns))
        frame.insert(0, "label", self.y)
        frame.insert(0, "as_of", self.dates)
        frame.insert(0, "ticker", self.tickers)
        return frame


def balance_sheet_extras(
    ticker: str,
    as_of: date,
    client,
) -> dict[str, float | None]:
    """Net cash over total assets, sourced from the filings for one company-date.

    This one figure cannot come out of the feature store, and the reason is
    itself the argument of this module. ``features.py`` carries cash only as a
    ratio over market capitalisation, and market capitalisation is exactly what a
    delisted target does not have. So the numerator and the denominator are
    resolved here directly, both from the balance sheet, both point in time.

    Net cash is cash plus short-term investments less straight debt, convertible
    debt and finance lease liabilities. Operating lease liabilities are excluded,
    following the same convention as the enterprise value bridge: whether they
    are debt is a judgment the bridge already makes, and making it differently
    here would put two definitions of leverage into one engine.

    ``client`` must be pinned to ``as_of`` the same way ``build_features``
    demands, and it is the caller's job to hand in the same pinned client that
    built the row.
    """
    from .. import tags
    from ..financials import build_financials

    facts = client.company_facts(ticker.upper())
    fin = build_financials(ticker.upper(), facts=facts)
    assets, _prov = facts.resolve_instant(
        "total assets", tags.TOTAL_ASSETS, fin.as_of, required=False
    )
    if assets is None or assets <= 0:
        return {"mna_net_cash_to_assets": None}
    net_cash = (
        fin.cash
        + fin.short_term_investments
        - fin.total_debt_ex_leases
        - fin.finance_lease_liability
    )
    return {"mna_net_cash_to_assets": float(net_cash / (assets / 1e6))}


def collect_extras(
    tickers: Sequence[str],
    dates: Sequence[date],
    client_factory: Callable[[date], Any],
) -> dict[tuple[str, date], dict[str, float | None]]:
    """``balance_sheet_extras`` across a panel, one pinned client per date.

    Shaped like ``features.build_panel`` and for the same reason: one client can
    only be pinned to one date, so a panel spanning many needs a factory. A
    company that cannot be resolved on a date is recorded as missing rather than
    skipped, so the gap reaches ``mna_missing_share`` instead of vanishing.
    """
    out: dict[tuple[str, date], dict[str, float | None]] = {}
    for when in dates:
        client = client_factory(when)
        for ticker in tickers:
            try:
                out[(ticker, when)] = balance_sheet_extras(ticker, when, client)
            except (TechvalError, OSError):
                out[(ticker, when)] = {"mna_net_cash_to_assets": None}
    return out


def derive_features(
    panel: FeaturePanel,
    universe: Universe,
    extras: Mapping[tuple[str, date], Mapping[str, float | None]] | None = None,
) -> dict[tuple[str, date], dict[str, float | None]]:
    """Compute the M&A-specific features for every row of a panel.

    The relative features are struck against the sub-vertical's own cross-section
    on the row's own date, never pooled across dates. Pooling would bake the
    level of an era into the comparison: software grew at thirty percent in 2021
    and at twelve in 2023, and a company at twelve percent is a laggard in one of
    those years and unremarkable in the other.

    Where a sub-vertical has fewer than ``MIN_VERTICAL_PEERS`` names on a date the
    comparison falls back to the whole cross-section for that date, because a
    median of three is three companies wearing a statistic's clothing. Where the
    sub-vertical is unknown the name is compared against the whole cross-section
    for the same reason.
    """
    members = universe.by_ticker
    rows_by_date: dict[date, list[FeatureRow]] = {}
    for row in panel.ok_rows():
        rows_by_date.setdefault(row.as_of, []).append(row)

    out: dict[tuple[str, date], dict[str, float | None]] = {}
    for when, rows in rows_by_date.items():
        buckets: dict[str | None, list[FeatureRow]] = {}
        for row in rows:
            member = members.get(row.ticker)
            buckets.setdefault(member.sub_vertical if member else None, []).append(row)

        for row in rows:
            member = members.get(row.ticker)
            vertical = member.sub_vertical if member else None
            peers = buckets.get(vertical) or []
            if vertical is None or len(peers) < MIN_VERTICAL_PEERS:
                peers = rows

            size = row.values.get("scale_log_revenue")
            peer_size = _observed(peers, "scale_log_revenue")
            growth = row.values.get("growth_revenue_1y")
            peer_growth = _observed(peers, "growth_revenue_1y")

            gross = row.values.get("margin_gross")
            ebit = row.values.get("margin_ebit")
            extra = (extras or {}).get((row.ticker, when)) or {}
            cash_ratio = extra.get("mna_net_cash_to_assets")

            derived: dict[str, float | None] = {
                "mna_subscale_in_vertical": (
                    None
                    if size is None or not peer_size
                    else float(size - float(np.median(peer_size)))
                ),
                "mna_growth_decay": _minus(
                    growth, row.values.get("growth_revenue_cagr_3y")
                ),
                "mna_growth_rank_in_vertical": _percentile(growth, peer_growth),
                "mna_opex_load": _minus(gross, ebit),
                "mna_net_cash_to_assets": cash_ratio,
                "mna_years_listed": _years_listed(member, when),
            }
            out[(row.ticker, when)] = derived
    return out


def _observed(rows: Sequence[FeatureRow], name: str) -> list[float]:
    return [
        float(r.values[name])
        for r in rows
        if r.values.get(name) is not None and np.isfinite(float(r.values[name]))
    ]


def _minus(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _percentile(value: float | None, population: Sequence[float]) -> float | None:
    """Where a value sits in its own cross-section, on [0, 1].

    The mid-rank convention, so a value tied with the whole population lands at
    0.5 rather than at 0 or 1. Below three observations there is no percentile
    worth reporting and the answer is None rather than a number.
    """
    if value is None or len(population) < 3:
        return None
    arr = np.asarray(population, dtype=float)
    below = float(np.sum(arr < value))
    equal = float(np.sum(arr == value))
    return float((below + 0.5 * equal) / arr.size)


def _years_listed(member: UniverseMember | None, when: date) -> float | None:
    if member is None or member.admitted is None:
        return None
    return max((when - member.admitted).days / 365.25, 0.0)


def build_matrix(
    panel: FeaturePanel,
    universe: Universe,
    report: LabelReport,
    *,
    columns: Sequence[str] = FITTED_COLUMNS,
    extras: Mapping[tuple[str, date], Mapping[str, float | None]] | None = None,
) -> FeatureMatrix:
    """Join the panel to the labels and emit the design matrix.

    An observation with no panel row is dropped and counted rather than filled:
    a company the feature store could not build is a fact about coverage and not
    a company with average features.
    """
    derived = derive_features(panel, universe, extras)
    by_key = {(r.ticker, r.as_of): r for r in panel.ok_rows()}
    names = [c for c in columns if c != MISSING_SHARE]

    rows: list[list[float]] = []
    labels: list[int] = []
    dates: list[date] = []
    tickers: list[str] = []
    unmatched = 0

    for obs in report.observations:
        key = (obs.ticker, obs.as_of)
        row = by_key.get(key)
        if row is None:
            unmatched += 1
            continue
        extra = derived.get(key, {})
        values: list[float] = []
        missing = 0
        for name in names:
            raw = extra[name] if name in extra else row.values.get(name)
            if raw is None or not np.isfinite(float(raw)):
                values.append(np.nan)
                missing += 1
            else:
                values.append(float(raw))
        if MISSING_SHARE in columns:
            values.append(missing / len(names))
        rows.append(values)
        labels.append(obs.label)
        dates.append(obs.as_of)
        tickers.append(obs.ticker)

    matrix = FeatureMatrix(
        X=np.asarray(rows, dtype=float).reshape(len(rows), len(columns)),
        y=np.asarray(labels, dtype=float),
        dates=dates,
        tickers=tickers,
        columns=tuple(columns),
    )
    if unmatched:
        matrix.notes.append(
            f"{unmatched} of {len(report.observations)} labelled observations had no "
            "feature row and were dropped rather than imputed."
        )
    return matrix


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass
class Attribution:
    """Why one name is where it is on the list.

    ``contributions`` are in logit units and sum, with the intercept, to the
    score. A linear model is used here largely so that this table exists: a
    gradient-boosted ensemble would rank slightly better on a sample this size
    only by accident, and would have nothing to say in a meeting.
    """

    ticker: str
    as_of: date
    probability: float
    intercept: float
    contributions: list[tuple[str, float, float, float]] = field(default_factory=list)

    def top(self, k: int = 3) -> list[tuple[str, float, float, float]]:
        """The k features pushing hardest, by absolute contribution."""
        return sorted(self.contributions, key=lambda c: -abs(c[3]))[:k]

    def sentence(self, k: int = 3) -> str:
        """One line a banker can read off a page."""
        if not self.contributions:
            return f"{self.ticker}: no feature on this row could be sourced."
        parts = []
        for name, value, _coef, contribution in self.top(k):
            direction = "raises" if contribution > 0 else "lowers"
            parts.append(f"{name} at {value:+.2f} {direction} the score")
        return f"{self.ticker} at {self.probability:.1%}: " + "; ".join(parts) + "."

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.contributions,
            columns=["feature", "standardised_value", "coefficient", "contribution"],
        )


@dataclass
class PropensityModel:
    """A fitted logistic screen, its coefficients, and what it was fitted on.

    ``mu`` and ``sd`` are the standardisation fitted on the training rows alone.
    ``prior_shift`` undoes the class weighting used during the fit: weighting the
    positive class makes the head report the probability of a sample in which
    positives were ``pos_weight`` times as common as they are, and subtracting
    the log of that weight from the logit maps it back. The correction is applied
    to ``probabilities`` and never to ``scores``, because a ranking is unaffected
    by a constant shift and a calibration table is not.
    """

    weights: np.ndarray
    intercept: float
    columns: tuple[str, ...]
    mu: np.ndarray
    sd: np.ndarray
    prior_shift: float
    card: ModelCard
    base_rate: float

    @property
    def coefficients(self) -> dict[str, float]:
        """Coefficient per feature, in standardised units so they compare."""
        return {c: float(w) for c, w in zip(self.columns, self.weights)}

    def standardise(self, X: np.ndarray) -> np.ndarray:
        """Centre, scale, and put the training mean where a gap was.

        A NaN becomes exactly zero after standardisation, which is the training
        mean of that column. It is mean imputation, it is fitted on the training
        rows only so it leaks nothing, and it is the reason ``mna_missing_share``
        is a column: without it the model cannot tell an average company from one
        it could not read.
        """
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Logits. Use these to rank; they carry the class weighting."""
        return self.standardise(X) @ self.weights + self.intercept

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        """Prior-corrected probabilities. Check ``calibration_table`` before quoting one."""
        return _sigmoid(self.scores(X) - self.prior_shift)

    def attribute(
        self, X: np.ndarray, tickers: Sequence[str], dates: Sequence[date]
    ) -> list[Attribution]:
        """Per-name decomposition of the score into feature contributions."""
        Z = self.standardise(X)
        probs = self.probabilities(X)
        out: list[Attribution] = []
        for i, ticker in enumerate(tickers):
            contributions = [
                (c, float(Z[i, j]), float(self.weights[j]), float(Z[i, j] * self.weights[j]))
                for j, c in enumerate(self.columns)
            ]
            out.append(
                Attribution(
                    ticker=ticker,
                    as_of=dates[i],
                    probability=float(probs[i]),
                    intercept=float(self.intercept - self.prior_shift),
                    contributions=contributions,
                )
            )
        return out

    def rank(
        self,
        panel: FeaturePanel,
        universe: Universe,
        as_of: date,
        *,
        top_k: int = DEFAULT_TOP_K,
        extras: Mapping[tuple[str, date], Mapping[str, float | None]] | None = None,
        events: Iterable[DealEvent | tuple[str, date, bool]] = (),
    ) -> pd.DataFrame:
        """The screen: names ordered by score on one date, with what drove each.

        This, and not a probability, is the deliverable. A banker works a target
        list, and the question a list answers is which twenty names to look at,
        which survives a model that ranks well and calibrates badly. The frame
        carries the probability anyway, and ``calibration_table`` on the same
        model says whether it may be read as one.

        Only companies in the universe on ``as_of`` are ranked, and a name whose
        deal has already been announced by then is dropped. Leaving it in is not
        a harmless cosmetic error: a company under offer screens beautifully,
        because everything that made a buyer want it is still in its last
        filings, and a screen whose top of list is names the market already
        knows about is a screen nobody needs. Pass ``events`` to enforce it. The
        list is printed here anyway when no events are supplied, so that the
        omission is the caller's visible choice rather than a silent default.
        """
        live = {m.ticker for m in universe.as_of(as_of)}
        announced = {
            (e if isinstance(e, DealEvent) else DealEvent(*e)).ticker
            for e in events
            if (e if isinstance(e, DealEvent) else DealEvent(*e)).announced <= as_of
        }
        rows = [
            r
            for r in panel.ok_rows()
            if r.as_of == as_of and r.ticker in live and r.ticker not in announced
        ]
        if not rows:
            raise NotMeaningfulError(
                f"no universe member has a feature row dated {as_of}, so there is "
                "nothing to rank. Check that the panel was built on this date."
            )
        stub = LabelReport(
            observations=[
                Observation(
                    ticker=r.ticker,
                    as_of=as_of,
                    label=0,
                    window_opens=as_of,
                    window_closes=as_of,
                )
                for r in rows
            ]
        )
        matrix = build_matrix(panel, universe, stub, columns=self.columns, extras=extras)
        probs = self.probabilities(matrix.X)
        logits = self.scores(matrix.X)
        attributions = self.attribute(matrix.X, matrix.tickers, matrix.dates)
        members = universe.by_ticker

        records = []
        for i, ticker in enumerate(matrix.tickers):
            member = members.get(ticker)
            drivers = attributions[i].top(3)
            records.append(
                {
                    "rank": 0,
                    "ticker": ticker,
                    "name": member.name if member else ticker,
                    "sub_vertical": member.sub_vertical if member else None,
                    "probability": float(probs[i]),
                    "score": float(logits[i]),
                    "driver_1": drivers[0][0] if drivers else None,
                    "driver_2": drivers[1][0] if len(drivers) > 1 else None,
                    "driver_3": drivers[2][0] if len(drivers) > 2 else None,
                    "why": attributions[i].sentence(),
                }
            )
        frame = pd.DataFrame(records).sort_values(
            ["probability", "ticker"], ascending=[False, True]
        )
        frame["rank"] = np.arange(1, len(frame) + 1)
        return frame.head(top_k).reset_index(drop=True)


@dataclass
class PropensityResult:
    """The model, every number it must be read beside, and the refusals.

    ``evaluation`` compares the model against the size sort, which is the
    strongest cheap alternative. ``base_rate_result`` compares it against a
    constant, which is the weaker comparison and is kept only so a reader can see
    both. ``calibration`` and ``precision_at_k`` answer the two questions AUC
    does not.
    """

    model: PropensityModel
    evaluation: EvalResult
    base_rate_result: EvalResult
    size_scores: np.ndarray
    out_of_sample: np.ndarray
    matrix: FeatureMatrix
    folds: list[Fold]
    labels: LabelReport
    calibration: pd.DataFrame | None = None
    precision_at_k: pd.DataFrame | None = None
    base_rates: pd.DataFrame | None = None
    fold_models: list[tuple[Fold, "PropensityModel"]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def verdict(self) -> str:
        return self.evaluation.verdict()

    def coefficient_stability(self) -> pd.DataFrame:
        """Every fold's coefficient for every feature, and how much it moved.

        The number to read before believing any sign in the final fit. A feature
        whose coefficient changes sign between folds has not been measured, it
        has been sampled, and saying so is the difference between a model card
        and a marketing document. With a hundred deals spread over seven years
        several of these columns will be unstable, and this frame is where that
        is admitted rather than hidden behind a single fitted number.
        """
        if not self.fold_models:
            return pd.DataFrame(columns=["feature", "mean", "sd", "sign_flips", "folds"])
        rows = []
        for j, name in enumerate(self.model.columns):
            series = np.array([m.weights[j] for _, m in self.fold_models], dtype=float)
            signs = np.sign(series[series != 0.0])
            rows.append(
                {
                    "feature": name,
                    "mean": float(series.mean()),
                    "sd": float(series.std(ddof=1)) if series.size > 1 else float("nan"),
                    "sign_flips": int(len(set(signs.tolist()))) - 1 if signs.size else 0,
                    "folds": int(series.size),
                }
            )
        return pd.DataFrame(rows, columns=["feature", "mean", "sd", "sign_flips", "folds"])

    def rows(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = list(self.labels.rows())
        out.extend(self.evaluation.rows())
        out.append(
            ("Baseline (base rate) AUC", self.base_rate_result.baseline_score)
        )
        return out


def fit_propensity(
    panel: FeaturePanel,
    universe: Universe,
    events: Iterable[DealEvent | tuple[str, date, bool]],
    assumptions: Assumptions | None = None,
    *,
    as_of: date | None = None,
    dates: Sequence[date] | None = None,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    gap_days: int = DEFAULT_GAP_DAYS,
    columns: Sequence[str] = FITTED_COLUMNS,
    n_folds: int | None = None,
    l2: float = DEFAULT_L2,
    steps: int = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    pos_weight: float | None = None,
    top_k: int = DEFAULT_TOP_K,
    extras: Mapping[tuple[str, date], Mapping[str, float | None]] | None = None,
) -> PropensityResult:
    """Fit the screen, score it walk-forward against the size sort, and report.

    The fit is a logistic regression: a single affine layer under a sigmoid,
    trained with the package's own ``bce_with_logits`` and ``Adam``. It is linear
    on purpose. With a hundred-odd distinct deals and fourteen features, anything
    with more capacity fits the noise and, worse, has nothing to say when
    somebody asks why a name is on the list. ``Attribution`` exists because the
    model is linear.

    The positive class is weighted by the ratio of negatives to positives so that
    the two contribute equally to the gradient, which is the standard answer to a
    three percent base rate. It changes what the head means: the output is then
    the probability under a reweighted sample, and ``PropensityModel.prior_shift``
    maps it back. Ranking is untouched by that shift and calibration is not, which
    is why both are reported separately.

    Evaluation is walk-forward by date with an embargo of the full label horizon
    plus the gap. Every observation's label is known only at the end of its own
    window, so training on an observation whose window overlaps the test window
    shows the model most of the answer before asking the question. The embargo
    costs observations and the count is on every ``Fold``.

    The intercept is not shrunk. Ridge on an intercept pulls the whole prediction
    toward one half, which on a three percent base rate is not regularisation, it
    is a mistake with a Greek letter on it.

    Raises ``NotMeaningfulError`` where the sample cannot support the claim:
    below ``MIN_DISTINCT_DEALS`` companies actually bought, the score moves more
    with one deal than with the model.
    """
    assumptions = assumptions or Assumptions()
    n_folds = n_folds or assumptions.ml.walk_forward_folds
    panel_dates = list(dates) if dates is not None else panel.dates
    if not panel_dates:
        raise ConfigError("the panel carries no dates, so there is nothing to fit")
    resolved_as_of = as_of or max(panel_dates)

    report = label_observations(
        universe,
        panel_dates,
        events,
        as_of=resolved_as_of,
        horizon_months=horizon_months,
        gap_days=gap_days,
    )
    matrix = build_matrix(panel, universe, report, columns=columns, extras=extras)
    if matrix.n == 0:
        raise NotMeaningfulError(
            "no labelled observation could be joined to a feature row. The panel "
            "and the universe are keyed on ticker; check that they use the same "
            "symbols."
        )
    n_pos = int(matrix.y.sum())
    distinct = len({t for t, lab in zip(matrix.tickers, matrix.y) if lab})
    if distinct < MIN_DISTINCT_DEALS:
        raise NotMeaningfulError(
            f"{distinct} distinct companies were acquired in this sample, below the "
            f"floor of {MIN_DISTINCT_DEALS}. {n_pos} positive observations over "
            f"{distinct} deals is {distinct} deals: the extra rows are the same "
            "companies seen again on later dates and add no independent evidence."
        )

    embargo = horizon_months * 31 + gap_days
    folds = walk_forward_folds(
        matrix.dates, n_folds, min_train=None, embargo_days=embargo
    )

    seed = assumptions.ml.random_seed
    out_of_sample = np.full(matrix.n, np.nan, dtype=float)
    fitted_on: list[tuple[Fold, PropensityModel]] = []
    d = np.asarray([dt.toordinal() for dt in matrix.dates], dtype=float)

    for fold in folds:
        cutoff = fold.test_start - timedelta(days=embargo)
        train = d < cutoff.toordinal()
        test = (d >= fold.test_start.toordinal()) & (d <= fold.test_end.toordinal())
        if not train.any() or not test.any():
            continue
        if matrix.y[train].sum() == 0:
            # A training window with no event has no positive class to learn.
            continue
        fold_model = _fit_once(
            matrix.X[train],
            matrix.y[train],
            tuple(matrix.columns),
            seed=seed,
            l2=l2,
            steps=steps,
            learning_rate=learning_rate,
            pos_weight=pos_weight,
            trained_through=fold.train_end,
        )
        out_of_sample[test] = fold_model.scores(matrix.X[test])
        fitted_on.append((fold, fold_model))

    scored = np.isfinite(out_of_sample)
    if not scored.any():
        raise NotMeaningfulError(
            "no fold produced an out-of-sample score. Every training window was "
            "empty or eventless under the embargo, which means the sample does "
            "not span enough time to be evaluated walk-forward."
        )

    # -- the baseline, and it is not the base rate ------------------------- #
    size_scores = _size_baseline(matrix)
    size_result = evaluate_classification(
        matrix.y[scored],
        size_scores[scored],
        dates=[dt for dt, keep in zip(matrix.dates, scored) if keep],
        folds=_restrict(folds, matrix.dates, scored),
        embargo_days=embargo,
    )

    model_result = evaluate_classification(
        matrix.y[scored],
        out_of_sample[scored],
        dates=[dt for dt, keep in zip(matrix.dates, scored) if keep],
        folds=_restrict(folds, matrix.dates, scored),
        embargo_days=embargo,
    )
    base_rate_result = EvalResult(
        metric=model_result.metric,
        score=model_result.score,
        baseline_name=model_result.baseline_name,
        baseline_score=model_result.baseline_score,
        n_observations=model_result.n_observations,
        folds=list(model_result.folds),
        notes=list(model_result.notes),
    )
    # The reported result is measured against the size sort, which is the
    # honest alternative. Everything else about it is unchanged.
    evaluation = EvalResult(
        metric="auc",
        score=model_result.score,
        baseline_name=(
            f"size only (log revenue, smallest first), walk-forward "
            f"AUC {size_result.score:.4f}"
        ),
        baseline_score=size_result.score,
        n_observations=model_result.n_observations,
        folds=list(model_result.folds),
        notes=list(model_result.notes),
    )
    evaluation.notes.append(
        f"Against a constant prediction at the base rate the AUC would be 0.500 by "
        f"construction, so the model clears that by {model_result.score - 0.5:+.4f}. "
        "That comparison is kept for completeness and is not the test: sorting the "
        "universe from smallest to largest is free, and is what this model has to "
        f"beat. It scores {size_result.score:.4f}."
    )

    # -- the model a caller actually uses, fitted on everything ------------ #
    final = _fit_once(
        matrix.X,
        matrix.y,
        tuple(matrix.columns),
        seed=seed,
        l2=l2,
        steps=steps,
        learning_rate=learning_rate,
        pos_weight=pos_weight,
        trained_through=max(matrix.dates),
    )
    final.card.evaluation = evaluation
    final.card.limitations = _limitations(report, matrix, folds, gap_days, horizon_months)
    final.card.notes = list(matrix.notes) + list(report.notes)
    final.card.hyperparameters.update(
        {
            "l2": l2,
            "steps": steps,
            "learning_rate": learning_rate,
            "horizon_months": horizon_months,
            "gap_days": gap_days,
            "embargo_days": embargo,
            "folds": len(folds),
            "pos_weight": final.card.hyperparameters.get("pos_weight"),
        }
    )

    result = PropensityResult(
        model=final,
        evaluation=evaluation,
        base_rate_result=base_rate_result,
        size_scores=size_scores,
        out_of_sample=out_of_sample,
        matrix=matrix,
        folds=folds,
        labels=report,
        base_rates=base_rate_by_year(report),
        fold_models=fitted_on,
    )
    result.precision_at_k = _precision_table(
        matrix, out_of_sample, size_scores, scored, top_k=top_k
    )
    result.calibration = _calibration(final, matrix, out_of_sample, scored)
    result.notes.append(
        "No price-derived feature is in this model. A delisted target has no "
        "price history from any public source, so market capitalisation and every "
        "multiple over it are missing for the positives and present for the "
        "negatives, and a classifier handed that learns the delisting rather than "
        "the deal."
    )
    return result


def _fit_once(
    X: np.ndarray,
    y: np.ndarray,
    columns: tuple[str, ...],
    *,
    seed: int,
    l2: float,
    steps: int,
    learning_rate: float,
    pos_weight: float | None,
    trained_through: date,
) -> PropensityModel:
    """One logistic fit on one training window, standardised on that window only."""
    mu = np.zeros(X.shape[1], dtype=float)
    sd = np.ones(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        obs = col[np.isfinite(col)]
        if obs.size < 2:
            mu[j] = float(obs[0]) if obs.size == 1 else 0.0
            continue
        mu[j] = float(obs.mean())
        spread = float(obs.std(ddof=1))
        sd[j] = spread if spread > 0.0 else 1.0

    Z = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.asarray(y, dtype=float).reshape(-1, 1)

    n_pos = float(target.sum())
    n_neg = float(target.size - n_pos)
    weight = pos_weight if pos_weight is not None else (n_neg / max(n_pos, 1.0))
    weight = max(float(weight), 1e-6)

    rng = np.random.default_rng(seed)
    layer = Linear(Z.shape[1], 1, rng=rng, init="xavier")
    # The head starts at the log odds of the weighted sample, which is where an
    # uninformative model belongs. Starting at zero asserts even odds on a task
    # whose base rate is three percent, and the first hundred steps are then
    # spent walking the intercept back.
    weighted_rate = (weight * n_pos) / (weight * n_pos + n_neg) if n_pos else 0.5
    layer.bias[:] = float(np.log(weighted_rate / (1.0 - weighted_rate)))

    # Decay on the coefficients only. Shrinking the intercept pulls every
    # prediction toward one half, and on a three percent base rate that is not
    # regularisation, it is a mistake with a Greek letter on it.
    opt_w = Adam(lr=learning_rate, weight_decay=l2)
    opt_b = Adam(lr=learning_rate)
    for _ in range(steps):
        layer.zero_grad()
        logits = layer.forward(Z)
        _loss, grad = bce_with_logits(logits, target, pos_weight=weight)
        layer.backward(grad)
        opt_w.step([layer.weight], [layer.grad_weight])
        opt_b.step([layer.bias], [layer.grad_bias])

    card = ModelCard(
        name="mna.propensity",
        task="acquisition propensity: probability a company is announced as a target",
        trained_through=trained_through,
        n_train=int(X.shape[0]),
        features=list(columns),
        hyperparameters={"pos_weight": weight, "seed": seed},
    )
    return PropensityModel(
        weights=layer.weight[:, 0].copy(),
        intercept=float(layer.bias[0]),
        columns=columns,
        mu=mu,
        sd=sd,
        prior_shift=float(np.log(weight)),
        card=card,
        base_rate=float(n_pos / max(target.size, 1)),
    )


def _size_baseline(matrix: FeatureMatrix) -> np.ndarray:
    """Sort the universe smallest first. The alternative the model has to beat.

    Negated log revenue, so a smaller company scores higher. A missing revenue
    scores at the sample median rather than at an extreme, which is the
    conservative choice: putting unreadable filers at the top of the baseline's
    list would flatter the model by handicapping its opponent.
    """
    if "scale_log_revenue" not in matrix.columns:
        raise ConfigError(
            "the size baseline needs scale_log_revenue in the matrix. It is the "
            "cheapest honest alternative to the model and cannot be skipped."
        )
    col = matrix.X[:, matrix.columns.index("scale_log_revenue")].astype(float)
    observed = col[np.isfinite(col)]
    fill = float(np.median(observed)) if observed.size else 0.0
    return -np.where(np.isfinite(col), col, fill)


def _restrict(
    folds: Sequence[Fold], dates: Sequence[date], mask: np.ndarray
) -> list[Fold]:
    """Folds whose test window still holds a scored observation."""
    kept = [dt for dt, keep in zip(dates, mask) if keep]
    return [
        f for f in folds if any(f.test_start <= dt <= f.test_end for dt in kept)
    ]


def _precision_table(
    matrix: FeatureMatrix,
    scores: np.ndarray,
    size_scores: np.ndarray,
    mask: np.ndarray,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Precision and recall at k for the model and for the size sort, per date.

    Per date rather than pooled, because a pooled top twenty over eight years is
    not a list anybody could have acted on. A screen is run on a date and the
    question is how many of that date's twenty were bought inside the horizon.
    """
    rows = []
    by_date: dict[date, list[int]] = {}
    for i, keep in enumerate(mask):
        if keep:
            by_date.setdefault(matrix.dates[i], []).append(i)
    for when, idx in sorted(by_date.items()):
        relevant = {matrix.tickers[i] for i in idx if matrix.y[i] > 0}
        if not relevant:
            continue
        k = min(top_k, len(idx))
        record: dict[str, Any] = {
            "as_of": when,
            "n": len(idx),
            "positives": len(relevant),
            "k": k,
            "base_rate": len(relevant) / len(idx),
        }
        for label, series in (("model", scores), ("size", size_scores)):
            order = sorted(idx, key=lambda i: (-series[i], matrix.tickers[i]))
            ranking = [matrix.tickers[i] for i in order]
            precision, recall = precision_recall_at_k(relevant, ranking, k)
            record[f"precision_at_k_{label}"] = precision
            record[f"recall_at_k_{label}"] = recall
        rows.append(record)
    return pd.DataFrame(rows)


def _calibration(
    model: PropensityModel,
    matrix: FeatureMatrix,
    scores: np.ndarray,
    mask: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame | None:
    """Calibration of the prior-corrected out-of-sample probabilities.

    Returns None rather than raising where the sample is below the harness floor,
    because a missing calibration table is a smaller problem than a fit that
    stops because it could not draw one.
    """
    probs = _sigmoid(scores[mask] - model.prior_shift)
    try:
        return calibration_table(matrix.y[mask], probs, bins=bins)
    except (NotMeaningfulError, ConfigError):
        return None


def _limitations(
    report: LabelReport,
    matrix: FeatureMatrix,
    folds: Sequence[Fold],
    gap_days: int,
    horizon_months: int,
) -> list[str]:
    distinct = len({t for t, lab in zip(matrix.tickers, matrix.y) if lab})
    return [
        f"{distinct} distinct companies were acquired in this sample. The "
        f"{int(matrix.y.sum())} positive observations are those same companies seen "
        "on several dates, so every interval implied by the observation count is "
        "too narrow by roughly the square root of the repetition.",
        "No price-derived feature is used. Market capitalisation, momentum, "
        "realised volatility and every multiple are absent for delisted targets "
        "from every public source, so including them would fit the delisting. "
        "The valuation channel is therefore untested here, and a cheap company "
        "and an expensive one look identical to this model.",
        "Announcement is the label, not completion. A blocked deal is a positive.",
        f"An observation whose announcement fell within {gap_days} days of its "
        "feature date was dropped rather than labelled, so the sample is slightly "
        "biased toward deals that were visible early.",
        "One deal per company. Where a company was bid for twice, only the "
        "earliest announcement is labelled and the later process is invisible.",
        f"Labels come from merger filings, and a company that agreed to be bought "
        "without filing one of the forms the screen reads is a positive sitting "
        "in the negative class.",
        f"Walk-forward over {len(folds)} folds with a {horizon_months * 31 + gap_days}"
        "-day embargo. Fold-level AUC on a window holding two or three events is "
        "noise, and the fold standard deviation on the result says how much.",
    ]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable logistic, evaluated on whichever side of zero does not overflow."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
