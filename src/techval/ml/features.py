"""The point-in-time financial feature store every model in this package trains on.

Fifty ratios and growth rates per company per date, each one built only from
filings that were on file at that date and prices that had printed by it. A leak
here is not a local defect: it contaminates every model fitted downstream, and it
does so invisibly, because a model that has seen the future fits beautifully.

**Point in time is the whole job.** Each row is built through a
``CompanyFacts`` carrying a ``knowledge_date`` and a ``MarketData`` whose price
series stops at the row date. Those two primitives already exist and this module
does not reimplement them, it insists on them: ``build_features`` refuses a
client that was not pinned and refuses a market feed that runs past the row date.
Afterwards ``assert_point_in_time`` walks the provenance of every figure that
reached the row, including the historical trailing-twelve-month resolutions this
module makes for growth and margin change, and checks the filing date behind each
one. The check is on the filing date, never the period end, and it is the
difference between a feature store that is evidence and one that is an assertion.

**The reporting lag is handled by that construction, not by a rule.** A fiscal
quarter ending 31 December is not public information on 2 January. It becomes
public when the 10-K or 10-Q carrying it is filed, typically four to eight weeks
later, and until then nobody could have traded on it. Because every fact is
filtered on its filing date, a row dated 15 January 2025 for a calendar-year
filer anchors on the September quarter and knows nothing of December. The row
records which balance-sheet date it actually landed on in ``statement_date``, so
the lag is visible rather than assumed, and the test suite pins that case.

**Ratios, not dollars.** A model given raw revenue learns that large companies
are large, which is true, already known, and useless. Every feature here is a
ratio, a growth rate, or an explicit log-scale size term. The three size terms
are logged because the cross-section of market capitalisation spans four orders
of magnitude and an untransformed one would make the largest name a leverage
point in every linear fit.

**Missing is a value, not a zero.** ``FeatureRow.values`` holds ``None`` where a
figure could not be sourced. Filling a missing gross margin with zero tells the
model the company broke even, which is a specific and usually false claim, and
the model has no way to tell that claim apart from a real one. Filling it with
the cross-sectional mean is worse, because it also leaks the cross-section. So
the value stays ``None``, the name goes into ``FeatureRow.missing``, and
``FeaturePanel.to_matrix`` emits one missing-share indicator per feature group
beside the design matrix. What to do about it is the consumer's decision to make
and to state.

**Two different reasons a feature is absent, kept apart.** A figure the filer
never reported is missing. A ratio whose denominator has the wrong sign is not
missing, it is not meaningful: net debt over a negative EBITDA is a negative
number that reads as conservative financing and means the opposite. Those return
``None`` with the reason recorded in ``notes``, in the same spirit as the NM that
the comps tables print. Winsorization is the answer to a denominator that is
small, never to one whose sign has flipped.

**Winsorize cross-sectionally, at each date.** A company with 1mm of trailing
EBITDA has a net-debt-to-EBITDA of 400x, and left alone that single observation
sets the scale of the whole column. The 1st and 99th percentiles are taken across
the names present on each date and values outside them are pulled to the
boundary, never dropped and never silently clipped: every touched value is
counted in a ``WinsorReport``. Cross-sectional rather than pooled across dates,
because a pooled percentile bakes the level of a whole era into the bound, and
the software cross-section of 2022 and of 2026 are not the same distribution.

**Standardisation fits on training rows only.** ``FeaturePanel.standardize``
will not run without ``fit_rows``. Standardising the full panel and splitting
afterwards leaks the mean and the dispersion of the test period into the fit,
which is a small leak that produces a large and entirely fake improvement in
walk-forward scores.

**Determinism.** Nothing in this module is random. Column order is fixed by
``FEATURE_NAMES``, row order by the order of the dates and tickers handed in, and
every statistic is a closed-form function of the inputs, so two runs on the same
inputs agree exactly rather than nearly. ``assumptions.ml.random_seed`` is
recorded on the panel so a model card fitted from it can cite one, but no
operation here consumes it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import tags
from ..backtest import LookaheadError, assert_no_lookahead
from ..config import Assumptions
from ..edgar import CompanyFacts, Fact, Provenance, derive_periods
from ..errors import ConfigError, MissingDataError, TechvalError
from ..ev_bridge import build_ev_bridge
from ..financials import Financials, build_financials
from ..market import MarketData, PriceSeries
from ..wacc import estimate_beta

# Money arrives from XBRL in units and leaves this module in USD millions, the
# same convention every other statement object in the engine uses.
_MM = 1e6


# --------------------------------------------------------------------------- #
# Tag ladders this module needs and ``tags.py`` does not carry
# --------------------------------------------------------------------------- #

# Operating expense lines. ``tags.py`` stops at the subtotals a valuation needs;
# research and sales spend are features rather than valuation inputs, so their
# ladders live here. Both are ordered most specific first: a filer that reports a
# combined research-and-development-and-engineering line is caught by the second
# entry only when the first is absent.
RND_EXPENSE = [
    "ResearchAndDevelopmentExpense",
    "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
]
SNM_EXPENSE = [
    "SellingAndMarketingExpense",
    "MarketingAndAdvertisingExpense",
    "SellingGeneralAndAdministrativeExpense",
]


# --------------------------------------------------------------------------- #
# The contract: names, order, groups
# --------------------------------------------------------------------------- #

# The ablation study reports by group, so the grouping is part of the contract
# and not a comment. Insertion order here defines column order everywhere.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "scale": (
        "scale_log_revenue",
        "scale_log_market_cap",
        "scale_log_enterprise_value",
        "scale_log_total_assets",
    ),
    "growth": (
        "growth_revenue_1y",
        "growth_revenue_cagr_2y",
        "growth_revenue_cagr_3y",
        "growth_acceleration",
        "growth_gross_profit_1y",
        "growth_deferred_revenue_1y",
    ),
    "margin": (
        "margin_gross",
        "margin_ebitda",
        "margin_ebit",
        "margin_net",
        "margin_fcf",
        "margin_gross_chg_1y",
        "margin_ebitda_chg_1y",
        "margin_ebit_chg_1y",
        "margin_net_chg_1y",
        "margin_fcf_chg_1y",
    ),
    "efficiency": (
        "efficiency_sbc_to_revenue",
        "efficiency_rnd_to_revenue",
        "efficiency_snm_to_revenue",
        "efficiency_capex_to_revenue",
        "efficiency_da_to_revenue",
        "efficiency_opex_to_revenue",
        "efficiency_sbc_to_gross_profit",
    ),
    "capital": (
        "capital_net_debt_to_ebitda",
        "capital_net_debt_to_market_cap",
        "capital_net_debt_to_ev",
        "capital_cash_to_market_cap",
        "capital_current_ratio",
        "capital_debt_to_capital",
    ),
    "returns": (
        "returns_roic_proxy",
        "returns_roe",
        "returns_asset_turnover",
        "returns_deferred_revenue_to_revenue",
        "returns_cash_conversion",
        "returns_rule_of_40",
    ),
    "market": (
        "market_momentum_12m",
        "market_momentum_6m",
        "market_momentum_3m",
        "market_relative_momentum_12m",
        "market_realised_volatility_1y",
        "market_beta",
        "market_52w_range_position",
    ),
    "quality": (
        "quality_accruals",
        "quality_revenue_volatility_3y",
        "quality_margin_volatility_3y",
        "quality_sbc_to_cfo",
    ),
}

FEATURE_NAMES: tuple[str, ...] = tuple(
    name for group in FEATURE_GROUPS.values() for name in group
)

GROUP_OF: dict[str, str] = {
    name: group for group, names in FEATURE_GROUPS.items() for name in names
}

# One indicator per group rather than per feature. Fifty binary columns beside
# fifty real ones would double the width of the design matrix to encode a
# pattern that is almost entirely group-level: a filer either reports a cash flow
# statement this engine can parse or it does not. The indicator is the share of
# the group that is missing rather than a flag, which carries strictly more
# information at the same width.
MISSING_INDICATORS: tuple[str, ...] = tuple(
    f"missing_share_{group}" for group in FEATURE_GROUPS
)

MATRIX_COLUMNS: tuple[str, ...] = FEATURE_NAMES + MISSING_INDICATORS

# Winsorization bounds. Not in ``Assumptions`` because they are a property of
# this feature store rather than of a valuation, and they are exposed as keyword
# arguments on ``winsorize`` and ``build_panel`` so a caller who wants a
# different trim states it at the call site rather than editing a constant.
WINSOR_LOWER_PCT = 1.0
WINSOR_UPPER_PCT = 99.0

# Below this many observed values in a date's cross-section the 1st and 99th
# percentiles fall between the first and second order statistics, so the
# operation degenerates into pulling the single most extreme name to its
# neighbour. That says something about the sample size and nothing about the
# data, so it is skipped and the skip is reported rather than performed quietly.
WINSOR_MIN_OBSERVATIONS = 20

# A price at a target date is taken from the last trading day at or before it.
# Beyond this many calendar days the series does not actually reach back that
# far, and returning its first close would label an eleven-month return as a
# twelve-month one.
_PRICE_LOOKBACK_SLACK_DAYS = 15

# A price feed that stops more than a fortnight before the row date is stale
# rather than capped, and the row says so. Long enough to absorb a Christmas
# week, short enough that a feed which quietly stopped updating is caught.
_STALE_PRICE_DAYS = 14

# A fiscal quarter is thirteen weeks, so the anniversary of a period end is 364
# or 371 days earlier rather than 365. The tolerance is well inside the ninety
# days that separate two quarter ends, so no anniversary is ambiguous.
_ANNIVERSARY_TOLERANCE_DAYS = 25

# Quarterly windows used for the volatility features. A thirteen-week quarter
# runs 90 to 92 days; the band allows for a filer whose quarter ends on a fixed
# weekday and for the 53-week fiscal year that stretches one of them.
_QUARTER_MIN_DAYS = 80
_QUARTER_MAX_DAYS = 100

# Weekly returns needed before a realised volatility is reported. Forty of the
# fifty-two weeks in a year, so a newly listed name is reported as missing rather
# than as unusually calm.
_MIN_VOL_OBSERVATIONS = 40


# --------------------------------------------------------------------------- #
# Small guarded arithmetic
# --------------------------------------------------------------------------- #


def _ratio(num: float | None, den: float | None) -> float | None:
    """Numerator over denominator, or None when either side is unavailable.

    The denominator is required to be strictly positive. Every ratio in this
    module divides by a quantity that is economically positive when it means
    anything at all: revenue, total assets, market capitalisation, book equity,
    gross profit. A zero or negative denominator produces a number whose sign
    says the opposite of what a reader would take it to say, which is the exact
    failure the engine prints NM for elsewhere.
    """
    if num is None or den is None or den <= 0:
        return None
    return num / den


def _growth(new: float | None, old: float | None) -> float | None:
    """Period-on-period growth, defined only from a positive base."""
    if new is None or old is None or old <= 0:
        return None
    return new / old - 1.0


def _cagr(new: float | None, old: float | None, years: float) -> float | None:
    """Compound annual growth, defined only where both ends are positive.

    A negative endpoint has no real root, and a company whose revenue went
    negative has a bigger problem than a missing feature.
    """
    if new is None or old is None or old <= 0 or new <= 0 or years <= 0:
        return None
    return (new / old) ** (1.0 / years) - 1.0


def _log_size(x: float | None) -> float | None:
    """Natural log of a size in USD millions, or None where the size is not positive."""
    if x is None or x <= 0:
        return None
    return math.log(x)


def _diff(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return new - old


def _add(*parts: float | None) -> float | None:
    """Sum, or None if any part is missing. Never treats an absent part as zero."""
    total = 0.0
    for p in parts:
        if p is None:
            return None
        total += p
    return total


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class FeatureRow:
    """One company's features as they stood on one date.

    ``as_of`` is the date the row claims to describe. ``knowledge_date`` is the
    date the client feeding it was actually pinned to, which is what the point-in
    -time guarantee rests on, and the two are kept apart so a row built through a
    client pinned earlier than its own date is visible rather than assumed away.

    ``statement_date`` is the balance-sheet date the trailing figures anchor on.
    For a calendar-year filer on a row dated 15 January it is the previous
    September, because the December quarter had not been filed. That is the
    reporting lag, and it is on the row rather than in a comment.

    ``values`` holds ``None``, not a filled float, wherever a figure could not be
    sourced. ``missing`` lists those names. ``notes`` carries the reason wherever
    the reason is a judgment rather than an absence: a ratio suppressed because
    its denominator had the wrong sign says so here.

    ``error`` is set only on a row that could not be built at all. Every value is
    ``None`` on such a row and it is kept in the panel rather than dropped,
    because the count of what could not be built is part of the result.
    """

    ticker: str
    as_of: date
    knowledge_date: date | None
    values: dict[str, float | None] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    statement_date: date | None = None
    error: str | None = None
    provenance: dict[str, Provenance] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def n_observed(self) -> int:
        return len(FEATURE_NAMES) - len(self.missing)

    def missing_share(self, group: str) -> float:
        """Share of one feature group that could not be sourced on this row."""
        names = FEATURE_GROUPS[group]
        absent = sum(1 for n in names if self.values.get(n) is None)
        return absent / len(names)

    def vector(self, *, include_missing_indicators: bool = True) -> np.ndarray:
        """One row of the design matrix, with NaN standing in for None.

        NaN rather than None because a float array cannot hold None, and the
        column names returned beside the matrix say which entries are indicators
        rather than measurements. The substitution happens here and nowhere else,
        so ``values`` stays the honest record.
        """
        vals = [
            np.nan if self.values.get(n) is None else float(self.values[n])
            for n in FEATURE_NAMES
        ]
        if include_missing_indicators:
            vals.extend(self.missing_share(g) for g in FEATURE_GROUPS)
        return np.asarray(vals, dtype=float)

    def rows(self) -> list[tuple[str, object]]:
        """Header rows for rendering. Formatting itself lives in cli.py."""
        return [
            ("Ticker", self.ticker),
            ("As of", str(self.as_of)),
            ("Knowledge date", str(self.knowledge_date or "-")),
            ("Statement date", str(self.statement_date or "-")),
            ("Features observed", f"{self.n_observed} of {len(FEATURE_NAMES)}"),
            ("Error", self.error or ""),
        ]


def _failed_row(ticker: str, as_of: date, reason: str) -> FeatureRow:
    """A row that could not be built, with every value left as None.

    A zero here would enter the cross-section as a genuine measurement of no
    growth and no leverage, and would be winsorized and standardised alongside
    real ones. Nothing was measured, so nothing is recorded.
    """
    return FeatureRow(
        ticker=ticker.upper(),
        as_of=as_of,
        knowledge_date=as_of,
        values={name: None for name in FEATURE_NAMES},
        missing=list(FEATURE_NAMES),
        error=reason,
    )


@dataclass
class WinsorReport:
    """What winsorization did to one feature in one date's cross-section.

    ``applied`` is False where the cross-section was too thin to take a
    percentile from, and ``note`` says so. A report is emitted either way: a
    trim that did not happen is as much a part of the record as one that did.
    """

    as_of: date
    feature: str
    n_observed: int
    lower: float | None = None
    upper: float | None = None
    n_pulled_low: int = 0
    n_pulled_high: int = 0
    applied: bool = True
    note: str | None = None

    @property
    def n_touched(self) -> int:
        return self.n_pulled_low + self.n_pulled_high


@dataclass
class FeaturePanel:
    """Feature rows for a set of companies across a set of dates.

    Rows are in the order they were built, dates outer and tickers inner, and
    include the ones that failed. ``winsorization`` is the full record of what
    the cross-sectional trim touched, so the count of altered values is auditable
    rather than a claim.
    """

    rows: list[FeatureRow]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    winsorization: list[WinsorReport] = field(default_factory=list)
    random_seed: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def dates(self) -> list[date]:
        """Distinct row dates, in the order they first appear."""
        seen: dict[date, None] = {}
        for r in self.rows:
            seen.setdefault(r.as_of, None)
        return list(seen)

    @property
    def tickers(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.rows:
            seen.setdefault(r.ticker, None)
        return list(seen)

    @property
    def n_winsorized(self) -> int:
        return sum(w.n_touched for w in self.winsorization)

    def ok_rows(self) -> list[FeatureRow]:
        return [r for r in self.rows if r.ok]

    def failures(self) -> list[FeatureRow]:
        return [r for r in self.rows if not r.ok]

    # -- shaping ----------------------------------------------------------- #

    def to_frame(self) -> pd.DataFrame:
        """A frame for inspection, one row per observation.

        NaN in a feature column means the figure could not be sourced. The frame
        is a rendering of the panel and not the panel itself: ``FeatureRow.values``
        keeps ``None``, because a frame cannot, and a consumer that reads NaN as a
        number will get a different answer from one that reads it as absent.
        """
        records = []
        for r in self.rows:
            rec: dict[str, object] = {
                "ticker": r.ticker,
                "as_of": r.as_of,
                "knowledge_date": r.knowledge_date,
                "statement_date": r.statement_date,
                "n_missing": len(r.missing),
                "error": r.error,
            }
            for name in self.feature_names:
                v = r.values.get(name)
                rec[name] = np.nan if v is None else float(v)
            records.append(rec)
        return pd.DataFrame.from_records(records)

    def to_matrix(
        self, *, include_missing_indicators: bool = True
    ) -> tuple[np.ndarray, list[str], list[date]]:
        """Design matrix, column names, and the date of each row.

        Shape is ``(n_rows, n_features [+ n_groups])``. Missing values are NaN.
        Nothing is imputed here and nothing is dropped, including rows that
        failed to build, which appear as all-NaN. Deciding what to do about a
        NaN is a modelling choice, and it belongs to the model rather than to the
        store that reports the gap.
        """
        names = list(MATRIX_COLUMNS if include_missing_indicators else FEATURE_NAMES)
        if not self.rows:
            return np.zeros((0, len(names)), dtype=float), names, []
        X = np.vstack(
            [
                r.vector(include_missing_indicators=include_missing_indicators)
                for r in self.rows
            ]
        )
        return X, names, [r.as_of for r in self.rows]

    # -- standardisation --------------------------------------------------- #

    def standardize(
        self,
        fit_rows: Sequence[FeatureRow] | Sequence[bool] | date | None = None,
        *,
        include_missing_indicators: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centre and scale the whole panel on statistics fitted to ``fit_rows`` only.

        ``fit_rows`` may be the training rows themselves, a boolean mask the
        length of ``rows``, or a cutoff date meaning every row at or before it,
        which is the shape a walk-forward split arrives in.

        It has no usable default. Standardising the full panel and splitting
        afterwards moves the mean and the dispersion of the test period into the
        fit, and the resulting out-of-sample score is better than the model
        deserves by an amount nobody can estimate after the fact. Passing None
        raises rather than quietly doing the convenient thing.

        Returns ``(X, mu, sd)`` where ``X`` is every row standardised and ``mu``
        and ``sd`` are the fitted statistics, so the transform can be reapplied
        to a later panel exactly. ``sd`` is the scale that was actually applied.
        A column with no variation across the fit rows cannot be scaled and is
        left at unit scale, which keeps ``X * sd + mu`` an exact inverse;
        dividing by zero would put infinities in the design matrix and dividing
        by an epsilon would put 1e15 there. Those columns are named in ``notes``.

        Sample standard deviation, denominator n-1, because these are estimates
        from a sample of companies and not the population of them.
        """
        if fit_rows is None:
            raise ConfigError(
                "FeaturePanel.standardize requires fit_rows. Fitting the mean and "
                "the standard deviation on the whole panel and splitting it "
                "afterwards leaks the test period's distribution into the "
                "transform, which inflates every out-of-sample score built on it. "
                "Pass the training rows, a boolean mask over rows, or the cutoff "
                "date of the training window."
            )

        mask = self._fit_mask(fit_rows)
        if not mask.any():
            raise ConfigError(
                "FeaturePanel.standardize was given a fit set containing no rows "
                "of this panel; there is nothing to fit the transform on"
            )

        X, names, _ = self.to_matrix(
            include_missing_indicators=include_missing_indicators
        )
        mu = np.zeros(X.shape[1], dtype=float)
        sd = np.ones(X.shape[1], dtype=float)
        degenerate: list[str] = []

        for j in range(X.shape[1]):
            col = X[mask, j]
            observed = col[np.isfinite(col)]
            if observed.size < 2:
                # Never observed, or observed once, on the fit rows. There is no
                # dispersion to divide by and centring on the test rows' own mean
                # would be fitting on the test set, so the column passes through
                # on its own scale and is named.
                mu[j] = float(observed[0]) if observed.size == 1 else 0.0
                sd[j] = 1.0
                degenerate.append(names[j])
                continue
            mu[j] = float(observed.mean())
            spread = float(observed.std(ddof=1))
            if spread == 0.0:
                sd[j] = 1.0
                degenerate.append(names[j])
            else:
                sd[j] = spread

        Z = (X - mu) / sd

        if degenerate:
            note = (
                f"{len(degenerate)} column(s) had no usable variation across the "
                f"{int(mask.sum())} fit rows and were centred but not scaled: "
                + ", ".join(degenerate)
            )
            if note not in self.notes:
                self.notes.append(note)

        return Z, mu, sd

    def _fit_mask(
        self, fit_rows: Sequence[FeatureRow] | Sequence[bool] | date
    ) -> np.ndarray:
        """Resolve the three accepted spellings of a training set into one mask."""
        if isinstance(fit_rows, date):
            return np.asarray([r.as_of <= fit_rows for r in self.rows], dtype=bool)

        items = list(fit_rows)
        if not items:
            # An empty training set is a real mistake with a clear message of its
            # own, raised by the caller above rather than dressed up as a mask of
            # the wrong length here.
            return np.zeros(len(self.rows), dtype=bool)
        if all(isinstance(x, (bool, np.bool_)) for x in items):
            if len(items) != len(self.rows):
                raise ConfigError(
                    f"fit_rows given as a boolean mask of length {len(items)} for a "
                    f"panel of {len(self.rows)} rows"
                )
            return np.asarray(items, dtype=bool)

        chosen = {id(r) for r in items}
        return np.asarray([id(r) in chosen for r in self.rows], dtype=bool)


# --------------------------------------------------------------------------- #
# The point-in-time proof
# --------------------------------------------------------------------------- #


def assert_point_in_time(
    row: FeatureRow,
    fin: Financials | None = None,
    *,
    prices: PriceSeries | None = None,
) -> None:
    """Prove nothing on this row was filed, or priced, after the row's own date.

    Three things are checked, and each of them catches a different real mistake.

    The provenance of the statements is walked through ``backtest.assert_no_lookahead``,
    which compares the filing date behind every line item against the row date.
    That catches a client built without a knowledge date, a cached payload reused
    from a later run, and a fixture loaded by the wrong helper, none of which any
    configuration check would notice.

    The provenance this module itself accumulated is walked the same way. The
    growth and margin-change features resolve trailing twelve months at earlier
    anchor dates, and those resolutions reach into the fact set separately from
    the ones ``build_financials`` made. Checking only the ``Financials`` would
    leave a third of the row unproven.

    The price series is checked for a close printed after the row date. A market
    feed built with the wrong ``today`` leaks the future through momentum rather
    than through a filing, and no amount of care with the fact set prevents it.

    A line item with no filing date is one defaulted to zero because no tag in
    the ladder reports it. It carries nothing from any filing, so there is
    nothing for it to have seen early.
    """
    if fin is not None:
        assert_no_lookahead(fin, row.as_of)

    late: list[str] = []
    for label, prov in row.provenance.items():
        if not prov.filed:
            continue
        filed = date.fromisoformat(prov.filed)
        if filed > row.as_of:
            late.append(
                f"{label} (tag {prov.tag or '-'}) filed {filed}, "
                f"{(filed - row.as_of).days} days after the row date"
            )
    if late:
        raise LookaheadError(
            f"{row.ticker} features dated {row.as_of} were built from "
            f"{len(late)} figure(s) not on file until later:\n  "
            + "\n  ".join(sorted(late))
            + "\n  Build the client with knowledge_date set to the row date."
        )

    if prices is not None and prices.last_date > row.as_of:
        raise LookaheadError(
            f"{row.ticker} features dated {row.as_of} were built from a price "
            f"series running to {prices.last_date}. The market feed was not "
            "capped. Build MarketData with today set to the row date."
        )


# --------------------------------------------------------------------------- #
# Period algebra over the fact set
# --------------------------------------------------------------------------- #


def _live_series(facts: CompanyFacts, ladder: Sequence[str]) -> list[Fact]:
    """Duration facts from the ladder tag the filer is currently reporting under.

    Not the first tag with any facts at all. A filer that migrated from one
    revenue concept to another leaves the old one in its fact set forever, and
    taking it would build a quarterly series that stopped whenever the migration
    happened. The tag whose newest period end is latest is the live one, which is
    the same test ``build_financials`` applies when it dates a balance sheet.
    """
    best: list[Fact] = []
    best_end: date | None = None
    for tag in ladder:
        series = [f for f in facts.facts(tag) if not f.is_instant]
        if not series:
            continue
        newest = max(f.end for f in series)
        if best_end is None or newest > best_end:
            best, best_end = series, newest
    return best


def _period_ends(facts: CompanyFacts) -> list[date]:
    """Sorted revenue period ends, which are the anchors a trailing window can end on.

    Revenue rather than any balance-sheet concept, for the same reason
    ``build_financials`` anchors on it: it is the one line every filer reports
    every period, so its ends are the filer's actual fiscal calendar. Ends are
    pooled across every revenue tag in the ladder, so a filer mid-migration keeps
    the anchors that sit on either side of the change.
    """
    return sorted(
        {f.end for tag in tags.REVENUE for f in facts.facts(tag) if not f.is_instant}
    )


def _anniversary(ends: Sequence[date], anchor: date, years: float) -> date | None:
    """The filer's own period end closest to ``years`` before ``anchor``.

    Subtracting 365 days from a quarter end lands between two fiscal quarters for
    a filer whose quarters are thirteen weeks, and lands a week out after a
    53-week year. Snapping to a reported period end instead means the trailing
    window that follows can actually be tiled from filed periods.
    """
    target = anchor - timedelta(days=round(365.25 * years))
    candidates = [
        e for e in ends if abs((e - target).days) <= _ANNIVERSARY_TOLERANCE_DAYS
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs((e - target).days))


def _quarterly_series(
    facts: CompanyFacts, ladder: Sequence[str], through: date, years: float
) -> dict[date, float]:
    """Discrete quarters for one concept, keyed by period end, in USD millions.

    Filers report cumulative year-to-date figures as well as discrete quarters,
    and never report the fourth quarter at all. ``derive_periods`` recovers the
    discrete windows by subtraction, which is the only way to get a quarterly
    series out of a US filer's fact set. Windows outside the thirteen-week band
    are dropped so a stray half-year does not enter the series as a quarter.
    """
    series = _live_series(facts, ladder)
    if not series:
        return {}
    cut = through - timedelta(days=round(365.25 * years))
    best: dict[date, Fact] = {}
    for f in derive_periods(series):
        if f.start is None or not (_QUARTER_MIN_DAYS <= f.days <= _QUARTER_MAX_DAYS):
            continue
        if f.end > through or f.end <= cut:
            continue
        prior = best.get(f.end)
        if prior is None or _quarter_rank(f) < _quarter_rank(prior):
            best[f.end] = f
    return {end: f.val / _MM for end, f in best.items()}


def _quarter_rank(f: Fact) -> tuple[int, int]:
    """Which of several windows ending on the same date is the better quarter.

    A filed period beats one recovered by subtraction, because the subtraction
    inherits the error of both operands. Among equals the window closest to
    thirteen weeks wins, so a 98-day stub left over from a 53-week year does not
    displace the real quarter beside it.
    """
    return (1 if f.derived_from else 0, abs(f.days - 91))


def _yoy_pairs(series: dict[date, float]) -> list[tuple[date, float, float]]:
    """Quarter ends paired with the same quarter a year earlier.

    Year on year rather than sequential, because software revenue has a fiscal
    seasonality that a sequential series would report as volatility. The pairing
    is by date within a tolerance, never by position: a filer with a gap in the
    series would otherwise have its Q3 paired against a Q2.
    """
    ends = sorted(series)
    pairs: list[tuple[date, float, float]] = []
    for e in ends:
        prior = _anniversary(ends, e, 1.0)
        if prior is not None and prior != e:
            pairs.append((e, series[e], series[prior]))
    return pairs


# --------------------------------------------------------------------------- #
# Market features
# --------------------------------------------------------------------------- #


def _close_at(series: PriceSeries, when: date) -> float | None:
    """Close on the last trading day at or before ``when``, if the series reaches it.

    A series that begins after ``when`` would otherwise return its own first
    close, and the return computed from it would be labelled twelve months while
    covering eight. The slack allows for a long weekend or a holiday week, not
    for a short history.
    """
    found: float | None = None
    found_date: date | None = None
    for d, c in zip(series.dates, series.closes):
        if d <= when:
            found, found_date = float(c), d
        else:
            break
    if found is None or found_date is None:
        return None
    if (when - found_date).days > _PRICE_LOOKBACK_SLACK_DAYS:
        return None
    return found


def _price_return(series: PriceSeries, months: int) -> float | None:
    """Price return over a trailing window ending at the series' last close.

    Price return, not total return: the price layer carries closes rather than a
    total-return index, so a dividend-paying name is understated by its yield.
    No name in the TMT universe this engine targets pays a material one, but the
    convention is stated rather than assumed.
    """
    target = series.last_date - timedelta(days=round(365.25 * months / 12.0))
    base = _close_at(series, target)
    if base is None or base <= 0:
        return None
    return series.last / base - 1.0


def _realised_volatility(series: PriceSeries) -> float | None:
    """Annualised standard deviation of weekly returns over the trailing year.

    Weekly rather than daily: a daily series of a mid-cap carries enough
    non-synchronous trading noise that the annualised figure overstates the
    volatility an investor experiences, and the same choice is already made for
    the beta regression, so the two features describe the same return series.
    """
    window = series.window(series.last_date - timedelta(days=365))
    _, returns = window.weekly_returns()
    if returns.size < _MIN_VOL_OBSERVATIONS:
        return None
    return float(np.std(returns, ddof=1)) * math.sqrt(52.0)


def _range_position(series: PriceSeries) -> float | None:
    """Where the last close sits in the 52-week range: 0 at the low, 1 at the high."""
    low, high = series.fifty_two_week_range()
    if high <= low:
        return None
    return (series.last - low) / (high - low)


# --------------------------------------------------------------------------- #
# One row
# --------------------------------------------------------------------------- #


def build_features(
    ticker: str,
    as_of: date,
    client,
    market: MarketData | None = None,
    assumptions: Assumptions | None = None,
) -> FeatureRow:
    """Build one company's feature row as it stood on one date.

    ``client`` must be pinned: ``EdgarClient(knowledge_date=as_of)``, or anything
    whose ``company_facts`` returns a ``CompanyFacts`` carrying a
    ``knowledge_date`` at or before ``as_of``. An unpinned client raises rather
    than producing a row, because a row built from an unpinned client is
    indistinguishable from a clean one until the model trained on it is wrong in
    production.

    ``market`` must be capped the same way: ``MarketData(today=as_of)``. Without
    one the seven market features are reported missing, which is a usable row for
    a fundamentals-only model and an honest one.

    The trailing figures anchor on the most recent period the filer had actually
    filed by ``as_of``, so the reporting lag falls out of the construction. The
    date they anchor on is recorded in ``statement_date``.

    Raises ``LookaheadError`` if anything on the finished row traces to a filing
    or a close dated after ``as_of``. It is raised rather than recorded, because
    unlike a parsing failure it invalidates the result instead of costing a row
    of it.
    """
    assumptions = assumptions or Assumptions()
    tk = ticker.upper()

    facts = client.company_facts(tk)
    knowledge = getattr(facts, "knowledge_date", None)
    if knowledge is None:
        raise LookaheadError(
            f"{tk}: the client feeding features dated {as_of} was not pinned to a "
            "knowledge date, so its fact set carries restatements, comparatives "
            "and split adjustments filed since. Build it as "
            f"EdgarClient(knowledge_date={as_of!r})."
        )
    if knowledge > as_of:
        raise LookaheadError(
            f"{tk}: the client feeding features dated {as_of} is pinned to "
            f"{knowledge}, which is {(knowledge - as_of).days} days later."
        )

    notes: list[str] = []
    if knowledge < as_of:
        notes.append(
            f"the client is pinned to {knowledge}, earlier than the row date "
            f"{as_of}; the row is conservative rather than leaking, but it does "
            "not see everything that was on file"
        )

    fin = build_financials(tk, facts=facts)

    # Every resolution this module makes on its own account is recorded here and
    # walked by ``assert_point_in_time`` beside the statements' own provenance.
    # The historical anchors are a third of the row and would otherwise be
    # unproven.
    row_prov: dict[str, Provenance] = {}

    computed: dict[str, float | None] = {}

    # -- history: the same concepts, resolved at earlier anchors ------------ #
    ends = _period_ends(facts)
    a1 = _anniversary(ends, fin.as_of, 1.0)
    a2 = _anniversary(ends, fin.as_of, 2.0)
    a3 = _anniversary(ends, fin.as_of, 3.0)

    def ttm_at(label: str, concept: str, ladder, anchor: date | None, *, kind="flow"):
        if anchor is None:
            return None
        value, p = facts.resolve_ttm(
            concept, ladder, anchor, kind=kind, required=False
        )
        row_prov[label] = p
        return None if value is None else value / _MM

    rev_1y = ttm_at("revenue t-1y", "revenue", tags.REVENUE, a1)
    rev_2y = ttm_at("revenue t-2y", "revenue", tags.REVENUE, a2)
    rev_3y = ttm_at("revenue t-3y", "revenue", tags.REVENUE, a3)
    gp_1y = ttm_at("gross profit t-1y", "gross profit", tags.GROSS_PROFIT, a1)
    ebit_1y = ttm_at("EBIT t-1y", "EBIT", tags.EBIT, a1)
    da_1y = ttm_at("D&A t-1y", "D&A", tags.DA, a1)
    ni_1y = ttm_at("net income t-1y", "net income", tags.NET_INCOME, a1)
    cfo_1y = ttm_at("cash from operations t-1y", "cash from operations", tags.CFO, a1)
    capex_1y = ttm_at("capital expenditure t-1y", "capital expenditure", tags.CAPEX, a1)

    # Operating expense lines the valuation path never resolves. Annual fallback
    # is off: a filer that reports research spend only once a year would put a
    # full-year figure over a trailing-twelve-month revenue, and the ratio would
    # be wrong by whatever the year grew.
    rnd, p_rnd = facts.resolve_ttm(
        "research and development expense", RND_EXPENSE, fin.as_of, required=False
    )
    row_prov["research and development expense"] = p_rnd
    rnd = None if rnd is None else rnd / _MM

    snm, p_snm = facts.resolve_ttm(
        "selling and marketing expense", SNM_EXPENSE, fin.as_of, required=False
    )
    row_prov["selling and marketing expense"] = p_snm
    snm = None if snm is None else snm / _MM

    assets, p_assets = facts.resolve_instant(
        "total assets", tags.TOTAL_ASSETS, fin.as_of, required=False
    )
    row_prov["total assets"] = p_assets
    assets = None if assets is None else assets / _MM

    equity, p_equity = facts.resolve_instant(
        "book equity", tags.EQUITY, fin.as_of, required=False
    )
    row_prov["book equity"] = p_equity
    equity = None if equity is None else equity / _MM

    # Deferred revenue a year back, summed current and non-current exactly as
    # ``build_financials`` sums the current one, so the growth rate compares like
    # with like. Both halves default to zero when no tag reports them, which is
    # the engine's own convention and is recorded in the provenance as a default
    # rather than as a reading. Where nothing at all is on file the sum is zero,
    # and growth from a zero base is reported as unavailable rather than as
    # infinite.
    dr_1y: float | None = None
    if a1 is not None:
        cur, p_cur = facts.resolve_instant(
            "deferred revenue, current t-1y",
            tags.DEFERRED_REVENUE_CURRENT,
            a1,
            default_when_absent=0.0,
        )
        nc, p_nc = facts.resolve_instant(
            "deferred revenue, non-current t-1y",
            tags.DEFERRED_REVENUE_NONCURRENT,
            a1,
            default_when_absent=0.0,
        )
        row_prov["deferred revenue, current t-1y"] = p_cur
        row_prov["deferred revenue, non-current t-1y"] = p_nc
        dr_1y = (cur or 0.0) / _MM + (nc or 0.0) / _MM

    # -- market ------------------------------------------------------------- #
    price: float | None = None
    series: PriceSeries | None = None
    market_cap: float | None = None
    enterprise_value: float | None = None
    net_debt: float | None = None

    if market is None:
        notes.append(
            "no market feed was supplied, so the seven market features and every "
            "ratio taken over market capitalisation or enterprise value are missing"
        )
    else:
        if market.today > as_of:
            raise LookaheadError(
                f"{tk}: the market feed for features dated {as_of} is built with "
                f"today={market.today}. Build MarketData with today set to the "
                "row date."
            )
        try:
            series = market.prices(tk)
            price = series.last
            # A series capped correctly but ending weeks early is not a leak and
            # will not trip the assertion, so it is reported instead. The
            # momentum windows are measured back from the last close, which means
            # a stale feed silently dates every market feature to whenever it
            # stopped.
            stale_days = (as_of - series.last_date).days
            if stale_days > _STALE_PRICE_DAYS:
                notes.append(
                    f"the price series ends {series.last_date}, {stale_days} days "
                    f"before the row date; the market features are measured back "
                    "from that close, not from the row date"
                )
        except TechvalError as exc:
            notes.append(f"price history unavailable: {exc}")

    if price is not None and price > 0:
        # Trailing diluted weighted-average shares unless a treasury-stock count
        # has been built onto the statements, which is the engine's own
        # convention. A weighted average lags the count outstanding at the
        # balance-sheet date, so for a company issuing heavily the capitalisation
        # here is a little low. The alternative is a cover-page share count that
        # is dated differently from every other figure on the row, and a feature
        # store gains more from one consistent date than from a closer number.
        market_cap = price * fin.shares_for_valuation
        try:
            bridge = build_ev_bridge(fin, price, assumptions)
            enterprise_value = bridge.enterprise_value
            net_debt = bridge.net_debt
        except TechvalError as exc:
            notes.append(f"enterprise value bridge unavailable: {exc}")

    # -- scale --------------------------------------------------------------- #
    computed["scale_log_revenue"] = _log_size(fin.revenue)
    computed["scale_log_market_cap"] = _log_size(market_cap)
    computed["scale_log_enterprise_value"] = _log_size(enterprise_value)
    computed["scale_log_total_assets"] = _log_size(assets)
    if enterprise_value is not None and enterprise_value <= 0:
        notes.append(
            "scale_log_enterprise_value: NM, the bridge gives a negative "
            f"enterprise value of {enterprise_value:,.0f}mm because net cash "
            "exceeds the equity value, and a log has no real value there"
        )

    # -- growth -------------------------------------------------------------- #
    g1 = _growth(fin.revenue, rev_1y)
    g2 = _cagr(fin.revenue, rev_2y, 2.0)
    computed["growth_revenue_1y"] = g1
    computed["growth_revenue_cagr_2y"] = g2
    computed["growth_revenue_cagr_3y"] = _cagr(fin.revenue, rev_3y, 3.0)
    # Acceleration against the two-year rate rather than against last year's
    # one-year rate, because the two-year rate needs one fewer historical anchor
    # and is available on filers with a shorter record.
    computed["growth_acceleration"] = _diff(g1, g2)
    computed["growth_gross_profit_1y"] = _growth(fin.gross_profit, gp_1y)
    computed["growth_deferred_revenue_1y"] = _growth(fin.deferred_revenue, dr_1y)

    # -- margin -------------------------------------------------------------- #
    fcf = _add(fin.cfo, None if fin.capex is None else -abs(fin.capex))
    fcf_1y = _add(cfo_1y, None if capex_1y is None else -abs(capex_1y))
    ebitda_1y = _add(ebit_1y, da_1y)

    m_gross = _ratio(fin.gross_profit, fin.revenue)
    m_ebitda = _ratio(fin.ebitda, fin.revenue)
    m_ebit = _ratio(fin.ebit, fin.revenue)
    m_net = _ratio(fin.net_income, fin.revenue)
    m_fcf = _ratio(fcf, fin.revenue)

    computed["margin_gross"] = m_gross
    computed["margin_ebitda"] = m_ebitda
    computed["margin_ebit"] = m_ebit
    computed["margin_net"] = m_net
    computed["margin_fcf"] = m_fcf
    computed["margin_gross_chg_1y"] = _diff(m_gross, _ratio(gp_1y, rev_1y))
    computed["margin_ebitda_chg_1y"] = _diff(m_ebitda, _ratio(ebitda_1y, rev_1y))
    computed["margin_ebit_chg_1y"] = _diff(m_ebit, _ratio(ebit_1y, rev_1y))
    computed["margin_net_chg_1y"] = _diff(m_net, _ratio(ni_1y, rev_1y))
    computed["margin_fcf_chg_1y"] = _diff(m_fcf, _ratio(fcf_1y, rev_1y))

    # -- efficiency ---------------------------------------------------------- #
    computed["efficiency_sbc_to_revenue"] = _ratio(fin.sbc, fin.revenue)
    computed["efficiency_rnd_to_revenue"] = _ratio(rnd, fin.revenue)
    computed["efficiency_snm_to_revenue"] = _ratio(snm, fin.revenue)
    # Capital expenditure is filed as a positive outflow under
    # ``PaymentsToAcquirePropertyPlantAndEquipment`` and as a negative under some
    # combined tags. The absolute value is taken so the intensity is a spend rate
    # either way rather than a sign convention.
    computed["efficiency_capex_to_revenue"] = _ratio(
        None if fin.capex is None else abs(fin.capex), fin.revenue
    )
    computed["efficiency_da_to_revenue"] = _ratio(fin.da, fin.revenue)
    computed["efficiency_opex_to_revenue"] = _ratio(
        _add(fin.gross_profit, -fin.ebit), fin.revenue
    )
    computed["efficiency_sbc_to_gross_profit"] = _ratio(fin.sbc, fin.gross_profit)

    # -- capital ------------------------------------------------------------- #
    # Net debt over EBITDA is suppressed rather than winsorized where EBITDA is
    # not positive. Winsorization is the answer to a denominator that is small:
    # a company with 1mm of EBITDA and 400mm of net debt belongs at the leveraged
    # end of the column, pulled to the 99th percentile. A negative EBITDA gives a
    # negative multiple that sorts to the conservative end, which is the opposite
    # of the truth, and no percentile bound repairs a sign.
    if fin.ebitda is not None and fin.ebitda <= 0 and net_debt is not None:
        computed["capital_net_debt_to_ebitda"] = None
        notes.append(
            "capital_net_debt_to_ebitda: NM, trailing EBITDA of "
            f"{fin.ebitda:,.1f}mm is not positive, so the multiple would read as "
            "conservative leverage when the opposite is true"
        )
    else:
        computed["capital_net_debt_to_ebitda"] = _ratio(net_debt, fin.ebitda)

    computed["capital_net_debt_to_market_cap"] = _ratio(net_debt, market_cap)
    computed["capital_net_debt_to_ev"] = _ratio(net_debt, enterprise_value)
    computed["capital_cash_to_market_cap"] = _ratio(
        fin.cash + fin.short_term_investments, market_cap
    )
    computed["capital_current_ratio"] = _ratio(
        fin.current_assets, fin.current_liabilities
    )
    total_debt = fin.total_debt_ex_leases + fin.finance_lease_liability
    computed["capital_debt_to_capital"] = _ratio(
        total_debt, None if market_cap is None else total_debt + market_cap
    )

    # -- returns -------------------------------------------------------------- #
    # ROIC on book invested capital: debt plus book equity less the cash and
    # securities that are not funding operations. A proxy and named one, because
    # the acquired goodwill inside book equity makes it a measure of what was
    # paid as much as of what is earned, and because the tax rate follows the
    # engine's convention of the configured marginal rate rather than the filed
    # effective one, which for a loss-making software filer is a reporting
    # artefact rather than the burden on an incremental dollar.
    tax_rate = assumptions.tax.marginal_tax_rate
    if assumptions.tax.use_effective_rate:
        if fin.effective_tax_rate is not None:
            tax_rate = fin.effective_tax_rate
        else:
            notes.append(
                "returns_roic_proxy: tax.use_effective_rate is set but the filed "
                "effective rate is not meaningful for this filer, so the "
                f"configured marginal rate of {tax_rate:.1%} is used"
            )
    invested = (
        None
        if equity is None
        else total_debt + equity - fin.cash - fin.short_term_investments
    )
    if invested is not None and invested <= 0:
        computed["returns_roic_proxy"] = None
        notes.append(
            "returns_roic_proxy: NM, cash and securities of "
            f"{fin.cash + fin.short_term_investments:,.0f}mm exceed debt plus book "
            "equity, so invested capital is negative and the return on it has no "
            "readable sign"
        )
    else:
        computed["returns_roic_proxy"] = _ratio(fin.ebit * (1.0 - tax_rate), invested)

    computed["returns_roe"] = _ratio(fin.net_income, equity)
    computed["returns_asset_turnover"] = _ratio(fin.revenue, assets)
    computed["returns_deferred_revenue_to_revenue"] = _ratio(
        fin.deferred_revenue, fin.revenue
    )
    if fin.ebitda is not None and fin.ebitda <= 0:
        computed["returns_cash_conversion"] = None
        notes.append(
            "returns_cash_conversion: NM, trailing EBITDA of "
            f"{fin.ebitda:,.1f}mm is not positive, so cash conversion against it "
            "would carry the sign of the denominator rather than of the cash"
        )
    else:
        computed["returns_cash_conversion"] = _ratio(fin.cfo, fin.ebitda)
    # The rule of forty: revenue growth plus free cash flow margin. The single
    # number the software cross-section is actually traded on, and a genuine
    # interaction rather than a restatement of its two parts, since the market
    # prices 40% growth at breakeven and 10% growth at a 30% margin alike.
    computed["returns_rule_of_40"] = _add(g1, m_fcf)

    # -- market --------------------------------------------------------------- #
    if series is None:
        for name in FEATURE_GROUPS["market"]:
            computed[name] = None
    else:
        r12 = _price_return(series, 12)
        computed["market_momentum_12m"] = r12
        computed["market_momentum_6m"] = _price_return(series, 6)
        computed["market_momentum_3m"] = _price_return(series, 3)

        index_symbol = assumptions.market.market_index
        index_series: PriceSeries | None = None
        try:
            index_series = market.prices(index_symbol)
        except TechvalError as exc:
            notes.append(f"{index_symbol} history unavailable: {exc}")

        if index_series is None:
            computed["market_relative_momentum_12m"] = None
            computed["market_beta"] = None
        else:
            computed["market_relative_momentum_12m"] = _diff(
                r12, _price_return(index_series, 12)
            )
            try:
                beta = estimate_beta(
                    series,
                    index_series,
                    adjustment=assumptions.market.beta_adjustment,
                    lookback_years=assumptions.market.beta_lookback_years,
                )
                computed["market_beta"] = beta.adjusted_beta
            except MissingDataError as exc:
                computed["market_beta"] = None
                notes.append(f"market_beta: {exc}")

        computed["market_realised_volatility_1y"] = _realised_volatility(series)
        computed["market_52w_range_position"] = _range_position(series)

    # -- quality --------------------------------------------------------------- #
    # Accruals as net income less cash from operations, over total assets. The
    # gap between the two is what accrual accounting added, and a persistently
    # large positive gap is the classic marker of earnings that are being
    # recognised faster than they are being collected.
    accrual = None if fin.cfo is None else fin.net_income - fin.cfo
    computed["quality_accruals"] = _ratio(accrual, assets)

    rev_q = _quarterly_series(facts, tags.REVENUE, fin.as_of, 4.0)
    growths = [
        g for _, new, old in _yoy_pairs(rev_q) if (g := _growth(new, old)) is not None
    ]
    computed["quality_revenue_volatility_3y"] = (
        float(np.std(growths, ddof=1)) if len(growths) >= 4 else None
    )

    ebit_q = _quarterly_series(facts, tags.EBIT, fin.as_of, 3.0)
    margins = [
        m
        for end, value in sorted(ebit_q.items())
        if (m := _ratio(value, rev_q.get(end))) is not None
    ]
    computed["quality_margin_volatility_3y"] = (
        float(np.std(margins, ddof=1)) if len(margins) >= 4 else None
    )

    computed["quality_sbc_to_cfo"] = _ratio(fin.sbc, fin.cfo)

    # The contract, enforced rather than trusted. A feature added to the
    # computation and not to FEATURE_GROUPS would silently change the width of
    # every matrix built from this store, and a name in FEATURE_GROUPS that
    # nothing computes would silently become a column of NaN.
    if set(computed) != set(FEATURE_NAMES):
        extra = sorted(set(computed) - set(FEATURE_NAMES))
        absent = sorted(set(FEATURE_NAMES) - set(computed))
        raise ConfigError(
            "the computed feature set does not match FEATURE_NAMES; "
            f"computed but not declared: {extra or 'none'}; "
            f"declared but not computed: {absent or 'none'}"
        )

    values = {name: computed[name] for name in FEATURE_NAMES}
    row = FeatureRow(
        ticker=tk,
        as_of=as_of,
        knowledge_date=knowledge,
        values=values,
        missing=[n for n in FEATURE_NAMES if values[n] is None],
        notes=notes,
        statement_date=fin.as_of,
        provenance=row_prov,
    )

    # Proven before the row leaves this function, so a leaked row never reaches a
    # panel, a matrix or a fit.
    assert_point_in_time(row, fin, prices=series)
    return row


# --------------------------------------------------------------------------- #
# Winsorization
# --------------------------------------------------------------------------- #


def winsorize(
    rows: Sequence[FeatureRow],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    lower_pct: float = WINSOR_LOWER_PCT,
    upper_pct: float = WINSOR_UPPER_PCT,
    min_observations: int = WINSOR_MIN_OBSERVATIONS,
) -> tuple[list[FeatureRow], list[WinsorReport]]:
    """Pull each date's cross-section inside its own 1st and 99th percentiles.

    Returns new rows and the full record of what was touched. Nothing is dropped
    and nothing is clipped silently: every value moved is counted, by feature and
    by date, so the share of the panel that was altered is a number a reader can
    check rather than a claim in a docstring.

    Percentiles are taken across the companies present on each date separately.
    Pooling every date first would set the bound from the level of whichever era
    dominates the sample, and a 2021 software cross-section and a 2026 one are
    not draws from the same distribution. Cross-sectional bounds also keep the
    trim neutral in time, which matters when the panel feeds a walk-forward split.

    Missing stays missing. A ``None`` is not a value outside the bounds, it is the
    absence of a value, and it is left alone.
    """
    if not 0.0 <= lower_pct < upper_pct <= 100.0:
        raise ConfigError(
            f"winsorization needs 0 <= lower_pct < upper_pct <= 100, got "
            f"{lower_pct} and {upper_pct}"
        )

    by_date: dict[date, list[int]] = {}
    for i, r in enumerate(rows):
        by_date.setdefault(r.as_of, []).append(i)

    new_values: list[dict[str, float | None]] = [dict(r.values) for r in rows]
    reports: list[WinsorReport] = []

    for when in sorted(by_date):
        idx = by_date[when]
        for name in feature_names:
            present = [i for i in idx if rows[i].values.get(name) is not None]
            observed = np.asarray(
                [float(rows[i].values[name]) for i in present], dtype=float
            )
            if observed.size < min_observations:
                reports.append(
                    WinsorReport(
                        as_of=when,
                        feature=name,
                        n_observed=int(observed.size),
                        applied=False,
                        note=(
                            f"{observed.size} observation(s) in the cross-section, "
                            f"below the {min_observations} at which a 1st and 99th "
                            "percentile describe the data rather than the sample size"
                        ),
                    )
                )
                continue

            lo = float(np.percentile(observed, lower_pct))
            hi = float(np.percentile(observed, upper_pct))
            low_hits = 0
            high_hits = 0
            for i in present:
                v = float(rows[i].values[name])
                if v < lo:
                    new_values[i][name] = lo
                    low_hits += 1
                elif v > hi:
                    new_values[i][name] = hi
                    high_hits += 1
            reports.append(
                WinsorReport(
                    as_of=when,
                    feature=name,
                    n_observed=int(observed.size),
                    lower=lo,
                    upper=hi,
                    n_pulled_low=low_hits,
                    n_pulled_high=high_hits,
                )
            )

    trimmed = [
        FeatureRow(
            ticker=r.ticker,
            as_of=r.as_of,
            knowledge_date=r.knowledge_date,
            values=new_values[i],
            missing=list(r.missing),
            notes=list(r.notes),
            statement_date=r.statement_date,
            error=r.error,
            provenance=r.provenance,
        )
        for i, r in enumerate(rows)
    ]
    return trimmed, reports


# --------------------------------------------------------------------------- #
# The panel
# --------------------------------------------------------------------------- #

ClientFactory = Callable[[date], object]
MarketFactory = Callable[[date], MarketData]


def build_panel(
    tickers: Sequence[str],
    dates: Sequence[date],
    client_factory: ClientFactory,
    market_factory: MarketFactory | None = None,
    assumptions: Assumptions | None = None,
    *,
    winsorize_cross_section: bool = True,
    lower_pct: float = WINSOR_LOWER_PCT,
    upper_pct: float = WINSOR_UPPER_PCT,
) -> FeaturePanel:
    """Build every ticker at every date, then trim each date's cross-section.

    ``client_factory`` and ``market_factory`` each take a date and return a feed
    pinned to it. They are factories rather than objects because a panel spans
    many dates and one feed can only be pinned to one of them; handing in a
    single client would mean every row after the first saw the wrong world.

    A company that cannot be built at a date is recorded as a failed row carrying
    the reason, never dropped and never filled in. Across a real universe the
    engine meets filers whose tagging it cannot resolve and filers that had not
    yet listed, and both are facts about the sample that a model's coverage
    statistics need to see. A ``LookaheadError`` is not caught: it says the panel
    is not evidence, which is not a per-row failure.
    """
    assumptions = assumptions or Assumptions()

    rows: list[FeatureRow] = []
    for when in dates:
        client = client_factory(when)
        market = None if market_factory is None else market_factory(when)
        for ticker in tickers:
            try:
                rows.append(
                    build_features(ticker, when, client, market, assumptions)
                )
            except TechvalError as exc:
                rows.append(_failed_row(ticker, when, f"{type(exc).__name__}: {exc}"))

    reports: list[WinsorReport] = []
    if winsorize_cross_section:
        rows, reports = winsorize(rows, lower_pct=lower_pct, upper_pct=upper_pct)

    panel = FeaturePanel(
        rows=rows,
        winsorization=reports,
        random_seed=assumptions.ml.random_seed,
    )
    failed = panel.failures()
    if failed:
        panel.notes.append(
            f"{len(failed)} of {len(rows)} observations could not be built and are "
            "carried as failures with the reason attached, not dropped"
        )
    return panel
