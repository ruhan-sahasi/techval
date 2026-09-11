"""Point-in-time backtesting: did the model's upside predict the next year's return?

A valuation model is a claim about the future. The only way to find out whether
the claim has ever been worth anything is to re-run it on dates in the past,
using only what was knowable then, and score the answer against what the stock
actually did afterwards. Everything in this module exists to make that honest.

**Lookahead is the whole game.** A backtest that reads today's restated
financials into a 2025 valuation will look brilliant and mean nothing. The engine
already has the primitive that prevents it: ``CompanyFacts`` and ``EdgarClient``
take a ``knowledge_date`` and discard every fact filed after it, and
``MarketData`` takes a ``today`` and stops the price series there. This module
constructs every run through those, then proves it afterwards.
``assert_no_lookahead`` walks the provenance of the ``Financials`` that came out
and checks that no line item was sourced from a filing dated after the valuation
date. It is called on every single observation. The proof matters more than the
precaution: a run that cannot show the filing dates behind its inputs is not
evidence, it is an assertion.

Datadog makes the point concretely. Valued as of 2026-03-01 the latest statements
on file end 2025-12-31 and show a 44mm operating loss. Run with full knowledge the
same company shows the twelve months to 2026-06-30 and a 16mm operating profit.
Those are different companies to a model, and the second one did not exist on
2026-03-01.

**Survivorship bias is present and is not fixed here.** The ticker list is
supplied by the caller, so it is a list of companies that exist today. Every name
that was delisted, acquired at a discount, or wound up between the valuation date
and now is missing from it, and those are disproportionately the losers. Results
from this harness are therefore biased upward, and the bias is not small in a
sector where a fifth of the 2021 listings no longer trade independently. The
honest fix is a point-in-time universe: the SEC ``company_tickers.json`` file as
it stood at each valuation date, or a delisting-complete vendor file. This
harness does not have one, and no statistic it produces should be read as though
it did. Where a company has no filings on record by the valuation date the
observation is recorded as skipped with the reason attached, never dropped, so
the count of what was not valued stays visible.

**The test statistic is Spearman, not Pearson.** The relationship between a DCF
upside and a subsequent return is monotonic at best and nothing like linear. A
handful of names at 300% upside, which is usually where the model has broken
rather than where the opportunity is, would drive a Pearson coefficient on their
own. Ranking first throws away the magnitudes and keeps the ordering, which is
the only part of the prediction anyone trades on. The rank correlation is
computed here with numpy rather than by taking a scipy dependency for one
function.

**Power.** With a handful of names over a few dates the sample is far too small
for any of this to mean anything, and a rank correlation on twelve observations
will happily print 0.4. Every statistic is reported with its n beside it, and the
checks say plainly when the sample is too thin to interpret. Valuing the same
name quarterly on a one-year horizon also produces overlapping holding periods,
whose observations are not independent, so the effective sample is smaller than
the raw count and any apparent significance is inflated. The checks report the
count of non-overlapping windows alongside the raw n.

Returns are price returns. The price layer carries closes, not a total-return
index, so for a name that pays a dividend the realised figure here understates
the holding-period return by its yield. No name in the technology universe this
engine targets pays a material one, but the convention is stated rather than
assumed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from . import tags
from .config import Assumptions
from .dcf import run_dcf
from .edgar import CompanyFacts, EdgarClient, HttpCache
from .errors import ConfigError, MissingDataError, TechvalError
from .ev_bridge import build_ev_bridge
from .financials import Financials, build_financials
from .market import MarketData, PriceSeries, PriceSource
from .wacc import compute_wacc

# Observation.method carries the terminal method used when a valuation
# succeeded, so the two failure states need labels that cannot collide with one.
FAILED = "failed"
SKIPPED = "skipped"


class LookaheadError(AssertionError):
    """A valuation dated D used a figure filed after D.

    Deliberately not a ``TechvalError``. Every other failure in a backtest is
    recorded against its observation and the run continues, because across fifty
    filers the engine will always meet one it cannot parse. Lookahead is not that
    kind of failure. It invalidates the result rather than costing one row of it,
    so it is raised as an assertion failure and travels straight past the
    per-observation handler to stop the run.
    """


class FactsClient(Protocol):
    """The part of ``EdgarClient`` a backtest needs."""

    def company_facts(self, ticker: str) -> CompanyFacts: ...


ClientFactory = Callable[[date], FactsClient]
MarketFactory = Callable[[date], MarketData]
PriceHistory = Callable[[str], PriceSeries]


# --------------------------------------------------------------------------- #
# The lookahead proof
# --------------------------------------------------------------------------- #


def assert_no_lookahead(fin: Financials, as_of: date) -> None:
    """Prove that nothing in these statements was filed after ``as_of``.

    Every line item in ``Financials`` carries a ``Provenance`` recording the tag,
    the periods and the filing date behind it. Walking that record is the only
    check that tests what actually reached the valuation, rather than what the
    caller intended to let through. A client built without a knowledge date, a
    cached payload reused from a later run, a fixture loaded by the wrong helper:
    all three pass every configuration check and all three are caught here.

    The test is the filing date, not the period end. A fiscal year ending
    2025-12-31 is not knowable on 2026-01-02; it becomes knowable when the 10-K
    is filed in February, and that is the date this compares.

    Line items with no filing date are those defaulted to zero because no tag in
    the ladder reports them. They carry no information from any filing, so there
    is nothing for them to have seen early.
    """
    late: list[str] = []
    for prov in fin.provenance.values():
        if not prov.filed:
            continue
        filed = date.fromisoformat(prov.filed)
        if filed > as_of:
            late.append(
                f"{prov.concept} (tag {prov.tag or '-'}) filed {filed}, "
                f"{(filed - as_of).days} days after the valuation date"
            )
    if late:
        raise LookaheadError(
            f"{fin.ticker} valued as of {as_of} used "
            f"{len(late)} figure(s) that were not on file until later:\n  "
            + "\n  ".join(sorted(late))
            + "\n  Build the client with knowledge_date set to the valuation date."
        )


def _has_filed_by(facts: CompanyFacts) -> bool:
    """True when this filer had reported revenue under the knowledge date in force.

    Revenue is the right thing to test because it is what the statement builder
    anchors on: the balance-sheet date is the end of the most recent period for
    which revenue was reported. A filer with no revenue fact visible at the
    valuation date has, as far as this engine can see, not yet filed, and that is
    a different fact about the world from a filer the engine failed to parse. The
    two are recorded separately so the count of each stays visible.
    """
    return any(facts.facts(tag) for tag in tags.REVENUE)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class Observation:
    """One valuation of one company on one date, successful or not.

    ``upside`` is the model's value per share over the market price, less one, so
    a positive figure says the model thought the stock was cheap. ``method`` is
    the terminal method that produced the headline value, or ``failed`` or
    ``skipped``. Everything numeric is None on those two, because a backtest that
    substituted a zero for a valuation it could not produce would be scoring its
    own filler.
    """

    ticker: str
    as_of: date
    price: float | None
    per_share_value: float | None
    upside: float | None
    wacc: float | None
    revenue: float | None
    ebit_margin: float | None
    method: str
    error: str | None = None

    @property
    def valued(self) -> bool:
        return self.upside is not None


def _unvalued(ticker: str, as_of: date, method: str, reason: str) -> Observation:
    """An observation that produced no value, with the reason it did not.

    Every numeric field stays None. A zero here would flow into the rank
    correlation as a genuine prediction of no upside, which is not what happened:
    nothing was predicted at all.
    """
    return Observation(
        ticker=ticker.upper(),
        as_of=as_of,
        price=None,
        per_share_value=None,
        upside=None,
        wacc=None,
        revenue=None,
        ebit_margin=None,
        method=method,
        error=reason,
    )


@dataclass
class ForwardReturn:
    """What the stock actually did over the holding period.

    Entry is the close on the last trading day at or before the valuation date,
    which is the same close the valuation itself priced against. Exit is the
    close on the first trading day at or after the valuation date plus
    ``horizon_days`` calendar days.

    Where the price series ends before that horizon, ``total_return`` is None and
    ``note`` says so. The alternative, cutting the horizon short to whatever the
    data allows, would put six-month and twelve-month holding periods into the
    same statistic, and a rank correlation across mixed holding periods measures
    the calendar as much as the model.

    ``entry_date`` records which trading day the entry close came from. A return
    whose start date is not on the record cannot be audited, and a price series
    that stops weeks before the valuation date produces an entry that looks
    perfectly ordinary without it.
    """

    ticker: str
    as_of: date
    horizon_days: int
    entry_price: float | None
    exit_price: float | None
    exit_date: date | None
    total_return: float | None
    entry_date: date | None = None
    note: str | None = None


@dataclass
class BacktestResult:
    """Observations, realised returns, and the statistics that score them.

    Lines in ``checks`` that begin with ``FLAG:`` failed a threshold; the rest are
    stated for the record. ``spearman_ic`` and ``hit_rate`` are None rather than
    a number whenever the paired sample is too thin to define them, because a
    coefficient printed on two observations is worse than no coefficient.
    """

    observations: list[Observation]
    returns: list[ForwardReturn]
    spearman_ic: float | None
    hit_rate: float | None
    quantile_returns: pd.DataFrame
    n_valued: int
    n_failed: int
    notes: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    n_skipped: int = 0
    n_paired: int = 0
    n_independent: int = 0
    horizon_days: int = 252

    def rows(self) -> list[tuple[str, float]]:
        """Headline statistics with the sample size that produced each one.

        The counts come first and the coefficient last, so that no reader meets
        the correlation before meeting the n behind it.
        """
        out: list[tuple[str, float]] = [
            ("Observations attempted", float(len(self.observations))),
            ("Valued", float(self.n_valued)),
            ("Failed", float(self.n_failed)),
            ("Skipped, no filings by the date", float(self.n_skipped)),
            ("Scored (n)", float(self.n_paired)),
            ("Non-overlapping windows", float(self.n_independent)),
        ]
        if self.hit_rate is not None:
            out.append(("Hit rate", self.hit_rate))
        if self.spearman_ic is not None:
            out.append(("Spearman rank IC", self.spearman_ic))
        return out

    def to_frame(self) -> pd.DataFrame:
        """One row per attempted observation, with its realised return beside it.

        Failures and skips keep their rows. Dropping them would make the table
        agree with the statistics while disagreeing with the run, and the count
        of what could not be valued is part of the result.
        """
        by_key = {(r.ticker, r.as_of): r for r in self.returns}
        records = []
        for obs in self.observations:
            fr = by_key.get((obs.ticker, obs.as_of))
            records.append(
                {
                    "Ticker": obs.ticker,
                    "As of": obs.as_of,
                    "Method": obs.method,
                    "Price": obs.price,
                    "Value per share": obs.per_share_value,
                    "Upside": obs.upside,
                    "WACC": obs.wacc,
                    "Revenue": obs.revenue,
                    "EBIT margin": obs.ebit_margin,
                    "Exit date": fr.exit_date if fr else None,
                    "Exit price": fr.exit_price if fr else None,
                    "Forward return": fr.total_return if fr else None,
                    # The valuation's own reason comes first. A skipped name has
                    # no price note worth reading, and a valued name with no
                    # return has no error, so the two never compete for the cell.
                    "Note": obs.error or (fr.note if fr else None),
                }
            )
        return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Factories: the valuation feed and the scoring feed, kept apart
# --------------------------------------------------------------------------- #


def edgar_client_factory(cache: HttpCache | None = None) -> ClientFactory:
    """Build an ``EdgarClient`` pinned to whatever date it is asked for."""
    return lambda as_of: EdgarClient(cache=cache, knowledge_date=as_of)


def market_factory_from(
    source: PriceSource, cache: HttpCache, history_years: float = 3.0
) -> MarketFactory:
    """Build a ``MarketData`` whose price series stops at the valuation date.

    ``history_years`` has to cover the beta regression window in
    ``market.beta_lookback_years`` with room to spare, since a two-year weekly
    regression run on exactly two years of closes loses observations to holidays
    at both ends.
    """
    return lambda as_of: MarketData(
        source, cache, today=as_of, history_years=history_years
    )


class ForwardPrices:
    """Realised prices for scoring, held apart from the valuation feed on purpose.

    The valuation path only ever sees a ``MarketData`` capped at the valuation
    date. Forward returns need closes from after that date, so they are read
    through this object instead, and this object is never handed to a valuation.
    The separation is structural rather than disciplinary: there is no route by
    which a valuation can reach a price it should not have seen, because the
    thing holding those prices is not in scope where the valuation runs.
    """

    def __init__(self, source: PriceSource, *, start: date, end: date) -> None:
        self.source = source
        self.start = start
        self.end = end
        self._series: dict[str, PriceSeries] = {}

    def __call__(self, symbol: str) -> PriceSeries:
        key = symbol.upper()
        if key not in self._series:
            self._series[key] = self.source.fetch(key, self.start, self.end)
        return self._series[key]


# --------------------------------------------------------------------------- #
# One valuation
# --------------------------------------------------------------------------- #


def value_at(
    ticker: str,
    as_of: date,
    assumptions: Assumptions,
    client_factory: ClientFactory,
    market_factory: MarketFactory,
) -> Observation:
    """Value one company as it was knowable on one date.

    The full chain runs: statements from facts filed by ``as_of``, the EV bridge
    at that date's close, a cost of capital from the price history up to that
    date, then the DCF. The headline per-share figure is whichever terminal
    method ``dcf.terminal.method`` selects, and the observation records which one
    it was. A Gordon value and a value-driver value are different predictions
    about the same company, so a score that pooled them without saying which was
    which would be reporting on two models at once.

    A ``TechvalError`` from anywhere in the chain is recorded against the
    observation and returned rather than raised. Across a real universe the
    engine will always meet filers whose tagging it cannot resolve, and a
    backtest that dies on the first of them reports nothing about the other
    forty-nine. A ``LookaheadError`` is not caught, for the reason given on that
    class.
    """
    if assumptions.market.risk_free_rate is None:
        raise ConfigError(
            "market.risk_free_rate must be set for a point-in-time valuation. "
            "Left unset, the cost of capital is anchored on the Treasury daily "
            "curve reader, which takes the latest quote in the calendar year "
            f"rather than the latest quote on or before {as_of}. That is a "
            "lookahead in the discount rate, and it would be invisible in the "
            "output. Pin the rate that prevailed at each valuation date, or pass "
            "risk_free_by_date to run_backtest."
        )

    client = client_factory(as_of)
    market = market_factory(as_of)

    try:
        facts = client.company_facts(ticker)
        if not _has_filed_by(facts):
            return _unvalued(
                ticker,
                as_of,
                SKIPPED,
                f"no revenue was on file by {as_of}, so there was nothing to "
                "value; the company had not filed, was not yet public, or reports "
                "under a tag outside the ladder",
            )

        fin = build_financials(ticker, facts=facts)
        # Called before anything downstream consumes the statements, so a run
        # that has seen the future stops here rather than producing a number
        # that would have to be withdrawn.
        assert_no_lookahead(fin, as_of)

        series = market.prices(ticker)
        if series.last_date > as_of:
            raise LookaheadError(
                f"{ticker.upper()} priced from a series running to "
                f"{series.last_date}, past the {as_of} valuation date. The "
                "market feed was not capped. Build MarketData with today set to "
                "the valuation date."
            )
        price = series.last
        if price <= 0:
            raise MissingDataError(
                "share price",
                ticker=ticker,
                hint=f"the close on {series.last_date} is {price}, which cannot "
                "be divided into a per-share value",
            )

        bridge = build_ev_bridge(fin, price, assumptions)
        wacc_result = compute_wacc(fin, bridge, market, assumptions)
        result = run_dcf(fin, bridge, wacc_result, assumptions)
    except TechvalError as exc:
        return _unvalued(ticker, as_of, FAILED, str(exc))

    return Observation(
        ticker=ticker.upper(),
        as_of=as_of,
        price=price,
        per_share_value=result.per_share,
        upside=result.per_share / price - 1.0,
        wacc=result.wacc,
        revenue=fin.revenue,
        ebit_margin=fin.ebit_margin,
        method=result.headline_method,
        error=None,
    )


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #


def forward_return(
    ticker: str, as_of: date, series: PriceSeries, horizon_days: int
) -> ForwardReturn:
    """Price return from the valuation date to the first close past the horizon.

    The horizon is counted in calendar days and then resolved to the first
    trading day at or after it, which is what a holding period actually is: you
    buy on the day you form the view and you sell a year later, on whatever day
    the exchange is next open.
    """
    dates = series.dates
    entry_idx = None
    for i in range(len(dates) - 1, -1, -1):
        if dates[i] <= as_of:
            entry_idx = i
            break
    if entry_idx is None:
        return ForwardReturn(
            ticker=ticker.upper(),
            as_of=as_of,
            horizon_days=horizon_days,
            entry_price=None,
            exit_price=None,
            exit_date=None,
            total_return=None,
            entry_date=None,
            note=(
                f"the price series for {ticker.upper()} begins {dates[0]}, after "
                f"the {as_of} valuation date"
            ),
        )

    entry_price = float(series.closes[entry_idx])
    entry_date = dates[entry_idx]
    target = as_of + timedelta(days=horizon_days)
    exit_idx = None
    for i in range(entry_idx, len(dates)):
        if dates[i] >= target:
            exit_idx = i
            break
    if exit_idx is None:
        return ForwardReturn(
            ticker=ticker.upper(),
            as_of=as_of,
            horizon_days=horizon_days,
            entry_price=entry_price,
            exit_price=None,
            exit_date=None,
            total_return=None,
            entry_date=entry_date,
            note=(
                f"the price series ends {dates[-1]}, before the {target} horizon; "
                "excluded from the statistics rather than scored over a shorter "
                "holding period"
            ),
        )

    exit_price = float(series.closes[exit_idx])
    return ForwardReturn(
        ticker=ticker.upper(),
        as_of=as_of,
        horizon_days=horizon_days,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_date=dates[exit_idx],
        total_return=exit_price / entry_price - 1.0,
        entry_date=entry_date,
    )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _midranks(x: np.ndarray) -> np.ndarray:
    """Ranks from 1, with tied values sharing the average of the ranks they span.

    Ties have to share a rank rather than be broken by position. Breaking them by
    the order the observations happened to arrive in would make the coefficient
    depend on the order of the ticker list, and a backtest whose answer moves
    when you sort the inputs differently is not reporting anything about the
    model.
    """
    order = np.argsort(x, kind="mergesort")
    ordered = x[order]
    ranks = np.empty(x.shape[0], dtype=float)
    i = 0
    n = x.shape[0]
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation: Pearson applied to the ranks.

    Returns None below three paired observations, where the statistic is not
    defined in any useful sense: on two points it is plus or minus one whatever
    the points are. Returns None too when either side is entirely tied, since a
    constant has no ranking to correlate with.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"spearman needs paired inputs, got {a.shape[0]} and {b.shape[0]}"
        )
    if a.shape[0] < 3:
        return None
    ra, rb = _midranks(a), _midranks(b)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _hit_rate(upsides: np.ndarray, rets: np.ndarray) -> tuple[float | None, int]:
    """Share of calls whose direction was right, and the count behind it.

    Only pairs where both the upside and the return are non-zero are counted. A
    prediction of exactly zero upside has no direction to be right or wrong
    about, and counting it either way would move the rate without any call having
    been made.
    """
    mask = (upsides != 0.0) & (rets != 0.0)
    n = int(mask.sum())
    if n == 0:
        return None, 0
    hits = int((np.sign(upsides[mask]) == np.sign(rets[mask])).sum())
    return hits / n, n


_QUANTILE_COLUMNS = [
    "n",
    "Upside from",
    "Upside to",
    "Mean upside",
    "Mean return",
    "Median return",
]


def _quantile_returns(
    upsides: np.ndarray, rets: np.ndarray, buckets: int = 5
) -> pd.DataFrame:
    """Realised returns by predicted-upside quintile, cheapest bucket first.

    Quintiles are cut on the ranked upside and split as evenly as the count
    allows, rather than on equal-width upside bands. Equal-width bands on a
    distribution with a long right tail put nearly everything in the bottom
    bucket and one name in the top, which is a picture of the distribution rather
    than of the model.

    A monotone rise in the return column from bucket one to bucket five is the
    result the model wants. Below about five observations per bucket there is
    nothing to read: a quintile of one name is a name, not a quintile.
    """
    if upsides.shape[0] == 0:
        return pd.DataFrame(
            columns=_QUANTILE_COLUMNS, index=pd.Index([], name="Quintile")
        )

    # Stable sort, so two identical upsides always land in the same buckets
    # whatever order the run happened to produce them in.
    order = np.argsort(upsides, kind="mergesort")
    groups = np.array_split(order, buckets)
    records = []
    labels = []
    for k, idx in enumerate(groups, start=1):
        if idx.size == 0:
            continue
        u, r = upsides[idx], rets[idx]
        labels.append(k)
        records.append(
            {
                "n": int(idx.size),
                "Upside from": float(u.min()),
                "Upside to": float(u.max()),
                "Mean upside": float(u.mean()),
                "Mean return": float(r.mean()),
                "Median return": float(np.median(r)),
            }
        )
    return pd.DataFrame(records, index=pd.Index(labels, name="Quintile"))


def _non_overlapping_windows(
    pairs: Sequence[tuple[str, date]], horizon_days: int
) -> int:
    """Count of holding periods that could be held one after another, per name.

    Two valuations of the same company three months apart on a one-year horizon
    share nine months of the same price path, so their returns are close to the
    same observation counted twice. Greedily taking the earliest date and then
    skipping everything inside its horizon gives the number of genuinely
    sequential windows the sample contains.

    This is an upper bound on independence, not a measure of it. Windows for
    different companies over the same calendar period still load on the same
    market, so a cross-section of ten names in one month is nothing like ten
    independent draws either.
    """
    by_ticker: dict[str, list[date]] = {}
    for ticker, when in pairs:
        by_ticker.setdefault(ticker, []).append(when)
    total = 0
    for ticker in sorted(by_ticker):
        last: date | None = None
        for when in sorted(by_ticker[ticker]):
            if last is None or (when - last).days >= horizon_days:
                total += 1
                last = when
    return total


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def run_backtest(
    tickers: Sequence[str],
    dates: Sequence[date],
    assumptions: Assumptions,
    client_factory: ClientFactory,
    market_factory: MarketFactory,
    price_history: PriceHistory,
    *,
    horizon_days: int = 252,
    min_observations: int = 30,
    risk_free_by_date: Mapping[date, float] | None = None,
    stale_price_days: int = 7,
) -> BacktestResult:
    """Value every ticker on every date, then score the calls against what happened.

    ``client_factory`` and ``market_factory`` each take a valuation date and
    return a feed pinned to it. ``price_history`` returns the whole price series
    for one symbol and is used only for scoring, never for valuing.

    ``risk_free_by_date`` supplies the ten-year Treasury yield that prevailed at
    each valuation date. Without it one rate from the assumptions file is applied
    across the whole sample, which is not lookahead but is wrong in its own way:
    the risk-free rate moved several hundred basis points across 2023 to 2026, so
    holding it flat suppresses the part of the discount rate that was genuinely
    different at each date and pushes the variation in measured upside onto the
    fundamentals alone. A missing date in the mapping is an error rather than a
    fallback to the file, because quietly reverting to a different rate for some
    dates and not others would make the sample incomparable with itself.

    The result is deterministic. Nothing here draws a random number, the
    observations come back in the order the dates and tickers were given, and the
    statistics are functions of those observations alone.
    """
    if assumptions.market.risk_free_rate is None and risk_free_by_date is None:
        raise ConfigError(
            "a backtest needs a risk-free rate it can date. Set "
            "market.risk_free_rate, or pass risk_free_by_date with the ten-year "
            "yield that prevailed at each valuation date."
        )

    observations: list[Observation] = []
    returns: list[ForwardReturn] = []
    notes: list[str] = []
    checks: list[str] = []

    for as_of in dates:
        dated = _assumptions_for(assumptions, as_of, risk_free_by_date)
        for ticker in tickers:
            obs = value_at(ticker, as_of, dated, client_factory, market_factory)
            observations.append(obs)
            try:
                series = price_history(ticker)
            except TechvalError as exc:
                returns.append(
                    ForwardReturn(
                        ticker=ticker.upper(),
                        as_of=as_of,
                        horizon_days=horizon_days,
                        entry_price=None,
                        exit_price=None,
                        exit_date=None,
                        total_return=None,
                        note=str(exc),
                    )
                )
                continue
            returns.append(forward_return(ticker, as_of, series, horizon_days))

    by_key = {(r.ticker, r.as_of): r for r in returns}
    scored = [
        (obs, by_key[(obs.ticker, obs.as_of)])
        for obs in observations
        if obs.upside is not None
        and (obs.ticker, obs.as_of) in by_key
        and by_key[(obs.ticker, obs.as_of)].total_return is not None
    ]
    upsides = np.array([o.upside for o, _ in scored], dtype=float)
    rets = np.array([r.total_return for _, r in scored], dtype=float)

    n_valued = sum(1 for o in observations if o.valued)
    n_failed = sum(1 for o in observations if o.method == FAILED)
    n_skipped = sum(1 for o in observations if o.method == SKIPPED)
    n_paired = len(scored)
    n_independent = _non_overlapping_windows(
        [(o.ticker, o.as_of) for o, _ in scored], horizon_days
    )

    ic = spearman(upsides, rets)
    hit, n_hit = _hit_rate(upsides, rets)

    notes.append(
        "Every valuation was built through a client pinned to its own date, so "
        "only facts filed by that date reached it, and the provenance of each "
        "one was walked afterwards to confirm it. Prices stop at the valuation "
        "date on the valuation side and are read from a separate series on the "
        "scoring side."
    )
    notes.append(
        "The ticker list was supplied by the caller and therefore contains only "
        "companies that exist now. Names delisted or acquired during the sample "
        "are absent, and they are disproportionately the losers, so every "
        "statistic below is biased upward by an amount this harness cannot "
        "measure."
    )
    notes.append(
        f"Forward returns are price returns over {horizon_days} calendar days, "
        "entry at the last close on or before the valuation date and exit at the "
        "first close on or after the horizon. Dividends are not included."
    )
    if risk_free_by_date is not None:
        notes.append(
            "The risk-free rate was set from the supplied per-date mapping, so "
            "each valuation discounts at the rate that prevailed when it was made."
        )
    else:
        notes.append(
            f"The risk-free rate was held at "
            f"{assumptions.market.risk_free_rate:.2%} across every date. The "
            "measured upsides therefore move with the fundamentals and the share "
            "price alone, and not with the rates that were actually quoted."
        )

    checks.append(
        f"{len(observations)} observations attempted: {n_valued} valued, "
        f"{n_failed} failed, {n_skipped} skipped with no filings on record."
    )
    checks.append(
        f"{n_paired} observations carry both an upside and a completed "
        f"{horizon_days}-day return and are the sample behind every statistic."
    )
    checks.append(
        "FLAG: the universe is survivorship biased. The honest fix is a "
        "point-in-time ticker list at each date; this run does not have one."
    )

    n_no_return = sum(
        1
        for o in observations
        if o.valued
        and by_key.get((o.ticker, o.as_of))
        and by_key[(o.ticker, o.as_of)].total_return is None
    )
    if n_no_return:
        checks.append(
            f"{n_no_return} valued observations have no completed horizon and "
            "were excluded rather than scored over a shorter holding period, "
            "which would have mixed holding periods inside one coefficient."
        )

    stale = [
        o
        for o, r in scored
        if r.entry_price is not None
        and o.price is not None
        and abs(o.price - r.entry_price) > 0.005 * r.entry_price
    ]
    if stale:
        checks.append(
            f"FLAG: {len(stale)} observations were valued at a price more than "
            "half a percent away from the scoring entry price. The two feeds "
            "disagree about what the stock closed at, and the return is being "
            "measured from a different point than the call was made."
        )

    old_prices = [
        r
        for r in returns
        if r.entry_date is not None
        and (r.as_of - r.entry_date).days > stale_price_days
    ]
    if old_prices:
        checks.append(
            f"FLAG: {len(old_prices)} observations were priced from a close more "
            f"than {stale_price_days} days before their valuation date. The price "
            "series does not really cover those dates, and both the call and the "
            "return start from a stale mark."
        )

    if n_paired < min_observations:
        checks.append(
            f"FLAG: n = {n_paired}, below the {min_observations} mark. A rank "
            "correlation on a sample this size is dominated by sampling noise "
            "and will happily print a number that looks like skill. Treat "
            "anything below as an illustration of the machinery, not as evidence "
            "about the model."
        )
    if n_paired and n_independent < n_paired:
        checks.append(
            f"FLAG: the {n_paired} scored observations contain only "
            f"{n_independent} non-overlapping {horizon_days}-day windows per "
            "name. Overlapping holding periods share most of the same price "
            "path, so they are not independent draws and any significance "
            "computed from the raw n is overstated. Windows across different "
            "names in the same months are correlated too, through the market."
        )
    if ic is None:
        checks.append(
            f"The Spearman coefficient is not reported: {n_paired} paired "
            "observations is not enough to define one, or every upside or every "
            "return was tied."
        )
    else:
        checks.append(f"Spearman rank IC {ic:+.3f} on n = {n_paired}.")
    if hit is None:
        checks.append("The hit rate is not reported: no directional call was made.")
    else:
        checks.append(f"Hit rate {hit:.0%} on n = {n_hit} directional calls.")

    return BacktestResult(
        observations=observations,
        returns=returns,
        spearman_ic=ic,
        hit_rate=hit,
        quantile_returns=_quantile_returns(upsides, rets),
        n_valued=n_valued,
        n_failed=n_failed,
        notes=notes,
        checks=checks,
        n_skipped=n_skipped,
        n_paired=n_paired,
        n_independent=n_independent,
        horizon_days=horizon_days,
    )


def _assumptions_for(
    assumptions: Assumptions,
    as_of: date,
    risk_free_by_date: Mapping[date, float] | None,
) -> Assumptions:
    """A copy of the assumptions carrying the rate that prevailed at ``as_of``.

    A deep copy per date, not a mutation of the caller's object, so that a run
    cannot leave the assumptions it was handed pointing at the last date it
    happened to reach.
    """
    dated = assumptions.model_copy(deep=True)
    dated.as_of = as_of.isoformat()
    if risk_free_by_date is not None:
        if as_of not in risk_free_by_date:
            raise ConfigError(
                f"risk_free_by_date has no rate for {as_of}. Supply one, or drop "
                "the mapping and pin a single rate in market.risk_free_rate. "
                "Falling back to the file for some dates and not others would "
                "make the observations incomparable with each other."
            )
        dated.market.risk_free_rate = float(risk_free_by_date[as_of])
    return dated
