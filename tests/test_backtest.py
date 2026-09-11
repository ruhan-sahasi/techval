"""Tests for the point-in-time backtest harness.

Everything here runs against the four committed fact payloads and their price
CSVs, so no test touches the network. The knowledge dates are chosen inside the
window the fixtures cover: facts were pruned to periods ending on or after
2023-01-01, which means a knowledge date earlier than the first of those filings
presents to the engine exactly as a company that had not yet filed. That is the
condition the skipped observation exists for, and it is reached here honestly
rather than by deleting anything from a payload.

The most important test in the file is the one that seeds a lookahead violation
and checks it is caught. Every other assertion here is about arithmetic; that one
is about whether the harness is evidence.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from techval.backtest import (
    FAILED,
    SKIPPED,
    BacktestResult,
    ForwardPrices,
    LookaheadError,
    assert_no_lookahead,
    forward_return,
    market_factory_from,
    run_backtest,
    spearman,
    value_at,
)
from techval.config import Assumptions
from techval.edgar import CompanyFacts, HttpCache
from techval.errors import ConfigError
from techval.financials import build_financials
from techval.market import CsvSource

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = FIXTURES / "prices"

# The fixture price CSVs run 2023-09-01 to 2026-09-09. Dates are picked so that
# the 252-day horizons of the first three complete inside that window and the
# fourth deliberately does not.
D_2024 = date(2024, 9, 3)
D_2025_H1 = date(2025, 3, 3)
D_2025_H2 = date(2025, 9, 2)
D_2026 = date(2026, 3, 2)
BEFORE_FIRST_FILING = date(2023, 1, 15)

TICKERS = ["DDOG", "CRWD", "MDB", "ZS"]


def _payload(ticker: str) -> dict:
    return json.loads((FIXTURES / f"companyfacts_{ticker.upper()}.json").read_text())


class DatedFixtureClient:
    """A fixture client that honours a knowledge date, as EdgarClient does."""

    def __init__(self, knowledge_date: date | None) -> None:
        self.knowledge_date = knowledge_date

    def company_facts(self, ticker: str) -> CompanyFacts:
        return CompanyFacts(
            _payload(ticker), ticker, knowledge_date=self.knowledge_date
        )


def dated_client_factory(as_of: date) -> DatedFixtureClient:
    return DatedFixtureClient(as_of)


def blind_client_factory(as_of: date) -> DatedFixtureClient:
    """The mistake the whole module exists to catch: no knowledge date at all.

    A caller who builds the client this way gets today's restated financials into
    a valuation dated years ago, and every other check in the engine passes.
    """
    return DatedFixtureClient(None)


@pytest.fixture
def cache() -> HttpCache:
    return HttpCache(enabled=False)


@pytest.fixture
def source() -> CsvSource:
    return CsvSource(PRICES)


@pytest.fixture
def market_factory(source, cache):
    return market_factory_from(source, cache)


@pytest.fixture
def prices(source) -> ForwardPrices:
    return ForwardPrices(source, start=date(2023, 1, 1), end=date(2026, 9, 10))


@pytest.fixture
def bt_assumptions() -> Assumptions:
    """Defaults with the risk-free rate pinned, so nothing reaches the Treasury."""
    a = Assumptions()
    a.market.risk_free_rate = 0.0483
    a.comps.peers = ["CRWD", "MDB", "ZS"]
    return a


# --------------------------------------------------------------------------- #
# The premise: a knowledge date changes the company being valued
# --------------------------------------------------------------------------- #


def test_knowledge_date_changes_the_statements_being_valued():
    """As of 2026-03-01 Datadog was loss making. With hindsight it was not.

    This is the whole reason the harness exists. The twelve months to 2025-12-31
    carry a 44mm operating loss and were the only thing on file at the start of
    March. The twelve months to 2026-06-30 carry a 16mm profit and were not
    filed until August. A backtest that valued the second one on the first one's
    date would be scoring a company that did not exist.
    """
    payload = _payload("DDOG")

    then = build_financials(
        "DDOG", facts=CompanyFacts(payload, "DDOG", knowledge_date=date(2026, 3, 1))
    )
    now = build_financials("DDOG", facts=CompanyFacts(payload, "DDOG"))

    assert then.as_of == date(2025, 12, 31)
    assert then.ebit == pytest.approx(-44.4, abs=0.5)
    assert now.as_of == date(2026, 6, 30)
    assert now.ebit == pytest.approx(16.3, abs=0.5)
    assert then.revenue < now.revenue


# --------------------------------------------------------------------------- #
# Requirement 1: the lookahead assertion, and a seeded violation
# --------------------------------------------------------------------------- #


def test_assert_no_lookahead_passes_on_a_properly_dated_run():
    as_of = date(2026, 3, 1)
    fin = build_financials(
        "DDOG", facts=CompanyFacts(_payload("DDOG"), "DDOG", knowledge_date=as_of)
    )
    assert_no_lookahead(fin, as_of)

    filed = [date.fromisoformat(p.filed) for p in fin.provenance.values() if p.filed]
    assert filed, "the assertion is worthless if there are no filing dates to walk"
    assert max(filed) <= as_of


def test_assert_no_lookahead_catches_a_seeded_violation():
    """Seed the violation the way a real caller would produce it.

    Nothing is doctored. The client is simply built without a knowledge date,
    which is the one-line mistake that ruins a backtest, and the statements that
    come back carry filings from months after the valuation date.
    """
    as_of = date(2026, 3, 1)
    fin = build_financials("DDOG", facts=CompanyFacts(_payload("DDOG"), "DDOG"))

    with pytest.raises(LookaheadError) as exc:
        assert_no_lookahead(fin, as_of)

    message = str(exc.value)
    assert "2026-03-01" in message
    assert "knowledge_date" in message
    # The message has to name the offending line items, not just say that one
    # exists, or the user cannot go and find what leaked.
    assert "revenue" in message


def test_assert_no_lookahead_ignores_defaulted_line_items():
    """A concept no tag reports has no filing date and cannot have seen anything."""
    as_of = date(2026, 3, 1)
    fin = build_financials(
        "DDOG", facts=CompanyFacts(_payload("DDOG"), "DDOG", knowledge_date=as_of)
    )
    defaulted = [p for p in fin.provenance.values() if p.filed is None]
    assert defaulted, "expected at least one concept defaulted to zero"
    assert_no_lookahead(fin, as_of)


def test_lookahead_error_is_not_swallowed_by_the_per_observation_handler(
    bt_assumptions, market_factory, prices
):
    """A leak must stop the run, not become one more row in the output.

    ``LookaheadError`` deliberately does not derive from ``TechvalError``, so it
    travels past the handler that records parse failures. If it were caught there
    the run would finish, print a correlation, and be wrong.
    """
    with pytest.raises(LookaheadError):
        value_at(
            "DDOG", D_2025_H1, bt_assumptions, blind_client_factory, market_factory
        )

    with pytest.raises(LookaheadError):
        run_backtest(
            ["DDOG"],
            [D_2025_H1],
            bt_assumptions,
            blind_client_factory,
            market_factory,
            prices,
        )


def test_lookahead_is_caught_on_the_price_side_too(bt_assumptions, source, cache):
    """A market feed that was never capped is a leak the provenance walk cannot see.

    ``Financials`` carries filing dates. A price series does not appear in it at
    all, so a ``MarketData`` built with the wrong ``today`` would hand the
    valuation a close from after the valuation date with nothing in the
    provenance to show for it.
    """
    from techval.market import MarketData

    capped = market_factory_from(source, cache)

    def uncapped(_as_of: date) -> MarketData:
        return MarketData(source, cache, today=date(2026, 9, 10))

    # The correctly capped factory values the same name without complaint, so
    # the failure below is the cap and nothing else.
    assert value_at(
        "DDOG", D_2025_H1, bt_assumptions, dated_client_factory, capped
    ).valued

    with pytest.raises(LookaheadError) as exc:
        value_at("DDOG", D_2025_H1, bt_assumptions, dated_client_factory, uncapped)
    assert "not capped" in str(exc.value)


def test_every_observation_in_a_run_passed_the_assertion(
    bt_assumptions, market_factory, prices
):
    """The run itself is the proof, so re-derive it here rather than trusting it.

    Each valued observation is rebuilt from the same knowledge-dated facts and
    walked again. If ``run_backtest`` had let a single one through unchecked this
    fails.
    """
    result = run_backtest(
        TICKERS,
        [D_2024, D_2025_H1],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    checked = 0
    for obs in result.observations:
        if not obs.valued:
            continue
        fin = build_financials(
            obs.ticker,
            facts=CompanyFacts(
                _payload(obs.ticker), obs.ticker, knowledge_date=obs.as_of
            ),
        )
        assert_no_lookahead(fin, obs.as_of)
        checked += 1
    assert checked >= 4


# --------------------------------------------------------------------------- #
# Requirement 2: survivorship, and skipped observations recorded not dropped
# --------------------------------------------------------------------------- #


def test_a_company_with_no_filings_is_skipped_and_kept(
    bt_assumptions, market_factory, prices
):
    """A name that had not filed is recorded with the reason, never dropped.

    Silently dropping it would quietly shrink the universe to whoever happened to
    be reporting, which is the same selection problem as survivorship wearing a
    different hat.
    """
    result = run_backtest(
        TICKERS,
        [BEFORE_FIRST_FILING, D_2025_H1],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )

    early = [o for o in result.observations if o.as_of == BEFORE_FIRST_FILING]
    assert len(early) == len(TICKERS)
    assert all(o.method == SKIPPED for o in early)
    assert all(o.upside is None and o.price is None for o in early)
    assert all("no revenue was on file" in (o.error or "") for o in early)

    assert result.n_skipped == len(TICKERS)
    # Still on the table, not just in the counter.
    frame = result.to_frame()
    assert (frame["Method"] == SKIPPED).sum() == len(TICKERS)
    assert len(frame) == len(result.observations)


def test_skipped_and_failed_are_counted_separately(
    bt_assumptions, market_factory, prices
):
    """Not filed and could not be parsed are different facts about the world.

    MongoDB is the live case: at several of these dates its diluted share count
    cannot be tiled into a trailing twelve months from the tags it reports, so
    the engine refuses rather than inventing a denominator. That is a parse
    failure against a company that had very much filed.
    """
    result = run_backtest(
        TICKERS,
        [BEFORE_FIRST_FILING, D_2024, D_2025_H1],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    assert result.n_skipped == 4
    assert result.n_failed >= 1

    failed = [o for o in result.observations if o.method == FAILED]
    assert {o.ticker for o in failed} == {"MDB"}
    assert all("diluted shares" in (o.error or "") for o in failed)
    assert result.n_valued + result.n_failed + result.n_skipped == len(
        result.observations
    )
    # Neither kind reaches the statistics. A failure scored as zero upside would
    # be the engine grading its own filler.
    assert result.n_paired <= result.n_valued


def test_a_parse_failure_does_not_abort_the_run(
    bt_assumptions, market_factory, prices
):
    """The names after the broken one still get valued."""
    result = run_backtest(
        ["MDB", "DDOG", "ZS"],
        [D_2025_H1],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    assert len(result.observations) == 3
    assert result.observations[0].method == FAILED
    assert result.observations[1].valued
    assert result.observations[2].valued


def test_the_survivorship_caveat_is_stated_in_every_result(
    bt_assumptions, market_factory, prices
):
    result = run_backtest(
        ["DDOG"],
        [D_2025_H1],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    flags = [c for c in result.checks if c.startswith("FLAG:")]
    assert any("survivorship" in c for c in flags)
    assert any("point-in-time ticker list" in c for c in flags)
    assert any("delisted or acquired" in n for n in result.notes)


# --------------------------------------------------------------------------- #
# Requirement 3: forward returns and the horizon rule
# --------------------------------------------------------------------------- #


def test_forward_return_enters_at_the_valuation_date_and_exits_past_the_horizon(
    prices,
):
    series = prices("DDOG")
    fr = forward_return("DDOG", D_2025_H1, series, 252)

    assert fr.entry_date is not None and fr.entry_date <= D_2025_H1
    assert fr.exit_date is not None
    assert fr.exit_date >= D_2025_H1 + timedelta(days=252)
    assert fr.total_return == pytest.approx(fr.exit_price / fr.entry_price - 1.0)
    assert fr.note is None

    # The exit is the FIRST trading day at or after the horizon, so there is no
    # close between the horizon date and the one chosen.
    horizon = D_2025_H1 + timedelta(days=252)
    between = [d for d in series.dates if horizon <= d < fr.exit_date]
    assert between == []


def test_a_horizon_past_the_end_of_the_series_records_none_and_is_excluded(
    bt_assumptions, market_factory, prices
):
    """Truncating the horizon would mix holding periods inside one coefficient.

    A March 2026 valuation on a 252-day horizon exits in November 2026, and the
    price series stops in September. The observation keeps its row and its
    upside; only the return is None, and the statistics are computed without it.
    """
    series = prices("DDOG")
    fr = forward_return("DDOG", D_2026, series, 252)
    assert fr.entry_price is not None
    assert fr.total_return is None
    assert fr.exit_date is None
    assert "before the" in (fr.note or "")

    result = run_backtest(
        ["DDOG", "ZS"],
        [D_2025_H1, D_2026],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    valued_2026 = [o for o in result.observations if o.as_of == D_2026 and o.valued]
    assert valued_2026, "the 2026 valuations should still be produced"
    assert result.n_paired == sum(
        1 for o in result.observations if o.valued and o.as_of == D_2025_H1
    )
    assert any("no completed horizon" in c for c in result.checks)


def test_the_horizon_is_counted_in_calendar_days(prices):
    series = prices("CRWD")
    short = forward_return("CRWD", D_2024, series, 30)
    long = forward_return("CRWD", D_2024, series, 252)
    assert short.exit_date >= D_2024 + timedelta(days=30)
    assert long.exit_date >= D_2024 + timedelta(days=252)
    assert short.exit_date < long.exit_date
    assert short.entry_price == long.entry_price
    assert short.horizon_days == 30 and long.horizon_days == 252


def test_the_entry_price_is_the_price_the_valuation_used(
    bt_assumptions, market_factory, prices
):
    """The call and the return must start from the same mark.

    A backtest that values at one close and measures the return from another is
    measuring a trade nobody could have made.
    """
    obs = value_at(
        "ZS", D_2025_H1, bt_assumptions, dated_client_factory, market_factory
    )
    fr = forward_return("ZS", D_2025_H1, prices("ZS"), 252)
    assert obs.price == pytest.approx(fr.entry_price)


# --------------------------------------------------------------------------- #
# Requirement 4: Spearman, hand computed
# --------------------------------------------------------------------------- #


def test_spearman_against_a_hand_computed_example():
    """rho = 1 - 6 * sum(d^2) / (n * (n^2 - 1)) on untied ranks.

    x ranks 1..5 against y ranks 2, 1, 4, 3, 5 gives d = -1, 1, -1, 1, 0 and
    sum(d^2) = 4, so rho = 1 - 24 / 120 = 0.8.
    """
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 1.0, 4.0, 3.0, 5.0]
    assert spearman(x, y) == pytest.approx(0.8)


def test_spearman_is_monotone_invariant_where_pearson_is_not():
    """The reason the harness ranks first.

    A perfectly ordered but violently convex relationship is rank correlation 1.
    Pearson on the same points is well below 1, and one extreme point is what
    moved it. A DCF upside distribution has exactly that shape.
    """
    upside = [-0.30, -0.10, 0.05, 0.20, 0.60, 4.00]
    realised = [-0.25, -0.05, 0.02, 0.10, 0.35, 0.40]
    assert spearman(upside, realised) == pytest.approx(1.0)
    pearson = float(np.corrcoef(np.array(upside), np.array(realised))[0, 1])
    assert pearson < 0.85


def test_spearman_averages_tied_ranks():
    """Ties share a rank, so the answer does not depend on input ordering."""
    x = [1.0, 2.0, 2.0, 3.0]
    y = [10.0, 20.0, 20.0, 30.0]
    assert spearman(x, y) == pytest.approx(1.0)

    shuffled_x = [2.0, 1.0, 3.0, 2.0]
    shuffled_y = [20.0, 10.0, 30.0, 20.0]
    assert spearman(shuffled_x, shuffled_y) == pytest.approx(1.0)


def test_spearman_perfectly_inverted_is_minus_one():
    assert spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_declines_to_answer_when_it_cannot():
    """Two points are plus or minus one whatever they are, and a constant has no rank."""
    assert spearman([1.0, 2.0], [5.0, 9.0]) is None
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    with pytest.raises(ValueError):
        spearman([1.0, 2.0, 3.0], [1.0, 2.0])


def test_hit_rate_and_quintiles_on_a_constructed_sample(
    bt_assumptions, market_factory, prices
):
    result = run_backtest(
        TICKERS,
        [D_2024, D_2025_H1, D_2025_H2],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    pairs = _scored_pairs(result)
    upsides = np.array([u for u, _ in pairs])
    rets = np.array([r for _, r in pairs])

    assert result.spearman_ic == pytest.approx(spearman(upsides, rets))

    hits = sum(1 for u, r in pairs if (u > 0) == (r > 0))
    assert result.hit_rate == pytest.approx(hits / len(pairs))

    q = result.quantile_returns
    assert q.index.name == "Quintile"
    assert int(q["n"].sum()) == len(pairs)
    assert list(q.index) == sorted(q.index)
    # Buckets are cut on the ranked upside, so they never interleave.
    assert list(q["Upside to"]) == sorted(q["Upside to"])
    assert q["Upside from"].iloc[0] == pytest.approx(upsides.min())
    assert q["Upside to"].iloc[-1] == pytest.approx(upsides.max())


# --------------------------------------------------------------------------- #
# Requirement 5 and 6: power and overlapping windows
# --------------------------------------------------------------------------- #


def test_a_thin_sample_is_flagged_and_n_travels_with_the_coefficient(
    bt_assumptions, market_factory, prices
):
    result = run_backtest(
        TICKERS,
        [D_2024, D_2025_H1],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    assert result.n_paired < 30
    assert any(c.startswith("FLAG:") and "below the 30 mark" in c for c in result.checks)

    # No coefficient is ever stated without its n on the same line.
    for line in result.checks:
        if "Spearman rank IC" in line:
            assert f"n = {result.n_paired}" in line
    labels = [label for label, _ in result.rows()]
    assert labels.index("Scored (n)") < labels.index("Spearman rank IC")


def test_the_threshold_for_a_thin_sample_is_the_callers(
    bt_assumptions, market_factory, prices
):
    result = run_backtest(
        TICKERS,
        [D_2024, D_2025_H1],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
        min_observations=2,
    )
    assert not any("below the 2 mark" in c for c in result.checks)


def test_overlapping_windows_are_counted_and_flagged(
    bt_assumptions, market_factory, prices
):
    """Quarterly valuations on a 252-day horizon are not independent draws.

    Two calls on the same name three months apart share most of the same price
    path. The greedy count of sequential windows says how many genuinely
    non-overlapping holding periods the sample contains, and it is well below the
    raw observation count here.
    """
    result = run_backtest(
        ["DDOG", "ZS"],
        [D_2024, D_2025_H1, D_2025_H2],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    assert result.n_paired == 6
    # Per name: 2024-09-03 is taken, 2025-03-03 falls inside its horizon and is
    # skipped, 2025-09-02 is more than 252 days later and is taken. Two each.
    assert result.n_independent == 4
    assert any(
        c.startswith("FLAG:") and "non-overlapping" in c for c in result.checks
    )


def test_dates_spaced_beyond_the_horizon_are_all_independent(
    bt_assumptions, market_factory, prices
):
    result = run_backtest(
        ["DDOG"],
        [D_2024, D_2025_H2],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    assert result.n_paired == 2
    assert result.n_independent == 2
    assert not any("non-overlapping" in c for c in result.checks)


# --------------------------------------------------------------------------- #
# The risk-free rate, which is the one lookahead the provenance walk cannot see
# --------------------------------------------------------------------------- #


def test_an_undated_risk_free_rate_is_refused(market_factory, prices):
    """Left unset, the discount rate would come from the wrong end of the year.

    The Treasury reader takes the latest quote in the calendar year rather than
    the latest quote at or before the valuation date, so an unpinned rate is a
    lookahead nobody would see in the output.
    """
    a = Assumptions()
    a.market.risk_free_rate = None

    with pytest.raises(ConfigError) as exc:
        run_backtest(
            ["DDOG"], [D_2025_H1], a, dated_client_factory, market_factory, prices
        )
    assert "risk_free_by_date" in str(exc.value)

    with pytest.raises(ConfigError):
        value_at("DDOG", D_2025_H1, a, dated_client_factory, market_factory)


def test_a_per_date_risk_free_rate_reaches_the_discount_rate(
    bt_assumptions, market_factory, prices
):
    rates = {D_2024: 0.0380, D_2025_H1: 0.0700}
    result = run_backtest(
        ["DDOG"],
        [D_2024, D_2025_H1],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
        risk_free_by_date=rates,
    )
    by_date = {o.as_of: o for o in result.observations}
    assert by_date[D_2025_H1].wacc > by_date[D_2024].wacc
    assert any("prevailed when it was made" in n for n in result.notes)

    # The caller's own object is untouched, so a second run is not contaminated.
    assert bt_assumptions.market.risk_free_rate == pytest.approx(0.0483)
    assert bt_assumptions.as_of is None


def test_the_per_date_mapping_is_enough_on_its_own(market_factory, prices):
    """A full mapping replaces the pinned rate rather than supplementing it."""
    a = Assumptions()
    a.market.risk_free_rate = None
    rates = {D_2024: 0.0380, D_2025_H1: 0.0700}

    result = run_backtest(
        ["DDOG"],
        [D_2024, D_2025_H1],
        a,
        dated_client_factory,
        market_factory,
        prices,
        risk_free_by_date=rates,
    )
    by_date = {o.as_of: o for o in result.observations}
    assert all(o.valued for o in result.observations)
    assert by_date[D_2025_H1].wacc > by_date[D_2024].wacc


def test_a_missing_rate_is_an_error_not_a_fallback(
    bt_assumptions, market_factory, prices
):
    with pytest.raises(ConfigError) as exc:
        run_backtest(
            ["DDOG"],
            [D_2024, D_2025_H1],
            bt_assumptions,
            dated_client_factory,
            market_factory,
            prices,
            risk_free_by_date={D_2024: 0.0380},
        )
    assert "2025-03-03" in str(exc.value)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_two_identical_runs_agree_exactly(bt_assumptions, source, cache):
    """Same inputs, same output, down to the last basis point.

    Fresh feeds on each run rather than shared ones, so the answer cannot be
    coming from a memo that survived between them.
    """
    dates = [D_2024, D_2025_H1, D_2025_H2]

    def one() -> BacktestResult:
        return run_backtest(
            TICKERS,
            dates,
            bt_assumptions,
            dated_client_factory,
            market_factory_from(source, cache),
            ForwardPrices(source, start=date(2023, 1, 1), end=date(2026, 9, 10)),
        )

    a, b = one(), one()

    assert a.observations == b.observations
    assert a.returns == b.returns
    assert a.spearman_ic == b.spearman_ic
    assert a.hit_rate == b.hit_rate
    assert (a.n_valued, a.n_failed, a.n_skipped, a.n_paired, a.n_independent) == (
        b.n_valued,
        b.n_failed,
        b.n_skipped,
        b.n_paired,
        b.n_independent,
    )
    assert a.checks == b.checks
    assert a.notes == b.notes
    assert a.quantile_returns.equals(b.quantile_returns)
    assert a.to_frame().equals(b.to_frame())


def test_the_order_of_the_ticker_list_does_not_move_the_statistics(
    bt_assumptions, market_factory, prices
):
    """Ranking with tied midranks and a stable sort makes the answer order free."""
    dates = [D_2024, D_2025_H1]
    forward = run_backtest(
        TICKERS, dates, bt_assumptions, dated_client_factory, market_factory, prices
    )
    backward = run_backtest(
        list(reversed(TICKERS)),
        dates,
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    assert forward.spearman_ic == pytest.approx(backward.spearman_ic)
    assert forward.hit_rate == pytest.approx(backward.hit_rate)
    assert forward.n_paired == backward.n_paired
    assert forward.quantile_returns["Mean return"].tolist() == pytest.approx(
        backward.quantile_returns["Mean return"].tolist()
    )


# --------------------------------------------------------------------------- #
# Shape of the result
# --------------------------------------------------------------------------- #


def test_the_frame_carries_one_row_per_attempt_with_its_return(
    bt_assumptions, market_factory, prices
):
    result = run_backtest(
        TICKERS,
        [BEFORE_FIRST_FILING, D_2025_H1],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    frame = result.to_frame()
    assert len(frame) == len(TICKERS) * 2
    for column in (
        "Ticker",
        "As of",
        "Method",
        "Price",
        "Value per share",
        "Upside",
        "WACC",
        "Revenue",
        "EBIT margin",
        "Exit date",
        "Exit price",
        "Forward return",
        "Note",
    ):
        assert column in frame.columns

    valued = frame[~frame["Method"].isin([FAILED, SKIPPED])]
    assert not valued.empty
    assert set(valued["Method"]) <= {"gordon", "value_driver"}
    assert valued["Upside"].notna().all()
    # Upside is value over price less one, on every row that has both.
    for _, row in valued.iterrows():
        assert row["Upside"] == pytest.approx(
            row["Value per share"] / row["Price"] - 1.0
        )


def test_the_observation_records_which_terminal_method_made_the_call(
    bt_assumptions, market_factory, prices
):
    """A Gordon call and a value-driver call are two different predictions.

    Scoring them together without recording which was which would report a
    correlation for a model that was never run as one model.
    """
    gordon = value_at(
        "DDOG", D_2025_H1, bt_assumptions, dated_client_factory, market_factory
    )
    assert gordon.method == "gordon"

    driven = bt_assumptions.model_copy(deep=True)
    driven.dcf.terminal.method = "value_driver"
    driven.dcf.terminal.terminal_roic = 0.14
    other = value_at("DDOG", D_2025_H1, driven, dated_client_factory, market_factory)

    assert other.method == "value_driver"
    assert other.price == pytest.approx(gordon.price)
    assert other.per_share_value != pytest.approx(gordon.per_share_value)


def test_an_empty_run_reports_nothing_rather_than_a_coefficient(
    bt_assumptions, market_factory, prices
):
    """No observations means no statistic, and the checks say why."""
    result = run_backtest(
        TICKERS,
        [BEFORE_FIRST_FILING],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
    )
    assert result.n_paired == 0
    assert result.spearman_ic is None
    assert result.hit_rate is None
    assert result.quantile_returns.empty
    assert any("not enough to define one" in c for c in result.checks)
    assert not any("Spearman rank IC" in label for label, _ in result.rows())


def test_the_notes_state_the_return_convention(
    bt_assumptions, market_factory, prices
):
    result = run_backtest(
        ["DDOG"],
        [D_2025_H1],
        bt_assumptions,
        dated_client_factory,
        market_factory,
        prices,
        horizon_days=180,
    )
    assert any(
        "price returns over 180 calendar days" in n and "Dividends are not included" in n
        for n in result.notes
    )
    assert result.horizon_days == 180


def _scored_pairs(result: BacktestResult) -> list[tuple[float, float]]:
    by_key = {(r.ticker, r.as_of): r for r in result.returns}
    out = []
    for obs in result.observations:
        if obs.upside is None:
            continue
        fr = by_key.get((obs.ticker, obs.as_of))
        if fr is None or fr.total_return is None:
            continue
        out.append((obs.upside, fr.total_return))
    return out
