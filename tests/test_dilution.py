"""Treasury stock share count.

Two filings are committed as pruned instance-fact fixtures and carry the real
cases: Datadog for a dual-class share count with deeply in-the-money options,
Zscaler for options that are out of the money and for an antidilutive-securities
table tagged under the options-outstanding concept. Everything else, bands and
the degradation paths, is built from hand-written facts so the arithmetic under
test is visible in the test.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from conftest import AS_OF, FIXTURES, PRICES
from techval.config import Assumptions
from techval.edgar import DimensionedFact, HttpCache
from techval.errors import MissingDataError, NotMeaningfulError
from techval.financials import build_financials
from techval.market import CsvSource, MarketData
from techval.dilution import (
    AwardBlock,
    build_share_count,
    shares_outstanding,
    treasury_stock_shares,
)

OPTION_COUNT = (
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingNumber"
)
OPTION_STRIKE = (
    "ShareBasedCompensationArrangementByShareBasedPaymentAward"
    "OptionsOutstandingWeightedAverageExercisePrice"
)
RSU_COUNT = (
    "ShareBasedCompensationArrangementByShareBasedPaymentAward"
    "EquityInstrumentsOtherThanOptionsNonvestedNumber"
)
BAND_AXIS = (
    "us-gaap:ShareBasedCompensationSharesAuthorizedUnderStockOptionPlans"
    "ExercisePriceRangeAxis"
)
CLASS_AXIS = "us-gaap:StatementClassOfStockAxis"
COVER = "EntityCommonStockSharesOutstanding"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def load_instance(ticker: str) -> tuple[list[DimensionedFact], str, date]:
    payload = json.loads((FIXTURES / f"instance_facts_{ticker}.json").read_text())
    facts = [
        DimensionedFact(
            tag=row["tag"],
            value=row["value"],
            unit=row["unit"],
            start=date.fromisoformat(row["start"]) if row["start"] else None,
            end=date.fromisoformat(row["end"]),
            dimensions=row["dimensions"],
        )
        for row in payload["facts"]
    ]
    return facts, payload["_accession"], date.fromisoformat(payload["_filed"])


class InstanceClient:
    """Serves committed instance facts, or whatever a test hands it instead."""

    def __init__(self, facts=None, accession="0000000000-00-000000", filed=AS_OF):
        self._override = (facts, accession, filed) if facts is not None else None

    def instance_facts(self, ticker, forms=("10-K", "10-Q")):
        if self._override is not None:
            return self._override
        return load_instance(ticker)


class BrokenClient:
    def instance_facts(self, ticker, forms=("10-K", "10-Q")):
        raise MissingDataError(
            "XBRL instance document",
            ticker=ticker,
            hint="filing 0001-00-000001 exposes no instance document to parse",
        )


def fact(tag, value, unit, end, dims=None) -> DimensionedFact:
    return DimensionedFact(
        tag=tag,
        value=value,
        unit=unit,
        start=None,
        end=date.fromisoformat(end),
        dimensions=dims or {},
    )


@pytest.fixture
def tsm(assumptions) -> Assumptions:
    assumptions.dilution.method = "treasury_stock"
    return assumptions


@pytest.fixture
def prices() -> MarketData:
    return MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=AS_OF)


@pytest.fixture
def ddog_count(ddog, tsm, prices):
    return build_share_count("DDOG", prices.spot("DDOG"), ddog, tsm, InstanceClient())


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #


def test_in_the_money_options_net_of_the_buyback(ddog_count):
    """Hand calculation, Datadog's Q2 2026 10-Q at the 2026-09-09 close.

        options outstanding      1,610,360 shares
        weighted-average strike  $7.83
        share price              $225.27

        exercise proceeds     1,610,360 x 7.83       = $12,609,118.80
        shares bought back    12,609,118.80 / 225.27 =     55,973.3 shares
        net new shares        1,610,360 - 55,973.3   =  1,554,386.7 shares

    The options are worth 1.61mm shares gross and 1.55mm net, because a $7.83
    strike against a $225.27 price buys back almost nothing. The whole point of
    the method is that those proceeds are real cash the company receives.
    """
    proceeds = 1_610_360 * 7.83
    bought_back = proceeds / 225.27
    expected = (1_610_360 - bought_back) / 1e6

    assert ddog_count.net_new_from_options == pytest.approx(expected, rel=1e-9)
    assert ddog_count.net_new_from_options == pytest.approx(1.554387, abs=5e-6)

    (block,) = ddog_count.option_awards
    assert (block.kind, block.count, block.strike) == ("option", 1.61036, 7.83)
    assert block.count > ddog_count.net_new_from_options  # the buyback is real


def test_walk_from_basic_to_fully_diluted_adds_up(ddog_count):
    """359.08mm outstanding plus 1.55mm net options plus 17.10mm units."""
    assert ddog_count.method == "treasury stock method"
    assert ddog_count.basic_outstanding == pytest.approx(359.075024)
    assert ddog_count.rsu_shares == pytest.approx(17.099356)
    assert ddog_count.fully_diluted == pytest.approx(
        359.075024 + ddog_count.net_new_from_options + 17.099356
    )

    labels = [label for label, _ in ddog_count.rows()]
    values = dict(ddog_count.rows())
    assert labels[0].startswith("Basic")
    assert values["Basic shares outstanding (mm)"] + values[
        "+ Net new shares from in-the-money options (mm)"
    ] + values["+ Unvested RSUs (mm)"] == pytest.approx(
        values["Fully diluted shares (mm)"]
    )


def test_diluted_waso_is_reported_alongside_with_the_gap(ddog_count, ddog):
    """The point-in-time count runs above the backward-looking average.

    Basic outstanding of 359.08mm is below the 366.93mm diluted weighted
    average, because the average already carries award overhang the walk adds
    back in full. Fully diluted lands at 377.73mm, 10.79mm and 2.9% above it.
    """
    assert ddog_count.diluted_waso == pytest.approx(366.934, abs=0.001)
    assert ddog_count.basic_outstanding < ddog_count.diluted_waso
    assert ddog_count.fully_diluted > ddog_count.diluted_waso

    assert ddog_count.gap_shares == pytest.approx(10.794, abs=0.001)
    assert ddog_count.gap_pct == pytest.approx(0.0294, abs=0.0001)
    assert ddog_count.diluted_waso == ddog.diluted_shares

    memo = dict(ddog_count.rows())
    assert memo["Memo: difference (mm)"] == pytest.approx(ddog_count.gap_shares)


def test_rsus_add_their_full_count_with_no_buyback(ddog_count):
    """An RSU has no strike, so there are no proceeds and no offset."""
    (block,) = ddog_count.rsu_awards
    assert block.kind == "rsu"
    assert block.strike is None
    assert block.count == pytest.approx(17.099356)
    assert ddog_count.rsu_shares == pytest.approx(block.count)

    fully, net_new, _ = treasury_stock_shares([block], price=225.27, basic=100.0)
    assert net_new == 0.0
    assert fully == pytest.approx(100.0 + 17.099356)


def test_out_of_the_money_options_add_nothing(zs, tsm, prices):
    """Zscaler: 150,000 options at $232.89 against a $166.10 close.

    Netted rather than dropped, the formula would return
    150,000 x (1 - 232.89/166.10) = -60,314 shares, and worthless options would
    shrink the share count. They contribute exactly zero instead.
    """
    count = build_share_count("ZS", prices.spot("ZS"), zs, tsm, InstanceClient())

    (block,) = count.option_awards
    assert block.strike == 232.89
    assert prices.spot("ZS") < block.strike
    assert count.net_new_from_options == 0.0
    assert count.fully_diluted == pytest.approx(
        count.basic_outstanding + count.rsu_shares
    )
    assert any("out of the money" in n for n in count.notes)


def test_out_of_the_money_block_never_subtracts():
    """The sign guard on its own, at the boundary and past it."""
    deep = AwardBlock("option", count=10.0, strike=200.0, source_tag="t")
    fully, net_new, _ = treasury_stock_shares([deep], price=50.0, basic=100.0)
    assert net_new == 0.0
    assert fully == 100.0

    at_the_money = AwardBlock("option", count=10.0, strike=50.0, source_tag="t")
    _, net_new, _ = treasury_stock_shares([at_the_money], price=50.0, basic=100.0)
    assert net_new == 0.0  # no intrinsic value, no net new shares


# --------------------------------------------------------------------------- #
# Share count
# --------------------------------------------------------------------------- #


def test_dual_class_shares_are_summed():
    """Datadog tags 334,904,614 Class A and 24,170,410 Class B, and no total.

    Reading either class alone is the failure this function exists to prevent:
    Class A alone loses 24.2mm shares, Class B alone loses 334.9mm.
    """
    facts, _accession, filed = load_instance("DDOG")
    total, notes = shares_outstanding(facts, filed)

    assert total == pytest.approx((334_904_614 + 24_170_410) / 1e6)
    assert total == pytest.approx(359.075024)
    assert any("2 share classes" in n for n in notes)
    assert any("CommonClassAMember" in n and "CommonClassBMember" in n for n in notes)


def test_single_class_issuer_is_not_double_counted():
    """Zscaler tags one undimensioned cover count, and that is the whole answer."""
    facts, _accession, filed = load_instance("ZS")
    total, notes = shares_outstanding(facts, filed)

    assert total == pytest.approx(163_054_698 / 1e6)
    assert any("single class" in n for n in notes)


def test_cover_page_count_beats_the_balance_sheet_count(ddog_count):
    """The cover count is struck on 2026-07-31, a month after the balance sheet.

    A point-in-time count should be as recent as the filing allows, and the
    balance-sheet count of 358,956,785 at 2026-06-30 is a month staler.
    """
    assert ddog_count.as_of == date(2026, 7, 31)
    assert ddog_count.basic_outstanding == pytest.approx(359.075024)
    assert ddog_count.accession == "0001628280-26-054458"


def test_shares_outstanding_raises_rather_than_guessing():
    with pytest.raises(MissingDataError) as exc:
        shares_outstanding([fact("Revenues", 100.0, "USD", "2026-06-30")], AS_OF)
    assert "shares outstanding" in str(exc.value)


# --------------------------------------------------------------------------- #
# Tag discipline
# --------------------------------------------------------------------------- #


def test_roll_forward_opening_balance_is_ignored(ddog_count):
    """One tag carries both ends of the option table, six months apart.

    Datadog tags 3,474,619 options at $7.26 for 2025-12-31 and 1,610,360 at
    $7.83 for 2026-06-30 under the identical concept. The opening balance is
    stale by a quarter and would more than double the option count.
    """
    (block,) = ddog_count.option_awards
    assert block.count == pytest.approx(1.61036)
    assert block.strike == 7.83
    assert block.count != pytest.approx(3.474619)


def test_dimensioned_lookalike_options_are_refused(zs, tsm, prices):
    """Zscaler tags its antidilutive table under the options-outstanding concept.

    At 2026-07-31 that concept carries 8,879,000 RSUs, 1,757,000 ESPP shares,
    997,000 performance shares and 324,000 committed performance awards, each
    under an award-type member, alongside the undimensioned 150,000 that are
    genuinely options. Summing the members would report 12.1mm options.
    """
    count = build_share_count("ZS", prices.spot("ZS"), zs, tsm, InstanceClient())
    (block,) = count.option_awards
    assert block.count == pytest.approx(0.15)
    assert sum(a.count for a in count.option_awards) < 1.0


def test_exercisable_subset_is_not_added_to_outstanding(ddog_count):
    """Options exercisable of 1,610,360 are a subset, not an addition.

    Datadog's exercisable count happens to equal its outstanding count, so a
    loose tag match would double the options to 3.22mm without looking odd.
    """
    assert sum(a.count for a in ddog_count.option_awards) == pytest.approx(1.61036)


# --------------------------------------------------------------------------- #
# Exercise-price bands
# --------------------------------------------------------------------------- #


def _banded_facts() -> list[DimensionedFact]:
    """Four million options struck at $10 and six million struck at $150.

    The weighted-average strike is (4 x 10 + 6 x 150) / 10 = $94.00, below a
    $100 price, so the average says the whole grant is in the money.
    """
    low = {BAND_AXIS: "ExercisePriceRangeOneMember"}
    high = {BAND_AXIS: "ExercisePriceRangeTwoMember"}
    return [
        fact(COVER, 200_000_000, "shares", "2026-06-30"),
        fact(OPTION_COUNT, 10_000_000, "shares", "2026-06-30"),
        fact(OPTION_STRIKE, 94.00, "USD/shares", "2026-06-30"),
        fact(OPTION_COUNT, 4_000_000, "shares", "2026-06-30", low),
        fact(OPTION_STRIKE, 10.00, "USD/shares", "2026-06-30", low),
        fact(OPTION_COUNT, 6_000_000, "shares", "2026-06-30", high),
        fact(OPTION_STRIKE, 150.00, "USD/shares", "2026-06-30", high),
    ]


def test_bands_are_valued_one_at_a_time(ddog, tsm):
    """Banded: 4mm at $10 adds 4 x (1 - 10/100) = 3.6mm, 6mm at $150 adds zero.

    The single $94.00 average would instead exercise all ten million and buy
    back 9.4mm shares for a net 0.6mm, six times less dilution, because it
    credits the company with $900mm of proceeds from options nobody would
    exercise. Banding is the larger number, and it is the right one.
    """
    count = build_share_count(
        "TEST", 100.0, ddog, tsm, InstanceClient(_banded_facts(), "0001-26-000001")
    )

    assert len(count.option_awards) == 2
    assert count.net_new_from_options == pytest.approx(3.6)

    single = 10.0 * (1 - 94.0 / 100.0)
    assert single == pytest.approx(0.6)
    assert count.net_new_from_options > single

    assert any("exercise-price band" in n for n in count.notes)
    assert any("out of the money" in n for n in count.notes)


def test_single_weighted_average_used_when_no_bands_are_tagged(ddog_count):
    """The notes have to say which of the two was used, and Datadog tags none."""
    assert any(
        "single weighted-average exercise price" in n and "no exercise-price bands" in n
        for n in ddog_count.notes
    )
    assert not any("taken by exercise-price band" in n for n in ddog_count.notes)
    assert len(ddog_count.option_awards) == 1


def test_partial_bands_fall_back_to_the_average_and_say_so(ddog, tsm):
    """A band with a count but no strike cannot be valued, so banding is off."""
    facts = [f for f in _banded_facts() if not (f.unit == "USD/shares" and f.value == 150.00)]
    count = build_share_count(
        "TEST", 100.0, ddog, tsm, InstanceClient(facts, "0001-26-000001")
    )

    assert count.net_new_from_options == pytest.approx(0.6)
    assert any("not all of them carry a strike" in n for n in count.notes)
    assert any("understated" in n for n in count.notes)


# --------------------------------------------------------------------------- #
# Assumptions
# --------------------------------------------------------------------------- #


def test_forfeiture_haircut_reaches_unvested_awards_only(ddog, tsm, prices):
    """Ten percent off 17,099,356 unvested units leaves 15,389,420.

    Options are left whole: the outstanding count includes vested options that
    can no longer be forfeited, and the filing does not separate them.
    """
    tsm.dilution.assumed_forfeiture_rate = 0.10
    count = build_share_count("DDOG", prices.spot("DDOG"), ddog, tsm, InstanceClient())

    assert count.rsu_shares == pytest.approx(17.099356 * 0.90)
    assert count.rsu_shares == pytest.approx(15.3894204)
    assert count.net_new_from_options == pytest.approx(1.554387, abs=5e-6)
    assert count.fully_diluted == pytest.approx(
        359.075024 + count.net_new_from_options + 15.3894204
    )
    assert any("forfeiture rate of 10.0%" in n for n in count.notes)


def test_forfeiture_default_is_zero(ddog_count):
    assert ddog_count.rsu_shares == pytest.approx(17.099356)
    assert not any("forfeiture" in n for n in ddog_count.notes)


def test_rsus_can_be_excluded(ddog, tsm, prices):
    tsm.dilution.include_rsus = False
    count = build_share_count("DDOG", prices.spot("DDOG"), ddog, tsm, InstanceClient())

    assert count.rsu_awards == []
    assert count.rsu_shares == 0.0
    assert count.fully_diluted == pytest.approx(
        359.075024 + count.net_new_from_options
    )
    assert any("excluded by assumption" in n for n in count.notes)


def test_waso_is_the_configured_default(ddog, assumptions, prices):
    """The default fetches nothing: a broken client must not even be reached."""
    assert assumptions.dilution.method == "waso"
    count = build_share_count("DDOG", prices.spot("DDOG"), ddog, assumptions, BrokenClient())

    assert count.method == "diluted WASO (configured)"
    assert count.fully_diluted == ddog.diluted_shares
    assert count.basic_outstanding is None
    assert count.flags == []


# --------------------------------------------------------------------------- #
# Degrading honestly
# --------------------------------------------------------------------------- #


def test_unreadable_instance_document_falls_back_to_waso(ddog, tsm, prices):
    count = build_share_count("DDOG", prices.spot("DDOG"), ddog, tsm, BrokenClient())

    assert count.method == "diluted WASO fallback"
    assert count.fully_diluted == ddog.diluted_shares
    assert count.basic_outstanding is None
    assert count.option_awards == [] and count.rsu_awards == []
    assert len(count.flags) == 1
    assert "instance document could not be read" in count.flags[0]
    assert count.rows()[0][0].startswith("Diluted weighted-average")


def test_missing_award_tags_fall_back_to_waso(ddog, tsm):
    """Shares outstanding without any award tags is not a share count."""
    facts = [fact(COVER, 200_000_000, "shares", "2026-06-30")]
    count = build_share_count(
        "TEST", 100.0, ddog, tsm, InstanceClient(facts, "0001-26-000002")
    )

    assert count.method == "diluted WASO fallback"
    assert count.fully_diluted == ddog.diluted_shares
    assert count.accession == "0001-26-000002"
    assert "no options outstanding and no unvested units" in count.flags[0]


def test_missing_share_count_falls_back_to_waso(ddog, tsm):
    facts = [
        fact(OPTION_COUNT, 1_000_000, "shares", "2026-06-30"),
        fact(OPTION_STRIKE, 10.0, "USD/shares", "2026-06-30"),
    ]
    count = build_share_count(
        "TEST", 100.0, ddog, tsm, InstanceClient(facts, "0001-26-000003")
    )

    assert count.method == "diluted WASO fallback"
    assert "shares outstanding are not tagged" in count.flags[0]


def test_options_without_a_strike_fall_back_rather_than_guess(ddog, tsm):
    """A strike is never invented, and the options are never silently dropped."""
    facts = [
        fact(COVER, 200_000_000, "shares", "2026-06-30"),
        fact(OPTION_COUNT, 1_000_000, "shares", "2026-06-30"),
        fact(RSU_COUNT, 5_000_000, "shares", "2026-06-30"),
    ]
    count = build_share_count(
        "TEST", 100.0, ddog, tsm, InstanceClient(facts, "0001-26-000004")
    )

    assert count.method == "diluted WASO fallback"
    assert "no weighted-average exercise price is tagged" in count.flags[0]


def test_strikeless_option_block_raises(ddog):
    with pytest.raises(MissingDataError) as exc:
        treasury_stock_shares(
            [AwardBlock("option", count=1.0, strike=None, source_tag="t")],
            price=100.0,
            basic=10.0,
        )
    assert "option exercise price" in str(exc.value)


def test_non_positive_price_is_not_meaningful():
    with pytest.raises(NotMeaningfulError):
        treasury_stock_shares(
            [AwardBlock("option", count=1.0, strike=5.0, source_tag="t")],
            price=0.0,
            basic=10.0,
        )


def test_convertibles_are_left_to_the_ev_bridge(ddog_count):
    """Datadog's notes convert into roughly 6.7mm shares. None of them are here."""
    assert any("ASU 2020-06" in n for n in ddog_count.notes)
    assert all(a.kind in ("option", "rsu") for a in ddog_count.option_awards)
    assert ddog_count.fully_diluted == pytest.approx(
        ddog_count.basic_outstanding
        + ddog_count.net_new_from_options
        + ddog_count.rsu_shares
    )


def test_fixtures_carry_the_near_miss_tags_the_ladders_must_exclude():
    """The exclusion tests are only worth anything if the traps are in the data.

    Both payloads keep the tags a loose match would pick up, so pruning can
    never quietly turn those tests green.
    """
    for ticker in ("DDOG", "ZS"):
        payload = json.loads((FIXTURES / f"instance_facts_{ticker}.json").read_text())
        assert payload["_retrieved"] == "2026-09-10"
        assert payload["_accession"]

        tags = {row["tag"] for row in payload["facts"]}
        assert any("OptionsExercisableNumber" in t for t in tags)
        assert any("OptionsExercisableWeightedAverageExercisePrice" in t for t in tags)

        # The counts the method reads are all instants. A duration context would
        # be a period movement, not a balance outstanding.
        awards = [
            row
            for row in payload["facts"]
            if "OptionsOutstandingNumber" in row["tag"]
            or "OtherThanOptionsNonvestedNumber" in row["tag"]
        ]
        assert awards and all(row["start"] is None for row in awards)

    poisoned = json.loads((FIXTURES / "instance_facts_ZS.json").read_text())
    assert any(
        "OptionsOutstandingNumber" in row["tag"]
        and row["dimensions"].get("AwardTypeAxis") == "RestrictedStockUnitsRSUMember"
        for row in poisoned["facts"]
    )
