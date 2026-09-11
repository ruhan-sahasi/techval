"""The warranted multiple: the split trap under it, the panel, and the fit.

Three fixtures, none of which touches the network.

``share_basis_facts.json`` holds the three weighted-average share tags for four
filers, pruned to periods ending 2019 or later and retrieved 2026-09-11. They are
chosen for what they break. Nvidia split four-for-one in July 2021 and ten-for-one
in June 2024, and each split is reflected in three separate filings. Arista split
four-for-one TWICE, in November 2021 and December 2024, which is the case that
separates a repeated restatement of one action from two genuine ones. Palo Alto's
two share counts differ by 5.998 rather than 6 because the counts are rounded.
Datadog has never split and must not acquire one.

``warranted/observations.json.gz`` is the recorded panel: US TMT filers at quarter
ends from 2021 to 2026, priced through ``build_observations`` against live SEC
payloads and daily closes. ``tests/fixtures/warranted/record.py`` is the script
that wrote it and is committed beside it, because a derived fixture nobody can
rebuild is an assertion.

The four committed ``companyfacts_*.json`` payloads carry the pipeline end to end
offline: features, financials, bridge, basis correction and multiple, on filers
whose numbers the rest of this suite already pins.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from techval.config import Assumptions
from techval.edgar import CompanyFacts, HttpCache
from techval.errors import ConfigError, NotMeaningfulError
from techval.ev_bridge import build_ev_bridge
from techval.financials import build_financials
from techval.market import CsvSource, MarketData
from techval.ml import warranted as W
from techval.ml.warranted import (
    BANNED_FEATURES,
    Observation,
    ObservationPanel,
    build_observations,
    fit_warranted,
    share_basis_factor,
)
from techval.tmt.metrics import build_metrics
from techval.tmt.taxonomy import SubVertical

FIXTURES = Path(__file__).parent.parent / "fixtures"
PRICES = FIXTURES / "prices"
PANEL_FIXTURE = FIXTURES / "warranted" / "observations.json.gz"

# Quarter ends inside the committed price history, which starts 2023-09-01.
TINY_DATES = [
    date(2024, 6, 30),
    date(2024, 9, 30),
    date(2024, 12, 31),
    date(2025, 3, 31),
    date(2025, 6, 30),
    date(2025, 9, 30),
    date(2025, 12, 31),
    date(2026, 3, 31),
    date(2026, 6, 30),
]
TINY_TICKERS = ["CRWD", "DDOG", "MDB", "ZS"]
TINY_VERTICALS = {t: "infrastructure_software" for t in TINY_TICKERS}


# --------------------------------------------------------------------------- #
# Fixture plumbing
# --------------------------------------------------------------------------- #


def _payloads() -> dict[str, dict]:
    return {
        t: json.loads((FIXTURES / f"companyfacts_{t}.json").read_text())
        for t in TINY_TICKERS
    }


class PinnedClient:
    """A fixture client that honours a knowledge date, which the panel requires.

    ``conftest.FixtureClient`` deliberately does not pin, so every row built
    through it would see restatements filed since. ``build_features`` refuses an
    unpinned client, and this is the pinned one.
    """

    def __init__(self, payloads: dict[str, dict], when: date | None) -> None:
        self.payloads = payloads
        self.when = when

    def company_facts(self, ticker: str) -> CompanyFacts:
        return CompanyFacts(self.payloads[ticker.upper()], ticker, knowledge_date=self.when)

    def ticker_to_cik(self, ticker: str) -> int:
        return int(self.payloads[ticker.upper()]["cik"])


@pytest.fixture(scope="module")
def split_facts() -> dict[str, dict]:
    return json.loads((FIXTURES / "share_basis_facts.json").read_text())


@pytest.fixture(scope="module")
def payloads() -> dict[str, dict]:
    return _payloads()


@pytest.fixture(scope="module")
def tiny(payloads) -> ObservationPanel:
    assumptions = Assumptions()
    assumptions.market.risk_free_rate = 0.0483
    return build_observations(
        TINY_TICKERS,
        TINY_DATES,
        lambda when: PinnedClient(payloads, when),
        lambda when: MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=when),
        TINY_VERTICALS,
        assumptions,
        current_client=PinnedClient(payloads, None),
    )


@pytest.fixture(scope="module")
def panel() -> ObservationPanel:
    with gzip.open(PANEL_FIXTURE, "rt") as handle:
        payload = json.load(handle)
    observations = [
        Observation(
            ticker=row["ticker"],
            as_of=date.fromisoformat(row["as_of"]),
            sub_vertical=row["sub_vertical"],
            multiple=row["multiple"],
            log_multiple=float(np.log(row["multiple"])),
            enterprise_value=row["enterprise_value"],
            equity_value=row["equity_value"],
            denominator=row["denominator"],
            statement_date=(
                None
                if row["statement_date"] is None
                else date.fromisoformat(row["statement_date"])
            ),
            basis_factor=row["basis_factor"],
            features=row["features"],
        )
        for row in payload["observations"]
    ]
    return ObservationPanel(
        target=payload["target"],
        observations=observations,
        notes=list(payload["notes"]),
        random_seed=7,
    )


@pytest.fixture(scope="module")
def fitted(panel) -> W.WarrantedModel:
    return fit_warranted(panel, Assumptions(), model="mlp")


@pytest.fixture(scope="module")
def fitted_ridge(panel) -> W.WarrantedModel:
    return fit_warranted(panel, Assumptions(), model="ridge")


@pytest.fixture(scope="module")
def short_panel(panel) -> ObservationPanel:
    """The first twelve dates, for the checks that do not need the whole history.

    Determinism is a property of the code rather than of the sample size, and a
    fit is the most expensive thing in this file, so the two determinism checks
    run on a panel less than half the size. It still clears
    ``min_train_observations`` comfortably, which is the only thing about it that
    has to be true.
    """
    keep = set(panel.dates[:12])
    return ObservationPanel(
        target=panel.target,
        observations=[o for o in panel.observations if o.as_of in keep],
        random_seed=panel.random_seed,
    )


# --------------------------------------------------------------------------- #
# The split detector, which was wrong before this branch
# --------------------------------------------------------------------------- #


def test_one_corporate_action_gets_one_threshold(split_facts):
    """Nvidia split twice, not six times.

    Each split is reflected in three separate filings, because a report restates
    only the comparative periods it happens to show. Counting each filing as its
    own action multiplied a pre-split share count by four three times and by ten
    three times over: Nvidia's fiscal 2022 diluted count came back as 2,535,000mm
    shares against a true 25,350mm, and the equity value built on it was wrong by
    a factor of a hundred.
    """
    facts = CompanyFacts(split_facts["NVDA"], "NVDA")
    assert facts._split_factors() == [
        (date(2021, 8, 20), 4.0),
        (date(2024, 8, 28), 10.0),
    ]


def test_two_splits_of_the_same_ratio_stay_two(split_facts):
    """Arista is the case that decides the collapsing rule.

    Two four-for-one splits three years apart across six filings. A rule that
    merged same-ratio detections on nothing but the ratio would report one, and
    every Arista share count before November 2021 would then be a quarter of what
    it should be. The period end separates them: the December 2024 filings restate
    periods that closed AFTER the November 2021 split, so they cannot be that
    split restating itself.
    """
    facts = CompanyFacts(split_facts["ANET"], "ANET")
    assert facts._split_factors() == [
        (date(2022, 2, 15), 4.0),
        (date(2025, 2, 19), 4.0),
    ]


def test_two_splits_of_different_ratios_are_both_kept(split_facts):
    facts = CompanyFacts(split_facts["PANW"], "PANW")
    assert facts._split_factors() == [
        (date(2023, 5, 24), 3.0),
        (date(2025, 2, 14), 2.0),
    ]


def test_no_split_is_invented_for_a_filer_that_never_split(split_facts):
    assert CompanyFacts(split_facts["DDOG"], "DDOG")._split_factors() == []


def test_a_split_the_knowledge_date_has_not_reached_has_not_happened(split_facts):
    """The point-in-time half of the same bug.

    A run pinned to March 2022 read Nvidia's June 2024 split off the 2024 filings
    and applied it, so a share count nobody would report for another two years
    landed in a valuation dated 2022. The detector now filters its own rows on the
    filing date exactly as ``facts`` does.
    """
    pinned = CompanyFacts(split_facts["NVDA"], "NVDA", knowledge_date=date(2022, 3, 31))
    assert pinned._split_factors() == [(date(2021, 8, 20), 4.0)]

    tag = "WeightedAverageNumberOfDilutedSharesOutstanding"
    fiscal_2022 = [
        f
        for f in pinned.facts(tag)
        if f.end == date(2022, 1, 30) and f.start == date(2021, 2, 1)
    ]
    assert len(fiscal_2022) == 1
    assert fiscal_2022[0].val == pytest.approx(2_535_000_000)

    current = CompanyFacts(split_facts["NVDA"], "NVDA")
    restated = [
        f
        for f in current.facts(tag)
        if f.end == date(2022, 1, 30) and f.start == date(2021, 2, 1)
    ]
    assert restated[0].val == pytest.approx(25_350_000_000)


# --------------------------------------------------------------------------- #
# The share basis correction
# --------------------------------------------------------------------------- #


def test_the_basis_factor_is_the_split_between_the_two_fact_sets(split_facts):
    """Nvidia needs a factor of ten in 2022 and of forty before July 2021.

    The price vendor restates its whole history for a split, so the close for 31
    March 2022 comes back as 27.29 rather than the 272.86 that printed. The share
    count from a pinned fact set is on the basis of the day. Multiplying them
    gives a tenth of the real equity value, and this is the conversion that
    repairs it.
    """
    current = CompanyFacts(split_facts["NVDA"], "NVDA")
    for when, expected in (
        (date(2021, 6, 30), 40.0),
        (date(2022, 3, 31), 10.0),
        (date(2024, 6, 30), 10.0),
        (date(2025, 3, 31), 1.0),
    ):
        pinned = CompanyFacts(split_facts["NVDA"], "NVDA", knowledge_date=when)
        factor, note = share_basis_factor(pinned, current)
        assert factor == pytest.approx(expected), when
        assert note is None


def test_a_filing_date_threshold_would_not_have_been_good_enough(split_facts):
    """Palo Alto is the counterexample to the tempting version of this fix.

    The three-for-one was September 2022 but the first filing to restate a
    comparative by three is dated May 2023. Any rule keyed on that filing date
    treats the whole of the intervening period as pre-split and multiplies by
    three when the price feed has already adjusted. Reading the ratio off the two
    fact sets has no such window: at the end of 2022 the answer is 2, the December
    2024 split alone.
    """
    current = CompanyFacts(split_facts["PANW"], "PANW")
    pinned = CompanyFacts(split_facts["PANW"], "PANW", knowledge_date=date(2022, 12, 31))
    factor, note = share_basis_factor(pinned, current)
    assert factor == pytest.approx(2.0)
    assert note is None

    threshold_rule = 1.0
    for when, ratio in current._split_factors():
        if when > date(2022, 12, 31):
            threshold_rule *= ratio
    assert threshold_rule == pytest.approx(6.0)


def test_a_rounding_restatement_is_snapped_rather_than_believed(split_facts):
    """The two counts differ by 5.998, and the answer is 6.

    Share counts are reported rounded to thousands, so the ratio between two of
    them is never exactly the split ratio. Snapping to a product of the splits the
    filer actually declared is what turns 5.998 into the 6 it obviously is, and
    the tolerance is tight enough that an ordinary restatement cannot reach it.
    """
    current = CompanyFacts(split_facts["PANW"], "PANW")
    pinned = CompanyFacts(split_facts["PANW"], "PANW", knowledge_date=date(2022, 3, 31))
    tag = "WeightedAverageNumberOfDilutedSharesOutstanding"
    now = {(f.start, f.end): f.val for f in current.facts(tag)}
    raw = [
        now[(f.start, f.end)] / f.val
        for f in pinned.facts(tag)
        if f.val and (f.start, f.end) in now
    ][-1]
    assert raw != pytest.approx(6.0, abs=1e-9)
    assert raw == pytest.approx(6.0, rel=1e-3)
    assert share_basis_factor(pinned, current)[0] == pytest.approx(6.0)


def test_a_filer_that_never_split_needs_no_conversion(split_facts):
    current = CompanyFacts(split_facts["DDOG"], "DDOG")
    pinned = CompanyFacts(split_facts["DDOG"], "DDOG", knowledge_date=date(2023, 6, 30))
    assert share_basis_factor(pinned, current) == (1.0, None)


def _synthetic_share_facts(old: float, new: float) -> dict:
    """One period, filed twice, moving by whatever ratio the caller wants."""
    return {
        "cik": 1,
        "entityName": "Synthetic",
        "facts": {
            "us-gaap": {
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": old,
                                "filed": "2025-02-01",
                                "form": "10-K",
                                "fy": 2024,
                                "fp": "FY",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": new,
                                "filed": "2026-02-01",
                                "form": "10-K",
                                "fy": 2025,
                                "fp": "FY",
                            },
                        ]
                    }
                }
            }
        },
    }


def test_a_basis_move_that_matches_no_declared_split_is_refused_not_guessed():
    """Seven-for-one is not a ratio any board declares, so nothing explains this.

    The detector needs two periods moving by a ratio it recognises, and one
    period moving by seven is neither. The share count has still moved by seven,
    which means the price and the count may be on different bases, and the honest
    answer is to say so and let the caller drop the row rather than to invent a
    conversion or to assume there was none.
    """
    payload = _synthetic_share_facts(100_000_000, 700_000_000)
    current = CompanyFacts(payload, "SYN")
    pinned = CompanyFacts(payload, "SYN", knowledge_date=date(2025, 6, 30))
    factor, note = share_basis_factor(pinned, current)
    assert factor == pytest.approx(7.0)
    assert note is not None
    assert "different bases" in note


def test_a_one_percent_restatement_is_not_a_corporate_action():
    payload = _synthetic_share_facts(100_000_000, 101_000_000)
    current = CompanyFacts(payload, "SYN")
    pinned = CompanyFacts(payload, "SYN", knowledge_date=date(2025, 6, 30))
    assert share_basis_factor(pinned, current) == (1.0, None)


def test_an_unresolved_basis_costs_the_row_and_names_the_reason(payloads, monkeypatch):
    """A row whose basis cannot be explained is skipped under its own category."""

    def unresolved(pinned, current):
        return 7.0, "the price and the share count may be on different bases"

    monkeypatch.setattr(W, "share_basis_factor", unresolved)
    assumptions = Assumptions()
    assumptions.market.risk_free_rate = 0.0483
    panel = build_observations(
        ["DDOG"],
        TINY_DATES[:2],
        lambda when: PinnedClient(payloads, when),
        lambda when: MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=when),
        TINY_VERTICALS,
        assumptions,
        current_client=PinnedClient(payloads, None),
    )
    assert not panel.observations
    assert {s.category for s in panel.skips} == {"share_basis_unresolved"}


def test_the_correction_reaches_the_enterprise_value(tiny, payloads):
    """CrowdStrike's four-for-one lands in mid-2026, so every earlier row needs it.

    The committed fixture set carries the split, which makes this the one place
    the whole chain is checked offline: pinned facts, an adjusted price, the
    conversion, and an enterprise value that is four times what it would otherwise
    have been.
    """
    early = [o for o in tiny.observations if o.ticker == "CRWD" and o.as_of.year < 2026]
    assert early, "the fixture should carry pre-split CrowdStrike rows"
    assert all(o.basis_factor == pytest.approx(4.0) for o in early)

    sample = early[-1]
    market = MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=sample.as_of)
    facts = CompanyFacts(payloads["CRWD"], "CRWD", knowledge_date=sample.as_of)
    fin = build_financials("CRWD", facts=facts)
    bridge = build_ev_bridge(fin, market.spot("CRWD"), Assumptions())
    assert sample.equity_value == pytest.approx(bridge.equity_value * 4.0)
    assert sample.enterprise_value == pytest.approx(
        bridge.equity_value * 4.0 + bridge.net_debt
    )


# --------------------------------------------------------------------------- #
# Building the panel
# --------------------------------------------------------------------------- #


def test_nothing_leaves_the_universe_without_a_reason(tiny):
    """Observed plus skipped equals every company-date that was asked for.

    A panel that quietly shrank is the commonest way a cross-sectional result gets
    quoted with more confidence than it earned, so the arithmetic is asserted
    rather than trusted.
    """
    assert len(tiny.observations) + len(tiny.skips) == len(TINY_TICKERS) * len(TINY_DATES)
    assert all(skip.reason for skip in tiny.skips)
    assert all(skip.category for skip in tiny.skips)


def test_the_enterprise_value_is_the_bridge_and_not_a_reimplementation(tiny, payloads):
    """Net debt, leases and convertibles are treated as they are everywhere else."""
    unsplit = [o for o in tiny.observations if o.basis_factor == 1.0]
    assert unsplit
    sample = unsplit[-1]
    market = MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=sample.as_of)
    facts = CompanyFacts(payloads[sample.ticker], sample.ticker, knowledge_date=sample.as_of)
    fin = build_financials(sample.ticker, facts=facts)
    bridge = build_ev_bridge(fin, market.spot(sample.ticker), Assumptions())
    assert sample.enterprise_value == pytest.approx(bridge.enterprise_value)
    assert sample.multiple == pytest.approx(bridge.enterprise_value / fin.revenue)


def test_an_unprofitable_filer_is_refused_on_ev_ebitda_and_named(payloads):
    """The selection decision the log forces, made visible instead of quiet.

    EV/EBITDA is undefined for a filer running negative GAAP EBITDA, and those are
    exactly the high-growth names the multiple gets quoted for. Dropping them
    silently restricts the model to profitable companies and then reports the
    result as though it were about the sector.
    """
    assumptions = Assumptions()
    assumptions.market.risk_free_rate = 0.0483
    panel = build_observations(
        TINY_TICKERS,
        TINY_DATES,
        lambda when: PinnedClient(payloads, when),
        lambda when: MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=when),
        TINY_VERTICALS,
        assumptions,
        current_client=PinnedClient(payloads, None),
        target="ev_ebitda",
    )
    lost = [s for s in panel.skips if s.category in ("not_meaningful", "denominator_missing")]
    assert lost, "at least one fixture filer should fail to carry a positive EBITDA"
    assert any("not meaningful" in s.reason or "not reported" in s.reason for s in lost)
    counts = panel.selection().set_index("outcome")["n"].to_dict()
    assert sum(counts.values()) == len(TINY_TICKERS) * len(TINY_DATES)


# --------------------------------------------------------------------------- #
# The three guards on the inputs, each written because this universe hit it
# --------------------------------------------------------------------------- #


def test_a_lessor_tagging_only_its_services_revenue_is_caught(split_facts):
    """Crown Castle resolves a thirty-first of its revenue and nothing complains.

    The ladder ranks the ASC 606 concept first, which is right for an operating
    company. A tower REIT's tenant income is lease revenue under ASC 842 and is
    not a contract with a customer, so only its site-services piece carries the
    606 tag and the total sits in ``Revenues``. Without this guard the towers and
    fibre bucket reads as though it trades at 125x revenue.
    """

    class Facts:
        def __init__(self, resolved):
            self.resolved = resolved

        def resolve_ttm(self, concept, ladder, as_of, **kwargs):
            return self.resolved.get(ladder[0]), None

    class Fin:
        revenue = 210.0
        as_of = date(2024, 12, 31)

    caught = W.revenue_is_a_component(
        Facts(
            {
                "RevenueFromContractWithCustomerExcludingAssessedTax": 210.0e6,
                "Revenues": 6_568.0e6,
            }
        ),
        Fin(),
    )
    assert caught is not None
    assert "Revenues" in caught
    assert "6,568.0mm" in caught
    assert "31.3 times larger" in caught

    # Two definitions of revenue that genuinely disagree, by sales tax or by
    # agency netting, differ by tens of percent and must not fire.
    assert (
        W.revenue_is_a_component(
            Facts(
                {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": 210.0e6,
                    "Revenues": 240.0e6,
                }
            ),
            Fin(),
        )
        is None
    )


def test_a_filer_reporting_debt_outside_the_ladder_is_caught():
    """Lumen resolves zero straight debt against roughly twenty billion of it.

    ``LongTermDebtAndCapitalLeaseObligations`` matches nothing in the three debt
    ladders, and ``LongTermDebt`` is present but retired in 2021, so a resolution
    that correctly asks for a value at the balance sheet date finds nothing and
    returns zero. The enterprise value then prints as equity less cash.
    """

    class Facts:
        def __init__(self, resolved):
            self.resolved = resolved

        def resolve_instant(self, concept, ladder, as_of, **kwargs):
            return self.resolved.get(ladder[0]), None

    class Fin:
        as_of = date(2023, 9, 30)

        def __init__(self, straight_debt):
            self.straight_debt = straight_debt

    found = {"LongTermDebtAndCapitalLeaseObligations": 20_000e6}
    caught = W.debt_is_outside_the_ladder(Facts(found), Fin(0.0))
    assert caught is not None
    assert "LongTermDebtAndCapitalLeaseObligations" in caught
    assert "20,000.0mm" in caught

    # A company that really has no straight debt, and one whose debt the ladder
    # did resolve, both pass.
    assert W.debt_is_outside_the_ladder(Facts({}), Fin(0.0)) is None
    assert W.debt_is_outside_the_ladder(Facts(found), Fin(4_000.0)) is None


def test_a_price_series_that_is_not_this_company_is_caught(payloads):
    """The only independent anchor on a price feed that free data offers.

    A vendor returning the wrong security gives a clean, monotone, entirely
    plausible series, and nothing inside a valuation can tell. The filer's own
    cover page can: ``dei:EntityPublicFloat`` is a number the company states.
    Priced on the vendor's series for BKNG, Booking's equity value comes out at
    7,796mm against a reported float of 133,100mm.

    The tolerance is ten times either way and is meant never to fire on anything
    real, so the test pins both ends: a sixteenfold gap is refused and a
    sixfold one, which is what a founder-led name that has run hard since its
    last cover page looks like, is not.
    """
    facts = CompanyFacts(payloads["DDOG"], "DDOG")
    floats = [f for f in facts.facts("EntityPublicFloat") if f.val]
    assert floats, "the fixture should carry a public float"
    reported = max(floats, key=lambda f: (f.end, f.filed)).val / 1e6

    class Fin:
        pass

    assert W.price_disagrees_with_the_public_float(facts, Fin(), reported) is None
    assert W.price_disagrees_with_the_public_float(facts, Fin(), reported * 6.0) is None
    too_big = W.price_disagrees_with_the_public_float(facts, Fin(), reported * 16.0)
    assert too_big is not None and "does not belong to this filer" in too_big
    too_small = W.price_disagrees_with_the_public_float(facts, Fin(), reported / 16.0)
    assert too_small is not None


def test_a_company_with_no_classification_is_recorded_not_dropped(payloads):
    assumptions = Assumptions()
    assumptions.market.risk_free_rate = 0.0483
    panel = build_observations(
        TINY_TICKERS,
        TINY_DATES[:2],
        lambda when: PinnedClient(payloads, when),
        lambda when: MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=when),
        {"DDOG": "infrastructure_software"},
        assumptions,
        current_client=PinnedClient(payloads, None),
    )
    unclassified = [s for s in panel.skips if s.category == "unclassified"]
    assert {s.ticker for s in unclassified} == {"CRWD", "MDB", "ZS"}


def test_tmt_metrics_cannot_be_run_across_this_universe(payloads):
    """Why the features come from the feature store and not from ``tmt.metrics``.

    The brief asked for both. Measurement says only one is possible.
    ``tmt.metrics.build_metrics`` routes on a sub-vertical string and refuses an
    unknown one, which is the right call, but its vocabulary and
    ``taxonomy.SubVertical`` overlap on exactly two values. Handed the other nine
    it raises, so it cannot be applied across a TMT panel at all.

    This is a tripwire as much as a test. If somebody reconciles the two
    vocabularies it will fail, and the paragraph in the module docstring
    explaining the omission should be deleted at the same time.
    """
    facts = CompanyFacts(payloads["DDOG"], "DDOG")
    fin = build_financials("DDOG", facts=facts)
    accepted, refused = [], []
    for vertical in SubVertical:
        try:
            build_metrics(fin, vertical.value, Assumptions())
            accepted.append(vertical.value)
        except ConfigError:
            refused.append(vertical.value)
    assert accepted == ["internet", "telecom"]
    assert len(refused) == 9
    assert "infrastructure_software" in refused
    assert "semiconductors" in refused


def test_an_unknown_target_is_refused(payloads):
    with pytest.raises(ConfigError, match="ev_revenue"):
        build_observations(
            ["DDOG"],
            TINY_DATES[:1],
            lambda when: PinnedClient(payloads, when),
            lambda when: MarketData(CsvSource(PRICES), HttpCache(enabled=False), today=when),
            TINY_VERTICALS,
            Assumptions(),
            current_client=PinnedClient(payloads, None),
            target="pe",
        )


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_it_reports_nothing_below_the_minimum_training_observations(tiny):
    """Thirty-six observations against a floor of a hundred and fifty."""
    assert len(tiny.observations) < Assumptions().ml.warranted.min_train_observations
    with pytest.raises(NotMeaningfulError, match="min_train_observations"):
        fit_warranted(tiny, Assumptions())


def test_a_market_capitalisation_feature_is_refused_with_the_reason(panel):
    """The answer written on the other side of the page.

    The target is log EV less log revenue. A model handed log market
    capitalisation discovers that log EV predicts log EV, scores beautifully and
    means nothing, and the failure is invisible in every diagnostic except this
    one.
    """
    for banned in ("scale_log_market_cap", "scale_log_enterprise_value", "capital_net_debt_to_ev"):
        with pytest.raises(ConfigError) as excinfo:
            fit_warranted(panel, Assumptions(), features=(*W.WARRANTED_FEATURES, banned))
        assert banned in str(excinfo.value)
        assert BANNED_FEATURES[banned][:30] in str(excinfo.value)


def test_a_feature_the_store_does_not_build_is_refused(panel):
    with pytest.raises(ConfigError, match="FEATURE_NAMES"):
        fit_warranted(panel, Assumptions(), features=("growth_revenue_1y", "vibes"))


def test_an_unknown_model_is_refused(panel):
    with pytest.raises(ConfigError, match="ridge"):
        fit_warranted(panel, Assumptions(), model="transformer")


# --------------------------------------------------------------------------- #
# The re-rating, which is trap one
# --------------------------------------------------------------------------- #


def test_the_calendar_is_a_small_share_of_the_pooled_variance(fitted, panel):
    """The re-rating is real, and it is not most of the variance. It is 2%.

    This is the one place measurement overturned the brief outright. The
    expectation was that a model fitted across 2021 and 2023 would be dominated by
    the rate cycle, and the software medians in this panel do move exactly as
    predicted: 17.1x at the end of 2021, 8.7x a year later, 7.7x by mid-2026. But
    pooled across the whole TMT universe that movement is 2.1% of the variance of
    the log multiple, because the dispersion BETWEEN sub-verticals swamps it. A
    carrier at 2x and an infrastructure software name at 13x are four times
    further apart than either of them moved across the cycle.

    Inside a single sub-vertical, where the brief's claim actually lives, the
    share rises to between 9% and 18%. Still not most of it, and still worth
    removing, because it is a real effect that has nothing to do with any company.
    Demeaning stays on by default for that reason rather than for the one given.
    """
    rerating = fitted.rerating
    assert rerating.n_dates >= 8
    assert rerating.total_variance > 0
    assert rerating.calendar_r_squared == pytest.approx(rerating.between_date_share, abs=1e-9)
    assert rerating.within_date_sd > 0
    assert "re-rating" in rerating.sentence()

    # Pooled: small.
    assert rerating.between_date_share < 0.10

    # Inside one sub-vertical: several times larger, and still a minority.
    shares = []
    for vertical in ("application_software", "infrastructure_software", "semiconductors"):
        rows = [o for o in panel.observations if o.sub_vertical == vertical]
        assert len(rows) > 150
        one = W._rerating(
            np.asarray([o.log_multiple for o in rows]),
            np.asarray([o.as_of for o in rows], dtype=object),
        )
        shares.append(one.between_date_share)
        assert 0.05 < one.between_date_share < 0.50
    assert min(shares) > rerating.between_date_share * 3


def test_the_variance_decomposition_adds_up(fitted):
    """Between plus within is the total, which is what makes the share a share."""
    r = fitted.rerating
    assert r.between_date_variance + r.within_date_variance == pytest.approx(
        r.total_variance
    )


def test_the_date_mean_leaves_out_the_row_it_is_applied_to():
    """A mean that includes the observation hands back 1/N of its own answer."""
    y = np.array([1.0, 2.0, 3.0, 10.0])
    dates = np.array([date(2024, 3, 31)] * 3 + [date(2024, 6, 30)], dtype=object)
    loo, plain = W._leave_one_out_means(y, dates)
    assert loo[0] == pytest.approx(2.5)
    assert loo[1] == pytest.approx(2.0)
    assert loo[2] == pytest.approx(1.5)
    assert plain[date(2024, 3, 31)] == pytest.approx(2.0)
    # A date holding one name has no others to average, so it falls back to the
    # value itself rather than dividing by zero.
    assert loo[3] == pytest.approx(10.0)


def test_demeaning_can_be_turned_off_and_the_fit_changes(panel, fitted_ridge):
    """Reported both ways, because the claim is that the difference is large.

    With the date mean removed the model is fitting a number centred on zero, so
    the intercept is near it. Left in, the model has to carry the level of the
    whole market and the intercept becomes the average log multiple of the sample.
    The decomposition itself does not move: it describes the panel, not the fit.
    """
    assumptions = Assumptions()
    assumptions.ml.warranted.demean_by_date = False
    raw = fit_warranted(panel, assumptions, model="ridge")
    demeaned = fitted_ridge

    assert raw.demeaned is False
    assert demeaned.demeaned is True
    assert raw.rerating.between_date_share == pytest.approx(
        demeaned.rerating.between_date_share
    )
    assert abs(raw.coefficients["intercept"]) > abs(demeaned.coefficients["intercept"])
    assert abs(demeaned.coefficients["intercept"]) < 0.25


# --------------------------------------------------------------------------- #
# Out of sample, which is trap three
# --------------------------------------------------------------------------- #


def test_every_reported_residual_comes_from_a_fold_that_did_not_train_on_it(fitted, panel):
    """An in-sample residual is partly the model fitting that very company."""
    first_test = min(fold.test_start for fold in fitted.folds)
    assert all(when >= first_test for _, when in fitted.reads)
    assert all(read.out_of_sample for read in fitted.reads.values())

    # Everything before the first test block is the initial training window and
    # is scored nowhere, so it must not have a read.
    early = {
        (o.ticker, o.as_of) for o in panel.observations if o.as_of < first_test
    }
    assert not (early & set(fitted.reads))


def test_the_training_window_never_reaches_into_its_own_test_block(fitted):
    for fold in fitted.folds:
        assert fold.train_end < fold.test_start
        assert fold.n_train >= Assumptions().ml.warranted.min_train_observations


def test_the_embargo_is_zero_and_says_so(fitted):
    assert fitted.card.hyperparameters["embargo_days"] == 0
    assert any("embargo is zero" in note for note in fitted.card.notes)


# --------------------------------------------------------------------------- #
# The honesty contract
# --------------------------------------------------------------------------- #


def test_every_baseline_is_reported_not_only_the_flattering_one(fitted):
    """Three baselines, all named, all scored, none of them a straw man."""
    names = " | ".join(fitted.baselines)
    assert "sub-vertical median" in names
    assert "persistence" in names
    assert "comps.py OLS" in names
    for result in fitted.baselines.values():
        assert result.n_observations >= 30
        assert result.metric == "spearman"


def test_the_card_is_scored_against_the_strongest_comparable_baseline(fitted):
    """The hardest baseline that answers the same question, not the flattering one.

    Persistence is the one exclusion and it is deliberate. It is the company's own
    multiple carried forward, so it is built from the very price a warranted
    multiple exists to be differenced against, and a construction that reproduced
    it would have a residual of zero and no signal. It stays in ``baselines`` and
    it gets its own note, so nobody has to take the exclusion on trust.
    """
    assert fitted.card.evaluation is not None
    comparable = {
        name: result
        for name, result in fitted.baselines.items()
        if "persistence" not in name
    }
    assert len(comparable) >= 2
    best = max(r.baseline_score for r in comparable.values())
    assert fitted.card.evaluation.baseline_score == pytest.approx(best)
    assert "persistence" not in fitted.card.evaluation.baseline_name


def test_the_persistence_ceiling_is_reported_even_though_it_is_excluded(fitted):
    name = next(n for n in fitted.baselines if "persistence" in n)
    assert fitted.baselines[name].n_observations >= 30
    assert any(
        "NOT the card's baseline" in note and "persistence" in note
        for note in fitted.card.notes
    )


def test_beat_baseline_is_computed_rather_than_asserted(fitted):
    result = fitted.card.evaluation
    assert result.beat_baseline == (result.score > result.baseline_score)
    verdict = fitted.verdict()
    if not result.beat_baseline:
        assert "Use the baseline" in verdict
    else:
        assert "lift" in verdict


def test_the_headline_travels_with_the_number_that_deflates_it(fitted):
    """A pooled score over the same companies quarter after quarter is the companies.

    The model is rewarded for recognising a name as much as for understanding it,
    because a feature vector barely moves in three months and neither does a
    relative multiple. Differencing both sides against the company's own previous
    observation asks what was added on top of knowing which company it is, and the
    answer is small. It is a field rather than a note, and ``verdict`` prints it
    beside the headline, so the two cannot be quoted apart.
    """
    assert fitted.change_rank_correlation is not None
    assert fitted.n_changes >= 30
    assert abs(fitted.change_rank_correlation) < abs(fitted.card.evaluation.score)
    verdict = fitted.verdict()
    assert "Differenced against each company's own previous observation" in verdict
    assert f"{fitted.change_rank_correlation:+.4f}" in verdict
    assert "not as a forecast" in verdict
    assert any("company identity" in lim for lim in fitted.card.limitations)


def test_the_change_diagnostic_is_zero_when_the_model_only_carries_the_level():
    """The diagnostic has to be able to return nothing, or it proves nothing.

    A "model" that simply repeats the previous observation predicts no change at
    all, so both sides of the difference are the same constant and the rank
    correlation is undefined rather than flattering. A model that predicts the
    change perfectly scores one. Both ends are pinned here so the +0.04 on the
    real panel can be read as a measurement rather than as an artefact.
    """
    rng = np.random.default_rng(0)
    previous = rng.normal(size=200)
    change = rng.normal(size=200) * 0.3
    y = previous + change
    mask = np.ones(200, dtype=bool)

    carries_the_level = W._change_in_relative_position(y, previous.copy(), previous, mask)
    assert carries_the_level[0] is None
    assert carries_the_level[1] == 200

    knows_the_change = W._change_in_relative_position(y, y.copy(), previous, mask)
    assert knows_the_change[0] == pytest.approx(1.0)

    # Below the evaluation module's own floor it reports nothing rather than a
    # correlation over a handful of points.
    thin = np.zeros(200, dtype=bool)
    thin[:10] = True
    assert W._change_in_relative_position(y, y.copy(), previous, thin) == (None, 10)


def test_the_card_carries_its_limitations(fitted):
    joined = " ".join(fitted.card.limitations).lower()
    assert "reflex" in joined
    assert "survivorship" in joined
    assert "independent" in joined


def test_the_comps_refusal_rate_is_reported(fitted):
    assert any("comps.py refused" in note for note in fitted.card.notes)


def test_the_harder_within_sub_vertical_number_is_reported_too(fitted):
    """The pooled score is mostly the ordering of the buckets, and that is said."""
    note = next(
        n for n in fitted.card.notes if "Inside one sub-vertical on one date" in n
    )
    assert "the ordering of the buckets" in note
    assert "sub-vertical median" in note


def test_a_grouped_rank_correlation_averages_cross_sections_not_observations():
    """A date with forty names must not drown one with twelve.

    Built so the two groups disagree: the first is ordered perfectly and the
    second is ordered backwards, so a mean over cross-sections is zero while a
    pooled correlation over all sixteen rows would be dominated by neither in an
    obvious way. The mean is the number this reports.
    """
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])
    pred = np.concatenate([y[:8], -y[8:]])
    dates = np.array([date(2025, 3, 31)] * 16, dtype=object)
    verticals = np.array(["telecom"] * 8 + ["gaming"] * 8)
    mask = np.ones(16, dtype=bool)
    value, groups = W._grouped_rank_correlation(y, pred, dates, verticals, mask, 8)
    assert groups == 2
    assert value == pytest.approx(0.0)

    # Raise the floor above either group and nothing is reportable.
    assert W._grouped_rank_correlation(y, pred, dates, verticals, mask, 9) == (None, 0)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_two_fits_from_one_seed_agree_to_the_bit(short_panel):
    """``assumptions.ml.random_seed`` is the only source of randomness here.

    Equality, not approximate equality. The weight initialisation, the dropout
    masks and the shuffling of batch order all come from generators seeded off
    that one number, so two runs produce the same parameters bit for bit and
    anything less would mean something in the stack is reading a clock or an
    address.
    """
    first = fit_warranted(short_panel, Assumptions(), model="mlp")
    second = fit_warranted(short_panel, Assumptions(), model="mlp")
    assert set(first.reads) == set(second.reads)
    for key, read in first.reads.items():
        assert read.warranted_multiple == second.reads[key].warranted_multiple
        assert read.residual_log == second.reads[key].residual_log
    assert first.card.evaluation.score == second.card.evaluation.score
    assert first.coefficients == second.coefficients


def test_a_different_seed_moves_the_network_and_not_the_ridge(short_panel):
    """The separation that makes the determinism claim mean something.

    A seed that changed nothing would make the test above vacuous. The ridge is a
    closed-form solve and must not move; the network is initialised from the seed
    and must.
    """
    other = Assumptions()
    other.ml.random_seed = 101
    base_mlp = fit_warranted(short_panel, Assumptions(), model="mlp")
    moved_mlp = fit_warranted(short_panel, other, model="mlp")
    assert base_mlp.card.evaluation.score != moved_mlp.card.evaluation.score
    assert base_mlp.coefficients == moved_mlp.coefficients


# --------------------------------------------------------------------------- #
# Reading it at a company
# --------------------------------------------------------------------------- #


def test_a_read_carries_its_residual_in_both_units(fitted):
    read = next(iter(fitted.reads.values()))
    assert read.residual_log == pytest.approx(
        float(np.log(read.actual_multiple) - np.log(read.warranted_multiple))
    )
    assert read.residual_turns == pytest.approx(
        read.actual_multiple - read.warranted_multiple
    )
    assert read.z == pytest.approx(read.residual_log / read.within_date_sd)


def test_the_one_liner_refuses_to_sound_like_a_valuation(fitted):
    read = max(fitted.reads.values(), key=lambda r: r.residual_log)
    sentence = read.sentence()
    assert read.ticker in sentence
    assert "not about value" in sentence
    assert "x against a warranted" in sentence


def test_a_company_that_was_never_scored_raises_rather_than_returning_nothing(fitted):
    with pytest.raises(NotMeaningfulError, match="no out-of-sample warranted multiple"):
        fitted.warranted("BRK.A")


def test_extremes_is_ordered_richest_first_and_names_both_ends(fitted):
    table = fitted.extremes(n=8)
    assert list(table["residual_log"]) == sorted(table["residual_log"], reverse=True)
    assert set(table["verdict"]) == {"rich", "cheap"}
    assert table["actual"].gt(0).all()
    assert table["warranted"].gt(0).all()


def test_extremes_refuses_a_date_the_panel_does_not_hold(fitted):
    with pytest.raises(NotMeaningfulError, match="no observations"):
        fitted.extremes(when=date(1999, 12, 31))


def test_the_information_coefficient_series_is_per_date(fitted):
    assert fitted.information_coefficients
    assert all(-1.0 <= ic <= 1.0 for ic in fitted.information_coefficients.values())
    assert set(fitted.information_coefficients) <= {when for _, when in fitted.reads}


def test_the_ridge_reports_a_coefficient_per_feature(fitted_ridge):
    assert set(fitted_ridge.coefficients) == {
        "intercept",
        *W.WARRANTED_FEATURES,
        "missing_share",
        *W.SUB_VERTICAL_COLUMNS,
    }
    assert all(np.isfinite(v) for v in fitted_ridge.coefficients.values())


def test_the_sub_vertical_is_a_fixed_effect_and_not_a_number(panel):
    """Eleven indicator columns in enum order, whatever the panel happens to hold.

    A sub-vertical encoded as an integer would tell the model that gaming sits
    ten units from infrastructure software and one unit from towers, which is
    not a fact about anything. The block is also fixed by the enum rather than by
    the sample, so a panel with no gaming names still produces a matrix of the
    same width.
    """
    X, dummies = W._design(panel.observations, W.WARRANTED_FEATURES)
    assert X.shape == (len(panel.observations), len(W.WARRANTED_FEATURES))
    assert dummies.shape == (len(panel.observations), len(W.SUB_VERTICAL_COLUMNS))
    assert set(np.unique(dummies)) <= {0.0, 1.0}
    assert dummies.sum(axis=1).max() == 1.0

    subset = panel.observations[:50]
    _, small = W._design(subset, W.WARRANTED_FEATURES)
    assert small.shape[1] == len(W.SUB_VERTICAL_COLUMNS)


def test_the_missing_share_column_ignores_the_indicator_block(panel):
    """It measures untagged FUNDAMENTALS, so a dummy must not dilute it."""
    X, dummies = W._design(panel.observations, W.WARRANTED_FEATURES)
    train = np.ones(len(panel.observations), dtype=bool)
    Z = W._prepare(X, dummies, train)
    assert Z.shape[1] == len(W.WARRANTED_FEATURES) + 1 + len(W.SUB_VERTICAL_COLUMNS)
    share = Z[:, len(W.WARRANTED_FEATURES)]
    expected = np.isnan(X).mean(axis=1)
    assert share == pytest.approx(expected)
    assert np.isfinite(Z).all()
