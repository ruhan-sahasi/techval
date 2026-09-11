"""The TMT fundamentals commands, and the two vocabulary gaps they bridge.

Nothing here touches the network. The commands run through
``typer.testing.CliRunner`` against a client that serves the committed fixtures,
so what a test asserts about the printed page is what the page really prints.

Three fixtures carry the work:

    DDOG  the software case. Company facts, the instance document its extension
          tags live in, and the full 10-K text, so the prose path runs.
    DIS   the conglomerate. Three segments that do not sum to consolidated
          revenue, the same revenue tagged again by geography, and no segment
          assets at all. Company facts and daily closes were recorded for this
          wave so the sum of the parts has a real company to run on.
    ZS    the single-reportable-segment filer, which is a valid answer rather
          than a failure.

The heart of the file is the pair of vocabulary tests. ``taxonomy.SubVertical``
and ``metrics.SUB_VERTICALS`` overlap on two values out of eleven, and
``kpis.KPISet`` and ``metrics.KPI_INPUTS`` overlap on almost nothing, so both
bridges live in ``commands_tmt`` and both are measured here rather than assumed.
Each of those tests is a tripwire as much as a test: reconcile either pair of
vocabularies upstream and the corresponding test fails, which is the signal that
the bridge in the command layer can be deleted.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from techval import commands_tmt as C
from techval.edgar import CompanyFacts, DimensionedFact, Provenance
from techval.errors import ConfigError
from techval.market import CsvSource
from techval.tmt import kpis as kpi_module
from techval.tmt import metrics as metric_module
from techval.tmt.taxonomy import SubVertical

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PRICES = FIXTURES / "prices"

# The date every fixture was retrieved, and the date every run below is pinned
# to. Without it ``build_kpis`` dates its prose metrics to today and the printed
# page changes every morning.
AS_OF = "2026-09-10"

runner = CliRunner()


# --------------------------------------------------------------------------- #
# an offline client
# --------------------------------------------------------------------------- #


def _instance(path: Path) -> tuple[list[DimensionedFact], str, date]:
    payload = json.loads(path.read_text())
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


SUBMISSIONS = json.loads((FIXTURES / "submissions_tmt.json").read_text())["companies"]


class FixtureClient:
    """Everything ``commands_tmt`` asks an ``EdgarClient`` for, off disk.

    ``instance`` names which committed instance document to serve per ticker,
    because the two families of fixture answer different questions: the
    ``instance_kpis_*`` files were pruned around the operating metrics and the
    ``instance_facts_*`` files around the segment axis.
    """

    def __init__(self, knowledge_date: date | None = None) -> None:
        self.knowledge_date = knowledge_date
        self.instance = {
            "DDOG": FIXTURES / "instance_kpis_DDOG.json",
            "DIS": FIXTURES / "instance_facts_DIS.json",
            "ZS": FIXTURES / "instance_facts_ZS.json",
        }

    def ticker_to_cik(self, ticker: str) -> int:
        return int(SUBMISSIONS[ticker.upper()]["cik"])

    def company_facts(self, ticker: str) -> CompanyFacts:
        path = FIXTURES / f"companyfacts_{ticker.upper()}.json"
        return CompanyFacts(
            json.loads(path.read_text()), ticker, knowledge_date=self.knowledge_date
        )

    def submissions(self, ticker: str) -> dict:
        return dict(SUBMISSIONS[ticker.upper()])

    def instance_facts(self, ticker, forms=("10-K", "10-Q")):
        return _instance(self.instance[ticker.upper()])

    def filings(self, ticker, forms=("10-K",), since=None, limit=20):
        if ticker.upper() != "DDOG":
            return []
        return [
            {
                "accession": "0001628280-26-008819",
                "filed": date(2026, 2, 18),
                "form": "10-K",
                "document": "ddog-20251231.htm",
                "period": "2025-12-31",
            }
        ]

    def filing_text(self, ticker, filing) -> str:
        with gzip.open(
            FIXTURES / "filing_text_DDOG_2025.json.gz", "rt", encoding="utf-8"
        ) as fh:
            return json.load(fh)["text"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Wire the command's own ``_setup`` to the fixtures.

    The constructors are replaced rather than ``_setup`` itself, so the knowledge
    date, the assumptions load and the market wiring inside it all still run. A
    test that stubbed ``_setup`` would prove the renderers work and nothing about
    whether ``--as-of`` reaches the client.
    """
    monkeypatch.setattr(
        C, "EdgarClient", lambda cache, knowledge_date=None: FixtureClient(knowledge_date)
    )
    monkeypatch.setattr(
        C, "make_price_source", lambda kind, cache, csv_dir=None: CsvSource(PRICES)
    )


def run(*args: str):
    result = runner.invoke(C.app, [*args, "--as-of", AS_OF])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


# --------------------------------------------------------------------------- #
# The sub-vertical map
# --------------------------------------------------------------------------- #


def test_every_sub_vertical_has_a_decision_recorded():
    """No bucket falls off the end of the map into a default pack.

    The failure this prevents is the one ``metrics.build_metrics`` refuses in the
    first place: a semiconductor company scored on the Rule of 40. A map that
    silently returned None for a bucket nobody had thought about would put the
    decision back in the hands of whatever the caller did with a None.
    """
    assert set(C.SUB_VERTICAL_TO_METRIC_PACK) == set(SubVertical)


def test_the_mapped_packs_are_ones_metrics_accepts():
    for vertical, pack in C.SUB_VERTICAL_TO_METRIC_PACK.items():
        if pack is None:
            continue
        assert pack in metric_module.SUB_VERTICALS, vertical


def test_the_buckets_with_no_pack_are_exactly_the_ones_kpis_leaves_empty():
    """The stated rule, measured rather than asserted in a comment.

    ``SUB_VERTICAL_TO_METRIC_PACK`` claims to carry across the judgment
    ``kpis.METRIC_PACKS`` already made about which disclosure pack a bucket
    belongs to. The checkable half of that claim is the negative space: the
    buckets that get no computed pack here should be the buckets that get no
    disclosure pack there. If somebody writes a semiconductor pack in either
    module, this fails and the other module needs the same decision.
    """
    no_computed_pack = {
        v.value for v, pack in C.SUB_VERTICAL_TO_METRIC_PACK.items() if pack is None
    }
    no_disclosure_pack = {
        name for name, pack in kpi_module.METRIC_PACKS.items() if not pack
    }
    assert no_computed_pack == no_disclosure_pack == {
        "semiconductors",
        "hardware",
        "it_services",
    }


def test_the_two_vocabularies_still_overlap_on_only_two_values():
    """The gap this map exists to close, measured here so it can be seen to close.

    A tripwire. ``metrics.SUB_VERTICALS`` is keyed on its own eleven strings and
    ``SubVertical`` on eleven others, and only ``internet`` and ``telecom`` are in
    both. Reconcile them upstream and this fails, at which point
    ``SUB_VERTICAL_TO_METRIC_PACK`` is dead code and should go with it.
    """
    shared = {v.value for v in SubVertical} & set(metric_module.SUB_VERTICALS)
    assert shared == {"internet", "telecom"}


def test_an_unmapped_bucket_is_refused_by_name_and_never_defaulted():
    with pytest.raises(ConfigError) as excinfo:
        C.resolve_metric_pack(SubVertical.SEMICONDUCTORS, None)
    message = str(excinfo.value)
    assert "semiconductors" in message
    assert "book to bill" in message
    # The whole point: the refusal must not quietly become the software pack.
    assert "software" not in message


def test_an_unclassified_filer_is_refused_with_the_ways_out():
    with pytest.raises(ConfigError) as excinfo:
        C.resolve_metric_pack(None, None)
    assert "--sub-vertical" in str(excinfo.value)
    assert "--metric-pack" in str(excinfo.value)


def test_a_forced_pack_is_validated_against_the_packs_that_exist():
    with pytest.raises(ConfigError, match="not a metric pack"):
        C.resolve_metric_pack(SubVertical.TELECOM, "wireless_towers")
    assert C.resolve_metric_pack(SubVertical.TOWERS_FIBER, "telecom")[0] == "telecom"


def test_the_weak_mappings_are_named_rather_than_hidden():
    """Gaming and towers_fiber are judgments, so the command says so out loud."""
    assert set(C.WEAK_MAPPINGS) == {SubVertical.GAMING, SubVertical.TOWERS_FIBER}
    for vertical in C.WEAK_MAPPINGS:
        assert C.SUB_VERTICAL_TO_METRIC_PACK[vertical] is not None


# --------------------------------------------------------------------------- #
# The KPI bridge
# --------------------------------------------------------------------------- #


def _fin(revenue: float = 1_000.0) -> SimpleNamespace:
    return SimpleNamespace(revenue=revenue, as_of=date(2026, 6, 30))


def _set(**kpis) -> kpi_module.KPISet:
    return kpi_module.KPISet(ticker="X", as_of=date(2026, 6, 30), kpis=dict(kpis))


def _kpi(name, value, unit, *, source="text", phrase="phrase", confidence=0.75, notes=""):
    return kpi_module.KPI(
        name=name,
        value=value,
        unit=unit,
        period_end=date(2026, 6, 30),
        source=source,
        tag_or_phrase=phrase,
        confidence=confidence,
        notes=notes,
    )


def test_the_naive_wiring_splits_three_ways_and_two_of_them_are_wrong():
    """Why ``_bridge_kpis`` exists at all, stated as arithmetic.

    Handing ``KPISet.as_dict()`` straight to ``build_metrics`` raises nothing and
    looks like it works. What it actually does splits three ways, and only the
    first is harmless.

    Four names agree on both sides and carry the same unit, so they cross
    cleanly. Two are spelled differently and reach nothing at all, which is
    visible as a blank row. The dangerous group is the remaining three: they
    share a name and do not share a unit. Subscribers are an absolute count on
    one side and millions on the other, ARPU is per whatever period the sentence
    named and the pack documents months, and churn has no period on it at all.
    Each of those crosses silently and lands wrong by a million, by three or
    twelve, and by twelve, with nothing on the page to say so.

    A tripwire too. Reconcile either side and this fails, at which point the
    bridge can be deleted.
    """
    extractor = set(kpi_module.ALL_METRICS)
    pack = set(metric_module.KPI_INPUTS)

    same_name = extractor & pack
    assert same_name == {
        "arr",
        "billings",
        "content_spend",
        "rpo",
        "arpu",
        "churn",
        "subscribers",
    }
    # Same name, same unit. These are the only four a naive wiring gets right.
    assert same_name - {"arpu", "churn", "subscribers"} == {
        "arr",
        "billings",
        "content_spend",
        "rpo",
    }
    # Different name, so they reach nothing.
    for renamed in ("net_revenue_retention", "content_amortisation"):
        assert renamed in extractor and renamed not in pack


def test_retention_crosses_under_the_name_the_pack_reads():
    bridged, lines = _bridge(net_revenue_retention=_kpi("net_revenue_retention", 1.2, "ratio"))
    assert bridged == {"nrr": 1.2}
    assert any("nrr" in line for line in lines)


def _bridge(revenue: float = 1_000.0, **kpis):
    return C._bridge_kpis(_set(**kpis), _fin(revenue))


def test_content_amortisation_crosses_the_spelling():
    """One letter, and without the bridge the whole media pack goes blank."""
    bridged, _ = _bridge(
        content_amortisation=_kpi("content_amortisation", 8_529.2, "usd_mm")
    )
    assert bridged == {"content_amortization": 8_529.2}


def test_a_subscriber_count_is_divided_into_millions():
    """An absolute count on one side of the bridge and millions on the other."""
    bridged, lines = _bridge(
        revenue=48_000.0, subscribers=_kpi("subscribers", 300_000_000.0, "count")
    )
    assert bridged["subscribers"] == pytest.approx(300.0)
    assert "divided by a million" in " ".join(lines)


def test_paid_subscribers_wins_over_a_total_that_includes_free_tiers():
    bridged, _ = _bridge(
        revenue=48_000.0,
        paid_subscribers=_kpi("paid_subscribers", 280_000_000.0, "count"),
        subscribers=_kpi("subscribers", 300_000_000.0, "count"),
    )
    assert bridged["subscribers"] == pytest.approx(280.0)


def test_a_net_additions_sentence_read_as_a_base_is_refused():
    """The T-Mobile case, which is a false positive with no flag of its own.

    The extractor's subscriber ladder matches "<count> postpaid phone customers".
    T-Mobile's Item 7 says it added 3,287,000 postpaid phone customers in the
    year, against a base of something over a hundred million, and the sentence
    does not contain the word "net" so the net-additions rule above it does not
    claim the figure first. Passed through, the pack reports a carrier with 3.3
    million subscribers and an implied 2,337 dollars of revenue per subscriber
    per month, and both numbers look like numbers.
    """
    bridged, lines = _bridge(
        revenue=92_189.0,
        subscribers=_kpi(
            "subscribers", 3_287_000.0, "count", phrase="3,287,000 postpaid phone customers"
        ),
    )
    assert "subscribers" not in bridged
    flagged = [line for line in lines if line.startswith("FLAG")]
    assert len(flagged) == 1
    assert "net additions" in flagged[0]
    assert "2,337" in flagged[0]


def test_a_plausible_count_still_crosses():
    """The gate has to let a real carrier through or it is just a refusal."""
    bridged, _ = _bridge(
        revenue=92_189.0, subscribers=_kpi("subscribers", 131_000_000.0, "count")
    )
    assert bridged["subscribers"] == pytest.approx(131.0)


@pytest.mark.parametrize(
    "unit,expected",
    [("usd_per_month", 17.0), ("usd_per_quarter", 17.0 / 3.0), ("usd_per_year", 17.0 / 12.0)],
)
def test_arpu_is_divided_down_to_the_month_the_pack_documents(unit, expected):
    bridged, _ = _bridge(arpu=_kpi("arpu", 17.0, unit))
    assert bridged["arpu"] == pytest.approx(expected)


def test_an_arpu_with_no_period_on_it_is_refused():
    """A price with no denominator is not a monthly price, it is a number."""
    bridged, lines = _bridge(arpu=_kpi("arpu", 50.37, "usd_per_period"))
    assert "arpu" not in bridged
    assert "wrong by three" in " ".join(lines)


def test_churn_crosses_only_where_the_evidence_says_month():
    monthly, _ = _bridge(
        churn=_kpi("churn", 0.0093, "ratio", phrase="monthly churn rate of 0.93%")
    )
    assert monthly == {"churn": 0.0093}

    unknown, lines = _bridge(churn=_kpi("churn", 0.0093, "ratio", phrase="churn of 0.93%"))
    assert "churn" not in unknown
    assert "wrong by twelve" in " ".join(lines)


def test_a_refused_figure_never_reaches_the_pack():
    bridged, lines = _bridge(
        arr=_kpi("arr", 0.0, "usd_mm", confidence=0.0, notes="no scale word")
    )
    assert bridged == {}
    assert "the extractor refused it" in " ".join(lines)


def test_a_figure_with_no_counterpart_says_so_instead_of_vanishing():
    _bridged, lines = _bridge(customers=_kpi("customers", 32_700.0, "count"))
    assert any("reaches no computed metric" in line for line in lines)


# --------------------------------------------------------------------------- #
# The figures that live on the statements
# --------------------------------------------------------------------------- #


def _payload(series: dict[str, list[tuple[str, str, float]]]) -> dict:
    """Company facts from (start, end, value) triples, one list per tag."""
    facts = {}
    for tag, rows in series.items():
        facts[tag] = {
            "units": {
                "USD": [
                    {
                        "start": start,
                        "end": end,
                        "val": value,
                        "form": "10-Q",
                        "fy": 2026,
                        "fp": "Q2",
                        "filed": end,
                    }
                    for start, end, value in rows
                ]
            }
        }
    return {"cik": 1, "entityName": "Fifty Three Week Inc", "facts": {"us-gaap": facts}}


def test_the_prior_year_anchor_survives_a_retired_revenue_tag():
    """Nvidia's case, and a bug this module had to route around.

    ``CompanyFacts.resolve_duration_series`` returns the **first** ladder tag
    that carries any duration facts at all. Nvidia stopped tagging
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` after fiscal 2022 and
    reports under ``Revenues`` now, but the retired tag still has facts, so a
    reader that stops at the first tag with a series gets period ends that stop
    years back, finds nothing near the target, falls back to a fixed 365 day step
    and lands one day off a 52/53 week filer's actual close. The window cannot be
    tiled and revenue growth, the Rule of 40 and the magic number all come back
    unavailable for the largest company in the sector.

    Pooling the candidate ends across every tag on the ladder fixes it. The
    fixture below is that shape in miniature: a stale tag ranked first, a live
    tag ranked third, and a period end 364 days back rather than 365.
    """
    facts = CompanyFacts(
        _payload(
            {
                "RevenueFromContractWithCustomerExcludingAssessedTax": [
                    ("2021-02-01", "2022-01-30", 26_914.0)
                ],
                "Revenues": [
                    ("2025-04-28", "2025-07-27", 46_743.0),
                    ("2025-07-28", "2025-10-26", 57_006.0),
                ],
            }
        ),
        "NVDA",
    )
    fin = SimpleNamespace(as_of=date(2026, 7, 26), revenue=302_970.0)
    # 364 days back, which is where a 52/53 week filer actually closed, and one
    # day away from where the naive step lands.
    assert C._prior_anchor(fin, facts) == date(2025, 7, 27)


def test_the_anchor_falls_back_to_the_step_when_nothing_is_near():
    facts = CompanyFacts(_payload({"Revenues": [("2019-01-01", "2019-12-31", 10.0)]}), "X")
    fin = SimpleNamespace(as_of=date(2026, 6, 30), revenue=100.0)
    assert C._prior_anchor(fin, facts) == date(2025, 6, 30)


def _quarters(tag: str, values: list[float]) -> list[tuple[str, str, float]]:
    """Four quarters to 2026-06-30 and four to 2025-06-30, newest last."""
    bounds = [
        ("2024-07-01", "2024-09-30"),
        ("2024-10-01", "2024-12-31"),
        ("2025-01-01", "2025-03-31"),
        ("2025-04-01", "2025-06-30"),
        ("2025-07-01", "2025-09-30"),
        ("2025-10-01", "2025-12-31"),
        ("2026-01-01", "2026-03-31"),
        ("2026-04-01", "2026-06-30"),
    ]
    return [(s, e, v) for (s, e), v in zip(bounds, values)]


def test_the_statement_reads_are_filed_figures_with_their_tag_and_window():
    """Prior revenue and selling spend, which the pack needs and cannot get.

    The normalized statement set is one trailing twelve month window, so it
    carries no prior period, and it has no sales and marketing line at all.
    Without both, the magic number and CAC payback never compute, and those are
    two of the metrics a software page leads with.
    """
    facts = CompanyFacts(
        _payload(
            {
                "Revenues": _quarters("Revenues", [100e6] * 4 + [130e6] * 4),
                "SellingAndMarketingExpense": _quarters(
                    "SellingAndMarketingExpense", [30e6] * 4 + [36e6] * 4
                ),
            }
        ),
        "X",
    )
    fin = SimpleNamespace(as_of=date(2026, 6, 30), revenue=520.0)
    values, rows = C._statement_kpis(fin, facts)
    assert values["revenue_prior"] == pytest.approx(400.0)
    assert values["sales_and_marketing"] == pytest.approx(144.0)
    assert values["sales_and_marketing_prior"] == pytest.approx(120.0)
    by_name = {r["name"]: r for r in rows}
    assert by_name["sales_and_marketing"]["tag_or_phrase"] == "SellingAndMarketingExpense"
    assert by_name["sales_and_marketing_prior"]["period_end"] == "2025-06-30"
    assert by_name["revenue_prior"]["source"] == "statements"


def test_sales_and_marketing_never_falls_back_to_sg_and_a():
    """SG&A is not sales and marketing, and the substitution is invisible.

    Folding general and administrative cost into the denominator makes the magic
    number smaller and the payback longer, and both stay plausible, so the error
    survives into a printed page. The ladder is one entry on purpose, and a filer
    that tags only SG&A reports no selling spend rather than a wrong one.
    """
    assert C.SALES_AND_MARKETING == ["SellingAndMarketingExpense"]
    facts = CompanyFacts(
        _payload(
            {
                "Revenues": _quarters("Revenues", [100e6] * 8),
                "SellingGeneralAndAdministrativeExpense": _quarters(
                    "SellingGeneralAndAdministrativeExpense", [50e6] * 8
                ),
            }
        ),
        "X",
    )
    fin = SimpleNamespace(as_of=date(2026, 6, 30), revenue=400.0)
    values, _rows = C._statement_kpis(fin, facts)
    assert "sales_and_marketing" not in values
    assert "revenue_prior" in values


def test_a_command_line_figure_is_validated_against_what_the_packs_read():
    values, rows = C._parse_kpi_overrides(["maintenance_capex=420"])
    assert values == {"maintenance_capex": 420.0}
    assert rows[0]["source"] == "supplied"
    with pytest.raises(ConfigError, match="not a figure any pack reads"):
        C._parse_kpi_overrides(["arpu_monthly=17"])
    with pytest.raises(ConfigError, match="not name=value"):
        C._parse_kpi_overrides(["arpu"])
    with pytest.raises(ConfigError, match="not a number"):
        C._parse_kpi_overrides(["arpu=seventeen"])


def test_a_defaulted_balance_sheet_line_is_flagged_against_every_ev_multiple():
    """T-Mobile again, in the other half of the engine.

    Its Q2 2026 10-Q tags ``ShortTermBorrowings`` and tags
    ``LongTermDebtNoncurrent`` only as of the last year end, so the staleness
    guard correctly refuses the stale figure and the default then reads long-term
    debt as zero. Enterprise value comes out around 202bn against something
    nearer 275bn and nothing else on the page says so.
    """
    fin = SimpleNamespace(
        warnings=[],
        provenance={
            "long-term debt": Provenance(
                concept="long-term debt",
                tag=None,
                method="absent, defaulted",
                note="only stale tags found (LongTermDebtNoncurrent (newest 2025-12-31)); read as zero",
            ),
            "revenue": Provenance(concept="revenue", tag="Revenues", method="ttm"),
        },
    )
    caveats = C.bridge_caveats(fin)
    assert len(caveats) == 1
    assert caveats[0].startswith("FLAG:")
    assert "read as zero" in caveats[0]
    assert "enterprise-value multiple" in caveats[0]


# --------------------------------------------------------------------------- #
# kpis
# --------------------------------------------------------------------------- #


def test_kpis_prints_the_evidence_beside_every_disclosed_figure():
    result = run("kpis", "DDOG", "--no-definitions")
    assert result.exit_code == 0
    out = result.stdout

    # Classified off Item 1 rather than off the code alone, and the pack that
    # follows is named with where the mapping came from.
    assert "infrastructure_software" in out
    assert "SUB_VERTICAL_TO_METRIC_PACK" in out

    # A tagged fact, a derived one and a prose one, each labelled as what it is.
    assert "RevenueRemainingPerformanceObligation" in out
    assert "XBRL, us-gaap" in out
    assert "derived" in out
    assert "prose" in out


def test_kpis_computes_the_pack_the_brief_asks_for():
    result = run("kpis", "DDOG", "--no-definitions")
    for metric in (
        "Net revenue retention",
        "Rule of 40 (EBITDA variant)",
        "Rule of 40 (FCF variant)",
        "Magic number",
        "CAC payback (months)",
    ):
        assert metric in result.stdout


def test_the_rule_of_40_never_appears_without_saying_which_variant():
    """One rule, three numbers, and the spread between them is the argument."""
    result = run("kpis", "DDOG")
    for variant in ("EBITDA variant", "FCF variant", "operating variant"):
        assert variant in result.stdout
    assert "Rule of 40 variant spread" in result.stdout
    # The definitions are on the page too, so no number is quoted without one.
    assert "Revenue growth plus free cash flow margin" in result.stdout


def test_a_prose_figure_carries_the_sentence_it_was_read_out_of():
    result = run("kpis", "DDOG", "--no-definitions")
    assert "net retention rate was about 120%" in result.stdout
    # And the hedge is on the page, because "about 120%" is the company's
    # estimate rather than a measurement.
    assert "prose, hedged" in result.stdout


def test_a_refused_figure_is_shown_and_withheld():
    result = run("kpis", "DDOG", "--no-definitions")
    assert "refused" in result.stdout
    # ARR was refused, so the pack reports no ARR and no EV/ARR rather than
    # carrying a hundred thousand dollars through as a hundred thousand million.
    assert "needs kpis['arr']" in result.stdout


def test_absence_is_reported_as_a_finding():
    result = run("kpis", "DDOG", "--no-definitions")
    assert "Looked for and not disclosed" in result.stdout
    assert "gross_revenue_retention" in result.stdout


def test_kpis_refuses_a_bucket_with_no_pack_and_still_prints_the_statements():
    """The refusal is the answer, and it does not take the whole page with it.

    Both packs go quiet together and for the same reason: ``kpis.METRIC_PACKS``
    looks for no sector disclosure on a chip maker and this map computes no
    sector metrics on one, because neither ARR nor the Rule of 40 is how that
    company is quoted. What still runs is the read off the filed statements,
    which is true of any filer, and the refusal names what is missing instead of
    scoring the company on somebody else's operating metrics.
    """
    result = run("kpis", "DDOG", "--sub-vertical", "semiconductors")
    assert result.exit_code == 0
    assert "No computed metric pack" in result.stdout
    assert "book to bill" in result.stdout
    assert "revenue_prior" in result.stdout
    assert "Rule of 40 (EBITDA variant)" not in result.stdout
    assert "Computed operating metrics" not in result.stdout


def test_a_forced_pack_lets_a_refused_bucket_through_deliberately():
    result = run("kpis", "DDOG", "--sub-vertical", "hardware", "--metric-pack", "software")
    assert result.exit_code == 0
    assert "forced with --metric-pack" in result.stdout
    assert "Rule of 40 (EBITDA variant)" in result.stdout


def test_the_bridge_prints_one_line_per_figure_that_crossed_or_did_not():
    result = run("kpis", "DDOG", "--no-definitions")
    assert "How the disclosed figures reached the pack" in result.stdout
    assert "kpis['nrr']" in result.stdout
    assert "reaches no computed metric" in result.stdout


def test_a_supplied_figure_is_labelled_as_typed_rather_than_read():
    result = run("kpis", "DDOG", "--no-definitions", "--kpi", "arr", "--kpi", "x=1")
    assert result.exit_code == 1

    result = run("kpis", "DDOG", "--no-definitions", "--kpi", "arr=3000")
    assert result.exit_code == 0
    assert "command line" in result.stdout
    assert "EV / ARR" in result.stdout


def test_skipping_the_text_gives_up_the_prose_metrics_and_says_so():
    result = run("kpis", "DDOG", "--no-definitions", "--no-text")
    assert result.exit_code == 0
    assert "Item 7 skipped with --no-text" in result.stdout
    assert "net retention rate was about 120%" not in result.stdout


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #


def test_segments_shows_the_parts_against_the_whole_and_the_gap():
    result = run("segments", "DIS")
    assert result.exit_code == 0
    out = result.stdout
    for segment in ("Entertainment", "Experiences", "Sports"):
        assert segment in out
    # The three segments, the consolidated total, and the eliminated difference
    # between them. A table that does not foot is the single most common way
    # segment analysis goes wrong.
    assert "96,294" in out
    assert "94,425" in out
    assert "(1,869)" in out
    assert "Reconciliation" in out


def test_segments_states_whether_it_foots_and_against_what_tolerance():
    result = run("segments", "DIS")
    assert "Within 2% tolerance" in result.stdout
    assert "foots" in result.stdout


def test_segment_margins_are_the_reason_the_command_exists():
    """The consolidated margin describes none of the businesses inside it."""
    result = run("segments", "DIS")
    assert "27.6%" in result.stdout  # Experiences
    assert "11.0%" in result.stdout  # Entertainment


def test_the_geography_cut_says_why_it_has_no_margin_column():
    result = run("segments", "DIS")
    out = result.stdout
    assert "Americas" in out and "Asia Pacific" in out
    assert "ASC 280-10-50-41" in out
    assert "Long-lived assets" in out


def test_the_geography_cut_can_be_left_off():
    result = run("segments", "DIS", "--no-geography")
    assert "Americas" not in result.stdout


def test_concentration_is_read_rather_than_printed():
    result = run("segments", "DIS")
    assert "Herfindahl" in result.stdout
    assert "a genuine conglomerate" in result.stdout


def test_a_single_segment_filer_is_an_answer_rather_than_a_failure():
    result = run("segments", "ZS")
    assert result.exit_code == 0
    assert "Herfindahl" in result.stdout
    assert "1.000" in result.stdout
    assert "one business with a rounding error" in result.stdout


# --------------------------------------------------------------------------- #
# sotp
# --------------------------------------------------------------------------- #

PLAN = """
segments:
  Entertainment:
    metric: ebitda
    multiple: 9.0
    source: US diversified media peers, spot EV/EBITDA
  Experiences:
    metric: ebitda
    multiple: 13.0
    source: listed regional park operators at a premium for pricing power
  Sports:
    metric: ebitda
    multiple: 7.0
    source: a discount for the affiliate fee runoff
corporate:
  cost: 1100.0
conglomerate_discount: 0.0
"""

# The segments come out of the fiscal 2025 annual report and the consolidated
# figures out of the trailing twelve months, so the run has to be pinned to the
# annual report or the coverage check refuses on growth rather than on a missing
# segment. That refusal is itself tested below.
ANNUAL = "2025-11-14"


def sotp(tmp_path: Path, plan: str = PLAN, *args: str, as_of: str = ANNUAL):
    path = tmp_path / "plan.yaml"
    path.write_text(plan)
    result = runner.invoke(
        C.app, ["sotp", "DIS", "--plan", str(path), "--as-of", as_of, *args]
    )
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


def test_sotp_refuses_without_a_plan_and_names_the_segments_to_price():
    result = run("sotp", "DIS")
    assert result.exit_code == 1
    assert "No plan supplied" in result.stdout
    assert "Entertainment, Experiences, Sports" in result.stdout
    assert "--emit-plan" in result.stdout


def test_the_skeleton_carries_the_filers_own_segment_names(tmp_path):
    path = tmp_path / "skeleton.yaml"
    result = run("sotp", "DIS", "--emit-plan", str(path))
    assert result.exit_code == 0
    written = path.read_text()
    for segment in ("Entertainment", "Experiences", "Sports"):
        assert f"  {segment}:" in written
    # No default multiple ships in the skeleton. A default here is a number
    # nobody chose turning up inside a valuation.
    assert "multiple: 0.0" in written
    assert 'source: ""' in written


def test_a_multiple_without_a_source_is_refused(tmp_path):
    result = sotp(
        tmp_path,
        "segments:\n  Entertainment:\n    metric: ebitda\n    multiple: 9.0\n",
    )
    assert result.exit_code == 1
    assert "missing: source" in result.stdout


def test_a_plan_with_no_segments_is_refused(tmp_path):
    result = sotp(tmp_path, "corporate:\n  cost: 10\n")
    assert result.exit_code == 1
    assert "carries no 'segments' mapping" in result.stdout


def test_sotp_shows_the_multiple_its_source_and_the_value_it_bought(tmp_path):
    result = sotp(tmp_path)
    assert result.exit_code == 0
    out = result.stdout
    assert "Where each multiple came from" in out
    assert "listed regional park operators" in out
    # Experiences EBITDA is 9,995 of operating income plus 2,823 of D&A, at 13x.
    assert "12,818" in out
    assert "166,634" in out


def test_sotp_shows_the_corporate_drag_and_how_it_was_allocated(tmp_path):
    """A sum of the parts that hides this is an argument for a chosen answer."""
    result = sotp(tmp_path)
    out = result.stdout
    assert "The holding company" in out
    assert "1,100" in out
    assert "enterprise-value-weighted average of the segment earnings multiples" in out
    assert "(12,808)" in out


def test_sotp_without_a_corporate_cost_says_the_total_is_overstated(tmp_path):
    plan = PLAN.replace("corporate:\n  cost: 1100.0\n", "")
    result = sotp(tmp_path, plan)
    assert result.exit_code == 0
    assert "not supplied" in result.stdout
    assert "values the holding company's own overhead at zero" in result.stdout


def test_the_bridge_walks_from_the_parts_to_a_price_per_share(tmp_path):
    result = sotp(tmp_path)
    out = result.stdout
    for line in (
        "Gross value of the parts",
        "Capitalised corporate cost",
        "Conglomerate discount",
        "Enterprise value, sum of the parts",
        "Equity value",
        "Value per share",
        "Enterprise value, market",
        "Gap per share",
    ):
        assert line in out


def test_the_conglomerate_question_is_reported_honestly(tmp_path):
    result = sotp(tmp_path)
    out = result.stdout
    assert "not automatically an opportunity" in out
    assert "break-up case rather than the status quo" in out
    assert "Lang and Stulz" in out


def test_a_discount_can_be_applied_and_is_argued_with_when_it_is(tmp_path):
    result = sotp(tmp_path, PLAN, "--discount", "0.10")
    assert result.exit_code == 0
    assert "10.0% conglomerate discount" in result.stdout
    assert "tuned until the sum of the parts agrees" in result.stdout


def test_a_zero_discount_prints_as_zero_rather_than_as_minus_zero(tmp_path):
    """The module stores deductions negative, so an unused line arrives as -0.0.

    Printed raw that reads as a rounding artefact on a line that was in fact not
    used at all.
    """
    assert C._money(-0.0) == "0"
    result = sotp(tmp_path)
    row = next(
        line for line in result.stdout.splitlines() if "Conglomerate discount" in line
    )
    assert row.split()[-1] == "0"


def test_the_period_mismatch_is_named_before_the_coverage_check_refuses(tmp_path):
    """The parts are annual and the whole is trailing twelve months.

    Disney's segments cover the year to 2025-09-27 and its consolidated revenue
    covers the twelve months to 2026-06-27, so the segments come to 97.4% of the
    company and ``run_sotp`` refuses. That refusal is right: a sum of the parts
    over segments that do not cover the company values the missing part at zero.
    What the command adds is saying which two periods are involved and how to
    pin them together, before the refusal lands.
    """
    result = sotp(tmp_path, PLAN, as_of="2026-09-10")
    assert result.exit_code == 1
    out = result.stdout
    assert "the segments cover the year ended 2025-09-27" in out
    assert "--as-of 2025-09-27" in out
    assert "97.4% of the company" in out


def test_the_cross_checks_name_the_wedge_between_revenue_and_value(tmp_path):
    """The mix is the valuation, and the check states it in one line."""
    result = sotp(tmp_path)
    assert "Largest segment 'Experiences'" in result.stdout
    assert "the consolidated multiple this company trades on is the wrong multiple" in (
        result.stdout
    )
