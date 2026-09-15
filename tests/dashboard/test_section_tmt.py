"""The TMT section of the dashboard: its shaping, its wiring, and the numbers it pins.

Three kinds of test, and none of them fits a model.

The shaping tests call the pure functions in ``sections/tmt.py`` with small
fakes, so what they assert is how module results become figures: every figure
has a kind something on the page can draw, every value is finite, every mark has
a label, the takeaway says the numbers it was given, and a figure that cannot be
built turns into a refusal with a reason rather than an empty chart.

One wiring test runs the collector against the committed fixtures, which reads
filings and does arithmetic but fits nothing and takes about a second. It is
here because the provenance and refusal plumbing only fails for real inputs.

The pinned truths read the committed snapshot and hold the page to the figures
previous audited runs reproduced. The snapshot is committed after every section
lands, so until then they skip.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from techval.dashboard.collect import CollectContext, collect_section
from techval.dashboard.sections import section_module
from techval.dashboard.sections import tmt as T
from techval.dashboard.snapshot import to_jsonable, validate_section

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"
KIT = ROOT / "src" / "techval" / "dashboard" / "assets" / "kit.js"
RENDERER = ROOT / "src" / "techval" / "dashboard" / "assets" / "sections" / "tmt.js"
# Spelled as a code point, so this file obeys the rule it checks.
EM_DASH = chr(0x2014)

COLUMNS = (
    ("ev_revenue", "EV/Revenue"),
    ("ev_ebitda", "EV/EBITDA"),
    ("premium_1d", "Premium to 1-day"),
    ("premium_30d", "Premium to 30-day"),
)


def _kit_charts() -> set[str]:
    source = KIT.read_text(encoding="utf-8")
    block = re.search(r"TV\.charts = \{(.*?)\};", source, flags=re.S).group(1)
    return set(re.findall(r"(\w+)\s*:", block))


def _renderer_tables() -> set[str]:
    source = RENDERER.read_text(encoding="utf-8")
    block = re.search(r"var TABLES = \{(.*?)\};", source, flags=re.S).group(1)
    return set(re.findall(r"(\w+)\s*:", block))


def _txn(ticker, **over):
    base = dict(
        target_ticker=ticker,
        target_name=f"{ticker} Inc.",
        acquirer_name=f"Buyer of {ticker}",
        announced=date(2026, 6, 1),
        status="pending",
        closed=None,
        consideration="cash",
        offer_price=50.0,
        cash_per_share=50.0,
        premium_1d=0.30,
        premium_30d=0.32,
        ev_revenue=4.0,
        sub_vertical="internet",
        notes=[],
        flags=[],
    )
    base.update(over)
    one, thirty = base["premium_1d"], base["premium_30d"]
    base["premium_gap_points"] = (
        (one - thirty) * 100.0 if one is not None and thirty is not None else None
    )
    return SimpleNamespace(**base)


def _stats(**counts):
    """Statistic rows per column: ``n`` from the keyword, a median where n clears five."""
    out = {}
    for _attr, label in COLUMNS:
        n = counts.get(label.replace("/", "_").replace(" ", "_").replace("-", "_"), 0)
        out[label] = {"n": float(n), "Median": 5.5 if n >= 5 else float("nan")}
    return out


def _shape(transactions, stats=None, **kw):
    return T.shape_precedents(
        transactions,
        stats if stats is not None else _stats(EV_Revenue=5),
        min_deals=5,
        leak_points=5.0,
        window_days=30,
        columns=COLUMNS,
        sub_vertical_stats=kw.pop("sub_vertical_stats", ()),
        **kw,
    )


def _walk_numbers(value, path="data"):
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _walk_numbers(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _walk_numbers(v, f"{path}[{i}]")


def _assert_well_formed(figures):
    kit, tables = _kit_charts(), _renderer_tables()
    for fid, fig in figures.items():
        assert fig["kind"] in kit or (fig["kind"] == "table" and fid in tables), fid
        assert isinstance(fig["title"], str) and isinstance(fig["subtitle"], str)
        if fig["kind"] != "tiles":
            assert fig["title"].strip(), f"{fid} has no title"
        # The snapshot writes NaN as null, which the kit draws as a missing mark,
        # so a non-finite value reaching a figure is a gap nobody chose.
        for path, number in _walk_numbers(fig["data"], fid):
            assert math.isfinite(number), path
        data = fig["data"]
        for row in data.get("rows", []):
            if fig["kind"] in ("dot", "hbar"):
                assert str(row["label"]).strip(), fid
        for step in data.get("steps", []):
            assert str(step["label"]).strip(), fid
        for tile in data.get("tiles", []):
            assert str(tile["label"]).strip(), fid
        assert EM_DASH not in json.dumps(fig, ensure_ascii=False), f"{fid} carries an em-dash"


# --------------------------------------------------------------------------- #
# shaping: precedents
# --------------------------------------------------------------------------- #


def test_precedent_figures_are_well_formed_and_the_refusals_have_reasons():
    deals = [
        _txn("AAA", premium_1d=0.10, premium_30d=0.35),
        _txn("BBB"),
        _txn("CCC", premium_1d=None, premium_30d=None, flags=[
            "CCC: no price series, so no unaffected price and no premium."
        ]),
        _txn("DDD", offer_price=None, premium_1d=None, premium_30d=None, ev_revenue=None,
             consideration="mixed", cash_per_share=27.0,
             notes=["the agreement states 0.2400, 0.4000 as alternative exchange ratios, which is a collar"],
             flags=["DDD: no offer price could be fixed, so no deal size and no multiple were computed"]),
        _txn("EEE", ev_revenue=None, flags=[
            "EEE: the target's financials could not be built (net income\n  hint: absent), "
            "so the deal size and the multiples are not reported"
        ]),
    ]
    part = _shape(deals, _stats(EV_Revenue=5, EV_EBITDA=2, Premium_to_1_day=3, Premium_to_30_day=3))
    figures = part["figures"]
    assert set(figures) == {"precedent_tiles", "premia", "ev_revenue", "deals"}
    _assert_well_formed(to_jsonable(figures))

    whats = {r["what"]: r["why"] for r in part["refusals"]}
    assert "0.2400, 0.4000" in whats["DDD offer price"]
    assert "27.00" in whats["DDD offer price"], "the cash leg is named as the cash leg"
    assert whats["Premia for CCC"].startswith("No price series")
    assert "\n" not in whats["EV/Revenue for EEE"], "multi-line module text is flattened"
    assert whats["EV/EBITDA statistic"].startswith("2 of 5 deals")
    assert "below the 5-deal floor" in whats["Premium to 1-day statistic"]
    assert "Statistics by sub-vertical" in whats
    for why in whats.values():
        assert why.strip() and not why[0].islower() and why.endswith((".", ")"))

    # Every refusal is named under the figure it concerns, so it hangs on that card.
    attached = [w for f in figures.values() for w in f["data"].get("refusals", [])]
    assert sorted(attached) == sorted(whats)
    assert "DDD offer price" in figures["deals"]["data"]["refusals"]
    assert "Premia for CCC" in figures["premia"]["data"]["refusals"]


def test_the_premia_figure_counts_the_flagged_deals_and_keeps_the_comp_rule():
    deals = [
        _txn("AAA", premium_1d=0.10, premium_30d=0.35),  # -25 points
        _txn("BBB", premium_1d=0.69, premium_30d=0.59),  # +10 points
        _txn("CCC", premium_1d=0.30, premium_30d=0.31),  # -1 point
    ]
    part = _shape(deals)
    premia = part["figures"]["premia"]
    assert premia["title"].startswith("2 of 3 premia move more than 5 points")
    assert "not a trading comp" in premia["subtitle"]
    assert [r["label"] for r in premia["data"]["rows"]] == ["AAA", "CCC", "BBB"]
    assert [r["flagged"] for r in premia["data"]["rows"]] == [True, False, True]
    assert [s["role"] for s in premia["data"]["series"]] == ["model", "alt"]
    assert part["summary"]["widest"] == {"ticker": "AAA", "one_day": 0.10, "thirty_day": 0.35}


def test_no_measured_premium_is_a_refusal_not_an_empty_chart():
    deals = [_txn("AAA", premium_1d=None, premium_30d=None), _txn("BBB", premium_1d=None, premium_30d=None)]
    part = _shape(deals)
    assert "premia" not in part["figures"]
    refusal = next(r for r in part["refusals"] if r["what"] == "Premia")
    assert "none of the 2 deals" in refusal["why"].lower()
    assert part["summary"]["widest"] is None


def test_a_median_below_the_floor_is_refused_on_the_tile_and_the_title():
    part = _shape([_txn("AAA"), _txn("BBB")], _stats(EV_Revenue=2))
    tile = part["figures"]["precedent_tiles"]["data"]["tiles"][3]
    assert tile["value"] is None and tile["status"] == "refused"
    assert "too few for a median" in part["figures"]["ev_revenue"]["title"]
    assert "median" not in part["figures"]["ev_revenue"]["data"] or (
        part["figures"]["ev_revenue"]["data"]["median"] is None
    )


def test_an_empty_precedent_set_refuses_the_read():
    with pytest.raises(T.NothingToShow):
        _shape([])


# --------------------------------------------------------------------------- #
# shaping: segments and the sum of the parts
# --------------------------------------------------------------------------- #


def _segment(name, revenue, income, da, share):
    return SimpleNamespace(
        name=name,
        revenue=revenue,
        operating_income=income,
        da=da,
        margin=income / revenue,
        revenue_share=share,
    )


def _report(segments, consolidated, **over):
    base = dict(
        ticker="TEST",
        as_of=date(2025, 12, 31),
        accession="0000000000-26-000001",
        segments=segments,
        consolidated_revenue=consolidated,
        unallocated=consolidated - sum(s.revenue for s in segments),
        reconciles=True,
        notes=[],
    )
    base.update(over)
    return SimpleNamespace(**base)


SOTP_REASON = "no multiple supplied for segment 'Cable'. Every segment needs one."


def _shape_segments(report, **over):
    kw = dict(
        operating_total=190.0,
        corporate_tags=["Depreciation"],
        sotp_reason=SOTP_REASON,
        price=(42.5, date(2026, 9, 10)),
        tolerance=0.02,
    )
    kw.update(over)
    return T.shape_segments(report, **kw)


def _two_segments(**over):
    return _report(
        [_segment("Cable", 600.0, 150.0, 90.0, 0.6), _segment("Studios", 420.0, 40.0, 10.0, 0.42)],
        1000.0,
        **over,
    )


def test_segment_revenue_bridges_to_consolidated_revenue_and_names_the_line_that_explains_it():
    report = _two_segments(
        notes=["The residual is the Intersegment Eliminations the filer tags, to the dollar, so no segment is missing."]
    )
    part = _shape_segments(report)
    figures = part["figures"]
    _assert_well_formed(to_jsonable(figures))
    assert set(figures) == {"segment_revenue", "segment_operating_income", "segment_margins"}

    bridge = figures["segment_revenue"]["data"]
    assert [(s["label"], s["value"]) for s in bridge["steps"]] == [
        ("Cable", 600.0),
        ("Studios", 420.0),
        ("Intersegment Eliminations", -20.0),
    ]
    assert bridge["total"]["value"] == 1000.0
    assert sum(s["value"] for s in bridge["steps"]) == pytest.approx(bridge["total"]["value"])
    assert figures["segment_revenue"]["title"] == (
        "Segment revenue overshoots consolidated revenue by 20, exactly the "
        "Intersegment Eliminations line the filer tags"
    )
    assert "inside the 2% tolerance" in figures["segment_revenue"]["subtitle"]


def test_operating_income_shows_the_gap_to_the_company_total_on_its_own_axis():
    part = _shape_segments(_two_segments(), operating_total=170.0)
    income = part["figures"]["segment_operating_income"]
    assert income["kind"] == "waterfall", "never a second axis on the revenue chart"
    steps = income["data"]["steps"]
    assert [(s["label"], s["value"]) for s in steps] == [
        ("Cable", 150.0),
        ("Studios", 40.0),
        ("Outside the segments", -20.0),
    ]
    assert income["data"]["total"]["value"] == 170.0
    assert sum(s["value"] for s in steps) == pytest.approx(170.0)
    assert income["title"] == (
        "Items outside the segments take 20 off segment operating income of 190"
    )


def test_a_company_total_equal_to_the_segment_sum_is_called_the_segment_total():
    """Disney's case: the undimensioned figure is the segments' own sum, to the dollar."""
    part = _shape_segments(_two_segments(), operating_total=190.0)
    income = part["figures"]["segment_operating_income"]
    gap = next(s for s in income["data"]["steps"] if s["label"] == "Outside the segments")
    assert gap["value"] == 0.0, "the gap is drawn even when it is zero"
    assert "is the segments' own sum" in income["title"]
    assert "says nothing about its size" in income["subtitle"]


def test_no_company_operating_total_is_a_refusal_not_a_total_made_up():
    part = _shape_segments(_two_segments(), operating_total=None)
    income = part["figures"]["segment_operating_income"]
    assert income["data"]["total"] == {"label": "Sum of the segments", "value": 190.0}
    assert all(s["label"] != "Outside the segments" for s in income["data"]["steps"])
    whats = {r["what"]: r["why"] for r in part["refusals"]}
    assert "no undimensioned OperatingIncomeLoss" in whats["Operating income reconciliation"]
    assert "Operating income reconciliation" in income["data"]["refusals"]


def test_margin_tiles_state_the_spread_between_the_segments():
    part = _shape_segments(_two_segments())
    tiles = part["figures"]["segment_margins"]
    assert tiles["title"] == "Cable earns a 25.0% operating margin, 2.6 times Studios' 9.5%"
    values = {t["label"]: t["value"] for t in tiles["data"]["tiles"]}
    assert values["Cable"] == pytest.approx(0.25)
    assert values["All segments"] == pytest.approx(190.0 / 1020.0)
    assert part["summary"]["top_margin"] == ("Cable", 0.25)


def test_the_sum_of_the_parts_is_refused_in_the_modules_words_with_what_the_fixtures_hold():
    part = _shape_segments(_two_segments())
    (refusal,) = part["refusals"]
    assert refusal["what"] == "Sum of the parts"
    why = refusal["why"]
    assert SOTP_REASON in why, "the module's own refusal, case kept"
    assert "240 in Cable and 50 in Studios" in why, "the EBITDA a plan would price"
    assert "carries only Depreciation" in why
    assert "42.50, TEST's close on 2026-09-10" in why
    assert part["figures"]["segment_operating_income"]["data"]["refusals"] == ["Sum of the parts"]

    no_price = _shape_segments(_two_segments(), price=None, corporate_tags=[])
    why = no_price["refusals"][0]["why"]
    assert "No TEST close is committed" in why
    assert "nothing is tagged against a corporate member" in why


def test_a_report_with_no_segment_revenue_refuses_the_read():
    report = _report([], 1000.0, unallocated=1000.0)
    with pytest.raises(T.NothingToShow):
        _shape_segments(report)


# --------------------------------------------------------------------------- #
# shaping: KPIs and the takeaway
# --------------------------------------------------------------------------- #


def _kpi_row(name, value, unit, source, confidence, evidence, notes="", period="2026-06-30"):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "period_end": period,
        "source": source,
        "tag_or_phrase": f"{name} tag",
        "confidence": confidence,
        "notes": notes,
        "evidence": evidence,
    }


def test_a_refused_kpi_stays_a_row_with_its_reason():
    filers = [
        {
            "ticker": "AAA",
            "rows": [
                _kpi_row("rpo", 100.0, "usd_mm", "xbrl_extension", 1.0, "XBRL, us-gaap"),
                _kpi_row("arr", 0.0, "usd_mm", "text", 0.0, "prose", notes="no scale word", period="2026-09-11"),
            ],
            "missing": 7,
            "instance_accession": "0000000000-26-000002",
            "instance_filed": "2026-08-01",
            "text_accession": "0000000000-26-000003",
            "text_period": "2025-12-31",
        },
        {"ticker": "BBB", "rows": [], "missing": 9},
    ]
    part = T.shape_kpis(filers)
    _assert_well_formed(to_jsonable(part["figures"]))
    rows = part["figures"]["kpis"]["data"]["rows"]
    refused = next(r for r in rows if r["metric"] == "arr")
    assert refused["refused"] is True
    assert refused["reason"] == "No scale word."
    assert refused["period_stated"] is False
    assert refused["period_end"] is None, "the read date is not passed off as a period end"
    assert refused["read_from"] == "10-K 0000000000-26-000003, year to 2025-12-31"
    assert next(r for r in rows if r["metric"] == "rpo")["reason"] is None
    assert part["figures"]["kpis"]["title"] == (
        "1 of 2 operating figures are refused, 1 of them among the 1 read from prose"
    )
    assert [s["missing"] for s in part["figures"]["kpis"]["data"]["sources"]] == [7, 9]
    assert any(r["what"] == "Disney subscriber count" for r in part["refusals"])


def test_no_kpi_anywhere_refuses_the_read():
    with pytest.raises(T.NothingToShow):
        T.shape_kpis([{"ticker": "AAA", "rows": [], "missing": 3}])


def test_the_takeaway_says_the_numbers_it_was_given():
    text = T.build_takeaway(
        {"n_deals": 7, "n_both": 6, "n_flagged": 4, "leak_points": 5.0,
         "widest": {"ticker": "QQQ", "one_day": 0.123, "thirty_day": 0.456}},
        {"ticker": "XYZ", "n_segments": 3, "allocated": 1234.0, "consolidated": 1200.0,
         "residual": -34.0, "explained": "Eliminations",
         "top_margin": ("Parks", 0.276), "low_margin": ("Film", 0.11)},
        {"n_total": 11, "n_refused": 2, "n_tagged": 5, "n_prose": 3, "n_prose_refused": 1},
    )
    for fragment in (
        "4 of the 6", "QQQ", "12.3%", "45.6%",
        "overshoots consolidated revenue by 34, exactly the Eliminations line",
        "Parks earns a 27.6% margin against Film's 11.0%",
        "Of 11", "2 are refused, 1 of them among the 3 read from prose",
    ):
        assert fragment in text, fragment
    assert EM_DASH not in text
    # A read that was refused leaves its sentence out rather than inventing one.
    assert T.build_takeaway(None, None, {"n_total": 3, "n_refused": 0, "n_tagged": 3}).startswith("Of 3")
    changed = T.build_takeaway(
        None,
        {"ticker": "XYZ", "n_segments": 2, "allocated": 1000.0, "consolidated": 1000.0,
         "residual": 0.0, "explained": None, "top_margin": None, "low_margin": None},
        None,
    )
    assert "foots exactly" in changed and "margin" not in changed


def test_the_renderer_draws_every_figure_id_the_collector_makes_and_carries_no_em_dash():
    source = RENDERER.read_text(encoding="utf-8")
    assert EM_DASH not in source
    ids = set(T.PRECEDENT_FIGURES) | {"segment_revenue", "segment_operating_income", "segment_margins", "kpis"}
    for fid in ids:
        assert f'"{fid}"' in source, f"{fid} is not placed by tmt.js"


# --------------------------------------------------------------------------- #
# wiring, against the committed fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def collected():
    ctx = CollectContext(FIXTURES, ROOT)
    section, run = collect_section(section_module("tmt"), ctx, use_cache=False)
    return section


def test_the_collector_produces_a_valid_section_with_every_figure_sourced(collected):
    validate_section(collected)
    assert collected["status"] == "ok"
    assert collected["headline"] is None
    assert set(collected["figures"]) == {
        "precedent_tiles", "premia", "ev_revenue", "deals",
        "segment_revenue", "segment_operating_income", "segment_margins", "kpis",
    }
    _assert_well_formed(collected["figures"])
    entry_points = {row["figure"]: row["entry_point"] for row in collected["provenance"]}
    assert entry_points["premia"] == "techval.tmt.precedents.build_precedents"
    assert entry_points["segment_revenue"] == "techval.tmt.segments.build_segments"
    assert entry_points["kpis"] == "techval.tmt.kpis.build_kpis"
    names = {r["what"] for r in collected["refusals"]}
    attached = {w for f in collected["figures"].values() for w in f["data"].get("refusals", [])}
    assert names == attached, "every refusal hangs under a figure"
    assert "prices/DIS.csv" in " ".join(
        i for row in collected["provenance"] if row["figure"] == "segment_revenue" for i in row["inputs"]
    )


def test_the_splunk_multiple_counts_the_current_debt_on_file_at_the_announcement(collected):
    """7.1x from the fixtures, which is what the live filings give.

    ``companyfacts_SPLK.json`` was pruned to the ladders as they stood before
    ``DebtCurrent`` joined them, so it lacked the 776.456mm Splunk tagged at
    2023-07-31 and the multiple came out 26,510 / 3,843.0 = 6.90x. Re-pruned by
    ``tests/fixtures/merger/record_companyfacts.py`` it is 27,286.8 / 3,843.0.
    """
    splk = next(r for r in collected["figures"]["deals"]["data"]["rows"] if r["ticker"] == "SPLK")
    assert splk["ev_revenue"] == pytest.approx(27_286.778 / 3_842.967, abs=5e-4)


def test_the_sum_of_the_parts_refusal_is_what_the_module_itself_says(collected):
    """Not a paraphrase: the collector asks value_segments and quotes the answer."""
    from techval.config import Assumptions
    from techval.errors import TechvalError
    from techval.tmt.sotp import value_segments

    segments = [SimpleNamespace(name=n, operating_income=1.0, revenue=1.0) for n in
                ("Entertainment", "Experiences", "Sports")]
    with pytest.raises(TechvalError) as exc:
        value_segments(segments, {}, Assumptions(), ticker="DIS")
    why = next(r["why"] for r in collected["refusals"] if r["what"] == "Sum of the parts")
    assert str(exc.value).split(".")[0] in why


# --------------------------------------------------------------------------- #
# pinned truths, against the committed snapshot
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def snapshot_tmt():
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    section = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"].get("tmt")
    if section is None or section["status"] != "ok":
        pytest.skip("the committed snapshot carries no collected tmt section")
    return section


def _deal(section, ticker):
    return next(r for r in section["figures"]["deals"]["data"]["rows"] if r["ticker"] == ticker)


def _refusal(section, what):
    return next(r["why"] for r in section["refusals"] if r["what"] == what)


def test_pinned_deal_prices_and_the_splunk_multiple(snapshot_tmt):
    splk = _deal(snapshot_tmt, "SPLK")
    assert splk["offer_price"] == pytest.approx(157.00, abs=0.005)
    assert splk["status"] == "completed"
    assert splk["acquirer"].startswith("Cisco")
    # 7.1x: 27,287 of enterprise value over 3,843.0 of trailing revenue.
    # Splunk's 10-Q for the quarter to 2023-07-31 reached EDGAR on 2023-08-24,
    # before the 2023-09-21 announcement, so the balance sheet and the trailing
    # twelve months both run to July: 3,653.7 for the year to January less
    # 1,472.8 for its first half plus 1,662.1 for the new one is 3,843.0. The
    # same 10-Q tags 776.5 of DebtCurrent, and that is the whole of the gap to
    # the 6.9x this page once printed: the fixture had been pruned before
    # DebtCurrent joined the current-debt ladder, so it carried none of it and
    # enterprise value read 26,510. The revenue was never the difference.
    assert round(splk["ev_revenue"], 1) == 7.1
    assert _deal(snapshot_tmt, "SLAB")["offer_price"] == pytest.approx(231.00, abs=0.005)
    assert _deal(snapshot_tmt, "IRDM")["offer_price"] is None
    assert "collar" in _refusal(snapshot_tmt, "IRDM offer price")


def test_pinned_premia_and_the_gaps_between_them(snapshot_tmt):
    rows = {r["label"]: r for r in snapshot_tmt["figures"]["premia"]["data"]["rows"]}
    roku = rows["ROKU"]["values"]
    assert round(roku["one_day"] * 100, 1) == 11.3
    assert round(roku["thirty_day"] * 100, 1) == 27.0
    payo = rows["PAYO"]["values"]
    assert round((payo["one_day"] - payo["thirty_day"]) * 100, 1) == -27.6
    assert round(rows["PAYO"]["gap_points"], 1) == -27.6
    ramp = rows["RAMP"]["values"]
    assert abs(ramp["one_day"] - ramp["thirty_day"]) * 100 < 1.0
    assert rows["ROKU"]["flagged"] and rows["PAYO"]["flagged"] and not rows["RAMP"]["flagged"]


def test_pinned_statistics_refuse_below_five_deals(snapshot_tmt):
    for column in ("EV/EBITDA", "Premium to 1-day", "Premium to 30-day"):
        assert "below the 5-deal floor" in _refusal(snapshot_tmt, f"{column} statistic")
    assert snapshot_tmt["figures"]["ev_revenue"]["data"]["median"] is not None


def test_pinned_disney_segments_and_the_refused_sum_of_the_parts(snapshot_tmt):
    steps = snapshot_tmt["figures"]["segment_revenue"]["data"]["steps"]
    assert [(s["label"], s["value"]) for s in steps] == [
        ("Entertainment", 42466.0),
        ("Experiences", 36156.0),
        ("Sports", 17672.0),
        ("Eliminations And Other", -1869.0),
    ]
    assert snapshot_tmt["figures"]["segment_revenue"]["data"]["total"]["value"] == 94425.0
    income = snapshot_tmt["figures"]["segment_operating_income"]["data"]
    assert [(s["label"], s["value"]) for s in income["steps"]] == [
        ("Entertainment", 4674.0),
        ("Experiences", 9995.0),
        ("Sports", 2882.0),
        ("Outside the segments", 0.0),
    ]
    assert income["total"]["value"] == 17551.0
    sotp = _refusal(snapshot_tmt, "Sum of the parts")
    assert "no multiple supplied for segment 'Entertainment'" in sotp
    assert "12,818 in Experiences" in sotp
    assert "105.82, DIS's close on 2026-09-10" in sotp


def test_pinned_kpis_and_their_refusals(snapshot_tmt):
    rows = {(r["ticker"], r["metric"]): r for r in snapshot_tmt["figures"]["kpis"]["data"]["rows"]}
    assert rows[("DDOG", "rpo")]["value"] == pytest.approx(3471.4)
    assert rows[("DDOG", "rpo")]["refused"] is False
    assert rows[("NFLX", "content_spend")]["value"] == pytest.approx(9774.44)
    assert rows[("TMUS", "rpo")]["refused"] is True
    assert "3 different undimensioned values" in rows[("TMUS", "rpo")]["reason"]
    assert "not reproduced" in _refusal(snapshot_tmt, "Disney subscriber count")
