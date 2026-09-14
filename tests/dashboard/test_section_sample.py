"""The sample section: counts read from the other sections, and refused when they are not there.

Two groups. The first hands ``read`` and ``shape`` a small fake of the results
the model sections publish, with values chosen to be unmistakable, so it
collects nothing and runs in milliseconds: every count reaches the figure
unchanged and names where it was read, a count that is absent, not a number or
stated in a reworded sentence becomes a refusal naming the field, and the words
move with the numbers.

The second reads the committed snapshot. It pins the counts the section was
audited at, and checks that the committed section is exactly what ``shape``
makes of the committed model sections beside it, so the page cannot drift from
the results it quotes. The snapshot is committed after every section lands, so
until then those tests skip.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from techval.dashboard import collect as C
from techval.dashboard.collect import CollectContext, entry_module
from techval.dashboard.sections import sample
from techval.dashboard.sections.sample import read, shape
from techval.dashboard.snapshot import to_jsonable, validate_section

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"


# --------------------------------------------------------------------------- #
# fakes, in the shapes the model sections publish
# --------------------------------------------------------------------------- #


def _section(sid: str, title: str, figures: dict, headline: dict | None = None, refusals=None) -> dict:
    return {
        "id": sid,
        "title": title,
        "takeaway": "",
        "status": "ok",
        "refusals": refusals or [],
        "headline": headline,
        "figures": {fid: {"kind": "tiles", "title": fid, "data": data} for fid, data in figures.items()},
        "provenance": [],
    }


def _results() -> dict:
    return {
        "signal": _section(
            "signal",
            "The value signal",
            {
                "evidence": {
                    "tiles": [
                        {"label": "Newey-West t", "value": -1.2},
                        {"label": "Effective observations", "value": 7.4, "sub": "From 1,234 company-dates on 20 dates"},
                    ]
                },
                "ic_by_date": {"rows": [{"label": f"{2015 + q // 4}-{3 * (q % 4) + 3:02d}", "value": 0.1} for q in range(20)]},
            },
            headline={"n": 20, "score": -0.05},
            refusals=[{"what": "Survivorship correction", "why": "No exit was seen."}],
        ),
        "encoder": _section(
            "encoder",
            "Peer encoder",
            {
                "truncation": {"series": []},
                "tower_ablation": {
                    "note": "9 of the panel's 40 features are empty on every one of its 700 rows, including all 5 market features.",
                    "table": {
                        "rows": [
                            {"cut_key": "ablation", "tower": "fundamentals", "damage": 0.5, "fold_sd": 0.5},
                            {"cut_key": "headline", "tower": "fundamentals", "damage": 0.0123, "fold_sd": 0.0456},
                            {"cut_key": "headline", "tower": "text", "damage": 0.2, "fold_sd": 0.1},
                        ]
                    },
                },
            },
            headline={"n": 81, "score": 0.4321},
        ),
        "warranted": _section(
            "warranted",
            "Warranted multiple",
            {
                "panel": {
                    "tiles": [
                        {"key": "observations", "label": "Company-quarters", "value": 1500},
                        {"key": "companies", "label": "Companies", "value": 60},
                        {"key": "dates", "label": "Quarter ends", "value": 25},
                    ]
                }
            },
            headline={"n": 700},
        ),
        "fade": _section(
            "fade",
            "Revenue fade",
            {
                "depth": {
                    "stats": {
                        "observations": 2400,
                        "filers_with_rows": 150,
                        "median_years": 9.5,
                        "max_years": 17,
                        "at_max": 11,
                        "first_filed": 2008,
                        "last_filed": 2025,
                    }
                }
            },
            headline={"n": 1900},
        ),
        "propensity": _section(
            "propensity",
            "M&A propensity",
            {"sample": {"counts": {"n_labelled": 8000, "n_scored": 5000, "scored_deals": 40, "distinct_deals": 55}}},
            headline={"n": 5000},
        ),
        "datalayer": _section(
            "datalayer",
            "Data layer",
            {
                "peer_label_counts": {"counts": {"pairs": 2000, "filers_kept": 50, "groups_kept": 180, "groups_recorded": 250}},
                "nvda_closes": {
                    "counts": {"days": 2012},
                    "series": [{"name": "Close", "values": [{"x": "2017-01-06", "y": 1.0}, {"x": "2025-01-03", "y": 2.0}]}],
                },
            },
        ),
    }


def _with_encoder_subtitle(results: dict, subtitle: str) -> dict:
    results["encoder"]["figures"]["truncation"]["subtitle"] = subtitle
    return results


def _fake() -> dict:
    return _with_encoder_subtitle(
        _results(), "Change in NDCG@10 from the committed excerpts, which stop at 1,800 characters, on 81 queries"
    )


def _shaped(results: dict) -> dict:
    return shape(read(results))


def _as_section(returned: dict) -> dict:
    section = to_jsonable(
        {
            "id": sample.ID,
            "title": sample.TITLE,
            "takeaway": returned.get("takeaway", ""),
            "status": returned["status"],
            "refusals": returned.get("refusals", []),
            "headline": returned.get("headline"),
            "figures": returned.get("figures", {}),
            "provenance": [
                {"figure": fid, "entry_point": sample.ENTRY_POINTS[fid], "inputs": [], "seconds": 0.0}
                for fid in returned.get("figures", {})
            ],
        }
    )
    validate_section(section)
    return section


def _rows(out: dict) -> dict:
    return {r["key"]: r for r in out["figures"]["counts"]["data"]["rows"]}


def _numbers(value, path="data"):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _numbers(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _numbers(v, f"{path}[{i}]")


# --------------------------------------------------------------------------- #
# counts flow through unchanged
# --------------------------------------------------------------------------- #


def test_every_count_reaches_the_figure_unchanged_and_names_where_it_was_read():
    rows = _rows(_shaped(_fake()))
    expected = {
        "signal.company_dates": (1234, 'signal.figures.evidence.data.tiles[label="Effective observations"].sub'),
        "signal.dates": (20, "signal.headline.n"),
        "signal.effective": (7.4, 'signal.figures.evidence.data.tiles[label="Effective observations"].value'),
        "encoder.pairs": (2000, "datalayer.figures.peer_label_counts.data.counts.pairs"),
        "encoder.queries": (81, "encoder.headline.n"),
        "encoder.filers": (50, "datalayer.figures.peer_label_counts.data.counts.filers_kept"),
        "warranted.company_quarters": (1500, 'warranted.figures.panel.data.tiles[key="observations"].value'),
        "warranted.scored": (700, "warranted.headline.n"),
        "warranted.companies": (60, 'warranted.figures.panel.data.tiles[key="companies"].value'),
        "fade.company_years": (2400, "fade.figures.depth.data.stats.observations"),
        "fade.scored": (1900, "fade.headline.n"),
        "fade.filers": (150, "fade.figures.depth.data.stats.filers_with_rows"),
        "propensity.labelled": (8000, "propensity.figures.sample.data.counts.n_labelled"),
        "propensity.scored": (5000, "propensity.figures.sample.data.counts.n_scored"),
        "propensity.deals": (40, "propensity.figures.sample.data.counts.scored_deals"),
    }
    assert {k: (r["values"]["count"], r["source"]) for k, r in rows.items()} == expected
    assert all(r["refusal"] is None for r in rows.values())


def test_the_table_view_carries_the_same_counts_and_sources_as_the_chart():
    data = _shaped(_fake())["figures"]["counts"]["data"]
    assert [(t["count"], t["source"]) for t in data["table"]["rows"]] == [
        (r["values"]["count"], r["source"]) for r in data["rows"]
    ]
    signal = [t for t in data["table"]["rows"] if t["model"] == "The value signal"]
    assert signal[0]["ratio"] == ""
    assert signal[2]["ratio"] == pytest.approx(1234 / 7.4)


def test_models_are_grouped_in_page_order_with_their_titles_from_the_results():
    data = _shaped(_fake())["figures"]["counts"]["data"]
    assert [g["key"] for g in data["groups"]] == ["signal", "encoder", "warranted", "fade", "propensity"]
    assert data["groups"][2]["label"] == "Warranted multiple"
    assert [r["step"] for r in data["rows"][:3]] == ["nominal", "headline", "limiting"]


def test_only_the_limiting_count_is_orange_and_nothing_is_the_model_blue():
    for row in _shaped(_fake())["figures"]["counts"]["data"]["rows"]:
        assert row["role"] == ("alt" if row["step"] == "limiting" else "baseline")


def test_ratios_titles_and_takeaway_follow_the_numbers():
    out = _shaped(_fake())
    groups = {g["key"]: g for g in out["figures"]["counts"]["data"]["groups"]}
    assert groups["signal"]["ratio"] == pytest.approx(1234 / 7.4)
    assert groups["signal"]["ratioText"] == "167 times fewer"
    assert groups["fade"]["ratioText"] == "16 times fewer"
    assert out["figures"]["counts"]["title"] == (
        "The counts that limit the five models run from 7 to 150, against largest counts of 1,234 to 8,000"
    )
    takeaway = out["takeaway"]
    assert takeaway.startswith("The five models count their evidence in thousands of rows")
    assert "16 to 200 times scarcer" in takeaway
    for clause in (
        "7 effective observations behind the value signal's 1,234 company-dates",
        "50 filers behind the peer encoder's 2,000 training pairs",
        "60 companies behind the warranted multiple's 1,500 company-quarters",
        "150 filers behind the revenue fade's 2,400 company-years",
        "40 deals behind the propensity screen's 8,000 labelled company-quarters",
    ):
        assert clause in takeaway, clause
    assert "Nvidia's closes reach back 8 years and the median filer's revenue 10 fiscal years" in takeaway


def test_the_record_is_years_between_published_dates_and_published_depths():
    out = _shaped(_fake())
    record = out["figures"]["record"]
    years = {r["key"]: r["value"] for r in record["data"]["rows"]}
    assert years["closes"] == pytest.approx((8 * 365 - 3 + 2) / 365.25)
    assert years["rebalance"] == pytest.approx(19 * 3 / 12)
    assert years["median_filer"] == 9.5 and years["deepest_filers"] == 17
    details = {r["label"]: r["detail"] for r in record["data"]["table"]["rows"]}
    assert "20 quarterly dates, 2015-03 to 2019-12" in details["The value signal's rebalance dates, first to last"]
    assert "11 of the 150 filers reach it" in details["Fiscal years of revenue, deepest filers"]
    assert record["title"] == "Nvidia's closes reach back 8 years, the median filer's revenue 10 fiscal years"


def test_limits_quote_published_numbers_with_their_sources():
    items = {i["what"]: i for i in _shaped(_fake())["figures"]["limits"]["data"]["items"]}
    assert len(items) == 7
    text = items["The text corpus is an excerpt"]
    assert "stops at 1,800 characters" in text["why"] and "0.4321 NDCG@10" in text["why"]
    assert {q["source"]: q["value"] for q in text["quotes"]} == {
        "encoder.figures.truncation.subtitle": 1800,
        "encoder.headline.score": 0.4321,
    }
    feed = items["There is no market feed in the peer panel"]
    assert feed["why"].startswith("9 of the peer panel's 40 features")
    assert "by 0.012, inside a fold standard deviation of 0.046" in feed["why"]
    assert "It is why the value signal refuses a survivorship correction across its 1,234 company-dates." in items[
        "There is no point-in-time universe"
    ]["why"]
    assert "the 180 of 250 disclosed groups" in items["The labels are compensation peers, not trading comparables"]["why"]
    for item in items.values():
        for q in item["quotes"]:
            assert q["source"] and math.isfinite(q["value"])


def test_the_section_meets_the_schema_with_finite_numbers_and_no_em_dash():
    section = _as_section(_shaped(_fake()))
    assert section["status"] == "ok" and section["headline"] is None and section["refusals"] == []
    assert set(section["figures"]) == {"counts", "record", "limits"}
    for fid, fig in section["figures"].items():
        assert fig["title"].strip(), fid
        assert chr(0x2014) not in json.dumps(fig, ensure_ascii=False), fid
        for where, number in _numbers(fig["data"]):
            assert math.isfinite(number), f"{fid} {where}"


# --------------------------------------------------------------------------- #
# refusals name the field
# --------------------------------------------------------------------------- #


def test_a_missing_count_is_a_refusal_naming_the_field_and_its_row_is_left_empty():
    results = _fake()
    del results["propensity"]["figures"]["sample"]["data"]["counts"]["scored_deals"]
    out = _shaped(results)
    field = "propensity.figures.sample.data.counts.scored_deals"
    assert out["refusals"] == [
        {
            "what": "M&A propensity: deals",
            "why": f"Not read from {field}: the field is absent. The count is not recomputed here, so its row is left empty.",
        }
    ]
    row = _rows(out)["propensity.deals"]
    assert row["values"]["count"] is None and row["source"] == field and row["refusal"] == "M&A propensity: deals"
    groups = {g["key"]: g for g in out["figures"]["counts"]["data"]["groups"]}
    assert groups["propensity"]["ratio"] is None and groups["propensity"]["ratioText"] == ""
    assert "propensity" not in out["takeaway"] and out["takeaway"].startswith("The four models")
    assert "the four models" in out["figures"]["counts"]["title"]
    _as_section(out)


@pytest.mark.parametrize("bad", ["8000", True, float("nan"), 0, -3, None])
def test_a_count_that_is_not_a_positive_number_is_refused_not_coerced(bad):
    results = _fake()
    results["propensity"]["figures"]["sample"]["data"]["counts"]["n_labelled"] = bad
    out = _shaped(results)
    [refusal] = out["refusals"]
    assert refusal["what"] == "M&A propensity: labelled company-quarters"
    assert "propensity.figures.sample.data.counts.n_labelled" in refusal["why"]
    assert _rows(out)["propensity.labelled"]["values"]["count"] is None


def test_a_reworded_signal_sentence_refuses_the_company_dates_rather_than_guessing():
    results = _fake()
    results["signal"]["figures"]["evidence"]["data"]["tiles"][1]["sub"] = "1,234 company-dates across 20 dates"
    out = _shaped(results)
    [refusal] = out["refusals"]
    assert refusal["what"] == "The value signal: company-dates"
    assert 'signal.figures.evidence.data.tiles[label="Effective observations"].sub' in refusal["why"]
    assert _rows(out)["signal.company_dates"]["values"]["count"] is None
    point_in_time = out["figures"]["limits"]["data"]["items"][1]
    assert "1,234" not in point_in_time["why"] and point_in_time["quotes"] == []


def test_a_signal_sentence_about_other_dates_than_the_headline_is_refused():
    results = _fake()
    results["signal"]["figures"]["evidence"]["data"]["tiles"][1]["sub"] = "From 1,234 company-dates on 19 dates"
    [refusal] = _shaped(results)["refusals"]
    assert "names 19 dates where signal.headline.n is 20" in refusal["why"]


def test_an_absent_section_is_one_refusal_for_its_model_not_one_per_row():
    results = _fake()
    del results["warranted"]
    out = _shaped(results)
    assert out["refusals"] == [
        {
            "what": "Warranted multiple: every count",
            "why": "None of its counts can be read, because the snapshot holds no warranted section. Nothing is recomputed in their place.",
        }
    ]
    rows = [r for r in out["figures"]["counts"]["data"]["rows"] if r["group"] == "warranted"]
    assert [r["refusal"] for r in rows] == ["Warranted multiple: every count"] * 3


def test_a_refused_section_upstream_refuses_the_counts_it_would_have_published():
    results = _fake()
    results["datalayer"]["status"] = "refused"
    out = _shaped(results)
    whats = [r["what"] for r in out["refusals"]]
    assert whats == [
        "Peer encoder: training pairs",
        "Peer encoder: filers",
        "Depth of the record: Nvidia's closes, first week to last",
    ]
    assert "the datalayer section is refused" in out["refusals"][0]["why"]
    assert _rows(out)["encoder.queries"]["values"]["count"] == 81
    assert out["figures"]["record"]["data"]["refusals"] == ["Depth of the record: Nvidia's closes, first week to last"]
    limits = {i["what"]: i for i in out["figures"]["limits"]["data"]["items"]}
    labels = limits["The labels are compensation peers, not trading comparables"]
    assert labels["quotes"] == [] and not any(c.isdigit() for c in labels["why"])


def test_a_limit_whose_number_is_unpublished_is_stated_without_it():
    results = _results()  # no truncation subtitle
    del results["encoder"]["figures"]["tower_ablation"]["data"]["note"]
    items = {i["what"]: i for i in _shaped(results)["figures"]["limits"]["data"]["items"]}
    text = items["The text corpus is an excerpt"]
    assert [q["label"] for q in text["quotes"]] == ["Encoder NDCG@10"]
    assert "characters" not in text["why"]
    feed = items["There is no market feed in the peer panel"]
    assert [q["source"].rsplit(".", 1)[-1] for q in feed["quotes"]] == ["damage", "fold_sd"]
    assert "features are empty" not in feed["why"]


def test_with_no_model_section_collected_the_section_is_refused():
    out = _shaped({})
    assert out["status"] == "refused"
    assert out["figures"] == {}
    assert out["refusals"][0]["what"] == sample.TITLE
    _as_section(out)


# --------------------------------------------------------------------------- #
# the collector
# --------------------------------------------------------------------------- #


def test_collect_records_every_figure_against_a_reader_that_resolves(tmp_path):
    ctx = CollectContext(root=FIXTURES, repo_root=ROOT, results=_fake())
    section, run = C.collect_section(sample, ctx, use_cache=False)
    assert run.status == "ok"
    assert {row["figure"] for row in section["provenance"]} == set(section["figures"])
    for row in section["provenance"]:
        assert row["entry_point"] == sample.ENTRY_POINTS[row["figure"]]
        assert row["inputs"] == []
        entry_module(row["entry_point"])
    assert sample.INPUTS == []


def test_collect_does_not_alter_the_results_it_reads():
    results = _fake()
    before = copy.deepcopy(results)
    ctx = CollectContext(root=FIXTURES, repo_root=ROOT, results=results)
    C.collect_section(sample, ctx, use_cache=False)
    assert results == before


# --------------------------------------------------------------------------- #
# pinned truths, against the committed snapshot
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def committed() -> dict:
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    sections = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"]
    if sections.get("sample", {}).get("status") != "ok":
        pytest.skip("the committed snapshot carries no collected sample section")
    return sections


def test_committed_section_is_what_shape_makes_of_the_sections_beside_it(committed):
    expected = to_jsonable(shape(read(committed)))
    section = committed["sample"]
    for key in ("status", "takeaway", "refusals", "figures"):
        assert section[key] == expected[key], key


def test_committed_counts_by_model(committed):
    rows = {r["key"]: r["values"]["count"] for r in committed["sample"]["figures"]["counts"]["data"]["rows"]}
    assert rows == {
        "signal.company_dates": 2604,
        "signal.dates": 35,
        "signal.effective": pytest.approx(13.03, abs=0.01),
        "encoder.pairs": 2534,
        "encoder.queries": 162,
        "encoder.filers": 63,
        "warranted.company_quarters": 1764,
        "warranted.scored": 888,
        "warranted.companies": 94,
        "fade.company_years": 2798,
        "fade.scored": 2188,
        "fade.filers": 223,
        "propensity.labelled": 9400,
        "propensity.scored": 5881,
        "propensity.deals": 67,
    }
    assert committed["sample"]["refusals"] == []


def test_committed_titles_and_record(committed):
    figures = committed["sample"]["figures"]
    assert figures["counts"]["title"] == (
        "The counts that limit the five models run from 13 to 223, against largest counts of 1,764 to 9,400"
    )
    years = {r["key"]: r["value"] for r in figures["record"]["data"]["rows"]}
    assert years["rebalance"] == 8.5
    assert years["closes"] == pytest.approx(10.0, abs=0.05)
    assert (years["median_filer"], years["deepest_filers"]) == (12, 19)
    assert len(figures["limits"]["data"]["items"]) == 7
