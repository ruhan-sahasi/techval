"""The data layer section: the page it shapes, and the numbers it must keep showing.

Three groups, and none of them runs the peer encoder.

The shaping tests hand ``shape`` small results built by hand and hold what comes
back to the page's rules: every figure a kind the kit draws, every number finite,
the takeaway and the titles following the values they were given, and a check
that cannot be computed becoming a refusal with a reason and no figure.

The check tests run the cheap measurements on tiny inputs, and the debt, split
and ticker checks once on the committed fixtures, because those take a fraction
of a second and are where the section's cross-checked numbers come from.

The pinned tests read the committed ``docs/dashboard/snapshot.json`` and check the
section's key numbers against the audited values. They skip until that file is
committed.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import pytest

from techval.dashboard.collect import entry_module
from techval.dashboard.sections import datalayer as D
from techval.dashboard.snapshot import to_jsonable, validate_section

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "src" / "techval" / "dashboard" / "assets" / "kit.js"
RENDERER = ROOT / "src" / "techval" / "dashboard" / "assets" / "sections" / "datalayer.js"
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"
FIXTURES = ROOT / "tests" / "fixtures"


def _kit_charts() -> set[str]:
    source = KIT.read_text(encoding="utf-8")
    body = re.search(r"TV\.charts = \{(.*?)\};", source, flags=re.S)
    assert body, "kit.js no longer assigns TV.charts"
    return set(re.findall(r"(\w+)\s*:", body.group(1)))


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _numbers(value: Any) -> Iterator[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _numbers(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _numbers(v)


def _as_snapshot_section(shaped: dict[str, Any]) -> dict[str, Any]:
    """What the collector would store, with one provenance row per figure."""
    return to_jsonable(
        {
            "id": D.ID,
            "title": D.TITLE,
            "provenance": [
                {"figure": fid, "entry_point": f"{D.__name__}.shape", "inputs": [], "seconds": 0.0}
                for fid in shaped["figures"]
            ],
            **shaped,
        }
    )


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


def _debt(**changes: Any) -> dict[str, Any]:
    base = {
        "legs": [
            {"leg": "long_term", "reporting": 40, "reached": 39, "unreached": ["DLR"]},
            {"leg": "current", "reporting": 30, "reached": 30, "unreached": []},
        ],
        "any": {"reporting": 42, "reached": 41},
        "unreached": [
            {
                "ticker": "DLR",
                "concepts": ["SecuredDebt", "SeniorNotes"],
                "has_total": False,
                "caught_by": ["SecuredDebt", "SeniorNotes"],
            }
        ],
        "filers_covered": 50,
        "universe_size": 52,
        "recorded": "2026-01-02",
        "material_usd": 100e6,
        "tolerance_days": 20,
        "census_ladders_match_tags": True,
    }
    base.update(changes)
    return base


def _splits(bad: int = 0) -> dict[str, Any]:
    return {
        "filers": [
            {"ticker": "AAA", "splits": [4.0], "checked": 20, "inconsistent": bad, "late_end": 3, "early_end": 0},
            {"ticker": "BBB", "splits": [2.0, 3.0], "checked": 30, "inconsistent": 0, "late_end": 0, "early_end": 5},
            {"ticker": "CTRL", "splits": [], "checked": 10, "inconsistent": 0, "late_end": 0, "early_end": 0},
        ],
        "quarter": {
            "ticker": "BBB",
            "start": "2021-08-01",
            "end": "2021-10-31",
            "filed": "2022-11-18",
            "as_filed": 100.0,
            "adjusted": 200.0,
            "late_end": 600.0,
        },
        "band": 1.01,
        "tags": 2,
    }


def _closes(step: bool = False) -> dict[str, Any]:
    worst = -0.88 if step else -0.05
    return {
        "ticker": "NVDA",
        "first": date(2020, 1, 3),
        "last": date(2020, 6, 26),
        "days": 120,
        "weekly": [(date(2020, 1, 3), 10.0), (date(2020, 3, 6), 12.0), (date(2020, 6, 26), 15.0)],
        "largest": {"move": 0.21, "date": date(2020, 3, 6), "close": 12.0},
        "windows": [
            {
                "from": "2020-02-01",
                "to": "2020-04-01",
                "factor": 10.0,
                "worst_day": worst,
                "worst_date": date(2020, 3, 2),
                "unadjusted_move": -0.9,
                "days": 40,
            }
        ],
    }


def _taxonomy() -> dict[str, Any]:
    return {
        "seed": 12,
        "payloads": 10,
        "classified": 10,
        "how": {"code_agrees": 6, "prior_fills": 3, "prior_overrides": 1, "unresolved": 2},
        "by_vertical": {
            "semiconductors": {"classified": 6, "by_code": 6},
            "payments": {"classified": 4, "by_code": 0},
        },
        "unresolved": ["EA", "JNPR"],
    }


def _peers(**lost: int) -> dict[str, Any]:
    groups_dropped = {"not_usable": 0, "before_panel": 7, "filer_not_encodable": 3, "no_peer_kept": 0}
    groups_dropped.update(lost)
    recorded = 40
    return {
        "groups_recorded": recorded,
        "filers_recorded": 12,
        "groups_kept": recorded - sum(groups_dropped.values()),
        "filers_kept": 10,
        "named_peers": 500,
        "pairs": 300,
        "lost": {"group": 120, "universe": 50, "encodable": 30, "self": 0},
        "groups_dropped": groups_dropped,
        "panel_dates": 4,
        "universe": 60,
    }


def _tickers() -> dict[str, Any]:
    return {
        "seeds": 20,
        "symbols": 18,
        "registrants": 17,
        "rungs": {"explicit CIK": 3, "former-ticker index": 17},
        "cik_only": 3,
        "cik_only_reasons": ["no trading symbol on the cover page"],
        "shared_symbols": [{"symbol": "AZPN", "registrants": 2, "undated_cik": 2, "reached_only_dated": 1}],
        "seed_universe": 12,
        "seed_absent": [
            {"ticker": "EA", "source": "unresolved", "cik": None, "name": None},
            {"ticker": "JNPR", "source": "former-ticker index", "cik": 1043604, "name": "JUNIPER NETWORKS INC"},
        ],
        "seed_offline": {
            "ticker_file_registrants": 8,
            "seed_in_ticker_file": 5,
            "rungs": {"SEC ticker file": 5, "former-ticker index": 1, "unresolved": 6},
            "unresolved_with_payload": 5,
        },
    }


def _results(**changes: Any) -> dict[str, Any]:
    base = {
        "debt": _debt(),
        "splits": _splits(),
        "closes": _closes(),
        "taxonomy": _taxonomy(),
        "peers": _peers(),
        "tickers": _tickers(),
    }
    base.update(changes)
    return base


NAMES = {"DLR": "DIGITAL REALTY TRUST, INC."}


# --------------------------------------------------------------------------- #
# shaping, with nothing run
# --------------------------------------------------------------------------- #


def test_every_figure_is_well_formed_and_in_page_order():
    shaped = D.shape(_results(), NAMES)
    charts = _kit_charts()
    assert tuple(shaped["figures"]) == D.FIGURE_IDS
    assert shaped["headline"] is None and shaped["status"] == "ok"
    for fid, fig in shaped["figures"].items():
        assert fig["kind"] in charts, fid
        assert fig["title"].strip(), fid
        assert all(math.isfinite(v) for v in _numbers(to_jsonable(fig["data"]))), fid
        data = fig["data"]
        for row in data.get("rows", []):
            assert isinstance(row["label"], str) and row["label"].strip(), fid
        for series in data.get("series", []):
            assert series["name"].strip() and series["role"] in ("model", "baseline", "alt"), fid
        for tile in data.get("tiles", []):
            assert tile["label"].strip() and tile["sub"].strip(), fid
        if fig["kind"] != "tiles":
            assert fig["subtitle"].strip(), fid
    validate_section(_as_snapshot_section(shaped))


def test_no_copy_on_the_page_carries_an_em_dash():
    for results in (_results(), _results(closes=_closes(step=True), splits=_splits(bad=2))):
        assert not [s for s in _strings(to_jsonable(D.shape(results, NAMES))) if "\u2014" in s]


def test_groups_and_filers_never_share_an_axis():
    figures = D.shape(_results(), NAMES)["figures"]
    counts = figures["peer_label_counts"]
    assert counts["kind"] == "tiles"
    assert [t["label"] for t in counts["data"]["tiles"]] == ["Peer groups kept", "Filers kept", "Training pairs"]
    for fid in ("peer_groups_lost", "peer_label_funnel"):
        assert figures[fid]["kind"] == "hbar"
    assert figures["peer_groups_lost"]["data"]["valueLabel"] == "Peer groups"
    assert figures["peer_label_funnel"]["data"]["valueLabel"] == "Named peers"


def test_the_takeaway_is_built_from_the_values_it_was_given():
    text = D.shape(_results(), NAMES)["takeaway"]
    assert "reach 39 of the 40 filers" in text
    assert "every one of 60 share-count periods" in text
    assert "10 of 40 disclosed peer groups never reach the encoder" in text

    broken = D.shape(_results(splits=_splits(bad=2)), NAMES)["takeaway"]
    assert "2 of 60 share-count periods filed more than once still disagree" in broken

    nothing = D.shape({}, NAMES)
    assert nothing["takeaway"] == "No data-layer check could be computed from the committed fixtures."
    assert nothing["figures"] == {}


def test_the_debt_title_names_who_the_ladders_miss_and_what_happens_to_them():
    title = D.shape(_results(), NAMES)["figures"]["debt_reach"]["title"]
    assert title == (
        "The debt ladders reach 39 of the 40 filers with material long-term debt; "
        "the cross-check refuses the one they miss, Digital Realty Trust"
    )
    silent = _debt(unreached=[{"ticker": "VZ", "concepts": ["LongTermDebtNoncurrent"], "has_total": False, "caught_by": []}])
    shaped = D.shape(_results(debt=silent), {"VZ": "VERIZON COMMUNICATIONS INC"})
    assert shaped["figures"]["debt_reach"]["title"].endswith("and Verizon Communications reads as zero")
    assert shaped["figures"]["debt_reach"]["data"]["counts"]["silent_zero"] == ["VZ"]


def test_the_before_the_fix_refusal_depends_on_whether_the_census_ladders_are_todays():
    def before(debt):
        refusals = D.shape(_results(debt=debt), NAMES)["refusals"]
        return next(r["why"] for r in refusals if r["what"] == D.REFUSED_DEBT_BEFORE)

    assert "are the ones techval.tags carries today" in before(_debt())
    assert "differ from the ones techval.tags carries today" in before(_debt(census_ladders_match_tags=False))
    near = D.shape(_results(), NAMES)["figures"]["debt_reach"]["data"]["near"]
    assert near == [D.REFUSED_DEBT_BEFORE]


def test_the_split_titles_follow_the_counts():
    figures = D.shape(_results(), NAMES)["figures"]
    assert figures["split_counts"]["title"] == "All 60 share-count periods filed more than once agree once split-adjusted"
    assert figures["split_oracle"]["title"] == (
        "Dating a split at either end of its window breaks 3 or 5 periods; the per-filing basis breaks none"
    )
    oracle = figures["split_oracle"]
    assert "every filing of one period must report the same quantity" in oracle["subtitle"]
    assert [r["label"] for r in oracle["data"]["rows"]] == [
        "AAA, 4-for-1",
        "BBB, 2-for-1 then 3-for-1",
        "CTRL, never split (control)",
    ]
    tiles = figures["split_counts"]["data"]["tiles"]
    assert "2 weighted-average share tags" in tiles[0]["sub"]
    assert tiles[2]["value"] == 200.0 and "600" in tiles[2]["sub"]

    broken = D.shape(_results(splits=_splits(bad=2)), NAMES)["figures"]
    assert broken["split_counts"]["title"].startswith("2 of 60 share-count periods")


def test_the_closes_title_says_whether_the_series_was_restated():
    restated = D.shape(_results(), NAMES)["figures"]["nvda_closes"]
    assert restated["title"] == (
        "Nvidia's closes carry no split step: no day inside either split window falls more than 5.0%"
    )
    assert restated["data"]["mark"]["label"] == "Largest day, +21.0% on 2020-03-06"
    stepped = D.shape(_results(closes=_closes(step=True)), NAMES)["figures"]["nvda_closes"]
    assert stepped["title"].startswith("Nvidia's closes fall −88.0% in a day inside the 10-for-1 window")
    assert stepped["data"]["counts"]["restated"] is False


def test_the_taxonomy_names_the_unresolved_and_what_the_prior_decided():
    figures = D.shape(_results(), NAMES)["figures"]
    how = figures["taxonomy_how"]
    assert how["title"] == "The analyst's prior, not the SIC code, places 4 of the 10 classified filers"
    assert how["data"]["rows"][-1]["label"] == "Unresolved: EA and JNPR"
    assert "JNPR resolves through the former-ticker index" in how["subtitle"]
    assert figures["taxonomy_by_vertical"]["title"] == "The SIC code alone places no filer in payments"

    alone = D.shape(_results(tickers={"refused": "no index"}), NAMES)["figures"]["taxonomy_how"]
    assert "former-ticker index" not in alone["subtitle"]


def test_the_peer_group_title_names_the_two_largest_losses_and_drops_empty_rows():
    fig = D.shape(_results(), NAMES)["figures"]["peer_groups_lost"]
    assert fig["title"] == (
        "10 of 40 peer groups are lost: 7 because the group predates the panel "
        "and 3 because the filer could not be encoded"
    )
    assert [r["value"] for r in fig["data"]["rows"]] == [7, 3]
    assert "No group is lost because it lacks a filing date or fiscal year or keeps no encodable peer" in fig["subtitle"]

    none = D.shape(_results(peers=_peers(before_panel=0, filer_not_encodable=0)), NAMES)["figures"]
    assert none["peer_groups_lost"]["title"] == "All 40 disclosed peer groups reach the encoder"


def test_the_ticker_figure_draws_only_what_the_pruned_file_cannot_move_and_refuses_the_rest():
    shaped = D.shape(_results(), NAMES)
    fig = shaped["figures"]["ticker_reach"]
    assert fig["title"] == "17 of 20 departed registrants resolve by their old ticker, and 1 seed ticker by nothing but a CIK"
    subs = {t["label"]: t["sub"] for t in fig["data"]["tiles"]}
    assert "JNPR resolves through the former-ticker index to Juniper Networks" in subs["Seed tickers the SEC file had dropped"]
    assert "EA through no rung but an explicit CIK" in subs["Seed tickers the SEC file had dropped"]
    why = next(r["why"] for r in shaped["refusals"] if r["what"] == D.REFUSED_SEED_RUNGS)
    assert "pruned copy of 8 registrants" in why and "5 of the 12 seed tickers" in why
    assert "6 unresolved, 5 by the SEC ticker file and 1 by the former-ticker index" in why
    assert "5 of the unresolved seeds carry a submissions payload" in why
    assert fig["data"]["near"] == [D.REFUSED_SEED_RUNGS]


def test_a_check_that_could_not_be_computed_is_a_refusal_with_its_reason_and_no_figure():
    shaped = D.shape(_results(closes={"refused": "the committed price series carries no Nvidia closes"}), NAMES)
    assert "nvda_closes" not in shaped["figures"]
    refusal = next(r for r in shaped["refusals"] if r["what"] == "Nvidia's split-adjusted closes")
    assert refusal["why"] == "the committed price series carries no Nvidia closes"
    validate_section(_as_snapshot_section(shaped))

    missing = _results()
    del missing["peers"]
    shaped = D.shape(missing, NAMES)
    assert not {"peer_label_counts", "peer_groups_lost", "peer_label_funnel"} & set(shaped["figures"])
    assert {"what": "Where peer labels are lost", "why": "the check was not run"} in shaped["refusals"]
    assert "peer groups" not in shaped["takeaway"]
    validate_section(_as_snapshot_section(shaped))


def test_a_builder_that_makes_the_wrong_figures_is_a_bug_not_a_page():
    original = D.FIGURES_BY_RESULT["tickers"]
    try:
        D.FIGURES_BY_RESULT["tickers"] = ("something_else",)
        with pytest.raises(ValueError):
            D.shape(_results(), NAMES)
    finally:
        D.FIGURES_BY_RESULT["tickers"] = original


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("DIGITAL REALTY TRUST, INC.", "Digital Realty Trust"),
        ("Juniper Networks, Inc.", "Juniper Networks"),
        ("AT&T INC.", "AT&T"),
        ("THE WALT DISNEY CO", "The Walt Disney"),
        ("", "DLR"),
        (None, "DLR"),
    ],
)
def test_display_name(name, expected):
    assert D.display_name(name, "DLR") == expected


# --------------------------------------------------------------------------- #
# the checks, on small inputs and on the committed fixtures
# --------------------------------------------------------------------------- #


def test_the_debt_concept_sets_are_the_guards_own():
    import test_tags

    assert D.REPORTS_NONCURRENT_DEBT == test_tags.REPORTS_NONCURRENT_DEBT
    assert D.REPORTS_CURRENT_DEBT == test_tags.REPORTS_CURRENT_DEBT


def test_debt_reach_counts_a_total_on_both_legs_and_names_the_cross_check():
    census = {
        "_techval_fixture": {"recorded": "2026-01-02", "material_usd": 1e8, "tolerance_days": 20},
        "universe_size": 4,
        "covered": ["A", "B", "C", "D"],
        "ladders": {},
        "live_by_filer": {
            "A": ["LongTermDebtNoncurrent", "DebtCurrent"],
            "B": ["DebtLongtermAndShorttermCombinedAmount"],
            "C": ["SeniorNotes"],
            "D": ["ShortTermBorrowings"],
        },
    }
    r = D.debt_ladder_reach(census)
    legs = {leg["leg"]: leg for leg in r["legs"]}
    assert (legs["long_term"]["reporting"], legs["long_term"]["reached"]) == (3, 2)
    assert (legs["current"]["reporting"], legs["current"]["reached"]) == (3, 3)
    assert r["any"] == {"reporting": 4, "reached": 3}
    assert [m["ticker"] for m in r["unreached"]] == ["C"]
    assert r["unreached"][0]["caught_by"] == ["SeniorNotes"]
    assert r["census_ladders_match_tags"] is False


def test_the_closes_check_refuses_without_a_series_or_a_split():
    closes = [(date(2020, 1, 1), 10.0), (date(2020, 1, 2), 11.0)]
    with pytest.raises(D._Unavailable, match="no Nvidia closes"):
        D.nvda_split_evidence([], [("2020-01-01", "2020-02-01", 4.0)])
    with pytest.raises(D._Unavailable, match="show no split"):
        D.nvda_split_evidence(closes, [])
    with pytest.raises(D._Unavailable, match="no close in the committed series falls inside"):
        D.nvda_split_evidence(closes, [("2021-01-01", "2021-02-01", 4.0)])


def test_the_closes_check_sees_an_unrestated_step():
    closes = [(date(2020, 1, d), c) for d, c in ((6, 40.0), (7, 41.0), (8, 10.2), (9, 10.0), (10, 10.5))]
    r = D.nvda_split_evidence(closes, [("2020-01-06", "2020-01-10", 4.0)])
    assert r["largest"]["date"] == date(2020, 1, 8)
    assert r["largest"]["move"] == pytest.approx(10.2 / 41.0 - 1)
    window = r["windows"][0]
    assert window["worst_day"] == pytest.approx(10.2 / 41.0 - 1)
    assert window["unadjusted_move"] == pytest.approx(-0.75)
    assert D.shape(_results(closes=r), NAMES)["figures"]["nvda_closes"]["data"]["counts"]["restated"] is False


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_debt_reach_on_the_committed_census():
    r = D.debt_ladder_reach(_load(D.CENSUS))
    long_term = next(leg for leg in r["legs"] if leg["leg"] == "long_term")
    assert (long_term["reached"], long_term["reporting"]) == (80, 81)
    assert [m["ticker"] for m in r["unreached"]] == ["DLR"]
    assert r["unreached"][0]["caught_by"]


def test_split_consistency_on_the_committed_payloads():
    payloads = {k: v for k, v in _load(D.SHARE_BASIS).items() if not k.startswith("_")}
    payloads["CRWD"] = _load(D.CRWD_FACTS)
    r = D.split_consistency(payloads)
    assert sum(f["checked"] for f in r["filers"]) == 298
    assert sum(f["inconsistent"] for f in r["filers"]) == 0
    assert {f["ticker"]: f["late_end"] for f in r["filers"]}["PANW"] == 6
    assert r["quarter"]["adjusted"] == pytest.approx(585_800_000)
    assert r["quarter"]["late_end"] == pytest.approx(1_757_400_000)


def test_ticker_reach_on_the_committed_index():
    from techval.tmt.taxonomy import SEED

    r = D.ticker_reach(
        _load(D.FORMER_INDEX),
        _load(D.MNA_UNIVERSE),
        _load(D.TICKER_FILE)["data"],
        SEED,
        _load(D.SUBMISSIONS)["companies"],
    )
    assert (r["symbols"], r["registrants"], r["seeds"]) == (100, 97, 105)
    assert r["rungs"] == {"explicit CIK": 8, "former-ticker index": 97}
    assert {row["ticker"]: row["source"] for row in r["seed_absent"]} == {
        "EA": "unresolved",
        "FI": "unresolved",
        "FYBR": "unresolved",
        "IPG": "unresolved",
        "JNPR": "former-ticker index",
    }
    assert [s["symbol"] for s in r["shared_symbols"]] == ["AZPN"]


def test_the_renderer_draws_every_figure_the_section_emits_in_page_order():
    source = RENDERER.read_text(encoding="utf-8")
    groups = re.search(r"var GROUPS = \[(.*?)\n  \];", source, flags=re.S)
    assert groups, "datalayer.js no longer declares its figure groups"
    assert tuple(re.findall(r'"(\w+)"', groups.group(1))) == D.FIGURE_IDS


def test_the_entry_points_resolve_and_the_inputs_are_committed():
    for entry in (
        f"{D.__name__}.debt_ladder_reach",
        f"{D.__name__}.split_consistency",
        f"{D.__name__}.nvda_split_evidence",
        "techval.tags.DEBT_CROSSCHECK",
        "techval.edgar.CompanyFacts._split_adjust",
        "techval.edgar.CompanyFacts._split_brackets",
        "techval.tmt.taxonomy.tmt_universe",
        "techval.tmt.taxonomy.SEED",
        "techval.ml.encoder.build_dataset",
        "techval.commands_peers._load_groups",
        "techval.edgar.EdgarClient.resolve_ticker",
        "techval.former_tickers.resolve",
    ):
        entry_module(entry)
    for name in D.INPUTS:
        assert (FIXTURES / name).is_file(), name


# --------------------------------------------------------------------------- #
# pinned truths, against the committed snapshot
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pinned() -> dict[str, Any]:
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    section = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"][D.ID]
    assert section["status"] == "ok", section["refusals"]
    return section


def _tiles(section: dict[str, Any], fid: str) -> dict[str, dict[str, Any]]:
    return {t["label"]: t for t in section["figures"][fid]["data"]["tiles"]}


def test_pinned_debt_reach(pinned):
    counts = pinned["figures"]["debt_reach"]["data"]["counts"]
    assert (counts["long_term"]["reached"], counts["long_term"]["reporting"]) == (80, 81)
    assert counts["unreached"] == ["DLR"]
    assert "Digital Realty" in pinned["figures"]["debt_reach"]["title"]
    assert any(r["what"] == D.REFUSED_DEBT_BEFORE for r in pinned["refusals"])


def test_pinned_split_consistency(pinned):
    counts = pinned["figures"]["split_counts"]["data"]["counts"]
    assert counts["inconsistent"] == 0
    assert counts["periods_checked"] == 298
    assert counts["quarter"]["adjusted"] == pytest.approx(585_800_000)
    rows = {r["label"].split(",")[0]: r["values"] for r in pinned["figures"]["split_oracle"]["data"]["rows"]}
    assert rows["PANW"]["late"] == 6 and rows["DDOG"] == {"current": 0, "late": 0, "early": 0}


def test_pinned_nvda_closes(pinned):
    counts = pinned["figures"]["nvda_closes"]["data"]["counts"]
    assert counts["restated"] is True
    assert counts["largest_move_date"] == "2016-11-11"
    assert counts["largest_move"] == pytest.approx(0.298, abs=5e-4)


def test_pinned_taxonomy(pinned):
    counts = pinned["figures"]["taxonomy_how"]["data"]["counts"]
    assert (counts["classified"], counts["seed"]) == (105, 110)
    assert counts["unresolved_tickers"] == ["EA", "FI", "FYBR", "IPG", "JNPR"]


def test_pinned_peer_labels(pinned):
    counts = pinned["figures"]["peer_label_counts"]["data"]["counts"]
    assert (counts["groups_recorded"], counts["filers_recorded"]) == (320, 75)
    assert (counts["groups_kept"], counts["filers_kept"]) == (220, 63)


def test_pinned_ticker_resolution(pinned):
    counts = pinned["figures"]["ticker_reach"]["data"]["counts"]
    assert (counts["symbols"], counts["registrants"]) == (100, 97)
    assert counts["rungs"]["former-ticker index"] == 97
    assert any(r["what"] == D.REFUSED_SEED_RUNGS for r in pinned["refusals"])
