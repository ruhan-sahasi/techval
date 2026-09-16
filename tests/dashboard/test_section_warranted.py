"""The warranted-multiple section: the shaping on small fakes, and the page's pinned numbers.

Nothing here fits the model. ``tests/ml/test_warranted.py`` pins the fit; these
tests pin what the page makes of a fit. The first group feeds ``shape`` hand-built
``Inputs`` and checks that every figure is well formed, that the takeaway and the
chip follow the values they were given, and that a figure the values cannot
support becomes a refusal with a reason. The enterprise-value audit gets its own
fakes: a clean audit keeps the screen, and a flagged name on it, or a partial
refit that moves it, refuses it with a reason naming what moved. A third group
reads the committed audit against the committed panel without a fit, so its
counts are held here even before a snapshot is committed. The last group reads
the committed snapshot and holds the section to the numbers the audited runs
reproduced from the committed panel, so a change that moves one of them has to
move this file too.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest

from techval.dashboard.collect import entry_module
from techval.dashboard.sections import warranted as S
from techval.dashboard.snapshot import to_jsonable, validate_section

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "src" / "techval" / "dashboard" / "assets" / "kit.js"
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"

CARD = "comps.py OLS refit on the date, sub-vertical peers"
UNIVERSE = "comps.py OLS refit on the date, whole TMT cross-section"
MEDIAN = "sub-vertical median multiple on the date (the comps convention)"
PERSISTENCE = (
    "the company's own multiple last quarter (persistence: NOT a warranted "
    "multiple, it reads the answer)"
)


def _kit_charts() -> set[str]:
    source = KIT.read_text(encoding="utf-8")
    body = re.search(r"TV\.charts = \{(.*?)\};", source, flags=re.S).group(1)
    return set(re.findall(r"(\w+)\s*:", body))


# The names a refusal carries on the page, keyed by the figure it stands in for.
names = {
    "deflation": "Level against change",
    "folds": "Score by walk-forward fold",
    "noise": "Lift against the noise",
    "screen": "Screen of rich and cheap names",
    "rerating": "The re-rating across dates",
    "audit": "Enterprise-value audit",
    "audit_filers": "Filers the audit moves",
}


def _folds(pairs):
    starts = [date(2022, 12, 31), date(2023, 12, 31), date(2024, 12, 31), date(2025, 12, 31)]
    return tuple(
        S.FoldResult(
            index=i,
            test_start=starts[i],
            test_end=date(starts[i].year + 1, 9, 30),
            n=100 + i,
            model_score=m,
            baseline_score=b,
        )
        for i, (m, b) in enumerate(pairs)
    )


SCREEN_DAY = date(2026, 6, 30)


def _row(ticker, recorded, rebuilt, when=SCREEN_DAY, refused=None, sub_vertical="semiconductors"):
    return S.AuditRow(ticker, when, sub_vertical, recorded, rebuilt, refused)


def _audit(extra=(), skipped=()) -> S.Audit:
    """An audit of the fake screen's names, all identical, plus one stale row on another date.

    The stale row belongs to a name on the screen, but not on the screen's date,
    so it must be counted without costing the screen.
    """
    rows = (
        _row("AAA", 1000.0, 1000.0),
        _row("BBB", 2000.0, 2000.001),
        _row("CCC", 500.0, 500.0, sub_vertical="application_software"),
        _row("DDD", 800.0, 800.0, sub_vertical="application_software"),
        _row("EEE", 900.0, 900.0),
        _row("AAA", 1000.0, 1100.0, when=date(2025, 12, 31)),
        _row("AAA", 1000.0, 1030.0, when=date(2025, 9, 30)),
    )
    return S.Audit(
        recorded="2026-09-14",
        code_commit="0123456789abcdef",
        observations=rows + tuple(extra),
        skipped=tuple(skipped),
    )


def _inputs(**overrides) -> S.Inputs:
    values = dict(
        target_label="EV/Revenue",
        model_kind="mlp",
        metric="spearman",
        higher_is_better=True,
        comparisons=(
            S.Comparison(CARD, 0.70, 0.60, 400, True),
            S.Comparison(MEDIAN, 0.80, 0.55, 700, False),
            S.Comparison(PERSISTENCE, 0.82, 0.95, 750, False),
        ),
        verdict_text="The model's own verdict sentence, carried verbatim.",
        fold_score_sd=0.05,
        folds=_folds([(0.80, 0.50), (0.75, 0.70), (0.60, 0.65)]),
        change=0.03,
        n_changes=750,
        pooled_on_change_rows=0.82,
        screen_date=date(2026, 6, 30),
        screen_names=40,
        screen=(
            S.ScreenRow("AAA", "semiconductors", 20.0, 8.0, 0.9, 1.0),
            S.ScreenRow("BBB", "semiconductors", 15.0, 9.0, 0.5, 0.6),
            S.ScreenRow("CCC", "application_software", 3.0, 9.0, -1.1, -1.2),
            S.ScreenRow("DDD", "application_software", 2.0, 7.0, -1.3, -1.4),
        ),
        date_means=((date(2021, 12, 31), 1.85), (date(2022, 12, 31), 1.35), (date(2023, 12, 31), 1.65)),
        between_date_share=0.021,
        within_date_sd=0.95,
        n_observations=300,
        n_companies=30,
        n_dates=3,
        n_refused=12,
        audit=_audit(),
        refit=S.PartialRefit(
            score=0.71,
            n=401,
            swapped=2,
            screen=(("AAA", True), ("BBB", True), ("CCC", False), ("DDD", False)),
        ),
    )
    values.update(overrides)
    return S.Inputs(**values)


def _numbers(value, path="data"):
    if isinstance(value, bool) or value is None or isinstance(value, str):
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
# Shaping, on fakes
# --------------------------------------------------------------------------- #


def test_every_figure_has_a_kind_the_kit_draws_finite_values_and_labels():
    section = S.shape(_inputs())
    charts = _kit_charts()
    assert section["status"] == "ok"
    assert section["refusals"] == []
    assert list(section["figures"]) == [fid for fid, _, _ in S.FIGURES]
    for fid, fig in section["figures"].items():
        assert fig["kind"] in charts, fid
        assert fig["title"].strip(), fid
        if fig["kind"] != "tiles":
            assert fig["subtitle"].strip(), fid
        for where, number in _numbers(fig["data"]):
            assert math.isfinite(number), f"{fid} {where}"
        data = fig["data"]
        for row in data.get("rows", []):
            assert str(row["label"]).strip(), fid
        for series in data.get("series", []):
            assert series["name"].strip(), fid
        for tile in data.get("tiles", []):
            assert tile["label"].strip(), fid
        table = data.get("table")
        if table:
            keys = {c["key"] for c in table["columns"]}
            assert table["rows"] and all(keys <= set(r) for r in table["rows"]), fid


def test_the_shaped_section_passes_the_snapshot_schema_and_carries_no_em_dash():
    section = S.shape(_inputs())
    full = to_jsonable(
        {
            "id": S.ID,
            "title": S.TITLE,
            **section,
            "provenance": [
                {"figure": fid, "entry_point": entry, "inputs": [], "seconds": 0.0}
                for fid, entry, _ in S.FIGURES
            ],
        }
    )
    validate_section(full)
    assert "\N{EM DASH}" not in json.dumps(full, ensure_ascii=False)


def test_every_entry_point_the_section_cites_resolves():
    for _, entry, _ in S.FIGURES:
        entry_module(entry)


def test_the_takeaway_is_built_from_the_values_it_was_given():
    text = S.takeaway(_inputs())
    assert "0.7000 against 0.6000 on 400 observations" in text
    assert "+0.0300 on 750" in text
    assert "almost all of the ranking is company identity" in text

    changed = S.takeaway(_inputs(change=0.25, n_changes=612))
    assert "+0.2500 on 612" in changed
    assert "survives differencing" in changed
    assert "almost all" not in changed

    losing = _inputs(comparisons=(S.Comparison(CARD, 0.55, 0.60, 400, True),))
    assert "does not rank" in S.takeaway(losing)


def test_the_chip_follows_the_stricter_noise_test():
    # Fold lifts +0.30, +0.05, -0.05: mean 0.10 against a spread of 0.18.
    assert S.verdict_status(_inputs()) == "inside_noise"
    steady = _inputs(folds=_folds([(0.80, 0.60), (0.78, 0.60), (0.79, 0.60)]))
    assert S.verdict_status(steady) == "beats"
    losing = _inputs(comparisons=(S.Comparison(CARD, 0.55, 0.60, 400, True),))
    assert S.verdict_status(losing) == "loses"
    one_fold = _inputs(folds=_folds([(0.80, 0.60)]))
    assert S.verdict_status(one_fold) == "not_significant"


def test_the_headline_is_the_card_and_the_verdict_is_verbatim():
    head = S.headline(_inputs())
    assert head["score"] == 0.70 and head["baseline_score"] == 0.60 and head["n"] == 400
    assert head["baseline_name"] == CARD
    assert head["lift"] == pytest.approx(0.10)
    assert head["verdict_text"] == "The model's own verdict sentence, carried verbatim."


def test_a_stronger_baseline_left_off_the_card_is_marked_and_explained():
    fig = S.build_baselines(_inputs())
    labels = {row["key"]: row["label"] for row in fig["data"]["rows"]}
    assert "the card's baseline" in labels[CARD]
    assert "left off the card" in labels[PERSISTENCE]
    assert "left off the card" not in labels[MEDIAN]
    assert "carried forward" in fig["subtitle"]
    assert [row["key"] for row in fig["data"]["rows"]][0] == CARD
    assert [row["key"] for row in fig["data"]["rows"]][-1] == PERSISTENCE


def test_the_noise_bars_are_each_lift_plus_and_minus_its_own_yardstick():
    fig = S.build_noise(_inputs())
    shipped, strict = fig["data"]["rows"]
    assert shipped["mid"] == pytest.approx(0.10)
    assert shipped["hi"] - shipped["mid"] == pytest.approx(0.05)
    assert strict["mid"] == pytest.approx(0.10)
    assert strict["hi"] - strict["mid"] == pytest.approx(math.sqrt(0.0325))
    calls = [r["call"] for r in fig["data"]["table"]["rows"]]
    assert calls == ["outside the noise, above zero", "inside the noise"]
    assert fig["title"].startswith("Outside the noise by the verdict sentence")


def test_the_screen_is_diverging_and_richest_first():
    fig = S.build_screen(_inputs())
    rows = fig["data"]["rows"]
    assert [r["label"] for r in rows] == ["AAA", "BBB", "CCC", "DDD"]
    assert [r["role"] for r in rows] == ["pos", "pos", "neg", "neg"]
    assert "never that the market is wrong" in fig["subtitle"]
    assert fig["title"].startswith("Semiconductors crowd the rich end")


def test_a_figure_the_values_cannot_support_is_refused_with_a_reason():
    section = S.shape(_inputs(change=None, pooled_on_change_rows=None, n_changes=12, screen=(), folds=_folds([(0.8, 0.6)])))
    refused = {r["what"]: r["why"] for r in section["refusals"]}
    assert set(refused) == {names[k] for k in ("deflation", "folds", "noise", "screen")}
    assert all(why.strip() for why in refused.values())
    assert "12" in refused[names["deflation"]]
    for fid in refused:
        assert fid not in section["figures"]
    assert {"baselines", "rerating", "panel"} <= set(section["figures"])
    assert "could not be computed" in section["takeaway"]


def test_provenance_is_recorded_only_for_the_figures_that_were_built():
    recorded = []

    @contextmanager
    def record(fid, entry):
        yield
        recorded.append((fid, entry))

    section = S.shape(_inputs(screen=()), record=record)
    assert [fid for fid, _ in recorded] == list(section["figures"])
    assert "screen" not in dict(recorded)


def test_every_figure_input_is_declared():
    for inputs in S.FIGURE_INPUTS.values():
        assert set(inputs) <= set(S.INPUTS)
    assert S.AUDIT in S.INPUTS


# --------------------------------------------------------------------------- #
# The enterprise-value audit, on fakes
# --------------------------------------------------------------------------- #


def test_the_audit_is_counted_the_way_the_page_quotes_it():
    audit = _audit(
        extra=(_row("FFF", 400.0, None, refused="not_built"), _row("GGG", 100.0, 101.5)),
        skipped=(
            S.SkippedRow("HHH", SCREEN_DAY, "admitted"),
            S.SkippedRow("III", SCREEN_DAY, "debt_outside_the_ladder"),
        ),
    )
    c = S.audit_counts(audit)
    assert (c.observations, c.compared) == (9, 8)
    # BBB's gap of a thousandth is rounding, not a difference.
    assert c.identical == 5
    # AAA +10% and +3%, GGG +1.5%: three past 1%, one past 5%.
    assert (c.off_noted, c.off_stale, c.filers_stale) == (3, 1, 1)
    assert (c.refused, c.skipped, c.admitted) == (1, 2, 1)

    fig = S.build_audit(_inputs(audit=audit))
    tiles = {t["key"]: t["value"] for t in fig["data"]["tiles"]}
    assert tiles == {"compared": 8, "identical": 5, "off_noted": 3, "off_stale": 1, "refused": 1, "admitted": 1}
    assert fig["title"].endswith("1 of the panel's enterprise values move by more than 5%, at 1 filer")
    assert "0123456" in fig["subtitle"]


def test_the_affected_filers_are_a_share_of_what_was_rebuilt_with_the_median_gap():
    audit = _audit(extra=(_row("AAA", 1000.0, 900.0, when=date(2025, 6, 30)),))
    (aaa,) = S.affected_filers(audit)
    # AAA: four rebuilt observations, +10% and -10% past the threshold.
    assert (aaa.ticker, aaa.compared, aaa.stale) == ("AAA", 4, 2)
    assert aaa.share == pytest.approx(0.5)
    assert aaa.median_gap == pytest.approx(0.0)
    fig = S.build_audit_filers(_inputs(audit=audit))
    (row,) = fig["data"]["rows"]
    assert row["values"]["share"] == pytest.approx(0.5)
    assert fig["data"]["series"][0]["role"] not in {"model", "model-muted"}
    assert fig["data"]["table"]["rows"][0]["largest_gap"] in {"+10%", "−10%"}


def test_a_clean_audit_keeps_the_screen_and_says_so_in_the_takeaway():
    clean = S.Audit("2026-09-14", "0123456", (_row("AAA", 1000.0, 1000.0),), ())
    section = S.shape(_inputs(audit=clean))
    assert "screen" in section["figures"]
    assert {r["what"] for r in section["refusals"]} == {names["audit_filers"]}
    assert "within 5% of the one recorded" in section["takeaway"]
    assert "notes" not in section["figures"]["audit"]["data"]
    assert "notes" not in section["figures"]["panel"]["data"]

    # Stale rows elsewhere, and a refit that moves nothing on the screen, keep it too.
    kept = S.shape(_inputs())
    assert "screen" in kept["figures"]
    assert S.screen_refusal(_inputs()) is None


@pytest.mark.parametrize(
    "extra, skipped, words",
    [
        ((_row("CCC", 500.0, 560.0),), (), "CCC, its enterprise value rebuilds 12% higher"),
        ((_row("DDD", 800.0, None, refused="not_built"),), (), "DDD, refused by today's code"),
        ((), (S.SkippedRow("BBB", SCREEN_DAY, "admitted"),), "BBB, skipped when recorded and admitted"),
    ],
)
def test_a_screen_name_the_audit_flags_refuses_the_screen_naming_it(extra, skipped, words):
    # The audit's rows for the name on the screen's date replace the identical ones.
    base = [r for r in _audit().observations if not any((r.ticker, r.as_of) == (e.ticker, e.as_of) for e in extra)]
    audit = S.Audit("2026-09-14", "0123456", tuple(base) + extra, skipped)
    section = S.shape(_inputs(audit=audit))
    assert "screen" not in section["figures"]
    why = {r["what"]: r["why"] for r in section["refusals"]}[names["screen"]]
    assert why.startswith("1 of the screen's 4 names on 2026-06-30 is flagged")
    assert words in why
    assert "re-recording the panel" in why
    assert "the screen is not drawn" in section["takeaway"]


def test_a_partial_refit_that_changes_the_screen_refuses_it_naming_the_change():
    refit = S.PartialRefit(0.72, 400, 3, (("AAA", True), ("EEE", True), ("CCC", True), ("DDD", False)))
    why = S.screen_refusal(_inputs(refit=refit))
    assert why is not None
    assert "None of the screen's 4 names on 2026-06-30 is itself flagged" in why
    assert "changes 2 of its 4 names: BBB leaves it, EEE joins and CCC changes between rich and cheap" in why
    assert "partial refit" in why


def test_without_an_audit_the_audit_and_the_screen_refuse_and_the_headline_stays():
    section = S.shape(_inputs(audit=None, audit_problem="the audit covers 3 company-dates", refit=None))
    refused = {r["what"]: r["why"] for r in section["refusals"]}
    assert set(refused) == {names[k] for k in ("audit", "audit_filers", "screen")}
    assert "the audit covers 3 company-dates" in refused[names["audit"]]
    assert "cannot be checked" in refused[names["screen"]]
    assert section["headline"]["score"] == 0.70


def test_the_caution_quotes_the_partial_refit_and_names_the_fix():
    audit = _audit(extra=(_row("FFF", 400.0, None, refused="not_built"),))
    fig = S.build_audit(_inputs(audit=audit))
    (note,) = fig["data"]["notes"]
    assert "0.7000 on 400 observations to 0.7100 on 401" in note["why"]
    assert "partial refit" in note["why"] and "understates" in note["why"]
    assert "Re-recording the panel" in note["why"]
    unfitted = S.build_audit(_inputs(audit=audit, refit=None))
    assert "No refit" in unfitted["data"]["notes"][0]["why"]


# --------------------------------------------------------------------------- #
# The committed audit against the committed panel, without a fit
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def committed_audit() -> S.Audit:
    from techval.commands_peers import _load_observations

    fixtures = ROOT / "tests" / "fixtures"
    panel = _load_observations(fixtures / S.PANEL)
    audit, problem = S.read_audit(fixtures / S.AUDIT, panel)
    assert problem is None
    return audit


def test_the_committed_audit_describes_the_committed_panel(committed_audit):
    c = S.audit_counts(committed_audit)
    assert (c.observations, c.compared, c.identical) == (1764, 1713, 1337)
    assert (c.off_noted, c.off_stale, c.filers_stale) == (188, 106, 13)
    assert (c.refused, c.skipped, c.admitted) == (51, 97, 97)
    filers = {f.ticker: (f.stale, f.compared) for f in S.affected_filers(committed_audit)}
    assert filers["VZ"] == (22, 22) and filers["WBD"] == (14, 14) and filers["TMUS"] == (2, 15)


def test_an_audit_of_another_recording_is_not_read_as_evidence(committed_audit, tmp_path):
    import gzip

    from techval.commands_peers import _load_observations

    panel = _load_observations(ROOT / "tests" / "fixtures" / S.PANEL)
    with gzip.open(ROOT / "tests" / "fixtures" / S.AUDIT, "rt") as handle:
        raw = json.load(handle)
    raw["observations"][0]["ev_recorded"] += 50.0
    moved = tmp_path / "ev_audit.json.gz"
    with gzip.open(moved, "wt") as handle:
        json.dump(raw, handle)
    audit, problem = S.read_audit(moved, panel)
    assert audit is None and "do not match" in problem


# --------------------------------------------------------------------------- #
# Pinned truths, against the committed snapshot
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def committed() -> dict:
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    section = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"][S.ID]
    assert section["status"] == "ok", section["refusals"]
    return section


def _at4(value, expected):
    assert round(value, 4) == pytest.approx(expected, abs=1e-9), (value, expected)


def test_the_headline_is_the_card_against_the_incumbent_ols(committed):
    head = committed["headline"]
    _at4(head["score"], 0.7651)
    _at4(head["baseline_score"], 0.6313)
    _at4(head["lift"], 0.1339)
    assert head["n"] == 888
    assert head["baseline_name"] == CARD
    assert head["verdict_status"] == "inside_noise"
    assert "outside the fold-to-fold noise" in head["verdict_text"]


def test_every_baseline_comparison_is_pinned(committed):
    rows = {r["key"]: r for r in committed["figures"]["baselines"]["data"]["rows"]}
    expected = {
        CARD: (0.7651, 0.6313, 888),
        UNIVERSE: (0.8391, 0.4480, 1500),
        MEDIAN: (0.8294, 0.6180, 1441),
        PERSISTENCE: (0.8412, 0.9722, 1548),
    }
    assert set(rows) == set(expected)
    for name, (model, baseline, n) in expected.items():
        _at4(rows[name]["values"]["model"], model)
        _at4(rows[name]["values"]["baseline"], baseline)
        assert rows[name]["n"] == n


def test_the_differenced_score_is_pinned_beside_the_pooled_one(committed):
    rows = {r["key"]: r for r in committed["figures"]["deflation"]["data"]["rows"]}
    _at4(rows["differenced"]["value"], 0.0420)
    _at4(rows["pooled"]["value"], 0.8412)
    assert rows["differenced"]["n"] == rows["pooled"]["n"] == 1548
    assert "+0.0420 on 1,548" in committed["takeaway"]


def test_the_folds_and_both_noise_tests_are_pinned(committed):
    folds = committed["figures"]["folds"]["data"]["rows"]
    for row, expected in zip(folds, (0.8086, 0.7981, 0.7720, 0.8515, 0.6306)):
        _at4(row["values"]["model"], expected)
    lifts = [r["values"]["model"] - r["values"]["baseline"] for r in folds]
    assert len(folds) == 5 and sum(1 for v in lifts if v > 0) == 4

    noise = {r["key"]: r for r in committed["figures"]["noise"]["data"]["rows"]}
    _at4(noise["shipped"]["mid"], 0.1339)
    _at4(noise["shipped"]["hi"] - noise["shipped"]["mid"], 0.0841)
    _at4(noise["strict"]["mid"], 0.1438)
    _at4(noise["strict"]["hi"] - noise["strict"]["mid"], 0.1736)


def test_the_panel_and_the_rerating_are_pinned(committed):
    tiles = {t["key"]: t["value"] for t in committed["figures"]["panel"]["data"]["tiles"]}
    assert tiles == {"observations": 1764, "companies": 94, "dates": 22, "refused": 546}
    rerating = committed["figures"]["rerating"]["data"]
    assert round(rerating["between_date_share"], 3) == pytest.approx(0.021)
    assert len(rerating["series"][0]["values"]) == 22


def test_the_audit_opens_the_section_with_the_committed_counts(committed):
    audit = committed["figures"]["audit"]
    tiles = {t["key"]: t["value"] for t in audit["data"]["tiles"]}
    assert tiles == {
        "compared": 1713,
        "identical": 1337,
        "off_noted": 188,
        "off_stale": 106,
        "refused": 51,
        "admitted": 97,
    }
    assert audit["title"].endswith("106 of the panel's enterprise values move by more than 5%, at 13 filers")
    filers = {r["key"]: r["values"]["share"] for r in committed["figures"]["audit_filers"]["data"]["rows"]}
    assert len(filers) == 13 and filers["VZ"] == 1.0 and filers["WBD"] == 1.0
    assert "106 of its 1,713 enterprise values move by more than 5%" in committed["takeaway"]


def test_the_caution_quotes_the_partial_refit_of_the_headline(committed):
    (note,) = committed["figures"]["audit"]["data"]["notes"]
    assert "from 0.7651 on 888 observations to 0.7720 on 889" in note["why"]
    assert "partial refit" in note["why"]


def test_the_screen_is_refused_because_the_partial_refit_moves_it(committed):
    assert "screen" not in committed["figures"]
    why = {r["what"]: r["why"] for r in committed["refusals"]}["screen"]
    assert "None of the screen's 16 names on 2026-06-30 is itself flagged" in why
    assert "(DLR, EBAY, TMUS, VZ and WBD)" in why
    assert "changes 1 of its 16 names: TXN leaves it and PLTR joins" in why


def test_the_takeaway_reads_the_lift_the_way_the_chip_does():
    """A reader who stops at the takeaway should not have to reconcile it with the square beside it.

    The chip judges the pooled lift against the spread of the per-fold lifts, so
    a takeaway that says only "better than" contradicts an INSIDE NOISE square
    two lines below it.
    """
    quiet = _inputs()
    assert S.verdict_status(quiet) == "inside_noise"
    assert "a lift inside the spread of the per-fold lifts" in S.takeaway(quiet)

    steady = _inputs(folds=_folds([(0.80, 0.60), (0.78, 0.60), (0.79, 0.60)]))
    assert S.verdict_status(steady) == "beats"
    assert "a lift outside the spread of the per-fold lifts" in S.takeaway(steady)
