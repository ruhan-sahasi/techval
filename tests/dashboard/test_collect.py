"""Collection: refusals caught, bugs propagated, provenance enforced, cache keyed on evidence.

Every test here installs stub section modules written to a temporary directory
in place of the real ones. The real modules are replaced wholesale as each
section is built and some of them fit models for minutes, so a collection test
that ran them would stop being fast the day its section landed, and would stop
testing the collector.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import textwrap
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.dashboard import collect as C
from techval.dashboard.sections import (
    COLLECT_ORDER,
    DEPENDENT_SECTIONS,
    MODEL_SECTIONS,
    SECTION_IDS,
    section_module,
)
from techval.errors import ConfigError

_serial = itertools.count()

NOT_BUILT = """
CALLS = []

def collect(ctx):
    CALLS.append(1)
    return {"status": "not_built"}
"""


class Sections:
    """Stub section modules, one per id, rewritable mid-test."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.dir = tmp_path / "section_modules"
        self.dir.mkdir()
        self.modules: dict = {}
        real = section_module
        monkeypatch.setattr(
            C, "section_module", lambda sid: self.modules[sid] if sid in self.modules else real(sid)
        )
        for sid in SECTION_IDS:
            self.install(sid, NOT_BUILT)

    def install(self, sid: str, body: str, inputs: list[str] | None = None, title: str | None = None):
        path = self.dir / f"{sid}.py"
        path.write_text(
            f"ID = {sid!r}\nTITLE = {title or sid.title()!r}\nINPUTS = {inputs or []!r}\n"
            + textwrap.dedent(body)
        )
        spec = importlib.util.spec_from_file_location(f"_stub_section_{sid}_{next(_serial)}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.modules[sid] = module
        return module


@pytest.fixture
def sections(tmp_path, monkeypatch) -> Sections:
    return Sections(tmp_path, monkeypatch)


@pytest.fixture
def root(tmp_path) -> Path:
    r = tmp_path / "fixtures"
    r.mkdir()
    (r / "panel.json").write_text('{"x": 1}')
    (r / "prices").mkdir()
    (r / "prices" / "DDOG.csv").write_text("date,close\n2026-09-10,120.0\n")
    return r


@pytest.fixture
def assumptions(tmp_path) -> Assumptions:
    a = Assumptions()
    a.ml.cache_dir = str(tmp_path / "cache")
    return a


def _collect(root, assumptions, **kw):
    runs: list[C.SectionRun] = []
    snap = C.collect_snapshot(
        root,
        "2026-09-14",
        assumptions=assumptions,
        repo_root=root.parent,
        on_section=runs.append,
        **kw,
    )
    return snap, {r.id: r for r in runs}


FIGURE = """
CALLS = []

def collect(ctx):
    CALLS.append(1)
    with ctx.record("scores", "techval.dashboard.snapshot.to_jsonable", ["panel.json"]):
        value = len(ctx.input("panel.json").read_text())
    return {
        "status": "ok",
        "takeaway": "A figure computed from the panel.",
        "figures": {"scores": {"kind": "hbar", "title": "Scores", "data": {"n": value}}},
    }
"""


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #


def test_every_section_id_has_a_module():
    assert len(set(SECTION_IDS)) == len(SECTION_IDS) == 10
    for sid in SECTION_IDS:
        module = section_module(sid)
        assert module.ID == sid
        assert module.TITLE.strip()
        assert all(isinstance(p, str) for p in module.INPUTS)
        assert callable(module.collect)


def test_collect_order_runs_the_dependent_sections_last():
    assert sorted(COLLECT_ORDER) == sorted(SECTION_IDS)
    assert COLLECT_ORDER[-2:] == ("sample", "overview")
    assert tuple(DEPENDENT_SECTIONS) == ("sample", "overview")
    assert set(MODEL_SECTIONS) == {"signal", "encoder", "warranted", "fade", "propensity"}


def test_unknown_section_is_a_config_error_naming_the_valid_ids():
    with pytest.raises(ConfigError, match="overview, signal"):
        section_module("scoreboard")


# --------------------------------------------------------------------------- #
# assembly, refusals and bugs
# --------------------------------------------------------------------------- #


def test_collects_every_section_in_order(sections, root, assumptions):
    order: list[str] = []
    snap, runs = _collect(root, assumptions)
    snap2 = C.collect_snapshot(
        root, "2026-09-14", assumptions=assumptions, repo_root=root.parent,
        use_cache=False, on_section=lambda r: order.append(r.id),
    )
    assert order == list(COLLECT_ORDER)
    assert list(snap.sections) == list(SECTION_IDS)
    assert snap.collected_at == "2026-09-14" and snap2.collected_at == "2026-09-14"
    assert snap.title == "Valuation engine and model results"
    for sid, section in snap.sections.items():
        assert section["id"] == sid
        assert section["title"] == sid.title()
        assert section["status"] == "not_built"
        assert section["headline"] is None
        assert section["refusals"] == [] and section["figures"] == {} and section["provenance"] == []


def test_a_techval_error_becomes_a_refusal_with_its_reason(sections, root, assumptions):
    sections.install(
        "fade",
        """
        from techval.errors import MissingDataError

        def collect(ctx):
            raise MissingDataError("fade panel", hint="record it with the fade recorder")
        """,
        title="Revenue fade",
    )
    snap, runs = _collect(root, assumptions)
    fade = snap.sections["fade"]
    from techval.errors import MissingDataError

    expected = str(MissingDataError("fade panel", hint="record it with the fade recorder"))
    assert fade["status"] == "refused"
    assert fade["refusals"] == [{"what": "Revenue fade", "why": expected}]
    assert fade["figures"] == {} and fade["headline"] is None
    assert runs["fade"].status == "refused"
    # The rest of the page still collects.
    assert snap.sections["encoder"]["status"] == "not_built"


def test_any_other_exception_propagates(sections, root, assumptions):
    sections.install(
        "encoder",
        """
        def collect(ctx):
            raise ValueError("a bug, not a refusal")
        """,
    )
    with pytest.raises(ValueError, match="a bug, not a refusal"):
        _collect(root, assumptions)


def test_a_missing_declared_input_is_refused_by_name(sections, root, assumptions):
    sections.install("tmt", NOT_BUILT, inputs=["absent.json"])
    snap, runs = _collect(root, assumptions)
    assert snap.sections["tmt"]["status"] == "refused"
    assert "absent.json" in snap.sections["tmt"]["refusals"][0]["why"]
    assert runs["tmt"].cache == "n/a"
    assert sections.modules["tmt"].CALLS == []


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def test_record_writes_repository_relative_provenance_and_digests(sections, root, assumptions):
    sections.install("encoder", FIGURE, inputs=["panel.json", "prices"])
    snap, _ = _collect(root, assumptions)
    enc = snap.sections["encoder"]
    assert enc["status"] == "ok"
    assert enc["figures"]["scores"]["data"] == {"n": 8}
    [row] = enc["provenance"]
    assert row["figure"] == "scores"
    assert row["entry_point"] == "techval.dashboard.snapshot.to_jsonable"
    assert row["inputs"] == ["fixtures/panel.json"]
    assert isinstance(row["seconds"], float)
    assert set(snap.fixtures) == {"fixtures/panel.json", "fixtures/prices/DDOG.csv"}


@pytest.mark.parametrize(
    "body, message",
    [
        (
            """
            def collect(ctx):
                with ctx.record("f", "techval.dashboard.snapshot.to_jsonable", ["other.json"]):
                    pass
                return {"status": "ok", "figures": {"f": {"kind": "hbar", "title": "t", "data": {}}}}
            """,
            "not declared in its INPUTS",
        ),
        (
            """
            def collect(ctx):
                with ctx.record("f", "techval.dashboard.snapshot.no_such_function", []):
                    pass
                return {"status": "ok", "figures": {"f": {"kind": "hbar", "title": "t", "data": {}}}}
            """,
            "has no attribute",
        ),
        (
            """
            def collect(ctx):
                with ctx.record("f", "techval.no_such_module.fit", []):
                    pass
                return {"status": "ok", "figures": {"f": {"kind": "hbar", "title": "t", "data": {}}}}
            """,
            "has no attribute|importable",
        ),
        (
            """
            def collect(ctx):
                return {"status": "ok", "figures": {"f": {"kind": "hbar", "title": "t", "data": {}}}}
            """,
            "no provenance for figure",
        ),
        (
            """
            def collect(ctx):
                return {"status": "ok", "provenance": []}
            """,
            "may return only",
        ),
    ],
)
def test_provenance_rules_are_bugs_not_refusals(sections, root, assumptions, body, message):
    sections.install("encoder", body, inputs=["panel.json"])
    with pytest.raises(ValueError, match=message):
        _collect(root, assumptions)


def test_only_model_sections_carry_a_headline(sections, root, assumptions):
    headline = {
        "metric": "auc", "score": 0.6, "baseline_name": "base rate", "baseline_score": 0.5,
        "lift": 0.1, "n": 100, "higher_is_better": True, "verdict_status": "beats",
        "verdict_text": "auc of 0.6000 against 0.5000 for base rate.",
    }
    body = f"""
    def collect(ctx):
        return {{"status": "ok", "headline": {headline!r}}}
    """
    sections.install("propensity", body)
    snap, _ = _collect(root, assumptions)
    assert snap.sections["propensity"]["headline"]["verdict_status"] == "beats"

    sections.install("engine", body)
    with pytest.raises(ValueError, match="only the model sections"):
        _collect(root, assumptions)


def test_dependent_sections_read_earlier_results(sections, root, assumptions):
    sections.install(
        "overview",
        """
        SEEN = []

        def collect(ctx):
            SEEN.append(sorted(ctx.results))
            ctx.results["encoder"]["status"] = "tampered"
            return {"status": "not_built"}
        """,
    )
    snap, _ = _collect(root, assumptions)
    [seen] = sections.modules["overview"].SEEN
    assert seen == sorted(s for s in SECTION_IDS if s != "overview")
    assert snap.sections["encoder"]["status"] == "not_built"


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #


def test_second_collect_hits_the_cache(sections, root, assumptions, tmp_path):
    sections.install("encoder", FIGURE, inputs=["panel.json"])
    first, runs1 = _collect(root, assumptions)
    assert {r.cache for r in runs1.values()} == {"miss"}
    assert len(list((tmp_path / "cache" / "dashboard").glob("encoder-*.json"))) == 1

    second, runs2 = _collect(root, assumptions)
    assert {r.cache for r in runs2.values()} == {"hit"}
    assert sections.modules["encoder"].CALLS == [1]
    assert second.sections == first.sections


def test_touching_a_declared_input_misses_exactly_that_section(sections, root, assumptions):
    sections.install("encoder", FIGURE, inputs=["panel.json"])
    sections.install("tmt", NOT_BUILT, inputs=["prices"])
    _collect(root, assumptions)

    (root / "panel.json").write_text('{"x": 2}')
    _, runs = _collect(root, assumptions)
    assert runs["encoder"].cache == "miss"
    assert runs["tmt"].cache == "hit"
    assert runs["signal"].cache == "hit"
    assert sections.modules["encoder"].CALLS == [1, 1]

    (root / "prices" / "DDOG.csv").write_text("date,close\n2026-09-10,121.0\n")
    _, runs = _collect(root, assumptions)
    assert runs["tmt"].cache == "miss" and runs["encoder"].cache == "hit"


def test_a_changed_collector_misses_its_cache(sections, root, assumptions):
    _collect(root, assumptions)
    sections.install("warranted", NOT_BUILT + "\n# edited\n")
    _, runs = _collect(root, assumptions)
    assert runs["warranted"].cache == "miss"
    assert runs["fade"].cache == "hit"


def test_a_changed_entry_point_module_misses_the_cache(sections, root, assumptions, tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "stub_model_for_dashboard.py").write_text("def fit():\n    return 1\n")
    monkeypatch.syspath_prepend(str(lib))
    sections.install(
        "signal",
        FIGURE.replace("techval.dashboard.snapshot.to_jsonable", "stub_model_for_dashboard.fit"),
        inputs=["panel.json"],
    )
    _collect(root, assumptions)
    _, runs = _collect(root, assumptions)
    assert runs["signal"].cache == "hit"

    (lib / "stub_model_for_dashboard.py").write_text("def fit():\n    return 2  # fixed\n")
    _, runs = _collect(root, assumptions)
    assert runs["signal"].cache == "miss"


def test_a_changed_result_misses_the_dependent_sections(sections, root, assumptions):
    sections.install("encoder", FIGURE, inputs=["panel.json"])
    _collect(root, assumptions)
    (root / "panel.json").write_text('{"x": 22}')
    _, runs = _collect(root, assumptions)
    assert runs["encoder"].cache == "miss"
    assert runs["sample"].cache == "miss" and runs["overview"].cache == "miss"
    assert runs["fade"].cache == "hit"


def test_refusals_are_not_cached(sections, root, assumptions):
    sections.install(
        "fade",
        """
        from techval.errors import DataSourceError
        CALLS = []

        def collect(ctx):
            CALLS.append(1)
            raise DataSourceError("no panel")
        """,
    )
    _collect(root, assumptions)
    _collect(root, assumptions)
    assert sections.modules["fade"].CALLS == [1, 1]


def test_no_cache_bypasses_it(sections, root, assumptions, tmp_path):
    sections.install("encoder", FIGURE, inputs=["panel.json"])
    _collect(root, assumptions)
    _, runs = _collect(root, assumptions, use_cache=False)
    assert {r.cache for r in runs.values()} == {"off"}
    assert sections.modules["encoder"].CALLS == [1, 1]


def test_an_unreadable_cache_entry_is_a_miss(sections, root, assumptions, tmp_path):
    sections.install("encoder", FIGURE, inputs=["panel.json"])
    _collect(root, assumptions)
    [entry] = (tmp_path / "cache" / "dashboard").glob("encoder-*.json")
    entry.write_text("{truncated")
    _, runs = _collect(root, assumptions)
    assert runs["encoder"].cache == "miss"
    json.loads(entry.read_text())  # rewritten whole


# --------------------------------------------------------------------------- #
# merging named sections into an existing snapshot
# --------------------------------------------------------------------------- #


def test_named_sections_merge_and_rerun_their_dependents(sections, root, assumptions):
    sections.install("encoder", FIGURE, inputs=["panel.json"])
    base, _ = _collect(root, assumptions, use_cache=False)

    snap, runs = _collect(root, assumptions, sections=["encoder"], base=base, use_cache=False)
    assert list(runs) == ["encoder", "sample", "overview"]
    assert list(snap.sections) == list(SECTION_IDS)
    assert snap.sections["fade"] == base.sections["fade"]
    assert snap.fixtures == base.fixtures


def test_merge_refuses_a_kept_section_whose_fixtures_changed(sections, root, assumptions):
    sections.install("encoder", FIGURE, inputs=["panel.json"])
    base, _ = _collect(root, assumptions, use_cache=False)
    (root / "panel.json").write_text('{"x": 3}')
    with pytest.raises(ConfigError, match="'encoder'.*fixtures/panel.json"):
        _collect(root, assumptions, sections=["fade"], base=base, use_cache=False)


def test_unknown_named_section_is_a_config_error(sections, root, assumptions):
    with pytest.raises(ConfigError, match="unknown dashboard section"):
        _collect(root, assumptions, sections=["nope"])


def test_collected_at_is_validated_and_the_commit_falls_back(root, tmp_path):
    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        C.collect_snapshot(root, "14/09/2026", repo_root=tmp_path)
    assert C.techval_commit(tmp_path) == "unknown"
