"""The reading section: the readings side by side until a key and recordings exist, a scored comparison after."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from techval.dashboard import collect as C
from techval.dashboard.collect import CollectContext, entry_module
from techval.dashboard.sections import reading as S
from techval.dashboard.snapshot import to_jsonable, validate_section
from techval.rag.key import KeyRow, KeyState
from techval.rag.passages import Passage
from techval.rag.readings import Reading
from techval.rag.recording import Recording
from techval.rag.run import Prepared, Run
from techval.rag.tasks import Task

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
UNRESOLVED = "NET: the 10-K text rag/filing_text_NET_2025.json.gz is not committed"


def _task(i: int) -> Task:
    return Task(f"T{i}:customers", "kpi", f"T{i}", "customers", f"A{i}", "10-K", date(2026, 1, 1), "7", None, None)


def _run(claude_status: str = "stated") -> Run:
    tasks = [_task(i) for i in range(4)]
    prepared = [
        Prepared(t, "x" * 200, 0, (Passage(f"{t.accession}:0", t.accession, "7", 0, 100, "x" * 100),))
        for t in tasks
    ]

    def stated(t, reader, right):
        return Reading(t.task_id, reader, "stated", value=100.0 if right else 150.0, unit="count")

    claude = {
        t.task_id: stated(t, "claude", True) if claude_status == "stated"
        else Reading(t.task_id, "claude", claude_status, reason="not run")
        for t in tasks
    }
    readings = {
        "claude": claude,
        "regex_retrieved": {t.task_id: stated(t, "regex_retrieved", i < 2) for i, t in enumerate(tasks)},
        "regex_native": {t.task_id: stated(t, "regex_native", i < 1) for i, t in enumerate(tasks)},
    }
    recordings = {}
    if claude_status == "stated":
        recordings = {
            t.task_id: Recording(
                "k", t.task_id, {}, {}, "claude-opus-5",
                {"input_tokens": 100, "output_tokens": 20}, "2026-09-16", "batch", "1.6.0",
            )
            for t in tasks
        }
    return Run(prepared, [UNRESOLVED], readings, recordings)


def _key() -> KeyState:
    return KeyState(
        rows={
            f"T{i}:customers": KeyRow(f"T{i}:customers", "stated", value=100.0, resolution=1.0, span=(10, 20))
            for i in range(4)
        }
    )


def _as_section(shaped: dict) -> dict:
    section = to_jsonable(
        {
            "id": S.ID,
            "title": S.TITLE,
            "provenance": [
                {"figure": f, "entry_point": S.ENTRY_POINTS[f], "inputs": [], "seconds": 0.0}
                for f in shaped["figures"]
            ],
            **shaped,
        }
    )
    validate_section(section)
    return section


def _why(section: dict) -> str:
    return " ".join(r["why"] for r in section["refusals"])


def test_a_scored_section_carries_a_headline_and_all_four_figures():
    section = _as_section(S.shape(_run(), _key(), 6))
    assert set(section["figures"]) == {"readings", "errors", "retrieval", "recording"}
    head = section["headline"]
    assert (head["metric"], head["score"], head["baseline_score"], head["n"]) == ("accuracy", 1.0, 0.5, 4)
    assert head["verdict_status"] == "not_significant" and "McNemar" in head["verdict_text"]
    assert section["takeaway"].startswith("On 4 filing facts the Claude reader is right on 4 and the regex readers on 2")
    table = section["figures"]["readings"]["data"]
    assert [c["key"] for c in table["columns"]] == [
        "subject", "fact", "key", "claude", "regex_retrieved", "regex_native", "verdict",
    ]
    assert (table["rows"][0]["claude"], table["rows"][3]["regex_native"]) == ("100", "150")
    assert [r["what"] for r in section["refusals"]] == ["Filings not yet committed"]


def test_without_a_key_the_readings_stand_and_nothing_is_scored():
    section = _as_section(S.shape(_run(), None, 6))
    assert section["headline"] is None
    assert set(section["figures"]) == {"readings", "recording"}
    assert "No answer key is committed" in _why(section)
    assert "key" not in [c["key"] for c in section["figures"]["readings"]["data"]["columns"]]
    assert "no answer key is committed yet" in section["takeaway"]


def test_without_recordings_the_claude_reader_is_named_as_missing():
    section = _as_section(S.shape(_run(claude_status="missing"), _key(), 6))
    assert section["headline"] is None and "recording" not in section["figures"]
    assert "4 of 4 requests have no recording" in _why(section)
    assert "techval rag record" in _why(section)
    assert section["figures"]["readings"]["data"]["rows"][0]["claude"] == "not recorded"


def test_an_incomplete_key_says_what_is_wrong_with_it():
    key = KeyState(unfilled=["T0:customers"], problems=["T1:customers: the quote is not in the filing"])
    why = _why(_as_section(S.shape(_run(), key, 6)))
    assert "1 blank row" in why and "the quote is not in the filing" in why


def test_values_are_shown_in_their_units():
    assert S.show(Reading("T", "claude", "stated", value=1.2e9, unit="usd"), "usd") == "$1.20bn"
    hedged = Reading("T", "claude", "stated", value=120.0, unit="percent", hedged=True)
    assert S.show(hedged, "percent") == "120% (hedged)"
    assert S.show(Reading("T", "claude", "stated", value=7.4, unit="usd_per_share"), "usd_per_share") == "$7.40"
    assert S.show(Reading("T", "claude", "refused", reason="two ratios"), "ratio") == "refused: two ratios"
    assert S.show(Reading("T", "claude", "not_stated"), "count") == "not stated"
    assert S.show_key(KeyRow("T", "ambiguous"), "ratio") == "ambiguous"


def test_collect_runs_offline_on_the_committed_fixtures():
    section, run = C.collect_section(S, CollectContext(root=FIXTURES, repo_root=ROOT), use_cache=False)
    assert run.status == "ok"
    assert len(section["figures"]["readings"]["data"]["rows"]) >= 49
    assert {r["figure"] for r in section["provenance"]} == set(section["figures"])
    for row in section["provenance"]:
        entry_module(row["entry_point"])
