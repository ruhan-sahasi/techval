"""Scoring: both readers against the key, on the same tasks, with their disagreements tested exactly."""

from __future__ import annotations

from datetime import date

import pytest

from techval.rag.evaluate import (
    IncompleteRun,
    compare,
    error_kind,
    is_correct,
    mcnemar_p,
    recall_at_k,
    verdict_text,
)
from techval.rag.key import KeyRow, KeyState
from techval.rag.passages import Passage
from techval.rag.readings import Reading
from techval.rag.run import Prepared, Run
from techval.rag.tasks import Task

RIGHT = ("stated", 100.0)
WRONG = ("stated", 150.0)
MISS = ("not_stated",)


def _task(i: int) -> Task:
    return Task(f"T{i:02d}:customers", "kpi", f"T{i:02d}", "customers", f"A{i}", "10-K", date(2026, 1, 1), "7", None, None)


def _reading(task: Task, reader: str, status: str, value: float | None = None) -> Reading:
    return Reading(task.task_id, reader, status, value=value, unit="count" if value is not None else None)


def _build(claude, native, retrieved=None, key=None):
    tasks = [_task(i) for i in range(len(claude))]
    passage = lambda t: Passage(f"{t.accession}:0", t.accession, "7", 0, 100, "x" * 100)  # noqa: E731
    prepared = [Prepared(t, "x" * 200, 0, (passage(t),)) for t in tasks]
    retrieved = retrieved or native
    readings = {
        "claude": {t.task_id: _reading(t, "claude", *c) for t, c in zip(tasks, claude)},
        "regex_native": {t.task_id: _reading(t, "regex_native", *c) for t, c in zip(tasks, native)},
        "regex_retrieved": {t.task_id: _reading(t, "regex_retrieved", *c) for t, c in zip(tasks, retrieved)},
    }
    key = key or [RIGHT] * len(tasks)
    rows = {
        t.task_id: KeyRow(
            t.task_id, status, value=value,
            resolution=1.0 if value is not None else None,
            span=(10, 20) if status == "stated" else None,
        )
        for t, (status, value) in zip(tasks, key)
    }
    return Run(prepared, [], readings), KeyState(rows=rows)


def test_a_value_within_half_the_quote_s_last_digit_is_right():
    key = KeyRow("T", "stated", value=100.0, resolution=1.0)
    assert is_correct(Reading("T", "claude", "stated", value=100.4, unit="count"), key, "count")
    assert not is_correct(Reading("T", "claude", "stated", value=100.6, unit="count"), key, "count")


def test_a_refusal_is_right_only_where_the_key_states_nothing():
    refused = Reading("T", "claude", "refused", reason="declined")
    assert is_correct(refused, KeyRow("T", "not_stated"), "count")
    assert is_correct(refused, KeyRow("T", "ambiguous"), "count")
    assert not is_correct(refused, KeyRow("T", "stated", value=1.0, resolution=1.0), "count")


def test_text_folds_case_and_punctuation_and_dates_do_not_fold():
    name = KeyRow("T", "stated", text_value="Neon Maple Parent Inc.")
    assert is_correct(Reading("T", "claude", "stated", text_value="neon maple parent inc", unit="text"), name, "text")
    day = KeyRow("T", "stated", text_value="2026-06-12")
    assert not is_correct(Reading("T", "claude", "stated", text_value="2026-06-13", unit="date"), day, "date")


def test_every_error_has_a_kind():
    stated = KeyRow("T", "stated", value=100.0, resolution=1.0)
    absent = KeyRow("T", "not_stated")
    assert error_kind(Reading("T", "claude", "stated", value=150.0, unit="count"), stated, "count") == "wrong_value"
    assert error_kind(Reading("T", "claude", "not_stated"), stated, "count") == "missed"
    assert error_kind(Reading("T", "claude", "stated", value=5.0, unit="count"), absent, "count") == "invented"
    assert error_kind(Reading("T", "claude", "ambiguous"), absent, "count") == "wrong_status"
    assert error_kind(Reading("T", "claude", "not_stated"), absent, "count") is None


def test_mcnemar_is_exact_and_two_sided():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(8, 0) == pytest.approx(2 * 0.5**8)
    assert mcnemar_p(3, 3) == pytest.approx(1.0)
    assert mcnemar_p(8, 2) == pytest.approx(112 / 1024)


def test_the_comparison_counts_the_disagreements_and_calls_the_verdict():
    claude = [RIGHT] * 10 + [RIGHT] * 8 + [WRONG] * 2
    native = [RIGHT] * 10 + [WRONG] * 8 + [RIGHT] * 2
    c = compare(*_build(claude, native))
    assert (c.claude.correct, c.regex.correct, c.claude_only, c.regex_only) == (18, 12, 8, 2)
    assert c.mcnemar_p == pytest.approx(112 / 1024)
    assert c.verdict_status == "not_significant"
    assert (c.claude.errors["wrong_value"], c.regex.errors["wrong_value"]) == (2, 8)
    assert c.verdicts["T10:customers"] == "Claude only" and c.verdicts["T00:customers"] == "both right"
    e = c.evaluation
    assert (e.metric, e.baseline_name, e.fold_unit, e.paired.n) == ("accuracy", "regex readers", "query", 20)
    text = verdict_text(c)
    assert text.startswith("accuracy of 0.9000 against 0.6000 for regex readers")
    assert "the Claude reader alone is right on 8 and the regex readers alone on 2" in text
    assert "McNemar p of 0.109" in text


def test_a_one_sided_run_of_wins_is_significant_either_way():
    assert compare(*_build([RIGHT] * 10, [WRONG] * 10)).verdict_status == "beats"
    assert compare(*_build([WRONG] * 10, [RIGHT] * 10)).verdict_status == "loses"


def test_the_stronger_regex_run_is_the_baseline():
    c = compare(*_build([RIGHT] * 4, [WRONG] * 4, [RIGHT] * 3 + [WRONG]))
    assert (c.regex.reader, c.regex.correct) == ("regex_retrieved", 3)
    assert set(c.regex_runs) == {"regex_native", "regex_retrieved"}


def test_a_run_with_unrecorded_readings_is_not_scored():
    with pytest.raises(IncompleteRun):
        compare(*_build([("missing",), RIGHT], [RIGHT, RIGHT]))


def test_an_incomplete_key_is_not_scored():
    run, key = _build([RIGHT], [RIGHT])
    key.unfilled.append("T99:customers")
    with pytest.raises(IncompleteRun):
        compare(run, key)


def test_recall_counts_the_stated_rows_whose_quote_a_retrieved_passage_covers():
    run, key = _build([RIGHT, RIGHT, MISS], [RIGHT, RIGHT, MISS], key=[RIGHT, RIGHT, ("not_stated", None)])
    tid = run.prepared[1].task.task_id
    rows = dict(key.rows)
    rows[tid] = KeyRow(tid, "stated", value=100.0, resolution=1.0, span=(150, 160))
    assert recall_at_k(run, KeyState(rows=rows)) == (0.5, 1, 2)
