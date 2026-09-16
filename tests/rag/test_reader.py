"""The two readers: a recorded Claude reader held to its quotes, and the regex rules as they stand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.rag.passages import Passage, chunk
from techval.rag.reader import (
    ANSWER_SCHEMA,
    ClaudeReader,
    RegexReader,
    build_request,
    parse_response,
    system_prompt,
)
from techval.rag.recording import Recording, RecordingMissing, request_key
from techval.rag.store import FilingStore
from techval.rag.tasks import KPI_DOCUMENTS, build_tasks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SETTINGS = Assumptions().ml.rag
PAYO_SENTENCE = "will be converted into the right to receive $7.40 in cash, without interest"


@pytest.fixture(scope="module")
def store() -> FilingStore:
    return FilingStore.from_fixtures(FIXTURES, annual_reports=KPI_DOCUMENTS.values())


@pytest.fixture(scope="module")
def tasks(store):
    return {t.task_id: t for t in build_tasks(store)[0]}


def _answer(**fields) -> dict:
    base = dict(
        status="stated", value=7.4, text_value=None, unit="usd_per_share", period_end=None,
        passage_id="A:0", quote="the right to receive $7.40 in cash", hedged=False,
        reason="the announcement states it",
    )
    base.update(fields)
    return base


def _response(answer, stop: str = "end_turn") -> dict:
    return {
        "model": "claude-opus-5",
        "stop_reason": stop,
        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": json.dumps(answer)}],
    }


def _recording(task_id: str, response: dict, served_by: str = "claude-opus-5") -> Recording:
    return Recording("k" * 64, task_id, {}, response, served_by, {}, "2026-09-16", "batch", "1.6.0")


class _Replayer:
    def __init__(self, stored: Recording | None = None) -> None:
        self.stored = stored
        self.asked: list[dict] = []

    def recording(self, task_id, params):
        self.asked.append(params)
        if self.stored is None:
            raise RecordingMissing(task_id, request_key(params))
        return self.stored


def test_the_request_has_the_spec_s_shape(tasks):
    passages = [Passage("A:0", "A", None, 0, 10, "Some text.")]
    params = build_request(tasks["PAYO:cash_per_share"], passages, SETTINGS)
    assert (params["model"], params["max_tokens"]) == ("claude-opus-5", 16000)
    assert "thinking" not in params
    [system] = params["system"]
    assert system["cache_control"] == {"type": "ephemeral"} and system["text"] == system_prompt()
    assert params["output_config"] == {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
    [message] = params["messages"]
    assert message["role"] == "user"
    assert '<passage id="A:0">' in message["content"]
    assert "How much cash does each share" in message["content"]


def test_every_task_shares_one_cacheable_system_prompt(tasks):
    passages = [Passage("A:0", "A", None, 0, 4, "Text")]
    assert len({json.dumps(build_request(t, passages, SETTINGS)["system"]) for t in tasks.values()}) == 1


def test_the_schema_requires_every_field_and_allows_no_other():
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert set(ANSWER_SCHEMA["required"]) == set(ANSWER_SCHEMA["properties"])


def test_a_well_formed_answer_becomes_a_reading_with_its_model(tasks):
    task = tasks["PAYO:cash_per_share"]
    reading = parse_response(task, _recording(task.task_id, _response(_answer()), served_by="claude-opus-4-8"))
    assert (reading.status, reading.value, reading.unit, reading.served_by, reading.reader) == (
        "stated", 7.4, "usd_per_share", "claude-opus-4-8", "claude",
    )


@pytest.mark.parametrize(
    "response,why",
    [
        ({"model": "m", "stop_reason": "refusal", "stop_details": {"category": "cyber"}, "content": []}, "declined"),
        (_response(_answer(), stop="max_tokens"), "max_tokens"),
        ({"model": "m", "stop_reason": "end_turn", "content": [{"type": "text", "text": "no"}]}, "not JSON"),
        (_response({"status": "stated"}), "schema"),
        (_response(_answer(status="probably")), "schema"),
        (_response(_answer(value="7.40")), "schema"),
        (_response(_answer(unit="dollars")), "schema"),
    ],
)
def test_a_declined_or_malformed_answer_is_refused(tasks, response, why):
    task = tasks["PAYO:cash_per_share"]
    reading = parse_response(task, _recording(task.task_id, response))
    assert reading.status == "refused" and why in reading.reason


def test_no_passages_means_no_request_and_no_statement(tasks):
    replayer = _Replayer()
    reading, recording = ClaudeReader(replayer, SETTINGS).read(tasks["PAYO:cash_per_share"], [])
    assert reading.status == "not_stated" and recording is None and replayer.asked == []


def test_an_unrecorded_request_is_missing_rather_than_answered(tasks):
    passages = [Passage("A:0", "A", None, 0, 4, "Text")]
    reading, recording = ClaudeReader(_Replayer(), SETTINGS).read(tasks["PAYO:cash_per_share"], passages)
    assert reading.status == "missing" and "techval rag record" in reading.reason
    assert recording is None


def test_a_recorded_answer_is_held_to_its_quote(tasks):
    passage = Passage("A:0", "A", None, 0, len(PAYO_SENTENCE), PAYO_SENTENCE)
    good = _recording("PAYO:cash_per_share", _response(_answer()))
    reading, recording = ClaudeReader(_Replayer(good), SETTINGS).read(tasks["PAYO:cash_per_share"], [passage])
    assert reading.status == "stated" and recording is good
    bad = _recording("PAYO:cash_per_share", _response(_answer(value=7.5, quote="the right to receive $7.50 in cash")))
    reading, _ = ClaudeReader(_Replayer(bad), SETTINGS).read(tasks["PAYO:cash_per_share"], [passage])
    assert reading.status == "refused"


def test_the_regex_rules_read_datadog_s_item_7_as_they_always_have(store, tasks):
    text, _ = store.section(store.by_path(KPI_DOCUMENTS["DDOG"]), "7")
    regex = RegexReader()

    def read(metric):
        return regex.read(tasks[f"DDOG:{metric}"], text, "native", [])

    assert read("customers").status == "refused"
    assert read("arr").status == "refused"
    nrr = read("net_revenue_retention")
    assert (nrr.reader, nrr.status, nrr.value, nrr.unit, nrr.hedged) == (
        "regex_native", "stated", 120.0, "percent", True,
    )
    assert read("subscribers").status == "not_stated"


def test_the_regex_rules_read_the_payoneer_announcement(store, tasks):
    text = store.text(store.get("0000950103-26-008945"))
    regex = RegexReader()

    def read(metric):
        return regex.read(tasks[f"PAYO:{metric}"], text, "retrieved", [])

    cash = read("cash_per_share")
    assert (cash.reader, cash.status, cash.value, cash.unit) == ("regex_retrieved", "stated", 7.4, "usd_per_share")
    assert read("consideration_form").text_value == "cash"
    assert read("exchange_ratio").status == "not_stated"
    assert read("agreement_date").text_value == "2026-06-12"
    assert read("acquirer").text_value == "Neon Maple Parent Inc."


def test_a_collared_exchange_ratio_is_refused_by_the_regex_rules(store, tasks):
    text = store.text(store.get("0001104659-26-078482"))
    reading = RegexReader().read(tasks["IRDM:exchange_ratio"], text, "native", [])
    assert reading.status == "refused" and "collar" in reading.reason


def test_a_regex_reading_points_at_the_passage_holding_its_match(store, tasks):
    doc = store.by_path(KPI_DOCUMENTS["DDOG"])
    text, base = store.section(doc, "7")
    passages = chunk(text, accession=doc.accession, item="7", base_offset=base)
    reading = RegexReader().read(tasks["DDOG:net_revenue_retention"], text, "native", passages)
    cited = next(p for p in passages if p.id == reading.passage_id)
    assert reading.quote in cited.text


def test_blank_text_is_not_read(tasks):
    reading = RegexReader().read(tasks["PAYO:acquirer"], "  ", "retrieved", [])
    assert reading.status == "not_stated"
