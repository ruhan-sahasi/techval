"""Recordings: one file per request, found by the request's own content, never by a live call."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from techval.errors import DataSourceError
from techval.rag.recording import (
    FALLBACK_BETA,
    Recorder,
    Recording,
    RecordingMissing,
    Replayer,
    request_key,
)

PARAMS_A = {"model": "claude-opus-5", "max_tokens": 16000, "messages": [{"role": "user", "content": "A"}]}
PARAMS_B = {"model": "claude-opus-5", "max_tokens": 16000, "messages": [{"role": "user", "content": "B"}]}


def _message(text: str, model: str = "claude-opus-5") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class _Message:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return dict(self._payload)


class FakeBatches:
    """Returns results in reverse order, and leaves unanswered requests errored."""

    def __init__(self, answers: dict[str, dict], statuses=("in_progress", "ended")) -> None:
        self.answers = answers
        self.statuses = list(statuses)
        self.created: list[list[dict]] = []

    def create(self, requests):
        self.created.append(list(requests))
        return SimpleNamespace(id="batch_1")

    def retrieve(self, batch_id):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return SimpleNamespace(processing_status=status)

    def results(self, batch_id):
        for request in reversed(self.created[-1]):
            payload = self.answers.get(request["custom_id"])
            if payload is None:
                yield SimpleNamespace(custom_id=request["custom_id"], result=SimpleNamespace(type="errored"))
            else:
                yield SimpleNamespace(
                    custom_id=request["custom_id"],
                    result=SimpleNamespace(type="succeeded", message=_Message(payload)),
                )


class FakeClient:
    def __init__(self, batches: FakeBatches | None = None, live: dict | None = None) -> None:
        self.messages = SimpleNamespace(batches=batches)
        self.calls: list[dict] = []

        def create(**kwargs):
            self.calls.append(kwargs)
            return _Message(live)

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def _recorder(client, root, sleeps=None):
    return Recorder(
        client, root, today="2026-09-16", sdk_version="1.6.0",
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


def test_the_key_ignores_key_order_and_follows_content():
    shuffled = {"messages": PARAMS_A["messages"], "max_tokens": 16000, "model": "claude-opus-5"}
    assert request_key(PARAMS_A) == request_key(shuffled)
    assert request_key(PARAMS_A) != request_key(PARAMS_B)
    assert len(request_key(PARAMS_A)) == 64


def test_a_missing_recording_is_an_error_not_a_live_call(tmp_path):
    with pytest.raises(RecordingMissing) as caught:
        Replayer(tmp_path).recording("PAYO:acquirer", PARAMS_A)
    assert caught.value.task_id == "PAYO:acquirer"
    assert caught.value.key == request_key(PARAMS_A)


def test_a_recording_of_a_different_request_is_refused(tmp_path):
    wrong = Recording(request_key(PARAMS_A), "T", PARAMS_B, _message("{}"), "claude-opus-5", {}, "2026-09-16", "batch", "1.6.0")
    (tmp_path / f"{request_key(PARAMS_A)}.json").write_text(json.dumps(wrong.to_json()))
    with pytest.raises(DataSourceError):
        Replayer(tmp_path).recording("T", PARAMS_A)


def test_a_batch_is_recorded_by_custom_id_whatever_order_results_arrive_in(tmp_path):
    batches = FakeBatches({request_key(PARAMS_A): _message("a"), request_key(PARAMS_B): _message("b")})
    sleeps: list[float] = []
    recorder = _recorder(FakeClient(batches), tmp_path, sleeps)
    written = recorder.record([("T:a", PARAMS_A), ("T:b", PARAMS_B)])
    assert sorted(r.task_id for r in written) == ["T:a", "T:b"]
    assert sleeps == [30]
    [sent] = batches.created
    assert all("fallbacks" not in r["params"] for r in sent)
    replayed = Replayer(tmp_path).recording("T:b", PARAMS_B)
    assert replayed.response["content"][0]["text"] == "b"
    assert (replayed.mode, replayed.served_by, replayed.recorded_at) == ("batch", "claude-opus-5", "2026-09-16")


def test_a_failed_batch_entry_is_reported_and_nothing_is_written_for_it(tmp_path):
    batches = FakeBatches({request_key(PARAMS_A): _message("a")})
    recorder = _recorder(FakeClient(batches), tmp_path)
    written = recorder.record([("T:a", PARAMS_A), ("T:b", PARAMS_B)])
    assert [r.task_id for r in written] == ["T:a"]
    assert recorder.failures == [("T:b", "errored")]
    assert not (tmp_path / f"{request_key(PARAMS_B)}.json").exists()


def test_a_request_already_recorded_is_not_sent_again(tmp_path):
    batches = FakeBatches({request_key(PARAMS_A): _message("a")})
    _recorder(FakeClient(batches), tmp_path).record([("T:a", PARAMS_A)])
    again = FakeBatches({})
    assert _recorder(FakeClient(again), tmp_path).record([("T:a", PARAMS_A)]) == []
    assert again.created == []


def test_a_live_recording_turns_on_server_side_fallbacks_and_names_the_model_that_answered(tmp_path):
    client = FakeClient(live=_message("x", model="claude-opus-4-8"))
    [rec] = _recorder(client, tmp_path).record([("T:a", PARAMS_A)], live=True)
    [call] = client.calls
    assert call["betas"] == [FALLBACK_BETA]
    assert call["extra_body"] == {"fallbacks": "default"}
    assert {k: call[k] for k in PARAMS_A} == PARAMS_A
    assert (rec.mode, rec.served_by) == ("live", "claude-opus-4-8")
