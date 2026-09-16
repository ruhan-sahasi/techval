"""The commands: template and score read committed files only; record is the one door to the API."""

from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from techval import cli as C
from techval import commands_rag as R
from techval.errors import ConfigError

runner = CliRunner()
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _invoke(*args):
    return runner.invoke(C.app, ["rag", *args])


def test_the_rag_group_is_mounted_with_its_three_commands():
    result = _invoke("--help")
    assert result.exit_code == 0
    for name in ("template", "record", "score"):
        assert name in result.output


def test_template_writes_a_blank_key(tmp_path):
    key = tmp_path / "answer_key.csv"
    result = _invoke("template", "--ml-data", str(FIXTURES), "--key", str(key))
    assert result.exit_code == 0, result.output
    assert key.exists() and "rows to" in result.output


def test_template_keeps_answers_unless_forced(tmp_path):
    key = tmp_path / "answer_key.csv"
    _invoke("template", "--ml-data", str(FIXTURES), "--key", str(key))
    with key.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["status"] = "not_stated"
    with key.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = _invoke("template", "--ml-data", str(FIXTURES), "--key", str(key))
    assert result.exit_code == 1 and "--force" in result.output
    assert _invoke("template", "--ml-data", str(FIXTURES), "--key", str(key), "--force").exit_code == 0


def test_score_says_what_it_is_waiting_for(tmp_path):
    result = _invoke(
        "score", "--ml-data", str(FIXTURES),
        "--key", str(tmp_path / "absent.csv"), "--recordings", str(tmp_path / "recordings"),
    )
    assert result.exit_code == 0, result.output
    assert "not recorded" in result.output and "No answer key" in result.output


def test_record_without_the_sdk_says_how_to_install_it(tmp_path, monkeypatch):
    def refuse():
        raise ConfigError("recording needs the Anthropic SDK: pip install 'techval[rag]'")

    monkeypatch.setattr(R, "_client", refuse)
    result = _invoke("record", "--ml-data", str(FIXTURES), "--recordings", str(tmp_path))
    assert result.exit_code == 1 and "techval[rag]" in result.output


def test_record_sends_only_the_tasks_asked_for(tmp_path, monkeypatch):
    sent: list[tuple[str, dict]] = []

    class FakeRecorder:
        def __init__(self, client, root, **kwargs):
            self.failures = [("PAYO:acquirer", "errored")]

        def record(self, requests, live=False):
            sent.extend(requests)
            return []

    monkeypatch.setattr(R, "_client", lambda: object())
    monkeypatch.setattr(R, "Recorder", FakeRecorder)
    result = _invoke(
        "record", "--ml-data", str(FIXTURES), "--recordings", str(tmp_path),
        "--tasks", "PAYO:acquirer, DDOG:arr",
    )
    assert result.exit_code == 0, result.output
    assert sorted(t for t, _ in sent) == ["DDOG:arr", "PAYO:acquirer"]
    assert "PAYO:acquirer" in result.output and "errored" in result.output


def test_record_refuses_a_task_it_does_not_know(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_client", lambda: object())
    result = _invoke("record", "--ml-data", str(FIXTURES), "--recordings", str(tmp_path), "--tasks", "NOPE:arr")
    assert result.exit_code == 1 and "NOPE:arr" in result.output
