"""``techval dashboard``: render without collecting, collect when asked, refuse contradictions.

Collection itself is tested in test_collect.py against stub sections. Here the
collector is replaced by a fake that records how it was called, so these tests
pin what the command decides (whether to collect, what to merge, which date to
stamp) and what it prints, without running a single section.
"""

from __future__ import annotations

from datetime import date

import pytest
from typer.testing import CliRunner

from techval import cli as C
from techval.dashboard import render as R
from techval.dashboard.collect import SectionRun
from techval.dashboard.render import render_dashboard
from techval.dashboard.sections import COLLECT_ORDER, SECTION_IDS
from techval.dashboard.snapshot import dump_snapshot, load_snapshot

from dashboard.test_render import make_assets, make_snapshot

runner = CliRunner()


@pytest.fixture
def assets(tmp_path, monkeypatch):
    path = make_assets(tmp_path)
    monkeypatch.setattr(R, "default_assets_dir", lambda: path)
    return path


@pytest.fixture
def no_collection(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the command collected when it should only have rendered")

    monkeypatch.setattr(C, "collect_snapshot", refuse)


@pytest.fixture
def fake_collect(monkeypatch):
    calls: list[dict] = []

    def fake(root, collected_at, **kwargs):
        calls.append({"root": root, "collected_at": collected_at, **kwargs})
        snap = make_snapshot()
        snap.collected_at = collected_at
        for i, sid in enumerate(COLLECT_ORDER):
            kwargs["on_section"](
                SectionRun(
                    sid, snap.sections[sid]["status"], 0.25 * i, "hit" if i % 2 else "miss"
                )
            )
        return snap

    monkeypatch.setattr(C, "collect_snapshot", fake)
    return calls


def _invoke(tmp_path, *args):
    return runner.invoke(
        C.app,
        [
            "dashboard",
            "--snapshot", str(tmp_path / "snapshot.json"),
            "--out", str(tmp_path / "site" / "index.html"),
            *args,
        ],
    )


def test_renders_an_existing_snapshot_without_collecting(tmp_path, assets, no_collection):
    dump_snapshot(make_snapshot(), tmp_path / "snapshot.json")
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    page = (tmp_path / "site" / "index.html").read_text()
    assert page == render_dashboard(make_snapshot(), assets)
    assert "Nothing was re-collected" in result.output
    for sid in SECTION_IDS:
        assert sid in result.output
    assert "not collected" in result.output


def test_collects_when_there_is_no_snapshot(tmp_path, assets, fake_collect):
    result = _invoke(tmp_path, "--ml-data", str(tmp_path))
    assert result.exit_code == 0, result.output
    [call] = fake_collect
    assert call["collected_at"] == date.today().isoformat()
    assert call["sections"] is None and call["base"] is None
    assert call["use_cache"] is True
    assert load_snapshot(tmp_path / "snapshot.json").collected_at == date.today().isoformat()
    assert (tmp_path / "site" / "index.html").is_file()
    # One line per section: id, status, seconds, cache.
    lines = [
        words
        for words in (line.split() for line in result.output.splitlines())
        if words and words[0] in SECTION_IDS
    ]
    assert [l[0] for l in lines] == list(COLLECT_ORDER)
    assert lines[0][2:] == ["0.00s", "cache", "miss"]
    assert lines[1][2:] == ["0.25s", "cache", "hit"]
    assert "Refused" in result.output and "fade" in result.output


def test_sections_merge_into_the_existing_snapshot(tmp_path, assets, fake_collect):
    dump_snapshot(make_snapshot(), tmp_path / "snapshot.json")
    result = _invoke(
        tmp_path, "--sections", "encoder, fade", "--no-cache", "--collected-at", "2026-09-01"
    )
    assert result.exit_code == 0, result.output
    [call] = fake_collect
    assert call["sections"] == ["encoder", "fade"]
    assert call["base"] == make_snapshot()
    assert call["use_cache"] is False
    assert call["collected_at"] == "2026-09-01"


@pytest.mark.parametrize(
    "args, exists, message",
    [
        (["--no-collect"], False, "there is none at"),
        (["--sections", "encoder"], False, "Collect every section first"),
        (["--sections", "encoder", "--no-collect"], True, "Drop one of them"),
        (["--collected-at", "2026-09-01", "--no-collect"], True, "Drop one of them"),
        (["--sections", "encoder,bogus"], True, "unknown dashboard section 'bogus'"),
    ],
)
def test_contradictions_and_absences_are_refused(
    tmp_path, assets, no_collection, args, exists, message
):
    if exists:
        dump_snapshot(make_snapshot(), tmp_path / "snapshot.json")
    result = _invoke(tmp_path, *args)
    assert result.exit_code == 1
    assert "ConfigError" in result.output
    assert message in " ".join(result.output.split())
    assert not (tmp_path / "site" / "index.html").exists()


def test_no_mounted_command_shadows_dashboard():
    """The command modules are merged in flat, so a second `dashboard` would shadow silently."""
    from techval import commands_forecast, commands_mna, commands_peers, commands_tmt

    for module in (commands_tmt, commands_peers, commands_forecast, commands_mna):
        names = [
            c.name or c.callback.__name__.replace("_", "-")
            for c in module.app.registered_commands
        ]
        assert "dashboard" not in names, module.__name__


def test_dashboard_is_listed_under_its_own_heading():
    result = runner.invoke(C.app, ["--help"])
    assert result.exit_code == 0
    assert "Results" in result.output and "dashboard" in result.output
    result = runner.invoke(C.app, ["dashboard", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--ml-data",
        "--snapshot",
        "--out",
        "--collect",
        "--sections",
        "--no-cache",
        "--collected-at",
    ):
        assert flag in result.output
