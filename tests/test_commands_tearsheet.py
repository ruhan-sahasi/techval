"""techval tearsheet, end to end and offline, on the committed fixtures."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from techval.cli import app

from conftest import FIXTURES, flat, offline_config, usage_error

runner = CliRunner()


def config(tmp_path: Path, peers: str = "[CRWD, MDB, ZS]") -> Path:
    return offline_config(tmp_path, f"comps:\n  peers: {peers}\n")


def snapshot_in(page: str) -> dict:
    block = re.search(r'<script type="application/json" id="tv-snapshot">(.*?)</script>', page, re.S)
    return json.loads(block.group(1).replace("<\\/", "</"))


def test_a_tearsheet_is_written_offline_as_a_one_section_page(tmp_path):
    out = tmp_path / "ddog.html"
    result = runner.invoke(
        app,
        ["tearsheet", "DDOG", "--config", str(config(tmp_path)), "--facts-dir", str(FIXTURES),
         "--panels", str(FIXTURES), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert f"Wrote {out}" in flat(result.output)
    snap = snapshot_in(out.read_text(encoding="utf-8"))
    assert snap["title"] == "DDOG, valued by the engine"
    assert list(snap["sections"]) == ["engine"] and snap["fixtures"] == {}
    assert snap["collected_at"] == "2026-09-10"
    assert "trading comps against CRWD, MDB, ZS" in snap["page"]["standfirst"]
    assert "scenarios" in snap["sections"]["engine"]["figures"]


def test_a_name_with_no_recorded_facts_is_refused_with_the_path_it_looked_at(tmp_path):
    result = runner.invoke(
        app,
        ["tearsheet", "NOPE", "--config", str(config(tmp_path, "[]")), "--facts-dir", str(FIXTURES),
         "--out", str(tmp_path / "nope.html")],
    )
    assert result.exit_code == 1
    assert "companyfacts_NOPE.json" in flat(result.output)
    assert not (tmp_path / "nope.html").exists()


def test_a_tearsheet_on_the_record_case_says_so_and_shows_its_figures(tmp_path):
    out = tmp_path / "record.html"
    result = runner.invoke(
        app,
        ["tearsheet", "DDOG", "--config", str(config(tmp_path)), "--facts-dir", str(FIXTURES),
         "--panels", str(FIXTURES), "--out", str(out), "--from-record"],
    )
    assert result.exit_code == 0, result.output
    snap = snapshot_in(out.read_text(encoding="utf-8"))
    assert "record" in snap["sections"]["engine"]["figures"]
    assert "The DCF is the record case" in snap["page"]["standfirst"]


def test_compare_writes_one_page_for_the_set_and_prints_the_table(tmp_path):
    out = tmp_path / "compare.html"
    result = runner.invoke(
        app,
        ["compare", "DDOG", "CRWD", "DIS", "NOPE", "--config", str(config(tmp_path, "[]")), "--facts-dir", str(FIXTURES),
         "--panels", str(FIXTURES), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "Record case" in text and "Implied return" in text
    assert "Not valued NOPE" in text and f"Wrote {out}" in text
    snap = snapshot_in(out.read_text(encoding="utf-8"))
    assert snap["title"] == "DDOG, CRWD, DIS, NOPE, compared"
    assert set(snap["sections"]["compare"]["figures"]) == {"value", "returns", "table"}


def test_compare_json_carries_the_rows_and_the_refusals(tmp_path):
    out = tmp_path / "compare.html"
    result = runner.invoke(
        app,
        ["compare", "DDOG", "NOPE", "--config", str(config(tmp_path, "[]")), "--facts-dir", str(FIXTURES),
         "--panels", str(FIXTURES), "--out", str(out), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [r["company"] for r in payload["rows"]] == ["DDOG"]
    assert payload["refused"][0]["what"] == "NOPE" and out.exists()


@pytest.mark.parametrize("command", [["tearsheet", "DDOG"], ["compare", "DDOG", "ZS"]])
def test_a_folder_named_as_the_page_is_refused_before_the_page_is_built(
    tmp_path, command
):
    """The page was built and then written to the folder: IsADirectoryError."""
    result = runner.invoke(app, [*command, "--out", str(tmp_path)])
    assert result.exit_code == 2
    assert "Invalid value for '--out'" in usage_error(result.output)

