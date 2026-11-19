"""techval expectations, end to end and offline, on Datadog's committed fixtures."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from techval.cli import app

from conftest import FIXTURES, flat, offline_config

runner = CliRunner()


def run(tmp_path, *extra):
    return runner.invoke(
        app,
        [
            "expectations", "DDOG",
            "--config", str(offline_config(tmp_path)),
            "--facts", str(FIXTURES / "companyfacts_DDOG.json"),
            "--panels", str(FIXTURES),
            *extra,
        ],
    )


def test_expectations_prints_every_lever_the_frontier_and_the_base_rates(tmp_path):
    result = run(tmp_path)
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    for phrase in ("Discount rate", "First-year growth", "Terminal margin", "Duration", "Frontier"):
        assert phrase in text, phrase
    assert "3.71% a year" in text
    assert "155.2%" in text
    assert "of 1,662" in text


def test_a_price_override_asks_what_another_price_assumes(tmp_path):
    result = run(tmp_path, "--price", "36.65")
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    # The base case's own value implies roughly the engine's own cost of capital.
    assert "11.28% a year" in text or "11.27% a year" in text


def test_json_output_is_the_whole_result(tmp_path):
    result = run(tmp_path, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ticker"] == "DDOG"
    assert {s["lever"] for s in payload["solves"]} == {"discount_rate", "first_year_growth", "terminal_margin", "duration"}
    assert payload["base_rates"]["first_year_growth"]["all"]["n"] == 1662


def test_without_panels_it_runs_and_says_there_are_no_base_rates(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(
        app,
        [
            "expectations", "DDOG",
            "--config", str(offline_config(tmp_path)),
            "--facts", str(FIXTURES / "companyfacts_DDOG.json"),
            "--panels", str(empty),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no fade panel" in flat(result.output)


def test_history_prints_each_quarter_and_names_the_refused_ones(tmp_path):
    result = run(tmp_path, "--history")
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "Implied return" in text and "2026-06-30" in text and "2024-03-31" in text
    assert "refused" in text.lower()


def test_history_rides_in_the_json(tmp_path):
    result = run(tmp_path, "--history", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    rows = payload["history"]
    assert rows[-1]["date"] == "2026-06-30" or rows[-1]["date"] == "2026-09-10"
    assert any(r["refused"] is None and r["implied_return"] for r in rows)


def test_the_screen_ranks_names_by_how_rarely_their_implied_growth_happened(tmp_path):
    result = runner.invoke(
        app,
        [
            "expectations-screen", "DDOG", "CRWD", "MDB", "ZS", "DIS",
            "--config", str(offline_config(tmp_path)),
            "--facts-dir", str(FIXTURES),
            "--panels", str(FIXTURES),
        ],
    )
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert text.index("CRWD") < text.index("DIS")
    assert "Implied return" in text and "Growth needed" in text
    assert "Vs scenarios" in text
    assert "3.1x the bull" in text and "86% bear" in text


def test_a_screen_name_without_facts_is_named_not_fatal(tmp_path):
    result = runner.invoke(
        app,
        ["expectations-screen", "DDOG", "NOPE", "--config", str(offline_config(tmp_path)), "--facts-dir", str(FIXTURES), "--json"],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    nope = next(r for r in rows if r["ticker"] == "NOPE")
    assert "companyfacts_NOPE.json" in nope["refused"]


def test_a_screen_name_whose_payload_is_not_json_is_named_not_fatal(tmp_path):
    """A blocked SEC request saved as the payload ended the whole screen.

    It raised a JSONDecodeError, which the command does not catch; it is now one
    refused row, as a missing payload already was.
    """
    facts = tmp_path / "facts"
    facts.mkdir()
    ddog = (FIXTURES / "companyfacts_DDOG.json").read_bytes()
    (facts / "companyfacts_DDOG.json").write_bytes(ddog)
    (facts / "companyfacts_MDB.json").write_text(
        "<html>Request Rate Threshold Exceeded</html>", encoding="utf-8"
    )
    result = runner.invoke(
        app,
        [
            "expectations-screen", "DDOG", "MDB",
            "--config", str(offline_config(tmp_path)),
            "--facts-dir", str(facts),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = {r["ticker"]: r for r in json.loads(result.output)}
    assert "could not be read as a companyfacts payload" in rows["MDB"]["refused"]
    assert rows["DDOG"]["refused"] is None


def test_a_facts_file_that_is_not_there_is_refused_not_raised(tmp_path):
    result = runner.invoke(
        app,
        [
            "expectations", "DDOG",
            "--config", str(offline_config(tmp_path)),
            "--facts", str(tmp_path / "companyfacts_DDOG.json"),
            "--panels", str(FIXTURES),
        ],
    )
    assert result.exit_code == 1
    assert "could not be opened: No such file or directory" in flat(result.output)


def test_the_record_case_is_named_when_it_is_on(tmp_path):
    path = offline_config(tmp_path)
    path.write_text(path.read_text() + "dcf:\n  from_record: true\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["expectations", "DDOG", "--config", str(path), "--facts", str(FIXTURES / "companyfacts_DDOG.json"), "--panels", str(FIXTURES)],
    )
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "On the record case: year 1 revenue growth 31.5%" in text
    assert "terminal ebit margin 0.4%" in text


def test_the_from_record_flag_reaches_expectations_and_the_screen(tmp_path):
    result = run(tmp_path, "--from-record")
    assert result.exit_code == 0, result.output
    assert "On the record case" in flat(result.output)
    args = ["expectations-screen", "DDOG", "ZS", "--config", str(offline_config(tmp_path)), "--facts-dir", str(FIXTURES), "--json"]
    plain = {r["ticker"]: r for r in json.loads(runner.invoke(app, args).output)}
    record = {r["ticker"]: r for r in json.loads(runner.invoke(app, [*args, "--from-record"]).output)}
    assert record["DDOG"]["base_value"] != plain["DDOG"]["base_value"]
