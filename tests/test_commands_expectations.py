"""techval expectations, end to end and offline, on Datadog's committed fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from techval.cli import app

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def flat(text: str) -> str:
    import re

    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())


def config(tmp_path: Path) -> Path:
    path = tmp_path / "assumptions.yaml"
    path.write_text(
        "market:\n  risk_free_rate: 0.0483\n"
        "price_source: csv\n"
        f"price_csv_dir: {FIXTURES / 'prices'}\n"
        "as_of: '2026-09-10'\n",
        encoding="utf-8",
    )
    return path


def run(tmp_path, *extra):
    return runner.invoke(
        app,
        [
            "expectations", "DDOG",
            "--config", str(config(tmp_path)),
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
            "--config", str(config(tmp_path)),
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
            "--config", str(config(tmp_path)),
            "--facts-dir", str(FIXTURES),
            "--panels", str(FIXTURES),
        ],
    )
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert text.index("CRWD") < text.index("DIS")
    assert "Implied return" in text and "Growth needed" in text


def test_a_screen_name_without_facts_is_named_not_fatal(tmp_path):
    result = runner.invoke(
        app,
        ["expectations-screen", "DDOG", "NOPE", "--config", str(config(tmp_path)), "--facts-dir", str(FIXTURES), "--json"],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    nope = next(r for r in rows if r["ticker"] == "NOPE")
    assert "companyfacts_NOPE.json" in nope["refused"]
