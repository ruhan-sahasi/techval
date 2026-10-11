"""techval scenarios, end to end and offline, on Datadog's committed fixtures."""

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


def run(tmp_path, *extra, panels=FIXTURES):
    config = tmp_path / "assumptions.yaml"
    config.write_text(
        "market:\n  risk_free_rate: 0.0483\n"
        f"price_source: csv\nprice_csv_dir: {FIXTURES / 'prices'}\n"
        "as_of: '2026-09-10'\n",
        encoding="utf-8",
    )
    return runner.invoke(
        app,
        ["scenarios", "DDOG", "--config", str(config), "--facts", str(FIXTURES / "companyfacts_DDOG.json"),
         "--panels", str(panels), *extra],
    )


def test_scenarios_print_three_valued_paths_the_weighting_and_the_calibration(tmp_path):
    result = run(tmp_path)
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    for phrase in ("Bear", "Base", "Bull", "28.52", "44.20", "71.90", "47.81"):
        assert phrase in text, phrase
    assert "above even the bull case" in text
    assert "held 69.1% of 674" in text


def test_a_price_inside_the_scenarios_is_read_as_a_weight(tmp_path):
    result = run(tmp_path, "--price", "60")
    assert result.exit_code == 0, result.output
    assert "on the bull case, against 30%" in flat(result.output)


def test_json_carries_the_scenarios_and_the_calibration(tmp_path):
    result = run(tmp_path, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [s["name"] for s in payload["scenarios"]] == ["bear", "base", "bull"]
    assert payload["calibration"]["n"] == 674
    assert payload["implied"]["side"] == "above"


def test_without_a_panel_there_is_nothing_to_read_scenarios_from(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run(tmp_path, panels=empty)
    assert result.exit_code == 1
    assert "no fade panel" in flat(result.output)
