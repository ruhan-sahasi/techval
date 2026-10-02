"""techval invest, end to end and offline.

The build test runs the fixture portfolio through the real command with csv
prices and --offline, so it exercises the ledger, the quotes, both model fits,
the snapshot and the page, and never a network.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from techval.commands_invest import app
from techval.invest.snapshot import validate_snapshot

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def flat(text: str) -> str:
    """Collapse whitespace before matching, because Rich wraps at the console width."""
    return " ".join(text.split())


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "assumptions.yaml"
    path.write_text(
        "market:\n  risk_free_rate: 0.0483\n"
        "price_source: csv\n"
        f"price_csv_dir: {FIXTURES / 'prices'}\n",
        encoding="utf-8",
    )
    return path


def test_init_scaffolds_and_refuses_to_overwrite(tmp_path):
    target = tmp_path / "book"
    first = runner.invoke(app, ["init", "--dir", str(target)])
    assert first.exit_code == 0, first.output
    assert (target / "portfolio.yaml").is_file()
    assert "techval invest" in first.output
    second = runner.invoke(app, ["init", "--dir", str(target)])
    assert second.exit_code == 1
    assert "not overwriting" in flat(second.output)


def test_a_missing_ledger_points_at_init(tmp_path):
    result = runner.invoke(app, ["--dir", str(tmp_path), "--config", str(write_config(tmp_path))])
    assert result.exit_code == 1
    assert "techval invest init" in flat(result.output)


def test_the_offline_build_writes_snapshot_and_page(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    result = runner.invoke(
        app, ["--dir", str(book), "--config", str(write_config(tmp_path)), "--offline"]
    )
    assert result.exit_code == 0, result.output
    snapshot = json.loads((book / "snapshot.json").read_text(encoding="utf-8"))
    validate_snapshot(snapshot)
    page = (book / "index.html").read_text(encoding="utf-8")
    assert "iv-snapshot" in page
    # Offline means offline: every covered name explains the missing DCF.
    ddog = snapshot["engine"]["DDOG"]
    assert ddog["dcf"] is None
    assert any("--offline" in r["why"] for r in ddog["refusals"])
    assert "Wrote" in result.output


def test_render_redraws_from_the_snapshot_alone(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    built = runner.invoke(
        app, ["--dir", str(book), "--config", str(write_config(tmp_path)), "--offline"]
    )
    assert built.exit_code == 0, built.output
    (book / "index.html").unlink()
    again = runner.invoke(app, ["--dir", str(book), "--render"])
    assert again.exit_code == 0, again.output
    assert (book / "index.html").is_file()


def test_render_without_a_snapshot_refuses(tmp_path):
    result = runner.invoke(app, ["--dir", str(tmp_path), "--render"])
    assert result.exit_code == 1
    assert "snapshot" in result.output


def test_a_run_without_the_model_panels_still_renders_and_says_why(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    empty = tmp_path / "no-panels"
    empty.mkdir()
    result = runner.invoke(
        app,
        ["--dir", str(book), "--config", str(write_config(tmp_path)), "--offline", "--panels", str(empty)],
    )
    assert result.exit_code == 0, result.output
    assert "model panels" in flat(result.output)
    assert (book / "index.html").is_file()


def test_open_hands_the_rendered_page_to_the_browser(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    built = runner.invoke(
        app, ["--dir", str(book), "--config", str(write_config(tmp_path)), "--offline", "--open"]
    )
    assert built.exit_code == 0, built.output
    assert opened == [(book / "index.html").resolve().as_uri()]
    again = runner.invoke(app, ["--dir", str(book), "--render", "--open"])
    assert again.exit_code == 0, again.output
    assert len(opened) == 2


def test_without_open_nothing_is_launched(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    runner.invoke(app, ["--dir", str(book), "--config", str(write_config(tmp_path)), "--offline"])
    assert opened == []


def test_the_tracker_builds_its_cache_with_an_age_limit(tmp_path, monkeypatch):
    import techval.edgar as edgar

    built = []
    real = edgar.HttpCache

    def spy(*args, **kwargs):
        cache = real(*args, **{**kwargs, "root": tmp_path / "http"})
        built.append(cache)
        return cache

    monkeypatch.setattr(edgar, "HttpCache", spy)
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    result = runner.invoke(
        app,
        ["--dir", str(book), "--config", str(write_config(tmp_path)), "--offline", "--max-age", "6"],
    )
    assert result.exit_code == 0, result.output
    assert built and built[0].max_age == 6 * 3600
