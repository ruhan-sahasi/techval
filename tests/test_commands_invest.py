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


def test_check_replays_the_ledger_without_quotes(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    result = runner.invoke(app, ["check", "--dir", str(book)])
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "17 transactions" in text
    assert "9 open positions" in text
    assert "DDOG 80" in text
    assert "cash" in text.lower()
    assert "targets 47%" in text


def test_check_names_the_row_that_cannot_be_true(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    (book / "portfolio.yaml").write_text(
        "transactions:\n"
        "  - {date: 2024-01-02, type: deposit, amount: 100}\n"
        "  - {date: 2024-01-03, type: buy, symbol: AAA, shares: 5, price: 100}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["check", "--dir", str(book)])
    assert result.exit_code == 1
    assert "2024-01-03" in flat(result.output)


STARTER_WITH_TARGETS = """\
# my book
name: Mine
transactions:
  - {date: 2026-01-02, type: deposit, amount: 10000}  # opening cash
  - {date: 2026-01-05, type: buy, symbol: DDOG, shares: 10, price: 120}
# targets after the list, with a comment between
targets:
  DDOG: 0.5
"""


def test_add_appends_to_the_list_and_keeps_every_comment(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    (book / "portfolio.yaml").write_text(STARTER_WITH_TARGETS, encoding="utf-8")
    result = runner.invoke(
        app, ["add", "buy", "net", "5", "80.5", "--date", "2026-02-03", "--fee", "1", "--dir", str(book)]
    )
    assert result.exit_code == 0, result.output
    text = (book / "portfolio.yaml").read_text(encoding="utf-8")
    assert "# my book" in text and "# opening cash" in text and "# targets after the list" in text
    lines = text.splitlines()
    added = lines.index("  - {date: 2026-02-03, type: buy, symbol: NET, shares: 5, price: 80.5, fee: 1}")
    assert lines[added - 1].startswith("  - {date: 2026-01-05")
    from techval.invest.ledger import Ledger

    ledger = Ledger.load(book / "portfolio.yaml")
    assert ledger.positions()["NET"].shares == 5
    assert ledger.targets == {"DDOG": 0.5}


def test_add_each_kind_of_row(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    (book / "portfolio.yaml").write_text(STARTER_WITH_TARGETS, encoding="utf-8")
    for args in (
        ["deposit", "500", "--date", "2026-03-01"],
        ["dividend", "DDOG", "4.25", "--date", "2026-03-02"],
        ["split", "DDOG", "2", "--date", "2026-03-03"],
        ["sell", "DDOG", "4", "70", "--date", "2026-03-04"],
        ["withdraw", "100", "--date", "2026-03-05"],
        ["buy", "VOO", "1", "500", "--kind", "etf", "--date", "2026-03-06"],
    ):
        result = runner.invoke(app, ["add", *args, "--dir", str(book)])
        assert result.exit_code == 0, (args, result.output)
    from techval.invest.ledger import Ledger

    ledger = Ledger.load(book / "portfolio.yaml")
    assert [t.type for t in ledger.transactions][-6:] == ["deposit", "dividend", "split", "sell", "withdraw", "buy"]
    assert ledger.positions()["DDOG"].shares == 16
    assert ledger.kind("VOO") == "etf"


def test_add_refuses_a_row_the_ledger_cannot_hold_and_leaves_the_file_alone(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    path = book / "portfolio.yaml"
    path.write_text(STARTER_WITH_TARGETS, encoding="utf-8")
    before = path.read_bytes()
    result = runner.invoke(app, ["add", "buy", "AAPL", "1000", "200", "--date", "2026-02-01", "--dir", str(book)])
    assert result.exit_code == 1
    assert "cash" in flat(result.output).lower()
    assert path.read_bytes() == before


def test_add_refuses_the_wrong_number_of_values(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    (book / "portfolio.yaml").write_text(STARTER_WITH_TARGETS, encoding="utf-8")
    result = runner.invoke(app, ["add", "buy", "NET", "5", "--dir", str(book)])
    assert result.exit_code == 1
    assert "SYMBOL SHARES PRICE" in flat(result.output)


def _export(tmp_path, what):
    book = tmp_path / "book"
    book.mkdir(exist_ok=True)
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    out = tmp_path / f"{what}.csv"
    result = runner.invoke(app, ["export", what, "--dir", str(book), "--out", str(out)])
    assert result.exit_code == 0, result.output
    import csv

    with out.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), result


def test_export_realized_writes_one_row_per_closed_lot_slice(tmp_path):
    rows, result = _export(tmp_path, "realized")
    assert list(rows[0]) == [
        "description", "date_acquired", "date_sold", "proceeds", "cost_basis", "gain", "term",
    ]
    assert [(r["description"], r["date_sold"], r["term"]) for r in rows] == [
        ("30 sh NET", "2024-08-15", "short"),
        ("10 sh MDB", "2026-04-15", "long"),
    ]
    assert float(rows[1]["gain"]) == -1300.0
    assert "2 rows" in flat(result.output)


def test_export_lots_and_activity(tmp_path):
    lots, _ = _export(tmp_path, "lots")
    assert len(lots) == 10 and {"symbol", "opened", "shares", "cost_per_share", "cost"} <= set(lots[0])
    activity, _ = _export(tmp_path, "activity")
    assert len(activity) == 17
    assert activity[0]["date"] == "2023-10-02" and activity[0]["type"] == "deposit"


def test_export_refuses_an_unknown_table(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    result = runner.invoke(app, ["export", "taxes", "--dir", str(book), "--out", str(tmp_path / "x.csv")])
    assert result.exit_code == 1
    assert "realized, lots or activity" in flat(result.output)


def test_a_run_prints_the_headline_numbers(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    result = runner.invoke(
        app, ["--dir", str(book), "--config", str(write_config(tmp_path)), "--offline"]
    )
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    for label in ("Total value", "Last session", "Time-weighted", "vs SPY", "Max drawdown", "Engine coverage", "Prices as of"):
        assert label in text, label
    assert "$116,785" in text


def test_benchmark_overrides_the_ledgers_for_one_run(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    result = runner.invoke(
        app,
        ["--dir", str(book), "--config", str(write_config(tmp_path)), "--offline", "--benchmark", "dis"],
    )
    assert result.exit_code == 0, result.output
    snapshot = json.loads((book / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["meta"]["benchmark"] == "DIS"
    assert snapshot["performance"]["benchmark"] == "DIS"
    assert "vs DIS" in flat(result.output)
    # The ledger file itself is untouched.
    assert "benchmark: SPY" in (book / "portfolio.yaml").read_text(encoding="utf-8")


def test_check_warns_about_a_row_entered_twice_but_does_not_refuse(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    (book / "portfolio.yaml").write_text(
        "transactions:\n"
        "  - {date: 2024-01-02, type: deposit, amount: 1000}\n"
        "  - {date: 2024-01-03, type: buy, symbol: AAA, shares: 1, price: 100}\n"
        "  - {date: 2024-01-03, type: buy, symbol: AAA, shares: 1, price: 100}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["check", "--dir", str(book)])
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "entered 2 times" in text and "rows 2 and 3" in text


def test_quiet_prints_only_what_needs_attention(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(FIXTURES / "invest" / "portfolio.yaml", book / "portfolio.yaml")
    result = runner.invoke(
        app, ["--dir", str(book), "--config", str(write_config(tmp_path)), "--offline", "--quiet"]
    )
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "Total value" not in text and "Wrote" not in text
    # The committed closes are weeks old by now: that still gets said.
    assert "days old" in text
    assert (book / "index.html").is_file()
