"""The techval invest command group: init, refresh and render.

The private directory ``portfolio/`` is gitignored, and everything the group
writes stays inside it. ``techval invest`` with no subcommand quotes the
holdings, runs the engine over them and renders the page; ``--offline`` prices
from local CSVs and skips the live DCFs entirely; ``--render`` redraws
the page from the last snapshot without touching a network at all.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from .config import Assumptions
from .errors import TechvalError

app = typer.Typer(
    name="invest",
    help="Track a private portfolio and read the engine over it.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()

_DIR = typer.Option(Path("portfolio"), "--dir", help="The private portfolio directory.")
_CFG = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file.")
_OFFLINE = typer.Option(
    False,
    "--offline",
    help="Prices from price_csv_dir only, and no live DCFs: offline means offline.",
)
_RENDER = typer.Option(
    False,
    "--render",
    help="Re-render the page from the existing snapshot. Touches no network.",
)
_PANELS = typer.Option(
    None,
    "--panels",
    help=(
        "Directory holding the committed model panels (fade_companyfacts.json.gz, "
        "warranted/observations.json.gz). Defaults to tests/fixtures in this checkout."
    ),
)
_OPEN = typer.Option(False, "--open", help="Open the rendered page in the default browser.")
_QUIET = typer.Option(
    False,
    "--quiet",
    "-q",
    help="Print only warnings and errors, for a scheduled run: stale prices, unquotable holdings, rows entered twice.",
)
_BENCHMARK = typer.Option(
    None,
    "--benchmark",
    help="Compare against this symbol for one run, QQQ say, instead of the ledger's benchmark.",
)
_MAX_AGE = typer.Option(
    20.0,
    "--max-age",
    min=0.0,
    help=(
        "Hours a cached price or filing stays fresh. A tracker is run every day, so "
        "the default refetches anything older than 20 hours; 0 refetches everything."
    ),
)

# A morning run should see last night's close and any 10-K filed since the
# last run. Twenty hours covers a daily habit without refetching twice a day.
DEFAULT_MAX_AGE_HOURS = 20.0
_REFIT = typer.Option(
    False,
    "--refit",
    help="Refit the fade and warranted models instead of loading them from ml.cache_dir.",
)

STARTER = """\
# Your portfolio, as a ledger. techval invest reads this file and writes
# snapshot.json and index.html beside it. The whole directory is gitignored;
# nothing in it leaves your machine except price and filing requests.
#
# Transaction types: buy, sell, deposit, withdraw, dividend, split, interest, fee.
# interest and fee are cash the account earned or paid, a cash sweep's interest
# or an advisory fee; they count as return, unlike a deposit or a withdrawal.
#   - {date: 2024-02-01, type: interest, amount: 4.10}
# transfer_in brings shares from another broker at their original cost basis
# per share, with no cash; acquired keeps the original purchase date.
#   - {date: 2024-03-01, type: transfer_in, symbol: AAPL, shares: 20, price: 142.10, acquired: 2021-06-14}
#   - {date: 2024-01-15, type: deposit, amount: 5000}
#   - {date: 2024-01-16, type: buy, symbol: MSFT, shares: 10, price: 390.00}
#   - {date: 2024-06-12, type: dividend, symbol: MSFT, amount: 7.50}
#   - {date: 2024-08-05, type: split, symbol: NVDA, ratio: 10}
# A buy or sell may carry a fee: it joins a buy's cost basis and comes off a
# sell's proceeds.
#   - {date: 2024-09-03, type: sell, symbol: MSFT, shares: 2, price: 410.00, fee: 1.50}
# kind marks non-stocks once per symbol: etf or crypto.
#   - {date: 2024-02-01, type: buy, symbol: VOO, shares: 4, price: 460, kind: etf}
name: My portfolio
benchmark: SPY
transactions:
  - {date: 2026-01-02, type: deposit, amount: 10000}
# targets are optional, for the drift pane: weights of total value.
# targets:
#   VOO: 0.40
#   cash: 0.10
"""


@app.callback()
def invest(
    ctx: typer.Context,
    directory: Path = _DIR,
    config: Path = _CFG,
    offline: bool = _OFFLINE,
    render_only: bool = _RENDER,
    refit: bool = _REFIT,
    panels: Path = _PANELS,
    open_page: bool = _OPEN,
    max_age: float = _MAX_AGE,
    benchmark: str = _BENCHMARK,
    quiet: bool = _QUIET,
) -> None:
    """Refresh quotes and the engine read, then render portfolio/index.html."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        _build(directory, config, offline, render_only, refit, panels, max_age, benchmark, quiet)
    except TechvalError as err:
        console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(1)
    if open_page:
        import webbrowser

        webbrowser.open((directory / "index.html").resolve().as_uri())


@app.command()
def init(directory: Path = _DIR) -> None:
    """Write a commented starter portfolio.yaml into the private directory."""
    path = directory / "portfolio.yaml"
    if path.exists():
        console.print(f"[red]{escape(str(path))} already exists; not overwriting your ledger.[/red]")
        raise typer.Exit(1)
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(STARTER, encoding="utf-8")
    console.print(f"Wrote {escape(str(path))}. Edit it, then run: techval invest")


@app.command()
def add(
    kind_of: str = typer.Argument(..., metavar="TYPE", help="buy, sell, deposit, withdraw, dividend, split, interest, fee or transfer_in."),
    values: list[str] = typer.Argument(..., metavar="VALUES", help="buy/sell/transfer_in: SYMBOL SHARES PRICE. deposit/withdraw/interest/fee: AMOUNT. dividend: SYMBOL AMOUNT. split: SYMBOL RATIO."),
    when: str = typer.Option(None, "--date", help="The transaction date, YYYY-MM-DD. Defaults to today."),
    fee: float = typer.Option(None, "--fee", help="A trade's fee, for a buy or a sell."),
    kind: str = typer.Option(None, "--kind", help="etf or crypto, the first time a symbol appears."),
    note: str = typer.Option(None, "--note", help="A note kept on the row."),
    acquired: str = typer.Option(None, "--acquired", help="A transfer_in's original purchase date, for its holding period."),
    directory: Path = _DIR,
) -> None:
    """Append one transaction to the ledger, after replaying the whole ledger with it."""
    import os
    import tempfile

    from .invest.ledger import ADD_SHAPES, Ledger, append_row, flow_row

    path = directory / "portfolio.yaml"
    try:
        shape = ADD_SHAPES.get(kind_of)
        if shape is None:
            raise TechvalError(f"type must be one of {', '.join(ADD_SHAPES)}, not {kind_of!r}")
        if len(values) != len(shape):
            raise TechvalError(
                f"a {kind_of} takes {' '.join(f.upper() for f in shape)}; got {len(values)} value"
                f"{'s' if len(values) != 1 else ''}"
            )
        row: dict = {"date": when or date.today().isoformat(), "type": kind_of}
        for field_name, raw in zip(shape, values):
            if field_name == "symbol":
                row[field_name] = raw.upper()
                continue
            try:
                row[field_name] = float(raw)
            except ValueError:
                raise TechvalError(f"{field_name} must be a number, not {raw!r}") from None
        row["fee"] = fee
        row["acquired"] = acquired
        row["kind"] = kind
        row["note"] = note
        if not path.is_file():
            raise TechvalError(f"no portfolio file at {path}. techval invest init writes a starter.")
        line = flow_row(row)
        updated = append_row(path.read_text(encoding="utf-8"), line)
        # Replay the whole ledger with the row before anything touches the file.
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".portfolio-", suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(updated)
            ledger = Ledger.load(Path(tmp))
            ledger.positions()
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except TechvalError as err:
        console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(1)
    console.print(f"Added {escape(line.strip())}", highlight=False)
    console.print(f"Cash after the ledger's last row: {ledger.cash():,.2f}", highlight=False)


EXPORTS = ("realized", "lots", "activity")


def _share_text(n: float) -> str:
    return f"{n:g}" if float(n).is_integer() else f"{n:.6f}".rstrip("0")


@app.command()
def export(
    what: str = typer.Argument(..., metavar="TABLE", help="realized, lots or activity."),
    out: Path = typer.Option(..., "--out", "-o", help="The CSV file to write."),
    directory: Path = _DIR,
) -> None:
    """Write the ledger's realized sales, open lots or raw activity as CSV.

    realized has the columns of IRS Form 8949, one row per lot slice a sale
    closed, so it can be checked line by line against a broker's 1099-B.
    """
    import csv

    from .invest.ledger import Ledger

    try:
        if what not in EXPORTS:
            raise TechvalError(f"export takes realized, lots or activity, not {what!r}")
        ledger = Ledger.load(directory / "portfolio.yaml")
        if what == "realized":
            header = ["description", "date_acquired", "date_sold", "proceeds", "cost_basis", "gain", "term"]
            rows = [
                [
                    f"{_share_text(e.shares)} sh {e.symbol}",
                    e.opened.isoformat(),
                    e.sold.isoformat(),
                    f"{e.proceeds:.2f}",
                    f"{e.basis:.2f}",
                    f"{e.gain:.2f}",
                    e.term,
                ]
                for e in ledger.realized_events()
            ]
        elif what == "lots":
            header = ["symbol", "kind", "opened", "shares", "cost_per_share", "cost"]
            rows = [
                [p.symbol, p.kind, lot.opened.isoformat(), _share_text(lot.shares), f"{lot.cost_per_share:.6f}", f"{lot.shares * lot.cost_per_share:.2f}"]
                for p in sorted(ledger.positions().values(), key=lambda p: p.symbol)
                for lot in p.lots
            ]
        else:
            header = ["date", "type", "symbol", "shares", "price", "amount", "ratio", "fee", "acquired", "kind", "note"]
            rows = [
                [
                    t.date.isoformat(), t.type, t.symbol or "",
                    "" if t.shares is None else _share_text(t.shares),
                    "" if t.price is None else f"{t.price:g}",
                    "" if t.amount is None else f"{t.amount:g}",
                    "" if t.ratio is None else f"{t.ratio:g}",
                    f"{t.fee:g}" if t.fee else "",
                    t.acquired.isoformat() if t.acquired else "",
                    t.kind or "", t.note or "",
                ]
                for t in ledger.transactions
            ]
    except TechvalError as err:
        console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    console.print(f"Wrote {len(rows)} rows to {escape(str(out))}", highlight=False)


@app.command()
def check(directory: Path = _DIR) -> None:
    """Replay the ledger offline and print what it holds; no quotes, no models."""
    from .invest.ledger import Ledger

    try:
        ledger = Ledger.load(directory / "portfolio.yaml")
        positions = ledger.positions()
        cash = ledger.cash()
    except TechvalError as err:
        console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(1)
    held = sorted((p for p in positions.values() if p.shares > 0), key=lambda p: p.symbol)
    console.print(
        f"{escape(ledger.name)}: {len(ledger.transactions)} transactions from "
        f"{ledger.first_date} to {ledger.transactions[-1].date}, {len(held)} open positions."
    )
    for p in held:
        shares = f"{p.shares:,.0f}" if float(p.shares).is_integer() else f"{p.shares:,.4f}"
        console.print(f"  {p.symbol} {shares} {p.kind}, cost {p.cost:,.2f}", highlight=False)
    console.print(f"Cash {cash:,.2f}", highlight=False)
    if ledger.targets:
        console.print(f"Drift targets {sum(ledger.targets.values()):.0%} of the book", highlight=False)
    _warn_duplicates(ledger)
    console.print("The ledger replays cleanly.")


def _signed_pct(v: float | None, dp: int = 2) -> str:
    return "n/a" if v is None else f"{v * 100:+.{dp}f}%"


def _print_summary(snapshot: dict, quiet: bool = False) -> None:
    """The headline numbers in the terminal, for a run nobody opens the page after."""
    from rich.table import Table

    meta, over = snapshot["meta"], snapshot["overview"]
    risk = snapshot["performance"]["risk"]["portfolio"]
    gap = over["twr_pct"] - over["benchmark_twr_pct"]
    table = Table(title=escape(meta["name"]), show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column(justify="right")
    rows = [
        ("Total value", f"${over['value']:,.0f}"),
        ("Last session", f"{_signed_pct(over['day_pct'])} ({over['day_abs']:+,.0f})"),
        ("Time-weighted", f"{_signed_pct(over['twr_pct'], 1)} since {meta['first_date']}"),
        (f"vs {meta['benchmark']}", f"{gap * 100:+.1f} pts"),
        ("Max drawdown", _signed_pct(risk["max_drawdown"], 1) if risk["max_drawdown"] else "none"),
        ("Positions", f"{over['n_positions']} and ${over['cash']:,.0f} cash"),
        ("Engine coverage", f"{over['covered_value_share']:.0%}: {over['cheap']} cheap, {over['rich']} rich, {over['uncovered']} unvalued"),
        ("Prices as of", meta["prices_as_of"]),
    ]
    for label, value in rows:
        table.add_row(label, escape(value))
    if not quiet:
        console.print(table)
    if meta["prices_age_days"] > 4:
        console.print(f"[yellow]Prices are {meta['prices_age_days']} days old.[/yellow]")
    for u in over["unpriced"]:
        console.print(
            f"[yellow]{escape(u['symbol'])} could not be quoted, so it is valued at the last trade "
            f"price, {u['priced_at']:,.2f} on {u['price_date']}: {escape(u['reason'])}[/yellow]"
        )
    if over["mismatched_closes"]:
        console.print(
            f"[yellow]{escape(', '.join(over['mismatched_closes']))} closed in a different session from "
            f"{escape(meta['benchmark'])}.[/yellow]"
        )


def _warn_duplicates(ledger) -> None:
    """Name rows entered more than once; a warning, since two equal trades can be real."""
    for dupe in ledger.possible_duplicates():
        rows = dupe["rows"]
        listed = ", ".join(str(r) for r in rows[:-1]) + f" and {rows[-1]}"
        console.print(
            f"[yellow]{escape(dupe['summary'])} is entered {dupe['count']} times, rows {listed}. "
            "If that is one trade pasted twice, delete the copy.[/yellow]"
        )


def _cache_root(assumptions: Assumptions) -> Path:
    configured = assumptions.ml.cache_dir
    return Path(configured).expanduser() if configured else Path.home() / ".techval" / "ml"


# The panels ship with a checkout, not with the package: resolved from here
# they are found in a source tree and missing from an installed wheel, where
# --panels names them instead.
CHECKOUT_PANELS = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


def _build(
    directory: Path,
    config: Path | None,
    offline: bool,
    render_only: bool,
    refit: bool = False,
    panels: Path | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    benchmark: str | None = None,
    quiet: bool = False,
) -> None:
    from .invest.render import write_page
    from .invest.snapshot import validate_snapshot

    snapshot_path = directory / "snapshot.json"
    page_path = directory / "index.html"

    if render_only:
        if not snapshot_path.is_file():
            raise TechvalError(
                f"no snapshot at {snapshot_path}; run techval invest without --render first"
            )
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        validate_snapshot(snapshot)
        write_page(snapshot, page_path)
        if not quiet:
            console.print(f"Rendered {escape(str(page_path))} from the existing snapshot.")
        return

    from .edgar import EdgarClient, HttpCache
    from .invest.ledger import Ledger
    from .invest.quotes import Quotes
    from .invest.snapshot import build_snapshot, write_snapshot
    from .market import MarketData, make_price_source

    ledger = Ledger.load(directory / "portfolio.yaml")
    _warn_duplicates(ledger)
    if benchmark:
        # For this run only; the ledger file keeps its own benchmark.
        ledger.benchmark = benchmark.upper()
    assumptions = Assumptions.load(config)
    # Unlike the valuation commands, the tracker wants yesterday's close and
    # this quarter's filings, so its cache entries age out.
    cache = HttpCache(max_age=max_age_hours * 3600)
    source_kind = "csv" if offline else assumptions.price_source
    source = make_price_source(source_kind, cache, assumptions.price_csv_dir)
    today = date.today()
    quotes = Quotes(source, start=ledger.first_date, today=today)

    # Offline means offline: no DCFs rather than a half-guess about the cache.
    # Each covered name then carries the refusal naming the fix.
    facts_for = None
    market = None
    if not offline:
        client = EdgarClient(cache)
        facts_for = client.company_facts
        market = MarketData(source, cache, today=today)

    from .invest.engine_read import missing_panels

    fixtures = Path(panels) if panels is not None else CHECKOUT_PANELS
    absent = missing_panels(fixtures)
    if absent:
        console.print(
            "[yellow]"
            + escape(
                f"The model panels are not all under {fixtures} "
                f"({', '.join(p.name for p in absent.values())} missing), so "
                f"{' and '.join(sorted(absent))} refuse by name on the page. "
                "Point --panels at a techval checkout's tests/fixtures to restore them."
            )
            + "[/yellow]"
        )
    snapshot = build_snapshot(
        ledger,
        quotes,
        fixtures=fixtures,
        assumptions=assumptions,
        today=today,
        facts_for=facts_for,
        market=market,
        cache_root=_cache_root(assumptions),
        refit=refit,
    )
    write_snapshot(snapshot, snapshot_path)
    write_page(snapshot, page_path)
    _print_summary(snapshot, quiet)
    if not quiet:
        console.print(f"Wrote {escape(str(snapshot_path))}")
        console.print(f"Wrote {escape(str(page_path))}")
