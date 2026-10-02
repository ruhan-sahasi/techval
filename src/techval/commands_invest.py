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
# Transaction types: buy, sell, deposit, withdraw, dividend, split.
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
) -> None:
    """Refresh quotes and the engine read, then render portfolio/index.html."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        _build(directory, config, offline, render_only, refit, panels, max_age)
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
    kind_of: str = typer.Argument(..., metavar="TYPE", help="buy, sell, deposit, withdraw, dividend or split."),
    values: list[str] = typer.Argument(..., metavar="VALUES", help="buy/sell: SYMBOL SHARES PRICE. deposit/withdraw: AMOUNT. dividend: SYMBOL AMOUNT. split: SYMBOL RATIO."),
    when: str = typer.Option(None, "--date", help="The transaction date, YYYY-MM-DD. Defaults to today."),
    fee: float = typer.Option(None, "--fee", help="A trade's fee, for a buy or a sell."),
    kind: str = typer.Option(None, "--kind", help="etf or crypto, the first time a symbol appears."),
    note: str = typer.Option(None, "--note", help="A note kept on the row."),
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
    console.print("The ledger replays cleanly.")


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
        console.print(f"Rendered {escape(str(page_path))} from the existing snapshot.")
        return

    from .edgar import EdgarClient, HttpCache
    from .invest.ledger import Ledger
    from .invest.quotes import Quotes
    from .invest.snapshot import build_snapshot, write_snapshot
    from .market import MarketData, make_price_source

    ledger = Ledger.load(directory / "portfolio.yaml")
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
    over = snapshot["overview"]
    console.print(
        f"{escape(ledger.name)}: {len(snapshot['positions'])} positions, "
        f"total {over['value']:,.0f}."
    )
    console.print(f"Wrote {escape(str(snapshot_path))}")
    console.print(f"Wrote {escape(str(page_path))}")
