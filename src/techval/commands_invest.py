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
) -> None:
    """Refresh quotes and the engine read, then render portfolio/index.html."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        _build(directory, config, offline, render_only)
    except TechvalError as err:
        console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(1)


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


def _build(directory: Path, config: Path | None, offline: bool, render_only: bool) -> None:
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
    cache = HttpCache(enabled=True)
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

    fixtures = Path(__file__).parent.parent.parent / "tests" / "fixtures"
    snapshot = build_snapshot(
        ledger,
        quotes,
        fixtures=fixtures,
        assumptions=assumptions,
        today=today,
        facts_for=facts_for,
        market=market,
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
