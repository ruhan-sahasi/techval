"""techval expectations: what a market price assumes, read off the engine's DCF.

The forward DCF is built exactly as ``techval value`` builds it, then solved
for the price one lever at a time by ``techval.reverse_dcf``. With the fade
panel available the implied growth is set against base rates from its
point-in-time company-years; without it the command still runs and says so.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .config import Assumptions
from .errors import TechvalError

app = typer.Typer()
console = Console()

CHECKOUT_PANELS = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"

LEVER_NAMES = {
    "discount_rate": "Discount rate",
    "first_year_growth": "First-year growth",
    "terminal_margin": "Terminal margin",
    "duration": "Duration of current growth",
}


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _cell(solve) -> str:
    if solve.lever == "duration":
        return f"{solve.implied:.0f} years" if solve.reached else f"over {solve.edge:.0f} years"
    if solve.reached:
        return _pct(solve.implied)
    return f"beyond {solve.edge:.0%}"


@app.command()
def expectations(
    ticker: str,
    config: Path = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the HTTP cache."),
    facts_file: Path = typer.Option(
        None, "--facts", help="Recorded companyfacts payload for TICKER, instead of a live SEC call."
    ),
    price: float = typer.Option(
        None, "--price", help="Ask what this price assumes, instead of the last close."
    ),
    panels: Path = typer.Option(
        None,
        "--panels",
        help="Directory holding fade_companyfacts.json.gz for base rates. Defaults to tests/fixtures in this checkout.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the whole result as JSON."),
) -> None:
    """What the price assumes: the DCF solved for the market price, lever by lever, with base rates."""
    from .edgar import CompanyFacts, EdgarClient, HttpCache
    from .ev_bridge import build_ev_bridge
    from .financials import build_financials
    from .market import MarketData, make_price_source
    from .reverse_dcf import Case, market_expectations
    from .wacc import compute_wacc

    symbol = ticker.upper()
    try:
        assumptions = Assumptions.load(config)
        knowledge = date.fromisoformat(assumptions.as_of) if assumptions.as_of else None
        cache = HttpCache(enabled=not no_cache)
        source = make_price_source(assumptions.price_source, cache, assumptions.price_csv_dir)
        market = MarketData(source, cache, today=knowledge or date.today())
        if facts_file is not None:
            payload = json.loads(Path(facts_file).read_text(encoding="utf-8"))
            fin = build_financials(symbol, facts=CompanyFacts(payload, symbol, knowledge_date=knowledge))
        else:
            fin = build_financials(symbol, client=EdgarClient(cache, knowledge_date=knowledge))
        spot = market.spot(symbol)
        bridge = build_ev_bridge(fin, spot, assumptions)
        wacc = compute_wacc(fin, bridge, market, assumptions)
        case = Case(fin=fin, bridge=bridge, wacc=wacc, assumptions=assumptions)

        observations, trailing = _panel(Path(panels) if panels else CHECKOUT_PANELS, symbol)
        result = market_expectations(
            case, price if price is not None else spot, trailing_growth=trailing, observations=observations
        )
    except TechvalError as err:
        console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(1)

    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, default=str))
        return
    _print(result)


def _panel(directory: Path, symbol: str):
    """The fade panel's observations and the company's own trailing growth, if both exist."""
    from .invest.engine_read import FADE_PANEL, fade_panel

    if not (directory / FADE_PANEL).is_file():
        return None, None
    observations = fade_panel(directory).observations
    own = [o for o in observations if o.ticker == symbol and o.growth is not None]
    trailing = max(own, key=lambda o: o.fiscal_year_end).growth if own else None
    return observations, trailing


def _print(result) -> None:
    levers = Table(title=f"What {escape(result.ticker)} at {result.price:,.2f} assumes", title_justify="left")
    levers.add_column("Lever")
    levers.add_column("Assumed", justify="right")
    levers.add_column("Implied by the price", justify="right")
    for solve in result.solves:
        assumed = "" if solve.assumed is None else _pct(solve.assumed)
        if solve.lever == "duration":
            assumed = "5-year fade"
        levers.add_row(LEVER_NAMES[solve.lever], assumed, _cell(solve))
    console.print(levers)

    frontier = Table(title="Frontier: first-year growth needed at each terminal margin", title_justify="left")
    frontier.add_column("Terminal margin", justify="right")
    frontier.add_column("First-year growth", justify="right")
    for row in result.frontier:
        need = _pct(row["implied_growth"]) if row["implied_growth"] is not None else "beyond 300%"
        frontier.add_row(_pct(row["margin"]), need)
    console.print(frontier)

    for sentence in result.summary():
        console.print(f"  {escape(sentence)}", highlight=False)
