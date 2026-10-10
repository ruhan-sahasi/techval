"""techval expectations: what a market price assumes, read off the engine's DCF.

The forward DCF is built exactly as ``techval value`` builds it, then solved
for the price one lever at a time by ``techval.reverse_dcf``. With the fade
panel available the implied growth is set against base rates from its
point-in-time company-years; without it the command still runs and says so.

``techval scenarios`` reads the same panel the other way round: not how often
the price's path happened, but what the company-years like this one did, as a
bear, base and bull case valued through the same DCF.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .config import Assumptions
from .errors import TechvalError

app = typer.Typer()
# As in cli.py: a piped or redirected run gets a width that fits the tables
# rather than Rich's eighty columns.
console = Console(width=None if sys.stdout.isatty() else 120)

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


def _company(symbol: str, config: Path | None, no_cache: bool, facts_file: Path | None):
    """The company as ``techval value`` builds it, with what the commands here also need.

    Returns the assumptions, the price source, the market, the raw companyfacts
    payload, the reverse-DCF ``Case`` and the last close.
    """
    from types import SimpleNamespace

    from .edgar import CompanyFacts, EdgarClient, HttpCache
    from .ev_bridge import build_ev_bridge
    from .financials import build_financials
    from .market import MarketData, make_price_source
    from .reverse_dcf import Case
    from .wacc import compute_wacc

    assumptions = Assumptions.load(config)
    knowledge = date.fromisoformat(assumptions.as_of) if assumptions.as_of else None
    cache = HttpCache(enabled=not no_cache)
    source = make_price_source(assumptions.price_source, cache, assumptions.price_csv_dir)
    market = MarketData(source, cache, today=knowledge or date.today())
    if facts_file is not None:
        payload = json.loads(Path(facts_file).read_text(encoding="utf-8"))
    else:
        payload = EdgarClient(cache).company_facts(symbol).raw
    fin = build_financials(symbol, facts=CompanyFacts(payload, symbol, knowledge_date=knowledge))
    spot = market.spot(symbol)
    bridge = build_ev_bridge(fin, spot, assumptions)
    wacc = compute_wacc(fin, bridge, market, assumptions)
    return SimpleNamespace(
        assumptions=assumptions, source=source, market=market, payload=payload,
        case=Case(fin=fin, bridge=bridge, wacc=wacc, assumptions=assumptions), spot=spot,
    )


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
    history: bool = typer.Option(
        False,
        "--history",
        help="Also read what the price assumed at each quarter-end of the last three years, point in time.",
    ),
) -> None:
    """What the price assumes: the DCF solved for the market price, lever by lever, with base rates."""
    from .reverse_dcf import market_expectations

    symbol = ticker.upper()
    try:
        co = _company(symbol, config, no_cache, facts_file)
        observations, trailing = _panel(Path(panels) if panels else CHECKOUT_PANELS, symbol)
        result = market_expectations(
            co.case, price if price is not None else co.spot, trailing_growth=trailing, observations=observations
        )
        rows = None
        if history:
            from datetime import timedelta

            from .reverse_dcf import expectations_history, quarter_ends

            today = co.market.today
            dates = quarter_ends(today - timedelta(days=3 * 365), today)
            rows = expectations_history(co.payload, symbol, co.source, co.assumptions, dates)
    except TechvalError as err:
        console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(1)

    if as_json:
        out = result.to_dict()
        if rows is not None:
            out["history"] = rows
        typer.echo(json.dumps(out, indent=2, default=str))
        return
    _print(result)
    if rows is not None:
        _print_history(symbol, rows)


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


def _print_history(symbol: str, rows: list[dict]) -> None:
    table = Table(
        title=f"What {escape(symbol)}'s price assumed at each quarter-end, from what was public then",
        title_justify="left",
    )
    for name in ("Quarter", "Close", "Base case", "Cost of capital", "Implied return", "Growth needed"):
        table.add_column(name, justify="left" if name == "Quarter" else "right")
    refused = []
    for r in rows:
        if r["refused"]:
            refused.append(r)
            continue
        table.add_row(
            r["date"],
            f"{r['price']:,.2f}",
            f"{r['base_value']:,.2f}",
            _pct(r["wacc"]),
            _pct(r["implied_return"]),
            _pct(r["implied_growth"]),
        )
    console.print(table)
    for r in refused:
        console.print(f"  {r['date']} refused: {escape(r['refused'])}", highlight=False)


@app.command("expectations-screen")
def expectations_screen(
    tickers: list[str] = typer.Argument(..., help="The names to compare."),
    config: Path = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the HTTP cache."),
    facts_dir: Path = typer.Option(
        None, "--facts-dir", help="A directory of recorded companyfacts_<TICKER>.json payloads, instead of live SEC calls."
    ),
    panels: Path = typer.Option(None, "--panels", help="Directory holding fade_companyfacts.json.gz for base rates."),
    as_json: bool = typer.Option(False, "--json", help="Print the rows as JSON."),
) -> None:
    """What each price assumes, side by side, the most demanding first."""
    from .edgar import EdgarClient, HttpCache
    from .market import make_price_source
    from .reverse_dcf import screen

    try:
        assumptions = Assumptions.load(config)
        when = date.fromisoformat(assumptions.as_of) if assumptions.as_of else date.today()
        cache = HttpCache(enabled=not no_cache)
        source = make_price_source(assumptions.price_source, cache, assumptions.price_csv_dir)
        payloads, missing = {}, []
        client = None if facts_dir is not None else EdgarClient(cache)
        for raw in tickers:
            symbol = raw.upper()
            if facts_dir is not None:
                path = Path(facts_dir) / f"companyfacts_{symbol}.json"
                if not path.is_file():
                    missing.append({"ticker": symbol, "refused": f"no recorded payload at {path}"})
                    continue
                payloads[symbol] = json.loads(path.read_text(encoding="utf-8"))
            else:
                payloads[symbol] = client.company_facts(symbol).raw
        observations, _ = _panel(Path(panels) if panels else CHECKOUT_PANELS, "")
        rows = screen(payloads, source, assumptions, when, observations) + missing
    except TechvalError as err:
        console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(1)

    if as_json:
        typer.echo(json.dumps(rows, indent=2, default=str))
        return
    table = Table(title=f"What each price assumes on {when}, the most demanding first", title_justify="left")
    for name in (
        "Ticker", "Close", "Base case", "Implied return", "Growth needed",
        "Company-years that did it", "Simulated above", "Vs scenarios",
    ):
        table.add_column(name, justify="left" if name == "Ticker" else "right")
    for r in rows:
        if r["refused"]:
            continue
        did = "n/a" if r["base_n"] is None else f"{r['base_hits']:,} of {r['base_n']:,}"
        above = "n/a" if r["simulated_above"] is None else f"{r['simulated_above']:.1%}"
        table.add_row(
            r["ticker"], f"{r['price']:,.2f}", f"{r['base_value']:,.2f}",
            _pct(r["implied_return"]),
            "beyond 300%" if r["implied_growth"] is None else _pct(r["implied_growth"]),
            did, above, _against(r),
        )
    console.print(table)
    for r in rows:
        if r["refused"]:
            console.print(f"  {r['ticker']} refused: {escape(r['refused'])}", highlight=False)


@app.command()
def scenarios(
    ticker: str,
    config: Path = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the HTTP cache."),
    facts_file: Path = typer.Option(
        None, "--facts", help="Recorded companyfacts payload for TICKER, instead of a live SEC call."
    ),
    price: float = typer.Option(None, "--price", help="Weigh the scenarios against this price instead of the last close."),
    panels: Path = typer.Option(
        None,
        "--panels",
        help="Directory holding fade_companyfacts.json.gz. Defaults to tests/fixtures in this checkout.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the whole result as JSON."),
) -> None:
    """Bear, base and bull from what similar company-years went on to do, valued through the engine's DCF."""
    from .scenarios import band_coverage, value_scenarios

    symbol = ticker.upper()
    directory = Path(panels) if panels else CHECKOUT_PANELS
    try:
        observations, trailing = _panel(directory, symbol)
        if observations is None:
            console.print(
                f"[red]There is no fade panel in {escape(str(directory))}, and the scenarios are read "
                "from it; point --panels at a checkout's tests/fixtures.[/red]"
            )
            raise typer.Exit(1)
        co = _company(symbol, config, no_cache, facts_file)
        result = value_scenarios(co.case, price if price is not None else co.spot, observations, trailing=trailing)
        calibration = band_coverage(observations)
    except TechvalError as err:
        console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(1)

    if as_json:
        typer.echo(json.dumps({**result.to_dict(), "calibration": calibration}, indent=2, default=str))
        return
    table = Table(title=f"{escape(symbol)}: bear, base and bull from what similar company-years did", title_justify="left")
    for name in ("Scenario", "Percentile", "Company-years", "5-year CAGR", "Growth, years 1 to 5", "Value", "Weight"):
        table.add_column(name, justify="left" if name in ("Scenario", "Growth, years 1 to 5") else "right")
    for s in result.scenarios:
        table.add_row(
            s.name.capitalize(), f"P{s.quantile * 100:.0f}", f"{s.n:,}", _pct(s.cagr),
            "  ".join(f"{g:.1%}" for g in s.growth[:5]), f"{s.value:,.2f}", f"{s.weight:.0%}",
        )
    console.print(table)
    for sentence in [*result.sentences, *result.notes, _calibration_sentence(calibration)]:
        console.print(f"  {escape(sentence)}", highlight=False)


def _calibration_sentence(c: dict) -> str:
    return (
        f"Held out from {c['test_from'][:4]}, the band from the 10th to the 90th percentile held "
        f"{c['share_inside']:.1%} of {c['n']:,} company-years it had not seen, against a nominal "
        f"{c['nominal']:.0%}; {c['share_above']:.1%} beat it and {c['share_below']:.1%} fell below."
    )


def _against(row: dict) -> str:
    """Where a price sits against its bear, base and bull, in a table cell."""
    side = row.get("scenario_side")
    if side is None:
        return "n/a"
    if side == "above":
        return f"{row['price'] / row['scenario_bull']:.1f}x the bull"
    if side == "below":
        return "below the bear"
    return f"{row['scenario_weight']:.0%} {side}"
