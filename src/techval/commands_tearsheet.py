"""techval tearsheet: one company's valuation, written as a page you can send.

The page is the results dashboard's valuation engine section measured for the
ticker asked about, from EDGAR and the configured price feed, or from recorded
payloads with --facts-dir. See ``techval.tearsheet``.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from .errors import TechvalError

app = typer.Typer()
console = Console(width=None if sys.stdout.isatty() else 120)

CHECKOUT = Path(__file__).resolve().parent.parent.parent


@app.command()
def tearsheet(
    ticker: str,
    config: Path = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the HTTP cache."),
    out: Path = typer.Option(None, "--out", "-o", help="Where to write the page. Default: out/tearsheet_<TICKER>.html."),
    facts_dir: Path = typer.Option(
        None, "--facts-dir", help="A directory of recorded companyfacts_<TICKER>.json payloads, instead of live SEC calls."
    ),
    panels: Path = typer.Option(
        None,
        "--panels",
        help="Directory holding fade_companyfacts.json.gz for base rates and scenarios. Defaults to tests/fixtures in this checkout.",
    ),
    open_page: bool = typer.Option(False, "--open", help="Open the page in the default browser once it is written."),
) -> None:
    """One company's valuation as a single page: the results page's engine section, measured for TICKER."""
    from .cli import _setup
    from .commands_expectations import CHECKOUT_PANELS, _panel
    from .config import ml_cache_root
    from .dashboard.collect import techval_commit
    from .dashboard.render import write_dashboard
    from .invest.engine_read import FADE_PANEL
    from .tearsheet import RecordedFacts, build_tearsheet

    symbol = ticker.upper()
    directory = Path(panels) if panels else CHECKOUT_PANELS
    path = Path(out) if out else Path("out") / f"tearsheet_{symbol}.html"
    try:
        assumptions, client, market = _setup(config, no_cache)
        if facts_dir is not None:
            knowledge = date.fromisoformat(assumptions.as_of) if assumptions.as_of else None
            client = RecordedFacts(facts_dir, knowledge)
        observations, _ = _panel(directory, symbol, ml_cache_root(assumptions))
        snap = build_tearsheet(
            symbol,
            assumptions,
            client=client,
            prices=market.source,
            price_feed=assumptions.price_source,
            observations=observations,
            panel=str(directory / FADE_PANEL),
            today=market.today,
            collected_at=market.today.isoformat(),
            commit=techval_commit(CHECKOUT),
        )
        write_dashboard(snap, path)
    except TechvalError as err:
        console.print(f"[red]{escape(str(err))}[/red]")
        raise typer.Exit(1)

    section = snap.sections["engine"]
    console.print(f"  {escape(section['takeaway'])}", highlight=False)
    refused = [r["what"] for r in section["refusals"]]
    if refused:
        console.print(
            f"  Not drawn: {escape('; '.join(refused))}. The page says why under each.", highlight=False
        )
    # A path is copied, so it is never wrapped.
    console.print(f"Wrote {escape(str(path))}.", soft_wrap=True)
    if open_page:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
