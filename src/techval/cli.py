"""Command line interface.

All rendering lives here. The numeric modules return dataclasses and DataFrames
and know nothing about terminals, so the same objects drive the CLI, the notebook
and anything else built on the package.

Four commands:

    techval value  TICKER      full valuation: statements, bridge, WACC, DCF, comps, chart
    techval comps  TICKER      the comp table alone
    techval merger ACQ TGT     accretion/dilution
    techval fetch  TICKER      warm the cache and print the provenance table
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .comps import run_comps
from .apv import run_apv
from .backtest import (
    ForwardPrices,
    edgar_client_factory,
    market_factory_from,
    run_backtest,
)
from .dilution import build_share_count
from .config import Assumptions
from .dcf import run_dcf, sensitivity_wacc_exit, sensitivity_wacc_growth
from .edgar import EdgarClient, HttpCache
from .errors import TechvalError
from .ev_bridge import build_ev_bridge
from .financials import build_financials
from .football import FootballRow, football_field
from .market import MarketData, make_price_source
from .merger import run_merger
from .simulation import run_simulation
from .wacc import compute_wacc, estimate_beta

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="DCF, trading comps and merger analysis for US-listed technology companies.",
)
# A comp table is thirteen columns wide. Rich falls back to eighty when stdout
# is not a terminal, which crushes every column to three characters in a piped
# or redirected run, so a redirected run is given a width that fits the tables.
console = Console(width=None if sys.stdout.isatty() else 120)

_CFG = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file.")
_NOCACHE = typer.Option(False, "--no-cache", help="Bypass the HTTP cache.")
_OUT = typer.Option("out", "--out", "-o", help="Directory for chart output.")
_ASOF = typer.Option(
    None,
    "--as-of",
    help=(
        "Value the company as it was knowable on this date (YYYY-MM-DD). Facts filed "
        "later are discarded and prices stop there, so the run uses only information "
        "that existed at the time."
    ),
)


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #


def _rule(title: str) -> None:
    console.print()
    console.rule(Text(title, style="bold"), style="dim")


def _money(v: float | None, dp: int = 0) -> str:
    if v is None:
        return "n/a"
    return f"({abs(v):,.{dp}f})" if v < 0 else f"{v:,.{dp}f}"


def _pct(v: float | None, dp: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{dp}%}"


def _mult(v: float | None) -> str:
    return "NM" if v is None else f"{v:,.1f}x"


# Rows whose value is a fraction, not an amount. Printing 0.176 where the label
# says "%" understates the figure by a hundred and reads as seventeen cents.
_PERCENT_ROWS = (
    "premium to unaffected price",
    "accretion / (dilution), %",
    "mix: cash",
    "mix: stock",
)


def _kv_table(rows: list[tuple[str, float]], *, title: str, dp: int = 1) -> Table:
    t = Table(title=title, title_justify="left", box=None, pad_edge=False)
    t.add_column("", style="", no_wrap=True)
    t.add_column("", justify="right")
    for label, value in rows:
        style = "bold" if label.lower().startswith(("enterprise", "pro forma eps")) else ""
        shown = (
            _pct(value) if label.lower() in _PERCENT_ROWS else _money(value, dp)
        )
        t.add_row(Text(label, style=style), Text(shown, style=style))
    return t


def _opt(v, fmt):
    return "n/a" if v is None else fmt(v)


def _frame_table(df: pd.DataFrame, *, title: str, index_label: str = "") -> Table:
    t = Table(title=title, title_justify="left", box=None, pad_edge=False)
    t.add_column(index_label, no_wrap=True)
    for c in df.columns:
        t.add_column(str(c), justify="right")
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("NM")
            elif isinstance(v, float):
                cells.append(f"{v:,.1f}" if abs(v) < 1000 else f"{v:,.0f}")
            else:
                cells.append(str(v))
        t.add_row(str(idx), *cells)
    return t


def _notes(items: list[str], *, heading: str = "Notes") -> None:
    if not items:
        return
    console.print(f"\n[bold]{heading}[/bold]")
    for n in items:
        style = "yellow" if n.upper().startswith(("FLAG", "WARN")) else "dim"
        console.print(Text(f"  - {n}", style=style))


def _setup(config: Path | None, no_cache: bool, as_of: str | None = None):
    assumptions = Assumptions.load(config)
    if as_of:
        assumptions.as_of = as_of

    # A dated run sees only what was filed by that date and prices that stopped
    # there. Both halves are needed: filings alone would still be marked to
    # today's price, which is the more obvious half of the same hindsight.
    knowledge = date.fromisoformat(assumptions.as_of) if assumptions.as_of else None

    cache = HttpCache(enabled=not no_cache)
    client = EdgarClient(cache, knowledge_date=knowledge)
    source = make_price_source(
        assumptions.price_source, cache, assumptions.price_csv_dir
    )
    market = MarketData(source, cache, today=knowledge or date.today())
    return assumptions, client, market


def _load(ticker: str, client, market, assumptions):
    fin = build_financials(ticker, client=client)
    price = market.spot(ticker)
    bridge = build_ev_bridge(fin, price, assumptions)
    return fin, price, bridge


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #


def _render_financials(fin) -> None:
    _rule(f"{fin.entity_name} ({fin.ticker})  normalized TTM, USD millions")
    console.print(
        f"[dim]CIK {fin.cik}   twelve months ended {fin.as_of}[/dim]\n"
    )
    rows = [
        ("Revenue", fin.revenue),
        ("Gross profit", fin.gross_profit),
        ("EBIT", fin.ebit),
        ("D&A", fin.da),
        ("EBITDA", fin.ebitda),
        ("Stock-based compensation", fin.sbc),
        ("Net income", fin.net_income),
        ("Capital expenditure", fin.capex),
        ("Diluted shares (mm)", fin.diluted_shares),
    ]
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", justify="right")
    t.add_column("", justify="right", style="dim")
    for label, v in rows:
        margin = ""
        if v is not None and fin.revenue and label in (
            "Gross profit",
            "EBIT",
            "EBITDA",
            "Stock-based compensation",
        ):
            margin = f"{v / fin.revenue:.1%} of revenue"
        t.add_row(label, _money(v, 1), margin)
    console.print(t)
    _notes(fin.warnings, heading="Data quality")


def _render_share_count(sc) -> None:
    """The walk from shares outstanding to a fully diluted count."""
    _rule("Share count")
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", justify="right")
    for label, value in sc.rows():
        bold = label.lower().startswith("fully diluted")
        t.add_row(Text(label, style="bold" if bold else ""),
                  Text(f"{value:,.2f}", style="bold" if bold else ""))
    console.print(t)
    console.print(f"\n[dim]Method: {sc.method}"
                  + (f", from {sc.accession}" if sc.accession else "")
                  + (f" as of {sc.as_of}" if sc.as_of else "") + ".[/dim]")
    _notes(list(sc.flags), heading="Flags")
    _notes(list(sc.notes))


def _render_regression(fit) -> None:
    """The multiple the target's own fundamentals warrant, and the residual."""
    _rule("Warranted multiple from fundamentals")
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", justify="right")
    for label, value in fit.rows():
        low = label.lower()
        if value is None:
            shown = "n/a"
        elif "multiple" in low or "residual" in low:
            shown = f"{value:,.2f}x"
        elif "price" in low or "implied ev" in low:
            shown = _money(value)
        elif "r-squared" in low:
            shown = f"{value:.3f}"
        else:
            shown = f"{value:,.3f}"
        bold = "warranted" in low or "residual" in low
        t.add_row(Text(label, style="bold" if bold else ""),
                  Text(shown, style="bold" if bold else ""))
    console.print(t)
    _notes(list(fit.notes))


def _render_purchase_accounting(pa, acq_ticker: str) -> None:
    """Opening balance sheet and the pro forma years it feeds."""
    _rule("Purchase accounting: opening balance sheet")
    console.print(_kv_table(pa.opening.rows(), title="", dp=0))

    _rule("Pro forma")
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    for y in pa.years:
        t.add_column(f"Year {y.year}", justify="right")
    rows = [
        ("Revenue", "revenue", 0), ("EBITDA", "ebitda", 0),
        ("Intangible amortisation", "intangible_amortisation", 0),
        ("EBIT", "ebit", 0), ("Interest", "interest", 0),
        ("Taxes", "taxes", 0), ("Net income", "net_income", 0),
        ("Diluted shares (mm)", "shares", 1), ("EPS", "eps", 3),
        ("Debt balance", "debt_balance", 0), ("Debt repaid", "debt_repaid", 0),
    ]
    for label, attr, dp in rows:
        bold = label in ("EPS", "Net income")
        cells = [Text(_money(getattr(y, attr), dp), style="bold" if bold else "")
                 for y in pa.years]
        t.add_row(Text(label, style="bold" if bold else ""), *cells)
    acc = [Text(_money(a, 3) if a is not None else "NM") for a in pa.accretion_by_year]
    t.add_row(Text("Accretion / (dilution)", style="bold"), *acc)
    console.print(t)
    _notes(list(pa.checks), heading="Cross-checks")
    _notes(list(pa.notes))


def _render_simulation(sim) -> None:
    """The distribution, and what it is and is not a probability about."""
    _rule("Monte Carlo")
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    for c in ("5th", "25th", "Median", "75th", "95th", "Mean", "SD"):
        t.add_column(c, justify="right")
    for label, pc, fmt in (
        ("Value per share", sim.per_share, lambda v: f"{v:,.2f}"),
        ("Enterprise value", sim.enterprise_value, lambda v: f"{v:,.0f}"),
    ):
        t.add_row(label, *[fmt(v) for v in
                           (pc.p5, pc.p25, pc.p50, pc.p75, pc.p95, pc.mean, pc.sd)])
    console.print(t)
    console.print(
        f"\n  [bold]P(value > {sim.current_price:,.2f} market price)   "
        f"{sim.prob_above_price:.1%}[/bold]"
    )
    console.print(
        f"  [bold]P(value > 20% above market price)       "
        f"{sim.prob_upside_20pct:.1%}[/bold]"
    )
    console.print(
        f"\n[dim]{sim.draws:,} draws, seed {sim.seed}, {sim.failed_draws:,} rejected. "
        "These are probabilities under the ASSUMED distribution, not about the world: "
        "a tight distribution around a wrong central case is confidently wrong.[/dim]"
    )

    _rule("Tornado")
    t = Table(box=None, pad_edge=False)
    t.add_column("Driver")
    t.add_column("Low", justify="right")
    t.add_column("High", justify="right")
    t.add_column("Price low", justify="right")
    t.add_column("Price high", justify="right")
    t.add_column("Swing", justify="right")
    for d in sim.tornado:
        t.add_row(d.label, f"{d.low_value:,.3f}", f"{d.high_value:,.3f}",
                  f"{d.low_price:,.2f}", f"{d.high_price:,.2f}",
                  Text(f"{d.swing:,.2f}", style="bold"))
    console.print(t)
    _notes(list(sim.checks), heading="Cross-checks")
    _notes(list(sim.notes))


def _render_apv(a) -> None:
    """Unlevered value plus financing side effects, reconciled to the WACC answer."""
    _rule("Adjusted present value")
    rows = [
        ("Unlevered cost of equity", a.unlevered_cost_of_equity),
        ("Unlevered value", a.unlevered_value),
        ("+ PV of tax shield, explicit period", a.pv_shield_explicit),
        ("+ PV of tax shield, terminal", a.pv_shield_terminal),
        ("Total value (APV)", a.total_value),
        ("Enterprise value (WACC)", a.wacc_enterprise_value),
        ("Difference", a.difference),
    ]
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", justify="right")
    for label, v in rows:
        bold = label.startswith(("Total value", "Difference"))
        shown = f"{v:.2%}" if "cost of equity" in label.lower() else _money(v)
        t.add_row(Text(label, style="bold" if bold else ""),
                  Text(shown, style="bold" if bold else ""))
    console.print(t)
    console.print(
        f"\n[dim]Difference of {a.difference_pct:+.2%} against the WACC answer. "
        f"Shield discounted at the {a.shield_discount_rate.replace('_', ' ')} "
        f"({a.shield_rate:.2%}).[/dim]"
    )
    _notes(list(a.checks), heading="Reconciliation")
    _notes(list(a.notes))


def _render_bridge(bridge, fin) -> None:
    _rule("Enterprise value bridge")
    console.print(_kv_table(bridge.rows(), title="", dp=1))
    den, label = bridge.multiple_denominator(fin)
    console.print(
        f"\n[dim]Convention: {bridge.lease_convention}. "
        f"Multiples pair this EV with {label} = {_money(den, 1)}.[/dim]"
    )
    console.print(
        f"[dim]EV excluding operating leases {_money(bridge.ev_excluding_leases)}   "
        f"including {_money(bridge.ev_including_leases)}[/dim]"
    )
    _notes(bridge.notes)


def _render_wacc(w) -> None:
    _rule("WACC buildup")
    t = Table(box=None, pad_edge=False)
    t.add_column("Input")
    t.add_column("Value", justify="right")
    t.add_column("Source", style="dim")
    for r in w.buildup_rows():
        kind = r.get("kind", "")
        v = r["value"]
        if kind == "rate" or kind == "ratio":
            shown = f"{v:.2%}" if kind == "rate" else f"{v:,.2f}"
        elif kind == "beta":
            shown = f"{v:,.3f}"
        elif kind == "usd_mm":
            shown = _money(v)
        else:
            shown = f"{v:,.0f}"
        bold = r["label"] in ("WACC", "Cost of equity")
        t.add_row(
            Text(r["label"], style="bold" if bold else ""),
            Text(shown, style="bold" if bold else ""),
            r["source"],
        )
    console.print(t)
    _notes(w.notes)


def _render_dcf(d, fin) -> None:
    _rule("Discounted cash flow")
    # Transposed: line items down the side, years across the top. That is how a
    # projection is read, and it also keeps thirteen metrics inside a terminal
    # width that would otherwise crush every column to three characters.
    df = d.to_frame().copy()
    pct_rows = {"Growth", "EBIT margin"}
    df.index = [f"Year {y}" for y in df.index]
    t = df.transpose()

    table = Table(title="Projection, USD mm", title_justify="left", box=None,
                  pad_edge=False)
    table.add_column("")
    for c in t.columns:
        table.add_column(str(c), justify="right")
    for label, row in t.iterrows():
        cells = []
        for v in row:
            if v is None or pd.isna(v):
                cells.append("n/a")
            elif label in pct_rows:
                cells.append(f"{v:.1%}")
            elif label == "Discount factor":
                cells.append(f"{v:.3f}")
            else:
                cells.append(f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.1f}")
        bold = label in ("FCFF", "PV")
        table.add_row(Text(str(label), style="bold" if bold else ""),
                      *(Text(c, style="bold" if bold else "") for c in cells))
    console.print(table)

    summary = [
        ("Enterprise value, Gordon growth", d.enterprise_value_gordon),
        ("Enterprise value, exit multiple", d.enterprise_value_exit),
        ("Enterprise value, value driver", d.enterprise_value_value_driver),
        ("Equity value, Gordon growth", d.equity_value_gordon),
        ("Equity value, exit multiple", d.equity_value_exit),
        ("Equity value, value driver", d.equity_value_value_driver),
    ]
    console.print()
    console.print(_kv_table(summary, title="", dp=0))

    # All three terminal methods are always computed. The assumption picks which
    # one carries the headline, and the reader should see which that was.
    headline = getattr(d, "headline_method", "gordon")
    for key, label, v in (
        ("gordon", "Gordon growth", d.per_share_gordon),
        ("exit_multiple", "exit multiple", d.per_share_exit),
        ("value_driver", "value driver", d.per_share_value_driver),
    ):
        shown = f"${v:,.2f}" if v is not None else "not computed"
        mark = "  <- headline" if key == headline else ""
        console.print(
            f"\n  [bold]Implied per share, {label:<14s} {shown}[/bold][dim]{mark}[/dim]"
        )
    console.print(
        f"\n[dim]Per-share figures divide by {d.shares_for_value:,.1f}mm shares.[/dim]"
    )
    _notes(d.checks, heading="Cross-checks")
    _notes(d.notes)


def _render_comps(result) -> None:
    _rule(f"Trading comparables: {result.target.ticker}")
    df = result.table()
    show = df[
        [
            c
            for c in (
                "Ticker",
                "Price",
                "Market cap",
                "EV",
                "Revenue",
                "Rev growth",
                "EBITDA margin",
                "Rule of 40",
                "EV/Revenue",
                "EV/Gross Profit",
                "EV/EBITDA",
                "EV/EBIT",
                "P/E",
            )
            if c in df.columns
        ]
    ].copy()
    for c in ("Rev growth", "EBITDA margin"):
        if c in show:
            show[c] = show[c].map(lambda v: _pct(v) if pd.notna(v) else "NM")
    for c in ("EV/Revenue", "EV/Gross Profit", "EV/EBITDA", "EV/EBIT", "P/E"):
        if c in show:
            show[c] = show[c].map(lambda v: _mult(v) if pd.notna(v) else "NM")
    for c in ("Price", "Rule of 40"):
        if c in show:
            show[c] = show[c].map(lambda v: f"{v:,.1f}" if pd.notna(v) else "NM")
    for c in ("Market cap", "EV", "Revenue"):
        if c in show:
            show[c] = show[c].map(lambda v: _money(v) if pd.notna(v) else "NM")
    console.print(_frame_table(show.set_index("Ticker"), title="", index_label="Ticker"))

    console.print()
    console.print(_frame_table(result.stats, title="Peer statistics", index_label=""))
    console.print()
    console.print(
        _frame_table(result.implied, title="Implied value for the target", index_label="")
    )

    if getattr(result, "regression", None) is not None:
        _render_regression(result.regression)

    if result.exclusions:
        console.print("\n[bold yellow]Peers excluded[/bold yellow]")
        for e in result.exclusions:
            console.print(f"  [yellow]- {e.ticker}: {e.reason}[/yellow]")
    flags = [
        f"{p.ticker}: {f}" for p in result.peers for f in p.flags
    ]
    _notes(flags, heading="Suppressed multiples")
    _notes(result.notes)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


@app.command()
def value(
    ticker: str,
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    out: str = _OUT,
    as_of: str = _ASOF,
) -> None:
    """Full valuation: statements, EV bridge, WACC, DCF, comps and football field."""
    try:
        assumptions, client, market = _setup(config, no_cache, as_of)
        if assumptions.as_of:
            console.print(
                f"[yellow]Point in time: valuing {ticker.upper()} as it was knowable "
                f"on {assumptions.as_of}.[/yellow]"
            )
        fin, price, bridge = _load(ticker, client, market, assumptions)
        _render_financials(fin)

        if assumptions.dilution.method == "treasury_stock":
            try:
                _render_share_count(
                    build_share_count(ticker, price, fin, assumptions, client)
                )
            except TechvalError as exc:
                _rule("Share count")
                console.print(
                    f"[yellow]Treasury stock count unavailable, falling back to "
                    f"diluted WASO: {exc}[/yellow]"
                )

        _render_bridge(bridge, fin)

        comps_result = None
        peer_betas = None
        peer_median_ev_ebitda = None
        if assumptions.comps.peers:
            comps_result = run_comps(ticker, assumptions, client, market)
            median = comps_result.stats.loc["Median"].get("EV/EBITDA")
            if median is not None and pd.notna(median):
                peer_median_ev_ebitda = float(median)
            peer_betas = _peer_betas(comps_result, market, assumptions)

        w = compute_wacc(fin, bridge, market, assumptions, peer_betas=peer_betas)
        _render_wacc(w)

        # A DCF that cannot be formed, because the terminal year is a loss or an
        # assumption is inconsistent, is reported and stepped past. The comps
        # and the chart stand on their own evidence, and discarding them because
        # one methodology declined would throw away the part of the report that
        # still means something.
        d = None
        growth_grid = exit_grid = None
        try:
            d = run_dcf(
                fin,
                bridge,
                w,
                assumptions,
                peer_median_ev_ebitda=peer_median_ev_ebitda,
            )
            _render_dcf(d, fin)

            _rule("Sensitivities")
            # Centred on the discount rate the DCF actually used, which is the
            # override when one is set, not the computed WACC beside it.
            growth_grid = sensitivity_wacc_growth(fin, bridge, assumptions, d.wacc)
            console.print(
                _frame_table(
                    growth_grid,
                    title="Implied price per share: WACC by terminal growth",
                    index_label="WACC",
                )
            )
            if peer_median_ev_ebitda:
                exit_grid = sensitivity_wacc_exit(
                    fin, bridge, assumptions, d.wacc, peer_median_ev_ebitda
                )
                console.print()
                console.print(
                    _frame_table(
                        exit_grid,
                        title="Implied price per share: WACC by exit multiple",
                        index_label="WACC",
                    )
                )
        except TechvalError as exc:
            _rule("Discounted cash flow")
            console.print(
                f"[yellow]The DCF could not be formed and is omitted: {exc}[/yellow]"
            )

        # Both read the DCF that was just built, so a failed DCF takes them with it
        # rather than being reported against nothing.
        if d is not None and assumptions.simulation.enabled:
            try:
                _render_simulation(
                    run_simulation(
                        fin, bridge, w, assumptions,
                        peer_median_ev_ebitda=peer_median_ev_ebitda,
                    )
                )
            except TechvalError as exc:
                _rule("Monte Carlo")
                console.print(f"[yellow]Simulation not run: {exc}[/yellow]")

        if d is not None and assumptions.apv.enabled:
            try:
                _render_apv(run_apv(fin, bridge, w, d, assumptions))
            except TechvalError as exc:
                _rule("Adjusted present value")
                console.print(f"[yellow]APV not run: {exc}[/yellow]")

        if comps_result is not None:
            _render_comps(comps_result)

        path = _chart(
            ticker, fin, market, d, comps_result, price, out,
            growth_grid=growth_grid, exit_grid=exit_grid,
        )
        _rule("Output")
        console.print(f"Football field written to [bold]{path}[/bold]")
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


def _peer_betas(comps_result, market, assumptions):
    """Levered peer betas, unlevered at each peer's own capital structure.

    The leverage ratio here has to be the same one the target is relevered at,
    which is gross book debt over market equity. Using net debt instead, and
    clamping it at zero for the cash-rich, unlevers every peer at 0.0x, leaves
    the asset beta equal to the levered beta, and then reapplies the target's
    leverage on top. That inflates beta and the cost of equity, and it does so
    silently because both halves look reasonable in isolation.
    """
    out = []
    for p in comps_result.peers:
        try:
            de = (p.gross_debt / p.market_cap) if p.market_cap else 0.0
            out.append(
                estimate_beta(
                    market.prices(p.ticker),
                    market.prices(assumptions.market.market_index),
                    adjustment=assumptions.market.beta_adjustment,
                    debt_to_equity=max(de, 0.0),
                    tax_rate=assumptions.tax.marginal_tax_rate,
                    lookback_years=assumptions.market.beta_lookback_years,
                )
            )
        except TechvalError:
            continue
    return out or None


def _grid_band(grid) -> tuple[float, float] | None:
    """The span of a sensitivity grid, ignoring cells that were withheld."""
    if grid is None:
        return None
    values = pd.to_numeric(grid.stack(), errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        return None
    return float(values.min()), float(values.max())


def _chart(
    ticker, fin, market, dcf_result, comps_result, price, out_dir,
    *, growth_grid=None, exit_grid=None,
):
    rows: list[FootballRow] = []
    lo, hi = market.prices(ticker).fifty_two_week_range()
    rows.append(FootballRow("52-week trading range", lo, hi))

    if comps_result is not None:
        for label in ("EV/Revenue", "EV/Gross Profit", "EV/EBITDA"):
            if label not in comps_result.implied.index:
                continue
            r = comps_result.implied.loc[label]
            low, high = r.get("Implied price low"), r.get("Implied price high")
            if pd.notna(low) and pd.notna(high) and low > 0:
                rows.append(FootballRow(f"Comps: {label}", float(low), float(high)))

    # The DCF bands are the full span of the sensitivity grids, so the bar shows
    # the same range the tables print rather than a decorative percentage. A
    # point estimate implies a precision the inputs do not support.
    subtitle_dcf = ""
    if dcf_result is not None:
        for label, base, band in (
            ("DCF: Gordon growth", dcf_result.per_share_gordon, _grid_band(growth_grid)),
            ("DCF: exit multiple", dcf_result.per_share_exit, _grid_band(exit_grid)),
        ):
            if band is not None:
                rows.append(FootballRow(label, band[0], band[1]))
            elif base and base > 0:
                rows.append(FootballRow(label, base * 0.85, base * 1.15))
        subtitle_dcf = (
            " DCF bands span the sensitivity grids."
            if (growth_grid is not None or exit_grid is not None)
            else " DCF bands are the point estimate plus or minus 15%."
        )

    path = Path(out_dir) / f"{ticker.upper()}_football.png"
    return football_field(
        rows,
        price,
        ticker.upper(),
        path,
        subtitle=f"TTM to {fin.as_of}." + subtitle_dcf,
    )


@app.command()
def comps(
    ticker: str,
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    as_of: str = _ASOF,
) -> None:
    """Trading comparables only."""
    try:
        assumptions, client, market = _setup(config, no_cache, as_of)
        if not assumptions.comps.peers:
            console.print("[red]No peers in the assumptions file (comps.peers).[/red]")
            raise typer.Exit(1)
        _render_comps(run_comps(ticker, assumptions, client, market))
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


@app.command()
def merger(
    acquirer: str,
    target: str,
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    as_of: str = _ASOF,
) -> None:
    """Accretion/(dilution), contribution and breakeven synergies."""
    try:
        assumptions, client, market = _setup(config, no_cache, as_of)
        acq_fin, acq_price, acq_bridge = _load(acquirer, client, market, assumptions)
        tgt_fin, tgt_price, tgt_bridge = _load(target, client, market, assumptions)
        r = run_merger(
            acq_fin, tgt_fin, acq_bridge, tgt_bridge, acq_price, tgt_price, assumptions
        )

        _rule(f"{acq_fin.ticker} acquires {tgt_fin.ticker}")
        console.print(_kv_table(r.consideration.rows(), title="Consideration", dp=2))

        _rule("Sources and uses")
        t = Table(box=None, pad_edge=False)
        t.add_column("")
        t.add_column("")
        t.add_column("USD mm", justify="right")
        for side, label, amount in r.sources_uses:
            style = "bold" if side == "Total" else ""
            t.add_row(Text(side, style=style), Text(label, style=style),
                      Text(_money(amount), style=style))
        console.print(t)

        _rule("Accretion / (dilution), year one on TTM figures")
        console.print(_kv_table(r.accretion.rows(), title="", dp=3))
        if r.accretion.breakeven_synergies is not None:
            console.print(
                f"\n  Pre-tax synergies to break even: "
                f"[bold]{_money(r.accretion.breakeven_synergies)}mm[/bold]"
            )

        _rule("Contribution versus ownership")
        t = Table(box=None, pad_edge=False)
        t.add_column("Metric")
        t.add_column(acq_fin.ticker, justify="right")
        t.add_column("%", justify="right")
        t.add_column(tgt_fin.ticker, justify="right")
        t.add_column("%", justify="right")
        for row in r.contribution:
            t.add_row(
                row.metric,
                _money(row.acquirer),
                _pct(row.acquirer_pct) if row.acquirer_pct is not None else "NM",
                _money(row.target),
                _pct(row.target_pct) if row.target_pct is not None else "NM",
            )
        t.add_row(
            Text("Pro forma ownership", style="bold"),
            "",
            Text(_pct(r.pro_forma_ownership["acquirer"]), style="bold"),
            "",
            Text(_pct(r.pro_forma_ownership["target"]), style="bold"),
        )
        console.print(t)

        _rule("Sensitivity")
        # The grid reports percent accretion or cents per share depending on
        # whether standalone EPS is a base worth dividing by. The module says
        # which in .attrs, and the heading has to carry it or the numbers are
        # unreadable.
        console.print(
            _frame_table(
                r.sensitivity,
                title=r.sensitivity.attrs.get(
                    "label", "Accretion / (dilution) by premium and consideration mix"
                ),
                index_label="Premium",
            )
        )
        _notes(r.checks, heading="Cross-checks")
        _notes(r.accretion.notes + r.notes)

        if getattr(r, "purchase_accounting", None) is not None:
            _render_purchase_accounting(r.purchase_accounting, acq_fin.ticker)
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)



def _quarter_ends(start: date, end: date) -> list[date]:
    """Calendar quarter ends inside a window, inclusive."""
    out: list[date] = []
    y, q = start.year, (start.month - 1) // 3
    while True:
        m = q * 3 + 3
        last = date(y, m, 1)
        last = date(y + (m == 12), 1 if m == 12 else m + 1, 1) - timedelta(days=1)
        if last > end:
            break
        if last >= start:
            out.append(last)
        q += 1
        if q == 4:
            q, y = 0, y + 1
    return out


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. DDOG,MDB,ZS"),
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    start: str = typer.Option(..., "--from", help="First valuation date (YYYY-MM-DD)."),
    end: str = typer.Option(..., "--to", help="Last valuation date (YYYY-MM-DD)."),
    horizon: int = typer.Option(252, "--horizon", help="Forward return horizon in calendar days."),
    dates: str = typer.Option(None, "--dates", help="Explicit comma-separated dates, instead of quarter ends."),
) -> None:
    """Value a set of names at past dates and score against realised returns.

    Every valuation is built with a knowledge date, so it sees only filings that
    existed then and prices that stopped there. Forward prices are read through a
    separate object that the valuation path never receives, which is what makes
    the separation structural rather than a matter of discipline.
    """
    try:
        assumptions, _client, _market = _setup(config, no_cache)
        names = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        first, last = date.fromisoformat(start), date.fromisoformat(end)
        when = (
            [date.fromisoformat(d.strip()) for d in dates.split(",") if d.strip()]
            if dates
            else _quarter_ends(first, last)
        )
        if not when:
            console.print("[red]No valuation dates in that window.[/red]")
            raise typer.Exit(1)

        cache = HttpCache(enabled=not no_cache)
        source = make_price_source(
            assumptions.price_source, cache, assumptions.price_csv_dir
        )
        result = run_backtest(
            names,
            when,
            assumptions,
            edgar_client_factory(cache),
            market_factory_from(source, cache),
            ForwardPrices(
                source,
                start=first - timedelta(days=400),
                end=last + timedelta(days=horizon + 30),
            ),
            horizon_days=horizon,
        )

        _rule(f"Backtest: {len(names)} names over {len(when)} dates")
        df = result.to_frame()
        if not df.empty:
            console.print(_frame_table(df.head(40), title="Observations", index_label=""))

        console.print()
        stats = Table(box=None, pad_edge=False)
        stats.add_column("")
        stats.add_column("", justify="right")
        for label, v in (
            ("Observations valued", result.n_valued),
            ("Failed", result.n_failed),
            ("Skipped", result.n_skipped),
            ("Scored against a forward return", result.n_paired),
            ("Non-overlapping of those", result.n_independent),
        ):
            stats.add_row(label, f"{v:,}")
        ic = result.spearman_ic
        stats.add_row(
            Text("Spearman rank IC", style="bold"),
            Text("n/a" if ic is None else f"{ic:+.3f}", style="bold"),
        )
        hr = result.hit_rate
        stats.add_row("Hit rate", "n/a" if hr is None else f"{hr:.1%}")
        console.print(stats)

        if result.quantile_returns is not None and not result.quantile_returns.empty:
            console.print()
            console.print(
                _frame_table(
                    result.quantile_returns,
                    title="Forward return by predicted-upside quintile",
                    index_label="Quintile",
                )
            )
        _notes(list(result.checks), heading="Cross-checks")
        _notes(list(result.notes))
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


@app.command()
def fetch(
    ticker: str,
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    as_of: str = _ASOF,
) -> None:
    """Warm the cache and print where every figure came from."""
    try:
        assumptions, client, market = _setup(config, no_cache, as_of)
        fin = build_financials(ticker, client=client)
        _render_financials(fin)

        _rule("Provenance")
        t = Table(box=None, pad_edge=False)
        for col in ("concept", "tag", "method", "period", "form", "filed"):
            t.add_column(col, overflow="fold")
        for row in fin.provenance_rows():
            t.add_row(*(row[c] for c in ("concept", "tag", "method", "period", "form", "filed")))
        console.print(t)

        notes = [r["note"] for r in fin.provenance_rows() if r["note"]]
        _notes(sorted(set(notes)), heading="Provenance notes")
        console.print(
            f"\n[dim]Price {market.spot(ticker):,.2f} from "
            f"{market.prices(ticker).source}, close of "
            f"{market.prices(ticker).last_date}.[/dim]"
        )
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
