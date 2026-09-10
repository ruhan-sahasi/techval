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
from datetime import date
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .comps import run_comps
from .config import Assumptions
from .dcf import run_dcf, sensitivity_wacc_exit, sensitivity_wacc_growth
from .edgar import EdgarClient, HttpCache
from .errors import TechvalError
from .ev_bridge import build_ev_bridge
from .financials import build_financials
from .football import FootballRow, football_field
from .market import MarketData, make_price_source
from .merger import run_merger
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


def _setup(config: Path | None, no_cache: bool):
    assumptions = Assumptions.load(config)
    cache = HttpCache(enabled=not no_cache)
    client = EdgarClient(cache)
    source = make_price_source(
        assumptions.price_source, cache, assumptions.price_csv_dir
    )
    market = MarketData(source, cache, today=date.today())
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
        ("Equity value, Gordon growth", d.equity_value_gordon),
        ("Equity value, exit multiple", d.equity_value_exit),
    ]
    console.print()
    console.print(_kv_table(summary, title="", dp=0))
    for label, v in (
        ("Gordon growth", d.per_share_gordon),
        ("exit multiple", d.per_share_exit),
    ):
        shown = f"${v:,.2f}" if v is not None else "not computed"
        console.print(f"\n  [bold]Implied per share, {label:<14s} {shown}[/bold]")
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
) -> None:
    """Full valuation: statements, EV bridge, WACC, DCF, comps and football field."""
    try:
        assumptions, client, market = _setup(config, no_cache)
        fin, price, bridge = _load(ticker, client, market, assumptions)
        _render_financials(fin)
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

        d = run_dcf(
            fin,
            bridge,
            w,
            assumptions,
            peer_median_ev_ebitda=peer_median_ev_ebitda,
        )
        _render_dcf(d, fin)

        _rule("Sensitivities")
        console.print(
            _frame_table(
                sensitivity_wacc_growth(fin, bridge, assumptions, w.wacc),
                title="Implied price per share: WACC by terminal growth",
                index_label="WACC",
            )
        )
        if peer_median_ev_ebitda:
            console.print()
            console.print(
                _frame_table(
                    sensitivity_wacc_exit(
                        fin, bridge, assumptions, w.wacc, peer_median_ev_ebitda
                    ),
                    title="Implied price per share: WACC by exit multiple",
                    index_label="WACC",
                )
            )

        if comps_result is not None:
            _render_comps(comps_result)

        path = _chart(ticker, fin, market, d, comps_result, price, out)
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


def _chart(ticker, fin, market, dcf_result, comps_result, price, out_dir):
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

    for label, base in (
        ("DCF: Gordon growth", dcf_result.per_share_gordon),
        ("DCF: exit multiple", dcf_result.per_share_exit),
    ):
        if base and base > 0:
            # The sensitivity grid is the honest width of a DCF: a single point
            # estimate implies a precision the inputs do not support.
            rows.append(FootballRow(label, base * 0.85, base * 1.15))

    path = Path(out_dir) / f"{ticker.upper()}_football.png"
    return football_field(
        rows,
        price,
        ticker.upper(),
        path,
        subtitle=(
            f"TTM to {fin.as_of}. DCF bands are the point estimate plus or minus 15%."
        ),
    )


@app.command()
def comps(ticker: str, config: Path = _CFG, no_cache: bool = _NOCACHE) -> None:
    """Trading comparables only."""
    try:
        assumptions, client, market = _setup(config, no_cache)
        if not assumptions.comps.peers:
            console.print("[red]No peers in the assumptions file (comps.peers).[/red]")
            raise typer.Exit(1)
        _render_comps(run_comps(ticker, assumptions, client, market))
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


@app.command()
def merger(
    acquirer: str, target: str, config: Path = _CFG, no_cache: bool = _NOCACHE
) -> None:
    """Accretion/(dilution), contribution and breakeven synergies."""
    try:
        assumptions, client, market = _setup(config, no_cache)
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
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


@app.command()
def fetch(ticker: str, config: Path = _CFG, no_cache: bool = _NOCACHE) -> None:
    """Warm the cache and print where every figure came from."""
    try:
        assumptions, client, market = _setup(config, no_cache)
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
