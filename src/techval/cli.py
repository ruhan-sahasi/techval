"""Command line interface.

All rendering lives here. The numeric modules return dataclasses and DataFrames
and know nothing about terminals, so the same objects drive the CLI, the notebook
and anything else built on the package.

The core valuation:

    techval value  TICKER      full valuation: statements, bridge, WACC, DCF, comps, chart
    techval comps  TICKER      the comp table alone
    techval merger ACQ TGT     accretion/dilution
    techval backtest           value a set of names at past dates and score them
    techval fetch  TICKER      warm the cache and print the provenance table

Nine more are mounted from the four ``commands_*`` modules, which own their own
rendering: ``kpis``, ``segments`` and ``sotp`` over ``techval.tmt``; ``peers``
and ``screen`` over the fitted encoder and the warranted multiple; ``fade`` and
``signal`` over the forecast and the harness; ``precedents`` and ``targets``
over M&A. They are merged in flat rather than nested behind a group, so the
invocation is the one each module documents for itself.

One more renders results rather than a valuation. ``techval dashboard`` runs
each dashboard section's collector against the committed fixtures, writes what
they return to ``docs/dashboard/snapshot.json``, and renders
``docs/dashboard/index.html`` from that file alone. It sits under its own help
heading.

Seven assumption flags additionally fold those capabilities into the ``value``
report as optional sections. Every one defaults to off and the base report is
byte-identical without them, which is asserted in tests/test_cli_integration.py
rather than merely intended.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from .comps import run_comps
from .apv import run_apv
from .commands_forecast import app as _forecast_app
from .commands_mna import app as _mna_app
from .commands_peers import app as _peers_app
from .commands_rag import app as _rag_app
from .commands_tmt import app as _tmt_app

# The optional sections below reuse the command modules' own loaders and
# renderers rather than growing second copies here. A warranted-multiple table
# printed by the value report and one printed by `techval screen` should not be
# able to drift apart, and the way to guarantee that is to have one of them.
from .commands_forecast import (
    _horizon_table as _render_fade_baselines,
    _prices_from_csv,
    _recorded_panel as _load_fade_panel,
    _render_curve as _render_fade_curve,
    _render_paths as _render_fade_paths,
    _render_valuation as _render_fade_valuation,
    _truncate_panel as _truncate_fade_panel,
)
from .commands_mna import load_dataset as _load_mna_dataset
from .commands_peers import (
    _bundle as _peer_bundle,
    _cache_root as _ml_cache_root,
    _load_observations,
)
from .commands_tmt import _render_sotp, load_plan as _load_sotp_plan
from .dashboard import (
    collect_snapshot,
    dump_snapshot,
    load_snapshot,
    write_dashboard,
)
from .dashboard.collect import SectionRun
from .dashboard.sections import SECTION_IDS, section_module
from .errors import MissingDataError
from .ml.forecast import compare_fade, fit_fade
from .ml.mna import fit_propensity
from .ml.signals import Score, test_signal
from .ml.warranted import fit_warranted
from .tmt.precedents import build_precedents
from .tmt.segments import build_segments
from .tmt.sotp import run_sotp
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
from .errors import ConfigError, TechvalError
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


def _mount(sub_app: typer.Typer, panel: str) -> None:
    """Mount a command module's commands at the top level, under a help heading.

    ``add_typer`` with no name merges the sub-app's commands into the parent
    rather than nesting them behind a group, which is the shape all four command
    modules document in their own docstrings: ``techval sotp DIS``, not
    ``techval tmt sotp DIS``. Nesting would have read better in ``--help`` and
    would have made those docstrings wrong, so the help panel carries the
    grouping instead and the invocation stays the one the modules advertise.

    The panel is set on each command here rather than in the modules themselves
    because a module does not know what it will be mounted beside, and because
    ``rich_help_panel`` passed to ``add_typer`` is silently ignored on a merge.
    """
    for command in sub_app.registered_commands:
        command.rich_help_panel = panel
    app.add_typer(sub_app)


_mount(_tmt_app, "TMT fundamentals")
_mount(_peers_app, "Learned comparables")
_mount(_forecast_app, "Forecasts and signal testing")
_mount(_mna_app, "M&A")
# The filing reader is a group, not a merge: "template", "record" and "score"
# are too generic to stand at the top level beside "value" and "comps".
app.add_typer(_rag_app, name="rag", rich_help_panel="Filing reader")

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
    # ``accretion_by_year`` holds (year, dollars per share, percent or None):
    # the contract is on PurchaseAccounting in merger.py. The percent is None
    # wherever standalone EPS is too thin to divide by, and the dollars figure
    # is then the only one that carries meaning, so it gets its own row and the
    # percent row prints NM in exactly the years the model refused to quote one.
    t.add_row(
        Text("Accretion / (dilution), $", style="bold"),
        *[Text(_money(dollars, 3)) for _, dollars, _ in pa.accretion_by_year],
    )
    if any(pct is not None for _, _, pct in pa.accretion_by_year):
        t.add_row(
            Text("Accretion / (dilution), %", style="bold"),
            *[
                Text(_pct(pct) if pct is not None else "NM")
                for _, _, pct in pa.accretion_by_year
            ],
        )
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
# optional sections of the value report
#
# Seven assumption flags switch these on and every one defaults to False, so the
# base report is what it has always been unless a user asks for more. That is
# not politeness about defaults. The premise of this package is that a DCF and a
# comp table trace to filings, and a fitted model's output does not trace the
# same way: it traces to a recorded panel, a training window and a seed. Mixing
# the two by default would put a number nobody can audit next to numbers anybody
# can, which is the one thing the engine is built not to do. A reader who turns
# a flag on has asked for the fitted number and is told, in the section itself,
# what it rests on.
#
# Each section refuses rather than invents. Where the artifact a model needs is
# absent the section says which file it wanted and which command writes it, and
# the report carries on: an optional section is not allowed to take the
# valuation down with it.
# --------------------------------------------------------------------------- #


def _ml_root(assumptions, ml_data: Path | None) -> Path:
    """Where the recorded model artifacts live.

    ``--ml-data`` overrides ``ml.cache_dir``, which is the same root and the same
    filenames ``techval peers`` and ``techval screen`` already default to. The
    path is a function argument rather than a new assumptions entry deliberately:
    a directory of recorded panels is a property of the machine a run happens on,
    not a property of the company being valued, and the assumptions file is the
    company's.
    """
    return Path(ml_data) if ml_data else _ml_cache_root(assumptions)


# Where each recorded artifact is known to live, under a single ``--ml-data``
# root, in the order the root is searched.
#
# One name per artifact would have been tidier and it does not survive contact
# with the repository. The recorders do not agree on a layout: the peer
# artifacts are committed with a ``_tmt`` suffix naming the universe they were
# recorded over, the warranted panel sits in ``warranted/`` beside its recorder
# script, the close prices sit in ``signals/``, and the M&A dataset is a
# directory of its own. The ``~/.techval/ml`` cache, meanwhile, writes the
# unsuffixed names flat. A single root therefore has to reach two layouts.
#
# The alternative was to hardcode one layout and let the other fail, which is
# what shipped, and it is worse than it sounds. Every refusal below tells the
# reader that "tests/fixtures carries a committed set". Pointing --ml-data at
# tests/fixtures then refused three of the four sections, and a hint that names
# a path which does not work teaches the reader that the feature is broken
# rather than that they typed the wrong directory. So the candidates are listed
# here, first hit wins, and a refusal names every place it looked.
#
# ``is_file`` rather than ``exists`` because ``~/.techval/ml`` really does hold
# a ``peer_groups/`` DIRECTORY beside the artifacts, and a directory that
# happens to share a name is not a panel.
_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "peer_groups": ("peer_groups.json", "peer_groups_tmt.json"),
    "peer_panel": ("peer_panel.json", "peer_panel_tmt.json"),
    "peer_item1": ("peer_item1.json", "peer_item1_tmt.json"),
    "warranted_panel": ("observations.json.gz", "warranted/observations.json.gz"),
    "signal_closes": ("closes.csv.gz", "signals/closes.csv.gz"),
    "fade_panel": ("fade_companyfacts.json.gz", "fade/fade_companyfacts.json.gz"),
    "mna_universe": ("mna/universe.json",),
    "mna_events": ("mna/events.json",),
    "mna_panel": ("mna/panel.csv.gz",),
}


def _resolve(root: Path, keys: list[str], *, what: str, writer: str) -> list[Path]:
    """The artifacts under ``root``, or a refusal naming every path tried.

    Refuse rather than invent, and say which file was wanted: the filename is
    the fix, and a reader who is given it can supply the artifact or point the
    run somewhere else. A reader who is given "unavailable" can do neither.
    """
    found: list[Path] = []
    missing: list[list[Path]] = []
    for key in keys:
        candidates = [root / name for name in _ARTIFACTS[key]]
        hit = next((p for p in candidates if p.is_file()), None)
        if hit is None:
            missing.append(candidates)
        else:
            found.append(hit)
    if missing:
        raise MissingDataError(
            what,
            hint=(
                "missing "
                + "; ".join(
                    " or ".join(str(p) for p in group) for group in missing
                )
                + f". {writer} Point the run at another directory with --ml-data."
            ),
        )
    return found


def _render_optional_sotp(ticker, fin, bridge, assumptions, client, plan_path) -> None:
    """The sum of the parts, beside the DCF rather than instead of it.

    A sum of the parts needs a multiple and a stated source for every segment and
    there is no default anywhere in this package, because a peer multiple for a
    cable network has no business living in a config shared with a software DCF.
    So this section refuses without a plan and names the command that writes one.
    """
    report = build_segments(ticker, client, assumptions)
    if plan_path is None:
        console.print(
            f"[yellow]tmt.sotp is on but no plan was supplied, so the parts are "
            f"not valued. {ticker.upper()} reports "
            f"{len(report.segments)} segment(s): "
            + ", ".join(s.name for s in report.segments)
            + ".[/yellow]"
        )
        console.print(
            "[yellow]Run [bold]techval sotp "
            f"{ticker.upper()} --emit-plan plan.yaml[/bold], fill in a multiple and "
            "a source for every segment, then rerun with --sotp-plan plan.yaml. "
            "There is no default multiple and inventing one would be the exact "
            "failure this engine refuses.[/yellow]"
        )
        return

    loaded = _load_sotp_plan(Path(plan_path))
    if report.as_of != fin.as_of:
        # The fix is deliberately NOT stated as --as-of {report.as_of}. Pinning
        # the knowledge date to the last day of the fiscal year rolls the run
        # back to before that year's annual report was filed, so the engine
        # falls back to the PRIOR year's segment footnote and the two windows
        # are still a period apart. Measured on DIS: --as-of 2025-09-27 returned
        # segments for the year ended 2024-09-28. The date that actually closes
        # the gap is shortly after the annual report reached EDGAR.
        console.print(
            f"[yellow]FLAG: the segments cover the year ended {report.as_of} and "
            f"the consolidated figures cover the twelve months to {fin.as_of}. "
            "Segment detail is annual, so the parts are measured a period behind "
            "the whole. To bring them together, pin the run with --as-of set to a "
            f"date shortly after the {report.as_of} annual report was filed, which "
            "is a month or two after the year end rather than the year end "
            "itself.[/yellow]"
        )
    result = run_sotp(
        fin,
        bridge,
        report.segments,
        loaded["multiples"],
        assumptions,
        corporate_cost=loaded["corporate_cost"],
        corporate_multiple=loaded["corporate_multiple"],
        corporate_multiple_source=loaded["corporate_multiple_source"],
        conglomerate_discount=float(loaded["conglomerate_discount"]),
    )
    _render_sotp(result, fin, report)


def _render_optional_peers(ticker, assumptions, root) -> list[str]:
    """The encoder's comp set beside the hand-written one, and the disagreement.

    Both are printed and neither replaces the other. The hand-written list is
    somebody's judgment and the ranked list is a cosine from a model trained on
    peer groups companies disclosed in their own proxies, so the interesting
    output is not either list: it is the names one has and the other does not.

    Returns the model's proposed tickers so a later section can reuse them.
    """
    groups, panel, text = _resolve(
        root,
        ["peer_groups", "peer_panel", "peer_item1"],
        what="the fitted peer encoder's training inputs",
        writer=(
            "These are recorded artifacts: an encoder is fitted on several "
            "thousand proxy filings and cannot be built from one ticker. "
            "tests/fixtures carries a committed set."
        ),
    )
    target = ticker.upper()
    bundle = _peer_bundle(
        assumptions,
        groups_path=groups,
        panel_path=panel,
        text_path=text,
        through=None,
        k=10,
        # The walk-forward evaluation costs about a minute and belongs to
        # `techval peers`, which prints it in full. A section inside the value
        # report that silently spent a minute on it would be a surprise, and one
        # that printed the warm number without the cold-start split beside it
        # would be quoting the flattering half. So it is skipped here and the
        # section says where to get it.
        evaluate=False,
        ablate=False,
        refit=False,
        use_cache=True,
    )
    encoder = bundle.encoder
    shown = int(assumptions.ml.peers.n_peers)
    ranked = encoder.neighbours(target, shown, apply_size_gate=True)
    names: dict[str, str] = {}
    for corpus in bundle.dataset.corpora.values():
        names.update(corpus.entity_names)

    console.print(
        f"[dim]{len(encoder.tickers)} companies in the fitted universe, embedded "
        f"at {encoder.fit_date}, text weight {encoder.text_weight:.2f}. This "
        "ranking is a similarity, not a valuation, and it is shown beside the "
        "hand-written set rather than in place of it.[/dim]\n"
    )
    hand = [p.upper() for p in assumptions.comps.peers]
    hand_set = set(hand)
    t = Table(box=None, pad_edge=False)
    t.add_column("#", justify="right")
    t.add_column("Ticker", no_wrap=True)
    t.add_column("Company", overflow="ellipsis", max_width=34)
    t.add_column("Similarity", justify="right")
    t.add_column("In the hand-written set", justify="center")
    for i, (peer, similarity) in enumerate(ranked, 1):
        t.add_row(
            str(i),
            Text(peer, style="bold"),
            names.get(peer, ""),
            f"{similarity:.4f}",
            "yes" if peer in hand_set else "-",
        )
    console.print(t)

    proposed = [p for p, _s in ranked]
    if hand:
        added = [p for p in proposed if p not in hand_set]
        dropped = [p for p in hand if p != target and p not in set(proposed)]
        console.print("\n[bold]Disagreement[/bold]")
        console.print(f"  The model adds  {', '.join(added) if added else 'nothing'}")
        console.print(f"  The model drops {', '.join(dropped) if dropped else 'nothing'}")
    console.print(
        "\n[dim]The comp table above was built from the hand-written set. This "
        "section proposes, it does not substitute: run [bold]techval peers "
        f"{target}[/bold] for the cold-start split and the baselines that say "
        "what the ranking is worth.[/dim]"
    )
    return proposed


def _fit_warranted_panel(assumptions, root):
    """Fit the warranted multiple on the recorded observation panel.

    Returns the fitted model, the panel it was fitted on and the path that panel
    came from, because the section that renders it has to say all three. A
    residual printed without the panel behind it is a number a reader cannot
    check, and checking it is the only thing that separates this from an opinion.
    """
    (panel_path,) = _resolve(
        root,
        ["warranted_panel"],
        what="the warranted-multiple observation panel",
        writer=(
            "The panel is every company in the universe priced at every quarter "
            "end, which is why it is recorded rather than fetched: fit_warranted "
            "refuses below 150 observations precisely so nobody runs it on a "
            "sample small enough to fetch live. tests/fixtures/warranted/record.py "
            "is the recorder."
        ),
    )
    observations = _load_observations(panel_path)
    return fit_warranted(observations, assumptions), observations, panel_path


def _render_optional_warranted(ticker, model, observations, panel_path) -> None:
    """The fitted warranted multiple and this company's residual.

    The residual is the whole point and it is the easiest number here to
    over-read, so the sentence the model writes about itself is printed rather
    than a bare figure: it carries the within-date standard deviation, whether
    the read was out of sample, and the reminder that a residual is a statement
    about relative pricing and never about value.

    The card underneath it is not decoration either. ``techval screen`` renders
    this same model with its training window, its walk-forward score and its
    limitations attached; this section rendered five rows and a sentence, so one
    model had two disclosure standards and the weaker one was the one sitting
    inside a valuation. ``ml/__init__`` sets the rule the weaker one broke: an
    unauditable model has no place beside a valuation whose every other number
    traces to a filing.
    """
    read = model.warranted(ticker.upper())
    rows = [
        ("Actual multiple", f"{read.actual_multiple:,.1f}x"),
        ("Warranted multiple", f"{read.warranted_multiple:,.1f}x"),
        ("Residual, turns", f"{read.residual_turns:+,.1f}x"),
        ("Residual, within-date SD", f"{read.z:+,.2f}"),
        ("Out of sample", "yes" if read.out_of_sample else "NO, in sample"),
    ]
    t = Table(box=None, pad_edge=False)
    t.add_column("", no_wrap=True)
    t.add_column("", justify="right")
    for label, shown in rows:
        t.add_row(label, shown)
    console.print(t)
    console.print(f"\n{read.sentence()}")

    _render_model_card(
        model.card,
        source=(
            f"{len(observations.observations):,} observations over "
            f"{len(observations.dates)} dates and {len(observations.tickers)} "
            f"companies, from {panel_path}. Demeaned by date: {model.demeaned}."
        ),
        # The pooled score in the card is mostly company identity: a feature
        # vector barely moves in three months and neither does a relative
        # multiple, so a model fitted on the past is rewarded for recognising a
        # name. The differenced figure asks whether it predicted the CHANGE, and
        # it is the size of what is actually being added. Printing the first
        # without the second is how a screen gets oversold.
        extra=[
            ("Differenced against the company's own prior read", model.change_rank_correlation),
            ("Differenced observations", model.n_changes),
        ],
    )


def _render_model_card(card, *, source: str, extra=()) -> None:
    """What a fitted model is, printed beside what it said.

    ``ml/__init__`` sets three rules for the package and the third is that every
    fitted model carries what it was trained on, when, with what features and how
    it scored out of sample. A model output printed without that is a number
    nobody can argue with, which is the same defect as a number nobody can
    trace, and it is the defect this report had: the warranted residual appeared
    with no training window, no sample size and no evaluation, while the same
    model rendered by ``techval screen`` carried all three.

    The rows come from ``ModelCard.rows`` rather than being assembled here, so a
    card printed inside the valuation and a card printed by a standalone command
    cannot describe one model two different ways.
    """
    t = Table(box=None, pad_edge=False)
    t.add_column("", no_wrap=True)
    t.add_column("", justify="right")
    for label, value in list(card.rows()) + list(extra):
        t.add_row(str(label), _card_value(value))
    console.print("\n[bold]The model behind that number[/bold]")
    console.print(t)
    console.print(Text(f"  Fitted on {source}", style="dim"))
    _notes(list(card.limitations), heading="What this model cannot do")


def _card_value(value) -> str:
    """Format a card cell without deciding what it means.

    ``bool`` before ``int`` because a bool IS an int in Python, and a card row
    reading "Beats baseline 1" would be both true and unreadable.
    """
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        # No forced sign. A rank correlation prints its own minus when it has
        # one, and a mean absolute error cannot be negative, so a leading plus
        # on it would only invite the reader to look for the comparison it is
        # not making.
        return f"{value:,.4f}"
    return str(value)


def _render_optional_signal(assumptions, model, root) -> None:
    """Has the score driving that residual ever predicted anything?

    This is the section the repository exists to be able to print. A warranted
    residual is only worth reading if the ordering it produces has earned
    anything in the years since, and the honest answer is allowed to be no. The
    harness is handed the residuals as scores with the sign fixed before the
    coefficient is seen: cheap means a negative residual, so the score is negated
    so that a positive coefficient means cheap names earned more.
    """
    (closes,) = _resolve(
        root,
        ["signal_closes"],
        what="the close price history the signal test scores against",
        writer=(
            "A forward return needs prices that run past the last score date by "
            "the full horizon. tests/fixtures/signals carries a recorded set."
        ),
    )
    scores = [
        Score(ticker=t, as_of=when, value=-read.residual_log)
        for (t, when), read in model.reads.items()
    ]
    prices = _prices_from_csv(closes, {s.ticker for s in scores})
    result = test_signal(
        scores,
        prices,
        assumptions=assumptions,
        label="warranted residual (negated: positive means cheap)",
    )
    console.print(Text(result.verdict(), style="yellow bold"))

    # The verdict alone was what this section printed, and it is the flattering
    # half. ``test_signal`` also writes a census of how every holding period
    # ended and a FLAG when none of them ended in an acquisition or a delisting,
    # which is the survivorship hole the harness was built to expose: a universe
    # of names that still trade has had its failures removed before the first
    # coefficient is computed, and every figure in the verdict is biased upward
    # by the outcomes it cannot see. Measured on the committed panel that FLAG
    # fires, so what was being suppressed here is not hypothetical.
    #
    # Deliberately NOT fixed by handing this call a deals list of its own. The
    # harness takes ``deals`` and ``delisting_return`` and neither is passed here
    # or by `techval signal`, so passing one only here would make the value
    # report and the standalone command disagree about the same model, which is
    # the drift this file exists to prevent. The census is printed instead, with
    # the convention in force named, so the reader can see the size of what is
    # missing rather than being told nothing is.
    flags = [line for line in result.checks if line.startswith("FLAG:")]
    census = [line for line in result.checks if "holding periods ran their course" in line]
    _notes(census + flags, heading="What the sample was")
    console.print(
        "\n[dim]No deal terms and no delisting return were supplied, so a name "
        "acquired or delisted inside a holding period is excluded rather than "
        "terminated at what the holder received. Run [bold]techval signal --score "
        "warranted --delisting shumway_nasdaq[/bold] to see how far the answer "
        "moves when the failures are priced instead of dropped.[/dim]"
    )
    console.print(
        "[dim]Sign fixed before the coefficient was seen. Run [bold]techval "
        "signal --score warranted[/bold] for the per-date coefficient series, the "
        "bucket table and the overlap correction behind this verdict.[/dim]"
    )


def _render_optional_propensity(ticker, assumptions, root) -> None:
    """This company's own acquisition probability, beside the base rate.

    The base rate is not decoration. A three percent unconditional chance of
    being bought in a year means a model output of four percent is barely a
    statement, and a probability printed on its own invites a reader to treat it
    as a forecast. Both numbers or neither.
    """
    _resolve(
        root,
        ["mna_universe", "mna_events", "mna_panel"],
        what="the acquisition propensity dataset",
        writer=(
            "The universe has to keep companies after they delist, because a "
            "company that is acquired stops filing and vanishes from every "
            "current ticker list. tests/fixtures/mna carries one with its "
            "manifest."
        ),
    )
    dataset = root / "mna"
    panel, universe, events, extras = _load_mna_dataset(dataset)
    when = max(panel.dates)
    result = fit_propensity(
        panel, universe, events, assumptions, as_of=when, extras=extras
    )
    frame = result.model.rank(
        panel, universe, when, top_k=len(universe.as_of(when)),
        extras=extras, events=events,
    )
    row = frame[frame["ticker"].str.upper() == ticker.upper()]
    base = result.labels.base_rate
    if row.empty:
        console.print(
            f"[yellow]{ticker.upper()} is not in the recorded propensity panel at "
            f"{when}, so it has no probability. The universe's base rate at this "
            f"horizon is {base:.1%}.[/yellow]"
        )
        return
    probability = float(row.iloc[0]["probability"])
    ev = result.evaluation
    t = Table(box=None, pad_edge=False)
    t.add_column("", no_wrap=True)
    t.add_column("", justify="right")
    t.add_row(f"{ticker.upper()} probability, {when}", _pct(probability))
    t.add_row("Universe base rate", _pct(base))
    t.add_row("Ratio to base rate", f"{probability / base:,.2f}x" if base else "NM")
    t.add_row(f"Rank of {len(frame):,}", str(int(row.iloc[0]["rank"])))
    console.print(t)
    console.print(
        Text(
            f"\n  Walk-forward AUC {ev.score:.4f} against {ev.baseline_score:.4f} "
            f"for the size sort, a lift of {ev.lift:+.4f}"
            + (
                f", inside a fold standard deviation of {ev.fold_sd:.4f}."
                if ev.fold_sd is not None
                else ", with no fold dispersion available."
            ),
            style="yellow bold",
        )
    )
    console.print(
        Text(
            "  This is the weakest model in the package. Fundamentals only: no "
            "valuation channel is in this ranking.",
            style="yellow",
        )
    )


def _render_optional_fade(ticker, fin, bridge, wacc_result, assumptions, root) -> None:
    """The DCF's typed-in growth path against one fitted on filings, both valued.

    This is the most consequential of the optional sections and it was the one
    with no wiring at all. ``ml.forecast.enabled`` existed, was documented, and
    reached nothing: a reader who set it got a byte-identical report, so the
    engine neither refused nor acted, which is the single behaviour the cardinal
    rule forbids. A switch that announces nothing is worse than a switch that
    says no.

    What it is worth wiring for is the shape of a DCF. Every other assumption in
    ``assumptions.dcf`` moves the answer by a few percent; the growth path moves
    it by multiples, and it is typed in. ``dcf.revenue_growth_start`` is a
    judgment about one company with no panel behind it, and the fade model
    replaces it with a number fitted on what more than two hundred TMT filers
    actually did next, out of sample, against three baselines.

    Four valuations are printed rather than two, and that is the point of the
    section rather than a flourish. A fitted point estimate set beside the
    assumption it replaces invites the reader to treat the fitted one as the
    answer; it is the middle of a band tens of growth points wide whose two ends
    are different companies, so the band is valued as well. The baseline table
    above it prints the model against persistence, the training mean and the
    sub-vertical mean at every horizon, because the model card's claim is that
    the curve ties persistence at one year and beats it at two and three, and a
    lift quoted without the baseline it was measured against is not a result.

    The renderers are the ones ``techval fade`` uses, imported rather than
    copied, so the fitted path inside a valuation and the fitted path from the
    standalone command cannot drift apart.
    """
    (panel_path,) = _resolve(
        root,
        ["fade_panel"],
        what="the revenue fade panel",
        writer=(
            "The curve is fitted across the whole TMT universe, which is one "
            "companyfacts call per filer at the SEC fair-access throttle and not "
            "something a valuation should do on its way past. "
            "tests/fixtures/fade_companyfacts.json.gz is a recorded blob of those "
            "payloads, and `techval fade TICKER --panel <file>` reads the same "
            "shape."
        ),
    )
    symbol = ticker.upper()
    console.print(
        f"[dim]Fitting the fade curve on the recorded panel at {panel_path}. The "
        "whole universe is fitted, not this company alone, because a curve "
        "estimated on one filer's history is that filer's history.[/dim]"
    )
    panel = _load_fade_panel(panel_path, assumptions, None)
    if assumptions.as_of:
        # Without this the option would be a lie. The panel builder pins every
        # FEATURE to the filing date of the report that carried it and stops
        # there, so a curve fitted on filings through 2026 and handed to a
        # valuation struck in 2020 knows how the intervening six years went. The
        # valuation would look excellent and the failure would be silent.
        panel = _truncate_fade_panel(panel, date.fromisoformat(assumptions.as_of))
    model = fit_fade(panel, assumptions)

    _render_fade_curve(model, panel)
    # Includes the point-in-time cut where one was made, which says how many
    # observations and how many forward labels were removed. A pinned run whose
    # panel was silently left at full depth would be the worst kind of wrong.
    _notes(list(panel.notes), heading="Panel")
    _render_model_card(
        model.card,
        source=(
            f"{len(panel.observations):,} company-years over "
            f"{len(panel.tickers)} filers, from {panel_path}"
        ),
    )
    _render_fade_baselines(model)

    comparison = compare_fade(
        fin, bridge, wacc_result, assumptions, model, ticker=symbol
    )
    _render_fade_paths(comparison, model, panel, symbol)
    _render_fade_valuation(comparison)
    console.print(
        "\n[dim]The DCF above this section is the 'Assumed fade' row. Nothing in "
        "it has been replaced: the fitted path is shown beside the typed one and "
        "the reader chooses. Run [bold]techval fade "
        f"{symbol}[/bold] for the reversion table, which assumes no functional "
        "form at all, and for the survivorship measurement, which is the part of "
        "this model most likely to be wrong in one direction.[/dim]"
    )


def _render_optional_precedents(assumptions, client, market, candidates) -> None:
    """Precedent transactions among the company's own sub-vertical.

    ``build_precedents`` reads each target's own merger filing, so the tickers it
    is handed are the question being asked. The comp set is the available stand
    in for the sub-vertical: those are the names a banker would already have
    agreed are the neighbours. Most of them were never acquired and contribute
    nothing, which is not an error but the reason a propensity model has a
    negative class.
    """
    if not candidates:
        console.print(
            "[yellow]No comp set and no proposed peers, so there is no "
            "sub-vertical to draw precedents from. Set comps.peers.[/yellow]"
        )
        return
    result = build_precedents(candidates, client, assumptions, prices=market)
    if not result.transactions:
        console.print(
            f"[yellow]None of the {len(candidates)} names searched has a merger "
            f"agreement on file inside {assumptions.ml.mna.lookback_years} years. "
            "Most companies are not acquired, so an empty precedent set is a "
            "finding rather than a failure.[/yellow]"
        )
        # Measured on live data rather than assumed. A reader who stops at the
        # line above would conclude that nothing in this sub-vertical has ever
        # been bought, and that conclusion would be wrong for a structural
        # reason worth stating.
        console.print(
            "[yellow]Read that with the selection effect in mind. This section "
            "searches the comp set, and a comp set is made of companies that "
            "still trade. A company that was acquired was delisted, drops out of "
            "the SEC ticker file, and can no longer be resolved from a ticker at "
            "all, so the names most likely to carry a precedent are the ones that "
            "cannot be in the list being searched. For a real precedent set, pass "
            "the targets directly to [bold]techval precedents[/bold] with "
            "price_source: csv and the closes supplied.[/yellow]"
        )
        return
    t = Table(box=None, pad_edge=False)
    t.add_column("Target", no_wrap=True)
    t.add_column("Acquirer", overflow="ellipsis", max_width=28)
    t.add_column("Announced", no_wrap=True)
    t.add_column("Equity value", justify="right")
    t.add_column("EV/Revenue", justify="right")
    t.add_column("Premium", justify="right")
    for deal in result.transactions:
        t.add_row(
            Text(str(deal.ticker), style="bold"),
            str(getattr(deal, "acquirer", "") or ""),
            str(getattr(deal, "announced", "") or ""),
            _money(getattr(deal, "equity_value", None)),
            _mult(getattr(deal, "ev_revenue", None)),
            _pct(getattr(deal, "premium", None)),
        )
    console.print(t)
    console.print(
        "\n[dim]Precedents are not trading comps: a control premium and the "
        "acquirer's expected synergies are inside every multiple above. Run "
        "[bold]techval precedents[/bold] for the accession behind each row and "
        "the deals that declined to price.[/dim]"
    )


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
    ml_data: Path = typer.Option(
        None,
        "--ml-data",
        help=(
            "Directory holding the recorded model artifacts the optional sections "
            "read. Defaults to ml.cache_dir. Only consulted when one of the ml.* "
            "flags is on. Both committed layouts are accepted, so --ml-data "
            "tests/fixtures reaches every section."
        ),
    ),
    sotp_plan: Path = typer.Option(
        None,
        "--sotp-plan",
        help=(
            "Segment plan naming the multiple, the metric and the source for every "
            "segment. Only consulted when tmt.sotp is on."
        ),
    ),
) -> None:
    """Full valuation: statements, EV bridge, WACC, DCF, comps and football field."""
    try:
        assumptions, client, market = _setup(config, no_cache, as_of)
        if assumptions.as_of:
            console.print(
                f"[yellow]Point in time: valuing {ticker.upper()} as it was knowable "
                f"on {assumptions.as_of}.[/yellow]"
            )
        fin = build_financials(ticker, client=client)
        price = market.spot(ticker)
        _render_financials(fin)

        # The share count is settled BEFORE the bridge is priced, because equity
        # value and every multiple built on it divide by whatever it decides.
        if assumptions.dilution.method == "treasury_stock":
            try:
                sc = build_share_count(ticker, price, fin, assumptions, client)
                _render_share_count(sc)
                fin.valuation_shares = sc.fully_diluted
            except TechvalError as exc:
                _rule("Share count")
                console.print(
                    f"[yellow]Treasury stock count unavailable, falling back to "
                    f"diluted WASO: {exc}[/yellow]"
                )

        bridge = build_ev_bridge(fin, price, assumptions)
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

        # Where the recorded model artifacts live, resolved once and shared by
        # every fitted section below. A pure function of the flag and the
        # assumptions, so computing it before any flag is read costs a default
        # run nothing.
        root = _ml_root(assumptions, ml_data)

        # Directly under the DCF, because this is the one optional section that
        # argues with the DCF rather than standing beside it: it refits the
        # growth path the projection was built on. It is not gated on ``d``
        # because it forms its own valuations, and a DCF that could not be made
        # will refuse again here with the same reason.
        if assumptions.ml.forecast.enabled:
            _rule("Growth: the assumed fade against a fitted one")
            try:
                _render_optional_fade(ticker, fin, bridge, w, assumptions, root)
            except TechvalError as exc:
                console.print(f"[yellow]Fitted growth path not run: {exc}[/yellow]")

        # Beside the DCF, not instead of it. The sum of the parts needs only the
        # statements and the bridge, so unlike the simulation and the APV it is
        # not taken down by a DCF that could not be formed.
        if assumptions.tmt.sotp:
            # The heading is printed by the caller, before the attempt, so a
            # section that refuses is still a section: it appears in the report
            # under its own name with the reason underneath, rather than either
            # vanishing or printing its title twice.
            _rule("Sum of the parts")
            try:
                _render_optional_sotp(
                    ticker, fin, bridge, assumptions, client, sotp_plan
                )
            except TechvalError as exc:
                console.print(f"[yellow]Sum of the parts not run: {exc}[/yellow]")

        if comps_result is not None:
            _render_comps(comps_result)

        # The remaining fitted sections, each behind its own flag and each
        # defaulting off. Every one is wrapped, because an optional section is
        # never allowed to take the valuation down with it: a missing recorded
        # panel should cost a reader that section and nothing else.
        proposed_peers: list[str] = []
        if assumptions.ml.peers.enabled:
            _rule("Learned comp set")
            try:
                proposed_peers = _render_optional_peers(ticker, assumptions, root)
            except TechvalError as exc:
                console.print(f"[yellow]Learned comp set not run: {exc}[/yellow]")

        # The warranted multiple and the signal test share one fit. The signal
        # section asks whether the residual the section above it just printed has
        # ever predicted anything, so fitting the model twice would be wasteful
        # and, worse, would let the two sections disagree.
        warranted_model = None
        warranted_panel = warranted_path = None
        warranted_error: TechvalError | None = None
        if assumptions.ml.warranted.enabled or assumptions.ml.signals.enabled:
            try:
                warranted_model, warranted_panel, warranted_path = (
                    _fit_warranted_panel(assumptions, root)
                )
            except TechvalError as exc:
                warranted_error = exc

        if assumptions.ml.warranted.enabled:
            _rule("Warranted multiple")
            if warranted_error is not None:
                console.print(
                    f"[yellow]Warranted multiple not fitted: {warranted_error}[/yellow]"
                )
            else:
                try:
                    _render_optional_warranted(
                        ticker, warranted_model, warranted_panel, warranted_path
                    )
                except TechvalError as exc:
                    console.print(
                        f"[yellow]No warranted read for {ticker.upper()}: {exc}[/yellow]"
                    )

        if assumptions.ml.signals.enabled:
            _rule("Has that residual ever predicted anything?")
            if warranted_error is not None:
                console.print(
                    "[yellow]Signal test not run: the warranted fit whose residual "
                    f"it scores could not be made: {warranted_error}[/yellow]"
                )
            else:
                try:
                    _render_optional_signal(assumptions, warranted_model, root)
                except TechvalError as exc:
                    console.print(f"[yellow]Signal test not run: {exc}[/yellow]")

        if assumptions.ml.mna.propensity_enabled:
            _rule("Acquisition propensity")
            try:
                _render_optional_propensity(ticker, assumptions, root)
            except TechvalError as exc:
                console.print(f"[yellow]Propensity not run: {exc}[/yellow]")

        if assumptions.ml.mna.precedents_enabled:
            _rule("Precedent transactions")
            try:
                hand = [p.upper() for p in assumptions.comps.peers]
                _render_optional_precedents(
                    assumptions, client, market, hand or proposed_peers
                )
            except TechvalError as exc:
                console.print(f"[yellow]Precedents not run: {exc}[/yellow]")

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


# --------------------------------------------------------------------------- #
# the results dashboard
# --------------------------------------------------------------------------- #


def _section_line(sid: str, status: str, seconds: str, cache: str) -> None:
    console.print(Text(f"  {sid:<11} {status:<10} {seconds:>9}   {cache}"))


def _print_run(run: SectionRun) -> None:
    _section_line(run.id, run.status, f"{run.seconds:,.2f}s", f"cache {run.cache}")


@app.command(rich_help_panel="Results")
def dashboard(
    config: Path = _CFG,
    ml_data: Path = typer.Option(
        Path("tests/fixtures"),
        "--ml-data",
        help=(
            "Directory the sections read their recorded artifacts from. The default "
            "is the committed fixtures, which is what makes the page reproducible."
        ),
    ),
    snapshot: Path = typer.Option(
        Path("docs/dashboard/snapshot.json"),
        "--snapshot",
        help="The results snapshot to collect into, or to render from.",
    ),
    out: Path = typer.Option(
        Path("docs/dashboard/index.html"), "--out", "-o", help="Where to write the page."
    ),
    collect: bool = typer.Option(
        None,
        "--collect/--no-collect",
        help=(
            "Run the sections, or render the existing snapshot as it stands. "
            "Default: collect only when the snapshot file does not exist."
        ),
    ),
    sections: str = typer.Option(
        None,
        "--sections",
        help=(
            "Comma-separated section ids to collect and merge into the existing "
            "snapshot. The sample and scoreboard sections that read them are "
            "collected again too."
        ),
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Recompute every section instead of reading the dashboard result cache.",
    ),
    collected_at: str = typer.Option(
        None,
        "--collected-at",
        help="The date the snapshot is stamped with (YYYY-MM-DD). Default: today.",
    ),
) -> None:
    """Collect every model's results from the fixtures and render the dashboard page.

    Each section runs the package's own entry points against the recorded
    artifacts under --ml-data, and each figure it produces is written to the
    snapshot with the entry point that computed it and the sha256 of every
    fixture it read. A section that cannot be reproduced from those artifacts
    is written as a refusal with its reason, and the page shows the refusal
    rather than a number. The page is then rendered from the snapshot alone, so
    a render never computes anything and the same snapshot always produces the
    same page.

    Section results are cached under ml.cache_dir, keyed on the section's code,
    its fixtures' bytes and the assumptions, and each line below says whether
    the cache was hit. --no-cache recomputes everything, which is the answer
    when model code outside a section's own entry point module has changed.
    """
    try:
        names = (
            [n.strip() for n in sections.split(",") if n.strip()] if sections else None
        )
        for name in names or ():
            section_module(name)  # an unknown id is refused before anything runs
        exists = snapshot.is_file()
        if names is not None and collect is False:
            raise ConfigError(
                "--sections names sections to collect and --no-collect says collect "
                "nothing. Drop one of them."
            )
        if collected_at is not None and collect is False:
            raise ConfigError(
                "--collected-at dates a collection and --no-collect renders an "
                "existing snapshot with the date it already carries. Drop one of them."
            )
        if collect is None:
            collect = names is not None or not exists
        if not collect and not exists:
            raise ConfigError(
                f"--no-collect renders an existing snapshot and there is none at "
                f"{snapshot}. Collect one first by leaving the flag off."
            )

        _rule("Dashboard")
        if collect:
            stamp = collected_at or date.today().isoformat()
            assumptions = Assumptions.load(config)
            base = None
            if names is not None:
                if not exists:
                    raise ConfigError(
                        f"--sections merges into an existing snapshot and there is "
                        f"none at {snapshot}. Collect every section first by leaving "
                        "--sections off."
                    )
                base = load_snapshot(snapshot)
            console.print(
                Text(
                    f"Collecting {', '.join(names) if names else 'every section'} from "
                    f"{ml_data}, stamped {stamp}, "
                    + ("cache off." if no_cache else "cache on."),
                    style="dim",
                ),
                soft_wrap=True,
            )
            snap = collect_snapshot(
                ml_data,
                stamp,
                sections=names,
                use_cache=not no_cache,
                assumptions=assumptions,
                base=base,
                on_section=_print_run,
            )
            dump_snapshot(snap, snapshot)
            refused = [s for s in snap.sections.values() if s["status"] == "refused"]
            _notes(
                [
                    f"{s['id']}: {r['what']}: {r['why']}"
                    for s in refused
                    for r in s["refusals"]
                ],
                heading="Refused",
            )
            console.print(
                Text(
                    f"\nWrote {snapshot}: commit {snap.techval_commit}, "
                    f"{len(snap.fixtures)} fixture digests.",
                    style="dim",
                ),
                soft_wrap=True,
            )
        else:
            snap = load_snapshot(snapshot)
            console.print(
                Text(
                    f"Rendering {snapshot} as collected on {snap.collected_at} at "
                    f"commit {snap.techval_commit}. Nothing was re-collected.",
                    style="dim",
                ),
                soft_wrap=True,
            )
            for sid in SECTION_IDS:
                section = snap.sections.get(sid)
                _section_line(
                    sid, section["status"] if section else "absent", "", "not collected"
                )

        write_dashboard(snap, out)
        console.print(Text(f"Wrote {out}.", style="dim"), soft_wrap=True)
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{escape(str(exc))}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
