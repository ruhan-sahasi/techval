"""Precedent transactions and the acquisition target screen, at the command line.

Two commands, fronting ``techval.tmt.precedents`` and ``techval.ml.mna``:

    techval precedents TICKERS   what acquirers paid, from the merger filings
    techval targets --dataset D  which companies get bought, ranked and scored

Both modules refuse in places, and most of the work here is making the refusals
as visible on a terminal as the numbers are. A precedent table and a target
screen are the two outputs in this package a reader is most likely to quote
without reading the note underneath, so the notes are not underneath.

**A precedent is not a trading comparable, and the gap between them is not
upside.** Every price in the precedent table contains a control premium, which
has historically run roughly 25 to 40 percent across TMT, and whatever synergies
the buyer underwrote. A precedent multiple therefore sits above a trading
multiple for the same company by construction. Reading the gap as evidence the
company is cheap today counts the control premium twice, once in the precedent
and once in the conclusion. That sentence is printed at the top of every
precedent run, in yellow, above the table, because a reader who reaches it after
the numbers has already formed the view it exists to prevent.

**Two premia are computed and they are allowed to disagree.** The headline is the
premium to the last close strictly before the announcement, because it is the
convention every reader can reconstruct. The engine also computes the premium to
the mean close over the thirty days before, and where the two differ by more
than ``LEAK_FLAG_POINTS`` the transaction is flagged. Both are always in the
table, side by side with the gap in points, and every flagged deal is restated
in its own section underneath. On the committed fixtures Roku is 11.3 percent
against 27.0 percent and Payoneer is 9.6 against 37.2: in each case the shares
had already run, and the one-day premium understates what the buyer paid over
the standalone value. Silicon Labs moves the other way at +10.3 points, which is
a stock that had fallen into the announcement and a one-day premium that
flatters the deal. One convention would have hidden all three.

**Some deals have no offer price, and printing one would be the error.** Iridium
is paid for in cash plus a collared number of Rocket Lab shares, so the exchange
ratio is fixed from the buyer's price at closing and does not exist at
announcement. ``Transaction.offer_price`` is ``None`` and the reason sits in
``notes``. The command prints the reason in full and prints nothing in the price
column. The cash leg of 27.00 dollars is shown once, in that section, labelled
as the cash leg and not as the offer, because substituting it would report a
40 percent discount to a price nobody agreed to pay.

**The target screen is the weakest model in this package and the output says
so.** Walk-forward AUC is 0.5685 against 0.5474 for sorting the universe from
smallest to largest, a lift of 0.0212 inside a fold standard deviation of
0.0910. ``beat_baseline`` is True and it is true by less than the noise, which is
a different claim from working. Nine of the fifteen coefficients change sign
between folds. The ranked list is printed because a banker works a list, and the
score is printed beside it in the same block rather than in a footnote.

**The screen cannot say whether cheap companies get bought.** Every price
derived feature is absent for exactly the companies that were acquired, because
no public source serves history for a delisted symbol, so market capitalisation
and every multiple over it would separate the classes perfectly by being
missing. The model is fundamentals only. The valuation channel, which is the one
a banker would most expect to matter, is untested here, and a screen that quietly
omitted it would invite the reader to conclude that value is not what drives
takeouts. The blind spot is printed above the list, not below it.

Money is USD millions, prices are dollars per share, premia and ``% cash`` are
decimal fractions rendered as percentages. Nothing here draws a random number.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import Assumptions
from .edgar import EdgarClient, HttpCache
from .errors import ConfigError, MissingDataError, TechvalError
from .market import make_price_source
from .ml.features import FEATURE_NAMES, FeaturePanel, FeatureRow
from .ml.mna import DealEvent, PropensityResult, build_universe, fit_propensity
from .tmt.precedents import (
    LEAK_FLAG_POINTS,
    MIN_DEALS_FOR_STATS,
    PrecedentSet,
    build_precedents,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Precedent transactions and the acquisition target screen.",
)

# Rich falls back to eighty when stdout is not a terminal, which crushes every
# column to four characters in a piped run. The rest of the CLI redirects at
# 120. This goes to 160, which is the width the precedent table actually needs:
# fourteen columns, of which one is an acquirer's legal name out of a merger
# agreement and one is a sub-vertical whose longest value is the twenty
# characters of "application_software".
#
# The number is not decoration. Below it rich shrinks the columns that are left,
# and the first casualties are the numeric ones: at 150 this table rendered
# Roku's unaffected close of 143.66 as "143." followed by an ellipsis, which
# still reads as a price. A truncated numeral is a wrong number that looks like
# a right one, so the width is chosen to fit the figures and the prose columns
# are the ones capped with an ellipsis instead.
console = Console(width=None if sys.stdout.isatty() else 160)

_CFG = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file.")
_NOCACHE = typer.Option(False, "--no-cache", help="Bypass the HTTP cache.")
_ASOF = typer.Option(
    None,
    "--as-of",
    help=(
        "Read the filing record as it stood on this date (YYYY-MM-DD). Documents "
        "filed later are discarded and prices stop there, so a historical run "
        "uses only what existed at the time."
    ),
)


# --------------------------------------------------------------------------- #
# rendering helpers
#
# Deliberately local rather than imported from cli.py. cli.py mounts this module
# with add_typer, so importing back out of it would close a cycle at import
# time, and the two files are the same handful of formatters either way.
# --------------------------------------------------------------------------- #


def _money(v: float | None, dp: int = 0) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"({abs(v):,.{dp}f})" if v < 0 else f"{v:,.{dp}f}"


def _pct(v: float | None, dp: int = 1) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{v:.{dp}%}"


def _mult(v: float | None) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NM"
    return f"{v:,.1f}x"


def _price(v: float | None) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{v:,.2f}"


def _rule(title: str) -> None:
    console.print()
    console.rule(Text(title, style="bold"), style="dim")


def _notes(items: list[str], *, heading: str = "Notes") -> None:
    if not items:
        return
    console.print(f"\n[bold]{heading}[/bold]")
    for n in items:
        style = "yellow" if n.upper().startswith(("FLAG", "WARN")) else "dim"
        console.print(Text(f"  - {n}", style=style))


def _warn_block(paragraphs: list[str], *, heading: str) -> None:
    """A standing caution, printed above the numbers it qualifies.

    Above rather than below, and in colour rather than dim, because the whole
    failure mode these two commands guard against is a reader who quotes the
    table and never reaches the note. A dim line after a table is a note; a
    coloured block before one is a condition of reading it.

    Each entry is a paragraph and is left to the console to wrap, rather than
    broken at a width chosen here. A block hard-wrapped at eighty on a terminal
    at a hundred and fifty reads as ragged, and the ragged edge is the first
    thing that makes a caution look like boilerplate.
    """
    console.print(f"\n[bold yellow]{heading}[/bold yellow]")
    for i, para in enumerate(paragraphs):
        if i:
            console.print()
        console.print(Text(f"  {para}", style="yellow"))


# --------------------------------------------------------------------------- #
# precedents: rendering
# --------------------------------------------------------------------------- #

#: The sentence the whole precedent output exists to keep attached to the
#: numbers. ``PrecedentSet.notes`` carries the same argument at length; this is
#: the version that goes above the table where it cannot be skipped.
_CONTROL_PREMIUM_WARNING = [
    "A precedent is not a trading comparable. Every price below is what a buyer "
    "paid to own the whole company, and it contains a control premium, roughly 25 "
    "to 40 percent across TMT historically, plus whatever synergies that buyer "
    "underwrote for itself.",
    "So a precedent multiple sits ABOVE a trading multiple for the same company by "
    "construction, not by coincidence. Reading the gap between the two as evidence "
    "that a company is cheap today counts the control premium twice: once inside "
    "the precedent, and once again in the conclusion drawn from it. These numbers "
    "do not belong in the same column as a comp table, and they are never averaged "
    "with one.",
]


def _premium_style(gap_points: float | None) -> str:
    """Yellow where the two unaffected conventions disagree enough to matter."""
    if gap_points is None:
        return ""
    return "yellow" if abs(gap_points) > LEAK_FLAG_POINTS else ""


def _render_deal_table(result: PrecedentSet) -> None:
    """One row per transaction, with both premia and the gap between them.

    Both premia are columns rather than a headline and a footnote. The engine has
    a house convention and it is the one-day close, but the reason the thirty-day
    figure is computed at all is that the two disagree on precisely the deals
    where the disagreement changes the answer, and a table that reported only the
    convention would hide that on exactly those rows.
    """
    t = Table(box=None, pad_edge=False)
    t.add_column("Announced", no_wrap=True)
    t.add_column("Target", no_wrap=True)
    # Truncated on one line rather than wrapped over three. The full legal name
    # of the buyer is in the provenance and detail sections below; here it only
    # has to identify the row.
    t.add_column("Acquirer", max_width=18, no_wrap=True, overflow="ellipsis")
    t.add_column("Status", no_wrap=True)
    t.add_column("Cons.", no_wrap=True)
    t.add_column("% cash", justify="right", min_width=6, no_wrap=True)
    # Wide enough for the cross reference that stands in for a price that does
    # not exist, because a marker wrapped over two lines stops being a marker.
    t.add_column("Offer", justify="right", min_width=9, no_wrap=True)
    t.add_column("Unaff. 1d", justify="right", min_width=9, no_wrap=True)
    t.add_column("Prem. 1d", justify="right", min_width=9, no_wrap=True)
    t.add_column("Prem. 30d", justify="right", min_width=9, no_wrap=True)
    t.add_column("Gap, pts", justify="right", min_width=9, no_wrap=True)
    t.add_column("EV/Rev", justify="right", min_width=6, no_wrap=True)
    t.add_column("EV/EBITDA", justify="right", min_width=9, no_wrap=True)
    # Pinned at the length of the longest sub-vertical name. Without the floor
    # rich crops it to whatever is left over, and a crop with no ellipsis turns
    # application_software into application_softwa with nothing to say it did.
    t.add_column("Sub-vertical", no_wrap=True, min_width=20)

    for txn in result.transactions:
        gap = txn.premium_gap_points
        style = _premium_style(gap)
        # A missing offer price is marked rather than blank. The marker points
        # at the section below that says why, so the empty cell is a cross
        # reference and not an omission the reader has to notice.
        offer = "see below" if txn.offer_price is None else _price(txn.offer_price)
        t.add_row(
            str(txn.announced) if txn.announced else "n/a",
            Text(txn.target_ticker, style="bold"),
            txn.acquirer_name or "n/a",
            txn.status,
            txn.consideration or "n/a",
            _pct(txn.pct_cash, 0),
            Text(offer, style="" if txn.offer_price is not None else "yellow"),
            _price(txn.unaffected_1d),
            Text(_pct(txn.premium_1d), style=style),
            Text(_pct(txn.premium_30d), style=style),
            Text("n/a" if gap is None else f"{gap:+,.1f}", style=style),
            _mult(txn.ev_revenue),
            _mult(txn.ev_ebitda),
            txn.sub_vertical or "unclassified",
        )
    console.print(t)


def _render_stats(
    stats: pd.DataFrame, minimum: int, n_transactions: int, *, title: str, indent: str = ""
) -> None:
    """Quartiles per column, with a refused column saying so in every cell.

    The alternative, a blank row, is worse than useless: a reader cannot tell a
    statistic that was declined from one the renderer dropped. ``n`` stays on the
    first row either way, because the difference between a median of eleven deals
    and a median of two is the whole of what a reader needs before quoting one.
    """
    t = Table(title=title, title_justify="left", box=None, pad_edge=False)
    t.add_column(indent, no_wrap=True)
    for c in stats.columns:
        t.add_column(str(c), justify="right")

    counts = {c: int(stats.loc["n", c]) for c in stats.columns}
    for row_label in stats.index:
        cells = []
        for c in stats.columns:
            v = stats.loc[row_label, c]
            if row_label == "n":
                cells.append(Text(f"{int(v)}", style="bold"))
            elif counts[c] < minimum:
                cells.append(Text("refused", style="yellow"))
            elif pd.isna(v):
                cells.append(Text("NM", style="dim"))
            elif "Premium" in str(c):
                cells.append(Text(_pct(float(v)), style="bold" if row_label == "Median" else ""))
            else:
                cells.append(Text(_mult(float(v)), style="bold" if row_label == "Median" else ""))
        t.add_row(Text(str(row_label), style="bold" if row_label in ("n", "Median") else ""), *cells)
    console.print(t)
    refused = [c for c in stats.columns if counts[c] < minimum]
    if refused:
        # The count that matters here is the transactions in the set, not the
        # largest column count in the frame. Quoting the latter would say "fewer
        # than 5 of the 6" over a nine-deal set, which reads as though three
        # deals had gone missing rather than as three columns declining.
        console.print(
            Text(
                f"  {len(refused)} of {len(stats.columns)} column(s) refused: fewer "
                f"than {minimum} of the {n_transactions} transactions carried a value "
                "for them, and a median of three transactions is three transactions.",
                style="yellow",
            )
        )


def _render_unpriced(result: PrecedentSet) -> None:
    """Deals where no offer price exists, and the reason in the filing's terms.

    The section exists so that the empty price cell above is an argument rather
    than a gap. Iridium is the case the fixtures carry: a collar sets the ratio
    off the buyer's price at closing, so there is no exchange ratio at
    announcement and no offer price to report. The cash leg is shown here and
    labelled, and nowhere else. Substituting it for the offer would report a
    price the parties never agreed and a premium struck against it.
    """
    unpriced = [t for t in result.transactions if t.offer_price is None]
    if not unpriced:
        return
    _rule("Deals with no offer price, and why that is the right answer")
    for txn in unpriced:
        console.print(
            f"\n[bold]{txn.target_ticker}[/bold]  {txn.target_name or ''}"
            f"  <- {txn.acquirer_name or 'buyer not identified'}"
            f"  announced {txn.announced}"
        )
        console.print(
            Text(
                "  No offer price is reported and none is substituted.",
                style="yellow bold",
            )
        )
        if txn.cash_per_share is not None:
            console.print(
                Text(
                    f"  The cash leg alone is {txn.cash_per_share:,.2f} per share. That "
                    "is the CASH LEG and not the offer: quoting it as the price would "
                    "understate the consideration by the whole of the stock leg and "
                    "strike every premium and multiple against a number nobody agreed.",
                    style="yellow",
                )
            )
        for note in txn.notes:
            console.print(Text(f"  - {note}", style="dim"))
        for flag in txn.flags:
            console.print(Text(f"  - {flag}", style="yellow"))


def _render_premium_disagreement(result: PrecedentSet) -> None:
    """Every deal where the two unaffected conventions disagree by more than the flag.

    Restated out of the table because the argument needs more than a column can
    hold. A negative gap is the stock having run into the announcement, which is
    what a leak looks like on a chart, and on those deals the one-day premium
    understates what the buyer paid over the standalone value. A positive gap is
    a stock that had fallen into the announcement, and there the one-day premium
    flatters the deal. Both are the same measurement problem and neither is a
    reason to prefer one convention silently.
    """
    flagged = [
        t
        for t in result.transactions
        if t.premium_gap_points is not None
        and abs(t.premium_gap_points) > LEAK_FLAG_POINTS
    ]
    if not flagged:
        return
    _rule(f"Premium: the two conventions disagree by more than {LEAK_FLAG_POINTS:.0f} points")
    t = Table(box=None, pad_edge=False)
    t.add_column("Target", no_wrap=True)
    t.add_column("Offer", justify="right")
    t.add_column("Unaff. 1d", justify="right")
    t.add_column("Prem. 1d", justify="right")
    t.add_column("Unaff. 30d", justify="right")
    t.add_column("Prem. 30d", justify="right")
    t.add_column("Gap, pts", justify="right")
    t.add_column("Reading", overflow="fold")
    for txn in flagged:
        gap = txn.premium_gap_points
        reading = (
            "ran INTO the announcement; the 1-day premium understates what was paid"
            if gap < 0
            else "FELL into the announcement; the 1-day premium flatters the deal"
        )
        t.add_row(
            Text(txn.target_ticker, style="bold"),
            _price(txn.offer_price),
            _price(txn.unaffected_1d),
            Text(_pct(txn.premium_1d), style="yellow"),
            _price(txn.unaffected_30d),
            Text(_pct(txn.premium_30d), style="yellow"),
            Text(f"{gap:+,.1f}", style="yellow bold"),
            reading,
        )
    console.print(t)
    console.print(
        "\n[dim]The headline convention is the last close strictly before the "
        "announcement, because it is the one any reader can reconstruct. Both are "
        "reported on every row above. Where they disagree, only one of the two is "
        "a control premium, and which one depends on whether the market already "
        "knew.[/dim]"
    )


def _render_sub_vertical_stats(result: PrecedentSet) -> None:
    """Statistics per sub-vertical, and a plain list of the buckets refused."""
    _rule("By sub-vertical")
    if result.sub_vertical_stats:
        counts = {}
        for txn in result.transactions:
            key = txn.sub_vertical or "unclassified"
            counts[key] = counts.get(key, 0) + 1
        for name in sorted(result.sub_vertical_stats):
            console.print()
            _render_stats(
                result.sub_vertical_stats[name],
                result.min_deals_for_stats,
                counts.get(name, 0),
                title=name,
                indent="",
            )
    else:
        console.print(
            Text(
                f"\n  No sub-vertical reached the {result.min_deals_for_stats}-deal "
                "floor, so no bucket gets a statistic. Every refusal and its count "
                "is in the flags below.",
                style="yellow",
            )
        )


# --------------------------------------------------------------------------- #
# precedents: the command
# --------------------------------------------------------------------------- #


@app.command()
def precedents(
    tickers: str = typer.Argument(
        ..., help="Comma-separated target tickers, e.g. SPLK,ZEN,MNDT,WORK"
    ),
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    as_of: str = _ASOF,
    sub_vertical: str = typer.Option(
        None,
        "--sub-vertical",
        help="Keep only deals classified into this sub-vertical, and recompute.",
    ),
    min_size: float = typer.Option(
        None,
        "--min-size",
        help=(
            "Floor on equity purchase price, USD millions, applied after the build "
            "and stricter than ml.mna.min_deal_size: a deal whose size could not be "
            "built is dropped here rather than waved through, and the count is flagged."
        ),
    ),
    min_deals: int = typer.Option(
        MIN_DEALS_FOR_STATS,
        "--min-deals",
        help="Deals a column needs before any statistic is reported for it.",
    ),
) -> None:
    """Precedent transactions from the targets' own merger filings.

    Every deal is read out of an 8-K, DEFM14A, SC 14D9, S-4 or DEFM14C filed by
    the target itself. Nothing is taken from a database, which is why the output
    can say which accession each price came from and why some rows decline.

    Three refusals are load-bearing and all three are printed rather than
    smoothed. A deal whose consideration cannot be reduced to one number at
    announcement, because a collar fixes the exchange ratio off the buyer's price
    at closing, gets no offer price and an explanation. A column with fewer than
    ``--min-deals`` values gets no statistic. And a completed target has been
    delisted, so no public source serves its price history and its premium is
    absent rather than zero: real premia over closed deals need
    ``price_source: csv`` in the assumptions with the closes supplied.
    """
    try:
        assumptions = Assumptions.load(config)
        if as_of:
            assumptions.as_of = as_of
        knowledge = (
            date.fromisoformat(assumptions.as_of) if assumptions.as_of else None
        )
        cache = HttpCache(enabled=not no_cache)
        client = EdgarClient(cache, knowledge_date=knowledge)
        prices = make_price_source(
            assumptions.price_source, cache, assumptions.price_csv_dir
        )

        names = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not names:
            console.print("[red]No tickers given.[/red]")
            raise typer.Exit(1)

        result = build_precedents(
            names,
            client,
            assumptions,
            prices=prices,
            min_deals_for_stats=min_deals,
        )
        if sub_vertical or min_size is not None:
            result = result.filter(sub_vertical=sub_vertical, min_size=min_size)

        mna = assumptions.ml.mna
        _rule(
            f"Precedent transactions: {len(result.transactions)} of "
            f"{len(names)} ticker(s), as of {result.as_of}"
        )
        console.print(
            f"[dim]Lookback {mna.lookback_years} years from {result.as_of}. "
            f"Size floor {mna.min_deal_size:,.0f}mm from ml.mna.min_deal_size"
            + (f", narrowed to {min_size:,.0f}mm here" if min_size is not None else "")
            + (f". Sub-vertical {sub_vertical}" if sub_vertical else "")
            + f". Prices from {assumptions.price_source}. Statistics need "
            f"{result.min_deals_for_stats} deals per column.[/dim]"
        )

        _warn_block(_CONTROL_PREMIUM_WARNING, heading="Read this before the table")

        if not result.transactions:
            console.print(
                "\n[yellow]No transaction survived. Most companies are not "
                "acquired, so an empty precedent set is an ordinary answer and not "
                "an error. The flags below say what was dropped and why.[/yellow]"
            )
            _notes(list(result.flags), heading="Flags")
            return

        console.print()
        _render_deal_table(result)

        _rule("Statistics")
        _render_stats(
            result.stats,
            result.min_deals_for_stats,
            len(result.transactions),
            title="All transactions",
        )

        _render_sub_vertical_stats(result)
        _render_premium_disagreement(result)
        _render_unpriced(result)

        # Deal size sits here rather than in the table above. A fifteenth column
        # there made rich shrink the price columns until 143.66 rendered as
        # 143.…, and a truncated numeral still reads as a number.
        _rule("Deal size and provenance")
        t = Table(box=None, pad_edge=False)
        t.add_column("Target", no_wrap=True)
        t.add_column("Equity, mm", justify="right")
        t.add_column("EV, mm", justify="right")
        t.add_column("Form", no_wrap=True)
        t.add_column("Accession", no_wrap=True)
        t.add_column("Conf.", justify="right")
        t.add_column("Basis for the unaffected price", overflow="fold")
        for txn in result.transactions:
            t.add_row(
                txn.target_ticker,
                _money(txn.equity_value),
                _money(txn.enterprise_value),
                txn.source_form or "n/a",
                txn.source_accession or "n/a",
                f"{txn.confidence:.2f}",
                txn.unaffected_basis or "none: no price series",
            )
        console.print(t)
        console.print(
            "\n[dim]Equity purchase price is what ml.mna.min_deal_size and --min-size "
            "are struck on, so a floor is legible against the quantity it filtered. "
            "An n/a is a deal whose size could not be built, and it is excluded by a "
            "size floor rather than waved through: an unknown size cannot be shown to "
            "clear one.[/dim]"
        )
        console.print(
            "\n[dim]Confidence is a coarse ladder over how much of the deal the "
            "filing text pinned down, not a probability: 0.90 an unambiguous all-cash "
            "price, 0.75 a stock or mixed price completed with the buyer's own close, "
            "0.50 consideration identified but no per-share value, 0.25 a merger "
            "agreement with no consideration parsed.[/dim]"
        )

        _notes(list(result.flags), heading="Flags and refusals")
        _notes(list(result.notes), heading="Standing notes")
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# targets: the prepared dataset
#
# The panel this model needs is every TMT registrant on every quarter end for
# seven years, each built through a CompanyFacts pinned to the row date. That is
# tens of thousands of point-in-time reads against EDGAR, and it is not a thing
# a command line builds while a user waits. It is built once and kept, which is
# also the only way the departed registrants survive: a delisted ticker cannot be
# recovered from today's company_tickers.json, so the roster has to have been
# written down while the company still existed.
# --------------------------------------------------------------------------- #

#: Columns of the panel CSV that are identity rather than features. Anything
#: else that is not in FEATURE_NAMES is carried as a derived extra, which is how
#: mna_net_cash_to_assets reaches build_matrix.
_PANEL_KEYS = ("ticker", "as_of", "knowledge_date", "statement_date")


def _open_panel(directory: Path):
    """The panel CSV, gzipped or not, or a refusal naming both paths tried."""
    gz, plain = directory / "panel.csv.gz", directory / "panel.csv"
    if gz.exists():
        return gzip.open(gz, "rt", newline="")
    if plain.exists():
        return plain.open("rt", newline="")
    raise MissingDataError(
        "feature panel",
        hint=f"no panel.csv.gz and no panel.csv in {directory}",
    )


def load_dataset(
    directory: Path,
) -> tuple[FeaturePanel, Any, list[DealEvent], dict[tuple[str, date], dict[str, float | None]]]:
    """Read a prepared propensity dataset: the panel, the roster, the deals.

    Three files, in the shape ``tests/fixtures/mna`` carries and the shape the
    manifest there documents:

    ``universe.json``   a roster of registrants with ``ticker``, ``cik``,
                        ``admitted`` and ``departed``. The departed entries are
                        the point of the file. A roster whose departed count is
                        zero is a survivor list, every positive label is missing
                        from it, and a model fitted on it measures nothing while
                        looking perfectly well behaved.

    ``events.json``     announced acquisitions, with the announcement date and a
                        completion flag. Announcement is the label; a blocked
                        deal is still a positive.

    ``panel.csv.gz``    one row per registrant per date, each column a feature
                        built through a ``CompanyFacts`` pinned to that row's
                        knowledge date. An empty cell is a figure that could not
                        be sourced and stays ``None`` rather than becoming a
                        zero, because the model has a missing-share feature that
                        depends on the difference.

    A column in the CSV that is neither an identity key nor one of
    ``FEATURE_NAMES`` is passed through as a derived extra rather than dropped,
    which is how ``mna_net_cash_to_assets`` reaches the design matrix without
    this loader having to know the model's feature list.
    """
    if not directory.exists():
        raise MissingDataError("propensity dataset", hint=f"no directory at {directory}")

    roster_path = directory / "universe.json"
    events_path = directory / "events.json"
    for path, what in ((roster_path, "universe roster"), (events_path, "deal events")):
        if not path.exists():
            raise MissingDataError(what, hint=f"no file at {path}")

    universe = build_universe(json.loads(roster_path.read_text()))

    events = [
        DealEvent(
            ticker=str(e["ticker"]).upper(),
            announced=date.fromisoformat(e["announced"]),
            completed=bool(e.get("completed")),
            name=str(e.get("name") or ""),
            acquirer=e.get("acquirer"),
            source_form=e.get("source_form"),
        )
        for e in json.loads(events_path.read_text())
    ]

    rows: list[FeatureRow] = []
    extras: dict[tuple[str, date], dict[str, float | None]] = {}
    with _open_panel(directory) as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or ())
        missing_keys = [k for k in _PANEL_KEYS if k not in header]
        if missing_keys:
            raise ConfigError(
                f"the panel CSV in {directory} is missing the identity column(s) "
                f"{', '.join(missing_keys)}. A feature row with no ticker and no "
                "date cannot be joined to a label."
            )
        feature_cols = [c for c in header if c in set(FEATURE_NAMES)]
        extra_cols = [c for c in header if c not in set(FEATURE_NAMES) and c not in _PANEL_KEYS]
        for rec in reader:
            when = date.fromisoformat(rec["as_of"])
            values: dict[str, float | None] = {
                name: (None if rec.get(name, "") == "" else float(rec[name]))
                for name in FEATURE_NAMES
            }
            rows.append(
                FeatureRow(
                    ticker=str(rec["ticker"]).upper(),
                    as_of=when,
                    knowledge_date=date.fromisoformat(rec["knowledge_date"]),
                    values=values,
                    missing=[n for n, v in values.items() if v is None],
                    statement_date=(
                        date.fromisoformat(rec["statement_date"])
                        if rec["statement_date"]
                        else None
                    ),
                )
            )
            if extra_cols:
                extras[(str(rec["ticker"]).upper(), when)] = {
                    c: (None if rec[c] == "" else float(rec[c])) for c in extra_cols
                }
    if not rows:
        raise MissingDataError(
            "feature panel rows", hint=f"the panel CSV in {directory} holds no rows"
        )

    panel = FeaturePanel(rows=rows)
    panel.notes.append(
        f"Loaded {len(rows):,} feature rows over {len(panel.dates)} dates and "
        f"{len(panel.tickers):,} registrants from {directory}. "
        f"{len(feature_cols)} of the {len(FEATURE_NAMES)} known features are present "
        f"as columns and {len(extra_cols)} derived extra(s) were carried through."
    )
    return panel, universe, events, extras


# --------------------------------------------------------------------------- #
# targets: rendering
# --------------------------------------------------------------------------- #

#: Printed above the ranked list on every run. Both halves are findings from the
#: module's own evaluation rather than modesty: the first is what the score is,
#: the second is what the feature set cannot see.
def _distinct_deals(result: PropensityResult) -> int:
    """Companies actually bought, counted on the matrix the model was scored on.

    Not ``LabelReport.n_distinct_deals``, which counts labelled observations
    before the join to the feature store and is therefore larger. The verdict is
    computed on the matrix, so the number that governs it is the one here, and
    quoting the larger count beside a score the smaller sample produced would
    overstate the evidence by exactly the companies the panel could not build.
    """
    return len({t for t, lab in zip(result.matrix.tickers, result.matrix.y) if lab})


def _worth_block(result: PropensityResult) -> list[str]:
    """What this model is worth, in the numbers rather than in adjectives.

    Returned as paragraphs and printed above the ranked list. The order is
    deliberate: the score, then what the score means, then the sample it rests
    on. A reader who stops after the first paragraph has still been told the
    lift is inside the noise.
    """
    ev = result.evaluation
    sd = ev.fold_sd
    inside = sd is not None and abs(ev.lift) < sd

    head = (
        f"Walk-forward AUC {ev.score:.4f} against {ev.baseline_score:.4f} for sorting "
        f"the universe from smallest to largest, which costs one line of code. Lift "
        f"{ev.lift:+.4f} on {ev.n_observations:,} observations over {len(ev.folds)} folds."
    )
    if sd is not None:
        head += (
            f" Fold standard deviation {sd:.4f}, so the lift is "
            + ("INSIDE" if inside else "outside")
            + " the fold-to-fold noise."
        )
    paragraphs = [head]

    if ev.beat_baseline and inside:
        paragraphs.append(
            "beat_baseline is technically True and it is true by less than the noise. "
            "That is a different claim from the model working. The honest reading is "
            "that this screen has not been shown to add anything to knowing how big a "
            "company is, and the size sort is free."
        )
    elif not ev.beat_baseline:
        paragraphs.append(
            "The model does NOT beat the size sort on this sample. Use the size sort."
        )

    stability = result.coefficient_stability()
    if not stability.empty:
        n_flip = int((stability["sign_flips"] > 0).sum())
        paragraphs.append(
            f"{n_flip} of the {len(stability)} coefficients change sign between folds. "
            "A coefficient that changes sign has not been measured, it has been "
            "sampled, and no story told off its sign survives the next fold."
        )

    paragraphs.append(
        f"The verdict rests on {_distinct_deals(result)} distinct deals that reached "
        f"the design matrix, not on the {int(result.matrix.y.sum()):,} positive "
        "observations: the extra rows are the same companies seen again on later "
        "dates and add no independent evidence. A sample this small does not settle "
        "the sign of a lift this size."
    )
    return paragraphs


_BLIND_SPOT = [
    "No price-derived feature is in this model, and the omission is not an "
    "oversight. A company that is acquired is delisted, and no public source "
    "serves price history for a delisted symbol, so market capitalisation, "
    "momentum, volatility and every multiple taken over a price are missing for "
    "exactly the companies that were bought and present for the ones that were "
    "not. A classifier handed that scores near a perfect AUC by learning that "
    "companies with no share price get acquired, which is true, circular and "
    "useless.",
    "The cost is real and is stated rather than worked around: this screen is "
    "fundamentals only, and it CANNOT SAY WHETHER CHEAP COMPANIES GET BOUGHT. A "
    "cheap company and an expensive one look identical to it. Do not read the "
    "absence of valuation from this list as evidence that valuation does not "
    "drive takeouts. It is evidence that this model was not allowed to look.",
]


def _render_screen(frame: pd.DataFrame, result: PropensityResult, screen_on: date) -> None:
    """The ranked list, with the score printed in the same block as the names.

    Beside the list rather than under it. A target screen is the output in this
    package most likely to be photographed and pasted into a deck, and a caveat
    on a different page from the names is a caveat nobody carries with them.
    """
    ev = result.evaluation
    _rule(f"Acquisition target screen, {screen_on}")
    console.print(
        Text(
            f"  Walk-forward AUC {ev.score:.4f} against {ev.baseline_score:.4f} for the "
            f"size sort, a lift of {ev.lift:+.4f} "
            + (
                f"inside a fold standard deviation of {ev.fold_sd:.4f}."
                if ev.fold_sd is not None
                else "with no fold dispersion available."
            ),
            style="yellow bold",
        )
    )
    console.print(
        Text(
            "  Fundamentals only: no valuation channel is in this ranking.",
            style="yellow",
        )
    )
    console.print()
    t = Table(box=None, pad_edge=False)
    t.add_column("#", justify="right", no_wrap=True)
    t.add_column("Ticker", no_wrap=True)
    t.add_column("Name", max_width=30, overflow="ellipsis")
    t.add_column("Sub-vertical", no_wrap=True)
    t.add_column("Prob.", justify="right")
    t.add_column("Score", justify="right")
    t.add_column("What put it on the list", overflow="fold")
    for _, row in frame.iterrows():
        drivers = [row[c] for c in ("driver_1", "driver_2", "driver_3") if row.get(c)]
        t.add_row(
            str(int(row["rank"])),
            Text(str(row["ticker"]), style="bold"),
            str(row["name"]),
            str(row["sub_vertical"] or "unclassified"),
            _pct(float(row["probability"])),
            f"{float(row['score']):+.2f}",
            ", ".join(str(d) for d in drivers),
        )
    console.print(t)
    console.print(
        "\n[dim]Probability is prior-corrected for the class weighting used in the "
        "fit. Check the calibration table below before quoting one as a likelihood: "
        "a model can rank well and calibrate badly, and the two fail "
        "independently.[/dim]"
    )


def _render_base_rates(result: PropensityResult) -> None:
    """Positives per year, which moves with the cycle and not with the companies."""
    frame = result.base_rates
    if frame is None or frame.empty:
        return
    _rule("Base rate by year")
    t = Table(box=None, pad_edge=False)
    t.add_column("Year", no_wrap=True)
    t.add_column("Observations", justify="right")
    t.add_column("Positives", justify="right")
    t.add_column("Base rate", justify="right")
    for _, row in frame.iterrows():
        t.add_row(
            str(int(row["year"])),
            f"{int(row['n']):,}",
            f"{int(row['positives']):,}",
            Text(_pct(float(row["base_rate"]), 2), style="bold"),
        )
    console.print(t)
    rates = frame["base_rate"].astype(float)
    lo, hi = float(rates.min()), float(rates.max())
    lo_year = int(frame.loc[rates.idxmin(), "year"])
    hi_year = int(frame.loc[rates.idxmax(), "year"])
    console.print(
        Text(
            f"\n  The base rate runs from {lo:.2%} in {lo_year} to {hi:.2%} in {hi_year}, "
            f"a factor of {hi / lo:,.1f}. 2021 and 2023 are different worlds for "
            "technology M&A: rates, the financing market and the antitrust posture all "
            "moved, and the deal count moved with them. A model scored across both "
            "without this table beside it can look skilful when it has learned which "
            "year it is.",
            style="yellow",
        )
    )


def _render_stability(result: PropensityResult) -> None:
    """Every fold's coefficient, and the ones that changed sign."""
    frame = result.coefficient_stability()
    if frame.empty:
        return
    _rule("Coefficient stability across folds")
    t = Table(box=None, pad_edge=False)
    t.add_column("Feature", no_wrap=True)
    t.add_column("Mean", justify="right")
    t.add_column("SD", justify="right")
    t.add_column("Folds", justify="right")
    t.add_column("Sign changes", justify="right")
    for _, row in frame.iterrows():
        flipped = int(row["sign_flips"]) > 0
        style = "yellow" if flipped else ""
        t.add_row(
            Text(str(row["feature"]), style=style),
            Text(f"{float(row['mean']):+.3f}", style=style),
            Text("n/a" if pd.isna(row["sd"]) else f"{float(row['sd']):.3f}", style=style),
            Text(f"{int(row['folds'])}", style=style),
            Text("yes" if flipped else "no", style="yellow bold" if flipped else "dim"),
        )
    console.print(t)
    n_flip = int((frame["sign_flips"] > 0).sum())
    console.print(
        Text(
            f"\n  {n_flip} of {len(frame)} coefficients change sign between folds. A "
            "coefficient that changes sign has not been measured, it has been "
            "sampled, and no story told off its sign survives the next fold. With "
            "this many deals spread over seven years several of these columns are "
            "unstable, and this table is where that is admitted rather than hidden "
            "behind a single fitted number. Read the verdict, not the sign.",
            style="yellow",
        )
    )


def _render_precision(result: PropensityResult, top_k: int) -> None:
    """Precision and recall at k, per date, for the model and for the size sort.

    Per date because a pooled top twenty over seven years is not a list anybody
    could have run. The size columns are here to be compared against, and where
    they are zero on a date the model's own zero means something different from
    what it would mean alone.
    """
    frame = result.precision_at_k
    if frame is None or frame.empty:
        return
    _rule(f"Precision and recall at k={top_k}, by screen date")
    t = Table(box=None, pad_edge=False)
    t.add_column("Date", no_wrap=True)
    t.add_column("n", justify="right")
    t.add_column("Positives", justify="right")
    t.add_column("Base rate", justify="right")
    t.add_column("Prec. model", justify="right")
    t.add_column("Prec. size", justify="right")
    t.add_column("Recall model", justify="right")
    t.add_column("Recall size", justify="right")
    for _, row in frame.iterrows():
        model_p = float(row["precision_at_k_model"])
        size_p = float(row["precision_at_k_size"])
        t.add_row(
            str(row["as_of"]),
            f"{int(row['n']):,}",
            f"{int(row['positives']):,}",
            _pct(float(row["base_rate"]), 2),
            Text(_pct(model_p, 1), style="bold" if model_p > size_p else ""),
            _pct(size_p, 1),
            _pct(float(row["recall_at_k_model"]), 1),
            _pct(float(row["recall_at_k_size"]), 1),
        )
    console.print(t)
    beats = int(
        (frame["precision_at_k_model"] > frame["precision_at_k_size"]).sum()
    )
    console.print(
        Text(
            f"\n  The model's top {top_k} holds more true targets than the size sort's "
            f"on {beats} of {len(frame)} dates. A precision of five percent on a base "
            "rate of three is one extra name in twenty, which is worth having and is "
            "not a result.",
            style="dim",
        )
    )


def _render_calibration(result: PropensityResult) -> None:
    """Predicted against realised, with the thin bins marked as thin."""
    frame = result.calibration
    if frame is None or frame.empty:
        console.print(
            "\n[yellow]No calibration table: the sample is below the harness "
            "floor, so the probabilities above may be ranked but not quoted.[/yellow]"
        )
        return
    _rule("Calibration of the out-of-sample probabilities")
    t = Table(box=None, pad_edge=False)
    t.add_column("Bucket", no_wrap=True)
    t.add_column("n", justify="right")
    t.add_column("Predicted", justify="right")
    t.add_column("Realised", justify="right")
    t.add_column("Gap", justify="right")
    for _, row in frame.iterrows():
        thin = bool(row["thin"])
        style = "dim" if thin else ""
        t.add_row(
            Text(f"{float(row['low']):.0%} to {float(row['high']):.0%}", style=style),
            Text(f"{int(row['n']):,}" + (" thin" if thin else ""), style=style),
            Text(_pct(float(row["predicted"]), 1), style=style),
            Text(_pct(float(row["realised"]), 1), style=style),
            Text(f"{float(row['gap']):+.1%}", style="yellow" if abs(float(row["gap"])) > 0.1 else style),
        )
    console.print(t)

    # The direction of the miss, computed rather than left to the reader. A
    # table of gaps is evidence; the sentence a reader needs is whether the
    # probabilities are too high or too low, and on a class-weighted fit over a
    # three percent base rate they are almost always too high.
    solid = frame[~frame["thin"].astype(bool)]
    over = solid[solid["gap"].astype(float) < -0.05]
    if len(over):
        worst = float(over["gap"].astype(float).min())
        console.print(
            Text(
                f"\n  {len(over)} of the {len(solid)} buckets that are not thin "
                f"over-predict by more than five points, the worst by {abs(worst):.1%}. "
                "These probabilities are ranks wearing a percentage sign. Read the "
                "ORDER of the list and do not quote a number off it as a likelihood.",
                style="yellow",
            )
        )
    console.print(
        "\n[dim]A bucket marked thin holds too few observations for its realised "
        "rate to mean anything. Ranking and calibration fail independently: a model "
        "that orders names correctly can still be badly wrong about the level, and "
        "this table is the only place that shows which.[/dim]"
    )


# --------------------------------------------------------------------------- #
# targets: the command
# --------------------------------------------------------------------------- #


@app.command()
def targets(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help=(
            "Directory holding universe.json, events.json and panel.csv.gz. See "
            "tests/fixtures/mna for the shape and its manifest for how it was built."
        ),
    ),
    config: Path = _CFG,
    as_of: str = _ASOF,
    screen_on: str = typer.Option(
        None,
        "--screen-on",
        help=(
            "Date to rank on (YYYY-MM-DD). Must be a date the panel carries. "
            "Defaults to the latest."
        ),
    ),
    top: int = typer.Option(20, "--top", "-k", help="Names on the printed list."),
    sub_vertical: str = typer.Option(
        None, "--sub-vertical", help="Show only this sub-vertical on the list."
    ),
    horizon_months: int = typer.Option(
        12, "--horizon-months", help="Label window: bought within this many months."
    ),
    gap_days: int = typer.Option(
        90,
        "--gap-days",
        help=(
            "An observation whose announcement falls inside this many days of its "
            "feature date is dropped, not labelled negative."
        ),
    ),
) -> None:
    """Rank acquisition candidates, and print what the ranking is worth.

    This is the weakest model in the package and the output is built to say so.
    The screen is fitted walk-forward by date with an embargo of the full label
    horizon plus the gap, scored against sorting the universe from smallest to
    largest, and reported with its fold dispersion, its per-year base rate and
    the count of coefficients that change sign between folds. On the recorded
    dataset the lift over the size sort is inside one fold standard deviation,
    which means the screen has not been shown to beat a free alternative.

    Two things about the sample govern everything else. The universe is
    constructed from a roster that keeps companies after they delist, because a
    company that is acquired stops filing and vanishes from every current ticker
    list, and a universe built the obvious way loses most of the positive class
    while remaining a perfectly well-behaved object. And no price-derived feature
    is used, because a delisted target has no price history from any public
    source, so every such feature is missing for the positives and present for
    the negatives and a classifier handed one learns the delisting instead of the
    deal. The second decision costs the valuation channel outright.
    """
    try:
        assumptions = Assumptions.load(config)
        if as_of:
            assumptions.as_of = as_of

        panel, universe, events, extras = load_dataset(dataset)
        resolved_as_of = (
            date.fromisoformat(assumptions.as_of)
            if assumptions.as_of
            else max(panel.dates)
        )
        when = date.fromisoformat(screen_on) if screen_on else max(panel.dates)
        if when not in set(panel.dates):
            raise ConfigError(
                f"the panel carries no rows dated {when}. Its dates run from "
                f"{min(panel.dates)} to {max(panel.dates)}."
            )

        result = fit_propensity(
            panel,
            universe,
            events,
            assumptions,
            as_of=resolved_as_of,
            horizon_months=horizon_months,
            gap_days=gap_days,
            extras=extras,
            top_k=top,
        )

        card = result.model.card
        _rule(f"Acquisition propensity: {card.name}, as of {resolved_as_of}")
        console.print(
            f"[dim]{len(universe.members):,} registrants, of which "
            f"{len(universe.departed_members()):,} have left the filing record. "
            f"{len(events):,} announced deals, {result.labels.n_distinct_deals} of them "
            f"labelled and {_distinct_deals(result)} reaching the design matrix. "
            f"Label horizon {horizon_months} months, "
            f"rumour gap {gap_days} days, embargo "
            f"{card.hyperparameters.get('embargo_days')} days, "
            f"{len(result.folds)} walk-forward folds, seed "
            f"{assumptions.ml.random_seed}.[/dim]"
        )

        _warn_block(_worth_block(result), heading="What this model is worth")
        _warn_block(_BLIND_SPOT, heading="What this model cannot see")

        cross_section = len(universe.as_of(when))
        frame = result.model.rank(
            panel,
            universe,
            when,
            top_k=max(cross_section, top),
            extras=extras,
            events=events,
        )
        if sub_vertical:
            frame = frame[frame["sub_vertical"] == sub_vertical]
            if frame.empty:
                console.print(
                    f"\n[yellow]No ranked name is classified {sub_vertical}.[/yellow]"
                )
            frame = frame.copy()
        frame = frame.head(top)
        if not frame.empty:
            _render_screen(frame, result, when)

        _render_base_rates(result)
        _render_precision(result, top)
        _render_stability(result)
        _render_calibration(result)

        _rule("Label sample")
        t = Table(box=None, pad_edge=False)
        t.add_column("")
        t.add_column("", justify="right")
        for label, value in result.labels.rows():
            shown = (
                _pct(float(value), 2)
                if label == "Base rate"
                else (f"{value:,}" if isinstance(value, (int, np.integer)) else str(value))
            )
            bold = label in ("Distinct deals", "Base rate")
            t.add_row(
                Text(label, style="bold" if bold else ""),
                Text(shown, style="bold" if bold else ""),
            )
        console.print(t)
        console.print(
            "\n[dim]An unresolved window is dropped rather than counted as a "
            "negative: a company whose twelve months are not yet up has not been "
            "shown not to be a target. An observation inside the rumour gap is "
            "dropped for the mirror reason, because labelling a company that was "
            "under offer six weeks later as a non-target teaches the opposite of "
            "the truth.[/dim]"
        )

        _notes(list(card.limitations), heading="Limitations")
        _notes(list(result.notes) + list(panel.notes), heading="Notes")
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
