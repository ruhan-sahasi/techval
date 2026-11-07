"""What the commands print with: a console, the rule over each section, the notes
under it, and how money, percentages and multiples read in a table.

The commands are spread over cli.py and the commands_* modules that cli.py
imports, so none of those modules can import these back from cli.py. They import
them from here instead, and each passes its own console, because the merger
commands print wider than the rest.
"""

from __future__ import annotations

import math
import sys

from rich.console import Console
from rich.text import Text


def make_console(width: int = 120) -> Console:
    """The terminal's own width when there is one, and ``width`` when output is piped."""
    return Console(width=None if sys.stdout.isatty() else width)


def print_rule(console: Console, title: str) -> None:
    console.print()
    console.rule(Text(title, style="bold"), style="dim")


def print_notes(console: Console, items, *, heading: str = "Notes") -> None:
    """Each item on its own line, a flag or warning in yellow; empty items are skipped."""
    items = [i for i in items if i]
    if not items:
        return
    console.print(f"\n[bold]{heading}[/bold]")
    for n in items:
        style = "yellow" if str(n).upper().startswith(("FLAG", "WARN")) else "dim"
        console.print(Text(f"  - {n}", style=style))


def _missing(v: float | None) -> bool:
    """None, or a float that is not a number: NaN or an infinity from a division."""
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def format_money(v: float | None, dp: int = 0) -> str:
    """Negatives in brackets, and a value rounded to nothing as a plain zero.

    A figure that rounds to zero would otherwise print as (0), or as -0 when it
    is a negative zero from a sign convention, both of which read as a rounding
    artefact rather than as a line that is not used.
    """
    if _missing(v):
        return "n/a"
    if abs(v) < 0.5 / (10**dp):
        return f"{0.0:,.{dp}f}"
    return f"({abs(v):,.{dp}f})" if v < 0 else f"{v:,.{dp}f}"


def format_pct(v: float | None, dp: int = 1) -> str:
    return "n/a" if _missing(v) else f"{v:.{dp}%}"


def format_multiple(v: float | None) -> str:
    """A valuation multiple; one that cannot be formed reads as not meaningful."""
    return "NM" if _missing(v) else f"{v:,.1f}x"
