"""What the commands print with: a console, the rule over each section, the notes
under it, and how money and percentages read in a table.

The commands are spread over cli.py and the commands_* modules that cli.py
imports, so none of those modules can import these back from cli.py. They import
them from here instead, and each passes its own console, because the merger
commands print wider than the rest.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.text import Text


def make_console(width: int = 120) -> Console:
    """The terminal's own width when there is one, and ``width`` when output is piped."""
    return Console(width=None if sys.stdout.isatty() else width)


def print_rule(console: Console, title: str) -> None:
    console.print()
    console.rule(Text(title, style="bold"), style="dim")


def print_notes(console: Console, items: list[str], *, heading: str = "Notes") -> None:
    if not items:
        return
    console.print(f"\n[bold]{heading}[/bold]")
    for n in items:
        style = "yellow" if n.upper().startswith(("FLAG", "WARN")) else "dim"
        console.print(Text(f"  - {n}", style=style))


def format_money(v: float | None, dp: int = 0) -> str:
    if v is None:
        return "n/a"
    return f"({abs(v):,.{dp}f})" if v < 0 else f"{v:,.{dp}f}"


def format_pct(v: float | None, dp: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{dp}%}"
