"""techval cache: what the HTTP cache holds, and pruning what has aged out.

The cache keeps every response forever by default, so a valuation reproduces;
a tracker run daily adds a price history per holding per day. These two
commands make that visible and give the owner the decision: ``info`` counts,
``prune`` removes entries older than a cutoff, and ``--dry-run`` says what it
would remove first. Removed entries are simply refetched when next needed.
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from .edgar import CACHE_ROOT

app = typer.Typer(name="cache", help="Inspect and prune the HTTP cache.", no_args_is_help=True)
console = Console()

_ROOT = typer.Option(CACHE_ROOT, "--root", help="The cache directory. Defaults to TECHVAL_CACHE or ~/.techval/cache.")

DAY = 86400.0


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GB"


def _entries(root: Path) -> list[Path]:
    return sorted(root.glob("*.cache"))


@app.command()
def info(root: Path = _ROOT) -> None:
    """Count the cached responses, their size and their age."""
    root = Path(root).expanduser()
    if not root.is_dir():
        console.print(f"No cache at {escape(str(root))} yet.")
        return
    entries = _entries(root)
    temps = list(root.glob("*.tmp"))
    total = sum(p.stat().st_size for p in entries)
    console.print(f"{escape(str(root))}: {len(entries):,} entries, {_size(total)}")
    if entries:
        now = time.time()
        ages = [(now - p.stat().st_mtime) / DAY for p in entries]
        console.print(f"Newest {min(ages):,.0f} days old, oldest {max(ages):,.0f} days old")
        for cutoff in (30, 90, 365):
            old = [p for p, a in zip(entries, ages) if a > cutoff]
            if old:
                freed = sum(p.stat().st_size for p in old)
                console.print(f"  older than {cutoff} days: {len(old):,} entries, {_size(freed)}")
    if temps:
        console.print(f"{len(temps)} unfinished write{'s' if len(temps) != 1 else ''} left by an interrupted fetch")


@app.command()
def prune(
    root: Path = _ROOT,
    older_than: float = typer.Option(..., "--older-than", min=0.0, help="Remove entries older than this many days."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Say what would be removed and remove nothing."),
) -> None:
    """Remove cached responses older than a cutoff; they are refetched when next needed."""
    root = Path(root).expanduser()
    if not root.is_dir():
        console.print(f"No cache at {escape(str(root))}; nothing to prune.")
        return
    cutoff = time.time() - older_than * DAY
    old = [p for p in _entries(root) if p.stat().st_mtime < cutoff]
    # An unfinished write is never valid, whatever its age.
    temps = list(root.glob("*.tmp"))
    freed = sum(p.stat().st_size for p in old + temps)
    if dry_run:
        console.print(f"Would remove {len(old):,} entries older than {older_than:g} days, {_size(freed)}.")
        return
    for path in old + temps:
        path.unlink(missing_ok=True)
    console.print(f"Removed {len(old):,} entries older than {older_than:g} days, {_size(freed)} freed.")
