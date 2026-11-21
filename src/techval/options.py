"""The command-line options more than one command module takes, each defined once.

cli.py imports the commands_* modules, so none of them can import an option
back from it. They import the shared ones from here, as they import what they
print with from terminal.py. An option that reads differently from one command
to the next, such as ``--as-of`` or ``--out``, stays with its command, and
takes ``iso_date`` from here when what it reads is a date, or ``one_of`` when it
reads one of a fixed set of words.
"""

from __future__ import annotations

from datetime import date

import typer

CONFIG = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file.")
NO_CACHE = typer.Option(False, "--no-cache", help="Bypass the HTTP cache.")
FROM_RECORD = typer.Option(
    False, "--from-record", help="Set the DCF from the company's own filings, as dcf.from_record does."
)
FACTS_DIR = typer.Option(
    None,
    "--facts-dir",
    help="A directory of recorded companyfacts_<TICKER>.json payloads, instead of live SEC calls.",
)
OPEN_PAGE = typer.Option(False, "--open", help="Open the page in the default browser once it is written.")


def iso_date(value: str | None) -> str | None:
    """A typed date, checked before the command runs and given in YYYY-MM-DD form.

    Each command reads its dates with ``date.fromisoformat`` inside a handler
    that catches only the package's own errors, so a date such as 2024-1-1
    raised a ValueError that nothing caught and the answer was a traceback.
    """
    if value is None:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise typer.BadParameter(f"{value!r} is not a YYYY-MM-DD date") from None


def iso_dates(value: str | None) -> str | None:
    """A comma-separated list of dates, each checked as ``iso_date`` checks one."""
    if value is None:
        return None
    return ",".join(iso_date(d.strip()) for d in value.split(",") if d.strip())


def one_of(*choices: str):
    """A callback taking one of ``choices``, in any case, and refusing the rest.

    A word outside the set was otherwise read as whichever choice the command
    falls back to, so a typo changed what ran without a word said about it.
    """

    def check(value: str | None) -> str | None:
        if value is None:
            return None
        if value.lower() not in choices:
            raise typer.BadParameter(
                f"{value!r} is not one of {', '.join(choices)}"
            )
        return value.lower()

    return check

