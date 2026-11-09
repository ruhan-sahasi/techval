"""The command-line options more than one command module takes, each defined once.

cli.py imports the commands_* modules, so none of them can import an option
back from it. They import the shared ones from here, as they import what they
print with from terminal.py. An option that reads differently from one command
to the next, such as ``--as-of`` or ``--out``, stays with its command.
"""

from __future__ import annotations

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
