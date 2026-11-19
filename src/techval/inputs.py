"""Reading the files a command is handed, or refusing them by name.

Every command catches ``TechvalError`` and prints the reason, and nothing else.
So what Python raises on a file it cannot use, an ``OSError``, a parser's error
or a ``ValueError`` from a malformed cell, reached the user as a traceback. Each
reader here turns those into a ``ConfigError`` that names the file and says
what is wrong with it.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import yaml

from .errors import ConfigError


def open_text(path: Path) -> io.TextIOBase:
    """Read a ``.gz`` the same way as a plain file.

    Every recorded panel in this repository is gzipped because the uncompressed
    ones run to tens of megabytes, and a user who unzips one to look at it should
    not then have to zip it again to feed it back in. ``utf-8-sig`` reads plain
    UTF-8 and also drops the byte-order mark a spreadsheet's "CSV UTF-8" export
    starts with, which plain UTF-8 keeps as part of the first column's name.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def read_yaml(path: Path) -> object:
    """A YAML file's contents, or a ConfigError naming the file and the reason.

    A file the parser rejects raised ``yaml.YAMLError``, and an impossible date
    such as ``as_of: 2024-13-01`` the ValueError from ``datetime.date``, and no
    command catches either, so a stray indent ended in a traceback. The file is
    handed over as bytes, so the parser settles the encoding, as YAML specifies,
    and the line it points at carries the file's name. A path that cannot be
    opened, a directory or a file without read permission, is refused too.
    """
    try:
        with Path(path).open("rb") as handle:
            return yaml.safe_load(handle)
    except OSError as exc:
        reason = exc.strerror or exc
        raise ConfigError(f"{path} could not be opened: {reason}.") from exc
    except (yaml.YAMLError, ValueError) as exc:
        raise ConfigError(f"{path} could not be read as YAML. {exc}") from exc


def read_field(path: Path, where: str, field: str, text: object, parse, what: str):
    """One field of a file a command was handed, read with ``parse``, or a refusal.

    A spreadsheet writes #N/A for a gap and 3/31/2024 for a date, and either
    raised a ValueError that no command catches. The ConfigError names the
    file, ``where`` in it (a line of a CSV, an entry of a JSON list), the field
    and the value, so the one bad cell can be found.
    """
    try:
        return parse(text)
    except (TypeError, ValueError):
        raise ConfigError(
            f"{path} {where}: {field} is {text!r}, which is not {what}."
        ) from None
