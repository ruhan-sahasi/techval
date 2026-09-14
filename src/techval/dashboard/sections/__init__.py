"""The dashboard's sections: their page order, their collection order, and where each lives.

Each section is one module, ``sections/<id>.py``, exposing four names and
nothing else the collector reads:

    ID       the section id, equal to the module name
    TITLE    the heading on the page
    INPUTS   fixture paths the collector reads, relative to the ml-data root
    collect  ``collect(ctx) -> dict``, run with a ``CollectContext``

A module is replaced wholesale when its section is built, so the registry here
knows ids and nothing about any section's contents. ``INPUTS`` is not
documentation: it is what the snapshot digests and what the result cache is
keyed on, and ``ctx.record`` refuses a figure that cites a file not declared in
it, since an undeclared input is one whose change would not invalidate the
cached figure built from it.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from ...errors import ConfigError

# Page order.
SECTION_IDS: tuple[str, ...] = (
    "overview",
    "signal",
    "encoder",
    "warranted",
    "fade",
    "propensity",
    "engine",
    "tmt",
    "datalayer",
    "sample",
)

# The five fitted models. Only these carry a headline score against a baseline.
MODEL_SECTIONS: tuple[str, ...] = ("signal", "encoder", "warranted", "fade", "propensity")

# Sections that read other sections' results rather than fixtures, in the order
# they must run: the sample section reads the model sections, and the scoreboard
# reads everything including the sample section.
DEPENDENT_SECTIONS: tuple[str, ...] = ("sample", "overview")

COLLECT_ORDER: tuple[str, ...] = (
    tuple(s for s in SECTION_IDS if s not in DEPENDENT_SECTIONS) + DEPENDENT_SECTIONS
)


def section_module(section_id: str) -> ModuleType:
    """The module for one section id, checked for the four names the collector reads.

    An unknown id is a ``ConfigError``, because it arrives from a user typing
    ``--sections``. A module that exists and is malformed is a ``TypeError``,
    because that is a bug in the section and must not be reported as a refusal.
    """
    if section_id not in SECTION_IDS:
        raise ConfigError(
            f"unknown dashboard section {section_id!r}; the sections are "
            + ", ".join(SECTION_IDS)
        )
    module = importlib.import_module(f"{__name__}.{section_id}")
    problems = []
    if getattr(module, "ID", None) != section_id:
        problems.append(f"ID is {getattr(module, 'ID', None)!r}")
    if not isinstance(getattr(module, "TITLE", None), str) or not module.TITLE.strip():
        problems.append("TITLE is not a non-empty string")
    inputs = getattr(module, "INPUTS", None)
    if not isinstance(inputs, (list, tuple)) or not all(isinstance(p, str) for p in inputs):
        problems.append("INPUTS is not a list of path strings")
    if not callable(getattr(module, "collect", None)):
        problems.append("collect is not callable")
    if problems:
        raise TypeError(f"section module {module.__name__}: " + "; ".join(problems))
    return module
