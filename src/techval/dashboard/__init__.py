"""The results dashboard: one self-contained page rendered from a collected snapshot.

The page is committed at ``docs/dashboard/index.html`` and rendered from
``docs/dashboard/snapshot.json``, which is collected by running the package's
own entry points against the committed fixtures. The rule is the engine's
rule: refuse rather than invent. No figure on the page is typed in, every
figure carries the entry point and the input digests behind it, and a figure
that cannot be reproduced offline is shown as a refusal with its reason.
"""

from __future__ import annotations

from .collect import CollectContext, SectionRun, collect_snapshot
from .render import render_dashboard, write_dashboard
from .sections import COLLECT_ORDER, SECTION_IDS, section_module
from .snapshot import (
    SCHEMA_VERSION,
    Snapshot,
    dump_snapshot,
    fixture_digests,
    load_snapshot,
    to_jsonable,
)

__all__ = [
    "COLLECT_ORDER",
    "CollectContext",
    "SCHEMA_VERSION",
    "SECTION_IDS",
    "SectionRun",
    "Snapshot",
    "collect_snapshot",
    "dump_snapshot",
    "fixture_digests",
    "load_snapshot",
    "render_dashboard",
    "section_module",
    "to_jsonable",
    "write_dashboard",
]
