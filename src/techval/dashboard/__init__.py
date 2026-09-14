"""The results dashboard: one self-contained page rendered from a collected snapshot.

The page is committed at ``docs/dashboard/index.html`` and rendered from
``docs/dashboard/snapshot.json``, which is collected by running the package's
own entry points against the committed fixtures. The rule is the engine's
rule: refuse rather than invent. No figure on the page is typed in, every
figure carries the entry point and the input digests behind it, and a figure
that cannot be reproduced offline is shown as a refusal with its reason.
"""

from __future__ import annotations

from .snapshot import (
    SCHEMA_VERSION,
    Snapshot,
    dump_snapshot,
    fixture_digests,
    load_snapshot,
    to_jsonable,
)

__all__ = [
    "SCHEMA_VERSION",
    "Snapshot",
    "dump_snapshot",
    "fixture_digests",
    "load_snapshot",
    "to_jsonable",
]
