"""Reading what a section hands the kit: the charts the kit knows, every number
or string inside a figure's data, and a section as the committed snapshot holds it.

Shared by the section tests, which each carried their own copy.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "src" / "techval" / "dashboard" / "assets" / "kit.js"
SNAPSHOT = ROOT / "docs" / "dashboard" / "snapshot.json"


def kit_charts() -> set[str]:
    """The chart kinds kit.js registers in ``TV.charts``."""
    source = KIT.read_text(encoding="utf-8")
    body = re.search(r"TV\.charts = \{(.*?)\};", source, flags=re.S)
    assert body, "kit.js no longer assigns TV.charts"
    return set(re.findall(r"(\w+)\s*:", body.group(1)))


def numbers_at(value: Any, path: str = "data") -> Iterator[tuple[str, float]]:
    """Every number inside ``value``, with where it sits. A bool is not a number."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from numbers_at(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from numbers_at(v, f"{path}[{i}]")


def numbers(value: Any) -> Iterator[float]:
    """Every number inside ``value``."""
    for _, number in numbers_at(value):
        yield number


def strings(value: Any) -> Iterator[str]:
    """Every string inside ``value``, the keys of its dicts included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from strings(v)


def committed_section(section_id: str) -> dict[str, Any]:
    """A section as the committed snapshot holds it, collected without a refusal.

    Skips until a snapshot is committed. A snapshot that lacks the section fails
    rather than skips, since that is a page that lost one of its parts.
    """
    if not SNAPSHOT.is_file():
        pytest.skip("docs/dashboard/snapshot.json is not committed yet")
    section = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sections"][section_id]
    assert section["status"] == "ok", section["refusals"]
    return section
