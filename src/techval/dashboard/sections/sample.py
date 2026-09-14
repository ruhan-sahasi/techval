"""What the sample can support. A stub until this section's collector is built."""

from __future__ import annotations

ID = "sample"
TITLE = "What the sample can support"
INPUTS: list[str] = []


def collect(ctx) -> dict:
    return {"status": "not_built"}
