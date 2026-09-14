"""The value signal. A stub until this section's collector is built."""

from __future__ import annotations

ID = "signal"
TITLE = "The value signal"
INPUTS: list[str] = []


def collect(ctx) -> dict:
    return {"status": "not_built"}
