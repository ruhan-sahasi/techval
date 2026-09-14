"""Data layer. A stub until this section's collector is built."""

from __future__ import annotations

ID = "datalayer"
TITLE = "Data layer"
INPUTS: list[str] = []


def collect(ctx) -> dict:
    return {"status": "not_built"}
