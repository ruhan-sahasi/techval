"""The committed demo page is exactly what the committed snapshot renders."""

from __future__ import annotations

import json
from pathlib import Path

from techval.invest.render import render_page

ROOT = Path(__file__).resolve().parents[2]


def test_the_demo_page_rebuilds_byte_identical():
    snapshot = json.loads((ROOT / "docs" / "invest" / "snapshot.json").read_text(encoding="utf-8"))
    committed = (ROOT / "docs" / "invest" / "index.html").read_text(encoding="utf-8")
    assert render_page(snapshot) == committed
