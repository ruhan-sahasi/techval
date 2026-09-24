"""The rendered page: every asset once, the snapshot readable, bytes stable."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from techval.invest.render import ASSETS, asset_names, render_page

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "docs" / "invest" / "snapshot.json"


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def page(snapshot) -> str:
    return render_page(snapshot)


def test_every_asset_is_inlined_once_in_order(page):
    names = re.findall(r'data-asset="([^"]+)"', page)
    scripts = [n for n in asset_names() if n.endswith(".js")]
    assert names == scripts
    for sheet in ("tokens.css", "layout.css", "app.css"):
        assert sheet in asset_names()


def test_the_snapshot_block_parses_back(page, snapshot):
    match = re.search(r'<script type="application/json" id="iv-snapshot">(.*?)</script>', page, flags=re.S)
    assert match
    assert json.loads(match.group(1)) == snapshot


def test_two_renders_are_byte_identical(snapshot):
    assert render_page(snapshot) == render_page(snapshot)


def test_the_title_names_the_portfolio(page, snapshot):
    assert f"<title>{snapshot['meta']['name']} · techval invest</title>" in page


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_invest_script_parses():
    for path in sorted(ASSETS.rglob("*.js")):
        result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"
