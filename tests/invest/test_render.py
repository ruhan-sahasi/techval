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


def test_the_page_carries_the_house_favicon(page):
    from techval.dashboard.render import FAVICON_HREF

    assert f'<link rel="icon" href="{FAVICON_HREF}">' in page


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_invest_script_parses():
    for path in sorted(ASSETS.rglob("*.js")):
        result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"


INVEST_JS = ASSETS / "invest.js"
KIT_JS = ASSETS.parent.parent / "dashboard" / "assets" / "kit.js"


def _run_invest(expr: str):
    import json as _json

    script = (
        "const vm = require('vm'), fs = require('fs');"
        "const stub = { setAttribute() {}, appendChild() {}, style: {},"
        " getContext: () => ({ measureText: () => ({ width: 5 }) }) };"
        "const document = { readyState: 'loading', createElementNS: () => stub, createElement: () => stub,"
        " body: stub, addEventListener() {}, documentElement: stub };"
        "const window = { location: { search: '' }, matchMedia: () => ({ matches: false }) };"
        "const ctx = { window, document, console };"
        "vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), ctx);"
        "vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);"
        f"process.stdout.write(JSON.stringify({expr}));"
    )
    out = subprocess.run(["node", "-e", script, str(KIT_JS), str(INVEST_JS)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return _json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_range_starts_are_calendar_offsets_from_the_last_date():
    dates = ["2025-01-02", "2025-06-30", "2025-12-31", "2026-03-31", "2026-08-31", "2026-09-09"]
    got = _run_invest(
        "['1M','3M','YTD','1Y','ALL'].map(k => window.IV.range.startIndex(" + json.dumps(dates) + ", k))"
    )
    assert got == [4, 4, 3, 2, 0]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_range_rebases_both_lines_to_one_at_its_start():
    got = _run_invest("window.IV.range.rebase([1.0, 1.2, 1.5, 1.8], 1)")
    assert got == pytest.approx([1.0, 1.25, 1.5])


def _run_with_storage(storage_js: str, expr: str):
    import json as _json

    script = (
        "const vm = require('vm'), fs = require('fs');"
        "const stub = { setAttribute() {}, getAttribute() { return null; }, appendChild() {}, style: {},"
        " getContext: () => ({ measureText: () => ({ width: 5 }) }) };"
        "const document = { readyState: 'loading', createElementNS: () => stub, createElement: () => stub,"
        " body: stub, addEventListener() {}, documentElement: stub };"
        f"const window = {{ location: {{ search: '' }}, matchMedia: () => ({{ matches: false }}), {storage_js} }};"
        "const ctx = { window, document, console };"
        "vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), ctx);"
        "vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);"
        f"process.stdout.write(JSON.stringify({expr}));"
    )
    out = subprocess.run(["node", "-e", script, str(KIT_JS), str(INVEST_JS)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return _json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_preferences_round_trip_through_local_storage():
    store = "localStorage: (() => { const m = {}; return { getItem: k => (k in m ? m[k] : null), setItem: (k, v) => { m[k] = String(v); } }; })()"
    got = _run_with_storage(store, "(window.IV.pref.set('range', '1Y'), [window.IV.pref.get('range'), window.IV.pref.get('nothing', 'ALL')])")
    assert got == ["1Y", "ALL"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_storage_that_throws_falls_back_to_defaults():
    throwing = "get localStorage() { throw new Error('denied'); }"
    got = _run_with_storage(throwing, "(window.IV.pref.set('range', '1Y'), window.IV.pref.get('range', 'ALL'))")
    assert got == "ALL"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_number_keys_and_brackets_pick_a_pane():
    got = _run_invest(
        "[['1','engine'],['7','overview'],['8','overview'],[']','overview'],[']','activity'],['[','overview'],['x','overview']]"
        ".map(p => window.IV.keyTarget(p[0], p[1]))"
    )
    assert got == ["overview", "activity", None, "holdings", "overview", "activity", None]
