"""Tests for the dashboard front end: the stylesheets, the chart kit and the gallery.

Most of these read the asset files as text and hold them to the rules a browser
cannot enforce for us: the theme structure that keeps a colour from existing in
only one mode, the ban on markup built from strings, the ban on dashed strokes,
and the ban on em-dashes in anything a reader sees.

The rest draw the gallery in headless Chrome and measure what landed on the
page: that a direct label names the value of the dot beside it, that a reference
label touches no mark, that coincident points are drawn apart, that the rail
marks the section being read. They are skipped where Chrome is not installed.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "src" / "techval" / "dashboard" / "assets"
KIT = ASSETS / "kit.js"

SECTION_IDS = (
    "overview",
    "signal",
    "encoder",
    "warranted",
    "fade",
    "propensity",
    "engine",
    "tmt",
    "reading",
    "datalayer",
    "sample",
)

DARK_MEDIA = "@media (prefers-color-scheme: dark)"
DARK_AUTO = ':root:not([data-theme="light"])'
DARK_FORCED = ':root[data-theme="dark"]'
PRINT_MEDIA = "@media print"


def _asset_files() -> list[Path]:
    return sorted(p for p in ASSETS.rglob("*") if p.is_file())


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _blocks(css: str) -> list[tuple[str, str]]:
    """Split CSS into its top-level ``(prelude, body)`` pairs, braces matched."""
    out: list[tuple[str, str]] = []
    depth = 0
    start = 0
    prelude = ""
    for i, ch in enumerate(css):
        if ch == "{":
            if depth == 0:
                prelude = css[start:i].strip()
                start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append((prelude, css[start:i]))
                start = i + 1
    assert depth == 0, "unbalanced braces"
    return out


def _defined(body: str) -> set[str]:
    return set(re.findall(r"(--[\w-]+)\s*:", body))


def _referenced(body: str) -> set[str]:
    return set(re.findall(r"var\(\s*(--[\w-]+)", body))


@pytest.fixture(scope="module")
def token_blocks() -> dict[str, str]:
    css = _strip_comments((ASSETS / "tokens.css").read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for prelude, body in _blocks(css):
        if prelude == DARK_MEDIA:
            inner = _blocks(body)
            assert [p for p, _ in inner] == [DARK_AUTO], "the dark media block holds one rule"
            found["auto"] = inner[0][1]
        elif prelude == PRINT_MEDIA:
            inner = _blocks(body)
            # Named for the forced dark theme as well, or a page printed while
            # stamped dark would take the dark tokens over these.
            preludes = [" ".join(p.split()) for p, _ in inner]
            assert preludes == [f":root, {DARK_FORCED}"], "the print media block holds one rule"
            found["print"] = inner[0][1]
        elif prelude in (":root", DARK_FORCED):
            key = "root" if prelude == ":root" else "forced"
            assert key not in found, f"{prelude} is defined twice"
            found[key] = body
        else:
            pytest.fail(f"tokens.css holds a rule other than the theme blocks: {prelude}")
    assert set(found) == {"root", "auto", "forced", "print"}
    return found


# Stylesheets ---------------------------------------------------------------------


def test_every_dark_token_is_defined_on_bare_root(token_blocks):
    root = _defined(token_blocks["root"])
    for key in ("auto", "forced", "print"):
        used = _defined(token_blocks[key]) | _referenced(token_blocks[key])
        missing = sorted(used - root)
        assert not missing, f"dark block ({key}) uses tokens with no light definition: {missing}"


def test_the_two_dark_blocks_define_the_same_tokens_with_the_same_values(token_blocks):
    def pairs(body: str) -> dict[str, str]:
        return {
            k.strip(): v.strip()
            for k, v in re.findall(r"([\w-]+)\s*:\s*([^;]+);", body)
        }

    assert pairs(token_blocks["auto"]) == pairs(token_blocks["forced"])


def test_dark_blocks_hold_tokens_only(token_blocks):
    for key in ("auto", "forced", "print"):
        names = re.findall(r"([\w-]+)\s*:", token_blocks[key])
        stray = [n for n in names if not n.startswith("--") and n != "color-scheme"]
        assert not stray, f"dark block ({key}) styles properties directly: {stray}"


def test_every_token_referenced_on_root_is_defined_there(token_blocks):
    root = token_blocks["root"]
    assert _referenced(root) <= _defined(root)


def test_layout_uses_tokens_rather_than_colours_or_theme_blocks():
    css = _strip_comments((ASSETS / "layout.css").read_text(encoding="utf-8"))
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css), "layout.css names a hex colour"
    assert not re.search(r"\b(rgb|rgba|hsl|hsla|oklch)\(", css), "layout.css names a colour"
    assert "prefers-color-scheme" not in css
    assert "data-theme" not in css
    tokens = _defined(_strip_comments((ASSETS / "tokens.css").read_text(encoding="utf-8")))
    unknown = sorted(_referenced(css) - tokens - {"--key", "--chip-hue", "--chip-on"})
    assert not unknown, f"layout.css reads tokens nobody defines: {unknown}"


def test_body_takes_its_background_from_a_token():
    css = _strip_comments((ASSETS / "layout.css").read_text(encoding="utf-8"))
    bodies = [body for prelude, body in _blocks(css) if prelude == "body"]
    assert len(bodies) == 1
    assert re.search(r"background(-color)?\s*:\s*var\(--page\)", bodies[0])
    assert re.search(r"(^|;|\s)color\s*:\s*var\(--ink-1\)", bodies[0])


def test_fonts_carry_real_fallback_stacks(token_blocks):
    root = token_blocks["root"]
    assert '"Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif' in root
    assert '"Newsreader", "Source Serif 4", Georgia, "Times New Roman", serif' in root
    assert '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace' in root


# Every asset ---------------------------------------------------------------------


def test_no_asset_contains_an_em_dash():
    offenders = [
        str(p.relative_to(ROOT))
        for p in _asset_files()
        if "\u2014" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"em-dash in {offenders}"


def test_no_dashed_strokes_anywhere_in_the_assets():
    for path in _asset_files():
        text = path.read_text(encoding="utf-8")
        assert "stroke-dasharray" not in text, path.name
        assert "strokeDasharray" not in text, path.name
        assert not re.search(r"border[\w-]*\s*:[^;]*\bdashed\b", text), path.name
        assert not re.search(r"outline[\w-]*\s*:[^;]*\bdashed\b", text), path.name


def _scripts() -> list[Path]:
    return [p for p in _asset_files() if p.suffix == ".js"]


def test_no_script_builds_markup_from_strings():
    # Series names, tickers and refusal reasons are data. They enter the DOM as
    # text nodes, so no script may hand a string to the HTML parser.
    banned = re.compile(r"\.(innerHTML|outerHTML)\s*\+?=|insertAdjacentHTML|document\.write|createContextualFragment")
    offenders = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in _scripts()
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if banned.search(line)
    ]
    assert not offenders, f"markup built from a string at {offenders}"


def test_no_asset_would_close_its_inlined_element_early():
    # The page inlines every asset; a literal closing tag inside one would end
    # its <script> or <style> element at that point.
    for path in _asset_files():
        text = path.read_text(encoding="utf-8").lower()
        assert "</script" not in text, path.name
        assert "</style" not in text, path.name
        assert "<!--" not in text, path.name


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_script_parses():
    for path in _scripts():
        result = subprocess.run(
            ["node", "--check", str(path)], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"{path.name}: {result.stderr}"


# Chart kit -----------------------------------------------------------------------


def _object_keys(source: str, name: str) -> set[str]:
    """The keys of the object literal assigned to ``name`` in ``source``."""
    start = source.index(f"{name} = {{")
    depth = 0
    for i in range(source.index("{", start), len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                body = source[source.index("{", start) + 1 : i]
                break
    # Top-level keys only: blank out nested braces before reading them.
    flat = body
    while re.search(r"\{[^{}]*\}", flat):
        flat = re.sub(r"\{[^{}]*\}", "", flat)
    return set(re.findall(r"(?:^|,)\s*(\w+)\s*:", flat, flags=re.M))


@pytest.fixture(scope="module")
def kit_source() -> str:
    return KIT.read_text(encoding="utf-8")


def test_kit_exposes_the_contract_api(kit_source):
    for name in (
        "el", "svg", "fmt", "token", "scale", "chip", "figure",
        "legend", "tooltip", "tableView", "charts", "sections",
    ):
        assert re.search(rf"\bTV\.{name}\s*=", kit_source), f"TV.{name} is not defined"
    assert {"num", "pct", "signed", "compact", "mm"} <= _object_keys(kit_source, "TV.fmt")
    assert {"linear", "band"} <= _object_keys(kit_source, "TV.scale")
    assert {"register", "render"} <= _object_keys(kit_source, "TV.sections")
    assert {"attach"} <= _object_keys(kit_source, "var tooltip")


def test_kit_has_every_chart_the_contract_names(kit_source):
    assert _object_keys(kit_source, "TV.charts") == {
        "hbar", "column", "dot", "line", "heat", "hist", "range", "waterfall", "table", "tiles",
    }


def test_kit_exports_the_helpers_the_sections_used_to_copy(kit_source):
    # A section that needs a resize-aware container, axis ticks or a text measure
    # reaches for the kit's, so no section carries its own copy.
    for name in (
        "frame", "axis", "tickFormat", "measure", "wrapText", "thinLabels",
        "markGroup", "chartSvg", "crisp", "cautionNote", "refusalNote",
    ):
        assert re.search(rf"\bTV\.{name}\s*=", kit_source), f"TV.{name} is not exported"
    assert {"x", "y", "yWidth"} <= _object_keys(kit_source, "TV.axis")


def test_kit_header_documents_the_options_it_reads(kit_source):
    header = kit_source[: kit_source.index("(function (global)")]
    for option in (
        "legend", "table", "tableHeaders", "reference", "lo and hi", "labels", "dodge",
        "aside", "yScale", "yTitle", "shade", "points", "breaks", "mono", "groups",
        "colTitle", "order", "card", "part", "note or notes", "refusals", "parts",
    ):
        assert option in header, f"the kit header does not document {option}"


def test_chips_cover_every_verdict_status(kit_source):
    assert _object_keys(kit_source, "var STATUS") == {
        "beats", "inside_noise", "ties", "loses", "not_significant", "refused",
    }


def test_every_chart_builds_a_table_view(kit_source):
    for chart in ("hbar", "column", "dot", "line", "heat", "hist", "range", "waterfall"):
        match = re.search(rf"\n  function {chart}\(body, spec\) \{{(.*?)\n  \}}\n", kit_source, flags=re.S)
        assert match, f"chart {chart} not found"
        assert re.search(r"tableFor\(\s*body", match.group(1)), f"{chart} has no table view"


def test_kit_marks_follow_the_mark_specs(kit_source):
    assert re.search(r"var BAR_MAX = 24;", kit_source)
    assert re.search(r"var RADIUS = 4;", kit_source)
    assert re.search(r"var GAP = 2;", kit_source)
    assert re.search(r"var HIT_MIN = 24;", kit_source)
    # The kit paints through tokens; a hex colour in it would not follow the theme.
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", re.sub(r"/\*.*?\*/", "", kit_source, flags=re.S))
    layout = _strip_comments((ASSETS / "layout.css").read_text(encoding="utf-8"))
    line = next(body for prelude, body in _blocks(layout) if prelude == ".tv-line")
    assert "stroke-width: 2" in line and "stroke-linejoin: round" in line and "stroke-linecap: round" in line
    dot = next(body for prelude, body in _blocks(layout) if prelude == ".tv-dot")
    assert "stroke: var(--surface)" in dot and "stroke-width: 2" in dot


def test_the_note_prints_as_a_note():
    # Screen furniture goes, a section starts a page, and nothing a reader is
    # meant to read as one piece is split across a break.
    layout = _strip_comments((ASSETS / "layout.css").read_text(encoding="utf-8"))
    block = next(body for prelude, body in _blocks(layout) if prelude == "@media print")
    rules = {" ".join(prelude.split()): body for prelude, body in _blocks(block)}
    hidden = next(body for prelude, body in rules.items() if ".tv-rail" in prelude and ".tv-btn" in prelude)
    assert "display: none" in hidden
    assert "break-before: page" in rules[".tv-section"]
    assert "break-before: auto" in rules[".tv-section:first-of-type"]
    whole = next(body for prelude, body in rules.items() if ".tv-figure," in prelude and ".tv-tileset," in prelude)
    assert "break-inside: avoid" in whole
    assert "overflow: visible" in rules[".tv-table-wrap"]
    assert "max-width: 100%" in rules[".tv-svg"] and "height: auto" in rules[".tv-svg"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_signed_cells_are_classed_by_sign_and_zero_stays_plain():
    script = (
        "const vm = require('vm'), fs = require('fs');"
        "const stub = { setAttribute() {}, appendChild() {}, style: {},"
        " getContext: () => ({ measureText: () => ({ width: 5 }) }) };"
        "const document = { createElementNS: () => stub, createElement: () => stub,"
        " body: stub, addEventListener() {}, documentElement: {} };"
        "const window = {};"
        "vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), { window, document, console });"
        "const s = window.TV.signedClass;"
        "process.stdout.write(JSON.stringify([s(3), s(-0.5), s(0), s(null), s('x'), s(NaN)]));"
    )
    out = subprocess.run(["node", "-e", script, str(KIT)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == ["tv-pos", "tv-neg", None, None, None, None]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_figures_marks_are_one_tab_stop_and_the_arrows_walk_them():
    # A stop on every mark put 75 presses of Tab inside the coefficient chart alone.
    script = r"""
const vm = require('vm'), fs = require('fs');
const listeners = {};
const body = {
  querySelectorAll: () => marks,
  addEventListener: (t, fn) => { (listeners[t] = listeners[t] || []).push(fn); },
  fire: (t, target, key) => {
    let prevented = false;
    (listeners[t] || []).forEach((fn) => fn({ target, key, preventDefault: () => { prevented = true; } }));
    return prevented;
  },
};
function mark() {
  const attrs = {};
  const m = {
    classList: { contains: (c) => c === 'tv-markg' },
    getAttribute: (k) => (k in attrs ? attrs[k] : null),
    setAttribute: (k, v) => { attrs[k] = String(v); },
    focus: () => body.fire('focusin', m),
  };
  return m;
}
const marks = [mark(), mark(), mark(), mark()];
const stub = { setAttribute() {}, appendChild() {}, style: {}, getContext: () => ({ measureText: () => ({ width: 5 }) }) };
const document = { createElementNS: () => stub, createElement: () => stub, body: stub, addEventListener() {}, documentElement: {} };
const window = {};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), { window, document, console });
const settle = window.TV.roveMarks(body);
const stops = () => marks.map((m) => (m.getAttribute('tabindex') === '0' ? 'x' : '.')).join('');
settle();
const out = [stops()];
for (const [i, key] of [[0, 'ArrowRight'], [1, 'End'], [3, 'ArrowDown'], [3, 'Home'], [0, 'ArrowUp']]) {
  body.fire('keydown', marks[i], key);
  out.push(stops());
}
out.push(body.fire('keydown', marks[0], 'Tab'));
process.stdout.write(JSON.stringify(out));
"""
    out = subprocess.run(["node", "-e", script, str(KIT)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == ["x...", ".x..", "...x", "...x", "x...", "x...", False]


def test_reduced_motion_and_focus_are_respected():
    layout = _strip_comments((ASSETS / "layout.css").read_text(encoding="utf-8"))
    assert "@media (prefers-reduced-motion: no-preference)" in layout
    assert re.search(r":focus-visible\s*\{[^}]*outline:\s*2px solid var\(--focus\)", layout)


# App shell and section renderers ---------------------------------------------------


APP = ASSETS / "app.js"


def test_every_section_has_a_renderer_that_registers_its_own_id():
    sections = ASSETS / "sections"
    assert sorted(p.stem for p in sections.glob("*.js")) == sorted(SECTION_IDS)
    for sid in SECTION_IDS:
        text = (sections / f"{sid}.js").read_text(encoding="utf-8")
        registered = re.findall(r"TV\.sections\.register\(\s*\"([\w-]+)\"", text)
        assert registered == [sid], f"sections/{sid}.js registers {registered}"


def test_no_section_renderer_carries_a_copy_of_the_kit():
    # The kit exports its frame, axes and measures; a section that copies them
    # drifts from the kit the first time either changes.
    copies = {
        "a chart frame": re.compile(r"function frame\s*\(|new ResizeObserver"),
        "a kind fallback": re.compile(r"has no chart named"),
        "a text measure": re.compile(r"measureText\s*\("),
        "a local table": re.compile(r'el\(\s*"table"'),
    }
    for sid in SECTION_IDS:
        text = (ASSETS / "sections" / f"{sid}.js").read_text(encoding="utf-8")
        for what, pattern in copies.items():
            assert not pattern.search(text), f"sections/{sid}.js carries {what}"


def test_app_stamps_the_theme_from_the_query_string():
    text = APP.read_text(encoding="utf-8")
    assert re.search(r"theme=\(dark\|light\)", text)
    assert 'setAttribute("data-theme"' in text
    # Stamped when the script runs, not at boot, so the page never paints in the wrong theme.
    assert re.search(r"\n  stampTheme\(\);\n", text)
    # With no query the page opens dark unless the system prefers light.
    assert "prefers-color-scheme: light" in text
    assert 'lighter ? "light" : "dark"' in text


def test_app_waits_for_the_section_scripts_before_drawing():
    # The page inlines app.js before the section scripts, so boot must wait for the document.
    text = APP.read_text(encoding="utf-8")
    assert 'addEventListener("DOMContentLoaded", boot)' in text
    assert "TV.sections.order()" in text


def test_app_reads_the_snapshot_block_and_draws_every_state():
    text = APP.read_text(encoding="utf-8")
    assert 'SNAPSHOT_ID = "tv-snapshot"' in text
    for status in ("not_built", "refused", "ok"):
        assert re.search(rf'status [!=]== "{status}"', text), status
    # A snapshot the page cannot read is a refusal, not a blank page.
    assert "readSnapshot" in text and "read.error" in text


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_verdict_opening_with_an_initialism_shows_it_in_capitals():
    # A model's verdict() opens with its metric in lower case. Capitalising only
    # the first letter printed "Mae of 0.1474", "Auc of 0.5685", "Ndcg@10 of 0.5407".
    cases = {
        "mae of 0.1474 against 0.1510": "MAE of 0.1474 against 0.1510",
        "auc of 0.5685 against 0.5474": "AUC of 0.5685 against 0.5474",
        "ndcg@10 of 0.5407 against 0.2213": "NDCG@10 of 0.5407 against 0.2213",
        "ndcg@k of 0.5": "NDCG@k of 0.5",
        "ic of 0.0412, rmse of 3.1": "IC of 0.0412, rmse of 3.1",
        "rmse of 3.2 on 2,188 observations.": "RMSE of 3.2 on 2,188 observations.",
        "Mae of 0.1": "MAE of 0.1",
        # Anything else has its first letter capitalised and nothing more.
        "cheapness (negative trailing EV/Revenue): mean IC -0.0984": "Cheapness (negative trailing EV/Revenue): mean IC -0.0984",
        "icarus of 0.1": "Icarus of 0.1",
        "maestro": "Maestro",
        "log MAE of 0.4120": "Log MAE of 0.4120",
        "spearman of 0.77": "Spearman of 0.77",
        "2% of the variance": "2% of the variance",
        "": "",
    }
    script = (
        "const vm = require('vm');"
        "const fs = require('fs');"
        "const document = {readyState: 'loading', addEventListener: function () {},"
        " documentElement: {setAttribute: function () {}}};"
        "const window = {TV: {el: function () {}}, location: {search: ''}};"
        "vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {window: window, document: document});"
        "const cases = JSON.parse(process.argv[2]);"
        "process.stdout.write(JSON.stringify(cases.map(function (c) { return window.TV.app.verdictCase(c); })));"
    )
    out = subprocess.run(
        ["node", "-e", script, str(APP), json.dumps(list(cases))],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert dict(zip(cases, json.loads(out.stdout))) == cases


# Gallery -------------------------------------------------------------------------


GALLERY = ROOT / "src" / "techval" / "dashboard" / "gallery.py"


@pytest.fixture(scope="module")
def gallery():
    # Loaded by path: the gallery must work without the rest of the dashboard package.
    import importlib.util

    spec = importlib.util.spec_from_file_location("techval_dashboard_gallery", GALLERY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gallery_page(gallery) -> str:
    return gallery.render_gallery()


def _snapshot_block(page: str):
    import json

    match = re.search(r'<script type="application/json" id="tv-snapshot">(.*?)</script>', page, flags=re.S)
    assert match, "the gallery has no snapshot block"
    return json.loads(match.group(1))


def test_gallery_imports_nothing_from_the_package():
    source = GALLERY.read_text(encoding="utf-8")
    assert not re.search(r"^\s*from\s+\.|^\s*(from|import)\s+techval", source, flags=re.M)


def test_gallery_follows_the_page_order(gallery):
    assert tuple(gallery.PAGE_ORDER) == SECTION_IDS


def test_gallery_inlines_every_asset_in_contract_order(gallery, gallery_page):
    names = gallery.asset_names()
    assert names == ["tokens.css", "layout.css", "kit.js", "app.js", *(f"sections/{s}.js" for s in SECTION_IDS)]
    style = re.search(r"<style>\n(.*?)</style>", gallery_page, flags=re.S).group(1)
    tokens = (ASSETS / "tokens.css").read_text(encoding="utf-8").rstrip("\n")
    layout = (ASSETS / "layout.css").read_text(encoding="utf-8").rstrip("\n")
    assert style == tokens + "\n" + layout + "\n"
    inlined = re.findall(r'<script data-asset="([^"]+)">\n(.*?)</script>', gallery_page, flags=re.S)
    assert [name for name, _ in inlined] == names[2:]
    for name, text in inlined:
        assert text == (ASSETS / name).read_text(encoding="utf-8").rstrip("\n") + "\n", name
    assert gallery_page.index('id="tv-snapshot"') < gallery_page.index('data-asset="kit.js"')


def test_the_gallery_shell_matches_the_page_shell(gallery):
    # The gallery copies the embedding contract so it can be rendered without
    # the package. The copy is only worth having while it agrees with the page
    # it stands in for.
    from techval.dashboard.render import FONTS_HREF, SCRIPTS, STYLESHEETS

    assert gallery.STYLESHEETS == STYLESHEETS
    assert gallery.SCRIPTS == SCRIPTS
    assert gallery.FONTS_HREF == FONTS_HREF


def test_gallery_is_deterministic_and_self_contained(gallery, gallery_page):
    assert gallery.render_gallery() == gallery_page
    # Nothing is loaded by src; the only links are the Plex stylesheet and its font host.
    assert not re.search(r"\ssrc=", gallery_page)
    external = re.findall(r'href="(https?:[^"]+)"', gallery_page)
    assert external and all(
        url.startswith(("https://fonts.googleapis.com", "https://fonts.gstatic.com")) for url in external
    )
    assert "Public+Sans:wght@400;500;650" in gallery_page
    assert "IBM+Plex+Mono:wght@400;500" in gallery_page
    assert "\u2014" not in GALLERY.read_text(encoding="utf-8")


def test_gallery_snapshot_escapes_closing_tags(gallery):
    assert gallery._embed_json({"why": "a </script> and <!-- in a filing"}) == (
        '{"why":"a <\\/script> and \\u003c!-- in a filing"}'
    )


def test_gallery_exercises_every_chart_every_chip_and_every_state(gallery_page, kit_source):
    snapshot = _snapshot_block(gallery_page)
    assert snapshot["schema"] == 1
    sections = list(snapshot["sections"].values())
    kinds = {f["kind"] for s in sections for f in s["figures"].values()}
    assert kinds == _object_keys(kit_source, "TV.charts")

    statuses = {s["headline"]["verdict_status"] for s in sections if s["headline"]}
    for s in sections:
        for f in s["figures"].values():
            if f["kind"] == "tiles":
                statuses |= {t["status"] for t in f["data"]["tiles"] if t.get("status")}
    # A refused section draws the refused chip, in its refusal card and on the rail.
    statuses |= {"refused" for s in sections if s["status"] == "refused"}
    assert statuses == _object_keys(kit_source, "var STATUS")

    assert {s["status"] for s in sections} == {"ok", "refused", "not_built"}
    # A refusal inside an ok section that names one of its figures.
    assert any(r["what"] in s["figures"] for s in sections if s["status"] == "ok" for r in s["refusals"])
    # The gallery never passes itself off as results.
    assert snapshot["fixtures"] == {}
    assert all(
        p["entry_point"].startswith("techval.dashboard.gallery.")
        for s in sections
        for p in s["provenance"]
    )
    assert 'id="tv-notice"' in gallery_page and "Synthetic data" in gallery_page


def _figures(snapshot):
    return {fid: f for s in snapshot["sections"].values() for fid, f in s["figures"].items()}


def test_gallery_exercises_what_the_kit_absorbed_from_the_sections(gallery_page):
    snapshot = _snapshot_block(gallery_page)
    figures = _figures(snapshot)
    of_kind = lambda kind: [f for f in figures.values() if f["kind"] == kind]  # noqa: E731
    rows = lambda kind: [r for f in of_kind(kind) for r in f["data"].get("rows", [])]  # noqa: E731

    assert any(isinstance(f.get("order"), (int, float)) for f in figures.values()), "a figure order"
    assert any("lo" in r and "hi" in r for r in rows("dot")), "whiskers on a dot"
    assert any("lo" in r and "hi" in r for r in rows("hbar")), "whiskers on a bar"
    assert any(r.get("group") for r in rows("dot")) and any(r.get("text") for r in rows("dot"))
    assert any(isinstance(f["data"].get("labels"), dict) for f in of_kind("dot")), "explicit dot labels"
    assert any(
        len(set(r["values"].values())) == 1 and len(r["values"]) > 1 for r in rows("dot")
    ), "a row whose points coincide"
    assert any(
        all(r["values"][f["data"]["series"][0]["key"]] == 0 for r in f["data"]["rows"]) for f in of_kind("dot")
    ), "a dot chart with nothing that stands out"
    assert any(f["data"].get("legend") is False for f in figures.values()), "a legend turned off"
    assert any("tableHeaders" in f["data"] for f in figures.values())
    assert any(isinstance(f["data"].get("table"), dict) for f in figures.values()), "a fuller table"
    assert any(isinstance(f["data"].get("table"), list) for f in figures.values()), "two fuller tables"
    assert any(all(s["value"] >= 0 for s in f["data"]["steps"]) for f in of_kind("waterfall"))
    heat = [f["data"] for f in of_kind("heat")]
    for option in ("mono", "groups", "breaks", "colTitle", "rowNotes", "rowTips", "labelAlign", "cellMax"):
        assert any(option in d for d in heat), f"heat {option}"
    line = [f["data"] for f in of_kind("line")]
    assert any(d.get("yScale") == "log" and d.get("x", {}).get("type") == "date" for d in line)
    for option in ("shade", "shadeLegend", "points", "yTitle", "endLabels"):
        assert any(option in d for d in line), f"line {option}"
    for kind in ("column", "line"):
        assert any(f["data"].get("reference") for f in of_kind(kind)), f"a reference rule on a {kind}"
    assert any(f.get("note") for f in figures.values()), "a caution under a figure"
    assert any(f.get("card") for f in of_kind("tiles")) and any(f.get("part") for f in figures.values())
    entry_points = {}
    for s in snapshot["sections"].values():
        for p in s["provenance"]:
            entry_points.setdefault(p["figure"], set()).add(p["entry_point"])
    assert any(len(e) > 1 for e in entry_points.values()), "a figure with two entry points"


def test_gallery_line_charts_stay_within_four_series(gallery_page):
    snapshot = _snapshot_block(gallery_page)
    for s in snapshot["sections"].values():
        for fid, f in s["figures"].items():
            if f["kind"] == "line":
                assert len(f["data"]["series"]) <= 4, fid


# The gallery, drawn ----------------------------------------------------------------
#
# One headless Chrome run draws the gallery at 1440 by 900 and a probe script
# measures the page once the charts have redrawn for their fonts. Each check is
# caught on its own, so one failure reports itself rather than hiding the rest.

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
)


def _chrome() -> str | None:
    if os.environ.get("TECHVAL_CHROME"):
        return os.environ["TECHVAL_CHROME"]
    for candidate in CHROME_CANDIDATES:
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if found:
            return found
    return None


PROBE = r"""<script>
(function () {
  function fig(id) { return document.querySelector('[data-figure-id="' + id + '"]'); }
  function box(node) { var b = node.getBBox(); return { x: b.x, y: b.y, w: b.width, h: b.height }; }
  function overlap(a, b) { return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h; }
  function figureOf(node) { var f = node.closest('[data-figure-id]'); return f ? f.getAttribute('data-figure-id') : null; }
  function current() { var a = document.querySelector('.tv-rail [aria-current="true"]'); return a ? a.getAttribute('data-section') : null; }
  var checks = {
    dotLabels: function () {
      var out = [];
      document.querySelectorAll('.tv-chart--dot svg').forEach(function (chart) {
        var circles = Array.prototype.slice.call(chart.querySelectorAll('circle[data-series]'));
        chart.querySelectorAll('text.tv-value[data-series]').forEach(function (label) {
          var b = box(label);
          var nearest = null;
          var distance = Infinity;
          circles.forEach(function (c) {
            var cx = +c.getAttribute('cx'), cy = +c.getAttribute('cy'), r = +c.getAttribute('r');
            var dx = Math.max(b.x - (cx + r), cx - r - (b.x + b.w), 0);
            var dy = Math.max(b.y - (cy + r), cy - r - (b.y + b.h), 0);
            var d = Math.sqrt(dx * dx + dy * dy);
            if (d < distance) { distance = d; nearest = c; }
          });
          out.push({
            figure: figureOf(label), text: label.textContent, series: label.getAttribute('data-series'),
            nearestSeries: nearest && nearest.getAttribute('data-series'), nearestValue: nearest && nearest.getAttribute('data-value'),
          });
        });
      });
      return out;
    },
    foldScoreLabels: function () {
      // Every value label on the fold chart, with the paint of the dot nearest to it.
      var chart = fig('fold_scores').querySelector('svg');
      var circles = Array.prototype.slice.call(chart.querySelectorAll('circle'));
      return Array.prototype.map.call(chart.querySelectorAll('text.tv-value'), function (label) {
        var b = box(label);
        var nearest = null;
        var distance = Infinity;
        circles.forEach(function (c) {
          var cx = +c.getAttribute('cx'), cy = +c.getAttribute('cy');
          var dx = Math.max(b.x - cx, cx - (b.x + b.w), 0);
          var dy = Math.max(b.y - cy, cy - (b.y + b.h), 0);
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d < distance) { distance = d; nearest = c; }
        });
        return { text: label.textContent, paint: nearest ? nearest.style.fill : null };
      });
    },
    misdatedLabels: function () { return fig('misdated').querySelectorAll('text.tv-value').length; },
    coincident: function () {
      var out = [];
      document.querySelectorAll('.tv-chart--dot g.tv-markg').forEach(function (g) {
        var cs = Array.prototype.slice.call(g.querySelectorAll('circle'));
        for (var i = 0; i < cs.length; i++) for (var j = i + 1; j < cs.length; j++) {
          var dx = +cs[i].getAttribute('cx') - +cs[j].getAttribute('cx');
          var dy = +cs[i].getAttribute('cy') - +cs[j].getAttribute('cy');
          if (Math.sqrt(dx * dx + dy * dy) < 8) out.push(figureOf(g) + ': ' + g.getAttribute('aria-label'));
        }
      });
      return out;
    },
    crowding: function () {
      // Points in different rows of one chart closer than a marker and a gap.
      var out = [];
      document.querySelectorAll('.tv-chart--dot svg').forEach(function (chart) {
        var groups = Array.prototype.slice.call(chart.querySelectorAll('g.tv-markg'));
        var points = [];
        groups.forEach(function (g, row) {
          g.querySelectorAll('circle').forEach(function (c) {
            points.push({ row: row, x: +c.getAttribute('cx'), y: +c.getAttribute('cy'), label: g.getAttribute('aria-label') });
          });
        });
        for (var i = 0; i < points.length; i++) for (var j = i + 1; j < points.length; j++) {
          if (points[i].row === points[j].row) continue;
          var dx = points[i].x - points[j].x, dy = points[i].y - points[j].y;
          if (Math.sqrt(dx * dx + dy * dy) < 18) out.push(figureOf(chart) + ': ' + points[i].label + ' / ' + points[j].label);
        }
      });
      return out;
    },
    refLabels: function () {
      var out = { labels: 0, overlaps: [] };
      document.querySelectorAll('svg.tv-svg').forEach(function (chart) {
        var marks = Array.prototype.slice.call(chart.querySelectorAll('path.tv-mark, rect.tv-mark')).map(box).filter(function (m) { return m.w > 0 && m.h > 0; });
        var lines = Array.prototype.slice.call(chart.querySelectorAll('path.tv-line'));
        chart.querySelectorAll('text.tv-ref-label').forEach(function (label) {
          out.labels++;
          var b = box(label);
          if (marks.some(function (m) { return overlap(b, m); })) out.overlaps.push(figureOf(label) + ': "' + label.textContent + '" on a bar');
          lines.forEach(function (path) {
            var length = path.getTotalLength();
            for (var s = 0; s <= length; s += 2) {
              var p = path.getPointAtLength(s);
              if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) {
                out.overlaps.push(figureOf(label) + ': "' + label.textContent + '" on a line');
                break;
              }
            }
          });
        });
      });
      return out;
    },
    foldTicks: function () {
      return Array.prototype.map.call(fig('ic_lift_by_fold').querySelectorAll('text.tv-tick'), function (t) { return t.textContent; });
    },
    rail: function () {
      var links = document.querySelectorAll('.tv-rail [data-section]');
      var out = { first: links[0].getAttribute('data-section'), last: links[links.length - 1].getAttribute('data-section') };
      window.scrollTo(0, document.documentElement.scrollHeight);
      window.dispatchEvent(new Event('scroll'));
      out.atFoot = current();
      window.scrollTo(0, 0);
      window.dispatchEvent(new Event('scroll'));
      out.atTop = current();
      return out;
    },
    headlines: function () {
      var out = {};
      document.querySelectorAll('section.tv-section').forEach(function (s) {
        var h = s.querySelector('.tv-headline__text');
        if (!h) return;
        var more = h.querySelector('details');
        out[s.id] = { shown: h.querySelector('p').textContent, rest: more ? more.querySelector('p').textContent : null, open: more ? more.open : null };
      });
      return out;
    },
    waterfallLegend: function () {
      return Array.prototype.map.call(fig('segment_income').querySelectorAll('.tv-legend__item'), function (li) { return li.textContent; });
    },
    tilesCard: function () {
      var card = fig('year_five');
      return {
        tag: card.tagName.toLowerCase(),
        part: !!card.querySelector('.tv-figure__part[data-figure-id="year_five_value"]'),
        tables: card.querySelectorAll('.tv-figure__table table').length,
        toggle: !card.querySelector('.tv-figure__foot .tv-btn').hidden,
        source: card.querySelector('.tv-figure__source').textContent,
      };
    },
    tilesets: function () {
      return Array.prototype.map.call(document.querySelectorAll('.tv-tileset'), function (set) {
        var btn = set.querySelector('.tv-figure__foot .tv-btn');
        var table = set.querySelector('.tv-figure__table');
        var body = set.querySelector('.tv-tileset__body');
        var out = {
          id: set.getAttribute('data-figure-id'),
          tiles: set.querySelectorAll('.tv-tile').length,
          rows: set.querySelectorAll('.tv-figure__table tbody tr').length,
          toggle: btn ? !btn.hidden : false,
          buttons: set.querySelectorAll('.tv-btn').length,
          source: (set.querySelector('.tv-figure__source') || {}).textContent || '',
          tableHidden: table ? table.hidden : null,
        };
        if (btn) {
          btn.click();
          out.opened = { table: table.hidden === false, body: body.hidden === true, text: btn.textContent };
          btn.click();
          out.closed = { table: table.hidden === true, body: body.hidden === false, text: btn.textContent };
        }
        return out;
      });
    },
    shadeKey: function () {
      var fig = document.querySelector('[data-figure-id="closes"]');
      return Array.prototype.map.call(fig.querySelectorAll('.tv-legend__item'), function (li) { return li.textContent; });
    },
    zeroStep: function () {
      var fig = document.querySelector('[data-figure-id="segment_income"]');
      var out = [];
      fig.querySelectorAll('.tv-markg').forEach(function (g) {
        var label = g.getAttribute('aria-label') || '';
        if (label.indexOf('Outside the segments') !== 0) return;
        var mark = g.querySelector('.tv-mark');
        var box = mark ? mark.getBBox() : null;
        out.push({ label: label, drawn: !!mark, height: box ? Math.round(box.height) : null, width: box ? Math.round(box.width) : null });
      });
      return out;
    },
    toggleNames: function () {
      var shown = Array.prototype.filter.call(document.querySelectorAll('.tv-figure__foot .tv-btn'), function (btn) {
        return !btn.hidden;
      });
      return shown.map(function (btn) {
        var fig = btn.closest('[data-figure-id]');
        var title = fig ? (fig.querySelector('.tv-figure__title, .tv-tileset__title') || {}).textContent || '' : '';
        return {
          text: btn.textContent,
          expanded: btn.getAttribute('aria-expanded'),
          label: btn.getAttribute('aria-label'),
          named: !!title && (btn.getAttribute('aria-label') || '').indexOf(title) > 0,
        };
      });
    },
    engineOrder: function () {
      return Array.prototype.map.call(document.querySelectorAll('#engine [data-figure-id]'), function (f) { return f.getAttribute('data-figure-id'); });
    },
    scoreboard: function () {
      var board = fig('scoreboard');
      var sections = fig('sections');
      var collection = fig('collection');
      return {
        order: Array.prototype.map.call(document.querySelectorAll('#overview [data-figure-id]'), function (f) { return f.getAttribute('data-figure-id'); }),
        tiles: board.querySelectorAll('.tv-overview-grid a.tv-tile[data-state="scored"]').length,
        chips: board.querySelectorAll('.tv-overview-grid .tv-chip').length,
        tableRows: sections.querySelectorAll('.tv-figure__body table tbody tr').length,
        collection: collection.tagName.toLowerCase(),
        collectionTiles: collection.querySelectorAll('.tv-tile').length,
      };
    },
    legendOff: function () { return fig('precision').querySelectorAll('.tv-legend__item').length; },
    heatLabels: function () {
      var chart = fig('coefficients').querySelector('svg');
      var cellLeft = Math.min.apply(null, Array.prototype.map.call(chart.querySelectorAll('rect.tv-mark'), function (r) { return +r.getAttribute('x'); }));
      return Array.prototype.map.call(chart.querySelectorAll('text.tv-label'), function (t) {
        var b = box(t);
        return { text: t.textContent, left: b.x, right: b.x + b.w, cellLeft: cellLeft };
      });
    },
    absorbed: function () {
      var ablation = fig('ablation');
      var closes = fig('closes');
      return {
        whiskers: ablation.querySelectorAll('.tv-whisker').length + fig('ndcg_by_method').querySelectorAll('.tv-whisker').length,
        groups: ablation.querySelectorAll('text.tv-group-label').length,
        cautions: document.querySelectorAll('.tv-note--caution').length,
        tableFigure: fig('methods').querySelectorAll('.tv-figure__body table tbody tr').length,
        shades: closes.querySelectorAll('rect.tv-shade').length,
        closeTicks: Array.prototype.map.call(closes.querySelectorAll('text.tv-tick'), function (t) { return t.textContent; }),
        fullerTables: closes.querySelectorAll('.tv-figure__table table').length,
      };
    },
    failures: function () {
      return Array.prototype.map.call(document.querySelectorAll('.tv-note'), function (n) { return n.textContent; }).filter(function (t) {
        return /renderer for this section failed|has no chart named/.test(t);
      });
    },
  };
  function run() {
    var out = {};
    Object.keys(checks).forEach(function (name) {
      try { out[name] = checks[name](); } catch (err) { out[name] = { error: String(err) }; }
    });
    var pre = document.createElement('pre');
    pre.id = 'tv-probe';
    pre.textContent = JSON.stringify(out);
    document.body.appendChild(pre);
  }
  window.addEventListener('load', function () { setTimeout(run, 3000); });
})();
</script>"""


@pytest.fixture(scope="module")
def drawn(gallery, tmp_path_factory):
    chrome = _chrome()
    if chrome is None:
        pytest.skip("Chrome is not installed, so the gallery cannot be drawn")
    work = tmp_path_factory.mktemp("drawn")
    page = work / "gallery.html"
    page.write_text(gallery.render_gallery().replace("</body>", PROBE + "</body>"), encoding="utf-8")
    command = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=1440,900",
        "--virtual-time-budget=15000",
        "--dump-dom",
        page.as_uri() + "?theme=light",
    ]
    # A private --user-data-dir makes Chrome on macOS dump the page and then never
    # exit, so the run uses headless Chrome's own throwaway profile. Should a run
    # still hang after the dump, what it printed is enough.
    try:
        result = subprocess.run(command, capture_output=True, timeout=180)
        out, err = result.stdout, result.stderr
    except subprocess.TimeoutExpired as hung:
        out, err = hung.stdout or b"", hung.stderr or b""
    out = out.decode("utf-8", "replace")
    match = re.search(r'<pre id="tv-probe">(.*?)</pre>', out, flags=re.S)
    assert match, f"the probe did not report: {err.decode('utf-8', 'replace')[-2000:]}"
    return json.loads(html.unescape(match.group(1)))


def _checked(drawn, name):
    value = drawn[name]
    assert not (isinstance(value, dict) and "error" in value), f"{name}: {value}"
    return value


def test_the_drawn_gallery_has_no_renderer_failure(drawn):
    assert _checked(drawn, "failures") == []


def test_a_dot_label_names_only_the_value_of_the_dot_beside_it(drawn):
    labels = _checked(drawn, "dotLabels")
    assert any(label["figure"] == "fold_scores" for label in labels), "the fold chart carries no label"
    for label in labels:
        assert label["nearestSeries"] == label["series"], label
        assert label["nearestValue"] == label["text"], label


def test_the_fold_label_sits_beside_the_model_not_the_baseline_it_trails(drawn):
    # Fold 2 has the widest gap and the model sits left of the median there. The
    # label reads the model's 0.352, so the dot nearest it must be the model's,
    # judged by paint alone so the check does not lean on the kit's own markup.
    labels = _checked(drawn, "foldScoreLabels")
    assert [label["text"] for label in labels] == ["0.352"], labels
    assert labels[0]["paint"] == "var(--c-model)", labels


def test_a_dot_chart_where_nothing_stands_out_carries_no_label(drawn):
    assert _checked(drawn, "misdatedLabels") == 0


def test_coincident_points_are_drawn_apart(drawn):
    assert _checked(drawn, "coincident") == []


def test_points_drawn_apart_stay_clear_of_the_next_row(drawn):
    assert _checked(drawn, "crowding") == []


def test_reference_labels_touch_no_bar_and_no_line(drawn):
    refs = _checked(drawn, "refLabels")
    assert refs["labels"] >= 4
    assert refs["overlaps"] == []


def test_column_labels_keep_the_first_and_the_last(drawn):
    # Twelve fold labels do not all fit the card, so some are dropped, never the ends.
    ticks = [t for t in _checked(drawn, "foldTicks") if t.startswith("Fold")]
    assert len(ticks) < 12, ticks
    assert ticks[0] == "Fold 1" and ticks[-1] == "Fold 12", ticks


def test_the_rail_marks_the_last_section_at_the_foot_of_the_page_and_the_first_at_the_top(drawn):
    rail = _checked(drawn, "rail")
    assert rail["atFoot"] == rail["last"]
    assert rail["atTop"] == rail["first"]


def test_a_headline_shows_its_first_sentence_and_keeps_the_rest_verbatim(drawn, gallery_page):
    headlines = _checked(drawn, "headlines")
    snapshot = _snapshot_block(gallery_page)
    for sid, shown in headlines.items():
        verdict = snapshot["sections"][sid]["headline"]["verdict_text"]
        whole = shown["shown"] + (" " + shown["rest"] if shown["rest"] else "")
        # Only the leading token is cased for display; everything after it is verbatim.
        token = verdict.split(" ", 1)[0]
        assert whole[: len(token)].lower() == token.lower(), sid
        assert whole[len(token) :] == verdict[len(token) :], sid
        assert shown["open"] in (None, False), sid
    warranted = headlines["warranted"]
    assert warranted["shown"] == "Log MAE of 0.4120 against 0.4090 for the sector median, a lift of -0.0030 on 162 observations."
    assert warranted["rest"].endswith("Use the sector median.")
    # A verdict that opens with an initialism shows it in capitals, not as "Ndcg@10" or "Auc".
    assert headlines["encoder"]["shown"].startswith("NDCG@10 of 0.5407 against 0.2213")
    assert headlines["propensity"]["shown"].startswith("AUC of 0.561 against 0.548")


def test_a_waterfall_legend_lists_only_the_steps_it_draws(drawn):
    assert _checked(drawn, "waterfallLegend") == ["Segment operating income", "Operating income"]


def test_tiles_in_a_card_carry_a_table_a_part_and_every_entry_point(drawn):
    card = _checked(drawn, "tilesCard")
    assert card["tag"] == "figure" and card["part"] and card["toggle"]
    assert card["tables"] == 2
    assert "render_gallery" in card["source"] and "gallery_snapshot" in card["source"]


def test_every_tileset_reads_as_numbers_too(drawn):
    # Tiles on the page plane are a figure like any other: the same table view,
    # the same data toggle, and the source line naming what computed them.
    sets = {s["id"]: s for s in _checked(drawn, "tilesets")}
    assert "panel_reach" in sets and "scoreboard" in sets
    for sid, s in sets.items():
        assert s["toggle"], sid
        assert s["buttons"] == 1, sid  # one toggle, never a second one drawn by a section
        assert s["rows"] >= 1, sid
        assert s["tableHidden"], sid
        assert s["source"], sid
        assert s["opened"] == {"table": True, "body": True, "text": "Hide data"}, sid
        assert s["closed"] == {"table": True, "body": True, "text": "Show data"}, sid
    plain = sets["panel_reach"]
    assert plain["tiles"] == 3 and plain["rows"] == 3
    assert "render_gallery" in plain["source"]


def test_a_table_that_scrolls_says_so():
    # A table wider than its box hides columns. Two covers painted on the
    # content scroll away and uncover a shadow on whichever side still holds
    # columns, so a cut-off table never looks finished.
    layout = _strip_comments((ASSETS / "layout.css").read_text(encoding="utf-8"))
    wrap = next(body for prelude, body in _blocks(layout) if prelude == ".tv-table-wrap")
    assert "overflow-x: auto" in wrap
    assert wrap.count("linear-gradient") == 4
    assert "background-attachment: local, local, scroll, scroll" in wrap


def test_a_shaded_window_keeps_its_key_when_it_is_the_only_one(drawn):
    # One series needs no legend: the title names it. A shaded window is named
    # nowhere else, so its key is drawn even when it stands alone.
    keys = _checked(drawn, "shadeKey")
    assert keys == ["Split window, from the last filing on the old basis to the first on the new"]


def test_a_bridge_step_of_exactly_zero_draws_a_rule(drawn):
    # The step is the finding ("nothing is taken off outside the segments"), so
    # it is drawn rather than left as a value label over an empty column.
    (step,) = _checked(drawn, "zeroStep")
    assert step["drawn"] and step["height"] == 2 and step["width"] > 2


def test_every_data_toggle_says_which_figure_it_opens_and_whether_it_is_open(drawn):
    toggles = _checked(drawn, "toggleNames")
    assert len(toggles) >= 10
    for t in toggles:
        assert t["expanded"] == "false", t
        assert t["label"].startswith(t["text"]), t
        assert t["named"], t


def test_figures_follow_their_order_number_not_their_key(drawn):
    # The engine renderer names the football field; the rest follow their order numbers.
    assert _checked(drawn, "engineOrder") == ["football", "sotp", "segment_income", "methods"]


def test_the_gallery_overview_is_drawn_by_the_scoreboard_renderer(drawn, gallery_page):
    # The collector's figure ids and shapes, so the real renderer draws them, not the kit's default.
    board = _checked(drawn, "scoreboard")
    snapshot = _snapshot_block(gallery_page)
    assert board["order"] == ["scoreboard", "sections", "collection"]
    assert board["tiles"] == board["chips"] == 5
    # A row for every section but the overview, and one for the totals.
    assert board["tableRows"] == (len(snapshot["sections"]) - 1) + 1
    assert board["collection"] == "figure" and board["collectionTiles"] == 6


def test_a_legend_turned_off_draws_no_legend(drawn):
    assert _checked(drawn, "legendOff") == 0


def test_heat_row_labels_are_never_clipped(drawn):
    for label in _checked(drawn, "heatLabels"):
        assert label["left"] >= 0 and label["right"] <= label["cellLeft"], label


def test_the_kit_draws_what_the_sections_used_to_draw_locally(drawn):
    absorbed = _checked(drawn, "absorbed")
    assert absorbed["whiskers"] >= 24  # three strokes a whisker, four on the ablation, four on the bars
    assert absorbed["groups"] == 2
    assert absorbed["cautions"] >= 2
    assert absorbed["tableFigure"] == 3
    assert absorbed["shades"] == 2 and absorbed["fullerTables"] == 2
    assert {"2022", "2023", "2024"} <= set(absorbed["closeTicks"])


def test_a_deep_link_is_held_while_the_charts_above_it_draw():
    # Charts draw after the first layout, so a section linked by #id drifts
    # down the page unless the anchor is held until the layout settles, and
    # released the moment the reader moves so it never fights a scroll.
    text = APP.read_text(encoding="utf-8")
    assert "function holdAnchor(" in text
    assert re.search(r"holdAnchor\(target,", text)
    assert "watcher.observe(document.body)" in text
    for name in ("wheel", "touchstart", "keydown", "pointerdown"):
        assert f'"{name}"' in text
