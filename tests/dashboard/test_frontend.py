"""Tests for the dashboard front end: the stylesheets, the chart kit and the gallery.

None of these start a browser. They read the asset files as text and hold them
to the rules a browser cannot enforce for us: the theme structure that keeps a
colour from existing in only one mode, the ban on markup built from strings,
the ban on dashed strokes, and the ban on em-dashes in anything a reader sees.
The drawing itself is checked by rendering the gallery and looking at it.
"""

from __future__ import annotations

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
    "datalayer",
    "sample",
)

DARK_MEDIA = "@media (prefers-color-scheme: dark)"
DARK_AUTO = ':root:not([data-theme="light"])'
DARK_FORCED = ':root[data-theme="dark"]'


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
        elif prelude in (":root", DARK_FORCED):
            key = "root" if prelude == ":root" else "forced"
            assert key not in found, f"{prelude} is defined twice"
            found[key] = body
        else:
            pytest.fail(f"tokens.css holds a rule other than the theme blocks: {prelude}")
    assert set(found) == {"root", "auto", "forced"}
    return found


# Stylesheets ---------------------------------------------------------------------


def test_every_dark_token_is_defined_on_bare_root(token_blocks):
    root = _defined(token_blocks["root"])
    for key in ("auto", "forced"):
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
    for key in ("auto", "forced"):
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
    assert '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif' in root
    assert '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace' in root


# Every asset ---------------------------------------------------------------------


def test_no_asset_contains_an_em_dash():
    offenders = [
        str(p.relative_to(ROOT))
        for p in _asset_files()
        if "—" in p.read_text(encoding="utf-8")
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
        "hbar", "column", "dot", "line", "heat", "hist", "range", "waterfall", "tiles",
    }


def test_chips_cover_every_verdict_status(kit_source):
    assert _object_keys(kit_source, "var STATUS") == {
        "beats", "inside_noise", "ties", "loses", "not_significant", "refused",
    }


def test_every_chart_builds_a_table_view(kit_source):
    for chart in ("hbar", "column", "dot", "line", "heat", "hist", "range", "waterfall"):
        match = re.search(rf"\n  function {chart}\(body, spec\) \{{(.*?)\n  \}}\n", kit_source, flags=re.S)
        assert match, f"chart {chart} not found"
        assert "tableFor(body" in match.group(1), f"{chart} has no table view"


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


def test_app_stamps_the_theme_from_the_query_string():
    text = APP.read_text(encoding="utf-8")
    assert re.search(r"theme=\(dark\|light\)", text)
    assert 'setAttribute("data-theme"' in text
    # Stamped when the script runs, not at boot, so the page never paints in the wrong theme.
    assert re.search(r"\n  stampTheme\(\);\n", text)


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
