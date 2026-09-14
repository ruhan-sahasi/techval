"""Tests for the dashboard front end: the stylesheets, the chart kit and the gallery.

None of these start a browser. They read the asset files as text and hold them
to the rules a browser cannot enforce for us: the theme structure that keeps a
colour from existing in only one mode, the ban on markup built from strings,
the ban on dashed strokes, and the ban on em-dashes in anything a reader sees.
The drawing itself is checked by rendering the gallery and looking at it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "src" / "techval" / "dashboard" / "assets"

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
