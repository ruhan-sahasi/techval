"""The invest assets follow the dashboard's rules, and its tokens match, theme for theme.

The parity test is structural rather than a curated list: every token the
results dashboard defines in its dark block must carry the same value on the
invest page's bare root (the app opens dark), and every light value must match
the invest page's light block. Scalars defined once must agree once. One
family, two files, no drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "src" / "techval" / "dashboard" / "assets"
INVEST = ROOT / "src" / "techval" / "invest" / "assets"


def _pairs(body: str) -> dict[str, str]:
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body))


def _block(css: str, pattern: str) -> str:
    match = re.search(pattern, css, flags=re.S)
    assert match, f"no block matches {pattern}"
    return match.group(1)


def _strip(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


@pytest.fixture(scope="module")
def dashboard_tokens() -> dict[str, dict[str, str]]:
    css = _strip((DASH / "tokens.css").read_text(encoding="utf-8"))
    return {
        "light": _pairs(_block(css, r":root \{(.*?)\n\}")),
        "dark": _pairs(_block(css, r":root\[data-theme=\"dark\"\] \{(.*?)\n\}")),
    }


@pytest.fixture(scope="module")
def invest_tokens() -> dict[str, dict[str, str]]:
    css = _strip((INVEST / "tokens.css").read_text(encoding="utf-8"))
    return {
        "dark": _pairs(_block(css, r":root \{(.*?)\n\}")),
        "light": _pairs(_block(css, r":root\[data-theme=\"light\"\] \{(.*?)\n\}")),
    }


def test_every_dark_token_matches_the_dashboards_dark_theme(dashboard_tokens, invest_tokens):
    for name, value in dashboard_tokens["dark"].items():
        assert invest_tokens["dark"].get(name) == value, name


def test_every_light_override_matches_the_dashboards_light_theme(dashboard_tokens, invest_tokens):
    for name, value in invest_tokens["light"].items():
        assert dashboard_tokens["light"].get(name) == value, name
    # And the light block covers exactly the names the dark themes redefine.
    assert set(invest_tokens["light"]) == set(dashboard_tokens["dark"])


def test_scalars_agree_and_are_defined_once(dashboard_tokens, invest_tokens):
    scalars = set(dashboard_tokens["light"]) - set(dashboard_tokens["dark"])
    for name in scalars:
        assert invest_tokens["dark"].get(name) == dashboard_tokens["light"][name], name
        assert name not in invest_tokens["light"], name


def _asset_files() -> list[Path]:
    return sorted(p for p in INVEST.rglob("*") if p.is_file())


def test_assets_exist():
    assert _asset_files(), "no invest assets yet"


def test_no_asset_contains_an_em_dash():
    offenders = [p.name for p in _asset_files() if "—" in p.read_text(encoding="utf-8")]
    assert not offenders, offenders


def test_no_dashed_strokes_anywhere():
    for path in _asset_files():
        text = path.read_text(encoding="utf-8")
        assert "stroke-dasharray" not in text, path.name
        assert not re.search(r"border[\w-]*\s*:[^;]*\bdashed\b", text), path.name


def test_no_script_builds_markup_from_strings():
    banned = re.compile(r"\.(innerHTML|outerHTML)\s*\+?=|insertAdjacentHTML|document\.write|createContextualFragment")
    for path in _asset_files():
        if path.suffix != ".js":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert not banned.search(line), f"{path.name}:{n}"


def test_styles_reach_colours_through_tokens_only():
    for path in _asset_files():
        if path.suffix != ".css" or path.name == "tokens.css":
            continue
        css = _strip(path.read_text(encoding="utf-8"))
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css), path.name
        assert not re.search(r"\b(rgb|rgba|hsl|hsla|oklch)\(", css), path.name
