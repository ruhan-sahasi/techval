"""Render the investing dashboard: one self-contained page from one snapshot.

The page inlines the invest tokens, the dashboard's component stylesheet (the
chart kit's figures, tables, tooltips and chips are styled there and reused
as-is), the app shell styles, the chart kit, the shell script and one script
per pane, with the snapshot embedded as JSON. Same discipline as the results
page: the same snapshot and assets always produce the same bytes.
"""

from __future__ import annotations

import html
from pathlib import Path

from ..dashboard.render import FONTS_HREF, embed_json
from .snapshot import validate_snapshot

ASSETS = Path(__file__).parent / "assets"
DASHBOARD_ASSETS = Path(__file__).parent.parent / "dashboard" / "assets"

SNAPSHOT_ELEMENT_ID = "iv-snapshot"

# (directory, name) pairs, in inline order. Panes follow the shell sorted by
# name, so adding one never reorders the others.
_STYLES = (
    (ASSETS, "tokens.css"),
    (DASHBOARD_ASSETS, "layout.css"),
    (ASSETS, "app.css"),
)
_SCRIPTS = (
    (DASHBOARD_ASSETS, "kit.js"),
    (ASSETS, "invest.js"),
)


def asset_names() -> list[str]:
    """Every asset the page inlines, in order, as the page labels them."""
    return [name for _, name in _STYLES] + [name for _, name in _SCRIPTS] + [
        f"panes/{p.name}" for p in sorted((ASSETS / "panes").glob("*.js"))
    ]


def _read(directory: Path, name: str, closing: str) -> str:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"invest asset {name} is missing from {directory}")
    text = path.read_text(encoding="utf-8").rstrip("\n") + "\n"
    if closing.lower() in text.lower():
        raise ValueError(f"invest asset {name} contains {closing!r} and would close its element early")
    return text


def render_page(snapshot: dict) -> str:
    """The whole page for one snapshot, as a string."""
    validate_snapshot(snapshot)
    styles = "".join(_read(d, n, "</style") for d, n in _STYLES)
    scripts = [(n, _read(d, n, "</script")) for d, n in _SCRIPTS]
    for pane in sorted((ASSETS / "panes").glob("*.js")):
        scripts.append((f"panes/{pane.name}", _read(pane.parent, pane.name, "</script")))
    title = f"{snapshot['meta']['name']} · techval invest"
    out = [
        "<!doctype html>\n",
        '<html lang="en">\n',
        "<head>\n",
        '<meta charset="utf-8">\n',
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
        f"<title>{html.escape(title)}</title>\n",
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n',
        f'<link rel="stylesheet" href="{html.escape(FONTS_HREF)}">\n',
        "<style>\n",
        styles,
        "</style>\n",
        "</head>\n",
        "<body>\n",
        "<noscript>This page draws itself with JavaScript. Everything it shows "
        "is in the snapshot JSON it was rendered from.</noscript>\n",
        f'<script type="application/json" id="{SNAPSHOT_ELEMENT_ID}">',
        embed_json(snapshot),
        "</script>\n",
    ]
    for name, text in scripts:
        out += [f'<script data-asset="{html.escape(name)}">\n', text, "</script>\n"]
    out += ["</body>\n", "</html>\n"]
    return "".join(out)


def write_page(snapshot: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_page(snapshot), encoding="utf-8")
    return path
