"""Render a snapshot into one self-contained HTML page.

The page is a single file with everything inlined: the two stylesheets, the
snapshot as a JSON data block, the chart kit, the boot script and one renderer
per section. The only external resource is the Google Fonts stylesheet for Public
Sans, Newsreader and IBM Plex Mono, and the type tokens carry system fallback
stacks, so a reader offline sees the same figures in a different face.

Rendering is pure. The same snapshot and the same assets produce the same
bytes every time: no clock, no environment, no ordering left to a dictionary,
so the committed ``docs/dashboard/index.html`` changes exactly when the
results or the frontend do.

The snapshot is untrusted text as far as HTML is concerned. A company name, a
refusal message quoting a filing, or an Item 1 excerpt can contain
``</script>``, and inside a script element the HTML tokenizer ends the element
at the first one it sees, whatever the JSON around it says. Every ``</`` in the
data block is therefore written ``<\\/`` and every ``<!--`` as ``\\u003c!--``;
both are valid JSON escapes that ``JSON.parse`` reads back as the original
characters, and between them the tokenizer can neither leave the element early
nor be tricked by a comment opener into missing its real end.
"""

from __future__ import annotations

import html
import json
import re
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from .sections import SECTION_IDS
from .snapshot import Snapshot

STYLESHEETS = ("tokens.css", "layout.css")
SCRIPTS = ("kit.js", "app.js")

FONTS_HREF = (
    "https://fonts.googleapis.com/css2"
    "?family=IBM+Plex+Mono:wght@400;500"
    "&family=Newsreader:ital,wght@0,400;0,500;1,400"
    "&family=Public+Sans:wght@400;500;650"
    "&display=swap"
)

SNAPSHOT_ELEMENT_ID = "tv-snapshot"


def default_assets_dir() -> Traversable:
    """The frontend assets shipped inside the package."""
    return files("techval.dashboard") / "assets"


def _read_asset(assets: Traversable | Path, name: str) -> str:
    node = assets
    for part in name.split("/"):
        node = node / part
    if not node.is_file():
        raise FileNotFoundError(
            f"dashboard asset {name} is missing from {assets}; the page cannot be "
            "rendered without it"
        )
    text = node.read_text(encoding="utf-8")
    # One trailing newline, whatever the file ends with, so an editor adding or
    # removing a final newline does not move the page.
    return text.rstrip("\n") + "\n"


def _guard(name: str, text: str, closing: str) -> str:
    if re.search(re.escape(closing), text, flags=re.IGNORECASE):
        raise ValueError(
            f"dashboard asset {name} contains {closing!r}, which would close its "
            "inlined element early. Split the string in the source, for example "
            f"'<' + '/{closing[2:]}'."
        )
    return text


def embed_json(data: Any) -> str:
    """JSON that is safe to place inside a ``<script type="application/json">`` element."""
    text = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.replace("</", "<\\/").replace("<!--", "\\u003c!--")


def render_dashboard(
    snapshot: Snapshot, assets_dir: Traversable | Path | str | None = None
) -> str:
    """The whole page for one snapshot, as a string.

    The snapshot is revalidated through its own schema first, so a snapshot
    assembled in code is held to the same rules as one read from disk and a
    malformed one is refused before a page is drawn from it.
    """
    data = Snapshot.from_dict(snapshot.to_dict(), source="snapshot").to_dict()
    assets = Path(assets_dir) if isinstance(assets_dir, str) else assets_dir
    if assets is None:
        assets = default_assets_dir()

    styles = "".join(
        _guard(name, _read_asset(assets, name), "</style") for name in STYLESHEETS
    )
    scripts = [
        (name, _guard(name, _read_asset(assets, name), "</script"))
        for name in (*SCRIPTS, *(f"sections/{sid}.js" for sid in SECTION_IDS))
    ]

    out = [
        "<!doctype html>\n",
        '<html lang="en">\n',
        "<head>\n",
        '<meta charset="utf-8">\n',
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
        f"<title>{html.escape(data['title'])}</title>\n",
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n',
        f'<link rel="stylesheet" href="{html.escape(FONTS_HREF)}">\n',
        "<style>\n",
        styles,
        "</style>\n",
        "</head>\n",
        "<body>\n",
        "<noscript>This page draws its figures with JavaScript. The results it "
        "shows are in the snapshot JSON it was rendered from.</noscript>\n",
        f'<script type="application/json" id="{SNAPSHOT_ELEMENT_ID}">',
        embed_json(data),
        "</script>\n",
    ]
    for name, text in scripts:
        out += [f'<script data-asset="{html.escape(name)}">\n', text, "</script>\n"]
    out += ["</body>\n", "</html>\n"]
    return "".join(out)


def write_dashboard(
    snapshot: Snapshot,
    out_path: str | Path,
    assets_dir: Traversable | Path | str | None = None,
) -> Path:
    """Render the page and write it, creating the directory if needed."""
    path = Path(out_path)
    page = render_dashboard(snapshot, assets_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF on every platform, so a page rendered on Windows diffs clean.
    path.write_text(page, encoding="utf-8", newline="\n")
    return path
