"""Rendering: one self-contained page, byte-identical for the same snapshot, unbreakable by its data.

These tests render against a small assets directory built in a temporary
directory rather than the real frontend, so they test the embedding contract
and not the charts: what is inlined, in what order, what is escaped, and that
nothing is left to chance between two renders of the same results.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from techval.dashboard.render import (
    FONTS_HREF,
    SNAPSHOT_ELEMENT_ID,
    render_dashboard,
    write_dashboard,
)
from techval.dashboard.sections import SECTION_IDS
from techval.dashboard.snapshot import Snapshot, dump_snapshot, load_snapshot
from techval.errors import ConfigError

HOSTILE = "</script><script>alert(1)</script> <!--<script> </SCRIPT > &  "


def make_assets(root: Path) -> Path:
    assets = root / "assets"
    (assets / "sections").mkdir(parents=True)
    (assets / "tokens.css").write_text(":root { --c-model: #2a6fdb; }\n")
    (assets / "layout.css").write_text("body { margin: 0; }")
    (assets / "kit.js").write_text("window.TV = {};\n")
    (assets / "app.js").write_text("/* app */\n\n\n")
    for sid in SECTION_IDS:
        (assets / "sections" / f"{sid}.js").write_text(f"/* section {sid} */\n")
    return assets


def make_snapshot(takeaway: str = "The encoder beats the popularity prior.") -> Snapshot:
    sections = {
        sid: {
            "id": sid,
            "title": sid.title(),
            "takeaway": "",
            "status": "not_built",
            "refusals": [],
            "headline": None,
            "figures": {},
            "provenance": [],
        }
        for sid in SECTION_IDS
    }
    sections["encoder"].update(
        {
            "status": "ok",
            "takeaway": takeaway,
            "figures": {
                "names": {
                    "kind": "hbar",
                    "title": "Candidates",
                    "subtitle": "ndcg@10",
                    "data": {"rows": [{"label": HOSTILE, "value": 0.25}]},
                }
            },
            "provenance": [
                {
                    "figure": "names",
                    "entry_point": "techval.ml.encoder.evaluate_peer_encoder",
                    "inputs": ["tests/fixtures/peer_panel_tmt.json"],
                    "seconds": 1.5,
                }
            ],
        }
    )
    sections["fade"].update(
        {"status": "refused", "refusals": [{"what": "Revenue fade", "why": HOSTILE}]}
    )
    return Snapshot(
        title="techval <Results> & more",
        techval_commit="b37dde5",
        collected_at="2026-09-14",
        fixtures={"tests/fixtures/peer_panel_tmt.json": "0" * 64},
        sections=sections,
    )


class Scripts(HTMLParser):
    """Collects every element the way a browser's tokenizer delimits it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[tuple[dict, str]] = []
        self.links: list[dict] = []
        self.styles: list[str] = []
        self.title = ""
        self._in: str | None = None
        self._attrs: dict = {}
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "link":
            self.links.append(dict(attrs))
        if tag in ("script", "style", "title"):
            self._in, self._attrs, self._buf = tag, dict(attrs), []

    def handle_endtag(self, tag):
        if tag == self._in:
            text = "".join(self._buf)
            if tag == "script":
                self.scripts.append((self._attrs, text))
            elif tag == "style":
                self.styles.append(text)
            else:
                self.title = text
            self._in = None

    def handle_data(self, data):
        if self._in:
            self._buf.append(data)


def parse(page: str) -> Scripts:
    p = Scripts()
    p.feed(page)
    p.close()
    return p


def test_same_snapshot_renders_byte_identical_html(tmp_path):
    assets = make_assets(tmp_path)
    first = render_dashboard(make_snapshot(), assets)
    second = render_dashboard(make_snapshot(), assets)
    assert first == second
    # And through the file: dump, load, render again.
    path = dump_snapshot(make_snapshot(), tmp_path / "snapshot.json")
    assert render_dashboard(load_snapshot(path), assets) == first


def test_page_layout_follows_the_embedding_contract(tmp_path):
    page = render_dashboard(make_snapshot(), make_assets(tmp_path))
    doc = parse(page)
    assert page.startswith("<!doctype html>\n")
    assert doc.title == "techval <Results> & more"
    assert "<title>techval &lt;Results&gt; &amp; more</title>" in page

    [style] = doc.styles
    assert style.index("--c-model") < style.index("body { margin: 0; }")

    stylesheet = [l for l in doc.links if l.get("rel") == "stylesheet"]
    assert [l["href"] for l in stylesheet] == [FONTS_HREF]
    assert "Public+Sans:wght@400;500;650" in FONTS_HREF
    assert "Newsreader:ital,wght@0,400;0,500;1,400" in FONTS_HREF
    assert "IBM+Plex+Mono:wght@400;500" in FONTS_HREF

    names = [a.get("data-asset") for a, _ in doc.scripts[1:]]
    assert names == ["kit.js", "app.js", *(f"sections/{s}.js" for s in SECTION_IDS)]
    assert doc.scripts[2][1] == "\n/* app */\n"


def test_no_external_resource_but_the_fonts(tmp_path):
    page = render_dashboard(make_snapshot(), make_assets(tmp_path))
    urls = re.findall(r'(?:src|href)="([^"]+)"', page)
    assert urls, "the font links should be found"
    for url in urls:
        assert url.startswith(("https://fonts.googleapis.com", "https://fonts.gstatic.com")), url
    assert not re.search(r"<script[^>]*\ssrc=", page)


def test_data_cannot_break_out_of_the_json_block(tmp_path):
    snap = make_snapshot(takeaway=HOSTILE)
    page = render_dashboard(snap, make_assets(tmp_path))
    doc = parse(page)

    # Exactly the scripts the renderer wrote, and no element the data opened.
    assert len(doc.scripts) == 1 + 2 + len(SECTION_IDS)
    attrs, block = doc.scripts[0]
    assert attrs == {"type": "application/json", "id": SNAPSHOT_ELEMENT_ID}
    assert json.loads(block) == snap.to_dict()
    assert json.loads(block)["sections"]["encoder"]["takeaway"] == HOSTILE

    assert "</script" not in block.lower()
    assert "<!--" not in block


def test_write_dashboard_writes_the_page(tmp_path):
    assets = make_assets(tmp_path)
    out = write_dashboard(make_snapshot(), tmp_path / "docs" / "dashboard" / "index.html", assets)
    assert out.read_bytes() == render_dashboard(make_snapshot(), assets).encode("utf-8")


def test_a_missing_asset_is_named(tmp_path):
    assets = make_assets(tmp_path)
    (assets / "sections" / "tmt.js").unlink()
    with pytest.raises(FileNotFoundError, match="sections/tmt.js"):
        render_dashboard(make_snapshot(), assets)


def test_an_asset_that_would_close_its_element_is_refused(tmp_path):
    assets = make_assets(tmp_path)
    (assets / "kit.js").write_text('const s = "</SCRIPT>";\n')
    with pytest.raises(ValueError, match="kit.js"):
        render_dashboard(make_snapshot(), assets)


def test_a_malformed_snapshot_is_refused_before_rendering(tmp_path):
    snap = make_snapshot()
    snap.sections["encoder"]["provenance"] = []
    with pytest.raises(ConfigError, match="no provenance"):
        render_dashboard(snap, make_assets(tmp_path))
