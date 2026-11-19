"""edgar.strip_markup: what a filing loses on the way to text, and what it keeps."""

from __future__ import annotations

import lxml.html

from techval.edgar import strip_markup

COVER = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION"

# The shape of an inline XBRL 10-K's opening: the header, hidden from a reader,
# holding a hidden fact and a context, and then the cover page.
INLINE_XBRL = (
    b'<html><body><div style="display:none"><ix:header><ix:hidden>'
    b'<ix:nonNumeric name="dei:AmendmentFlag">false</ix:nonNumeric></ix:hidden>'
    b'<ix:resources><xbrli:context id="c-1"><xbrli:entity>0001561550'
    b"</xbrli:entity></xbrli:context></ix:resources></ix:header></div>"
    b"<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p></body></html>"
)


def test_the_text_after_a_dropped_script_or_style_is_kept():
    """lxml's ``remove`` takes an element's tail, which is the text after it."""
    body = (
        b"<html><body><p>Revenue grew <script>track()</script>12% in 2025.</p>"
        b"<style>p { margin: 0 }</style>Gross margin was 81%.</body></html>"
    )
    text = strip_markup(body)
    assert "Revenue grew 12% in 2025." in text
    assert "Gross margin was 81%." in text
    assert "track()" not in text
    assert "margin: 0" not in text


def test_the_inline_xbrl_header_is_dropped():
    """The namespace query meant to drop it matched nothing in the parsed page."""
    assert strip_markup(INLINE_XBRL) == COVER


def test_an_html_header_element_is_kept():
    """Only the iXBRL header is dropped, not HTML's own header element."""
    body = b"<html><body><header>Annual Report</header><p>2025</p></body></html>"
    assert strip_markup(body) == "Annual Report2025"


def test_the_fallback_for_a_page_lxml_refuses_drops_the_same_parts(monkeypatch):
    def refuse(_body):
        raise ValueError("unparseable")

    monkeypatch.setattr(lxml.html, "fromstring", refuse)
    assert strip_markup(INLINE_XBRL) == COVER
