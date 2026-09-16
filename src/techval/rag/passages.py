"""Filing text cut into passages that can always be quoted back to the filing.

A passage carries its character range in the whole document, and its text is
that range, byte for byte. Everything downstream relies on the identity: a
reader cites a passage, the checker finds the quote inside it, and the page can
point a reader at the exact place in the filing. ``nlp.sections`` keeps the same
identity for the Items it splits, so a passage cut from an Item is offset by the
Item's own start.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A sentence ends at a full stop, question or exclamation mark, optionally
# followed by a closing quote or bracket, and then whitespace.
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]?\s")


@dataclass(frozen=True)
class Passage:
    id: str
    accession: str
    item: str | None
    start_char: int
    end_char: int
    text: str


def chunk(
    source: str,
    *,
    accession: str,
    item: str | None = None,
    base_offset: int = 0,
    size: int = 1200,
    overlap: int = 200,
    snap: int = 150,
) -> list[Passage]:
    """Windows of about ``size`` characters, each sharing ``overlap`` with the one before.

    A window that stops short of the end is cut back to the last sentence end
    inside its final ``snap`` characters, so a passage rarely stops mid-sentence.
    ``base_offset`` is where ``source`` starts in the document, so for every
    passage ``document[p.start_char:p.end_char] == p.text``. Windows holding
    nothing but whitespace are skipped.
    """
    if size <= snap or overlap < 0 or overlap >= size:
        raise ValueError(
            f"a window of {size} characters needs size above snap ({snap}) and "
            f"0 <= overlap < size (overlap is {overlap})"
        )
    out: list[Passage] = []
    start, n = 0, len(source)
    while start < n:
        end = min(start + size, n)
        if end < n:
            last = None
            for match in _SENTENCE_END.finditer(source, end - snap, end):
                last = match.end()
            if last is not None:
                end = last
        text = source[start:end]
        if text.strip():
            at = base_offset + start
            out.append(Passage(f"{accession}:{at}", accession, item, at, base_offset + end, text))
        if end >= n:
            break
        start = end - overlap if end - overlap > start else end
    return out
