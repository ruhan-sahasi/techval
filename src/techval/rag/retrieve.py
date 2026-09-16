"""BM25 over one document's passages: which passages a question should read.

One document is one index. A question about Datadog's customers is asked of
Datadog's filing and of nothing else, so no other filing, earlier or later, can
weight the terms, and the point-in-time rule holds by construction. The scoring
is Okapi BM25 with the textbook ``k1`` and ``b``. It is a keyword search, it
knows nothing about meaning, and that is what makes its misses visible: the
page reports how often the answer's passage was among those retrieved.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from .passages import Passage

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, passages: Sequence[Passage], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = list(passages)
        self.k1, self.b = k1, b
        self._counts = [Counter(tokens(p.text)) for p in self.passages]
        self._lengths = [sum(c.values()) for c in self._counts]
        self._average = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        n = len(self._counts)
        frequency = Counter(t for counts in self._counts for t in counts)
        self._idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def score(self, index: int, terms: Sequence[str]) -> float:
        counts, length = self._counts[index], self._lengths[index]
        norm = 1.0 - self.b + self.b * (length / self._average if self._average else 0.0)
        total = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if tf:
                total += self._idf[term] * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        return total

    def top_k(self, query: Sequence[str], k: int) -> list[Passage]:
        """The ``k`` best passages for the query, best first, ties to the earlier passage.

        Passages that share no term with the query are never returned, so a
        filing that never mentions the metric yields nothing rather than its
        first few pages.
        """
        terms = sorted({t for phrase in query for t in tokens(phrase)})
        ranked = sorted(
            ((self.score(i, terms), p.start_char, i) for i, p in enumerate(self.passages)),
            key=lambda row: (-row[0], row[1]),
        )
        return [self.passages[i] for score, _, i in ranked[:k] if score > 0]
