"""BM25 over one document: the passage that answers ranks first, and nothing else leaks in."""

from __future__ import annotations

from techval.rag.passages import Passage
from techval.rag.retrieve import BM25, tokens


def _p(at: int, text: str) -> Passage:
    return Passage(f"A:{at}", "A", None, at, at + len(text), text)


CORPUS = [
    _p(0, "The company had approximately 32,700 customers at year end."),
    _p(100, "Revenue grew and the company invested in research."),
    _p(200, "Customers with ARR of $100,000 or more numbered 4,310."),
    _p(300, "The company describes its platform and its products."),
]


def test_tokens_keep_numbers_whole():
    assert tokens("About 32,700 customers, 1.5 billion") == ["about", "32,700", "customers", "1.5", "billion"]


def test_the_passages_that_answer_rank_first_and_ties_go_to_the_earlier():
    top = BM25(CORPUS).top_k(["customers", "customer count"], 2)
    assert [p.start_char for p in top] == [0, 200]


def test_passages_sharing_no_term_are_never_returned():
    assert BM25(CORPUS).top_k(["subscribers"], 3) == []


def test_a_rare_term_outweighs_a_common_one():
    index = BM25(CORPUS)
    assert index.score(2, ["arr"]) > index.score(1, ["company"])


def test_the_same_query_gives_the_same_order_every_time():
    index = BM25(CORPUS)
    first = [p.id for p in index.top_k(["company", "customers"], 4)]
    assert all([p.id for p in index.top_k(["customers", "company"], 4)] == first for _ in range(3))
