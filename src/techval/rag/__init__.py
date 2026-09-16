"""A retrieval-augmented reader for facts stated in SEC filings.

The package asks one question of one filing at a time: how much cash a target's
holders receive, how many customers a company states, and so on. It answers it
three ways, over the same retrieved passages: a recorded Claude reader answers
once, and the regex rules techval already has answer twice, on the passages
and on the whole Item or document. An owner-written answer key then says which
was right.

Nothing here is a live dependency. The Claude reader's requests and responses
are committed as fixtures and replayed, so the dashboard and the tests read
them offline with no SDK and no key. Only ``techval rag record`` calls the API.
Nothing here changes a valuation, a label or a model score either: the readings
are measured here, and used elsewhere only if the measurement supports it.
"""
