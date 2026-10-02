"""The HTTP cache: forever by default, with an age limit for callers that want fresh data.

Valuation commands keep every response forever, because a number in a memo
must reproduce. A portfolio tracker wants the opposite for the same URLs, so
an age limit is a property of the cache a caller builds, never a change to the
default.
"""

from __future__ import annotations

import os
import time

from techval.edgar import HttpCache


def test_by_default_an_entry_never_expires(tmp_path):
    cache = HttpCache(root=tmp_path)
    cache.put("https://example.test/a", b"body")
    path = next(tmp_path.glob("*.cache"))
    ancient = time.time() - 10 * 365 * 86400
    os.utime(path, (ancient, ancient))
    assert cache.get("https://example.test/a") == b"body"


def test_an_entry_older_than_max_age_is_a_miss(tmp_path):
    cache = HttpCache(root=tmp_path, max_age=3600)
    cache.put("https://example.test/a", b"body")
    assert cache.get("https://example.test/a") == b"body"
    path = next(tmp_path.glob("*.cache"))
    stale = time.time() - 7200
    os.utime(path, (stale, stale))
    assert cache.get("https://example.test/a") is None


def test_a_stale_entry_is_overwritten_by_the_next_put(tmp_path):
    cache = HttpCache(root=tmp_path, max_age=3600)
    cache.put("https://example.test/a", b"old")
    path = next(tmp_path.glob("*.cache"))
    stale = time.time() - 7200
    os.utime(path, (stale, stale))
    cache.put("https://example.test/a", b"new")
    assert cache.get("https://example.test/a") == b"new"


def test_a_disabled_cache_stays_disabled_with_an_age_limit(tmp_path):
    cache = HttpCache(root=tmp_path, enabled=False, max_age=3600)
    cache.put("https://example.test/a", b"body")
    assert cache.get("https://example.test/a") is None
    assert not list(tmp_path.glob("*.cache"))
