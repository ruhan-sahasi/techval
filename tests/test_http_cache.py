"""The HTTP cache keeps every response for ever, because a number in a memo must reproduce."""

from __future__ import annotations

import os
import time

from techval.edgar import HttpCache


def test_an_entry_never_expires(tmp_path):
    cache = HttpCache(root=tmp_path)
    cache.put("https://example.test/a", b"body")
    path = next(tmp_path.glob("*.cache"))
    ancient = time.time() - 10 * 365 * 86400
    os.utime(path, (ancient, ancient))
    assert cache.get("https://example.test/a") == b"body"


def test_a_disabled_cache_reads_and_writes_nothing(tmp_path):
    cache = HttpCache(root=tmp_path, enabled=False)
    cache.put("https://example.test/a", b"body")
    assert cache.get("https://example.test/a") is None
    assert not list(tmp_path.glob("*.cache"))
