"""techval cache: see what the HTTP cache holds, and prune what has aged out."""

from __future__ import annotations

import os
import time
from pathlib import Path

from typer.testing import CliRunner

from techval.commands_cache import app

runner = CliRunner()
DAY = 86400


def flat(text: str) -> str:
    return " ".join(text.split())


def seed(root: Path) -> None:
    """Three entries: one fresh, two old, and a leftover temp file."""
    root.mkdir()
    now = time.time()
    for name, age_days, size in (("a", 1, 1000), ("b", 40, 2000), ("c", 400, 3000)):
        path = root / f"{name}.cache"
        path.write_bytes(b"x" * size)
        os.utime(path, (now - age_days * DAY, now - age_days * DAY))
    (root / "d.tmp").write_bytes(b"y" * 10)


def test_info_counts_entries_and_bytes(tmp_path):
    root = tmp_path / "cache"
    seed(root)
    result = runner.invoke(app, ["info", "--root", str(root)])
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "3 entries" in text
    assert "5.9 KB" in text or "6.0 KB" in text
    assert "1 unfinished write" in text


def test_prune_removes_only_entries_older_than_the_cutoff(tmp_path):
    root = tmp_path / "cache"
    seed(root)
    result = runner.invoke(app, ["prune", "--root", str(root), "--older-than", "30"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in root.iterdir()) == ["a.cache"]
    assert "Removed 2 entries" in flat(result.output)


def test_dry_run_removes_nothing_and_says_what_it_would(tmp_path):
    root = tmp_path / "cache"
    seed(root)
    result = runner.invoke(app, ["prune", "--root", str(root), "--older-than", "30", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert len(list(root.iterdir())) == 4
    assert "Would remove 2 entries" in flat(result.output)


def test_a_missing_cache_is_reported_not_an_error(tmp_path):
    result = runner.invoke(app, ["info", "--root", str(tmp_path / "nowhere")])
    assert result.exit_code == 0, result.output
    assert "no cache" in flat(result.output).lower()
