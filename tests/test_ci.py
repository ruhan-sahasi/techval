"""The CI workflow tests every Python version the package says it supports."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_the_matrix_covers_requires_python():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))
    versions = workflow["jobs"]["pytest"]["strategy"]["matrix"]["python"]
    floor = re.search(r'requires-python = ">=(\d+\.\d+)"', (ROOT / "pyproject.toml").read_text()).group(1)
    assert floor in versions, f"requires-python is >={floor} and CI never runs {floor}"
