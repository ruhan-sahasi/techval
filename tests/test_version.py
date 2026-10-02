"""techval --version, and one version number in two places that must agree."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

import techval
from techval.cli import app

ROOT = Path(__file__).resolve().parents[1]


def test_version_flag_prints_the_package_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"techval {techval.__version__}"


def test_pyproject_and_the_package_agree_on_the_version():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', text, flags=re.M).group(1)
    assert declared == techval.__version__
