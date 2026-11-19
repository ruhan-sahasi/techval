"""Every text file the package reads or writes names its encoding.

Without one, Python reads and writes with the locale's encoding, which is UTF-8
on macOS and Linux and cp1252 on a stock Windows install, so the same file
decodes one way here and another there. Binary modes are exempt, which covers
``config.read_yaml``: it hands the parser bytes so that YAML settles the
encoding itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "techval"

# The mode each opener takes when none is passed, and where its mode sits.
OPENERS = {"open": ("r", 1), "io.open": ("r", 1), "gzip.open": ("rb", 1)}


def _mode(call: ast.Call, func: str) -> str:
    default, position = OPENERS.get(func, ("r", 0))  # Path.open takes the mode first
    for keyword in call.keywords:
        if keyword.arg == "mode":
            return ast.literal_eval(keyword.value)
    if len(call.args) > position:
        return ast.literal_eval(call.args[position])
    return default


def _unnamed(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if any(keyword.arg == "encoding" for keyword in node.keywords):
            continue
        func = ast.unparse(node.func)
        name = func.rsplit(".", 1)[-1]
        opens_text = name == "open" and func != "webbrowser.open"
        opens_text = opens_text and "b" not in _mode(node, func)
        if name in ("read_text", "write_text") or opens_text:
            found.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    return found


def test_every_text_read_and_write_in_the_package_names_its_encoding():
    found = [hit for path in sorted(PACKAGE.rglob("*.py")) for hit in _unnamed(path)]
    assert found == [], "no encoding named at " + ", ".join(found)
