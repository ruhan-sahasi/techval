"""The committed demo snapshot regenerates from committed inputs, exactly."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "fixtures" / "invest" / "record_demo.py"
SNAPSHOT = ROOT / "docs" / "invest" / "snapshot.json"


def test_the_demo_snapshot_is_a_pure_function_of_the_fixtures():
    spec = importlib.util.spec_from_file_location("record_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rebuilt = module.build()
    committed = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert rebuilt == committed
