"""The fixture recorder writes exactly the files the KPI tasks look for."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from techval.rag.tasks import KPI_DOCUMENTS

SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "rag" / "record_filing_text.py"


def test_the_recorder_writes_the_files_the_tasks_expect():
    spec = importlib.util.spec_from_file_location("record_filing_text", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = {f"rag/filing_text_{t}_{module.FISCAL_YEAR}.json.gz" for t in module.TICKERS}
    assert written == {path for ticker, path in KPI_DOCUMENTS.items() if ticker != "DDOG"}
