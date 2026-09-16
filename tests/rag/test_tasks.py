"""The task list: every committed filing becomes its questions, and a missing one is named."""

from __future__ import annotations

from pathlib import Path

import pytest

from techval.rag.readings import UNITS
from techval.rag.store import FilingStore
from techval.rag.tasks import DEAL_METRICS, DEALS, KPI_DOCUMENTS, KPI_METRICS, METRICS, build_tasks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def built():
    store = FilingStore.from_fixtures(FIXTURES, annual_reports=KPI_DOCUMENTS.values())
    return build_tasks(store)


def test_the_nine_deals_become_forty_five_tasks(built):
    tasks, _ = built
    deals = [t for t in tasks if t.kind == "deal"]
    assert len(deals) == 45 == len(DEALS) * len(DEAL_METRICS)
    assert {t.ticker for t in deals} == {ticker for ticker, _ in DEALS}
    assert all(t.item is None and t.form == "8-K" for t in deals)


def test_kpi_tasks_exist_exactly_for_the_committed_annual_reports(built):
    tasks, unresolved = built
    kpis = [t for t in tasks if t.kind == "kpi"]
    assert all(t.item == "7" and t.form == "10-K" for t in kpis)
    for ticker, path in KPI_DOCUMENTS.items():
        committed = (FIXTURES / path).exists()
        mine = [t for t in kpis if t.ticker == ticker]
        assert len(mine) == (len(KPI_METRICS) if committed else 0), ticker
        assert any(line.startswith(f"{ticker}:") for line in unresolved) != committed, ticker
    assert "DDOG" in {t.ticker for t in kpis}


def test_task_ids_are_unique_sorted_and_read_as_of_their_own_filing(built):
    tasks, _ = built
    ids = [t.task_id for t in tasks]
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    assert all(t.as_of == t.filed for t in tasks)
    assert next(t for t in tasks if t.task_id == "PAYO:acquirer").target_name == "Payoneer Global Inc."


def test_every_metric_is_fully_described():
    assert set(METRICS) == set(DEAL_METRICS) | set(KPI_METRICS)
    for name, m in METRICS.items():
        assert m.name == name and m.unit in UNITS
        assert m.question.endswith("?") and m.definition and m.terms and m.label
        assert m.kind == ("deal" if name in DEAL_METRICS else "kpi")
