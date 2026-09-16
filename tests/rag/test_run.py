"""One pass: every committed task retrieved once, within its own filing, and read three ways."""

from __future__ import annotations

from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.rag.run import prepare, read_all, requests
from techval.rag.store import FilingStore
from techval.rag.tasks import KPI_DOCUMENTS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SETTINGS = Assumptions().ml.rag


@pytest.fixture(scope="module")
def store() -> FilingStore:
    return FilingStore.from_fixtures(FIXTURES, annual_reports=KPI_DOCUMENTS.values())


@pytest.fixture(scope="module")
def prepared(store):
    return prepare(store, SETTINGS)[0]


@pytest.fixture(scope="module")
def run(store, tmp_path_factory):
    return read_all(store, SETTINGS, tmp_path_factory.mktemp("recordings"))


def test_every_passage_comes_from_its_task_s_own_input(prepared):
    assert len(prepared) >= 49
    for p in prepared:
        assert len(p.passages) <= SETTINGS.top_k
        for x in p.passages:
            assert x.accession == p.task.accession
            assert p.text[x.start_char - p.base : x.end_char - p.base] == x.text


def test_kpis_read_item_7_and_deals_read_the_whole_announcement(store, prepared):
    ddog = next(p for p in prepared if p.task.task_id == "DDOG:customers")
    assert ddog.base > 0 and "dollar-based net retention" in ddog.text
    payo = next(p for p in prepared if p.task.task_id == "PAYO:cash_per_share")
    assert payo.base == 0 and payo.text == store.text(store.get(payo.task.accession))
    assert any("$7.40" in x.text for x in payo.passages)


def test_a_request_is_built_only_where_something_was_retrieved(prepared):
    assert [t for t, _ in requests(prepared, SETTINGS)] == [p.task.task_id for p in prepared if p.passages]


def test_without_recordings_the_claude_reader_is_missing_and_the_regex_readings_stand(run):
    assert run.missing == sorted(p.task.task_id for p in run.prepared if p.passages)
    assert set(run.readings) == {"claude", "regex_retrieved", "regex_native"}
    ids = {p.task.task_id for p in run.prepared}
    assert all(set(by_task) == ids for by_task in run.readings.values())
    assert run.readings["regex_native"]["PAYO:cash_per_share"].value == 7.4
    assert run.recordings == {}


def test_the_same_inputs_give_the_same_readings(store, run, tmp_path):
    assert read_all(store, SETTINGS, tmp_path).readings == run.readings
