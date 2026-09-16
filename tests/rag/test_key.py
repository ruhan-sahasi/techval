"""The answer key: a blind template, and a loader that holds every row to the filing."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.rag.key import COLUMNS, KEY_FILE, TO_FILL, has_answers, load_key, write_template
from techval.rag.run import prepare
from techval.rag.store import FilingStore
from techval.rag.tasks import KPI_DOCUMENTS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def prepared():
    store = FilingStore.from_fixtures(FIXTURES, annual_reports=KPI_DOCUMENTS.values())
    return prepare(store, Assumptions().ml.rag)[0]


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fill(path: Path, answers: dict[str, dict]) -> None:
    rows = _rows(path)
    for row in rows:
        row.update(answers.get(row["task_id"], {}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_the_template_lists_every_task_with_nothing_answered(tmp_path, prepared):
    path = tmp_path / KEY_FILE
    assert write_template(path, prepared) == len(prepared)
    rows = _rows(path)
    assert [r["task_id"] for r in rows] == sorted(p.task.task_id for p in prepared)
    assert list(rows[0]) == list(COLUMNS)
    assert all(r[c] == "" for r in rows for c in TO_FILL)
    assert not has_answers(path)
    payo = next(r for r in rows if r["task_id"] == "PAYO:cash_per_share")
    assert payo["unit"] == "usd_per_share" and payo["filing_url"].startswith("https://www.sec.gov/")
    assert payo["question"].startswith("How much cash does each share")


def test_the_template_shows_no_reader_output(tmp_path, prepared):
    write_template(tmp_path / KEY_FILE, prepared)
    body = (tmp_path / KEY_FILE).read_text(encoding="utf-8").lower()
    assert "claude" not in body and "regex" not in body and "refused" not in body


def test_no_key_file_means_no_key(tmp_path, prepared):
    assert load_key(tmp_path / KEY_FILE, prepared) is None


def test_an_unfilled_key_names_every_blank_row(tmp_path, prepared):
    path = tmp_path / KEY_FILE
    write_template(path, prepared)
    _fill(path, {"PAYO:cash_per_share": {"status": "stated", "value": "7.40", "quote": "the right to receive $7.40 in cash"}})
    state = load_key(path, prepared)
    assert not state.complete and has_answers(path)
    assert set(state.rows) == {"PAYO:cash_per_share"}
    assert len(state.unfilled) == len(prepared) - 1
    assert state.rows["PAYO:cash_per_share"].resolution == pytest.approx(0.01)


def test_every_stated_row_is_held_to_the_filing(tmp_path, prepared):
    path = tmp_path / KEY_FILE
    write_template(path, prepared)
    _fill(
        path,
        {
            "PAYO:cash_per_share": {"status": "stated", "value": "7.50", "quote": "the right to receive $7.40 in cash"},
            "PAYO:agreement_date": {"status": "stated", "text_value": "2026-06-12", "quote": "a sentence the filing never wrote"},
            "PAYO:exchange_ratio": {"status": "maybe"},
            "PAYO:consideration_form": {
                "status": "stated", "text_value": "cash",
                "quote": "the right to receive $7.40 in cash", "start_char": "5",
            },
            "PAYO:acquirer": {
                "status": "stated", "text_value": "Neon Maple Parent Inc.",
                "quote": "Neon Maple Parent Inc., a corporation incorporated pursuant to the laws of Canada",
            },
            "DDOG:arr": {"status": "stated", "value": "seven"},
        },
    )
    state = load_key(path, prepared)
    problems = "\n".join(state.problems)
    assert "PAYO:cash_per_share: the quote does not state 7.5" in problems
    assert "PAYO:agreement_date: the quote is not in the filing" in problems
    assert "PAYO:exchange_ratio: status 'maybe'" in problems
    assert "PAYO:consideration_form: the quote occurs at" in problems
    assert "DDOG:arr: value and start_char must be numbers" in problems
    acquirer = state.rows["PAYO:acquirer"]
    assert acquirer.text_value == "Neon Maple Parent Inc." and acquirer.span[1] > acquirer.span[0]


def test_a_fully_answered_key_is_complete(tmp_path, prepared):
    path = tmp_path / KEY_FILE
    write_template(path, prepared)
    _fill(path, {p.task.task_id: {"status": "not_stated"} for p in prepared})
    state = load_key(path, prepared)
    assert state.complete and all(r.status == "not_stated" for r in state.rows.values())


def test_a_task_the_key_leaves_out_is_reported(tmp_path, prepared):
    path = tmp_path / KEY_FILE
    write_template(path, prepared[:-1])
    state = load_key(path, prepared)
    assert state.missing == [prepared[-1].task.task_id] and not state.complete
