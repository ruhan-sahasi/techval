"""Committed filing text, served point in time.

Two kinds of document are committed:

- **Annual reports.** Stored whole as ``filing_text_*.json.gz``, each payload
  carrying its own accession, form and filing date.
- **Merger filings.** The primary documents under ``merger/text/``. They are
  dated from the filer's own ``submissions_<TICKER>.json`` beside them, because
  a merger text with no date cannot be served point in time and is therefore
  not served at all.

``documents`` never returns a filing dated after the date asked about. A task
reads its own filing as of that filing's date, so a later amendment, proxy or
10-K cannot reach it.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from ..errors import MissingDataError
from ..nlp.sections import FilingSections, split_items

EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{bare}/"


@dataclass(frozen=True)
class Document:
    accession: str
    ticker: str
    form: str
    filed: date
    path: str
    url: str | None = None
    entity_name: str | None = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entity(root: Path, name: str) -> tuple[int | None, str | None]:
    path = root / name
    if not path.exists():
        return None, None
    facts = _read_json(path)
    return facts.get("cik"), facts.get("entityName")


def _annual_report(root: Path, rel: str) -> Document | None:
    path = root / rel
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        head = json.load(handle)
    ticker = str(head["ticker"]).upper()
    cik, name = _entity(root, f"companyfacts_{ticker}.json")
    url = head.get("url")
    if url is None and cik is not None:
        url = EDGAR_ARCHIVE.format(cik=cik, bare=head["accession"].replace("-", ""))
    return Document(
        accession=head["accession"],
        ticker=ticker,
        form=head["form"],
        filed=date.fromisoformat(head["filed"]),
        path=rel,
        url=url,
        entity_name=head.get("entity_name") or name,
    )


def _submissions(directory: Path) -> dict[str, tuple[str, str, date]]:
    """accession -> (ticker, form, filed), from every submissions file in the directory."""
    out: dict[str, tuple[str, str, date]] = {}
    for path in sorted(directory.glob("submissions_*.json")):
        ticker = path.stem.split("_", 1)[1].upper()
        recent = (_read_json(path).get("filings") or {}).get("recent") or {}
        for accession, form, filed in zip(
            recent.get("accessionNumber", []), recent.get("form", []), recent.get("filingDate", [])
        ):
            out[accession] = (ticker, form, date.fromisoformat(filed))
    return out


def _merger_documents(root: Path, merger_dir: str) -> list[Document]:
    directory = root / merger_dir
    manifest = directory / "MANIFEST.json"
    if not manifest.exists():
        return []
    dated = _submissions(directory)
    docs = []
    for name, entry in sorted(_read_json(manifest).get("files", {}).items()):
        if not name.startswith("text/"):
            continue
        accession = Path(name).name.split(".", 1)[0]
        if accession not in dated:
            continue
        ticker, form, filed = dated[accession]
        _, entity = _entity(directory, f"companyfacts_{ticker}.json")
        docs.append(
            Document(
                accession=accession,
                ticker=ticker,
                form=form,
                filed=filed,
                path=f"{merger_dir}/{name}",
                url=entry.get("source"),
                entity_name=entity,
            )
        )
    return docs


class FilingStore:
    def __init__(self, root: str | Path, documents: Iterable[Document]) -> None:
        self.root = Path(root)
        self._docs = {d.accession: d for d in documents}
        self._by_path = {d.path: d for d in self._docs.values()}
        self._text: dict[str, str] = {}
        self._splits: dict[str, FilingSections] = {}

    @classmethod
    def from_fixtures(
        cls,
        root: str | Path,
        *,
        annual_reports: Iterable[str] = (),
        merger_dir: str = "merger",
    ) -> FilingStore:
        root = Path(root)
        docs = [d for rel in annual_reports if (d := _annual_report(root, rel)) is not None]
        docs.extend(_merger_documents(root, merger_dir))
        return cls(root, docs)

    def get(self, accession: str) -> Document | None:
        return self._docs.get(accession)

    def by_path(self, path: str) -> Document | None:
        return self._by_path.get(path)

    def documents(
        self, ticker: str, as_of: date, forms: tuple[str, ...] | None = None
    ) -> list[Document]:
        """The ticker's documents filed on or before ``as_of``, newest first."""
        rows = [
            d
            for d in self._docs.values()
            if d.ticker == ticker.upper() and d.filed <= as_of and (forms is None or d.form in forms)
        ]
        return sorted(rows, key=lambda d: (d.filed, d.accession), reverse=True)

    def text(self, doc: Document) -> str:
        if doc.accession not in self._text:
            path = self.root / doc.path
            if doc.path.endswith(".json.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    body = json.load(handle)["text"]
            elif doc.path.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    body = handle.read()
            else:
                body = path.read_text(encoding="utf-8")
            self._text[doc.accession] = body
        return self._text[doc.accession]

    def section(self, doc: Document, item: str) -> tuple[str, int]:
        """One Item's text and where it starts in the document."""
        if doc.accession not in self._splits:
            self._splits[doc.accession] = split_items(self.text(doc), doc.form)
        found = self._splits[doc.accession].get(item)
        if found is None:
            raise MissingDataError(
                f"{doc.ticker} {doc.form} {doc.accession} has no Item {item} the section splitter could find"
            )
        return found.text, found.start_char
