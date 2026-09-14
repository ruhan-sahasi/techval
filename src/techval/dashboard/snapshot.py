"""The results snapshot: what the dashboard shows, and where each figure came from.

The dashboard is a page rendered from a file, and the file is the contract. It
is collected once by running the package's own entry points against the
committed fixtures, written to ``docs/dashboard/snapshot.json``, and read back
by the renderer. Nothing on the page is typed in. A figure that is on the page
is in this file, and a figure in this file carries the entry point that
computed it, the inputs it read and the digest of every one of those inputs.

Three rules keep the file honest and its diffs readable.

**Serialisation is canonical.** Floats are rounded to six decimal places, keys
are sorted, the file ends in a newline, and NaN and infinity become null. A
re-collection that changes nothing therefore changes no byte, and one that
changes a score changes the lines that carry it and no others. Six places is
beyond the precision any figure here is reported at and well short of the
last-bit noise a refit on another machine produces.

**Nothing is coerced silently.** ``to_jsonable`` converts the types the models
actually return (numpy scalars and arrays, dates, dataclasses, tuples, paths)
and raises on anything else. Falling back to ``str()`` would have turned an
unexpected object into a plausible looking string on the page, which is the
dashboard's version of a guessed number.

**The schema version is checked, not assumed.** A snapshot written by a later
layout is refused by name rather than half-read, because a renderer that finds
a key missing and draws an empty chart has invented an absence.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from ..errors import ConfigError

SCHEMA_VERSION = 1

# Decimal places every float is rounded to before it is written.
FLOAT_DP = 6

STATUSES = ("ok", "refused", "not_built")
VERDICT_STATUSES = ("beats", "inside_noise", "ties", "loses", "not_significant")
HEADLINE_KEYS = (
    "metric",
    "score",
    "baseline_name",
    "baseline_score",
    "lift",
    "n",
    "higher_is_better",
    "verdict_status",
    "verdict_text",
)
SECTION_KEYS = (
    "id",
    "title",
    "takeaway",
    "status",
    "refusals",
    "headline",
    "figures",
    "provenance",
)
PROVENANCE_KEYS = ("figure", "entry_point", "inputs", "seconds")


@dataclass
class Snapshot:
    """One collection of every section's results, in the schema-1 layout.

    ``sections`` holds plain dictionaries rather than dataclasses because each
    one is produced by a different collector and read by a different renderer,
    and the JSON shape is the thing both sides agree on. ``validate_section``
    is what holds them to it.
    """

    title: str
    techval_commit: str
    collected_at: str
    fixtures: dict[str, str] = field(default_factory=dict)
    sections: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "schema": self.schema,
                "title": self.title,
                "techval_commit": self.techval_commit,
                "collected_at": self.collected_at,
                "fixtures": self.fixtures,
                "sections": self.sections,
            }
        )

    @classmethod
    def from_dict(cls, raw: Any, *, source: str = "snapshot") -> "Snapshot":
        if not isinstance(raw, dict):
            raise ConfigError(f"{source} is not a JSON object.")
        version = raw.get("schema")
        if version != SCHEMA_VERSION:
            raise ConfigError(
                f"{source} declares schema version {version!r} and this build of "
                f"techval reads version {SCHEMA_VERSION} only. Re-collect it with "
                "`techval dashboard --collect` rather than rendering a layout the "
                "renderer does not know."
            )
        missing = [
            k
            for k in ("title", "techval_commit", "collected_at", "fixtures", "sections")
            if k not in raw
        ]
        if missing:
            raise ConfigError(f"{source} is missing {', '.join(missing)}.")
        sections = raw["sections"]
        if not isinstance(sections, dict):
            raise ConfigError(f"{source}: sections is not an object.")
        for key, section in sections.items():
            try:
                validate_section(section)
            except ValueError as exc:
                raise ConfigError(f"{source}: section {key!r}: {exc}") from exc
            if section["id"] != key:
                raise ConfigError(
                    f"{source}: section stored under {key!r} says its id is "
                    f"{section['id']!r}."
                )
        return cls(
            title=raw["title"],
            techval_commit=raw["techval_commit"],
            collected_at=raw["collected_at"],
            fixtures=dict(raw["fixtures"]),
            sections=dict(sections),
            schema=version,
        )


# --------------------------------------------------------------------------- #
# canonical JSON
# --------------------------------------------------------------------------- #


def _float(value: float) -> float | None:
    # float() first: numpy's float64 subclasses float, and round() on one hands
    # back another float64 rather than the Python float the writer expects.
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    rounded = round(value, FLOAT_DP)
    # round(-1e-9, 6) is -0.0, which serialises as "-0.0" and would make two
    # collections of the same zero differ by a byte.
    return 0.0 if rounded == 0 else rounded


def _key(key: Any) -> str:
    if isinstance(key, enum.Enum):
        return _key(key.value)
    if isinstance(key, str):
        return str(key)
    if isinstance(key, bool):
        raise TypeError(f"refusing a boolean dictionary key: {key!r}")
    if isinstance(key, (datetime, date)):
        return key.isoformat()
    if hasattr(key, "item") and callable(key.item):  # numpy integer keys
        return _key(key.item())
    if isinstance(key, int):
        return str(key)
    raise TypeError(f"refusing a dictionary key of type {type(key).__name__}: {key!r}")


def to_jsonable(value: Any) -> Any:
    """Convert a result into plain JSON types, canonically, or raise.

    numpy scalars become Python scalars and numpy arrays become lists; floats
    are rounded to six places, with NaN and infinity written as null; dates and
    datetimes become ISO strings; dataclasses become dictionaries of their
    fields; tuples become lists; paths become POSIX strings. Any other type
    raises ``TypeError``, because a result the collector did not expect is a
    bug to fix and not a string to print.
    """
    if isinstance(value, enum.Enum):
        # Before str and int: a StrEnum or IntEnum member is both.
        return to_jsonable(value.value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, datetime):
        # pandas' NaT is a datetime that is not equal to itself.
        return None if value != value else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {_key(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    # numpy, detected by protocol so this module does not import it. ``tolist``
    # on an array and ``item`` on a scalar both return native Python types.
    module = type(value).__module__
    if module == "numpy" or module.startswith("numpy."):
        if type(value).__name__ == "datetime64":
            # ``item`` on a nanosecond datetime64 returns an int, which would be
            # written as a plausible looking number. Go through microseconds.
            if value != value:
                return None
            import numpy as np  # present whenever a numpy value is

            unit = np.datetime_data(value.dtype)[0]
            coarse = unit in ("Y", "M", "W", "D")
            return to_jsonable(
                value.astype("datetime64[D]" if coarse else "datetime64[us]").item()
            )
        if hasattr(value, "tolist") and getattr(value, "ndim", 0) > 0:
            return to_jsonable(value.tolist())
        if hasattr(value, "item"):
            return to_jsonable(value.item())
    raise TypeError(
        f"cannot put a {type(value).__module__}.{type(value).__name__} in a "
        f"snapshot: {value!r}"
    )


def dumps_snapshot(snapshot: Snapshot) -> str:
    """The canonical text of a snapshot: sorted keys, two-space indent, final newline."""
    return (
        json.dumps(
            snapshot.to_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def dump_snapshot(snapshot: Snapshot, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_snapshot(snapshot), encoding="utf-8", newline="\n")
    return path


def load_snapshot(path: str | Path) -> Snapshot:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no dashboard snapshot at {path}.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    return Snapshot.from_dict(raw, source=str(path))


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


def validate_section(section: Any) -> None:
    """Hold one section to the schema, or raise ``ValueError`` saying where it breaks.

    Two rules here are the cardinal rule made structural rather than left to
    each collector. A refused section must say why, since a refusal with no
    reason is indistinguishable from a broken page. And every figure must have
    at least one provenance row, and every provenance row must name a figure
    that exists, since a figure with no entry point behind it is exactly the
    typed-in number the dashboard is not allowed to show.
    """
    if not isinstance(section, dict):
        raise ValueError("a section must be an object")
    missing = [k for k in SECTION_KEYS if k not in section]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    extra = sorted(set(section) - set(SECTION_KEYS))
    if extra:
        raise ValueError(f"keys outside the schema: {', '.join(extra)}")
    for key in ("id", "title", "takeaway"):
        if not isinstance(section[key], str):
            raise ValueError(f"{key} must be a string")
    if section["status"] not in STATUSES:
        raise ValueError(
            f"status {section['status']!r} is not one of {', '.join(STATUSES)}"
        )

    refusals = section["refusals"]
    if not isinstance(refusals, list):
        raise ValueError("refusals must be a list")
    for r in refusals:
        if (
            not isinstance(r, dict)
            or set(r) != {"what", "why"}
            or not all(isinstance(r[k], str) and r[k].strip() for k in ("what", "why"))
        ):
            raise ValueError(f"a refusal must be {{what, why}} with both stated: {r!r}")
    if section["status"] == "refused" and not refusals:
        raise ValueError("a refused section must say what was refused and why")

    headline = section["headline"]
    if headline is not None:
        if not isinstance(headline, dict):
            raise ValueError("headline must be null or an object")
        missing = [k for k in HEADLINE_KEYS if k not in headline]
        extra = sorted(set(headline) - set(HEADLINE_KEYS))
        if missing or extra:
            raise ValueError(
                "headline keys do not match the schema"
                + (f"; missing {', '.join(missing)}" if missing else "")
                + (f"; unexpected {', '.join(extra)}" if extra else "")
            )
        if headline["verdict_status"] not in VERDICT_STATUSES:
            raise ValueError(
                f"verdict_status {headline['verdict_status']!r} is not one of "
                f"{', '.join(VERDICT_STATUSES)}"
            )

    figures = section["figures"]
    if not isinstance(figures, dict):
        raise ValueError("figures must be an object keyed by figure id")
    for fid, fig in figures.items():
        if not isinstance(fig, dict) or not isinstance(fig.get("kind"), str):
            raise ValueError(f"figure {fid!r} must be an object with a kind")
        if not isinstance(fig.get("title"), str) or "data" not in fig:
            raise ValueError(f"figure {fid!r} must carry a title and data")

    provenance = section["provenance"]
    if not isinstance(provenance, list):
        raise ValueError("provenance must be a list")
    for row in provenance:
        if not isinstance(row, dict) or set(row) != set(PROVENANCE_KEYS):
            raise ValueError(
                f"a provenance row must be {{{', '.join(PROVENANCE_KEYS)}}}: {row!r}"
            )
        if row["figure"] not in figures:
            raise ValueError(
                f"provenance names figure {row['figure']!r}, which the section "
                "does not contain"
            )
    sourced = {row["figure"] for row in provenance}
    unsourced = sorted(set(figures) - sourced)
    if unsourced:
        raise ValueError(
            "no provenance for figure(s) "
            + ", ".join(unsourced)
            + ": record each one with ctx.record(figure_id, entry_point, inputs)"
        )


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def _expand(path: Path) -> list[Path]:
    """A file, or every file under a directory, in a stable order.

    Directories are accepted because several fixtures are directories of
    recordings (``prices/``, ``mna/``) and a collector that reads one reads all
    of it. Dotfiles and bytecode caches are not evidence and are skipped, so a
    Finder ``.DS_Store`` cannot move a digest.
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p
            for p in path.rglob("*")
            if p.is_file()
            and not any(
                part.startswith(".") or part == "__pycache__"
                for part in p.relative_to(path).parts
            )
        )
    raise FileNotFoundError(path)


def relative_key(path: Path, repo_root: Path | None) -> str:
    """How a path is named in a snapshot: relative to the repository where it can be."""
    path = Path(path)
    if repo_root is not None:
        try:
            return path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def fixture_digests(
    paths: Iterable[str | Path], *, repo_root: str | Path | None = None
) -> dict[str, str]:
    """sha256 of each file's bytes, keyed by its repository-relative POSIX path.

    File contents rather than modification times, for the reason the model
    caches give: a fixture copied or regenerated byte for byte is the same
    evidence, and one that changed by a single value is not. A directory
    contributes one entry per file under it. A path that does not exist raises
    ``FileNotFoundError``; the collector decides whether that is a refusal.
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    out: dict[str, str] = {}
    for p in paths:
        for f in _expand(Path(p)):
            out[relative_key(f, root)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return dict(sorted(out.items()))
