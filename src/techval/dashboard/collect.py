"""Collect the snapshot: run every section's collector, cache it, and capture refusals.

Collection runs each section in ``COLLECT_ORDER`` against a ``CollectContext``
and assembles what they return into a ``Snapshot``. Three behaviours carry the
weight.

**A refusal is caught; a bug is not.** A section that raises ``TechvalError``
becomes ``status: "refused"`` with the message as its reason, which is how the
engine already reports a figure it could not source. Any other exception
propagates. Catching ``Exception`` would have been more forgiving and would
have turned a ``KeyError`` in a collector into a polite refusal on the page,
where it reads as a limit of the data rather than a defect of the code.

**Provenance is recorded, not written.** A section cannot return provenance
rows. It wraps each computation in ``ctx.record(figure_id, entry_point,
inputs)``, which times the block, checks that the entry point is a real
attribute of a real module, and checks that every input was declared in the
section's ``INPUTS``. A figure citing an entry point that does not resolve, or
a file the cache key does not cover, is a ``ValueError`` at collection time
rather than a false line on the page.

**The cache is keyed on evidence.** The encoder's ablation alone takes minutes,
so each section's result is cached under
``<ml cache dir>/dashboard/<id>-<digest>.json``. The digest covers the section
module's source bytes, the bytes of every declared input, the assumptions, and
for the sections that read other sections, those sections' results. A changed
fixture or a changed collector therefore misses exactly the section that reads
it. The entry points a section recorded are checked too: the source digest of
each entry point's module is stored with the entry and a hit is only a hit if
those modules are byte-identical, so a fix to the model code cannot be hidden
behind a figure the old code computed. What that check does not cover is code
the entry point imports from elsewhere; ``--no-cache`` is the answer when that
has changed. Refusals are never cached: they are cheap to reproduce and often
depend on the machine.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping

from ..config import Assumptions
from ..errors import ConfigError, TechvalError
from .sections import (
    COLLECT_ORDER,
    DEPENDENT_SECTIONS,
    MODEL_SECTIONS,
    SECTION_IDS,
    section_module,
)
from .snapshot import (
    Snapshot,
    fixture_digests,
    relative_key,
    to_jsonable,
    validate_section,
)

DEFAULT_ROOT = Path("tests/fixtures")
DEFAULT_TITLE = "Valuation engine and model results"

# Bumped when the cached payload's layout or meaning changes, which invalidates
# every entry at once.
CACHE_FORMAT = 1

# The keys a collector may return. id, title and provenance are the collector's
# to set, not the section's.
_RETURNABLE = ("status", "takeaway", "refusals", "headline", "figures")


@dataclass
class SectionRun:
    """What happened to one section during a collection, for the command to print."""

    id: str
    status: str
    seconds: float
    cache: str  # "hit", "miss", "off", or "n/a" when no key could be computed


class CollectContext:
    """What a section's ``collect`` is handed.

    ``root`` is the ml-data root the section's ``INPUTS`` are relative to.
    ``repo_root`` is the checkout, which snapshot paths are written relative to.
    ``results`` holds the sections already collected, as they will appear in the
    snapshot, for the sections that read other sections; it is a copy, so a
    section cannot alter another's result by reading it.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        repo_root: str | Path | None = None,
        assumptions: Assumptions | None = None,
        results: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.assumptions = assumptions if assumptions is not None else Assumptions()
        self.results: dict[str, dict[str, Any]] = copy.deepcopy(dict(results or {}))
        self.section_id: str | None = None
        self.declared: tuple[str, ...] = ()
        self.provenance: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ inputs

    def _normalise(self, name: str | Path) -> str:
        p = Path(name)
        if p.is_absolute():
            try:
                p = p.relative_to(self.root.resolve())
            except ValueError:
                try:
                    p = p.relative_to(self.root.absolute())
                except ValueError:
                    raise ValueError(
                        f"input {name} is not under the ml-data root {self.root}"
                    ) from None
        return PurePosixPath(p.as_posix()).as_posix()

    def _check_declared(self, name: str | Path) -> str:
        norm = self._normalise(name)
        for d in self.declared:
            if norm == d or norm.startswith(d.rstrip("/") + "/"):
                return norm
        raise ValueError(
            f"section {self.section_id!r} reads {norm!r}, which is not declared in "
            "its INPUTS. Declare it: an undeclared input is neither digested in the "
            "snapshot nor part of the cache key, so a change to it would leave a "
            "stale figure on the page."
        )

    def input(self, name: str | Path) -> Path:
        """The path of a declared input under the ml-data root."""
        return self.root / self._check_declared(name)

    # -------------------------------------------------------------- provenance

    @contextmanager
    def record(
        self, figure_id: str, entry_point: str, inputs: list[str | Path] | tuple = ()
    ) -> Iterator[None]:
        """Time the block and, if it completes, append a provenance row for the figure.

        ``entry_point`` is the dotted path of the function that computed the
        figure and must resolve. ``inputs`` are paths relative to the ml-data
        root, each inside a declared input; they are written repository-relative.
        A block that raises appends nothing, since the figure it was computing
        does not exist.
        """
        if not isinstance(figure_id, str) or not figure_id:
            raise ValueError("record needs a figure id")
        entry_module(entry_point)
        names = [self._check_declared(i) for i in inputs]
        start = time.perf_counter()
        yield
        self.provenance.append(
            {
                "figure": figure_id,
                "entry_point": entry_point,
                "inputs": [relative_key(self.root / n, self.repo_root) for n in names],
                "seconds": round(time.perf_counter() - start, 2),
            }
        )


def entry_module(entry_point: str) -> ModuleType:
    """The module an entry point lives in, after checking the attribute exists.

    ``techval.ml.encoder.evaluate_peer_encoder`` resolves to ``techval.ml.encoder``
    with ``evaluate_peer_encoder`` found on it. A dotted path that names nothing
    is a ``ValueError``: provenance that points at a function that does not
    exist is a false statement about where a figure came from.
    """
    parts = entry_point.split(".") if isinstance(entry_point, str) else []
    if len(parts) < 2 or not all(parts):
        raise ValueError(f"entry point {entry_point!r} is not a dotted module path")
    for cut in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:cut])
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # Only a miss on this very name means "try a shorter prefix". A
            # module that exists and fails to import one of ITS dependencies is
            # a real error and must surface as one.
            if exc.name != name and not name.startswith(f"{exc.name}."):
                raise
            continue
        obj: Any = module
        for attr in parts[cut:]:
            if not hasattr(obj, attr):
                raise ValueError(
                    f"entry point {entry_point!r}: {name} has no attribute "
                    f"{'.'.join(parts[cut:])!r}"
                )
            obj = getattr(obj, attr)
        return module
    raise ValueError(f"entry point {entry_point!r} does not name an importable module")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.strip()
    return text if out.returncode == 0 and text else None


def techval_commit(repo_root: Path) -> str:
    """``git rev-parse --short HEAD`` in the checkout, or ``"unknown"`` outside one."""
    return _git(["rev-parse", "--short", "HEAD"], repo_root) or "unknown"


def _repo_root_for(root: Path) -> Path:
    start = root if root.is_dir() else Path.cwd()
    top = _git(["rev-parse", "--show-toplevel"], start)
    return Path(top) if top else Path.cwd()


def ml_cache_dir(assumptions: Assumptions) -> Path:
    """The ml cache root, resolved the way ``techval peers`` resolves it."""
    configured = assumptions.ml.cache_dir
    return Path(configured).expanduser() if configured else Path.home() / ".techval" / "ml"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _module_digest(module: ModuleType) -> str:
    origin = getattr(module, "__file__", None)
    if not origin:
        raise ValueError(f"module {module.__name__} has no source file to digest")
    return _sha(Path(origin).read_bytes())


def _declared_digests(module: ModuleType, ctx: CollectContext) -> dict[str, str]:
    return fixture_digests([ctx.root / p for p in module.INPUTS], repo_root=ctx.repo_root)


def _missing_inputs(module: ModuleType, root: Path) -> list[str]:
    return [p for p in module.INPUTS if not (root / p).exists()]


def _assumptions_fingerprint(assumptions: Assumptions) -> Any:
    dumped = assumptions.model_dump(mode="json")
    # Where the cache lives does not change what is in it.
    dumped.get("ml", {}).pop("cache_dir", None)
    return dumped


def _results_fingerprint(results: Mapping[str, dict[str, Any]], section_id: str) -> Any:
    # Timings move on every recomputation and move no figure, so they are left
    # out: a scoreboard should not miss its cache because the encoder ran faster.
    out = {}
    for sid, section in sorted(results.items()):
        if sid == section_id:
            continue
        section = copy.deepcopy(section)
        for row in section.get("provenance", []):
            row.pop("seconds", None)
        out[sid] = section
    return out


def cache_key(
    module: ModuleType,
    input_digests: Mapping[str, str],
    assumptions: Assumptions,
    results: Mapping[str, dict[str, Any]],
) -> str:
    payload = {
        "format": CACHE_FORMAT,
        "id": module.ID,
        "source": _module_digest(module),
        "inputs": dict(sorted(input_digests.items())),
        "assumptions": _assumptions_fingerprint(assumptions),
        "results": (
            _results_fingerprint(results, module.ID)
            if module.ID in DEPENDENT_SECTIONS
            else None
        ),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return _sha(text.encode())[:16]


def _read_cache(path: Path) -> dict[str, Any] | None:
    """A cached section, or None when the entry is absent, unreadable or stale."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
        return None
    section = payload.get("section")
    code = payload.get("code")
    if not isinstance(code, dict):
        return None
    try:
        validate_section(section)
        for name, digest in code.items():
            if _module_digest(importlib.import_module(name)) != digest:
                return None
    except (ValueError, ImportError, OSError):
        return None
    return section


def _write_cache(path: Path, section: dict[str, Any]) -> None:
    code = {
        entry_module(row["entry_point"]).__name__: None for row in section["provenance"]
    }
    payload = {
        "format": CACHE_FORMAT,
        "code": {
            name: _module_digest(importlib.import_module(name)) for name in sorted(code)
        },
        "section": section,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _refused(module: ModuleType, why: str) -> dict[str, Any]:
    return {
        "id": module.ID,
        "title": module.TITLE,
        "takeaway": "",
        "status": "refused",
        "refusals": [{"what": module.TITLE, "why": why}],
        "headline": None,
        "figures": {},
        "provenance": [],
    }


def _normalise(module: ModuleType, returned: Any, ctx: CollectContext) -> dict[str, Any]:
    """Fill in what the collector owns, then hold the section to the schema."""
    where = f"section {module.ID!r}"
    if not isinstance(returned, dict):
        raise TypeError(f"{where}: collect returned {type(returned).__name__}, not a dict")
    stray = sorted(set(returned) - set(_RETURNABLE))
    if stray:
        raise ValueError(
            f"{where}: collect may return only {', '.join(_RETURNABLE)}; it returned "
            f"{', '.join(stray)}. id and title come from the module and provenance "
            "from ctx.record."
        )
    if "status" not in returned:
        raise ValueError(f"{where}: collect must return a status")
    section = to_jsonable(
        {
            "id": module.ID,
            "title": module.TITLE,
            "takeaway": returned.get("takeaway", ""),
            "status": returned["status"],
            "refusals": returned.get("refusals", []),
            "headline": returned.get("headline"),
            "figures": returned.get("figures", {}),
            "provenance": ctx.provenance,
        }
    )
    try:
        validate_section(section)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    if section["headline"] is not None and module.ID not in MODEL_SECTIONS:
        raise ValueError(
            f"{where}: only the model sections ({', '.join(MODEL_SECTIONS)}) carry a "
            "headline score against a baseline"
        )
    if section["status"] == "not_built" and (section["figures"] or section["headline"]):
        raise ValueError(f"{where}: a section that is not built cannot carry figures")
    return section


def _plan(sections: list[str] | tuple[str, ...] | None) -> list[str]:
    """The sections to run, in collection order, with the dependents they invalidate.

    Re-collecting a model section without re-collecting the scoreboard would
    leave the scoreboard showing the old score beside the new one, so every
    dependent section after the earliest requested one runs too.
    """
    if sections is None:
        return list(COLLECT_ORDER)
    requested = list(dict.fromkeys(sections))
    for sid in requested:
        section_module(sid)  # ConfigError naming the valid ids
    if not requested:
        raise ConfigError("no dashboard sections were named")
    first = min(COLLECT_ORDER.index(s) for s in requested)
    return [
        s
        for i, s in enumerate(COLLECT_ORDER)
        if s in requested or (s in DEPENDENT_SECTIONS and i > first)
    ]


def _check_kept_sections_current(
    base: Snapshot, kept: list[str], ctx: CollectContext
) -> None:
    """Refuse to merge into a snapshot whose kept sections read other fixture bytes.

    A merged snapshot writes one ``fixtures`` table for every section in it. If
    a section carried over from the old snapshot was collected from a fixture
    that has since changed, the table would certify bytes that section never
    read.
    """
    for sid in kept:
        module = section_module(sid)
        prefixes = [
            relative_key(ctx.root / p, ctx.repo_root) for p in module.INPUTS
        ]
        present = [p for p in module.INPUTS if (ctx.root / p).exists()]
        now = fixture_digests([ctx.root / p for p in present], repo_root=ctx.repo_root)
        then = {
            k: v
            for k, v in base.fixtures.items()
            if any(k == pre or k.startswith(pre.rstrip("/") + "/") for pre in prefixes)
        }
        if now != then:
            changed = sorted(
                k for k in set(now) | set(then) if now.get(k) != then.get(k)
            )
            raise ConfigError(
                f"section {sid!r} in the existing snapshot was collected from fixture "
                f"bytes that have changed since ({', '.join(changed)}). Collect it "
                f"again: add it to --sections, or collect every section."
            )


def _collected_at(value: str | date) -> str:
    text = value.isoformat() if isinstance(value, date) else str(value)
    try:
        date.fromisoformat(text)
    except ValueError:
        raise ConfigError(f"collected_at must be a YYYY-MM-DD date, got {value!r}") from None
    return text


# --------------------------------------------------------------------------- #
# the collection
# --------------------------------------------------------------------------- #


def collect_section(
    module: ModuleType,
    ctx: CollectContext,
    *,
    use_cache: bool = True,
) -> tuple[dict[str, Any], SectionRun]:
    """Run one section, or read it from the cache, and say which happened."""
    start = time.perf_counter()

    def run(section: dict[str, Any], cache: str) -> tuple[dict[str, Any], SectionRun]:
        seconds = round(time.perf_counter() - start, 2)
        return section, SectionRun(module.ID, section["status"], seconds, cache)

    missing = _missing_inputs(module, ctx.root)
    if missing:
        return run(
            _refused(
                module,
                "declared input(s) not found under the ml-data root "
                f"{ctx.root}: {', '.join(missing)}",
            ),
            "n/a",
        )

    path: Path | None = None
    cache_state = "off"
    if use_cache:
        key = cache_key(
            module, _declared_digests(module, ctx), ctx.assumptions, ctx.results
        )
        path = ml_cache_dir(ctx.assumptions) / "dashboard" / f"{module.ID}-{key}.json"
        cached = _read_cache(path)
        if cached is not None:
            return run(cached, "hit")
        cache_state = "miss"

    ctx.section_id = module.ID
    ctx.declared = tuple(PurePosixPath(p).as_posix() for p in module.INPUTS)
    ctx.provenance = []
    try:
        returned = module.collect(ctx)
    except TechvalError as exc:
        return run(_refused(module, str(exc)), cache_state)

    section = _normalise(module, returned, ctx)
    if path is not None:
        _write_cache(path, section)
    return run(section, cache_state)


def collect_snapshot(
    root: str | Path,
    collected_at: str | date,
    *,
    sections: list[str] | tuple[str, ...] | None = None,
    use_cache: bool = True,
    assumptions: Assumptions | None = None,
    repo_root: str | Path | None = None,
    base: Snapshot | None = None,
    on_section: Callable[[SectionRun], None] | None = None,
) -> Snapshot:
    """Collect every section, or the named ones merged into ``base``.

    ``collected_at`` is required and is never read from the clock here: the
    date on a snapshot is a statement the caller makes, and a collection run
    twice on the same fixtures must be able to produce the same file.

    With ``sections``, only those sections run (plus the dependent sections
    that read them), and every other section is carried over from ``base``
    after checking that its fixtures have not changed underneath it.
    ``on_section`` is called as each section finishes.
    """
    stamp = _collected_at(collected_at)
    root = Path(root)
    if not root.is_dir():
        raise ConfigError(f"ml-data root {root} is not a directory")
    repo = Path(repo_root) if repo_root is not None else _repo_root_for(root)
    assumptions = assumptions if assumptions is not None else Assumptions()

    plan = _plan(sections)
    results: dict[str, dict[str, Any]] = {}
    title = DEFAULT_TITLE
    if base is not None:
        title = base.title
        kept = [s for s in base.sections if s not in plan]
        _check_kept_sections_current(
            base, kept, CollectContext(root, repo, assumptions)
        )
        results.update({s: copy.deepcopy(base.sections[s]) for s in kept})

    for sid in plan:
        module = section_module(sid)
        ctx = CollectContext(root, repo, assumptions, results)
        section, run = collect_section(module, ctx, use_cache=use_cache)
        results[sid] = section
        if on_section is not None:
            on_section(run)

    ordered = {s: results[s] for s in SECTION_IDS if s in results}
    declared = [
        root / p for s in ordered for p in section_module(s).INPUTS if (root / p).exists()
    ]
    return Snapshot(
        title=title,
        techval_commit=techval_commit(repo),
        collected_at=stamp,
        fixtures=fixture_digests(declared, repo_root=repo),
        sections=ordered,
    )


__all__ = [
    "CollectContext",
    "DEFAULT_ROOT",
    "SectionRun",
    "collect_section",
    "collect_snapshot",
    "entry_module",
    "ml_cache_dir",
    "techval_commit",
]
