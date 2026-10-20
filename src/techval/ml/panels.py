"""The committed fade panel, and the disk cache model fits are kept in.

Building the point-in-time fade panel from its recorded payloads takes a few
seconds, and every command that reads base rates or scenarios needs it, so it
is built once per process and kept on disk between runs. A cached fit is keyed
on the bytes of its inputs and the source of the code that builds it, so it
goes stale the moment either changes.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
from functools import lru_cache
from pathlib import Path

# Bumped whenever what a cached fit means changes without its inputs changing.
FIT_SCHEMA = 1

FADE_PANEL = "fade_companyfacts.json.gz"

# The modules whose source moves the fade panel. A change to any of them is a
# cache miss, the same rule the dashboard collector keys its sections on. The
# panel is built through CompanyFacts and the tag ladders as much as through the
# forecast module, so those count; a test profiles the build to keep it so.
FADE_MODULES = (
    "techval.ml.forecast",
    "techval.ml.features",
    "techval.edgar",
    "techval.tags",
    "techval.config",
    "techval.tmt.taxonomy",
    "techval.ml.panels",
)


def fit_key(kind: str, inputs: list[Path], modules: tuple[str, ...], extra: str = "") -> str:
    """A key over the input bytes, the fitting code, the cache schema and anything else named."""
    h = hashlib.sha256(f"{kind}:{FIT_SCHEMA}:{extra}".encode())
    for path in inputs:
        h.update(Path(path).read_bytes())
    for name in modules:
        h.update(Path(importlib.import_module(name).__file__).read_bytes())
    return h.hexdigest()[:16]


def cached_fit(
    kind: str,
    inputs: list[Path],
    modules: tuple[str, ...],
    build,
    cache_root: Path | None,
    refit: bool = False,
    extra: str = "",
):
    """Load a fit from the disk cache, or build it and store it.

    An unreadable or unwritable entry is a miss, never a failed run; with no
    cache root there is no disk cache at all. ``extra`` carries anything else
    the build reads, such as the assumptions, into the key.
    """
    if cache_root is None:
        return build()
    import joblib

    path = Path(cache_root).expanduser() / f"{kind}_{fit_key(kind, inputs, modules, extra)}.joblib"
    if path.is_file() and not refit:
        try:
            return joblib.load(path)
        except Exception:  # noqa: BLE001 - an unreadable cache is a miss
            pass
    model = build()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path, compress=3)
    except Exception:  # noqa: BLE001 - an unwritable cache slows the next run only
        pass
    return model


def fade_panel(fixtures: Path, cache_root: Path | None = None, refit: bool = False):
    """The point-in-time fade panel, for base rates and scenarios."""
    return _fade_panel(Path(fixtures), None if cache_root is None else Path(cache_root), bool(refit))


@lru_cache(maxsize=4)
def _fade_panel(fixtures: Path, cache_root: Path | None, refit: bool):
    def build():
        from ..edgar import CompanyFacts
        from .forecast import build_fade_panel, fade_universe

        blob = json.load(gzip.open(fixtures / FADE_PANEL, "rt"))

        class _Client:
            def company_facts(self, ticker: str) -> CompanyFacts:
                return CompanyFacts(blob["payloads"][ticker.upper()], ticker)

        universe = {t: v for t, v in fade_universe().items() if t in blob["payloads"]}
        return build_fade_panel(universe, _Client())

    return cached_fit("fade_panel", [fixtures / FADE_PANEL], FADE_MODULES, build, cache_root, refit)
