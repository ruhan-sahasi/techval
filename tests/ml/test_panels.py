"""The fade panel loader and the disk cache model fits are kept in."""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_a_cached_fit_is_built_once_and_loaded_after(tmp_path):
    from techval.ml.panels import cached_fit

    panel = tmp_path / "panel.bin"
    panel.write_bytes(b"one")
    built = []

    def build():
        built.append(1)
        return {"fit": len(built)}

    modules = ("techval.ml.panels",)
    first = cached_fit("probe", [panel], modules, build, tmp_path / "cache")
    second = cached_fit("probe", [panel], modules, build, tmp_path / "cache")
    assert first == second == {"fit": 1}
    assert len(built) == 1
    assert len(list((tmp_path / "cache").glob("probe_*.joblib"))) == 1


def test_changed_panel_bytes_are_a_cache_miss(tmp_path):
    from techval.ml.panels import cached_fit, fit_key

    panel = tmp_path / "panel.bin"
    modules = ("techval.ml.panels",)
    panel.write_bytes(b"one")
    before = fit_key("probe", [panel], modules)
    panel.write_bytes(b"two")
    assert fit_key("probe", [panel], modules) != before
    built = []
    cached_fit("probe", [panel], modules, lambda: built.append(1) or "a", tmp_path / "cache")
    panel.write_bytes(b"three")
    cached_fit("probe", [panel], modules, lambda: built.append(1) or "b", tmp_path / "cache")
    assert len(built) == 2


def test_an_unreadable_entry_is_a_miss_and_refit_rebuilds(tmp_path):
    from techval.ml.panels import cached_fit, fit_key

    panel = tmp_path / "panel.bin"
    panel.write_bytes(b"one")
    modules = ("techval.ml.panels",)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"probe_{fit_key('probe', [panel], modules)}.joblib").write_bytes(b"not a pickle")
    assert cached_fit("probe", [panel], modules, lambda: "rebuilt", cache) == "rebuilt"
    assert cached_fit("probe", [panel], modules, lambda: "refit", cache, refit=True) == "refit"
    assert cached_fit("probe", [panel], modules, lambda: "never", cache) == "refit"


def test_no_cache_root_means_no_disk_cache(tmp_path):
    from techval.ml.panels import cached_fit

    calls = []
    for _ in range(2):
        cached_fit("probe", [], (), lambda: calls.append(1), None)
    assert len(calls) == 2


def test_the_fade_panel_cache_key_covers_every_module_the_build_runs():
    """A cached panel is stale the moment code it was built through changes.

    The key hashes the source of FADE_MODULES, so every techval module whose
    code runs while the panel is built has to be among them; the build is
    profiled on a few companies and the modules it actually entered are read
    off the profile.
    """
    import cProfile
    import gzip
    import json as _json

    import techval
    from techval.edgar import CompanyFacts
    from techval.ml import panels
    from techval.ml.forecast import build_fade_panel, fade_universe

    blob = _json.load(gzip.open(FIXTURES / panels.FADE_PANEL, "rt"))

    class Client:
        def company_facts(self, ticker):
            return CompanyFacts(blob["payloads"][ticker.upper()], ticker)

    universe = {t: v for t, v in list(fade_universe().items()) if t in blob["payloads"]}
    universe = dict(list(universe.items())[:5])
    profile = cProfile.Profile()
    profile.runcall(build_fade_panel, universe, Client())
    root = Path(techval.__file__).parent
    entered = set()
    for stat in profile.getstats():
        code = stat.code
        if isinstance(code, str):
            continue
        path = Path(code.co_filename)
        if root in path.parents:
            entered.add(".".join(("techval", *path.relative_to(root).with_suffix("").parts)))
    assert entered and entered <= set(panels.FADE_MODULES), entered - set(panels.FADE_MODULES)
