"""The snapshot file: canonical, lossless, and strict about what it accepts.

The dashboard is only as trustworthy as the file it is rendered from, so the
properties tested here are the ones a reviewer relies on when reading a diff of
``docs/dashboard/snapshot.json``: that writing and reading it loses nothing,
that the same results always produce the same bytes, that the types the models
return are converted rather than stringified, and that a file from a layout this
build does not know is refused by name instead of half-drawn.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from techval.dashboard.snapshot import (
    SCHEMA_VERSION,
    Snapshot,
    dump_snapshot,
    dumps_snapshot,
    fixture_digests,
    load_snapshot,
    to_jsonable,
    validate_section,
)
from techval.errors import ConfigError


def _section(sid: str = "encoder", **over) -> dict:
    base = {
        "id": sid,
        "title": "Peer encoder",
        "takeaway": "The encoder ranks disclosed peers above the popularity prior.",
        "status": "ok",
        "refusals": [],
        "headline": {
            "metric": "ndcg@10",
            "score": 0.5407,
            "baseline_name": "popularity prior",
            "baseline_score": 0.2213,
            "lift": 0.3194,
            "n": 162,
            "higher_is_better": True,
            "verdict_status": "beats",
            "verdict_text": "ndcg@10 of 0.5407 against 0.2213 for popularity prior.",
        },
        "figures": {
            "baselines": {
                "kind": "hbar",
                "title": "Every baseline on the same queries",
                "subtitle": "ndcg@10, n = 162",
                "data": {"rows": [{"label": "encoder", "value": 0.5407}]},
            }
        },
        "provenance": [
            {
                "figure": "baselines",
                "entry_point": "techval.ml.encoder.evaluate_peer_encoder",
                "inputs": ["tests/fixtures/peer_panel_tmt.json"],
                "seconds": 37.1,
            }
        ],
    }
    base.update(over)
    return base


def _snapshot() -> Snapshot:
    return Snapshot(
        title="techval Results",
        techval_commit="b37dde5",
        collected_at="2026-09-14",
        fixtures={"tests/fixtures/peer_panel_tmt.json": "ab" * 32},
        sections={"encoder": _section()},
    )


def test_round_trip_is_lossless(tmp_path):
    snap = _snapshot()
    path = dump_snapshot(snap, tmp_path / "nested" / "snapshot.json")
    back = load_snapshot(path)
    assert back == snap
    assert dumps_snapshot(back) == path.read_text()


def test_file_is_canonical(tmp_path):
    text = dump_snapshot(_snapshot(), tmp_path / "s.json").read_text()
    assert text.endswith("}\n") and not text.endswith("\n\n")
    raw = json.loads(text)
    assert list(raw) == sorted(raw)
    assert raw["schema"] == SCHEMA_VERSION
    # Insertion order must not reach the file.
    snap = _snapshot()
    snap.fixtures = {"z.json": "1", "a.json": "2"}
    again = _snapshot()
    again.fixtures = {"a.json": "2", "z.json": "1"}
    assert dumps_snapshot(snap) == dumps_snapshot(again)


def test_to_jsonable_converts_numpy_nan_and_dates():
    class Colour(enum.Enum):
        BLUE = "blue"

    @dataclass
    class Point:
        at: date
        value: float
        tags: tuple[str, ...]

    out = to_jsonable(
        {
            "i": np.int64(7),
            "f": np.float64(0.123456789),
            "b": np.bool_(True),
            "arr": np.array([[1.0, np.nan], [np.inf, -np.inf]]),
            "nan": float("nan"),
            "negzero": -1e-12,
            "d": date(2026, 9, 14),
            "dt": datetime(2026, 9, 14, 8, 30),
            "ts": pd.Timestamp("2026-01-01"),
            "nat": pd.NaT,
            "np_date": np.datetime64("2026-03-31"),
            "np_ns": np.datetime64("2026-03-31T00:00:00.000000000"),
            "np_nat": np.datetime64("NaT"),
            "tuple": (1, 2.0000004),
            "enum": Colour.BLUE,
            "path": Path("tests") / "fixtures" / "x.json",
            "dc": Point(date(2025, 1, 2), 1 / 3, ("a", "b")),
            date(2024, 12, 31): "date key",
            np.int64(3): "numpy key",
        }
    )
    assert out["i"] == 7 and type(out["i"]) is int
    assert out["f"] == 0.123457 and type(out["f"]) is float
    assert out["b"] is True
    assert out["arr"] == [[1.0, None], [None, None]]
    assert out["nan"] is None
    assert out["negzero"] == 0.0 and str(out["negzero"]) == "0.0"
    assert out["d"] == "2026-09-14"
    assert out["dt"] == "2026-09-14T08:30:00"
    assert out["ts"] == "2026-01-01T00:00:00"
    assert out["nat"] is None
    assert out["np_date"] == "2026-03-31"
    assert out["np_ns"] == "2026-03-31T00:00:00"
    assert out["np_nat"] is None
    assert out["tuple"] == [1, 2.0]
    assert out["enum"] == "blue"
    assert out["path"] == "tests/fixtures/x.json"
    assert out["dc"] == {"at": "2025-01-02", "value": 0.333333, "tags": ["a", "b"]}
    assert out["2024-12-31"] == "date key"
    assert out["3"] == "numpy key"
    # And the result is strict JSON.
    json.dumps(out, allow_nan=False)


def test_to_jsonable_refuses_unknown_types():
    with pytest.raises(TypeError, match="object"):
        to_jsonable({"x": object()})
    with pytest.raises(TypeError):
        to_jsonable({"s": {1, 2}})


def test_unknown_schema_version_is_refused_by_name(tmp_path):
    raw = _snapshot().to_dict()
    raw["schema"] = 2
    path = tmp_path / "s.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="schema version 2"):
        load_snapshot(path)


def test_missing_and_malformed_files_are_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="no dashboard snapshot"):
        load_snapshot(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_snapshot(bad)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"status": "maybe"}, "status"),
        ({"status": "refused"}, "must say what was refused"),
        ({"refusals": [{"what": "x", "why": ""}]}, "both stated"),
        ({"headline": {"metric": "auc"}}, "headline keys"),
        ({"provenance": []}, "no provenance for figure"),
        (
            {
                "provenance": [
                    {"figure": "ghost", "entry_point": "m.f", "inputs": [], "seconds": 1.0}
                ]
            },
            "does not contain",
        ),
        ({"extra": 1}, "outside the schema"),
    ],
)
def test_section_structure_is_enforced(change, message):
    with pytest.raises(ValueError, match=message):
        validate_section(_section(**change))


def test_a_bad_section_in_a_file_is_a_config_error(tmp_path):
    raw = _snapshot().to_dict()
    raw["sections"]["encoder"]["provenance"] = []
    path = tmp_path / "s.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="encoder"):
        load_snapshot(path)


def test_fixture_digests_hash_bytes_and_expand_directories(tmp_path):
    (tmp_path / "prices").mkdir()
    (tmp_path / "prices" / "B.csv").write_bytes(b"b")
    (tmp_path / "prices" / "A.csv").write_bytes(b"a")
    (tmp_path / "prices" / ".DS_Store").write_bytes(b"noise")
    (tmp_path / "panel.json").write_bytes(b"{}")

    out = fixture_digests(
        [tmp_path / "panel.json", tmp_path / "prices"], repo_root=tmp_path
    )
    assert out == {
        "panel.json": hashlib.sha256(b"{}").hexdigest(),
        "prices/A.csv": hashlib.sha256(b"a").hexdigest(),
        "prices/B.csv": hashlib.sha256(b"b").hexdigest(),
    }
    with pytest.raises(FileNotFoundError):
        fixture_digests([tmp_path / "absent.json"], repo_root=tmp_path)


def test_committed_fixture_is_keyed_repository_relative():
    repo = Path(__file__).resolve().parents[2]
    out = fixture_digests([repo / "tests/fixtures/peer_groups_tmt.json"], repo_root=repo)
    assert list(out) == ["tests/fixtures/peer_groups_tmt.json"]
