"""Ideas: the screen's rich and cheap, led by the signal's own failure.

The pane exists to be the honest version of a recommendations tab: the
residuals are real, the ordering is real, and the first thing on it is the
measured fact that this score lost to a random one on forward returns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from techval.invest.engine_read import warranted_model
from techval.invest.ideas import SIGNAL_VERDICT, ideas

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def model():
    return warranted_model(FIXTURES)


def test_ideas_split_rich_from_cheap_and_exclude_held_names(model):
    everything = ideas(model, held=set(), n=6)
    assert len(everything["cheap"]) == 6 and len(everything["rich"]) == 6
    assert all(i.residual_log < 0 for i in everything["cheap"])
    assert all(i.residual_log > 0 for i in everything["rich"])
    excluded = everything["cheap"][0].ticker
    without = ideas(model, held={excluded}, n=6)
    assert excluded not in [i.ticker for i in without["cheap"]]


def test_cheapest_first_and_richest_first(model):
    out = ideas(model, held=set(), n=8)
    cheap = [i.residual_log for i in out["cheap"]]
    rich = [i.residual_log for i in out["rich"]]
    assert cheap == sorted(cheap)
    assert rich == sorted(rich, reverse=True)


def test_the_verdict_travels_verbatim(model):
    out = ideas(model, held=set(), n=3)
    assert out["verdict"] == SIGNAL_VERDICT
    assert "-0.0984" in SIGNAL_VERDICT
    assert "not significant" in SIGNAL_VERDICT.lower()
    assert "techval signal" in SIGNAL_VERDICT
    assert out["as_of"] >= "2026-01-01"


def test_two_calls_agree(model):
    assert ideas(model, held={"NET"}, n=5) == ideas(model, held={"NET"}, n=5)
