"""The assumptions file as a user actually writes it.

Every other test in this suite builds ``Assumptions`` in Python, which is the
one way of constructing it that nobody uses in anger. A user writes YAML and
hands it to ``--config``, and the path from a line of YAML to a validated model
had three defects that only that path can reach: an obviously correct date was
rejected for being a date, a mistyped key produced a pydantic traceback rather
than a refusal, and two fields accepted a value and reached no code.

The three are one defect wearing three faces. The engine's rule is that it
refuses with a reason rather than either inventing or exploding, and a config
layer that explodes on a typo and stays silent on a setting it ignores keeps the
rule in neither direction.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.errors import ConfigError

EXAMPLE = Path(__file__).resolve().parents[1] / "assumptions.example.yaml"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "assumptions.yaml"
    path.write_text(body)
    return path


# --------------------------------------------------------------------------- #
# as_of, written the way a reader writes it
# --------------------------------------------------------------------------- #


def test_an_unquoted_iso_date_is_accepted_as_the_knowledge_date(tmp_path):
    """``as_of: 2026-09-11`` is a date to YAML and was a type error to pydantic.

    YAML 1.1 resolves an unquoted ISO date to a native ``datetime.date``, so the
    most natural spelling of this line arrived at the model as a date object and
    was rejected for not being a string. The reader saw a validation error on a
    line that looks obviously correct, which is the worst kind of error message:
    it points at the right line and gives the wrong reason.
    """
    path = _write(tmp_path, "as_of: 2026-09-11\n")
    assert Assumptions.load(path).as_of == "2026-09-11"


def test_a_quoted_iso_date_still_means_the_same_thing(tmp_path):
    """The spelling that already worked has to keep working."""
    path = _write(tmp_path, 'as_of: "2026-09-11"\n')
    assert Assumptions.load(path).as_of == "2026-09-11"


def test_both_spellings_produce_the_same_assumptions(tmp_path):
    """Not merely both accepted. The same run, so neither is a second-class form."""
    quoted = Assumptions.load(_write(tmp_path, 'as_of: "2026-09-11"\n'))
    unquoted = Assumptions.load(_write(tmp_path, "as_of: 2026-09-11\n"))
    assert quoted.model_dump() == unquoted.model_dump()


def test_a_timestamp_loses_its_time_rather_than_carrying_it(tmp_path):
    """A knowledge date is a date. YAML will hand over a datetime if one is written.

    The normalised value goes into filenames and into the EDGAR client, neither
    of which has any use for a time of day, so it is dropped here rather than
    somewhere further downstream where the failure would be a path with a colon
    in it.
    """
    path = _write(tmp_path, "as_of: 2026-09-11 13:45:00\n")
    assert Assumptions.load(path).as_of == "2026-09-11"


def test_a_date_passed_in_python_is_normalised_too():
    """The CLI's ``--as-of`` is a string, but the API is callable from a notebook."""
    assert Assumptions(as_of=date(2026, 9, 11)).as_of == "2026-09-11"


def test_a_date_that_is_not_a_date_is_still_refused(tmp_path):
    """Widening the field must not have widened it to anything at all."""
    with pytest.raises(ConfigError) as exc:
        Assumptions.load(_write(tmp_path, "as_of: [2026, 9, 11]\n"))
    assert "as_of" in str(exc.value)


# --------------------------------------------------------------------------- #
# a rejected file is a refusal, not a traceback
# --------------------------------------------------------------------------- #


def test_a_mistyped_key_is_a_refusal_naming_the_key(tmp_path):
    """Every command catches TechvalError and no command catches ValidationError.

    So a typo in a YAML file used to come back as a forty line traceback through
    the pydantic internals, with the one useful sentence at the bottom. The
    engine's own standard is a typed refusal carrying the reason and the fix.
    """
    path = _write(tmp_path, "dcf:\n  projectoin_years: 5\n")
    with pytest.raises(ConfigError) as exc:
        Assumptions.load(path)
    message = str(exc.value)
    assert "projectoin_years" in message
    assert str(path) in message


def test_a_refusal_names_every_field_that_failed(tmp_path):
    """One pass over the file, not one error at a time."""
    path = _write(tmp_path, "dcf:\n  projection_years: hello\ncomps:\n  peers: 4\n")
    with pytest.raises(ConfigError) as exc:
        Assumptions.load(path)
    message = str(exc.value)
    assert "dcf.projection_years" in message
    assert "comps.peers" in message


def test_a_hand_written_consistency_check_is_not_swallowed(tmp_path):
    """``DCFAssumptions`` raises ConfigError from inside a validator already.

    Those messages are the good ones in this file and the wrapper must not
    flatten them into a generic "was not accepted".
    """
    path = _write(tmp_path, "dcf:\n  terminal_growth: 0.09\n")
    with pytest.raises(ConfigError) as exc:
        Assumptions.load(path)
    assert "outgrows its own economy" in str(exc.value)


# --------------------------------------------------------------------------- #
# a setting that reaches no code says so
# --------------------------------------------------------------------------- #


def test_require_same_sub_vertical_refuses_rather_than_being_ignored(tmp_path):
    """It was the only field in the file that read as a setting and did nothing.

    The peer encoder gates its candidate list on market capitalisation and size
    ratio and on nothing else. A reader who set this got a ranking that was not
    what the assumptions file said it was, and got it silently, which is the
    exact failure mode this package refuses.
    """
    path = _write(tmp_path, "ml:\n  peers:\n    require_same_sub_vertical: true\n")
    with pytest.raises(ConfigError) as exc:
        Assumptions.load(path)
    message = str(exc.value)
    assert "not implemented" in message
    assert "max_size_ratio" in message


def test_overlapping_windows_refuses_to_be_switched_off(tmp_path):
    """Its own description promised behaviour behind a false that does not exist.

    ``signals.py`` computes the Newey-West standard error unconditionally and
    prints it beside the naive one, so false changed nothing. A reader who set
    it would have believed a naive t-statistic was what came back.
    """
    path = _write(tmp_path, "ml:\n  signals:\n    overlapping_windows: false\n")
    with pytest.raises(ConfigError) as exc:
        Assumptions.load(path)
    assert "cannot be set false" in str(exc.value)


@pytest.mark.parametrize(
    "body",
    [
        "ml:\n  peers:\n    require_same_sub_vertical: false\n",
        "ml:\n  signals:\n    overlapping_windows: true\n",
    ],
)
def test_the_default_value_of_each_is_still_writable(body, tmp_path):
    """Refusing a non-default must not refuse a file that spells out the default."""
    assert Assumptions.load(_write(tmp_path, body)) is not None


# --------------------------------------------------------------------------- #
# the worked example
# --------------------------------------------------------------------------- #


def test_the_committed_example_file_loads():
    """It is the first thing a reader copies, so it is the first thing to be right."""
    assert Assumptions.load(EXAMPLE).ticker == "DDOG"


def test_the_example_carries_every_capability_switch():
    """The file claims every judgment that moves a valuation is in it.

    Seven flags fold fitted sections into the value report and the example
    showed none of them, so a reader copying it could not discover that the
    capabilities existed at all. Each is named with its default so that the
    claim in the header is true.
    """
    text = EXAMPLE.read_text()
    for flag in (
        "sotp:",
        "enabled:",
        "propensity_enabled:",
        "precedents_enabled:",
        "as_of:",
    ):
        assert flag in text, f"{flag} is missing from the worked example"


def test_the_example_does_not_switch_a_fitted_section_on():
    """Copying the example must not silently put a fitted number in a valuation."""
    a = Assumptions.load(EXAMPLE)
    assert a.tmt.sotp is False
    assert a.ml.peers.enabled is False
    assert a.ml.forecast.enabled is False
    assert a.ml.warranted.enabled is False
    assert a.ml.signals.enabled is False
    assert a.ml.mna.propensity_enabled is False
    assert a.ml.mna.precedents_enabled is False
