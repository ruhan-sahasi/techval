"""inputs.read_json: a recorded file is read, or refused with its name and why."""

from __future__ import annotations

import gzip

import pytest

from techval.errors import ConfigError
from techval.inputs import read_json

from conftest import FIXTURES

# What the SEC serves a request it will not answer, saved under a payload's name.
SEC_REFUSAL = (
    "<html><head><title>SEC.gov | Request Rate Threshold Exceeded</title></head>"
    "<body>Your request has been identified as part of a network of automated "
    "tools.</body></html>"
)


def test_an_html_page_saved_as_a_payload_is_refused_by_name(tmp_path):
    path = tmp_path / "companyfacts_DDOG.json"
    path.write_text(SEC_REFUSAL, encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        read_json(path, "a companyfacts payload", dict)
    assert str(exc.value).startswith(
        f"{path} could not be read as a companyfacts payload: Expecting value"
    )


@pytest.mark.parametrize(
    "body, held",
    [
        ("[]", "an array"),
        ('"companyfacts"', "a string"),
        ("3", "a number"),
        ("null", "null"),
        ("true", "true or false"),
    ],
)
def test_json_of_the_wrong_shape_is_refused_by_what_it_holds(tmp_path, body, held):
    """An array where an object belongs raised an AttributeError further in."""
    path = tmp_path / "companyfacts_DDOG.json"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        read_json(path, "a companyfacts payload", dict)
    assert str(exc.value) == (
        f"{path} could not be read as a companyfacts payload: it holds {held}, "
        "not an object."
    )


def test_a_gzip_cut_short_or_not_gzip_at_all_is_refused(tmp_path):
    """EOFError and BadGzipFile, which no command caught."""
    whole = gzip.compress(b'{"payloads": {}, "meta": {}}')
    short = tmp_path / "short.json.gz"
    short.write_bytes(whole[:-8])
    plain = tmp_path / "plain.json.gz"
    plain.write_bytes(b'{"payloads": {}, "meta": {}}')
    for path in (short, plain):
        with pytest.raises(ConfigError, match="could not be read as a recorded fade"):
            read_json(path, "a recorded fade panel", dict)


def test_a_missing_file_or_a_folder_is_refused_as_unopenable(tmp_path):
    for path, reason in (
        (tmp_path / "companyfacts_NOPE.json", "No such file or directory"),
        (tmp_path, "Is a directory"),
    ):
        with pytest.raises(ConfigError) as exc:
            read_json(path, "a companyfacts payload")
        assert str(exc.value) == f"{path} could not be opened: {reason}."


def test_a_gzipped_object_is_read_as_a_plain_one():
    """The suffix decides, as it did in each reader this replaced."""
    path = FIXTURES / "fade_companyfacts.json.gz"
    blob = read_json(path, "a recorded fade panel", dict)
    assert {"payloads", "meta"} <= set(blob)
