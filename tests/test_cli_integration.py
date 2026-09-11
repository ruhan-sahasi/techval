"""Everything is mounted, and the base valuation does not move when it is.

Two claims are tested here and the second one is the reason the file exists.

**Every command is reachable.** Seventeen thousand lines of ``techval.tmt`` and
``techval.ml`` were unreachable from the command line until the sub-apps were
mounted, and a module nobody can invoke is indistinguishable from one that does
not work. The mount test walks the four command modules and asserts that each of
their commands appears in ``techval --help`` and answers ``--help`` itself, so a
module that is added later and not mounted fails here rather than being
discovered by a user who cannot find it.

**The base report does not move.** This is the load-bearing test. The premise of
this repository is that a discounted cash flow and a comp table trace to
filings, and the optional machinery bolted on beside them is allowed to be
switched on by a reader who wants it and is not allowed to change a single
character of the report for a reader who does not. Two tests defend that from
opposite directions: one asserts the six renderers are never called on a default
run, and one asserts the rendered text is byte-identical to a run that never had
the optional code path compiled into the call at all. The first would pass if a
flag were read but its section printed nothing; the second would pass if a
section printed something identical by coincidence. Together they close both
doors.

The refusal tests matter nearly as much. Every optional section reads a recorded
artifact, and pointing a section at a directory that does not have one is the
normal case rather than the exotic one: the fixtures live in the repository and
a user's ``ml.cache_dir`` starts empty. A section that cannot find its panel has
to say which file it wanted and leave the valuation standing, because a missing
model artifact is not a reason to withhold a DCF.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from techval import cli as C
from techval import commands_forecast, commands_mna, commands_peers, commands_tmt
from techval.config import Assumptions
from techval.edgar import CompanyFacts, DimensionedFact
from techval.market import CsvSource

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = FIXTURES / "prices"
AS_OF = "2026-09-10"

runner = CliRunner()

# The six flags, each with the heading its section prints. The headings are
# asserted against rendered output rather than against the renderer names,
# because what a reader sees is the thing that must not change.
OPTIONAL_SECTIONS = {
    "tmt.sotp": "Sum of the parts",
    "ml.peers.enabled": "Learned comp set",
    "ml.warranted.enabled": "Warranted multiple",
    "ml.signals.enabled": "Has that residual ever predicted anything?",
    "ml.mna.propensity_enabled": "Acquisition propensity",
    "ml.mna.precedents_enabled": "Precedent transactions",
}

# The renderers the default path is forbidden to touch.
OPTIONAL_RENDERERS = (
    "_render_optional_sotp",
    "_render_optional_peers",
    "_fit_warranted_panel",
    "_render_optional_warranted",
    "_render_optional_signal",
    "_render_optional_propensity",
    "_render_optional_precedents",
)


SUBMISSIONS = json.loads((FIXTURES / "submissions_tmt.json").read_text())["companies"]


def _instance(path: Path) -> tuple[list[DimensionedFact], str, date]:
    payload = json.loads(path.read_text())
    facts = [
        DimensionedFact(
            tag=row["tag"],
            value=row["value"],
            unit=row["unit"],
            start=date.fromisoformat(row["start"]) if row["start"] else None,
            end=date.fromisoformat(row["end"]),
            dimensions=row["dimensions"],
        )
        for row in payload["facts"]
    ]
    return facts, payload["_accession"], date.fromisoformat(payload["_filed"])


class FixtureClient:
    """The committed payloads, standing in for a network client.

    Wider than the conftest client because two of the optional sections are
    live rather than recorded: the sum of the parts reads a segment axis out of
    an instance document, and the precedent set reads merger filings. Both are
    served from the fixtures so this file stays offline.
    """

    def __init__(self, knowledge_date: date | None = None) -> None:
        self.knowledge_date = knowledge_date

    def ticker_to_cik(self, ticker: str) -> int:
        return self.company_facts(ticker).cik

    def company_facts(self, ticker: str) -> CompanyFacts:
        path = FIXTURES / f"companyfacts_{ticker.upper()}.json"
        return CompanyFacts(
            json.loads(path.read_text()), ticker, knowledge_date=self.knowledge_date
        )

    def submissions(self, ticker: str) -> dict:
        return dict(SUBMISSIONS.get(ticker.upper(), {}))

    def instance_facts(self, ticker, forms=("10-K", "10-Q")):
        return _instance(FIXTURES / f"instance_facts_{ticker.upper()}.json")

    def filings(self, ticker, forms=("10-K",), since=None, limit=20):
        """No merger agreements on file for any fixture company.

        Which is the ordinary case: most companies are not acquired, and an
        empty precedent set is a finding rather than a failure. The section
        under test is the one that says so.
        """
        return []

    def filing_text(self, ticker, filing) -> str:
        return ""


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Point the CLI's own constructors at the fixtures.

    The constructors are replaced rather than ``_setup``, for the reason the tmt
    command tests give: stubbing ``_setup`` would prove the renderers work and
    nothing about whether the assumptions actually reach them, and the
    assumptions are the entire subject of this file.
    """
    monkeypatch.setattr(
        C, "EdgarClient", lambda cache, knowledge_date=None: FixtureClient(knowledge_date)
    )
    monkeypatch.setattr(
        C, "make_price_source", lambda kind, cache, csv_dir=None: CsvSource(PRICES)
    )
    # The football field writes a PNG. Sending it to a temp directory keeps the
    # suite from depositing files in the working tree.
    monkeypatch.chdir(tmp_path)


def _config(tmp_path: Path, body: str = "") -> Path:
    """An assumptions file with a pinned risk-free rate and a real peer set."""
    path = tmp_path / "assumptions.yaml"
    path.write_text(
        "market:\n"
        "  risk_free_rate: 0.0483\n"
        "comps:\n"
        "  peers: [CRWD, MDB, ZS]\n" + body
    )
    return path


def run_value(*args: str):
    result = runner.invoke(C.app, ["value", "DDOG", "--as-of", AS_OF, *args])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


# --------------------------------------------------------------------------- #
# the mount
# --------------------------------------------------------------------------- #


ALL_MODULES = (commands_tmt, commands_peers, commands_forecast, commands_mna)


def _command_names(module) -> list[str]:
    names = []
    for command in module.app.registered_commands:
        names.append(command.name or command.callback.__name__.replace("_", "-"))
    return names


def test_every_sub_app_command_is_mounted():
    """A module that is not mounted cannot be invoked, so it is not shipped."""
    result = runner.invoke(C.app, ["--help"])
    assert result.exit_code == 0
    for module in ALL_MODULES:
        for name in _command_names(module):
            assert name in result.output, f"{name} is missing from --help"


def test_the_five_original_commands_survived_the_mount():
    """Mounting merges commands into the parent, so a collision would be silent."""
    result = runner.invoke(C.app, ["--help"])
    for name in ("value", "comps", "merger", "backtest", "fetch"):
        assert name in result.output


def test_no_command_name_collides():
    """Flat mounting means two modules cannot both own a name.

    ``add_typer`` with no name merges rather than nesting, which is what keeps
    the invocation the command modules document in their own docstrings. The
    price is that a duplicate name would quietly shadow, so it is asserted here.
    """
    seen: dict[str, str] = {}
    for module in ALL_MODULES:
        for name in _command_names(module):
            assert name not in seen, f"{name} claimed by {seen[name]} and {module.__name__}"
            seen[name] = module.__name__
    for name in ("value", "comps", "merger", "backtest", "fetch"):
        assert name not in seen, f"{name} collides with an original command"


@pytest.mark.parametrize(
    "name",
    [n for module in ALL_MODULES for n in _command_names(module)],
)
def test_each_mounted_command_answers_help(name):
    """Reachable, not merely listed."""
    result = runner.invoke(C.app, [name, "--help"])
    assert result.exit_code == 0
    assert name in result.output


# --------------------------------------------------------------------------- #
# the base report does not move
# --------------------------------------------------------------------------- #


def test_default_run_never_touches_the_optional_machinery(monkeypatch, tmp_path):
    """With every flag off, not one of the optional renderers is called.

    Booby-trapped rather than mocked. If a later change reads an artifact or
    fits a model before checking the flag, this fails with the name of the
    renderer that fired.
    """

    def trap(name):
        def _boom(*args, **kwargs):
            raise AssertionError(f"{name} ran on a default value run")

        return _boom

    for name in OPTIONAL_RENDERERS:
        monkeypatch.setattr(C, name, trap(name))

    result = run_value("-c", str(_config(tmp_path)))
    assert result.exit_code == 0


def test_default_report_carries_no_optional_heading(tmp_path):
    """None of the six sections appears unless it is asked for."""
    result = run_value("-c", str(_config(tmp_path)))
    assert result.exit_code == 0
    for flag, heading in OPTIONAL_SECTIONS.items():
        assert heading not in result.output, f"{flag} leaked into the default report"


def test_default_report_still_carries_the_base_sections(tmp_path):
    """The guard above would also pass on an empty report, so pin the content."""
    result = run_value("-c", str(_config(tmp_path)))
    assert result.exit_code == 0
    for heading in (
        "normalized TTM",
        "Enterprise value bridge",
        "WACC buildup",
        "Trading comparables",
    ):
        assert heading in result.output


def test_base_report_is_byte_identical_with_the_flags_off(tmp_path, monkeypatch):
    """The whole premise, asserted on the rendered characters.

    The comparison is against a run whose optional block cannot execute at all,
    achieved by trapping every renderer. If any flag defaulted to True, or a
    section printed so much as a blank line on the default path, the two strings
    would differ.
    """
    baseline = run_value("-c", str(_config(tmp_path)))

    def _boom(*args, **kwargs):
        raise AssertionError("optional machinery ran")

    for name in OPTIONAL_RENDERERS:
        monkeypatch.setattr(C, name, _boom)
    trapped = run_value("-c", str(_config(tmp_path)))

    assert baseline.exit_code == trapped.exit_code == 0
    assert baseline.output == trapped.output


def test_every_optional_flag_defaults_off():
    """The defaults are the contract, so they are asserted directly on the model."""
    a = Assumptions()
    assert a.tmt.sotp is False
    assert a.ml.peers.enabled is False
    assert a.ml.warranted.enabled is False
    assert a.ml.signals.enabled is False
    assert a.ml.mna.propensity_enabled is False
    assert a.ml.mna.precedents_enabled is False


# --------------------------------------------------------------------------- #
# the flags are actually consumed
# --------------------------------------------------------------------------- #


FLAG_BODIES = {
    "tmt.sotp": "tmt:\n  sotp: true\n",
    "ml.peers.enabled": "ml:\n  peers:\n    enabled: true\n",
    "ml.warranted.enabled": "ml:\n  warranted:\n    enabled: true\n",
    "ml.signals.enabled": "ml:\n  signals:\n    enabled: true\n",
    "ml.mna.propensity_enabled": "ml:\n  mna:\n    propensity_enabled: true\n",
    "ml.mna.precedents_enabled": "ml:\n  mna:\n    precedents_enabled: true\n",
}


@pytest.mark.parametrize("flag", sorted(FLAG_BODIES))
def test_each_flag_switches_its_section_on(flag, tmp_path):
    """A flag nothing reads is the defect this wave existed to fix.

    The section is pointed at an empty artifact directory, so most of these
    refuse. Refusing is still consuming the flag: the heading appears, which is
    what distinguishes a flag that is read from one that is ignored.
    """
    config = _config(tmp_path, FLAG_BODIES[flag])
    empty = tmp_path / "no-artifacts"
    empty.mkdir()
    result = run_value("-c", str(config), "--ml-data", str(empty))
    assert result.exit_code == 0
    assert OPTIONAL_SECTIONS[flag] in result.output


@pytest.mark.parametrize("flag", sorted(FLAG_BODIES))
def test_a_section_that_cannot_run_does_not_take_the_valuation_down(flag, tmp_path):
    """A missing recorded panel costs a reader that section and nothing else."""
    config = _config(tmp_path, FLAG_BODIES[flag])
    empty = tmp_path / "no-artifacts"
    empty.mkdir()
    result = run_value("-c", str(config), "--ml-data", str(empty))
    assert result.exit_code == 0
    # The valuation is still there, underneath the refusal.
    assert "Enterprise value bridge" in result.output
    assert "Trading comparables" in result.output
    assert "Football field written to" in result.output


@pytest.mark.parametrize(
    "flag,wanted",
    [
        ("ml.peers.enabled", "peer_groups.json"),
        ("ml.warranted.enabled", "observations.json.gz"),
        # Signals fits the warranted model to get the residual it scores, so a
        # missing observation panel is what it reports, not a missing price file.
        ("ml.signals.enabled", "observations.json.gz"),
        ("ml.mna.propensity_enabled", "universe.json"),
    ],
)
def test_a_missing_artifact_is_named_rather_than_guessed_at(flag, wanted, tmp_path):
    """Refuse rather than invent, and say which file was wanted.

    A section that printed "unavailable" and stopped would leave a reader with
    no way to fix it. The filename is the fix.

    Whitespace is stripped before the comparison because rich hard-wraps a long
    absolute path at the console width, which splits the filename across two
    lines without changing what a reader sees.
    """
    config = _config(tmp_path, FLAG_BODIES[flag])
    empty = tmp_path / "no-artifacts"
    empty.mkdir()
    result = run_value("-c", str(config), "--ml-data", str(empty))
    assert result.exit_code == 0
    flat = "".join(result.output.split())
    assert "missing" in flat
    assert wanted in flat, f"{flag} did not name the file it wanted"


def test_sotp_refuses_without_a_plan_and_names_the_command_that_writes_one(tmp_path):
    """There is no default multiple anywhere in this package, and that is the point."""
    config = _config(tmp_path, FLAG_BODIES["tmt.sotp"])
    result = run_value("-c", str(config))
    assert result.exit_code == 0
    assert "Sum of the parts" in result.output
    assert "--emit-plan" in result.output
    assert "--sotp-plan" in result.output


def test_ml_data_defaults_to_the_configured_cache_dir(tmp_path):
    """``--ml-data`` overrides ``ml.cache_dir``; without it the config decides.

    The same root and the same filenames ``techval peers`` and ``techval screen``
    already use, so a user who has run either has the artifacts in place.
    """
    root = tmp_path / "configured"
    root.mkdir()
    config = _config(
        tmp_path,
        "ml:\n  cache_dir: " + str(root) + "\n  warranted:\n    enabled: true\n",
    )
    result = run_value("-c", str(config))
    assert result.exit_code == 0
    flat = "".join(result.output.split())
    assert "".join(str(root / "observations.json.gz").split()) in flat


def test_ml_root_prefers_the_flag_over_the_config(tmp_path):
    """The unit underneath the test above, asserted without a full valuation."""
    a = Assumptions()
    a.ml.cache_dir = str(tmp_path / "from-config")
    assert C._ml_root(a, None) == tmp_path / "from-config"
    assert C._ml_root(a, tmp_path / "from-flag") == tmp_path / "from-flag"
