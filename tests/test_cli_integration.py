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
from techval.errors import MissingDataError
from techval.market import CsvSource
from techval.merger import run_merger

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = FIXTURES / "prices"
AS_OF = "2026-09-10"

runner = CliRunner()

# The seven flags, each with the heading its section prints. The headings are
# asserted against rendered output rather than against the renderer names,
# because what a reader sees is the thing that must not change.
#
# This map is also the census, and a census is only worth what its enumeration
# is worth. It listed six while config.py carried seven capability switches, so
# ml.forecast.enabled reached no valuation, printed neither a heading nor a
# refusal, and nothing here could notice. A flag that is inert is worse than a
# flag that refuses, because a reader who sets it concludes the engine agreed.
# test_the_census_below_covers_every_switch_in_the_config closes that door: it
# walks the assumptions model rather than this list.
OPTIONAL_SECTIONS = {
    "tmt.sotp": "Sum of the parts",
    "ml.forecast.enabled": "Growth: the assumed fade against a fitted one",
    "ml.peers.enabled": "Learned comp set",
    "ml.warranted.enabled": "Warranted multiple",
    "ml.signals.enabled": "Has that residual ever predicted anything?",
    "ml.mna.propensity_enabled": "Acquisition propensity",
    "ml.mna.precedents_enabled": "Precedent transactions",
}

# The renderers the default path is forbidden to touch.
OPTIONAL_RENDERERS = (
    "_render_optional_sotp",
    "_render_optional_fade",
    "_render_optional_peers",
    "_fit_warranted_panel",
    "_render_optional_warranted",
    "_render_optional_signal",
    "_render_optional_propensity",
    "_render_optional_precedents",
)

# Every capability switch in the assumptions model, by dotted path. Written out
# rather than discovered so that adding a switch to config.py and forgetting to
# wire it fails here with the name of the switch.
CAPABILITY_SWITCHES = (
    "tmt.sotp",
    "ml.forecast.enabled",
    "ml.peers.enabled",
    "ml.warranted.enabled",
    "ml.signals.enabled",
    "ml.mna.propensity_enabled",
    "ml.mna.precedents_enabled",
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
    assert a.ml.forecast.enabled is False
    assert a.ml.peers.enabled is False
    assert a.ml.warranted.enabled is False
    assert a.ml.signals.enabled is False
    assert a.ml.mna.propensity_enabled is False
    assert a.ml.mna.precedents_enabled is False


def _switch(assumptions, dotted: str):
    node = assumptions
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


@pytest.mark.parametrize("dotted", CAPABILITY_SWITCHES)
def test_every_capability_switch_is_covered_by_this_file(dotted):
    """The census defends only the flags it knows about, so the list is the subject.

    ``ml.forecast.enabled`` was in ``config.py`` and in neither of the maps
    above, and it reached no valuation: the byte-identity test passed on it
    because there was nothing to be identical to. Three assertions rather than
    one, because a switch can go missing from any of the three maps and each
    omission costs a different guarantee.
    """
    assert _switch(Assumptions(), dotted) is False, f"{dotted} does not default off"
    assert dotted in OPTIONAL_SECTIONS, f"{dotted} has no heading in the census"
    assert dotted in FLAG_BODIES, f"{dotted} is never switched on by a test"


# --------------------------------------------------------------------------- #
# the flags are actually consumed
# --------------------------------------------------------------------------- #


FLAG_BODIES = {
    "tmt.sotp": "tmt:\n  sotp: true\n",
    "ml.forecast.enabled": "ml:\n  forecast:\n    enabled: true\n",
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
        ("ml.forecast.enabled", "fade_companyfacts.json.gz"),
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


# --------------------------------------------------------------------------- #
# --ml-data reaches the committed artifacts
# --------------------------------------------------------------------------- #


def test_every_recorded_artifact_resolves_inside_the_committed_fixture_tree():
    """One value of ``--ml-data`` has to reach all of them, and it did not.

    The refusals tell a reader that "tests/fixtures carries a committed set".
    Pointing ``--ml-data`` at tests/fixtures then reached the M&A dataset and
    nothing else: the peer artifacts are committed with a ``_tmt`` suffix, the
    warranted panel sits one directory deeper and so do the close prices. A hint
    that names a path which does not work is worse than no hint at all, because
    the reader concludes the feature is broken rather than that they pointed at
    the wrong directory.

    This walks the resolver's own table, so a fixture that is renamed or a
    candidate list that drifts fails here with the name of the artifact rather
    than being discovered by somebody running the flagship command.
    """
    for key in C._ARTIFACTS:
        (found,) = C._resolve(
            FIXTURES, [key], what="a committed artifact", writer=""
        )
        assert found.is_file(), f"{key} did not resolve under tests/fixtures"


def test_the_cache_layout_is_still_the_first_candidate():
    """``techval peers`` and ``techval screen`` write flat names into ml.cache_dir.

    Tolerating the fixture tree's layout must not have cost the layout the
    recorders actually write, so the unsuffixed flat name is asserted to be the
    one a refusal names first.
    """
    for key, candidates in C._ARTIFACTS.items():
        assert candidates, f"{key} has no candidate path at all"
        if key.startswith("mna_"):
            continue
        assert "/" not in candidates[0], f"{key} does not try the flat cache name first"


def test_a_refusal_names_every_place_it_looked(tmp_path):
    """Both candidates, so a reader with either layout can see which one to fix."""
    empty = tmp_path / "no-artifacts"
    empty.mkdir()
    with pytest.raises(MissingDataError) as exc:
        C._resolve(empty, ["warranted_panel"], what="the panel", writer="written by x")
    flat = "".join(str(exc.value).split())
    assert "observations.json.gz" in flat
    assert "warranted/observations.json.gz" in flat


def test_a_directory_of_the_right_name_is_not_an_artifact(tmp_path):
    """``~/.techval/ml`` really does hold a peer_groups/ directory beside the files."""
    root = tmp_path / "root"
    (root / "peer_groups.json").mkdir(parents=True)
    with pytest.raises(MissingDataError):
        C._resolve(root, ["peer_groups"], what="the groups", writer="")


# --------------------------------------------------------------------------- #
# the fitted growth path, which reached nothing at all
# --------------------------------------------------------------------------- #


def _fade_result(tmp_path, extra: str = ""):
    config = _config(tmp_path, FLAG_BODIES["ml.forecast.enabled"] + extra)
    return run_value("-c", str(config), "--ml-data", str(FIXTURES))


def test_the_fitted_growth_path_reaches_the_valuation(tmp_path):
    """``ml.forecast.enabled`` was the only switch that reached no valuation.

    Setting it produced a byte-identical report: the engine neither refused nor
    acted, which is the one behaviour the cardinal rule forbids. This asserts the
    section renders rather than merely that a heading appeared, by pinning the
    four cases the comparison exists to print.
    """
    result = _fade_result(tmp_path)
    assert result.exit_code == 0
    assert OPTIONAL_SECTIONS["ml.forecast.enabled"] in result.output
    for case in ("Assumed fade", "Fitted fade", "Fitted, low band", "Fitted, high band"):
        assert case in result.output, f"{case} is missing from the fade section"


def test_the_fitted_path_is_shown_beside_the_assumed_one_and_replaces_nothing(tmp_path):
    """The section is a comparison. A fitted number that quietly replaced the
    typed one would be the fitted model doing exactly what the flag exists to
    prevent, so the DCF above is asserted to be untouched: the same headline is
    still printed and the assumed path is still the one it was struck on."""
    with_fade = _fade_result(tmp_path)
    without = run_value("-c", str(_config(tmp_path)))
    headline = [line for line in without.output.splitlines() if "<- headline" in line]
    assert headline, "the base report lost its headline"
    assert headline[0] in with_fade.output


def test_the_fade_section_prints_a_band_rather_than_a_point_estimate(tmp_path):
    """A fitted point estimate beside the assumption it replaces invites a reader
    to treat it as the answer. It is the middle of a band whose two ends are
    different companies, so both ends are valued and the note saying what the
    band is and is not travels with them."""
    result = _fade_result(tmp_path)
    assert "What the path is and is not" in result.output
    assert "the band the fit actually supports runs" in result.output.lower()


def test_the_fade_section_prints_the_baselines_at_every_horizon(tmp_path):
    """The model card's claim is a tie at one year and a win at two and three.

    A lift quoted without the baseline it was measured against is not a result,
    and at two and three years the baseline that matters stops being persistence
    and becomes the company's own sub-vertical. Both are asserted, because
    printing only the one the model beats is the flattering half.
    """
    result = _fade_result(tmp_path)
    assert "Model against the baselines, every horizon" in result.output
    for column in ("Persistence", "Training mean", "Sub-vertical", "Best baseline"):
        assert column in result.output
    assert "Against persistence alone:" in result.output


def test_the_fade_section_carries_the_model_card(tmp_path):
    """A fitted path inside a valuation without its training window is unauditable."""
    result = _fade_result(tmp_path)
    flat = " ".join(result.output.split())
    assert "The model behind that number" in flat
    assert "Trained through" in flat
    assert "What this model cannot do" in flat


def test_the_fade_section_honours_the_knowledge_date(tmp_path):
    """Without the cut the option would be a lie.

    The panel builder pins every feature to the filing date of the report that
    carried it and then stops, so a curve fitted on filings through 2026 and
    handed to a valuation struck in 2023 would know how the intervening years
    went. The valuation would look excellent and the failure would be silent.
    """
    config = _config(tmp_path, FLAG_BODIES["ml.forecast.enabled"])
    result = runner.invoke(
        C.app,
        ["value", "DDOG", "--as-of", "2024-06-30", "-c", str(config),
         "--ml-data", str(FIXTURES)],
    )
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    # The panel's own note, not the report's point-in-time banner, which is
    # printed whether or not anything downstream honoured the date.
    assert "observation(s) filed after 2024-06-30 were removed" in flat
    assert "forward label(s) that had not been filed by then were detached" in flat


# --------------------------------------------------------------------------- #
# a model output inside a valuation carries its card
# --------------------------------------------------------------------------- #


def test_the_warranted_section_carries_the_model_card(tmp_path):
    """``techval screen`` printed this model's card and the value report did not.

    One model, two disclosure standards, and the weaker one was the copy sitting
    inside a valuation. ml/__init__ states the rule it broke: an unauditable
    model has no place beside a valuation whose every other number traces to a
    filing.
    """
    config = _config(tmp_path, FLAG_BODIES["ml.warranted.enabled"])
    result = run_value("-c", str(config), "--ml-data", str(FIXTURES))
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "Trained through" in flat
    assert "Training observations" in flat
    assert "observations.json.gz" in "".join(result.output.split())
    assert "What this model cannot do" in flat
    # The pooled score is mostly company identity. Printing it without the
    # differenced figure beside it is how a screen gets oversold.
    assert "Differenced against the company's own prior read" in flat


def test_the_signal_section_says_what_the_sample_was(tmp_path):
    """The verdict alone was the flattering half.

    ``test_signal`` writes a census of how every holding period ended and flags a
    sample in which none of them ended in an acquisition or a delisting, which is
    the survivorship hole the harness exists to expose. The section printed the
    verdict and dropped the census, so the reader was told the coefficient and
    not that the failures had been removed before it was computed.
    """
    config = _config(tmp_path, FLAG_BODIES["ml.signals.enabled"])
    result = run_value("-c", str(config), "--ml-data", str(FIXTURES))
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "holding periods ran their course" in flat
    assert "FLAG:" in flat
    assert "biased upward by the outcomes it cannot see" in flat
    # And the convention that produced it, so the exclusion is a stated choice
    # rather than a default nobody saw.
    assert "--delisting" in flat


# --------------------------------------------------------------------------- #
# the merger command renders to its last row with purchase accounting on
# --------------------------------------------------------------------------- #


MERGER_BODY = (
    "merger:\n"
    "  offer_premium: 0.30\n"
    "  purchase_accounting:\n"
    "    enabled: true\n"
)


def run_merger_command(*args: str):
    result = runner.invoke(C.app, ["merger", "DDOG", "MDB", "--as-of", AS_OF, *args])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


def test_merger_with_purchase_accounting_renders_to_the_last_row(tmp_path):
    """The flag's own section prints rather than dying at the accretion row.

    ``accretion_by_year`` holds (year, dollars, percent or None) tuples, and the
    renderer iterated them as if they were scalars, so every run with
    ``merger.purchase_accounting.enabled`` computed a correct model and then
    crashed formatting the last row of the pro forma table. Nothing in the suite
    ran the command end to end with the flag on, which is how it survived: this
    is that test.
    """
    result = run_merger_command("-c", str(_config(tmp_path, MERGER_BODY)))
    assert result.exit_code == 0
    assert "Purchase accounting: opening balance sheet" in result.output
    assert "Pro forma" in result.output
    assert "Accretion / (dilution)" in result.output


def test_the_accretion_row_prints_the_dollars_the_model_computed(tmp_path):
    """The rendered cells are the tuple's per-share dollars, not any other field.

    The section could print SOMETHING and still lie, so the same fixtures are run
    through ``run_merger`` directly and every per-share figure the model computed
    is asserted to appear in the rendered row, formatted exactly as ``_money``
    formats it. The percent half follows the PurchaseAccounting contract: shown
    where the model quoted one, NM where standalone EPS was too thin to divide by.
    """
    config = _config(tmp_path, MERGER_BODY)
    assumptions, client, market = C._setup(config, True, AS_OF)
    acq_fin, acq_price, acq_bridge = C._load("DDOG", client, market, assumptions)
    tgt_fin, tgt_price, tgt_bridge = C._load("MDB", client, market, assumptions)
    r = run_merger(
        acq_fin, tgt_fin, acq_bridge, tgt_bridge, acq_price, tgt_price, assumptions
    )
    pa = r.purchase_accounting
    assert pa is not None and pa.accretion_by_year

    result = run_merger_command("-c", str(config))
    assert result.exit_code == 0
    flat = "".join(result.output.split())
    for _, dollars, _ in pa.accretion_by_year:
        shown = f"({abs(dollars):,.3f})" if dollars < 0 else f"{dollars:,.3f}"
        assert shown in flat, f"{shown} is missing from the accretion row"
    for _, _, pct in pa.accretion_by_year:
        if pct is not None:
            assert f"{pct:.1%}" in flat, f"{pct:.1%} is missing from the percent row"
