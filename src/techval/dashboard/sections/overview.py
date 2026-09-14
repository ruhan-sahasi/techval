"""The scoreboard: five models, each against its baseline, and the state of everything else.

This section computes nothing from a fixture. It is collected last, reads the
other sections' results from ``ctx.results`` and puts three things at the top of
the page: one tile per model section with its headline score against its
baseline, a table of every other section's status with its figure and refusal
counts, and what the collection ran under. A reader who reads nothing else
should leave knowing what each verdict was, and the takeaway leads with the one
model that was scored against realised market returns, because that is the
result the rest of the page has to be read against.

The collector is split as the other sections are. ``scoreboard``,
``section_states`` and ``assumption_overrides`` are pure functions over results
and assumptions, ``shape`` turns what they return into the section, and
``collect`` records each of them as the entry point of the figure it builds.

Four judgments are made here and stated where they are made.

**A model section that has no score shows its state, never a number.** A
refused section's tile carries the reason it refused; one that has not been
collected says so. The section is refused only when no model section produced a
headline, since then there is no verdict to put at the top of the page.

**The model scored against returns is named, not inferred.** ``RETURNS_MODEL``
is the value signal, whose collector tests whether cheapness predicted forward
returns. The other four are scored against what filings and deal records say
(disclosed peers, traded multiples, reported revenue growth, announced deals).
That is a fact about what each section measures and no headline field carries
it, so it is held here like an identity and not derived from a metric's name.

**Timings, the commit and the date are read at render, not here.** The result
cache for this section is keyed on the other sections' results with their
timings left out, so that a re-run which moves no figure does not miss it. A
total of seconds written here could therefore be the total of an earlier run
than the provenance rows the page prints beside it, and the commit and date are
not in ``ctx`` at all. The renderer reads all three from the snapshot it draws,
where they cannot disagree with the rest of the page.

**The assumptions are shown as what changed, not as a file name.** The
collector is handed the assumptions it ran under and not the path they were
read from, so the figure lists every setting that differs from the engine
defaults, which is what a reader needs to reproduce the page. ``ml.cache_dir``
is left out: where the cache lives changes no figure, and the result cache
leaves it out for the same reason.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from ...config import Assumptions
from ...errors import NotMeaningfulError
from . import MODEL_SECTIONS, SECTION_IDS, section_module

ID = "overview"
TITLE = "Scoreboard"
INPUTS: list[str] = []

# The one model section scored against realised market returns. See the module
# docstring for why this is held and not derived.
RETURNS_MODEL = "signal"

_HERE = "techval.dashboard.sections.overview"
SCOREBOARD_ENTRY = f"{_HERE}.scoreboard"
STATES_ENTRY = f"{_HERE}.section_states"
ASSUMPTIONS_ENTRY = f"{_HERE}.assumption_overrides"
DEFAULTS_ENTRY = "techval.config.Assumptions"

# Figure ids in page order.
FIGURES = ("scoreboard", "sections", "collection")

# The order verdicts are tallied in, best first, and how each reads in a sentence
# as the predicate of "one model ..." and "two models ...".
VERDICT_ORDER = ("beats", "inside_noise", "ties", "not_significant", "loses")
_PREDICATE = {
    "beats": ("beats it outside the noise", "beat it outside the noise"),
    "inside_noise": ("sits inside the noise", "sit inside the noise"),
    "ties": ("ties it", "tie it"),
    "not_significant": ("is not significant", "are not significant"),
    "loses": ("loses to it", "lose to it"),
}
# How a verdict qualifies a lift: "a lift of -0.0986 that is not significant".
_LIFT_CLAUSE = {
    "beats": "that clears the noise",
    "inside_noise": "that sits inside the noise",
    "ties": "that counts as a tie",
    "not_significant": "that is not significant",
    "loses": "that loses to the baseline",
}

MINUS = "−"
_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


# --------------------------------------------------------------------------- #
# words and numbers
# --------------------------------------------------------------------------- #


def _word(n: int) -> str:
    return _WORDS[n] if 0 <= n < len(_WORDS) else f"{n:,}"


def _dp(v: float) -> int:
    """Decimal places that follow the magnitude, as the page's headline strip formats."""
    a = abs(v)
    if a >= 100:
        return 0
    if a >= 10:
        return 1
    if a >= 1:
        return 2
    return 4


def _num(v: float, dp: int | None = None) -> str:
    dp = _dp(v) if dp is None else dp
    body = f"{abs(v):,.{dp}f}"
    return (MINUS if v < 0 and any(c in "123456789" for c in body) else "") + body


def _signed(v: float, dp: int | None = None) -> str:
    dp = _dp(v) if dp is None else dp
    body = f"{abs(v):,.{dp}f}"
    if not any(c in "123456789" for c in body):
        return body
    return ("+" if v > 0 else MINUS) + body


def _name(title: str) -> str:
    """A section title as a noun phrase inside a sentence: "the peer encoder"."""
    rest = title[4:] if title.lower().startswith("the ") else title
    # Lower the first letter of a word ("Peer encoder"), never of an initialism
    # ("M&A propensity", "TMT layer").
    if len(rest) > 1 and rest[1].islower():
        rest = rest[0].lower() + rest[1:]
    return f"the {rest}"


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _title(section_id: str, section: Mapping[str, Any] | None) -> str:
    if section and isinstance(section.get("title"), str) and section["title"].strip():
        return section["title"]
    return section_module(section_id).TITLE


# --------------------------------------------------------------------------- #
# the three computations
# --------------------------------------------------------------------------- #


def scoreboard(results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One tile per model section, in page order.

    A scored tile carries the headline as the section wrote it, plus the kit's
    tile keys (``label``, ``value``, ``sub``, ``status``, ``delta``) so the
    default drawing of a tiles figure still shows it truthfully. A tile for a
    section without a score carries its state and no numeric field at all.
    """
    tiles = []
    for sid in MODEL_SECTIONS:
        section = results.get(sid)
        title = _title(sid, section)
        tile: dict[str, Any] = {"section": sid, "href": f"#{sid}", "label": title}
        status = section.get("status") if section else None
        headline = section.get("headline") if section else None
        if status == "ok" and headline:
            score = float(headline["score"])
            base = float(headline["baseline_score"])
            lift = float(headline["lift"])
            if not all(math.isfinite(v) for v in (score, base, lift)):
                raise ValueError(f"section {sid!r} carries a non-finite headline number")
            dp = _dp(score)
            tile.update(
                {
                    "state": "scored",
                    "metric": headline["metric"],
                    "score": score,
                    "baseline_name": headline["baseline_name"],
                    "baseline_score": base,
                    "lift": lift,
                    "n": int(headline["n"]),
                    "higher_is_better": bool(headline["higher_is_better"]),
                    "verdict_status": headline["verdict_status"],
                    "value": score,
                    "format": f"num:{dp}",
                    "sub": (
                        f"{headline['metric']} against {_num(base)} for "
                        f"{headline['baseline_name']}, n = {int(headline['n']):,}"
                    ),
                    "status": headline["verdict_status"],
                    "delta": {"value": lift, "format": f"signed:{_dp(lift)}", "label": "lift"},
                }
            )
        elif status == "refused":
            refusals = section.get("refusals") or []
            first = refusals[0] if refusals else {"what": title, "why": "No reason was recorded."}
            tile.update(
                {
                    "state": "refused",
                    "value": "Refused",
                    "status": "refused",
                    "reason": {"what": first["what"], "why": first["why"]},
                    "sub": first["why"],
                }
            )
        elif status == "ok":
            tile.update(
                {
                    "state": "no_headline",
                    "value": "No score",
                    "sub": "Collected without a headline score against a baseline",
                }
            )
        else:
            tile.update(
                {
                    "state": "not_built",
                    "value": "Not collected",
                    "sub": "This snapshot holds no results for this model",
                }
            )
        tiles.append(tile)
    return tiles


def section_states(results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every section but this one, in page order, with its status and its counts."""
    rows = []
    for sid in SECTION_IDS:
        if sid == ID:
            continue
        section = results.get(sid)
        status = section.get("status") if section else "not_built"
        headline = section.get("headline") if section else None
        rows.append(
            {
                "id": sid,
                "title": _title(sid, section),
                "href": f"#{sid}",
                "status": status,
                "verdict_status": headline.get("verdict_status") if headline else None,
                "figures": len(section.get("figures") or {}) if section else 0,
                "refusals": len(section.get("refusals") or []) if section else 0,
                "provenance": len(section.get("provenance") or []) if section else 0,
            }
        )
    return rows


def _flatten(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten(value[key], f"{prefix}{key}.")
    else:
        yield prefix[:-1], value


def assumption_overrides(assumptions: Assumptions) -> list[dict[str, Any]]:
    """Every setting that differs from the engine default, by dotted name."""
    applied = dict(_flatten(assumptions.model_dump(mode="json")))
    default = dict(_flatten(Assumptions().model_dump(mode="json")))
    return [
        {"key": key, "value": applied[key], "default": default.get(key)}
        for key in sorted(applied)
        if key != "ml.cache_dir" and applied[key] != default.get(key)
    ]


# --------------------------------------------------------------------------- #
# shape: results in, page out
# --------------------------------------------------------------------------- #


def _tally(tiles: list[dict[str, Any]]) -> dict[str, int]:
    scored = [t for t in tiles if t["state"] == "scored"]
    return {v: sum(t["verdict_status"] == v for t in scored) for v in VERDICT_ORDER}


def _returns_sentence(tile: dict[str, Any]) -> str:
    lift = tile["lift"]
    where = "behind" if lift < 0 else "ahead of" if lift > 0 else "level with"
    clause = _LIFT_CLAUSE.get(tile["verdict_status"], f"with the verdict {tile['verdict_status']}")
    return (
        f"{_capital(_name(tile['label']))}, the one model scored against realised returns, "
        f"came back {where} its baseline: {_num(tile['score'])} on {tile['metric']} against "
        f"{_num(tile['baseline_score'])}, a lift of {_signed(lift)} {clause}."
    )


def _tally_sentence(tiles: list[dict[str, Any]]) -> str:
    scored = [t for t in tiles if t["state"] == "scored"]
    total = len(scored)
    beats = [t for t in scored if t["verdict_status"] == "beats"]
    if total == len(tiles):
        lead = f"{_capital(_word(total))} models were each held to a baseline"
    else:
        verb = "was" if total == 1 else "were"
        lead = f"{_capital(_word(total))} of the {_word(len(tiles))} models {verb} held to a baseline"
    if not beats:
        first = "none beats it outside the noise"
    elif len(beats) == total:
        first = (
            f"{_name(beats[0]['label'])} beats it outside the noise"
            if total == 1
            else f"all {_word(total)} beat it outside the noise"
        )
    elif len(beats) == 1:
        first = f"only {_name(beats[0]['label'])} beats it outside the noise"
    else:
        first = f"{_join([_name(t['label']) for t in beats])} beat it outside the noise"
    counts = _tally(tiles)
    rest = [
        f"{_word(counts[v])} {_PREDICATE[v][0 if counts[v] == 1 else 1]}"
        for v in VERDICT_ORDER[1:]
        if counts[v]
    ]
    unknown = total - sum(counts.values())
    if unknown:
        rest.append(f"{_word(unknown)} carr{'ies' if unknown == 1 else 'y'} a verdict this page does not know")
    return f"{lead}, and {first}" + (f"; {_join(rest)}." if rest else ".")


def _unscored_sentence(tiles: list[dict[str, Any]]) -> str:
    parts = []
    for t in tiles:
        name = _name(t["label"])
        if t["state"] == "refused":
            parts.append(f"{name} refused its score")
        elif t["state"] == "no_headline":
            parts.append(f"{name} was collected without a headline score")
        elif t["state"] == "not_built":
            parts.append(f"{name} has not been collected")
    if not parts:
        return ""
    sentence = _capital(_join(parts))
    returns = next((t for t in tiles if t["section"] == RETURNS_MODEL), None)
    if returns is not None and returns["state"] != "scored":
        sentence += ", so no model on this page is scored against realised returns"
    return sentence + "."


def takeaway(tiles: list[dict[str, Any]]) -> str:
    """The sentence at the top of the page, built from the tiles and nothing else."""
    sentences = []
    returns = next((t for t in tiles if t["section"] == RETURNS_MODEL), None)
    if returns is not None and returns["state"] == "scored":
        sentences.append(_returns_sentence(returns))
    sentences.append(_tally_sentence(tiles))
    unscored = _unscored_sentence(tiles)
    if unscored:
        sentences.append(unscored)
    return " ".join(sentences)


def _scoreboard_title(tiles: list[dict[str, Any]]) -> str:
    scored = [t for t in tiles if t["state"] == "scored"]
    beats = [t for t in scored if t["verdict_status"] == "beats"]
    if not beats:
        return "No model beats its baseline outside the noise"
    if len(beats) == len(scored):
        if len(beats) == 1:
            return f"{_capital(_name(beats[0]['label']))} beats its baseline outside the noise"
        return f"All {_word(len(beats))} scored models beat their baselines outside the noise"
    names = _join([_name(t["label"]) for t in beats])
    return (
        f"{_capital(_word(len(beats)))} of {_word(len(scored))} models "
        f"beat{'s' if len(beats) == 1 else ''} {'its baseline' if len(beats) == 1 else 'their baselines'} "
        f"outside the noise: {names}"
    )


def _sections_figure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    figures = sum(r["figures"] for r in rows)
    refusals = sum(r["refusals"] for r in rows)
    refusing = [r for r in rows if r["refusals"]]
    if refusals:
        most = max(refusing, key=lambda r: r["refusals"])
        title = (
            f"{figures:,} figure{'s' if figures != 1 else ''} drawn and {refusals:,} "
            f"refusal{'s' if refusals != 1 else ''} stated, "
            f"{most['refusals']:,} of them in {_name(most['title'])}"
            if len(refusing) > 1
            else f"{figures:,} figures drawn and {refusals:,} refusal"
            f"{'s' if refusals != 1 else ''} stated, all in {_name(most['title'])}"
        )
    else:
        title = f"{figures:,} figures drawn and nothing refused"
    collected = sum(r["status"] == "ok" for r in rows)
    subtitle = (
        "Every other section, in page order. A refusal is a number a section would not "
        "print because the fixtures could not reproduce it; each is stated with its reason."
    )
    return {
        "kind": "hbar",
        "title": title,
        "subtitle": subtitle,
        "data": {
            "rows": [
                {"label": r["title"], "value": r["refusals"], "role": "baseline"} for r in rows
            ],
            "format": "int",
            "valueLabel": "Refusals",
            "labelHeader": "Section",
            "table": rows,
            "totals": {
                "sections": len(rows),
                "collected": collected,
                "figures": figures,
                "refusals": refusals,
                "refusing_sections": len(refusing),
            },
        },
    }


def _value_text(v: Any) -> str:
    if v is None:
        return "unset"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return repr(v) if not isinstance(v, str) else v


def _collection_figure(overrides: list[dict[str, Any]]) -> dict[str, Any]:
    k = len(overrides)
    if k == 0:
        title = "Every assumption is the engine default"
        sub = "No setting differs from the defaults"
    elif k == 1:
        title = "Every assumption but one is the engine default"
        o = overrides[0]
        sub = f"{o['key']} = {_value_text(o['value'])}; the default is {_value_text(o['default'])}"
    else:
        title = f"{_capital(_word(k))} assumptions differ from the engine defaults"
        sub = "; ".join(f"{o['key']} = {_value_text(o['value'])}" for o in overrides)
    return {
        "kind": "tiles",
        "title": title,
        "subtitle": "What every number on this page was computed from, and under what",
        "data": {
            "tiles": [
                {
                    "label": "Assumptions changed from the defaults",
                    "value": k,
                    "format": "int",
                    "sub": sub,
                }
            ],
            "overrides": overrides,
        },
    }


def shape(
    tiles: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    overrides: list[dict[str, Any]],
) -> dict[str, Any]:
    """The section as the collector returns it, from the three computations alone.

    Raises ``NotMeaningfulError`` when no model section carries a score, which
    the collector turns into a refusal of the section.
    """
    if not any(t["state"] == "scored" for t in tiles):
        states = _join([f"{_name(t['label'])} {t['state'].replace('_', ' ')}" for t in tiles])
        raise NotMeaningfulError(
            "no model section produced a headline score against its baseline, so there "
            f"is no verdict to put at the top of the page ({states})"
        )
    counts = _tally(tiles)
    return {
        "status": "ok",
        "takeaway": takeaway(tiles),
        "refusals": [],
        "headline": None,
        "figures": {
            "scoreboard": {
                "kind": "tiles",
                "title": _scoreboard_title(tiles),
                "subtitle": "Each model's headline score against the baseline its section names",
                "data": {"tiles": tiles, "counts": counts, "returns_model": RETURNS_MODEL},
            },
            "sections": _sections_figure(rows),
            "collection": _collection_figure(overrides),
        },
    }


def collect(ctx) -> dict[str, Any]:
    results = ctx.results
    with ctx.record("scoreboard", SCOREBOARD_ENTRY):
        tiles = scoreboard(results)
    with ctx.record("sections", STATES_ENTRY):
        rows = section_states(results)
    # The overrides are measured against the defaults in techval.config, so that
    # module is recorded too: the result cache then misses when a default moves,
    # even though the applied assumptions it is keyed on may not.
    with ctx.record("collection", DEFAULTS_ENTRY), ctx.record("collection", ASSUMPTIONS_ENTRY):
        overrides = assumption_overrides(ctx.assumptions)
    return shape(tiles, rows, overrides)
