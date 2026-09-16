"""Both readers scored against the owner's key, on the same tasks.

**What counts as right.**

- A reader is right on a task when its status is the key's and, for a stated
  fact, its value is the key's.
- A number matches within half a unit of the last digit the key's quote shows.
  A date must match exactly. Text is compared after folding case and
  punctuation.
- A refusal is right only where the key states nothing.

**What counts as wrong.** Every wrong answer has one kind:

- a wrong value accepted, the one that reaches a valuation;
- a stated value missed;
- a value invented where the filing states none;
- a wrong status between the answers that state nothing.

**The baseline.** The regex rules are scored twice, on the passages and on the
whole input, and the stronger score is the baseline, so the comparison is
never against a handicapped incumbent.

**The test.** Whether the difference is more than chance is an exact McNemar
test on the tasks where exactly one reader is right. With about sixty tasks, a
real gap can still come out not significant, and the page says so when it
does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

from ..errors import TechvalError
from ..ml.protocol import EvalResult, PairedDelta
from .key import KeyRow, KeyState
from .readings import Reading
from .run import Run
from .tasks import METRICS
from .verify import fold_words

ERROR_KINDS = ("wrong_value", "missed", "invented", "wrong_status")
REGEX_READERS = ("regex_native", "regex_retrieved")
SIGNIFICANCE = 0.05
_VERDICTS = {
    (True, True): "both right",
    (True, False): "Claude only",
    (False, True): "regex only",
    (False, False): "neither",
}


class IncompleteRun(TechvalError):
    """A comparison asked for before every reading is recorded and every key row is filled."""


def value_matches(unit: str, reading: Reading, key: KeyRow) -> bool:
    if unit == "date":
        return (reading.text_value or "") == (key.text_value or "")
    if unit == "text":
        return fold_words(reading.text_value or "") == fold_words(key.text_value or "")
    if reading.value is None or key.value is None:
        return False
    tolerance = key.resolution / 2 if key.resolution else 0.005 * abs(key.value)
    return abs(reading.value - key.value) <= max(tolerance, 1e-9)


def is_correct(reading: Reading, key: KeyRow, unit: str) -> bool:
    if reading.status == "refused":
        return key.status in ("not_stated", "ambiguous")
    if reading.status != key.status:
        return False
    return key.status != "stated" or value_matches(unit, reading, key)


def error_kind(reading: Reading, key: KeyRow, unit: str) -> str | None:
    if is_correct(reading, key, unit):
        return None
    if key.status == "stated":
        return "wrong_value" if reading.status == "stated" else "missed"
    return "invented" if reading.status == "stated" else "wrong_status"


def mcnemar_p(claude_only: int, regex_only: int) -> float:
    discordant = claude_only + regex_only
    if discordant == 0:
        return 1.0
    return float(binomtest(claude_only, discordant, 0.5).pvalue)


@dataclass(frozen=True)
class ReaderScore:
    reader: str
    n: int
    correct: int
    errors: dict[str, int]

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


@dataclass(frozen=True)
class Comparison:
    claude: ReaderScore
    regex: ReaderScore
    regex_runs: dict[str, ReaderScore]
    verdicts: dict[str, str]
    claude_only: int
    regex_only: int
    mcnemar_p: float
    verdict_status: str
    evaluation: EvalResult
    recall: float | None
    recall_hits: int
    recall_n: int


def _score(reader: str, run: Run, key: KeyState) -> tuple[ReaderScore, dict[str, bool]]:
    errors = dict.fromkeys(ERROR_KINDS, 0)
    marks: dict[str, bool] = {}
    for p in run.prepared:
        task_id, unit = p.task.task_id, METRICS[p.task.metric].unit
        reading, row = run.readings[reader][task_id], key.rows[task_id]
        marks[task_id] = is_correct(reading, row, unit)
        if not marks[task_id]:
            errors[error_kind(reading, row, unit)] += 1
    return ReaderScore(reader, len(marks), sum(marks.values()), errors), marks


def recall_at_k(run: Run, key: KeyState) -> tuple[float | None, int, int]:
    hits = n = 0
    for p in run.prepared:
        row = key.rows.get(p.task.task_id)
        if row is None or row.status != "stated" or row.span is None:
            continue
        n += 1
        start, end = row.span
        hits += any(x.start_char < end and start < x.end_char for x in p.passages)
    return (hits / n if n else None), hits, n


def _paired(differences: np.ndarray) -> PairedDelta | None:
    if differences.size < 2:
        return None
    sd = float(np.std(differences, ddof=1))
    standard_error = sd / float(np.sqrt(differences.size))
    return PairedDelta(
        mean=float(differences.mean()),
        sd=sd,
        standard_error=standard_error,
        t=float(differences.mean() / standard_error) if standard_error else float("inf"),
        win_rate=float((differences > 0).mean()),
        n=int(differences.size),
    )


def compare(run: Run, key: KeyState) -> Comparison:
    if run.missing:
        raise IncompleteRun(f"{len(run.missing)} Claude readings have no recording")
    if not key.complete:
        raise IncompleteRun("the answer key is not complete")
    claude, claude_marks = _score("claude", run, key)
    runs = {reader: _score(reader, run, key) for reader in REGEX_READERS}
    best = max(REGEX_READERS, key=lambda r: (runs[r][0].correct, r == "regex_native"))
    regex, regex_marks = runs[best]
    ids = [p.task.task_id for p in run.prepared]
    verdicts = {t: _VERDICTS[(claude_marks[t], regex_marks[t])] for t in ids}
    claude_only = sum(claude_marks[t] and not regex_marks[t] for t in ids)
    regex_only = sum(regex_marks[t] and not claude_marks[t] for t in ids)
    p = mcnemar_p(claude_only, regex_only)
    if p < SIGNIFICANCE:
        status = "beats" if claude_only > regex_only else "loses"
    else:
        status = "not_significant"
    differences = np.array([float(claude_marks[t]) - float(regex_marks[t]) for t in ids])
    evaluation = EvalResult(
        metric="accuracy",
        score=claude.accuracy,
        baseline_name="regex readers",
        baseline_score=regex.accuracy,
        n_observations=len(ids),
        higher_is_better=True,
        fold_unit="query",
        paired=_paired(differences),
        notes=[f"The regex score is the stronger of its two runs, {best.replace('_', ' ')}."],
    )
    recall, hits, n = recall_at_k(run, key)
    return Comparison(
        claude=claude,
        regex=regex,
        regex_runs={reader: score for reader, (score, _) in runs.items()},
        verdicts=verdicts,
        claude_only=claude_only,
        regex_only=regex_only,
        mcnemar_p=p,
        verdict_status=status,
        evaluation=evaluation,
        recall=recall,
        recall_hits=hits,
        recall_n=n,
    )


def verdict_text(c: Comparison) -> str:
    return (
        f"{c.evaluation.verdict()} The readers disagree on {c.claude_only + c.regex_only} of "
        f"{c.claude.n} tasks: the Claude reader alone is right on {c.claude_only} and the "
        f"regex readers alone on {c.regex_only}, an exact McNemar p of {c.mcnemar_p:.3f}."
    )
