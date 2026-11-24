"""Failure kinds that a log reader classifies from the printed text alone.

A process that dies on a missing input (a file, record, row, argument or environment value it needs) raises
InputMissingError, or prints a `✗` line through input_missing(). Both put INPUT_MISSING into the text, and
record.driver_spec.bundle_rc reads that marker from the bundle log to report rc 12 (INPUT_MISSING) instead of rc 1
(reasons stored before the marker existed carry no marker and read as rc 1).
Other failure kinds travel in a cause field `cause=<kind>` of the failure sentence (CAUSES: a field a machine reads,
not a word of the text): timeout (a time cap), crash (a child rc != 0), no_done (rc 0 without the end line), stop
(a SystemExit of our code), error (a Python exception). Input missing is not a field value: its one statement is the
INPUT_MISSING marker, and cause_of reads it. Unknown values die (cause, cause_of).
No dependencies: any package and any process may import it.
"""

from __future__ import annotations

import re

INPUT_MISSING = "[INPUT_MISSING]"

_LEAD_TAG = re.compile(r"^(\s*\[[^\]\n]+\])(\s*)")


def input_missing(msg: str) -> str:
    """msg with the INPUT_MISSING marker. The marker goes after a leading `[TAG]` so readers keyed on the tag keep
    matching, otherwise in front."""
    msg = str(msg)
    if INPUT_MISSING in msg:
        return msg
    m = _LEAD_TAG.match(msg)
    if m:
        return f"{m.group(1)} {INPUT_MISSING} {msg[m.end() :]}"
    return f"{INPUT_MISSING} {msg}"


class InputMissingError(SystemExit):
    """SystemExit for a missing input: the process exits as with SystemExit(msg) (rc 1, message on stderr) and the
    message carries INPUT_MISSING."""

    def __init__(self, msg: str):
        super().__init__(input_missing(msg))


def is_input_missing(text: str) -> bool:
    """Whether a log line (or failure reason) is an input-missing failure."""
    return INPUT_MISSING in (text or "")


CAUSES = ("timeout", "crash", "no_done", "stop", "error")
CAUSE_INPUT_MISSING = "input_missing"  # cause_of's name for the INPUT_MISSING marker (never a field value)
_CAUSE_RE = re.compile(r"\bcause=(\w+)")


def cause(kind: str) -> str:
    """The cause field for a failure sentence: `cause=<kind>`, or the INPUT_MISSING marker for input_missing (its one
    statement). An unknown kind dies."""
    if kind == CAUSE_INPUT_MISSING:
        return INPUT_MISSING
    if kind not in CAUSES:
        raise ValueError(f"cause {kind!r}: one of {CAUSES} or {CAUSE_INPUT_MISSING!r}")
    return f"cause={kind}"


def cause_of(text: str) -> str | None:
    """The cause of a failure text: input_missing when it carries the INPUT_MISSING marker, else the first cause field
    value, else None. Any unknown field value dies."""
    vals = _CAUSE_RE.findall(text or "")
    bad = [v for v in vals if v not in CAUSES]
    if bad:
        raise ValueError(f"cause field {bad[0]!r}: one of {CAUSES} (input missing = the {INPUT_MISSING} marker)")
    if is_input_missing(text):
        return CAUSE_INPUT_MISSING
    return vals[0] if vals else None
