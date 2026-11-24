"""Failure kinds that a log reader classifies from the printed text alone.

A process that dies on a missing input (a file, record, row, argument or environment value it needs) raises
InputMissingError, or prints a `✗` line through input_missing(). Both put INPUT_MISSING into the text, and
record.driver_spec.bundle_rc reads that marker from the bundle log to report rc 12 (INPUT_MISSING) instead of rc 1
(reasons stored before the marker are rewritten to the marked text by the site migration).
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
