#!/usr/bin/env python3
"""The "reference" condition of PARK gate C's self-proof: one owner, without torch (the judge sidecar's python3 reads
it).

Rule: when gate C's self-proof (the one-step fall reference against the measured in-air median) disagrees, but the
batch's real fall (free fall of lifted objects, ``n_ff``) is 0 and the gate D count is 0 on every side, the batch is a
reference line, not invalid. Any count > 0, or gate D not measured (no count, None), keeps it invalid. A live judge
calls ``proof_reference`` and ``proof_text`` from ``dexengine.rl.judge.park.park_verdict``; recounting old judge logs
(the sidecar's park_bad) decides with the same functions through ``line_still_invalid``.

Reading an old log line (``line_still_invalid``): the only invalid segment of an ``[EVAL_PARK] batch ...`` line
(``INVALID(gate X — ``) is gate C's self-proof segment (PROOF_HEAD ... PROOF_MARK); the same line's
``lifted-object free fall N(`` and ``gate D free fall R a[ / L b] (gate > `` are read (per side, '?' = not measured)
and when ``proof_reference`` holds the line is not invalid under today's rule. An unreadable line stays invalid. The
patterns are the judge log's own words; logs written with the older Korean words are read after the legacy-value
migration rewrites them.
CLI: ``python3 -m dexengine.rl.judge.park_recount still-invalid`` prints the stdin lines that are still invalid under
today's rule (bytes unchanged).
"""

from __future__ import annotations

import re
import sys

PROOF_HEAD = "INVALID(gate C — fixed-pose one-step fall reference"
PROOF_MARK = "the reference formula disagrees with physics"
_SEG = re.compile(r"INVALID\(gate [A-Z] — ")
_NFF = re.compile(r"lifted-object free fall (\d+)\(")
_DSEG = re.compile(r"gate D free fall (.+?) \(gate > ")  # park_batch: "R a / L b (gate > lim)"; unmeasured side "?"


def proof_reference(n_ff: int, d_count: dict) -> bool:
    """Whether a gate C self-proof mismatch is lowered to a reference line: real fall 0 and the gate D count 0 on every
    side (no count or an empty count is not)."""
    return n_ff == 0 and bool(d_count) and all(c == 0 for c in d_count.values())


def proof_text(ref: bool, d_mm: float, k_air: int, med: float) -> str:
    """The judge-line segment of a gate C self-proof mismatch: a reference segment when ref (proof_reference) holds,
    else the invalid segment (its text is read by the judge table's reason column, the sidecar's park_gates and
    line_still_invalid, so it never changes)."""
    if ref:
        return (
            f"NOTE(gate C self-proof — reference −{d_mm:.2f} mm ≠ in-air {med:+.2f} mm (samples {k_air}) — "
            f"{PROOF_MARK}, but fall 0 · gate D 0)"
        )
    return (
        f"INVALID(gate C — fixed-pose one-step fall reference −{d_mm:.2f} mm ≠ measured in-air median "
        f"{med:+.2f} mm (samples {k_air}) — {PROOF_MARK}; gate C cannot be measured)"
    )


def line_still_invalid(line: str) -> bool:
    """Whether an old [EVAL_PARK] line is still invalid under today's rule (module docstring)."""
    if "INVALID(" not in line:
        return False
    segs = _SEG.findall(line)
    if len(segs) != 1 or PROOF_HEAD not in line or PROOF_MARK not in line or line.count("INVALID(") != 1:
        return True
    nff, dseg = _NFF.search(line), _DSEG.search(line)
    if nff is None or dseg is None:
        return True
    parts = dseg.group(1).split(" / ")
    d_count = {}
    for part in parts:
        m = re.fullmatch(r"([RL]) (\d+)", part)
        if m is None:  # '?' (not measured) or an unknown form: not lowered
            return True
        d_count[m.group(1)] = int(m.group(2))
    return not proof_reference(int(nff.group(1)), d_count)


def main(argv: list[str]) -> int:
    if argv != ["still-invalid"]:
        print("[PARK_RECOUNT] ✗ usage: still-invalid (stdin -> the lines still invalid)", file=sys.stderr)
        return 2
    for raw in sys.stdin.buffer:  # bytes: UTF-8 whatever the judge host locale (C, POSIX)
        if line_still_invalid(raw.decode("utf-8", errors="replace")):
            sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
