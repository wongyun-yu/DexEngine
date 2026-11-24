"""Whether a process has ended — the one owner of the rule, for Python and for the shell (its CLI): a process has
ended when /proc/<pid>/stat is gone or its state is in ENDED_STATES — Z (zombie: its parent has not waited yet, though
kill -0 still answers for it) · X. Nothing else counts: neither the group nor the session it led. A stat that cannot
be read for another reason is no proof of an end (a claim is kept while in doubt).

Python: the campaign's process lifecycle module re-exports it (alive · stop build on ended). Shell: the host gate
(scripts/rl/host_gate.sh host_gate_ended) runs `python3 proc_ended.py ended <pid>` → rc 0 ended · 1 not · 2 usage.
Standard library only — a pod's stage 1 imports it under the image's interpreter; the judge's gate runs it, so it is
on the judge's code path (not an orchestration file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: /proc/<pid>/stat states of a process that has ended — zombie · dead.
ENDED_STATES = ("Z", "X")


def proc_stat(pid: int) -> tuple[str, int, int, int] | None:
    """(state, process group, session, start time in clock ticks) of /proc/<pid>/stat — the fields after the last ')'
    (the command name may hold one): 3 · 5 · 6 · 22. None = no such process (ended and reaped); any other read
    failure raises (OSError · ValueError) — the caller decides what a doubt means."""
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return f[0], int(f[2]), int(f[3]), int(f[19])


def ended(pid: int) -> bool:
    """The pid's process has ended (module doc). A stat that cannot be read for another reason → False."""
    try:
        st = proc_stat(pid)
    except (OSError, IndexError, ValueError):
        return False
    return st is None or st[0] in ENDED_STATES


if __name__ == "__main__":  # CLI (module doc) — rc 0 ended · 1 not · 2 usage (argparse's own rc for a bad line)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("ended",))
    ap.add_argument("pid", help="decimal digits only")
    a = ap.parse_args()
    if not a.pid.isdigit():
        ap.error(f"pid {a.pid!r}: decimal digits only")
    sys.exit(0 if ended(int(a.pid)) else 1)
