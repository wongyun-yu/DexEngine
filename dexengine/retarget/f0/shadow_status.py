"""f0 shadow statuses: which row statuses are OK or judgeable, data defects vs run failures, the per-clip ledger and
its last row (read by prejudge, the f0 ledger and the stage-1 plan)."""

from __future__ import annotations

from pathlib import Path

from dexengine.core.paths import qpos_layout
from dexengine.retarget.f0 import ledger as f0_summary

# The success set of ledger statuses — **this tuple is the only owner**. Anything that is not a success
# counts as a failure (the rule keeps applying when a new status is added — a "FAIL" prefix
# check would miss REPAIR_FAIL and drop it out of stage2's failure list).
OK_STATUSES = ("REPAIRED", "CLEAN", "SHALLOW", "CACHED")


# FAIL_NO_BUDGET_MARK: a run in which f0.repair did not print its per-side `[F0_BUDGET]` marker once per side. It
# fails loudly so an old tree or interpreter is never passed over silently: without the marker proving the switch was
# applied, the run is invalid. It is in neither the success set nor the judgeable set: its output cannot say what the
# budget was.
# Statuses that **may enter** judging = statuses for which the shadow (the staging copy) exists.
# REPAIR_FAIL (hold pad kept) and REFUSED_BIG (an over-large repair refused, original kept) could not be
# repaired, but they can still go through judging, baking and the gate, and the gate makes the final
# call. Excluding these two would lose clips the gate adopts.
# Counting failures (OK_STATUSES) and eligibility to be judged (JUDGEABLE_STATUSES) are different
# questions, so both are owned here.
JUDGEABLE_STATUSES = OK_STATUSES + ("REPAIR_FAIL", "REFUSED_BIG")


def is_judgeable(status: str) -> bool:
    return status in JUDGEABLE_STATUSES


def is_ok_status(status: str) -> bool:
    return status in OK_STATUSES


# The two kinds of non-judgeable status; these three (DATA_DEFECT, SourceDefectError, is_run_failure) are the only
# owners.
#   data defect  = only what a producer declares by raising `SourceDefectError` (status DATA_DEFECT). It goes through
#                  stage2's f0_failed.txt to the status-ledger status SOURCE_DEFECT.
#   run failure  = every other non-judgeable status (FAIL_EXC, FAIL_F0, FAIL_MIRROR, FAIL_GPU_WORKER,
#                  FAIL_NO_BUDGET_MARK, FAIL_NOLEDGER, and names added later). Not a clip verdict: stage2 fails the
#                  chunk.
# The split is by declaration, not by exception text or type name: the same ValueError could be a data verdict ('no
# manipulating hand') or a missing mesh file (trimesh 'string is not a file').
# The GPU worker side (f0.repair) has no declaration path: whatever it raises, rc != 0 = FAIL_F0 = a run failure
# (the unknown goes to failure, not to a verdict; declaring data defects from the worker needs a protocol field first).
DATA_DEFECT = "DATA_DEFECT"


class SourceDefectError(ValueError):
    """Declaration of a defect in the content of the input data. Raised by the producer function that judges the defect
    (today only `clip_sides`: D23 no_main_object). Subclasses ValueError, so callers catching ValueError still get it.
    The message is recorded as the clip's ledger note and keeps its exact text."""


def is_data_defect(status: str) -> bool:
    return status == DATA_DEFECT


# Held = the f0 object repair (repair_obj) refused a side: the hand cannot be moved out of the object within the rule
# (a contact link would be lost, or the check fails after the retries). The clip is not judged; it is a rule rejection
# (the reference cannot be made penetration-free within the rule, so the trajectory is dropped), counted (completion
# line, [F0_FAILCLASS] held=, stage2's f0_held.txt → placement ledger HELD, ledger_states.REJECTED) with the refusal
# reason as its ledger note. It writes no shadow marker, so a later f0 run solves it again — a solver fix re-runs
# exactly the held clips.
HELD = "HOLD_OBJ"


def is_held(status: str) -> bool:
    return status == HELD


def is_run_failure(status: str) -> bool:
    """A status that is neither judgeable, a data defect nor held = a run failure; unknown new status names land here
    too."""
    return not is_judgeable(status) and not is_data_defect(status) and not is_held(status)


def failclass_marker(rows) -> str:
    """Marker with this run's counts of the two non-judgeable kinds. The consumer (stage2) reads the ledger, not this
    line (ledger.py split-failures)."""
    dd = sum(1 for r in rows if is_data_defect(r["status"]))
    rf = sum(1 for r in rows if is_run_failure(r["status"]))
    hd = sum(1 for r in rows if is_held(r["status"]))
    return f0_summary.failclass_line(dd, rf, len(rows) - dd - rf - hd, hd)  # line format owned by f0_summary


SIDES = qpos_layout.SIDES  # owner of the side vocabulary


#: Per-clip ledger, one file right under this hand's shadow root (`<F0>/<hand spec>/`); append-only (written by main
# per clip).
LEDGER_NAME = "LEDGER.tsv"


def ledger_last(led: Path, want=None) -> dict:
    """Last row per clip of the ledger -> {clip: (status, note)}. The ledger is append-only, so the last row is the
    clip's current status. Readers (stage2 split-failures, stage-1 step 7 data.stage1.plan.f0_cells) call this one
    function. With want (a set), only those clips are kept. A missing ledger file gives an empty dict; the caller
    decides what that means."""
    last = {}
    if not Path(led).is_file():
        return last
    with open(led, encoding="utf-8", errors="replace") as fh:
        next(fh, None)  # header
        for ln in fh:
            p = ln.rstrip("\n").split("\t")  # clip · status · … · note(last column)
            if len(p) >= 2 and (want is None or p[0] in want):
                last[p[0]] = (p[1], p[-1] if len(p) >= 8 else "")
    return last


def unsettled_reason(status, shadow_marker: Path) -> str:
    """One clip of stage-1 step 7: "" = settled, else the kind of gap (`no_ledger`, `run_failure`, `no_marker`).
    Settled = the last ledger status is not a run failure (judgeable, a declared data defect or held, the complement
    of is_run_failure) and, when judgeable, its shadow marker (MARKER) exists (judgeable = a status with a shadow,
    JUDGEABLE_STATUSES). A data defect normally has no shadow. An old marker does not count when the last row is a run
    failure: counting markers would hide run-failure clips behind reuse."""
    if status is None:
        return "no_ledger"
    if is_run_failure(status):
        return "run_failure"
    if is_judgeable(status) and not Path(shadow_marker).is_file():
        return "no_marker"
    return ""
