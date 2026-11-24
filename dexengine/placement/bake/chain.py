"""Run several bake stages in **one process** — the stage table and its runner.

The chain (`scripts/stage2/chain_fast.sh`) sends one clip's stage list — bake → body → wrist (both sides) → ik
(both sides) [→ xr] → retime → body → wrist (both sides) — to a resident worker (`bake/server.py`), which calls
`run_specs`. All stages run in the Isaac venv interpreter the chain starts the worker with; there is no other
entry. A stage = '<stage>[:arg,arg,...]': its module's `main()` is called with sys.argv swapped. A stage that
returns a non-zero code or raises stops the list with that code.

The stage modules are imported lazily by module name (`importlib.import_module` over the STAGES table), so
importing this module stays light (ledger readers and shell heredocs use `abort_line` / `halfwrite_lengths`).
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

from dexengine.core.repo import REPO_ROOT as WT

#: Stage name → module. The order of a stage list is the caller's.
STAGES = {
    "bake": "dexengine.placement.bake.combo",
    "body": "dexengine.placement.fk.body_positions",
    "wrist": "dexengine.placement.fk.wrist_poses",
    "retime": "dexengine.placement.retime.retime",
    "ik": "dexengine.placement.coarse_path.dp",
    # cross repair: one clip's PRE → xr → POST (retime → body/wrist) in one interpreter
    "xr": "dexengine.placement.repair.cross_repair",
}
#: Stage name → repository-relative source file (readers hash these files as the bake code identity).
SCRIPTS = {name: mod.replace(".", "/") + ".py" for name, mod in STAGES.items()}
#: Modules split off a stage script: their code is part of that stage, so the bake code identity hashes them too.
STAGE_PARTS = (
    "dexengine/placement/coarse_path/dp_solve.py",
    "dexengine/placement/repair/cross_steps.py",
    "dexengine/placement/repair/cross_report.py",
)
# The pass rc convention is owned by cross_repair (0 = clean / repaired · 3 = residual, recorded and continue ·
# anything else = failure); the output-level check (the report file cross_repair.REPORT_REL exists) is the same
# condition chain_fast uses.
XR_PASS_RC = (0, 3)


#: Head of every stop line (abort_line · fault_line · the cross-repair stops) — read by bake_drive.sh (ledger reason
#: grep), chain_fast.sh (cross-repair case) and record.augment.bake_run (variant detail).
STOP = "[runner] stopped —"


def abort_line(name: str) -> str:
    """The line printed when a stage **returns** a non-zero rc and the list stops (printed by run_specs, read by
    ledger.bake_fail_status — the byte value is part of the ledger contract). Only a returned rc is the stage's
    verdict (the DP stage's rejection rc 3 · 4). A stage that ends with SystemExit prints fault_line; an unhandled
    exception (the resident server turns it into rc 1) and a chain killed by timeout leave neither line."""
    return f"{STOP} {name} failed"


def fault_line(name: str) -> str:
    """The line printed when a stage ends with SystemExit — a fixed line different from abort_line, so whatever
    precedes it is never read as a verdict. The bake ledger reason grep (bake_drive.sh, 'stopped') carries it."""
    return f"{STOP} {name} SystemExit exit"


_CACHE: dict[str, object] = {}
# Pure cache the resident worker (bake/server.py) may keep between clips (keys do not depend on the clip); every
# other global is reset to its import-time value per request.
_SERVER_KEEP = ("_CACHE",)  # stage module objects (code)


def _mod(name: str):
    """The module of stage `name` (imported once)."""
    if name in _CACHE:
        return _CACHE[name]
    m = importlib.import_module(STAGES[name])
    _CACHE[name] = m
    return m


def run_one(spec: str) -> tuple[int, bool]:
    """One stage → (rc, ended with SystemExit) — the second value picks run_specs' stop line (abort_line ·
    fault_line)."""
    name, _, argstr = spec.partition(":")
    rel = SCRIPTS.get(name)
    if rel is None:
        print(f"[runner] unknown stage: {name}", flush=True)
        return 2, False
    argv = [x for x in argstr.split(",") if x != ""] if argstr else []
    # the bake module has subcommands: the stage name is its first argument
    if name == "bake":
        argv = [name] + argv
    t0 = time.time()
    m = _mod(name)
    old = sys.argv
    sys.argv = [str(WT / rel)] + argv
    exited = False
    try:
        if name == "xr":
            # An unhandled exception is rc 1 (the same meaning as for a separate Python process).
            try:
                rc = m.main()
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception:  # noqa: BLE001 — an unhandled exception = rc 1, as for a separate process
                import traceback

                traceback.print_exc()
                rc = 1
        else:
            rc = m.main()
    except SystemExit as e:  # argparse errors (int code) or contract violations (str code)
        exited = e.code is not None
        if e.code is None:
            rc = 0
        elif isinstance(e.code, int):
            rc = e.code
        else:
            # A string code (e.g. a bake contract violation SystemExit(f"...")) is kept on stderr as is and
            # counts as rc 1 (int(e.code) would hide it behind a ValueError).
            print(str(e.code), file=sys.stderr, flush=True)
            rc = 1
    finally:
        sys.argv = old
    rc = 0 if rc is None else int(rc)
    print(f"[runner] {name} rc={rc} ({time.time() - t0:.1f}s)", flush=True)
    return rc, exited


def halfwrite_lengths(ik_right, ik_left, kin) -> tuple[int, int, int]:
    """Half-write lengths (R, L, qpos) — rows of `arm_joint_positions_all` in the right and left arm-IK npz and of
    the tree `qpos`; −1 when missing or unreadable. The meaning is the caller's: the chain's failure path calls
    different values a half write, the chain's length gate fails on any −1 or difference, the end check of
    bake_drive calls a difference or R < 0 a half write (a chain killed by timeout never runs its own check)."""
    import numpy as np  # noqa: PLC0415 — shell heredocs call only this function (no stage module import)

    def n(p, k):
        try:
            return len(np.load(p)[k])
        except Exception:  # noqa: BLE001 — any unreadable file or member counts as -1 (a half write)
            return -1

    return n(ik_right, "arm_joint_positions_all"), n(ik_left, "arm_joint_positions_all"), n(kin, "qpos")


def run_specs(specs, before=None) -> int:
    """Run a stage list in order (the resident server calls this per request).
    before(name, module) is the server's hook right before each stage (None = no hook, same behaviour)."""
    t0 = time.time()
    for spec in specs:
        if before is not None:
            _n = spec.partition(":")[0]
            if _n in SCRIPTS:
                before(_n, _mod(_n))
        rc, exited = run_one(spec)
        name = spec.split(":")[0]
        if name == "xr":
            if rc not in XR_PASS_RC:
                print(f"{STOP} xr failed rc={rc}", flush=True)
                return rc
            _a = [x for x in spec.partition(":")[2].split(",") if x]
            _tree = Path(_a[_a.index("--tree") + 1]) if "--tree" in _a else None
            _rep = (_tree / _mod("xr").REPORT_REL) if _tree else None  # report location = cross_repair's
            if _rep is None or not _rep.is_file() or _rep.stat().st_size == 0:
                print(f"{STOP} xr report missing ({_rep})", flush=True)
                return 98
            continue
        if rc != 0:
            print((fault_line if exited else abort_line)(name), flush=True)
            return rc
    print(f"[runner] all {len(specs)} stages done ({time.time() - t0:.1f}s)", flush=True)
    return 0
