"""Launch steps of the record driver (record.driver main; rules in its module doc): input and machine gates
(check_env), operations place, ledger copy and queue (prepare), the run claim (claim: pilot.json, driver.pid).
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import sys
import time
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import input_missing
from dexengine.core.hashing import sha256_file
from dexengine.record import driver_spec as SPM  # spec mode, final tally
from dexengine.record import piece as PC
from dexengine.record import sources as OWN  # launch owner ledger vs frozen copy check
from dexengine.record.driver_check import adopted, bdir_name, bundle_queue, die, now, say, write_atomic
from dexengine.record.piece import merge_gate  # merge call


def pid_alive_as_driver(pid: int) -> bool:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "collect_pilot.py" in cmd


def check_env(a) -> None:
    """Input and machine gates (no defaults; no launch when not met)."""
    a.queue_root = os.environ.get("DEXENGINE_RL_QUEUE_ROOT", "")
    if not a.queue_root:
        die(input_missing("DEXENGINE_RL_QUEUE_ROOT is missing (no default)"))
    if not os.environ.get("DEXENGINE_MAIN_ROOT"):
        die(input_missing("DEXENGINE_MAIN_ROOT (data root) is not set — launch with scripts/stage4_record.py drive"))
    a.tree = a.tree.resolve()
    pkg = str(a.tree)
    if a.tree != P.main_root().resolve() and pkg not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        die(
            input_missing(
                f"worktree {a.tree} but {pkg} is not on PYTHONPATH — every session's tree guard would fail; "
                "put the tree on PYTHONPATH"
            )
        )
    a.main = P.main_root().resolve()
    if not getattr(a, "arm", None):
        die(
            input_missing(
                "--arm is required (no default): the record combo is <arm>_<hand> (registered: paths --what combo-ids)"
            )
        )
    a.combo = P.combo_for(a.hand, a.arm)
    a.gpus = [x for x in a.gpus.split(",") if x]
    a.exclusive = {x for x in a.exclusive_gpus.split(",") if x}
    if a.exclusive - set(a.gpus):
        die(f"--exclusive-gpus {sorted(a.exclusive)} outside --gpus {a.gpus}")
    a.only = {x for x in a.only.split(",") if x}
    a.py = sys.executable
    PC.round_gate(a)
    if a.spec_mode:
        SPM.spec_timeout(a)
    a.merge_fn = merge_gate(a.merge)  # before the claim: on without a merge module dies


def prepare(a) -> None:
    """Operations place, ledger copy (fixed condition; a relaunch gets the same copy), queue. Missing ledger file = rc
    12 · sha mismatch = rc 13."""
    SPM.need_file(a.owner)
    SPM.owner_gate(OWN.verify_src, a.owner, a.owner_sha)
    a.run_dir = P.run_path("run_collect", a.hand, root=a.main, tag=a.tag)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    a.ledger = a.run_dir / "ledger.tsv"
    if not a.ledger.is_file():
        shutil.copyfile(a.owner, a.ledger)
        say(f"ledger copy {a.owner} -> {a.ledger}")
    a.ledger_sha = sha256_file(a.ledger)
    a.queue = SPM.spec_queue(
        a, bundle_queue(a.ledger, SPM.spec_order(a), a.min_rate, adopted(a.hand), subset=a.clip_subset)
    )
    a.owner = a.owner.resolve()
    SPM.owner_gate(OWN.check, a.owner, a.ledger, [c for b in a.queue for c in b["clips"]])  # copy != owner = 13
    if a.only - {bdir_name(b) for b in a.queue}:
        die(input_missing(f"--only {sorted(a.only - {bdir_name(b) for b in a.queue})} not in the queue"))
    by = collections.Counter(b["dataset"] for b in a.queue)
    cl = collections.Counter()
    for b in a.queue:
        cl[b["dataset"]] += len(b["clips"])
    a.order = [x for x in a.order.split(",") if x]
    say(
        f"queue bundles {len(a.queue)} · clips {sum(cl.values())} · "
        + " · ".join(f"{d} {by[d]}/{cl[d]}" for d in a.order)
        + f" · ledger sha {a.ledger_sha[:12]} · combo {a.combo}"
    )
    # applied marker: root of the work place run_collect_work
    wr = PC.work_root(a.main)
    print(f"[COLLECT_WORK_ROOT] {wr} · {'job tree (env)' if wr != Path(a.main) else 'data root'}", flush=True)


def claim(a) -> None:
    """pilot.json (first start, fixed deadline, launch record) · driver.pid (duplicates refused)."""
    pj = a.run_dir / "pilot.json"
    meta = json.loads(pj.read_text()) if pj.is_file() else {}
    PC.round_claim(meta, a.round, pj)
    a.started = meta.get("started") or now()
    a.deadline_ts = meta.get("deadline_ts") or (time.time() + a.hours * 3600)
    launches = meta.get("launches", []) + [{"at": now(), "pid": os.getpid(), "argv": sys.argv[1:]}]
    write_atomic(
        pj,
        json.dumps(
            dict(
                meta,
                round=a.round,
                started=a.started,
                deadline_ts=a.deadline_ts,
                hours=meta.get("hours", a.hours),
                launches=launches,
            ),
            indent=1,
            ensure_ascii=False,
        ),
    )
    pf = a.run_dir / "driver.pid"
    if pf.is_file():
        old = int(pf.read_text().split()[0] or 0)
        if old and old != os.getpid() and pid_alive_as_driver(old):
            die(f"a driver already runs pid {old} ({pf}): duplicate refused")
    write_atomic(pf, f"{os.getpid()} {now()} collect_pilot.py --tag {a.tag}\n")
