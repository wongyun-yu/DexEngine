"""One tick of the RL campaign launch loop (dexengine/cluster/local.py runs the ticks; dexengine/rl/campaign.py
--launch runs the loop) — the tick joins the runner's pieces:

  reserve   the campaign's judge GPUs stay judge-only (host gate kind rsv under the loop's pid — taken at the start,
            re-asserted each tick, released at the end): training and data claims, holds and the data lanes' room skip
            them, and non-judge claims leave one judge's RAM free for each (the gate's judge share)
  read      every order bundle's state from its cell (bundle_state — derived from the cell's
            files, no second statement); a bundle without a launch record is queued
  stop      a failed bundle's live sidecar (its training failed — nothing left to judge) is stopped by its own session
            (the pid is checked by its command line first — never a pattern kill)
  sidecar   a launched bundle whose training lives or ended with rc 0, with no judge rc and no live sidecar, gets its
            sidecar started (again) with the recorded command and its launch id — judge.starts counts the starts and
            bundle_state turns the bundle failed past the sidecar start limit
  overflow  at a training's end (train.rc 0) the overflow step decides once (overflow.overflow_at_train_end —
            the model's rule and quantity) and records it (overflow.txt); undecided (the training still lives, the gate
            refused the hold for now) = next tick
  finalize  a done, unfinalized cell is finalized (rl.cells.finalize, prune per the campaign) — a refusal is
            reported, never retried silently
  launch    none while stopping (SIGTERM), while a drain is requested (drain.json — the loop ends drained once
            nothing is in flight) or while the code tree (ctx.code) is off the loop's start. Else
            the free training GPUs (no live claim or hold — host gate list) take the next queued bundles in order whose
            launch evidence (ctx.canary — canaries + run key smoke/probe) counts under the
            loop's training-path fingerprint and whose disk fits (in_flight · disk_gate): input hash here (ctx.inputs —
            refused = launch_refused, stays queued) → launch record (cold start, code and training-path fingerprints) →
            launch_cmd.txt → gate claim + fork (rl.cell_state.launch) → train.gpu → sidecar; a refusal rolls back
  status    the tick's status dict (summarize — states, GPUs, idle GPUs with their reason, events); the loop around the
            ticks (lock, frozen settings, errors, SIGTERM, status files) is dexengine/cluster/local.py
A failed bundle is never relaunched here (`python -m dexengine.rl.cell_tools relaunch` clears it with a
ledger). The campaign ends when every queued bundle is terminal (done and finalized, or failed) and nothing is
in flight: complete only when the done set equals the queue set (a set difference, not a count).
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

TERMINAL = ("done", "failed")


@dataclass
class Ctx:
    """What a tick needs — built by the campaign runner (production) or a selftest (fakes). bundles = the campaign order
    (dicts with hand); blocked = {(hand, id): why} of order bundles that may not be queued."""

    campaign: str
    host: str
    train_gpus: list[str]
    judge_gpus: list[str]
    overflow_h: float | None
    prune: bool
    root: Path
    bundles: list[dict]
    blocked: dict
    mans: dict
    cdirs: dict
    cell_of: Callable[[dict], Path]
    commands: Callable[[dict, Path], dict]
    canary: Callable[[dict, str, bool], str]  # (bundle, training-path fp, exempt) → "" ready · else why not
    inputs: Callable[[dict], str]  # P5 at launch: "" = inputs here hash to the audit record
    git: Callable[[], dict]
    code: Callable[[], str]  # the code tree fingerprint now
    train: Callable[[], str]  # the training-path fingerprint now
    per_run_gb: float
    code0: str = ""  # the fingerprint the loop started on (run_loop / first tick)
    train0: str = ""  # the training-path fingerprint the loop started on
    ident: dict = field(default_factory=lambda: {"cmd_fp": "", "inherit_env": []})  # other parts of loop_files.IDENT
    ident_first: str = ""  # the first start's launch identity (loop_files.ident_of)
    exempt: bool = True  # CANARY_VALIDATED holds (identity = the first start's)
    resources: dict = field(default_factory=dict)  # declared resources — {} = host path
    stop: list = field(default_factory=list)  # SIGTERM seen — no launch, the loop ends at the next check
    notes: dict = field(default_factory=dict)  # once-only events (a lost reservation · a code change)
    events: list = field(default_factory=list)


def in_flight(cells: list[Path]) -> int:
    """Runs holding their full checkpoint set on disk: every launched cell not yet finalized (D9)."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    return sum(CH.cell_file(c, "launch").is_file() and not CH.cell_file(c, "finalized").is_file() for c in cells)


def disk_gate(path: Path, in_flight_runs: int, per_run_gb: float) -> tuple[bool, str]:
    """A launch needs free disk for every run in flight plus this one to write all its checkpoints (pruned only at
    finalize)."""
    free = shutil.disk_usage(path).free / 1e9
    need = (in_flight_runs + 1) * per_run_gb
    return free >= need, (
        f"free disk {free:.0f} GB · needed {need:.0f} GB ((in flight {in_flight_runs} + 1) × {per_run_gb:.1f})"
    )


def _run_of(sdir: Path) -> str:
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    return json.loads(CH.cell_file(sdir, "launch").read_text(encoding="utf-8"))["run"]


def _pid(sdir: Path, key: str) -> int | None:
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    try:
        return int(CH.cell_file(sdir, key).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _rc(sdir: Path) -> str | None:
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    f = CH.cell_file(sdir, "train_rc")
    return f.read_text().strip() if f.is_file() else None


def _event(ctx: Ctx, kind: str, b: dict | None, why: str) -> None:
    """One events.tsv row — a tab or newline inside `why` (an exception text) would break the row, so they go."""
    ctx.events.append(
        (
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            kind,
            b["hand"] if b else "-",
            b["id"] if b else "-",
            " ".join(str(why).split()),
        )
    )


def step_reserve(ctx: Ctx) -> None:
    """Re-assert the judge-only reservation of the judge GPUs (module doc) — a refusal is an event once and the
    status' judge_reserve (the judge backlog then counts what sits on a judge GPU)."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    ok, why = CH.reserve_judges(ctx.judge_gpus, os.getpid(), ctx.campaign)
    if ok:
        ctx.notes.pop("reserve", None)
    elif ctx.notes.get("reserve") != why:
        ctx.notes["reserve"] = why
        _event(ctx, "judge_reserve_lost", None, why)


def read_all(ctx: Ctx) -> dict:
    """{(hand, id): state} of every order bundle — a record outside the rule's cells dies (read_states)."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415
    from dexengine.rl.cell_state import bkey  # noqa: PLC0415

    byid = {bkey(b): b for b in ctx.bundles}
    got = CH.read_states(ctx.cdirs, lambda h, bid: ctx.cell_of(byid[(h, bid)]))
    extra = sorted(set(got) - set(byid))
    if extra:
        raise ValueError(f"launch records outside the order {extra[:3]} — cells hold bundles not in this campaign")
    return {k: got.get(k, "queued") for k in byid}


def step_cleanup(ctx: Ctx, states: dict) -> None:
    """Stop the live sidecar of a bundle whose training failed (its session — the pid's command line names the run)."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415
    from dexengine.rl.cell_state import bkey  # noqa: PLC0415

    for b in ctx.bundles:
        sdir = ctx.cell_of(b)
        if states[bkey(b)] != "failed" or not CH.cell_file(sdir, "launch").is_file():
            continue
        run, jp = _run_of(sdir), _pid(sdir, "judge_pid")
        if jp is not None and CH.alive(jp, run) and CH.judge_verdict(run, sdir) is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(jp, signal.SIGTERM)
            _event(ctx, "sidecar_stop", b, f"training failed (train.rc {_rc(sdir)}) — stopped sidecar pid {jp} session")


def start_sidecar(ctx: Ctx, b: dict, sdir: Path) -> bool:
    """Start the bundle's sidecar with its recorded command (launch_cmd.txt judge=) and its launch id; one line in
    judge.starts per start."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    cmds = dict(
        ln.split("=", 1)
        for ln in CH.cell_file(sdir, "launch_cmd").read_text(encoding="utf-8").splitlines()
        if "=" in ln
    )
    lid = CH.launch_id_of(sdir)
    if not cmds.get("judge") or cmds["judge"] == "None" or not lid:
        _event(ctx, "sidecar_none", b, "no judge command or launch id — not starting the sidecar")
        return False
    try:
        pid = CH.spawn(sdir, "judge", cmds["judge"].replace(CH.LAUNCH_ID_SLOT, lid), ctx.root)
    except OSError as e:
        _event(ctx, "sidecar_fail", b, f"fork failed {e}")
        return False
    with open(CH.cell_file(sdir, "judge_starts"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={pid}\n")
    _event(ctx, "sidecar_start", b, f"pid {pid} · start {CH.sidecar_starts(sdir)}/{CH.SIDECAR_MAX_STARTS}")
    return True


def step_sidecars(ctx: Ctx, states: dict) -> None:
    from dexengine.rl import cell_state as CH  # noqa: PLC0415
    from dexengine.rl.cell_state import bkey  # noqa: PLC0415

    for b in ctx.bundles:
        if states[bkey(b)] not in ("training", "trained"):
            continue
        sdir = ctx.cell_of(b)
        run = _run_of(sdir)
        jp = _pid(sdir, "judge_pid")
        if (
            (jp is None or not CH.alive(jp, run))
            and CH.judge_verdict(run, sdir) is None
            and CH.sidecar_starts(sdir) < CH.SIDECAR_MAX_STARTS
        ):
            start_sidecar(ctx, b, sdir)


def step_overflow(ctx: Ctx, states: dict, cells: list[Path]) -> None:
    """The overflow decision once per training end (overflow.txt) — a live training = next tick."""
    from dexengine.cluster import overflow as CO  # noqa: PLC0415
    from dexengine.rl import cell_state as CH  # noqa: PLC0415
    from dexengine.rl.cell_state import bkey  # noqa: PLC0415

    for b in ctx.bundles:
        sdir = ctx.cell_of(b)
        if (
            states[bkey(b)] not in ("trained", "judging")
            or _rc(sdir) != "0"
            or CH.cell_file(sdir, "overflow").is_file()
        ):
            continue
        gpu = _gpu_of(sdir)
        if not gpu:
            CH.cell_file(sdir, "overflow").write_text("held=0 no training GPU recorded\n", encoding="utf-8")
            continue
        d, why = CO.overflow_at_train_end(_run_of(sdir), sdir, gpu, cells, ctx.judge_gpus, ctx.overflow_h)
        if d == "wait":  # not decided — asked again next tick (structure, not words)
            continue
        CH.cell_file(sdir, "overflow").write_text(f"held={int(d == 'held')} gpu={gpu} {why}\n", encoding="utf-8")
        _event(ctx, "overflow" if d == "held" else "overflow_no", b, why)


def step_finalize(ctx: Ctx, states: dict) -> dict:
    """Finalize every done cell not yet finalized → {(hand, id): refusal} of those refused."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415
    from dexengine.rl import cells as CL  # noqa: PLC0415
    from dexengine.rl.cell_state import bkey  # noqa: PLC0415

    errs = {}
    for b in ctx.bundles:
        sdir = ctx.cell_of(b)
        if states[bkey(b)] != "done" or CH.cell_file(sdir, "finalized").is_file():
            continue
        try:
            res = CL.finalize(_run_of(sdir), b["hand"], sdir, prune=ctx.prune, apply=True, root=ctx.root)
            _event(ctx, "finalized", b, f"moved {res['move']} · deleted {res['delete']} · {res['freed_gb']} GB")
        except (ValueError, RuntimeError) as e:
            errs[bkey(b)] = str(e)[:300]
            _event(ctx, "finalize_refused", b, str(e)[:300])
    return errs


def gpu_view(ctx: Ctx) -> dict[str, dict]:
    """{gpu: {claims: [...], held: run|None}} of the host's training and judge GPUs from the gate's live list."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    view = {g: {"claims": [], "held": None} for g in ctx.train_gpus + ctx.judge_gpus}
    for c in CH.gate_claims():
        if c["gpu"] in view:
            if c["kind"] == "hold":
                view[c["gpu"]]["held"] = c["run"]
            else:
                view[c["gpu"]]["claims"].append(f"{c['kind']}:{c['run']}")
    return view


def _launch_one(ctx: Ctx, b: dict, sdir: Path, gpus: list[str]) -> tuple[bool, str]:
    """Launch record → launch_cmd.txt → gate + fork (candidates `gpus`) → train.gpu → sidecar. (launched, why). A gate
    refusal rolls the record back (the cell reads queued again)."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415
    from dexengine.rl import cells as CL  # noqa: PLC0415

    why = ctx.inputs(b)  # before any record: a refusal leaves the cell untouched
    if why:
        raise ValueError(f"P5 input rehash refused — {why}")
    man = ctx.mans[b["hand"]]
    cmds = ctx.commands(b, sdir)
    CH.launch_record(sdir, b, man, cmds["run"], dict(ctx.git(), code=ctx.code0, train=ctx.train0), root=ctx.root)
    try:
        CL.write_launch_cmd(sdir, cmds)
        pid, g = CH.launch(
            "train",
            gpus,
            cmds["run"],
            cmds["train"],
            sdir,
            CH.TRAIN_RESERVE_GB,
            CH.TRAIN_RAMP_S,
            cwd=ctx.root,
            maxc=ctx.resources.get("train_par", 0),
        )
    except (ValueError, OSError, subprocess.SubprocessError):
        _rollback(sdir)  # nothing ran — the cell reads queued, the error is reported
        raise
    if pid is None:
        _rollback(sdir)
        return False, g
    CH.cell_file(sdir, "train_gpu").write_text(f"{g}\n", encoding="utf-8")
    _event(ctx, "launched", b, f"{cmds['run']} GPU {g} pid {pid}")
    if cmds.get("judge"):
        start_sidecar(ctx, b, sdir)
    return True, g


def _rollback(sdir: Path) -> None:
    """Undo a launch the gate refused: the record and command file go (the cell was proven empty) — it reads queued."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    for k in ("launch", "launch_cmd"):
        CH.cell_file(sdir, k).unlink(missing_ok=True)


def launch_gate(ctx: Ctx) -> str:
    """Why this tick launches nothing ("" = it may): stopping (SIGTERM) · a drain request · the code tree moved off
    the loop's start fingerprint (an event once per new fingerprint — new code comes in by drain, then restart)."""
    from dexengine.cluster import loop_files as LF  # noqa: PLC0415

    if ctx.stop:
        return "SIGTERM — the loop is stopping (no new launches)"
    if d := LF.drain_why(LF.loop_dir(ctx.campaign, ctx.root)):
        return d
    cur = ctx.code()
    if cur == ctx.code0:
        ctx.notes.pop("code", None)
        return ""
    why = (
        f"the code tree differs from the loop start (start {ctx.code0} · now {cur}) — launches stopped. "
        'To move to the new code, restart the loop with --change-params "<why>"'
    )
    if ctx.notes.get("code") != cur:
        ctx.notes["code"] = cur
        _event(ctx, "code_changed", None, why)
    return why


def step_launch(ctx: Ctx, states: dict, cells: list[Path], stop_why: str = "") -> dict[str, str]:
    """Fill the free training GPUs in queue order (none when stop_why — launch_gate) → {gpu: why idle} for every
    training GPU without a claim (every idle GPU has a reason, or it is a bug the status names; a held GPU waiting
    for its judge claim says so)."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415
    from dexengine.rl.cell_state import bkey  # noqa: PLC0415

    view = gpu_view(ctx)
    pending = {
        _gpu_of(c)
        for b, c in zip(ctx.bundles, cells, strict=True)
        if states[bkey(b)] in ("trained", "judging") and _rc(c) == "0" and not CH.cell_file(c, "overflow").is_file()
    }
    free = [g for g in ctx.train_gpus if not view[g]["claims"] and not view[g]["held"] and g not in pending]
    queued = [b for b in ctx.bundles if states[bkey(b)] == "queued" and bkey(b) not in ctx.blocked]
    why_all, skipped = stop_why, []
    n = 0
    for b in [] if stop_why else queued:
        if n >= len(free):
            break
        miss = ctx.canary(b, ctx.train0, ctx.exempt)
        if miss:
            skipped.append(f"{b['hand']}/{b['id']}: {miss}")
            continue
        ok, why = disk_gate(ctx.root, in_flight(cells) + n, ctx.per_run_gb)
        if ok:  # again before each launch: a TERM or a merge mid-tick
            why = launch_gate(ctx)
            ok = not why
        if not ok:
            why_all = why
            break
        try:
            ok, why = _launch_one(ctx, b, ctx.cell_of(b), free)
        except (ValueError, OSError, subprocess.SubprocessError) as e:
            from dexengine.rl.cell_state import UncommittedTreeError  # noqa: PLC0415

            _event(ctx, "launch_refused", b, str(e)[:300])
            skipped.append(f"{b['hand']}/{b['id']}: launch refused {str(e)[:120]}")
            if isinstance(e, UncommittedTreeError):  # every later bundle would launch from the same tree
                why_all = str(e)[:200]
                break
            continue
        if not ok:
            why_all = f"host gate: {why}"
            break
        n += 1
    reason = why_all or (
        "no bundle can launch — " + " · ".join(skipped[:3])
        if skipped
        else "end of queue (no bundles left)"
        if n == len(queued)
        else "IDLE_BUG — no reason"
    )
    view = gpu_view(ctx)
    out = {}
    for g in ctx.train_gpus:
        if view[g]["claims"]:
            continue
        if view[g]["held"]:
            out[g] = f"overflow hold — {view[g]['held']} final judge next (pre-claim: host RAM gate, sidecar POLL)"
        elif g in pending:
            out[g] = "waiting for the overflow decision (training just ended — next step)"
        else:
            u = _mem_used(g)
            out[g] = f"process outside the gate (memory.used {u} MiB)" if u and u >= 500 else reason
    return out


def _gpu_of(sdir: Path) -> str:
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    f = CH.cell_file(sdir, "train_gpu")
    return f.read_text().strip() if f.is_file() else ""


def _mem_used(g: str) -> int | None:
    """memory.used (MiB) of a GPU for an idle GPU's reason — no answer (no nvidia-smi, a hang) is None, not an error."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", g, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    v = r.stdout.strip().splitlines()[:1]
    return int(v[0]) if r.returncode == 0 and v and v[0].strip().isdigit() else None


def tick(ctx: Ctx) -> dict:
    """One step (module doc) → the status dict (the loop in local.py writes it)."""
    ctx.events = []
    ctx.code0 = ctx.code0 or ctx.code()  # run_loop fingerprints first; a bare tick (selftests) here
    ctx.train0 = ctx.train0 or ctx.train()
    states = read_all(ctx)
    cells = [ctx.cell_of(b) for b in ctx.bundles]
    step_reserve(ctx)
    step_cleanup(ctx, states)
    step_sidecars(ctx, states)
    step_overflow(ctx, states, cells)
    errs = step_finalize(ctx, states)
    states = read_all(ctx)
    idle = step_launch(ctx, states, cells, launch_gate(ctx))
    states = read_all(ctx)
    return summarize(ctx, states, idle, errs)


def summarize(ctx: Ctx, states: dict, idle: dict, errs: dict) -> dict:
    """The status: counts by state, finalized, idle GPUs, the end test (sets, not counts)."""
    from dexengine.cluster import loop_files as LF  # noqa: PLC0415
    from dexengine.rl import cell_state as CH  # noqa: PLC0415
    from dexengine.rl.cell_state import bkey  # noqa: PLC0415

    queue = {bkey(b) for b in ctx.bundles if bkey(b) not in ctx.blocked}
    fin = {bkey(b) for b in ctx.bundles if CH.cell_file(ctx.cell_of(b), "finalized").is_file()}
    done = {k for k in queue if states[k] == "done" and k in fin}
    failed = {k for k in queue if states[k] == "failed"}
    live = {k for k in queue if states[k] not in TERMINAL or (states[k] == "done" and k not in fin)}
    ended = not live
    view = gpu_view(ctx)
    return dict(
        campaign=ctx.campaign,
        host=ctx.host,
        time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        pid=os.getpid(),
        counts=dict(collections.Counter(states[k] for k in queue)),
        manifest=len(ctx.bundles),
        queue=len(queue),
        blocked=len(ctx.blocked),
        blocked_why=dict(collections.Counter(ctx.blocked.values())),
        done_finalized=len(done),
        failed=sorted(f"{h}/{i}" for h, i in failed),
        judge_reserve=ctx.notes.get("reserve", "ok"),
        code=ctx.code0,
        train_fp=ctx.train0,
        **LF.swap_status(ctx.bundles, states, ctx.cell_of, LF.loop_dir(ctx.campaign, ctx.root)),
        finalize_refused={f"{h}/{i}": e for (h, i), e in errs.items()},
        gpus={g: dict(role="judge" if g in ctx.judge_gpus else "train", **view[g], idle=idle.get(g)) for g in view},
        idle_bug=sorted(g for g, w in idle.items() if w.startswith("IDLE_BUG")),
        ended=ended,
        complete=ended and done == queue,
        missing=sorted(f"{h}/{i}" for h, i in queue - done),
        states={f"{h}/{i}": states[(h, i)] for h, i in sorted(states)},
        events=[list(e) for e in ctx.events],
    )
