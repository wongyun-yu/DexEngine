"""One Isaac record session: run, progress watch, failure classification, and a single retry of boot crashes,
stalls and native crashes. Called by the bundle (record.bundle).

Classification (classify, from the session log and rc). The boundary is the stage's hook line HOOK_LINE: the session
entry prints it right after it has imported the target module, so it proves that the Isaac app has booted (the
canonical background, lighting and record look are applied only after the hook).
  boot   no hook line and a crash sign = crash before the marker (crash words BOOT_SIGNS in the log: breakpad, CUDA
         state, segfault, CUDA error, out of memory; or a signal death rc >= 128; or timeout rc 124). The words are
         appended to the reason. -> only that session (one clip for render) is started once more, with the marker
         [COLLECT_RETRY] (record log and the head of the new session log); the first log is kept as <name>.boot1.log.
         A boot failure on the retry dies = the bundle fails (a per-clip render failure is not one of the exclusion
         reasons of the record check driver_check.verify, so no exclusion without a reason is made).
  pre    no hook line and no crash sign = an error before the hook (a deterministic entry or setup error, e.g. a
         tree mismatch rc 2); a retry would fail the same way: no retry (dies, the last line is the reason).
  stall  the progress watch (watch_stall) reaped the session (STALL_LINE in the log; checked before hook and rc):
         the progress lines (PROGRESS[stage]) did not grow for STALL_S seconds, so the session's process group (the
         group GNU timeout made with setpgid: bash, Isaac python, torch compile workers) got TERM then KILL. -> one
         retry like boot (first log <name>.stall1.log); a stall on the retry dies. The group is reaped rather than a
         new session (setsid) so that the outer reaper (cluster.proc.run_session) still reaches it.
  crash  the hook ran, then a native signal death (death_signal in NATIVE_CRASH_SIGNALS: ABRT, BUS, FPE, ILL, SEGV) =
         a native crash during the session (a carb BaseMutex unlock assertion abort, a segfault: inside Kit, not our
         code). -> one retry like boot (first log <name>.crash1.log, no pause); a crash on the retry dies and the
         assigner hands the clip to the next launch through RETRY_KINDS. The cause inside Kit is not identified (no
         stack; it also happens with the Kit extension folder watch off); the retry stops one session crash from
         turning every clip of a spec into a technical failure. The render output does not change (the same session
         is run again).
  fail   the hook ran but the end marker is missing or rc != 0 (outside crash) = died during the session; no retry.
  marks  it ran to the end but a marker of the canonical look (looks.look_d: applied, frame-guard token, settle) is
         missing = a canonical defect; no retry (dies). The record stage checks the contact-partner
         marker (rollout.recorder.log_problems); the render stage checks the head-view marker
         (render.visibility.log_problems) and the unused-camera-off marker (render.clip_session) too.
  ok
(A 0.4 s breakpad crash was once reported as "canonical background and lighting not applied", hiding the cause,
because the marker check ran before the rc and boot checks.)
"""

from __future__ import annotations

import contextlib
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from dexengine.cluster.proc_ended import ENDED_STATES, proc_stat  # owner of /proc stat reading and the ended states
from dexengine.core import log_markers as LM
from dexengine.record.looks import look_d as LOOK
from dexengine.record.render import body as BODY
from dexengine.record.render import clip_session as RCL
from dexengine.record.render import idle as IDLE
from dexengine.record.render import visibility as HV
from dexengine.record.render import visibility as VIS
from dexengine.record.render import writer as CW  # owner of the render progress line head (FRAME_HEAD)
from dexengine.record.rollout import recorder as REC
from dexengine.record.scene import extras as EXV
from dexengine.record.scene import supports as SUP
from dexengine.record.session import entry as SS
from dexengine.record.session.attempts import RETRY_KINDS, keep_dump, keep_name
from dexengine.rl import session_render as SR_SEAM  # the trainer's render seam (hook target name)

MARK = "[COLLECT]"
RETRY = "[COLLECT_RETRY]"
# Isaac session caps (render one clip measured 27 to 45 s; per clip = 4 trajectories x 5 looks, estimated 11 to 16 min)
SESSION_TIMEOUT_S = {"record": 3600, "render": 900, "render_clip": 2400}
# stage -> head of its hook line (printed by the session entry; the plugin stages bring their own)
HOOK_LINE = {
    "record": f"{SS.HOOK} traj_record.TrajRecorder",
    "render": f"{SS.HOOK} {SR_SEAM.HOOK_TARGET}",
    "render_clip": f"{SS.HOOK} {SR_SEAM.HOOK_TARGET}",
}
BOOT_SIGNS = LM.BOOT_SIGNS  # log readers read these words too (core.log_markers owns them)
TIMEOUT_RC = 124  # timeout(1) expired; a SIGKILL after -k is caught as a signal death (rc >= 128)
KIND_TEXT = LM.KIND_TEXT  # the death-line text per kind (core.log_markers owns it; log readers read it)
#: a death by one of these signals after the hook is a crash (module doc); the names are the rule, the numbers come
#: from this machine's signal module (death_signal)
NATIVE_CRASH_SIGNALS = ("SIGABRT", "SIGBUS", "SIGFPE", "SIGILL", "SIGSEGV")
CLIP_RETRY_MAX = 2  # how often a clip that died only of RETRY_KINDS is handed to a new job by the next launch
#   (read by the assigner: failed kind retry, HOLD beyond it)
BOOT_RETRY_WAIT_S = (60.0, 180.0)  # random pause (s) before a boot retry: app boot crashes cluster in a job's first
#   session (jobs of the same node starting together) and an immediate retry lands in the same window (measured)
# Progress watch. STALL_S = a stall when the progress lines do not grow for this long (set from the canary frame-gap
# distribution of record sessions). The clock starts at session start (boot and setup up to the first progress line
# are inside STALL_S; measured boot -> render function 44 s).
STALL_S = 600
STALL_POLL_S = 5.0  # watch period (stall verdict delay <= STALL_S + period)
STALL_GRACE_S = 15.0  # TERM -> KILL
STALL_MARK = "[COLLECT_STALL]"
STALL_LINE = f"{STALL_MARK} stalled"  # head of the line written at the end of a reaped session log (classify reads it)
#: carb assertion line (head printed by the Carbonite assert handler); the session then hangs silently (measured), so
#: it is reaped at once
CARB_ASSERT = "Assertion Failed!"
CARB_KILL_MARK = "[CARB_ASSERT_KILL]"
DUMP_MARK = "[STALL_DUMP]"
DUMP_TIMEOUT_S = 30.0  # cap of one py-spy dump (does not hold the reaping long)
PROC_GAP_S = 5.0  # between two /proc reads (io bytes grow = busy in IO; same = stuck)
STACK_WAIT_S = 2.0  # after the stack signal, until the session has written its stacks
# stage -> head of its progress line (printed by the render frame lines and the trajectory recorder). None = a stage
# without progress lines (no watch; said in the marker)
PROGRESS = {
    "record": "[TRAJ_REC] ",
    "render": CW.FRAME_HEAD,
    "render_clip": CW.FRAME_HEAD,
}


def death_signal(rc: int) -> str | None:
    """Signal name of a signal death: bash's 128 + n and a direct Popen child's -n follow the same rule. None when it
    is not a signal death; a number without a name on this machine gives 'SIG?<n>' (still a death, number in marker)."""
    n = rc - 128 if rc > 128 else (-rc if rc < 0 else None)
    if n is None:
        return None
    try:
        return signal.Signals(n).name
    except ValueError:
        return f"SIG?{n}"


def crash_signs(rc: int, text: str) -> list[str]:
    """Evidence words of a crash; empty = not a crash (a deterministic error)."""
    why = [w for sign, w in BOOT_SIGNS if sign in text]
    if rc == TIMEOUT_RC:
        why.append(f"timeout (rc {rc})")
    elif sig := death_signal(rc):
        why.append(f"signal death (rc {rc} = {sig})")
    return why


def tail_line(text: str) -> str:
    return next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")[:160]


def isaac_cmd(module: str, args: list[str]) -> list[str]:
    """argv that runs ``python -m <module> <args>`` in the Isaac interpreter set up by scripts/_isaac_env.sh.
    The caller's cwd must be the tree root."""
    return [
        "bash",
        "-c",
        'set -euo pipefail; source scripts/_isaac_env.sh; exec "$ISAACLAB_DIR/isaaclab.sh" -p -m "$0" "$@"',
        module,
        *args,
    ]


def plugin_stage(stage: str):
    """The plugin SessionStage registered under this --stage name (after hooks.load_plugins), or None."""
    from dexengine.record import hooks

    if stage in HOOK_LINE:
        return None
    hooks.load_plugins()
    return hooks.get("session_stage", stage)


def stage_timeout(stage: str) -> int:
    plug = plugin_stage(stage)
    return plug.timeout_s if plug is not None else SESSION_TIMEOUT_S[stage]


def stage_progress(stage: str) -> str | None:
    plug = plugin_stage(stage)
    return plug.progress if plug is not None else PROGRESS[stage]


def _persist_kit():
    from dexengine.record.session import persist_kit as PK

    return PK


def classify(stage: str, rc: int, text: str, done_pat: str) -> tuple[str, str]:
    """(kind, reason); kind in boot · pre · crash · fail · marks · ok (module doc). An unknown stage dies."""
    plug = plugin_stage(stage)
    if stage not in HOOK_LINE and plug is None:
        raise SystemExit(f"{MARK} unknown session stage {stage!r} — {tuple(HOOK_LINE)} or a plugin stage")
    hook_line = plug.hook_line if plug is not None else HOOK_LINE[stage]
    stalled = [ln for ln in text.splitlines() if ln.startswith(STALL_LINE)]
    if stalled:
        return "stall", f"{stalled[-1][len(STALL_LINE) :].strip().lstrip('— ')} · rc {rc}"
    if not any(ln.startswith(hook_line) for ln in text.splitlines()):
        why = crash_signs(rc, text)
        return (
            "boot" if why else "pre"
        ), f"{' · '.join(why) or 'no crash words'} · rc {rc} · last line {tail_line(text)!r}"
    done = done_pat in text
    if death_signal(rc) in NATIVE_CRASH_SIGNALS:
        why = crash_signs(rc, text) + (["carb mutex assertion"] if "BaseMutex" in text else [])
        return "crash", (
            f"{' · '.join(why)} · rc {rc} · {done_pat} {'present' if done else 'missing'} · "
            f"last line {tail_line(text)!r}"
        )
    if rc != 0 or not done:
        sig = death_signal(rc)
        return "fail", f"rc {rc}{f' = {sig}' if sig else ''} · {done_pat} {'present' if done else 'missing'}"
    per_ep = not (
        stage == "render_clip"
        and (
            RCL.zero_by_rule(text)  # 0 episodes by rule · 0 left after resume
            or _persist_kit().resumed_all(text)
        )
    )
    marks = (
        REC.log_problems(text)
        if stage == "record"
        else plug.log_problems(text)
        if plug is not None
        else (
            LOOK.log_problems(text, settle=per_ep)
            + SUP.log_problems(text)
            + (
                BODY.log_problems(text)
                + IDLE.log_problems(text)
                + EXV.log_problems(text)
                + RCL.log_problems(text)
                + (VIS.vis_log_problems(text) + HV.log_problems(text) if per_ep else [])
                if stage in ("render", "render_clip")
                else []
            )
        )
    )
    if marks:
        return "marks", str(marks)
    return "ok", ""


def _reap_group(pgid: int, p: subprocess.Popen, grace_s: float) -> int:
    """TERM the session group pgid, KILL after grace_s -> number of processes left (unkillable in the kernel; the outer
    reaper looks again). pgid must be timeout itself (the group leader, setpgid), else it could hit another group:
    dies."""
    st = proc_stat(pgid)
    if st is not None and (st[1] != pgid or pgid == os.getpgid(0)):
        raise SystemExit(f"{MARK} session {pgid}: group {st[1]} is not its own; not reaped (timeout made no group)")
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        t_end = time.time() + wait_s
        while time.time() < t_end and _group_alive(pgid):
            p.poll()
            time.sleep(0.2)
    p.wait()
    return sum(1 for _ in _group_pids(pgid))


def _group_pids(pgid: int):
    """Unfinished processes of group pgid (reads /proc stat via proc_ended.proc_stat, which owns ENDED_STATES)."""
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            with contextlib.suppress(OSError, ValueError, IndexError):
                st = proc_stat(int(d.name))
                if st and st[1] == pgid and st[0] not in ENDED_STATES:
                    yield int(d.name)


def _group_alive(pgid: int) -> bool:
    return next(_group_pids(pgid), None) is not None


def py_spy() -> str | None:
    """The py-spy executable: PATH, then next to this interpreter (Isaac env bin); None if neither."""
    import shutil

    for c in (shutil.which("py-spy"), str(Path(sys.executable).with_name("py-spy"))):
        if c and os.access(c, os.X_OK):
            return c
    return None


def proc_snap(pgid: int) -> dict[int, str]:
    """Per group pid, the state readable without ptrace: State · wchan · io read/write_bytes ('-' when unreadable)."""
    out = {}
    for pid in _group_pids(pgid):
        d = Path(f"/proc/{pid}")
        st = io = wch = "-"
        with contextlib.suppress(OSError):
            st = next(
                (x.split(":", 1)[1].strip() for x in (d / "status").read_text().splitlines() if x.startswith("State:")),
                "-",
            )
        with contextlib.suppress(OSError):
            wch = (d / "wchan").read_text().strip() or "0"
        with contextlib.suppress(OSError):
            kv = dict(x.split(": ") for x in (d / "io").read_text().splitlines() if ": " in x)
            io = f"r={kv.get('read_bytes', '-')} w={kv.get('write_bytes', '-')}"
        with contextlib.suppress(OSError):
            comm = (d / "comm").read_text().strip()
            out[pid] = f"{comm} State={st} wchan={wch} {io}"
    return out


def stall_dump(pgid: int) -> str:
    """Evidence of where the session hung, right before reaping (DUMP_MARK lines): (1) /proc state twice (PROC_GAP_S
    apart: State, wchan, io); (2) stall_stack.SIGNAL to the group's python (the session writes every thread's stack to
    its log with faulthandler; wait STACK_WAIT_S); (3) py-spy dump ('none <host>' when missing, or the ptrace refusal).
    No failure here holds the reaping."""
    import socket

    from dexengine.core import debug as stall_stack

    a = proc_snap(pgid)
    time.sleep(PROC_GAP_S)
    b = proc_snap(pgid)
    out = [
        f"{DUMP_MARK} proc pid {pid} first {a.get(pid, '-')} · after {PROC_GAP_S:g} s {b.get(pid, 'ended')}"
        for pid in sorted(set(a) | set(b))
    ]
    py = [pid for pid, v in b.items() if v.startswith("python")]
    for pid in py:
        with contextlib.suppress(OSError):
            os.kill(pid, stall_stack.SIGNAL)
    out.append(
        f"{DUMP_MARK} stack signal {stall_stack.SIGNAL.name} -> python {py} · wait {STACK_WAIT_S:g} s"
        f" (stacks are the session log's 'Current thread' and 'Thread 0x' lines)"
    )
    time.sleep(STACK_WAIT_S)
    exe = py_spy()
    if exe is None:
        return "\n".join(
            out
            + [f"{DUMP_MARK} none {socket.gethostname()}: no py-spy executable(PATH · {Path(sys.executable).parent})"]
        )
    for pid in _group_pids(pgid):
        with contextlib.suppress(OSError):
            if not Path(f"/proc/{pid}/comm").read_text().strip().startswith("python"):
                continue
        try:
            r = subprocess.run([exe, "dump", "--pid", str(pid)], capture_output=True, text=True, timeout=DUMP_TIMEOUT_S)
            out.append(f"{DUMP_MARK} pid {pid} rc {r.returncode}\n{(r.stdout + r.stderr).strip()[-20000:]}")
        except (subprocess.TimeoutExpired, OSError) as e:
            out.append(f"{DUMP_MARK} pid {pid} failed {type(e).__name__}")
    return "\n".join(out)


def watch_stall(p: subprocess.Popen, log: Path, stage: str, stall_s: float) -> str:
    """Watch the number of progress lines (PROGRESS[stage] head) in the log until session p ends -> '' (ended) or a
    stall line (STALL_LINE …) after reaping the group when no progress line was added for stall_s seconds. Marker
    `[COLLECT_STALL] watch on STALL_S=<value>` (once per session)."""
    head = stage_progress(stage)
    if head is None:
        print(f"{STALL_MARK} no watch · stage {stage} has no progress line (PROGRESS)", flush=True)
        p.wait()
        return ""
    print(f"{STALL_MARK} watch on STALL_S={stall_s:g} · stage {stage} · progress line {head.strip()!r}", flush=True)
    poll = min(STALL_POLL_S, max(0.2, stall_s / 5))
    t_last, n, last, pos, rest = time.time(), 0, "(none)", 0, b""
    while True:
        try:
            p.wait(timeout=poll)
            return ""
        except subprocess.TimeoutExpired:
            pass
        with open(log, "rb") as f:
            f.seek(pos)
            chunk = rest + f.read()
            pos = f.tell()
        *full, rest = chunk.split(b"\n")
        lines = [x.decode(errors="replace") for x in full]
        got = [ln for ln in lines if ln.startswith(head)]
        if got:
            t_last, n, last = time.time(), n + len(got), got[-1].strip()[:120]
        carb = next((ln for ln in lines if ln.startswith(CARB_ASSERT)), None)
        if carb is not None or time.time() - t_last > stall_s:
            dump = stall_dump(p.pid)
            left = _reap_group(p.pid, p, STALL_GRACE_S)
            why = (
                f"{CARB_KILL_MARK} carb assertion line: reaped without waiting"
                if carb is not None
                else f"no progress for {stall_s:g} s (STALL_S)"
            )
            line = (
                f"{STALL_LINE} — {why} · progress lines {n} · last progress {last!r} · "
                f"group {p.pid} TERM -> KILL · processes left {left}"
            )
            print(dump.splitlines()[0], flush=True)
            print(line, flush=True)
            return dump + "\n" + line


def run_once(
    tree: Path, gpu: str, args: list[str], log: Path, head: str = "", stall_s: float = STALL_S
) -> tuple[int, str]:
    """One session -> (rc, log text). head, when given, is written as the first log line (the retry marker). When the
    progress watch (watch_stall) reaps a hung session, a stall line is written at the end of the log (classify ->
    stall). stall_s is changed only by tests (default STALL_S)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    with open(log, "w") as f:
        if head:
            f.write(head + "\n")
            f.flush()
        p = subprocess.Popen(
            ["timeout", "-k", "60", str(stage_timeout(args[1])), *isaac_cmd("dexengine.record.session.entry", args)],
            cwd=tree,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        stalled = watch_stall(p, log, args[1], stall_s)
        if stalled:
            f.write("\n" + stalled + "\n")
            f.flush()
    return p.returncode, log.read_text(errors="replace")


def session(tree: Path, gpu: str, args: list[str], log: Path, done_pat: str, runner=run_once, pause=time.sleep) -> None:
    """An Isaac record session; anything but ok dies. Only RETRY_KINDS (boot · stall · crash) start the same session
    once more (module doc)."""
    stage = args[1]
    from dexengine.record.session import persist as PST

    if runner is run_once and stage == "render_clip":  # the Kit-reuse worker (canonical; session.persist)

        def runner(t, g, ar, lg, hd):
            return PST.run(t, g, ar, lg, hd, stall_s=STALL_S, timeout_s=SESSION_TIMEOUT_S["render_clip"])

    head = ""
    for attempt in (1, 2):
        t0 = time.time()
        rc, text = runner(tree, gpu, args, log, head)
        kind, why = classify(stage, rc, text, done_pat)
        print(
            f"{MARK} session {args[:2]} rc {rc} · kind {kind} · {done_pat} "
            f"{'present' if done_pat in text else 'missing'} · {time.time() - t0:.0f} s · attempt {attempt}/2 · "
            f"{log}",
            flush=True,
        )
        if kind == "ok":
            return
        if kind in RETRY_KINDS and attempt == 1:
            keep = keep_name(log, kind)
            log.rename(keep)
            if dl := keep_dump(text, keep):
                print(dl, flush=True)
            head = f"{RETRY} 2/2 — first attempt {KIND_TEXT[kind]}: {why} · first log {keep}"
            print(head, flush=True)
            if kind == "boot":
                w = random.uniform(*BOOT_RETRY_WAIT_S)
                print(
                    f"{RETRY} boot crash: pause {w:.0f} s before the retry (BOOT_RETRY_WAIT_S {BOOT_RETRY_WAIT_S}; "
                    "steps out of the window where the node's jobs boot together)",
                    flush=True,
                )
                pause(w)
            continue
        if dl := keep_dump(text, log):
            print(dl, flush=True)
        again = (
            " — after the retry (2/2) · the bundle fails (a per-clip render failure is no exclusion reason)"
            if attempt == 2
            else ""
        )
        raise SystemExit(f"{MARK} {KIND_TEXT[kind]}{again} · {why} — {log}")
