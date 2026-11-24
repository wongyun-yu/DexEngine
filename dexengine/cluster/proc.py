"""Process lifecycle of the RL campaign's launches — the one owner (Python side) of whether a launched process lives
and of how it is reaped:

  ended(pid)          a process has ended: /proc/<pid>/stat gone or its state Z (zombie) · X — the rule's owner is
                      proc_ended (re-exported here with ENDED_STATES · proc_stat; the shell's host gate runs it too)
  alive(pid, run)     liveness: not ended and the command line carries the
                      run name (a reused pid is not ours)
  stop(pid, run)      SIGTERM to the session `pid` leads while it is alive — it may end in between (the spawner reaps
                      at once), which is no error
  reap_when_done(p)   the spawner reaps what it forks — no ended child lingers as a zombie
  run_session(cmd, …) a child as the leader of a new session under a wall limit — on the limit and after its end
                      every process left in the session dies (a nested GNU timeout leaves the group, never the
                      session) · session_pids(sid) lists them
  pod_init(say, cl)   the pod's init (as_init) — a pod's first call in its stage 1: a Kubernetes pod's PID 1 · a Slurm
                      job's pod (a child of the Slurm script, never PID 1) as the child subreaper of its descendants
                      (subreaper — their orphans come to it as they come to a PID 1); elsewhere nothing
  REAPER_START_RE     as_init's start line (reaper_start_line) as a Slurm probe counts it — line and pattern here
  init_line()         that line again for a later bundle of the same job (a chain of bundles) — stated only
                      while this process is still the reaper's child (INIT_PID · measured, else RuntimeError)

Standard library only — stage 1 of a cluster pod imports it under the image's interpreter.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from dexengine.cluster.proc_ended import (  # noqa: F401 — the rule's owner (re-exported: CH · selftests)
    ENDED_STATES,
    ended,
    proc_stat,
)


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def alive(pid: int, run: str) -> bool:
    """A pid is ours only while it has not ended (ended) AND its command line carries the run name (a reused pid is
    not). A zombie's command line is empty — it is dead on both counts."""
    return not ended(pid) and run in _cmdline(pid)


def stop(pid: int, run: str, sig: int = signal.SIGTERM) -> bool:
    """`sig` to the session (= process group) `pid` leads while alive(pid, run) → True when sent. The session may end
    between the check and the signal — the spawner reaps at once (reap_when_done), so its group is gone: False."""
    if not alive(pid, run):
        return False
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def reap_when_done(p) -> None:
    """Reap a forked child (a subprocess.Popen) as soon as it ends: a daemon thread waits on it. A caller that only
    polls (a loop reading files) would otherwise leave it a zombie until its next Popen — and a zombie answers kill -0,
    so the host gate keeps its claim alive and its GPU idle. The thread ends with the caller; the child then falls to
    init (a pod's PID 1 reaps — as_init)."""
    threading.Thread(target=p.wait, name=f"reap-{p.pid}", daemon=True).start()


#: rc of a run cut by its wall limit — GNU timeout's, so a caller's rc → reason rule reads it unchanged
RC_TIMEOUT = 124


def session_pids(sid: int) -> list[int]:
    """Processes of session `sid` that have not ended (proc_stat · ENDED_STATES), its leader included while alive."""
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            with contextlib.suppress(OSError, IndexError, ValueError):
                st = proc_stat(int(d.name))
                if st and st[2] == sid and st[0] not in ENDED_STATES:
                    out.append(int(d.name))
    return out


def _signal_all(pids: list[int], sig: int) -> None:
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, sig)


def run_session(cmd: list[str], limit_s: float, kill_after_s: float, **popen) -> tuple[int, int]:
    """(rc, processes swept) of `cmd` run as the leader of a new session. At `limit_s` every process of the session gets
    SIGTERM, what is left after `kill_after_s` SIGKILL, and rc = RC_TIMEOUT. After the leader ends — limit or not —
    whatever is left of the session gets SIGKILL (swept). A group kill cannot do this: GNU timeout moves itself and its
    child into a new process group, so `timeout -k` at this level never reached a nested timeout's render — it kept its
    GPU after the run ended. A process leaves the session only by setsid (nothing on the collect path calls it) — so one
    level only: a run_session inside the run would take its child out of the leader's session."""
    p = subprocess.Popen(cmd, start_new_session=True, **popen)
    try:
        rc = p.wait(timeout=limit_s)
    except subprocess.TimeoutExpired:
        _signal_all(session_pids(p.pid), signal.SIGTERM)
        t_end = time.time() + kill_after_s
        while session_pids(p.pid) and time.time() < t_end:
            time.sleep(1)
        _signal_all(session_pids(p.pid), signal.SIGKILL)
        p.wait()
        rc = RC_TIMEOUT
    left = session_pids(p.pid)
    _signal_all(left, signal.SIGKILL)
    return rc, len(left)


#: prctl option: the orphans of this process' descendants are handed to it (linux/prctl.h) instead of to PID 1
PR_SET_CHILD_SUBREAPER = 36
#: as_init's start line (reaper_start_line) in grep basic regex — a Slurm probe counts it; a pod log reader reads
#: the PID 1 form
REAPER_START_RE = r"\[REAPER\] PID [0-9]* = init"


def reaper_start_line(pid: int, child: int) -> str:
    return f"[REAPER] PID {pid} = init — job = child {child} · reaps all children/orphans (waitpid -1) · TERM·INT→child"


INIT_PID: int | None = None  # as_init's process (the reaper), as the job child it forked knows it


def init_line() -> str:
    if INIT_PID is None or os.getppid() != INIT_PID:
        raise RuntimeError(f"this process's parent {os.getppid()} is not this job's init {INIT_PID} — no reaper")
    return reaper_start_line(INIT_PID, os.getpid()) + " · next bundle of the same job (chain)"


def subreaper(say) -> None:
    """This process becomes the child subreaper of its descendants (prctl PR_SET_CHILD_SUBREAPER): an orphan below it
    comes to it, not to the node's init — what a pod's PID 1 is by construction. Refused → [REAPER] FAIL, rc 2, before
    any work (as_init's start line never stands for a reaper that receives no orphan). ctypes is imported here,
    not at the top — a pod's stage 1 (PID 1) never needs it."""
    import ctypes

    if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        say(f"[REAPER] FAIL prctl(PR_SET_CHILD_SUBREAPER) errno {ctypes.get_errno()} — cannot receive orphans")
        sys.exit(2)
    say(f"[REAPER] subreaper on — PID {os.getpid()} receives its descendants' orphans (not PID 1)")


def descendants(root: int) -> list[int]:
    """Live (not ENDED_STATES) descendants of `root` by the /proc parent links — children first is not promised."""
    kids: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            with contextlib.suppress(OSError, IndexError, ValueError):
                f = Path(f"/proc/{d.name}/stat").read_text().rsplit(")", 1)[1].split()
                if f[0] not in ENDED_STATES:
                    kids.setdefault(int(f[1]), []).append(int(d.name))
    out, todo = [], [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def _ident(pid: int) -> tuple[int, int] | None:
    with contextlib.suppress(OSError, IndexError, ValueError):
        st = proc_stat(pid)
        return (pid, st[3]) if st else None
    return None


def sweep_left(keep: set[tuple[int, int]], rounds: int = 20) -> int:
    """SIGKILL every live descendant of this process except `keep` ((pid, start time) — what existed before the job:
    the log's tee) and reap them — what a pod's PID 1 exit does to the pod. Off a pod (host · node · Slurm) nothing
    else does it: a session whose spawner died (a pilot killed before its run_session sweep · a persistent Kit
    worker whose owner watch never fired) would stay under the node's init holding its GPU, and new jobs on that
    GPU die of CUDA OOM at boot. A killed parent's children come back to this subreaper, so it repeats until a
    round finds none. → the number killed."""
    n = 0
    for _ in range(rounds):
        left = [p for p in descendants(os.getpid()) if _ident(p) not in keep]
        if not left:
            break
        _signal_all(left, signal.SIGKILL)
        n += len(left)
        end = time.monotonic() + 1.0
        with contextlib.suppress(ChildProcessError):  # no children left: stop waiting
            while time.monotonic() < end:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    time.sleep(0.02)
    return n


def as_init(say, drain_s: float = 5.0) -> None:
    """Be the pod's init: fork; the child returns and does the job (stage1 → its execve keeps the pid → stage2), this
    process reaps every child and every orphan the kernel hands to PID 1 (waitpid(-1)) until the job's child ends,
    passes SIGTERM · SIGINT on to it and exits with its status (killed by signal N → 128 + N). A PID 1 that never waits
    leaves each orphan a zombie, and a zombie answers kill -0 like a live process — a claim bound to one never dies.
    The reaping runs in another process than the job, so no waitpid here can take a child the job's own subprocess
    calls wait on. The child keeps the signal dispositions it had; the two signals are blocked across the fork, so one
    sent in between reaches the child. say(text) writes the [REAPER] lines (the pod's log).
    Before it exits, PID 1 closes its stdout · stderr and goes on reaping for at most drain_s seconds until no child is
    left: the pod log's tee is a child of PID 1, and PID 1's exit kills every process of the pod — without the wait the
    tee may die before it writes the last [REAPER] line. The deadline bounds it (another process may hold the pipe).
    Before the end line, every descendant still alive except those that existed before the fork (the tee) gets
    SIGKILL (sweep_left) — PID 1's exit does that in a pod; off a pod nothing else does."""
    fwd = (signal.SIGTERM, signal.SIGINT)
    sys.stdout.flush()
    sys.stderr.flush()
    old = signal.pthread_sigmask(signal.SIG_BLOCK, fwd)
    global INIT_PID
    INIT_PID = os.getpid()
    keep = {i for c in descendants(INIT_PID) if (i := _ident(c))}  # the log's tee — it was there before the job
    child = os.fork()
    if child == 0:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)
        return

    def pass_on(signum, _frame) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signum)

    for s in fwd:
        signal.signal(s, pass_on)
    say(reaper_start_line(os.getpid(), child))
    signal.pthread_sigmask(signal.SIG_SETMASK, old)
    others, rc = 0, None
    while rc is None:
        try:
            pid, status = os.waitpid(-1, 0)
        except ChildProcessError:  # no child at all while the job's child was never reaped — cannot be
            say(f"[REAPER] no child at all before child {child} was reaped — rc 1")
            rc = 1
            break
        if pid != child:
            others += 1
            continue
        rc = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 128 + os.WTERMSIG(status)  # any python 3
    swept = sweep_left(keep)
    say(
        f"[REAPER] child {child} ended rc={rc} · other children/orphans reaped {others} · leftover descendants "
        f"KILLed {swept}"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    for fd in (1, 2):
        with contextlib.suppress(OSError):
            os.close(fd)
    end = time.monotonic() + drain_s
    with contextlib.suppress(ChildProcessError):  # no children left: stop waiting
        while time.monotonic() < end:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                time.sleep(0.02)
    os._exit(rc)
