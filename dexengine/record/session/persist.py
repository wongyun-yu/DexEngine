"""Kit-reuse worker for the stage-2 render (always on, no switch).

  A render_clip session is not started as a new Isaac process: it is handed as a job to the worker of this record
  process and GPU (session.entry --stage render_clips --queue, one Kit). The clip program (render.clip: C1, per look,
  brightness redraw, head stage) is unchanged; only the session layer differs. Caller: session.launch.session (the
  runner branch, the render_clip stage of the default runner).
If the worker dies (rc 134 and so on), that job is classified as crash and the session's single retry redraws only
that job on a new worker. The worker closes the app and exits when its caller dies or no job arrives for SERVE_IDLE_S.
Job log = the job's span of the worker log (<queue>/server.log, from the start marker off to the end marker end); it
is written to the session log path (log) and classified like a plain render_clip session (the worker prints the
process hook lines again for every job).
Marker [COLLECT_PERSIST].
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path

# This header uses the standard library only: session.launch imports it (host job tools read it with system python).

PERSIST_MARK = "[COLLECT_PERSIST]"  # the worker marker (one owner; persist_kit and persist_guard import it)
SERVER_TIMEOUT_S = 6 * 3600  # cap of one worker (the per-job cap is the caller's SESSION_TIMEOUT_S)
POLL_S = 0.1  # how often the caller looks for the end marker (costs POLL_S/2 per job)
_SRV: dict = {}  # {queue: Popen}: workers this process started (for reaping)


def queue_dir(tree: Path, gpu: str) -> Path:
    """One queue per GPU and code tree, under the job TMPDIR: each job has its own HOME, tree links, assets and
    scratch, so a worker is shared only inside its job (2 record slots per GPU = 2 workers)."""
    tag = hashlib.sha1(str(Path(tree).resolve()).encode()).hexdigest()[:8]
    q = Path(tempfile.gettempdir()) / f"collect_persist_gpu{gpu}_{tag}"
    q.mkdir(parents=True, exist_ok=True)
    return q


def _alive(q: Path) -> int:
    """Live worker pid of the queue (pidfile, and the queue in its /proc cmdline); 0 if none."""
    try:
        pid = int((q / "server.pid").read_text().split()[0])
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError, IndexError):
        return 0
    return pid if str(q).encode() in cmd and b"render_clips" in cmd else 0


def _server(tree: Path, gpu: str, q: Path) -> int:
    """Live worker pid; start one if none (one launch lock; the worker rc is left in <queue>/server.rc)."""
    with open(q / ".launch.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if pid := _alive(q):
            return pid
        (q / "server.rc").unlink(missing_ok=True)
        owner = str(os.getpid())  # the worker ends when this process ends
        from dexengine.record.session.launch import isaac_cmd

        inner = " ".join(
            shlex.quote(x)
            for x in [
                "timeout",
                "-k",
                "60",
                str(SERVER_TIMEOUT_S),
                *isaac_cmd(
                    "dexengine.record.session.entry",
                    ["--stage", "render_clips", "--queue", str(q), "--owner-pid", owner],
                ),
            ]
        )
        with open(q / "server.log", "ab") as log:
            p = subprocess.Popen(
                ["bash", "-c", f"{inner}; echo $? > {shlex.quote(str(q / 'server.rc'))}"],
                cwd=tree,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (q / "server.pid").write_text(f"{p.pid}\n")
        _SRV[str(q)] = p
        t0 = time.time()
        while not _alive(q) and time.time() - t0 < 10:  # until bash -c has started timeout (queue in the cmdline)
            time.sleep(0.2)
        print(
            f"{PERSIST_MARK} worker started pid {p.pid} · GPU {gpu} · queue {q} · owner {owner} owner=collect",
            flush=True,
        )
        return p.pid


def _submit(q: Path, job: dict) -> int:
    with open(q / ".submit.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        n = 1 + max((int(x.stem) for x in q.glob("*.job")), default=0)
        tmp = q / f".{n}.job.tmp"
        tmp.write_text(json.dumps(job))
        os.replace(tmp, q / f"{n}.job")
    return n


def _server_rc(q: Path) -> int:
    try:
        return int((q / "server.rc").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return 1


def _session(sid: int) -> list[int]:
    """Processes of session sid (field 6 of /proc stat). The worker uses start_new_session, so sid = the pid of the
    bash -c that started it. timeout below it makes its own group with setpgid, so a group kill (killpg(bash -c))
    does not reach timeout or Kit (test N3)."""
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                f = (d / "stat").read_text().rsplit(")", 1)[1].split()
                if int(f[3]) == sid and f[0] != "Z":  # a zombie has ended (reaping is the parent's job)
                    out.append(int(d.name))
            except (OSError, ValueError, IndexError):
                continue
    return out


def _signal(p: int, sig: int) -> None:
    """Send sig to one process; a process that is gone or not ours is skipped."""
    try:
        os.kill(p, sig)
    except (ProcessLookupError, PermissionError):
        return


def _kill(pid: int) -> None:
    """TERM, then after 30 s KILL, every process of the worker session (bash -c, timeout, the session entry,
    isaaclab.sh, Kit — including the group timeout made)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _session(pid)
        if not pids:
            return
        for p in pids:
            _signal(p, sig)
        t0 = time.time()
        while time.time() - t0 < 30 and _session(pid):
            time.sleep(0.5)


def _slice(q: Path, off: int, end: int | None) -> str:
    with open(q / "server.log", "rb") as f:
        f.seek(off)
        return f.read(None if end is None else max(0, end - off)).decode(errors="replace")


def _start_off(q: Path, n: int, off0: int) -> int:
    st = q / f"{n}.start"
    return json.loads(st.read_text()).get("off", off0) if st.exists() else off0


def run(
    tree: Path, gpu: str, args: list[str], log: Path, head: str = "", stall_s: float = 600.0, timeout_s: float = 2400.0
) -> tuple[int, str]:
    """One render_clip session as a worker job -> (rc, the job's log text), the same shape as launch.run_once.
    rc: 0 = end marker · the worker rc (it died) · 124 = job cap exceeded; a stall (the worker log did not grow for
    stall_s) appends the stall line. If the worker closes before it takes this job (app key changed, memory gate, idle
    end), a new worker is started (one Kit, serial)."""
    if args[1] != "render_clip":
        raise SystemExit(f"{PERSIST_MARK} the worker takes render_clip only: {args[:2]}")
    q = queue_dir(tree, gpu)
    pid = _server(tree, gpu, q)
    job = {
        "clip": Path(args[args.index("--plan") + 1]).parent.name,
        "dump": args[args.index("--dump") + 1],
        "seq_dir": args[args.index("--seq-dir") + 1],
        "plan": args[args.index("--plan") + 1],
    }
    off0 = (q / "server.log").stat().st_size
    n = _submit(q, job)
    t0, size, t_grow, stalled = time.time(), -1, time.time(), ""
    while True:
        if (q / f"{n}.done").exists():
            d = json.loads((q / f"{n}.done").read_text())
            rc, text = 0, _slice(q, d["off"], d["end"])
            break
        if not _alive(q):
            time.sleep(0.5)  # look once more (server.rc write vs .done write race)
            if (q / f"{n}.done").exists():
                continue
            if not (q / f"{n}.start").exists():  # the worker closed before taking this job: a new worker takes it
                pid = _server(tree, gpu, q)
                continue
            rc, text = _server_rc(q), _slice(q, _start_off(q, n, off0), None)
            break
        z = (q / "server.log").stat().st_size
        if z != size:
            size, t_grow = z, time.time()
        if time.time() - t_grow > stall_s or time.time() - t0 > timeout_s:
            if time.time() - t_grow > stall_s:
                from dexengine.record.session import (
                    launch as _CLA,  # noqa: PLC0415  (owner of STALL_LINE; already loaded)
                )

                stalled = f"{_CLA.STALL_LINE} — worker log did not grow for {stall_s:.0f} s"
            _kill(pid)
            rc, text = (_server_rc(q) if stalled else 124), _slice(q, _start_off(q, n, off0), None)
            break
        time.sleep(POLL_S)
    body = (head + "\n" if head else "") + text + ("\n" + stalled + "\n" if stalled else "")
    log.write_text(body)
    print(f"{PERSIST_MARK} job {n} {job['clip']} rc {rc} · {time.time() - t0:.0f} s · worker pid {pid}", flush=True)
    return rc, body
