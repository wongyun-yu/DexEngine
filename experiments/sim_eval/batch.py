"""Run a list of closed-loop scene jobs on local GPUs: one Isaac session per scene (`run_scene`), `per_gpu` at a time.

Outputs: `<runs_root>/<batch>_<scene folder>/` per scene and the ledger `<runs_root>/<batch>.ledger.tsv`
(columns start, end, gpu, rc, clip). A scene whose every mode already has an `episode.npz` written by the current
recorder is skipped (rc `done`). A session that died because the scene moves the robot base (the harness refuses
world-moving scenes) is recorded as rc `na_world`, apart from failures. Completion is by count: every listed job has one
ledger row.

Launch gate per GPU (shared by every batch on this machine through `<runs_root>/gpu<g>.lock`): free VRAM must cover a
session for each session started in the last `ramp_s` seconds plus the new one, and at most `ramp_max` sessions may be
ramping at once. An optional host-RAM floor (`min_avail_gb`) holds new launches too.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

MARK = "[E5_BATCH]"
#: Session failure text of a scene whose body mode moves the world (not run by the closed loop).
NA_WORLD_RE = re.compile(r"^SystemExit: .*body method desk_stretch .*closed loop not implemented", re.M)
NEED_MB = 6656 + 2048  # VRAM of one session plus margin
RAMP_S = 75  # a session takes this long to allocate its VRAM
RAMP_MAX = 3  # sessions allowed to ramp up at once on one GPU


@dataclass
class BatchSpec:
    """One batch: job files (absolute), outputs under runs_root, sessions spread over gpus."""

    name: str
    jobs: list[Path]
    modes: list[str]
    gpus: list[int]
    per_gpu: int
    runs_root: Path
    run_scene_args: list[str] = field(default_factory=list)  # passed to every run_scene call (policy, eval cfg, …)
    cap_mult: float = 1.5
    min_avail_gb: int | None = None


def scene_of(job: Path) -> str:
    """Scene folder name of a job file `<…>/<scene>/job.json`."""
    return job.parent.name


def out_dir(spec: BatchSpec, job: Path) -> Path:
    """Session output folder of one job."""
    return spec.runs_root / f"{spec.name}_{scene_of(job)}"


def recorder() -> int:
    """Recorder version written into every episode.json by the closed loop."""
    from experiments.sim_eval.loop.common import RECORDER

    return int(RECORDER)


def all_done(o: Path, modes: list[str], rec: int) -> bool:
    """Every mode has an episode written by the current recorder."""
    try:
        for m in modes:
            if (
                not (o / m / "episode.npz").is_file()
                or f'"recorder": {rec},' not in (o / m / "episode.json").read_text()
            ):
                return False
    except OSError:
        return False
    return True


def _free_mib(g: int) -> int | None:
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(g)],
        capture_output=True,
        text=True,
    )
    try:
        return int(p.stdout.strip())
    except ValueError:
        return None


def _avail_gb() -> int:
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) // 1048576
    return 0


def claim_gpu(spec: BatchSpec, g: int) -> str | None:
    """Pass the launch gate of GPU g and stamp the launch → None, or the reason to wait."""
    lock = spec.runs_root / f"gpu{g}.lock"
    stamp = spec.runs_root / f"gpu{g}.recent"
    with open(lock, "w") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        now = int(time.time())
        recent = []
        if stamp.is_file():
            recent = [int(t) for t in stamp.read_text().split() if t.isdigit() and now - int(t) < RAMP_S]
        free = _free_mib(g)
        need = NEED_MB * (1 + len(recent))
        if free is None or free < need:
            return f"GPU {g} free {free} MiB < {need} (ramping {len(recent)})"
        if len(recent) >= RAMP_MAX:
            return f"GPU {g} ramping {len(recent)} ≥ {RAMP_MAX} (within {RAMP_S} s)"
        if spec.min_avail_gb is not None and _avail_gb() < spec.min_avail_gb:
            return f"host RAM available {_avail_gb()} GB < {spec.min_avail_gb}"
        tmp = stamp.with_name(f"{stamp.name}.{os.getpid()}")
        tmp.write_text("".join(f"{t}\n" for t in recent + [now]))
        os.replace(tmp, stamp)
    return None


def _session_cmd(spec: BatchSpec, job: Path, o: Path, g: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.sim_eval.run_scene",
        "--job",
        str(job),
        "--out",
        str(o),
        "--gpu",
        str(g),
        "--modes",
        ",".join(spec.modes),
        "--cap-mult",
        str(spec.cap_mult),
        *spec.run_scene_args,
    ]


def _check_jobs(spec):
    """Refuse an empty job list or two jobs that share a scene folder name."""
    names = [scene_of(j) for j in spec.jobs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"{MARK} two jobs share a scene folder name (outputs would collide): {dup[:3]}")
    if not spec.jobs:
        raise SystemExit(f"{MARK} empty job list")


def _launch_round(spec, stopped, i, rec, row, log, running, root, env):
    """One pass over the GPUs: skip done jobs, launch the next job on each GPU that passes its gate."""
    launched = False
    for g in spec.gpus:
        if stopped or i >= len(spec.jobs):
            break
        job = spec.jobs[i]
        o = out_dir(spec, job)
        if all_done(o, spec.modes, rec):
            row(time.strftime("%T"), "-", "done", scene_of(job))
            log(f"{MARK} skipped (already done) {scene_of(job)}")
            i += 1
            continue
        if sum(1 for v in running.values() if v[1] == g) >= spec.per_gpu:
            continue
        why = claim_gpu(spec, g)
        if why:
            (spec.runs_root / f"{spec.name}.wait").write_text(why + "\n")
            continue
        o.mkdir(parents=True, exist_ok=True)
        with open(f"{o}.run.log", "w") as lf:
            p = subprocess.Popen(
                _session_cmd(spec, job, o, g),
                cwd=root,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        running[p.pid] = (p, g, scene_of(job), time.strftime("%T"))
        log(f"{MARK} launch {scene_of(job)} GPU {g} pid {p.pid}")
        i += 1
        launched = True
        time.sleep(5)
    return launched, i


def _summarize(rows, spec, led, log):
    """Count ledger rows by outcome, log the DONE line and return the result dict."""
    ok = sum(1 for r in rows if r == 0)
    na = sum(1 for r in rows if r == "na_world")
    done = sum(1 for r in rows if r == "done")
    res = {
        "n": len(spec.jobs),
        "ok": ok,
        "na": na,
        "done": done,
        "fail": len(rows) - ok - na - done,
        "not_run": len(spec.jobs) - len(rows),
        "ledger": str(led),
    }
    log(f"{MARK} DONE ok={ok}/{len(spec.jobs)} na={na} fail={res['fail']} done={done} not_run={res['not_run']}")
    return res


def run_batch(spec: BatchSpec, stop: Callable[[], bool] | None = None, log=print) -> dict:
    """Run every job; `stop()` returning True stops new launches (running sessions finish, the rest are not run).

    Returns {"ok", "na", "fail", "done", "not_run", "n", "ledger"}; one ledger row per job that ran or was skipped."""
    spec.runs_root.mkdir(parents=True, exist_ok=True)
    _check_jobs(spec)
    led = spec.runs_root / f"{spec.name}.ledger.tsv"
    led.write_text("start\tend\tgpu\trc\tclip\n")
    rec = recorder()
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root))
    running: dict = {}  # pid → (proc, gpu, clip, start)
    rows = []

    def row(start, gpu, rc, clip):
        with open(led, "a") as f:
            f.write(f"{start}\t{time.strftime('%T')}\t{gpu}\t{rc}\t{clip}\n")
        rows.append(rc)

    def reap():
        for pid, (p, g, c, t0) in list(running.items()):
            if p.poll() is None:
                continue
            rc: object = p.returncode
            log_txt = spec.runs_root / f"{spec.name}_{c}" / "session.log"
            if rc != 0 and log_txt.is_file() and NA_WORLD_RE.search(log_txt.read_text(errors="replace")):
                rc = "na_world"
            row(t0, g, rc, c)
            log(f"{MARK} end {c} GPU {g} rc {rc}")
            del running[pid]

    i, stopped = 0, False
    log(
        f"{MARK} start {spec.name} · jobs {len(spec.jobs)} · modes {','.join(spec.modes)} · GPU {spec.gpus} × "
        f"{spec.per_gpu} · recorder {rec}"
    )
    while i < len(spec.jobs) or running:
        reap()
        if not stopped and stop is not None and stop():
            stopped = True
            log(f"{MARK} stop requested — no new sessions; {len(spec.jobs) - i} job(s) not run")
        launched, i = _launch_round(spec, stopped, i, rec, row, log, running, root, env)
        if not launched:
            time.sleep(10)
    return _summarize(rows, spec, led, log)


def rc0_scenes(ledger: Path) -> set[str]:
    """Scenes whose session ended with rc 0 in a batch ledger."""
    if not ledger.is_file():
        return set()
    out = set()
    for ln in ledger.read_text().splitlines()[1:]:
        f = ln.split("\t")
        if len(f) >= 5 and f[3] == "0":
            out.add(f[4])
    return out


def write_json(path: Path, obj) -> None:
    """Atomic JSON write."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)
