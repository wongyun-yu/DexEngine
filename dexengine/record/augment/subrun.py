"""Child runs of the placement augment step (the generation and post-gate children augment.stage starts): their time
cap and the cause field of their failure sentence. The single owner of these rules.

  run(cmd, log, env, done, limit, mark)  runs the child as a new process group in the same session (the bundle
          session reaper of record.driver_bundle still reaches it) and writes its output to log. limit = seconds
          (fixed: the post gate) | GenBound (the generation). Past the cap the whole group gets SIGTERM, SIGKILL after
          KILL_AFTER_S, rc RC_TIMEOUT. End: rc 0 and the done line -> return; otherwise SystemExit whose sentence
          starts with the cause field (core.errors.cause: timeout | crash | no_done) and '<child name> rc <n>' and
          ends with '— <log>' (record.driver_check.reason_of follows that log).
  GenBound  generation cap = min(deadline side, max(FLOOR_S, UNIT_S x checks x T / workers x MARGIN)). The work
          (checks, T, workers) is the work line (WORK_MARK) the generator prints after reading its judged cells and
          before baking; before that line there is no work (= the floor). Deadline side = the bundle end time (the
          record driver gives it in the bundle session env DEADLINE_ENV) - reserve (the caller's share: this clip's
          later steps) - start time. Without the env (a run outside the driver) the marker says there is no deadline.
          Each decision prints one BOUND_MARK line (the value and every input).
  pool_map  the results of Executor.map (input order) with one progress line (PROGRESS) per finished item, so a stall
          and a slow run look different (the generator's collision gate). The pool itself is sized by core.capacity.
UNIT_S basis: 24 measured generation records (2 workers), the maximum per (checks x T / workers): bake 1.16 s +
collision gate 0.30 s. Needs Python 3.11 (Popen process_group; the Isaac runtime is 3.11).
Tests: tests/record/test_aug_subrun.py
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from dexengine.cluster.proc import RC_TIMEOUT  # GNU timeout's rc (the owner)
from dexengine.core import errors as ERR  # owner of the cause field
from dexengine.core.repo import REPO_ROOT

BOUND_MARK = "[AUG_GEN_BOUND]"
WORK_MARK = "[AUG_GEN_WORK]"
PROGRESS = "collision gate progress"
DEADLINE_ENV = "DEXENGINE_COLLECT_BUNDLE_DEADLINE"
FLOOR_S, UNIT_S, MARGIN, MIN_LEFT_S = 1800, 1.5, 1.5, 60
POLL_S, KILL_AFTER_S = 5.0, 60.0
WORK_RE = re.compile(rf"^{re.escape(WORK_MARK)} checks=(\d+) T=(\d+) jobs=(\d+)$", re.M)


def work_line(checks: int, T: int, jobs: int) -> str:
    """The work line the generator prints (the form GenBound reads)."""
    return f"{WORK_MARK} checks={int(checks)} T={int(T)} jobs={int(jobs)}"


def deadline_env(start: float, limit_s: float) -> dict:
    """The env entry a bundle session gets: its end time = start + its cap (epoch seconds)."""
    return {DEADLINE_ENV: str(int(start + limit_s))}


def bound_s(work: dict | None, cap: float | None) -> tuple[float, float | None]:
    """(cap s, work size s | None): work = {checks, T, jobs} | None · cap = deadline side s | None (no deadline)."""
    w = None if work is None else UNIT_S * work["checks"] * work["T"] / max(1, work["jobs"]) * MARGIN
    base = float(FLOOR_S) if w is None else max(float(FLOOR_S), w)
    return (base if cap is None else min(base, cap)), w


class GenBound:
    """The generation cap of one child (module doc): decided at start, decided again on the first work line."""

    def __init__(self, reserve_s: float, environ=None, now: float | None = None):
        raw = (os.environ if environ is None else environ).get(DEADLINE_ENV, "")
        self.deadline = float(raw) if raw else None  # not a number -> ValueError (unknown input dies)
        self.reserve, self.work = float(reserve_s), None
        t0 = time.time() if now is None else now
        self.cap = None if self.deadline is None else self.deadline - self.reserve - t0
        self.limit, self.work_s = bound_s(None, self.cap)

    def update(self, text: str) -> bool:
        """On the first work line the cap is decided again -> True (print the marker line)."""
        if self.work is not None or not (m := WORK_RE.search(text)):
            return False
        self.work = {"checks": int(m[1]), "T": int(m[2]), "jobs": int(m[3])}
        self.limit, self.work_s = bound_s(self.work, self.cap)
        return True

    def line(self) -> str:
        w = self.work
        ws = (
            "before the work line (no work)"
            if w is None
            else f"work = unit {UNIT_S} x checks {w['checks']} x T {w['T']} / workers {w['jobs']} x margin {MARGIN} "
            f"= {self.work_s:.0f} s"
        )
        dl = (
            f"no deadline (env {DEADLINE_ENV} unset: a run outside the record driver)"
            if self.deadline is None
            else f"deadline side {self.cap:.0f} s (bundle end "
            f"{dt.datetime.fromtimestamp(self.deadline).isoformat(timespec='seconds')} - reserve {self.reserve:.0f} s)"
        )
        return f"{BOUND_MARK} cap {self.limit:.0f} s = min(deadline side, max(floor {FLOOR_S} s, work)) · {ws} · {dl}"


def _kill(p: subprocess.Popen) -> None:
    """SIGTERM to the child's group, SIGKILL to what is left after KILL_AFTER_S (grandchildren included)."""
    for sig, wait in ((signal.SIGTERM, KILL_AFTER_S), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            break
        try:
            p.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGTERM:  # the leader ended: what is left of its group gets SIGKILL too
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
        break
    p.wait()


def run(cmd: list, log: Path, env: dict, done: str, limit, mark: str) -> None:
    """Run one child (module doc); returns when it ended with rc 0 and printed `done`, else SystemExit."""
    name = Path(str(cmd[1])).name
    gb = limit if isinstance(limit, GenBound) else None
    lim = gb.limit if gb else float(limit)
    if gb:
        print(f"{mark} {gb.line()} — {name}", flush=True)
        if gb.cap is not None and gb.cap < MIN_LEFT_S:
            raise SystemExit(
                f"{mark} {ERR.cause('timeout')} · {name} not launched — deadline side {gb.cap:.0f} s < {MIN_LEFT_S} s"
                f" — {log}"
            )
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with open(log, "w") as f:
        p = subprocess.Popen(
            [str(x) for x in cmd], cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, process_group=0
        )
    timed_out = False
    while True:
        try:
            rc = p.wait(timeout=POLL_S)
            break
        except subprocess.TimeoutExpired:
            pass
        if gb and gb.update(log.read_text(errors="replace")):
            lim = gb.limit
            print(f"{mark} {gb.line()} — {name}", flush=True)
        if time.monotonic() - t0 > lim:
            timed_out = True
            _kill(p)
            rc = RC_TIMEOUT
            break
    rc = 128 - rc if rc < 0 else rc  # a signal death reads as the shell's 128 + signal number
    has = done in log.read_text(errors="replace")
    if rc == 0 and has:
        return
    why = "timeout" if timed_out else ("crash" if rc else "no_done")
    raise SystemExit(
        f"{mark} {ERR.cause(why)} · {name} rc {rc} · {done} {'present' if has else 'missing'} · "
        f"{time.monotonic() - t0:.0f} s / cap {lim:.0f} s — {log}"
    )


def pool_map(ex, fn, items: list, mark: str) -> list:
    """Executor ex (sized by the caller through core.capacity) -> fn over items in input order, one PROGRESS line per
    finished item (name, ok, sec, hand-hand sec, elapsed)."""
    from concurrent.futures import as_completed

    t0, out = time.monotonic(), [None] * len(items)
    fut = {ex.submit(fn, x): i for i, x in enumerate(items)}
    for n, f in enumerate(as_completed(fut), 1):
        r = out[fut[f]] = f.result()
        g = r if isinstance(r, dict) else {}
        print(
            f"{mark} {PROGRESS} {n}/{len(items)} · {g.get('name')} · ok={g.get('ok')} · {g.get('sec')} s · "
            f"hand-hand {g.get('hh_sec')} s · elapsed {time.monotonic() - t0:.0f} s",
            flush=True,
        )
    return out
