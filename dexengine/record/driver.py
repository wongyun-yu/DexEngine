#!/usr/bin/env python3
"""Public-data record driver (one hand, many bundles; --arm names the robot — required, no default).
From the hand's canonical owner ledger (rl_clip_owner.tsv; owner dexengine.rl.acceptance) the clips whose RL is finished
(state canon ∧ owning checkpoint completion rate >= --min-rate) become a queue of bundles (campaign, bundle); one
worker per GPU handles the bundles one by one with record.bundle (owner-ledger entry). Entry = scripts/stage4_record.py
drive.

  queue   copy of the ledger into the operations place (ledger.tsv, sha256: the same tag = the same queue) -> clips
          per (dataset, campaign, bundle). Checkpoints come from the owner ledger at launch (--owner = --owner-sha); a
          copy whose owners differ dies (record.sources). Order = --order (datasets; missing from the list dies) ·
          bundle name.
  output  bundle = the work place run_collect_work (bundle --out; root piece.work_root; one job owns it) · piece =
          (bundle, --round) (record.piece; one tag = one round). DONE bundles are skipped, and a bundle run again
          skips the clips its cache finished. Check passed -> with --merge on (required argument, no default) merged
          into the final place, the data layer collect (hand x one dataset) (piece.merge_into -> store.merge
          merge_piece; module missing = dies before starting · result.json merge field · a failed merge = FAILED,
          nothing cleaned) -> the intermediate outputs are cleaned (record.clean).
  ops     run layer run_collect/<tag>: progress.json (60 s) · pilot.json (first start, deadline, launch record) ·
          driver.pid · ledger.tsv · bundles/<campaign>__<bundle>/{state, target.txt, collect.log, result.json} · STOP.
  state   first line of the bundle state: RUNNING pid= gpu= · DONE rc= · FAILED rc= reason=; waiting = no file.
  counts  driver_check.verify (targets = stored ⊔ reasoned exclusions, frames, cameras, augment sets); a mismatch or an
          augment failure (RC_AUG_FAILED, failure line) is FAILED.
  GPU     one worker per --gpus entry (the same GPU twice = two workers <GPU>.<n>) · --exclusive-gpus pauses before
          each bundle while someone else's compute process runs · MemAvailable >= --min-mem-gb before a start.
          --cpu-tail on: the bundle's GPU part ends at its render (bundle --until render); its store (the same command
          with --gpu none: caches only, no GPU visible), check, merge and cleanup run in that worker's tail thread
          while the worker starts its next bundle (at most one tail per worker; a failure streak counts when the tail
          ends). off (default) = the worker holds its GPU to the bundle's end.
  stop    deadline (--hours, from the first start) · STOP · tree changed -> no new bundles (running ones finish).
          Three consecutive bundles failing for the same reason stop that worker ([COLLECT_PILOT] ✗ worker stopped).
  spec, rc  --spec mode and the final rc rule are record.driver_spec (module doc) · pyarrow is store.parquet.
  modules   driver_launch (gates, queue, claim) · driver_bundle (one bundle's calls and end) · driver_check (checks).
  index     before the claim, each queued dataset's render-only object index is built when missing
          (scene.render_only_entry.ensure_index; it is not a shipped input).
Markers: [COLLECT_PILOT] … · end line [COLLECT_PILOT] END <reason> bundles done n · failed m · left k · rc <rc> <name>.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import sys
import threading
import time
from pathlib import Path

from dexengine.core import config

# completion threshold default (MIN_RATE) · RC_PARTIAL; owner of bundle processing
from dexengine.record import bundle as C
from dexengine.record import driver_spec as SPM  # spec mode, final tally
from dexengine.record import piece as PC
from dexengine.record.driver_bundle import BundleRun
from dexengine.record.driver_check import (
    bdir_name,
    gpu_foreign,
    mem_available_gb,
    reason_key,
    say,
    tree_state,
)
from dexengine.record.driver_launch import check_env, claim, prepare

HERE = Path(__file__).resolve().parent


ROOT = HERE.parents[1]
FAIL_STREAK = 3
TICK_S = 60


class Pilot(BundleRun):  # one bundle's calls and end: record.driver_bundle
    def __init__(self, a):
        self.a = a
        self.me = os.getpid()
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.stop_reason = ""
        self.workers: dict[str, dict] = {}
        self.run = a.run_dir
        (self.run / "bundles").mkdir(parents=True, exist_ok=True)
        self.queue = a.queue
        self.taken: set[str] = set()
        self.tree0 = tree_state(a.tree)

    # ── bundle state ──
    def state(self, b: dict) -> str:
        f = self.run / "bundles" / bdir_name(b) / "state"
        return f.read_text().split("\n", 1)[0] if f.is_file() else ""

    def take(self) -> dict | None:
        with self.lock:
            for b in self.queue:
                n = bdir_name(b)
                if n in self.taken:
                    continue
                st = self.state(b)
                if st.startswith("DONE") or (st.startswith("FAILED") and not self.a.retry_failed):
                    continue
                if self.a.only and n not in self.a.only:
                    continue
                if self.a.max_bundles and self.started() >= self.a.max_bundles:
                    return None
                self.taken.add(n)
                return b
        return None

    def started(self) -> int:
        return len(self.taken)

    def halt(self, why: str) -> None:
        with self.lock:
            if not self.stop_reason:
                self.stop_reason = why
                say(f"✗ new bundles stopped: {why}")
        self.stop_ev.set()

    def gate(self) -> str:
        """Gate before a new bundle: an empty string = may start."""
        if self.stop_reason:
            return self.stop_reason
        if time.time() >= self.a.deadline_ts:
            self.halt(f"deadline {dt.datetime.fromtimestamp(self.a.deadline_ts).isoformat(timespec='seconds')}")
            return self.stop_reason
        if (self.run / "STOP").exists():
            self.halt("STOP file")
            return self.stop_reason
        t = tree_state(self.a.tree)
        if t != self.tree0:
            self.halt(f"tree changed {self.tree0} -> {t}")
            return self.stop_reason
        return ""

    # ── workers ──
    def worker(self, key: str, gpu: str) -> None:
        try:
            self._worker(key, gpu)
        except BaseException as e:  # noqa: BLE001 — driver defect: no silently dead worker; stops all new bundles
            self.workers[key].update(state="crashed", note=f"{type(e).__name__}: {e}"[:300], bundle=None)
            self.halt(f"worker {key} (GPU {gpu}) defect {type(e).__name__}: {e}"[:300])

    def _worker(self, key: str, gpu: str) -> None:
        W = self.workers[key]
        streak: list[str] = []
        tail: list = []  # [thread, box] of this worker's running CPU tail (--cpu-tail on); box = [result | error]
        try:
            self._work(key, gpu, W, streak, tail)
        finally:
            self._join_tail(tail)

    @staticmethod
    def _join_tail(tail: list) -> dict | None:
        """Wait for the worker's CPU tail → its bundle result (None = no tail); a tail that raised re-raises here (a
        driver defect, as in the worker itself)."""
        if not tail:
            return None
        th, box = tail
        th.join()
        tail.clear()
        if isinstance(box[0], BaseException):
            raise box[0]
        return box[0]

    @staticmethod
    def _start_tail(tail: list, fn, name: str) -> None:
        box: list = []

        def run() -> None:
            try:
                box.append(fn())
            except BaseException as e:  # noqa: BLE001 — handed to the worker at the join
                box.append(e)

        th = threading.Thread(target=run, name=name, daemon=True)
        th.start()
        tail[:] = [th, box]

    def _stop_streak(self, key: str, gpu: str, W: dict, streak: list[str], res: dict) -> bool:
        """Record a finished bundle in the worker's failure streak; True = the worker stops (same reason
        FAIL_STREAK times in a row)."""
        if res["state"] != "FAILED":
            streak.clear()
            return False
        streak.append(reason_key(res["reason"]))
        if len(streak) >= FAIL_STREAK and len(set(streak[-FAIL_STREAK:])) == 1:
            say(f"✗ worker stopped {key} (GPU {gpu}): same reason {FAIL_STREAK} times in a row: {res['reason'][:160]}")
            W.update(
                state="stopped", note=f"same failure {FAIL_STREAK} times in a row: {res['reason'][:160]}", bundle=None
            )
            return True
        return False

    def _work(self, key: str, gpu: str, W: dict, streak: list[str], tail: list) -> None:
        while True:
            why = self.gate()
            if why:
                W.update(state="stopped", note=why, bundle=None)
                return
            if gpu in self.a.exclusive:
                fo = gpu_foreign(gpu, self.me)
                if fo:
                    W.update(
                        state="waiting", note=f"someone else's GPU processes {fo[:4]}: this GPU pauses", bundle=None
                    )
                    self.stop_ev.wait(TICK_S)
                    continue
            mem = mem_available_gb()
            if mem < self.a.min_mem_gb:
                W.update(state="waiting", note=f"MemAvailable {mem:.0f} GB < {self.a.min_mem_gb}", bundle=None)
                self.stop_ev.wait(TICK_S)
                continue
            b = self.take()
            if b is None:
                W.update(state="idle", note="queue end (or --max-bundles)", bundle=None)
                return
            res = self.run_bundle(b, gpu, W)
            if "tail" in res:  # GPU part done: the previous tail first (one per worker), then this one
                prev = self._join_tail(tail)
                self._start_tail(tail, res["tail"], f"tail{key}")  # a rendered bundle always gets its store
                if prev is not None and self._stop_streak(key, gpu, W, streak, prev):
                    return
                continue
            prev = self._join_tail(tail)  # results count in bundle order
            if prev is not None and self._stop_streak(key, gpu, W, streak, prev):
                return
            if self._stop_streak(key, gpu, W, streak, res):
                return

    # ── progress summary ──
    def progress(self, final: str = "") -> None:
        SPM.write(self, final)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--spec", default="", help="spec jsonl: spec mode (record.driver_spec); otherwise the 7 below are required"
    )
    ap.add_argument("--spec-id", default="", help="spec_id of spec mode (with --spec only)")
    ap.add_argument("--hand")
    ap.add_argument("--arm", help="arm id — required, no default (in spec mode too): the record combo = <arm>_<hand>")
    ap.add_argument(
        "--owner", type=Path, help="received canonical owner ledger (checkpoints; its copy = the queue condition)"
    )
    ap.add_argument("--owner-sha", help="canonical owner ledger sha256 measured at launch (sources canon)")
    ap.add_argument("--tag", help="operations place run_collect/<tag> (spec mode = spec_id)")
    ap.add_argument("--merge", required=True, choices=("on", "off"), help="checked pieces -> final")
    ap.add_argument("--order", help="dataset order (comma separated): every dataset of the ledger")
    ap.add_argument("--gpus", help="worker GPUs (comma separated; spec mode default 0)")
    ap.add_argument(
        "--exclusive-gpus", default="", help="GPUs that pause while someone else computes (comma separated)"
    )
    ap.add_argument(
        "--hours", type=float, required=True, help="time budget (from the first start); past it no new bundles"
    )
    ap.add_argument("--min-rate", type=float, default=C.MIN_RATE)
    ap.add_argument("--min-mem-gb", type=float, default=40.0)
    ap.add_argument("--bundle-timeout-base", type=int, default=1800)
    ap.add_argument("--round", choices=PC.EPI.ROUNDS, help="collect_episodes")
    ap.add_argument("--bundle-timeout-per-clip", type=int, default=None, help="collect_piece.round_gate")
    ap.add_argument("--only", default="", help="these bundles only (<campaign>__<bundle>, comma separated)")
    ap.add_argument("--clip-subset", type=Path, default=None, help="only the clips of this cover list")
    ap.add_argument("--max-bundles", type=int, default=0, help="cap of bundles started in this run (0 = none)")
    ap.add_argument("--retry-failed", action="store_true", help="run FAILED bundles again too (default skipped)")
    ap.add_argument("--plan", action="store_true", help="print the queue and end")
    ap.add_argument("--tree", type=Path, default=ROOT, help="code tree that runs record.bundle (default this tree)")
    ap.add_argument(
        "--cpu-tail",
        choices=("off", "on"),
        default="off",
        help="on = free the GPU at the bundle's render end; store, check, merge and cleanup run on the CPU "
        "while the worker starts its next bundle (module doc GPU) · off = as before (default)",
    )
    config.add_data_root_arg(ap)
    a = ap.parse_args()
    config.apply_data_root(a)
    return a


def drive(a) -> tuple[list[dict], int]:
    S = Pilot(a)
    for b in a.queue:  # RUNNING left by a dead earlier driver runs again (the cache skips finished clips)
        if S.state(b).startswith("RUNNING"):
            say(f"RUNNING bundle of an earlier run {bdir_name(b)}: runs again (cache resume)")
            (a.run_dir / "bundles" / bdir_name(b) / "state").unlink()
    say(
        f"start · tree {a.tree} · {S.tree0} · data root {a.main} · operations {a.run_dir} · GPU {a.gpus} "
        f"(exclusive {sorted(a.exclusive)}) · deadline "
        f"{dt.datetime.fromtimestamp(a.deadline_ts).isoformat(timespec='seconds')}"
    )
    signal.signal(signal.SIGTERM, lambda sig, _f: S.halt(f"signal {sig}"))
    threads = []
    for i, g in enumerate(a.gpus):
        n = a.gpus[:i].count(g)  # which worker of the same GPU; the first keeps the GPU number
        key = g if n == 0 else f"{g}.{n}"
        S.workers[key] = {"state": "starting", "bundle": None, "note": "", "gpu": g}
        t = threading.Thread(target=S.worker, args=(key, g), name=f"gpu{key}", daemon=True)
        t.start()
        threads.append(t)
        time.sleep(5)  # two workers must not start Isaac at the same moment
    while any(t.is_alive() for t in threads):
        S.progress()
        for t in threads:
            t.join(timeout=TICK_S / max(1, len(threads)))
    why = S.stop_reason or "queue end"
    S.progress(final=f"ended: {why}")
    results, left = SPM.outcome(a.run_dir, a.queue, a.only)
    rc = SPM.run_rc(results, left)
    done = sum(r.get("state") == "DONE" for r in results)
    say(f"END {why} bundles done {done} · failed {len(results) - done} · left {left} · rc {rc} {SPM.name(rc)}")
    return results, left


def ensure_render_only(a) -> None:
    """Each queued dataset's render-only object index, built once here when missing, before any worker starts (the
    bundles then find it; record.bundle.ensure_render_only makes the same call for a bundle run alone)."""
    from dexengine.record.scene import render_only_entry as ROE

    for ds in sorted({b["dataset"] for b in a.queue}):
        ROE.ensure_index(ds)


def main() -> int:
    a = parse_args()
    SPM.resolve(a)
    check_env(a)
    prepare(a)
    if a.plan:
        for b in a.queue:
            print(f"{b['dataset']}\t{bdir_name(b)}\t{len(b['clips'])}")
        return 0
    ensure_render_only(a)
    claim(a)
    return SPM.run_rc(*drive(a))


if __name__ == "__main__":
    sys.exit(main())
