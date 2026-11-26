"""The training mode: cold-start PPO, automatic resume after a GPU memory failure, results row.

A run is complete when its final checkpoint agent_<MI × rollouts>.pt exists; a launch whose final checkpoint
already exists does not train again (it only writes the results row). Judging is the separate fresh-process judge
(`dexengine.rl.judge`), never part of training. The training entry draws nothing: training runs without the renderer,
and robot videos come only from the record render (scripts/stage4_record.py).

Automatic resume (bimanual): when the trainer log shows a GPU memory failure (PhysX/CUDA allocation errors) or the
trainer's planned restart marker `[MEM_RESTART] iter`, training resumes from the last checkpoint saved before the
first such line (`[CKPT_SAVED]` markers) as run `<run>_R<iter>`; its checkpoints are copied back into the run folder
and its log is appended to the run's log. Limits: MAX_FAILURE_RESUMES failures and MAX_PLANNED_RESTARTS planned
restarts; a resume that makes no progress stops.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from dexengine.rl.launch import modes, process, records
from dexengine.rl.launch import plan as PL
from dexengine.rl.launch.plan import Plan

#: Trainer log lines that mean "died for GPU memory" (or a planned memory restart).
MAX_FAILURE_RESUMES = 4  # automatic resumes after GPU memory failures
MAX_PLANNED_RESTARTS = 20  # resumes after the trainer's planned [MEM_RESTART]
MEMORY_FAILURE = re.compile(
    r"PxgCudaDeviceMemoryAllocator failed|CUDA out of memory|CUDA error, code 2|"
    r"cudaErrorMemoryAllocation|\[MEM_RESTART\] iter"
)


def done_without_eval(log: Path, rc: int, mi: str) -> bool:
    """True when a training finished all `mi` iterations although MI < eval interval (no evaluation ran)."""
    if rc != 0 or not log.is_file():
        return False
    t = log.read_text(errors="replace")
    ei = re.findall(r"^  Eval interval: ([0-9]+)", t, re.M)
    return (
        bool(re.search(rf"^  Max iterations: {mi}\s*$", t, re.M))
        and bool(ei)
        and int(mi) < int(ei[-1])
        and "\n  Training Complete" in "\n" + t
        and "\n  Checkpoints: " in "\n" + t
        and not re.search(r"\[EXP1\] training stopped early|\[MEM_RESTART\] iter", t)
        and "FROZEN EVAL iter" not in t
    )


def train(plan: Plan, ref: Path, need: int) -> int:
    return _train_bi(plan, ref, need) if plan.req.path == "bi" else _train_uni(plan, ref)


def _row(plan: Plan, status: str, secs: int, tail: str = "") -> None:
    req, v = plan.req, plan.values
    seq = f"dir:{Path(req.seq_dir).name}"
    fields = (
        [req.run, seq, v["NE"], v["MI"], status, secs]
        if req.path == "bi"
        else [req.run, seq, req.side, v["NE"], v["MI"], status, secs]
    )
    records.append_summary(records.log_dir(False, req.run_tree), req.path, fields, tail)


def _train_bi(plan: Plan, ref: Path, need: int) -> int:
    req, root, tree = plan.req, plan.req.root, plan.req.run_tree  # code tree (the trainer's cwd) · run tree
    ck = PL.final_ckpt(plan)
    t0, rc, ran, noeval = time.time(), 1, False, False
    log = ref / f"{req.run}_train.log"
    if not ck.is_file():
        ran = True
        records.check_coll_preset(tree, req.run, plan.preset, "train", fresh=True)
        launch = process.resolve_call(plan, modes.train_call(plan), ref)
        if not req.dry_run:
            records.write_coll_preset(tree, req.run, plan.preset, plan.env, PL.PRESET_TRAIN_FLAGS, root)
            records.write_switches_record(tree, req.run, launch)
            if need:
                process.wait_gpu(req.gpu, need)
        print(
            f"[launch {time.strftime('%T')}] TRAIN {req.run} NE{plan.values['NE']} MI{plan.values['MI']} "
            f"mb{plan.values['MB']} GPU{req.gpu}",
            flush=True,
        )
        rc = process.start(launch, root, req.dry_run)
        if req.dry_run:
            print(
                f"[launch] TRAIN {req.run} dry run: argv dump written, nothing launched"
                if rc == 0
                else f"[launch] TRAIN {req.run} dry run failed rc={rc}",
                flush=True,
            )
            return rc
        noeval = done_without_eval(log, rc, plan.values["MI"])
        if not noeval:
            _auto_resume(plan, ref, ck)
    secs = int(time.time() - t0)
    if ran and rc != 0 and not noeval and not ck.is_file():
        _row(plan, "TRAIN_FAIL", secs)
        print(f"[launch] {req.run} TRAIN_FAIL — trainer rc={rc}, no final checkpoint ({ck}); log {log}", flush=True)
        return rc
    if not ck.is_file():
        if done_without_eval(log, rc, plan.values["MI"]):
            _row(plan, "TRAIN_DONE_NOEVAL", secs)
            print(f"[launch] {req.run} TRAIN_DONE_NOEVAL (MI < eval interval; no judge value)", flush=True)
            return 8
        _row(plan, "TRAIN_FAIL", secs)
        print(f"[launch] {req.run} TRAIN_FAIL", flush=True)
        return 1
    _row(plan, "SIDECAR", secs, "JUDGE=sidecar")
    print(f"[launch] JUDGE {req.run} = judge sidecar (fresh-process PARK judge)", flush=True)
    print(f"[launch {time.strftime('%T')}] DONE {req.run} train={secs}s", flush=True)
    return 0


def _auto_resume(plan: Plan, ref: Path, ck: Path) -> None:
    req, root, tree, v = plan.req, plan.req.root, plan.req.run_tree, plan.values
    env = plan.env
    max_fail, max_planned = MAX_FAILURE_RESUMES, MAX_PLANNED_RESTARTS
    roll = int(env.get("DEXENGINE_ROLLOUTS") or PL.ROLLOUTS_DEFAULT)
    fails = planned = 0
    prev_iter = None
    log = ref / f"{req.run}_train.log"
    ckdir = records.run_dir(req.run, tree) / "checkpoints"
    while not ck.is_file() and fails < max_fail and planned < max_planned:
        text = log.read_text(errors="replace") if log.is_file() else ""
        m = MEMORY_FAILURE.search(text)
        if not m:
            print(f"[launch] {req.run} failed without a GPU memory signature — no automatic resume ({log})", flush=True)
            return
        head = text[: m.start()]
        line = text[m.start() :].split("\n", 1)[0]
        saved = records.ckpt_saved(head)
        good_step = saved[-1][1] * roll if saved else 0
        cands = sorted(
            (int(p.stem.split("_")[1]), p) for p in ckdir.glob("agent_*.pt") if p.stem.split("_")[1].isdigit()
        )
        pick = [p for s, p in cands if good_step > 0 and s <= good_step]
        if not saved:
            print(f"[launch] {req.run}: no [CKPT_SAVED] marker before the first memory error", flush=True)
        if not pick:
            print(f"[launch] {req.run}: no checkpoint to resume from", flush=True)
            return
        res_ck = pick[-1]
        res_iter = int(res_ck.stem.split("_")[1]) // roll
        if res_iter >= int(v["MI"]):
            print(f"[launch] {req.run}: last checkpoint is already final (iter {res_iter})", flush=True)
            return
        if prev_iter == res_iter:
            print(f"[launch] {req.run}: resumed twice at iter {res_iter} without progress — stop", flush=True)
            return
        prev_iter = res_iter
        res_run = f"{req.run}_R{res_iter}"
        if "MEM_RESTART" in line:
            planned += 1
            print(f"[launch] planned memory restart {planned}/{max_planned}: iter {res_iter} -> {res_run}", flush=True)
        else:
            fails += 1
            print(
                f"[launch] GPU memory failure -> automatic resume {fails}/{max_fail}: iter {res_iter} -> {res_run}",
                flush=True,
            )
        log.rename(log.with_name(f"{log.stem}_seg{fails}.log"))
        marker = records.coll_preset_marker(tree, req.run)
        res_dir = records.run_dir(res_run, tree)
        res_dir.mkdir(parents=True, exist_ok=True)
        if marker.is_file():
            shutil.copy(marker, res_dir / "coll_preset.txt")
        call = modes.train_call(plan, exp=res_run, extra_tail=f"--resume_from {res_ck} --resume_iteration {res_iter}")
        process.start(process.resolve_call(plan, call, ref), root, False)
        for p in (res_dir / "checkpoints").glob("agent_*.pt"):
            if not (ckdir / p.name).exists():
                shutil.copy(p, ckdir / p.name)
        seg = ref / f"{res_run}_train.log"
        if seg.is_file():
            with open(log, "a") as f:
                f.write(seg.read_text(errors="replace"))


def _train_uni(plan: Plan, ref: Path) -> int:
    req, root, tree = plan.req, plan.req.root, plan.req.run_tree  # code tree (the trainer's cwd) · run tree
    ck = PL.final_ckpt(plan)
    sentinel = records.run_dir(req.run, tree) / "TRAIN_DONE"
    t0, rc, ran = time.time(), 0, False
    log = ref / f"{req.run}_train.log"
    if not sentinel.is_file():
        ran = True
        print(
            f"[launch {time.strftime('%T')}] TRAIN {req.run} SIDE={req.side} NE{plan.values['NE']} "
            f"MI{plan.values['MI']} GPU{req.gpu}",
            flush=True,
        )
        launch = process.resolve_call(plan, modes.train_call(plan), ref)
        if not req.dry_run:
            records.write_switches_record(tree, req.run, launch)
        rc = process.start(launch, root, req.dry_run)
        if req.dry_run:
            print(
                f"[launch] TRAIN {req.run} dry run: argv dump written, nothing launched"
                if rc == 0
                else f"[launch] TRAIN {req.run} dry run failed rc={rc}",
                flush=True,
            )
            return rc
        if rc == 0 and ck.is_file():
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text(
                f"RUN={req.run} MI={plan.values['MI']} NE={plan.values['NE']} side={req.side} "
                f"end={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            )
    else:
        print(f"[launch] SKIP TRAIN {req.run} — TRAIN_DONE exists ({sentinel.read_text().strip()})", flush=True)
    secs = int(time.time() - t0)
    if ran and rc != 0:
        _row(plan, "TRAIN_FAIL", secs)
        print(f"[launch] {req.run} TRAIN_FAIL — trainer rc={rc}", flush=True)
        return rc
    if not ck.is_file():
        _row(plan, "TRAIN_FAIL", secs)
        print(f"[launch] {req.run} TRAIN_FAIL", flush=True)
        return 1
    if ran and log.is_file() and "[SCREEN-STOP]" in log.read_text(errors="replace"):
        _row(plan, "0.0", secs)
        print(f"[launch] {req.run} SCREEN-STOP — recorded rate 0.0", flush=True)
        return 0
    _row(plan, "SIDECAR", secs, "JUDGE=sidecar")
    print(f"[launch] JUDGE {req.run} = judge sidecar (fresh-process PARK judge, SIDE={req.side})", flush=True)
    return 0
