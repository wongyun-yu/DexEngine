"""Run one launch end to end: manifest, mode guards, trainer processes, results tables and exit code.

Exit codes: 0 done · 1 training failed without a trainer error code · 2 bad input · 6 GPU memory never freed ·
7 collision-preset mismatch (or a caller value the preset forbids) · 8 training reached its last iteration without
any evaluation · otherwise the failing trainer's code. A dry run (`Request.dry_run`) writes the manifest and the
argv dumps into the dry log folder; for a training or judging launch (`Request.preflight`) it also starts each trainer
process once in its preflight mode (CPU, no Kit; `process.preflight`), and a failing preflight is its exit code.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from dexengine.rl.launch import modes, process, records
from dexengine.rl.launch import plan as PL
from dexengine.rl.launch.plan import Plan
from dexengine.rl.launch.request import Request

#: MODE= token of the manifest (readers count training runs by it).
MANIFEST_MODE = {"train": "train", "smoke": "smoke", "judge": "frozenreplay", "render": "render", "replay": "replay"}
#: GPU memory (MiB) a launch waits for before it starts.
NEED_MIB = {"train": 24000, "judge": 24000, "render": 10000, "replay": 10000}


def out_name(req: Request) -> str:
    """Output name (manifest, logs) — the bimanual replay keeps its historical `_render` suffix."""
    return req.run + "_render" if (req.mode == "replay" and req.path == "bi") else req.out


def manifest(plan: Plan, out: str) -> str:
    """Launch manifest text (see dexengine.rl.launch.records)."""
    req, v, root = plan.req, plan.values, plan.req.root
    mode = "render" if out.endswith("_render") else MANIFEST_MODE[req.mode]
    seq = f"dir:{Path(req.seq_dir).name}"
    lines = [records.manifest_head(root, mode, out), f"SNAPSHOT=none REPO_ROOT={root}"]
    if req.path == "bi":
        lines += [
            f"RUN={req.run} SIDE={req.side} SEQ={seq} TOOL=seqdir LOBJ=seqdir NE={v['NE']} MI={v['MI']} "
            f"MB={v['MB']} TIGHTEN={v['TIGHTEN']} NOCOL=0 ROBOT_Z=0",
            f"COLL_PRESET={plan.preset}",
            f"RESET_SCHEME={v['RESET_SCHEME']} PSA_FLAGS=--per_side_advantage {v['PER_SIDE_ADV']} "
            f"WRIST_CAPS={PL.wrist_caps(v)} HINGE={PL.hinge(v)} strip_power={v['STRIP_POWER']} "
            f"force_window_post={v['FDIR_POST']} FORCE_DIR={v['FORCE_DIR']} OBJ_OBS_FIXED={v['OBJ_OBS_FIXED']}",
            f"MULTIGPU= GPU={req.gpu}",
            f"EXTRA={plan.extra} LGATE_RAMP={v['LGATE_RAMP']}",
        ]
    else:
        lines += [
            f"RUN={req.run} SIDE={req.side} SEQ={seq} OBJ=(seqdir) NE={v['NE']} MI={v['MI']} "
            f"MB={v.get('MB') or 'yaml'} TIGHTEN={v['TIGHTEN']} NOCOL=0",
            f"GATE=POS_CONST={v['POSC']} ROT_CONST={v['ROTC']} LAGW={v['LAGW']} "
            f"WRIST={PL.wrist_caps(v)} {PL.hinge(v)} OBJ_OBS_FIXED={v['OBJ_OBS_FIXED']}",
            f"MULTIGPU= GPU={req.gpu}",
            f"EXTRA={plan.extra}",
            "TRAIN_EXTRA=",
        ]
    roster = records.arg_path(req, records.log_dir(False, req.run_tree) / f"{out}_roster.txt")
    lines.append(f"ROSTER_RECORD={roster} SEQ_DIR={req.seq_dir}")
    lines += [f"RECIPE::{k}={v[k]}" for k in sorted(v)]
    lines += [f"{k}={val}" for k, val in sorted(plan.env.items())]
    return "\n".join(lines) + "\n"


def run(req: Request) -> int:
    """Run (or dry-run) a launch; return its exit code."""
    plan = PL.prepare(req)
    tree, out = req.run_tree, out_name(req)
    ref = records.log_dir(req.dry_run, tree)
    need = req.need_mib if req.need_mib is not None else NEED_MIB.get(req.mode, 0)
    if req.mode == "judge" and req.path == "bi" and req.ckpt != "none":
        records.check_coll_preset(tree, req.run, plan.preset, "judge")
    if req.mode == "render":
        records.check_coll_preset(tree, req.run, plan.preset, "render", render_rundir=req.render_rundir)
    records.write_manifest(ref / f"{out}_cmd.txt", manifest(plan, out))
    if req.mode == "smoke":
        return _smoke(plan, ref)
    if req.mode == "judge":
        return _judge(plan, ref, need)
    if req.mode == "render":
        return _render(plan, ref, need)
    if req.mode == "replay":
        return _replay(plan, ref, need)
    from dexengine.rl.launch import training

    return training.train(plan, ref, need)


def _smoke(plan: Plan, ref: Path) -> int:
    req = plan.req
    launch = process.resolve_call(plan, modes.smoke_call(plan), ref)
    rc = process.start(launch, req.root, req.dry_run)
    if req.dry_run or req.path == "bi":
        return process.dry_done(rc, f"SMOKE {req.run}", req.preflight) if req.dry_run else rc
    passed = rc == 0 and "==== REFBASE SMOKE PASS ====" in _read(launch.log)
    records.append_summary(
        records.log_dir(False, req.run_tree),
        "uni",
        [
            req.run,
            f"dir:{Path(req.seq_dir).name}",
            req.side,
            plan.values["NE"],
            plan.values["MI"],
            "SMOKE_PASS" if passed else "SMOKE_FAIL",
            0,
        ],
    )
    print(f"[launch] {req.run} {'SMOKE_PASS' if passed else 'SMOKE_FAIL'} (rc={rc})", flush=True)
    return 0 if passed else (rc or 1)


def _judge(plan: Plan, ref: Path, need: int) -> int:
    req = plan.req
    if req.ckpt != "none" and not req.dry_run and not Path(req.ckpt).is_file():
        print(f"[launch] judge checkpoint missing: {req.ckpt}", flush=True)
        return 2
    call = modes.judge_call(plan)
    print(f"[launch] judge ckpt={req.ckpt} NE={call.ne} (training NE={plan.values['NE']}) -> {req.out}", flush=True)
    if not req.dry_run and need:
        process.wait_gpu(call.gpu, need)
    rc = process.start(process.resolve_call(plan, call, ref), req.root, req.dry_run)
    return process.dry_done(rc, f"JUDGE {req.out}", req.preflight) if req.dry_run else rc


def _render(plan: Plan, ref: Path, need: int) -> int:
    req = plan.req
    look = modes.render_look_env(plan, physx_lite=True)
    calls = modes.render_calls(plan, look)
    if not req.dry_run and need and calls:
        process.wait_gpu(calls[0].gpu, need)
    first_rc = 0
    for call in calls:
        rc = process.start(process.resolve_call(plan, call, ref), req.root, req.dry_run)
        out = call.extra_flags.rsplit("--videos_out_dir ", 1)[1].split()[0]
        n = len(list(records.P.run_out(out, req.run_tree).glob("*.mp4"))) if not req.dry_run else 0
        print(f"[launch] RENDER {call.exp} rc={rc} mp4={n}", flush=True)
        first_rc = first_rc or rc
    print(f"[launch] render done ({len(calls)} checkpoints)", flush=True)
    return first_rc


def _replay(plan: Plan, ref: Path, need: int) -> int:
    """Kinematic replay, one session per clip; clip videos are renamed <clip>_<view>_kinematic.mp4."""
    req = plan.req
    look = modes.render_look_env(plan, physx_lite=req.path == "bi")
    if not req.dry_run and need:
        process.wait_gpu(req.gpu, need)
    od = modes.replay_out_dir(plan)
    odw = records.P.run_out(records.P.mode_record(od, "dryrun") if req.dry_run else od, req.run_tree)
    odw.mkdir(parents=True, exist_ok=True)
    ledger = odw / "_replay.done"
    ledger.write_text("")
    clips = modes.replay_clips(plan)
    ok, first_rc = 0, 0
    tmp = Path(tempfile.mkdtemp(prefix="replay_seq_"))
    try:
        for c in clips:
            t0 = time.time()
            for e in tmp.iterdir():
                e.unlink()
            (tmp / c).symlink_to(os.path.realpath(Path(req.seq_dir) / c))
            rc = process.start(
                process.resolve_call(plan, modes.replay_call(plan, str(tmp), c, look), ref), req.root, req.dry_run
            )
            first_rc = first_rc or rc
            got = 0
            for view in ("ego", "front", "side", "side_front"):
                src = sorted(odw.glob(f"[0-9][0-9]_*_{view}_kinematic.mp4"))
                if src:
                    shutil.move(src[0], odw / f"{c}_{view}_kinematic.mp4")
                    got += 1
            (odw / "manifest.txt").unlink(missing_ok=True)
            good = got == 4 and rc == 0
            ok += good
            with open(ledger, "a") as f:
                f.write(
                    f"{c} ok (views 4, {time.time() - t0:.0f}s)\n"
                    if good
                    else f"{c} FAIL (views {got}, rc={rc}, log {ref}/{req.run}_{c}_replay_render.log)\n"
                )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(ledger, "a") as f:
        f.write(f"ALL_REPLAY ok={ok}/{len(clips)}\n")
    print(f"[launch] replay {ok}/{len(clips)} clips -> {od} (ledger {ledger})", flush=True)
    if req.dry_run:
        return process.dry_done(first_rc, f"REPLAY {req.run}", req.preflight)
    return 0 if ok == len(clips) else (first_rc or 1)


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""
