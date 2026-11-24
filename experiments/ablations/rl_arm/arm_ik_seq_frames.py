#!/usr/bin/env python3
"""RL ablation "sequential IK" — the arm path solved by ONE run that continues from the previous frame.

What the canonical arm path is: for the adopted placement, a coarse path over the whole clip (pool DP), frames
re-solved from that coarse path, several runs and the least-swivel passing run kept. This builder switches all of
that off and keeps only what a plain differential / warm-started IK tracker does (OATH · DexMan · SPIDER on a fixed
alignment):

  * one run, no selection;
  * frame 0 = no random starts, one solution, and the seed is an identity of the run (`--f0`, no default):
    `--f0 approach` (the ablation row) uses nothing from the canonical run: the arm starts at the rest pose
    (`idle_arm.idle_qpos`) and the wrist target is moved from the rest wrist pose to the frame-0 wrist pose
    over `APPROACH_STEPS` interpolated poses, each solved from the previous one — what a differential-IK
    tracker does when it reaches for the first pose;
    `--f0 canon` (the control) seeds frame 0 with the canonical frame-0 joints, so frame 0 is the canonical
    pose and every difference comes from the continuation alone;
    `--f0 rest` is ONE damped-least-squares solve of frame 0 from the rest pose — measured on the
    64-clip bundle: it reaches the frame-0 wrist pose (2 mm · 1 deg) in 13/64 (right) · 14/64 (left) clips
    only, so it is kept as a count, not trained;
  * frame t = one solve started at the solution of frame t−1;
  * a frame that does not converge is kept as it is and the chain continues from it (no restart, no repair).
    The residual is recorded — the limits report (`arm_ik_limits_report.py`) counts it.

What stays the same: the wrist targets are the FK of the canonical joints (the palm frame of
`dexengine.placement.arm_ik.resolve_combo`) — the same placement, the same retiming, the same frames, the same wrist
path. Every other key of the canonical cache (base_xyz · base_z_off · num_frames · names) is copied.

Output = a sibling arm-IK cache dir with the same file names. Read by the training env (an Isaac process) through
the env switches `DEXENGINE_ARM_IK_DIR_SUFFIX=<sfx>` (every consumer: reset states · arm feedforward · ENVZ) or
`DEXENGINE_ARM_IK_RESET_SUFFIX=<sfx>` (reset states only); the paper runs used the same switches under the pinned
commit's older flag prefix (experiments.workspace.LEGACY_ENV_PREFIX).

  python -m experiments.ablations.rl_arm.arm_ik_seq_frames --arm rby1 --hand sharpa_wave --side right --f0 approach \
      --src <...>/arm_ik_poses_right_cont25 --out <...>/arm_ik_poses_right_cont25_seqikapp --jobs 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from dexengine.placement import arm_ik as AP

# Solver tolerance of one frame. Tighter than the solver default (1 mm · 0.5 deg): with the default a warm-started
# solve returns without moving while the target is still inside 1 mm, which makes a staircase in the joints. That
# would be a handicap of this script, not of sequential IK.
EPS_POS_M, EPS_ROT_DEG = 1e-5, 0.01
APPROACH_STEPS = 100  # 2 s at 50 Hz — the reach from the rest pose to the frame-0 wrist pose
#: Seed descriptions — stored values (written into every output record as seq_f0_seed · f0_seed), kept as written.
F0_SEEDS = {
    "canon": "canonical arm_joint_positions_all[0]",
    "rest": "collision.IDLE_ARM_QPOS",
    "approach": f"collision.IDLE_ARM_QPOS, wrist moved to the frame-0 pose over {APPROACH_STEPS} poses",
}

_W: dict = {}


def _init(arm: str, hand: str, side: str) -> None:
    import pinocchio as pin

    from dexengine.placement.collision import idle_arm as IA  # the rest pose has one owner

    urdf, ee_frame, joints, _, _ = AP.resolve_combo(arm, hand, side)
    model = pin.buildModelFromUrdf(str(urdf))
    _W.update(
        pin=pin,
        model=model,
        data=model.createData(),
        ee=model.getFrameId(ee_frame),
        idx=tuple(model.joints[model.getJointId(n)].idx_q for n in joints),
        joints=tuple(joints),
        rest=np.asarray(IA.idle_qpos(arm, side), dtype=np.float64),
    )


def _approach_warmstart(f0, _fk, seed, qa, pin, model, data, ee, idx, iters, q_prev):
    """f0 = approach: walk the IK seed from the rest pose to frame 0 along the interpolated wrist poses."""
    if f0 == "approach":
        m0, m1 = _fk(seed), _fk(qa[0])
        for s in range(1, APPROACH_STEPS):  # the last pose (s = APPROACH_STEPS) is frame 0, solved in the loop below
            ms = pin.SE3.Interpolate(m0, m1, s / APPROACH_STEPS)
            q_prev = AP.solve_ik_pose(
                model,
                data,
                ms.translation.copy(),
                ms.rotation.copy(),
                ee,
                idx,
                max_iters=iters,
                eps_pos=EPS_POS_M,
                eps_rot_deg=EPS_ROT_DEG,
                q0=q_prev,
            )[0]
    return q_prev


def _write_cache(z, q_new, pe, re, f0, nit, out, src):
    """Write the re-solved cache next to the source keys (atomic replace)."""
    keep = {k: z[k] for k in z.files}
    keep.update(
        arm_joint_positions=q_new[0].astype(z["arm_joint_positions"].dtype),
        arm_joint_positions_all=q_new.astype(z["arm_joint_positions_all"].dtype),
        wrist_pos_error=np.float32(pe[0]),
        wrist_pos_error_all=pe.astype(np.float32),
        wrist_rot_error_deg=np.float32(re[0]),
        wrist_rot_error_deg_all=re.astype(np.float32),
        solver=np.array(f"seq_warmstart_f0{f0}"),
        seq_source_solver=np.array(str(z["solver"])),
        seq_f0_seed=np.array(F0_SEEDS[f0]),
        seq_iters_all=nit,
    )
    tmp = Path(out) / (Path(src).stem + ".tmp.npz")
    np.savez(tmp, **keep)
    os.replace(tmp, Path(out) / Path(src).name)


def _clip_stats(T, q_new, src, pe, re, nit, iters, qa):
    """Per-clip error, step and deviation statistics."""
    step = np.linalg.norm(np.diff(q_new, axis=0), axis=1) if T > 1 else np.zeros(1)
    return {
        "clip": Path(src).stem,
        "frames": int(T),
        "pos_err_mm_p99": float(np.quantile(pe, 0.99) * 1e3),
        "pos_err_mm_max": float(pe.max() * 1e3),
        "rot_err_deg_max": float(re.max()),
        "frames_over_2mm_1deg": int(((pe > 2e-3) | (re > 1.0)).sum()),
        "f0_pos_err_mm": float(pe[0] * 1e3),
        "f0_rot_err_deg": float(re[0]),
        "f0_iters": int(nit[0]),
        "step_l2_max_rad": float(step.max()),
        "frames_hit_iter_cap": int((nit >= iters).sum()),
        "dev_from_canonical_deg_max": float(np.degrees(np.abs(q_new - qa).max())),
        "frames_over_5deg_from_canonical": int((np.degrees(np.abs(q_new - qa)).max(axis=1) > 5.0).sum()),
    }


def _one(job) -> dict:
    src, out, iters, f0 = job
    pin, model, data, ee, idx = _W["pin"], _W["model"], _W["data"], _W["ee"], _W["idx"]
    z = np.load(src)
    names = tuple(str(n) for n in z["arm_joint_names"])
    if names != _W["joints"]:
        raise RuntimeError(f"[SEQIK] {Path(src).name}: joint names {names} != combo {_W['joints']}")
    qa = np.asarray(z["arm_joint_positions_all"], dtype=np.float64)
    T = qa.shape[0]
    q_new = np.zeros_like(qa)
    pe, re, nit = np.zeros(T), np.zeros(T), np.zeros(T, dtype=np.int32)
    q_prev = pin.neutral(model)
    seed = qa[0] if f0 == "canon" else _W["rest"]
    for k, qi in enumerate(idx):
        q_prev[qi] = seed[k]

    def _fk(vec):
        q = pin.neutral(model)
        for k, qi in enumerate(idx):
            q[qi] = vec[k]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return pin.SE3(data.oMf[ee])

    q_prev = _approach_warmstart(f0, _fk, seed, qa, pin, model, data, ee, idx, iters, q_prev)
    for t in range(T):
        M = _fk(qa[t])
        qs, pe[t], re[t], nit[t] = AP.solve_ik_pose(
            model,
            data,
            M.translation.copy(),
            M.rotation.copy(),
            ee,
            idx,
            max_iters=iters,
            eps_pos=EPS_POS_M,
            eps_rot_deg=EPS_ROT_DEG,
            q0=q_prev,
        )
        q_new[t] = [qs[qi] for qi in idx]
        q_prev = qs
    _write_cache(z, q_new, pe, re, f0, nit, out, src)
    return _clip_stats(T, q_new, src, pe, re, nit, iters, qa)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--side", required=True, choices=("right", "left"))
    ap.add_argument("--f0", required=True, choices=tuple(F0_SEEDS), help="frame-0 seed (no default — an identity)")
    ap.add_argument("--max_iters", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    files = sorted(src.glob("*.npz"))
    if not files:
        raise SystemExit(f"[SEQIK] no npz in {src}")
    if out.resolve() == src.resolve():
        raise SystemExit("[SEQIK] --out is the canonical dir — refused")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(a.arm, a.hand, a.side)) as pool:
        rows = pool.map(_one, [(str(f), str(out), a.max_iters, a.f0) for f in files], chunksize=1)
    made = sorted(p.name for p in out.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if made != [f.name for f in files]:
        raise SystemExit(f"[SEQIK] set mismatch: {len(made)} written vs {len(files)} source")
    summ = {
        "side": a.side,
        "src": str(src),
        "out": str(out),
        "solver": f"seq_warmstart_f0{a.f0}",
        "f0_seed": F0_SEEDS[a.f0],
        "eps_pos_m": EPS_POS_M,
        "eps_rot_deg": EPS_ROT_DEG,
        "max_iters": a.max_iters,
        "clips": len(rows),
        "frames": sum(r["frames"] for r in rows),
        "clips_f0_inside_2mm_1deg": sum(r["f0_pos_err_mm"] <= 2.0 and r["f0_rot_err_deg"] <= 1.0 for r in rows),
        "clips_all_frames_inside_2mm_1deg": sum(r["frames_over_2mm_1deg"] == 0 for r in rows),
        "frames_over_2mm_1deg": sum(r["frames_over_2mm_1deg"] for r in rows),
        "frames_over_5deg_from_canonical": sum(r["frames_over_5deg_from_canonical"] for r in rows),
        "seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    (out / f"SEQIK_{a.side}.json").write_text(json.dumps(summ, indent=1))
    n = summ["clips"]
    print(
        f"[SEQIK] side={a.side} f0={a.f0} clips={n} frames={summ['frames']} · frame 0 inside 2 mm / 1 deg: "
        f"{summ['clips_f0_inside_2mm_1deg']}/{n} · all frames inside: {summ['clips_all_frames_inside_2mm_1deg']}/{n}"
        f" · frames outside {summ['frames_over_2mm_1deg']} · frames > 5 deg from canonical "
        f"{summ['frames_over_5deg_from_canonical']} · {summ['seconds']} s → {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
