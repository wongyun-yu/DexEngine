#!/usr/bin/env python3
"""RL ablation A2 (paper Tab. 3, "IK re-solve"): an arm-joint cache solved at every frame INDEPENDENTLY.

The canonical cache (precompute/<ds>/robots/<combo>/arm_ik_poses_<side>_cont25, solver pool_dp_v5_reuse) picks one
consistent IK branch through the clip. A2 keeps every wrist pose and re-solves each frame alone: the target of frame t
is FK(canonical q_t) — the same wrist pose the canonical cache reaches, so the placement, palm offset and base frame
cannot drift — and the solve is solve_ik_multistart (dexengine.placement.arm_ik, the canonical solver: neutral start +
random starts from one fixed seed, best 6-D residual) with no warm start from frame t-1. Every other key of the
canonical file is copied unchanged (base_xyz · base_z_off · num_frames · names), so ENVZ and the mount guards read
the same values.

Writes <out>/<clip>.npz (+ solver = cold_multistart_s<seed>) and <out>/FLIPS.json: per side, the share of adjacent
frames whose largest arm-joint jump exceeds 10 / 20 / 45 deg (any joint and joint 3 = elbow), for this cache and for the
canonical one, plus the IK residuals. Dies if a frame misses the tolerance (pos > 2 mm or rot > 1 deg).
Usage: python -m experiments.ablations.rl_arm.arm_ik_cold_frames --src <canonical dir> --out <dir> --arm rby1 --hand
sharpa_wave --side right [--jobs 16]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from dexengine.placement import arm_ik as AP  # the canonical solver and robot facts

TOL_POS_M, TOL_ROT_DEG = 0.002, 1.0
_W = {}


def _init(arm, hand, side):
    import pinocchio as pin

    urdf, ee_frame, joints, _, _ = AP.resolve_combo(arm, hand, side)
    model = pin.buildModelFromUrdf(str(urdf))
    _W.update(
        pin=pin,
        model=model,
        data=model.createData(),
        ee=model.getFrameId(ee_frame),
        idx=tuple(model.joints[model.getJointId(n)].idx_q for n in joints),
        joints=joints,
    )


def _one(job):
    src, out, seed, starts, iters = job
    pin, model, data, ee, idx = _W["pin"], _W["model"], _W["data"], _W["ee"], _W["idx"]
    z = np.load(src)
    if [str(x) for x in z["arm_joint_names"]] != list(_W["joints"]):
        raise SystemExit(f"[A2] joint names differ {src}: {list(z['arm_joint_names'])} vs {_W['joints']}")
    qa = np.asarray(z["arm_joint_positions_all"], dtype=np.float64)
    T = qa.shape[0]
    q_new = np.zeros_like(qa, dtype=np.float32)
    pe = np.zeros(T, dtype=np.float32)
    re = np.zeros(T, dtype=np.float32)
    for t in range(T):
        q = pin.neutral(model)
        for k, qi in enumerate(idx):
            q[qi] = qa[t, k]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        M = data.oMf[ee]
        qs, pe[t], re[t], _ = AP.solve_ik_multistart(
            model, data, M.translation.copy(), M.rotation.copy(), ee, idx, num_starts=starts, max_iters=iters, seed=seed
        )
        if pe[t] > TOL_POS_M or re[t] > TOL_ROT_DEG:
            # a hard frame: the same independent solve with 5x the random starts (seed+1, still no warm start) —
            # measured: 3 of 18,467 left frames missed with 20 starts (up to 27.5 mm / 6.7 deg)
            q2, p2, r2, _ = AP.solve_ik_multistart(
                model,
                data,
                M.translation.copy(),
                M.rotation.copy(),
                ee,
                idx,
                num_starts=5 * starts,
                max_iters=iters,
                seed=seed + 1,
            )
            if p2 * 1000 + r2 < pe[t] * 1000 + re[t]:
                qs, pe[t], re[t] = q2, p2, r2
        q_new[t] = [qs[qi] for qi in idx]
    keep = {k: z[k] for k in z.files}
    keep.update(
        arm_joint_positions=q_new[0],
        arm_joint_positions_all=q_new,
        wrist_pos_error=pe[0],
        wrist_pos_error_all=pe,
        wrist_rot_error_deg=re[0],
        wrist_rot_error_deg_all=re,
        solver=np.array(f"cold_multistart_s{seed}"),
        a2_source_solver=np.array(str(z["solver"])),
    )
    tmp = Path(out) / (Path(src).stem + ".tmp.npz")
    np.savez(tmp, **keep)
    os.replace(tmp, Path(out) / Path(src).name)
    return Path(src).name, qa.astype(np.float32), q_new, float(pe.max()), float(re.max())


def _flip_share(dq, thr_deg):
    """dq = adjacent-frame joint differences (pairs, 7) [rad] → share of pairs past thr (any joint · joint 3)."""
    d = np.degrees(np.abs(dq))
    return {"any": float((d.max(axis=1) > thr_deg).mean()), "elbow_j3": float((d[:, 3] > thr_deg).mean())}


def main():
    ap = argparse.ArgumentParser()
    for k in ("--src", "--out", "--arm", "--hand"):
        ap.add_argument(k, required=True)
    ap.add_argument("--side", required=True, choices=("right", "left"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_starts", type=int, default=20)
    ap.add_argument("--max_iters", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="", help="comma list of clip stems to (re)solve")
    ap.add_argument(
        "--summary_only",
        action="store_true",
        help="solve nothing: FLIPS + tolerance over every <out> file against its <src> file",
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    srcs = sorted(Path(a.src).glob("*.npz"))
    if a.only:
        srcs = [s for s in srcs if s.stem in a.only.split(",")]
    if not srcs:
        raise SystemExit(f"[A2] no npz in {a.src}")
    t0 = time.time()
    if a.summary_only or a.only:
        if a.only:  # re-solve the named clips, then summarise over every clip
            jobs = [(str(s), str(out), a.seed, a.num_starts, a.max_iters) for s in srcs]
            with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(a.arm, a.hand, a.side)) as pool:
                pool.map(_one, jobs, chunksize=1)
        res = []
        for s in sorted(Path(a.src).glob("*.npz")):
            o = out / s.name
            if not o.is_file():
                raise SystemExit(f"[A2] missing {o}")
            zs, zo = np.load(s), np.load(o)
            res.append(
                (
                    s.name,
                    np.asarray(zs["arm_joint_positions_all"], dtype=np.float32),
                    np.asarray(zo["arm_joint_positions_all"], dtype=np.float32),
                    float(zo["wrist_pos_error_all"].max()),
                    float(zo["wrist_rot_error_deg_all"].max()),
                )
            )
    else:
        jobs = [(str(s), str(out), a.seed, a.num_starts, a.max_iters) for s in srcs]
        with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(a.arm, a.hand, a.side)) as pool:
            res = pool.map(_one, jobs, chunksize=1)
    old = np.concatenate([np.diff(r[1], axis=0) for r in res])
    new = np.concatenate([np.diff(r[2], axis=0) for r in res])
    flips = {
        "side": a.side,
        "clips": len(res),
        "adjacent_pairs": int(old.shape[0]),
        "seed": a.seed,
        "num_starts": a.num_starts,
        "src": str(a.src),
        "src_solver": "see a2_source_solver in each npz",
        "max_pos_err_m": max(r[3] for r in res),
        "max_rot_err_deg": max(r[4] for r in res),
        "seconds": round(time.time() - t0, 1),
    }
    for thr in (10, 20, 45):
        flips[f"a2_gt{thr}deg"] = _flip_share(new, thr)
        flips[f"canonical_gt{thr}deg"] = _flip_share(old, thr)
    (out / f"FLIPS_{a.side}.json").write_text(json.dumps(flips, indent=1))
    bad = [r[0] for r in res if r[3] > TOL_POS_M or r[4] > TOL_ROT_DEG]
    print(
        f"[A2] side={a.side} clips={len(res)} pairs={flips['adjacent_pairs']} "
        f"max_err={flips['max_pos_err_m'] * 1000:.2f}mm/"
        f"{flips['max_rot_err_deg']:.2f}deg flips>20deg a2={flips['a2_gt20deg']} "
        f"canonical={flips['canonical_gt20deg']} "
        f"{flips['seconds']}s",
        flush=True,
    )
    if bad:
        raise SystemExit(f"[A2] FAIL tolerance ({TOL_POS_M * 1000:.0f} mm / {TOL_ROT_DEG} deg) missed in {bad}")


if __name__ == "__main__":
    main()
