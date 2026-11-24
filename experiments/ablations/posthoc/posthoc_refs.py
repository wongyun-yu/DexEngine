#!/usr/bin/env python3
"""Floating-hand result + post-hoc arm IK (paper Sec. E.4): the derived reference set (no training).

Input = the trajectory record of a floating-hand judgment (`evaluation/traj_record`, schema 2: per clip the kept
episodes, per step the root link pose + joint positions of both hands in the env-local frame). For every clip of the
record this builder writes, into a SANDBOX root (the canonical root is read only):

  1. a tree whose hand columns of `0/trajectory_kinematic.npz` are the floating rollout — the 6 wrist dofs solved so
     that the MuJoCo palm body is where the rollout palm was, the finger joints copied by joint name. Object columns,
     contacts, the time axis and every other key stay canonical;
  2. an arm-IK cache `arm_ik_poses_<side><suffix>/`: ONE warm-started run on the rollout palm path (frame t starts at
     the solution of frame t−1, frame 0 starts at the canonical frame-0 joints; no coarse path, no re-solve, no run
     selection) — the post-hoc arm.

Frames (measured, not assumed — the builder dies if they do not hold):
  env-local pose = G · MuJoCo-world pose, G = Rz(−90°) + t_clip. t_clip comes from the record's own reference object
    columns (the env's transform of the tree's object columns): both objects must give the same t_clip on every step,
    which also proves step k of the record ↔ frame k of the tree and that the tree is the one the rollout ran on;
  palm frame of the arm URDF (what the record's hand pose gives) = MuJoCo palm body ∘ c, c = a fixed translation in
    the body frame (measured 29.5 mm for sharpa_wave) — fitted over the clips at frame 0 (the reset state = canonical
    arm FK), median, and the spread must be < 0.1 mm; the rotation part must be the identity;
  robot-base-frame palm pose = env-local palm pose − cache `base_xyz`  (frame 0 must reproduce the canonical arm FK).

Episode per clip = the record's selected episode (`sel`, rank 0 of the successes); a clip with no success uses the
record's failure_best episode, and frames past its end hold its last state. Which clips those are is written to the
summary — the caller reports them.

`--fingers pos` = finger reference = the ACHIEVED finger positions of the rollout (what schema 2 stores).
`--fingers tgt` = finger reference = the PD targets of the rollout (`targets` array, written only by a record made with
DEXENGINE_TRAJ_RECORD_TARGETS=1). The two differ by the squeeze the policy commands: a PD finger holds an object with
target − position, so `pos` replays the pose without the grip force.

After this builder the canonical tools make the derived caches (run by the caller on the sandbox):
  dexengine/placement/fk/body_positions.py <tree> -q · dexengine/placement/fk/wrist_poses.py

  python experiments/ablations/posthoc/posthoc_refs.py --traj <rec.npz> --canon-root <main root> --out-root <sandbox> \
      --floating-urdf-dir <.../floating_sharpa_wave_vj> --arm rby1 --hand sharpa_wave --dataset taco \
      --suffix _cont25_fhpost --src-suffix _cont25 --fingers pos --jobs 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from dexengine.core import qpos_layout
from dexengine.placement import arm_ik as AP
from experiments.ablations.posthoc import refs_geometry as G

EPS_POS_M, EPS_ROT_DEG = 1e-5, 0.01  # the solver tolerance of arm_ik_seq_frames (same reason)
RZ = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Rz(−90°): x_env = y_mj · y_env = −x_mj
TOL_T_CLIP_M = 1e-5  # t_clip from the two reference objects, over every step, must agree within this
TOL_PALM_OFF_MAD_M = 1e-4  # spread (median absolute deviation) of the palm-frame offset c over the clips
TOL_FRAME_ROT_DEG = 0.5  # palm frame of the floating URDF vs the MuJoCo palm body (after G) at frame 0
# Frame 0: the rollout starts at the reference wrist pose, and the canonical arm solution has its own frame-0
# residual against that pose (the cache records it — measured up to 8.2 mm · 3.4 deg on this bundle). So
# (rollout palm − base_xyz) vs the canonical arm FK must EQUAL the recorded residual within this.
TOL_F0_ARM_M, TOL_F0_ARM_DEG = 5e-5, 0.02
TOL_FK_M, TOL_FK_DEG = 1e-5, 0.01  # written wrist dofs → MuJoCo FK vs the rollout palm
SIDES = (("right", "robot"), ("left", "robot_left"))
TREE_FILES = ("task_info.json", "scene.xml", "scene_eq.xml")

_W: dict = {}


def _init(a) -> None:
    import pinocchio as pin

    z = np.load(a.traj, allow_pickle=True)
    ents = {e["name"]: e for e in json.loads(str(z["entities"]))}
    arr = "targets" if a.fingers == "tgt" else "states"
    if arr not in z.files:
        raise SystemExit(f"[FHPOST] --fingers {a.fingers} needs the '{arr}' array — this record has {z.files}")
    mods = {}
    for side, en in SIDES:
        urdf, ee, joints, _, _ = AP.resolve_combo(a.arm, a.hand, side)
        am = pin.buildModelFromUrdf(str(urdf))
        fu = next(Path(a.floating_urdf_dir).glob(f"*_split_{side[0].upper()}.urdf"))
        fm = pin.buildModelFromUrdf(str(fu))
        jn = list(ents[en]["joint_names"])
        mods[side] = {
            "am": am,
            "ad": am.createData(),
            "ee_a": am.getFrameId(ee),
            "joints": tuple(joints),
            "aidx": tuple(am.joints[am.getJointId(n)].idx_q for n in joints),
            "fm": fm,
            "fd": fm.createData(),
            "ee_f": fm.getFrameId(ee),
            "ee_name": ee,
            "fidx": [fm.joints[fm.getJointId(n)].idx_q for n in jn],
            "lo": int(ents[en]["lo"]),
            "jn": jn,
        }
    meta = json.loads(str(z["meta"]))
    ref_lo = max(int(e["hi"]) for e in ents.values())  # traj_record: the reference object poses follow the
    refc = {sd["side"]: ref_lo + 7 * i for i, sd in enumerate(meta["sides"])}  # entities, 7 per side, in `sides` order
    if z["states"].shape[1] != ref_lo + 7 * len(meta["sides"]) or set(refc) != {s for s, _ in SIDES}:
        raise SystemExit(
            f"[FHPOST] record columns {z['states'].shape[1]} != entities {ref_lo} + 7 × sides {list(refc)}"
        )
    _W.update(
        pin=pin,
        a=a,
        mods=mods,
        refc=refc,
        S=np.asarray(z["states"], dtype=np.float64),
        F=np.asarray(z[arr], dtype=np.float64),
        names=[str(c) for c in z["clip_names"]],
        sel=z["sel"],
        ko=z["kept_offset"],
        kl=z["kept_length"],
        fbc=[int(v) for v in z["fb_clip"]],
        fbo=z["fb_offset"],
        fbl=z["fb_length"],
    )


def _episode(ci: int):
    k = int(_W["sel"][ci])
    if k >= 0:
        return int(_W["ko"][k]), int(_W["kl"][k]), "success_rank0"
    if ci not in _W["fbc"]:
        raise RuntimeError(f"[FHPOST] clip {_W['names'][ci]}: no selected episode and no failure_best")
    j = _W["fbc"].index(ci)
    return int(_W["fbo"][j]), int(_W["fbl"][j]), "failure_best"


def _palm_env(m, s):
    """Rollout palm pose (R, p) in the env-local frame: root link pose ∘ FK of the floating joints."""
    pin = _W["pin"]
    q = pin.neutral(m["fm"])
    lo = m["lo"]
    for i, qi in enumerate(m["fidx"]):
        q[qi] = s[lo + 7 + i]
    pin.forwardKinematics(m["fm"], m["fd"], q)
    pin.updateFramePlacements(m["fm"], m["fd"])
    Rr, o = G.quat_R(s[lo + 3 : lo + 7]), m["fd"].oMf[m["ee_f"]]
    return Rr @ o.rotation, Rr @ o.translation + s[lo : lo + 3]


def _write_dofs_per_side(mods, lay, model, data, mujoco, off, n, row, clip, qc, qn, t_clip) -> dict:
    """Per hand: the 6 wrist dofs solved frame by frame onto the rollout palm path and the finger joints copied by
    name, into qn (rows of row are added); returns side → (palm body, rollout palm poses)."""
    palms = {}
    for side, _ in SIDES:
        m = mods[side]
        hs = int(lay.hands[side])
        b, R0, p0, sl, hi = G.wrist_chain(model, mujoco, m["ee_name"], hs)
        A = [_palm_env(m, _W["S"][off + k]) for k in range(n)]
        row[f"{side}_frame_rot_deg"] = G.ang_deg((RZ @ data.xmat[b].reshape(3, 3)).T @ A[0][0])
        if row[f"{side}_frame_rot_deg"] > TOL_FRAME_ROT_DEG:
            raise RuntimeError(
                f"[FHPOST] {clip} {side}: floating palm frame vs MuJoCo palm body differ by "
                f"{row[f'{side}_frame_rot_deg']:.3f} deg at frame 0 (G = Rz(−90°) does not hold)"
            )
        c = _W["c"][side]
        # wrist dofs, frame by frame (frame 0 continues from the canonical frame-0 dofs)
        prev = qc[0, hs + 3 : hs + 6].copy()
        for k in range(n):
            Rm = RZ.T @ A[k][0]
            pm = RZ.T @ (A[k][1] - t_clip) - Rm @ c
            qn[k, hs : hs + 6] = G.wrist_dofs(R0, p0, sl, hi, Rm, pm, prev)
            prev = qn[k, hs + 3 : hs + 6]
        # fingers by joint name (the reference of the fingers: achieved positions or PD targets)
        seen = 0
        for i, name in enumerate(m["jn"]):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                continue
            adr = int(model.jnt_qposadr[j])
            if not hs + 6 <= adr < hs + lay.hand_qpos_dim:
                raise RuntimeError(f"[FHPOST] {clip} {side}: joint {name} at qpos {adr} is outside the finger block")
            qn[:n, adr] = _W["F"][off : off + n, m["lo"] + 7 + i]
            seen += 1
        if seen != lay.hand_qpos_dim - 6:
            raise RuntimeError(
                f"[FHPOST] {clip} {side}: {seen} finger joints matched by name, tree has {lay.hand_qpos_dim - 6}"
            )
        qn[n:, hs : hs + lay.hand_qpos_dim] = qn[n - 1, hs : hs + lay.hand_qpos_dim]
        row[f"{side}_f0_wrist_dof_dev"] = float(np.abs(qn[0, hs : hs + 6] - qc[0, hs : hs + 6]).max())
        row[f"{side}_f0_finger_dev_rad"] = float(
            np.abs(qn[0, hs + 6 : hs + lay.hand_qpos_dim] - qc[0, hs + 6 : hs + lay.hand_qpos_dim]).max()
        )
        palms[side] = (b, A)
    return palms


def _check_palm_match(model, data, mujoco, n, qn, palms, t_clip, row, clip) -> None:
    """The written dofs put the MuJoCo palm where the rollout palm was, every frame (dies otherwise)."""
    # check: the written dofs put the MuJoCo palm where the rollout palm was, every frame
    fk_p, fk_r = 0.0, 0.0
    for k in range(n):
        data.qpos[:] = qn[k]
        mujoco.mj_kinematics(model, data)
        for side, _ in SIDES:
            b, A = palms[side]
            fk_p = max(
                fk_p,
                float(
                    np.linalg.norm(RZ @ (data.xpos[b] + data.xmat[b].reshape(3, 3) @ _W["c"][side]) + t_clip - A[k][1])
                ),
            )
            fk_r = max(fk_r, G.ang_deg((RZ @ data.xmat[b].reshape(3, 3)).T @ A[k][0]))
    row.update(fk_pos_err_m_max=fk_p, fk_rot_err_deg_max=fk_r)
    if fk_p > TOL_FK_M or fk_r > TOL_FK_DEG:
        raise RuntimeError(
            f"[FHPOST] {clip}: written wrist dofs miss the rollout palm ({fk_p * 1e3:.4f} mm · {fk_r:.4f} deg)"
        )


def _copy_tree(a, clip, src_tree, dst_tree, kin, lay, T, qn, kind, L) -> None:
    """The derived tree: canonical tree files copied (no links), trajectory_kinematic.npz with the new hand columns."""
    # the tree
    (dst_tree / "0").mkdir(parents=True, exist_ok=True)
    for f in TREE_FILES + ("F0_REPAIR.json",):
        if (src_tree / f).is_file():
            shutil.copy2(src_tree / f, dst_tree / f)
        elif f in TREE_FILES:
            raise RuntimeError(f"[FHPOST] {clip}: canonical tree has no {f}")
    # Every other per-frame file is COPIED. A link into the canonical tree is a write path into it: the tree tools
    # that run on a derived tree afterwards (retime.py rewrites the contact sidecars and trajectory_ikrollout.npz in
    # place and copies the kinematic to .pre_rt) follow links.
    for f in sorted((src_tree / "0").iterdir()):
        d = dst_tree / "0" / f.name
        if f.name not in ("trajectory_kinematic.npz", "body_positions.npz") and f.is_file() and not d.exists():
            shutil.copy2(f, d)
    links = [str(x) for x in dst_tree.rglob("*") if x.is_symlink()]
    if links:
        raise RuntimeError(f"[FHPOST] {clip}: derived tree holds links (a write path out of it): {links[:3]}")
    keep = {k: kin[k] for k in kin.files}
    dt = 1.0 / float(kin["frequency"])
    hb = slice(0, 2 * lay.hand_qpos_dim) if sorted(lay.hands.values()) == [0, lay.hand_qpos_dim] else None
    if hb is None or "qvel" not in keep or keep["qvel"].shape[0] != T:
        raise RuntimeError(f"[FHPOST] {clip}: hand blocks are not qpos 0..{2 * lay.hand_qpos_dim} or qvel is missing")
    qv = np.asarray(keep["qvel"]).copy()
    qv[:, hb] = np.gradient(qn[:, hb], dt, axis=0)
    keep.update(
        qpos=qn.astype(kin["qpos"].dtype),
        qvel=qv.astype(kin["qvel"].dtype),
        fhpost_source=np.array(Path(a.traj).name),
        fhpost_fingers=np.array(a.fingers),
        fhpost_episode=np.array(kind),
        fhpost_episode_len=np.int32(L),
    )
    tmp = dst_tree / "0" / "trajectory_kinematic.tmp.npz"
    np.savez(tmp, **keep)
    os.replace(tmp, dst_tree / "0" / "trajectory_kinematic.npz")


def _posthoc_arm_per_side(a, pin, mods, canon, out, rel_pre, clip, T, n, palms, kind, row) -> None:
    """The post-hoc arm per side: one warm-started IK run on the rollout palm path → the arm-IK cache."""
    # the post-hoc arm: one warm-started run on the rollout palm path
    for side, _ in SIDES:
        m = mods[side]
        src = canon / rel_pre / f"arm_ik_poses_{side}{a.src_suffix}" / f"{clip}.npz"
        dst = out / rel_pre / f"arm_ik_poses_{side}{a.suffix}"
        dst.mkdir(parents=True, exist_ok=True)
        z = np.load(src)
        if tuple(str(v) for v in z["arm_joint_names"]) != m["joints"]:
            raise RuntimeError(f"[FHPOST] {clip} {side}: cache joint names != combo")
        qa = np.asarray(z["arm_joint_positions_all"], dtype=np.float64)
        if qa.shape[0] != T:
            raise RuntimeError(f"[FHPOST] {clip} {side}: cache frames {qa.shape[0]} != tree {T}")
        base = np.asarray(z["base_xyz"], dtype=np.float64)
        A = palms[side][1]
        q_prev = pin.neutral(m["am"])
        for i, qi in enumerate(m["aidx"]):
            q_prev[qi] = qa[0][i]
        pin.forwardKinematics(m["am"], m["ad"], q_prev)
        pin.updateFramePlacements(m["am"], m["ad"])
        Ma = m["ad"].oMf[m["ee_a"]]
        d0, r0 = float(np.linalg.norm(A[0][1] - base - Ma.translation)), G.ang_deg(Ma.rotation.T @ A[0][0])
        row[f"{side}_f0_target_vs_canon_mm"], row[f"{side}_f0_target_vs_canon_deg"] = d0 * 1e3, r0
        c0, cr0 = float(z["wrist_pos_error_all"][0]), float(z["wrist_rot_error_deg_all"][0])
        if abs(d0 - c0) > TOL_F0_ARM_M or abs(r0 - cr0) > TOL_F0_ARM_DEG:
            raise RuntimeError(
                f"[FHPOST] {clip} {side}: rollout palm − base_xyz vs the canonical frame-0 arm FK "
                f"({d0 * 1e3:.3f} mm · {r0:.3f} deg) is not the cache's own frame-0 residual "
                f"({c0 * 1e3:.3f} mm · {cr0:.3f} deg) — the rollout did not start at the reference"
            )
        q_new, pe, re, nit = np.zeros_like(qa), np.zeros(T), np.zeros(T), np.zeros(T, dtype=np.int32)
        for t in range(T):
            Rt, pt = A[min(t, n - 1)]
            qs, pe[t], re[t], nit[t] = AP.solve_ik_pose(
                m["am"],
                m["ad"],
                (pt - base).copy(),
                Rt.copy(),
                m["ee_a"],
                m["aidx"],
                max_iters=a.max_iters,
                eps_pos=EPS_POS_M,
                eps_rot_deg=EPS_ROT_DEG,
                q0=q_prev,
            )
            q_new[t] = [qs[qi] for qi in m["aidx"]]
            q_prev = qs
        kz = {k: z[k] for k in z.files}
        kz.update(
            arm_joint_positions=q_new[0].astype(z["arm_joint_positions"].dtype),
            arm_joint_positions_all=q_new.astype(z["arm_joint_positions_all"].dtype),
            wrist_pos_error=np.float32(pe[0]),
            wrist_pos_error_all=pe.astype(np.float32),
            wrist_rot_error_deg=np.float32(re[0]),
            wrist_rot_error_deg_all=re.astype(np.float32),
            solver=np.array("fhpost_seq_warmstart_f0canon"),
            seq_source_solver=np.array(str(z["solver"])),
            seq_f0_seed=np.array("canonical arm_joint_positions_all[0]"),
            seq_iters_all=nit,
            fhpost_source=np.array(Path(a.traj).name),
            fhpost_episode=np.array(kind),
        )
        tmp = dst / f"{clip}.tmp.npz"
        np.savez(tmp, **kz)
        os.replace(tmp, dst / f"{clip}.npz")
        step = np.linalg.norm(np.diff(q_new, axis=0), axis=1) if T > 1 else np.zeros(1)
        row.update(
            {
                f"{side}_pos_err_mm_p99": float(np.quantile(pe, 0.99) * 1e3),
                f"{side}_pos_err_mm_max": float(pe.max() * 1e3),
                f"{side}_rot_err_deg_max": float(re.max()),
                f"{side}_frames_over_2mm_1deg": int(((pe > 2e-3) | (re > 1.0)).sum()),
                f"{side}_step_l2_max_rad": float(step.max()),
                f"{side}_dev_from_canonical_deg_max": float(np.degrees(np.abs(q_new - qa).max())),
            }
        )


def _one(ci: int) -> dict:
    import mujoco

    pin, a, mods = _W["pin"], _W["a"], _W["mods"]
    clip = _W["names"][ci]
    canon, out = Path(a.canon_root), Path(a.out_root)
    rel_tree = Path("data") / a.dataset / "trees" / f"{a.arm}_{a.hand}" / "bimanual" / clip
    rel_pre = Path("precompute") / a.dataset / "robots" / f"{a.arm}_{a.hand}"
    src_tree, dst_tree = canon / rel_tree, out / rel_tree
    lay = qpos_layout.read(str(src_tree))
    kin = np.load(src_tree / "0" / "trajectory_kinematic.npz", allow_pickle=True)
    qc = np.asarray(kin["qpos"], dtype=np.float64)
    T = qc.shape[0]
    off, L, kind = _episode(ci)
    n = min(L, T)
    model = mujoco.MjModel.from_xml_path(str(src_tree / "scene.xml"))
    data = mujoco.MjData(model)
    if model.nq != qc.shape[1]:
        raise RuntimeError(f"[FHPOST] {clip}: model nq {model.nq} != qpos {qc.shape[1]}")
    data.qpos[:] = qc[0]
    mujoco.mj_kinematics(model, data)
    row = {
        "clip": clip,
        "frames": int(T),
        "episode": kind,
        "episode_len": int(L),
        "frames_held": int(T - n),
        "record_longer": int(max(0, L - T)),
    }
    qn = qc.copy()
    tt = np.stack(
        [
            _W["S"][off : off + n, _W["refc"][s] : _W["refc"][s] + 3] - qc[:n, lay.main[s] : lay.main[s] + 3] @ RZ.T
            for s, _ in SIDES
        ]
    )
    t_clip = tt[0, 0].copy()
    row.update(t_clip=[float(v) for v in t_clip], t_clip_dev_m=float(np.abs(tt - t_clip).max()))
    if row["t_clip_dev_m"] > TOL_T_CLIP_M:
        raise RuntimeError(
            f"[FHPOST] {clip}: the record's reference objects are not G · the tree's objects on every step "
            f"(dev {row['t_clip_dev_m'] * 1e3:.3f} mm) — another tree, or step k is not frame k"
        )
    if _W.get("c") is None:  # the pre-pass: the palm-frame offset of this clip at frame 0, nothing written
        res = {}
        for side, _ in SIDES:
            m = mods[side]
            b = G.wrist_chain(model, mujoco, m["ee_name"], int(lay.hands[side]))[0]
            Ra, pa = _palm_env(m, _W["S"][off])
            Rb = data.xmat[b].reshape(3, 3)
            res[side] = (Rb.T @ (RZ.T @ (pa - t_clip) - data.xpos[b]), G.ang_deg((RZ @ Rb).T @ Ra))
        return res
    palms = _write_dofs_per_side(mods, lay, model, data, mujoco, off, n, row, clip, qc, qn, t_clip)
    _check_palm_match(model, data, mujoco, n, qn, palms, t_clip, row, clip)
    _copy_tree(a, clip, src_tree, dst_tree, kin, lay, T, qn, kind, L)
    _posthoc_arm_per_side(a, pin, mods, canon, out, rel_pre, clip, T, n, palms, kind, row)
    # how far the rollout hands are from the reference hands (the tree this replaces)
    row["palm_dev_from_reference_mm_max"] = float(
        1e3
        * max(
            np.abs(qn[:n, lay.hands[s] : lay.hands[s] + 3] - qc[:n, lay.hands[s] : lay.hands[s] + 3]).max()
            for s, _ in SIDES
        )
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--canon-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--floating-urdf-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--suffix", required=True, help="suffix of the written arm-IK cache dirs (an identity, no default)")
    ap.add_argument("--src-suffix", required=True, help="suffix of the canonical arm-IK cache dirs")
    ap.add_argument("--fingers", required=True, choices=("pos", "tgt"))
    ap.add_argument("--max_iters", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (a dry run)")
    a = ap.parse_args()
    if Path(a.out_root).resolve() == Path(a.canon_root).resolve() or a.suffix == a.src_suffix:
        raise SystemExit("[FHPOST] --out-root is the canonical root or --suffix is the canonical suffix — refused")
    _init(a)
    n = len(_W["names"])
    ids = list(range(n))[: a.limit or n]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.jobs) as pool:  # pre-pass: the palm-frame offset, one value for the set
        pre = pool.map(_one, ids, chunksize=1)
    cfit, cmad = {}, {}
    for side, _ in SIDES:
        cs = np.array([p[side][0] for p in pre])
        cfit[side] = np.median(cs, axis=0)
        cmad[side] = float(np.median(np.linalg.norm(cs - cfit[side], axis=1)))
        rot = max(p[side][1] for p in pre)
        if cmad[side] > TOL_PALM_OFF_MAD_M or rot > TOL_FRAME_ROT_DEG:
            raise SystemExit(
                f"[FHPOST] {side}: palm-frame offset is not one value over the clips (spread {cmad[side] * 1e3:.3f} mm"
                f" · rotation up to {rot:.3f} deg)"
            )
    _W["c"] = cfit
    print(
        "[FHPOST] palm-frame offset (MuJoCo palm body → arm URDF palm frame), body frame, mm: "
        + " · ".join(f"{s} {np.round(cfit[s] * 1e3, 2).tolist()} (spread {cmad[s] * 1e3:.4f})" for s, _ in SIDES),
        flush=True,
    )
    with mp.get_context("fork").Pool(a.jobs) as pool:
        rows = pool.map(_one, ids, chunksize=1)
    out = Path(a.out_root)
    summ = {
        "traj": str(a.traj),
        "fingers": a.fingers,
        "suffix": a.suffix,
        "src_suffix": a.src_suffix,
        "clips": len(rows),
        "episodes": {k: sum(r["episode"] == k for r in rows) for k in ("success_rank0", "failure_best")},
        "failure_best_clips": [r["clip"] for r in rows if r["episode"] == "failure_best"],
        "clips_with_held_frames": sum(r["frames_held"] > 0 for r in rows),
        "palm_offset_m": {s: cfit[s].tolist() for s, _ in SIDES},
        "palm_offset_spread_m": cmad,
        "t_clip_dev_m_max": max(r["t_clip_dev_m"] for r in rows),
        "t_clip_min": np.min([r["t_clip"] for r in rows], axis=0).tolist(),
        "t_clip_max": np.max([r["t_clip"] for r in rows], axis=0).tolist(),
        "fk_pos_err_m_max": max(r["fk_pos_err_m_max"] for r in rows),
        "palm_dev_from_reference_mm_max_median": float(np.median([r["palm_dev_from_reference_mm_max"] for r in rows])),
        "seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    (out / f"FHPOST{a.suffix}.json").write_text(json.dumps(summ, indent=1))
    print(
        f"[FHPOST] fingers={a.fingers} clips={len(rows)} episodes={summ['episodes']} "
        f"held={summ['clips_with_held_frames']} "
        f"t_clip=[{np.round(summ['t_clip_min'], 4).tolist()} … {np.round(summ['t_clip_max'], 4).tolist()}] "
        f"fk_max={summ['fk_pos_err_m_max'] * 1e3:.4f} mm · {summ['seconds']} s → {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
