"""Guards and the idle arm of the bake's arm IK stage (`coarse_path.dp`).

The guards are the judge's and the gate's own functions (`collision.side_guard`, `collision.cross_guard`); they run
on the bake's final joints just before each save. The idle arm (one-hand clips) solves no IK: it holds the canonical
rest pose and its hand block in the tree is rewritten to match. The stage module `dp` owns the per-run state these
functions need (`_T0`, `DESK_XY`) and passes it in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from dexengine.placement.coords import core as _VC

#: Failure word that ends a guard's `[stage_dp]` verdict line — the bake driver copies lines with "fail" into the
#: ledger; arm_flip.bake_fix relabels it for a candidate rejection.
FAIL_MARK = "— failed"


def _side_guard(q_all, arm, hand, side, z_off, seq, tree, desk_xy=None) -> bool:
    """Final per-side arm guard (desk · support objects · torso) — collision.side_guard, the judge's and the gate's
    own step, on the bake's final joints (the f0-inserted head frames included, which the judge never sees). The
    torso guard's fingers = the tree's finger values (an idle side: the hand's rest pose, hand_surface.guard_finger_q).
    Supports come from the support_surface module that collision imports (a plain import: a second module copy
    would make a second SceneSupports class, which the guard refuses)."""
    from dexengine.placement.collision import core as SCL  # noqa: PLC0415
    from dexengine.placement.collision import hand_surface as HSF  # noqa: PLC0415
    from dexengine.placement.collision import idle_arm as IA  # noqa: PLC0415
    from dexengine.placement.collision import support_surface as SSF  # noqa: PLC0415

    fq = HSF.guard_finger_q(tree, hand, idle=IA.idle_sides_of_tree(tree))[side]
    ok, why = SCL.side_guard(
        SCL.build_arm_ctx(arm, hand, side),
        q_all,
        side,
        z_off=z_off,
        desk_xy=desk_xy,
        fingers=fq,
        supports=SSF.scene_supports(tree),
    )
    if not ok:
        print(f"[stage_dp] {seq}/{side}: arm guard {why} {FAIL_MARK}", flush=True)
    return ok


def _cross_guard(q_all, arm, hand, side, z_off, seq, out_dir, t0: float) -> bool:
    """Final two-arm cross guard — collision.cross_guard, the same function and threshold as the judge and the gate.

    It is pairwise: the chain solves the right arm, then the left arm in the same run, so the later side reads the
    partner npz the earlier side has just written. A missing partner (the first side) passes; the pair is checked
    when the partner arrives. `t0` = when the stage module was loaded for this run: a partner file older than that is
    a leftover of another anchor or generation and is not used (the final guard checks again after retiming).
    """
    from dexengine.placement.collision import core as SCL  # noqa: PLC0415

    other = "left" if side == "right" else "right"
    od = Path(str(out_dir).replace(f"_{side}_", f"_{other}_"))
    op = od / f"{seq}.npz"
    if od == Path(out_dir) or not op.exists():
        return True
    if op.stat().st_mtime < t0:
        return True
    z = np.load(op)
    q_o = np.asarray(z["arm_joint_positions_all"], np.float64)
    # Second check: the stamped length and base_z_off. z_off is stamped as float32, so the threshold absorbs the
    # round trip (-0.1925 → -0.19249999523); anchors are one grid cell (3.5 cm) apart, so 1e-6 separates them.
    zo = float(z["base_z_off"]) if "base_z_off" in z.files else z_off
    if len(q_o) != len(q_all) or abs(zo - z_off) > 1e-6:
        print(
            f"[stage_dp] {seq}/{side}: the opposite side's npz is not this placement's partner "
            f"(length {len(q_o)} vs {len(q_all)}, z_off {zo} vs {z_off}) — cross check skipped",
            flush=True,
        )
        return True
    qR, qL = (q_all, q_o) if side == "right" else (q_o, q_all)
    ok, why = SCL.cross_guard(SCL.build_arm_ctx(arm, hand, "right"), SCL.build_arm_ctx(arm, hand, "left"), qR, qL)
    if not ok:
        print(f"[stage_dp] {seq}/{side}: both arms {why} {FAIL_MARK} (the driver tries the next anchor)", flush=True)
    return ok


def _hand_root_from_palm(
    tree: Path, side: str, palm_body: str, P: np.ndarray, R: np.ndarray, off: int, H: int
) -> np.ndarray:
    """World pose of the palm body (replay convention = body_positions.npz frame) → the 6 virtual-wrist qpos of the
    hand block in the tree.

    Rule (read from the MJCF of any hand): the palm body is a direct child of world with 6 joints = slide 3 + hinge
    3, joint origins 0. MuJoCo applies the joints in order, so
    xpos = p0 + R0·Σ q_i a_i, xmat = R0·Rot(b1,q4)·Rot(b2,q5)·Rot(b3,q6).
    Raise if a hinge axis is not a principal axis (±e). The closed-form solution is read back with mj_forward and
    compared; a mismatch raises.
    """
    import mujoco
    from scipy.spatial.transform import Rotation

    m = mujoco.MjModel.from_xml_path(str(tree / "scene.xml"))
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, palm_body)
    if b < 0 or m.body_parentid[b] != 0:
        raise RuntimeError(f"[idle] {tree.name}/{side}: palm body {palm_body} missing or not a direct child of world")
    ja, jn = int(m.body_jntadr[b]), int(m.body_jntnum[b])
    if jn != 6 or int(m.jnt_qposadr[ja]) != off or any(int(m.jnt_qposadr[ja + k]) != off + k for k in range(6)):
        raise RuntimeError(
            f"[idle] {tree.name}/{side}: the 6 virtual-wrist joints are not contiguous at qpos[{off}:{off + 6}]"
        )
    types = [int(m.jnt_type[ja + k]) for k in range(6)]
    if types[:3] != [int(mujoco.mjtJoint.mjJNT_SLIDE)] * 3 or types[3:] != [int(mujoco.mjtJoint.mjJNT_HINGE)] * 3:
        raise RuntimeError(f"[idle] {tree.name}/{side}: joint types are not slide×3 + hinge×3: {types}")
    if np.abs(m.jnt_pos[ja : ja + 6]).max() > 1e-9:
        raise RuntimeError(f"[idle] {tree.name}/{side}: hinge origins are not 0 — outside the closed-form rule")
    p0 = np.asarray(m.body_pos[b], float)
    R0 = Rotation.from_quat(np.asarray(m.body_quat[b], float)[[1, 2, 3, 0]]).as_matrix()  # MuJoCo wxyz → scipy xyzw
    A = np.stack([np.asarray(m.jnt_axis[ja + k], float) for k in range(3)], axis=1)  # columns = slide axes
    q_slide = np.linalg.solve(A, R0.T @ (np.asarray(P, float) - p0))
    seq, sgn = "", []
    for k in range(3, 6):
        ax = np.asarray(m.jnt_axis[ja + k], float)
        i = int(np.abs(ax).argmax())
        if not np.allclose(np.abs(ax), np.eye(3)[i], atol=1e-9):
            raise RuntimeError(f"[idle] {tree.name}/{side}: hinge axis is not a principal axis: {ax}")
        seq += "XYZ"[i]
        sgn.append(float(np.sign(ax[i])))
    ang = Rotation.from_matrix(R0.T @ np.asarray(R, float)).as_euler(seq)  # upper = intrinsic R1·R2·R3
    q_hinge = np.asarray(ang) / np.asarray(sgn)
    root6 = np.concatenate([q_slide, q_hinge])
    # Read it back
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[off : off + 6] = root6
    mujoco.mj_forward(m, d)
    dp = float(np.linalg.norm(d.xpos[b] - P))
    Rm = Rotation.from_quat(np.asarray(d.xquat[b], float)[[1, 2, 3, 0]]).as_matrix()
    dang = float(np.degrees(np.linalg.norm((Rotation.from_matrix(Rm.T @ R)).as_rotvec())))
    if dp > 1e-6 or dang > 1e-4:
        raise RuntimeError(
            f"[idle] {tree.name}/{side}: virtual-wrist read-back mismatch Δp={dp:.2e}m Δθ={dang:.2e}° (rule error)"
        )
    return root6


def _write_idle_hand(HSF, a, H, root6, qpos, off, z, lay):
    """The idle hand at its rest posture: root from the palm, fingers at rest_finger_q; qvel of those columns zeroed."""
    rest = HSF.rest_finger_q(a.hand)
    if len(rest) != H - 6:
        raise RuntimeError(
            f"[idle] {a.sequence}/{a.side}: hand width {H} − virtual wrist 6 != fingers {len(rest)} ({a.hand})"
        )
    qpos[:, off : off + 6] = root6[None, :]
    qpos[:, off + 6 : off + H] = rest[None, :]
    z["qpos"] = qpos.astype(np.asarray(z["qpos"]).dtype)
    if "qvel" in z and np.asarray(z["qvel"]).ndim == 2:
        starts = sorted({int(c) for c in lay.main.values() if c is not None} | {int(c) for c in lay.extras})
        voff = int(off) - sum(1 for c in starts if c < off)  # per preceding object block, qpos 7 ↔ qvel 6
        qv = np.array(z["qvel"])
        qv[:, voff : voff + H] = 0.0
        z["qvel"] = qv


def idle_rest(a, joints, base_xyz, DESK: float, palm_off: np.ndarray, rz, guards) -> int:
    """Idle arm: solve no IK — apply the rest pose (idle_arm.idle_qpos) to every frame, write the IK npz, and replace
    that hand block in the tree with the idle arm's FK wrist pose + the fingers' rest pose (hand_surface.rest_finger_q,
    so render, contact and the gate all see one pose). `guards(q)` = the stage's guards (same as for an active arm);
    returns the stage rc (0 · 4 = guard failure)."""
    from scipy.spatial.transform import Rotation

    from dexengine.core import hands  # noqa: PLC0415
    from dexengine.core import qpos_layout as QL  # noqa: PLC0415
    from dexengine.placement.collision import core as SCL  # noqa: PLC0415
    from dexengine.placement.collision import hand_surface as HSF  # noqa: PLC0415
    from dexengine.placement.collision import idle_arm as IA  # noqa: PLC0415

    tree = Path(a.tree)
    kin_p = tree / "0/trajectory_kinematic.npz"
    z = dict(np.load(kin_p, allow_pickle=True))
    qpos = np.array(z["qpos"], np.float64)
    T = len(qpos)
    lay = QL.read(str(tree), nq=qpos.shape[1])
    off, H = lay.hands.get(a.side), int(lay.hand_qpos_dim)
    if off is None:
        raise RuntimeError(f"[idle] {a.sequence}/{a.side}: the tree has no hand block on that side (hands={lay.hands})")
    q_all = IA.idle_arm_q(a.arm, a.side, T)
    base = np.asarray(base_xyz, float) + np.array([0.0, 0.0, a.base_z_off])
    # Idle arm EE (robot-local) → the inverse of the stage's target transform:
    # pb = rz·(P + rot·palm_off) + (0,0,DESK) − base, Rb = rz·rot
    T_ee = IA.idle_ee_pose(SCL.build_arm_ctx(a.arm, a.hand, a.side))
    rot = rz.inv() * Rotation.from_matrix(T_ee[:3, :3])
    P = rz.inv().apply(T_ee[:3, 3] + base - np.array([0.0, 0.0, DESK])) - rot.apply(palm_off)
    _spec = hands.HAND_REGISTRY[a.hand]
    palm_body = _spec.palm_fk_body_right.replace("right_", f"{a.side}_", 1)
    root6 = _hand_root_from_palm(tree, a.side, palm_body, P, rot.as_matrix(), int(off), H)
    if getattr(a, "no_idle_guard", False):  # kinematic-data variant: guards recorded, never rejecting
        ok = guards(q_all)
        print(
            f"[stage_dp] {a.sequence}/{a.side}: idle-arm guard off (--no-idle-guard) · guard result "
            f"{'pass' if ok else 'hit'}",
            flush=True,
        )
    elif not guards(q_all):
        return 4
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"{a.sequence}.npz",
        base_xyz=np.asarray(base, dtype=np.float64),
        arm_joint_names=np.array(joints),
        arm_joint_positions=q_all[0].astype(np.float32),
        arm_joint_positions_all=q_all.astype(np.float32),
        wrist_pos_error=np.float32(0.0),
        wrist_pos_error_all=np.zeros(T, np.float32),
        wrist_rot_error_deg=np.float32(0.0),
        wrist_rot_error_deg_all=np.zeros(T, np.float32),
        base_z_off=np.float32(a.base_z_off),
        num_frames=np.int32(T),
        solver=np.array(IA.IDLE_SOLVER_TAG, dtype="<U24"),
    )
    _write_idle_hand(HSF, a, H, root6, qpos, off, z, lay)
    tmp = kin_p.with_suffix(".tmp.npz")
    np.savez(tmp, **z)
    os.replace(tmp, kin_p)
    # Whoever writes a baked tree trajectory re-stamps the supports right after.
    if _VC.is_baked_tree(json.loads((tree / "task_info.json").read_text())):
        _VC.stamp_scene(tree, keep_height=True)
    print(
        f"[IDLE_SIDE] {a.sequence}/{a.side}: T={T} q(deg)={np.round(np.degrees(q_all[0]), 1).tolist()} "
        f"palm_world={np.round(P, 4).tolist()} root6={np.round(root6, 4).tolist()} → tree qpos[{off}:{off + H}] "
        f"(virtual wrist updated · finger rest pose) · solver={IA.IDLE_SOLVER_TAG} · error 0",
        flush=True,
    )
    return 0
