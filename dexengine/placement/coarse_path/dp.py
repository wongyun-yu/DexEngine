#!/usr/bin/env python3
"""Bake arm IK stage (`ik` in the stage chain): coarse path by dynamic programming over the reach-map valleys.

Per frame, the candidates are the valley solutions of the 3×3×3 map cells around the wrist target and its N nearest
orientation atoms (deduplicated by joint distance), polished by a few DLS steps. DP then picks one candidate per
frame minimising the per-joint step relative to the joint velocity budget (steps over the budget are penalised),
with a small bonus for joint-limit margin; frames without a candidate are filled by linear interpolation of the
picks. A final DLS polish over every frame gives the arm trajectory, which must pass the same checks as the judge:
tracking error p99 on the post-retiming time axis (`--max-err-mm`, stretch.weighted_p99), the total retiming
stretch (`--max-stretch`), the arm-flip gate, and the desk · support · torso · cross guards.

When the judge already solved the clip (`--reuse-npz`), its joints are reused and only the frames the f0 repair
prepended are solved; reuse that fails verification falls back to the DP path. An idle arm (one-hand clips) solves
no IK (`guards.idle_rest`). The output npz is the arm-IK format every downstream step reads.

Return codes: 0 written · 3 rejected (coverage, error, stretch, arm flip) · 4 guard failure.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# isort: split
from dexengine.core.paths import pool_root  # (owner of the canonical map root)
from dexengine.placement import arm_flip as _AF  # (owner of the arm-flip gate + re-solve)
from dexengine.placement.coarse_path import guards as _G
from dexengine.placement.coords import core as _VC  # (owner of the scene z offset and desk bounds)
from dexengine.placement.reach_map import store as _PM  # (reach map identity check)

# The moment this module was loaded — the cross guard's test of "was the opposite side's npz written by this run".
# The chain solves the right and the left arm back to back in one process, so only this run's outputs are newer.
_T0 = time.time()
# Resident worker (bake/server.py): _T0 means "when this module was loaded", so the server refills it right before
# the stage is first called in each request.
_SERVER_LOADTIME = ("_T0",)
# Arm IK objects (pytorch_kinematics chain, ≈0.5 s to build per side), keyed by (side, URDF, EE, joint names) only;
# `_dls` does not change them — a clip-independent cache the resident server keeps between clips.
_IK_MEMO: dict = {}
_SERVER_KEEP = ("_IK_MEMO",)

DESK_XY = None  # set by main from the tree (coords.desk_bounds_scene_for)

#: DLS damping of every polish in this stage.
DLS_LAMBDA = 1e-2
#: A polished candidate is kept when its wrist error is below these (m, deg).
POLISH_OK_POS_M, POLISH_OK_ROT_DEG = 0.01, 8.0
#: A map valley is a candidate when its stored error is below these (m, deg) and its arm and hand link heights are
#: above the desk minus VALLEY_DESK_SLACK_M (robot-local, table scenes).
VALLEY_OK_POS_M, VALLEY_OK_ROT_DEG, VALLEY_DESK_SLACK_M = 0.03, 20.0, 0.05
#: Frames the f0 repair prepends (n_blend) and the frame stride of the judge's trajectory — used only to describe
#: where tracking errors sit in the rejection line.
F0_BLEND_FRAMES, JUDGE_STRIDE = 20, 5
#: Iteration multiplier of the one extra polish a borderline reused trajectory gets.
REPOLISH_ITERS_MULT = 3

RC_PIN_RETRY = 90  # finish_dp_and_write → caller: solve once more, pinned around the judge trajectory


def stage_guards(q_all, a, out_dir) -> bool:
    """Hard penetration / cross constraints just before saving — every save point has to pass this."""
    return _G._side_guard(
        q_all, a.arm, a.hand, a.side, a.base_z_off, a.sequence, a.tree, desk_xy=DESK_XY
    ) and _G._cross_guard(q_all, a.arm, a.hand, a.side, a.base_z_off, a.sequence, out_dir, _T0)


def arm_flip_fix(a, *args, **kw):
    """Arm-flip gate + re-solve + bake end gate — the body is arm_flip.bake_fix (owner of the rule and values)."""
    return _AF.bake_fix(a, *args, guards=lambda q: stage_guards(q, a, a.output_dir), **kw)


def solver_stamp(name: str, flip: str):
    """The `solver` field of the output npz — an unchanged trajectory keeps the plain name (<U24), a trajectory the
    arm-flip fix changed gets the fix suffix (<U32)."""
    return np.array(name, dtype="<U24") if not flip else np.array(name + flip, dtype="<U32")


def _add_io_args(ap):
    """The clip, side, combo and the input/output folders."""
    ap.add_argument("--sequence", required=True, help="<base>_pl")
    ap.add_argument(
        "--tree",
        required=True,
        help="tree directory (holds task_info.json). The desk z offset of the IK targets = "
        "coords.scene_z_offset(tree), the one owner of the capture z → world z offset",
    )
    ap.add_argument("--side", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--base_z_off", type=float, default=0.0)
    ap.add_argument("--wrist_dir", required=True)
    ap.add_argument(
        "--no-idle-guard",
        action="store_true",
        help="do not reject on the idle arm's penetration / cross guards (the result is only printed; default off)",
    )
    ap.add_argument("--output_dir", required=True)


def _add_solver_args(ap):
    """Reach-map root, IK polish, dedup, atoms and the DP/retime/guard thresholds."""
    ap.add_argument("--pool-root", default=str(pool_root()))  # canonical owner = paths.pool_root
    ap.add_argument("--polish-iters", type=int, default=8)
    ap.add_argument("--final-iters", type=int, default=8)
    ap.add_argument(
        "--budget-mult",
        type=float,
        default=1.0,
        help="allowed transition = per-joint velocity budget (LIM*DT) × this; larger steps are penalised",
    )
    ap.add_argument("--dedup-deg", type=float, default=12.0)
    ap.add_argument("--atoms", type=int, default=3, help="orientation atoms looked up per frame (same as the judge)")
    # The three below are one criterion shared with the judge and the gate: change them together.
    ap.add_argument(
        "--min-cov",
        type=float,
        default=0.5,
        help="minimum fraction of frames with a candidate. Missing frames are filled by interpolation "
        "and judged by the all-frame tracking error, so this is only the minimum for a DP path; "
        "1.0 would drop a clip for a single missing frame",
    )
    ap.add_argument(
        "--max-err-mm",
        type=float,
        default=30.0,
        help="tracking error p99 limit (mm), measured on the post-retiming time axis "
        "(stretch.weighted_p99) — the same value as the judge's --dp-err-mm",
    )
    ap.add_argument(
        "--max-stretch",
        type=float,
        default=3.0,
        help="limit on how many times longer the clip becomes after retiming (same value as the judge's "
        "--dp-max-stretch). Measured as a stretch factor, not a peak speed ratio: retiming slows the "
        "fast spans by 1/k, so the question is how much the clip must be stretched",
    )
    ap.add_argument("--safety", type=float, default=1.15, help="retiming safety factor (same as the bake)")
    ap.add_argument(
        "--max-trim",
        type=int,
        default=2,
        help="a reused trajectory up to this many frames longer than the tree has its tail cut "
        "(the 1~2 frame tail trim of the staging)",
    )
    ap.add_argument(
        "--max-insert",
        type=int,
        default=40,
        help="maximum number of f0-inserted frames accepted on reuse (n_blend default 20 plus slack); "
        "more means a retimed wrist was given and reuse is refused",
    )
    ap.add_argument(
        "--arm-flip-fail",
        choices=("off", "dp"),
        default=_AF.BAKE_GATE_DEFAULT,
        help="bake end gate (arm_flip.bake_gate · canonical BAKE_GATE_DEFAULT) · off = explicitly off",
    )
    ap.add_argument(
        "--dp-pin-rad",
        type=float,
        default=_AF.BAKE_PIN_RAD,
        help="a DP stopped by the gate is solved again pinned within this distance (rad) of the judge "
        "trajectory (BAKE_PIN_RAD) · 0 = never",
    )
    ap.add_argument(
        "--reuse-npz",
        default="",
        help="all-frame joint trajectory confirmed by the judge (judge --save-ik). Skips candidate "
        "collection and DP and solves only the frames f0 prepended",
    )


def build_parser():
    """CLI of the stage (the chain passes the same arguments to `main` through sys.argv)."""
    ap = argparse.ArgumentParser()
    _add_io_args(ap)
    # A map built at another robot base disagrees with the canonical base — the identity check stops it.
    _add_solver_args(ap)
    return ap


def _tree_idle_sides(a):
    """The tree's idle sides (task_info.idle_sides); env DEXENGINE_ACTIVE_SIDES must agree with what the tree states."""
    # One-hand rule: an idle side solves no IK. The owner of "which side is idle" is the tree's
    # task_info.idle_sides (copied from adopted.json by the bake); env DEXENGINE_ACTIVE_SIDES (stage2 --active-sides)
    # must agree with what the tree states.
    import os as _os0

    from dexengine.placement.collision import idle_arm as IA  # noqa: PLC0415

    _idle = IA.idle_sides_of_tree(a.tree)
    _as_env = _os0.environ.get("DEXENGINE_ACTIVE_SIDES", "")
    _has_key = IA.IDLE_SIDES_KEY in json.loads((Path(a.tree) / "task_info.json").read_text())
    if _as_env == "auto" and not _has_key:
        raise RuntimeError(
            f"[stage_dp] {a.sequence}: DEXENGINE_ACTIVE_SIDES=auto but the tree task_info has no "
            f"{IA.IDLE_SIDES_KEY} — the judge did not run with --active-sides auto"
        )
    if _has_key and _as_env != "auto":
        raise RuntimeError(
            f"[stage_dp] {a.sequence}: the tree states {IA.IDLE_SIDES_KEY}={_idle} but the bake is "
            f"not DEXENGINE_ACTIVE_SIDES=auto (now '{_as_env}')"
        )
    return _idle


def _reach_map(a, combo, base_xyz, palm_off, wd_inputs, _SCENE, DESK, base):
    """The reach map of this combo/side, verified against the judge's identity and inputs (wd_inputs = the builder's
    wrist set, arm_ik.resolve_combo) and against the map id the tree records (the judge's, via the bake). Returns its
    folder, axes, atoms, step, sizes and the desk clearance offset."""
    # Reach map (valley q + validity)
    pd = Path(a.pool_root) / f"{combo}_{a.side}"
    if not (pd / "meta.npz").exists():
        # RuntimeError, not SystemExit: the stage runner reports an exception as rc 1.
        raise RuntimeError(
            f"[stage_dp] reach map missing: {pd}\n"
            f"  The current combo is {combo}_{a.side}; a map built for another posture or combo must be given "
            f"explicitly with `DEXENGINE_POOL_ROOT=<root>` or `--pool-root <root>`."
        )
    # Map identity (base + palm_off + arm · hand · side) — the judge's own function (store.verify).
    meta = _PM.verify(
        pd,
        arm=a.arm,
        hand=a.hand,
        side=a.side,
        base_xyz=base_xyz,
        palm_off=palm_off,
        wrist_dir=wd_inputs,
        who="stage_dp",
    )
    _ti = json.loads((Path(a.tree) / "task_info.json").read_text())
    _PM.check_recorded(pd, _PM.recorded(_ti, a.side), who="stage_dp")
    axx, axy, axz = meta["axes_x"], meta["axes_y"], meta["axes_z"]
    atoms = meta["atoms_xyzw"].astype(np.float64)
    step = float(meta["pos_step"])
    nx, ny, nz, A = len(axx), len(axy), len(axz), len(atoms)
    desk_rel = (DESK - base[2]) if _SCENE == "table" else -1e9  # aerial scene: map clearance off (same as the judge)
    return pd, axx, axy, axz, atoms, step, nx, ny, nz, A, desk_rel


def _wrist_targets(a, Rotation, rz, palm_off, DESK, base):
    """Wrist targets in the robot base frame (the same transform as the judge)."""
    # Targets (wrist → robot base frame) — the same transform as the judge
    wz = np.load(Path(a.wrist_dir) / f"{a.sequence}.npz")
    P, Q = wz["positions"], wz["quaternions"]
    if P.ndim == 3:
        P, Q = P[0], Q[0]
    rot = Rotation.from_quat(np.stack([Q[:, 1], Q[:, 2], Q[:, 3], Q[:, 0]], 1))
    pb = rz.apply(P.astype(np.float64) + rot.apply(palm_off)) + np.array([0, 0, DESK]) - base
    Rb = (rz * rot).as_matrix()
    T = len(pb)
    return rot, pb, Rb, T


def _finger_floor(a, T):
    """The fingers' stretch floor on this wrist cache's time axis (another length = another time axis: die)."""
    # The fingers' stretch floor (hand_surface.finger_floor_for — the function retime.py and the judge call), on
    # this wrist cache's time axis; another length = another time axis → die before solving.
    from dexengine.placement.collision import hand_surface as _HSF  # noqa: PLC0415

    FF = _HSF.finger_floor_for(a.tree, a.hand, a.safety)
    if len(FF) != max(T - 1, 1):
        raise RuntimeError(
            f"[stage_dp] {a.sequence}/{a.side}: finger floor {len(FF)} spans != wrist cache {T} frames "
            "— the tree and the wrist cache have different time axes"
        )
    return FF


def _target_tensor(torch, T, DT, DEV, Rb, pb):
    """The (T, 4, 4) target transforms on the IK device."""
    Tt = torch.zeros((T, 4, 4), dtype=DT, device=DEV)
    Tt[:, 3, 3] = 1
    Tt[:, :3, :3] = torch.tensor(Rb, dtype=DT, device=DEV)
    Tt[:, :3, 3] = torch.tensor(pb, dtype=DT, device=DEV)
    return Tt


def _arm_ik(urdf, a, ee, joints, OPG):
    """The arm's pinocchio model and its batched IK (memoised per side, URDF, end effector and joints)."""
    from dexengine.robots import urdf_load  # noqa: PLC0415

    model = urdf_load.build_model(str(urdf))
    _k = (a.side, str(urdf), str(ee), tuple(joints))
    ik = _IK_MEMO.get(_k)
    if ik is None:
        ik = _IK_MEMO[_k] = OPG.ArmIK(a.side, model, urdf, ee, joints)
    return model, ik


def main() -> int:
    """Stage entry → rc (0 written · 3 rejected · 4 guard failure)."""
    # the solve half imports this module's guards and constants (deferred: no import cycle)
    from dexengine.placement.coarse_path.dp_solve import reuse_or_collect_then_dp  # noqa: PLC0415

    ap = build_parser()
    a = ap.parse_args()

    import torch
    from scipy.spatial.transform import Rotation

    from dexengine.placement import arm_ik as AIP  # noqa: PLC0415
    from dexengine.placement.reach_map import batch_ik as OPG  # noqa: PLC0415
    from dexengine.placement.retime import stretch as VST  # noqa: PLC0415

    DEV, DT = OPG.DEV, OPG.DT
    rz = Rotation.from_euler("z", -90, degrees=True)
    # Capture (tree) z → world z offset (coords prints `[DESK_Z]` once per tree).
    DESK = _VC.scene_z_offset(a.tree)
    global DESK_XY
    DESK_XY = _VC.desk_bounds_scene_for(a.tree, margin=0.0)  # the tree's dataset desk top — read by the guards
    _SCENE = _VC.scene_mode(a.tree)  # table|aerial — the map clearance below
    palm_off = AIP.palm_frame_offset(a.hand)
    combo = f"{a.arm}_{a.hand}"

    urdf, ee, joints, base_xyz, wd_inputs = AIP.resolve_combo(a.arm, a.hand, a.side)
    _idle = _tree_idle_sides(a)
    if a.side in _idle:
        return _G.idle_rest(a, joints, base_xyz, DESK, palm_off, rz, lambda q: stage_guards(q, a, a.output_dir))
    model, ik = _arm_ik(urdf, a, ee, joints, OPG)
    # Velocity budget: the joint step allowed per frame = velocityLimit × dt, per joint.
    _vidx = [model.joints[model.getJointId(n)].idx_v for n in joints]
    BUDGET = np.asarray(model.velocityLimit)[_vidx].astype(np.float64) * VST.FRAME_DT
    base = np.asarray(base_xyz) + np.array([0.0, 0.0, a.base_z_off])

    pd, axx, axy, axz, atoms, step, nx, ny, nz, A, desk_rel = _reach_map(
        a, combo, base_xyz, palm_off, wd_inputs, _SCENE, DESK, base
    )

    def _load_pool():
        """Valley joints VQ and validity VOK of every map cell. Loaded only on the DP path: the reuse path never
        reads the shards, so its output does not depend on them."""
        from dexengine.placement.reach_map import clearance as _CLR  # noqa: PLC0415 — owner of the desk-top check

        _CLR.check_desk_top(pd, _VC.desk_top_for(_VC.tree_dataset(a.tree)))  # shard vclr_* = tabletop values
        sh0 = np.load(pd / "shard_z000.npz")
        K = sh0["vpe"].shape[-1]
        VQ = np.empty((nx, ny, nz, A, K, ik.n), np.float16)
        VOK = np.empty((nx, ny, nz, A, K), bool)
        for k in range(nz):
            sh = np.load(pd / f"shard_z{k:03d}.npz")
            VQ[:, :, k] = sh["vq"].reshape(nx, ny, A, K, ik.n)
            VOK[:, :, k] = (
                (sh["vpe"].reshape(nx, ny, A, K) < VALLEY_OK_POS_M)
                & (sh["vre"].reshape(nx, ny, A, K) < VALLEY_OK_ROT_DEG)
                & (sh["vclr_arm"].reshape(nx, ny, A, K) > desk_rel - VALLEY_DESK_SLACK_M)
                & (sh["vclr_hand"].reshape(nx, ny, A, K) > desk_rel - VALLEY_DESK_SLACK_M)
            )
        return K, VQ, VOK

    lo3 = np.array([axx[0], axy[0], axz[0]])

    rot, pb, Rb, T = _wrist_targets(a, Rotation, rz, palm_off, DESK, base)
    FF = _finger_floor(a, T)
    Tt = _target_tensor(torch, T, DT, DEV, Rb, pb)

    t0 = time.time()
    qb = (rz * rot).as_quat()
    simA = np.abs(qb @ atoms.T)
    aiN = np.argsort(-simA, axis=1)[:, : a.atoms]  # top N atoms, identical to the judge
    gi = np.round((pb - lo3) / step).astype(int)

    return reuse_or_collect_then_dp(
        a, T, ik, Tt, torch, DT, DEV, VST, BUDGET, FF, base, joints, t0, _load_pool, gi, nx, ny, nz, aiN
    )


if __name__ == "__main__":
    raise SystemExit(main())
