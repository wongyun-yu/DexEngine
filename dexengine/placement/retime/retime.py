#!/usr/bin/env python3
"""Retiming — stretch the tree and the joints onto the same time map, at the speed of the settled joint trajectory.

The chain is: placement -> f0 repair -> one IK solve (DP) -> retiming (here). The joints are never re-solved; only the
time axis is resampled, so a joint that runs at k times its limit becomes exactly 1/k after its interval is stretched
by k (a property of linear interpolation). Object and hand data are resampled with the same tau, so the time axes
cannot drift apart. (Solving IK twice — once to plan the retiming and once for the data — breaks that guarantee when
the two solutions pick different valleys.)

Joints: both arms (vendor URDF velocityLimit) and every finger of every hand block in the tree (HandSpec's applied
limit, the value the sim actuator gets — hand_surface.finger_floor_for). The fingers are resampled with the same tau,
so they also end at <= 1/safety of their limit. A tree with no hand block dies before anything is written.

Input:  tree (<tree>/0/trajectory_kinematic.npz, f0 repair done) + joint npz (both sides)
Output: resampled tree + joints at the same paths (originals kept as .pre_rt), the time map in the tree npz
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from dexengine.core.paths import IK_SIDECAR_SFX
from dexengine.placement.coords import (
    core as _VC,
)  # owner of the support stamp (restamped after the trajectory is written)


def _resample_marked(MCV, z, tau, s, hqd, layout, tree):
    """Resample the tree, then measure the hand-root reference mark on the bytes about to be written.

    The hand roots' angle convention is read from the tree's own scene by its owner (retime_variants.maps_for ->
    BaseMap), and only when a frame needs it: a clip whose inserted frames are already proven close from the angle
    values alone never loads the model. Mark: retimed frames whose hand root sits further than ROOT_TURN_MAX_DEG
    (measured as a rotation) from the original motion at the same instant are counted and the clip is named on a
    `[ROOT_TURN] over` line — the clip is still written (one clip does not stop a bundle)."""
    owners: list = []

    def root_maps():
        if not owners:
            owners.append(MCV.maps_for(tree, hqd))
        return owners[0]

    out = MCV.resample_all(z, tau, s, hqd, layout=layout, root_maps=root_maps)
    turn, over, exact = MCV._rt.root_turn_mark(z["qpos"], out["qpos"], tau, hqd, root_maps=root_maps)
    lim = MCV._rt.ROOT_TURN_MAX_DEG
    if over:
        print(
            f"[ROOT_TURN] over {Path(tree).name}: {over} frames whose hand-root rotation is more than {lim:.0f}° off "
            f"the original · max {turn:.1f}° (frames inserted by the retiming, measured as a rotation)",
            flush=True,
        )
    else:
        print(
            f"[ROOT_TURN] 0 frames over · deviation at most {turn:.2f}° <= {lim:.0f}° (rotation measured on {exact} "
            "hands · the rest by the per-step angle bound)",
            flush=True,
        )
    return out


def _record_safety(ti_p: Path, safety: float) -> None:
    """Write task_info.retime_safety = the factor that actually produced s.

    The bake writes a placeholder before any retiming exists; it is corrected only when it differs, so a tree whose
    placeholder already equals --safety keeps its task_info.json bytes."""
    import json as _json  # noqa: PLC0415

    if not ti_p.is_file():
        return
    ti = _json.loads(ti_p.read_text())
    if ti.get("retime_safety") == safety:
        return
    ti["retime_safety"] = safety
    tmp = ti_p.with_suffix(".tmp.json")
    tmp.write_text(_json.dumps(ti, indent=1))
    tmp.replace(ti_p)
    print(
        f"[RETIME_SAFETY] {ti_p.parent.name}: task_info.retime_safety <- {safety} (the value actually used)", flush=True
    )


def _parse_args():
    """Command line: tree, arm, hand, the two IK files, safety and the stretch cap."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tree", required=True, help="one combo tree (<clip>_pl_rt under the dexengine.core.layout trees layer)"
    )
    ap.add_argument("--arm", required=True)
    ap.add_argument(
        "--hand", required=True, help="hand id (HandSpec registry name) — owner of the finger speed limits; no default"
    )
    ap.add_argument("--ik-right", required=True)
    ap.add_argument("--ik-left", required=True)
    ap.add_argument("--safety", type=float, default=1.15)
    ap.add_argument("--max-stretch", type=float, default=3.0, help="cap on the total stretch — above it the clip fails")
    a = ap.parse_args()
    return a


def _tree_layout(QL, tree, qpos, _spec, P, a):
    """Hand qpos width and the articulated layout (or None) of this hand's tree; another hand's tree dies in the
    reader."""
    _lay_rt = QL.read(
        str(tree), nq=qpos.shape[1], expect_hand_qpos_dim=int(_spec.hand_qpos_dim), expect_hand=P.hand_spec_dir(a.hand)
    )
    hqd = int(_lay_rt.hand_qpos_dim)
    print(f"[retime] hand_qpos_dim={hqd} nq={qpos.shape[1]} (qpos_layout)", flush=True)
    # Articulated objects: only a tree that has joint columns is resampled from the layout declaration (the quaternion
    # slots are read from the object blocks and the joint sidecars qpos_obj_arti_* follow the same time map). A tree
    # without joint columns passes None = the arithmetic slot path. The decision is made on the column list, not on the
    # per-side `artic` key, which misses background articulated objects.
    _lay_pass = _lay_rt if _lay_rt.artic_cols() else None
    if _lay_pass is not None:
        print(
            f"[ARTIC] retime {tree.name}: joint columns {_lay_rt.artic_cols()} passed through · sidecars "
            f"{[r.get('npz_key') for r in _lay_rt.artic_records()]} resampled",
            flush=True,
        )
    return hqd, _lay_pass


def _stretch_profile(AS, a, DT, HSF, tree, qpos, T, Q, VST):
    """Per-interval stretch: the worst of the two arms' velocity ratios and the finger floor (the judge's and the bake's
    functions)."""

    # Per-joint velocity budget (velocityLimit × dt)
    VLIM = np.asarray(AS.ARM_REGISTRY[a.arm].per_joint_limits("velocity"), dtype=float)

    # Stretch ratio: the worst of the two arms and the finger floor (every hand block of the tree)
    # Computed with the same functions as the judge and the bake (stretch · hand_surface.finger_floor_for), so a
    # trajectory an earlier stage passed is not rejected here by a different computation.
    BUD = VLIM * DT
    fq = HSF.tree_finger_q(tree, a.hand, qpos=qpos)  # the hand blocks the tree has (zero blocks die below)
    ff = HSF.finger_floor_for(tree, a.hand, a.safety, fq=fq)
    s = ff.copy()
    s_arm = np.ones(max(T - 1, 1))
    rmax0 = 0.0
    for q, _ in Q.values():
        s = np.maximum(s, VST.stretch_profile(q, BUD, a.safety, floor=ff))
        s_arm = np.maximum(s_arm, VST.stretch_profile(q, BUD, a.safety))
        rmax0 = max(rmax0, VST.raw_ratio_max(q, BUD))
    n_ff = int((ff > s_arm).sum())  # intervals the fingers lengthened beyond the arms
    stretch = float(np.mean(s))
    return BUD, fq, ff, s, rmax0, n_ff, stretch


def _resample_tree(T_new, DT, MCV, z, tau, s, hqd, _lay_pass, tree, a, T):
    """Resample the tree on the time map tau and store the map in the npz (retime_s, retime_tau, retime_dt,
    retime_safety, retime_src_idx)."""
    new_t = np.arange(T_new) * DT

    # 1) Resample the tree (object and hand qpos, etc.) · 1a) hand-root mark
    out = _resample_marked(MCV, z, tau, s, hqd, _lay_pass, tree)
    # 1b) Store the time map in the tree npz, so consumers (contact derivation, render overlay) read the exact warp
    # instead of re-deriving it. Keys:
    #   retime_s        (T-1,)   float64  per-interval factor k (= s); 1.0 = no slowdown
    #   retime_tau      (T,)     float64  original frame i -> new time (s); tau[0] = 0
    #   retime_dt       scalar   float64  frame interval (s) of the new time axis
    #   retime_src_idx  (T_new,) float32  tree frame k -> original (pre_rt) fractional index
    #                                     = interp(k*dt, tau, arange(T))
    #   retime_safety   scalar   float64  the safety factor that produced s
    # Contact derivation uses p_pre = retime_src_idx, p_base = p_pre - P (P = the f0 pad).
    out["retime_s"] = s.astype(np.float64)
    out["retime_tau"] = tau.astype(np.float64)
    out["retime_dt"] = np.float64(DT)
    out["retime_safety"] = np.float64(a.safety)
    out["retime_src_idx"] = np.interp(new_t, tau, np.arange(T, dtype=np.float64)).astype(np.float32)
    return new_t, out


def _resample_arms(Q, T_new, new_t, tau, VST, BUD, HSF, tree, a, out):
    """Resample both arms with the same tau and measure the worst post-resample speed ratio (arms, fingers)."""
    newQ = {}
    for side, (q, zz) in Q.items():
        qn = np.empty((T_new, q.shape[1]), np.float64)
        for j in range(q.shape[1]):
            qn[:, j] = np.interp(new_t, tau, q[:, j])
        newQ[side] = (qn, zz)

    # Verify that the speed is within the limit after resampling (guaranteed by construction, but checked) — arms and
    # fingers (the resampled tree's finger columns, the same columns the floor was built on)
    worst = worst_f = 0.0
    for qn, _ in newQ.values():
        if len(qn) > 1:
            worst = max(worst, float(VST.frame_ratio(qn, BUD).max()))
    if T_new > 1:
        worst_f = max(
            float(VST.frame_ratio(f, HSF.finger_budget(a.hand)).max())
            for f in HSF.tree_finger_q(tree, a.hand, qpos=out["qpos"]).values()
        )
    return newQ, worst, worst_f


def _write_retimed(kin_p, out, tree, a, tau, new_t, T, newQ, T_new, ikp):
    """Write the retimed tree, contact sidecars and IK files (each original kept as the sidecar)."""

    # 3) Write (originals preserved)
    shutil.copy2(kin_p, kin_p.with_suffix(IK_SIDECAR_SFX))
    _tmp = kin_p.with_suffix(".tmp.npz")
    np.savez(_tmp, **out)
    _tmp.replace(kin_p)  # published by one rename
    import json as _json

    _ti_p = tree / "task_info.json"
    _record_safety(_ti_p, float(a.safety))
    # The trajectory changed, so the support stamp is recomputed from it (retiming moves the rest intervals' frames)
    if _ti_p.is_file() and _VC.is_baked_tree(_json.loads(_ti_p.read_text())):
        _VC.stamp_scene(tree, keep_height=True)
    if (tree / "0/trajectory_ikrollout.npz").exists():
        np.savez(tree / "0/trajectory_ikrollout.npz", qpos=out.get("qpos_rollout", out["qpos"]))
    # Sidecars (contact, etc.): nearest mapping on the time axis — the original contacts are not recomputed
    idx = np.clip(np.searchsorted(tau, new_t), 0, T - 1)
    for cf in sorted((tree / "0").glob("contact_mano_*.npz")):
        if ".pre_rt" in cf.name:
            continue
        cz = dict(np.load(cf))
        shutil.copy2(cf, cf.with_suffix(IK_SIDECAR_SFX))
        for k, v in list(cz.items()):
            v = np.asarray(v)
            if v.ndim >= 1 and v.shape[0] == T:
                cz[k] = v[idx]
        np.savez(cf, **cz)
    for side, (qn, zz) in newQ.items():
        d = dict(zz)
        d["arm_joint_positions_all"] = qn.astype(np.float32)
        d["arm_joint_positions"] = qn[0].astype(np.float32)
        d["num_frames"] = np.int32(T_new)
        for k in ("wrist_pos_error_all", "wrist_rot_error_deg_all"):
            if k in d:
                v = np.asarray(d[k]).astype(np.float64)
                d[k] = np.interp(new_t, tau, v).astype(np.float32)
        p = ikp[side]
        shutil.copy2(p, p.with_suffix(IK_SIDECAR_SFX))
        np.savez(p, **d)


def main() -> int:
    a = _parse_args()

    from dexengine.core import arms as AS  # noqa: PLC0415
    from dexengine.core import hands as HS  # noqa: PLC0415
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.core import qpos_layout as QL  # noqa: PLC0415
    from dexengine.placement import retime_variants as MCV  # noqa: PLC0415
    from dexengine.placement.collision import hand_surface as HSF  # noqa: PLC0415
    from dexengine.placement.retime import stretch as VST  # noqa: PLC0415

    DT = VST.FRAME_DT  # the tree frame interval (owner: stretch)

    tree = Path(a.tree)
    kin_p = tree / "0/trajectory_kinematic.npz"
    z = dict(np.load(kin_p))
    qpos = z["qpos"]
    T = len(qpos)
    # The hand qpos width comes from the qpos_layout reader (task_info.hand_qpos_dim), never from (D - 14) // 2, which
    # assumes two objects and would pick the quaternion slots off by one on 1-object or 3+-object trees. The tree must
    # be this hand's (stated width and identity): an unknown hand or another hand's tree dies before anything is
    # written.
    _spec = HS.HAND_REGISTRY.get(a.hand)
    if _spec is None:
        print(f"[v5retime] hand {a.hand!r} is not in HandSpec — finger limits unknown · failed", flush=True)
        return 2
    hqd, _lay_pass = _tree_layout(QL, tree, qpos, _spec, P, a)

    # Joint trajectories (both sides) — the length must equal the tree's (they were solved on the same time axis)
    ikp = {"right": Path(a.ik_right), "left": Path(a.ik_left)}
    Q = {}
    for side, p in ikp.items():
        zz = np.load(p)
        q = zz["arm_joint_positions_all"].astype(np.float64)
        if len(q) != T:
            print(f"[v5retime] {side} joint length {len(q)} != tree {T} — failed", flush=True)
            return 3
        Q[side] = (q, zz)
    BUD, fq, ff, s, rmax0, n_ff, stretch = _stretch_profile(AS, a, DT, HSF, tree, qpos, T, Q, VST)
    if stretch > a.max_stretch:
        print(
            f"[v5retime] stretch {stretch:.2f}x > cap {a.max_stretch} — failed"
            f" (r_max={rmax0:.2f} · finger floor mean {float(ff.mean()):.3f} contributing spans {n_ff})",
            flush=True,
        )
        return 3

    # Uniform slowdown: the factor s is one scalar per interval, and both arms, every joint, the object and the hand
    # channels are resampled with that same s (-> the same tau). There is no per-channel factor; the shape check dies
    # if s ever becomes (T-1, n).
    assert s.ndim == 1 and len(s) == max(T - 1, 1), (
        f"[v5retime] uniform slowdown violated: s.shape={s.shape} (expected ({max(T - 1, 1)},))"
    )
    assert float(s.min()) >= 1.0, f"[v5retime] s<1 (speed-up) — {float(s.min()):.4f}"
    tau = np.concatenate([[0.0], np.cumsum(s * DT)])
    T_new = int(np.floor(tau[-1] / DT + 1e-9)) + 1  # floating-point correction (same as resample_all)
    new_t, out = _resample_tree(T_new, DT, MCV, z, tau, s, hqd, _lay_pass, tree, a, T)
    # 2) Resample the joints with the same tau. Linear interpolation, so the speed becomes exactly 1/s.
    newQ, worst, worst_f = _resample_arms(Q, T_new, new_t, tau, VST, BUD, HSF, tree, a, out)
    # A violated guarantee is a failure, not a warning: nothing is written — the same rc 3 as the stretch cap.
    if max(worst, worst_f) > 1.0 / 0.95:  # 5% margin
        print(
            f"[v5retime] after resampling r_max arm {worst:.2f} · fingers {worst_f:.2f} > {1.0 / 0.95:.4f} "
            f"(guarantee violated) — failed · the tree is not written",
            flush=True,
        )
        return 3
    _write_retimed(kin_p, out, tree, a, tau, new_t, T, newQ, T_new, ikp)
    print(
        f"[v5retime] T {T}→{T_new} (stretch {T_new / max(T, 1):.2f}x) · "
        f"r_max before {rmax0:.2f} → after {worst:.2f} · "
        f"slowed intervals {int((s > 1).sum())}/{len(s)} (max factor {float(s.max()):.2f}) · "
        f"fingers {a.hand} nf={len(HSF.finger_names(a.hand, 'right'))}×{len(fq)} floor mean {float(ff.mean()):.4f} "
        f"contributing intervals {n_ff} after r_max {worst_f:.2f} · "
        f"time map saved (retime_s·retime_tau·retime_src_idx)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
