"""f0 repair outputs: write the repaired frame-0 pose into the tree in place, or as separate outputs (pad frames,
wrist sidecar and qpos)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from dexengine.core.paths import PLACEMENT_SFX, WRIST_SIDECAR_SFX
from dexengine.core.paths import main_root as _data_root
from dexengine.core.paths import qpos_layout as QL
from dexengine.data.objects import scene_parts as SP
from dexengine.retarget.f0 import phrases as PH
from dexengine.retarget.f0.repair_cfg import HC
from dexengine.retarget.f0.repair_geom import _aa_to_R, _R_to_quat_wxyz, _slerp, fit_wrist_qpos_map


def _tree_sides(tree) -> list[str]:
    """The hands in the tree's qpos_layout (right, then left). A single-hand emb tree has one —
    the wrist cache of an absent hand is never read (2a)."""
    return list(QL.read(str(tree), nq=None).manip_sides())  # D23: no main object (inactive) = not processed


def _fit_wrists(qpos, seq, sides_repaired, wrist_root):
    """Fit the qpos→world wrist map of every repaired side before anything is written (a side that dies leaves
    no half-written tree)."""
    # measure all wrist fits before writing, so a side that dies never leaves a half-written tree with only the other
    # side's cache patched
    fits = {}
    for side, r in sides_repaired.items():
        wp_f = wrist_root / f"wrist_poses_{side}" / f"{seq}.npz"
        with np.load(wp_f) as wr:
            fits[side] = fit_wrist_qpos_map(
                qpos[:, r["_internal"][6] : r["_internal"][6] + 6],
                wr["positions"].astype(np.float64),
                where=f"{seq} [{side}] {wp_f}",
            )

    return fits


def _blend_rows(H, P, P_base, a_grid, mode, qpos, sides_repaired):
    """The repaired frame-0 pose blended back onto the original: rewritten pad rows (rewrite) or new pad rows
    (prepend), plus each side's start pose."""
    new_qpos = qpos.copy()
    blend = None  # the prepended pad rows (prepend mode only)
    if mode == "prepend":
        blend = np.repeat(qpos[0:1], P, axis=0)
    starts = {}
    for side, r in sides_repaired.items():
        dq, wpos0, wquat0, wR0, indep0, fingers0, off, _ = r["_internal"]
        rp = wpos0 + dq[:3]
        rq = _R_to_quat_wxyz(_aa_to_R(dq[3:6]) @ wR0)
        f_rep = HC.expand(indep0 + dq[6:])
        starts[side] = (rp, rq, f_rep, off)
        tgt = qpos[P_base if mode == "rewrite" else 0, off + 6 : off + H]
        rows = f_rep[None, :] * (1 - a_grid)[:, None] + tgt[None, :] * a_grid[:, None]
        if mode == "rewrite":
            new_qpos[:P, off + 6 : off + H] = rows
        else:
            blend[:, off + 6 : off + H] = rows
    return blend, new_qpos, starts


def write_in_place(seq: str, sides_repaired: dict, data_root: Path, wrist_root: Path, n_blend: int):
    """In-place repair of a gen tree — the sequence name and the timeline are preserved.

    WHY: in a combo gen tree the `_f0` sequence already has the baseline (legacy) pad
    of length P prepended by the variant generator (a hold on its own frame-0). Making yet another
    suffixed tree here would split every name the bundle, the caches and the env use. So
      * if a pad exists (P>0):  rows 0..P-1 are **rewritten** as a blend 'repaired pose →
        original frame-0 (row P)'. The length is unchanged — timeline, object columns and the
        other hand stay as they were.
      * if there is no pad (P=0):  P frames are **prepended** per the v6 convention (the name is
        unchanged, the length is +P — only for the tree of the hand concerned, and recorded in
        F0_REPAIR_INPLACE).
    The original is backed up 1× as *.pre_f0r.npz (idempotent: if the marker exists the caller skips).
    """
    d = data_root / seq
    kin = d / "0/trajectory_kinematic.npz"
    npz = np.load(kin, allow_pickle=True)
    qpos = npz["qpos"]
    qvel = npz["qvel"]
    H = HC.H
    freq = float(npz["frequency"]) if "frequency" in npz.files else 50.0
    dt = 1.0 / max(freq, 1.0)
    F_CAP = HC.f_cap_rad_s * dt
    W_CAP = 0.010
    fj = d / "F0_REPAIR.json"
    P_base = 0
    if fj.exists():  # a marker that exists but cannot be read dies (never a silent prepend)
        P_base = int(json.loads(fj.read_text()).get("n_blend_effective", 0) or 0)
    mode = "rewrite" if P_base > 0 else "prepend"

    pad_vel = {}
    if mode == "rewrite":
        P = P_base
        for side, r in sides_repaired.items():
            dq, wpos0, wquat0, wR0, indep0, fingers0, off, _ = r["_internal"]
            f_rep = HC.expand(indep0 + dq[6:])
            pad_vel[side] = {
                "finger_max_rad_s": round(float(np.abs(f_rep - fingers0).max()) / (P * dt), 3),
                "wrist_m_s": round(float(np.linalg.norm(dq[:3])) / (P * dt), 4),
                "cap_rad_s": round(HC.f_cap_rad_s, 3),
            }
    else:
        P = int(n_blend)
        for r in sides_repaired.values():
            dq, wpos0, wquat0, wR0, indep0, fingers0, off, _ = r["_internal"]
            f_rep = HC.expand(indep0 + dq[6:])
            d_f = float(np.abs(f_rep - fingers0).max())
            d_w = float(np.linalg.norm(dq[:3]))
            P = max(P, int(np.ceil(d_f / F_CAP)), int(np.ceil(d_w / W_CAP)))
        if P > 60:
            return None, PH.pad_too_big(P)
    a_grid = np.arange(P, dtype=np.float64) / max(P, 1)

    fits = _fit_wrists(qpos, seq, sides_repaired, wrist_root)
    # backup (1×)
    bkin = d / f"0/trajectory_kinematic{WRIST_SIDECAR_SFX}"
    if not bkin.exists():
        shutil.copy2(kin, bkin)

    blend, new_qpos, starts = _blend_rows(H, P, P_base, a_grid, mode, qpos, sides_repaired)

    _rewrite_wrist_caches(
        seq, sides_repaired, data_root, wrist_root, mode, P, P_base, a_grid, fits, starts, new_qpos, blend
    )
    _write_trees(d, kin, npz, qpos, qvel, new_qpos, blend, mode, P)
    _write_markers(d, seq, sides_repaired, mode, P, P_base, dt, pad_vel)
    return mode, None


def _rewrite_wrist_caches(
    seq, sides_repaired, data_root, wrist_root, mode, P, P_base, a_grid, fits, starts, new_qpos, blend
) -> None:
    """write_in_place step: rewrite the wrist caches of the hands in the tree (repaired hands get the blend; in prepend
    mode the other hand gets a hold pad); fills the wrist columns of `new_qpos` (rewrite) or `blend` (prepend)."""
    # wrist cache (canonical for that hand's tree) — only the hands in the tree
    # (2a single-hand emb: owned by qpos_layout) ----
    for side in _tree_sides(data_root / seq):
        wp_f = wrist_root / f"wrist_poses_{side}" / f"{seq}.npz"
        wr = np.load(wp_f)
        pos, quat = wr["positions"].astype(np.float64), wr["quaternions"].astype(np.float64)
        bwp = wp_f.with_suffix(WRIST_SIDECAR_SFX)
        if not bwp.exists():
            shutil.copy2(wp_f, bwp)
        if side in sides_repaired:
            rp, rq, f_rep, off = starts[side]
            tgt_p = pos[P_base] if mode == "rewrite" else pos[0]
            tgt_q = quat[P_base] if mode == "rewrite" else quat[0]
            blend_pos = np.array([rp + (tgt_p - rp) * a for a in a_grid])
            blend_quat = np.array([_slerp(rq, tgt_q, float(a)) for a in a_grid])
            A, b, _res = fits[side]
            if mode == "rewrite":
                npos = pos.copy()
                nquat = quat.copy()
                npos[:P], nquat[:P] = blend_pos, blend_quat
            else:
                npos = np.vstack([blend_pos, pos])
                nquat = np.vstack([blend_quat, quat])
            Ainv = np.linalg.inv(A)
            for k in range(P):
                new_qpos[k, off : off + 3] = (
                    (Ainv @ (blend_pos[k] - b)) if mode == "rewrite" else new_qpos[k, off : off + 3]
                )
            if mode == "prepend":
                qwrist = np.array([Ainv @ (blend_pos[k] - b) for k in range(P)])
                blend[:, off : off + 3] = qwrist
            np.savez(wp_f, positions=npos.astype(np.float32), quaternions=nquat.astype(np.float32))
        elif mode == "prepend":
            np.savez(
                wp_f,
                positions=np.vstack([np.repeat(pos[0:1], P, 0), pos]).astype(np.float32),
                quaternions=np.vstack([np.repeat(quat[0:1], P, 0), quat]).astype(np.float32),
            )


def _write_trees(d, kin, npz, qpos, qvel, new_qpos, blend, mode, P) -> None:
    """write_in_place step: the main kinematic npz, and in prepend mode the sidecars on the same frame axis."""
    # main npz
    out = {k: npz[k] for k in npz.files}
    if mode == "rewrite":
        out["qpos"] = new_qpos  # qvel/rollout/contact keep their length and pad
    else:
        T = qpos.shape[0]
        out["qpos"] = np.vstack([blend, qpos])
        out["qvel"] = np.vstack([np.zeros((P, qvel.shape[1]), qvel.dtype), qvel])
        for k in ("qpos_rollout", "contact", "contact_pos"):
            if k in out and getattr(out[k], "ndim", 0) >= 1 and out[k].shape[0] == T:
                out[k] = np.concatenate([np.repeat(out[k][0:1], P, 0), out[k]], 0)
    np.savez(kin, **out)
    if mode == "prepend":
        # Pad sync: the sidecars that share the same frame axis (the marker blocks an idempotent re-run)
        T = qpos.shape[0]
        for f in ("contact_mano_right.npz", "contact_mano_left.npz"):
            sfp = d / "0" / f
            if sfp.exists():
                z = np.load(sfp, allow_pickle=True)
                zz = {
                    k: (
                        np.concatenate([np.repeat(z[k][0:1], P, 0), z[k]], 0)
                        if getattr(z[k], "ndim", 0) >= 1 and abs(z[k].shape[0] - T) <= 4
                        else z[k]
                    )
                    for k in z.files
                }
                np.savez(sfp, **zz)
        rfp = d / "0/trajectory_ikrollout.npz"
        if rfp.exists():
            z = np.load(rfp, allow_pickle=True)
            zz = {k: z[k] for k in z.files}
            for k in ("qpos", "qpos_rollout", "qvel", "contact", "contact_pos"):
                if k in zz and getattr(zz[k], "ndim", 0) >= 1 and abs(zz[k].shape[0] - T) <= 4:
                    zz[k] = np.concatenate([np.repeat(zz[k][0:1], P, 0), zz[k]], 0)
            if "qpos" in zz:
                zz["qpos"] = out["qpos"]
            np.savez(rfp, **zz)


def _write_markers(d, seq, sides_repaired, mode, P, P_base, dt, pad_vel) -> None:
    """write_in_place step: F0_REPAIR_INPLACE.json, and in prepend mode the F0_REPAIR.json the overlay reads."""
    meta = {
        "hand": HC.name,
        "mode": mode,
        "P": int(P),
        "P_base": int(P_base),
        "dt": dt,
        "pad_velocities": pad_vel,
        "sides": {sd: {k: v for k, v in r.items() if k != "_internal"} for sd, r in sides_repaired.items()},
        "note": (
            "rewrite: the existing hold pad is rewritten as a repaired → original-frame-0 blend (length "
            "unchanged). prepend: a new v6 pad is attached (length +P). body_positions keep the original — "
            "the overlay reads the frames shifted by the P of F0_REPAIR(_INPLACE)."
        ),
    }
    with open(d / "F0_REPAIR_INPLACE.json", "w") as fout:
        json.dump(meta, fout, indent=1)
    # Consumption contract: the render overlay and evaluation use n_blend_effective from
    # F0_REPAIR.json to read the MANO frames pulled back by P. A prepend-mode
    # tree had no such file, so it is created here (for rewrite the P of the existing file still
    # holds).
    if mode == "prepend":
        # The base tree records no pad of its own (pad_baseline 0, the bake's record schema) — the total pad
        # the overlay reads is this P.
        with open(d / "F0_REPAIR.json", "w") as fh:
            json.dump(
                {
                    "source": seq,
                    "design": "inplace_prepend_v6",
                    "n_blend_effective": int(P),
                    "pad_new": int(P),
                    "pad_baseline": 0,
                    "dt": dt,
                },
                fh,
                indent=1,
            )


def _pad_length(dt, n_blend, qpos, seq, sides_repaired, wrist_root):
    """Pad length P (floor n_blend, raised by the finger and wrist step caps; over 60 frames stops) and each side's
    wrist fit."""
    # v6 PAD DESIGN
    #   * P starts at n_blend (20 frames = 0.4 s; 10 is too short) and the step caps raise it.
    #   * New frame 0   = repaired pose.
    #   * New frames 0..P-1 interpolate repaired -> ORIGINAL FRAME 0 (wrist pos lerp,
    #     wrist rot slerp, finger joints lerp — each channel eases in naturally).
    #   * New frame P   = original frame 0, and frames P.. are the ORIGINAL trajectory
    #     UNCHANGED, byte for byte. Nothing is discarded; total length = P + T.
    #   * The UNREPAIRED hand and BOTH objects hold STILL at their original frame-0 pose
    #     for the P pad frames (not slow-motion) — they simply start 0.4 s later.
    #   * MANO is untouched; its sbs video gets a 0.4 s freeze at composition time.
    #   * P adapts UPWARD from the floor (scale rule): every sequence gets
    #     the same closed-form check — P = max(floor, ceil(Δq/(v_lim·dt)), ceil(Δwrist/1cm)),
    #     where Δq is over ALL finger joints (every registered hand is fully actuated, and
    #     sim drives each joint with its own velocity-limited actuator, so the max over
    #     joints is the binding number). No rollout is
    #     needed — the pad is pure kinematics, so feasibility is decidable at write time.
    #     Hard cap 60 frames (1.2 s): needing more signals a pathological repair — FAIL loudly.
    P = int(n_blend)
    F_CAP = HC.f_cap_rad_s * dt  # rad/frame (per-hand minimum applied velocity limit / stretch.SAFETY)
    W_CAP = 0.010  # m/frame
    fits = {}
    for side, r in sides_repaired.items():
        dq, wpos0, wquat0, wR0, indep0, fingers0, off, _ = r["_internal"]
        f_rep = HC.expand(indep0 + dq[6:])
        wr0 = np.load(wrist_root / f"wrist_poses_{side}" / f"{seq}.npz")
        # measure the wrist fit before writing (same rule as write_in_place: on failure nothing of this clip is written)
        fits[side] = fit_wrist_qpos_map(
            qpos[:, off : off + 6], wr0["positions"].astype(np.float64), where=f"{seq} [{side}]"
        )
        d_f = float(np.abs(f_rep - fingers0).max())
        d_w = float(np.linalg.norm((wpos0 + dq[:3]) - wr0["positions"][0].astype(np.float64)))
        P = max(P, int(np.ceil(d_f / F_CAP)), int(np.ceil(d_w / W_CAP)))
    if P > 60:
        raise SystemExit(f"{seq}: {PH.pad_too_big(P)}, stopped")
    return P, fits


def _pad_rows(P, a_grid, dt, qpos, seq, sides_repaired, wrist_root):
    """Pad rows frozen at the original frame 0 except the repaired hands, which ease from the repaired pose; plus start
    poses and pad velocities."""
    # pad rows: EVERYTHING frozen at original frame 0 ...
    blend = np.repeat(qpos[0:1], P, axis=0)
    starts = {}
    pad_vel = {}
    for side, r in sides_repaired.items():
        dq, wpos0, wquat0, wR0, indep0, fingers0, off, _ = r["_internal"]
        rp = wpos0 + dq[:3]
        rq = _R_to_quat_wxyz(_aa_to_R(dq[3:6]) @ wR0)
        f_rep = HC.expand(indep0 + dq[6:])
        starts[side] = (rp, rq, f_rep, off)
        # ... except the repaired hand, which eases from the repaired pose to orig[0]
        blend[:, off + 6 : off + HC.H] = f_rep[None, :] * (1 - a_grid)[:, None] + fingers0[None, :] * a_grid[:, None]
        # pad-segment velocity report (limits are NOT enforced here by design):
        wr0 = np.load(wrist_root / f"wrist_poses_{side}" / f"{seq}.npz")
        pad_vel[side] = {
            "finger_max_rad_s": round(float(np.abs(f_rep - fingers0).max()) / (P * dt), 3),
            "wrist_m_s": round(float(np.linalg.norm(rp - wr0["positions"][0].astype(np.float64))) / (P * dt), 4),
        }
    return blend, pad_vel, starts


def _write_wrist_caches(P, a_grid, data_root, fits, new_qpos, seq, sides_repaired, starts, suffix, wrist_root):
    """Write the suffixed wrist caches (repaired side eases to cache[0], the other holds it) and set the repaired wrist
    qpos rows."""
    # wrist caches: repaired side eases repaired -> cache[0]; other side holds cache[0].
    # Frames P.. are the ORIGINAL cache rows unchanged.  (only the hands in the tree — 2a)
    for side in _tree_sides(data_root / seq):
        wr = np.load(wrist_root / f"wrist_poses_{side}" / f"{seq}.npz")
        pos, quat = wr["positions"].astype(np.float64), wr["quaternions"].astype(np.float64)
        if side in sides_repaired:
            rp, rq, f_rep, off = starts[side]
            blend_pos = np.array([rp + (pos[0] - rp) * a for a in a_grid])
            blend_quat = np.array([_slerp(rq, quat[0], float(a)) for a in a_grid])
            A, b, _res = fits[side]
            Ainv = np.linalg.inv(A)
            for k in range(P):
                new_qpos[k, off : off + 3] = Ainv @ (blend_pos[k] - b)
        else:
            blend_pos = np.repeat(pos[0:1], P, axis=0)
            blend_quat = np.repeat(quat[0:1], P, axis=0)
        np.savez(
            wrist_root / f"wrist_poses_{side}" / f"{seq}{suffix}.npz",
            positions=np.vstack([blend_pos, pos]).astype(np.float32),
            quaternions=np.vstack([blend_quat, quat]).astype(np.float32),
        )


def _copy_side_files(P, T, dst, src):
    """Copy task_info, scene xml files, contact and annotation files into the new tree (frame-indexed arrays get the
    pad)."""
    SP.verify_dir(src)  # scene convex parts = the current decomposition; otherwise the upstream tree is stale
    for f in ("task_info.json", *SP.SCENE_XMLS):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    for f in ("contact_mano_right.npz", "contact_mano_left.npz", "text_annotations.json"):
        s = src / "0" / f
        if not s.exists():
            continue
        if f.endswith(".npz"):
            z = np.load(s, allow_pickle=True)
            zz = {}
            for k in z.files:
                a = z[k]
                zz[k] = (
                    np.concatenate([np.repeat(a[0:1], P, axis=0), a], axis=0)
                    if getattr(a, "ndim", 0) >= 1 and abs(a.shape[0] - T) <= 4
                    else a
                )
            np.savez(dst / "0" / f, **zz)
        else:
            shutil.copy2(s, dst / "0" / f)
    if (src / "0/body_positions.npz").exists():
        shutil.copy2(src / "0/body_positions.npz", dst / "0/body_positions.npz")


def _write_ikrollout(P, T, dst, new_qpos, new_qvel, src):
    """Write trajectory_ikrollout.npz (the loader enumerates sequences by it) with the repaired arrays."""
    # DISCOVERY MARKER. The trajectory loader globs "*/0/trajectory_ikrollout.npz" to ENUMERATE
    # sequences, then reads trajectory_kinematic.npz for the actual motion
    # (rl/tasks/mdp/trajectory/loader.py). Without this file the repaired sequence is
    # invisible and training dies with "No trajectories match filters".
    # We write the SAME repaired arrays, so even the
    # fallback branch (kinematic missing) would be correct; the project rule "object
    # motion comes from kinematic only" is unaffected because kinematic is present and
    # always preferred.
    src_roll = src / "0/trajectory_ikrollout.npz"
    if src_roll.exists():
        z = np.load(src_roll, allow_pickle=True)
        zz = {k: z[k] for k in z.files}
        for k in ("qpos", "qpos_rollout", "qvel", "contact", "contact_pos"):
            if k in zz and getattr(zz[k], "ndim", 0) >= 1 and abs(zz[k].shape[0] - T) <= 4:
                zz[k] = np.concatenate([np.repeat(zz[k][0:1], P, axis=0), zz[k]], axis=0)
        if "qpos" in zz:
            zz["qpos"] = new_qpos  # repaired frames win
        np.savez(dst / "0/trajectory_ikrollout.npz", **zz)
    else:
        np.savez(dst / "0/trajectory_ikrollout.npz", qpos=new_qpos, qvel=new_qvel)


def _write_sidecars(P, dst, dt, pad_vel, seq, sides_repaired, suffix):
    """Write the suffixed placement sidecar and the F0_REPAIR.json ledger of the new tree."""
    pl = _data_root() / "precompute/taco/placement" / f"{seq}{PLACEMENT_SFX}"
    if pl.exists():
        with open(pl) as fin:
            j = json.load(fin)
        j["sequence"] = f"{seq}{suffix}"
        j["f0_repair"] = True
        with open(pl.parent / f"{seq}{suffix}{PLACEMENT_SFX}", "w") as fout:
            json.dump(j, fout, indent=1)

    meta = {
        "source": seq,
        "design": "v6_pad20_land_on_orig0_freeze_other",
        "n_blend_effective": P,
        "dt": dt,
        "pad_velocities": pad_vel,
        "sides": {},
    }
    for side, r in sides_repaired.items():
        meta["sides"][side] = {k: v for k, v in r.items() if k != "_internal"}
    meta["note"] = (
        "body_positions.npz is a stale copy of the source (frame axis NOT "
        "shifted); wrist caches for this variant were written directly — do "
        "NOT re-run wrist_poses_from_body_positions on this dir."
    )
    with open(dst / "F0_REPAIR.json", "w") as fh:
        json.dump(meta, fh, indent=1)


def write_outputs(seq: str, sides_repaired: dict, data_root: Path, wrist_root: Path, n_blend: int, suffix: str):
    """Assemble <seq><suffix> tree + wrist caches for BOTH sides + placement sidecar."""
    src = data_root / seq
    dst = data_root / f"{seq}{suffix}"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "0").mkdir(exist_ok=True)
    npz = np.load(src / "0/trajectory_kinematic.npz", allow_pickle=True)
    qpos, qvel = npz["qpos"], npz["qvel"]
    T = qpos.shape[0]
    freq = float(npz["frequency"]) if "frequency" in npz.files else 50.0
    dt = 1.0 / max(freq, 1.0)

    P, fits = _pad_length(dt, n_blend, qpos, seq, sides_repaired, wrist_root)
    a_grid = np.arange(P, dtype=np.float64) / P  # a=0 → repaired, a→1 → orig[0]

    blend, pad_vel, starts = _pad_rows(P, a_grid, dt, qpos, seq, sides_repaired, wrist_root)
    new_qpos = np.vstack([blend, qpos])  # frames P.. = original, unchanged
    new_qvel = np.vstack([np.zeros((P, qvel.shape[1]), qvel.dtype), qvel])
    new_roll = np.vstack([np.repeat(npz["qpos_rollout"][0:1], P, axis=0), npz["qpos_rollout"]])

    _write_wrist_caches(P, a_grid, data_root, fits, new_qpos, seq, sides_repaired, starts, suffix, wrist_root)

    out = {k: npz[k] for k in npz.files}
    out["qpos"], out["qvel"], out["qpos_rollout"] = new_qpos, new_qvel, new_roll
    for k in ("contact", "contact_pos"):
        arr = npz[k]
        out[k] = np.concatenate([np.repeat(arr[0:1], P, axis=0), arr], axis=0)
    np.savez(dst / "0/trajectory_kinematic.npz", **out)

    _copy_side_files(P, T, dst, src)

    _write_ikrollout(P, T, dst, new_qpos, new_qvel, src)

    _write_sidecars(P, dst, dt, pad_vel, seq, sides_repaired, suffix)
    return dst
